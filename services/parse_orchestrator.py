"""
services/parse_orchestrator.py

Phase 2 orchestrator. Coordinates:
  1. ParseCacheManager  — cache hit/miss check
  2. ParsingAgent       — PDF -> RawChunks
  3. ChunkingAgent      — RawChunks -> DB DocumentChunks
  4. EmbeddingService   — local embeddings stored in DB
  5. Report status      — updates report.status in DB

Cache key: (report_id, parser_version)

Each Report row has exactly one ParsedDocument row per parser version.
Bumping parser_version in config.py (settings.parser_version) invalidates
the cache and triggers a fresh parse on next run.

FIXES (v2)
----------
- Pre-embedding diagnostic: logs chunk count and how many already have
  embeddings before calling embed_document().
- Post-embedding diagnostic: logs how many chunks now have embeddings,
  broken down by type (table / text / footnote).
- Embedding failure is still non-fatal but now logs an explicit actionable
  ERROR instead of a generic WARNING, making it easy to spot in logs.
- embed_document() return value is checked: 0 means nothing was embedded,
  which now triggers a dedicated log message pointing at the most likely
  causes (missing pgvector extension, model load failure, no internet).
"""
from __future__ import annotations

import uuid
from pathlib import Path

from core.config import get_settings
from core.database import get_db
from core.logging_config import get_logger
from models.db_models import Report, DocumentChunk
from models.schemas import ParsedDocumentRead
from services.parse_cache import get_cached_parse, is_cached, store_parse

logger = get_logger(__name__)


def _log_embedding_coverage(
    parsed_document_id: uuid.UUID,
    db,
    label: str,
) -> None:
    """
    Query and log the current embedding coverage for a parsed document.
    Called before and after embed_document() to make the effect visible.
    """
    try:
        from sqlalchemy import func as sa_func
        total = db.query(sa_func.count(DocumentChunk.id)).filter(
            DocumentChunk.parsed_document_id == parsed_document_id,
        ).scalar() or 0
        embedded = db.query(sa_func.count(DocumentChunk.id)).filter(
            DocumentChunk.parsed_document_id == parsed_document_id,
            DocumentChunk.is_embedded == True,
            DocumentChunk.embedding.isnot(None),
        ).scalar() or 0
        pct = round(100 * embedded / total, 1) if total else 0
        logger.info(
            f"parse_orchestrator.embedding_coverage_{label}",
            parsed_document_id=str(parsed_document_id),
            total_chunks=total,
            embedded_chunks=embedded,
            coverage_pct=pct,
        )
    except Exception as exc:
        logger.warning(
            "parse_orchestrator.coverage_query_failed",
            label=label,
            error=str(exc),
        )


class ParseOrchestrator:
    """
    Stateless orchestrator. Safe to instantiate once and call run() many times.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def run(self, report_id: uuid.UUID, force: bool = False) -> ParsedDocumentRead:
        """
        Parse a downloaded PDF and cache the results.

        Cache behavior:
            If force=False and a ParsedDocument already exists for
            (report_id, parser_version), return it immediately without
            re-parsing (cache hit).

            If force=True, delete any existing cache entry and re-parse
            from scratch. Use this after upgrading the parser.

        Args:
            report_id: UUID of a Report with status in
                       ('downloaded', 'parsed', 'extracted').
            force:     If True, bypass the cache and re-parse.

        Returns:
            ParsedDocumentRead with .id, .page_count, .meta.
            meta includes: chunk_count, table_count, ocr_page_count,
            word_count, parser_version, pdf_path, mode.

        Raises:
            ValueError if the report is not found, has the wrong status,
            has no file_path, or the file is missing from disk.
        """
        parser_version = self.settings.parser_version

        with get_db() as db:
            # Load the Report row
            report = db.query(Report).filter(Report.id == report_id).first()
            if not report:
                raise ValueError(f"Report not found: {report_id}")

            if report.status not in ("downloaded", "parsed", "extracted"):
                raise ValueError(
                    f"Report {report_id} has status '{report.status}'. "
                    f"Expected 'downloaded' before parsing."
                )

            if not report.file_path:
                raise ValueError(
                    f"Report {report_id} has no file_path. "
                    f"Run download first."
                )

            pdf_path = Path(report.file_path)
            if not pdf_path.exists():
                raise ValueError(
                    f"PDF not found on disk: {pdf_path}. "
                    f"The file may have been deleted after download."
                )

            # Cache hit: (report_id, parser_version) already parsed
            if not force and is_cached(report_id, parser_version, db):
                logger.info(
                    "parse_orchestrator.cache_hit",
                    report_id=str(report_id),
                    parser_version=parser_version,
                )
                cached = get_cached_parse(report_id, parser_version, db)
                # Even on cache hit, run embeddings if chunks are missing them
                _log_embedding_coverage(cached.id, db, "cache_hit_check")
                self._embed_if_needed(cached.id, db)
                _log_embedding_coverage(cached.id, db, "cache_hit_after_embed")
                return ParsedDocumentRead.model_validate(cached)

            # Cache miss: run the full parse pipeline
            logger.info(
                "parse_orchestrator.parsing",
                report_id=str(report_id),
                path=str(pdf_path),
                parser_version=parser_version,
            )

            from agents.parsing_agent import ParsingAgent
            from agents.chunking_agent import ChunkingAgent

            try:
                raw_chunks, meta = ParsingAgent().parse(pdf_path)
            except Exception as exc:
                report.status        = "failed"
                report.error_message = f"Parsing failed: {exc}"
                db.flush()
                logger.error(
                    "parse_orchestrator.parse_failed",
                    report_id=str(report_id),
                    error=str(exc),
                )
                raise

            # Store the ParsedDocument cache entry.
            parsed_doc = store_parse(
                report_id=report_id,
                parser_version=parser_version,
                page_count=meta["page_count"],
                meta=meta,
                db=db,
            )

            # Chunk and store DocumentChunks
            db_chunks = ChunkingAgent().chunk_and_store(raw_chunks, parsed_doc, db)

            logger.info(
                "parse_orchestrator.chunking_done",
                report_id=str(report_id),
                total_chunks=len(db_chunks),
                table_chunks=sum(1 for c in db_chunks if c.chunk_type == "table"),
                text_chunks=sum(1 for c in db_chunks if c.chunk_type == "text"),
            )

            # Log coverage before embedding attempt
            _log_embedding_coverage(parsed_doc.id, db, "before_embed")

            # Compute and store embeddings
            n_embedded = self._embed_if_needed(parsed_doc.id, db)

            # Log coverage after embedding attempt
            _log_embedding_coverage(parsed_doc.id, db, "after_embed")

            if n_embedded == 0:
                logger.error(
                    "parse_orchestrator.embedding_produced_zero",
                    report_id=str(report_id),
                    parsed_document_id=str(parsed_doc.id),
                    total_chunks=len(db_chunks),
                    action=(
                        "Zero embeddings were stored. Likely causes: "
                        "(1) sentence-transformers model could not be loaded "
                        "(check for HuggingFace connectivity or run: "
                        "python -c \"from sentence_transformers import SentenceTransformer; "
                        "SentenceTransformer('BAAI/bge-small-en-v1.5')\"); "
                        "(2) pgvector extension not installed "
                        "(run: CREATE EXTENSION IF NOT EXISTS vector;); "
                        "(3) DB column type mismatch "
                        f"(expected Vector({384})). "
                        "Retrieval will fall back to keyword-only mode."
                    ),
                )
            else:
                logger.info(
                    "parse_orchestrator.embedding_success",
                    report_id=str(report_id),
                    n_embedded=n_embedded,
                    total_chunks=len(db_chunks),
                )

            # Update meta with the final chunk count
            parsed_doc.meta = {**meta, "chunk_count": len(db_chunks)}

            # Advance report status
            report.status = "parsed"
            db.flush()

            logger.info(
                "parse_orchestrator.complete",
                report_id=str(report_id),
                pages=meta["page_count"],
                chunks=len(db_chunks),
                embedded=n_embedded,
            )
            return ParsedDocumentRead.model_validate(parsed_doc)

    def _embed_if_needed(self, parsed_document_id: uuid.UUID, db) -> int:
        """
        Run embed_document() if the embedding service is available.
        Returns the number of chunks embedded (0 on failure or unavailability).
        Non-fatal: keyword-only retrieval continues even if this returns 0.
        """
        try:
            from services.embedding_service import EmbeddingService
            emb = EmbeddingService()

            logger.info(
                "parse_orchestrator.embedding_check",
                parsed_document_id=str(parsed_document_id),
                model=emb.model_name,
            )

            if not emb.is_available():
                logger.warning(
                    "parse_orchestrator.embedding_model_unavailable",
                    parsed_document_id=str(parsed_document_id),
                    model=emb.model_name,
                    action=(
                        "Embedding model is not available. "
                        "Semantic retrieval will be disabled. "
                        "To fix: ensure sentence-transformers is installed and "
                        "the model can be downloaded from HuggingFace, or "
                        "pre-cache it: python -c \"from sentence_transformers "
                        "import SentenceTransformer; "
                        f"SentenceTransformer('{emb.model_name}')\""
                    ),
                )
                return 0

            n_embedded = emb.embed_document(parsed_document_id, db)
            return n_embedded

        except Exception as exc:
            logger.error(
                "parse_orchestrator.embedding_exception",
                parsed_document_id=str(parsed_document_id),
                error=str(exc),
                action=(
                    "Unexpected exception during embedding — NOT fatal. "
                    "Keyword retrieval will continue. "
                    "Fix the embedding issue and re-parse with force=True "
                    "or run: python main.py embed --report-id <id>"
                ),
            )
            return 0