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

No content-level (SHA-256) deduplication is performed here. That was
reverted per the design decision to keep parse dedup simple and keyed
only on (report_id, parser_version).

Entry point: ParseOrchestrator.run(report_id, force=False)
"""
from __future__ import annotations

import uuid
from pathlib import Path

from core.config import get_settings
from core.database import get_db
from core.logging_config import get_logger
from models.db_models import Report
from models.schemas import ParsedDocumentRead
from services.parse_cache import get_cached_parse, is_cached, store_parse

logger = get_logger(__name__)


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
            # store_parse() clears any stale entry for this (report_id, parser_version)
            # before inserting the new one.
            parsed_doc = store_parse(
                report_id=report_id,
                parser_version=parser_version,
                page_count=meta["page_count"],
                meta=meta,
                db=db,
            )

            # Chunk and store DocumentChunks
            db_chunks = ChunkingAgent().chunk_and_store(raw_chunks, parsed_doc, db)

            # Compute and store embeddings using the local sentence-transformers
            # model. This is zero API cost. Falls back gracefully if the model
            # is not installed.
            try:
                from services.embedding_service import EmbeddingService
                emb = EmbeddingService()
                if emb.is_available():
                    n_embedded = emb.embed_document(parsed_doc.id, db)
                    logger.info(
                        "parse_orchestrator.embedded",
                        chunks_embedded=n_embedded,
                    )
                else:
                    logger.warning(
                        "parse_orchestrator.embedding_skipped",
                        reason="sentence-transformers model not available",
                    )
            except Exception as exc:
                # Embedding failure is non-fatal. Keyword-only retrieval
                # still works without embeddings.
                logger.warning(
                    "parse_orchestrator.embedding_failed",
                    error=str(exc),
                )

            # Update meta with the final chunk count (known only after chunking)
            parsed_doc.meta = {**meta, "chunk_count": len(db_chunks)}

            # Advance report status
            report.status = "parsed"
            db.flush()

            logger.info(
                "parse_orchestrator.complete",
                report_id=str(report_id),
                pages=meta["page_count"],
                chunks=len(db_chunks),
            )
            return ParsedDocumentRead.model_validate(parsed_doc)