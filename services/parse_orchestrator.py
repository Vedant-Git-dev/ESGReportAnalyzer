"""
services/parse_orchestrator.py

Phase 2 orchestrator. Coordinates:
  1. ParseCacheManager  — cache hit/miss check
  2. ParsingAgent       — PDF → RawChunks
  3. ChunkingAgent      — RawChunks → DB DocumentChunks
  4. Report status      — updates report.status in DB

Entry point: ParseOrchestrator.run(report_id)
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
        Parse a downloaded PDF and cache results.

        Args:
            report_id: UUID of a Report with status='downloaded'
            force:     If True, re-parse even if cache exists

        Returns:
            ParsedDocumentRead schema (with id, page_count, meta, chunk count in meta)

        Raises:
            ValueError: report not found, wrong status, or file missing
        """
        parser_version = self.settings.parser_version

        with get_db() as db:
            # --- Load report ---
            report = db.query(Report).filter(Report.id == report_id).first()
            if not report:
                raise ValueError(f"Report {report_id} not found")
            if report.status not in ("downloaded", "parsed", "extracted"):
                raise ValueError(
                    f"Report {report_id} has status '{report.status}' — must be 'downloaded' to parse"
                )
            if not report.file_path:
                raise ValueError(f"Report {report_id} has no file_path")

            pdf_path = Path(report.file_path)

            # --- Cache check ---
            if not force and is_cached(report_id, parser_version, db):
                logger.info(
                    "parse_orchestrator.cache_hit",
                    report_id=str(report_id),
                    parser_version=parser_version,
                )
                cached = get_cached_parse(report_id, parser_version, db)
                return ParsedDocumentRead.model_validate(cached)

            # --- Parse ---
            logger.info(
                "parse_orchestrator.parsing",
                report_id=str(report_id),
                path=str(pdf_path),
                parser_version=parser_version,
            )

            from agents.parsing_agent import ParsingAgent
            from agents.chunking_agent import ChunkingAgent

            parsing_agent = ParsingAgent()
            chunking_agent = ChunkingAgent()

            try:
                raw_chunks, meta = parsing_agent.parse(pdf_path)
            except Exception as exc:
                report.status = "failed"
                report.error_message = f"Parsing failed: {exc}"
                db.flush()
                logger.error("parse_orchestrator.parse_failed", error=str(exc))
                raise

            # --- Store parse cache entry ---
            parsed_doc = store_parse(
                report_id=report_id,
                parser_version=parser_version,
                page_count=meta["page_count"],
                meta=meta,
                db=db,
            )

            # --- Chunk and store ---
            db_chunks = chunking_agent.chunk_and_store(raw_chunks, parsed_doc, db)

            # --- Compute embeddings (local model, zero API cost) ---
            try:
                from services.embedding_service import EmbeddingService
                emb_service = EmbeddingService()
                if emb_service.is_available():
                    embedded = emb_service.embed_document(parsed_doc.id, db)
                    logger.info("parse_orchestrator.embedded", chunks=embedded)
                else:
                    logger.warning("parse_orchestrator.embedding_skipped",
                                  reason="model not available")
            except Exception as exc:
                logger.warning("parse_orchestrator.embedding_failed", error=str(exc))

            # Update meta with final chunk count
            parsed_doc.meta = {**meta, "chunk_count": len(db_chunks)}

            # --- Update report status ---
            report.status = "parsed"
            db.flush()

            logger.info(
                "parse_orchestrator.complete",
                report_id=str(report_id),
                pages=meta["page_count"],
                chunks=len(db_chunks),
            )

            return ParsedDocumentRead.model_validate(parsed_doc)