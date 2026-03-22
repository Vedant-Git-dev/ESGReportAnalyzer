"""
services/parse_cache.py

Parse Cache Manager.

Key insight: parsing is expensive (fitz + pdfplumber + optional OCR).
We parse ONCE per (report_id, parser_version) and store structured chunks in DB.
When parser_version is bumped in config, the cache miss triggers a fresh parse.

Public API
----------
get_cached_parse(report_id, parser_version, db) -> ParsedDocument | None
store_parse(report_id, parser_version, pages, meta, db) -> ParsedDocument
is_cached(report_id, parser_version, db) -> bool
"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from core.logging_config import get_logger
from models.db_models import ParsedDocument

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


def is_cached(report_id: uuid.UUID, parser_version: str, db: Session) -> bool:
    """Return True if a valid parse cache entry exists for this report + version."""
    exists = (
        db.query(ParsedDocument.id)
        .filter(
            ParsedDocument.report_id == report_id,
            ParsedDocument.parser_version == parser_version,
        )
        .first()
    )
    return exists is not None


def get_cached_parse(
    report_id: uuid.UUID,
    parser_version: str,
    db: Session,
) -> ParsedDocument | None:
    """
    Fetch cached ParsedDocument (with chunks eagerly loaded).
    Returns None on cache miss.
    """
    result = (
        db.query(ParsedDocument)
        .filter(
            ParsedDocument.report_id == report_id,
            ParsedDocument.parser_version == parser_version,
        )
        .first()
    )
    if result:
        logger.info(
            "parse_cache.hit",
            report_id=str(report_id),
            parser_version=parser_version,
            chunks=len(result.chunks),
        )
    else:
        logger.info(
            "parse_cache.miss",
            report_id=str(report_id),
            parser_version=parser_version,
        )
    return result


def store_parse(
    report_id: uuid.UUID,
    parser_version: str,
    page_count: int,
    meta: dict,
    db: Session,
) -> ParsedDocument:
    """
    Create a new ParsedDocument cache entry.
    Caller is responsible for adding DocumentChunks afterwards.
    Deletes any existing entry for the same (report_id, parser_version) first
    to avoid constraint violations on re-runs.
    """
    # Clean up stale entry if it exists (e.g. aborted previous parse)
    existing = (
        db.query(ParsedDocument)
        .filter(
            ParsedDocument.report_id == report_id,
            ParsedDocument.parser_version == parser_version,
        )
        .first()
    )
    if existing:
        db.delete(existing)
        db.flush()
        logger.info("parse_cache.stale_entry_deleted", report_id=str(report_id))

    parsed_doc = ParsedDocument(
        report_id=report_id,
        parser_version=parser_version,
        page_count=page_count,
        meta=meta,
    )
    db.add(parsed_doc)
    db.flush()

    logger.info(
        "parse_cache.stored",
        report_id=str(report_id),
        parser_version=parser_version,
        pages=page_count,
    )
    return parsed_doc