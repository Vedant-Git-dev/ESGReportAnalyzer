"""
agents/parsing_agent.py

Parsing Agent — Phase 2.

Extracts structured content from downloaded PDFs using a 3-layer strategy:
  Layer 1 — PyMuPDF (fitz):  fast text extraction with bounding boxes
  Layer 2 — pdfplumber:       table detection and extraction
  Layer 3 — pytesseract OCR:  fallback for scanned / image-heavy pages

Output: list of RawChunk dicts (type, page, content) passed to ChunkingAgent.

NEVER called directly if parse cache already exists for (report_id, parser_version).
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core.config import get_settings
from core.logging_config import get_logger

logger = get_logger(__name__)

# Minimum non-whitespace characters for a text block to be kept
_MIN_TEXT_LEN = 30
# Minimum rows × cols for a pdfplumber table to be considered valid
_MIN_TABLE_CELLS = 4


@dataclass
class RawChunk:
    """Intermediate unit produced by the parser, consumed by ChunkingAgent."""
    chunk_type: str          # "text" | "table" | "footnote"
    page_number: int
    content: str
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Layer 1 — PyMuPDF text extraction
# ---------------------------------------------------------------------------

def _extract_text_fitz(pdf_path: Path) -> tuple[list[RawChunk], int]:
    """
    Extract text blocks from all pages using fitz (PyMuPDF).
    Returns (chunks, page_count).
    Uses get_text("blocks") for reading-order-aware extraction.
    """
    import fitz  # PyMuPDF

    chunks: list[RawChunk] = []
    doc = fitz.open(str(pdf_path))
    page_count = doc.page_count

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks")  # list of (x0,y0,x1,y1,text,block_no,block_type)
        for block in blocks:
            block_type = block[6]  # 0=text, 1=image
            if block_type != 0:
                continue
            text = block[4].strip()
            if len(text) < _MIN_TEXT_LEN:
                continue

            # Classify footnotes: small blocks near bottom of page with superscript markers
            chunk_type = "text"
            if _is_footnote(text, page, block):
                chunk_type = "footnote"

            chunks.append(RawChunk(
                chunk_type=chunk_type,
                page_number=page_num,
                content=_clean_text(text),
                meta={"bbox": block[:4]},
            ))

    doc.close()
    logger.debug("parsing.fitz_done", pages=page_count, chunks=len(chunks))
    return chunks, page_count


def _is_footnote(text: str, page, block) -> bool:
    """Heuristic: short block in bottom 15% of page starting with digit/symbol."""
    try:
        page_height = page.rect.height
        block_top = block[1]
        near_bottom = block_top > (page_height * 0.85)
        short_block = len(text) < 200
        starts_with_marker = bool(re.match(r"^[\d\*†‡§¹²³]", text))
        return near_bottom and short_block and starts_with_marker
    except Exception:
        return False


def _clean_text(text: str) -> str:
    """Normalise whitespace and remove control characters."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Layer 2 — pdfplumber table extraction
# ---------------------------------------------------------------------------

def _extract_tables_pdfplumber(pdf_path: Path) -> list[RawChunk]:
    """
    Extract tables from all pages using pdfplumber.
    Tables are serialised to pipe-delimited markdown-like format for LLM-friendliness.
    """
    import pdfplumber

    chunks: list[RawChunk] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.extract_tables()
            except Exception as exc:
                logger.warning("parsing.pdfplumber_page_error", page=page_num, error=str(exc))
                continue

            for table_idx, table in enumerate(tables):
                if not table:
                    continue
                # Filter out near-empty tables
                total_cells = sum(1 for row in table for cell in row if cell and str(cell).strip())
                if total_cells < _MIN_TABLE_CELLS:
                    continue

                serialised = _serialise_table(table)
                if not serialised:
                    continue

                chunks.append(RawChunk(
                    chunk_type="table",
                    page_number=page_num,
                    content=serialised,
                    meta={"table_index": table_idx},
                ))

    logger.debug("parsing.pdfplumber_done", tables=len(chunks))
    return chunks


def _serialise_table(table: list[list]) -> str:
    """Convert a pdfplumber table (list of rows) to pipe-delimited text."""
    rows = []
    for row in table:
        cells = [str(cell).strip().replace("\n", " ") if cell is not None else "" for cell in row]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Layer 3 — OCR fallback (pytesseract)
# ---------------------------------------------------------------------------

def _page_needs_ocr(text_chunks: list[RawChunk], page_num: int) -> bool:
    """Return True if a page yielded no text — likely a scanned image."""
    page_chunks = [c for c in text_chunks if c.page_number == page_num]
    return len(page_chunks) == 0


def _ocr_page(pdf_path: Path, page_num: int) -> Optional[RawChunk]:
    """Rasterise a single page and run pytesseract OCR on it."""
    try:
        import fitz
        import pytesseract
        from PIL import Image
        import io

        doc = fitz.open(str(pdf_path))
        page = doc[page_num - 1]
        # Render at 200 DPI for good OCR accuracy
        mat = fitz.Matrix(200 / 72, 200 / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        doc.close()

        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img, lang="eng")
        text = _clean_text(text)

        if len(text) < _MIN_TEXT_LEN:
            return None

        return RawChunk(
            chunk_type="text",
            page_number=page_num,
            content=text,
            meta={"ocr": True},
        )
    except Exception as exc:
        logger.warning("parsing.ocr_failed", page=page_num, error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ParsingAgent:
    """
    Stateless agent. Call parse(pdf_path) to get RawChunks.
    The agent does NOT interact with the DB — that is ParseCacheManager's job.

    Two extraction modes, controlled by settings.use_spatial_chunker:
      True  (default) — SpatialChunker: word-level glyph extraction with
                        column detection. Handles design-heavy PDFs correctly.
      False           — Block-based: fitz blocks + pdfplumber tables. Faster
                        but breaks on multi-column layouts.
    """

    def parse(self, pdf_path: Path) -> tuple[list[RawChunk], dict]:
        settings = get_settings()

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info("parsing.start", path=str(pdf_path),
                    mode="spatial" if settings.use_spatial_chunker else "block")

        if settings.use_spatial_chunker:
            return self._parse_spatial(pdf_path, settings)
        else:
            return self._parse_block(pdf_path, settings)

    def _parse_spatial(self, pdf_path: Path, settings) -> tuple[list[RawChunk], dict]:
        """
        Spatial mode: word-level extraction with column detection.
        Correctly handles multi-column tables and design-heavy layouts.
        """
        from services.spatial_chunker import extract_spatial_chunks

        all_chunks, spatial_meta = extract_spatial_chunks(pdf_path)

        meta = {
            **spatial_meta,
            "text_chunk_count": sum(1 for c in all_chunks if c.chunk_type == "text"),
            "parser_version": settings.parser_version,
            "pdf_path": str(pdf_path),
            "mode": "spatial",
        }

        logger.info(
            "parsing.complete",
            mode="spatial",
            pages=meta["page_count"],
            chunks=len(all_chunks),
            tables=meta["table_count"],
            ocr_pages=meta["ocr_page_count"],
        )
        return all_chunks, meta

    def _parse_block(self, pdf_path: Path, settings) -> tuple[list[RawChunk], dict]:
        """
        Block mode (legacy): fitz blocks + pdfplumber tables + OCR.
        Use when spatial chunker is disabled via USE_SPATIAL_CHUNKER=false.
        """
        logger.info("parsing.start", path=str(pdf_path))

        # Layer 1 — text
        text_chunks, page_count = _extract_text_fitz(pdf_path)

        # Layer 2 — tables
        table_chunks = _extract_tables_pdfplumber(pdf_path)

        # Layer 3 — OCR on blank pages
        ocr_chunks: list[RawChunk] = []
        ocr_page_count = 0
        for page_num in range(1, page_count + 1):
            if _page_needs_ocr(text_chunks, page_num):
                chunk = _ocr_page(pdf_path, page_num)
                if chunk:
                    ocr_chunks.append(chunk)
                    ocr_page_count += 1

        all_chunks = text_chunks + table_chunks + ocr_chunks
        all_chunks.sort(key=lambda c: (c.page_number, 0 if c.chunk_type == "table" else 1))

        word_count = sum(len(c.content.split()) for c in all_chunks)
        meta = {
            "page_count": page_count,
            "word_count": word_count,
            "text_chunk_count": len(text_chunks),
            "table_count": len(table_chunks),
            "ocr_page_count": ocr_page_count,
            "parser_version": settings.parser_version,
            "pdf_path": str(pdf_path),
            "mode": "block",
        }

        logger.info(
            "parsing.complete",
            mode="block",
            pages=page_count,
            text_chunks=len(text_chunks),
            tables=len(table_chunks),
            ocr_pages=ocr_page_count,
        )
        return all_chunks, meta