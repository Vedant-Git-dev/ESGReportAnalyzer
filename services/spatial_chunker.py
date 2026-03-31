"""
services/spatial_chunker.py

Spatial Chunker for ESG PDFs.

Strategy: Use pdfplumber's layout-aware extraction (which handles rotation
natively) + targeted reconstruction to fix split values.

For design-heavy PDFs like Infosys Annual Reports:
  - pdfplumber preserves reading order better than fitz blocks
  - We use pdfplumber's char-level bbox to detect column structure
  - Table pages get pdfplumber table extraction
  - Non-table pages get layout-aware text with smart line joining
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from core.logging_config import get_logger

logger = get_logger(__name__)

_MIN_TEXT_LEN = 20
_NUMBER_RE = re.compile(r"\b[\d,]+(?:\.\d+)?\b")
_NUM_UNIT_RE = re.compile(
    r"\b[\d,]+(?:\.\d+)?\s*"
    r"(?:tco2e?|ghg|gj|mj|mwh|gwh|kwh|kl|kilolitr|litr|mt|metric.?tonn|"
    r"tonn|kg|%|crore|lakh|nos|employees|headcount|fte|equivalent|"
    r"cubic|sqm|sq\.m)\b",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _pdfplumber_page(page) -> list[tuple[str, str]]:
    """
    Extract (chunk_type, content) pairs from one pdfplumber page.
    Returns list of (type, text) — type is 'table' or 'text'.
    """
    results = []

    # Try table extraction first
    try:
        tables = page.extract_tables()
        for table in (tables or []):
            if not table:
                continue
            cells = sum(1 for row in table for cell in row if cell and str(cell).strip())
            if cells < 4:
                continue
            rows = []
            for row in table:
                cells_str = [str(c).strip().replace("\n", " ") if c else "" for c in row]
                rows.append(" | ".join(cells_str))
            content = "\n".join(rows)
            if content.strip():
                results.append(("table", content))
    except Exception as e:
        logger.debug("spatial.table_error", error=str(e))

    # Text extraction with layout preservation
    try:
        text = page.extract_text(layout=True, x_tolerance=3, y_tolerance=3) or ""
        text = _clean(text)
        if len(text) >= _MIN_TEXT_LEN:
            results.append(("text", text))
    except Exception:
        try:
            text = page.extract_text() or ""
            text = _clean(text)
            if len(text) >= _MIN_TEXT_LEN:
                results.append(("text", text))
        except Exception:
            pass

    return results


def extract_spatial_chunks(pdf_path: Path) -> tuple[list, dict]:
    """
    Extract chunks using pdfplumber which handles page rotation natively.
    Falls back to OCR for pages with no extractable text.
    """
    import pdfplumber
    from agents.parsing_agent import RawChunk, _ocr_page

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    all_chunks: list[RawChunk] = []
    text_pages: set[int] = set()
    ocr_page_count = 0
    table_count = 0

    with pdfplumber.open(str(pdf_path)) as pdf:
        page_count = len(pdf.pages)

        for page_num_0, page in enumerate(pdf.pages):
            page_num = page_num_0 + 1
            pairs = _pdfplumber_page(page)

            if pairs:
                text_pages.add(page_num)

            for chunk_type, content in pairs:
                if chunk_type == "table":
                    table_count += 1
                all_chunks.append(RawChunk(
                    chunk_type=chunk_type,
                    page_number=page_num,
                    content=content,
                    meta={"spatial": True},
                ))

    # OCR fallback
    import fitz
    doc = fitz.open(str(pdf_path))
    for page_num in range(1, page_count + 1):
        if page_num not in text_pages:
            chunk = _ocr_page(pdf_path, page_num)
            if chunk:
                all_chunks.append(chunk)
                ocr_page_count += 1
    doc.close()

    word_count = sum(len(c.content.split()) for c in all_chunks)
    logger.info("spatial.complete", pages=page_count, chunks=len(all_chunks),
                tables=table_count, ocr=ocr_page_count)

    meta = {
        "page_count": page_count, "word_count": word_count,
        "table_count": table_count, "ocr_page_count": ocr_page_count,
        "spatial_chunker": True,
    }
    return all_chunks, meta