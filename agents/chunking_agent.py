"""
agents/chunking_agent.py

Chunking Agent — Phase 2 (revised v2).

Changes vs v1:
  1. is_table_chunk()  — detects pipe-delimited / high-numeric-density blocks
  2. _split_with_overlap() — skips splitting when content is table-like
  3. _merge_label_value_chunks() — merges adjacent chunks where one holds a
     KPI keyword and the next holds the corresponding number, so label+value
     never land in separate retrieval units
  4. Sentence-boundary splitting, overlap stitching, numeric-density tagging
     unchanged from v1
"""
from __future__ import annotations

import re
import uuid

from sqlalchemy.orm import Session

from agents.parsing_agent import RawChunk
from core.config import get_settings
from core.logging_config import get_logger
from models.db_models import DocumentChunk, ParsedDocument

logger = get_logger(__name__)

_WORDS_PER_TOKEN = 0.75
_OVERLAP_SENTENCES = 1
_NUMBER_RE = re.compile(r"\b\d[\d,\.]*\b")

_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "will", "would", "could", "should", "may", "might",
    "that", "this", "these", "those", "it", "its", "as", "not", "no", "our",
    "we", "their", "they", "he", "she", "his", "her", "which", "who", "also",
}

_ESG_TERMS = {
    "ghg", "co2", "nox", "sox", "esg", "kl", "mwh", "gwh", "gj", "tj",
    "fte", "csr", "epi", "re", "pv",
}

# KPI keywords that trigger label+value merge
_KPI_LABEL_KEYWORDS: frozenset[str] = frozenset([
    "scope 1", "scope 2", "scope 3", "total scope",
    "total energy", "energy consumed", "energy consumption",
    "water consumption", "water consumed",
    "waste generated", "total waste",
    "ghg emission", "greenhouse gas", "carbon",
    "tco2e", "tco2", "co2e",
    "employee", "headcount", "workforce",
    "csr expenditure", "csr spend",
])

# Patterns that suggest numeric data rows in a table
_PIPE_ROW_RE = re.compile(r"\|")
_MULTI_NUM_RE = re.compile(r"(\b[\d,]+(?:\.\d+)?\b[\s|,]{0,10}){2,}")
_UNIT_SUFFIX_RE = re.compile(
    r"\b(tco2e?|t\s*co2e?|mwh|gwh|gj|tj|kl|mt|metric\s*tonn?\w*|%|crore|lakh)\b",
    re.IGNORECASE,
)

# A "real" data number: 3+ digits, or comma-formatted (21,949).
# Excludes single/double digits like the "1" in "Scope 1" or "2" in "Scope 2".
_DATA_NUMBER_RE = re.compile(r"\b\d[\d,]*\d{2,}\b|\b\d{3,}\b")


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) / _WORDS_PER_TOKEN))


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

def _extract_keywords(text: str) -> str:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    keywords = {
        t for t in tokens
        if (len(t) > 2 and t not in _STOP_WORDS) or t in _ESG_TERMS
    }
    if _NUMBER_RE.search(text):
        keywords.add("has_numbers")
    return " ".join(sorted(keywords))


# ---------------------------------------------------------------------------
# is_table_chunk  (NEW)
# ---------------------------------------------------------------------------

def is_table_chunk(text: str) -> bool:
    """
    Return True when *text* looks like a table or numeric data block.

    Heuristics (any one is sufficient):
      1. Pipe separators — pdfplumber serialises tables with '|'
      2. High numeric density — >30% of whitespace-split tokens are numbers
      3. Multiple numbers on multiple lines — repeated number patterns
      4. Contains a recognised ESG unit alongside at least two numbers

    Kept cheap (no regex compilation at call time — all patterns are
    module-level constants).
    """
    if not text or len(text.strip()) < 10:
        return False

    # 1. Pipe separators — strong signal
    pipe_count = text.count("|")
    if pipe_count >= 2:
        return True

    tokens = text.split()
    if not tokens:
        return False

    # 2. Numeric density > 30 %
    num_tokens = sum(1 for t in tokens if _NUMBER_RE.fullmatch(t.strip("(),.")))
    if len(tokens) >= 4 and num_tokens / len(tokens) > 0.30:
        return True

    # 3. Two or more numeric values on the same / adjacent lines
    if _MULTI_NUM_RE.search(text):
        return True

    # 4. ESG unit + at least two numbers  → data row
    if _UNIT_SUFFIX_RE.search(text):
        all_nums = _NUMBER_RE.findall(text)
        if len(all_nums) >= 2:
            return True

    return False


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

def _split_into_sentences(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z\d])", text)
    sentences: list[str] = []
    for part in raw:
        for line in part.split("\n"):
            line = line.strip()
            if line:
                sentences.append(line)
    return [s for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Overlapping chunk builder  (MODIFIED — table protection)
# ---------------------------------------------------------------------------

def _split_with_overlap(
    text: str,
    max_tokens: int,
    min_tokens: int,
    overlap: int = _OVERLAP_SENTENCES,
) -> list[str]:
    """
    Split text into token-bounded chunks with sentence-level overlap.

    MODIFICATION: If *text* is detected as a table/numeric block via
    is_table_chunk(), the whole block is returned as a single chunk rather
    than being split, preserving label↔value adjacency within tables.
    """
    # Table protection: never split table-like content
    if is_table_chunk(text):
        stripped = text.strip()
        if stripped:
            return [stripped]
        return []

    max_words = int(max_tokens * _WORDS_PER_TOKEN)
    min_words = int(min_tokens * _WORDS_PER_TOKEN)

    sentences = _split_into_sentences(text)
    if not sentences:
        return [text] if text.strip() else []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sent in sentences:
        sent_words = len(sent.split())
        if current_words + sent_words > max_words and current_words >= min_words:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
            current_words = sum(len(s.split()) for s in current)
        current.append(sent)
        current_words += sent_words

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Label+value chunk merge  (NEW)
# ---------------------------------------------------------------------------

def _merge_label_value_chunks(
    chunks: list[DocumentChunk],
    max_merged_tokens: int,
) -> list[DocumentChunk]:
    """
    Scan consecutive chunk pairs on the same page.  When chunk[i] contains
    a KPI label keyword but no numeric value, and chunk[i+1] on the same
    page has numeric values but no KPI keyword, merge them into chunk[i].

    This prevents the common BRSR/ESG PDF split pattern:
        chunk A: "Total Scope 1 emissions (Metric tonnes CO2e)"   ← label only
        chunk B: "21,949   20,972"                                 ← values only

    After merging chunk B's content is appended to chunk A and chunk B is
    removed from the list.

    Returns a new list; the DB objects in the returned list are mutated
    in-place (content, token_count, keywords updated) but not yet flushed.
    """
    if len(chunks) < 2:
        return chunks

    result: list[DocumentChunk] = []
    skip: set[int] = set()

    for i, chunk in enumerate(chunks):
        if i in skip:
            continue

        text_i = chunk.content.lower()
        has_kpi_label = any(kw in text_i for kw in _KPI_LABEL_KEYWORDS)
        # Use _DATA_NUMBER_RE so single digits in "Scope 1" / "Scope 2" do not
        # count as a data value — we only want to suppress merge when a real
        # numeric value (3+ digits) is already present in the label chunk.
        has_number_i = bool(_DATA_NUMBER_RE.search(chunk.content))

        if has_kpi_label and not has_number_i and i + 1 < len(chunks):
            nxt = chunks[i + 1]
            same_page = (
                chunk.page_number is None
                or nxt.page_number is None
                or chunk.page_number == nxt.page_number
            )
            has_number_nxt = bool(_DATA_NUMBER_RE.search(nxt.content))
            merged_tokens = (chunk.token_count or 0) + (nxt.token_count or 0)

            if same_page and has_number_nxt and merged_tokens <= max_merged_tokens:
                merged_content = chunk.content.rstrip() + "\n" + nxt.content.lstrip()
                chunk.content = merged_content
                chunk.token_count = _estimate_tokens(merged_content)
                chunk.keywords = _extract_keywords(merged_content)
                skip.add(i + 1)
                logger.debug(
                    "chunking.label_value_merged",
                    page=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    merged_from=nxt.chunk_index,
                )

        result.append(chunk)

    return result


# ---------------------------------------------------------------------------
# Public Agent
# ---------------------------------------------------------------------------

class ChunkingAgent:

    def __init__(self) -> None:
        self.settings = get_settings()

    def chunk_and_store(
        self,
        raw_chunks: list[RawChunk],
        parsed_document: ParsedDocument,
        db: Session,
    ) -> list[DocumentChunk]:
        max_t = self.settings.max_chunk_tokens
        min_t = self.settings.min_chunk_tokens

        db_chunks: list[DocumentChunk] = []
        chunk_index = 0

        for raw in raw_chunks:
            if raw.chunk_type == "table":
                # Tables: never split, store as-is
                chunk = self._make_chunk(
                    parsed_document_id=parsed_document.id,
                    chunk_index=chunk_index,
                    chunk_type="table",
                    page_number=raw.page_number,
                    content=raw.content,
                )
                db.add(chunk)
                db_chunks.append(chunk)
                chunk_index += 1

            else:
                tokens = _estimate_tokens(raw.content)

                # If this text block is table-like keep it whole regardless
                # of the token budget — we must not split numeric rows
                if is_table_chunk(raw.content):
                    if raw.content.strip():
                        chunk = self._make_chunk(
                            parsed_document_id=parsed_document.id,
                            chunk_index=chunk_index,
                            chunk_type="table",   # promote to table type
                            page_number=raw.page_number,
                            content=raw.content,
                        )
                        db.add(chunk)
                        db_chunks.append(chunk)
                        chunk_index += 1
                    continue

                if tokens <= max_t:
                    segments = [raw.content]
                else:
                    segments = _split_with_overlap(raw.content, max_t, min_t)

                for seg in segments:
                    if not seg.strip():
                        continue
                    chunk = self._make_chunk(
                        parsed_document_id=parsed_document.id,
                        chunk_index=chunk_index,
                        chunk_type=raw.chunk_type,
                        page_number=raw.page_number,
                        content=seg,
                    )
                    db.add(chunk)
                    db_chunks.append(chunk)
                    chunk_index += 1

        # Flush to get IDs before the merge step
        db.flush()

        # Merge label-only chunks with adjacent value chunks
        max_merged = int(max_t * 1.5)   # allow slightly larger merged chunks
        db_chunks = _merge_label_value_chunks(db_chunks, max_merged)

        # Re-flush to persist merged content
        db.flush()

        logger.info(
            "chunking.complete",
            parsed_document_id=str(parsed_document.id),
            total_chunks=len(db_chunks),
            table_chunks=sum(1 for c in db_chunks if c.chunk_type == "table"),
            text_chunks=sum(1 for c in db_chunks if c.chunk_type == "text"),
        )
        return db_chunks

    def _make_chunk(
        self,
        parsed_document_id: uuid.UUID,
        chunk_index: int,
        chunk_type: str,
        page_number: int,
        content: str,
    ) -> DocumentChunk:
        return DocumentChunk(
            parsed_document_id=parsed_document_id,
            chunk_index=chunk_index,
            chunk_type=chunk_type,
            page_number=page_number,
            content=content,
            token_count=_estimate_tokens(content),
            keywords=_extract_keywords(content),
        )