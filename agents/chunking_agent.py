"""
agents/chunking_agent.py

Chunking Agent — Phase 2 (revised).

Key improvements over v1:
  1. Sentence-boundary splitting — never cuts mid-sentence
  2. Overlap stitching — each chunk includes 1 trailing sentence from
     the previous chunk, so values split across chunk boundaries are captured
  3. Numeric-density flag — chunks with numbers get a keyword tag "has_numbers"
     so retrieval can boost them
  4. Tables never split (unchanged)
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
_OVERLAP_SENTENCES = 1        # sentences to carry over between chunks
_NUMBER_RE = re.compile(r"\b\d[\d,\.]*\b")

_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "will", "would", "could", "should", "may", "might",
    "that", "this", "these", "those", "it", "its", "as", "not", "no", "our",
    "we", "their", "they", "he", "she", "his", "her", "which", "who", "also",
}

# ESG-domain terms always kept in keyword index even if short
_ESG_TERMS = {
    "ghg", "co2", "nox", "sox", "esg", "kl", "mwh", "gwh", "gj", "tj",
    "fte", "csr", "epi", "re", "pv",
}


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
    # Tag chunks that contain numbers — retrieval boosts these
    if _NUMBER_RE.search(text):
        keywords.add("has_numbers")
    return " ".join(sorted(keywords))


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences on . ! ? followed by whitespace + capital.
    Also splits on newlines that look like list items or table rows.
    """
    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Split on sentence-ending punctuation
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z\d])", text)

    sentences: list[str] = []
    for part in raw:
        # Further split on newlines that start new logical lines
        lines = part.split("\n")
        for line in lines:
            line = line.strip()
            if line:
                sentences.append(line)

    return [s for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Overlapping chunk builder
# ---------------------------------------------------------------------------

def _split_with_overlap(
    text: str,
    max_tokens: int,
    min_tokens: int,
    overlap: int = _OVERLAP_SENTENCES,
) -> list[str]:
    """
    Split text into chunks bounded by [min_tokens, max_tokens].
    Each chunk includes `overlap` trailing sentences from the previous chunk
    so values split across a boundary are still captured in context.
    """
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
            # Save current chunk
            chunks.append(" ".join(current))
            # Carry over last N sentences as overlap into the next chunk
            current = current[-overlap:] if overlap else []
            current_words = sum(len(s.split()) for s in current)

        current.append(sent)
        current_words += sent_words

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]


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