"""
agents/chunking_agent.py

Chunking Agent — Phase 3 (part of Phase 2 pipeline).

Responsibilities:
  1. Receive RawChunks from ParsingAgent
  2. Split oversized text chunks into token-bounded segments (200–500 tokens)
  3. Keep table chunks intact (never split tables)
  4. Build a lowercase keyword index per chunk for retrieval
  5. Persist DocumentChunk rows under a ParsedDocument

Token counting uses a simple word-based approximation (1 token ≈ 0.75 words)
to avoid a hard dependency on tiktoken in Phase 2. Can be swapped later.
"""
from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from agents.parsing_agent import RawChunk
from core.config import get_settings
from core.logging_config import get_logger
from models.db_models import DocumentChunk, ParsedDocument

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Approximate: 1 token ≈ 0.75 words  →  word_count = token_count * 0.75
_WORDS_PER_TOKEN = 0.75

# Stop-words excluded from keyword index
_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "will", "would", "could", "should", "may", "might",
    "that", "this", "these", "those", "it", "its", "as", "not", "no", "our",
    "we", "their", "they", "he", "she", "his", "her", "which", "who", "also",
}


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    words = len(text.split())
    return max(1, int(words / _WORDS_PER_TOKEN))


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

def _extract_keywords(text: str) -> str:
    """
    Return space-separated lowercase keywords for fast ILIKE retrieval.
    Filters stop-words and very short tokens.
    """
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    keywords = {t for t in tokens if len(t) > 2 and t not in _STOP_WORDS}
    return " ".join(sorted(keywords))


# ---------------------------------------------------------------------------
# Text splitter
# ---------------------------------------------------------------------------

def _split_text(text: str, max_tokens: int, min_tokens: int) -> list[str]:
    """
    Split text into segments within [min_tokens, max_tokens] token range.
    Splits on paragraph boundaries first, then sentence boundaries.
    Tries to keep semantic units together.
    """
    max_words = int(max_tokens * _WORDS_PER_TOKEN)
    min_words = int(min_tokens * _WORDS_PER_TOKEN)

    # Split on double newlines (paragraphs)
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    segments: list[str] = []
    current: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        if para_words > max_words:
            # Para itself is too long — split by sentences
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                sent_words = len(sent.split())
                if current_words + sent_words > max_words and current_words >= min_words:
                    segments.append(" ".join(current))
                    current = [sent]
                    current_words = sent_words
                else:
                    current.append(sent)
                    current_words += sent_words
        else:
            if current_words + para_words > max_words and current_words >= min_words:
                segments.append("\n\n".join(current))
                current = [para]
                current_words = para_words
            else:
                current.append(para)
                current_words += para_words

    if current:
        segments.append("\n\n".join(current))

    # Filter truly empty segments
    return [s for s in segments if s.strip()]


# ---------------------------------------------------------------------------
# Public Agent
# ---------------------------------------------------------------------------

class ChunkingAgent:
    """
    Converts RawChunks → DocumentChunk ORM rows stored in DB.
    Tables are never split. Text is split to [min_chunk_tokens, max_chunk_tokens].
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def chunk_and_store(
        self,
        raw_chunks: list[RawChunk],
        parsed_document: ParsedDocument,
        db: Session,
    ) -> list[DocumentChunk]:
        """
        Process all RawChunks for a ParsedDocument, persist to DB, return list.

        Args:
            raw_chunks:       Output of ParsingAgent.parse()
            parsed_document:  Already-flushed ParsedDocument ORM object
            db:               Active SQLAlchemy session (caller commits)
        """
        max_t = self.settings.max_chunk_tokens
        min_t = self.settings.min_chunk_tokens

        db_chunks: list[DocumentChunk] = []
        chunk_index = 0

        for raw in raw_chunks:
            if raw.chunk_type == "table":
                # Tables are stored as single chunks — never split
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
                # Text / footnote — split if over max_tokens
                tokens = _estimate_tokens(raw.content)
                if tokens <= max_t:
                    segments = [raw.content]
                else:
                    segments = _split_text(raw.content, max_t, min_t)

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
            footnote_chunks=sum(1 for c in db_chunks if c.chunk_type == "footnote"),
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