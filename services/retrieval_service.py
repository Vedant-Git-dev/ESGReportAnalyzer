"""
services/retrieval_service.py

Retrieval Service — Phase 3.

Given a KPI definition and a parsed document, returns the top-K most relevant
chunks for LLM extraction. Never returns the full document.

Retrieval pipeline
------------------
Step 1 — Keyword filter
    Match chunk.keywords against KPI retrieval_keywords.
    Tables get a 2x priority boost over plain text.

Step 2 — TF-IDF style scoring
    Score = sum of keyword hit weights, normalised by chunk token count.
    Tables that contain numeric values score higher.

Step 3 — Page-proximity bonus
    Chunks on the same page as a high-scoring chunk get a small bonus
    (ESG tables and their captions tend to be on the same page).

Step 4 — Rank and return top-K
    Hard limit: never exceed settings.retrieval_top_k (default 7).

Future: Step 2 can be replaced with pgvector cosine similarity once
embeddings are generated. The interface stays identical.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy.orm import Session

from core.config import get_settings
from core.logging_config import get_logger
from models.db_models import DocumentChunk, KPIDefinition, ParsedDocument

logger = get_logger(__name__)

# Weights
_TABLE_BOOST = 2.0        # tables are 2x more likely to contain KPI values
_NUMERIC_BOOST = 0.3      # extra score if chunk contains a number
_PAGE_PROXIMITY_BONUS = 0.1
_FOOTNOTE_PENALTY = 0.5   # footnotes are less likely to have primary values


@dataclass
class ScoredChunk:
    chunk: DocumentChunk
    score: float
    matched_keywords: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> set[str]:
    """Lowercase word tokens from text."""
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def _has_numeric(text: str) -> bool:
    return bool(re.search(r"\d", text))


def _score_chunk(
    chunk: DocumentChunk,
    query_keywords: list[str],
) -> ScoredChunk:
    """
    Compute relevance score for a single chunk against a keyword list.
    Returns ScoredChunk with score=0 if no keywords match.
    """
    if not chunk.keywords:
        return ScoredChunk(chunk=chunk, score=0.0)

    chunk_kw_set = set(chunk.keywords.split())
    matched: list[str] = []

    raw_score = 0.0
    for kw in query_keywords:
        kw_lower = kw.lower()
        # Exact token match
        if kw_lower in chunk_kw_set:
            matched.append(kw)
            raw_score += 1.0
        else:
            # Partial match — keyword is a substring of a chunk token
            for ck in chunk_kw_set:
                if kw_lower in ck or ck in kw_lower:
                    matched.append(kw)
                    raw_score += 0.5
                    break

    if raw_score == 0:
        return ScoredChunk(chunk=chunk, score=0.0)

    # Normalise by token count (prefer dense matches)
    token_count = max(chunk.token_count or 1, 1)
    score = raw_score / (token_count ** 0.3)  # soft normalisation

    # Type boosts
    if chunk.chunk_type == "table":
        score *= _TABLE_BOOST
    elif chunk.chunk_type == "footnote":
        score *= _FOOTNOTE_PENALTY

    # Numeric boost
    if _has_numeric(chunk.content):
        score += _NUMERIC_BOOST

    return ScoredChunk(chunk=chunk, score=score, matched_keywords=list(set(matched)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RetrievalService:
    """
    Stateless retrieval service.
    Instantiate once and call retrieve() per KPI per report.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def retrieve(
        self,
        parsed_document_id: uuid.UUID,
        kpi: KPIDefinition,
        db: Session,
        top_k: Optional[int] = None,
    ) -> list[ScoredChunk]:
        """
        Retrieve the top-K most relevant chunks for a KPI from a parsed document.

        Args:
            parsed_document_id: UUID of the ParsedDocument to search in
            kpi:                KPIDefinition ORM object with retrieval_keywords
            db:                 Active SQLAlchemy session
            top_k:              Override settings.retrieval_top_k if provided

        Returns:
            List of ScoredChunk sorted by score descending, max top_k items.
            Empty list if no relevant chunks found.
        """
        k = top_k or self.settings.retrieval_top_k
        query_keywords: list[str] = kpi.retrieval_keywords or []

        if not query_keywords:
            logger.warning("retrieval.no_keywords", kpi=kpi.name)
            return []

        # --- Step 1: Keyword pre-filter via DB ILIKE ---
        # Build a broad OR filter to avoid loading all chunks into memory
        from sqlalchemy import or_
        keyword_filters = [
            DocumentChunk.keywords.ilike(f"%{kw.lower()}%")
            for kw in query_keywords
        ]

        candidate_chunks = (
            db.query(DocumentChunk)
            .filter(
                DocumentChunk.parsed_document_id == parsed_document_id,
                or_(*keyword_filters),
            )
            .all()
        )

        logger.debug(
            "retrieval.candidates",
            kpi=kpi.name,
            candidates=len(candidate_chunks),
        )

        if not candidate_chunks:
            # Fallback: return top chunks by position (first N pages often have summaries)
            logger.info("retrieval.no_keyword_match_fallback", kpi=kpi.name)
            fallback = (
                db.query(DocumentChunk)
                .filter(DocumentChunk.parsed_document_id == parsed_document_id)
                .order_by(DocumentChunk.chunk_index)
                .limit(k)
                .all()
            )
            return [ScoredChunk(chunk=c, score=0.0) for c in fallback]

        # --- Step 2: Score all candidates ---
        scored = [_score_chunk(c, query_keywords) for c in candidate_chunks]
        scored = [s for s in scored if s.score > 0]

        # --- Step 3: Page-proximity bonus ---
        # Find pages of top-3 scorers and boost neighbours
        scored.sort(key=lambda s: s.score, reverse=True)
        top_pages: set[int] = {
            s.chunk.page_number
            for s in scored[:3]
            if s.chunk.page_number is not None
        }
        for s in scored[3:]:
            if s.chunk.page_number in top_pages:
                s.score += _PAGE_PROXIMITY_BONUS

        # --- Step 4: Final sort and top-K ---
        scored.sort(key=lambda s: s.score, reverse=True)
        result = scored[:k]

        logger.info(
            "retrieval.complete",
            kpi=kpi.name,
            candidates=len(candidate_chunks),
            scored=len(scored),
            returned=len(result),
            top_score=round(result[0].score, 3) if result else 0,
        )

        return result

    def retrieve_by_keywords(
        self,
        parsed_document_id: uuid.UUID,
        keywords: list[str],
        db: Session,
        top_k: Optional[int] = None,
        chunk_types: Optional[list[str]] = None,
    ) -> list[ScoredChunk]:
        """
        Ad-hoc retrieval by keyword list without a KPIDefinition.
        Useful for exploratory queries and testing.

        Args:
            parsed_document_id: UUID of ParsedDocument
            keywords:           List of search keywords
            db:                 Active SQLAlchemy session
            top_k:              Max results (default from settings)
            chunk_types:        Filter to specific types e.g. ["table"]
        """
        k = top_k or self.settings.retrieval_top_k

        from sqlalchemy import or_
        keyword_filters = [
            DocumentChunk.keywords.ilike(f"%{kw.lower()}%")
            for kw in keywords
        ]

        query = db.query(DocumentChunk).filter(
            DocumentChunk.parsed_document_id == parsed_document_id,
            or_(*keyword_filters),
        )

        if chunk_types:
            query = query.filter(DocumentChunk.chunk_type.in_(chunk_types))

        candidates = query.all()

        # Build a temporary KPI-like object for scoring
        class _FakeKPI:
            name = "adhoc"
            retrieval_keywords = keywords

        scored = [_score_chunk(c, keywords) for c in candidates]
        scored = [s for s in scored if s.score > 0]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]