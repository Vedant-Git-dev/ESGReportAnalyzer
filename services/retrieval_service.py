"""
services/retrieval_service.py

Retrieval Service — Phase 3 (revised).

Key improvements:
  1. Numeric-density scoring — chunks with numbers rank much higher
  2. Data-sentence detection — chunks containing "X unit" patterns get a
     large boost (these are the actual KPI value sentences)
  3. Adjacent chunk stitching — retrieval returns a window of ±1 chunk
     around each top match, so split values are always in context
  4. Dedup by chunk_index after stitching
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

# Scoring weights
_TABLE_BOOST        = 2.5
_FOOTNOTE_PENALTY   = 0.4
_NUMERIC_BOOST      = 0.5      # chunk contains any number
_DATA_SENTENCE_BOOST = 1.5    # chunk matches "number unit" pattern
_PAGE_PROXIMITY_BONUS = 0.15
_NEIGHBOR_SCORE     = 0.6     # score assigned to stitched neighbor chunks

# Detects "number unit" patterns — strong signal this chunk has actual data
_DATA_RE = re.compile(
    r"\b[\d,]+(?:\.\d+)?\s*"
    r"(?:tco2e?|t\s*co2e?|mwh|gwh|gj|tj|kl|kilolitr\w*|mt|tonne\w*|%|percent|crore|lakh|employees|headcount)\b",
    re.IGNORECASE,
)


@dataclass
class ScoredChunk:
    chunk: DocumentChunk
    score: float
    matched_keywords: list[str] = field(default_factory=list)
    is_neighbor: bool = False    # True if stitched in as adjacent context


def _has_data_sentence(text: str) -> bool:
    return bool(_DATA_RE.search(text))


def _score_chunk(chunk: DocumentChunk, query_keywords: list[str]) -> ScoredChunk:
    if not chunk.keywords:
        return ScoredChunk(chunk=chunk, score=0)

    chunk_kw_set = set(chunk.keywords.split())
    matched: list[str] = []
    raw_score = 0

    for kw in query_keywords:
        kw_parts = kw.lower().split()   # handle multi-word keywords
        for part in kw_parts:
            if part in chunk_kw_set:
                matched.append(kw)
                raw_score += 1
                break
            else:
                for ck in chunk_kw_set:
                    if part in ck or ck in part:
                        matched.append(kw)
                        raw_score += 0.4
                        break

    if raw_score == 0:
        return ScoredChunk(chunk=chunk, score=0)

    # Soft normalise — prefer dense keyword matches
    token_count = max(chunk.token_count or 1, 1)
    score = raw_score / (token_count ** 0.25)

    # Type boosts
    if chunk.chunk_type == "table":
        score *= _TABLE_BOOST
    elif chunk.chunk_type == "footnote":
        score *= _FOOTNOTE_PENALTY

    # Numeric content boosts
    content = chunk.content
    if "has_numbers" in chunk_kw_set or re.search(r"\d", content):
        score += _NUMERIC_BOOST

    # Strongest signal: chunk contains an actual "value unit" pair
    if _has_data_sentence(content):
        score += _DATA_SENTENCE_BOOST

    return ScoredChunk(chunk=chunk, score=score, matched_keywords=list(set(matched)))


def _stitch_neighbors(
    top_scored: list[ScoredChunk],
    all_chunks_by_index: dict[int, DocumentChunk],
    window: int = 1,
) -> list[ScoredChunk]:
    """
    For each top-scoring chunk, include its immediate neighbors (±window).
    This captures values that were split across chunk boundaries.
    Deduplicates by chunk_index.
    """
    seen: set[int] = set()
    result: list[ScoredChunk] = []

    for sc in top_scored:
        idx = sc.chunk.chunk_index
        if idx not in seen:
            seen.add(idx)
            result.append(sc)

        for offset in range(-window, window + 1):
            if offset == 0:
                continue
            neighbor_idx = idx + offset
            if neighbor_idx in seen:
                continue
            neighbor = all_chunks_by_index.get(neighbor_idx)
            if neighbor is None:
                continue
            seen.add(neighbor_idx)
            result.append(ScoredChunk(
                chunk=neighbor,
                score=_NEIGHBOR_SCORE,
                matched_keywords=[],
                is_neighbor=True,
            ))

    # Re-sort: primary chunks first by score, neighbors interleaved by chunk_index
    result.sort(key=lambda s: (s.is_neighbor, -s.score))
    return result


class RetrievalService:

    def __init__(self) -> None:
        self.settings = get_settings()

    def retrieve(
        self,
        parsed_document_id: uuid.UUID,
        kpi: KPIDefinition,
        db: Session,
        top_k: Optional[int] = None,
    ) -> list[ScoredChunk]:
        k = top_k or self.settings.retrieval_top_k
        query_keywords: list[str] = list(kpi.retrieval_keywords or [])

        if not query_keywords:
            logger.warning("retrieval.no_keywords", kpi=kpi.name)
            return []

        # --- Step 1: DB keyword pre-filter ---
        from sqlalchemy import or_
        keyword_filters = [
            DocumentChunk.keywords.ilike(f"%{kw.lower().split()[0]}%")
            for kw in query_keywords
        ]
        # Also include chunks tagged has_numbers that mention any keyword token
        keyword_filters.append(DocumentChunk.keywords.ilike("%has_numbers%"))

        candidate_chunks = (
            db.query(DocumentChunk)
            .filter(
                DocumentChunk.parsed_document_id == parsed_document_id,
                or_(*keyword_filters),
            )
            .all()
        )

        logger.debug("retrieval.candidates", kpi=kpi.name, candidates=len(candidate_chunks))

        if not candidate_chunks:
            # Fallback: return first K chunks
            logger.info("retrieval.fallback_positional", kpi=kpi.name)
            fallback = (
                db.query(DocumentChunk)
                .filter(DocumentChunk.parsed_document_id == parsed_document_id)
                .order_by(DocumentChunk.chunk_index)
                .limit(k)
                .all()
            )
            return [ScoredChunk(chunk=c, score=0) for c in fallback]

        # --- Step 2: Score ---
        scored = [_score_chunk(c, query_keywords) for c in candidate_chunks]
        scored = [s for s in scored if s.score > 0]

        # --- Step 3: Page-proximity bonus ---
        scored.sort(key=lambda s: s.score, reverse=True)
        top_pages = {s.chunk.page_number for s in scored[:3] if s.chunk.page_number}
        for s in scored[3:]:
            if s.chunk.page_number in top_pages:
                s.score += _PAGE_PROXIMITY_BONUS

        # --- Step 4: Take top-K primary results ---
        scored.sort(key=lambda s: s.score, reverse=True)
        top = scored[:k]

        # --- Step 5: Stitch in adjacent neighbors ---
        all_chunks_map = {c.chunk_index: c for c in candidate_chunks}
        # Also load immediate neighbors from DB that may not be in candidates
        if top:
            min_idx = min(s.chunk.chunk_index for s in top)
            max_idx = max(s.chunk.chunk_index for s in top)
            neighbor_chunks = (
                db.query(DocumentChunk)
                .filter(
                    DocumentChunk.parsed_document_id == parsed_document_id,
                    DocumentChunk.chunk_index >= max(0, min_idx - 1),
                    DocumentChunk.chunk_index <= max_idx + 1,
                )
                .all()
            )
            for nc in neighbor_chunks:
                all_chunks_map[nc.chunk_index] = nc

        stitched = _stitch_neighbors(top, all_chunks_map, window=1)

        # Cap at 2*k after stitching to avoid flooding LLM
        final = stitched[:k * 2]

        logger.info(
            "retrieval.complete",
            kpi=kpi.name,
            candidates=len(candidate_chunks),
            scored=len(scored),
            returned=len(final),
            top_score=round(top[0].score, 3) if top else 0,
        )
        return final

    def retrieve_by_keywords(
        self,
        parsed_document_id: uuid.UUID,
        keywords: list[str],
        db: Session,
        top_k: Optional[int] = None,
        chunk_types: Optional[list[str]] = None,
    ) -> list[ScoredChunk]:
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
        scored = [_score_chunk(c, keywords) for c in candidates]
        scored = [s for s in scored if s.score > 0]
        scored.sort(key=lambda s: s.score, reverse=True)

        # Stitch neighbors for ad-hoc queries too
        all_chunks_map = {c.chunk_index: c for c in candidates}
        if scored:
            stitched = _stitch_neighbors(scored[:k], all_chunks_map, window=1)
            return stitched[:k * 2]
        return scored[:k]
    
# ---------------------------------------------------------------------------
# Hybrid retrieval — combines semantic + keyword results
# ---------------------------------------------------------------------------

def _build_kpi_queries(kpi: KPIDefinition) -> list[str]:
    """
    Build natural-language query strings for semantic search from a KPI definition.
    Multiple queries improve recall — results are merged by max similarity.
    """
    from agents.extraction_agent import _KPI_ALIASES

    base_queries = [kpi.display_name]

    # Add unit to query — "Scope 1 GHG Emissions tCO2e"
    base_queries.append(f"{kpi.display_name} {kpi.expected_unit}")

    # Add keyword-based queries
    for kw in (kpi.retrieval_keywords or [])[:3]:
        base_queries.append(kw)

    # Add alias-based queries
    for alias in _KPI_ALIASES.get(kpi.name, [])[:3]:
        base_queries.append(alias)

    return base_queries


class HybridRetrievalService(RetrievalService):
    """
    Extends RetrievalService with semantic (embedding) retrieval.

    Strategy:
      1. Run semantic search (embedding cosine similarity) — top-K results
      2. Run keyword search (existing pipeline) — top-K results
      3. Merge: take union, score = max(semantic_score, keyword_score)
      4. Re-rank merged list by combined score
      5. Stitch neighbors around top results
      6. Return top-K*2 for LLM context

    Falls back to keyword-only if embeddings are not computed or model unavailable.
    """

    def __init__(self) -> None:
        super().__init__()
        self._embedding_service = None

    def _get_embedding_service(self):
        if self._embedding_service is None:
            from services.embedding_service import EmbeddingService
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    def retrieve(
        self,
        parsed_document_id: uuid.UUID,
        kpi: KPIDefinition,
        db: Session,
        top_k: Optional[int] = None,
    ) -> list[ScoredChunk]:
        k = top_k or self.settings.retrieval_top_k

        if not self.settings.use_embedding_retrieval:
            return super().retrieve(parsed_document_id, kpi, db, top_k)

        emb_service = self._get_embedding_service()

        # Check if this document has embeddings at all
        has_embeddings = db.query(DocumentChunk.id).filter(
            DocumentChunk.parsed_document_id == parsed_document_id,
            DocumentChunk.is_embedded == True,
        ).first() is not None

        if not has_embeddings or not emb_service.is_available():
            logger.info("retrieval.semantic_unavailable_fallback", kpi=kpi.name,
                       has_embeddings=has_embeddings)
            return super().retrieve(parsed_document_id, kpi, db, top_k)

        # --- Semantic retrieval ---
        queries = _build_kpi_queries(kpi)
        try:
            semantic_results = emb_service.search_multi_query(
                parsed_document_id=parsed_document_id,
                queries=queries,
                db=db,
                top_k=k,
            )

            semantic_scored = {
                chunk.id: ScoredChunk(
                    chunk=chunk,
                    score=sim * 10,  # scale to be comparable with keyword scores
                    matched_keywords=["[semantic]"],
                )
                for chunk, sim in semantic_results
            }
        except Exception as exc:
            semantic_results = emb_service.search_multi_query(
                parsed_document_id=parsed_document_id,
                queries=queries,
                db=db,
                top_k=k,
            )
            semantic_scored = {
                chunk.id: ScoredChunk(
                    chunk=chunk,
                    score=sim * 10,  # scale to be comparable with keyword scores
                    matched_keywords=["[semantic]"],
                )
                for chunk, sim in semantic_results
            }
            keyword_results = super().retrieve(parsed_document_id, kpi, db, top_k)
            keyword_scored = {sc.chunk.id: sc for sc in keyword_results}

            all_ids = set(semantic_scored) | set(keyword_scored)
            for chunk_id in all_ids[:2]:
                sem = semantic_scored.get(chunk_id)
                kw = keyword_scored.get(chunk_id)
                print(type(sem.score), "sem.score")
                print(type(kw.score), "kw.score")
            logger.warning("retrieval.semantic_failed", kpi=kpi.name, error=str(exc))
            semantic_scored = {}

        # --- Keyword retrieval ---
        keyword_results = super().retrieve(parsed_document_id, kpi, db, top_k)
        keyword_scored = {sc.chunk.id: sc for sc in keyword_results}

        max_kw = max((kw.score for kw in keyword_scored.values()), default=1)
        max_sem = max((sem.score for sem in semantic_scored.values()), default=1)

        for kw in keyword_scored.values():
            kw.score = kw.score / max_kw

        for sem in semantic_scored.values():
            sem.score = sem.score / max_sem
        
        # --- Merge: union of both result sets ---
        all_ids = set(semantic_scored) | set(keyword_scored)
        merged: list[ScoredChunk] = []

        for chunk_id in all_ids:
            sem = semantic_scored.get(chunk_id)
            kw = keyword_scored.get(chunk_id)
            if sem and kw:
                # Both found — boost combined score
                combined_score = sem.score * 0.6 + kw.score * 0.4 + 0.5
                merged.append(ScoredChunk(
                    chunk=sem.chunk,
                    score=combined_score,
                    matched_keywords=list(set(sem.matched_keywords + kw.matched_keywords)),
                ))
            elif sem:
                merged.append(sem)
            else:
                merged.append(kw)

        merged.sort(key=lambda s: s.score, reverse=True)
        top = merged[:k]

        # --- Neighbor stitching ---
        all_chunks_map: dict[int, DocumentChunk] = {}
        for sc in merged:
            all_chunks_map[sc.chunk.chunk_index] = sc.chunk

        if top:
            min_idx = min(s.chunk.chunk_index for s in top)
            max_idx = max(s.chunk.chunk_index for s in top)
            neighbor_chunks = (
                db.query(DocumentChunk)
                .filter(
                    DocumentChunk.parsed_document_id == parsed_document_id,
                    DocumentChunk.chunk_index >= max(0, min_idx - 1),
                    DocumentChunk.chunk_index <= max_idx + 1,
                )
                .all()
            )
            for nc in neighbor_chunks:
                all_chunks_map[nc.chunk_index] = nc

        stitched = _stitch_neighbors(top, all_chunks_map, window=1)
        final = stitched[:k * 2]

        logger.info(
            "retrieval.hybrid_complete",
            kpi=kpi.name,
            semantic=len(semantic_scored),
            keyword=len(keyword_scored),
            merged=len(merged),
            returned=len(final),
        )
        return final