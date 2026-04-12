"""
services/retrieval_service.py

Retrieval Service — Phase 3 (v5, multi-query expansion + dedup).

Changes vs v4
-------------
1. _expand_kpi_queries(kpi) -> list[str]
   Builds a ranked list of query strings for one KPI using its retrieval
   keywords, display name, expected unit, and aliases from extraction_agent.
   Longer / more specific phrases first (higher precision), shorter terms
   last (fallback recall).

2. RetrievalService.retrieve() — multi-query execution
   Runs _expand_kpi_queries() to produce up to _MAX_QUERIES_PER_KPI query
   strings, executes each as a separate keyword-filter pass against the DB,
   then merges and deduplicates results by chunk ID, keeping the highest
   score seen for any given chunk.

3. HybridRetrievalService.retrieve() — same multi-query expansion applied
   to the semantic search path as well.

Everything else (strict filters, scoring, neighbor stitching) is unchanged.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy.orm import Session

from core.config import get_settings
from core.logging_config import get_logger
from models.db_models import DocumentChunk, KPIDefinition

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Scoring weights (unchanged)
# ---------------------------------------------------------------------------
_TABLE_BOOST          = 2.0
_FOOTNOTE_PENALTY     = 0.4
_NUMERIC_BOOST        = 0.3
_DATA_SENTENCE_BOOST  = 1.2
_EXACT_PHRASE_BONUS   = 2.0
_UNIT_MATCH_BONUS     = 1.5
_PENALTY_PER_TERM     = 0.7
_PAGE_PROXIMITY_BONUS = 0.15
_NEIGHBOR_SCORE       = 0.5
_MIN_RELEVANT_CHUNKS  = 2

# Maximum number of query strings run per KPI in multi-query mode
_MAX_QUERIES_PER_KPI  = 4

_DATA_RE = re.compile(
    r"\b[\d,]+(?:\.\d+)?\s*"
    r"(?:tco2e?|t\s*co2e?|mwh|gwh|gj|tj|kl|kilolitr\w*|mt|tonne\w*|%|percent|crore|lakh|employees|headcount)\b",
    re.IGNORECASE,
)

_UNIT_NUMBER_RE = re.compile(
    r"\b[\d,]+(?:\.\d+)?[\s|,]{0,8}"
    r"(?:tco2e?|t\s*co2e?|mt\s*co2e?|gj|mwh|gwh|tj|kl|kilolitr\w*|crore|lakh)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# KPI strict filters  (unchanged from v4)
# ---------------------------------------------------------------------------
KPI_STRICT_FILTERS: dict[str, dict] = {
    "scope_1_emissions": {
        "must_match":    [
            "scope 1", "scope-1", "direct emissions", "direct ghg",
            # "stationary combustion", "fugitive", "owned vehicle",
            # "greenhouse gas emission", "ghg emission",
            # "carbon neutral", "co2 equivalent",
            # "total scope", "scope 1 and", "scope i",
        ],
        "must_exclude":  ["scope 1"],
        "unit_fallback": ["tco2e", "t co2e", "tonne co2", "mt co2", "co2e"],
        "penalty_terms": [
            "employee", "salary", "turnover", "headcount",
            "water consumption", "waste generated", "csr spend",
            "total employees", "workforce", "remuneration",
        ],
        "unit_hints":    ["tco2e", "t co2e", "tonne co2e", "mt co2e"],
    },
    "scope_2_emissions": {
        "must_match":    [
            "scope 2", "scope-2", "indirect emissions",
            "purchased electricity", "market-based", "location-based",
            "electricity ghg", "grid emission",
            "total scope", "scope 1 and", "scope ii",
        ],
        "must_exclude":  ["stationary combustion", "scope 3"],
        "unit_fallback": ["tco2e", "t co2e", "tonne co2", "co2e"],
        "penalty_terms": [
            "employee", "salary", "turnover", "headcount",
            "water consumption", "waste generated",
        ],
        "unit_hints":    ["tco2e", "t co2e", "tonne co2e"],
    },
    "total_ghg_emissions": {
        "must_match":    [
            "total ghg", "total emissions", "scope 1 and 2",
            "scope 1+2", "scope 1 & 2", "carbon footprint",
            "greenhouse gas", "ghg inventory", "co2 equivalent",
            "carbon neutral", "net zero",
        ],
        "must_exclude":  [],
        "unit_fallback": ["tco2e", "t co2e", "mtco2e", "co2e"],
        "penalty_terms": [
            "employee", "salary", "turnover", "headcount",
            "water consumption", "waste generated",
        ],
        "unit_hints":    ["tco2e", "t co2e", "mtco2e"],
    },
    "energy_consumption": {
        "must_match":    [
            "total energy", "energy consumption", "energy consumed",
            "energy use", "fuel consumed", "electricity consumed",
            "gigajoule", "megawatt", "fuel consumption",
        ],
        "must_exclude":  ["emissions tco2e"],
        "unit_fallback": ["gj", "mwh", "gwh", "tj"],
        "penalty_terms": [
            "employee", "salary", "turnover", "headcount",
            "water consumption", "waste generated", "csr",
        ],
        "unit_hints":    ["gj", "mwh", "gwh", "tj"],
    },
    "renewable_energy_percentage": {
        "must_match":    [
            "renewable energy", "solar energy", "wind energy",
            "clean energy", "green energy", "renewable electricity",
            "non-fossil", "re share", "percent renewable",
            "solar power", "wind power",
        ],
        "must_exclude":  [],
        "unit_fallback": [],
        "penalty_terms": [
            "employee", "salary", "turnover", "headcount",
            "water consumption", "waste generated",
        ],
        "unit_hints":    ["%", "percent"],
    },
    "water_consumption": {
        "must_match":    [
            "water consumption", "water consumed", "water use",
            "water withdrawal", "freshwater", "water recycled",
            "water intensity", "kilolitre", "water discharge",
            "zero liquid discharge",
        ],
        "must_exclude":  [],
        "unit_fallback": ["kl", "kilolitre", "m3"],
        "penalty_terms": [
            "employee", "salary", "turnover", "headcount",
            "waste generated", "csr", "tco2e",
        ],
        "unit_hints":    ["kl", "kilolitre", "m3", "cubic met"],
    },
    "waste_generated": {
        "must_match":    [
            "waste generated", "total waste", "solid waste",
            "hazardous waste", "non-hazardous waste",
            "waste disposed", "waste diverted", "waste management",
            "incineration", "landfill",
        ],
        "must_exclude":  [],
        "unit_fallback": ["metric tonne", "metric ton"],
        "penalty_terms": [
            "employee", "salary", "turnover", "headcount", "csr",
            "tco2e", "water consumption",
        ],
        "unit_hints":    ["mt", "metric tonne", "metric ton", "kg"],
    },
    "employee_count": {
        "must_match":    [
            "total employees", "total workforce", "headcount",
            "number of employees", "permanent employees",
            "workforce strength", "employee base", "fte",
            "employee strength",
        ],
        "must_exclude":  [],
        "unit_fallback": [],
        "penalty_terms": [
            "salary", "remuneration", "csr", "turnover rate",
            "energy consumption", "tco2e",
        ],
        "unit_hints":    ["headcount", "number", "nos", "employees"],
    },
    "women_in_workforce_percentage": {
        "must_match":    [
            "women", "female", "gender diversity", "gender ratio",
            "women employees", "female employees",
        ],
        "must_exclude":  [],
        "unit_fallback": [],
        "penalty_terms": [
            "salary", "remuneration", "csr", "energy consumption",
            "tco2e", "water",
        ],
        "unit_hints":    ["%", "percent"],
    },
    "csr_spend": {
        "must_match":    [
            "csr expenditure", "csr spend", "csr investment",
            "amount spent on csr", "corporate social responsibility",
            "social spend", "community investment", "csr amount",
        ],
        "must_exclude":  [],
        "unit_fallback": ["crore", "lakh"],
        "penalty_terms": [
            "energy consumption", "ghg", "tco2e", "water",
            "waste", "employee count",
        ],
        "unit_hints":    ["crore", "lakh", "inr"],
    },
}

_DEFAULT_FILTER: dict = {
    "must_match":    [],
    "must_exclude":  [],
    "unit_fallback": [],
    "penalty_terms": ["employee", "salary", "turnover"],
    "unit_hints":    [],
}


# ---------------------------------------------------------------------------
# Query expansion  (NEW)
# ---------------------------------------------------------------------------

def _expand_kpi_queries(kpi: KPIDefinition) -> list[str]:
    """
    Return a ranked list of query strings for *kpi* (at most _MAX_QUERIES_PER_KPI).

    Longer / more specific phrases come first for higher precision; short
    single-term fallbacks come last.  Deduped by lower-case value.
    """
    try:
        from agents.extraction_agent import _KPI_ALIASES
        aliases = _KPI_ALIASES.get(kpi.name, [])
    except ImportError:
        aliases = []

    seen: set[str] = set()
    queries: list[str] = []

    def _add(q: str) -> None:
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            queries.append(q)

    # Most specific first
    if kpi.display_name and kpi.expected_unit:
        _add(f"{kpi.display_name} {kpi.expected_unit}")

    kw_list = sorted(kpi.retrieval_keywords or [], key=len, reverse=True)
    for kw in kw_list:
        _add(kw)
        if len(queries) >= _MAX_QUERIES_PER_KPI:
            break

    if kpi.display_name:
        _add(kpi.display_name)

    for alias in aliases:
        _add(alias)
        if len(queries) >= _MAX_QUERIES_PER_KPI:
            break

    flt = KPI_STRICT_FILTERS.get(kpi.name, _DEFAULT_FILTER)
    for hint in flt.get("unit_hints", []):
        _add(hint)
        if len(queries) >= _MAX_QUERIES_PER_KPI:
            break

    return queries[:_MAX_QUERIES_PER_KPI]


# ---------------------------------------------------------------------------
# is_relevant_chunk  (unchanged from v4)
# ---------------------------------------------------------------------------

def is_relevant_chunk(
    text: str,
    kpi_name: str,
    *,
    unit_fallback_mode: bool = False,
) -> tuple[bool, str]:
    flt = KPI_STRICT_FILTERS.get(kpi_name, _DEFAULT_FILTER)
    text_lower = text.lower()

    if unit_fallback_mode:
        for hint in flt.get("unit_fallback", []):
            if hint.lower() in text_lower and _UNIT_NUMBER_RE.search(text):
                return True, f"unit_fallback hit: '{hint}'"
        return False, "unit_fallback: no unit+number pattern"

    must_match = flt.get("must_match", [])
    matched_term: Optional[str] = None
    if must_match:
        matched_term = next((t for t in must_match if t.lower() in text_lower), None)
        if matched_term is None:
            return False, "no must_match term found"

    if matched_term is None:
        for term in flt.get("must_exclude", []):
            if term.lower() in text_lower:
                return False, f"must_exclude hit: '{term}'"

    return True, f"must_match: '{matched_term}'" if matched_term else "ok"


# ---------------------------------------------------------------------------
# Scoring  (unchanged)
# ---------------------------------------------------------------------------

def _score_chunk_precise(
    chunk: DocumentChunk,
    query_keywords: list[str],
    kpi_name: str,
) -> tuple[float, dict]:
    flt = KPI_STRICT_FILTERS.get(kpi_name, _DEFAULT_FILTER)
    text = chunk.content
    text_lower = text.lower()
    chunk_kws = set((chunk.keywords or "").split())
    breakdown: dict = {}

    kw_score = 0.0
    matched_kws: list[str] = []
    for kw in query_keywords:
        kw_lower = kw.lower()
        first = kw_lower.split()[0]
        if first in chunk_kws:
            matched_kws.append(kw)
            kw_score += 1.0
        else:
            for ck in chunk_kws:
                if first in ck or ck in first:
                    matched_kws.append(kw)
                    kw_score += 0.3
                    break

    token_count = max(chunk.token_count or 1, 1)
    kw_score = kw_score / (token_count ** 0.2)
    breakdown["kw_score"] = round(kw_score, 3)

    exact_bonus = 0.0
    for kw in query_keywords:
        if len(kw.split()) > 1 and kw.lower() in text_lower:
            exact_bonus += _EXACT_PHRASE_BONUS
    breakdown["exact_bonus"] = round(exact_bonus, 3)

    unit_bonus = 0.0
    for hint in flt.get("unit_hints", []):
        if hint.lower() in text_lower:
            unit_bonus = _UNIT_MATCH_BONUS
            break
    breakdown["unit_bonus"] = round(unit_bonus, 3)

    numeric_bonus = _NUMERIC_BOOST if re.search(r"\d", text) else 0.0
    data_bonus    = _DATA_SENTENCE_BOOST if _DATA_RE.search(text) else 0.0
    breakdown["numeric_bonus"] = round(numeric_bonus, 3)
    breakdown["data_bonus"]    = round(data_bonus, 3)

    base = kw_score + exact_bonus + unit_bonus + numeric_bonus + data_bonus
    if chunk.chunk_type == "table":
        base *= _TABLE_BOOST
        breakdown["type_mult"] = _TABLE_BOOST
    elif chunk.chunk_type == "footnote":
        base *= _FOOTNOTE_PENALTY
        breakdown["type_mult"] = _FOOTNOTE_PENALTY
    else:
        breakdown["type_mult"] = 1.0

    penalty = 0.0
    hit_penalties: list[str] = []
    for term in flt.get("penalty_terms", []):
        if term.lower() in text_lower:
            penalty += _PENALTY_PER_TERM
            hit_penalties.append(term)
    breakdown["penalty"]      = round(penalty, 3)
    breakdown["penalty_hits"] = hit_penalties

    final = max(0.0, base - penalty)
    breakdown["final"]       = round(final, 3)
    breakdown["matched_kws"] = matched_kws
    return final, breakdown


# ---------------------------------------------------------------------------
# ScoredChunk + neighbor stitching  (unchanged)
# ---------------------------------------------------------------------------

@dataclass
class ScoredChunk:
    chunk: DocumentChunk
    score: float
    matched_keywords: list[str] = field(default_factory=list)
    is_neighbor: bool = False


def _stitch_neighbors_page_scoped(
    top_scored: list[ScoredChunk],
    all_chunks_by_index: dict[int, DocumentChunk],
    window: int = 1,
) -> list[ScoredChunk]:
    seen: set[int] = set()
    result: list[ScoredChunk] = []

    for sc in top_scored:
        idx = sc.chunk.chunk_index
        anchor_page = sc.chunk.page_number
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
            if (anchor_page is not None
                    and neighbor.page_number is not None
                    and neighbor.page_number != anchor_page):
                continue
            seen.add(neighbor_idx)
            result.append(ScoredChunk(
                chunk=neighbor,
                score=_NEIGHBOR_SCORE,
                matched_keywords=[],
                is_neighbor=True,
            ))

    result.sort(key=lambda s: (s.is_neighbor, -s.score))
    return result


# ---------------------------------------------------------------------------
# Internal helpers for multi-query candidate collection
# ---------------------------------------------------------------------------

def _db_candidates_for_keywords(
    parsed_document_id: uuid.UUID,
    keywords: list[str],
    db: Session,
) -> list[DocumentChunk]:
    """Broad DB keyword pre-filter for a list of keyword strings."""
    from sqlalchemy import or_
    kf = [
        DocumentChunk.keywords.ilike(f"%{kw.lower().split()[0]}%")
        for kw in keywords
    ]
    kf.append(DocumentChunk.keywords.ilike("%has_numbers%"))
    return (
        db.query(DocumentChunk)
        .filter(
            DocumentChunk.parsed_document_id == parsed_document_id,
            or_(*kf),
        )
        .all()
    )


def _merge_candidate_sets(sets: list[list[DocumentChunk]]) -> list[DocumentChunk]:
    """Deduplicate multiple candidate lists by chunk ID."""
    seen: dict[uuid.UUID, DocumentChunk] = {}
    for candidates in sets:
        for c in candidates:
            if c.id not in seen:
                seen[c.id] = c
    return list(seen.values())


# ---------------------------------------------------------------------------
# RetrievalService
# ---------------------------------------------------------------------------

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
        k = min(top_k or self.settings.retrieval_top_k, 5)
        query_keywords: list[str] = list(kpi.retrieval_keywords or [])

        if not query_keywords:
            logger.warning("retrieval.no_keywords", kpi=kpi.name)
            return []

        # ── Step 1: Multi-query candidate collection ──────────────────────
        expanded_queries = _expand_kpi_queries(kpi)
        candidate_sets: list[list[DocumentChunk]] = []

        for qry in expanded_queries:
            cands = _db_candidates_for_keywords(
                parsed_document_id, qry.split(), db
            )
            if cands:
                candidate_sets.append(cands)

        # Fallback: original keyword list
        if not candidate_sets:
            cands = _db_candidates_for_keywords(
                parsed_document_id, query_keywords, db
            )
            if cands:
                candidate_sets.append(cands)

        if not candidate_sets:
            logger.info("retrieval.fallback_positional", kpi=kpi.name)
            fallback = (
                db.query(DocumentChunk)
                .filter(DocumentChunk.parsed_document_id == parsed_document_id)
                .order_by(DocumentChunk.chunk_index)
                .limit(k)
                .all()
            )
            return [ScoredChunk(chunk=c, score=0) for c in fallback]

        candidate_chunks = _merge_candidate_sets(candidate_sets)

        logger.debug(
            "retrieval.candidates",
            kpi=kpi.name,
            queries=len(expanded_queries),
            candidates=len(candidate_chunks),
        )

        # ── Step 2: Strict relevance filter ───────────────────────────────
        relevant: list[DocumentChunk] = []
        for chunk in candidate_chunks:
            ok, reason = is_relevant_chunk(chunk.content, kpi.name)
            if ok:
                relevant.append(chunk)
            else:
                logger.debug(
                    "retrieval.chunk_filtered",
                    kpi=kpi.name,
                    chunk_index=chunk.chunk_index,
                    reason=reason,
                )

        logger.info(
            "retrieval.after_strict_filter",
            kpi=kpi.name,
            before=len(candidate_chunks),
            after=len(relevant),
        )

        # ── Step 3: Unit-based fallback ────────────────────────────────────
        if len(relevant) < _MIN_RELEVANT_CHUNKS:
            already_ids = {c.id for c in relevant}
            for chunk in candidate_chunks:
                if chunk.id in already_ids:
                    continue
                ok, _ = is_relevant_chunk(
                    chunk.content, kpi.name, unit_fallback_mode=True
                )
                if ok:
                    relevant.append(chunk)
            if not relevant:
                relevant = candidate_chunks

        # ── Step 4: Scoring ────────────────────────────────────────────────
        scored: list[ScoredChunk] = []
        for chunk in relevant:
            score, breakdown = _score_chunk_precise(chunk, query_keywords, kpi.name)
            if score <= 0:
                continue
            scored.append(ScoredChunk(
                chunk=chunk,
                score=score,
                matched_keywords=breakdown.get("matched_kws", []),
            ))

        if not scored:
            return []

        # ── Step 5: Page-proximity bonus ──────────────────────────────────
        scored.sort(key=lambda s: s.score, reverse=True)
        top_pages = {s.chunk.page_number for s in scored[:3] if s.chunk.page_number}
        for s in scored[3:]:
            if s.chunk.page_number in top_pages:
                s.score += _PAGE_PROXIMITY_BONUS

        # ── Step 6: Top-K + neighbor stitching ────────────────────────────
        scored.sort(key=lambda s: s.score, reverse=True)
        top = scored[:k]

        all_chunks_map = {c.chunk_index: c for c in relevant}
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

        stitched = _stitch_neighbors_page_scoped(top, all_chunks_map, window=1)
        final = stitched[: k * 2]

        logger.info(
            "retrieval.complete",
            kpi=kpi.name,
            candidates=len(candidate_chunks),
            after_filter=len(relevant),
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
        kf = [DocumentChunk.keywords.ilike(f"%{kw.lower()}%") for kw in keywords]
        query = db.query(DocumentChunk).filter(
            DocumentChunk.parsed_document_id == parsed_document_id,
            or_(*kf),
        )
        if chunk_types:
            query = query.filter(DocumentChunk.chunk_type.in_(chunk_types))
        candidates = query.all()

        scored_list: list[ScoredChunk] = []
        for chunk in candidates:
            score, breakdown = _score_chunk_precise(chunk, keywords, kpi_name="")
            if score > 0:
                scored_list.append(ScoredChunk(
                    chunk=chunk,
                    score=score,
                    matched_keywords=breakdown.get("matched_kws", []),
                ))

        scored_list.sort(key=lambda s: s.score, reverse=True)
        all_chunks_map = {c.chunk_index: c for c in candidates}
        if scored_list:
            stitched = _stitch_neighbors_page_scoped(scored_list[:k], all_chunks_map)
            return stitched[: k * 2]
        return scored_list[:k]


# ---------------------------------------------------------------------------
# Semantic query builder used by HybridRetrievalService
# ---------------------------------------------------------------------------

def _build_kpi_queries(kpi: KPIDefinition) -> list[str]:
    try:
        from agents.extraction_agent import _KPI_ALIASES
        aliases = _KPI_ALIASES.get(kpi.name, [])
    except ImportError:
        aliases = []
    base = [kpi.display_name, f"{kpi.display_name} {kpi.expected_unit}"]
    for kw in (kpi.retrieval_keywords or [])[:3]:
        base.append(kw)
    for alias in aliases[:3]:
        base.append(alias)
    return base


# ---------------------------------------------------------------------------
# HybridRetrievalService
# ---------------------------------------------------------------------------

class HybridRetrievalService(RetrievalService):

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
        k = min(top_k or self.settings.retrieval_top_k, 5)

        if not self.settings.use_embedding_retrieval:
            return super().retrieve(parsed_document_id, kpi, db, top_k)

        emb_service = self._get_embedding_service()
        has_embeddings = db.query(DocumentChunk.id).filter(
            DocumentChunk.parsed_document_id == parsed_document_id,
            DocumentChunk.is_embedded == True,
        ).first() is not None

        if not has_embeddings or not emb_service.is_available():
            return super().retrieve(parsed_document_id, kpi, db, top_k)

        # ── Semantic retrieval with expanded queries ───────────────────────
        semantic_queries = _build_kpi_queries(kpi)
        try:
            semantic_results = emb_service.search_multi_query(
                parsed_document_id=parsed_document_id,
                queries=semantic_queries,
                db=db,
                top_k=k,
            )
            semantic_scored = {
                chunk.id: ScoredChunk(
                    chunk=chunk,
                    score=sim * 10,
                    matched_keywords=["[semantic]"],
                )
                for chunk, sim in semantic_results
            }
        except Exception as exc:
            logger.warning("retrieval.semantic_failed", kpi=kpi.name, error=str(exc))
            semantic_scored = {}

        # ── Keyword retrieval (multi-query via parent) ─────────────────────
        keyword_results = super().retrieve(parsed_document_id, kpi, db, top_k)
        keyword_scored = {sc.chunk.id: sc for sc in keyword_results}

        # ── Strict filter on semantic results ─────────────────────────────
        filtered_semantic: dict = {}
        for cid, sc in semantic_scored.items():
            ok, _ = is_relevant_chunk(sc.chunk.content, kpi.name)
            if not ok:
                ok, _ = is_relevant_chunk(
                    sc.chunk.content, kpi.name, unit_fallback_mode=True
                )
            if ok:
                filtered_semantic[cid] = sc
        semantic_scored = filtered_semantic

        # ── Normalise + merge ─────────────────────────────────────────────
        max_kw  = max((s.score for s in keyword_scored.values()), default=1.0) or 1.0
        max_sem = max((s.score for s in semantic_scored.values()), default=1.0) or 1.0
        for s in keyword_scored.values():
            s.score /= max_kw
        for s in semantic_scored.values():
            s.score /= max_sem

        all_ids = set(semantic_scored) | set(keyword_scored)
        merged: list[ScoredChunk] = []
        for chunk_id in all_ids:
            sem = semantic_scored.get(chunk_id)
            kw  = keyword_scored.get(chunk_id)
            if sem and kw:
                merged.append(ScoredChunk(
                    chunk=sem.chunk,
                    score=sem.score * 0.5 + kw.score * 0.5 + 0.3,
                    matched_keywords=list(set(sem.matched_keywords + kw.matched_keywords)),
                ))
            elif sem:
                merged.append(sem)
            else:
                merged.append(kw)  # type: ignore[arg-type]

        merged.sort(key=lambda s: s.score, reverse=True)
        top = merged[:k]

        all_chunks_map: dict[int, DocumentChunk] = {
            sc.chunk.chunk_index: sc.chunk for sc in merged
        }
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

        stitched = _stitch_neighbors_page_scoped(top, all_chunks_map, window=1)
        final = stitched[: k * 2]

        logger.info(
            "retrieval.hybrid_complete",
            kpi=kpi.name,
            semantic=len(semantic_scored),
            keyword=len(keyword_scored),
            merged=len(merged),
            returned=len(final),
        )
        return final