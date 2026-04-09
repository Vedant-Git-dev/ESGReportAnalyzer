"""
services/retrieval_service.py

Retrieval Service — Phase 3 (v4, precision + recall balanced).

Fixes vs v3
-----------
Fix 1 — scope_1_emissions must_match too restrictive for TCS BRSR
  TCS BRSR GHG table chunks contain "greenhouse gas" and "tCO2e" but NOT
  "scope 1" verbatim — the scope label is in the table header (a separate
  chunk). Added broader GHG terms and the exact BRSR column phrase to
  must_match so the value row is admitted.

Fix 2 — must_exclude fired on combined-scope chunks (already in v3 but
  confirmed needed): A chunk saying "scope 1 ... scope 2 ... total" was
  dropped because "scope 2" is a must_exclude for scope_1_emissions.
  must_exclude only fires when NO must_match term is also present.

Fix 3 — unit_fallback now also fires for scope_1/scope_2 when fewer than
  MIN_RELEVANT_CHUNKS pass the strict filter. This is required for TCS where
  the GHG value row only contains "tCO2e" and a number, not a scope label.

Architecture: unchanged. No new dependencies.
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
# Scoring weights
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

# Minimum relevant chunks before activating unit-based fallback
_MIN_RELEVANT_CHUNKS = 2

# Detects "number unit" pairs — strong signal the chunk has actual data
_DATA_RE = re.compile(
    r"\b[\d,]+(?:\.\d+)?\s*"
    r"(?:tco2e?|t\s*co2e?|mwh|gwh|gj|tj|kl|kilolitr\w*|mt|tonne\w*|%|percent|crore|lakh|employees|headcount)\b",
    re.IGNORECASE,
)

# Matches "number near unit" — used in unit_fallback_mode.
# Handles both inline "21,949 tCO2e" and table formats "21,949 | tCO2e".
# The separator between number and unit may be spaces, pipes, or other delimiters.
_UNIT_NUMBER_RE = re.compile(
    r"\b[\d,]+(?:\.\d+)?[\s|,]{0,8}"
    r"(?:tco2e?|t\s*co2e?|mt\s*co2e?|gj|mwh|gwh|tj|kl|kilolitr\w*|crore|lakh)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# KPI strict filters
# ---------------------------------------------------------------------------
# must_match    — chunk must contain AT LEAST ONE of these (exact phrase, CI).
# must_exclude  — drops chunk ONLY IF no must_match term is also present.
# unit_fallback — strings checked when strict filter yields < MIN_RELEVANT_CHUNKS.
#                 Chunk is admitted if it has a unit_fallback term + a number.
# penalty_terms — soft: -PENALTY_PER_TERM per hit (does not drop the chunk).
# unit_hints    — +UNIT_MATCH_BONUS if present.

KPI_STRICT_FILTERS: dict[str, dict] = {
    "scope_1_emissions": {
        # EXPANDED: TCS BRSR uses "greenhouse gas" / "total scope" / "tCO2e" without
        # saying "scope 1" verbatim in the value row. Include broader GHG terms.
        "must_match":    [
            "scope 1", "scope-1", "direct emissions", "direct ghg",
            "stationary combustion", "fugitive", "owned vehicle",
            # Broader terms that appear in GHG tables even without "scope 1" label:
            "greenhouse gas emission", "ghg emission",
            "carbon neutral", "co2 equivalent",
            # BRSR table phrases that accompany scope 1 data:
            "total scope", "scope 1 and", "scope i",
        ],
        "must_exclude":  ["indirect emission", "purchased electricity", "scope 3"],
        # Note: "scope 2 emission" removed from must_exclude because combined rows
        # ("Scope 1 and Scope 2") should NOT be excluded when scope 1 is also present.
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
            # Broader terms:
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
# Public helper: is_relevant_chunk
# ---------------------------------------------------------------------------

def is_relevant_chunk(
    text: str,
    kpi_name: str,
    *,
    unit_fallback_mode: bool = False,
) -> tuple[bool, str]:
    """
    Relevance gate. Returns (is_relevant, reason_string).

    Normal mode:
      - must_exclude fires ONLY when no must_match term is also present.
        (A combined scope1+scope2 row should not be excluded just because
        "scope 2" is in the must_exclude list for scope_1_emissions.)
      - Drops chunk if must_match is defined and none are found.

    unit_fallback_mode:
      - Admits chunk if it has a unit_fallback string AND a number pattern.
      - Used when strict mode returns fewer than MIN_RELEVANT_CHUNKS.
    """
    flt = KPI_STRICT_FILTERS.get(kpi_name, _DEFAULT_FILTER)
    text_lower = text.lower()

    if unit_fallback_mode:
        for hint in flt.get("unit_fallback", []):
            if hint.lower() in text_lower and _UNIT_NUMBER_RE.search(text):
                return True, f"unit_fallback hit: '{hint}'"
        return False, "unit_fallback: no unit+number pattern"

    # Check must_match
    must_match = flt.get("must_match", [])
    matched_term: Optional[str] = None
    if must_match:
        matched_term = next((t for t in must_match if t.lower() in text_lower), None)
        if matched_term is None:
            return False, "no must_match term found"

    # Contextual exclusion: only fire when must_match did NOT hit.
    # This allows combined "Scope 1 and Scope 2" rows to pass for both KPIs.
    if matched_term is None:
        for term in flt.get("must_exclude", []):
            if term.lower() in text_lower:
                return False, f"must_exclude hit: '{term}'"

    return True, f"must_match: '{matched_term}'" if matched_term else "ok"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_chunk_precise(
    chunk: DocumentChunk,
    query_keywords: list[str],
    kpi_name: str,
) -> tuple[float, dict]:
    """
    Returns (score, breakdown_dict).
    breakdown_dict is used exclusively for debug logging.
    """
    flt = KPI_STRICT_FILTERS.get(kpi_name, _DEFAULT_FILTER)
    text = chunk.content
    text_lower = text.lower()
    chunk_kws = set((chunk.keywords or "").split())
    breakdown: dict = {}

    # --- Keyword score (partial token match) ---
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

    # --- Exact phrase bonus (multi-word keywords found verbatim) ---
    exact_bonus = 0.0
    for kw in query_keywords:
        if len(kw.split()) > 1 and kw.lower() in text_lower:
            exact_bonus += _EXACT_PHRASE_BONUS
    breakdown["exact_bonus"] = round(exact_bonus, 3)

    # --- Unit hint bonus ---
    unit_bonus = 0.0
    for hint in flt.get("unit_hints", []):
        if hint.lower() in text_lower:
            unit_bonus = _UNIT_MATCH_BONUS
            break
    breakdown["unit_bonus"] = round(unit_bonus, 3)

    # --- Numeric / data-sentence bonus ---
    numeric_bonus = _NUMERIC_BOOST if re.search(r"\d", text) else 0.0
    data_bonus    = _DATA_SENTENCE_BOOST if _DATA_RE.search(text) else 0.0
    breakdown["numeric_bonus"] = round(numeric_bonus, 3)
    breakdown["data_bonus"]    = round(data_bonus, 3)

    # --- Base score with type multiplier ---
    base = kw_score + exact_bonus + unit_bonus + numeric_bonus + data_bonus
    if chunk.chunk_type == "table":
        base *= _TABLE_BOOST
        breakdown["type_mult"] = _TABLE_BOOST
    elif chunk.chunk_type == "footnote":
        base *= _FOOTNOTE_PENALTY
        breakdown["type_mult"] = _FOOTNOTE_PENALTY
    else:
        breakdown["type_mult"] = 1.0

    # --- Penalty for noise-domain terms ---
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
# Dataclass + page-scoped neighbor stitching
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
    """
    Stitch in adjacent chunks — SAME PAGE only.

    Only include a neighbor if its page_number matches the anchor chunk's
    page_number. Unknown page numbers are always included.
    """
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
            # Page-scope gate
            if (anchor_page is not None
                    and neighbor.page_number is not None
                    and neighbor.page_number != anchor_page):
                logger.debug(
                    "retrieval.neighbor_skipped_cross_page",
                    anchor_chunk=idx,
                    anchor_page=anchor_page,
                    neighbor_chunk=neighbor_idx,
                    neighbor_page=neighbor.page_number,
                )
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

        # ── Step 1: Broad DB pre-filter ───────────────────────────────────
        from sqlalchemy import or_
        keyword_filters = [
            DocumentChunk.keywords.ilike(f"%{kw.lower().split()[0]}%")
            for kw in query_keywords
        ]
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
            logger.info("retrieval.fallback_positional", kpi=kpi.name)
            fallback = (
                db.query(DocumentChunk)
                .filter(DocumentChunk.parsed_document_id == parsed_document_id)
                .order_by(DocumentChunk.chunk_index)
                .limit(k)
                .all()
            )
            return [ScoredChunk(chunk=c, score=0) for c in fallback]

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
                    page=chunk.page_number,
                    reason=reason,
                    preview=chunk.content[:80].replace("\n", " "),
                )

        logger.info(
            "retrieval.after_strict_filter",
            kpi=kpi.name,
            before=len(candidate_chunks),
            after=len(relevant),
        )

        # ── Step 3: Unit-based fallback ────────────────────────────────────
        # When the GHG table is split across chunks, the value row may not
        # contain "scope 1" — only the header chunk does. The unit fallback
        # re-admits any candidate that has tCO2e + a number.
        if len(relevant) < _MIN_RELEVANT_CHUNKS:
            already_ids = {c.id for c in relevant}
            fallback_candidates: list[DocumentChunk] = []
            for chunk in candidate_chunks:
                if chunk.id in already_ids:
                    continue
                ok, reason = is_relevant_chunk(
                    chunk.content, kpi.name, unit_fallback_mode=True
                )
                if ok:
                    fallback_candidates.append(chunk)
                    logger.debug(
                        "retrieval.unit_fallback_admit",
                        kpi=kpi.name,
                        chunk_index=chunk.chunk_index,
                        page=chunk.page_number,
                        reason=reason,
                        preview=chunk.content[:80].replace("\n", " "),
                    )

            if fallback_candidates:
                logger.info(
                    "retrieval.unit_fallback_activated",
                    kpi=kpi.name,
                    strict_passed=len(relevant),
                    fallback_added=len(fallback_candidates),
                )
                relevant = relevant + fallback_candidates
            elif not relevant:
                logger.warning("retrieval.full_fallback", kpi=kpi.name)
                relevant = candidate_chunks

        # ── Step 4: Precise scoring ────────────────────────────────────────
        scored: list[ScoredChunk] = []
        for chunk in relevant:
            score, breakdown = _score_chunk_precise(chunk, query_keywords, kpi.name)
            if score <= 0:
                continue
            sc = ScoredChunk(
                chunk=chunk,
                score=score,
                matched_keywords=breakdown.get("matched_kws", []),
            )
            scored.append(sc)

            logger.debug(
                "retrieval.chunk_scored",
                kpi=kpi.name,
                chunk_index=chunk.chunk_index,
                page=chunk.page_number,
                chunk_type=chunk.chunk_type,
                score=breakdown["final"],
                kw=breakdown["kw_score"],
                exact=breakdown["exact_bonus"],
                unit=breakdown["unit_bonus"],
                data=breakdown["data_bonus"],
                penalty=breakdown["penalty"],
                penalty_hits=breakdown["penalty_hits"],
                preview=chunk.content[:80].replace("\n", " "),
            )

        if not scored:
            logger.info("retrieval.no_scored_chunks", kpi=kpi.name)
            return []

        # ── Step 5: Page-proximity bonus (top-3 pages) ────────────────────
        scored.sort(key=lambda s: s.score, reverse=True)
        top_pages = {s.chunk.page_number for s in scored[:3] if s.chunk.page_number}
        for s in scored[3:]:
            if s.chunk.page_number in top_pages:
                s.score += _PAGE_PROXIMITY_BONUS

        # ── Step 6: Top-K primary ─────────────────────────────────────────
        scored.sort(key=lambda s: s.score, reverse=True)
        top = scored[:k]

        # ── Step 7: Page-scoped neighbor stitching ────────────────────────
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
# Hybrid retrieval — semantic + keyword fusion
# ---------------------------------------------------------------------------

def _build_kpi_queries(kpi: KPIDefinition) -> list[str]:
    from agents.extraction_agent import _KPI_ALIASES
    base_queries = [kpi.display_name, f"{kpi.display_name} {kpi.expected_unit}"]
    for kw in (kpi.retrieval_keywords or [])[:3]:
        base_queries.append(kw)
    for alias in _KPI_ALIASES.get(kpi.name, [])[:3]:
        base_queries.append(alias)
    return base_queries


class HybridRetrievalService(RetrievalService):
    """
    Extends RetrievalService with semantic (embedding) retrieval.
    Strict filtering + page-scoped stitching inherited from RetrievalService.
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
        k = min(top_k or self.settings.retrieval_top_k, 5)

        if not self.settings.use_embedding_retrieval:
            return super().retrieve(parsed_document_id, kpi, db, top_k)

        emb_service = self._get_embedding_service()
        has_embeddings = db.query(DocumentChunk.id).filter(
            DocumentChunk.parsed_document_id == parsed_document_id,
            DocumentChunk.is_embedded == True,
        ).first() is not None

        if not has_embeddings or not emb_service.is_available():
            logger.info("retrieval.semantic_unavailable_fallback", kpi=kpi.name)
            return super().retrieve(parsed_document_id, kpi, db, top_k)

        # ── Semantic retrieval ────────────────────────────────────────────
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
                    score=sim * 10,
                    matched_keywords=["[semantic]"],
                )
                for chunk, sim in semantic_results
            }
        except Exception as exc:
            logger.warning("retrieval.semantic_failed", kpi=kpi.name, error=str(exc))
            semantic_scored = {}

        # ── Keyword retrieval (strict filter + page-scoped stitching) ─────
        keyword_results = super().retrieve(parsed_document_id, kpi, db, top_k)
        keyword_scored = {sc.chunk.id: sc for sc in keyword_results}

        # ── Apply strict filter to semantic results ───────────────────────
        filtered_semantic: dict = {}
        for cid, sc in semantic_scored.items():
            ok, reason = is_relevant_chunk(sc.chunk.content, kpi.name)
            if not ok:
                ok, reason = is_relevant_chunk(
                    sc.chunk.content, kpi.name, unit_fallback_mode=True
                )
            if ok:
                filtered_semantic[cid] = sc
            else:
                logger.debug(
                    "retrieval.semantic_chunk_filtered",
                    kpi=kpi.name,
                    chunk_index=sc.chunk.chunk_index,
                    reason=reason,
                )
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
                combined = sem.score * 0.5 + kw.score * 0.5 + 0.3
                merged.append(ScoredChunk(
                    chunk=sem.chunk,
                    score=combined,
                    matched_keywords=list(set(sem.matched_keywords + kw.matched_keywords)),
                ))
            elif sem:
                merged.append(sem)
            else:
                merged.append(kw)  # type: ignore[arg-type]

        merged.sort(key=lambda s: s.score, reverse=True)
        top = merged[:k]

        # ── Page-scoped neighbor stitching ────────────────────────────────
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