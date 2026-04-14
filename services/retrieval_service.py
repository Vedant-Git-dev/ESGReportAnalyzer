"""
services/retrieval_service.py

Retrieval Service — Phase 3 (v5, answerability-first scoring).

Changes vs v4
-------------
(See original file header for full change log.)

NEW in this version:
  Added strict filters for three new KPIs:
    - scope_3_emissions   : must match scope-3 terms; must NOT match scope-1/2 only
    - energy_consumption  : must match energy terms; must NOT match GHG/emissions terms
    - water_consumption   : must match water terms; must NOT match waste/disposal terms
  All existing filters are preserved unchanged.
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
# Scoring weights (unchanged from v5)
# ---------------------------------------------------------------------------
_TABLE_BOOST          = 3.0
_FOOTNOTE_PENALTY     = 0.4
_NUMERIC_BOOST        = 0.8
_DATA_SENTENCE_BOOST  = 1.2
_EXACT_PHRASE_BONUS   = 2.0
_UNIT_MATCH_BONUS     = 1.5
_PENALTY_PER_TERM     = 0.7
_PAGE_PROXIMITY_BONUS = 0.15
_NEIGHBOR_SCORE       = 0.5

_ANSWERABILITY_BOOST  = 5.0
_NO_NUMERIC_PENALTY   = 1.5
_NO_UNIT_PENALTY      = 0.8
_PROSE_TABLE_FACTOR   = 0.5
_MIN_NUMERIC_DENSITY  = 0.10
_MIN_PIPE_COUNT       = 2
_MIN_RELEVANT_CHUNKS  = 2

_DATA_RE = re.compile(
    r"\b[\d,]+(?:\.\d+)?\s*"
    r"(?:tco2e?|t\s*co2e?|mwh|gwh|gj|tj|kl|kilolitr\w*|mt|tonne\w*|%|percent|crore|lakh|employees|headcount)\b",
    re.IGNORECASE,
)

_UNIT_CONTENT_RE = re.compile(
    r"(?:"
    r"\b(?:"
    r"tco2e?|t\s*co2e?|mt\s*co2e?|kt\s*co2e?|"
    r"mj|gj|tj|pj|mwh|gwh|twh|kwh|"
    r"kl|kilolitr\w*|m3|m\xb3|litr\w*|megalitr\w*|"
    r"mt|metric\s*tonn?\w*|tonn?\w*|kg|"
    r"crore|lakh|"
    r"employees|headcount|nos|fte|"
    r"percent"
    r")\b"
    r"|%"
    r")",
    re.IGNORECASE,
)

_UNIT_NUMBER_RE = re.compile(
    r"\b[\d,]+(?:\.\d+)?[\s|,]{0,8}"
    r"(?:tco2e?|t\s*co2e?|mt\s*co2e?|gj|mwh|gwh|tj|kl|kilolitr\w*|crore|lakh)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# KPI strict filters
# ---------------------------------------------------------------------------
KPI_STRICT_FILTERS: dict[str, dict] = {
    # ── Existing KPIs (unchanged) ────────────────────────────────────────────
    "scope_1_emissions": {
        "must_match":    [
            "scope 1", "scope-1", "direct emissions", "direct ghg",
            "stationary combustion", "fugitive", "owned vehicle",
            "greenhouse gas emission", "ghg emission",
            "carbon neutral", "co2 equivalent",
            "total scope", "scope 1 and", "scope i",
        ],
        "must_exclude":  ["indirect emission", "purchased electricity", "scope 3"],
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
    "energy_consumption": {
        # Energy must NOT match emissions-only chunks (GHG kPIs) or water/waste
        "must_match":    [
            "total energy", "energy consumption", "energy consumed",
            "energy use", "fuel consumed", "electricity consumed",
            "gigajoule", "megawatt", "fuel consumption",
            "energy used", "power consumption", "thermal energy",
            "energy from grid", "total energy consumed",
            "total energy consumption", "electricity consumption",
            "non-renewable energy", "renewable energy consumed",
        ],
        "must_exclude":  [
            "emissions tco2e",      # GHG table rows
            "scope 1 emissions",    # GHG rows that mention energy intensity
            "scope 2 emissions",
        ],
        "unit_fallback": ["gj", "mwh", "gwh", "tj", "kwh", "mj"],
        "penalty_terms": [
            "employee", "salary", "turnover", "headcount",
            "water consumption", "waste generated", "csr",
            "scope 1 emissions", "scope 2 emissions",
            "tco2e",  # penalise GHG-only chunks
        ],
        "unit_hints":    ["gj", "mwh", "gwh", "tj", "kwh", "mj"],
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
        # Water must NOT match waste-disposal chunks
        "must_match":    [
            "water consumption", "water consumed", "water use",
            "water withdrawal", "freshwater", "water recycled",
            "water intensity", "kilolitre", "water discharge",
            "zero liquid discharge", "water used", "water intake",
            "total water", "water withdrawn", "water abstraction",
            "water sourced", "ground water", "surface water",
            "municipal water", "third party water",
            "total freshwater", "water footprint",
            "net water consumption",
        ],
        "must_exclude":  [
            "waste disposed",     # waste chunks sometimes mention "water" in passing
            "landfill",           # solid waste disposal
            "hazardous waste",    # waste-type context, not water
            "solid waste",
        ],
        "unit_fallback": ["kl", "kilolitre", "m3"],
        "penalty_terms": [
            "employee", "salary", "turnover", "headcount",
            "waste generated", "csr", "tco2e",
            "solid waste", "hazardous waste",
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

    # ── NEW KPIs ─────────────────────────────────────────────────────────────

    "scope_3_emissions": {
        # Must NOT match scope-1-only or scope-2-only chunks.
        # "total scope 1 and 2" is also excluded — that's total_ghg, not scope 3.
        "must_match":    [
            "scope 3", "scope-3", "scope iii",
            "value chain emissions", "upstream emissions", "downstream emissions",
            "supply chain emissions", "indirect value chain",
            "scope 3 category", "total scope 3",
            "business travel emissions", "employee commute",
            "purchased goods and services", "use of sold products",
            "end-of-life treatment", "capital goods",
            "transportation and distribution", "waste in operations",
            "processing of sold products", "franchises", "investments",
            "other indirect emissions",
            "scope 3 ghg", "scope 3 tco2e", "scope 3 carbon",
            "scope 3 footprint",
        ],
        "must_exclude":  [
            # Prevent false matches on scope-1/2-only rows
            "scope 1 and scope 2",   # combined scope 1+2 rows
            "scope 1 emissions only",
            "scope 2 emissions only",
            "scope 1 and 2 intensity",  # intensity rows
        ],
        "unit_fallback": ["tco2e", "t co2e", "tonne co2", "co2e", "kt co2e"],
        "penalty_terms": [
            "employee", "salary", "turnover", "headcount",
            "water consumption", "waste generated", "csr",
            "energy consumption", "gigajoule", "mwh",
            # Penalise chunks that mention scope 1 or 2 as primary subject
            "scope 1 direct emissions",
            "scope 2 indirect emissions",
        ],
        "unit_hints":    ["tco2e", "t co2e", "tonne co2e", "kt co2e", "mt co2e"],
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
    (Logic unchanged from v5 — new KPIs covered by KPI_STRICT_FILTERS entries.)
    """
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
# v5 helpers: answerability and table-structure detection (unchanged)
# ---------------------------------------------------------------------------

def _has_esg_unit(text: str) -> bool:
    return bool(_UNIT_CONTENT_RE.search(text))


def _has_numeric(text: str) -> bool:
    return bool(re.search(r"\d", text))


def _numeric_density(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if re.search(r"\d", t)) / len(tokens)


def _is_structurally_table(text: str) -> bool:
    return (
        text.count("|") >= _MIN_PIPE_COUNT
        or _numeric_density(text) >= _MIN_NUMERIC_DENSITY
    )


# ---------------------------------------------------------------------------
# Scoring (unchanged from v5)
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

    # 1. Keyword score
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

    # 2. Exact phrase bonus
    exact_bonus = 0.0
    for kw in query_keywords:
        if len(kw.split()) > 1 and kw.lower() in text_lower:
            exact_bonus += _EXACT_PHRASE_BONUS
    breakdown["exact_bonus"] = round(exact_bonus, 3)

    # 3. Unit hint bonus
    unit_bonus = 0.0
    for hint in flt.get("unit_hints", []):
        if hint.lower() in text_lower:
            unit_bonus = _UNIT_MATCH_BONUS
            break
    breakdown["unit_bonus"] = round(unit_bonus, 3)

    # 4. Numeric bonus
    chunk_has_numeric = _has_numeric(text)
    numeric_bonus = _NUMERIC_BOOST if chunk_has_numeric else 0.0
    breakdown["numeric_bonus"] = round(numeric_bonus, 3)

    # 5. Data-sentence bonus
    data_bonus = _DATA_SENTENCE_BOOST if _DATA_RE.search(text) else 0.0
    breakdown["data_bonus"] = round(data_bonus, 3)

    # 6. Answerability boost
    must_match_terms = flt.get("must_match", [])
    chunk_has_kpi_kw = any(t.lower() in text_lower for t in must_match_terms)
    chunk_has_unit   = _has_esg_unit(text)

    answerable = chunk_has_kpi_kw and chunk_has_numeric and chunk_has_unit
    ans_boost  = _ANSWERABILITY_BOOST if answerable else 0.0
    breakdown["ans_boost"]   = round(ans_boost, 3)
    breakdown["answerable"]  = answerable

    # Base score (pre-multiplier)
    base = kw_score + exact_bonus + unit_bonus + numeric_bonus + data_bonus + ans_boost

    # 7. Chunk type multiplier
    if chunk.chunk_type == "table":
        real_table = _is_structurally_table(text)
        effective_mult = _TABLE_BOOST if real_table else _TABLE_BOOST * _PROSE_TABLE_FACTOR
        base *= effective_mult
        breakdown["type_mult"]     = effective_mult
        breakdown["is_real_table"] = real_table
    elif chunk.chunk_type == "footnote":
        base *= _FOOTNOTE_PENALTY
        breakdown["type_mult"]     = _FOOTNOTE_PENALTY
        breakdown["is_real_table"] = False
    else:
        breakdown["type_mult"]     = 1.0
        breakdown["is_real_table"] = False

    # 8. No-numeric penalty
    no_numeric_pen = 0.0
    if not chunk_has_numeric:
        no_numeric_pen = _NO_NUMERIC_PENALTY
        base -= no_numeric_pen
    breakdown["no_numeric_pen"] = round(no_numeric_pen, 3)

    # 9. No-unit penalty
    no_unit_pen = 0.0
    if chunk_has_kpi_kw and not chunk_has_unit:
        no_unit_pen = _NO_UNIT_PENALTY
        base -= no_unit_pen
    breakdown["no_unit_pen"] = round(no_unit_pen, 3)

    # 10. Penalty for noise-domain terms
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
# Dataclass + page-scoped neighbor stitching (unchanged)
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
                logger.debug(
                    "retrieval.neighbor_skipped_cross_page",
                    anchor_chunk=idx, anchor_page=anchor_page,
                    neighbor_chunk=neighbor_idx, neighbor_page=neighbor.page_number,
                )
                continue
            seen.add(neighbor_idx)
            result.append(ScoredChunk(
                chunk=neighbor, score=_NEIGHBOR_SCORE,
                matched_keywords=[], is_neighbor=True,
            ))

    result.sort(key=lambda s: (s.is_neighbor, -s.score))
    return result


# ---------------------------------------------------------------------------
# RetrievalService (unchanged from v5 — new KPIs handled by filters above)
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

        relevant: list[DocumentChunk] = []
        for chunk in candidate_chunks:
            ok, reason = is_relevant_chunk(chunk.content, kpi.name)
            if ok:
                relevant.append(chunk)
            else:
                logger.debug(
                    "retrieval.chunk_filtered",
                    kpi=kpi.name, chunk_index=chunk.chunk_index,
                    page=chunk.page_number, reason=reason,
                    preview=chunk.content[:80].replace("\n", " "),
                )

        logger.info(
            "retrieval.after_strict_filter",
            kpi=kpi.name, before=len(candidate_chunks), after=len(relevant),
        )

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
                        kpi=kpi.name, chunk_index=chunk.chunk_index,
                        page=chunk.page_number, reason=reason,
                        preview=chunk.content[:80].replace("\n", " "),
                    )

            if fallback_candidates:
                logger.info(
                    "retrieval.unit_fallback_activated",
                    kpi=kpi.name, strict_passed=len(relevant),
                    fallback_added=len(fallback_candidates),
                )
                relevant = relevant + fallback_candidates
            elif not relevant:
                logger.warning("retrieval.full_fallback", kpi=kpi.name)
                relevant = candidate_chunks

        scored: list[ScoredChunk] = []
        for chunk in relevant:
            score, breakdown = _score_chunk_precise(chunk, query_keywords, kpi.name)
            if score <= 0:
                continue
            sc = ScoredChunk(
                chunk=chunk, score=score,
                matched_keywords=breakdown.get("matched_kws", []),
            )
            scored.append(sc)

            logger.debug(
                "retrieval.chunk_scored",
                kpi=kpi.name, chunk_index=chunk.chunk_index,
                page=chunk.page_number, chunk_type=chunk.chunk_type,
                score=breakdown["final"], kw=breakdown["kw_score"],
                exact=breakdown["exact_bonus"], unit=breakdown["unit_bonus"],
                data=breakdown["data_bonus"], answerable=breakdown["answerable"],
                ans_boost=breakdown["ans_boost"], is_real_table=breakdown["is_real_table"],
                no_numeric_pen=breakdown["no_numeric_pen"],
                no_unit_pen=breakdown["no_unit_pen"],
                penalty=breakdown["penalty"], penalty_hits=breakdown["penalty_hits"],
                preview=chunk.content[:80].replace("\n", " "),
            )

        if not scored:
            logger.info("retrieval.no_scored_chunks", kpi=kpi.name)
            return []

        scored.sort(key=lambda s: s.score, reverse=True)
        top_pages = {s.chunk.page_number for s in scored[:3] if s.chunk.page_number}
        for s in scored[3:]:
            if s.chunk.page_number in top_pages:
                s.score += _PAGE_PROXIMITY_BONUS

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
            kpi=kpi.name, candidates=len(candidate_chunks),
            after_filter=len(relevant), scored=len(scored),
            returned=len(final), top_score=round(top[0].score, 3) if top else 0,
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
                    chunk=chunk, score=score,
                    matched_keywords=breakdown.get("matched_kws", []),
                ))

        scored_list.sort(key=lambda s: s.score, reverse=True)
        all_chunks_map = {c.chunk_index: c for c in candidates}
        if scored_list:
            stitched = _stitch_neighbors_page_scoped(scored_list[:k], all_chunks_map)
            return stitched[: k * 2]
        return scored_list[:k]


# ---------------------------------------------------------------------------
# Hybrid retrieval — semantic + keyword fusion (unchanged from v5)
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

        queries = _build_kpi_queries(kpi)
        try:
            semantic_results = emb_service.search_multi_query(
                parsed_document_id=parsed_document_id,
                queries=queries, db=db, top_k=k,
            )
            semantic_scored = {
                chunk.id: ScoredChunk(
                    chunk=chunk, score=sim * 10,
                    matched_keywords=["[semantic]"],
                )
                for chunk, sim in semantic_results
            }
        except Exception as exc:
            logger.warning("retrieval.semantic_failed", kpi=kpi.name, error=str(exc))
            semantic_scored = {}

        keyword_results = super().retrieve(parsed_document_id, kpi, db, top_k)
        keyword_scored = {sc.chunk.id: sc for sc in keyword_results}

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
                    kpi=kpi.name, chunk_index=sc.chunk.chunk_index, reason=reason,
                )
        semantic_scored = filtered_semantic

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
                    chunk=sem.chunk, score=combined,
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
            kpi=kpi.name, semantic=len(semantic_scored),
            keyword=len(keyword_scored), merged=len(merged), returned=len(final),
        )
        return final