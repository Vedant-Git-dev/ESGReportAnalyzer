"""
agents/extraction_agent.py

KPI Extraction Agent — Phase 4 (enhanced).

3-layer extraction designed for naturally-worded ESG PDFs:

  Layer 1 — Regex (deterministic, zero API cost)
    - Broad patterns that catch narrative phrasing, not just tables
    - Tables searched first
    - Value-only patterns (unit inferred from context)
    - Skip LLM only when confidence is very high

  Layer 2 — LLM (fires whenever regex confidence < threshold)
    - Wider retrieval: top-K chunks + full-doc fallback pages
    - Rich prompt with aliases, unit examples, and extraction hints
    - Asks model to scan for ANY mention of the metric

  Layer 3 — Validation
    - Range check, unit consistency, confidence floor
"""
from __future__ import annotations

import re
import uuid
from typing import Optional

from sqlalchemy import or_
from sqlalchemy.orm import Session

from core.config import get_settings
from core.logging_config import get_logger
from models.db_models import DocumentChunk, KPIDefinition, ParsedDocument, Report
from models.schemas import ExtractedKPI
from services.kpi_service import KPIService
from services.llm_service import LLMService
from services.retrieval_service import HybridRetrievalService, RetrievalService, ScoredChunk

logger = get_logger(__name__)

_REGEX_HIGH_CONFIDENCE = 0.88
_LLM_CHUNK_CHAR_LIMIT = 6000

# Patterns that indicate a number is NOT a stock value but a delta/target/intensity.
# Applied as a pre-check on the sentence containing the regex match.
_DELTA_CONTEXT_RE = re.compile(
    r"\b("
    r"reduc(?:ed|es?|tion|ing)|"                  # "reduced scope 1 by", "reduces emissions"
    r"declin(?:ed|e|ing)|"
    r"decreas(?:ed|es?|e|ing)|"
    r"avoid(?:ed|ance|ing|s?)|"
    r"offset(?:ting|s)?|"
    r"sequester(?:ed|ing)?|"
    r"sav(?:ed|ing|ings)|"
    r"target(?:ed|ing|s?)?\s+(?:of|to|at)|"
    r"goal\s+(?:of|to)|"
    r"aim(?:s|ing)?\s+(?:to|for)|"
    r"commit(?:ted|ment)\s+(?:to|of)|"
    r"plan(?:ned|s|ning)?\s+(?:to|for)|"
    r"by\s+(?:fy|20)\d{2}|"
    r"per\s+(?:employee|fte|unit|sqm|sq\.?\s*m|revenue|capita|tonne|kwh)|"
    r"intensit(?:y|ies)|"
    r"baseline\s+(?:of|year|value)|"
    r"compared\s+to\s+(?:fy|20)\d{2}|"
    r"vs\.?\s+(?:fy|20)\d{2}|"
    r"from\s+(?:fy|20)\d{2}\s+(?:to|level)|"
    r"improvement\s+(?:of|in)|"
    r"increase[sd]?\s+by|"                        # "increased by X"
    r"net\s+(?:zero|neutral)|"                    # "net zero target of X"
    r"emission\s+factor"                           # intensity-related
    r")\b",
    re.IGNORECASE,
)


def _get_sentence_context(text: str, match_start: int, window: int = 120) -> str:
    """Extract the sentence around a regex match position."""
    start = max(0, match_start - window)
    end = min(len(text), match_start + window)
    snippet = text[start:end]
    # Find sentence boundaries within the snippet
    sentences = re.split(r"[.!?\n]", snippet)
    # Return the sentence most likely containing the match
    for sent in sentences:
        if str(match_start - start) and len(sent) > 5:
            return sent
    return snippet


def _is_delta_context(text: str, match_start: int) -> bool:
    """Check if the text around a match position describes a delta/target/intensity."""
    context = _get_sentence_context(text, match_start, window=150)
    return bool(_DELTA_CONTEXT_RE.search(context))

# ---------------------------------------------------------------------------
# Unit synonym map
# ---------------------------------------------------------------------------
_UNIT_SYNONYMS: dict[str, str] = {
    "tco2e": "tCO2e", "tco2": "tCO2e", "t co2e": "tCO2e",
    "mtco2e": "tCO2e", "mt co2e": "tCO2e",
    "ktco2e": "tCO2e", "kt co2e": "tCO2e",
    "tonnes co2e": "tCO2e", "tons co2e": "tCO2e",
    "mwh": "MWh", "gwh": "GWh", "twh": "TWh",
    "gj": "GJ", "tj": "TJ", "pj": "PJ",
    "kl": "KL", "kilolitre": "KL", "kiloliter": "KL", "kilo litre": "KL",
    "m3": "m³", "cubic meter": "m³", "cubic metre": "m³",
    "mt": "MT", "metric ton": "MT", "metric tonne": "MT",
    "kg": "kg", "tonne": "MT",
    "%": "%", "percent": "%", "per cent": "%",
    "inr crore": "INR Crore", "crore": "INR Crore", "cr": "INR Crore",
    "inr lakh": "INR Lakh", "lakh": "INR Lakh",
    "number": "count", "nos": "count", "headcount": "count",
}

# KPI-specific aliases used to widen retrieval and prompt context
_KPI_ALIASES: dict[str, list[str]] = {
    "scope_1_emissions": [
        "scope 1", "direct emissions", "direct ghg", "fuel combustion",
        "stationary combustion", "owned vehicles", "fugitive emissions",
    ],
    "scope_2_emissions": [
        "scope 2", "indirect emissions", "purchased electricity",
        "electricity consumption ghg", "market-based", "location-based",
    ],
    "total_ghg_emissions": [
        "total emissions", "total ghg", "scope 1 and 2", "scope 1+2",
        "combined emissions", "carbon footprint", "co2 equivalent",
        "greenhouse gas", "ghg inventory", "carbon neutral",
    ],
    "energy_consumption": [
        "total energy", "energy consumed", "energy use", "energy usage",
        "electricity consumed", "fuel consumed", "energy intensity",
        "gigajoules", "megawatt hours",
    ],
    "renewable_energy_percentage": [
        "renewable energy", "solar energy", "wind energy", "clean energy",
        "green energy", "renewable electricity", "non-fossil", "re share",
        "percent renewable", "% renewable",
    ],
    "water_consumption": [
        "water consumed", "water usage", "water use", "water withdrawal",
        "freshwater", "water recycled", "water intensity", "kilolitres",
    ],
    "waste_generated": [
        "waste generated", "total waste", "solid waste", "hazardous waste",
        "non-hazardous waste", "waste disposed", "waste diverted",
    ],
    "employee_count": [
        "total employees", "total workforce", "number of employees",
        "headcount", "fte", "full time", "permanent employees",
        "workforce strength", "employee base",
    ],
    "women_in_workforce_percentage": [
        "women", "female", "gender diversity", "women employees",
        "female employees", "gender ratio", "women in workforce",
    ],
    "csr_spend": [
        "csr expenditure", "csr spend", "csr investment", "social spend",
        "community investment", "corporate social responsibility spend",
    ],
}


def _normalise_unit(unit_str: str) -> str:
    if not unit_str:
        return unit_str
    key = unit_str.lower().strip()
    return _UNIT_SYNONYMS.get(key, unit_str.strip())


def _parse_number(s: str) -> Optional[float]:
    """Parse number strings including Indian number format (1,00,000)."""
    if not s:
        return None
    try:
        cleaned = s.replace(",", "").replace(" ", "").strip()
        return float(cleaned)
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Layer 1 — Regex (broad, narrative-aware patterns)
# ---------------------------------------------------------------------------

# Indian number format: 8,50,434 or 19,55,525 (lakh-crore grouping)
# Also handles footnote markers: 8,745(2) → capture 8,745
_NUM = r"([\d,]+(?:\.\d+)?)(?:\(\d+\))?"   # number optionally followed by footnote marker
_WS  = r"[\s:–\-|]*"
_UNIT_PAT = (
    r"(tco2e?|t\s*co2e?|mt\s*co2e?|kt\s*co2e?|tonnes?\s*co2e?|"
    r"mwh|gwh|gj|tj|kl|kilolitr\w*|m3|mt|metric\s*tonn?\w*|"
    r"%|percent|crore|lakh|number|nos|headcount)"
)

# Patterns for "label\nvalue" block structure (fitz splits label and value by \n)
# These handle the most common BRSR/ESG PDF block format:
#   "Total energy consumed (A+B+C+D+E+F)(1)\n8,50,434\n8,39,448"
_BLOCK_PATTERNS_WITH_UNIT_IN_LABEL = [
    # Energy: "total energy consumed ... (in GJ)\n<value>"
    (r"total\s+energy\s+consumed[^\n]*\n" + _NUM, "GJ"),
    (r"total\s+energy\s+consumption[^\n]*\n" + _NUM, "GJ"),
    # Water consumption: "total volume of water consumption[^\n]*\n<value>"
    (r"total\s+volume\s+of\s+water\s+consumption[^\n]*\n" + _NUM, "KL"),
    (r"total\s+water\s+consumption[^\n]*\n" + _NUM, "KL"),
    # Waste total: "Total (A + B + ... + H)\n<value>"
    (r"^total\s*\([a-h\s+]+\)\s*\n" + _NUM, "MT"),
    (r"total\s+waste\s+generated[^\n]*\n[^\n]*\n" + _NUM, "MT"),
    # Scope 1: "Total Scope 1 emissions...\nMetric tonnes...\n<value>"
    (r"total\s+scope\s*1\s+emissions[^\n]*\n[^\n]*\n" + _NUM, "tCO2e"),
    (r"total\s+scope\s*1\s+emissions[^\n]*equivalent\n" + _NUM, "tCO2e"),
    # Scope 2: same pattern
    (r"total\s+scope\s*2\s+emissions[^\n]*\n[^\n]*\n" + _NUM, "tCO2e"),
    (r"total\s+scope\s*2\s+emissions[^\n]*equivalent\n" + _NUM, "tCO2e"),
]

# Built-in broad patterns (inline, no newlines)
_BROAD_PATTERNS = [
    rf"(?:was|is|were|of|:)\s*{_NUM}\s*{_UNIT_PAT}",
    rf"{_NUM}\s*{_UNIT_PAT}\s*(?:in|for|during|fy|fiscal)",
    rf"\|\s*{_NUM}\s*\|\s*{_UNIT_PAT}",
    rf"(?:total(?:led|s)?|amount(?:ed|s)?|reached|stood at|equat\w+)\s*{_WS}{_NUM}\s*{_UNIT_PAT}",
]


def _try_block_patterns(chunk, kpi: "KPIDefinition") -> Optional["ExtractedKPI"]:
    """
    Try block-aware patterns where unit is in the label and value is on the next line.
    This handles: "Total energy consumed (in GJ)\n8,50,434\n8,39,448"
    """
    text = chunk.content

    # Only apply to KPIs where we know the unit from the label
    block_patterns_for_kpi = {
        "energy_consumption": [
            # Grand total: requires A in parenthetical — excludes subtotals like (D + E + F)
            r"total\s+energy\s+consumed\s*\(A[^)]*D[^)]*F\)[^\n]*\n([\d,]+(?:\.\d+)?)(?:\(\d+\))?",
            r"total\s+energy\s+consumed\s*\((?:A[^)]*)\)[^\n]*\n([\d,]+(?:\.\d+)?)(?:\(\d+\))?",
        ],
        "water_consumption": [
            r"total\s+volume\s+of\s+water\s+consumption[^\n]*\n([\d,]+(?:\.\d+)?)(?:\(\d+\))?",
            r"total\s+water\s+consumption[^\n]*\n([\d,]+(?:\.\d+)?)(?:\(\d+\))?",
        ],
        "waste_generated": [
            r"^total\s*\([a-h\s\+]+\)\s*\n([\d,]+(?:\.\d+)?)(?:\(\d+\))?",
            r"total\s+\(A\s*\+\s*B[^\n]*\)\s*\n([\d,]+(?:\.\d+)?)(?:\(\d+\))?",
        ],
        "scope_1_emissions": [
            r"total\s+scope\s*1\s+emissions[\s\S]{0,120}?equivalent\n([\d,]+(?:\.\d+)?)(?:\(\d+\))?",
            r"total\s+scope\s*1\s+emissions[\s\S]{0,80}?\n\n?([\d,]+(?:\.\d+)?)(?:\(\d+\))?",
        ],
        "scope_2_emissions": [
            r"total\s+scope\s*2\s+emissions[\s\S]{0,120}?equivalent\n([\d,]+(?:\.\d+)?)(?:\(\d+\))?",
            r"total\s+scope\s*2\s+emissions[\s\S]{0,80}?\n\n?([\d,]+(?:\.\d+)?)(?:\(\d+\))?",
        ],
        "total_ghg_emissions": [
            # Sum is not directly stated — skip block pattern, LLM handles this
        ],
    }

    patterns_for_this_kpi = block_patterns_for_kpi.get(kpi.name, [])
    if not patterns_for_this_kpi:
        return None

    expected_unit = kpi.expected_unit

    for pat_str in patterns_for_this_kpi:
        try:
            m = re.search(pat_str, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if not m:
                continue
            raw_val = m.group(1).strip()
            value = _parse_number(raw_val)
            if value is None or value <= 0:
                continue

            # Sanity: Indian format numbers like 8,50,434 → 850434
            # _parse_number strips commas so 8,50,434 → 850434.0 ✓

            # Skip intensity values (very small numbers like 0.000000522)
            if value < 0.1 and kpi.name not in ("renewable_energy_percentage",):
                continue

            confidence = 0.93 if chunk.chunk_type == "table" else 0.88
            logger.info("extraction.block_pattern_hit",
                       kpi=kpi.name, value=value, unit=expected_unit,
                       chunk_type=chunk.chunk_type, page=chunk.page_number)

            return ExtractedKPI(
                kpi_name=kpi.name,
                raw_value=raw_val,
                normalized_value=value,
                unit=expected_unit,
                extraction_method="regex",
                confidence=confidence,
                source_chunk_id=chunk.id,
                validation_passed=True,
            )
        except re.error:
            continue

    return None


def _try_regex(
    chunks: list[DocumentChunk],
    kpi: KPIDefinition,
) -> Optional[ExtractedKPI]:
    """
    Try KPI-specific patterns first, then broad fallback patterns.
    Tables before text. Returns best match (highest confidence).
    """
    kpi_patterns = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in (kpi.regex_patterns or [])]
    broad_patterns = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _BROAD_PATTERNS]

    sorted_chunks = sorted(chunks, key=lambda c: 0 if c.chunk_type == "table" else 1)
    best: Optional[ExtractedKPI] = None

    for chunk in sorted_chunks:
        text = chunk.content

        # --- Block-aware patterns (label\nvalue — handles BRSR/ESG block format) ---
        block_result = _try_block_patterns(chunk, kpi)
        if block_result:
            if _is_delta_context(text, 0):
                pass  # skip delta blocks
            else:
                if best is None or block_result.confidence > best.confidence:
                    best = block_result
                if block_result.confidence >= _REGEX_HIGH_CONFIDENCE:
                    return best

        # Try KPI-specific patterns (higher confidence)
        for pattern in kpi_patterns:
            match = pattern.search(text)
            if not match or len(match.groups()) < 2:
                continue
            # Reject if the match sits in a delta/target/intensity sentence
            if _is_delta_context(text, match.start()):
                logger.debug("extraction.regex_delta_rejected",
                           kpi=kpi.name, snippet=text[max(0,match.start()-60):match.start()+60])
                continue
            value = _parse_number(match.group(1))
            if value is None:
                continue
            unit = _normalise_unit(match.group(2))
            confidence = 0.92 if chunk.chunk_type == "table" else 0.80
            logger.info("extraction.regex_hit", kpi=kpi.name, value=value, unit=unit,
                       chunk_type=chunk.chunk_type, page=chunk.page_number, pattern="specific")
            result = ExtractedKPI(
                kpi_name=kpi.name, raw_value=match.group(1),
                normalized_value=value, unit=unit,
                extraction_method="regex", confidence=confidence,
                source_chunk_id=chunk.id, validation_passed=True,
            )
            if best is None or result.confidence > best.confidence:
                best = result
            if confidence >= _REGEX_HIGH_CONFIDENCE:
                return best

        # Try broad patterns against chunks that mention relevant keywords
        kpi_keywords = set(
            w.lower() for kw in (kpi.retrieval_keywords or [])
            for w in kw.split()
        )
        kpi_keywords.update(
            w.lower() for alias in _KPI_ALIASES.get(kpi.name, [])
            for w in alias.split()
        )
        text_lower = text.lower()
        if not any(kw in text_lower for kw in kpi_keywords):
            continue

        for pattern in broad_patterns:
            match = pattern.search(text)
            if not match or len(match.groups()) < 2:
                continue
            # Reject delta/target/intensity context
            if _is_delta_context(text, match.start()):
                continue
            value = _parse_number(match.group(1))
            if value is None:
                continue
            unit = _normalise_unit(match.group(2))
            if kpi.expected_unit and unit.lower() != kpi.expected_unit.lower():
                continue
            confidence = 0.65 if chunk.chunk_type == "table" else 0.55
            logger.info("extraction.regex_broad_hit", kpi=kpi.name, value=value, unit=unit,
                       chunk_type=chunk.chunk_type, page=chunk.page_number)
            result = ExtractedKPI(
                kpi_name=kpi.name, raw_value=match.group(1),
                normalized_value=value, unit=unit,
                extraction_method="regex", confidence=confidence,
                source_chunk_id=chunk.id, validation_passed=True,
            )
            if best is None or result.confidence > best.confidence:
                best = result

    return best


# ---------------------------------------------------------------------------
# Layer 2 — LLM (richer prompt + wider context)
# ---------------------------------------------------------------------------

def _build_chunks_text(scored_chunks: list[ScoredChunk], max_chars: int = _LLM_CHUNK_CHAR_LIMIT) -> str:
    """
    Build LLM prompt text from scored chunks.
    - Primary chunks (non-neighbors) first
    - Then neighbors for context
    - Tables labeled [TABLE], others labeled [Page N]
    - Hard cap at max_chars
    """
    # Sort: primary high-score first, then neighbors
    primary = [sc for sc in scored_chunks if not sc.is_neighbor]
    neighbors = [sc for sc in scored_chunks if sc.is_neighbor]
    ordered = primary + neighbors

    parts = []
    total = 0
    seen_indices: set = set()

    for sc in ordered:
        idx = sc.chunk.chunk_index
        if idx in seen_indices:
            continue
        seen_indices.add(idx)

        if sc.chunk.chunk_type == "table":
            prefix = f"[TABLE | Page {sc.chunk.page_number}]\n"
        else:
            prefix = f"[Page {sc.chunk.page_number}]\n"

        block = f"{prefix}{sc.chunk.content}\n\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)

    return "".join(parts).strip()


def _try_llm(
    scored_chunks: list[ScoredChunk],
    kpi: KPIDefinition,
    llm: LLMService,
    report_year: int,
) -> Optional[ExtractedKPI]:
    if not scored_chunks:
        return None

    aliases = _KPI_ALIASES.get(kpi.name, [])
    alias_str = ", ".join(aliases) if aliases else "none"
    chunks_text = _build_chunks_text(scored_chunks)

    result = llm.extract_kpi(
        kpi_name=kpi.name,
        kpi_display=kpi.display_name,
        expected_unit=kpi.expected_unit,
        chunks_text=chunks_text,
        aliases=alias_str,
        report_year=report_year,
    )

    if result is None:
        return None

    raw_value = result.get("value")
    if raw_value is None:
        logger.info("extraction.llm_not_found", kpi=kpi.name, notes=result.get("notes"))
        return None

    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        logger.warning("extraction.llm_bad_value", kpi=kpi.name, raw=raw_value)
        return None

    unit = _normalise_unit(str(result.get("unit") or kpi.expected_unit))
    confidence = float(result.get("confidence") or 0.5)
    year = result.get("year") or report_year
    source_chunk_id = scored_chunks[0].chunk.id if scored_chunks else None

    logger.info("extraction.llm_hit", kpi=kpi.name, value=value, unit=unit, confidence=confidence)

    return ExtractedKPI(
        kpi_name=kpi.name, raw_value=str(raw_value),
        normalized_value=value, unit=unit, year=year,
        extraction_method="llm", confidence=confidence,
        source_chunk_id=source_chunk_id,
        validation_passed=True,
        validation_notes=result.get("notes"),
    )


# ---------------------------------------------------------------------------
# Layer 3 — Validation
# ---------------------------------------------------------------------------

def _validate(extracted: ExtractedKPI, kpi: KPIDefinition) -> ExtractedKPI:
    notes: list[str] = []
    val = extracted.normalized_value

    if val is not None:
        if kpi.valid_min is not None and val < kpi.valid_min:
            extracted.validation_passed = False
            notes.append(f"Value {val} below minimum {kpi.valid_min}")
        if kpi.valid_max is not None and val > kpi.valid_max:
            extracted.validation_passed = False
            notes.append(f"Value {val} exceeds maximum {kpi.valid_max}")

    if extracted.unit and kpi.expected_unit:
        if extracted.unit.lower() != kpi.expected_unit.lower():
            notes.append(f"Unit mismatch: got '{extracted.unit}', expected '{kpi.expected_unit}'")

    if extracted.confidence is not None and extracted.confidence < 0.4:
        extracted.validation_passed = False
        notes.append(f"Confidence too low: {extracted.confidence:.2f}")

    if notes:
        existing = extracted.validation_notes or ""
        extracted.validation_notes = existing + (" | " if existing else "") + " | ".join(notes)
        if not extracted.validation_passed:
            logger.warning("extraction.validation_failed", kpi=kpi.name, value=val, notes=notes)

    return extracted


# ---------------------------------------------------------------------------
# Wider retrieval — fallback to summary/overview pages
# ---------------------------------------------------------------------------

def _get_wider_chunks(
    parsed_doc: ParsedDocument,
    kpi: KPIDefinition,
    db: Session,
    already_retrieved_ids: set,
    extra_k: int = 5,
) -> list[ScoredChunk]:
    """
    Fallback retrieval: scan all tables + first 10 pages for KPI aliases.
    Returns chunks not already in the primary retrieval set.
    """
    aliases = _KPI_ALIASES.get(kpi.name, [])
    all_keywords = list(kpi.retrieval_keywords or []) + aliases

    if not all_keywords:
        return []

    keyword_filters = [
        DocumentChunk.keywords.ilike(f"%{kw.lower().split()[0]}%")
        for kw in all_keywords
    ]

    extra_chunks = (
        db.query(DocumentChunk)
        .filter(
            DocumentChunk.parsed_document_id == parsed_doc.id,
            DocumentChunk.id.notin_(list(already_retrieved_ids)),
            or_(*keyword_filters),
        )
        .order_by(
            # tables first, then by page
            DocumentChunk.chunk_type.desc(),
            DocumentChunk.page_number,
        )
        .limit(extra_k)
        .all()
    )

    return [ScoredChunk(chunk=c, score=0.3, matched_keywords=[]) for c in extra_chunks]


# ---------------------------------------------------------------------------
# Public Agent
# ---------------------------------------------------------------------------

class ExtractionAgent:

    def __init__(self) -> None:
        self.settings = get_settings()
        self.retrieval = HybridRetrievalService()
        self.llm = LLMService()
        self.kpi_service = KPIService()

    def extract_all(
        self,
        report_id: uuid.UUID,
        db: Session,
        kpi_names: Optional[list[str]] = None,
        fallback_search: bool = True,
        max_fallback_reports: int = 3,
    ) -> list[ExtractedKPI]:
        """
        Extract all active KPIs for a report.

        Args:
            report_id:            UUID of a parsed report
            db:                   Active session
            kpi_names:            Optional filter — only extract these KPIs
            fallback_search:      If True, search other reports when a KPI is not found
            max_fallback_reports: Max additional reports to try per missing KPI
        """
        report = db.query(Report).filter(Report.id == report_id).first()
        if not report:
            raise ValueError(f"Report {report_id} not found")

        parsed_doc = (
            db.query(ParsedDocument)
            .filter(ParsedDocument.report_id == report_id)
            .order_by(ParsedDocument.parsed_at.desc())
            .first()
        )
        if not parsed_doc:
            raise ValueError(f"No parse cache for report {report_id}. Run parse first.")

        kpis = self.kpi_service.get_by_names(kpi_names, db) if kpi_names else self.kpi_service.get_all_active(db)

        logger.info("extraction.start", report_id=str(report_id), kpis=len(kpis))

        results: list[ExtractedKPI] = []
        not_found_kpis: list[KPIDefinition] = []

        for kpi in kpis:
            extracted = self._extract_one(kpi=kpi, parsed_doc=parsed_doc, report=report, db=db)
            results.append(extracted)

            if extracted.normalized_value is not None:
                self.kpi_service.store_record(
                    company_id=report.company_id,
                    report_id=report_id,
                    report_year=report.report_year,
                    kpi_definition_id=kpi.id,
                    extracted=extracted,
                    source_chunk_id=extracted.source_chunk_id,
                    db=db,
                )
            else:
                not_found_kpis.append(kpi)

        logger.info(
            "extraction.complete",
            report_id=str(report_id),
            total_kpis=len(kpis)
        )
        return results

    def _extract_one(
        self,
        kpi: KPIDefinition,
        parsed_doc: ParsedDocument,
        report: Report,
        db: Session,
    ) -> ExtractedKPI:
        logger.debug("extraction.kpi_start", kpi=kpi.name)

        # Primary retrieval
        scored_chunks = self.retrieval.retrieve(
            parsed_document_id=parsed_doc.id,
            kpi=kpi,
            db=db,
        )
        chunks = [sc.chunk for sc in scored_chunks]

        # Layer 1: Regex on primary chunks
        extracted = _try_regex(chunks, kpi)

        if extracted and extracted.confidence >= _REGEX_HIGH_CONFIDENCE:
            logger.info("extraction.regex_confident", kpi=kpi.name, value=extracted.normalized_value)
            return _validate(extracted, kpi)

        # Widen retrieval for LLM (add alias-based extra chunks)
        retrieved_ids = {sc.chunk.id for sc in scored_chunks}
        extra_chunks = _get_wider_chunks(parsed_doc, kpi, db, retrieved_ids, extra_k=5)
        all_scored = scored_chunks + extra_chunks

        # Layer 2: LLM
        if not extracted or extracted.confidence < _REGEX_HIGH_CONFIDENCE:
            llm_result = _try_llm(all_scored, kpi, self.llm, report.report_year)
            if llm_result:
                # Take LLM result if it has higher confidence than regex
                if extracted is None or llm_result.confidence > extracted.confidence:
                    extracted = llm_result

        if extracted is None or extracted.normalized_value is None:
            logger.info("extraction.not_found", kpi=kpi.name)
            return ExtractedKPI(
                kpi_name=kpi.name,
                extraction_method="regex",
                confidence=0.0,
                validation_passed=False,
                validation_notes="Not found by regex or LLM",
            )

        # Layer 3: Validation
        return _validate(extracted, kpi)