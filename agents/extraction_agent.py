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
from services.retrieval_service import RetrievalService, ScoredChunk

logger = get_logger(__name__)

_REGEX_HIGH_CONFIDENCE = 0.88
_LLM_CHUNK_CHAR_LIMIT = 6000   # increased from 4000

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

# Generic number pattern: matches 1,00,000 or 12.34 or 1234
_NUM = r"([\d,]+(?:\.\d+)?)"
_WS = r"[\s:–\-|]*"          # flexible separator
_UNIT_PAT = r"(tco2e?|t\s*co2e?|mt\s*co2e?|kt\s*co2e?|tonnes?\s*co2e?|mwh|gwh|gj|tj|kl|kilolitr\w*|m3|mt|metric\s*tonn?\w*|%|percent|crore|lakh|number|nos|headcount)"

# Built-in broad patterns applied to ALL KPIs as a fallback
_BROAD_PATTERNS = [
    # "was 12,345 tCO2e" / "of 12,345 tCO2e"
    rf"(?:was|is|were|of|:)\s*{_NUM}\s*{_UNIT_PAT}",
    # "12,345 tCO2e in FY" / "12,345 tCO2e for"
    rf"{_NUM}\s*{_UNIT_PAT}\s*(?:in|for|during|fy|fiscal)",
    # table cell: "| 12,345 | tCO2e |"
    rf"\|\s*{_NUM}\s*\|\s*{_UNIT_PAT}",
    # "totalled 12,345 tCO2e"
    rf"(?:total(?:led|s)?|amount(?:ed|s)?|reached|stood at|equat\w+)\s*{_WS}{_NUM}\s*{_UNIT_PAT}",
]


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

        # Try KPI-specific patterns (higher confidence)
        for pattern in kpi_patterns:
            match = pattern.search(text)
            if not match or len(match.groups()) < 2:
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
            value = _parse_number(match.group(1))
            if value is None:
                continue
            unit = _normalise_unit(match.group(2))
            # Only accept broad match if unit is plausible for this KPI
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
        self.retrieval = RetrievalService()
        self.llm = LLMService()
        self.kpi_service = KPIService()

    def extract_all(
        self,
        report_id: uuid.UUID,
        db: Session,
        kpi_names: Optional[list[str]] = None,
    ) -> list[ExtractedKPI]:
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

        found = sum(1 for r in results if r.normalized_value is not None)
        logger.info("extraction.complete", report_id=str(report_id),
                   total_kpis=len(kpis), found=found, not_found=len(kpis) - found)
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