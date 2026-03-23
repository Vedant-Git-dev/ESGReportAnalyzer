"""
agents/extraction_agent.py

KPI Extraction Agent — Phase 4.

3-layer extraction per KPI:

  Layer 1 — Regex (deterministic, zero API cost)
    - Try each regex pattern from kpi_definitions.regex_patterns
    - Tables searched first (higher signal density)
    - If value found with high confidence → skip LLM entirely

  Layer 2 — LLM (only when regex fails)
    - Retrieve top-K chunks via RetrievalService
    - Send ONLY those chunks (never full doc)
    - Parse JSON response

  Layer 3 — Validation
    - Unit consistency check
    - Range check (valid_min / valid_max from KPI definition)
    - Reject outliers, flag low confidence

Output: list[ExtractedKPI] per report
"""

from __future__ import annotations

import re
import uuid
from typing import Optional

from sqlalchemy.orm import Session

from core.config import get_settings
from core.logging_config import get_logger
from models.db_models import DocumentChunk, KPIDefinition, ParsedDocument, Report
from models.schemas import ExtractedKPI
from services.kpi_service import KPIService
from services.llm_service import LLMService
from services.retrieval_service import RetrievalService, ScoredChunk

logger = get_logger(__name__)

# Confidence threshold above which we skip LLM even if regex found something
_REGEX_HIGH_CONFIDENCE = 0.85

# Unit synonym map — normalise common variations before storing
_UNIT_SYNONYMS: dict[str, str] = {
    "tco2e": "tCO2e",
    "tco2": "tCO2e",
    "mtco2e": "tCO2e",   # will need value * 1e6 — handled in normalization agent
    "ktco2e": "tCO2e",   # value * 1e3
    "mwh": "MWh",
    "gwh": "GWh",
    "gj": "GJ",
    "tj": "TJ",
    "kl": "KL",
    "kilolitre": "KL",
    "kiloliter": "KL",
    "m3": "m³",
    "mt": "MT",
    "metric ton": "MT",
    "metric tonne": "MT",
    "%": "%",
    "percent": "%",
    "inr crore": "INR Crore",
    "crore": "INR Crore",
}


def _normalise_unit(unit_str: str) -> str:
    if not unit_str:
        return unit_str
    return _UNIT_SYNONYMS.get(unit_str.lower().strip(), unit_str.strip())


def _parse_number(s: str) -> Optional[float]:
    """Parse a number string with commas and spaces."""
    try:
        return float(s.replace(",", "").replace(" ", ""))
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Layer 1 — Regex extraction
# ---------------------------------------------------------------------------

def _try_regex(
    chunks: list[DocumentChunk],
    kpi: KPIDefinition,
) -> Optional[ExtractedKPI]:
    """
    Try each regex pattern against each chunk.
    Tables are tried first. Returns first confident match.
    """
    patterns = kpi.regex_patterns or []
    if not patterns:
        return None

    # Sort: tables first
    sorted_chunks = sorted(chunks, key=lambda c: 0 if c.chunk_type == "table" else 1)

    for chunk in sorted_chunks:
        text = chunk.content.lower()
        for pattern_str in patterns:
            try:
                pattern = re.compile(pattern_str, re.IGNORECASE | re.DOTALL)
                match = pattern.search(text)
                if not match:
                    continue

                # Group 1 = value, Group 2 = unit (both required)
                if len(match.groups()) < 2:
                    continue

                raw_val_str = match.group(1)
                raw_unit_str = match.group(2)

                value = _parse_number(raw_val_str)
                if value is None:
                    continue

                unit = _normalise_unit(raw_unit_str)
                confidence = 0.9 if chunk.chunk_type == "table" else 0.75

                logger.info(
                    "extraction.regex_hit",
                    kpi=kpi.name,
                    value=value,
                    unit=unit,
                    chunk_type=chunk.chunk_type,
                    page=chunk.page_number,
                )

                return ExtractedKPI(
                    kpi_name=kpi.name,
                    raw_value=raw_val_str,
                    normalized_value=value,
                    unit=unit,
                    extraction_method="regex",
                    confidence=confidence,
                    source_chunk_id=chunk.id,
                    validation_passed=True,
                )
            except re.error as exc:
                logger.warning("extraction.bad_regex", pattern=pattern_str, error=str(exc))
                continue

    return None


# ---------------------------------------------------------------------------
# Layer 2 — LLM extraction
# ---------------------------------------------------------------------------

def _build_chunks_text(scored_chunks: list[ScoredChunk], max_chars: int = 4000) -> str:
    """
    Concatenate top-K chunks into a single prompt string.
    Tables are prefixed with [TABLE] for LLM context.
    Hard cap at max_chars to stay within token limits.
    """
    parts = []
    total = 0
    for sc in scored_chunks:
        prefix = "[TABLE]\n" if sc.chunk.chunk_type == "table" else ""
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
    """
    Call LLM with top-K chunks. Parse and return result or None.
    """
    if not scored_chunks:
        logger.info("extraction.llm_skip_no_chunks", kpi=kpi.name)
        return None

    chunks_text = _build_chunks_text(scored_chunks)
    result = llm.extract_kpi(
        kpi_name=kpi.name,
        kpi_display=kpi.display_name,
        expected_unit=kpi.expected_unit,
        chunks_text=chunks_text,
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

    # Use the top chunk as source reference
    source_chunk_id = scored_chunks[0].chunk.id if scored_chunks else None

    logger.info(
        "extraction.llm_hit",
        kpi=kpi.name,
        value=value,
        unit=unit,
        confidence=confidence,
    )

    return ExtractedKPI(
        kpi_name=kpi.name,
        raw_value=str(raw_value),
        normalized_value=value,
        unit=unit,
        year=year,
        extraction_method="llm",
        confidence=confidence,
        source_chunk_id=source_chunk_id,
        validation_passed=True,
        validation_notes=result.get("notes"),
    )


# ---------------------------------------------------------------------------
# Layer 3 — Validation
# ---------------------------------------------------------------------------

def _validate(extracted: ExtractedKPI, kpi: KPIDefinition) -> ExtractedKPI:
    """
    Apply validation rules. Modifies extracted in-place.
    Rules:
      1. Range check (valid_min / valid_max)
      2. Unit consistency
      3. Flag suspiciously low confidence
    """
    notes: list[str] = []

    # Range check
    val = extracted.normalized_value
    if val is not None:
        if kpi.valid_min is not None and val < kpi.valid_min:
            extracted.validation_passed = False
            notes.append(f"Value {val} below minimum {kpi.valid_min}")
        if kpi.valid_max is not None and val > kpi.valid_max:
            extracted.validation_passed = False
            notes.append(f"Value {val} exceeds maximum {kpi.valid_max}")

    # Unit consistency
    if extracted.unit and kpi.expected_unit:
        if extracted.unit.lower() != kpi.expected_unit.lower():
            # Not a hard fail — units can be converted by normalization agent
            notes.append(f"Unit mismatch: got '{extracted.unit}', expected '{kpi.expected_unit}'")

    # Confidence flag
    if extracted.confidence is not None and extracted.confidence < 0.4:
        extracted.validation_passed = False
        notes.append(f"Confidence too low: {extracted.confidence:.2f}")

    if notes:
        extracted.validation_notes = (extracted.validation_notes or "") + " | ".join(notes)
        if not extracted.validation_passed:
            logger.warning(
                "extraction.validation_failed",
                kpi=kpi.name,
                value=val,
                notes=notes,
            )

    return extracted


# ---------------------------------------------------------------------------
# Public Agent
# ---------------------------------------------------------------------------

class ExtractionAgent:
    """
    Extracts all active KPIs from a parsed report.
    Follows: regex → LLM fallback → validation per KPI.
    """

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
        """
        Extract all active KPIs for a report.

        Args:
            report_id:  UUID of a parsed report
            db:         Active session
            kpi_names:  Optional filter — only extract these KPIs

        Returns:
            List of ExtractedKPI (includes failed/not-found with validation_passed=False)
        """
        # Load report + parsed doc
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

        # Load KPI definitions
        if kpi_names:
            kpis = self.kpi_service.get_by_names(kpi_names, db)
        else:
            kpis = self.kpi_service.get_all_active(db)

        logger.info(
            "extraction.start",
            report_id=str(report_id),
            kpis=len(kpis),
            parsed_doc_id=str(parsed_doc.id),
        )

        results: list[ExtractedKPI] = []

        for kpi in kpis:
            extracted = self._extract_one(
                kpi=kpi,
                parsed_doc=parsed_doc,
                report=report,
                db=db,
            )
            results.append(extracted)

            # Persist immediately (append-only)
            if extracted.normalized_value is not None:
                source_chunk_id = extracted.source_chunk_id
                self.kpi_service.store_record(
                    company_id=report.company_id,
                    report_id=report_id,
                    report_year=report.report_year,
                    kpi_definition_id=kpi.id,
                    extracted=extracted,
                    source_chunk_id=source_chunk_id,
                    db=db,
                )

        found = sum(1 for r in results if r.normalized_value is not None)
        logger.info(
            "extraction.complete",
            report_id=str(report_id),
            total_kpis=len(kpis),
            found=found,
            not_found=len(kpis) - found,
        )

        return results

    def _extract_one(
        self,
        kpi: KPIDefinition,
        parsed_doc: ParsedDocument,
        report: Report,
        db: Session,
    ) -> ExtractedKPI:
        """Run the 3-layer extraction pipeline for a single KPI."""

        logger.debug("extraction.kpi_start", kpi=kpi.name)

        # --- Retrieve relevant chunks ---
        scored_chunks = self.retrieval.retrieve(
            parsed_document_id=parsed_doc.id,
            kpi=kpi,
            db=db,
        )

        # Flatten to just chunk objects for regex layer
        chunks = [sc.chunk for sc in scored_chunks]

        # --- Layer 1: Regex ---
        extracted = _try_regex(chunks, kpi)

        if extracted and extracted.confidence >= _REGEX_HIGH_CONFIDENCE:
            logger.info("extraction.regex_confident", kpi=kpi.name, value=extracted.normalized_value)
            return _validate(extracted, kpi)

        # --- Layer 2: LLM (if regex failed or low confidence) ---
        if not extracted:
            llm_result = _try_llm(scored_chunks, kpi, self.llm, report.report_year)
            if llm_result:
                extracted = llm_result
            else:
                # Total miss — return empty record
                logger.info("extraction.not_found", kpi=kpi.name)
                return ExtractedKPI(
                    kpi_name=kpi.name,
                    extraction_method="regex",
                    confidence=0.0,
                    validation_passed=False,
                    validation_notes="Not found by regex or LLM",
                )

        # --- Layer 3: Validation ---
        return _validate(extracted, kpi)