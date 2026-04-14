"""
services/kpi_cache_service.py

KPI-Level Cache and Fallback Selection Service.

Replaces the old report-level selection (pick one report, read all KPIs from it)
with a KPI-level approach:

  1. For every requested KPI, independently query kpi_records.
  2. Apply report-type priority (Integrated > BRSR > ESG) per KPI.
  3. Only extract KPIs that have no valid cached value.
  4. Store newly extracted values with a dedup guard on
     (company_id, report_id, kpi_definition_id).

Public API
----------
KPICacheService.load(company_id, fy, kpi_names, db)
    -> {"kpis": {name: record}, "missing": [name]}

KPICacheService.store(company_id, report_id, fy, kpi_records, revenue_result, db)
    -> None  (idempotent, dedup-safe)

KPICacheService.select_best_per_kpi(company_id, fy, kpi_names, db)
    -> {name: {"value", "unit", "method", "confidence", "report_type"}}
    Applies Integrated > BRSR > ESG priority per KPI.
"""
from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import case

from core.logging_config import get_logger

logger = get_logger(__name__)

# Report-type priority: lower integer = higher priority.
# Integrated is most complete for large Indian conglomerates;
# BRSR is SEBI-mandated and has standardised disclosures;
# ESG is the catch-all fallback.
REPORT_TYPE_PRIORITY: dict[str, int] = {
    "Integrated": 0,
    "BRSR":       1,
    "ESG":        2,
}

# Minimum confidence to treat a cached value as valid.
# Values below this threshold are treated as cache misses and re-extracted.
MIN_CONFIDENCE = 0.40

# Plausibility ranges per KPI (min, max) in canonical units.
# Values outside this range are silently dropped — they are extraction errors.
KPI_PLAUSIBILITY: dict[str, tuple[float, float]] = {
    "scope_1_emissions":   (1,           5_000_000),
    "scope_2_emissions":   (1,           5_000_000),
    "total_ghg_emissions": (1,          10_000_000),
    "energy_consumption":  (1_000,     500_000_000),
    "water_consumption":   (100,       100_000_000),
    "waste_generated":     (0.1,           500_000),
    "employee_count":      (1,           5_000_000),
    "renewable_energy_percentage": (0,         100),
    "women_in_workforce_percentage": (0,        100),
}


def _is_plausible(kpi_name: str, value: float) -> bool:
    """Return True when value falls within the known plausible range."""
    limits = KPI_PLAUSIBILITY.get(kpi_name)
    if limits is None:
        return True
    lo, hi = limits
    return lo <= value <= hi


def _type_priority_expr(Report):
    """SQLAlchemy CASE expression for report-type priority ordering."""
    return case(
        (Report.report_type == "Integrated", 0),
        (Report.report_type == "BRSR",       1),
        (Report.report_type == "ESG",        2),
        else_=99,
    )


class KPICacheService:
    """
    Stateless service.  All methods accept an open SQLAlchemy Session.
    Thread-safe as long as each request gets its own session (standard).
    """

    # ------------------------------------------------------------------
    # Core read: per-KPI priority selection
    # ------------------------------------------------------------------

    def select_best_per_kpi(
        self,
        company_id: uuid.UUID,
        fy: int,
        kpi_names: list[str],
        db: Session,
    ) -> dict[str, dict]:
        """
        For each KPI in kpi_names, find the best cached record applying
        Integrated > BRSR > ESG priority, then recency as tiebreaker.

        Returns only KPIs that have a valid (non-null, plausible, confident)
        cached value.  KPIs with no valid record are absent from the result.

        Args:
            company_id: UUID of the company row.
            fy:         Fiscal year (integer, e.g. 2025).
            kpi_names:  List of KPI definition names to look up.
            db:         Active SQLAlchemy session.

        Returns:
            {
                kpi_name: {
                    "value":       float,
                    "unit":        str,
                    "method":      str,
                    "confidence":  float,
                    "report_type": str,   # source report type
                    "report_id":   uuid,  # source report_id
                }
            }
        """
        from models.db_models import Report, KPIRecord, KPIDefinition

        type_priority = _type_priority_expr(Report)
        result: dict[str, dict] = {}

        for kpi_name in kpi_names:
            kdef = (
                db.query(KPIDefinition)
                .filter(KPIDefinition.name == kpi_name)
                .first()
            )
            if not kdef:
                logger.warning("kpi_cache.kpi_def_not_found", kpi=kpi_name)
                continue

            rec = (
                db.query(KPIRecord)
                .join(Report, KPIRecord.report_id == Report.id)
                .filter(
                    KPIRecord.company_id        == company_id,
                    KPIRecord.kpi_definition_id == kdef.id,
                    KPIRecord.report_year       == fy,
                    KPIRecord.normalized_value.isnot(None),
                )
                .order_by(type_priority, KPIRecord.extracted_at.desc())
                .first()
            )

            if rec is None:
                logger.debug("kpi_cache.miss", kpi=kpi_name, company_id=str(company_id)[:8])
                continue

            val  = rec.normalized_value
            conf = rec.confidence or 0.0
            unit = rec.unit or kdef.expected_unit

            # Validate: confidence gate
            if conf < MIN_CONFIDENCE:
                logger.debug(
                    "kpi_cache.low_confidence",
                    kpi=kpi_name, confidence=conf, threshold=MIN_CONFIDENCE,
                )
                continue

            # Validate: plausibility gate
            if not _is_plausible(kpi_name, val):
                logger.warning(
                    "kpi_cache.implausible_value",
                    kpi=kpi_name, value=val, unit=unit,
                )
                continue

            # Fetch the report type for provenance display
            src_report = db.query(Report).filter(Report.id == rec.report_id).first()
            src_type   = src_report.report_type if src_report else "unknown"

            result[kpi_name] = {
                "value":       val,
                "unit":        unit,
                "method":      rec.extraction_method,
                "confidence":  conf,
                "report_type": src_type,
                "report_id":   rec.report_id,
            }
            logger.debug(
                "kpi_cache.hit",
                kpi=kpi_name, value=val, unit=unit,
                report_type=src_type, confidence=round(conf, 3),
            )

        return result

    # ------------------------------------------------------------------
    # Convenience: split into cached / missing
    # ------------------------------------------------------------------

    def load(
        self,
        company_id: uuid.UUID,
        fy: int,
        kpi_names: list[str],
        db: Session,
    ) -> dict:
        """
        Check kpi_records for all requested KPIs.

        Returns:
            {
                "kpis":    {name: record_dict},   # valid cached records
                "missing": [name],                 # KPIs with no valid cache
            }
        """
        cached  = self.select_best_per_kpi(company_id, fy, kpi_names, db)
        missing = [k for k in kpi_names if k not in cached]

        logger.info(
            "kpi_cache.load",
            company_id=str(company_id)[:8],
            fy=fy,
            requested=len(kpi_names),
            cached=len(cached),
            missing=missing,
        )
        return {"kpis": cached, "missing": missing}

    # ------------------------------------------------------------------
    # Write: dedup-safe store
    # ------------------------------------------------------------------

    def store(
        self,
        company_id:     uuid.UUID,
        report_id:      uuid.UUID,
        fy:             int,
        kpi_records:    dict,
        revenue_result,
        db:             Session,
    ) -> None:
        """
        Persist newly extracted KPIs and revenue for ONE report.

        Dedup guard: (company_id, report_id, kpi_definition_id, normalized_value).
        If an identical record already exists it is silently skipped — no
        duplicate rows are written.

        Revenue is written only when the report row has no existing revenue_cr.

        Args:
            company_id:     Company UUID.
            report_id:      Source report UUID (KPIRecord.report_id).
            fy:             Fiscal year integer.
            kpi_records:    {kpi_name: {"value", "unit", "method", "confidence"}}.
            revenue_result: RevenueResult | None.
            db:             Active session (caller owns commit/rollback).
        """
        from models.db_models import Report, KPIRecord, KPIDefinition
        from services.revenue_extractor import store_revenue

        # Revenue: write only when absent
        if revenue_result:
            report_row = db.query(Report).filter(Report.id == report_id).first()
            if report_row and getattr(report_row, "revenue_cr", None) is None:
                try:
                    store_revenue(report_row, revenue_result, db)
                    logger.info(
                        "kpi_cache.revenue_stored",
                        report_id=str(report_id)[:8],
                        value_cr=revenue_result.value_cr,
                    )
                except Exception as exc:
                    logger.warning("kpi_cache.revenue_store_failed", error=str(exc))

        # KPIs
        stored = 0
        for kpi_name, rec in kpi_records.items():
            kdef = (
                db.query(KPIDefinition)
                .filter(KPIDefinition.name == kpi_name)
                .first()
            )
            if not kdef:
                logger.warning("kpi_cache.kpi_def_missing_on_store", kpi=kpi_name)
                continue

            val  = rec.get("value")
            unit = rec.get("unit", kdef.expected_unit)
            conf = float(rec.get("confidence", 0.5))

            if val is None:
                continue

            # Plausibility gate before writing
            if not _is_plausible(kpi_name, float(val)):
                logger.warning(
                    "kpi_cache.store_skipped_implausible",
                    kpi=kpi_name, value=val, unit=unit,
                )
                continue

            # Dedup: same report + KPI + value
            exists = (
                db.query(KPIRecord)
                .filter(
                    KPIRecord.company_id        == company_id,
                    KPIRecord.report_id         == report_id,
                    KPIRecord.kpi_definition_id == kdef.id,
                    KPIRecord.report_year       == fy,
                    KPIRecord.normalized_value  == float(val),
                )
                .first()
            )
            if exists:
                logger.debug(
                    "kpi_cache.store_skipped_duplicate",
                    kpi=kpi_name, report_id=str(report_id)[:8],
                )
                continue

            db.add(KPIRecord(
                company_id        = company_id,
                report_id         = report_id,
                kpi_definition_id = kdef.id,
                report_year       = fy,
                raw_value         = str(val),
                normalized_value  = float(val),
                unit              = unit,
                extraction_method = rec.get("method", "regex"),
                confidence        = conf,
                is_validated      = conf >= 0.85,
                validation_notes  = "kpi_cache_service",
            ))
            stored += 1

        if stored:
            db.flush()
            logger.info(
                "kpi_cache.stored",
                report_id=str(report_id)[:8],
                stored=stored,
                total_input=len(kpi_records),
            )

    # ------------------------------------------------------------------
    # Revenue helper: load best cached revenue for a company+year
    # ------------------------------------------------------------------

    def load_revenue(
        self,
        company_id: uuid.UUID,
        fy: int,
        db: Session,
    ):
        """
        Return the best cached RevenueResult for a company+year, or None.

        Priority: Integrated > BRSR > ESG, then most recent.
        """
        from models.db_models import Report
        from services.revenue_extractor import RevenueResult

        type_priority = _type_priority_expr(Report)

        report_with_rev = (
            db.query(Report)
            .filter(
                Report.company_id == company_id,
                Report.report_year == fy,
                Report.revenue_cr.isnot(None),
            )
            .order_by(type_priority, Report.created_at.desc())
            .first()
        )

        if not report_with_rev:
            return None

        rev_cr = getattr(report_with_rev, "revenue_cr", None)
        if rev_cr is None:
            return None

        try:
            return RevenueResult(
                value_cr   = float(rev_cr),
                raw_value  = str(rev_cr),
                raw_unit   = getattr(report_with_rev, "revenue_unit", None) or "INR_Crore",
                source     = getattr(report_with_rev, "revenue_source", None) or "db",
                page_number = 0,
                confidence  = 0.99,
                pattern_name = "cached",
            )
        except Exception:
            return None
