"""
run_benchmark.py

Terminal CLI — DB-first multi-source ESG benchmarking pipeline.

Multi-source pipeline (v2)
---------------------------
In v1 the CLI selected ONE report per company (BRSR preferred) and ran
extraction on that single source only.

In v2, ALL downloaded reports for a company+year are processed:

  1. DB lookup
     Check KPI records and revenue across ALL report types.
     Fast path: if all required KPIs and revenue are present, skip extraction.

  2. Full pipeline when data is missing
     Resolve ALL downloaded reports (BRSR + ESG + Integrated).
     For EACH report, independently:
       a. ParseOrchestrator.run()  — idempotent
       b. ExtractionAgent.extract_all() — extracts from that report's chunks
       c. Plausibility guard on extracted values
       d. Store results linked to that specific report_id
     If one report fails, the others continue (error isolation).

  3. KPI selection for comparison
     Read from DB across ALL reports for the company+year.
     Priority order: BRSR first, then Integrated, then ESG.
     Within the same report type, take the most recently extracted value.

  4. No merging at write time
     Each KPIRecord is linked to its source report_id.
     TCS BRSR scope_2 = X and TCS ESG scope_2 = Y both exist in DB.
     Only the BRSR value is used for comparison (priority rule at read time).

Interface:
    python -m esg_bench.run_benchmark \\
        --company1 "Infosys" --fy1 2025 \\
        --company2 "TCS"     --fy2 2024 \\
        [--force]   # skip DB cache, re-run full pipeline
        [--no-llm]  # rule-based summary only

Bugs fixed in original version:
  [BUG-1] DetachedInstanceError on _report.file_path — fixed by extracting
          all ORM values to plain dataclasses while session is open.
  [BUG-2] LLM returns absurd values — fixed by plausibility guard before
          storing KPI records.
"""
from __future__ import annotations

import argparse
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.logging_config import get_logger
from services.normalizer import normalize, NormalizationError
from services.benchmark import build_company_profile, compare_profiles, print_report
from services.summary_generator import generate_summary
from services.revenue_extractor import (
    extract_revenue,
    store_revenue,
    ensure_revenue_columns,
    RevenueResult,
)

logger = get_logger(__name__)
W = 72

TARGET_KPI_NAMES: list[str] = [
    "scope_1_emissions",
    "scope_2_emissions",
    "total_ghg_emissions",
    "waste_generated",
]

_DEFAULT_REVENUE_CR = 315322.0

# Report-type priority for KPI selection (lower = preferred).
# Applied at read time when multiple records exist for the same KPI.
_REPORT_TYPE_PRIORITY: dict[str, int] = {
    "BRSR":       0,
    "Integrated": 1,
    "ESG":        2,
}

# Plausibility limits: (min, max) in canonical unit
_KPI_PLAUSIBILITY: dict[str, tuple[float, float]] = {
    "energy_consumption":  (1_000,     500_000_000),
    "scope_1_emissions":   (1,           5_000_000),
    "scope_2_emissions":   (1,           5_000_000),
    "total_ghg_emissions": (1,          10_000_000),
    "water_consumption":   (100,        100_000_000),
    "waste_generated":     (0.1,            500_000),
}

_RATIO_CEILINGS: dict[str, float] = {
    "energy_consumption":  1_000,
    "scope_1_emissions":   10,
    "scope_2_emissions":   10,
    "total_ghg_emissions": 20,
    "water_consumption":   500,
    "waste_generated":     5,
}


# ─────────────────────────────────────────────────────────────────────────────
# Plausibility guards
# ─────────────────────────────────────────────────────────────────────────────

def _validate_kpi_plausibility(kpi_name: str, value: float, unit: str) -> bool:
    limits = _KPI_PLAUSIBILITY.get(kpi_name)
    if limits is None:
        return True
    lo, hi = limits
    try:
        norm = normalize(kpi_name=kpi_name, value=value, unit=unit)
        v = norm.normalized_value
    except NormalizationError:
        return False
    if v < lo or v > hi:
        logger.warning(
            "run_benchmark.implausible_value",
            kpi=kpi_name, value=value, unit=unit,
            normalized=v, limit_lo=lo, limit_hi=hi,
        )
        return False
    return True


def _validate_ratio_plausibility(kpi_name: str, ratio_value: float) -> bool:
    ceiling = _RATIO_CEILINGS.get(kpi_name)
    if ceiling is None:
        return True
    if ratio_value <= 0 or ratio_value > ceiling:
        logger.warning(
            "run_benchmark.implausible_ratio",
            kpi=kpi_name, ratio=ratio_value, ceiling=ceiling,
        )
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# DB schema bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_schema() -> None:
    try:
        from core.database import get_db
        with get_db() as db:
            ensure_revenue_columns(db)
    except Exception as exc:
        logger.warning("run_benchmark.migration_failed", error=str(exc))
        print(f"  [WARN] DB migration: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Plain-data containers (no live ORM objects — prevents DetachedInstanceError)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _CompanyInfo:
    id:   uuid.UUID
    name: str


@dataclass
class _ReportInfo:
    """One downloaded report row, as plain data."""
    id:          uuid.UUID
    report_type: str
    file_path:   Optional[str]
    status:      str


# ─────────────────────────────────────────────────────────────────────────────
# Revenue helper
# ─────────────────────────────────────────────────────────────────────────────

def _revenue_from_row(revenue_cr, revenue_unit, revenue_source) -> Optional[RevenueResult]:
    if revenue_cr is None:
        return None
    try:
        return RevenueResult(
            value_cr=float(revenue_cr),
            raw_value=str(revenue_cr),
            raw_unit=revenue_unit or "INR_Crore",
            source=revenue_source or "db",
            page_number=0,
            confidence=0.99,
            pattern_name="cached",
        )
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# total_ghg derivation
# ─────────────────────────────────────────────────────────────────────────────

def _derive_total_ghg(kpi_records: dict) -> Optional[dict]:
    if "total_ghg_emissions" in kpi_records:
        return None
    s1 = kpi_records.get("scope_1_emissions")
    s2 = kpi_records.get("scope_2_emissions")
    if not s1 or not s2:
        return None
    try:
        n1 = normalize("scope_1_emissions", float(s1["value"]), s1["unit"])
        n2 = normalize("scope_2_emissions", float(s2["value"]), s2["unit"])
        total = round(n1.normalized_value + n2.normalized_value, 2)
    except Exception:
        return None
    return {
        "value":      total,
        "unit":       "tCO2e",
        "method":     "derived",
        "confidence": round(min(s1["confidence"], s2["confidence"]) * 0.99, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DB read — priority-ordered, returns plain data only
# ─────────────────────────────────────────────────────────────────────────────

def _db_lookup(company_name: str, fy: int) -> dict:
    """
    Check DB for cached KPI records and revenue.

    Multi-source selection:
      When multiple KPIRecords exist for the same KPI (one per report type),
      select by BRSR(0) > Integrated(1) > ESG(2) priority, then newest first.
      This is applied via a JOIN to Report and a CASE-based ORDER BY.

    Returns plain Python dicts/dataclasses only — no live ORM objects.
    """
    empty: dict = {
        "kpis":    {},
        "revenue": None,
        "company": None,
        "reports": [],   # list[_ReportInfo] — ALL downloaded reports
    }
    try:
        from core.database import get_db
        from models.db_models import Company, Report, KPIRecord, KPIDefinition
        from sqlalchemy import case

        with get_db() as db:
            company_row = (
                db.query(Company)
                .filter(Company.name.ilike(f"%{company_name}%"))
                .first()
            )
            if not company_row:
                return empty

            company_info = _CompanyInfo(id=company_row.id, name=company_row.name)

            # All downloaded reports for this company+year
            report_rows = (
                db.query(Report)
                .filter(
                    Report.company_id == company_row.id,
                    Report.report_year == fy,
                    Report.status.in_(["downloaded", "parsed", "extracted"]),
                )
                .order_by(Report.created_at.desc())
                .all()
            )

            report_infos = [
                _ReportInfo(
                    id=r.id,
                    report_type=r.report_type,
                    file_path=r.file_path,
                    status=r.status,
                )
                for r in report_rows
            ]

            # Priority expression for KPI selection
            type_priority = case(
                (Report.report_type == "BRSR",       0),
                (Report.report_type == "Integrated",  1),
                (Report.report_type == "ESG",         2),
                else_=99,
            )

            # Revenue: prefer BRSR, then most recent
            rev_report = (
                db.query(Report)
                .filter(
                    Report.company_id == company_row.id,
                    Report.report_year == fy,
                    Report.revenue_cr.isnot(None),
                )
                .order_by(type_priority, Report.created_at.desc())
                .first()
            )

            cached_rev: Optional[RevenueResult] = None
            if rev_report:
                cached_rev = _revenue_from_row(
                    getattr(rev_report, "revenue_cr",     None),
                    getattr(rev_report, "revenue_unit",   None),
                    getattr(rev_report, "revenue_source", None),
                )

            # KPIs: priority-ordered selection
            kpis: dict = {}
            if report_rows:
                for kpi_name in TARGET_KPI_NAMES:
                    kdef = (
                        db.query(KPIDefinition)
                        .filter(KPIDefinition.name == kpi_name)
                        .first()
                    )
                    if not kdef:
                        continue

                    rec = (
                        db.query(KPIRecord)
                        .join(Report, KPIRecord.report_id == Report.id)
                        .filter(
                            KPIRecord.company_id        == company_row.id,
                            KPIRecord.kpi_definition_id == kdef.id,
                            KPIRecord.report_year       == fy,
                            KPIRecord.normalized_value.isnot(None),
                        )
                        .order_by(type_priority, KPIRecord.extracted_at.desc())
                        .first()
                    )
                    if rec:
                        val  = rec.normalized_value
                        unit = rec.unit or kdef.expected_unit
                        if _validate_kpi_plausibility(kpi_name, val, unit):
                            # Get source report type for logging
                            src_report = db.query(Report).filter(
                                Report.id == rec.report_id
                            ).first()
                            src_type = src_report.report_type if src_report else "unknown"
                            kpis[kpi_name] = {
                                "value":       val,
                                "unit":        unit,
                                "method":      rec.extraction_method,
                                "confidence":  rec.confidence or 0.9,
                                "report_type": src_type,
                            }
                        else:
                            print(f"  [DROP] {kpi_name} from DB: value {val} {unit} "
                                  f"outside plausible range — will re-extract")

            return {
                "kpis":    kpis,
                "revenue": cached_rev,
                "company": company_info,
                "reports": report_infos,
            }

    except Exception as exc:
        logger.warning("run_benchmark.db_lookup_failed", error=str(exc))
        return empty


# ─────────────────────────────────────────────────────────────────────────────
# DB write — per report_id, append-only
# ─────────────────────────────────────────────────────────────────────────────

def _db_store(
    company_id:     uuid.UUID,
    report_id:      uuid.UUID,
    fy:             int,
    new_kpis:       dict,
    revenue_result: Optional[RevenueResult],
) -> None:
    """
    Persist extracted KPIs + revenue for ONE report (append-only).

    Each KPIRecord is linked to the specific report_id it was extracted from.
    Duplicate detection: same company + report + KPI + value already in DB → skip.
    """
    try:
        from core.database import get_db
        from models.db_models import Report, KPIRecord, KPIDefinition

        with get_db() as db:
            if revenue_result:
                report_row = db.query(Report).filter(Report.id == report_id).first()
                if report_row and getattr(report_row, "revenue_cr", None) is None:
                    try:
                        store_revenue(report_row, revenue_result, db)
                    except Exception as exc:
                        logger.warning("run_benchmark.revenue_store_failed", error=str(exc))

            for kpi_name, rec in new_kpis.items():
                kdef = (
                    db.query(KPIDefinition)
                    .filter(KPIDefinition.name == kpi_name)
                    .first()
                )
                if not kdef:
                    continue

                # Dedup: same report + KPI + value
                exists = (
                    db.query(KPIRecord)
                    .filter(
                        KPIRecord.company_id        == company_id,
                        KPIRecord.report_id         == report_id,
                        KPIRecord.kpi_definition_id == kdef.id,
                        KPIRecord.report_year       == fy,
                        KPIRecord.normalized_value  == rec["value"],
                    )
                    .first()
                )
                if exists:
                    continue

                db.add(KPIRecord(
                    company_id        = company_id,
                    report_id         = report_id,
                    kpi_definition_id = kdef.id,
                    report_year       = fy,
                    raw_value         = str(rec["value"]),
                    normalized_value  = rec["value"],
                    unit              = rec["unit"],
                    extraction_method = rec["method"],
                    confidence        = rec["confidence"],
                    is_validated      = rec["confidence"] >= 0.85,
                    validation_notes  = "extracted by esg_bench",
                ))

    except Exception as exc:
        logger.warning("run_benchmark.db_store_failed", error=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Report resolution — returns ALL usable reports as plain _ReportInfo list
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_all_reports(
    company_name:     str,
    fy:               int,
    existing_company: Optional[_CompanyInfo],
    existing_reports: list[_ReportInfo],
) -> tuple[Optional[_CompanyInfo], list[_ReportInfo]]:
    """
    Return Company info + ALL usable downloaded reports.

    No priority selection here — all report types are returned so the
    caller can process each one independently.

    Returns plain _CompanyInfo / list[_ReportInfo] (no live ORM objects).
    """
    try:
        from core.database import get_db
        from models.db_models import Company, Report

        with get_db() as db:
            if existing_company:
                cid   = existing_company.id
                cname = existing_company.name
            else:
                row = (
                    db.query(Company)
                    .filter(Company.name.ilike(f"%{company_name}%"))
                    .first()
                )
                if not row:
                    print(f"  [WARN] Company '{company_name}' not found in DB.")
                    print(f"         Run: python main.py ingest --company \"{company_name}\" "
                          f"--year {fy}")
                    return None, []
                cid   = row.id
                cname = row.name

            company_info = _CompanyInfo(id=cid, name=cname)

            if existing_reports:
                return company_info, existing_reports

            report_rows = (
                db.query(Report)
                .filter(
                    Report.company_id == cid,
                    Report.report_year == fy,
                    Report.status.in_(["downloaded", "parsed", "extracted"]),
                )
                .order_by(Report.created_at.desc())
                .all()
            )

            if not report_rows:
                # Check if reports exist with wrong status
                any_row = (
                    db.query(Report)
                    .filter(Report.company_id == cid, Report.report_year == fy)
                    .first()
                )
                if any_row:
                    print(f"  [WARN] Report exists but status='{any_row.status}'. "
                          f"Must be downloaded first.")
                else:
                    print(f"  [WARN] No report for {company_name} FY{fy}.")
                print(f"         Run: python main.py ingest --company \"{company_name}\" "
                      f"--year {fy}")
                return company_info, []

            infos = [
                _ReportInfo(
                    id=r.id,
                    report_type=r.report_type,
                    file_path=r.file_path,
                    status=r.status,
                )
                for r in report_rows
            ]

            types_found = [ri.report_type for ri in infos]
            print(f"  Reports found: {types_found}")
            return company_info, infos

    except Exception as exc:
        logger.error("run_benchmark.resolve_reports_failed", error=str(exc))
        return None, []


# ─────────────────────────────────────────────────────────────────────────────
# Per-report extraction pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline_for_report(
    report_info:  _ReportInfo,
    fy:           int,
    missing_kpis: list[str],
    need_revenue: bool,
    llm_service,
) -> tuple[dict, Optional[RevenueResult]]:
    """
    Run ParseOrchestrator + ExtractionAgent + revenue extractor for ONE report.

    Returns:
        (new_kpi_records, revenue_result)
        new_kpi_records: only values that pass plausibility check

    Does NOT raise on failure — returns empty results and logs the error.
    Caller continues with the next report (error isolation).
    """
    from services.parse_orchestrator import ParseOrchestrator
    from agents.extraction_agent import ExtractionAgent
    from core.database import get_db

    new_kpis: dict              = {}
    new_rev:  Optional[RevenueResult] = None

    if not report_info.file_path or not Path(report_info.file_path).exists():
        print(f"       [SKIP] file not on disk: {report_info.file_path}")
        return new_kpis, new_rev

    pdf_path = Path(report_info.file_path)

    # ── Parse (idempotent) ────────────────────────────────────────────────────
    print(f"       [a] ParseOrchestrator (cache check)...")
    try:
        parsed = ParseOrchestrator().run(report_id=report_info.id)
        print(f"           {parsed.page_count} pages, "
              f"{parsed.meta.get('chunk_count','?')} chunks")
    except Exception as exc:
        logger.error("run_benchmark.parse_failed",
                     report_id=str(report_info.id)[:8], error=str(exc))
        print(f"       [ERROR] Parsing failed: {exc}")
        return new_kpis, new_rev   # can't extract without parse

    # ── ExtractionAgent ───────────────────────────────────────────────────────
    kpis_to_extract = [k for k in missing_kpis if k != "total_ghg_emissions"]
    if kpis_to_extract:
        print(f"       [b] ExtractionAgent -> {kpis_to_extract}")
        try:
            with get_db() as db:
                extracted_list = ExtractionAgent().extract_all(
                    report_id=report_info.id,
                    db=db,
                    kpi_names=kpis_to_extract,
                )
            for ext in extracted_list:
                if ext.normalized_value is None:
                    continue
                val  = ext.normalized_value
                unit = ext.unit or ""
                if not _validate_kpi_plausibility(ext.kpi_name, val, unit):
                    print(f"           [DROP] {ext.kpi_name}: {val:,.2f} {unit} "
                          f"— outside plausible range")
                    continue
                new_kpis[ext.kpi_name] = {
                    "value":      val,
                    "unit":       unit,
                    "method":     ext.extraction_method,
                    "confidence": ext.confidence or 0.5,
                }
                print(f"           OK  {ext.kpi_name}: {val:,.2f} {unit} "
                      f"[{ext.extraction_method} conf={ext.confidence:.2f}]")
            for k in kpis_to_extract:
                if k not in new_kpis:
                    print(f"           --  {k}: not found")
        except Exception as exc:
            logger.error("run_benchmark.agent_failed",
                         report_id=str(report_info.id)[:8], error=str(exc))
            print(f"       [ERROR] ExtractionAgent: {exc}")

    # ── Revenue ───────────────────────────────────────────────────────────────
    if need_revenue:
        print(f"       [c] Revenue extraction...")
        try:
            new_rev = extract_revenue(
                pdf_path=pdf_path,
                fiscal_year_hint=fy,
                llm_service=llm_service,
            )
            if new_rev:
                print(f"           OK  revenue: INR {new_rev.value_cr:,.0f} Cr "
                      f"[{new_rev.pattern_name} conf={new_rev.confidence:.2f}]")
            else:
                print(f"           --  revenue: not found in this report")
        except Exception as exc:
            logger.warning("run_benchmark.revenue_failed",
                           report_id=str(report_info.id)[:8], error=str(exc))

    return new_kpis, new_rev


# ─────────────────────────────────────────────────────────────────────────────
# Per-company pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_company(
    company_name: str,
    fy:           int,
    force:        bool,
    llm_service=None,
):
    """
    Multi-source DB-first pipeline for one company.

    Phase 1 — DB lookup
      Check existing KPIRecords (with BRSR > Integrated > ESG priority selection).
      If all KPIs and revenue are present and plausible, skip extraction.

    Phase 2 — Multi-report extraction loop (when data is missing)
      Resolve ALL downloaded reports for the company+year.
      For EACH report, independently:
        a. Parse (idempotent)
        b. Extract KPIs and revenue
        c. Store results linked to that report_id
      If one report fails, others continue (error isolation).

    Phase 3 — Merge + derive + normalise
      Merge cached KPIs with newly extracted ones.
      Derive total_ghg = scope_1 + scope_2 if missing.
      Persist any new derived values.

    Phase 4 — Build CompanyProfile
      Normalise to canonical units and compute intensity ratios.
      Drop ratios above plausibility ceiling.

    Returns CompanyProfile for benchmarking.
    """
    print(f"\n{'─' * W}")
    print(f"  {company_name}  FY{fy}")
    print(f"{'─' * W}")

    # ── Phase 1: DB lookup ────────────────────────────────────────────────────
    cached_kpis:  dict                     = {}
    cached_rev:   Optional[RevenueResult]  = None
    company_info: Optional[_CompanyInfo]   = None
    report_infos: list[_ReportInfo]        = []

    if not force:
        print("  [1/5] Checking DB cache...")
        db_data      = _db_lookup(company_name, fy)
        cached_kpis  = db_data["kpis"]
        cached_rev   = db_data["revenue"]
        company_info = db_data["company"]
        report_infos = db_data["reports"]

        if cached_kpis:
            for kpi, rec in cached_kpis.items():
                src = rec.get("report_type", "?")
                print(f"        {kpi}: {rec['value']:,.2f} {rec['unit']}  [{src}]")
        else:
            print("        No cached KPIs")
        print(f"        Revenue: "
              f"{'INR ' + f'{cached_rev.value_cr:,.0f} Cr' if cached_rev else 'not cached'}")
        if report_infos:
            types = [ri.report_type for ri in report_infos]
            print(f"        Reports in DB: {types}")
    else:
        print("  [1/5] --force: skipping DB cache")
        db_data      = _db_lookup(company_name, fy)
        company_info = db_data["company"]
        report_infos = db_data["reports"]

    # ── Phase 2: Multi-report extraction ──────────────────────────────────────
    extractable  = [k for k in TARGET_KPI_NAMES if k != "total_ghg_emissions"]
    missing_kpis = [k for k in extractable if k not in cached_kpis]
    need_revenue = cached_rev is None

    new_kpi_recs_all: dict              = {}
    new_revenue:      Optional[RevenueResult] = None

    if missing_kpis or need_revenue or force:
        needs = []
        if missing_kpis: needs.append(f"KPIs: {missing_kpis}")
        if need_revenue:  needs.append("revenue")
        print(f"  [2/5] Multi-report pipeline -> {' + '.join(needs)}")

        # Resolve all downloaded reports
        company_info, report_infos = _resolve_all_reports(
            company_name=company_name,
            fy=fy,
            existing_company=company_info,
            existing_reports=report_infos,
        )

        if not company_info:
            print("  [SKIP] Company not found in DB.")
            print("         Run: python main.py ingest --company "
                  f"\"{company_name}\" --year {fy}")
        elif not report_infos:
            print("  [SKIP] No downloaded reports found.")
        else:
            # Process each report independently
            for ri in report_infos:
                print(f"\n     [{ri.report_type}] report_id={str(ri.id)[:8]}")

                kpis_still_needed = (
                    [k for k in extractable if k not in cached_kpis
                     and k not in new_kpi_recs_all]
                    if not force else extractable
                )
                rev_still_needed = need_revenue and new_revenue is None

                if not kpis_still_needed and not rev_still_needed and not force:
                    print(f"       All needed data already extracted — skipping.")
                    continue

                try:
                    new_kpis, new_rev = _run_pipeline_for_report(
                        report_info=ri,
                        fy=fy,
                        missing_kpis=kpis_still_needed if not force else extractable,
                        need_revenue=rev_still_needed or force,
                        llm_service=llm_service,
                    )
                except Exception as exc:
                    # Error isolation: log and continue with next report
                    logger.error("run_benchmark.report_pipeline_failed",
                                 report_type=ri.report_type,
                                 report_id=str(ri.id)[:8],
                                 error=str(exc))
                    print(f"       [ERROR] {ri.report_type} pipeline failed: {exc}")
                    print(f"       Continuing with next report...")
                    continue

                # Accumulate results from this report
                new_kpi_recs_all.update(new_kpis)
                if new_rev and new_revenue is None:
                    new_revenue = new_rev

                # Store immediately linked to this report_id
                to_store = dict(new_kpis)
                if to_store or new_rev:
                    print(f"       Storing {len(to_store)} KPI(s) for [{ri.report_type}]...")
                    _db_store(
                        company_id=company_info.id,
                        report_id=ri.id,
                        fy=fy,
                        new_kpis=to_store,
                        revenue_result=new_rev,
                    )
    else:
        print("  [2/5] All data in DB — skipping pipeline")

    # ── Phase 3: Merge + derive total_ghg ─────────────────────────────────────
    merged_kpis = {**cached_kpis, **new_kpi_recs_all}

    if "total_ghg_emissions" not in merged_kpis:
        ghg = _derive_total_ghg(merged_kpis)
        if ghg:
            merged_kpis["total_ghg_emissions"] = ghg
            s1 = merged_kpis.get("scope_1_emissions", {}).get("value", 0)
            s2 = merged_kpis.get("scope_2_emissions", {}).get("value", 0)
            if isinstance(s1, (int, float)) and isinstance(s2, (int, float)):
                print(f"  [3/5] Derived total_ghg: {ghg['value']:,.2f} tCO2e "
                      f"({s1:,.0f} + {s2:,.0f})")
            else:
                print(f"  [3/5] Derived total_ghg: {ghg['value']:,.2f} tCO2e")
        else:
            print("  [3/5] Cannot derive total_ghg (missing scope_1 or scope_2)")
    else:
        print("  [3/5] total_ghg_emissions already present")

    # ── Phase 4: Persist new derived values ───────────────────────────────────
    # Store derived total_ghg against the highest-priority report that contributed
    derived_to_store: dict = {}
    if "total_ghg_emissions" not in cached_kpis and "total_ghg_emissions" in merged_kpis:
        derived_to_store["total_ghg_emissions"] = merged_kpis["total_ghg_emissions"]

    revenue_result = cached_rev or new_revenue

    if (derived_to_store or new_revenue) and company_info and report_infos:
        # Pick highest-priority report for storing derived KPIs
        sorted_reports = sorted(
            report_infos,
            key=lambda r: _REPORT_TYPE_PRIORITY.get(r.report_type, 99),
        )
        target_report = sorted_reports[0]
        print(f"  [4/5] Storing derived KPIs -> [{target_report.report_type}]...")
        _db_store(
            company_id=company_info.id,
            report_id=target_report.id,
            fy=fy,
            new_kpis=derived_to_store,
            revenue_result=new_revenue if not cached_rev else None,
        )
        print(f"        {len(derived_to_store)} derived KPI(s) stored")
    else:
        print("  [4/5] Nothing new to store")

    # ── Phase 5: Normalise + display ──────────────────────────────────────────
    print(f"\n  [5/5] Normalised KPIs — {company_name} FY{fy}")
    print(f"  {'KPI':<35} {'Value':>16}  Unit   [Source / Report]")
    print("  " + "─" * 72)

    for kpi_name in TARGET_KPI_NAMES:
        rec = merged_kpis.get(kpi_name)
        if not rec:
            print(f"  {'  ' + kpi_name:<35} {'NOT FOUND':>16}")
            continue
        try:
            n = normalize(kpi_name=kpi_name, value=float(rec["value"]), unit=str(rec["unit"]))
            conv = (f"  <- {rec['value']:,.2f} {rec['unit']}"
                    if n.conversion_factor != 1.0 else "")
            src  = rec.get("report_type", rec.get("method", "?"))
            print(f"  {'  ' + kpi_name:<35} {n.normalized_value:>14,.2f}  "
                  f"{n.normalized_unit}  [{src}]{conv}")
        except NormalizationError as e:
            print(f"  {'  ' + kpi_name:<35} {rec['value']:>14,.2f}  "
                  f"{rec['unit']}  [WARN: {e}]")

    rev_cr  = revenue_result.value_cr if revenue_result else _DEFAULT_REVENUE_CR
    rev_src = revenue_result.source   if revenue_result else "default"
    print(f"  {'  revenue':<35} {rev_cr:>14,.0f}  INR_Crore  [{rev_src}]")

    # ── Build CompanyProfile ──────────────────────────────────────────────────
    page_texts: list[str] = []
    # Use highest-priority report for PDF text (intensity ratio detection)
    sorted_for_pdf = sorted(
        [ri for ri in report_infos if ri.file_path and Path(ri.file_path).exists()],
        key=lambda r: _REPORT_TYPE_PRIORITY.get(r.report_type, 99),
    )
    if sorted_for_pdf:
        try:
            import fitz
            doc = fitz.open(str(sorted_for_pdf[0].file_path))
            for pg in doc:
                page_texts.append(pg.get_text())
            doc.close()
        except Exception:
            pass

    profile = build_company_profile(
        kpi_records=merged_kpis,
        revenue_cr=rev_cr,
        revenue_source=rev_src,
        company_name=company_name,
        fiscal_year=fy,
        page_texts=page_texts,
    )

    # Drop implausible ratios
    bad_ratios = [
        kpi for kpi, ratio in profile.ratios.items()
        if not _validate_ratio_plausibility(kpi, ratio.ratio_value)
    ]
    for kpi in bad_ratios:
        print(f"  [DROP ratio] {kpi}: ratio {profile.ratios[kpi].ratio_value:.4e} "
              f"{profile.ratios[kpi].ratio_unit} exceeds ceiling")
        del profile.ratios[kpi]

    print(f"\n  Intensity ratios ({company_name} FY{fy}):")
    if profile.ratios:
        for kpi_name, ratio in profile.ratios.items():
            src = "[reported]" if "reported" in ratio.ratio_source else "[computed]"
            v   = (f"{ratio.ratio_value:.4e}" if ratio.ratio_value < 0.001
                   else f"{ratio.ratio_value:.4f}")
            print(f"  {'  ' + kpi_name:<35} {v:>14}  {ratio.ratio_unit}  {src}")
    else:
        print("  (no valid intensity ratios)")

    return profile


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "ESG Competitive Benchmarking — multi-source, DB-first.\n"
            "Processes ALL downloaded report types (BRSR, ESG, Integrated).\n"
            "Companies must be ingested first: python main.py ingest ..."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m esg_bench.run_benchmark \\
      --company1 "Infosys" --fy1 2025 \\
      --company2 "TCS"     --fy2 2024

  # Force re-extraction across all report types:
  python -m esg_bench.run_benchmark \\
      --company1 "Infosys" --fy1 2025 \\
      --company2 "TCS"     --fy2 2024 --force

  # Ingest all report types first:
  python main.py ingest --company "Infosys" --year 2025
  python main.py ingest --company "TCS"     --year 2024
        """,
    )
    parser.add_argument("--company1", required=True)
    parser.add_argument("--fy1",      required=True, type=int)
    parser.add_argument("--company2", required=True)
    parser.add_argument("--fy2",      required=True, type=int)
    parser.add_argument("--force",   action="store_true",
                        help="Skip DB cache, re-run extraction on all reports")
    parser.add_argument("--no-llm",  action="store_true",
                        help="Disable LLM (rule-based summary, regex-only extraction)")
    args = parser.parse_args()

    print("\n" + "=" * W)
    print("  ESG COMPETITIVE BENCHMARKING — MULTI-SOURCE PIPELINE")
    print(f"  {args.company1} FY{args.fy1}  vs  {args.company2} FY{args.fy2}")
    print(f"  KPI priority: BRSR > Integrated > ESG")
    print("=" * W)

    print("\n  Ensuring DB schema...")
    _ensure_schema()

    llm_service = None
    if not args.no_llm:
        try:
            from services.llm_service import LLMService
            from core.config import get_settings
            if get_settings().llm_api_key:
                llm_service = LLMService()
                print("  LLM: Gemini enabled")
            else:
                print("  LLM: no API key — rule-based fallback")
        except Exception as exc:
            print(f"  LLM: unavailable ({exc})")

    profile1 = process_company(args.company1, args.fy1, args.force, llm_service)
    profile2 = process_company(args.company2, args.fy2, args.force, llm_service)

    report = compare_profiles([profile1, profile2])
    print_report(report)

    print("\n" + "=" * W)
    print("  NARRATIVE SUMMARY")
    print("=" * W + "\n")
    print(generate_summary([profile1, profile2], report, llm=llm_service))
    print()


if __name__ == "__main__":
    main()