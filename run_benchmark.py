"""
run_benchmark.py

Terminal CLI — DB-first multi-source ESG benchmarking pipeline.

Multi-source pipeline (v3 — KPI-level cache + fallback)
---------------------------------------------------------
v2 selected ONE report per company (BRSR preferred) and ran extraction
on that single source only.

v3 introduces:

  KPI-Level Cache
    Before ANY extraction, check kpi_records for (company_id, fy, kpi_name).
    If a valid record exists (confidence >= threshold, value plausible),
    use it — skip extraction for that KPI entirely.
    Only extract KPIs that have no valid cached record.

  KPI-Level Report Fallback
    When selecting from kpi_records, apply priority PER KPI:
        Integrated (0) > BRSR (1) > ESG (2)
    Within the same type, take the most recently extracted record.
    This means scope_1 can come from BRSR while waste_generated comes
    from Integrated — the best available source wins per metric.

  No global report selection
    The old "pick one best report, ignore the others" logic is gone.
    Every KPI independently finds its best source.

  Dedup-safe writes
    KPIRecord rows are keyed on (company_id, report_id, kpi_definition_id,
    normalized_value).  Duplicate writes are silently skipped.

Interface (unchanged):
    python -m esg_bench.run_benchmark \\
        --company1 "Infosys" --fy1 2025 \\
        --company2 "TCS"     --fy2 2024 \\
        [--force]   # skip cache, re-run full pipeline
        [--no-llm]  # rule-based summary only
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
from services.kpi_cache_service import KPICacheService

logger = get_logger(__name__)
W = 72

TARGET_KPI_NAMES: list[str] = [
    "scope_1_emissions",
    "scope_2_emissions",
    "total_ghg_emissions",
    "waste_generated",
]

_DEFAULT_REVENUE_CR = 315322.0

_REPORT_TYPE_PRIORITY: dict[str, int] = {
    "Integrated": 0,
    "BRSR":       1,
    "ESG":        2,
}

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

_kpi_cache_svc = KPICacheService()


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
# Plain-data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _CompanyInfo:
    id:   uuid.UUID
    name: str


@dataclass
class _ReportInfo:
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
# DB lookup — KPI-level cache read
# ─────────────────────────────────────────────────────────────────────────────

def _db_lookup(company_name: str, fy: int) -> dict:
    """
    Check DB for cached KPI records (KPI-level, not report-level).

    Uses KPICacheService.select_best_per_kpi() which applies
    Integrated > BRSR > ESG priority independently per KPI.

    Returns plain Python dicts only — no live ORM objects.
    """
    empty: dict = {
        "kpis":    {},
        "revenue": None,
        "company": None,
        "reports": [],
    }
    try:
        from core.database import get_db
        from models.db_models import Company, Report

        with get_db() as db:
            company_row = (
                db.query(Company)
                .filter(Company.name.ilike(f"%{company_name}%"))
                .first()
            )
            if not company_row:
                return empty

            company_info = _CompanyInfo(id=company_row.id, name=company_row.name)

            # All downloaded reports for this company+year (for pipeline use)
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

            # KPI-level cache read — Integrated > BRSR > ESG per KPI
            extractable = [k for k in TARGET_KPI_NAMES if k != "total_ghg_emissions"]
            kpis = _kpi_cache_svc.select_best_per_kpi(
                company_id=company_row.id,
                fy=fy,
                kpi_names=extractable,
                db=db,
            )

            # Revenue: best cached across all report types
            cached_rev = _kpi_cache_svc.load_revenue(company_row.id, fy, db)

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
# DB write — per report_id, dedup-safe via KPICacheService
# ─────────────────────────────────────────────────────────────────────────────

def _db_store(
    company_id:     uuid.UUID,
    report_id:      uuid.UUID,
    fy:             int,
    new_kpis:       dict,
    revenue_result: Optional[RevenueResult],
) -> None:
    """Persist extracted KPIs + revenue using KPICacheService (dedup-safe)."""
    try:
        from core.database import get_db
        with get_db() as db:
            _kpi_cache_svc.store(
                company_id=company_id,
                report_id=report_id,
                fy=fy,
                kpi_records=new_kpis,
                revenue_result=revenue_result,
                db=db,
            )
    except Exception as exc:
        logger.warning("run_benchmark.db_store_failed", error=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Report resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_all_reports(
    company_name:     str,
    fy:               int,
    existing_company: Optional[_CompanyInfo],
    existing_reports: list[_ReportInfo],
) -> tuple[Optional[_CompanyInfo], list[_ReportInfo]]:
    """Return Company info + ALL usable downloaded reports as plain objects."""
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
# Per-report extraction pipeline (unchanged logic)
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

    Only extracts the KPIs listed in missing_kpis — KPIs already in cache
    are never passed here.

    Returns (new_kpi_records, revenue_result).
    Does NOT raise on failure — returns empty results and logs the error.
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

    # Parse (idempotent)
    print(f"       [a] ParseOrchestrator (cache check)...")
    try:
        parsed = ParseOrchestrator().run(report_id=report_info.id)
        print(f"           {parsed.page_count} pages, "
              f"{parsed.meta.get('chunk_count','?')} chunks")
    except Exception as exc:
        logger.error("run_benchmark.parse_failed",
                     report_id=str(report_info.id)[:8], error=str(exc))
        print(f"       [ERROR] Parsing failed: {exc}")
        return new_kpis, new_rev

    # ExtractionAgent — only for missing KPIs
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

    # Revenue
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
# Per-company pipeline — v3 with KPI-level cache
# ─────────────────────────────────────────────────────────────────────────────

def process_company(
    company_name: str,
    fy:           int,
    force:        bool,
    llm_service=None,
):
    """
    Multi-source DB-first pipeline for one company (v3).

    Phase 1 — KPI-level cache check
      For each target KPI, independently check kpi_records using
      Integrated > BRSR > ESG priority.
      Cached KPIs are used as-is — no extraction for them.
      Only missing KPIs proceed to Phase 2.

    Phase 2 — Multi-report extraction (missing KPIs only)
      Resolve ALL downloaded reports for the company+year.
      For EACH report, extract ONLY the still-missing KPIs.
      Once a KPI is found in any report, remove it from the missing list
      so subsequent reports skip it (no redundant extraction).
      Error isolation: if one report fails, others continue.

    Phase 3 — Merge + derive + normalise
      Merge cached KPIs with newly extracted ones.
      Derive total_ghg = scope_1 + scope_2 if missing.
      Persist any new derived values.

    Phase 4 — Build CompanyProfile
      Normalise to canonical units and compute intensity ratios.
      Drop ratios above plausibility ceiling.
    """
    print(f"\n{'─' * W}")
    print(f"  {company_name}  FY{fy}")
    print(f"{'─' * W}")

    # ── Phase 1: KPI-level cache check ───────────────────────────────────────
    cached_kpis:  dict                    = {}
    cached_rev:   Optional[RevenueResult] = None
    company_info: Optional[_CompanyInfo]  = None
    report_infos: list[_ReportInfo]       = []

    extractable = [k for k in TARGET_KPI_NAMES if k != "total_ghg_emissions"]

    if not force:
        print("  [1/5] KPI-level cache check (Integrated > BRSR > ESG per KPI)...")
        db_data      = _db_lookup(company_name, fy)
        cached_kpis  = db_data["kpis"]
        cached_rev   = db_data["revenue"]
        company_info = db_data["company"]
        report_infos = db_data["reports"]

        if cached_kpis:
            for kpi, rec in cached_kpis.items():
                src = rec.get("report_type", "?")
                print(f"        HIT  {kpi}: {rec['value']:,.2f} {rec['unit']}  "
                      f"[{src} conf={rec['confidence']:.2f}]")
        else:
            print("        No cached KPIs")

        missing_kpis = [k for k in extractable if k not in cached_kpis]
        if not missing_kpis:
            print("        All KPIs cached — skipping extraction entirely.")
        else:
            print(f"        Missing: {missing_kpis}")

        print(f"        Revenue: "
              f"{'INR ' + f'{cached_rev.value_cr:,.0f} Cr' if cached_rev else 'not cached'}")
    else:
        print("  [1/5] --force: clearing cache, will re-extract all KPIs.")
        db_data      = _db_lookup(company_name, fy)
        company_info = db_data["company"]
        report_infos = db_data["reports"]
        missing_kpis = list(extractable)
        cached_kpis  = {}
        cached_rev   = None

    # ── Phase 2: Multi-report extraction (missing KPIs only) ─────────────────
    need_revenue      = cached_rev is None
    new_kpi_recs_all: dict                    = {}
    new_revenue:      Optional[RevenueResult] = None

    if missing_kpis or need_revenue:
        needs = []
        if missing_kpis: needs.append(f"KPIs: {missing_kpis}")
        if need_revenue:  needs.append("revenue")
        print(f"  [2/5] Multi-report extraction -> {' + '.join(needs)}")

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
            # Sort reports by type priority so we always try the best source first.
            # This minimises redundant extraction across reports.
            sorted_reports = sorted(
                report_infos,
                key=lambda r: _REPORT_TYPE_PRIORITY.get(r.report_type, 99),
            )

            # Track which KPIs still need extraction as we iterate reports.
            still_missing = list(missing_kpis)
            rev_still_needed = need_revenue

            for ri in sorted_reports:
                if not still_missing and not rev_still_needed and not force:
                    print(f"\n     All KPIs found — stopping early (skipping remaining reports).")
                    break

                print(f"\n     [{ri.report_type}] report_id={str(ri.id)[:8]}")

                kpis_for_this_report = (
                    still_missing if not force else list(extractable)
                )

                if not kpis_for_this_report and not rev_still_needed and not force:
                    print(f"       All needed KPIs already found — skipping.")
                    continue

                try:
                    new_kpis, new_rev = _run_pipeline_for_report(
                        report_info=ri,
                        fy=fy,
                        missing_kpis=kpis_for_this_report,
                        need_revenue=rev_still_needed or force,
                        llm_service=llm_service,
                    )
                except Exception as exc:
                    logger.error(
                        "run_benchmark.report_pipeline_failed",
                        report_type=ri.report_type,
                        report_id=str(ri.id)[:8],
                        error=str(exc),
                    )
                    print(f"       [ERROR] {ri.report_type} pipeline failed: {exc}")
                    print(f"       Continuing with next report...")
                    continue

                # Merge results from this report
                new_kpi_recs_all.update(new_kpis)

                # Remove newly found KPIs from the still-missing list so
                # subsequent reports don't redundantly extract them.
                for found_kpi in new_kpis:
                    if found_kpi in still_missing:
                        still_missing.remove(found_kpi)

                if new_rev and new_revenue is None:
                    new_revenue = new_rev
                    rev_still_needed = False

                # Store immediately linked to this specific report_id
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

            if still_missing:
                print(f"\n     KPIs not found in any report: {still_missing}")
    else:
        print("  [2/5] All KPIs and revenue cached — skipping pipeline entirely.")

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

    # ── Phase 4: Persist derived values ───────────────────────────────────────
    derived_to_store: dict = {}
    if "total_ghg_emissions" not in cached_kpis and "total_ghg_emissions" in merged_kpis:
        derived_to_store["total_ghg_emissions"] = merged_kpis["total_ghg_emissions"]

    revenue_result = cached_rev or new_revenue

    if (derived_to_store or new_revenue) and company_info and report_infos:
        sorted_for_store = sorted(
            report_infos,
            key=lambda r: _REPORT_TYPE_PRIORITY.get(r.report_type, 99),
        )
        target_report = sorted_for_store[0]
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
    print(f"  {'KPI':<35} {'Value':>16}  Unit   [Source]")
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
            src = rec.get("report_type", rec.get("method", "?"))
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
            "ESG Competitive Benchmarking — KPI-level cache + multi-source fallback.\n"
            "Applies Integrated > BRSR > ESG priority independently per KPI.\n"
            "Companies must be ingested first: python main.py ingest ..."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m esg_bench.run_benchmark \\
      --company1 "Infosys" --fy1 2025 \\
      --company2 "TCS"     --fy2 2024

  # Force re-extraction (bypass KPI cache):
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
                        help="Bypass KPI cache, re-run extraction on all reports")
    parser.add_argument("--no-llm",  action="store_true",
                        help="Disable LLM (rule-based summary, regex-only extraction)")
    args = parser.parse_args()

    print("\n" + "=" * W)
    print("  ESG COMPETITIVE BENCHMARKING — KPI-LEVEL CACHE + FALLBACK")
    print(f"  {args.company1} FY{args.fy1}  vs  {args.company2} FY{args.fy2}")
    print(f"  KPI priority per metric: Integrated > BRSR > ESG")
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