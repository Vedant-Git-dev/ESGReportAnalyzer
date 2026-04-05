"""
esg_bench/run_benchmark.py

Terminal CLI — DB-first ESG benchmarking pipeline.

Interface (no PDF arguments — PDF path comes from reports.file_path in DB):
    python -m esg_bench.run_benchmark \
        --company1 "Infosys" --fy1 2025 \
        --company2 "TCS"     --fy2 2024 \
        [--force]   # skip DB cache, re-run full pipeline
        [--no-llm]  # rule-based summary only

Flow per company:
  1. DB lookup  — check kpi_records + reports.revenue_cr
     → if all KPIs + revenue found: skip to normalise (fast path)
     → if missing: run full pipeline
  2. Full pipeline (ExtractionAgent architecture):
       a. ParseOrchestrator  → parse PDF → DocumentChunks in DB
       b. ExtractionAgent    → regex → LLM → validation on DB chunks
       c. extract_revenue()  → regex → LLM → validation
  3. Store results to DB (append-only, skip duplicates)
  4. Derive total_ghg = scope_1 + scope_2 (never explicitly stated in BRSR)
  5. Plausibility guard — drop any KPI whose absolute value is outside the
     realistic range for an Indian large-cap; prevents inflated LLM values
     from poisoning intensity ratios
  6. Normalise → intensity ratios → compare → summarise

Bugs fixed in this version:
  [BUG-1] DetachedInstanceError on _report.file_path
          Root cause: _db_lookup returned live ORM objects; accessing them
          after the session closed raised DetachedInstanceError.
          Fix: _db_lookup now returns plain Python dicts/dataclasses only.
          File path stored as a plain str, IDs stored as uuid.UUID.

  [BUG-2] LLM returns absurd values (77M tCO2e scope_1, 20M MT waste)
          Root cause: no plausibility guard between extraction and ratio
          computation. Bad LLM values passed straight to benchmark.py.
          Fix: _validate_kpi_plausibility() checks every extracted value
          against hard physical limits per KPI before building ratios.
          Values outside the range are dropped (logged as warnings).
"""
from __future__ import annotations

import argparse
import sys
import uuid
from dataclasses import dataclass
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

# ─────────────────────────────────────────────────────────────────────────────
# Plausibility limits for KPI absolute values (after normalisation to canonical unit)
# These are hard physical limits — values outside them are extraction errors.
#
# Calibrated for Indian large-cap companies (IT, manufacturing, energy sectors).
# IT sector values are typically much lower than manufacturing/energy.
# Upper limits chosen to be 100× typical large-cap maxima to allow for
# genuine outliers while still catching absurd LLM hallucinations.
#
# Units match canonical units in services/normalizer.py:
#   energy      → GJ
#   scope_1/2   → tCO2e
#   total_ghg   → tCO2e
#   water       → KL
#   waste       → MT
# ─────────────────────────────────────────────────────────────────────────────
_KPI_PLAUSIBILITY: dict[str, tuple[float, float]] = {
    # (min, max) in canonical unit
    "energy_consumption":  (1_000,       500_000_000),   # 1 GJ → 500 PJ
    "scope_1_emissions":   (1,           5_000_000),     # 1 tCO2e → 5M tCO2e
    "scope_2_emissions":   (1,           5_000_000),
    "total_ghg_emissions": (1,           10_000_000),    # scope1+scope2 combined
    "water_consumption":   (100,         100_000_000),   # 100 KL → 100B KL
    "waste_generated":     (0.1,         500_000),       # 0.1 MT → 500k MT
}

# Intensity ratio ceilings per KPI (per INR Crore).
# A ratio above this ceiling is almost certainly a unit error or LLM hallucination.
_RATIO_CEILINGS: dict[str, float] = {
    "energy_consumption":  1_000,    # GJ/Cr   — TCS ≈ 7100 only because MJ not converted
    "scope_1_emissions":   10,       # tCO2e/Cr
    "scope_2_emissions":   10,
    "total_ghg_emissions": 20,
    "water_consumption":   500,      # KL/Cr
    "waste_generated":     5,        # MT/Cr
}


def _validate_kpi_plausibility(kpi_name: str, value: float, unit: str) -> bool:
    """
    Return True if the normalised absolute value is within realistic range.
    Normalises the raw value to canonical unit first, then checks limits.
    """
    limits = _KPI_PLAUSIBILITY.get(kpi_name)
    if limits is None:
        return True  # unknown KPI — let it through
    lo, hi = limits
    try:
        norm = normalize(kpi_name=kpi_name, value=value, unit=unit)
        v = norm.normalized_value
    except NormalizationError:
        return False  # can't normalise → not plausible
    if v < lo or v > hi:
        logger.warning(
            "run_benchmark.implausible_value",
            kpi=kpi_name, value=value, unit=unit,
            normalized=v, limit_lo=lo, limit_hi=hi,
        )
        return False
    return True


def _validate_ratio_plausibility(kpi_name: str, ratio_value: float) -> bool:
    """Return True if the intensity ratio (KPI/Cr) is within realistic range."""
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
# Plain-data containers returned from DB queries
# (NO live ORM objects — avoids DetachedInstanceError)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _CompanyInfo:
    id:   uuid.UUID
    name: str

@dataclass
class _ReportInfo:
    id:        uuid.UUID
    file_path: Optional[str]
    status:    str


# ─────────────────────────────────────────────────────────────────────────────
# Revenue helper — reads from plain values, not ORM
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
# DB read — returns plain dicts/dataclasses only (BUG-1 fix)
# ─────────────────────────────────────────────────────────────────────────────

def _db_lookup(company_name: str, fy: int) -> dict:
    """
    Check DB for cached KPI records and revenue.

    BUG-1 FIX: All ORM objects are converted to plain Python values before
    the session closes. No ORM object is ever returned to the caller.
    This prevents DetachedInstanceError when attributes are accessed later.

    Returns:
        {
          "kpis":    {kpi_name: {"value","unit","method","confidence"}},
          "revenue": RevenueResult | None,
          "company": _CompanyInfo | None,
          "report":  _ReportInfo | None,
        }
    """
    empty: dict = {"kpis": {}, "revenue": None, "company": None, "report": None}
    try:
        from core.database import get_db
        from models.db_models import Company, Report, KPIRecord, KPIDefinition
        with get_db() as db:
            company_row = (
                db.query(Company)
                .filter(Company.name.ilike(f"%{company_name}%"))
                .first()
            )
            if not company_row:
                return empty

            # Extract plain values immediately while session is open
            company_info = _CompanyInfo(
                id=company_row.id,
                name=company_row.name,
            )

            report_row = (
                db.query(Report)
                .filter(
                    Report.company_id == company_row.id,
                    Report.report_year == fy,
                )
                .order_by(Report.created_at.desc())
                .first()
            )

            report_info: Optional[_ReportInfo] = None
            cached_rev:  Optional[RevenueResult] = None

            if report_row:
                # Extract ALL values from ORM while session is alive
                report_info = _ReportInfo(
                    id=report_row.id,
                    file_path=report_row.file_path,
                    status=report_row.status,
                )
                cached_rev = _revenue_from_row(
                    getattr(report_row, "revenue_cr",     None),
                    getattr(report_row, "revenue_unit",   None),
                    getattr(report_row, "revenue_source", None),
                )

            kpis: dict = {}
            if report_row:
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
                        .filter(
                            KPIRecord.company_id == company_row.id,
                            KPIRecord.kpi_definition_id == kdef.id,
                            KPIRecord.report_year == fy,
                            KPIRecord.normalized_value.isnot(None),
                        )
                        .order_by(KPIRecord.extracted_at.desc())
                        .first()
                    )
                    if rec:
                        # Only keep values that pass plausibility check
                        val  = rec.normalized_value
                        unit = rec.unit or kdef.expected_unit
                        if _validate_kpi_plausibility(kpi_name, val, unit):
                            kpis[kpi_name] = {
                                "value":      val,
                                "unit":       unit,
                                "method":     rec.extraction_method,
                                "confidence": rec.confidence or 0.9,
                            }
                        else:
                            print(f"  [DROP] {kpi_name} from DB: value {val} {unit} "
                                  f"outside plausible range — will re-extract")

            return {
                "kpis":    kpis,
                "revenue": cached_rev,
                "company": company_info,
                "report":  report_info,
            }

    except Exception as exc:
        logger.warning("run_benchmark.db_lookup_failed", error=str(exc))
        return empty


# ─────────────────────────────────────────────────────────────────────────────
# DB write
# ─────────────────────────────────────────────────────────────────────────────

def _db_store(
    company_id: uuid.UUID,
    report_id:  uuid.UUID,
    fy:         int,
    new_kpis:   dict,
    revenue_result: Optional[RevenueResult],
) -> None:
    """Persist extracted KPIs + revenue (append-only, skip exact duplicates)."""
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
                exists = (
                    db.query(KPIRecord)
                    .filter(
                        KPIRecord.company_id == company_id,
                        KPIRecord.kpi_definition_id == kdef.id,
                        KPIRecord.report_year == fy,
                        KPIRecord.normalized_value == rec["value"],
                    )
                    .first()
                )
                if exists:
                    continue
                db.add(KPIRecord(
                    company_id=company_id,
                    report_id=report_id,
                    kpi_definition_id=kdef.id,
                    report_year=fy,
                    raw_value=str(rec["value"]),
                    normalized_value=rec["value"],
                    unit=rec["unit"],
                    extraction_method=rec["method"],
                    confidence=rec["confidence"],
                    is_validated=rec["confidence"] >= 0.85,
                    validation_notes="extracted by esg_bench",
                ))
    except Exception as exc:
        logger.warning("run_benchmark.db_store_failed", error=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Report resolution — returns plain _ReportInfo or None
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_report(
    company_name: str,
    fy: int,
    existing_company: Optional[_CompanyInfo],
    existing_report:  Optional[_ReportInfo],
) -> tuple[Optional[_CompanyInfo], Optional[_ReportInfo]]:
    """
    Ensure Company + Report exist in DB and report has a usable file_path.
    Returns plain _CompanyInfo / _ReportInfo (no live ORM objects).
    """
    try:
        from core.database import get_db
        from models.db_models import Company, Report
        with get_db() as db:
            # Company
            if existing_company:
                cid = existing_company.id
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
                    return None, None
                cid   = row.id
                cname = row.name

            company_info = _CompanyInfo(id=cid, name=cname)

            # Report — prefer already-resolved one, otherwise query
            if existing_report:
                return company_info, existing_report

            report_row = (
                db.query(Report)
                .filter(
                    Report.company_id == cid,
                    Report.report_year == fy,
                    Report.status.in_(["downloaded", "parsed", "extracted"]),
                )
                .order_by(Report.created_at.desc())
                .first()
            )

            if report_row is None:
                any_row = (
                    db.query(Report)
                    .filter(Report.company_id == cid, Report.report_year == fy)
                    .first()
                )
                if any_row:
                    print(f"  [WARN] Report exists but status='{any_row.status}'. "
                          f"Must be downloaded first.")
                    print(f"         Run: python main.py ingest --company \"{company_name}\" "
                          f"--year {fy}")
                else:
                    print(f"  [WARN] No report for {company_name} FY{fy}.")
                    print(f"         Run: python main.py ingest --company \"{company_name}\" "
                          f"--year {fy}")
                return company_info, None

            # Extract plain values while session is alive
            report_info = _ReportInfo(
                id=report_row.id,
                file_path=report_row.file_path,
                status=report_row.status,
            )

            if not report_info.file_path or not Path(report_info.file_path).exists():
                print(f"  [WARN] PDF not found at: {report_info.file_path}")
                print(f"         Re-run ingestion.")
                return company_info, None

            return company_info, report_info

    except Exception as exc:
        logger.error("run_benchmark.resolve_report_failed", error=str(exc))
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Full extraction pipeline (ExtractionAgent architecture)
# ─────────────────────────────────────────────────────────────────────────────

def _run_full_pipeline(
    report_info:  _ReportInfo,
    fy:           int,
    missing_kpis: list[str],
    need_revenue: bool,
    llm_service,
) -> tuple[dict, Optional[RevenueResult]]:
    """
    Run ParseOrchestrator + ExtractionAgent + revenue extractor.

    Returns:
        (new_kpi_records, revenue_result)
        new_kpi_records: only values that pass plausibility check
    """
    from services.parse_orchestrator import ParseOrchestrator
    from agents.extraction_agent import ExtractionAgent
    from core.database import get_db

    pdf_path   = Path(report_info.file_path)
    new_kpis:  dict = {}
    new_rev:   Optional[RevenueResult] = None

    # ── Parse + chunk (idempotent — orchestrator checks cache) ────────────────
    print("  [2a] ParseOrchestrator (parse cache check)...")
    try:
        parsed = ParseOrchestrator().run(report_id=report_info.id)
        print(f"       {parsed.page_count} pages, "
              f"{parsed.meta.get('chunk_count','?')} chunks")
    except Exception as exc:
        logger.error("run_benchmark.parse_failed", error=str(exc))
        print(f"  [ERROR] Parsing failed: {exc}")
        return new_kpis, new_rev

    # ── ExtractionAgent (regex → LLM → validation) ───────────────────────────
    kpis_to_extract = [k for k in missing_kpis if k != "total_ghg_emissions"]
    if kpis_to_extract:
        print(f"  [2b] ExtractionAgent → {kpis_to_extract}")
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
                # BUG-2 FIX: reject implausible extracted values
                if not _validate_kpi_plausibility(ext.kpi_name, val, unit):
                    print(f"       [DROP] {ext.kpi_name}: {val:,.2f} {unit} — "
                          f"outside plausible range (likely LLM error)")
                    continue
                new_kpis[ext.kpi_name] = {
                    "value":      val,
                    "unit":       unit,
                    "method":     ext.extraction_method,
                    "confidence": ext.confidence or 0.5,
                }
                print(f"       ✓ {ext.kpi_name}: {val:,.2f} {unit} "
                      f"[{ext.extraction_method} conf={ext.confidence:.2f}]")
            for k in kpis_to_extract:
                if k not in new_kpis:
                    print(f"       ✗ {k}: not found")
        except Exception as exc:
            logger.error("run_benchmark.agent_failed", error=str(exc))
            print(f"  [ERROR] ExtractionAgent: {exc}")

    # ── Revenue ───────────────────────────────────────────────────────────────
    if need_revenue:
        print("  [2c] Revenue extraction...")
        try:
            new_rev = extract_revenue(
                pdf_path=pdf_path,
                fiscal_year_hint=fy,
                llm_service=llm_service,
            )
            if new_rev:
                print(f"       ✓ revenue: ₹{new_rev.value_cr:,.0f} Cr "
                      f"[{new_rev.pattern_name} conf={new_rev.confidence:.2f}]")
            else:
                print(f"       ✗ revenue: not found — "
                      f"default ₹{_DEFAULT_REVENUE_CR:,.0f} Cr will be used")
        except Exception as exc:
            logger.warning("run_benchmark.revenue_failed", error=str(exc))

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
    DB-first pipeline for one company. Returns CompanyProfile.
    All intermediate state is plain Python — no live ORM objects.
    """
    print(f"\n{'─' * W}")
    print(f"  {company_name}  FY{fy}")
    print(f"{'─' * W}")

    # ── Step 1: DB lookup (returns only plain data) ────────────────────────────
    cached_kpis: dict = {}
    cached_rev:  Optional[RevenueResult] = None
    company_info: Optional[_CompanyInfo] = None
    report_info:  Optional[_ReportInfo]  = None

    if not force:
        print("  [1/5] Checking DB cache...")
        db_data      = _db_lookup(company_name, fy)
        cached_kpis  = db_data["kpis"]
        cached_rev   = db_data["revenue"]
        company_info = db_data["company"]
        report_info  = db_data["report"]

        if cached_kpis:
            print(f"        {len(cached_kpis)} KPIs cached: {list(cached_kpis)}")
        else:
            print("        No cached KPIs")
        print(f"        Revenue: {'₹' + f'{cached_rev.value_cr:,.0f} Cr' if cached_rev else 'not cached'}")
    else:
        print("  [1/5] --force: skipping DB cache")
        db_data      = _db_lookup(company_name, fy)
        company_info = db_data["company"]
        report_info  = db_data["report"]

    # ── Step 2: Full pipeline for missing data ─────────────────────────────────
    extractable  = [k for k in TARGET_KPI_NAMES if k != "total_ghg_emissions"]
    missing_kpis = [k for k in extractable if k not in cached_kpis]
    need_revenue = cached_rev is None

    new_kpi_recs: dict = {}
    new_revenue:  Optional[RevenueResult] = None

    if missing_kpis or need_revenue or force:
        needs = []
        if missing_kpis: needs.append(f"KPIs: {missing_kpis}")
        if need_revenue:  needs.append("revenue")
        print(f"  [2/5] Pipeline needed → {' + '.join(needs)}")

        company_info, report_info = _resolve_report(
            company_name=company_name,
            fy=fy,
            existing_company=company_info,
            existing_report=report_info,
        )

        if company_info and report_info:
            new_kpi_recs, new_revenue = _run_full_pipeline(
                report_info=report_info,
                fy=fy,
                missing_kpis=missing_kpis if not force else extractable,
                need_revenue=need_revenue or force,
                llm_service=llm_service,
            )
        else:
            print("  [SKIP] Cannot extract — report not available in DB.")
            print("         Continuing with partial cache only.")
    else:
        print("  [2/5] All data in DB — skipping pipeline ✓")

    # ── Step 3: Merge + derive total_ghg ──────────────────────────────────────
    merged_kpis = {**cached_kpis, **new_kpi_recs}

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
            print("  [3/5] Cannot derive total_ghg (scope_1 or scope_2 missing)")
    else:
        print("  [3/5] total_ghg_emissions already present ✓")

    # ── Step 4: Persist new results ────────────────────────────────────────────
    to_store = dict(new_kpi_recs)
    if "total_ghg_emissions" not in cached_kpis and "total_ghg_emissions" in merged_kpis:
        to_store["total_ghg_emissions"] = merged_kpis["total_ghg_emissions"]

    if (to_store or new_revenue) and company_info and report_info:
        print("  [4/5] Storing to DB...")
        _db_store(
            company_id=company_info.id,
            report_id=report_info.id,
            fy=fy,
            new_kpis=to_store,
            revenue_result=new_revenue,
        )
        print(f"        {len(to_store)} KPI record(s) stored")
    else:
        print("  [4/5] Nothing new to store")

    # ── Step 5: Normalise + display ────────────────────────────────────────────
    revenue_result = cached_rev or new_revenue

    print(f"\n  [5/5] Normalised KPIs — {company_name} FY{fy}")
    print(f"  {'KPI':<35} {'Value':>16}  Unit   [Source]")
    print("  " + "─" * 64)

    for kpi_name in TARGET_KPI_NAMES:
        rec = merged_kpis.get(kpi_name)
        if not rec:
            print(f"  {'  ' + kpi_name:<35} {'NOT FOUND':>16}")
            continue
        try:
            n = normalize(kpi_name=kpi_name, value=float(rec["value"]), unit=str(rec["unit"]))
            conv = (f"  ← {rec['value']:,.2f} {rec['unit']}"
                    if n.conversion_factor != 1.0 else "")
            print(f"  {'  ' + kpi_name:<35} {n.normalized_value:>14,.2f}  "
                  f"{n.normalized_unit}  [{rec.get('method','?')}]{conv}")
        except NormalizationError as e:
            print(f"  {'  ' + kpi_name:<35} {rec['value']:>14,.2f}  "
                  f"{rec['unit']}  [WARN: {e}]")

    rev_cr  = revenue_result.value_cr if revenue_result else _DEFAULT_REVENUE_CR
    rev_src = revenue_result.source   if revenue_result else "default"
    print(f"  {'  revenue':<35} {rev_cr:>14,.0f}  INR_Crore  [{rev_src}]")

    # ── Build CompanyProfile ──────────────────────────────────────────────────
    # Load page texts for reported-ratio detection (intensity per rupee in PDF)
    page_texts: list[str] = []
    if report_info and report_info.file_path:
        try:
            import fitz
            doc = fitz.open(str(report_info.file_path))
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

    # BUG-2 FIX: drop ratios that exceed plausibility ceilings after computation
    bad_ratios = [
        kpi for kpi, ratio in profile.ratios.items()
        if not _validate_ratio_plausibility(kpi, ratio.ratio_value)
    ]
    for kpi in bad_ratios:
        print(f"  [DROP ratio] {kpi}: ratio {profile.ratios[kpi].ratio_value:.4e} "
              f"{profile.ratios[kpi].ratio_unit} exceeds ceiling — excluded from comparison")
        del profile.ratios[kpi]

    print(f"\n  Intensity ratios ({company_name} FY{fy}):")
    if profile.ratios:
        for kpi_name, ratio in profile.ratios.items():
            src = "[reported]" if "reported" in ratio.ratio_source else "[computed]"
            v   = (f"{ratio.ratio_value:.4e}" if ratio.ratio_value < 0.001
                   else f"{ratio.ratio_value:.4f}")
            print(f"  {'  ' + kpi_name:<35} {v:>14}  {ratio.ratio_unit}  {src}")
    else:
        print("  (no valid intensity ratios — all values failed plausibility check)")

    return profile


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "ESG Competitive Benchmarking — DB-first, no PDF arguments.\n"
            "Companies must be ingested into the database first."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m esg_bench.run_benchmark \\
      --company1 "Infosys" --fy1 2025 \\
      --company2 "TCS"     --fy2 2024

  # Force re-extraction:
  python -m esg_bench.run_benchmark \\
      --company1 "Infosys" --fy1 2025 \\
      --company2 "TCS"     --fy2 2024 --force

  # Populate DB first:
  python main.py ingest --company "Infosys" --year 2025
  python main.py ingest --company "TCS"     --year 2024
        """,
    )
    parser.add_argument("--company1", required=True)
    parser.add_argument("--fy1",      required=True, type=int)
    parser.add_argument("--company2", required=True)
    parser.add_argument("--fy2",      required=True, type=int)
    parser.add_argument("--force",   action="store_true",
                        help="Skip DB cache, re-run full extraction")
    parser.add_argument("--no-llm",  action="store_true",
                        help="Disable LLM (rule-based summary, regex-only extraction)")
    args = parser.parse_args()

    print("\n" + "═" * W)
    print("  ESG COMPETITIVE BENCHMARKING PIPELINE")
    print(f"  {args.company1} FY{args.fy1}  vs  {args.company2} FY{args.fy2}")
    print("═" * W)

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

    print("\n" + "═" * W)
    print("  NARRATIVE SUMMARY")
    print("═" * W + "\n")
    print(generate_summary([profile1, profile2], report, llm=llm_service))
    print()


if __name__ == "__main__":
    main()