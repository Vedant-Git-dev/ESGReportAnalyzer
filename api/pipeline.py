"""
api/pipeline.py

Core pipeline logic extracted from dashboard/ui.py — pure Python, no Streamlit,
no HTTP concerns. Every function here calls the EXACT same backend services
ui.py called. Zero agent/service code was changed.

This module is imported by the FastAPI routers. The Streamlit ui.py can still
import from these same services directly — this file is additive only.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from api.schemas import (
    ALL_KPI_NAMES, DEFAULT_REVENUE_CR, KPI_GROUPS,
    REPORT_TYPE_PRIORITY,
)


# ---------------------------------------------------------------------------
# Internal data containers (mirror ui.py CompanyData / ReportInfo)
# ---------------------------------------------------------------------------

@dataclass
class ReportInfo:
    id:          uuid.UUID
    report_type: str
    file_path:   Optional[str]
    status:      str


@dataclass
class CompanyData:
    company_name:   str
    fy:             int
    sector:         str
    kpi_records:    dict
    revenue_result: object
    log:            list[str]
    company_id:     Optional[uuid.UUID] = None
    report_infos:   list[ReportInfo]    = field(default_factory=list)
    file_path:      Optional[str]       = None


# ---------------------------------------------------------------------------
# DB helpers (verbatim from ui.py, minus st.warning calls)
# ---------------------------------------------------------------------------

def check_db() -> bool:
    try:
        from core.database import check_connection
        return check_connection()
    except Exception:
        return False


def get_llm_service():
    try:
        from services.llm_service import LLMService
        from core.config import get_settings
        if get_settings().llm_api_key:
            return LLMService()
    except Exception:
        pass
    return None


def list_companies() -> list[dict]:
    """Return all non-deactivated companies, ordered by name."""
    try:
        from core.database import get_db
        from models.db_models import Company
        with get_db() as db:
            rows = (
                db.query(Company)
                # Treat NULL as active for backward compatibility with
                # legacy rows created before is_active was consistently set.
                .filter(Company.is_active.isnot(False))
                .order_by(Company.name)
                .all()
            )
            return [{"id": str(r.id), "name": r.name, "sector": r.sector} for r in rows]
    except Exception:
        return []


def list_companies_by_sector(sector: str) -> list[dict]:
    """
    Return active companies whose sector matches the given string
    (case-insensitive exact match).  Falls back to list_companies()
    when sector is empty/None so callers never get an empty list
    due to a missing sector filter.
    """
    if not sector or not sector.strip():
        return list_companies()

    try:
        from core.database import get_db
        from models.db_models import Company
        with get_db() as db:
            rows = (
                db.query(Company)
                .filter(
                    Company.is_active.isnot(False),
                    Company.sector.ilike(sector.strip()),
                )
                .order_by(Company.name)
                .all()
            )
            return [{"id": str(r.id), "name": r.name, "sector": r.sector} for r in rows]
    except Exception:
        return []


def db_check_exact_match(company_name: str, fy: int) -> dict:
    """
    Check for EXACT company name (case-insensitive) + EXACT year match.
    Returns exact match data if found, otherwise returns empty.
    This is used as the first check before falling back to search.
    """
    empty = {"exists": False, "company_id": None, "reports": [], "exact_match": False}
    try:
        from core.database import get_db
        from models.db_models import Company, Report

        with get_db() as db:
            # Exact match on company name (case-insensitive)
            company_row = (
                db.query(Company)
                .filter(Company.name.ilike(company_name.strip()))
                .first()
            )
            if not company_row:
                return {**empty, "exact_match": False}

            company_id = company_row.id

            # Exact match on year
            reports = (
                db.query(Report)
                .filter(
                    Report.company_id == company_id,
                    Report.report_year == fy,
                    Report.status.in_(["downloaded", "parsed", "extracted"]),
                )
                .order_by(Report.created_at.desc())
                .all()
            )

            if not reports:
                # Company exists but no report for this year
                return {**empty, "exists": False, "company_id": company_id, "exact_match": True}

            infos = sorted(
                [ReportInfo(id=r.id, report_type=r.report_type,
                            file_path=r.file_path, status=r.status)
                 for r in reports],
                key=lambda ri: (
                    REPORT_TYPE_PRIORITY.get(ri.report_type, 99),
                    0 if ri.file_path and Path(ri.file_path).exists() else 1,
                ),
            )
            return {"exists": True, "company_id": company_id, "reports": infos, "exact_match": True}

    except Exception:
        return empty


def db_get_all_reports(company_name: str, fy: int) -> dict:
    """
    Get reports for a company. First tries exact match, then falls back to partial match.
    Returns dict with exists, company_id, reports, and exact_match flag.
    """
    # First try exact match
    exact_result = db_check_exact_match(company_name, fy)
    if exact_result["exists"]:
        return exact_result

    # If company exists but no report for this year, return info for search
    if exact_result.get("company_id") and exact_result.get("exact_match"):
        return {"exists": False, "company_id": exact_result["company_id"], "reports": [], "exact_match": True, "company_exists": True}

    # Fall back to partial match (for backwards compatibility)
    empty = {"exists": False, "company_id": None, "reports": [], "exact_match": False}
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

            company_id = company_row.id
            reports = (
                db.query(Report)
                .filter(
                    Report.company_id == company_id,
                    Report.report_year == fy,
                    Report.status.in_(["downloaded", "parsed", "extracted"]),
                )
                .order_by(Report.created_at.desc())
                .all()
            )
            if not reports:
                return empty

            infos = sorted(
                [ReportInfo(id=r.id, report_type=r.report_type,
                            file_path=r.file_path, status=r.status)
                 for r in reports],
                key=lambda ri: (
                    REPORT_TYPE_PRIORITY.get(ri.report_type, 99),
                    0 if ri.file_path and Path(ri.file_path).exists() else 1,
                ),
            )
            return {"exists": True, "company_id": company_id, "reports": infos, "exact_match": False}

    except Exception:
        return empty


def ensure_company_sector(company_id: uuid.UUID, sector: str) -> None:
    """
    Backfill missing/blank sector for an existing company row.
    This keeps sector-filtered dropdowns consistent when a company was
    discovered earlier without sector metadata.
    """
    if not company_id or not sector or not sector.strip():
        return
    try:
        from core.database import get_db
        from models.db_models import Company

        wanted = sector.strip()
        with get_db() as db:
            row = db.query(Company).filter(Company.id == company_id).first()
            if not row:
                return
            if row.sector is None or not str(row.sector).strip():
                row.sector = wanted
                db.flush()
    except Exception:
        pass


def cache_load(company_id: uuid.UUID, fy: int) -> dict:
    try:
        from core.database import get_db
        from services.kpi_cache_service import KPICacheService
        with get_db() as db:
            return KPICacheService().select_best_per_kpi(
                company_id=company_id, fy=fy, kpi_names=ALL_KPI_NAMES, db=db,
            )
    except Exception:
        return {}


def cache_load_revenue(company_name: str, fy: int, kpi: str = "revenue_from_operations"):
    """
    Load revenue from revenue_search_cache table (authoritative source).
    Strict flow:
    1. Check revenue_search_cache table first (authoritative source)
    2. If not found, run web search and store in revenue_search_cache
    3. Return result from revenue_search_cache

    Args:
        company_name: Company name
        fy: Fiscal year
        kpi: Which KPI — "revenue_from_operations" or "net_revenue"
    """
    try:
        from core.database import get_db
        from models.db_models import RevenueSearchCache
        from services.revenue_extractor import RevenueResult

        with get_db() as db:
            # Step 1: Check revenue_search_cache first
            cached = (
                db.query(RevenueSearchCache)
                .filter(
                    RevenueSearchCache.company_name.ilike(company_name),
                    RevenueSearchCache.fiscal_year == fy,
                    RevenueSearchCache.kpi == kpi,
                    RevenueSearchCache.is_valid == True,
                    RevenueSearchCache.revenue_cr.isnot(None),
                )
                .order_by(RevenueSearchCache.confidence.desc())
                .first()
            )
            if cached:
                return RevenueResult(
                    value_cr=cached.revenue_cr,
                    raw_value=str(cached.revenue_cr),
                    raw_unit="INR_Crore",
                    source="web_search",
                    page_number=0,
                    confidence=cached.confidence,
                    pattern_name=f"web_{cached.source_domain}",
                )

            # Step 2: Web search if not in cache
            from services.revenue_llm_service import search_revenue, _store_in_cache
            web_result = search_revenue(company_name, fy, kpi=kpi)
            if web_result and web_result.confidence >= 0.80:
                # Store in revenue_search_cache
                _store_in_cache(web_result, kpi=kpi)
                return RevenueResult(
                    value_cr=web_result.value_cr,
                    raw_value=web_result.raw_value,
                    raw_unit="INR_Crore",
                    source="web_search",
                    page_number=0,
                    confidence=web_result.confidence,
                    pattern_name=f"web_{web_result.source_domain}",
                )

    except Exception:
        pass
    return None


def db_store_kpis(
    company_id: uuid.UUID,
    report_id: uuid.UUID,
    fy: int,
    kpi_records: dict,
    revenue_result,
) -> None:
    try:
        from core.database import get_db
        from services.kpi_cache_service import KPICacheService
        with get_db() as db:
            KPICacheService().store(
                company_id=company_id, report_id=report_id, fy=fy,
                kpi_records=kpi_records, revenue_result=revenue_result, db=db,
            )
    except Exception:
        pass


def db_load_kpis_and_revenue(company_id: uuid.UUID, fy: int, company_name: str = "") -> dict:
    """
    Load KPIs and revenue from database.
    Revenue always comes from revenue_search_cache (authoritative source).
    """
    empty = {"kpis": {}, "revenue": None, "file_path": None}
    try:
        from core.database import get_db
        from models.db_models import Report, KPIRecord, KPIDefinition, RevenueSearchCache
        from services.revenue_extractor import RevenueResult
        from sqlalchemy import case

        with get_db() as db:
            type_priority = case(
                (Report.report_type == "Integrated", 0),
                (Report.report_type == "BRSR",       1),
                (Report.report_type == "ESG",        2),
                else_=99,
            )
            best_report = (
                db.query(Report)
                .filter(
                    Report.company_id == company_id,
                    Report.report_year == fy,
                    Report.status.in_(["downloaded", "parsed", "extracted"]),
                    Report.file_path.isnot(None),
                )
                .order_by(type_priority, Report.created_at.desc())
                .first()
            )
            file_path = best_report.file_path if best_report else None

            # Revenue: ALWAYS from revenue_search_cache
            cached_rev = None
            if company_name:
                rev_cached = (
                    db.query(RevenueSearchCache)
                    .filter(
                        RevenueSearchCache.company_name.ilike(company_name),
                        RevenueSearchCache.fiscal_year == fy,
                        RevenueSearchCache.is_valid == True,
                        RevenueSearchCache.revenue_cr.isnot(None),
                    )
                    .order_by(RevenueSearchCache.confidence.desc())
                    .first()
                )
                if rev_cached:
                    try:
                        cached_rev = RevenueResult(
                            value_cr=float(rev_cached.revenue_cr),
                            raw_value=str(rev_cached.revenue_cr),
                            raw_unit="INR_Crore",
                            source="web_search",
                            page_number=0,
                            confidence=rev_cached.confidence,
                            pattern_name=f"web_{rev_cached.source_domain}",
                        )
                    except Exception:
                        pass

            kpis: dict = {}
            for kpi_name in ALL_KPI_NAMES:
                kdef = (
                    db.query(KPIDefinition)
                    .filter(KPIDefinition.name == kpi_name,
                            KPIDefinition.is_active == True)
                    .first()
                )
                if not kdef:
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
                if not rec:
                    continue
                val  = rec.normalized_value
                unit = rec.unit or kdef.expected_unit
                src_report = db.query(Report).filter(Report.id == rec.report_id).first()
                rec_type   = src_report.report_type if src_report else "unknown"
                kpis[kpi_name] = {
                    "value": val, "unit": unit,
                    "method": rec.extraction_method,
                    "confidence": rec.confidence or 0.9,
                    "report_type": rec_type,
                }

        return {"kpis": kpis, "revenue": cached_rev, "file_path": file_path}
    except Exception:
        return empty


def ensure_schema() -> None:
    try:
        from core.database import get_db
        from services.revenue_extractor import ensure_revenue_columns
        with get_db() as db:
            ensure_revenue_columns(db)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Pipeline steps (verbatim logic from ui.py, emit log lines instead of st.*)
# ---------------------------------------------------------------------------

def step_ingest(
    company_name: str, fy: int, sector: str, log: list[str],
    emit=None,
) -> dict:
    from agents.ingestion_agent import IngestionAgent
    from models.schemas import CompanyCreate

    def _emit(msg: str) -> None:
        log.append(msg)
        if emit:
            emit(msg)

    agent = IngestionAgent()
    company_data = CompanyCreate(name=company_name, sector=sector, country="India")
    _emit(f"Searching for {company_name} FY{fy} reports...")

    try:
        result = agent.run_multi_report_types(
            company_data=company_data, year=fy, auto_download=True,
        )
    except Exception as exc:
        _emit(f"  Search failed: {exc}")
        return {"company_id": None, "reports": []}

    company   = result["company"]
    downloads = result["downloaded_reports"]
    not_found = result.get("not_found_types", [])
    failed    = result.get("failed_types", [])

    for rtype in ["BRSR", "ESG", "Integrated"]:
        dl = next((d for d in downloads if d.report_type == rtype), None)
        if dl:
            fname   = Path(dl.file_path).name if dl.file_path else "unknown"
            size_mb = round((dl.file_size_bytes or 0) / (1024 * 1024), 1)
            _emit(f"  [{rtype}] downloaded: {fname} ({size_mb} MB)")
        elif rtype in not_found:
            _emit(f"  [{rtype}] not found.")
        elif rtype in failed:
            _emit(f"  [{rtype}] download failed.")

    if not downloads:
        _emit("No PDFs downloaded.")
        return {"company_id": company.id if company else None, "reports": []}

    reports = [
        ReportInfo(id=d.id, report_type=d.report_type,
                   file_path=d.file_path, status=d.status)
        for d in downloads if d.status == "downloaded"
    ]
    _emit(f"  {len(reports)} report(s) ready.")
    return {"company_id": company.id, "reports": reports}


def step_parse(
    report_id: uuid.UUID, report_type: str, log: list[str], emit=None,
) -> bool:
    from services.parse_orchestrator import ParseOrchestrator

    msg = f"Reading {report_type} report..."
    log.append(msg)
    if emit:
        emit(msg)

    try:
        result = ParseOrchestrator().run(report_id=report_id, force=False)
        detail = f"  {result.page_count} pages, {result.meta.get('chunk_count','?')} sections."
        log.append(detail)
        if emit:
            emit(detail)
        return True
    except Exception as exc:
        err = f"  Could not read report: {exc}"
        log.append(err)
        if emit:
            emit(err)
        return False


def step_extract_missing(
    report_id: uuid.UUID,
    report_type: str,
    fy: int,
    log: list[str],
    llm_service,
    missing_kpi_names: list[str],
    need_revenue: bool,
    company_name: str = "",
    emit=None,
) -> dict:
    from agents.extraction_agent import ExtractionAgent
    from services.revenue_extractor import extract_revenue
    from core.database import get_db

    new_kpis: dict = {}
    new_revenue = None

    if not missing_kpi_names and not need_revenue:
        msg = f"  [{report_type}] All KPIs cached — skipping extraction."
        log.append(msg)
        if emit:
            emit(msg)
        return {"kpis": new_kpis, "revenue": new_revenue}

    prefix = f"  [{report_type}] (id={str(report_id)[:8]})"

    if missing_kpi_names:
        msg = f"Extracting {len(missing_kpi_names)} ESG metric(s)..."
        log.append(msg)
        if emit:
            emit(msg)
        try:
            with get_db() as db:
                extracted_list = ExtractionAgent().extract_all(
                    report_id=report_id, db=db, kpi_names=missing_kpi_names,
                )
            for ext in extracted_list:
                if ext.normalized_value is None:
                    continue
                val  = ext.normalized_value
                unit = ext.unit or ""
                new_kpis[ext.kpi_name] = {
                    "value":      val,
                    "unit":       unit,
                    "method":     ext.extraction_method,
                    "confidence": ext.confidence or 0.5,
                }
                detail = (
                    f"    {ext.kpi_name}: {val:,.2f} {unit} "
                    f"[{ext.extraction_method} conf={ext.confidence:.2f}]"
                )
                log.append(detail)
                if emit:
                    emit(detail)
        except Exception as exc:
            log.append(f"    Extraction failed: {exc}")
    else:
        log.append(f"{prefix} No missing KPIs — skipping.")

    if need_revenue:
        try:
            from models.db_models import Report
            with get_db() as db:
                rpt = db.query(Report).filter(Report.id == report_id).first()
                pdf_path_str = rpt.file_path if rpt else None
        except Exception:
            pdf_path_str = None

        if pdf_path_str and Path(pdf_path_str).exists():
            try:
                new_revenue = extract_revenue(
                    pdf_path=Path(pdf_path_str),
                    fiscal_year_hint=fy,
                    llm_service=llm_service,
                    company_name=company_name,
                )
                if new_revenue:
                    rev_msg = f"    Revenue: INR {new_revenue.value_cr:,.0f} Cr"
                    log.append(rev_msg)
                    if emit:
                        emit(rev_msg)
            except Exception as exc:
                log.append(f"    Revenue extraction failed: {exc}")

    return {"kpis": new_kpis, "revenue": new_revenue}


# ---------------------------------------------------------------------------
# Main company pipeline (verbatim logic from ui.py run_company_pipeline)
# ---------------------------------------------------------------------------

def run_company_pipeline(
    company_name: str,
    fy: int,
    sector: str,
    llm_service,
    emit=None,
) -> CompanyData:
    """
    Full pipeline for one company. Identical logic to ui.py run_company_pipeline.
    emit(msg) is called for each progress step (used for SSE streaming).
    """
    log: list[str] = []

    def _emit(msg: str) -> None:
        log.append(msg)
        if emit:
            emit(msg)

    # Step 1 — resolve company + reports
    # First check: exact company name + exact year match in DB
    _emit(f"Checking for {company_name} FY{fy} in database...")
    db_data    = db_get_all_reports(company_name, fy)
    company_id = db_data.get("company_id")
    report_infos: list[ReportInfo] = []
    exact_match = db_data.get("exact_match", False)

    if db_data["exists"]:
        if exact_match:
            _emit(f"✓ Found exact match in database (FY{fy})")
        else:
            _emit(f"✓ Found company in database (FY{fy})")
        report_infos = db_data["reports"]
    else:
        if exact_match:
            # Company exists but no report for this year - need to search
            _emit(f"No report found for {company_name} FY{fy}. Searching...")
        else:
            # Company not found in DB - need to search
            _emit(f"{company_name} not in database. Searching for reports...")
        ingest_result = step_ingest(company_name, fy, sector, log, emit)
        company_id    = ingest_result.get("company_id") or company_id
        report_infos  = ingest_result.get("reports", [])
        if not report_infos:
            _emit(f"No report found for {company_name} FY{fy}.")
            return CompanyData(
                company_name=company_name, fy=fy, sector=sector,
                kpi_records={}, revenue_result=None, log=log,
                company_id=company_id,
            )
        db_data    = db_get_all_reports(company_name, fy)
        company_id = db_data.get("company_id") or company_id
        if db_data["exists"]:
            report_infos = db_data["reports"]

    # Ensure companies discovered in older runs (often with NULL sector)
    # become visible in sector-scoped dropdowns.
    if company_id:
        ensure_company_sector(company_id, sector)

    # Step 2 — KPI-level cache check
    _emit("Checking for previously extracted data...")
    cached_kpis: dict = {}
    cached_revenue_ops = None
    cached_net_revenue = None

    if company_id:
        cached_kpis       = cache_load(company_id, fy)
        cached_revenue_ops = cache_load_revenue(company_name, fy, kpi="revenue_from_operations")
        cached_net_revenue = cache_load_revenue(company_name, fy, kpi="net_revenue")

    missing_kpis = [k for k in ALL_KPI_NAMES if k not in cached_kpis]
    need_rev_ops  = cached_revenue_ops is None
    need_net_rev  = cached_net_revenue is None
    _emit(
        f"Cache check: {len(cached_kpis)} KPI(s) already available, "
        f"{len(missing_kpis)} still missing."
    )

    if not missing_kpis and not need_rev_ops and not need_net_rev:
        _emit(f"All data loaded from cache ({len(cached_kpis)} metrics).")
        final_db = db_load_kpis_and_revenue(company_id, fy, company_name) if company_id else {}
        # Build revenue_result dict for both KPIs
        rev_ops_result = cached_revenue_ops
        net_rev_result = cached_net_revenue
        return CompanyData(
            company_name=company_name, fy=fy, sector=sector,
            kpi_records=cached_kpis, revenue_result=rev_ops_result,
            log=log, company_id=company_id,
            report_infos=report_infos, file_path=final_db.get("file_path"),
        )

    # Step 3 — multi-report extraction
    sorted_reports = sorted(
        report_infos, key=lambda r: REPORT_TYPE_PRIORITY.get(r.report_type, 99),
    )
    still_missing  = list(missing_kpis)
    still_need_rev_ops = need_rev_ops
    still_need_net_rev = need_net_rev
    all_new_kpis:  dict = {}
    final_revenue_ops = cached_revenue_ops
    final_net_revenue = cached_net_revenue

    for ri in sorted_reports:
        if not still_missing and not still_need_rev_ops and not still_need_net_rev:
            break

        _emit(f"Reading {ri.report_type} report...")
        parse_ok = step_parse(ri.id, ri.report_type, log, emit=None)
        if not parse_ok:
            continue

        _emit(f"Extracting ESG metrics from {ri.report_type} report...")
        ext_result = step_extract_missing(
            report_id=ri.id, report_type=ri.report_type, fy=fy,
            log=log, llm_service=llm_service,
            missing_kpi_names=list(still_missing),
            need_revenue=still_need_rev_ops, emit=None,
            company_name=company_name,
        )
        new_kpis    = ext_result["kpis"]
        new_revenue = ext_result["revenue"]

        all_new_kpis.update(new_kpis)
        for found in list(new_kpis.keys()):
            if found in still_missing:
                still_missing.remove(found)

        if new_revenue and final_revenue_ops is None:
            final_revenue_ops = new_revenue
            still_need_rev_ops = False

        # Net revenue: search if not yet cached
        if still_need_net_rev:
            from services.revenue_llm_service import search_revenue, _store_in_cache
            net_rev = search_revenue(company_name, fy, kpi="net_revenue")
            if net_rev and net_rev.confidence >= 0.80:
                _store_in_cache(net_rev, kpi="net_revenue")
                final_net_revenue = net_rev
                still_need_net_rev = False
                log.append(f"    Net Revenue: INR {net_rev.value_cr:,.0f} Cr")

        if company_id and (new_kpis or new_revenue):
            db_store_kpis(company_id, ri.id, fy, new_kpis, new_revenue)

    # Step 4 — merge + final DB read
    merged_kpis  = {**all_new_kpis, **cached_kpis}
    final_db     = db_load_kpis_and_revenue(company_id, fy) if company_id else {}
    final_kpis   = final_db.get("kpis", {})
    final_merged = {**merged_kpis, **final_kpis}
    final_revenue_ops = final_revenue_ops or final_db.get("revenue")

    _emit(
        f"Comparison-ready KPI set: {len(final_merged)} total "
        f"({len(cached_kpis)} cached, {len(all_new_kpis)} newly extracted)."
    )

    if final_merged:
        _emit(f"Found {len(final_merged)} ESG metric(s).")
    else:
        _emit(f"No ESG data found for {company_name} FY{fy}.")

    return CompanyData(
        company_name=company_name, fy=fy, sector=sector,
        kpi_records=final_merged, revenue_result=final_revenue_ops,
        log=log, company_id=company_id,
        report_infos=report_infos, file_path=final_db.get("file_path"),
    )


# ---------------------------------------------------------------------------
# Benchmark builder (verbatim from ui.py _build_benchmark)
# ---------------------------------------------------------------------------

def build_benchmark(data1: CompanyData, data2: CompanyData, sector: str) -> dict:
    from services.benchmark import build_company_profile, compare_profiles
    from services.summary_generator import generate_summary, generate_recommendation
    from services.comparison_cache_service import ComparisonCacheService

    profiles = []
    for data in [data1, data2]:
        rev    = data.revenue_result
        rev_cr = rev.value_cr if rev else DEFAULT_REVENUE_CR
        rev_src = rev.source if rev else "default"

        # Inject financial KPIs from revenue result
        kpi_records = dict(data.kpi_records)
        if rev and rev.value_cr:
            kpi_records["revenue_from_operations"] = {
                "value":      rev.value_cr,
                "unit":       "INR_Crore",
                "method":     rev.source,
                "confidence": rev.confidence,
                "report_type": None,
            }
            emp_rec = kpi_records.get("employee_count")
            emp_val = float(emp_rec["value"]) if emp_rec and emp_rec.get("value") else 0
            if emp_val > 0:
                kpi_records["revenue_per_employee"] = {
                    "value":      rev.value_cr / emp_val,
                    "unit":       "INR_Crore",
                    "method":     "computed",
                    "confidence": rev.confidence,
                    "report_type": None,
                }
                # Store value_per_employee for Financial KPI display
                kpi_records["revenue_from_operations"]["value_per_employee"] = rev.value_cr / emp_val

        # Net revenue: load from revenue_search_cache if not in kpi_records
        if "net_revenue" not in kpi_records:
            net_rev = cache_load_revenue(data.company_name, data.fy, kpi="net_revenue")
            if net_rev:
                emp_rec = kpi_records.get("employee_count")
                emp_val = float(emp_rec["value"]) if emp_rec and emp_rec.get("value") else 0
                kpi_records["net_revenue"] = {
                    "value":      net_rev.value_cr,
                    "unit":       "INR_Crore",
                    "method":     net_rev.source,
                    "confidence": net_rev.confidence,
                    "report_type": None,
                }
                if emp_val > 0:
                    kpi_records["net_revenue"]["value_per_employee"] = net_rev.value_cr / emp_val

        page_texts: list[str] = []
        if data.file_path and Path(data.file_path).exists():
            try:
                import fitz
                doc = fitz.open(str(data.file_path))
                for pg in doc:
                    page_texts.append(pg.get_text())
                doc.close()
            except Exception:
                pass

        profile = build_company_profile(
            kpi_records=kpi_records, revenue_cr=rev_cr,
            revenue_source=rev_src, company_name=data.company_name,
            fiscal_year=data.fy, page_texts=page_texts,
        )
        profiles.append(profile)

    report   = compare_profiles(profiles)

    # Include all comparisons (no ceiling filtering to prevent data loss)
    # The max_ratio values in KPI_GROUPS are informational/validation only
    filtered = report.comparisons

    # Check comparison cache first
    cache_service = ComparisonCacheService()
    cache_result = cache_service.get(
        company1=data1.company_name,
        fy1=data1.fy,
        company2=data2.company_name,
        fy2=data2.fy,
    )

    llm = get_llm_service()

    if cache_result:
        # Use cached summary and recommendation
        summary = cache_result.summary
        recommendation = cache_result.recommendation
    else:
        # Generate new summary and recommendation using LLM
        summary = generate_summary(profiles, report, llm=llm)
        recommendation = generate_recommendation(profiles, report, llm=llm)

        # Store in cache
        cache_service.store(
            company1=data1.company_name,
            fy1=data1.fy,
            company2=data2.company_name,
            fy2=data2.fy,
            sector=sector,
            summary=summary,
            recommendation=recommendation,
        )

    return {
        "profiles":           profiles,
        "report":             report,
        "filtered":           filtered,
        "summary":            summary,
        "recommendation":     recommendation,
        "company1_records":   _profile_to_kpi_records(profiles[0]) if len(profiles) > 0 else {},
        "company2_records":   _profile_to_kpi_records(profiles[1]) if len(profiles) > 1 else {},
    }


def _profile_to_kpi_records(profile) -> dict:
    """Convert a CompanyProfile's raw_kpis to dict format for API response."""
    records = {}
    for kpi_name, norm_kpi in profile.raw_kpis.items():
        rec = {
            "value": norm_kpi.raw_value,
            "unit": norm_kpi.raw_unit,
        }
        if hasattr(norm_kpi, 'value_per_employee') and norm_kpi.value_per_employee is not None:
            rec["value_per_employee"] = norm_kpi.value_per_employee
        records[kpi_name] = rec
    return records