"""
dashboard/ui.py

ESG Competitive Intelligence Dashboard.

Multi-source pipeline (v2)
---------------------------
In v1, the pipeline selected ONE report per company (BRSR preferred) and
extracted KPIs from that single source.  This left data in the other
downloaded reports unused.

In v2, ALL downloaded reports for a company+year are processed:

  1. DB existence check
     Query ALL reports for (company_name, fiscal_year) with status in
     ('downloaded', 'parsed', 'extracted').  Return the full list.
     If any exist: skip ingestion. Proceed to parse+extract all of them.
     If none exist: run ingestion (downloads all three types), then process all.

  2. Multi-report parse + extract loop
     For EACH downloaded report, independently:
       a. ParseOrchestrator.run() — idempotent (cache-keyed on report_id + parser_version)
       b. ExtractionAgent.extract_all() — extracts KPIs from that report's chunks
       c. Store results linked to that specific report_id
     If one report fails, the others continue (error isolation).

  3. KPI selection for comparison
     After all reports are processed, read from DB across ALL reports.
     When multiple records exist for the same KPI (one per report type),
     select by priority: BRSR > Integrated > ESG (then newest as tiebreaker).
     This implements multi-source awareness at read time, not write time.

  4. No merging or conflict resolution at write time
     Each KPIRecord in the DB is linked to its source report_id.
     Raw values from BRSR and ESG coexist independently.
     The priority selection happens only when building the benchmark profile.

Pipeline contract
-----------------
  Step 1 — DB existence check
    Returns list of all usable report rows.  No priority selection here.

  Step 2 — Full pipeline for Company 1 (all reports)
    Ingestion -> Parse ALL -> Extract ALL -> Store each.
    Complete before Company 2 starts.

  Step 3 — Full pipeline for Company 2
    Same as Step 2.

  Step 4 — Load both from DB
    Read KPI records with BRSR > Integrated > ESG priority ordering.
    Build benchmark profiles. Render charts.

Run:
    streamlit run dashboard/ui.py
"""
from __future__ import annotations

import io
import sys
import tempfile
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import plotly.graph_objects as go
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="ESG Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# DESIGN TOKENS  (unchanged)
# =============================================================================

C = {
    "bg":      "#0F1117",
    "surface": "#1A1D27",
    "border":  "#2D3142",
    "text":    "#E8EAF0",
    "sub":     "#8B92A9",
    "green":   "#10B981",
    "blue":    "#4B84DE",
    "amber":   "#F59E0B",
    "red":     "#EF4444",
    "ca":      "#4B84DE",
    "cb":      "#10B981",
    "grid":    "#2D3142",
    "font":    "Inter, 'Helvetica Neue', Arial, sans-serif",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{{font-family:{C['font']};background:{C['bg']};color:{C['text']};}}
.stApp{{background:{C['bg']};}}
[data-testid="stSidebar"]{{background:{C['surface']};border-right:1px solid {C['border']};}}
.stButton>button{{
    background:{C['blue']};color:#fff;border:none;border-radius:8px;
    padding:0.55rem 1.4rem;font-family:{C['font']};font-weight:600;
    font-size:14px;width:100%;letter-spacing:.02em;transition:background .2s;
}}
.stButton>button:hover{{background:#2563EB;}}
.stButton>button:disabled{{background:{C['border']};color:{C['sub']};}}
.stSelectbox>div>div,.stTextInput>div>div>input{{
    border-radius:8px;border:1px solid {C['border']};
    background:{C['surface']};color:{C['text']};
}}
hr{{border-color:{C['border']};margin:1.2rem 0;}}
.card{{
    background:{C['surface']};border:1px solid {C['border']};
    border-radius:12px;padding:18px 22px;margin-bottom:12px;
}}
.label{{
    font-size:11px;font-weight:600;text-transform:uppercase;
    letter-spacing:.08em;color:{C['sub']};margin-bottom:4px;
}}
.badge-green{{
    background:#064E3B;color:#34D399;border-radius:20px;
    font-size:11px;font-weight:700;padding:3px 12px;letter-spacing:.04em;
}}
.badge-blue{{
    background:#1E3A5F;color:#93C5FD;border-radius:6px;
    font-size:11px;font-weight:600;padding:2px 9px;
}}
.sec{{
    font-size:13px;font-weight:700;text-transform:uppercase;
    letter-spacing:.1em;color:{C['sub']};margin-bottom:16px;
    padding-bottom:8px;border-bottom:2px solid {C['border']};
}}
.summary-box{{
    background:{C['surface']};border-left:4px solid {C['blue']};
    border-radius:0 10px 10px 0;padding:20px 24px;
    line-height:1.75;font-size:14.5px;
}}
.step-log{{
    background:#0A0D14;border:1px solid {C['border']};border-radius:8px;
    padding:12px 16px;font-family:monospace;font-size:12px;
    color:#6EE7B7;line-height:1.6;
}}
.upload-zone{{
    background:{C['surface']};border:2px dashed {C['border']};
    border-radius:12px;padding:24px;text-align:center;
}}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================

SECTORS = [
    "Information Technology", "Banking & Financial Services",
    "Energy & Utilities", "Pharmaceuticals & Healthcare",
    "Automotive & Manufacturing", "Fast-Moving Consumer Goods (FMCG)",
    "Chemicals & Materials", "Telecommunications",
    "Infrastructure & Real Estate", "Metals & Mining", "Other",
]

INGESTION_REPORT_TYPES = ["BRSR", "ESG", "Integrated"]
UPLOAD_REPORT_TYPE_OPTIONS = ["BRSR", "ESG", "Integrated", "Annual", "CSR", "Other"]

KPI_META: dict[str, dict] = {
    "scope_1_emissions": {
        "label":     "Scope 1 GHG",
        "unit":      "tCO2e/Cr",
        "max_ratio": 10,
        "desc":      "Direct GHG emissions per INR Crore revenue",
    },
    "scope_2_emissions": {
        "label":     "Scope 2 GHG",
        "unit":      "tCO2e/Cr",
        "max_ratio": 10,
        "desc":      "Indirect GHG emissions per INR Crore revenue",
    },
    "total_ghg_emissions": {
        "label":     "Total GHG",
        "unit":      "tCO2e/Cr",
        "max_ratio": 20,
        "desc":      "Scope 1 + Scope 2 per INR Crore revenue",
    },
    "waste_generated": {
        "label":     "Waste Intensity",
        "unit":      "MT/Cr",
        "max_ratio": 5,
        "desc":      "Waste generated per INR Crore revenue",
    },
}

EXTRACTABLE_KPI_NAMES = [
    "scope_1_emissions",
    "scope_2_emissions",
    "waste_generated",
]

TARGET_KPI_NAMES = list(KPI_META.keys())

_KPI_PLAUSIBILITY: dict[str, tuple[float, float]] = {
    "scope_1_emissions":   (1,      5_000_000),
    "scope_2_emissions":   (1,      5_000_000),
    "total_ghg_emissions": (1,     10_000_000),
    "waste_generated":     (0.1,     500_000),
}

_DEFAULT_REVENUE_CR = 315_322.0

# Report-type priority for KPI selection: lower number = higher priority.
# When multiple KPI records exist (one per downloaded report), we prefer
# BRSR (most comprehensive, SEBI-mandated) > Integrated > ESG.
_REPORT_TYPE_PRIORITY: dict[str, int] = {
    "BRSR":       0,
    "Integrated": 1,
    "ESG":        2,
}


# =============================================================================
# DATA CONTAINERS
# =============================================================================

@dataclass
class ReportInfo:
    """
    Plain-data representation of one downloaded report row.
    No live ORM objects — safe to use outside a DB session.
    """
    id:          uuid.UUID
    report_type: str
    file_path:   Optional[str]
    status:      str


@dataclass
class CompanyData:
    """All data for one company after the pipeline completes."""
    company_name:   str
    fy:             int
    sector:         str
    kpi_records:    dict
    revenue_result: object
    log:            list[str]
    company_id:     Optional[uuid.UUID] = None
    # List of ALL reports processed — preserved for audit / UI display
    report_infos:   list[ReportInfo] = field(default_factory=list)
    # Primary file path (from highest-priority report, for PDF text extraction)
    file_path:      Optional[str]    = None


# =============================================================================
# DB LAYER
# =============================================================================

@st.cache_resource
def _check_db() -> bool:
    try:
        from core.database import check_connection
        return check_connection()
    except Exception:
        return False


@st.cache_resource(ttl=60)
def _get_company_names() -> list[str]:
    try:
        from core.database import get_db
        from models.db_models import Company
        with get_db() as db:
            rows = (
                db.query(Company.name)
                .filter(Company.is_active == True)
                .order_by(Company.name)
                .all()
            )
        return [r[0] for r in rows]
    except Exception:
        return []


def _db_get_all_reports(company_name: str, fy: int) -> dict:
    """
    Return ALL downloaded/parsed/extracted reports for a company+year.

    Unlike the old _db_check_report_exists() which picked ONE best report,
    this returns every usable report so the pipeline can process all of them.

    Returns
    -------
    {
        "exists":      bool,                 # True if any usable report found
        "company_id":  uuid.UUID | None,
        "reports":     list[ReportInfo],     # ALL usable reports, type-priority sorted
    }

    Sorting: BRSR first, then Integrated, then ESG, then by created_at desc.
    This ordering is used only to pick the "primary" file_path for PDF text
    extraction in _build_benchmark.  Extract/parse runs on ALL reports.
    """
    empty = {"exists": False, "company_id": None, "reports": []}
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

            # Convert to plain ReportInfo objects while session is open.
            # Sort by type priority then created_at descending.
            infos = sorted(
                [
                    ReportInfo(
                        id=r.id,
                        report_type=r.report_type,
                        file_path=r.file_path,
                        status=r.status,
                    )
                    for r in reports
                ],
                key=lambda ri: (
                    _REPORT_TYPE_PRIORITY.get(ri.report_type, 99),
                    # file_path present is better (for PDF extraction)
                    0 if ri.file_path and Path(ri.file_path).exists() else 1,
                ),
            )

            return {
                "exists":     True,
                "company_id": company_id,
                "reports":    infos,
            }

    except Exception as exc:
        st.warning(f"DB report lookup failed for {company_name}: {exc}")
        return empty


def _db_load_kpis_and_revenue(company_id: uuid.UUID, fy: int) -> dict:
    """
    Load the best KPI record (per KPI) and revenue for a company+FY.

    Multi-source selection rule:
      When multiple KPIRecords exist for the same KPI (one per report type),
      select by: BRSR first, Integrated second, ESG third.
      Within the same report type, take the most recently extracted record.

    This is implemented by joining KPIRecord → Report and applying a
    CASE-based sort on report_type before taking .first().

    Revenue: from the most recently updated report that has revenue_cr set,
    preferring BRSR if available.

    Returns
    -------
    {
        "kpis":      {kpi_name: {"value","unit","method","confidence","report_type"}},
        "revenue":   RevenueResult | None,
        "file_path": str | None,    # primary report file path
    }
    """
    empty = {"kpis": {}, "revenue": None, "file_path": None}
    try:
        from core.database import get_db
        from models.db_models import Report, KPIRecord, KPIDefinition
        from services.revenue_extractor import RevenueResult
        from sqlalchemy import case

        with get_db() as db:
            # Priority expression: BRSR=0, Integrated=1, ESG=2, others=99
            type_priority = case(
                (Report.report_type == "BRSR",       0),
                (Report.report_type == "Integrated",  1),
                (Report.report_type == "ESG",         2),
                else_=99,
            )

            # Revenue: prefer BRSR, then most recent
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

            # Primary file path: highest-priority report with existing file
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

            cached_rev = None
            if report_with_rev and getattr(report_with_rev, "revenue_cr", None) is not None:
                try:
                    cached_rev = RevenueResult(
                        value_cr=float(report_with_rev.revenue_cr),
                        raw_value=str(report_with_rev.revenue_cr),
                        raw_unit=getattr(report_with_rev, "revenue_unit", None) or "INR_Crore",
                        source=getattr(report_with_rev, "revenue_source", None) or "db",
                        page_number=0,
                        confidence=0.99,
                        pattern_name="cached",
                    )
                except Exception:
                    pass

            # KPIs: priority-ordered selection across all reports
            kpis: dict = {}
            for kpi_name in EXTRACTABLE_KPI_NAMES:
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
                lo, hi = _KPI_PLAUSIBILITY.get(kpi_name, (0, float("inf")))
                if not (lo <= val <= hi):
                    continue

                # Retrieve report_type for provenance display
                rec_report = db.query(Report).filter(Report.id == rec.report_id).first()
                rec_type   = rec_report.report_type if rec_report else "unknown"

                kpis[kpi_name] = {
                    "value":       val,
                    "unit":        unit,
                    "method":      rec.extraction_method,
                    "confidence":  rec.confidence or 0.9,
                    "report_type": rec_type,   # provenance tag
                }

        return {"kpis": kpis, "revenue": cached_rev, "file_path": file_path}

    except Exception as exc:
        st.warning(f"DB KPI load failed for company_id={company_id}: {exc}")
        return empty


def _db_ensure_schema() -> None:
    try:
        from core.database import get_db
        from services.revenue_extractor import ensure_revenue_columns
        with get_db() as db:
            ensure_revenue_columns(db)
    except Exception:
        pass


def _db_store_kpis(
    company_id:     uuid.UUID,
    report_id:      uuid.UUID,
    fy:             int,
    kpi_records:    dict,
    revenue_result,
) -> None:
    """
    Persist KPIs and revenue for ONE report.

    Skips exact duplicates (same company/KPI/year/value/report already present).
    Revenue is written only if the report row does not already have a value.
    Each KPIRecord is linked to its specific report_id — no merging.
    """
    try:
        from core.database import get_db
        from models.db_models import Report, KPIRecord, KPIDefinition
        from services.revenue_extractor import store_revenue

        with get_db() as db:
            if revenue_result:
                report_row = db.query(Report).filter(Report.id == report_id).first()
                if report_row and getattr(report_row, "revenue_cr", None) is None:
                    try:
                        store_revenue(report_row, revenue_result, db)
                    except Exception:
                        pass

            for kpi_name, rec in kpi_records.items():
                kdef = (
                    db.query(KPIDefinition)
                    .filter(KPIDefinition.name == kpi_name)
                    .first()
                )
                if not kdef:
                    continue

                # Dedup: same report + KPI + value already stored
                already = (
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
                if already:
                    continue

                db.add(KPIRecord(
                    company_id        = company_id,
                    report_id         = report_id,   # linked to specific report
                    kpi_definition_id = kdef.id,
                    report_year       = fy,
                    raw_value         = str(rec["value"]),
                    normalized_value  = rec["value"],
                    unit              = rec["unit"],
                    extraction_method = rec["method"],
                    confidence        = rec["confidence"],
                    is_validated      = rec["confidence"] >= 0.85,
                    validation_notes  = "esg_dashboard",
                ))

    except Exception as exc:
        st.warning(f"DB store failed for report {str(report_id)[:8]}: {exc}")


# =============================================================================
# PIPELINE STEPS
# =============================================================================

def _step_ingest(
    company_name: str,
    fy:           int,
    sector:       str,
    log:          list[str],
) -> dict:
    """
    Run full multi-type ingestion: BRSR + ESG + Integrated.

    Uses run_multi_report_types() which runs all query templates in one
    Tavily session, classifies URLs, and downloads one PDF per type.

    Returns
    -------
    {
        "company_id": uuid | None,
        "reports":    list[ReportInfo],  # all successfully downloaded reports
    }
    """
    from agents.ingestion_agent import IngestionAgent
    from models.schemas import CompanyCreate

    agent        = IngestionAgent()
    company_data = CompanyCreate(name=company_name, sector=sector, country="India")

    log.append(
        f"Searching BRSR, ESG, and Integrated reports for {company_name} FY{fy}."
    )

    try:
        result = agent.run_multi_report_types(
            company_data=company_data,
            year=fy,
            auto_download=True,
        )
    except Exception as exc:
        log.append(f"  Ingestion failed: {exc}")
        return {"company_id": None, "reports": []}

    company       = result["company"]
    downloads     = result["downloaded_reports"]
    not_found     = result.get("not_found_types", [])
    failed        = result.get("failed_types", [])

    for rtype in INGESTION_REPORT_TYPES:
        dl = next((d for d in downloads if d.report_type == rtype), None)
        if dl:
            fname   = Path(dl.file_path).name if dl.file_path else "unknown"
            size_mb = round((dl.file_size_bytes or 0) / (1024 * 1024), 1)
            log.append(f"  [{rtype}] downloaded: {fname} ({size_mb} MB)")
        elif rtype in not_found:
            log.append(
                f"  [{rtype}] not found — company may not publish a separate {rtype} report."
            )
        elif rtype in failed:
            log.append(f"  [{rtype}] download failed — all candidate URLs rejected.")
        else:
            log.append(f"  [{rtype}] no result.")

    if not downloads:
        log.append("No PDFs downloaded for any report type.")
        return {
            "company_id": company.id if company else None,
            "reports":    [],
        }

    # Convert DownloadedReport → ReportInfo
    reports = [
        ReportInfo(
            id=d.id,
            report_type=d.report_type,
            file_path=d.file_path,
            status=d.status,
        )
        for d in downloads
        if d.status == "downloaded"
    ]
    log.append(f"  {len(reports)} report(s) ready for processing.")
    return {"company_id": company.id, "reports": reports}


def _step_parse(report_id: uuid.UUID, report_type: str, log: list[str]) -> bool:
    """
    Run ParseOrchestrator for one report.  Idempotent via parse cache.

    Returns True on success, False on failure.
    Failure is logged but does NOT raise — caller continues with next report.
    """
    from services.parse_orchestrator import ParseOrchestrator

    log.append(f"  Parsing [{report_type}] (id={str(report_id)[:8]})...")
    try:
        result = ParseOrchestrator().run(report_id=report_id, force=False)
        log.append(
            f"    {result.page_count} pages, "
            f"{result.meta.get('chunk_count', '?')} chunks, "
            f"{result.meta.get('table_count', '?')} tables."
        )
        return True
    except Exception as exc:
        log.append(f"    Parse failed: {exc}")
        return False


def _step_extract(
    report_id:   uuid.UUID,
    report_type: str,
    fy:          int,
    log:         list[str],
    llm_service,
) -> dict:
    """
    Run KPI + revenue extraction for one report.

    Returns {"kpis": dict, "revenue": RevenueResult | None}.
    Failure is logged but does NOT raise — caller continues with next report.
    """
    from agents.extraction_agent import ExtractionAgent
    from services.revenue_extractor import extract_revenue
    from core.database import get_db

    new_kpis: dict = {}

    log.append(f"  Extracting [{report_type}] (id={str(report_id)[:8]})...")
    try:
        with get_db() as db:
            extracted_list = ExtractionAgent().extract_all(
                report_id=report_id,
                db=db,
                kpi_names=EXTRACTABLE_KPI_NAMES,
            )

        for ext in extracted_list:
            if ext.normalized_value is None:
                continue

            val  = ext.normalized_value
            unit = ext.unit or ""
            lo, hi = _KPI_PLAUSIBILITY.get(ext.kpi_name, (0, float("inf")))
            if not (lo <= val <= hi):
                log.append(
                    f"    {ext.kpi_name}: {val:,.2f} {unit} "
                    f"outside plausible range — dropped"
                )
                continue

            new_kpis[ext.kpi_name] = {
                "value":      val,
                "unit":       unit,
                "method":     ext.extraction_method,
                "confidence": ext.confidence or 0.5,
            }
            log.append(
                f"    {ext.kpi_name}: {val:,.2f} {unit} "
                f"[{ext.extraction_method} conf={ext.confidence:.2f}]"
            )

    except Exception as exc:
        log.append(f"    KPI extraction failed: {exc}")

    # Revenue extraction
    new_revenue = None
    try:
        from core.database import get_db
        from models.db_models import Report
        with get_db() as db:
            rpt = db.query(Report).filter(Report.id == report_id).first()
            pdf_path_str = rpt.file_path if rpt else None
    except Exception:
        pdf_path_str = None

    if pdf_path_str and Path(pdf_path_str).exists():
        try:
            new_revenue = extract_revenue(
                pdf_path         = Path(pdf_path_str),
                fiscal_year_hint = fy,
                llm_service      = llm_service,
            )
            if new_revenue:
                log.append(
                    f"    Revenue: INR {new_revenue.value_cr:,.0f} Cr "
                    f"[{new_revenue.pattern_name} conf={new_revenue.confidence:.2f}]"
                )
        except Exception as exc:
            log.append(f"    Revenue extraction failed: {exc}")

    return {"kpis": new_kpis, "revenue": new_revenue}


def _derive_total_ghg(kpi_records: dict) -> Optional[dict]:
    """
    Compute total_ghg = scope_1 + scope_2 when both are present.
    BRSR reports never state this as a standalone absolute value.
    """
    if "total_ghg_emissions" in kpi_records:
        return None
    s1 = kpi_records.get("scope_1_emissions")
    s2 = kpi_records.get("scope_2_emissions")
    if not s1 or not s2:
        return None
    try:
        from services.normalizer import normalize
        n1    = normalize("scope_1_emissions", float(s1["value"]), s1["unit"])
        n2    = normalize("scope_2_emissions", float(s2["value"]), s2["unit"])
        total = round(n1.normalized_value + n2.normalized_value, 2)
    except Exception:
        return None
    return {
        "value":      total,
        "unit":       "tCO2e",
        "method":     "derived",
        "confidence": round(min(s1["confidence"], s2["confidence"]) * 0.99, 3),
    }


# =============================================================================
# MAIN MULTI-REPORT PIPELINE
# =============================================================================

def run_company_pipeline(
    company_name:       str,
    fy:                 int,
    sector:             str,
    db_online:          bool,
    llm_service,
    status_placeholder,
) -> CompanyData:
    """
    Multi-source pipeline for one company+FY.

    Phase 1 — Discovery
        Check DB for existing reports.  If any exist, skip ingestion.
        If none exist, run run_multi_report_types() to download all three types.

    Phase 2 — Multi-report parse + extract loop
        For EACH downloaded report (BRSR, ESG, Integrated), independently:
          a. ParseOrchestrator.run() — idempotent
          b. ExtractionAgent.extract_all() — extracts KPIs from that report
          c. _db_store_kpis() — persists records linked to that report_id
        If one report fails, the others continue (error isolation).

    Phase 3 — Final DB read
        Load KPI state from DB for all reports.
        Priority-order selection (BRSR > Integrated > ESG) applied at read time.
        Build CompanyData from the priority-selected values.

    Returns CompanyData populated from the final DB read.
    """
    log: list[str] = []

    def _update(msg: str) -> None:
        log.append(msg)
        status_placeholder.markdown(
            "<div class='step-log'>" + "<br>".join(log[-16:]) + "</div>",
            unsafe_allow_html=True,
        )

    _update(f"Starting pipeline for {company_name} FY{fy}.")

    company_id:   Optional[uuid.UUID] = None
    report_infos: list[ReportInfo]    = []

    # ── Phase 1: Discovery ────────────────────────────────────────────────────
    if not db_online:
        _update("Database is offline. Cannot proceed.")
        return CompanyData(
            company_name=company_name, fy=fy, sector=sector,
            kpi_records={}, revenue_result=None, log=log,
        )

    _update("Checking database for existing reports...")
    db_data    = _db_get_all_reports(company_name, fy)
    company_id = db_data.get("company_id")

    if db_data["exists"]:
        report_infos = db_data["reports"]
        types_found  = [ri.report_type for ri in report_infos]
        _update(
            f"  Found {len(report_infos)} existing report(s): {types_found}. "
            f"Skipping search."
        )
    else:
        _update(f"  No reports in DB for {company_name} FY{fy}. Running ingestion...")
        ingest = _step_ingest(company_name, fy, sector, log)
        for line in log[-6:]:
            pass  # lines already in log from _step_ingest
        company_id   = ingest.get("company_id") or company_id
        report_infos = ingest.get("reports", [])

        if not report_infos:
            _update("Ingestion produced no downloadable reports. Cannot proceed.")
            return CompanyData(
                company_name=company_name, fy=fy, sector=sector,
                kpi_records={}, revenue_result=None, log=log,
                company_id=company_id,
            )

        # Re-read from DB to get consistent ReportInfo with all DB-assigned fields
        db_data    = _db_get_all_reports(company_name, fy)
        company_id = db_data.get("company_id") or company_id
        if db_data["exists"]:
            report_infos = db_data["reports"]

    _update(f"Processing {len(report_infos)} report(s)...")

    # ── Phase 2: Parse + extract each report independently ────────────────────
    for ri in report_infos:
        _update(f"--- [{ri.report_type}] report_id={str(ri.id)[:8]} ---")

        # 2a. Parse (idempotent)
        parse_ok = _step_parse(ri.id, ri.report_type, log)
        if not parse_ok:
            _update(f"  Parse failed for [{ri.report_type}] — skipping extraction.")
            continue   # error isolation: move to next report

        # 2b. Extract
        extract_result = _step_extract(ri.id, ri.report_type, fy, log, llm_service)
        new_kpis    = extract_result["kpis"]
        new_revenue = extract_result["revenue"]

        # 2c. Derive total_ghg from this report's scope components
        ghg = _derive_total_ghg(new_kpis)
        if ghg:
            new_kpis["total_ghg_emissions"] = ghg
            s1v = new_kpis.get("scope_1_emissions", {}).get("value", 0)
            s2v = new_kpis.get("scope_2_emissions", {}).get("value", 0)
            _update(
                f"  Derived total_ghg [{ri.report_type}]: "
                f"{ghg['value']:,.2f} tCO2e ({s1v:,.0f} + {s2v:,.0f})"
            )

        # 2d. Store — linked to this specific report_id
        if company_id and (new_kpis or new_revenue):
            _db_store_kpis(
                company_id     = company_id,
                report_id      = ri.id,
                fy             = fy,
                kpi_records    = new_kpis,
                revenue_result = new_revenue,
            )
            _update(
                f"  Stored {len(new_kpis)} KPI(s) for [{ri.report_type}]."
            )
        else:
            _update(f"  Nothing extracted from [{ri.report_type}].")

    # ── Phase 3: Final priority-ordered read from DB ──────────────────────────
    _update("Loading final KPI state from DB (BRSR > Integrated > ESG priority)...")

    if not company_id:
        _update("No company_id — cannot read from DB.")
        return CompanyData(
            company_name=company_name, fy=fy, sector=sector,
            kpi_records={}, revenue_result=None, log=log,
            report_infos=report_infos,
        )

    final_db = _db_load_kpis_and_revenue(company_id, fy)
    final_kpis    = final_db["kpis"]
    final_revenue = final_db["revenue"]
    final_fp      = final_db["file_path"]

    # Derive total_ghg from final priority-selected scope components
    ghg_final = _derive_total_ghg(final_kpis)
    if ghg_final:
        final_kpis["total_ghg_emissions"] = ghg_final

    if final_kpis:
        kpi_summary = {k: f"{v['value']:,.0f} [{v.get('report_type','?')}]"
                       for k, v in final_kpis.items()}
        _update(f"  Final KPIs ({len(final_kpis)}): {kpi_summary}")
    else:
        _update("  No KPIs available after processing all reports.")

    if final_revenue:
        _update(f"  Revenue: INR {final_revenue.value_cr:,.0f} Crore")
    else:
        _update(f"  Revenue: not found. Default INR {_DEFAULT_REVENUE_CR:,.0f} Cr used.")

    _update(f"Pipeline complete for {company_name} FY{fy}.")

    return CompanyData(
        company_name   = company_name,
        fy             = fy,
        sector         = sector,
        kpi_records    = final_kpis,
        revenue_result = final_revenue,
        log            = log,
        company_id     = company_id,
        report_infos   = report_infos,
        file_path      = final_fp,
    )


# =============================================================================
# UPLOAD PIPELINE  (unchanged logic, updated to use new DB helpers)
# =============================================================================

def run_upload_pipeline(
    uploaded_file,
    company_name: str,
    fy:           int,
    sector:       str,
    report_type:  str,
    db_online:    bool,
    llm_service,
    status_placeholder,
) -> dict:
    """Full pipeline for a user-uploaded PDF (single file)."""
    from agents.ingestion_agent import IngestionAgent

    log: list[str] = []

    def _update(msg: str) -> None:
        log.append(msg)
        status_placeholder.markdown(
            "<div class='step-log'>" + "<br>".join(log[-14:]) + "</div>",
            unsafe_allow_html=True,
        )

    _update(
        f"Processing upload: {uploaded_file.name} "
        f"({round(uploaded_file.size / (1024 * 1024), 1)} MB)"
    )

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, prefix="esg_upload_") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = Path(tmp.name)
        _update(f"  Saved to temp file: {tmp_path.name}")
    except Exception as exc:
        _update(f"  Failed to save upload to disk: {exc}")
        return {"success": False, "log": log}

    _update("Registering in database...")
    try:
        result = IngestionAgent().ingest_uploaded_pdf(
            source_path  = tmp_path,
            company_name = company_name,
            year         = fy,
            sector       = sector,
            report_type  = report_type,
        )
    except Exception as exc:
        _update(f"  Ingestion failed: {exc}")
        tmp_path.unlink(missing_ok=True)
        return {"success": False, "log": log}
    finally:
        tmp_path.unlink(missing_ok=True)

    company    = result["company"]
    report     = result["report"]
    is_dup     = result.get("is_duplicate", False)
    company_id = company.id
    report_id  = report.id

    if is_dup:
        _update(f"  Duplicate SHA-256: reusing report {str(report_id)[:8]}.")
    else:
        fname = Path(report.file_path).name if report.file_path else "unknown"
        _update(f"  Stored as: {fname}")

    ri = ReportInfo(id=report_id, report_type=report_type,
                    file_path=report.file_path, status=report.status)

    parsed_ok = _step_parse(report_id, report_type, log)
    if not parsed_ok:
        _update("  Parsing failed — extraction may be incomplete.")

    _update(f"Extracting KPIs from [{report_type}]...")
    extract_result = _step_extract(report_id, report_type, fy, log, llm_service)
    new_kpis    = extract_result["kpis"]
    new_revenue = extract_result["revenue"]

    ghg = _derive_total_ghg(new_kpis)
    if ghg:
        new_kpis["total_ghg_emissions"] = ghg
        _update(f"  Derived total_ghg: {ghg['value']:,.2f} tCO2e")

    if db_online and (new_kpis or new_revenue):
        _db_store_kpis(company_id, report_id, fy, new_kpis, new_revenue)
        _update(f"  Stored {len(new_kpis)} KPI record(s) to DB.")

    _update("Upload pipeline complete.")
    return {
        "success":      True,
        "kpi_records":  new_kpis,
        "revenue":      new_revenue,
        "company_id":   company_id,
        "report_id":    report_id,
        "report_infos": [ri],
        "log":          log,
    }


# =============================================================================
# BENCHMARK BUILDER  (unchanged logic)
# =============================================================================

def _build_benchmark(data1: CompanyData, data2: CompanyData, sector: str) -> dict:
    from services.benchmark import build_company_profile, compare_profiles
    from services.summary_generator import generate_summary
    from services.llm_service import LLMService
    from core.config import get_settings

    profiles = []
    for data in [data1, data2]:
        rev    = data.revenue_result
        rev_cr = rev.value_cr if rev else _DEFAULT_REVENUE_CR
        rev_src = rev.source  if rev else "default"

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
            kpi_records    = data.kpi_records,
            revenue_cr     = rev_cr,
            revenue_source = rev_src,
            company_name   = data.company_name,
            fiscal_year    = data.fy,
            page_texts     = page_texts,
        )
        profiles.append(profile)

    report   = compare_profiles(profiles)
    filtered = _filter_comparable(report.comparisons)

    llm = None
    try:
        if get_settings().llm_api_key:
            llm = LLMService()
    except Exception:
        pass

    summary = generate_summary(profiles, report, llm=llm)
    return {"profiles": profiles, "report": report, "filtered": filtered, "summary": summary}


def _filter_comparable(comparisons) -> list:
    out = []
    for comp in comparisons:
        ceiling = KPI_META.get(comp.kpi_name, {}).get("max_ratio")
        if all((not ceiling or v <= ceiling) and v > 0 for _, v, _ in comp.entries):
            out.append(comp)
    return out


# =============================================================================
# CHART HELPERS  (unchanged)
# =============================================================================

_CHART_FONT = dict(family="Inter, Arial, sans-serif", size=12, color=C["text"])


def _hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _radar_chart(filtered, la: str, lb: str):
    cats, sa, sb = [], [], []
    for comp in filtered:
        vals = {lbl: v for lbl, v, _ in comp.entries}
        va, vb = vals.get(la), vals.get(lb)
        if va is None or vb is None:
            continue
        cats.append(KPI_META.get(comp.kpi_name, {}).get("label", comp.display_name))
        total = va + vb
        sa.append(round(100 * (1 - va / total), 1) if total else 50)
        sb.append(round(100 * (1 - vb / total), 1) if total else 50)
    if len(cats) < 2:
        return None
    cats_c = cats + [cats[0]]; sa_c = sa + [sa[0]]; sb_c = sb + [sb[0]]
    fig = go.Figure()
    for name, scores, color in [
        (la.split(" FY")[0], sa_c, C["ca"]),
        (lb.split(" FY")[0], sb_c, C["cb"]),
    ]:
        fig.add_trace(go.Scatterpolar(
            r=scores, theta=cats_c, fill="toself", name=name,
            line=dict(color=color, width=2.5), fillcolor=_hex_rgba(color, 0.12),
            hovertemplate="<b>%{theta}</b><br>Score: %{r:.0f}<extra></extra>",
        ))
    fig.update_layout(
        polar=dict(bgcolor=C["surface"],
            radialaxis=dict(visible=True, range=[0, 100],
                            tickfont=dict(size=9, color=C["sub"]),
                            gridcolor=C["grid"], linecolor=C["border"]),
            angularaxis=dict(tickfont=dict(size=11, color=C["text"]),
                             gridcolor=C["grid"], linecolor=C["border"]),
        ),
        paper_bgcolor=C["bg"], height=360,
        margin=dict(l=40, r=40, t=40, b=20), font=_CHART_FONT,
        legend=dict(orientation="h", y=-0.08, font=dict(size=12)),
    )
    return fig


def _donut_chart(filtered, la: str, lb: str):
    na, nb = la.split(" FY")[0], lb.split(" FY")[0]
    wa = sum(1 for c in filtered if c.winner == la)
    wb = sum(1 for c in filtered if c.winner == lb)
    fig = go.Figure(go.Pie(
        labels=[na, nb], values=[wa, wb], hole=0.62,
        marker=dict(colors=[C["ca"], C["cb"]], line=dict(color=C["bg"], width=3)),
        textinfo="label+percent", textfont=dict(size=12),
        hovertemplate="<b>%{label}</b><br>%{value} KPI wins<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=C["bg"], height=260, margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False, font=_CHART_FONT,
        annotations=[dict(
            text=f"{wa+wb}<br><span style='font-size:10px'>KPIs</span>",
            x=0.5, y=0.5, font=dict(size=20, color=C["text"]), showarrow=False,
        )],
    )
    return fig


def _gap_bar_chart(filtered, la: str, lb: str):
    labels_list, va_list, vb_list = [], [], []
    for comp in filtered:
        vals = {lbl: v for lbl, v, _ in comp.entries}
        va, vb = vals.get(la), vals.get(lb)
        if va is None or vb is None:
            continue
        labels_list.append(KPI_META.get(comp.kpi_name, {}).get("label", comp.display_name))
        va_list.append(va); vb_list.append(vb)
    if not labels_list:
        return None
    fig = go.Figure()
    for name, vals, color, pat in [
        (la.split(" FY")[0], va_list, C["ca"], ""),
        (lb.split(" FY")[0], vb_list, C["cb"], "/"),
    ]:
        fig.add_trace(go.Bar(
            name=name, x=vals, y=labels_list, orientation="h",
            marker=dict(color=color, line=dict(color=color, width=1), pattern_shape=pat),
            text=[f"{v:.3g}" for v in vals], textposition="outside",
            textfont=dict(size=10),
            hovertemplate="<b>%{y}</b><br>%{x:.4e}<extra></extra>",
        ))
    fig.update_layout(
        barmode="group", height=300,
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"], font=_CHART_FONT,
        xaxis=dict(title="Intensity per INR Crore", showgrid=True,
                   gridcolor=C["grid"], zeroline=False, color=C["text"]),
        yaxis=dict(autorange="reversed", color=C["text"]),
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=160, r=60, t=20, b=40),
    )
    return fig


def _mini_bar_chart(comp, la: str, lb: str):
    vals   = {lbl: v for lbl, v, _ in comp.entries}
    va, vb = vals.get(la, 0), vals.get(lb, 0)
    meta   = KPI_META.get(comp.kpi_name, {})
    na, nb = la.split(" FY")[0], lb.split(" FY")[0]
    fig = go.Figure()
    for name, val, color, pat in [(na, va, C["ca"], ""), (nb, vb, C["cb"], "/")]:
        fig.add_trace(go.Bar(
            name=name, x=[val], y=[meta.get("label", "")], orientation="h",
            marker=dict(color=color, line=dict(color=color, width=1.5), pattern_shape=pat),
            text=[f"{val:.3g}"], textposition="outside",
            textfont=dict(size=11, color=C["text"]),
            hovertemplate=f"<b>{name}</b><br>{val:.4e} {meta.get('unit','')}<extra></extra>",
        ))
    fig.update_layout(
        barmode="group", height=110, showlegend=False,
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"], font=_CHART_FONT,
        yaxis=dict(visible=False),
        xaxis=dict(showgrid=True, gridcolor=C["grid"], zeroline=False, showticklabels=False),
        margin=dict(l=10, r=60, t=8, b=8),
    )
    return fig


# =============================================================================
# PDF REPORT EXPORT  (unchanged)
# =============================================================================

def _export_pdf_report(profiles, filtered, summary: str, sector: str) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm,
    )

    BLUE  = colors.HexColor("#3B82F6")
    GREEN = colors.HexColor("#10B981")
    GRAY  = colors.HexColor("#64748B")
    LIGHT = colors.HexColor("#F8F9FB")
    BDR   = colors.HexColor("#E2E8F0")
    BLK   = colors.HexColor("#1A202C")

    ss    = getSampleStyleSheet()
    s_title = ParagraphStyle("t",  parent=ss["Title"],   fontSize=22, textColor=BLK,
                              spaceAfter=4, leading=28, fontName="Helvetica-Bold")
    s_sub   = ParagraphStyle("s",  parent=ss["Normal"],  fontSize=11, textColor=GRAY,
                              spaceAfter=14, leading=16)
    s_h2    = ParagraphStyle("h2", parent=ss["Heading2"],fontSize=13, textColor=BLK,
                              spaceBefore=16, spaceAfter=8, fontName="Helvetica-Bold")
    s_body  = ParagraphStyle("b",  parent=ss["Normal"],  fontSize=10, textColor=BLK,
                              leading=16, spaceAfter=8)
    s_note  = ParagraphStyle("n",  parent=ss["Normal"],  fontSize=9, textColor=GRAY,
                              leading=14)

    labels = [f"{p.company_name} FY{p.fiscal_year}" for p in profiles]
    story  = [
        Paragraph("ESG Competitive Intelligence Report", s_title),
        Paragraph(f"{labels[0]} vs {labels[1]}", s_sub),
        Paragraph(f"Sector: {sector}", s_note),
        HRFlowable(width="100%", thickness=1, color=BDR, spaceAfter=14),
        Paragraph("Methodology", s_h2),
        Paragraph(
            "All metrics are intensity ratios (KPI divided by annual revenue in INR Crore). "
            "KPIs are extracted from ALL downloaded reports (BRSR, ESG, Integrated). "
            "When multiple records exist for the same KPI, the BRSR value is preferred, "
            "then Integrated, then ESG. KPIs with implausible intensity ratios are excluded.",
            s_body,
        ),
        Spacer(1, 8),
        Paragraph("KPI Intensity Comparison", s_h2),
    ]

    tdata = [["Metric", "Unit", labels[0], labels[1], "Gap", "Leader"]]
    for comp in filtered:
        meta = KPI_META.get(comp.kpi_name, {})
        vals = {lbl: v for lbl, v, _ in comp.entries}
        v0, v1 = vals.get(labels[0]), vals.get(labels[1])
        fmt = lambda v: f"{v:.2e}" if (v and v < 0.001) else (f"{v:.4f}" if v else "N/A")
        tdata.append([
            meta.get("label", comp.display_name), meta.get("unit", comp.unit),
            fmt(v0), fmt(v1), f"{comp.pct_gap:.1f}%",
            comp.winner.split(" FY")[0],
        ])

    tbl = Table(tdata, colWidths=[5.2*cm, 2.3*cm, 2.8*cm, 2.8*cm, 1.6*cm, 2.5*cm], repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1,  0), BLUE),
        ("TEXTCOLOR",     (0, 0), (-1,  0), colors.white),
        ("FONTNAME",      (0, 0), (-1,  0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1,  0), 9),
        ("BOTTOMPADDING", (0, 0), (-1,  0), 8),
        ("TOPPADDING",    (0, 0), (-1,  0), 8),
        ("FONTSIZE",      (0, 1), (-1, -1), 9),
        ("TOPPADDING",    (0, 1), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID",          (0, 0), (-1, -1), 0.5, BDR),
        ("ALIGN",         (2, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TEXTCOLOR",     (5, 1), ( 5, -1), GREEN),
        ("FONTNAME",      (5, 1), ( 5, -1), "Helvetica-Bold"),
    ]))

    story += [tbl, Spacer(1, 16), Paragraph("AI-Generated Narrative Summary", s_h2)]
    for para in summary.split("\n\n"):
        if para.strip():
            story.append(Paragraph(para.strip(), s_body))
    story += [
        Spacer(1, 12),
        HRFlowable(width="100%", thickness=0.5, color=BDR),
        Spacer(1, 6),
        Paragraph(
            "Generated by ESG Competitive Intelligence Pipeline. "
            "Data sourced from public BRSR, ESG, and Integrated reports. "
            "Multi-source extraction: all downloaded report types processed independently.",
            s_note,
        ),
    ]
    doc.build(story)
    return buf.getvalue()


# =============================================================================
# LLM SERVICE
# =============================================================================

@st.cache_resource
def _get_llm_service():
    try:
        from services.llm_service import LLMService
        from core.config import get_settings
        if get_settings().llm_api_key:
            return LLMService()
    except Exception:
        pass
    return None


# =============================================================================
# APP BOOTSTRAP
# =============================================================================

db_online       = _check_db()
known_companies = _get_company_names() if db_online else []
llm_service     = _get_llm_service()
if db_online:
    _db_ensure_schema()


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div style="padding:0 0 14px">
        <div style="font-size:22px;font-weight:800;color:#E8EAF0;letter-spacing:-0.5px">
            ESG Intel
        </div>
        <div style="font-size:12px;color:#8B92A9;margin-top:2px">
            Multi-Source Benchmarking Pipeline
        </div>
    </div>""", unsafe_allow_html=True)

    db_color = C["green"] if db_online else C["red"]
    db_label = (
        f"Database connected ({len(known_companies)} companies)"
        if db_online else "Database offline"
    )
    st.markdown(
        f"<div style='font-size:11px;color:{db_color};font-weight:600'>"
        f"{db_label}</div>",
        unsafe_allow_html=True,
    )
    if db_online:
        llm_color = C["green"] if llm_service else C["amber"]
        llm_label = "LLM enabled (Gemini)" if llm_service else "LLM disabled (no API key)"
        st.markdown(
            f"<div style='font-size:11px;color:{llm_color};font-weight:600'>"
            f"{llm_label}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    sector = st.selectbox("Sector", SECTORS, key="sector")
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # Company 1
    st.markdown(
        f'<div class="label" style="color:{C["ca"]}">Company 1</div>',
        unsafe_allow_html=True,
    )
    company1 = st.text_input(
        "c1name", placeholder="e.g. Infosys",
        label_visibility="collapsed", key="c1_name",
    )
    fy1 = st.number_input(
        "FY1", min_value=2015, max_value=2030, value=2025,
        label_visibility="collapsed", key="c1_fy",
    )
    if db_online and company1:
        check1 = _db_get_all_reports(company1, int(fy1))
        if check1["exists"]:
            types1 = [ri.report_type for ri in check1["reports"]]
            st.caption(f"{len(check1['reports'])} report(s) in DB: {', '.join(types1)}")
        else:
            st.caption("No reports in DB — full pipeline will run")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Company 2
    st.markdown(
        f'<div class="label" style="color:{C["cb"]}">Company 2</div>',
        unsafe_allow_html=True,
    )
    company2 = st.text_input(
        "c2name", placeholder="e.g. TCS",
        label_visibility="collapsed", key="c2_name",
    )
    fy2 = st.number_input(
        "FY2", min_value=2015, max_value=2030, value=2024,
        label_visibility="collapsed", key="c2_fy",
    )
    if db_online and company2:
        check2 = _db_get_all_reports(company2, int(fy2))
        if check2["exists"]:
            types2 = [ri.report_type for ri in check2["reports"]]
            st.caption(f"{len(check2['reports'])} report(s) in DB: {', '.join(types2)}")
        else:
            st.caption("No reports in DB — full pipeline will run")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    ready       = bool(company1 and company2 and company1.strip().lower() != company2.strip().lower())
    compare_btn = st.button("Compare", disabled=not ready, use_container_width=True)

    if not ready and (company1 or company2):
        if company1.strip().lower() == company2.strip().lower():
            st.caption("Enter two different company names.")

    st.markdown("---")


# =============================================================================
# MAIN CONTENT TABS
# =============================================================================

tab_compare, tab_upload = st.tabs(["Comparison", "Upload PDF"])


# ---------------------------------------------------------------------------
# TAB: COMPARISON
# ---------------------------------------------------------------------------

with tab_compare:

    if "result" not in st.session_state and not compare_btn:
        st.markdown(f"""
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;padding:80px 40px;text-align:center">
            <div style="font-size:28px;font-weight:800;color:#E8EAF0;
                        letter-spacing:-0.5px;margin-bottom:8px">
                ESG Competitive Intelligence
            </div>
            <div style="font-size:14px;color:#8B92A9;max-width:560px;
                        line-height:1.7;margin-bottom:24px">
                Enter two company names and click Compare.<br><br>
                The pipeline downloads and processes ALL available report types
                (BRSR, ESG, Integrated) independently. KPI values from all sources
                are stored separately in the database.<br><br>
                For comparison, the BRSR value is preferred, then Integrated, then ESG.
            </div>
            <div style="font-size:13px;color:#6EE7B7;background:#064E3B;
                        border-radius:8px;padding:10px 20px;display:inline-block">
                Multi-source | All report types | BRSR > Integrated > ESG priority
            </div>
        </div>""", unsafe_allow_html=True)

    if compare_btn and ready:
        st.markdown(f"### {company1} FY{fy1} vs {company2} FY{fy2}")
        st.caption(
            "Multi-source pipeline: parse + extract all report types independently. "
            "KPI selection for comparison: BRSR > Integrated > ESG."
        )

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown(f"**{company1} FY{fy1}**")
            placeholder1 = st.empty()
        with col_s2:
            st.markdown(f"**{company2} FY{fy2}**")
            placeholder2 = st.empty()

        pipeline_error = None
        data1 = data2 = None

        try:
            data1 = run_company_pipeline(
                company_name=company1, fy=int(fy1), sector=sector,
                db_online=db_online, llm_service=llm_service,
                status_placeholder=placeholder1,
            )
            data2 = run_company_pipeline(
                company_name=company2, fy=int(fy2), sector=sector,
                db_online=db_online, llm_service=llm_service,
                status_placeholder=placeholder2,
            )
        except Exception as exc:
            pipeline_error = exc
            st.error(f"Pipeline error: {exc}")
            with st.expander("Full traceback"):
                st.code(traceback.format_exc())

        if pipeline_error is None and data1 is not None and data2 is not None:
            if not data1.kpi_records and not data2.kpi_records:
                st.error("No KPIs extracted for either company. Check API keys and company names.")
                st.stop()

            result = _build_benchmark(data1, data2, sector)
            result["log1"]    = data1.log
            result["log2"]    = data2.log
            result["sector"]  = sector
            result["reports1"] = data1.report_infos
            result["reports2"] = data2.report_infos

            st.session_state.update({
                "result":   result,
                "company1": company1, "company2": company2,
                "fy1": fy1, "fy2": fy2,
            })

    if "result" in st.session_state:
        result   = st.session_state["result"]
        c1n      = st.session_state.get("company1", "Company 1")
        c2n      = st.session_state.get("company2", "Company 2")
        fy1v     = st.session_state.get("fy1", "")
        fy2v     = st.session_state.get("fy2", "")
        profiles = result["profiles"]
        report   = result["report"]
        filtered = result["filtered"]
        summary  = result["summary"]
        _sector  = result.get("sector", "")

        label_a = f"{profiles[0].company_name} FY{profiles[0].fiscal_year}"
        label_b = f"{profiles[1].company_name} FY{profiles[1].fiscal_year}"
        skipped = [c.kpi_name for c in report.comparisons if c not in filtered]

        # Header
        hc1, hc2 = st.columns([3, 1])
        with hc1:
            st.markdown(f"""
            <div style="margin-bottom:4px">
                <span style="font-size:24px;font-weight:800;color:{C['text']}">{c1n}</span>
                <span style="font-size:16px;color:{C['sub']};margin:0 10px">vs</span>
                <span style="font-size:24px;font-weight:800;color:{C['text']}">{c2n}</span>
            </div>
            <div style="font-size:13px;color:{C['sub']}">
                <span class="badge-blue">{_sector}</span>
                &nbsp;&nbsp;FY{fy1v} &middot; FY{fy2v}
                &nbsp;&middot;&nbsp;Intensity ratios (KPI / INR Crore)
                &nbsp;&middot;&nbsp;BRSR &gt; Integrated &gt; ESG priority
            </div>""", unsafe_allow_html=True)
        with hc2:
            pdf_bytes = _export_pdf_report(profiles, filtered, summary, _sector)
            st.download_button(
                "Download PDF Report", data=pdf_bytes,
                file_name=f"ESG_{c1n}_vs_{c2n}.pdf",
                mime="application/pdf", use_container_width=True,
            )

        # Source reports summary
        reports1 = result.get("reports1", [])
        reports2 = result.get("reports2", [])
        if reports1 or reports2:
            r1_str = ", ".join(ri.report_type for ri in reports1) if reports1 else "none"
            r2_str = ", ".join(ri.report_type for ri in reports2) if reports2 else "none"
            st.markdown(
                f"<div style='font-size:12px;color:{C['sub']};margin-bottom:8px'>"
                f"Sources processed — {c1n}: [{r1_str}]  |  {c2n}: [{r2_str}]"
                f"</div>",
                unsafe_allow_html=True,
            )

        if skipped:
            st.info(
                f"Excluded (intensity ratio exceeded ceiling): "
                f"{', '.join(KPI_META.get(k, {}).get('label', k) for k in skipped)}"
            )

        if not filtered:
            st.warning("No KPIs passed the sanity filter.")
            with st.expander("Extraction log"):
                for cname, lk in [(c1n, "log1"), (c2n, "log2")]:
                    lines = result.get(lk, [])
                    if lines:
                        st.markdown(f"**{cname}**")
                        for line in lines:
                            st.code(line, language=None)
            st.stop()

        st.markdown("---")

        # Scorecard
        st.markdown('<div class="sec">Overview</div>', unsafe_allow_html=True)
        wins_a = sum(1 for c in filtered if c.winner == label_a)
        wins_b = sum(1 for c in filtered if c.winner == label_b)
        leader = c1n if wins_a >= wins_b else c2n

        score_cols = st.columns(4)
        for col, (bc, bl, bv, bs) in zip(score_cols, [
            (C["blue"],  "KPIs Compared",   str(len(filtered)), f"of {len(report.comparisons)} total"),
            (C["ca"],    f"{c1n} FY{fy1v}",  str(wins_a),        "KPI wins"),
            (C["cb"],    f"{c2n} FY{fy2v}",  str(wins_b),        "KPI wins"),
            (C["green"], "Leader",            leader,             "More KPI wins"),
        ]):
            col.markdown(f"""
            <div class="card" style="border-top:3px solid {bc}">
                <div class="label">{bl}</div>
                <div style="font-size:28px;font-weight:800;color:{C['text']};
                            white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
                    {bv}
                </div>
                <div style="font-size:12px;color:{C['sub']};margin-top:2px">{bs}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Radar + donut
        st.markdown('<div class="sec">Performance Overview</div>', unsafe_allow_html=True)
        vc1, vc2 = st.columns([2, 1])
        with vc1:
            st.markdown("**Normalised Score Radar** — Higher score = lower environmental intensity")
            fig = _radar_chart(filtered, label_a, label_b)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with vc2:
            st.markdown("**Win Distribution**")
            st.plotly_chart(
                _donut_chart(filtered, label_a, label_b),
                use_container_width=True, config={"displayModeBar": False},
            )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Gap bars
        st.markdown('<div class="sec">Intensity Comparison</div>', unsafe_allow_html=True)
        st.caption("Lower bar = better environmental performance relative to revenue.")
        gap = _gap_bar_chart(filtered, label_a, label_b)
        if gap:
            st.plotly_chart(gap, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Per-KPI detail
        st.markdown('<div class="sec">KPI Detail</div>', unsafe_allow_html=True)
        for comp in filtered:
            meta   = KPI_META.get(comp.kpi_name, {})
            vals   = {lbl: v for lbl, v, _ in comp.entries}
            va, vb = vals.get(label_a), vals.get(label_b)
            wname  = comp.winner.split(" FY")[0]
            a_wins = comp.winner == label_a

            st.markdown(f"""
            <div class="card">
                <div style="display:flex;justify-content:space-between;
                            align-items:flex-start;margin-bottom:10px">
                    <div>
                        <span style="font-size:16px;font-weight:700">
                            {meta.get('label', comp.display_name)}
                        </span>
                        <div style="font-size:11px;color:{C['sub']};margin-top:2px">
                            {meta.get('desc', '')} &middot; {meta.get('unit', comp.unit)}
                        </div>
                    </div>
                    <div style="text-align:right">
                        <span class="badge-green">Leader: {wname}</span>
                        <div style="font-size:12px;color:{C['sub']};margin-top:4px">
                            {comp.pct_gap:.1f}% gap
                        </div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            st.plotly_chart(
                _mini_bar_chart(comp, label_a, label_b),
                use_container_width=True, config={"displayModeBar": False},
            )

            mc1, mc2 = st.columns(2)
            for col, val, color, is_win, dname in [
                (mc1, va, C["ca"], a_wins,     c1n),
                (mc2, vb, C["cb"], not a_wins, c2n),
            ]:
                vs = (
                    f"{val:.4e}" if (val is not None and val < 0.001)
                    else (f"{val:.4f}" if val is not None else "N/A")
                )
                col.markdown(f"""
                <div style="background:{C['surface']};border-radius:8px;
                            padding:10px 14px;border:1px solid {C['border']};
                            border-left:4px solid {color}">
                    <div style="font-size:11px;font-weight:600;color:{color}">
                        {dname}{' (leader)' if is_win else ''}
                    </div>
                    <div style="font-size:13px;color:{C['text']};margin-top:4px;
                                font-family:monospace">
                        {vs} {meta.get('unit', '')}
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        # Summary
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec">Summary</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="summary-box">{summary.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )

        # Methodology
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        with st.expander("Methodology and Data Provenance"):
            st.markdown("""
**Multi-source pipeline**: ALL downloaded report types are processed independently.
BRSR, ESG, and Integrated reports each go through parse → extract → store as separate
DB records linked to their own report_id.  No merging or conflict resolution at write time.

**KPI priority for comparison**: When a KPI exists in multiple report types, the value
from BRSR is used first, then Integrated, then ESG.  This preference order reflects
SEBI-mandated completeness of BRSR data for Indian listed companies.

**Idempotent processing**: Re-running the pipeline for the same company+year skips
ingestion (PDFs already downloaded) and parse (cached per report_id + parser_version).
Only new extractions are written to DB.

**Intensity ratios**: Every KPI is divided by annual revenue (INR Crore) before
comparison to normalise for company size differences.

**Sanity filter**: KPIs with intensity ratios above a per-type ceiling are excluded.
            """)

        # Pipeline logs
        log1 = result.get("log1", [])
        log2 = result.get("log2", [])
        if log1 or log2:
            with st.expander("Pipeline log"):
                lc1, lc2 = st.columns(2)
                with lc1:
                    if log1:
                        st.markdown(f"**{c1n} FY{fy1v}**")
                        for line in log1:
                            st.code(line, language=None)
                with lc2:
                    if log2:
                        st.markdown(f"**{c2n} FY{fy2v}**")
                        for line in log2:
                            st.code(line, language=None)


# ---------------------------------------------------------------------------
# TAB: UPLOAD PDF
# ---------------------------------------------------------------------------

with tab_upload:
    st.markdown('<div class="sec">Upload ESG / BRSR Report PDF</div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:13px;color:#8B92A9;margin-bottom:20px'>"
        "Upload a PDF directly. It is stored independently and processed through "
        "the same parse → extract pipeline. SHA-256 deduplication prevents re-processing "
        "an identical file. The extracted KPIs are linked to this specific report."
        "</div>",
        unsafe_allow_html=True,
    )

    up_col1, up_col2 = st.columns([2, 1])
    with up_col1:
        upload_company = st.text_input(
            "Company name (required)", placeholder="e.g. Wipro", key="upload_company",
        )
        upload_fy = st.number_input(
            "Fiscal year (required)", min_value=2010, max_value=2030, value=2024,
            key="upload_fy",
        )
    with up_col2:
        upload_sector = st.selectbox("Sector", SECTORS, key="upload_sector")
        upload_report_type = st.selectbox(
            "Report type", UPLOAD_REPORT_TYPE_OPTIONS, key="upload_report_type",
        )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Select a PDF file", type=["pdf"], key="pdf_uploader",
    )

    if uploaded_file is not None:
        size_mb = round(uploaded_file.size / (1024 * 1024), 2)
        if size_mb > 50:
            st.error(f"File is {size_mb} MB which exceeds the 50 MB limit.")
            uploaded_file = None
        else:
            st.markdown(
                f"<div style='font-size:12px;color:{C['green']};margin-top:4px'>"
                f"Ready: <strong>{uploaded_file.name}</strong> ({size_mb} MB)"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    upload_ready = bool(uploaded_file and upload_company)
    upload_btn   = st.button(
        "Process Upload",
        disabled=not upload_ready or not db_online,
        key="upload_btn",
        use_container_width=True,
    )
    if not db_online:
        st.warning("Database is offline. Cannot process upload.")

    if upload_btn and upload_ready and db_online:
        st.markdown("---")
        st.markdown(f"### Processing {uploaded_file.name} for {upload_company} FY{upload_fy}")
        upload_status = st.empty()

        upload_result = run_upload_pipeline(
            uploaded_file=uploaded_file,
            company_name=upload_company,
            fy=int(upload_fy),
            sector=upload_sector,
            report_type=upload_report_type,
            db_online=db_online,
            llm_service=llm_service,
            status_placeholder=upload_status,
        )

        if upload_result["success"]:
            n_kpis = len(upload_result.get("kpi_records", {}))
            st.success(
                f"Upload processed. {n_kpis} KPI(s) extracted. "
                f"{upload_company} FY{upload_fy} is available in the Comparison tab."
            )
            kpis = upload_result.get("kpi_records", {})
            rev  = upload_result.get("revenue")
            if kpis:
                rows = [
                    {
                        "KPI":        KPI_META.get(k, {}).get("label", k),
                        "Value":      f"{r['value']:,.2f}",
                        "Unit":       r["unit"],
                        "Method":     r["method"],
                        "Confidence": f"{r['confidence']:.0%}",
                    }
                    for k, r in kpis.items()
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            if rev:
                st.markdown(
                    f"**Revenue:** INR {rev.value_cr:,.0f} Crore "
                    f"[{rev.pattern_name}, confidence {rev.confidence:.0%}]"
                )
        else:
            st.error("Upload pipeline failed.")

        with st.expander("Processing log"):
            for line in upload_result.get("log", []):
                st.code(line, language=None)

    elif uploaded_file is None:
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size:15px;font-weight:600;color:#E8EAF0;margin-bottom:8px">
                Select a PDF to upload
            </div>
            <div style="font-size:13px;color:#8B92A9;line-height:1.7">
                Supported: BRSR, ESG, Integrated, Annual, CSR<br>
                Maximum size: 50 MB<br>
                Processed independently — does not affect other reports for this company.
            </div>
        </div>
        """, unsafe_allow_html=True)