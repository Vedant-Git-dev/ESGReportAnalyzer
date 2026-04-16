"""
dashboard/ui.py  — ESG Competitive Intelligence Dashboard  (v3)

KPI group changes vs v2
-----------------------
  Added:   scope_3_emissions, energy_consumption, water_consumption,
           employee_count, women_in_workforce_percentage,
           renewable_energy_percentage, complaints_filed, complaints_pending
  Removed: total_ghg_emissions

UI auto-displays any KPI returned from the backend.
KPIs are grouped into Environmental / Social / Governance sections.
Governance KPIs (complaints) are shown as absolute counts, not intensity ratios.
Percentage KPIs (women %, renewable %) are shown as-is.
All others are shown as KPI / revenue (intensity ratios).

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
    "bg":      "#FFFFFF",
    "surface": "#F8F9FA",
    "border":  "#DEE2E6",
    "text":    "#212529",
    "sub":     "#6C757D",
    "green":   "#198754",
    "blue":    "#0D6EFD",
    "amber":   "#E6A817",
    "red":     "#DC3545",
    "ca":      "#0D6EFD",
    "cb":      "#198754",
    "grid":    "#E9ECEF",
    "font":    "Inter, 'Helvetica Neue', Arial, sans-serif",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] {{ font-family: {C['font']}; background-color: {C['bg']} !important; color: {C['text']} !important; }}
.stApp {{ background-color: {C['bg']} !important; }}
[data-testid="stSidebar"] {{ background-color: {C['surface']} !important; border-right: 1px solid {C['border']}; }}
[data-testid="stSidebar"] * {{ color: {C['text']} !important; }}
p, span, div, label, h1, h2, h3, h4, h5, h6, .stMarkdown, .stText, .stCaption {{ color: {C['text']} !important; }}
[data-baseweb="input"], [data-baseweb="input-container"], [data-baseweb="base-input"] {{ background-color: {C['bg']} !important; color: {C['text']} !important; }}
.stTextInput > div > div > input {{ background-color: {C['bg']} !important; color: {C['text']} !important; border: 1px solid {C['border']} !important; border-radius: 6px; }}
.stNumberInput input {{ background-color: {C['bg']} !important; color: {C['text']} !important; border: 1px solid {C['border']} !important; }}
[data-testid="stNumberInputContainer"] {{ background-color: {C['bg']} !important; }}
.stSelectbox > div > div {{ background-color: {C['bg']} !important; color: {C['text']} !important; border: 1px solid {C['border']} !important; border-radius: 6px; }}
[data-baseweb="popover"], [data-baseweb="menu"] {{ background-color: {C['bg']} !important; }}
[data-baseweb="option"] {{ background-color: {C['bg']} !important; color: {C['text']} !important; }}
.stButton > button {{ background-color: {C['blue']}; color: #ffffff !important; border: none; border-radius: 6px; padding: 0.5rem 1.2rem; font-weight: 600; font-size: 14px; width: 100%; transition: background 0.2s; }}
.stButton > button:hover {{ background-color: #0b5ed7; color: #ffffff !important; }}
.stButton > button:disabled {{ background-color: {C['border']} !important; color: {C['sub']} !important; }}
[data-testid="stFileUploader"] {{ background-color: {C['surface']} !important; border-radius: 8px; }}
[data-testid="stFileUploaderDropzone"] {{ background-color: {C['surface']} !important; border: 2px dashed {C['border']} !important; border-radius: 8px; }}
[data-testid="stTabs"] button {{ color: {C['text']} !important; background: transparent !important; }}
[data-testid="stTabs"] button[aria-selected="true"] {{ border-bottom: 2px solid {C['blue']} !important; color: {C['blue']} !important; }}
[data-testid="stExpander"] {{ background-color: {C['surface']} !important; border: 1px solid {C['border']} !important; border-radius: 8px; }}
.stAlert, .stInfo, .stWarning, .stError, .stSuccess {{ color: {C['text']} !important; }}
[data-testid="stAlertContainer"] * {{ color: {C['text']} !important; }}
[data-testid="stDataFrame"] * {{ color: {C['text']} !important; background-color: {C['bg']} !important; }}
.stCode, .stCodeBlock, code, pre {{ background-color: #F1F3F5 !important; color: #212529 !important; border: 1px solid {C['border']}; border-radius: 6px; }}
hr {{ border-color: {C['border']}; margin: 1rem 0; }}
.card {{ background: {C['surface']}; border: 1px solid {C['border']}; border-radius: 10px; padding: 16px 20px; margin-bottom: 10px; color: {C['text']} !important; }}
.card * {{ color: {C['text']} !important; }}
.sec {{ font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .1em; color: {C['sub']}; margin-bottom: 14px; padding-bottom: 6px; border-bottom: 2px solid {C['border']}; }}
.label {{ font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .07em; color: {C['text']}; margin-bottom: 3px; }}
.label-tag {{ font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .07em; color: {C['text']}; margin-bottom: 3px; }}
.badge-green {{ background: #D1E7DD; color: #0F5132 !important; border-radius: 20px; font-size: 11px; font-weight: 700; padding: 2px 10px; }}
.badge-blue {{ background: #CFE2FF; color: #084298 !important; border-radius: 6px; font-size: 11px; font-weight: 600; padding: 2px 8px; }}
.badge-amber {{ background: #FFF3CD; color: #664D03 !important; border-radius: 6px; font-size: 11px; font-weight: 600; padding: 2px 8px; }}
.summary-box {{ background: {C['surface']}; border-left: 4px solid {C['blue']}; border-radius: 0 8px 8px 0; padding: 18px 22px; line-height: 1.75; font-size: 14px; color: {C['text']} !important; }}
.step-log {{ background: #F1F3F5; border: 1px solid {C['border']}; border-radius: 6px; padding: 10px 14px; font-family: monospace; font-size: 12px; color: {C['text']} !important; line-height: 1.6; min-height: 60px; }}
.upload-info {{ background: #E8F4FD; border: 1px solid #B8D9F5; border-radius: 6px; padding: 8px 12px; font-size: 12px; color: #084298 !important; margin-top: 4px; }}
.upload-zone {{ background: {C['surface']}; border: 2px dashed {C['border']}; border-radius: 12px; padding: 24px; text-align: center; }}
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

# =============================================================================
# KPI CATALOGUE
# All UI display logic derives from this single source of truth.
# Adding a new KPI here is all that is needed to display it everywhere.
# =============================================================================

# KPI groups with display metadata
# "ratio_denominator": "revenue"  → value / revenue_cr  → shown as X / INR_Crore
# "ratio_denominator": "none"     → absolute value (governance + percentages)
KPI_GROUPS: dict[str, dict] = {
    # ── Environmental ────────────────────────────────────────────────────────
    "scope_1_emissions": {
        "label": "Scope 1 GHG",
        "group": "Environmental",
        "unit": "tCO2e",
        "ratio_unit": "tCO2e/Cr",
        "ratio_denominator": "revenue",
        "max_ratio": 10,
        "higher_is_better": False,
        "desc": "Direct GHG emissions per INR Crore revenue",
    },
    "scope_2_emissions": {
        "label": "Scope 2 GHG",
        "group": "Environmental",
        "unit": "tCO2e",
        "ratio_unit": "tCO2e/Cr",
        "ratio_denominator": "revenue",
        "max_ratio": 10,
        "higher_is_better": False,
        "desc": "Indirect GHG emissions per INR Crore revenue",
    },
    "scope_3_emissions": {
        "label": "Scope 3 GHG",
        "group": "Environmental",
        "unit": "tCO2e",
        "ratio_unit": "tCO2e/Cr",
        "ratio_denominator": "revenue",
        "max_ratio": 50,
        "higher_is_better": False,
        "desc": "Value chain GHG emissions per INR Crore revenue",
    },
    "energy_consumption": {
        "label": "Energy Intensity",
        "group": "Environmental",
        "unit": "GJ",
        "ratio_unit": "GJ/Cr",
        "ratio_denominator": "revenue",
        "max_ratio": 1_000,
        "higher_is_better": False,
        "desc": "Total energy consumed per INR Crore revenue",
    },
    "water_consumption": {
        "label": "Water Intensity",
        "group": "Environmental",
        "unit": "KL",
        "ratio_unit": "KL/Cr",
        "ratio_denominator": "revenue",
        "max_ratio": 500,
        "higher_is_better": False,
        "desc": "Total water consumed per INR Crore revenue",
    },
    "waste_generated": {
        "label": "Waste Intensity",
        "group": "Environmental",
        "unit": "MT",
        "ratio_unit": "MT/Cr",
        "ratio_denominator": "revenue",
        "max_ratio": 5,
        "higher_is_better": False,
        "desc": "Waste generated per INR Crore revenue",
    },
    "renewable_energy_percentage": {
        "label": "Renewable Energy",
        "group": "Environmental",
        "unit": "%",
        "ratio_unit": "%",
        "ratio_denominator": "none",
        "max_ratio": 100,
        "higher_is_better": True,
        "desc": "Share of energy from renewable sources",
    },
    # ── Social ────────────────────────────────────────────────────────────────
    "employee_count": {
        "label": "Workforce",
        "group": "Social",
        "unit": "count",
        "ratio_unit": "employees/Cr",
        "ratio_denominator": "revenue",
        "max_ratio": 5_000,
        "higher_is_better": False,
        "desc": "Total employees per INR Crore revenue",
    },
    "women_in_workforce_percentage": {
        "label": "Women in Workforce",
        "group": "Social",
        "unit": "%",
        "ratio_unit": "%",
        "ratio_denominator": "none",
        "max_ratio": 100,
        "higher_is_better": True,
        "desc": "Percentage of women in workforce",
    },
    # ── Governance ────────────────────────────────────────────────────────────
    "complaints_filed": {
        "label": "Complaints Filed",
        "group": "Governance",
        "unit": "count",
        "ratio_unit": "count",
        "ratio_denominator": "none",      # absolute — no normalization
        "max_ratio": 1_000_000,
        "higher_is_better": False,
        "desc": "Total complaints filed during the year",
    },
    "complaints_pending": {
        "label": "Complaints Pending",
        "group": "Governance",
        "unit": "count",
        "ratio_unit": "count",
        "ratio_denominator": "none",      # absolute — no normalization
        "max_ratio": 100_000,
        "higher_is_better": False,
        "desc": "Complaints pending resolution at year end",
    },
}

# Derived helpers from KPI_GROUPS (used throughout the module)
ALL_KPI_NAMES       = list(KPI_GROUPS.keys())
EXTRACTABLE_KPI_NAMES = ALL_KPI_NAMES   # all are extractable now

# Legacy alias kept so existing code still references KPI_META
KPI_META = KPI_GROUPS

_KPI_PLAUSIBILITY: dict[str, tuple[float, float]] = {
    "scope_1_emissions":             (1,      5_000_000),
    "scope_2_emissions":             (1,      5_000_000),
    "scope_3_emissions":             (1,     10_000_000),
    "energy_consumption":            (100,  500_000_000),
    "water_consumption":             (100,  100_000_000),
    "waste_generated":               (0.1,      500_000),
    "renewable_energy_percentage":   (0,             100),
    "employee_count":                (1,         5_000_000),
    "women_in_workforce_percentage": (0,             100),
    "complaints_filed":              (0,       1_000_000),
    "complaints_pending":            (0,         100_000),
}

_DEFAULT_REVENUE_CR = 315_322.0

_REPORT_TYPE_PRIORITY: dict[str, int] = {
    "Integrated": 0,
    "BRSR":       1,
    "ESG":        2,
}


# =============================================================================
# RATIO COMPUTATION  (central, used by benchmark and UI)
# =============================================================================

def compute_ratio(kpi_name: str, value: float, revenue_cr: float) -> float:
    """
    Compute the display ratio for a KPI.
    Governance and percentage KPIs return the raw value unchanged.
    All others return value / revenue_cr.
    """
    meta = KPI_GROUPS.get(kpi_name, {})
    if meta.get("ratio_denominator") == "none":
        return value
    if revenue_cr > 0:
        return value / revenue_cr
    return value


def format_ratio(value: float, kpi_name: str) -> str:
    """Human-readable ratio string."""
    if value == 0:
        return "0"
    if abs(value) < 0.001:
        return f"{value:.4e}"
    if abs(value) < 1:
        return f"{value:.4f}"
    return f"{value:,.2f}"


# =============================================================================
# DATA CONTAINERS  (unchanged from v2)
# =============================================================================

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

            infos = sorted(
                [ReportInfo(id=r.id, report_type=r.report_type,
                            file_path=r.file_path, status=r.status)
                 for r in reports],
                key=lambda ri: (
                    _REPORT_TYPE_PRIORITY.get(ri.report_type, 99),
                    0 if ri.file_path and Path(ri.file_path).exists() else 1,
                ),
            )
            return {"exists": True, "company_id": company_id, "reports": infos}

    except Exception as exc:
        st.warning(f"DB report lookup failed for {company_name}: {exc}")
        return empty


def _db_load_kpis_and_revenue(company_id: uuid.UUID, fy: int) -> dict:
    """
    Load best KPI record per KPI (Integrated > BRSR > ESG priority).
    Returns all active KPIs — UI renders whatever comes back.
    """
    empty = {"kpis": {}, "revenue": None, "file_path": None}
    try:
        from core.database import get_db
        from models.db_models import Report, KPIRecord, KPIDefinition
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

            rev_report = (
                db.query(Report)
                .filter(
                    Report.company_id == company_id,
                    Report.report_year == fy,
                    Report.revenue_cr.isnot(None),
                )
                .order_by(type_priority, Report.created_at.desc())
                .first()
            )

            cached_rev = None
            if rev_report and getattr(rev_report, "revenue_cr", None) is not None:
                try:
                    cached_rev = RevenueResult(
                        value_cr=float(rev_report.revenue_cr),
                        raw_value=str(rev_report.revenue_cr),
                        raw_unit=getattr(rev_report, "revenue_unit", None) or "INR_Crore",
                        source=getattr(rev_report, "revenue_source", None) or "db",
                        page_number=0, confidence=0.99, pattern_name="cached",
                    )
                except Exception:
                    pass

            kpis: dict = {}
            for kpi_name in EXTRACTABLE_KPI_NAMES:
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
                lo, hi = _KPI_PLAUSIBILITY.get(kpi_name, (0, float("inf")))
                if not (lo <= val <= hi):
                    continue

                src_report = db.query(Report).filter(Report.id == rec.report_id).first()
                rec_type   = src_report.report_type if src_report else "unknown"

                kpis[kpi_name] = {
                    "value":       val,
                    "unit":        unit,
                    "method":      rec.extraction_method,
                    "confidence":  rec.confidence or 0.9,
                    "report_type": rec_type,
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
                    .filter(KPIDefinition.name == kpi_name,
                            KPIDefinition.is_active == True)
                    .first()
                )
                if not kdef:
                    continue

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
                    report_id         = report_id,
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
# PIPELINE STEPS  (unchanged logic, updated KPI list)
# =============================================================================

def _step_ingest(company_name: str, fy: int, sector: str, log: list[str]) -> dict:
    from agents.ingestion_agent import IngestionAgent
    from models.schemas import CompanyCreate

    agent = IngestionAgent()
    company_data = CompanyCreate(name=company_name, sector=sector, country="India")
    log.append(f"Searching BRSR, ESG, and Integrated reports for {company_name} FY{fy}.")

    try:
        result = agent.run_multi_report_types(company_data=company_data, year=fy, auto_download=True)
    except Exception as exc:
        log.append(f"  Ingestion failed: {exc}")
        return {"company_id": None, "reports": []}

    company   = result["company"]
    downloads = result["downloaded_reports"]
    not_found = result.get("not_found_types", [])
    failed    = result.get("failed_types", [])

    for rtype in INGESTION_REPORT_TYPES:
        dl = next((d for d in downloads if d.report_type == rtype), None)
        if dl:
            fname   = Path(dl.file_path).name if dl.file_path else "unknown"
            size_mb = round((dl.file_size_bytes or 0) / (1024 * 1024), 1)
            log.append(f"  [{rtype}] downloaded: {fname} ({size_mb} MB)")
        elif rtype in not_found:
            log.append(f"  [{rtype}] not found.")
        elif rtype in failed:
            log.append(f"  [{rtype}] download failed.")

    if not downloads:
        log.append("No PDFs downloaded.")
        return {"company_id": company.id if company else None, "reports": []}

    reports = [
        ReportInfo(id=d.id, report_type=d.report_type,
                   file_path=d.file_path, status=d.status)
        for d in downloads if d.status == "downloaded"
    ]
    log.append(f"  {len(reports)} report(s) ready.")
    return {"company_id": company.id, "reports": reports}


def _step_parse(report_id: uuid.UUID, report_type: str, log: list[str]) -> bool:
    from services.parse_orchestrator import ParseOrchestrator
    log.append(f"  Parsing [{report_type}] (id={str(report_id)[:8]})...")
    try:
        result = ParseOrchestrator().run(report_id=report_id, force=False)
        log.append(f"    {result.page_count} pages, {result.meta.get('chunk_count', '?')} chunks.")
        return True
    except Exception as exc:
        log.append(f"    Parse failed: {exc}")
        return False


def _step_extract(
    report_id: uuid.UUID, report_type: str, fy: int,
    log: list[str], llm_service,
) -> dict:
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
                log.append(f"    {ext.kpi_name}: {val:,.2f} {unit} out of range — dropped")
                continue
            new_kpis[ext.kpi_name] = {
                "value": val, "unit": unit,
                "method": ext.extraction_method,
                "confidence": ext.confidence or 0.5,
            }
            log.append(
                f"    {ext.kpi_name}: {val:,.2f} {unit} "
                f"[{ext.extraction_method} conf={ext.confidence:.2f}]"
            )
    except Exception as exc:
        log.append(f"    KPI extraction failed: {exc}")

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
                pdf_path=Path(pdf_path_str),
                fiscal_year_hint=fy,
                llm_service=llm_service,
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
    """Derive total_ghg from scope_1 + scope_2 — kept for backward compat."""
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
        "value": total, "unit": "tCO2e", "method": "derived",
        "confidence": round(min(s1["confidence"], s2["confidence"]) * 0.99, 3),
    }


# =============================================================================
# MAIN PIPELINE  (same logic as v2, references EXTRACTABLE_KPI_NAMES)
# =============================================================================

def run_company_pipeline(
    company_name: str, fy: int, sector: str, db_online: bool,
    llm_service, status_placeholder, uploaded_file=None, upload_report_type: str = "BRSR",
) -> CompanyData:
    log: list[str] = []

    def _update(msg: str) -> None:
        log.append(msg)
        status_placeholder.markdown(
            "<div class='step-log'>" + "<br>".join(log[-16:]) + "</div>",
            unsafe_allow_html=True,
        )

    _update(f"Starting pipeline for {company_name} FY{fy}.")

    if uploaded_file is not None:
        _update(f"PDF uploaded: {uploaded_file.name}. Running upload pipeline...")
        upload_result = run_upload_pipeline(
            uploaded_file=uploaded_file, company_name=company_name, fy=fy,
            sector=sector, report_type=upload_report_type, db_online=db_online,
            llm_service=llm_service, status_placeholder=status_placeholder,
        )
        for line in upload_result.get("log", []):
            if line not in log:
                log.append(line)
        if upload_result.get("success"):
            merged = {**upload_result.get("kpi_records", {})}
            rev    = upload_result.get("revenue")
            cid    = upload_result.get("company_id")
            if db_online and cid:
                db_fill = _db_load_kpis_and_revenue(cid, fy)
                for k, v in db_fill.get("kpis", {}).items():
                    if k not in merged:
                        merged[k] = v
                if not rev:
                    rev = db_fill.get("revenue")
            _update(f"Upload complete: {len(merged)} KPI(s).")
            return CompanyData(
                company_name=company_name, fy=fy, sector=sector,
                kpi_records=merged, revenue_result=rev, log=log,
                company_id=cid,
                report_infos=upload_result.get("report_infos", []),
                file_path=(upload_result.get("report_infos") or [{}])[0].file_path
                           if upload_result.get("report_infos") else None,
            )
        else:
            _update("Upload failed. Falling back to DB/search pipeline.")

    if not db_online:
        _update("Database offline.")
        return CompanyData(company_name=company_name, fy=fy, sector=sector,
                           kpi_records={}, revenue_result=None, log=log)

    _update("Checking database for existing reports...")
    db_data    = _db_get_all_reports(company_name, fy)
    company_id = db_data.get("company_id")
    report_infos: list[ReportInfo] = []

    if db_data["exists"]:
        report_infos = db_data["reports"]
        types_found  = [ri.report_type for ri in report_infos]
        _update(f"  Found {len(report_infos)} report(s): {types_found}. Skipping search.")
    else:
        _update(f"  No reports in DB. Running ingestion...")
        ingest = _step_ingest(company_name, fy, sector, log)
        company_id   = ingest.get("company_id") or company_id
        report_infos = ingest.get("reports", [])
        if not report_infos:
            _update("No downloadable reports.")
            return CompanyData(company_name=company_name, fy=fy, sector=sector,
                               kpi_records={}, revenue_result=None, log=log,
                               company_id=company_id)
        db_data    = _db_get_all_reports(company_name, fy)
        company_id = db_data.get("company_id") or company_id
        if db_data["exists"]:
            report_infos = db_data["reports"]

    _update(f"Processing {len(report_infos)} report(s)...")

    sorted_reports = sorted(report_infos,
                            key=lambda r: _REPORT_TYPE_PRIORITY.get(r.report_type, 99))
    still_missing = list(EXTRACTABLE_KPI_NAMES)
    need_revenue  = True

    for ri in sorted_reports:
        if not still_missing and not need_revenue:
            break
        _update(f"--- [{ri.report_type}] report_id={str(ri.id)[:8]} ---")
        parse_ok = _step_parse(ri.id, ri.report_type, log)
        if not parse_ok:
            continue
        extract_result = _step_extract(ri.id, ri.report_type, fy, log, llm_service)
        new_kpis    = extract_result["kpis"]
        new_revenue = extract_result["revenue"]

        for found in list(new_kpis.keys()):
            if found in still_missing:
                still_missing.remove(found)
        if new_revenue:
            need_revenue = False

        if company_id and (new_kpis or new_revenue):
            _db_store_kpis(company_id, ri.id, fy, new_kpis, new_revenue)
            _update(f"  Stored {len(new_kpis)} KPI(s) for [{ri.report_type}].")

    _update("Loading final KPI state from DB...")

    final_db      = _db_load_kpis_and_revenue(company_id, fy) if company_id else {"kpis": {}, "revenue": None, "file_path": None}
    final_kpis    = final_db["kpis"]
    final_revenue = final_db["revenue"]
    final_fp      = final_db["file_path"]

    if final_kpis:
        _update(f"  Final KPIs ({len(final_kpis)}): {list(final_kpis.keys())}")
    if final_revenue:
        _update(f"  Revenue: INR {final_revenue.value_cr:,.0f} Crore")
    else:
        _update(f"  Revenue: not found. Default {_DEFAULT_REVENUE_CR:,.0f} Cr used.")

    _update(f"Pipeline complete for {company_name} FY{fy}.")

    return CompanyData(
        company_name=company_name, fy=fy, sector=sector,
        kpi_records=final_kpis, revenue_result=final_revenue, log=log,
        company_id=company_id, report_infos=report_infos, file_path=final_fp,
    )


def run_upload_pipeline(
    uploaded_file, company_name: str, fy: int, sector: str,
    report_type: str, db_online: bool, llm_service, status_placeholder,
) -> dict:
    from agents.ingestion_agent import IngestionAgent
    log: list[str] = []

    def _update(msg: str) -> None:
        log.append(msg)
        status_placeholder.markdown(
            "<div class='step-log'>" + "<br>".join(log[-14:]) + "</div>",
            unsafe_allow_html=True,
        )

    _update(f"Processing upload: {uploaded_file.name}")

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, prefix="esg_upload_") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = Path(tmp.name)
    except Exception as exc:
        _update(f"  Failed to save: {exc}")
        return {"success": False, "log": log}

    try:
        result = IngestionAgent().ingest_uploaded_pdf(
            source_path=tmp_path, company_name=company_name,
            year=fy, sector=sector, report_type=report_type,
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
        _update(f"  Duplicate: reusing report {str(report_id)[:8]}.")
    ri = ReportInfo(id=report_id, report_type=report_type,
                    file_path=report.file_path, status=report.status)

    parsed_ok = _step_parse(report_id, report_type, log)
    if not parsed_ok:
        _update("  Parsing failed.")

    extract_result = _step_extract(report_id, report_type, fy, log, llm_service)
    new_kpis    = extract_result["kpis"]
    new_revenue = extract_result["revenue"]

    if db_online and (new_kpis or new_revenue):
        _db_store_kpis(company_id, report_id, fy, new_kpis, new_revenue)
        _update(f"  Stored {len(new_kpis)} KPI record(s).")

    _update("Upload pipeline complete.")
    return {
        "success": True, "kpi_records": new_kpis, "revenue": new_revenue,
        "company_id": company_id, "report_id": report_id,
        "report_infos": [ri], "log": log,
    }


# =============================================================================
# BENCHMARK BUILDER  (updated for new KPI groups)
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
        rev_src = rev.source if rev else "default"

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
            kpi_records=data.kpi_records, revenue_cr=rev_cr,
            revenue_source=rev_src, company_name=data.company_name,
            fiscal_year=data.fy, page_texts=page_texts,
        )
        profiles.append(profile)

    report   = compare_profiles(profiles)

    # Filter by ceiling — governance KPIs always pass (no ceiling exceeded in practice)
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
    """Keep only comparisons where both values pass their ceiling."""
    out = []
    for comp in comparisons:
        meta    = KPI_GROUPS.get(comp.kpi_name, {})
        ceiling = meta.get("max_ratio")
        if all(
            (not ceiling or v <= ceiling) and v >= 0
            for _, v, _ in comp.entries
        ):
            out.append(comp)
    return out


# =============================================================================
# CHART HELPERS
# =============================================================================

_CHART_FONT = dict(family="Inter, Arial, sans-serif", size=12, color=C["text"])


def _hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _radar_chart(filtered, la: str, lb: str):
    cats, sa, sb = [], [], []
    for comp in filtered:
        vals   = {lbl: v for lbl, v, _ in comp.entries}
        va, vb = vals.get(la), vals.get(lb)
        if va is None or vb is None:
            continue
        meta = KPI_GROUPS.get(comp.kpi_name, {})
        label = meta.get("label", comp.display_name)
        cats.append(label)
        total = va + vb
        # For higher-is-better KPIs, invert the score
        if meta.get("higher_is_better"):
            sa.append(round(100 * va / total, 1) if total else 50)
            sb.append(round(100 * vb / total, 1) if total else 50)
        else:
            sa.append(round(100 * (1 - va / total), 1) if total else 50)
            sb.append(round(100 * (1 - vb / total), 1) if total else 50)

    if len(cats) < 2:
        return None
    cats_c = cats + [cats[0]]
    sa_c   = sa   + [sa[0]]
    sb_c   = sb   + [sb[0]]
    fig = go.Figure()
    for name, scores, color in [(la.split(" FY")[0], sa_c, C["ca"]),
                                 (lb.split(" FY")[0], sb_c, C["cb"])]:
        fig.add_trace(go.Scatterpolar(
            r=scores, theta=cats_c, fill="toself", name=name,
            line=dict(color=color, width=2.5),
            fillcolor=_hex_rgba(color, 0.12),
            hovertemplate="<b>%{theta}</b><br>Score: %{r:.0f}<extra></extra>",
        ))
    fig.update_layout(
        polar=dict(
            bgcolor=C["surface"],
            radialaxis=dict(visible=True, range=[0, 100],
                            tickfont=dict(size=9, color=C["sub"]),
                            gridcolor=C["grid"], linecolor=C["border"]),
            angularaxis=dict(tickfont=dict(size=11, color=C["text"]),
                             gridcolor=C["grid"], linecolor=C["border"]),
        ),
        paper_bgcolor=C["bg"], height=380,
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
        paper_bgcolor=C["bg"], height=260,
        margin=dict(l=0, r=0, t=10, b=0),
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
        meta = KPI_GROUPS.get(comp.kpi_name, {})
        labels_list.append(meta.get("label", comp.display_name))
        va_list.append(va)
        vb_list.append(vb)
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
        barmode="group", height=max(300, len(labels_list) * 45),
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"], font=_CHART_FONT,
        xaxis=dict(title="Value", showgrid=True, gridcolor=C["grid"],
                   zeroline=False, color=C["text"]),
        yaxis=dict(autorange="reversed", color=C["text"]),
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=180, r=60, t=20, b=40),
    )
    return fig


def _mini_bar_chart(comp, la: str, lb: str):
    vals   = {lbl: v for lbl, v, _ in comp.entries}
    va, vb = vals.get(la, 0), vals.get(lb, 0)
    meta   = KPI_GROUPS.get(comp.kpi_name, {})
    na, nb = la.split(" FY")[0], lb.split(" FY")[0]
    fig = go.Figure()
    for name, val, color, pat in [(na, va, C["ca"], ""), (nb, vb, C["cb"], "/")]:
        fig.add_trace(go.Bar(
            name=name, x=[val], y=[meta.get("label", "")], orientation="h",
            marker=dict(color=color, line=dict(color=color, width=1.5), pattern_shape=pat),
            text=[f"{val:.3g}"], textposition="outside",
            textfont=dict(size=11, color=C["text"]),
            hovertemplate=f"<b>{name}</b><br>{val:.4e} {meta.get('ratio_unit', '')}<extra></extra>",
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
# PDF REPORT EXPORT  (updated for new KPI groups)
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
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             leftMargin=2*cm, rightMargin=2*cm,
                             topMargin=2*cm, bottomMargin=2*cm)

    BLUE  = colors.HexColor("#3B82F6")
    GREEN = colors.HexColor("#10B981")
    GRAY  = colors.HexColor("#64748B")
    LIGHT = colors.HexColor("#F8F9FB")
    BDR   = colors.HexColor("#E2E8F0")
    BLK   = colors.HexColor("#1A202C")

    ss = getSampleStyleSheet()
    s_title = ParagraphStyle("t", parent=ss["Title"],   fontSize=22, textColor=BLK, spaceAfter=4, leading=28, fontName="Helvetica-Bold")
    s_sub   = ParagraphStyle("s", parent=ss["Normal"],  fontSize=11, textColor=GRAY, spaceAfter=14, leading=16)
    s_h2    = ParagraphStyle("h2", parent=ss["Heading2"],fontSize=13, textColor=BLK, spaceBefore=16, spaceAfter=8, fontName="Helvetica-Bold")
    s_body  = ParagraphStyle("b", parent=ss["Normal"],  fontSize=10, textColor=BLK, leading=16, spaceAfter=8)
    s_note  = ParagraphStyle("n", parent=ss["Normal"],  fontSize=9,  textColor=GRAY, leading=14)

    labels = [f"{p.company_name} FY{p.fiscal_year}" for p in profiles]
    story  = [
        Paragraph("ESG Competitive Intelligence Report", s_title),
        Paragraph(f"{labels[0]} vs {labels[1]}", s_sub),
        Paragraph(f"Sector: {sector}", s_note),
        HRFlowable(width="100%", thickness=1, color=BDR, spaceAfter=14),
        Paragraph("Methodology", s_h2),
        Paragraph(
            "Environmental & Social KPIs: normalized by annual revenue (INR Crore). "
            "Governance KPIs (complaints): shown as absolute counts. "
            "Percentage KPIs (women %, renewable %): shown as-is. "
            "KPI source priority: Integrated > BRSR > ESG.",
            s_body,
        ),
        Spacer(1, 8),
        Paragraph("KPI Comparison", s_h2),
    ]

    tdata = [["Group", "Metric", "Unit", labels[0], labels[1], "Gap", "Leader"]]
    for comp in filtered:
        meta = KPI_GROUPS.get(comp.kpi_name, {})
        vals = {lbl: v for lbl, v, _ in comp.entries}
        v0, v1 = vals.get(labels[0]), vals.get(labels[1])
        fmt = lambda v: f"{v:.2e}" if (v and 0 < v < 0.001) else (f"{v:,.2f}" if v else "N/A")
        tdata.append([
            meta.get("group", ""),
            meta.get("label", comp.display_name),
            meta.get("ratio_unit", ""),
            fmt(v0), fmt(v1),
            f"{comp.pct_gap:.1f}%",
            comp.winner.split(" FY")[0],
        ])

    tbl = Table(tdata, colWidths=[2.8*cm, 4.0*cm, 2.2*cm, 2.4*cm, 2.4*cm, 1.5*cm, 2.2*cm], repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  BLUE),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  8),
        ("BOTTOMPADDING", (0, 0), (-1, 0),  7),
        ("TOPPADDING",    (0, 0), (-1, 0),  7),
        ("FONTSIZE",      (0, 1), (-1, -1), 8),
        ("TOPPADDING",    (0, 1), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID",          (0, 0), (-1, -1), 0.5, BDR),
        ("ALIGN",         (3, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TEXTCOLOR",     (6, 1), ( 6, -1), GREEN),
        ("FONTNAME",      (6, 1), ( 6, -1), "Helvetica-Bold"),
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
            "Source: public BRSR, ESG, and Integrated Annual Reports.",
            s_note,
        ),
    ]
    doc.build(story)
    return buf.getvalue()


# =============================================================================
# LLM
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
    st.markdown(
        f"""<div style="padding:0 0 12px">
          <div style="font-size:20px;font-weight:800;color:{C['text']}">
            ESG Intelligence
          </div>
          <div style="font-size:11px;color:{C['sub']};margin-top:2px">
            Multi-Source Benchmarking Pipeline
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    db_col = C["green"] if db_online else C["red"]
    st.markdown(
        f"<div style='font-size:11px;color:{db_col};font-weight:600'>"
        f"{'● DB connected' if db_online else '● DB offline'}"
        + (f" ({len(known_companies)} companies)" if db_online else "")
        + "</div>", unsafe_allow_html=True,
    )
    llm_col = C["green"] if llm_service else C["amber"]
    st.markdown(
        f"<div style='font-size:11px;color:{llm_col};font-weight:600'>"
        f"{'● LLM enabled' if llm_service else '⚠ LLM disabled'}"
        f"</div>", unsafe_allow_html=True,
    )

    st.markdown("---")
    sector = st.selectbox("Sector", SECTORS, key="sector")
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    for ckey, fkey, ukey, rkey, cname_hint in [
        ("c1_name", "c1_fy", "upload1", "rtype1", "e.g. Infosys"),
        ("c2_name", "c2_fy", "upload2", "rtype2", "e.g. TCS"),
    ]:
        color = C["ca"] if ckey == "c1_name" else C["cb"]
        label = "Company / Dataset 1" if ckey == "c1_name" else "Company / Dataset 2"
        st.markdown(
            f'<div class="label-tag" style="color:{color}">{label}</div>',
            unsafe_allow_html=True,
        )
        st.text_input("", placeholder=cname_hint, label_visibility="collapsed", key=ckey)
        st.number_input("FY", min_value=2015, max_value=2030, value=2025 if ckey == "c1_name" else 2024,
                        label_visibility="collapsed", key=fkey)
        upf = st.file_uploader("Upload PDF (optional)", type=["pdf"], key=ukey)
        if upf:
            rtype = st.selectbox("Report type", UPLOAD_REPORT_TYPE_OPTIONS,
                                  key=rkey, label_visibility="collapsed")
            st.markdown(
                f"<div class='upload-info'>{upf.name} ({round(upf.size/1e6,1)} MB)</div>",
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    company1 = st.session_state.get("c1_name", "")
    company2 = st.session_state.get("c2_name", "")
    fy1      = st.session_state.get("c1_fy", 2025)
    fy2      = st.session_state.get("c2_fy", 2024)
    upload1  = st.session_state.get("upload1")
    upload2  = st.session_state.get("upload2")
    rtype1   = st.session_state.get("rtype1", "BRSR")
    rtype2   = st.session_state.get("rtype2", "BRSR")

    ready       = bool(company1 and company2 and company1.strip().lower() != company2.strip().lower())
    compare_btn = st.button("Compare", disabled=not ready, use_container_width=True)
    st.markdown("---")


# =============================================================================
# MAIN CONTENT
# =============================================================================

tab_compare, tab_upload = st.tabs(["Comparison", "Upload PDF"])


# ---------------------------------------------------------------------------
# COMPARISON TAB
# ---------------------------------------------------------------------------

with tab_compare:

    if "result" not in st.session_state and not compare_btn:
        st.markdown(f"""
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;padding:60px 40px;text-align:center">
          <div style="font-size:44px;margin-bottom:12px"></div>
          <div style="font-size:26px;font-weight:800;color:{C['text']};
                      letter-spacing:-0.5px;margin-bottom:8px">
              ESG Competitive Intelligence
          </div>
          <div style="font-size:14px;color:{C['sub']};max-width:560px;line-height:1.8;margin-bottom:24px">
              {len(KPI_GROUPS)} KPIs across Environmental, Social &amp; Governance groups.<br>
              Governance KPIs shown as absolute counts.<br>
              All others normalized by revenue (INR Crore).
          </div>
          <div style="display:flex;gap:10px;justify-content:center;flex-wrap:wrap">
            <span class="badge-blue">Environmental</span>
            <span class="badge-green">Social</span>
            <span class="badge-amber">Governance</span>
          </div>
        </div>""", unsafe_allow_html=True)

    if compare_btn and ready:
        st.markdown(f"### {company1} FY{fy1} vs {company2} FY{fy2}")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown(f"**{company1} FY{fy1}**")
            placeholder1 = st.empty()
        with col_s2:
            st.markdown(f"**{company2} FY{fy2}**")
            placeholder2 = st.empty()

        data1 = data2 = None
        pipeline_error = None

        try:
            data1 = run_company_pipeline(
                company_name=company1, fy=int(fy1), sector=sector,
                db_online=db_online, llm_service=llm_service,
                status_placeholder=placeholder1,
                uploaded_file=upload1, upload_report_type=rtype1 or "BRSR",
            )
            data2 = run_company_pipeline(
                company_name=company2, fy=int(fy2), sector=sector,
                db_online=db_online, llm_service=llm_service,
                status_placeholder=placeholder2,
                uploaded_file=upload2, upload_report_type=rtype2 or "BRSR",
            )
        except Exception as exc:
            pipeline_error = exc
            st.error(f"Pipeline error: {exc}")
            with st.expander("Full traceback"):
                st.code(traceback.format_exc())

        if pipeline_error is None and data1 is not None and data2 is not None:
            if not data1.kpi_records and not data2.kpi_records:
                st.error("No KPIs extracted for either company.")
                st.stop()

            result = _build_benchmark(data1, data2, sector)
            result.update({
                "log1": data1.log, "log2": data2.log,
                "sector": sector,
                "reports1": data1.report_infos,
                "reports2": data2.report_infos,
            })
            st.session_state.update({
                "result": result,
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
                &nbsp;&middot;&nbsp;Integrated &gt; BRSR &gt; ESG priority
            </div>""", unsafe_allow_html=True)
        with hc2:
            pdf_bytes = _export_pdf_report(profiles, filtered, summary, _sector)
            st.download_button(
                "Download PDF Report", data=pdf_bytes,
                file_name=f"ESG_{c1n}_vs_{c2n}.pdf",
                mime="application/pdf", use_container_width=True,
            )

        if skipped:
            st.info(f"Excluded (ceiling exceeded): "
                    f"{', '.join(KPI_GROUPS.get(k, {}).get('label', k) for k in skipped)}")

        if not filtered:
            st.warning("No KPIs passed the sanity filter.")
            st.stop()

        st.markdown("---")

        # Scorecard
        st.markdown('<div class="sec">Overview</div>', unsafe_allow_html=True)
        wins_a = sum(1 for c in filtered if c.winner == label_a)
        wins_b = sum(1 for c in filtered if c.winner == label_b)
        leader = c1n if wins_a >= wins_b else c2n

        score_cols = st.columns(4)
        for col, (bc, bl, bv, bs) in zip(score_cols, [
            (C["blue"],  "KPIs Compared",  str(len(filtered)),  f"of {len(report.comparisons)} total"),
            (C["ca"],    f"{c1n} FY{fy1v}", str(wins_a),         "KPI wins"),
            (C["cb"],    f"{c2n} FY{fy2v}", str(wins_b),         "KPI wins"),
            (C["green"], "Leader",          leader,               "More KPI wins"),
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

        # Charts
        vc1, vc2 = st.columns([2, 1])
        with vc1:
            st.markdown('<div class="sec">Performance Radar</div>', unsafe_allow_html=True)
            st.caption("Higher score = better performance on that metric")
            fig = _radar_chart(filtered, label_a, label_b)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with vc2:
            st.markdown('<div class="sec">Win Distribution</div>', unsafe_allow_html=True)
            st.plotly_chart(
                _donut_chart(filtered, label_a, label_b),
                use_container_width=True, config={"displayModeBar": False},
            )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec">KPI Comparison by Group</div>', unsafe_allow_html=True)
        gap = _gap_bar_chart(filtered, label_a, label_b)
        if gap:
            st.plotly_chart(gap, use_container_width=True, config={"displayModeBar": False})

        # Per-group KPI detail — auto-generated from KPI_GROUPS
        for group_name, group_color in [
            ("Environmental", C["green"]),
            ("Social",        C["blue"]),
            ("Governance",    C["amber"]),
        ]:
            group_comps = [c for c in filtered
                           if KPI_GROUPS.get(c.kpi_name, {}).get("group") == group_name]
            if not group_comps:
                continue

            st.markdown(f"""
            <div class="sec" style="margin-top:24px;border-bottom-color:{group_color}">
                {group_name}
            </div>""", unsafe_allow_html=True)

            for comp in group_comps:
                meta   = KPI_GROUPS.get(comp.kpi_name, {})
                vals   = {lbl: v for lbl, v, _ in comp.entries}
                va, vb = vals.get(label_a), vals.get(label_b)
                wname  = comp.winner.split(" FY")[0]
                a_wins = comp.winner == label_a

                st.markdown(f"""
                <div class="card">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
                        <div>
                            <span style="font-size:16px;font-weight:700">
                                {meta.get('label', comp.display_name)}
                            </span>
                            <div style="font-size:11px;color:{C['sub']};margin-top:2px">
                                {meta.get('desc', '')} &middot; {meta.get('ratio_unit', comp.unit)}
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
                    vs = format_ratio(val, comp.kpi_name) if val is not None else "N/A"
                    col.markdown(f"""
                    <div style="background:{C['surface']};border-radius:8px;
                                padding:10px 14px;border:1px solid {C['border']};
                                border-left:4px solid {color}">
                        <div style="font-size:11px;font-weight:600;color:{color}">
                            {dname}{' (leader)' if is_win else ''}
                        </div>
                        <div style="font-size:13px;color:{C['text']};margin-top:4px;
                                    font-family:monospace">
                            {vs} {meta.get('ratio_unit', '')}
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

        # Logs
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
# UPLOAD TAB
# ---------------------------------------------------------------------------

with tab_upload:
    st.markdown('<div class="sec">Upload ESG / BRSR Report PDF</div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:13px;color:#8B92A9;margin-bottom:20px'>"
        "Upload a PDF directly. Extracted KPIs are available in the Comparison tab immediately."
        "</div>",
        unsafe_allow_html=True,
    )

    up_col1, up_col2 = st.columns([2, 1])
    with up_col1:
        upload_company = st.text_input("Company name (required)", placeholder="e.g. Wipro", key="upload_company")
        upload_fy = st.number_input("Fiscal year (required)", min_value=2010, max_value=2030, value=2024, key="upload_fy")
    with up_col2:
        upload_sector      = st.selectbox("Sector", SECTORS, key="upload_sector")
        upload_report_type = st.selectbox("Report type", UPLOAD_REPORT_TYPE_OPTIONS, key="upload_report_type")

    uploaded_file = st.file_uploader("Select a PDF file", type=["pdf"], key="pdf_uploader")

    if uploaded_file is not None:
        size_mb = round(uploaded_file.size / (1024 * 1024), 2)
        if size_mb > 50:
            st.error(f"File is {size_mb} MB which exceeds the 50 MB limit.")
            uploaded_file = None
        else:
            st.markdown(
                f"<div style='font-size:12px;color:{C['green']};margin-top:4px'>"
                f"Ready: <strong>{uploaded_file.name}</strong> ({size_mb} MB)</div>",
                unsafe_allow_html=True,
            )

    upload_ready = bool(uploaded_file and upload_company)
    upload_btn   = st.button(
        "Process Upload",
        disabled=not upload_ready or not db_online,
        key="upload_btn",
        use_container_width=True,
    )

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
            kpis = upload_result.get("kpi_records", {})
            rev  = upload_result.get("revenue")
            st.success(f"Upload processed. {len(kpis)} KPI(s) extracted.")
            if kpis:
                rows = []
                for k, r in kpis.items():
                    meta = KPI_GROUPS.get(k, {})
                    rows.append({
                        "Group":      meta.get("group", ""),
                        "KPI":        meta.get("label", k),
                        "Value":      f"{r['value']:,.2f}",
                        "Unit":       r["unit"],
                        "Method":     r["method"],
                        "Confidence": f"{r['confidence']:.0%}",
                    })
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