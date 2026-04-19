"""
dashboard/ui.py  — ESG Competitive Intelligence Dashboard  (v4)

KPI-LEVEL CACHING FIX (v4 vs v3)
----------------------------------
v3 always called ExtractionAgent().extract_all() for every report, regardless
of what was already stored in kpi_records.  This caused:
  - Redundant LLM/regex extraction on every page load
  - Slow UI response even when all data was already in DB
  - Duplicate kpi_records rows

v4 introduces KPI-level caching via KPICacheService:

  CASE 1 — Full cache hit (all KPIs in kpi_records):
    → Return cached values immediately, skip extraction entirely.

  CASE 2 — Partial cache hit (some KPIs missing):
    → Use cached values for existing KPIs.
    → Run ExtractionAgent ONLY for the missing KPI names.
    → Store new values.

  CASE 3 — Cache miss (nothing cached):
    → Run full pipeline as before.

The cache uses Integrated > BRSR > ESG priority per KPI, identical to
run_benchmark.py.  The extraction logic, retrieval logic, parsing logic,
DB schema, and UI theme are all unchanged.

New internal helpers
--------------------
  _cache_load(company_id, fy, db)
      Thin wrapper around KPICacheService.select_best_per_kpi().
      Returns {kpi_name: record_dict} for all KPIs that have valid
      cached values.  Used at the top of run_company_pipeline() before
      any extraction is attempted.

  _step_extract_missing(report_id, report_type, fy, log, llm_service,
                        missing_kpi_names)
      Replaces the old _step_extract().  Accepts an explicit list of
      KPI names to extract; skips the ExtractionAgent call entirely when
      the list is empty.  Revenue extraction is unchanged (always runs
      when needed).

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
# DESIGN TOKENS
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

/* ── Base ── */
html, body, [class*="css"] {{
    font-family: {C['font']};
    background-color: {C['bg']} !important;
    color: {C['text']} !important;
}}
.stApp {{ background-color: {C['bg']} !important; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {C['surface']} !important;
    border-right: 1px solid {C['border']};
}}
[data-testid="stSidebar"] * {{ color: {C['text']} !important; }}

/* ── Typography ── */
p, span, div, label, h1, h2, h3, h4, h5, h6,
.stMarkdown, .stText, .stCaption {{
    color: {C['text']} !important;
}}

/* ── Text inputs ── */
[data-baseweb="input"],
[data-baseweb="input-container"],
[data-baseweb="base-input"] {{
    background-color: {C['bg']} !important;
    color: {C['text']} !important;
}}
.stTextInput > div > div > input {{
    background-color: {C['bg']} !important;
    color: {C['text']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 6px;
}}
.stNumberInput input {{
    background-color: {C['bg']} !important;
    color: {C['text']} !important;
    border: 1px solid {C['border']} !important;
}}
[data-testid="stNumberInputContainer"] {{
    background-color: {C['bg']} !important;
}}

/* ── Select / dropdown – FIXED VISIBILITY ── */
.stSelectbox > div > div {{
    background-color: {C['bg']} !important;
    color: {C['text']} !important;
    border: 1.5px solid {C['border']} !important;
    border-radius: 6px;
}}
div[data-baseweb="select"] > div {{
    background-color: {C['bg']} !important;
    color: {C['text']} !important;
    border: 1.5px solid {C['border']} !important;
    border-radius: 6px;
}}
div[data-baseweb="select"] span {{
    color: {C['text']} !important;
}}
[data-baseweb="popover"],
[data-baseweb="menu"] {{
    background-color: {C['bg']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.12) !important;
}}
[data-baseweb="option"] {{
    background-color: {C['bg']} !important;
    color: {C['text']} !important;
}}
[data-baseweb="option"]:hover {{
    background-color: {C['surface']} !important;
    color: {C['text']} !important;
}}
/* Dropdown arrow icon */
[data-baseweb="select"] svg {{
    fill: {C['text']} !important;
    color: {C['text']} !important;
}}

/* ── Buttons – FIXED VISIBILITY ── */
.stButton > button {{
    background-color: {C['blue']} !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px;
    padding: 0.5rem 1.2rem;
    font-weight: 600;
    font-size: 14px;
    width: 100%;
    transition: background 0.2s, box-shadow 0.2s;
    box-shadow: 0 2px 6px rgba(13,110,253,0.25);
}}
.stButton > button:hover {{
    background-color: #0b5ed7 !important;
    color: #ffffff !important;
    box-shadow: 0 4px 12px rgba(13,110,253,0.35);
}}
.stButton > button:active {{
    background-color: #0a4ebf !important;
    color: #ffffff !important;
}}
.stButton > button:disabled {{
    background-color: {C['border']} !important;
    color: {C['sub']} !important;
    box-shadow: none;
    cursor: not-allowed;
}}

/* ── File uploader ── */
[data-testid="stFileUploader"] {{
    background-color: {C['surface']} !important;
    border-radius: 8px;
}}
[data-testid="stFileUploaderDropzone"] {{
    background-color: {C['surface']} !important;
    border: 2px dashed {C['border']} !important;
    border-radius: 8px;
}}

/* ── Tabs ── */
[data-testid="stTabs"] button {{
    color: {C['text']} !important;
    background: transparent !important;
}}
[data-testid="stTabs"] button[aria-selected="true"] {{
    border-bottom: 2px solid {C['blue']} !important;
    color: {C['blue']} !important;
}}

/* ── Expander ── */
[data-testid="stExpander"] {{
    background-color: {C['surface']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 8px;
}}

/* ── Alerts ── */
.stAlert, .stInfo, .stWarning, .stError, .stSuccess {{
    color: {C['text']} !important;
}}
[data-testid="stAlertContainer"] * {{ color: {C['text']} !important; }}

/* ── DataFrame ── */
[data-testid="stDataFrame"] * {{
    color: {C['text']} !important;
    background-color: {C['bg']} !important;
}}

/* ── Code blocks ── */
.stCode, .stCodeBlock, code, pre {{
    background-color: #F1F3F5 !important;
    color: #212529 !important;
    border: 1px solid {C['border']};
    border-radius: 6px;
}}

/* ── Divider ── */
hr {{ border-color: {C['border']}; margin: 1rem 0; }}

/* ── Custom cards & layout ── */
.card {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
    color: {C['text']} !important;
}}
.card * {{ color: {C['text']} !important; }}
.sec {{
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: {C['sub']};
    margin-bottom: 14px;
    padding-bottom: 6px;
    border-bottom: 2px solid {C['border']};
}}
.label {{
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .07em;
    color: {C['text']};
    margin-bottom: 3px;
}}
.label-tag {{
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .07em;
    color: {C['text']};
    margin-bottom: 3px;
}}

/* ── Badges ── */
.badge-green {{
    background: #D1E7DD;
    color: #0F5132 !important;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    padding: 2px 10px;
}}
.badge-blue {{
    background: #CFE2FF;
    color: #084298 !important;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
}}
.badge-amber {{
    background: #FFF3CD;
    color: #664D03 !important;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
}}

/* ── Content blocks ── */
.summary-box {{
    background: {C['surface']};
    border-left: 4px solid {C['blue']};
    border-radius: 0 8px 8px 0;
    padding: 18px 22px;
    line-height: 1.75;
    font-size: 14px;
    color: {C['text']} !important;
}}
.step-log {{
    background: #F1F3F5;
    border: 1px solid {C['border']};
    border-radius: 6px;
    padding: 10px 14px;
    font-family: monospace;
    font-size: 12px;
    color: {C['text']} !important;
    line-height: 1.6;
    min-height: 60px;
}}
.upload-info {{
    background: #E8F4FD;
    border: 1px solid #B8D9F5;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 12px;
    color: #084298 !important;
    margin-top: 4px;
}}
.upload-zone {{
    background: {C['surface']};
    border: 2px dashed {C['border']};
    border-radius: 12px;
    padding: 24px;
    text-align: center;
}}

/* ── Progress status box ── */
.status-box {{
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #1E40AF !important;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.status-box-warn {{
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #92400E !important;
    margin-bottom: 8px;
}}
.status-box-ok {{
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #14532D !important;
    margin-bottom: 8px;
}}

/* ── Spinner override ── */
[data-testid="stSpinner"] > div {{
    border-top-color: {C['blue']} !important;
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

# =============================================================================
# KPI CATALOGUE  (unchanged)
# =============================================================================

KPI_GROUPS: dict[str, dict] = {
    "scope_1_emissions": {
        "label": "Scope 1 GHG", "group": "Environmental", "unit": "tCO2e",
        "ratio_unit": "tCO2e/Cr", "ratio_denominator": "revenue",
        "max_ratio": 10, "higher_is_better": False,
        "desc": "Direct GHG emissions per INR Crore revenue",
    },
    "scope_2_emissions": {
        "label": "Scope 2 GHG", "group": "Environmental", "unit": "tCO2e",
        "ratio_unit": "tCO2e/Cr", "ratio_denominator": "revenue",
        "max_ratio": 10, "higher_is_better": False,
        "desc": "Indirect GHG emissions per INR Crore revenue",
    },
    "scope_3_emissions": {
        "label": "Scope 3 GHG", "group": "Environmental", "unit": "tCO2e",
        "ratio_unit": "tCO2e/Cr", "ratio_denominator": "revenue",
        "max_ratio": 50, "higher_is_better": False,
        "desc": "Value chain GHG emissions per INR Crore revenue",
    },
    "energy_consumption": {
        "label": "Energy Intensity", "group": "Environmental", "unit": "GJ",
        "ratio_unit": "GJ/Cr", "ratio_denominator": "revenue",
        "max_ratio": 1_000, "higher_is_better": False,
        "desc": "Total energy consumed per INR Crore revenue",
    },
    "water_consumption": {
        "label": "Water Intensity", "group": "Environmental", "unit": "KL",
        "ratio_unit": "KL/Cr", "ratio_denominator": "revenue",
        "max_ratio": 500, "higher_is_better": False,
        "desc": "Total water consumed per INR Crore revenue",
    },
    "waste_generated": {
        "label": "Waste Intensity", "group": "Environmental", "unit": "MT",
        "ratio_unit": "MT/Cr", "ratio_denominator": "revenue",
        "max_ratio": 5, "higher_is_better": False,
        "desc": "Waste generated per INR Crore revenue",
    },
    "renewable_energy_percentage": {
        "label": "Renewable Energy", "group": "Environmental", "unit": "%",
        "ratio_unit": "%", "ratio_denominator": "none",
        "max_ratio": 100, "higher_is_better": True,
        "desc": "Share of energy from renewable sources",
    },
    "employee_count": {
        "label": "Workforce", "group": "Social", "unit": "count",
        "ratio_unit": "employees/Cr", "ratio_denominator": "revenue",
        "max_ratio": 5_000, "higher_is_better": False,
        "desc": "Total employees per INR Crore revenue",
    },
    "women_in_workforce_percentage": {
        "label": "Women in Workforce", "group": "Social", "unit": "%",
        "ratio_unit": "%", "ratio_denominator": "none",
        "max_ratio": 100, "higher_is_better": True,
        "desc": "Percentage of women in workforce",
    },
    "complaints_filed": {
        "label": "Complaints Filed", "group": "Governance", "unit": "count",
        "ratio_unit": "count", "ratio_denominator": "none",
        "max_ratio": 1_000_000, "higher_is_better": False,
        "desc": "Total complaints filed during the year",
    },
    "complaints_pending": {
        "label": "Complaints Pending", "group": "Governance", "unit": "count",
        "ratio_unit": "count", "ratio_denominator": "none",
        "max_ratio": 100_000, "higher_is_better": False,
        "desc": "Complaints pending resolution at year end",
    },
}

ALL_KPI_NAMES          = list(KPI_GROUPS.keys())
EXTRACTABLE_KPI_NAMES  = ALL_KPI_NAMES
KPI_META               = KPI_GROUPS   # legacy alias

_KPI_PLAUSIBILITY: dict[str, tuple[float, float]] = {
    "scope_1_emissions":             (1,      5_000_000),
    "scope_2_emissions":             (1,      5_000_000),
    "scope_3_emissions":             (1,     10_000_000),
    "energy_consumption":            (100,  500_000_000),
    "water_consumption":             (100,  100_000_000),
    "waste_generated":               (0.1,      500_000),
    "renewable_energy_percentage":   (0,             100),
    "employee_count":                (1,       5_000_000),
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
# RATIO COMPUTATION  (unchanged)
# =============================================================================

def compute_ratio(kpi_name: str, value: float, revenue_cr: float) -> float:
    meta = KPI_GROUPS.get(kpi_name, {})
    if meta.get("ratio_denominator") == "none":
        return value
    if revenue_cr > 0:
        return value / revenue_cr
    return value


def format_ratio(value: float, kpi_name: str) -> str:
    if value == 0:
        return "0"
    if abs(value) < 0.001:
        return f"{value:.4e}"
    if abs(value) < 1:
        return f"{value:.4f}"
    return f"{value:,.2f}"


# =============================================================================
# DATA CONTAINERS  (unchanged)
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


# =============================================================================
# KPI-LEVEL CACHE LAYER  (unchanged)
# =============================================================================

def _cache_load(company_id: uuid.UUID, fy: int) -> dict:
    try:
        from core.database import get_db
        from services.kpi_cache_service import KPICacheService

        cache_svc = KPICacheService()
        with get_db() as db:
            cached = cache_svc.select_best_per_kpi(
                company_id=company_id,
                fy=fy,
                kpi_names=EXTRACTABLE_KPI_NAMES,
                db=db,
            )
        return cached
    except Exception:
        return {}


def _cache_load_revenue(company_id: uuid.UUID, fy: int):
    try:
        from core.database import get_db
        from services.kpi_cache_service import KPICacheService

        cache_svc = KPICacheService()
        with get_db() as db:
            return cache_svc.load_revenue(company_id=company_id, fy=fy, db=db)
    except Exception:
        return None


# =============================================================================
# EXTRACTION (CACHE-AWARE, v4) — unchanged backend logic
# =============================================================================

def _step_extract_missing(
    report_id: uuid.UUID,
    report_type: str,
    fy: int,
    log: list[str],
    llm_service,
    missing_kpi_names: list[str],
    need_revenue: bool,
) -> dict:
    from agents.extraction_agent import ExtractionAgent
    from services.revenue_extractor import extract_revenue
    from core.database import get_db

    new_kpis: dict = {}
    new_revenue     = None

    if not missing_kpi_names and not need_revenue:
        log.append(f"  [{report_type}] All KPIs cached — skipping extraction.")
        return {"kpis": new_kpis, "revenue": new_revenue}

    prefix = f"  [{report_type}] (id={str(report_id)[:8]})"

    if missing_kpi_names:
        log.append(f"{prefix} Extracting {len(missing_kpi_names)} missing KPI(s): "
                   f"{missing_kpi_names}")
        try:
            with get_db() as db:
                extracted_list = ExtractionAgent().extract_all(
                    report_id=report_id,
                    db=db,
                    kpi_names=missing_kpi_names,
                )
            for ext in extracted_list:
                if ext.normalized_value is None:
                    continue
                val  = ext.normalized_value
                unit = ext.unit or ""
                lo, hi = _KPI_PLAUSIBILITY.get(ext.kpi_name, (0, float("inf")))
                if not (lo <= val <= hi):
                    log.append(f"    {ext.kpi_name}: {val:,.2f} {unit} "
                               f"out of range — dropped")
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
    else:
        log.append(f"{prefix} No missing KPIs — skipping ExtractionAgent.")

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
                )
                if new_revenue:
                    log.append(
                        f"    Revenue: INR {new_revenue.value_cr:,.0f} Cr "
                        f"[{new_revenue.pattern_name} conf={new_revenue.confidence:.2f}]"
                    )
            except Exception as exc:
                log.append(f"    Revenue extraction failed: {exc}")

    return {"kpis": new_kpis, "revenue": new_revenue}


# =============================================================================
# DB helpers (unchanged)
# =============================================================================

def _db_load_kpis_and_revenue(company_id: uuid.UUID, fy: int) -> dict:
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
        from services.kpi_cache_service import KPICacheService

        with get_db() as db:
            KPICacheService().store(
                company_id=company_id,
                report_id=report_id,
                fy=fy,
                kpi_records=kpi_records,
                revenue_result=revenue_result,
                db=db,
            )
    except Exception as exc:
        st.warning(f"DB store failed for report {str(report_id)[:8]}: {exc}")


# =============================================================================
# PIPELINE STEPS  (unchanged backend, added status updates)
# =============================================================================

def _step_ingest(
    company_name: str,
    fy: int,
    sector: str,
    log: list[str],
    status_placeholder=None,
) -> dict:
    from agents.ingestion_agent import IngestionAgent
    from models.schemas import CompanyCreate

    def _update(msg: str) -> None:
        log.append(msg)
        if status_placeholder:
            status_placeholder.markdown(
                f'<div class="status-box">🔍 {msg}</div>',
                unsafe_allow_html=True,
            )

    agent = IngestionAgent()
    company_data = CompanyCreate(name=company_name, sector=sector, country="India")
    _update(f"Searching for {company_name} FY{fy} reports...")

    try:
        result = agent.run_multi_report_types(company_data=company_data, year=fy,
                                              auto_download=True)
    except Exception as exc:
        log.append(f"  Search failed: {exc}")
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


def _step_parse(
    report_id: uuid.UUID,
    report_type: str,
    log: list[str],
    status_placeholder=None,
) -> bool:
    from services.parse_orchestrator import ParseOrchestrator

    if status_placeholder:
        status_placeholder.markdown(
            f'<div class="status-box">📄 Reading {report_type} report...</div>',
            unsafe_allow_html=True,
        )

    log.append(f"  Reading [{report_type}] report (id={str(report_id)[:8]})...")
    try:
        result = ParseOrchestrator().run(report_id=report_id, force=False)
        log.append(f"    {result.page_count} pages, "
                   f"{result.meta.get('chunk_count', '?')} sections found.")
        return True
    except Exception as exc:
        log.append(f"    Could not read report: {exc}")
        return False


def _derive_total_ghg(kpi_records: dict) -> Optional[dict]:
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
# MAIN PIPELINE  (v4 — with improved status feedback)
# =============================================================================

def run_company_pipeline(
    company_name: str,
    fy: int,
    sector: str,
    db_online: bool,
    llm_service,
    status_placeholder,
    uploaded_file=None,
    upload_report_type: str = "BRSR",
) -> CompanyData:
    log: list[str] = []

    def _set_status(msg: str, kind: str = "info") -> None:
        """Update the status box. kind: info | warn | ok"""
        css_class = {
            "info": "status-box",
            "warn": "status-box-warn",
            "ok":   "status-box-ok",
        }.get(kind, "status-box")
        icon = {"info": "🔍", "warn": "⚠️", "ok": "✅"}.get(kind, "🔍")
        status_placeholder.markdown(
            f'<div class="{css_class}">{icon} {msg}</div>',
            unsafe_allow_html=True,
        )

    # ── Uploaded PDF path ─────────────────────────────────────────────────────
    if uploaded_file is not None:
        _set_status("Processing uploaded report...")
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
            _set_status("Report processed successfully.", "ok")
            return CompanyData(
                company_name=company_name, fy=fy, sector=sector,
                kpi_records=merged, revenue_result=rev, log=log,
                company_id=cid,
                report_infos=upload_result.get("report_infos", []),
                file_path=(upload_result.get("report_infos") or [{}])[0].file_path
                           if upload_result.get("report_infos") else None,
            )

    if not db_online:
        _set_status("Database is offline. Please check your connection.", "warn")
        return CompanyData(company_name=company_name, fy=fy, sector=sector,
                           kpi_records={}, revenue_result=None, log=log)

    # ── Step 1: Resolve company + reports ────────────────────────────────────
    _set_status(f"Searching for {company_name} FY{fy} reports...")
    db_data    = _db_get_all_reports(company_name, fy)
    company_id = db_data.get("company_id")
    report_infos: list[ReportInfo] = []

    if db_data["exists"]:
        report_infos = db_data["reports"]
    else:
        ingest     = _step_ingest(company_name, fy, sector, log, status_placeholder)
        company_id = ingest.get("company_id") or company_id
        report_infos = ingest.get("reports", [])
        if not report_infos:
            _set_status(
                f"No report found for {company_name} FY{fy}. "
                "Try uploading the PDF directly using the Upload tab.",
                "warn",
            )
            return CompanyData(company_name=company_name, fy=fy, sector=sector,
                               kpi_records={}, revenue_result=None, log=log,
                               company_id=company_id)
        db_data    = _db_get_all_reports(company_name, fy)
        company_id = db_data.get("company_id") or company_id
        if db_data["exists"]:
            report_infos = db_data["reports"]

    # ── Step 2: KPI-LEVEL CACHE CHECK ─────────────────────────────────────────
    _set_status("Checking for previously extracted data...")
    cached_kpis: dict = {}
    cached_revenue    = None

    if company_id:
        cached_kpis    = _cache_load(company_id, fy)
        cached_revenue = _cache_load_revenue(company_id, fy)

    missing_kpis   = [k for k in EXTRACTABLE_KPI_NAMES if k not in cached_kpis]
    need_revenue   = cached_revenue is None

    if not missing_kpis and not need_revenue:
        _set_status(f"Data loaded from cache for {company_name} FY{fy}.", "ok")
        final_db = _db_load_kpis_and_revenue(company_id, fy) if company_id else {}
        return CompanyData(
            company_name=company_name, fy=fy, sector=sector,
            kpi_records=cached_kpis,
            revenue_result=cached_revenue,
            log=log,
            company_id=company_id,
            report_infos=report_infos,
            file_path=final_db.get("file_path"),
        )

    # ── Step 3: Multi-report extraction (missing KPIs only) ──────────────────
    sorted_reports  = sorted(report_infos,
                             key=lambda r: _REPORT_TYPE_PRIORITY.get(r.report_type, 99))
    still_missing   = list(missing_kpis)
    still_need_rev  = need_revenue
    all_new_kpis:   dict = {}
    final_revenue         = cached_revenue

    for ri in sorted_reports:
        if not still_missing and not still_need_rev:
            break

        _set_status(f"Reading {ri.report_type} report...")
        parse_ok = _step_parse(ri.id, ri.report_type, log, status_placeholder=None)
        if not parse_ok:
            continue

        _set_status(f"Extracting ESG metrics from {ri.report_type} report...")
        extract_result = _step_extract_missing(
            report_id=ri.id,
            report_type=ri.report_type,
            fy=fy,
            log=log,
            llm_service=llm_service,
            missing_kpi_names=list(still_missing),
            need_revenue=still_need_rev,
        )
        new_kpis    = extract_result["kpis"]
        new_revenue = extract_result["revenue"]

        all_new_kpis.update(new_kpis)

        for found in list(new_kpis.keys()):
            if found in still_missing:
                still_missing.remove(found)

        if new_revenue and final_revenue is None:
            final_revenue  = new_revenue
            still_need_rev = False

        if company_id and (new_kpis or new_revenue):
            _db_store_kpis(company_id, ri.id, fy, new_kpis, new_revenue)

    # ── Step 4: Merge cached + newly extracted ────────────────────────────────
    merged_kpis = {**all_new_kpis, **cached_kpis}

    # ── Step 5: Final DB read ─────────────────────────────────────────────────
    final_db      = _db_load_kpis_and_revenue(company_id, fy) if company_id else {}
    final_kpis    = final_db.get("kpis", {})
    final_rev_db  = final_db.get("revenue")
    final_fp      = final_db.get("file_path")

    final_merged = {**merged_kpis, **final_kpis}
    final_revenue = final_revenue or final_rev_db

    if final_merged:
        found_count = len(final_merged)
        _set_status(f"Found {found_count} ESG metric(s) for {company_name} FY{fy}.", "ok")
    else:
        _set_status(
            f"No ESG data found for {company_name} FY{fy}. "
            "Try uploading the report PDF directly.",
            "warn",
        )

    return CompanyData(
        company_name=company_name, fy=fy, sector=sector,
        kpi_records=final_merged, revenue_result=final_revenue, log=log,
        company_id=company_id, report_infos=report_infos, file_path=final_fp,
    )


# =============================================================================
# UPLOAD PIPELINE  (cache-aware, with status feedback)
# =============================================================================

def run_upload_pipeline(
    uploaded_file, company_name: str, fy: int, sector: str,
    report_type: str, db_online: bool, llm_service, status_placeholder,
) -> dict:
    from agents.ingestion_agent import IngestionAgent
    log: list[str] = []

    def _update(msg: str, kind: str = "info") -> None:
        log.append(msg)
        if status_placeholder:
            css_class = {
                "info": "status-box",
                "warn": "status-box-warn",
                "ok":   "status-box-ok",
            }.get(kind, "status-box")
            icon = {"info": "📄", "warn": "⚠️", "ok": "✅"}.get(kind, "📄")
            status_placeholder.markdown(
                f'<div class="{css_class}">{icon} {msg}</div>',
                unsafe_allow_html=True,
            )

    _update("Saving uploaded report...")

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False,
                                        prefix="esg_upload_") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = Path(tmp.name)
    except Exception as exc:
        _update(f"Could not save uploaded file: {exc}", "warn")
        return {"success": False, "log": log}

    try:
        result = IngestionAgent().ingest_uploaded_pdf(
            source_path=tmp_path, company_name=company_name,
            year=fy, sector=sector, report_type=report_type,
        )
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        _update(f"Upload failed: {exc}", "warn")
        return {"success": False, "log": log}
    finally:
        tmp_path.unlink(missing_ok=True)

    company    = result["company"]
    report     = result["report"]
    is_dup     = result.get("is_duplicate", False)
    company_id = company.id
    report_id  = report.id

    ri = ReportInfo(id=report_id, report_type=report_type,
                    file_path=report.file_path, status=report.status)

    # ── Cache check ───────────────────────────────────────────────────────────
    _update("Checking for previously extracted data...")
    cached_kpis    = _cache_load(company_id, fy) if db_online else {}
    cached_revenue = _cache_load_revenue(company_id, fy) if db_online else None

    missing_kpi_names = [k for k in EXTRACTABLE_KPI_NAMES if k not in cached_kpis]
    need_revenue      = cached_revenue is None

    if not missing_kpi_names and not need_revenue:
        _update(f"All data loaded from cache ({len(cached_kpis)} metrics).", "ok")
        return {
            "success": True, "kpi_records": cached_kpis,
            "revenue": cached_revenue, "company_id": company_id,
            "report_id": report_id, "report_infos": [ri], "log": log,
        }

    # Parse
    _update("Reading report pages...")
    parsed_ok = _step_parse(report_id, report_type, log, status_placeholder=None)

    # Extract only missing KPIs
    _update(f"Extracting ESG metrics ({len(missing_kpi_names)} to find)...")
    extract_result = _step_extract_missing(
        report_id=report_id,
        report_type=report_type,
        fy=fy,
        log=log,
        llm_service=llm_service,
        missing_kpi_names=missing_kpi_names,
        need_revenue=need_revenue,
    )
    new_kpis    = extract_result["kpis"]
    new_revenue = extract_result["revenue"]

    merged_kpis   = {**new_kpis, **cached_kpis}
    final_revenue = cached_revenue or new_revenue

    if db_online and (new_kpis or new_revenue):
        _db_store_kpis(company_id, report_id, fy, new_kpis, new_revenue)
        _update(f"Saved {len(new_kpis)} metric(s) to database.")

    if merged_kpis:
        _update(f"Done — {len(merged_kpis)} ESG metric(s) extracted.", "ok")
    else:
        _update(
            "No metrics could be extracted from this report. "
            "The PDF may be image-only or use a non-standard format.",
            "warn",
        )

    return {
        "success": True, "kpi_records": merged_kpis, "revenue": final_revenue,
        "company_id": company_id, "report_id": report_id,
        "report_infos": [ri], "log": log,
    }


# =============================================================================
# BENCHMARK BUILDER  (unchanged)
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
        meta    = KPI_GROUPS.get(comp.kpi_name, {})
        ceiling = meta.get("max_ratio")
        if all(
            (not ceiling or v <= ceiling) and v >= 0
            for _, v, _ in comp.entries
        ):
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
        vals   = {lbl: v for lbl, v, _ in comp.entries}
        va, vb = vals.get(la), vals.get(lb)
        if va is None or vb is None:
            continue
        meta = KPI_GROUPS.get(comp.kpi_name, {})
        label = meta.get("label", comp.display_name)
        cats.append(label)
        total = va + vb
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
        xaxis=dict(showgrid=True, gridcolor=C["grid"], zeroline=False,
                   showticklabels=False),
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
            "Percentage KPIs (women %, renewable %): shown as-is. ",
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

    story += [tbl, Spacer(1, 16), Paragraph("Summary", s_h2)]
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

    # DB / LLM status indicators
    db_col  = C["green"] if db_online else C["red"]
    llm_col = C["green"] if llm_service else C["amber"]
    st.markdown(
        f"""<div style="display:flex;gap:8px;margin-bottom:12px">
          <span style="font-size:11px;padding:2px 8px;border-radius:12px;
                       background:{'#D1E7DD' if db_online else '#F8D7DA'};
                       color:{db_col};font-weight:600">
            {'● DB Online' if db_online else '● DB Offline'}
          </span>
          <span style="font-size:11px;padding:2px 8px;border-radius:12px;
                       background:{'#D1E7DD' if llm_service else '#FFF3CD'};
                       color:{llm_col};font-weight:600">
            {'● AI Ready' if llm_service else '● No AI Key'}
          </span>
        </div>""",
        unsafe_allow_html=True,
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
        st.number_input("FY", min_value=2015, max_value=2030,
                        value=2025 if ckey == "c1_name" else 2024,
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

    ready       = bool(company1 and company2)
    compare_btn = st.button("Compare", disabled=not ready, use_container_width=True)
    st.markdown("---")

    if not ready:
        st.markdown(
            f'<div style="font-size:11px;color:{C["sub"]};text-align:center">'
            'Enter both company names to compare</div>',
            unsafe_allow_html=True,
        )


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
          <div style="font-size:14px;color:{C['sub']};margin-bottom:16px;max-width:480px">
              Enter two company names in the sidebar and click Compare to benchmark
              their ESG performance across Environmental, Social, and Governance metrics.
          </div>
          <div style="display:flex;gap:10px;justify-content:center;flex-wrap:wrap">
            <span class="badge-blue">Environmental</span>
            <span class="badge-green">Social</span>
            <span class="badge-amber">Governance</span>
          </div>
        </div>""", unsafe_allow_html=True)

    if compare_btn and ready:
        st.markdown(f"### Comparing {company1} FY{fy1} vs {company2} FY{fy2}")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown(
                f'<div style="font-size:13px;font-weight:600;color:{C["ca"]};'
                f'margin-bottom:6px">{company1} FY{fy1}</div>',
                unsafe_allow_html=True,
            )
            placeholder1 = st.empty()
        with col_s2:
            st.markdown(
                f'<div style="font-size:13px;font-weight:600;color:{C["cb"]};'
                f'margin-bottom:6px">{company2} FY{fy2}</div>',
                unsafe_allow_html=True,
            )
            placeholder2 = st.empty()

        data1 = data2 = None
        pipeline_error = None

        # Show initial "searching" state in both columns
        placeholder1.markdown(
            '<div class="status-box">🔍 Starting search...</div>',
            unsafe_allow_html=True,
        )
        placeholder2.markdown(
            '<div class="status-box">⏳ Waiting...</div>',
            unsafe_allow_html=True,
        )

        try:
            data1 = run_company_pipeline(
                company_name=company1, fy=int(fy1), sector=sector,
                db_online=db_online, llm_service=llm_service,
                status_placeholder=placeholder1,
                uploaded_file=upload1, upload_report_type=rtype1 or "BRSR",
            )

            # Update col 2 to active state once col 1 is done
            placeholder2.markdown(
                '<div class="status-box">🔍 Starting search...</div>',
                unsafe_allow_html=True,
            )

            data2 = run_company_pipeline(
                company_name=company2, fy=int(fy2), sector=sector,
                db_online=db_online, llm_service=llm_service,
                status_placeholder=placeholder2,
                uploaded_file=upload2, upload_report_type=rtype2 or "BRSR",
            )

        except Exception as exc:
            pipeline_error = exc
            st.error(f"Something went wrong: {exc}")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())

        if pipeline_error is None and data1 is not None and data2 is not None:
            # Show appropriate final state in each placeholder
            if not data1.kpi_records:
                placeholder1.markdown(
                    f'<div class="status-box-warn">⚠️ No data found for {company1}. '
                    f'Try uploading the report PDF.</div>',
                    unsafe_allow_html=True,
                )
            if not data2.kpi_records:
                placeholder2.markdown(
                    f'<div class="status-box-warn">⚠️ No data found for {company2}. '
                    f'Try uploading the report PDF.</div>',
                    unsafe_allow_html=True,
                )

            if not data1.kpi_records and not data2.kpi_records:
                st.warning(
                    "No ESG data found for either company. "
                    "Please upload the report PDFs using the **Upload PDF** tab, "
                    "then return here to compare."
                )
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
            st.warning(
                "No comparable ESG metrics found between these two companies. "
                "This can happen when reports use very different formats, "
                "or when data for one company is incomplete."
            )
            st.stop()

        st.markdown("---")

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
                    <div style="display:flex;justify-content:space-between;
                                align-items:flex-start;margin-bottom:10px">
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

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec">Summary</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="summary-box">{summary.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )

        log1 = result.get("log1", [])
        log2 = result.get("log2", [])
        if log1 or log2:
            with st.expander("Processing details"):
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
    st.markdown('<div class="sec">Upload ESG / BRSR Report PDF</div>',
                unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:13px;color:#8B92A9;margin-bottom:20px'>"
        "Upload a PDF directly to extract and store ESG metrics. "
        "Processed reports are immediately available in the Comparison tab."
        "</div>",
        unsafe_allow_html=True,
    )

    up_col1, up_col2 = st.columns([2, 1])
    with up_col1:
        upload_company = st.text_input("Company name (required)",
                                       placeholder="e.g. Wipro", key="upload_company")
        upload_fy = st.number_input("Fiscal year (required)", min_value=2010,
                                    max_value=2030, value=2024, key="upload_fy")
    with up_col2:
        upload_sector      = st.selectbox("Sector", SECTORS, key="upload_sector")
        upload_report_type = st.selectbox("Report type", UPLOAD_REPORT_TYPE_OPTIONS,
                                          key="upload_report_type")

    uploaded_file = st.file_uploader("Select a PDF file", type=["pdf"],
                                     key="pdf_uploader")

    if uploaded_file is not None:
        size_mb = round(uploaded_file.size / (1024 * 1024), 2)
        if size_mb > 50:
            st.error(f"File is {size_mb} MB — maximum allowed size is 50 MB.")
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

    if not db_online:
        st.warning("Database is offline. Uploads require a database connection.")

    if upload_btn and upload_ready and db_online:
        st.markdown("---")
        st.markdown(f"### Processing: {uploaded_file.name}")
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
            st.success(f"✅ Report processed — {len(kpis)} metric(s) extracted.")
            if kpis:
                rows = []
                for k, r in kpis.items():
                    meta = KPI_GROUPS.get(k, {})
                    rows.append({
                        "Group":      meta.get("group", ""),
                        "Metric":     meta.get("label", k),
                        "Value":      f"{r['value']:,.2f}",
                        "Unit":       r["unit"],
                        "Method":     r["method"],
                        "Confidence": f"{r['confidence']:.0%}",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True,
                             hide_index=True)
            else:
                st.info(
                    "No metrics could be extracted automatically. "
                    "The PDF may be image-based or use an unusual format. "
                    "Try a different PDF or report type."
                )
            if rev:
                st.markdown(
                    f"**Revenue:** INR {rev.value_cr:,.0f} Crore "
                    f"[{rev.pattern_name}, confidence {rev.confidence:.0%}]"
                )
        else:
            st.error(
                "Upload failed. Please check that the file is a valid PDF "
                "and that the company name is correct."
            )

        with st.expander("Processing details"):
            for line in upload_result.get("log", []):
                st.code(line, language=None)