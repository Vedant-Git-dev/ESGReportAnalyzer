"""
ui.py  —  ESG Competitive Intelligence Dashboard (light theme, PDF upload)

Pipeline contract
-----------------
For each company slot the resolution order is:

  1. If a PDF is uploaded for that slot:
       a. Register + parse the uploaded PDF (temp ingest).
       b. Extract KPIs from the PDF.
       c. Merge with any existing DB records for the same company+FY
          (DB wins for KPIs that are already stored and pass plausibility;
           newly extracted values fill the gaps).

  2. If NO PDF is uploaded:
       a. Check DB for existing KPI records + revenue.
       b. If found → fast path (no network, no parsing).
       c. If not found → report an error; without a PDF there is nothing
          to extract from.

Same-company different-year comparison is fully supported — the two company
slots are completely independent (name + FY + optional PDF).

Run:
    streamlit run ui.py
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
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# DESIGN TOKENS  — white background, black text
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

html, body, [class*="css"] {{
    font-family: {C['font']};
    background-color: {C['bg']} !important;
    color: {C['text']} !important;
}}
.stApp {{ background-color: {C['bg']} !important; }}

/* Sidebar */
[data-testid="stSidebar"] {{
    background-color: {C['surface']} !important;
    border-right: 1px solid {C['border']};
}}
[data-testid="stSidebar"] * {{
    color: {C['text']} !important;
}}

/* All text elements */
p, span, div, label, h1, h2, h3, h4, h5, h6,
.stMarkdown, .stText, .stCaption {{
    color: {C['text']} !important;
}}

/* ── All baseweb inputs (text, number, select) ── */
[data-baseweb="input"],
[data-baseweb="input-container"],
[data-baseweb="base-input"] {{
    background-color: {C['bg']} !important;
    color: {C['text']} !important;
}}
[data-baseweb="input"] *,
[data-baseweb="input-container"] *,
[data-baseweb="base-input"] * {{
    background-color: {C['bg']} !important;
    color: {C['text']} !important;
}}

/* Text input */
.stTextInput > div > div > input {{
    background-color: {C['bg']} !important;
    color: {C['text']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 6px;
}}

/* Number input — the whole widget including stepper +/- buttons */
.stNumberInput {{
    background-color: {C['bg']} !important;
}}
.stNumberInput > div {{
    background-color: {C['bg']} !important;
}}
.stNumberInput input {{
    background-color: {C['bg']} !important;
    color: {C['text']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 6px 0 0 6px;
}}
.stNumberInput button {{
    background-color: {C['surface']} !important;
    color: {C['text']} !important;
    border: 1px solid {C['border']} !important;
}}
.stNumberInput button:hover {{
    background-color: {C['border']} !important;
    color: {C['text']} !important;
}}
.stNumberInput button svg {{
    fill: {C['text']} !important;
    stroke: {C['text']} !important;
}}
/* baseweb number spinner container */
[data-testid="stNumberInputContainer"] {{
    background-color: {C['bg']} !important;
}}
[data-testid="stNumberInputContainer"] * {{
    background-color: {C['bg']} !important;
    color: {C['text']} !important;
}}
[data-testid="stNumberInputContainer"] button {{
    background-color: {C['surface']} !important;
    color: {C['text']} !important;
    border: 1px solid {C['border']} !important;
}}

/* Selectbox */
.stSelectbox > div > div {{
    background-color: {C['bg']} !important;
    color: {C['text']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 6px;
}}
.stSelectbox [data-baseweb="select"],
.stSelectbox [data-baseweb="select"] * {{
    background-color: {C['bg']} !important;
    color: {C['text']} !important;
}}
/* Selectbox dropdown popover */
[data-baseweb="popover"] {{
    background-color: {C['bg']} !important;
}}
[data-baseweb="menu"] {{
    background-color: {C['bg']} !important;
}}
[data-baseweb="menu"] * {{
    background-color: {C['bg']} !important;
    color: {C['text']} !important;
}}
[data-baseweb="option"] {{
    background-color: {C['bg']} !important;
    color: {C['text']} !important;
}}
[data-baseweb="option"]:hover {{
    background-color: {C['surface']} !important;
    color: {C['text']} !important;
}}
li[role="option"] {{
    background-color: {C['bg']} !important;
    color: {C['text']} !important;
}}
li[role="option"]:hover {{
    background-color: {C['surface']} !important;
}}

/* ── Buttons ── */
.stButton > button {{
    background-color: {C['blue']};
    color: #ffffff !important;
    border: none;
    border-radius: 6px;
    padding: 0.5rem 1.2rem;
    font-weight: 600;
    font-size: 14px;
    width: 100%;
    transition: background 0.2s;
}}
.stButton > button:hover {{ background-color: #0b5ed7; color: #ffffff !important; }}
.stButton > button:disabled {{
    background-color: {C['border']} !important;
    color: {C['sub']} !important;
}}
.stButton > button p {{
    color: #ffffff !important;
}}
.stButton > button:disabled p {{
    color: {C['sub']} !important;
}}

/* ── File uploader ── */
[data-testid="stFileUploader"] {{
    background-color: {C['surface']} !important;
    border-radius: 8px;
}}
[data-testid="stFileUploader"] * {{
    color: {C['text']} !important;
}}
/* Dropzone area */
[data-testid="stFileUploaderDropzone"] {{
    background-color: {C['surface']} !important;
    border: 2px dashed {C['border']} !important;
    border-radius: 8px;
}}
[data-testid="stFileUploaderDropzone"] * {{
    color: {C['text']} !important;
}}
/* "Browse files" button inside file uploader */
[data-testid="stFileUploaderDropzoneButton"] {{
    background-color: {C['bg']} !important;
    color: {C['text']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 6px !important;
}}
[data-testid="stFileUploaderDropzoneButton"]:hover {{
    background-color: {C['surface']} !important;
    color: {C['text']} !important;
}}
[data-testid="stFileUploaderDropzoneButton"] p,
[data-testid="stFileUploaderDropzoneButton"] span {{
    color: {C['text']} !important;
}}

/* Tabs */
[data-testid="stTabs"] button {{
    color: {C['text']} !important;
    background: transparent !important;
}}
[data-testid="stTabs"] button[aria-selected="true"] {{
    border-bottom: 2px solid {C['blue']} !important;
    color: {C['blue']} !important;
}}

/* Expander */
[data-testid="stExpander"] {{
    background-color: {C['surface']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 8px;
}}
[data-testid="stExpander"] * {{
    color: {C['text']} !important;
}}

/* Alerts / info boxes */
.stAlert, .stInfo, .stWarning, .stError, .stSuccess {{
    color: {C['text']} !important;
}}
.stAlert p, .stInfo p, .stWarning p, .stError p, .stSuccess p {{
    color: {C['text']} !important;
}}
[data-testid="stAlertContainer"] {{
    color: {C['text']} !important;
}}
[data-testid="stAlertContainer"] * {{
    color: {C['text']} !important;
}}

/* Dataframe */
[data-testid="stDataFrame"] * {{
    color: {C['text']} !important;
    background-color: {C['bg']} !important;
}}

/* Code blocks (step log) */
.stCode, .stCodeBlock, code, pre {{
    background-color: #F1F3F5 !important;
    color: #212529 !important;
    border: 1px solid {C['border']};
    border-radius: 6px;
}}

/* Download button */
[data-testid="stDownloadButton"] button {{
    background-color: {C['surface']} !important;
    color: {C['text']} !important;
    border: 1px solid {C['border']} !important;
}}

/* Dividers */
hr {{ border-color: {C['border']}; margin: 1rem 0; }}

/* Custom component classes */
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

.label-tag {{
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .07em;
    color: {C['text']};
    margin-bottom: 3px;
}}

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

.summary-box {{
    background: {C['surface']};
    border-left: 4px solid {C['blue']};
    border-radius: 0 8px 8px 0;
    padding: 18px 22px;
    line-height: 1.75;
    font-size: 14px;
    color: {C['text']} !important;
}}
.summary-box * {{ color: {C['text']} !important; }}

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

KPI_META: dict[str, dict] = {
    "scope_1_emissions":   {"label": "Scope 1 GHG",     "unit": "tCO2e/Cr",
                             "max_ratio": 10,  "desc": "Direct GHG per ₹Crore revenue"},
    "scope_2_emissions":   {"label": "Scope 2 GHG",     "unit": "tCO2e/Cr",
                             "max_ratio": 10,  "desc": "Indirect GHG per ₹Crore revenue"},
    "total_ghg_emissions": {"label": "Total GHG",       "unit": "tCO2e/Cr",
                             "max_ratio": 20,  "desc": "Scope 1+2 per ₹Crore revenue"},
    "waste_generated":     {"label": "Waste Intensity", "unit": "MT/Cr",
                             "max_ratio": 5,   "desc": "Waste per ₹Crore revenue"},
}

EXTRACTABLE_KPI_NAMES = ["scope_1_emissions", "scope_2_emissions", "waste_generated"]
TARGET_KPI_NAMES      = list(KPI_META.keys())

_KPI_PLAUSIBILITY: dict[str, tuple[float, float]] = {
    "scope_1_emissions":   (1,        5_000_000),
    "scope_2_emissions":   (1,        5_000_000),
    "total_ghg_emissions": (1,       10_000_000),
    "waste_generated":     (0.1,        500_000),
}

_DEFAULT_REVENUE_CR = 315_322.0

# Temporary PDF files are stored here during the Streamlit session
_TEMP_DIR = Path(tempfile.gettempdir()) / "esg_ui_uploads"
_TEMP_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA CONTAINERS
# =============================================================================

@dataclass
class CompanyData:
    company_name:   str
    fy:             int
    sector:         str
    kpi_records:    dict
    revenue_result: object          # RevenueResult | None
    log:            list[str]
    company_id:     Optional[uuid.UUID] = None
    report_id:      Optional[uuid.UUID] = None
    file_path:      Optional[str]       = None
    pdf_source:     str = "db"          # "db" | "upload" | "db+upload"


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
            rows = db.query(Company.name).filter(Company.is_active == True).order_by(Company.name).all()
        return [r[0] for r in rows]
    except Exception:
        return []


def _db_lookup(company_name: str, fy: int) -> dict:
    """Load cached KPI records + revenue from DB. Returns plain Python dicts only."""
    empty = {"kpis": {}, "revenue": None, "company_id": None, "report_id": None, "file_path": None}
    try:
        from core.database import get_db
        from models.db_models import Company, Report, KPIRecord, KPIDefinition
        from services.revenue_extractor import RevenueResult

        with get_db() as db:
            company_row = (
                db.query(Company)
                .filter(Company.name.ilike(f"%{company_name}%"))
                .first()
            )
            if not company_row:
                return empty

            company_id = company_row.id

            report_row = (
                db.query(Report)
                .filter(Report.company_id == company_id, Report.report_year == fy)
                .order_by(Report.created_at.desc())
                .first()
            )
            report_id = report_row.id        if report_row else None
            file_path = report_row.file_path if report_row else None

            cached_rev = None
            if report_row and getattr(report_row, "revenue_cr", None) is not None:
                try:
                    cached_rev = RevenueResult(
                        value_cr=float(report_row.revenue_cr),
                        raw_value=str(report_row.revenue_cr),
                        raw_unit=getattr(report_row, "revenue_unit",   None) or "INR_Crore",
                        source  =getattr(report_row, "revenue_source", None) or "db",
                        page_number=0, confidence=0.99, pattern_name="cached",
                    )
                except Exception:
                    pass

            kpis: dict = {}
            if report_row:
                for kpi_name in EXTRACTABLE_KPI_NAMES:
                    kdef = db.query(KPIDefinition).filter(KPIDefinition.name == kpi_name).first()
                    if not kdef:
                        continue
                    rec = (
                        db.query(KPIRecord)
                        .filter(
                            KPIRecord.company_id == company_id,
                            KPIRecord.kpi_definition_id == kdef.id,
                            KPIRecord.report_year == fy,
                            KPIRecord.normalized_value.isnot(None),
                        )
                        .order_by(KPIRecord.extracted_at.desc())
                        .first()
                    )
                    if not rec:
                        continue
                    val  = rec.normalized_value
                    unit = rec.unit or kdef.expected_unit
                    lo, hi = _KPI_PLAUSIBILITY.get(kpi_name, (0, float("inf")))
                    if not (lo <= val <= hi):
                        continue
                    kpis[kpi_name] = {
                        "value": val, "unit": unit,
                        "method": rec.extraction_method,
                        "confidence": rec.confidence or 0.9,
                    }

        return {"kpis": kpis, "revenue": cached_rev,
                "company_id": company_id, "report_id": report_id, "file_path": file_path}

    except Exception as exc:
        st.warning(f"DB lookup failed for {company_name}: {exc}")
        return empty


def _db_ensure_schema() -> None:
    try:
        from core.database import get_db
        from services.revenue_extractor import ensure_revenue_columns
        with get_db() as db:
            ensure_revenue_columns(db)
    except Exception:
        pass


def _db_store_kpis(company_id, report_id, fy, kpi_records, revenue_result) -> None:
    try:
        from core.database import get_db
        from models.db_models import Report, KPIRecord, KPIDefinition
        from services.revenue_extractor import store_revenue

        with get_db() as db:
            if revenue_result:
                rpt = db.query(Report).filter(Report.id == report_id).first()
                if rpt and getattr(rpt, "revenue_cr", None) is None:
                    try:
                        store_revenue(rpt, revenue_result, db)
                    except Exception:
                        pass

            for kpi_name, rec in kpi_records.items():
                kdef = db.query(KPIDefinition).filter(KPIDefinition.name == kpi_name).first()
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
                    company_id=company_id, report_id=report_id,
                    kpi_definition_id=kdef.id, report_year=fy,
                    raw_value=str(rec["value"]), normalized_value=rec["value"],
                    unit=rec["unit"], extraction_method=rec["method"],
                    confidence=rec["confidence"],
                    is_validated=rec["confidence"] >= 0.85,
                    validation_notes="esg_dashboard",
                ))
    except Exception as exc:
        st.warning(f"DB store failed: {exc}")


# =============================================================================
# PDF UPLOAD HELPERS
# =============================================================================

def _save_upload_to_temp(uploaded_file, slot_label: str) -> Optional[Path]:
    """
    Write a Streamlit UploadedFile to a temp path and return the path.
    The file is kept until the session ends or the user uploads a new file.
    Returns None on error.
    """
    try:
        suffix = Path(uploaded_file.name).suffix or ".pdf"
        tmp_path = _TEMP_DIR / f"esg_{slot_label}_{uuid.uuid4().hex[:8]}{suffix}"
        tmp_path.write_bytes(uploaded_file.getbuffer())
        return tmp_path
    except Exception as exc:
        st.warning(f"Could not save uploaded file: {exc}")
        return None


def _ingest_uploaded_pdf(
    pdf_path: Path,
    company_name: str,
    fy: int,
    sector: str,
    report_type: str,
    log: list[str],
) -> dict:
    """
    Register uploaded PDF, parse it, extract KPIs + revenue.

    Returns {company_id, report_id, kpis, revenue, file_path} or empty dict on failure.
    """
    from agents.ingestion_agent import IngestionAgent

    log.append(f"Registering uploaded PDF: {pdf_path.name}")
    try:
        result = IngestionAgent().ingest_uploaded_pdf(
            source_path=pdf_path,
            company_name=company_name,
            year=fy,
            sector=sector,
            report_type=report_type,
        )
    except Exception as exc:
        log.append(f"  PDF ingestion failed: {exc}")
        return {}

    company   = result["company"]
    report    = result["report"]
    is_dup    = result.get("is_duplicate", False)
    company_id = company.id
    report_id  = report.id

    if is_dup:
        log.append(f"  Duplicate detected (SHA-256 match). Reusing report {str(report_id)[:8]}.")
    else:
        log.append(f"  Registered as {report_type} report (id={str(report_id)[:8]}).")

    # Parse
    log.append("Parsing uploaded PDF...")
    try:
        from services.parse_orchestrator import ParseOrchestrator
        parsed = ParseOrchestrator().run(report_id=report_id, force=False)
        log.append(f"  {parsed.page_count} pages, {parsed.meta.get('chunk_count','?')} chunks.")
    except Exception as exc:
        log.append(f"  Parsing failed: {exc}")
        return {}

    # Extract KPIs
    log.append("Extracting KPIs from uploaded PDF...")
    new_kpis: dict = {}
    try:
        from agents.extraction_agent import ExtractionAgent
        from core.database import get_db
        with get_db() as db:
            extracted_list = ExtractionAgent().extract_all(
                report_id=report_id, db=db, kpi_names=EXTRACTABLE_KPI_NAMES,
            )
        for ext in extracted_list:
            if ext.normalized_value is None:
                log.append(f"  {ext.kpi_name}: not found")
                continue
            val, unit = ext.normalized_value, ext.unit or ""
            lo, hi = _KPI_PLAUSIBILITY.get(ext.kpi_name, (0, float("inf")))
            if not (lo <= val <= hi):
                log.append(f"  {ext.kpi_name}: {val:,.2f} outside range — dropped")
                continue
            new_kpis[ext.kpi_name] = {
                "value": val, "unit": unit,
                "method": ext.extraction_method,
                "confidence": ext.confidence or 0.5,
            }
            log.append(f"  {ext.kpi_name}: {val:,.2f} {unit} [{ext.extraction_method} conf={ext.confidence:.2f}]")
    except Exception as exc:
        log.append(f"  KPI extraction failed: {exc}")

    # Revenue
    log.append("Extracting revenue from uploaded PDF...")
    new_revenue = None
    try:
        from services.revenue_extractor import extract_revenue
        new_revenue = extract_revenue(pdf_path=Path(report.file_path), fiscal_year_hint=fy)
        if new_revenue:
            log.append(f"  Revenue: ₹{new_revenue.value_cr:,.0f} Cr [{new_revenue.pattern_name}]")
        else:
            log.append("  Revenue: not found in PDF.")
    except Exception as exc:
        log.append(f"  Revenue extraction failed: {exc}")

    return {
        "company_id": company_id,
        "report_id":  report_id,
        "file_path":  report.file_path,
        "kpis":       new_kpis,
        "revenue":    new_revenue,
    }


# =============================================================================
# PIPELINE STEPS
# =============================================================================

def _step_search_and_ingest(
    company_name: str,
    fy: int,
    sector: str,
    log: list[str],
) -> tuple:
    """
    Run the full Tavily search + download pipeline for a company+FY.
    Returns (report_id, company_id, file_path) or (None, None, None) on failure.
    Called only when there is no DB data and no uploaded PDF.
    """
    from agents.ingestion_agent import IngestionAgent
    from models.schemas import CompanyCreate

    log.append(f"Searching for {company_name} FY{fy} reports via Tavily...")
    try:
        agent        = IngestionAgent()
        company_data = CompanyCreate(name=company_name, sector=sector, country="India")
        result       = agent.run_multi_report_types(
            company_data=company_data, year=fy, auto_download=True,
        )
    except Exception as exc:
        log.append(f"  Search failed: {exc}")
        return None, None, None

    company   = result.get("company")
    downloads = result.get("downloaded_reports", [])
    not_found = result.get("not_found_types", [])
    failed    = result.get("failed_types", [])

    for rtype in ["BRSR", "ESG", "Integrated"]:
        dl = next((d for d in downloads if d.report_type == rtype), None)
        if dl:
            size_mb = round((dl.file_size_bytes or 0) / 1e6, 1)
            log.append(f"  [{rtype}] downloaded ({size_mb} MB).")
        elif rtype in not_found:
            log.append(f"  [{rtype}] not found.")
        elif rtype in failed:
            log.append(f"  [{rtype}] download failed.")

    if not downloads:
        log.append("  No PDFs downloaded. Check the Tavily API key and company name.")
        return None, None, None

    # Prefer BRSR > ESG > Integrated
    type_priority = {"BRSR": 0, "ESG": 1, "Integrated": 2}
    with_file = [d for d in downloads if d.file_path and Path(d.file_path).exists()]
    if with_file:
        with_file.sort(key=lambda d: type_priority.get(d.report_type, 99))
        best = with_file[0]
    else:
        best = downloads[0]

    log.append(f"  Using {best.report_type} report (id={str(best.id)[:8]}).")
    return best.id, company.id if company else None, best.file_path


def _step_parse(report_id: uuid.UUID, log: list[str]) -> bool:
    from services.parse_orchestrator import ParseOrchestrator
    log.append("Parsing PDF (cache check)...")
    try:
        r = ParseOrchestrator().run(report_id=report_id, force=False)
        log.append(f"  {r.page_count} pages, {r.meta.get('chunk_count','?')} chunks.")
        return True
    except Exception as exc:
        log.append(f"  Parse failed: {exc}")
        return False


def _step_extract(report_id: uuid.UUID, fy: int, log: list[str], llm_service) -> dict:
    from agents.extraction_agent import ExtractionAgent
    from services.revenue_extractor import extract_revenue
    from core.database import get_db

    new_kpis: dict = {}
    log.append("Extracting KPIs (regex → LLM → validate)...")
    try:
        with get_db() as db:
            extracted_list = ExtractionAgent().extract_all(
                report_id=report_id, db=db, kpi_names=EXTRACTABLE_KPI_NAMES,
            )
        for ext in extracted_list:
            if ext.normalized_value is None:
                log.append(f"  {ext.kpi_name}: not found")
                continue
            val, unit = ext.normalized_value, ext.unit or ""
            lo, hi = _KPI_PLAUSIBILITY.get(ext.kpi_name, (0, float("inf")))
            if not (lo <= val <= hi):
                log.append(f"  {ext.kpi_name}: {val:,.2f} out of range — dropped")
                continue
            new_kpis[ext.kpi_name] = {
                "value": val, "unit": unit,
                "method": ext.extraction_method,
                "confidence": ext.confidence or 0.5,
            }
            log.append(f"  {ext.kpi_name}: {val:,.2f} {unit} [{ext.extraction_method} conf={ext.confidence:.2f}]")
    except Exception as exc:
        log.append(f"  Extraction failed: {exc}")

    log.append("Extracting revenue...")
    new_revenue = None
    try:
        from core.database import get_db as _gdb
        from models.db_models import Report
        with _gdb() as db:
            rpt = db.query(Report).filter(Report.id == report_id).first()
            pdf_str = rpt.file_path if rpt else None
    except Exception:
        pdf_str = None

    if pdf_str and Path(pdf_str).exists():
        try:
            new_revenue = extract_revenue(
                pdf_path=Path(pdf_str), fiscal_year_hint=fy, llm_service=llm_service
            )
            if new_revenue:
                log.append(f"  Revenue: ₹{new_revenue.value_cr:,.0f} Cr [{new_revenue.pattern_name}]")
            else:
                log.append("  Revenue: not found.")
        except Exception as exc:
            log.append(f"  Revenue extraction failed: {exc}")
    else:
        log.append("  Revenue: PDF unavailable.")

    return {"kpis": new_kpis, "revenue": new_revenue}


def _derive_total_ghg(kpi_records: dict) -> Optional[dict]:
    if "total_ghg_emissions" in kpi_records:
        return None
    s1 = kpi_records.get("scope_1_emissions")
    s2 = kpi_records.get("scope_2_emissions")
    if not s1 or not s2:
        return None
    try:
        from services.normalizer import normalize
        n1 = normalize("scope_1_emissions", float(s1["value"]), s1["unit"])
        n2 = normalize("scope_2_emissions", float(s2["value"]), s2["unit"])
        total = round(n1.normalized_value + n2.normalized_value, 2)
    except Exception:
        return None
    return {
        "value": total, "unit": "tCO2e", "method": "derived",
        "confidence": round(min(s1["confidence"], s2["confidence"]) * 0.99, 3),
    }


# =============================================================================
# MAIN PIPELINE — handles both DB-only and PDF-upload paths
# =============================================================================

def run_company_pipeline(
    company_name:    str,
    fy:              int,
    sector:          str,
    db_online:       bool,
    llm_service,
    status_placeholder,
    uploaded_file=None,         # Streamlit UploadedFile or None
    upload_report_type: str = "BRSR",
) -> CompanyData:
    """
    Resolve KPIs for one company slot.

    Resolution order
    ----------------
    1. If a PDF is uploaded:
         a. Ingest (register + parse + extract) the uploaded PDF.
         b. Load any existing DB records for the same company+FY.
         c. Merge: DB wins for KPIs already present; PDF fills gaps.
         d. Store newly extracted KPIs to DB (if DB is online).

    2. If no PDF is uploaded:
         a. Fast path if DB has all KPIs + revenue.
         b. If DB has a report with a parseable PDF, run parse+extract.
         c. If nothing is available, return empty CompanyData with an error.
    """
    log: list[str] = []

    def _upd(msg: str) -> None:
        log.append(msg)
        status_placeholder.markdown(
            "<div class='step-log'>" + "<br>".join(log[-14:]) + "</div>",
            unsafe_allow_html=True,
        )

    _upd(f"Starting: {company_name} FY{fy}")

    company_id: Optional[uuid.UUID] = None
    report_id:  Optional[uuid.UUID] = None
    file_path:  Optional[str]       = None
    pdf_source  = "db"

    # ── Step 1: DB lookup ─────────────────────────────────────────────────────
    db_kpis: dict  = {}
    db_rev         = None
    if db_online:
        _upd("Checking DB cache...")
        db_data    = _db_lookup(company_name, fy)
        db_kpis    = db_data["kpis"]
        db_rev     = db_data["revenue"]
        company_id = db_data["company_id"]
        report_id  = db_data["report_id"]
        file_path  = db_data["file_path"]
        if db_kpis:
            _upd(f"  DB: {len(db_kpis)} KPI(s) cached.")
        if db_rev:
            _upd(f"  DB: Revenue ₹{db_rev.value_cr:,.0f} Cr cached.")

    # ── Step 2: Handle uploaded PDF ───────────────────────────────────────────
    pdf_kpis: dict = {}
    pdf_rev        = None
    pdf_report_id  = None
    pdf_company_id = None
    pdf_file_path  = None

    if uploaded_file is not None:
        _upd(f"PDF uploaded: {uploaded_file.name} ({round(uploaded_file.size/1e6,1)} MB)")
        tmp_path = _save_upload_to_temp(uploaded_file, f"{company_name}_{fy}")
        if tmp_path is None:
            _upd("  Failed to save uploaded file.")
        else:
            upload_result = _ingest_uploaded_pdf(
                pdf_path=tmp_path,
                company_name=company_name,
                fy=fy,
                sector=sector,
                report_type=upload_report_type,
                log=log,
            )
            # Clean up temp file (permanent copy is in storage)
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

            if upload_result:
                pdf_kpis       = upload_result.get("kpis", {})
                pdf_rev        = upload_result.get("revenue")
                pdf_report_id  = upload_result.get("report_id")
                pdf_company_id = upload_result.get("company_id")
                pdf_file_path  = upload_result.get("file_path")
                _upd(f"  PDF extraction complete: {len(pdf_kpis)} KPI(s) found.")
                pdf_source = "upload"

    # ── Step 3: DB-only extraction if needed ──────────────────────────────────
    # Only run if no PDF was uploaded AND DB has a parseable report
    db_extracted_kpis: dict = {}
    db_extracted_rev       = None

    if uploaded_file is None:
        missing  = [k for k in EXTRACTABLE_KPI_NAMES if k not in db_kpis]
        need_rev = db_rev is None

        if (missing or need_rev) and db_online and report_id:
            _upd(f"  Missing from DB: {missing}. Running extraction...")
            if file_path and Path(file_path).exists():
                parse_ok = _step_parse(report_id, log)
                if parse_ok:
                    ext = _step_extract(report_id, fy, log, llm_service)
                    db_extracted_kpis = ext.get("kpis", {})
                    db_extracted_rev  = ext.get("revenue")
            else:
                _upd("  No PDF on disk — will try search pipeline.")

        # ── Search pipeline fallback ──────────────────────────────────────────
        # No DB data, no upload, no stored PDF → run full Tavily ingestion.
        if not db_kpis and not db_extracted_kpis:
            _upd(f"No local data found. Running search pipeline for {company_name} FY{fy}...")
            search_report_id, search_company_id, search_file_path = (
                _step_search_and_ingest(company_name, fy, sector, log)
            )
            if search_report_id:
                company_id = search_company_id
                report_id  = search_report_id
                file_path  = search_file_path
                parse_ok   = _step_parse(report_id, log)
                if parse_ok:
                    ext = _step_extract(report_id, fy, log, llm_service)
                    db_extracted_kpis = ext.get("kpis", {})
                    db_extracted_rev  = ext.get("revenue")
            else:
                _upd("Search pipeline found no reports. Cannot proceed.")
                return CompanyData(
                    company_name=company_name, fy=fy, sector=sector,
                    kpi_records={}, revenue_result=None, log=log,
                )

    # ── Step 4: Merge all sources ──────────────────────────────────────────────
    # Priority: DB cached (highest confidence) → PDF extracted → DB extracted
    # For each KPI: use DB if present; PDF fills gaps; DB-extracted fills remaining gaps.
    merged_kpis: dict = {}

    # DB cached (most trusted — already validated in previous runs)
    for k, v in db_kpis.items():
        merged_kpis[k] = {**v, "_src": "db"}

    # PDF extracted (new data from uploaded PDF)
    for k, v in pdf_kpis.items():
        if k not in merged_kpis:
            merged_kpis[k] = {**v, "_src": "pdf"}
        # If PDF has higher confidence than DB, prefer PDF
        elif v.get("confidence", 0) > merged_kpis[k].get("confidence", 0):
            merged_kpis[k] = {**v, "_src": "pdf"}

    # DB extracted (freshly extracted from DB's stored PDF)
    for k, v in db_extracted_kpis.items():
        if k not in merged_kpis:
            merged_kpis[k] = {**v, "_src": "db_extract"}

    # Strip internal _src key before storing/using
    kpi_records = {k: {kk: vv for kk, vv in v.items() if kk != "_src"}
                   for k, v in merged_kpis.items()}

    if pdf_kpis and db_kpis:
        pdf_source = "db+upload"
    elif pdf_kpis:
        pdf_source = "upload"

    # Resolve IDs: PDF takes precedence for report_id if it was just ingested
    if pdf_company_id:
        company_id = pdf_company_id
    if pdf_report_id:
        report_id = pdf_report_id
        file_path = pdf_file_path

    # Derive total_ghg
    ghg = _derive_total_ghg(kpi_records)
    if ghg:
        kpi_records["total_ghg_emissions"] = ghg
        s1v = kpi_records.get("scope_1_emissions", {}).get("value", 0)
        s2v = kpi_records.get("scope_2_emissions", {}).get("value", 0)
        _upd(f"  Derived total_ghg: {ghg['value']:,.2f} tCO2e ({s1v:,.0f}+{s2v:,.0f})")

    # Revenue: prefer PDF/DB-extracted over cached if available
    revenue = db_rev
    new_rev = pdf_rev or db_extracted_rev
    if new_rev and (revenue is None or new_rev.confidence > revenue.confidence):
        revenue = new_rev

    # ── Step 5: Store new extractions to DB ───────────────────────────────────
    new_kpis_to_store = {
        **{k: v for k, v in pdf_kpis.items()},
        **{k: v for k, v in db_extracted_kpis.items()},
    }
    if "total_ghg_emissions" not in db_kpis and "total_ghg_emissions" in kpi_records:
        new_kpis_to_store["total_ghg_emissions"] = kpi_records["total_ghg_emissions"]

    if db_online and company_id and report_id and new_kpis_to_store:
        _upd(f"Storing {len(new_kpis_to_store)} new KPI(s) to DB...")
        _db_store_kpis(company_id, report_id, fy, new_kpis_to_store, new_rev or db_extracted_rev)

    if not kpi_records:
        _upd("No KPIs found from any source.")
    else:
        _upd(f"Final: {len(kpi_records)} KPI(s) from source={pdf_source}.")

    if revenue:
        _upd(f"Revenue: ₹{revenue.value_cr:,.0f} Cr [{revenue.source}]")
    else:
        _upd(f"Revenue: not found — default ₹{_DEFAULT_REVENUE_CR:,.0f} Cr will be used.")

    _upd(f"Done: {company_name} FY{fy}.")
    return CompanyData(
        company_name=company_name, fy=fy, sector=sector,
        kpi_records=kpi_records, revenue_result=revenue,
        log=log, company_id=company_id, report_id=report_id,
        file_path=file_path, pdf_source=pdf_source,
    )


# =============================================================================
# BENCHMARK BUILDER
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
            kpi_records=data.kpi_records,
            revenue_cr=rev_cr, revenue_source=rev_src,
            company_name=data.company_name, fiscal_year=data.fy,
            page_texts=page_texts,
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
# CHART HELPERS
# =============================================================================

_CHART_FONT = dict(family="Inter, Arial, sans-serif", size=12, color=C["text"])


def _hex_rgba(h: str, a: float) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16)
    return f"rgba({r},{g},{b},{a})"


def _chart_layout(**kw) -> dict:
    return dict(
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        font=_CHART_FONT,
        **kw,
    )


def _radar_chart(filtered, la: str, lb: str):
    cats, sa, sb = [], [], []
    for comp in filtered:
        vals = {l: v for l, v, _ in comp.entries}
        va, vb = vals.get(la), vals.get(lb)
        if va is None or vb is None:
            continue
        cats.append(KPI_META.get(comp.kpi_name, {}).get("label", comp.display_name))
        total = va + vb
        sa.append(round(100*(1-va/total), 1) if total else 50)
        sb.append(round(100*(1-vb/total), 1) if total else 50)
    if len(cats) < 2:
        return None
    cats_c = cats + [cats[0]]; sa_c = sa + [sa[0]]; sb_c = sb + [sb[0]]
    fig = go.Figure()
    for name, scores, color in [(la.split(" FY")[0], sa_c, C["ca"]), (lb.split(" FY")[0], sb_c, C["cb"])]:
        fig.add_trace(go.Scatterpolar(
            r=scores, theta=cats_c, fill="toself", name=name,
            line=dict(color=color, width=2.5),
            fillcolor=_hex_rgba(color, 0.10),
            hovertemplate="<b>%{theta}</b><br>Score: %{r:.0f}<extra></extra>",
        ))
    fig.update_layout(
        polar=dict(
            bgcolor=C["surface"],
            radialaxis=dict(visible=True, range=[0,100],
                            tickfont=dict(size=9, color=C["sub"]),
                            gridcolor=C["grid"], linecolor=C["border"]),
            angularaxis=dict(tickfont=dict(size=11, color=C["text"]),
                             gridcolor=C["grid"], linecolor=C["border"]),
        ),
        height=360, margin=dict(l=40,r=40,t=40,b=20),
        legend=dict(orientation="h", y=-0.08, font=dict(size=12, color=C["text"])),
        **_chart_layout(),
    )
    return fig


def _donut_chart(filtered, la: str, lb: str):
    na, nb = la.split(" FY")[0], lb.split(" FY")[0]
    wa = sum(1 for c in filtered if c.winner == la)
    wb = sum(1 for c in filtered if c.winner == lb)
    fig = go.Figure(go.Pie(
        labels=[na, nb], values=[wa, wb], hole=0.62,
        marker=dict(colors=[C["ca"], C["cb"]], line=dict(color="#fff", width=3)),
        textinfo="label+percent", textfont=dict(size=12, color=C["text"]),
        hovertemplate="<b>%{label}</b><br>%{value} wins<extra></extra>",
    ))
    fig.update_layout(
        height=260, margin=dict(l=0,r=0,t=10,b=0), showlegend=False,
        annotations=[dict(text=f"{wa+wb}<br><span style='font-size:10px'>KPIs</span>",
                          x=0.5, y=0.5, font=dict(size=20, color=C["text"]), showarrow=False)],
        **_chart_layout(),
    )
    return fig


def _gap_bar_chart(filtered, la: str, lb: str):
    labels_list, va_list, vb_list = [], [], []
    for comp in filtered:
        vals = {l: v for l, v, _ in comp.entries}
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
            textfont=dict(size=10, color=C["text"]),
            hovertemplate="<b>%{y}</b><br>%{x:.4e}<extra></extra>",
        ))
    fig.update_layout(
        barmode="group", height=300,
        xaxis=dict(title="Intensity per ₹Crore", showgrid=True,
                   gridcolor=C["grid"], zeroline=False, color=C["text"],
                   title_font=dict(color=C["text"])),
        yaxis=dict(autorange="reversed", color=C["text"]),
        legend=dict(orientation="h", y=1.08, font=dict(color=C["text"])),
        margin=dict(l=160, r=60, t=20, b=40),
        **_chart_layout(),
    )
    return fig


def _mini_bar_chart(comp, la: str, lb: str):
    vals = {l: v for l, v, _ in comp.entries}
    va, vb = vals.get(la, 0), vals.get(lb, 0)
    meta = KPI_META.get(comp.kpi_name, {})
    na, nb = la.split(" FY")[0], lb.split(" FY")[0]
    fig = go.Figure()
    for name, val, color, pat in [(na, va, C["ca"], ""), (nb, vb, C["cb"], "/")]:
        fig.add_trace(go.Bar(
            name=name, x=[val], y=[meta.get("label","")], orientation="h",
            marker=dict(color=color, line=dict(color=color, width=1.5), pattern_shape=pat),
            text=[f"{val:.3g}"], textposition="outside",
            textfont=dict(size=11, color=C["text"]),
            hovertemplate=f"<b>{name}</b><br>{val:.4e} {meta.get('unit','')}<extra></extra>",
        ))
    fig.update_layout(
        barmode="group", height=110, showlegend=False,
        yaxis=dict(visible=False),
        xaxis=dict(showgrid=True, gridcolor=C["grid"], zeroline=False,
                   showticklabels=False),
        margin=dict(l=10, r=60, t=8, b=8),
        **_chart_layout(),
    )
    return fig


# =============================================================================
# PDF REPORT EXPORT
# =============================================================================

def _export_pdf_report(profiles, filtered, summary: str, sector: str) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, HRFlowable)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    BLUE  = colors.HexColor("#0D6EFD")
    GREEN = colors.HexColor("#198754")
    GRAY  = colors.HexColor("#6C757D")
    LIGHT = colors.HexColor("#F8F9FA")
    BDR   = colors.HexColor("#DEE2E6")
    BLK   = colors.HexColor("#212529")

    ss   = getSampleStyleSheet()
    s_t  = ParagraphStyle("t",  parent=ss["Title"],   fontSize=22, textColor=BLK,
                           spaceAfter=4, leading=28, fontName="Helvetica-Bold")
    s_s  = ParagraphStyle("s",  parent=ss["Normal"],  fontSize=11, textColor=GRAY,
                           spaceAfter=14, leading=16)
    s_h2 = ParagraphStyle("h2", parent=ss["Heading2"],fontSize=13, textColor=BLK,
                           spaceBefore=16, spaceAfter=8, fontName="Helvetica-Bold")
    s_b  = ParagraphStyle("b",  parent=ss["Normal"],  fontSize=10, textColor=BLK,
                           leading=16, spaceAfter=8)
    s_n  = ParagraphStyle("n",  parent=ss["Normal"],  fontSize=9,  textColor=GRAY,
                           leading=14)

    labels = [f"{p.company_name} FY{p.fiscal_year}" for p in profiles]
    story  = [
        Paragraph("ESG Competitive Intelligence Report", s_t),
        Paragraph(f"{labels[0]} vs {labels[1]}", s_s),
        Paragraph(f"Sector: {sector}", s_n),
        HRFlowable(width="100%", thickness=1, color=BDR, spaceAfter=14),
        Paragraph("Methodology", s_h2),
        Paragraph("All metrics are intensity ratios (KPI ÷ INR Crore revenue). "
                  "KPIs with implausible ratios are excluded.", s_b),
        Spacer(1, 8),
        Paragraph("KPI Intensity Comparison", s_h2),
    ]

    tdata = [["Metric", "Unit", labels[0], labels[1], "Gap", "Leader"]]
    for comp in filtered:
        meta = KPI_META.get(comp.kpi_name, {})
        vals = {l: v for l, v, _ in comp.entries}
        v0, v1 = vals.get(labels[0]), vals.get(labels[1])
        fmt = lambda v: f"{v:.2e}" if (v and v < 0.001) else (f"{v:.4f}" if v else "N/A")
        tdata.append([
            meta.get("label", comp.display_name), meta.get("unit", comp.unit),
            fmt(v0), fmt(v1), f"{comp.pct_gap:.1f}%", comp.winner.split(" FY")[0],
        ])

    tbl = Table(tdata, colWidths=[5.2*cm, 2.3*cm, 2.8*cm, 2.8*cm, 1.6*cm, 2.5*cm], repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), BLUE),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,0), 9),
        ("BOTTOMPADDING", (0,0), (-1,0), 8),
        ("TOPPADDING",    (0,0), (-1,0), 8),
        ("FONTSIZE",      (0,1), (-1,-1), 9),
        ("TOPPADDING",    (0,1), (-1,-1), 6),
        ("BOTTOMPADDING", (0,1), (-1,-1), 6),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, LIGHT]),
        ("GRID",          (0,0), (-1,-1), 0.5, BDR),
        ("ALIGN",         (2,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TEXTCOLOR",     (5,1), (5,-1), GREEN),
        ("FONTNAME",      (5,1), (5,-1), "Helvetica-Bold"),
    ]))

    story += [tbl, Spacer(1, 16), Paragraph("AI-Generated Narrative Summary", s_h2)]
    for para in summary.split("\n\n"):
        if para.strip():
            story.append(Paragraph(para.strip(), s_b))
    story += [
        Spacer(1, 12),
        HRFlowable(width="100%", thickness=0.5, color=BDR),
        Spacer(1, 6),
        Paragraph("Generated by ESG Competitive Intelligence Pipeline. "
                  "Data sourced from BRSR, ESG, and Integrated reports.", s_n),
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

UPLOAD_TYPES = ["BRSR", "ESG", "Integrated", "Annual", "Other"]


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    # Header
    st.markdown(
        f"""<div style="padding:0 0 12px">
          <div style="font-size:20px;font-weight:800;color:{C['text']}">
            🌿 ESG Intelligence
          </div>
          <div style="font-size:11px;color:{C['sub']};margin-top:2px">
            Competitive Benchmarking Pipeline
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Status indicators
    db_col = C["green"] if db_online else C["red"]
    st.markdown(
        f"<div style='font-size:11px;color:{db_col};font-weight:600'>"
        f"{'● DB connected' if db_online else '● DB offline'}"
        + (f" ({len(known_companies)} companies)" if db_online else "")
        + "</div>",
        unsafe_allow_html=True,
    )
    llm_col = C["green"] if llm_service else C["amber"]
    st.markdown(
        f"<div style='font-size:11px;color:{llm_col};font-weight:600'>"
        f"{'● LLM enabled' if llm_service else '⚠ LLM disabled (no API key)'}"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    sector = st.selectbox("Sector", SECTORS, key="sector")
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── Company 1 ─────────────────────────────────────────────────────────────
    st.markdown(
        f"<div class='label-tag' style='color:{C['text']}'>"
        f"Company / Dataset 1</div>",
        unsafe_allow_html=True,
    )
    company1 = st.text_input(
        "c1name", placeholder="e.g. Infosys",
        label_visibility="collapsed", key="c1_name",
    )
    fy1 = st.number_input(
        "FY1", min_value=2010, max_value=2030, value=2024,
        label_visibility="collapsed", key="c1_fy",
        help="Fiscal year end (e.g. 2024 for FY2023-24)",
    )
    upload1 = st.file_uploader(
        "Upload PDF (optional)", type=["pdf"], key="upload1",
        help="Upload a BRSR/ESG/Annual report PDF. If omitted, existing DB data is used.",
    )
    rtype1 = st.selectbox("Report type", UPLOAD_TYPES, key="rtype1",
                           label_visibility="collapsed") if upload1 else None

    if upload1:
        st.markdown(
            f"<div class='upload-info'>📄 {upload1.name} "
            f"({round(upload1.size/1e6,1)} MB)</div>",
            unsafe_allow_html=True,
        )
    elif db_online and company1:
        db1 = _db_lookup(company1, int(fy1))
        n1  = len(db1["kpis"])
        if n1 > 0:
            rev_hint = " · revenue cached" if db1["revenue"] else ""
            st.caption(f"✓ {n1} KPI(s) in DB{rev_hint}")
        else:
            st.caption("No DB data — will search & download automatically.")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Company 2 ─────────────────────────────────────────────────────────────
    st.markdown(
        f"<div class='label-tag' style='color:{C['cb']}'>"
        f"Company / Dataset 2</div>",
        unsafe_allow_html=True,
    )
    company2 = st.text_input(
        "c2name", placeholder="e.g. TCS (or same company, different FY)",
        label_visibility="collapsed", key="c2_name",
    )
    fy2 = st.number_input(
        "FY2", min_value=2010, max_value=2030, value=2023,
        label_visibility="collapsed", key="c2_fy",
        help="Fiscal year end for Company 2",
    )
    upload2 = st.file_uploader(
        "Upload PDF (optional)", type=["pdf"], key="upload2",
        help="Upload a PDF for company 2.",
    )
    rtype2 = st.selectbox("Report type ", UPLOAD_TYPES, key="rtype2",
                           label_visibility="collapsed") if upload2 else None

    if upload2:
        st.markdown(
            f"<div class='upload-info'>📄 {upload2.name} "
            f"({round(upload2.size/1e6,1)} MB)</div>",
            unsafe_allow_html=True,
        )
    elif db_online and company2:
        db2 = _db_lookup(company2, int(fy2))
        n2  = len(db2["kpis"])
        if n2 > 0:
            rev_hint2 = " · revenue cached" if db2["revenue"] else ""
            st.caption(f"✓ {n2} KPI(s) in DB{rev_hint2}")
        else:
            st.caption("No DB data — will search & download automatically.")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Same company different year note
    if (company1 and company2
            and company1.strip().lower() == company2.strip().lower()
            and int(fy1) != int(fy2)):
        st.markdown(
            f"<div style='font-size:11px;color:{C['amber']};font-weight:600'>"
            f"📅 Year-over-year comparison mode</div>",
            unsafe_allow_html=True,
        )

    # Validation: only need a company name — search pipeline runs if nothing else is available
    def _slot_ready(company, fy, upload, db_data_fn) -> tuple[bool, str]:
        if not company:
            return False, "Enter a company name."
        # PDF provided → always ready
        if upload is not None:
            return True, ""
        # DB has data → ready
        if db_online:
            d = db_data_fn()
            if d["kpis"] or d["report_id"]:
                return True, ""
        # No data locally but we have Tavily → search will run automatically
        # Allow the slot — the pipeline will search and download
        return True, ""

    r1_ok, r1_msg = _slot_ready(
        company1, int(fy1), upload1,
        lambda: _db_lookup(company1, int(fy1)) if company1 else {"kpis":{}, "report_id":None},
    )
    r2_ok, r2_msg = _slot_ready(
        company2, int(fy2), upload2,
        lambda: _db_lookup(company2, int(fy2)) if company2 else {"kpis":{}, "report_id":None},
    )

    # Reject only when BOTH name AND FY are identical with no PDFs (exact duplicate)
    same_slot = (
        company1 and company2
        and company1.strip().lower() == company2.strip().lower()
        and int(fy1) == int(fy2)
        and upload1 is None and upload2 is None
    )

    ready = r1_ok and r2_ok and not same_slot
    compare_btn = st.button("Compare", disabled=not ready, use_container_width=True)

    if not ready:
        hints = []
        if not company1: hints.append("Enter Company 1 name")
        elif not r1_ok:  hints.append(f"Slot 1: {r1_msg}")
        if not company2: hints.append("Enter Company 2 name")
        elif not r2_ok:  hints.append(f"Slot 2: {r2_msg}")
        if same_slot:    hints.append("Same company + same FY — change the year or upload different PDFs")
        for h in hints:
            st.caption(h)

    st.markdown("---")


# =============================================================================
# MAIN CONTENT
# =============================================================================

tab_compare, tab_help = st.tabs(["Comparison", "How it works"])

with tab_compare:

    # Landing state
    if "result" not in st.session_state and not compare_btn:
        st.markdown(f"""
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;padding:60px 40px;text-align:center">
            <div style="font-size:44px;margin-bottom:12px">🌿</div>
            <div style="font-size:26px;font-weight:800;color:{C['text']};
                        letter-spacing:-0.5px;margin-bottom:8px">
                ESG Competitive Intelligence
            </div>
            <div style="font-size:14px;color:{C['sub']};max-width:560px;
                        line-height:1.8;margin-bottom:24px">
                Compare ESG intensity metrics for any two companies or
                the same company across two fiscal years.<br><br>
                <strong>With PDF upload:</strong> upload BRSR / ESG / Annual report
                PDFs for instant extraction — no prior ingestion needed.<br>
                <strong>DB-first:</strong> if the company is already in the database,
                cached data is used automatically and the PDF fills any gaps.
            </div>
            <div style="display:flex;gap:10px;justify-content:center;flex-wrap:wrap">
                <span class="badge-blue">PDF Upload</span>
                <span class="badge-green">DB Cache Merge</span>
                <span class="badge-amber">Year-over-Year</span>
            </div>
        </div>""", unsafe_allow_html=True)

    # Pipeline trigger
    if compare_btn and ready:
        label1 = f"{company1} FY{fy1}"
        label2 = f"{company2} FY{fy2}"
        st.markdown(f"### {label1}  vs  {label2}")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown(
                f"<div style='font-size:13px;font-weight:600;color:{C['ca']}'>"
                f"{label1}</div>", unsafe_allow_html=True)
            ph1 = st.empty()
        with col_s2:
            st.markdown(
                f"<div style='font-size:13px;font-weight:600;color:{C['cb']}'>"
                f"{label2}</div>", unsafe_allow_html=True)
            ph2 = st.empty()

        pipeline_error = None
        data1 = data2 = None
        try:
            data1 = run_company_pipeline(
                company_name=company1, fy=int(fy1), sector=sector,
                db_online=db_online, llm_service=llm_service,
                status_placeholder=ph1,
                uploaded_file=upload1,
                upload_report_type=rtype1 or "BRSR",
            )
            data2 = run_company_pipeline(
                company_name=company2, fy=int(fy2), sector=sector,
                db_online=db_online, llm_service=llm_service,
                status_placeholder=ph2,
                uploaded_file=upload2,
                upload_report_type=rtype2 or "BRSR",
            )
        except Exception as exc:
            pipeline_error = exc
            st.error(f"Pipeline error: {exc}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())

        if pipeline_error is None and data1 and data2:
            if not data1.kpi_records and not data2.kpi_records:
                st.error(
                    "No KPIs found for either company. "
                    "Upload PDFs or ensure data exists in the database."
                )
                st.stop()

            result = _build_benchmark(data1, data2, sector)
            result.update({"log1": data1.log, "log2": data2.log,
                           "src1": data1.pdf_source, "src2": data2.pdf_source,
                           "sector": sector})

            st.session_state.update({
                "result": result, "company1": company1, "company2": company2,
                "fy1": fy1, "fy2": fy2,
            })

    # ── Results ────────────────────────────────────────────────────────────────
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
        src1     = result.get("src1", "db")
        src2     = result.get("src2", "db")

        label_a = f"{profiles[0].company_name} FY{profiles[0].fiscal_year}"
        label_b = f"{profiles[1].company_name} FY{profiles[1].fiscal_year}"
        skipped = [c.kpi_name for c in report.comparisons if c not in filtered]

        # Header row
        hc1, hc2 = st.columns([3, 1])
        with hc1:
            same_co = c1n.strip().lower() == c2n.strip().lower()
            title = (
                f"{c1n}: FY{fy1v} vs FY{fy2v}"
                if same_co else
                f"{c1n}  vs  {c2n}"
            )
            st.markdown(
                f"<div style='font-size:22px;font-weight:800;color:{C['text']};margin-bottom:4px'>"
                f"{title}</div>"
                f"<div style='font-size:12px;color:{C['sub']}'>"
                f"<span class='badge-blue'>{_sector}</span>"
                + (f"&nbsp;&nbsp;FY{fy1v} · FY{fy2v}&nbsp;·&nbsp;" if not same_co else
                   f"&nbsp;&nbsp;")
                + f"Intensity per ₹Crore revenue"
                + f"&nbsp;&nbsp;|&nbsp;&nbsp;Data sources: "
                + f"<span class='badge-amber'>{src1}</span> / <span class='badge-amber'>{src2}</span>"
                + "</div>",
                unsafe_allow_html=True,
            )

        with hc2:
            pdf_bytes = _export_pdf_report(profiles, filtered, summary, _sector)
            st.download_button(
                "⬇ Download Report",
                data=pdf_bytes,
                file_name=f"ESG_{c1n}_vs_{c2n}_FY{fy1v}_{fy2v}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        if skipped:
            skipped_labels = [KPI_META.get(k, {}).get("label", k) for k in skipped]
            st.info(f"Excluded (ratio exceeded ceiling): {', '.join(skipped_labels)}")

        if not filtered:
            st.warning(
                "No KPIs passed the sanity filter. "
                "Check that PDFs contain absolute values (not intensity-only rows)."
            )
            with st.expander("Extraction log"):
                lc1, lc2 = st.columns(2)
                with lc1:
                    st.markdown(f"**{label_a}**")
                    for line in result.get("log1", []):
                        st.code(line, language=None)
                with lc2:
                    st.markdown(f"**{label_b}**")
                    for line in result.get("log2", []):
                        st.code(line, language=None)
            st.stop()

        st.markdown("---")

        # Scorecard row
        st.markdown('<div class="sec">Overview</div>', unsafe_allow_html=True)
        wins_a = sum(1 for c in filtered if c.winner == label_a)
        wins_b = sum(1 for c in filtered if c.winner == label_b)
        leader = label_a.split(" FY")[0] if wins_a >= wins_b else label_b.split(" FY")[0]

        sc = st.columns(4)
        for col, (bcolor, blabel, bval, bsub) in zip(sc, [
            (C["blue"],  "KPIs Compared",  str(len(filtered)), f"of {len(report.comparisons)} total"),
            (C["ca"],    label_a,           str(wins_a),         "KPI wins"),
            (C["cb"],    label_b,           str(wins_b),         "KPI wins"),
            (C["green"], "Leader",          leader,              "More KPI wins"),
        ]):
            col.markdown(f"""
            <div class="card" style="border-top:3px solid {bcolor}">
              <div class="label-tag">{blabel}</div>
              <div style="font-size:26px;font-weight:800;color:{C['text']};
                          white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
                {bval}
              </div>
              <div style="font-size:12px;color:{C['sub']};margin-top:2px">{bsub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Radar + donut
        st.markdown('<div class="sec">Performance Overview</div>', unsafe_allow_html=True)
        vc1, vc2 = st.columns([2, 1])
        with vc1:
            st.markdown(
                f"<span style='font-size:13px;font-weight:600;color:{C['text']}'>"
                f"Normalised Score Radar</span> "
                f"<span style='font-size:11px;color:{C['sub']}'>Higher = better</span>",
                unsafe_allow_html=True,
            )
            fig = _radar_chart(filtered, label_a, label_b)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with vc2:
            st.markdown(
                f"<span style='font-size:13px;font-weight:600;color:{C['text']}'>"
                f"Win Distribution</span>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                _donut_chart(filtered, label_a, label_b),
                use_container_width=True, config={"displayModeBar": False},
            )
            st.markdown(f"""
            <div style="display:flex;gap:14px;justify-content:center;margin-top:4px">
              <div style="display:flex;align-items:center;gap:5px">
                <div style="width:10px;height:10px;border-radius:3px;background:{C['ca']}"></div>
                <span style="font-size:12px;color:{C['text']}">{label_a.split(' FY')[0]} FY{fy1v}</span>
              </div>
              <div style="display:flex;align-items:center;gap:5px">
                <div style="width:10px;height:10px;border-radius:3px;background:{C['cb']}"></div>
                <span style="font-size:12px;color:{C['text']}">{label_b.split(' FY')[0]} FY{fy2v}</span>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Gap bars
        st.markdown('<div class="sec">Intensity Comparison</div>', unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:12px;color:{C['sub']};margin-bottom:8px'>"
            f"Lower bar = better environmental performance per ₹Crore revenue.</div>",
            unsafe_allow_html=True,
        )
        gap = _gap_bar_chart(filtered, label_a, label_b)
        if gap:
            st.plotly_chart(gap, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Per-KPI detail cards
        st.markdown('<div class="sec">KPI Detail</div>', unsafe_allow_html=True)
        for comp in filtered:
            meta   = KPI_META.get(comp.kpi_name, {})
            vals   = {l: v for l, v, _ in comp.entries}
            va, vb = vals.get(label_a), vals.get(label_b)
            wname  = comp.winner.split(" FY")[0]
            a_wins = comp.winner == label_a

            st.markdown(f"""
            <div class="card">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
                <div>
                  <span style="font-size:15px;font-weight:700;color:{C['text']}">
                    {meta.get('label', comp.display_name)}
                  </span>
                  <div style="font-size:11px;color:{C['sub']};margin-top:2px">
                    {meta.get('desc','')} · {meta.get('unit', comp.unit)}
                  </div>
                </div>
                <div style="text-align:right">
                  <span class="badge-green">★ {wname}</span>
                  <div style="font-size:11px;color:{C['sub']};margin-top:3px">{comp.pct_gap:.1f}% gap</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            st.plotly_chart(
                _mini_bar_chart(comp, label_a, label_b),
                use_container_width=True, config={"displayModeBar": False},
            )

            mc1, mc2 = st.columns(2)
            for col, label_full, val, color, is_win, dname in [
                (mc1, label_a, va, C["ca"], a_wins,     label_a),
                (mc2, label_b, vb, C["cb"], not a_wins, label_b),
            ]:
                vs = (f"{val:.4e}" if (val is not None and val < 0.001)
                      else (f"{val:.4f}" if val is not None else "N/A"))
                col.markdown(f"""
                <div style="background:{C['surface']};border-radius:8px;padding:10px 14px;
                            border:1px solid {C['border']};border-left:4px solid {color}">
                  <div style="font-size:11px;font-weight:600;color:{color}">
                    {dname}{'  ★' if is_win else ''}
                  </div>
                  <div style="font-size:13px;color:{C['text']};margin-top:4px;font-family:monospace">
                    {vs} {meta.get('unit','')}
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

        # Methodology expander
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        with st.expander("Methodology & Data Provenance"):
            st.markdown(f"""
<p style="color:{C['text']}"><strong>Data sources</strong><br>
Each company slot accepts either an uploaded PDF or existing DB records (or both).
When a PDF is uploaded, it is registered, parsed, and KPIs are extracted immediately.
If DB records also exist for that company+FY, they are merged: DB values fill in
KPIs not found in the PDF and vice versa; for overlapping KPIs the higher-confidence
value wins.</p>

<p style="color:{C['text']}"><strong>Same-company year-over-year</strong><br>
Set both slots to the same company name with different fiscal years.
Each year can have its own PDF upload or use its own DB records independently.</p>

<p style="color:{C['text']}"><strong>Intensity ratios</strong><br>
Every KPI is divided by annual revenue (₹Crore) to normalise for company size.</p>

<p style="color:{C['text']}"><strong>Unit normalisation</strong><br>
Energy → GJ, Emissions → tCO₂e, Water → KL, Waste → MT.</p>

<p style="color:{C['text']}"><strong>Sanity filter</strong><br>
KPIs with implausible intensity ratios (likely unit errors) are excluded automatically.</p>
""", unsafe_allow_html=True)

        # Pipeline logs
        with st.expander("Pipeline log"):
            lc1, lc2 = st.columns(2)
            with lc1:
                st.markdown(
                    f"<span style='font-weight:600;color:{C['ca']}'>{label_a}</span>",
                    unsafe_allow_html=True,
                )
                for line in result.get("log1", []):
                    st.code(line, language=None)
            with lc2:
                st.markdown(
                    f"<span style='font-weight:600;color:{C['cb']}'>{label_b}</span>",
                    unsafe_allow_html=True,
                )
                for line in result.get("log2", []):
                    st.code(line, language=None)


# ---------------------------------------------------------------------------
# Tab: How it works
# ---------------------------------------------------------------------------

with tab_help:
    st.markdown(
        f"<h3 style='color:{C['text']}'>How to use this dashboard</h3>",
        unsafe_allow_html=True,
    )

    sections = [
        ("🔵 Compare two companies",
         f"Enter different company names in each slot. Upload their PDFs or rely on "
         f"previously ingested data. Click <strong>Compare</strong>."),
        ("📅 Year-over-year comparison",
         f"Enter the <strong>same company name</strong> in both slots with "
         f"<strong>different fiscal years</strong>. Upload separate PDFs for each year. "
         f"The dashboard will label it as a year-over-year comparison automatically."),
        ("📄 PDF upload",
         f"Upload a BRSR, ESG, Integrated, or Annual Report PDF for either company slot. "
         f"The PDF is saved to temporary storage, parsed, and KPIs are extracted immediately. "
         f"If DB records already exist for that company+FY, they are merged with the PDF results."),
        ("🗄️ DB-only (no PDF)",
         f"If no PDF is uploaded but the company+FY combination exists in the database, "
         f"the cached KPIs and revenue are used directly — no network calls, no parsing."),
        ("⚠️ No data at all",
         f"If neither a PDF is uploaded nor DB records exist, the slot will show an error. "
         f"You must provide at least one data source per slot."),
        ("📊 Charts",
         f"All KPIs are shown as intensity ratios (value ÷ revenue in ₹Crore). "
         f"Lower intensity = better environmental performance. "
         f"KPIs with ratios above a safety ceiling are excluded to prevent unit-error artefacts."),
    ]

    for title, body in sections:
        st.markdown(f"""
        <div class="card" style="margin-bottom:12px">
          <div style="font-size:14px;font-weight:700;color:{C['text']};margin-bottom:6px">
            {title}
          </div>
          <div style="font-size:13px;color:{C['text']};line-height:1.7">{body}</div>
        </div>""", unsafe_allow_html=True)