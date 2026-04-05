"""
dashboard/ui.py

ESG Competitive Intelligence Dashboard — Fully Automated Pipeline.

Architecture:
    No PDF uploads. When a user compares two companies, the dashboard:

    1. DB-FIRST CHECK
       Query kpi_records + reports.revenue_cr for each company+FY.
       If all required KPIs and revenue are cached and pass plausibility,
       skip to Step 5 immediately (fast path, no API calls).

    2. INGESTION  (if data missing)
       Call IngestionAgent.run() → Tavily search → download PDF → persist
       Report row in DB. Handles 406/403 with UA rotation automatically.

    3. PARSING
       Call ParseOrchestrator.run() → ParsingAgent → ChunkingAgent →
       EmbeddingService → ParsedDocument + DocumentChunks stored in DB.
       Idempotent — cache keyed on (report_id, parser_version).

    4. EXTRACTION
       Call ExtractionAgent.extract_all() → regex → batch LLM → validate
       → store KPIRecords. Then extract_revenue() for revenue figure.

    5. BENCHMARK
       Normalise KPIs → intensity ratios → compare_profiles() →
       generate_summary() → render charts.

Key design decisions:
    - Every step is idempotent and stores results in DB.
    - Second run for the same company+FY is always a fast DB-only path.
    - All failures surface as visible Streamlit errors with full tracebacks.
    - No global state — pipeline state flows through function arguments.
    - Plausibility guards on absolute values AND intensity ratios.

Run:
    cd /path/to/esg_pipeline
    streamlit run dashboard/ui.py
"""
from __future__ import annotations

import io
import sys
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ── Path bootstrap (must run before any project imports) ─────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import plotly.graph_objects as go
import streamlit as st
import pandas as pd

# ── Page config — MUST be the first Streamlit call ───────────────────────────
st.set_page_config(
    page_title="ESG Intelligence · Benchmark",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# DESIGN TOKENS
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
    "ca":      "#4B84DE",   # company A colour
    "cb":      "#10B981",   # company B colour
    "grid":    "#2D3142",
    "font":    "Inter, 'Helvetica Neue', Arial, sans-serif",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{{
    font-family:{C['font']};background:{C['bg']};color:{C['text']};
}}
.stApp{{background:{C['bg']};}}
[data-testid="stSidebar"]{{
    background:{C['surface']};border-right:1px solid {C['border']};
}}
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

# KPI display metadata — lower intensity is better for all
KPI_META = {
    "scope_1_emissions":   {
        "label": "Scope 1 GHG",     "unit": "tCO2e/Cr",
        "max_ratio": 10,   "desc": "Direct GHG per ₹Crore revenue",
    },
    "scope_2_emissions":   {
        "label": "Scope 2 GHG",     "unit": "tCO2e/Cr",
        "max_ratio": 10,   "desc": "Indirect GHG per ₹Crore revenue",
    },
    "total_ghg_emissions": {
        "label": "Total GHG",       "unit": "tCO2e/Cr",
        "max_ratio": 20,   "desc": "Scope 1+2 per ₹Crore revenue",
    },
    "waste_generated":     {
        "label": "Waste Intensity", "unit": "MT/Cr",
        "max_ratio": 5,    "desc": "Waste generated per ₹Crore revenue",
    },
}

# KPIs we attempt to extract (total_ghg is derived, not extracted directly)
EXTRACTABLE_KPI_NAMES = [
    "scope_1_emissions",
    "scope_2_emissions",
    "waste_generated",
]

# All KPIs shown in comparison (including derived)
TARGET_KPI_NAMES = list(KPI_META.keys())

# Plausibility limits (canonical units) — values outside are extraction errors
_KPI_PLAUSIBILITY: dict[str, tuple[float, float]] = {
    "scope_1_emissions":   (1,        5_000_000),
    "scope_2_emissions":   (1,        5_000_000),
    "total_ghg_emissions": (1,       10_000_000),
    "waste_generated":     (0.1,        500_000),
}

# Default fallback revenue if all extraction fails
_DEFAULT_REVENUE_CR = 315_322.0

# =============================================================================
# DATA CONTAINERS
# =============================================================================

@dataclass
class CompanyData:
    """
    All data needed for one company in the benchmark.
    Populated progressively through the pipeline steps.
    """
    company_name:   str
    fy:             int
    sector:         str
    kpi_records:    dict        # {kpi_name: {"value", "unit", "method", "confidence"}}
    revenue_result: object      # RevenueResult | None
    log:            list[str]   # human-readable pipeline steps
    company_id:     Optional[uuid.UUID] = None
    report_id:      Optional[uuid.UUID] = None
    file_path:      Optional[str] = None


# =============================================================================
# DB LAYER  (all calls wrapped — dashboard works even when DB is down)
# =============================================================================

@st.cache_resource
def _check_db() -> bool:
    """Check DB connectivity once per app lifetime."""
    try:
        from core.database import check_connection
        return check_connection()
    except Exception:
        return False


@st.cache_resource(ttl=60)
def _get_company_names() -> list[str]:
    """Return known company names for sidebar hints (refreshed every 60s)."""
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


def _db_lookup(company_name: str, fy: int) -> dict:
    """
    Pull cached KPI records and revenue for a company+FY.

    Returns plain Python dict — no live ORM objects (avoids DetachedInstanceError).
    Only values that pass the plausibility check are returned; implausible
    values are dropped here so the pipeline re-extracts them.

    Return shape:
        {
            "kpis":      {kpi_name: {"value","unit","method","confidence"}},
            "revenue":   RevenueResult | None,
            "company_id": UUID | None,
            "report_id":  UUID | None,
            "file_path":  str | None,
        }
    """
    empty = {
        "kpis": {}, "revenue": None,
        "company_id": None, "report_id": None, "file_path": None,
    }
    try:
        from core.database import get_db
        from models.db_models import Company, Report, KPIRecord, KPIDefinition
        from services.revenue_extractor import RevenueResult

        with get_db() as db:
            # ── Company ───────────────────────────────────────────────────
            company_row = (
                db.query(Company)
                .filter(Company.name.ilike(f"%{company_name}%"))
                .first()
            )
            if not company_row:
                return empty

            company_id   = company_row.id
            company_name_db = company_row.name

            # ── Report ────────────────────────────────────────────────────
            report_row = (
                db.query(Report)
                .filter(
                    Report.company_id == company_id,
                    Report.report_year == fy,
                )
                .order_by(Report.created_at.desc())
                .first()
            )
            report_id  = report_row.id        if report_row else None
            file_path  = report_row.file_path if report_row else None

            # ── Revenue ───────────────────────────────────────────────────
            cached_rev = None
            if report_row and getattr(report_row, "revenue_cr", None) is not None:
                try:
                    cached_rev = RevenueResult(
                        value_cr      = float(report_row.revenue_cr),
                        raw_value     = str(report_row.revenue_cr),
                        raw_unit      = getattr(report_row, "revenue_unit",   None) or "INR_Crore",
                        source        = getattr(report_row, "revenue_source", None) or "db",
                        page_number   = 0,
                        confidence    = 0.99,
                        pattern_name  = "cached",
                    )
                except Exception:
                    pass

            # ── KPI records ───────────────────────────────────────────────
            kpis: dict = {}
            if report_row:
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
                        .filter(
                            KPIRecord.company_id       == company_id,
                            KPIRecord.kpi_definition_id == kdef.id,
                            KPIRecord.report_year      == fy,
                            KPIRecord.normalized_value.isnot(None),
                        )
                        .order_by(KPIRecord.extracted_at.desc())
                        .first()
                    )
                    if not rec:
                        continue

                    val  = rec.normalized_value
                    unit = rec.unit or kdef.expected_unit

                    # Plausibility guard — drop implausible DB values
                    lo, hi = _KPI_PLAUSIBILITY.get(kpi_name, (0, float("inf")))
                    if not (lo <= val <= hi):
                        continue  # will be re-extracted

                    kpis[kpi_name] = {
                        "value":      val,
                        "unit":       unit,
                        "method":     rec.extraction_method,
                        "confidence": rec.confidence or 0.9,
                    }

        return {
            "kpis":       kpis,
            "revenue":    cached_rev,
            "company_id": company_id,
            "report_id":  report_id,
            "file_path":  file_path,
        }

    except Exception as exc:
        # Suppress — dashboard stays functional without DB
        st.warning(f"DB lookup failed for {company_name}: {exc}")
        return empty


def _db_ensure_schema() -> None:
    """Add revenue columns to reports table if not yet present."""
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
    Persist new KPI records and revenue to DB (append-only, skips duplicates).
    Called once after the extraction pipeline completes.
    """
    try:
        from core.database import get_db
        from models.db_models import Report, KPIRecord, KPIDefinition
        from services.revenue_extractor import store_revenue

        with get_db() as db:
            # Revenue
            if revenue_result:
                report_row = db.query(Report).filter(Report.id == report_id).first()
                if report_row and getattr(report_row, "revenue_cr", None) is None:
                    try:
                        store_revenue(report_row, revenue_result, db)
                    except Exception:
                        pass

            # KPIs
            for kpi_name, rec in kpi_records.items():
                kdef = (
                    db.query(KPIDefinition)
                    .filter(KPIDefinition.name == kpi_name)
                    .first()
                )
                if not kdef:
                    continue

                # Skip exact duplicates
                exists = (
                    db.query(KPIRecord)
                    .filter(
                        KPIRecord.company_id        == company_id,
                        KPIRecord.kpi_definition_id == kdef.id,
                        KPIRecord.report_year       == fy,
                        KPIRecord.normalized_value  == rec["value"],
                    )
                    .first()
                )
                if exists:
                    continue

                db.add(KPIRecord(
                    company_id          = company_id,
                    report_id           = report_id,
                    kpi_definition_id   = kdef.id,
                    report_year         = fy,
                    raw_value           = str(rec["value"]),
                    normalized_value    = rec["value"],
                    unit                = rec["unit"],
                    extraction_method   = rec["method"],
                    confidence          = rec["confidence"],
                    is_validated        = rec["confidence"] >= 0.85,
                    validation_notes    = "esg_dashboard",
                ))

    except Exception as exc:
        st.warning(f"DB store failed: {exc}")


# =============================================================================
# PIPELINE STEPS
# =============================================================================

def _step_ingest(company_name: str, fy: int, sector: str, log: list[str]) -> dict:
    """
    Step 1 — Ingestion.

    Calls IngestionAgent.run() which:
      - get_or_create Company in DB
      - Tavily multi-query search for PDF URLs
      - Downloads PDF with browser-UA rotation (handles 406/403)
      - Persists Report row in DB

    Returns dict with company_id, report_id, file_path (or None on failure).
    """
    from agents.ingestion_agent import IngestionAgent
    from models.schemas import CompanyCreate

    log.append("🔍  Searching for ESG/BRSR report via Tavily...")

    agent = IngestionAgent()
    company_data = CompanyCreate(
        name    = company_name,
        sector  = sector,
        country = "India",
    )

    try:
        result = agent.run(
            company_data   = company_data,
            year           = fy,
            report_type    = "BRSR",
            auto_download  = True,
            max_downloads  = 3,  # try up to 3 URLs if earlier ones fail
        )
    except Exception as exc:
        log.append(f"❌  Ingestion failed: {exc}")
        return {"company_id": None, "report_id": None, "file_path": None}

    company   = result.get("company")
    downloads = result.get("downloaded_reports", [])

    if not downloads:
        log.append("❌  No PDF downloaded — all discovered URLs failed.")
        return {
            "company_id": company.id if company else None,
            "report_id":  None,
            "file_path":  None,
        }

    # Use the first successfully downloaded report
    dl_report = next((r for r in downloads if r.status == "downloaded"), None)
    if not dl_report:
        log.append("❌  Downloads attempted but all failed.")
        return {
            "company_id": company.id if company else None,
            "report_id":  None,
            "file_path":  None,
        }

    log.append(
        f"✅  Downloaded PDF · {round((dl_report.file_size_bytes or 0) / 1e6, 1)} MB"
    )
    return {
        "company_id": company.id,
        "report_id":  dl_report.id,
        "file_path":  dl_report.file_path,
    }


def _step_parse(report_id: uuid.UUID, log: list[str]) -> bool:
    """
    Step 2 — Parsing.

    Calls ParseOrchestrator.run() which:
      - Checks parse cache (idempotent — skips if already parsed)
      - ParsingAgent: spatial chunker (pdfplumber) or block-mode (fitz)
      - ChunkingAgent: sentence-boundary splitting with overlap
      - EmbeddingService: local bge-small model (zero API cost)
      - Stores ParsedDocument + DocumentChunks in DB

    Returns True on success.
    """
    from services.parse_orchestrator import ParseOrchestrator

    log.append("📄  Parsing PDF (spatial chunker + embeddings)...")
    try:
        result = ParseOrchestrator().run(report_id=report_id, force=False)
        log.append(
            f"✅  Parsed · {result.page_count} pages · "
            f"{result.meta.get('chunk_count','?')} chunks · "
            f"{result.meta.get('table_count','?')} tables"
        )
        return True
    except Exception as exc:
        log.append(f"❌  Parsing failed: {exc}")
        return False


def _step_extract(
    report_id:  uuid.UUID,
    fy:         int,
    log:        list[str],
    llm_service,
) -> dict:
    """
    Step 3 — KPI Extraction + Revenue.

    KPI extraction (ExtractionAgent.extract_all):
      - Regex pass (free, deterministic)
      - Batch LLM pass for remaining KPIs (2-3 KPIs per call)
      - Validation + range checks
      - KPIRecords stored in DB by the agent

    Revenue extraction (extract_revenue):
      - Regex from P&L pages
      - Back-calculation from BRSR intensity ratios
      - LLM fallback

    Returns dict of extracted KPI records + revenue.
    """
    from agents.extraction_agent import ExtractionAgent
    from services.revenue_extractor import extract_revenue
    from core.database import get_db

    new_kpis: dict = {}

    # ── KPI extraction ────────────────────────────────────────────────────────
    log.append(f"🔬  Extracting KPIs (regex → batch-LLM → validate)...")
    try:
        with get_db() as db:
            extracted_list = ExtractionAgent().extract_all(
                report_id  = report_id,
                db         = db,
                kpi_names  = EXTRACTABLE_KPI_NAMES,
            )

        for ext in extracted_list:
            if ext.normalized_value is None:
                log.append(f"  ✗  {ext.kpi_name}: not found")
                continue

            val  = ext.normalized_value
            unit = ext.unit or ""

            # Plausibility guard
            lo, hi = _KPI_PLAUSIBILITY.get(ext.kpi_name, (0, float("inf")))
            if not (lo <= val <= hi):
                log.append(
                    f"  ⚠  {ext.kpi_name}: {val:,.2f} {unit} "
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
                f"  ✅  {ext.kpi_name}: {val:,.2f} {unit} "
                f"[{ext.extraction_method} conf={ext.confidence:.2f}]"
            )

    except Exception as exc:
        log.append(f"❌  Extraction agent failed: {exc}")
        # Continue — revenue extraction may still succeed

    # ── Revenue extraction ────────────────────────────────────────────────────
    log.append("💰  Extracting revenue...")
    new_revenue = None

    # Get file path for revenue extractor (needs direct PDF access)
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
                pdf_path          = Path(pdf_path_str),
                fiscal_year_hint  = fy,
                llm_service       = llm_service,
            )
            if new_revenue:
                log.append(
                    f"  ✅  Revenue: ₹{new_revenue.value_cr:,.0f} Cr "
                    f"[{new_revenue.pattern_name} conf={new_revenue.confidence:.2f}]"
                )
            else:
                log.append("  ✗  Revenue: not found — default will be used")
        except Exception as exc:
            log.append(f"  ⚠  Revenue extraction failed: {exc}")
    else:
        log.append("  ✗  Revenue: PDF path unavailable for revenue extraction")

    return {"kpis": new_kpis, "revenue": new_revenue}


def _derive_total_ghg(kpi_records: dict) -> Optional[dict]:
    """
    Derive total_ghg_emissions = scope_1 + scope_2.

    BRSR reports never state this as a standalone absolute value.
    The 'Total Scope 1 and 2' label in the PDF only appears on intensity
    rows (per rupee), never as an absolute tCO2e figure.
    This derivation is the correct approach.
    """
    if "total_ghg_emissions" in kpi_records:
        return None  # already present

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
        "value":      total,
        "unit":       "tCO2e",
        "method":     "derived",
        "confidence": round(min(s1["confidence"], s2["confidence"]) * 0.99, 3),
    }


# =============================================================================
# MAIN PIPELINE ORCHESTRATOR
# =============================================================================

def run_company_pipeline(
    company_name: str,
    fy:           int,
    sector:       str,
    db_online:    bool,
    llm_service,
    status_placeholder,
) -> CompanyData:
    """
    DB-first pipeline for one company+FY.

    Flow:
        1. DB lookup — return fast if all data cached
        2. Ingest (search + download PDF via Tavily)
        3. Parse (spatial chunker → chunks → embeddings in DB)
        4. Extract (regex → batch LLM → validate → store KPIs in DB)
        5. Derive total_ghg from scope_1 + scope_2
        6. Persist any new KPI records + revenue to DB

    All steps are idempotent. If interrupted and restarted, the pipeline
    resumes from the last completed step (cache-keyed at each layer).

    Args:
        company_name:        Company name (fuzzy-matched in DB)
        fy:                  Fiscal year (e.g. 2025)
        sector:              Sector string for Company creation
        db_online:           Whether DB is reachable
        llm_service:         LLMService instance or None
        status_placeholder:  Streamlit placeholder for live status updates

    Returns:
        CompanyData with all extracted/cached data
    """
    log: list[str] = []

    def _update(msg: str):
        """Push a status line to both the log and the live placeholder."""
        log.append(msg)
        status_placeholder.markdown(
            "<div class='step-log'>" +
            "<br>".join(log[-12:]) +   # show last 12 lines to avoid overflow
            "</div>",
            unsafe_allow_html=True,
        )

    _update(f"⏳  Starting pipeline for **{company_name} FY{fy}**...")

    company_id: Optional[uuid.UUID] = None
    report_id:  Optional[uuid.UUID] = None
    file_path:  Optional[str]       = None

    # ── Step 1: DB lookup ─────────────────────────────────────────────────────
    cached_kpis  = {}
    cached_rev   = None

    if db_online:
        _update("🗄️   Checking DB cache...")
        db_data     = _db_lookup(company_name, fy)
        cached_kpis = db_data["kpis"]
        cached_rev  = db_data["revenue"]
        company_id  = db_data["company_id"]
        report_id   = db_data["report_id"]
        file_path   = db_data["file_path"]

        if cached_kpis:
            _update(f"  ✅  {len(cached_kpis)} KPI(s) in DB: {list(cached_kpis.keys())}")
        else:
            _update("  ℹ️   No cached KPIs — will extract from PDF.")

        if cached_rev:
            _update(f"  ✅  Revenue in DB: ₹{cached_rev.value_cr:,.0f} Cr")
        else:
            _update("  ℹ️   Revenue not cached — will extract from PDF.")

    # Determine what is still missing
    missing_kpis = [k for k in EXTRACTABLE_KPI_NAMES if k not in cached_kpis]
    need_revenue = cached_rev is None

    # Fast path: everything is already cached
    if not missing_kpis and not need_revenue:
        _update("✅  All data available from DB — skipping pipeline.")
        merged = dict(cached_kpis)
        ghg = _derive_total_ghg(merged)
        if ghg:
            merged["total_ghg_emissions"] = ghg
        return CompanyData(
            company_name   = company_name,
            fy             = fy,
            sector         = sector,
            kpi_records    = merged,
            revenue_result = cached_rev,
            log            = log,
            company_id     = company_id,
            report_id      = report_id,
            file_path      = file_path,
        )

    # ── Step 2: Ingest (search + download) ───────────────────────────────────
    # Only ingest if we don't already have a downloaded PDF in DB
    pdf_available = (
        file_path is not None and Path(file_path).exists()
    )

    if not pdf_available:
        _update(f"📥  Ingesting — searching & downloading PDF for {company_name} FY{fy}...")
        ingest_result = _step_ingest(company_name, fy, sector, log)
        company_id = ingest_result["company_id"] or company_id
        report_id  = ingest_result["report_id"]  or report_id
        file_path  = ingest_result["file_path"]  or file_path
        _update(f"  file_path={file_path}")

        if not file_path or not Path(file_path).exists():
            _update("❌  Cannot proceed — no PDF available.")
            return CompanyData(
                company_name   = company_name,
                fy             = fy,
                sector         = sector,
                kpi_records    = dict(cached_kpis),
                revenue_result = cached_rev,
                log            = log,
                company_id     = company_id,
                report_id      = report_id,
                file_path      = None,
            )
    else:
        _update(f"📎  PDF already in DB at {file_path}")

    # ── Step 3: Parse ─────────────────────────────────────────────────────────
    if report_id:
        parsed_ok = _step_parse(report_id, log)
        if not parsed_ok:
            _update("⚠️  Parsing failed — extraction may be incomplete.")
    else:
        _update("⚠️  No report_id available — skipping parse.")

    # ── Step 4: Extract ───────────────────────────────────────────────────────
    new_kpis   = {}
    new_revenue = None

    if report_id:
        _update("🧬  Extracting KPIs from parsed chunks...")
        extract_result = _step_extract(report_id, fy, log, llm_service)
        new_kpis    = extract_result["kpis"]
        new_revenue = extract_result["revenue"]

    # ── Step 5: Merge + derive total_ghg ─────────────────────────────────────
    merged_kpis = {**cached_kpis, **new_kpis}
    ghg = _derive_total_ghg(merged_kpis)
    if ghg:
        merged_kpis["total_ghg_emissions"] = ghg
        s1v = merged_kpis.get("scope_1_emissions", {}).get("value", 0)
        s2v = merged_kpis.get("scope_2_emissions", {}).get("value", 0)
        _update(
            f"  🔗  Derived total_ghg: {ghg['value']:,.2f} tCO2e "
            f"({s1v:,.0f} + {s2v:,.0f})"
        )

    # ── Step 6: Persist new data to DB ───────────────────────────────────────
    to_store = dict(new_kpis)
    if "total_ghg_emissions" not in cached_kpis and "total_ghg_emissions" in merged_kpis:
        to_store["total_ghg_emissions"] = merged_kpis["total_ghg_emissions"]

    if db_online and company_id and report_id and (to_store or new_revenue):
        _update(f"💾  Storing {len(to_store)} KPI record(s) to DB...")
        _db_store_kpis(
            company_id     = company_id,
            report_id      = report_id,
            fy             = fy,
            kpi_records    = to_store,
            revenue_result = new_revenue,
        )
        _update("  ✅  Stored.")

    revenue = cached_rev or new_revenue
    if not revenue:
        _update(
            f"  ℹ️   No revenue found — using default "
            f"₹{_DEFAULT_REVENUE_CR:,.0f} Cr for intensity ratios."
        )

    _update(f"🏁  Pipeline complete for {company_name} FY{fy}.")

    return CompanyData(
        company_name   = company_name,
        fy             = fy,
        sector         = sector,
        kpi_records    = merged_kpis,
        revenue_result = revenue,
        log            = log,
        company_id     = company_id,
        report_id      = report_id,
        file_path      = file_path,
    )


# =============================================================================
# BENCHMARK BUILDER
# =============================================================================

def _build_benchmark(data1: CompanyData, data2: CompanyData, sector: str) -> dict:
    """
    Build CompanyProfiles → BenchmarkReport → narrative summary.
    Uses benchmark.py and summary_generator.py (unchanged).
    """
    from services.benchmark import build_company_profile, compare_profiles
    from services.summary_generator import generate_summary
    from services.llm_service import LLMService
    from core.config import get_settings

    profiles = []
    for data in [data1, data2]:
        rev = data.revenue_result
        rev_cr  = rev.value_cr  if rev else _DEFAULT_REVENUE_CR
        rev_src = rev.source    if rev else "default"

        # Collect page texts for reported-ratio detection
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

    # LLM for narrative summary
    llm = None
    try:
        if get_settings().llm_api_key:
            llm = LLMService()
    except Exception:
        pass

    summary = generate_summary(profiles, report, llm=llm)

    return {
        "profiles": profiles,
        "report":   report,
        "filtered": filtered,
        "summary":  summary,
    }


# =============================================================================
# SANITY FILTER
# =============================================================================

def _filter_comparable(comparisons) -> list:
    """
    Drop comparisons where either company's ratio exceeds the plausibility ceiling.
    Prevents unit errors (MJ not converted to GJ) from corrupting the chart.
    """
    out = []
    for comp in comparisons:
        meta = KPI_META.get(comp.kpi_name, {})
        mx   = meta.get("max_ratio")
        ok   = all(
            (not mx or v <= mx) and v > 0
            for _, v, _ in comp.entries
        )
        if ok:
            out.append(comp)
    return out


# =============================================================================
# CHART HELPERS
# =============================================================================
_FONT = dict(family="Inter, Arial, sans-serif", size=12, color=C["text"])


def _hex_rgba(h: str, a: float) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16)
    return f"rgba({r},{g},{b},{a})"


def _radar(filtered, la: str, lb: str):
    cats, sa, sb = [], [], []
    for comp in filtered:
        vals = {l: v for l, v, _ in comp.entries}
        va, vb = vals.get(la), vals.get(lb)
        if va is None or vb is None:
            continue
        cats.append(KPI_META.get(comp.kpi_name, {}).get("label", comp.display_name))
        total = va + vb
        sa.append(round(100 * (1 - va / total), 1) if total else 50)
        sb.append(round(100 * (1 - vb / total), 1) if total else 50)

    if len(cats) < 2:
        return None

    cats_c = cats + [cats[0]]
    sa_c   = sa   + [sa[0]]
    sb_c   = sb   + [sb[0]]

    fig = go.Figure()
    for name, scores, color in [
        (la.split(" FY")[0], sa_c, C["ca"]),
        (lb.split(" FY")[0], sb_c, C["cb"]),
    ]:
        fig.add_trace(go.Scatterpolar(
            r          = scores,
            theta      = cats_c,
            fill       = "toself",
            name       = name,
            line       = dict(color=color, width=2.5),
            fillcolor  = _hex_rgba(color, 0.12),
            hovertemplate = "<b>%{theta}</b><br>Score: %{r:.0f}<extra></extra>",
        ))

    fig.update_layout(
        polar = dict(
            bgcolor     = C["surface"],
            radialaxis  = dict(
                visible   = True,
                range     = [0, 100],
                tickfont  = dict(size=9, color=C["sub"]),
                gridcolor = C["grid"],
                linecolor = C["border"],
            ),
            angularaxis = dict(
                tickfont  = dict(size=11, color=C["text"]),
                gridcolor = C["grid"],
                linecolor = C["border"],
            ),
        ),
        paper_bgcolor = C["bg"],
        height        = 360,
        margin        = dict(l=40, r=40, t=40, b=20),
        font          = _FONT,
        legend        = dict(orientation="h", y=-0.08, font=dict(size=12)),
    )
    return fig


def _donut(filtered, la: str, lb: str):
    na = la.split(" FY")[0]
    nb = lb.split(" FY")[0]
    wa = sum(1 for c in filtered if c.winner == la)
    wb = sum(1 for c in filtered if c.winner == lb)

    fig = go.Figure(go.Pie(
        labels      = [na, nb],
        values      = [wa, wb],
        hole        = 0.62,
        marker      = dict(
            colors = [C["ca"], C["cb"]],
            line   = dict(color=C["bg"], width=3),
        ),
        textinfo    = "label+percent",
        textfont    = dict(size=12),
        hovertemplate = "<b>%{label}</b><br>%{value} wins<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor = C["bg"],
        height        = 260,
        margin        = dict(l=0, r=0, t=10, b=0),
        showlegend    = False,
        annotations   = [dict(
            text     = f"{wa + wb}<br><span style='font-size:10px'>KPIs</span>",
            x=0.5, y=0.5,
            font     = dict(size=20, color=C["text"]),
            showarrow= False,
        )],
        font = _FONT,
    )
    return fig


def _gap_bars(filtered, la: str, lb: str):
    labels_list, va_list, vb_list = [], [], []
    for comp in filtered:
        vals = {l: v for l, v, _ in comp.entries}
        va, vb = vals.get(la), vals.get(lb)
        if va is None or vb is None:
            continue
        labels_list.append(KPI_META.get(comp.kpi_name, {}).get("label", comp.display_name))
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
            name          = name,
            x             = vals,
            y             = labels_list,
            orientation   = "h",
            marker        = dict(color=color, line=dict(color=color, width=1), pattern_shape=pat),
            text          = [f"{v:.3g}" for v in vals],
            textposition  = "outside",
            textfont      = dict(size=10),
            hovertemplate = "<b>%{y}</b><br>%{x:.4e}<extra></extra>",
        ))

    fig.update_layout(
        barmode       = "group",
        height        = 300,
        paper_bgcolor = C["bg"],
        plot_bgcolor  = C["bg"],
        font          = _FONT,
        xaxis         = dict(
            title     = "Intensity per ₹Crore",
            showgrid  = True,
            gridcolor = C["grid"],
            zeroline  = False,
            color     = C["text"],
        ),
        yaxis         = dict(autorange="reversed", color=C["text"]),
        legend        = dict(orientation="h", y=1.08),
        margin        = dict(l=160, r=60, t=20, b=40),
    )
    return fig


def _mini_bar(comp, la: str, lb: str):
    vals   = {l: v for l, v, _ in comp.entries}
    va, vb = vals.get(la, 0), vals.get(lb, 0)
    meta   = KPI_META.get(comp.kpi_name, {})
    na, nb = la.split(" FY")[0], lb.split(" FY")[0]

    fig = go.Figure()
    for name, val, color, pat in [
        (na, va, C["ca"], ""),
        (nb, vb, C["cb"], "/"),
    ]:
        fig.add_trace(go.Bar(
            name          = name,
            x             = [val],
            y             = [meta.get("label", "")],
            orientation   = "h",
            marker        = dict(
                color         = color,
                line          = dict(color=color, width=1.5),
                pattern_shape = pat,
            ),
            text          = [f"{val:.3g}"],
            textposition  = "outside",
            textfont      = dict(size=11, color=C["text"]),
            hovertemplate = f"<b>{name}</b><br>{val:.4e} {meta.get('unit','')}<extra></extra>",
        ))

    fig.update_layout(
        barmode       = "group",
        height        = 110,
        showlegend    = False,
        paper_bgcolor = C["bg"],
        plot_bgcolor  = C["bg"],
        font          = _FONT,
        yaxis         = dict(visible=False),
        xaxis         = dict(
            showgrid       = True,
            gridcolor      = C["grid"],
            zeroline       = False,
            showticklabels = False,
        ),
        margin = dict(l=10, r=60, t=8, b=8),
    )
    return fig


# =============================================================================
# PDF EXPORT
# =============================================================================

def _export_pdf(profiles, filtered, summary: str, sector: str) -> bytes:
    """Generate a ReportLab PDF of the benchmark report."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    )

    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(
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
    title = ParagraphStyle("t",  parent=ss["Title"],   fontSize=22, textColor=BLK,
                            spaceAfter=4,  leading=28, fontName="Helvetica-Bold")
    sub   = ParagraphStyle("s",  parent=ss["Normal"],  fontSize=11, textColor=GRAY,
                            spaceAfter=14, leading=16)
    h2    = ParagraphStyle("h2", parent=ss["Heading2"],fontSize=13, textColor=BLK,
                            spaceBefore=16, spaceAfter=8, fontName="Helvetica-Bold")
    body  = ParagraphStyle("b",  parent=ss["Normal"],  fontSize=10, textColor=BLK,
                            leading=16, spaceAfter=8)
    note  = ParagraphStyle("n",  parent=ss["Normal"],  fontSize=9,  textColor=GRAY,
                            leading=14)

    labels = [f"{p.company_name} FY{p.fiscal_year}" for p in profiles]
    story  = []

    story.append(Paragraph("ESG Competitive Intelligence Report", title))
    story.append(Paragraph(f"{labels[0]} vs {labels[1]}", sub))
    story.append(Paragraph(f"Sector: {sector}", note))
    story.append(HRFlowable(width="100%", thickness=1, color=BDR, spaceAfter=14))

    story.append(Paragraph("Methodology", h2))
    story.append(Paragraph(
        "All metrics are intensity ratios (KPI ÷ INR Crore revenue). "
        "KPIs with unrealistic ratios are excluded automatically. "
        "No absolute values are shown.", body,
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("KPI Intensity Comparison", h2))
    tdata = [["Metric", "Unit", labels[0], labels[1], "Gap", "Leader"]]
    for comp in filtered:
        meta = KPI_META.get(comp.kpi_name, {})
        vals = {l: v for l, v, _ in comp.entries}
        v0, v1 = vals.get(labels[0]), vals.get(labels[1])

        def fmt(v):
            return f"{v:.2e}" if (v and v < 0.001) else (f"{v:.4f}" if v else "N/A")

        tdata.append([
            meta.get("label", comp.display_name),
            meta.get("unit", comp.unit),
            fmt(v0), fmt(v1),
            f"{comp.pct_gap:.1f}%",
            comp.winner.split(" FY")[0],
        ])

    cw  = [5.2*cm, 2.3*cm, 2.8*cm, 2.8*cm, 1.6*cm, 2.5*cm]
    tbl = Table(tdata, colWidths=cw, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING",    (0, 0), (-1, 0), 8),
        ("FONTSIZE",      (0, 1), (-1, -1), 9),
        ("TOPPADDING",    (0, 1), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID",          (0, 0), (-1, -1), 0.5, BDR),
        ("ALIGN",         (2, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TEXTCOLOR",     (5, 1), (5, -1), GREEN),
        ("FONTNAME",      (5, 1), (5, -1), "Helvetica-Bold"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 16))

    story.append(Paragraph("AI-Generated Narrative Summary", h2))
    for para in summary.split("\n\n"):
        para = para.strip()
        if para:
            story.append(Paragraph(para, body))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BDR))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Generated by ESG Competitive Intelligence Pipeline. "
        "Data extracted from public BRSR/Annual reports.", note,
    ))

    doc.build(story)
    return buf.getvalue()


# =============================================================================
# LLM SERVICE INIT
# =============================================================================

@st.cache_resource
def _get_llm_service():
    """Initialise LLMService once per app lifetime (lazy, cached)."""
    try:
        from services.llm_service import LLMService
        from core.config import get_settings
        if get_settings().llm_api_key:
            return LLMService()
    except Exception:
        pass
    return None


# =============================================================================
# SIDEBAR
# =============================================================================
db_online      = _check_db()
known_companies = _get_company_names() if db_online else []
llm_service    = _get_llm_service()

# Ensure revenue columns exist on startup
if db_online:
    _db_ensure_schema()

with st.sidebar:
    st.markdown("""
    <div style="padding:0 0 14px">
        <div style="font-size:22px;font-weight:800;color:#E8EAF0;letter-spacing:-0.5px">
            🌿 ESG Intel
        </div>
        <div style="font-size:12px;color:#8B92A9;margin-top:2px">
            Fully Automated Benchmarking
        </div>
    </div>""", unsafe_allow_html=True)

    # DB status indicator
    green_col = C["green"]
    amber_col = C["amber"]
    red_col   = C["red"]
    sub_col   = C["sub"]

    if db_online:
        st.markdown(
            f"<div style='font-size:11px;color:{green_col};font-weight:600'>"
            f"● Database connected &nbsp; {len(known_companies)} companies in DB</div>",
            unsafe_allow_html=True,
        )
        if llm_service:
            st.markdown(
                f"<div style='font-size:11px;color:{green_col};font-weight:600'>"
                f"● LLM enabled (Gemini)</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='font-size:11px;color:{amber_col};font-weight:600'>"
                f"⚠ LLM disabled (no API key)</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f"<div style='font-size:11px;color:{red_col};font-weight:600'>"
            f"● Database offline</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    sector = st.selectbox("Sector", SECTORS, key="sector")

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── Company 1 ─────────────────────────────────────────────────────────────
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
        help="Fiscal year of Company 1 report",
    )

    # Show cached KPI count for Company 1
    if db_online and company1:
        db1 = _db_lookup(company1, int(fy1))
        n1  = len(db1["kpis"])
        if n1 > 0:
            st.caption(
                f"✅ {n1} KPIs cached"
                + (" · revenue cached" if db1["revenue"] else "")
            )
        else:
            st.caption("⚪ Not in DB — pipeline will run automatically")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Company 2 ─────────────────────────────────────────────────────────────
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
        help="Fiscal year of Company 2 report",
    )

    # Show cached KPI count for Company 2
    if db_online and company2:
        db2 = _db_lookup(company2, int(fy2))
        n2  = len(db2["kpis"])
        if n2 > 0:
            st.caption(
                f"✅ {n2} KPIs cached"
                + (" · revenue cached" if db2["revenue"] else "")
            )
        else:
            st.caption("⚪ Not in DB — pipeline will run automatically")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Compare button ────────────────────────────────────────────────────────
    ready        = bool(company1 and company2 and company1 != company2)
    compare_btn  = st.button(
        "⚡  Compare",
        disabled = not ready,
        use_container_width = True,
    )

    if not ready and (company1 or company2):
        if company1 == company2:
            st.caption("⚠ Enter two different companies.")
        elif not company1:
            st.caption("⚠ Enter Company 1 name.")
        elif not company2:
            st.caption("⚠ Enter Company 2 name.")

    st.markdown("---")



# =============================================================================
# MAIN CONTENT AREA
# =============================================================================

tab_compare, = st.tabs(["📊 Comparison"])

with tab_compare:

    # ── Landing / idle state ──────────────────────────────────────────────────
    if "result" not in st.session_state and not compare_btn:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;padding:80px 40px;text-align:center">
            <div style="font-size:52px;margin-bottom:16px">🌿</div>
            <div style="font-size:28px;font-weight:800;color:#E8EAF0;
                        letter-spacing:-0.5px;margin-bottom:8px">
                ESG Competitive Intelligence
            </div>
            <div style="font-size:14px;color:#8B92A9;max-width:500px;
                        line-height:1.7;margin-bottom:24px">
                Enter two company names in the sidebar and click <strong>Compare</strong>.
                The system automatically searches for their BRSR/ESG reports,
                downloads, parses, and extracts ESG KPIs — no PDF upload needed.
            </div>
            <div style="font-size:13px;color:#6EE7B7;background:#064E3B;
                        border-radius:8px;padding:10px 20px;display:inline-block">
                ✓ DB-first · ✓ Auto ingestion · ✓ Regex + LLM extraction
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Pipeline trigger ──────────────────────────────────────────────────────
    if compare_btn and ready:
        # Create two status placeholders before starting (so they appear in order)
        st.markdown(
            f"### Running pipeline for **{company1} FY{fy1}** and **{company2} FY{fy2}**"
        )

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown(f"**{company1} FY{fy1}**")
            placeholder1 = st.empty()
        with col_s2:
            st.markdown(f"**{company2} FY{fy2}**")
            placeholder2 = st.empty()

        try:
            # Run both pipelines sequentially (parallel would need threading)
            data1 = run_company_pipeline(
                company_name        = company1,
                fy                  = int(fy1),
                sector              = sector,
                db_online           = db_online,
                llm_service         = llm_service,
                status_placeholder  = placeholder1,
            )
            data2 = run_company_pipeline(
                company_name        = company2,
                fy                  = int(fy2),
                sector              = sector,
                db_online           = db_online,
                llm_service         = llm_service,
                status_placeholder  = placeholder2,
            )

            # Warn if either company has no KPIs at all
            if not data1.kpi_records and not data2.kpi_records:
                st.error(
                    "No KPIs could be extracted for either company. "
                    "Check that the Tavily API key is set and the company names are correct."
                )
                st.stop()

            result = _build_benchmark(data1, data2, sector)
            result["log1"] = data1.log
            result["log2"] = data2.log
            result["data1"] = data1
            result["data2"] = data2
            result["sector"] = sector

            st.session_state["result"]   = result
            st.session_state["company1"] = company1
            st.session_state["company2"] = company2
            st.session_state["fy1"]      = fy1
            st.session_state["fy2"]      = fy2

        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            with st.expander("Full traceback"):
                st.code(traceback.format_exc())

    # ── Results rendering ─────────────────────────────────────────────────────
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

        # ── Header ────────────────────────────────────────────────────────────
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
                &nbsp;&nbsp;FY{fy1v} · FY{fy2v}
                &nbsp;·&nbsp;Intensity ratios (KPI / ₹Crore)
            </div>""", unsafe_allow_html=True)

        with hc2:
            pdf_bytes_out = _export_pdf(profiles, filtered, summary, _sector)
            st.download_button(
                "⬇ Download PDF",
                data            = pdf_bytes_out,
                file_name       = f"ESG_{c1n}_vs_{c2n}.pdf",
                mime            = "application/pdf",
                use_container_width = True,
            )

        if skipped:
            skipped_labels = [KPI_META.get(k, {}).get("label", k) for k in skipped]
            st.info(f"ℹ️ Excluded (unrealistic ratio): {', '.join(skipped_labels)}")

        if not filtered:
            st.warning(
                "No comparable KPIs after sanity filtering. "
                "This usually means extraction found only intensity rows "
                "(very small decimals) instead of absolute values."
            )
            # Still show extraction logs so the user can debug
            with st.expander("🔎 Extraction log"):
                for company_label, log_key in [(c1n, "log1"), (c2n, "log2")]:
                    log = result.get(log_key, [])
                    if log:
                        st.markdown(f"**{company_label}**")
                        for line in log:
                            st.code(line, language=None)
            st.stop()

        st.markdown("---")

        # ── Scorecard ─────────────────────────────────────────────────────────
        st.markdown('<div class="sec">📌 Overview</div>', unsafe_allow_html=True)
        wins_a = sum(1 for c in filtered if c.winner == label_a)
        wins_b = sum(1 for c in filtered if c.winner == label_b)
        leader = c1n if wins_a >= wins_b else c2n

        sc = st.columns(4)
        for col, (bcolor, blabel, bval, bsub) in zip(sc, [
            (C["blue"],  "KPIs Compared",   str(len(filtered)), f"of {len(report.comparisons)} extracted"),
            (C["ca"],    f"{c1n} FY{fy1v}",  str(wins_a),        "KPI wins"),
            (C["cb"],    f"{c2n} FY{fy2v}",  str(wins_b),        "KPI wins"),
            (C["green"], "Leader", leader, "More KPI wins"),
        ]):
            col.markdown(f"""
            <div class="card" style="border-top:3px solid {bcolor}">
                <div class="label">{blabel}</div>
                <div style="font-size:28px;font-weight:800;color:{C['text']};
                            white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
                    {bval}
                </div>
                <div style="font-size:12px;color:{C['sub']};margin-top:2px">{bsub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── Radar + donut ─────────────────────────────────────────────────────
        st.markdown('<div class="sec">📊 Performance Overview</div>', unsafe_allow_html=True)
        vc1, vc2 = st.columns([2, 1])
        with vc1:
            st.markdown("**Normalised Score Radar** — Higher = better")
            fig = _radar(filtered, label_a, label_b)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with vc2:
            st.markdown("**Win Distribution**")
            st.plotly_chart(
                _donut(filtered, label_a, label_b),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.markdown(f"""
            <div style="display:flex;gap:12px;justify-content:center;margin-top:6px">
                <div style="display:flex;align-items:center;gap:6px">
                    <div style="width:12px;height:12px;border-radius:3px;background:{C['ca']}"></div>
                    <span style="font-size:12px">{c1n}</span>
                </div>
                <div style="display:flex;align-items:center;gap:6px">
                    <div style="width:12px;height:12px;border-radius:3px;background:{C['cb']}"></div>
                    <span style="font-size:12px">{c2n}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── Gap bars ──────────────────────────────────────────────────────────
        st.markdown('<div class="sec">📉 Intensity Comparison</div>', unsafe_allow_html=True)
        st.caption("All KPIs normalised by ₹Crore revenue. Lower bar = better environmental performance.")
        gap = _gap_bars(filtered, label_a, label_b)
        if gap:
            st.plotly_chart(gap, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── Per-KPI cards ─────────────────────────────────────────────────────
        st.markdown('<div class="sec">🔍 KPI Detail</div>', unsafe_allow_html=True)
        for comp in filtered:
            meta = KPI_META.get(comp.kpi_name, {})
            vals = {l: v for l, v, _ in comp.entries}
            va, vb = vals.get(label_a), vals.get(label_b)
            winner_name = comp.winner.split(" FY")[0]
            wa = comp.winner == label_a

            st.markdown(f"""
            <div class="card">
                <div style="display:flex;justify-content:space-between;
                            align-items:flex-start;margin-bottom:10px">
                    <div>
                        <span style="font-size:16px;font-weight:700">
                            {meta.get('label', comp.display_name)}
                        </span>
                        <div style="font-size:11px;color:{C['sub']};margin-top:2px">
                            {meta.get('desc','')} · {meta.get('unit', comp.unit)}
                        </div>
                    </div>
                    <div style="text-align:right">
                        <span class="badge-green">★ {winner_name}</span>
                        <div style="font-size:12px;color:{C['sub']};margin-top:4px">
                            {comp.pct_gap:.1f}% gap
                        </div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            st.plotly_chart(
                _mini_bar(comp, label_a, label_b),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            mc1, mc2 = st.columns(2)
            for col, label_full, val, color, is_win, name in [
                (mc1, label_a, va, C["ca"], wa,      c1n),
                (mc2, label_b, vb, C["cb"], not wa,  c2n),
            ]:
                val_str = (
                    f"{val:.4e}" if (val and val < 0.001)
                    else (f"{val:.4f}" if val else "N/A")
                )
                col.markdown(f"""
                <div style="background:{C['surface']};border-radius:8px;
                            padding:10px 14px;border:1px solid {C['border']};
                            border-left:4px solid {color}">
                    <div style="font-size:11px;font-weight:600;color:{color}">
                        {name} {'★' if is_win else ''}
                    </div>
                    <div style="font-size:13px;color:{C['text']};margin-top:4px;
                                font-family:monospace">
                        {val_str} {meta.get('unit','')}
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        # ── AI Summary ────────────────────────────────────────────────────────
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec">💬 Summary</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="summary-box">{summary.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )

        # ── Methodology ───────────────────────────────────────────────────────
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        with st.expander("ℹ️ Methodology & Data Provenance"):
            st.markdown("""
**Intensity ratios** — Every KPI is divided by annual revenue (₹Crore) before
comparison, removing scale bias between companies of different sizes.

**Automated pipeline** — If a company is not in the DB, the system searches
for its BRSR/ESG report via Tavily, downloads the PDF, parses it with the
spatial chunker, and extracts KPIs using regex + batch LLM calls.

**Unit normalisation** — All values converted to canonical units:
energy → GJ, emissions → tCO₂e, water → KL, waste → MT.

**Sanity filtering** — KPIs where either company's ratio exceeds a plausible
ceiling are excluded to catch unit errors in source PDFs.

**DB-first** — KPI records already in the database are used without re-extracting.
New extractions are stored immediately for future use.

**Winner selection** — Lower intensity = better for all environmental KPIs.
Overall leader = company with more individual KPI wins.

**AI summary** — Gemini narrates from pre-computed verified ratios only.
Rule-based fallback when no API key is configured.
            """)

        # ── Extraction logs ───────────────────────────────────────────────────
        log1 = result.get("log1", [])
        log2 = result.get("log2", [])
        if log1 or log2:
            with st.expander("🔎 Extraction log"):
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
