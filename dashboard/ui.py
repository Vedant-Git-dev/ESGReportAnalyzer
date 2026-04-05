"""
dashboard/esg_dashboard.py

ESG Competitive Intelligence Dashboard — fully integrated with database.

DB Integration design:
  1. On load → check DB connection status and show indicator
  2. Sidebar → company name fields autocomplete from companies in DB
  3. Run comparison:
       a. Check DB for existing KPI records (DB-first, no redundant extraction)
       b. If PDF uploaded → extract only missing KPIs → store to DB
       c. If no PDF uploaded and company exists in DB → use cached data fully
  4. History tab → browse all past comparisons stored in DB
  5. All writes go through run_benchmark._db_store() / revenue_extractor.store_revenue()
     (same path as CLI) — no parallel write logic

What stays the same as before:
  - No raw KPI numbers shown on UI (only intensity ratios)
  - Sanity filter on unrealistic ratios
  - Radar, donut, gap-bar, per-KPI cards
  - PDF download via ReportLab
  - AI narrative summary

Run:
    cd /path/to/esg_pipeline
    streamlit run dashboard/esg_dashboard.py
"""
from __future__ import annotations

import io
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import plotly.graph_objects as go
import streamlit as st
import pandas as pd

# ── Page config (must be FIRST Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="ESG Intelligence · Benchmark",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Design tokens
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bg":       "#8E4D4D", "surface":  "#615EB1", "border":   "#D3D9E1",
    "text":     "#01060F", "sub":      "#CDDBEC",
    "green":    "#10B981", "blue":     "#A0BAE6",
    "amber":    "#F59E0B", "red":      "#EF4444",
    "ca":       "#4B84DE", "cb":       "#10B981",
    "grid":     "#993636",
    "font":     "Inter, 'Helvetica Neue', Arial, sans-serif",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{{font-family:{C['font']};background:{C['bg']};color:{C['text']};}}
.stApp{{background:{C['bg']};}}
[data-testid="stSidebar"]{{background:{C['surface']};border-right:1px solid {C['border']};}}
.stButton>button{{background:{C['blue']};color:#fff;border:none;border-radius:8px;
  padding:0.55rem 1.4rem;font-family:{C['font']};font-weight:600;font-size:14px;width:100%;
  letter-spacing:.02em;transition:background .2s;}}
.stButton>button:hover{{background:#2563EB;}}
.stButton>button:disabled{{background:{C['border']};color:{C['sub']};}}
[data-testid="stFileUploader"]{{border:1.5px dashed {C['border']};border-radius:10px;
  padding:8px;background:{C['surface']};}}
.stSelectbox>div>div,.stTextInput>div>div>input{{border-radius:8px;border:1px solid {C['border']};}}
hr{{border-color:{C['border']};margin:1.2rem 0;}}
.card{{background:{C['surface']};border:1px solid {C['border']};border-radius:12px;
  padding:18px 22px;margin-bottom:12px;}}
.label{{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;
  color:{C['sub']};margin-bottom:4px;}}
.badge-green{{background:#D1FAE5;color:#065F46;border-radius:20px;font-size:11px;
  font-weight:700;padding:3px 12px;letter-spacing:.04em;}}
.badge-blue{{background:#EFF6FF;color:#1D4ED8;border-radius:6px;font-size:11px;
  font-weight:600;padding:2px 9px;}}
.badge-amber{{background:#FEF3C7;color:#92400E;border-radius:6px;font-size:11px;
  font-weight:600;padding:2px 9px;}}
.sec{{font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;
  color:{C['sub']};margin-bottom:16px;padding-bottom:8px;border-bottom:2px solid {C['border']};}}
.summary-box{{background:{C['surface']};border-left:4px solid {C['blue']};
  border-radius:0 10px 10px 0;padding:20px 24px;line-height:1.75;font-size:14.5px;}}
.db-online{{color:#10B981;font-weight:600;font-size:12px;}}
.db-offline{{color:#EF4444;font-weight:600;font-size:12px;}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SECTORS = [
    "Information Technology", "Banking & Financial Services",
    "Energy & Utilities", "Pharmaceuticals & Healthcare",
    "Automotive & Manufacturing", "Fast-Moving Consumer Goods (FMCG)",
    "Chemicals & Materials", "Telecommunications",
    "Infrastructure & Real Estate", "Metals & Mining", "Other",
]

KPI_META = {
    "energy_consumption":  {"label":"Energy Intensity",     "unit":"GJ/Cr",  "max_ratio":1_000,  "desc":"Total energy per  Crore revenue"},
    "scope_1_emissions":   {"label":"Scope 1 GHG",          "unit":"tCO2e/Cr", "max_ratio":10,     "desc":"Direct GHG per  Crore revenue"},
    "scope_2_emissions":   {"label":"Scope 2 GHG",          "unit":"tCO2e/Cr", "max_ratio":10,     "desc":"Indirect GHG per  Crore revenue"},
    "total_ghg_emissions": {"label":"Total GHG",            "unit":"tCO2e/Cr", "max_ratio":20,     "desc":"Combined Scope 1+2 per  Crore revenue"},
    "water_consumption":   {"label":"Water Intensity",      "unit":"KL/Cr", "max_ratio":500,    "desc":"Water consumed per  Crore revenue"},
    "waste_generated":     {"label":"Waste Intensity",      "unit":"MT/Cr", "max_ratio":5,      "desc":"Waste generated per  Crore revenue"},
}

TARGET_KPI_NAMES = [
    "energy_consumption", "scope_1_emissions", "scope_2_emissions",
    "total_ghg_emissions", "water_consumption", "waste_generated",
]

# ─────────────────────────────────────────────────────────────────────────────
# DB layer — all calls wrapped so dashboard runs even when DB is down
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def _check_db() -> bool:
    """Check if PostgreSQL is reachable. Result cached for app lifetime."""
    try:
        from core.database import check_connection
        return check_connection()
    except Exception:
        return False


@st.cache_resource
def _get_companies_from_db() -> list[str]:
    """Return list of company names already in DB. Shown as autocomplete hints."""
    try:
        from core.database import get_db
        from models.db_models import Company
        with get_db() as db:
            rows = db.query(Company.name).filter(Company.is_active == True).order_by(Company.name).all()
            return [r[0] for r in rows]
    except Exception:
        return []


def _db_lookup(company_name: str, fy: int) -> dict:
    """
    Pull cached KPI records and revenue from DB for company+FY.
    Returns {"kpis": {kpi_name: {value,unit,method,confidence}}, "revenue": RevenueResult|None}
    """
    empty = {"kpis": {}, "revenue": None}
    try:
        from core.database import get_db
        from models.db_models import Company, Report, KPIRecord, KPIDefinition
        from services.revenue_extractor import RevenueResult
        with get_db() as db:
            company = (
                db.query(Company)
                .filter(Company.name.ilike(f"%{company_name}%"))
                .first()
            )
            if not company:
                return empty

            report = (
                db.query(Report)
                .filter(Report.company_id == company.id, Report.report_year == fy)
                .order_by(Report.created_at.desc())
                .first()
            )
            if not report:
                return empty

            # Revenue
            rev_val = getattr(report, "revenue_cr", None)
            cached_rev = None
            if rev_val:
                cached_rev = RevenueResult(
                    value_cr=float(rev_val),
                    raw_value=str(rev_val),
                    raw_unit=getattr(report, "revenue_unit", None) or "INR_Crore",
                    source=getattr(report, "revenue_source", None) or "db",
                    page_number=0, confidence=0.99, pattern_name="cached",
                )

            # KPI records — most recent non-null per KPI
            kpis: dict = {}
            for kpi_name in TARGET_KPI_NAMES:
                kdef = db.query(KPIDefinition).filter(KPIDefinition.name == kpi_name).first()
                if not kdef:
                    continue
                rec = (
                    db.query(KPIRecord)
                    .filter(
                        KPIRecord.company_id == company.id,
                        KPIRecord.kpi_definition_id == kdef.id,
                        KPIRecord.report_year == fy,
                        KPIRecord.normalized_value.isnot(None),
                    )
                    .order_by(KPIRecord.extracted_at.desc())
                    .first()
                )
                if rec:
                    kpis[kpi_name] = {
                        "value": rec.normalized_value,
                        "unit":  rec.unit or kdef.expected_unit,
                        "method": rec.extraction_method,
                        "confidence": rec.confidence or 0.9,
                    }
            return {"kpis": kpis, "revenue": cached_rev}
    except Exception as exc:
        st.warning(f"DB lookup failed: {exc}")
        return empty


def _db_ensure_schema() -> None:
    """Add revenue columns to reports table if missing."""
    try:
        from core.database import get_db
        from services.revenue_extractor import ensure_revenue_columns
        with get_db() as db:
            ensure_revenue_columns(db)
    except Exception:
        pass


def _db_upsert_company_report(
    company_name: str,
    sector: str,
    fy: int,
    pdf_path: Path,
) -> tuple[Optional[object], Optional[object]]:
    """
    Get-or-create Company + Report rows.
    Returns (company_orm, report_orm) or (None, None) on failure.
    """
    try:
        from core.database import get_db
        from models.db_models import Company, Report
        with get_db() as db:
            company = (
                db.query(Company)
                .filter(Company.name.ilike(f"%{company_name}%"))
                .first()
            )
            if not company:
                company = Company(name=company_name, sector=sector, country="India")
                db.add(company)
                db.flush()

            report = (
                db.query(Report)
                .filter(Report.company_id == company.id, Report.report_year == fy)
                .order_by(Report.created_at.desc())
                .first()
            )
            if not report:
                report = Report(
                    company_id=company.id,
                    report_year=fy,
                    report_type="BRSR",
                    file_path=str(pdf_path.resolve()),
                    status="downloaded",
                )
                db.add(report)
                db.flush()
            elif not report.file_path:
                report.file_path = str(pdf_path.resolve())
                report.status = "downloaded"
                db.flush()

            # Return plain IDs (not ORM objects — session will close)
            return str(company.id), str(report.id)
    except Exception as exc:
        st.warning(f"DB upsert failed: {exc}")
        return None, None


def _db_store_kpis(
    company_name: str,
    fy: int,
    kpi_records: dict,
    revenue_result,
) -> None:
    """Write new KPI records + revenue to DB. Skips exact duplicates."""
    try:
        from core.database import get_db
        from models.db_models import Company, Report, KPIRecord, KPIDefinition
        from services.revenue_extractor import store_revenue
        with get_db() as db:
            company = (
                db.query(Company).filter(Company.name.ilike(f"%{company_name}%")).first()
            )
            if not company:
                return
            report = (
                db.query(Report)
                .filter(Report.company_id == company.id, Report.report_year == fy)
                .order_by(Report.created_at.desc())
                .first()
            )
            if not report:
                return

            if revenue_result and getattr(report, "revenue_cr", None) is None:
                try:
                    store_revenue(report, revenue_result, db)
                except Exception:
                    pass

            for kpi_name, rec in kpi_records.items():
                kdef = db.query(KPIDefinition).filter(KPIDefinition.name == kpi_name).first()
                if not kdef:
                    continue
                exists = (
                    db.query(KPIRecord)
                    .filter(
                        KPIRecord.company_id == company.id,
                        KPIRecord.kpi_definition_id == kdef.id,
                        KPIRecord.report_year == fy,
                        KPIRecord.normalized_value == rec["value"],
                    )
                    .first()
                )
                if exists:
                    continue
                db.add(KPIRecord(
                    company_id=company.id,
                    report_id=report.id,
                    kpi_definition_id=kdef.id,
                    report_year=fy,
                    raw_value=str(rec["value"]),
                    normalized_value=rec["value"],
                    unit=rec["unit"],
                    extraction_method=rec["method"],
                    confidence=rec["confidence"],
                    is_validated=rec["confidence"] >= 0.85,
                    validation_notes="esg_dashboard",
                ))
    except Exception as exc:
        st.warning(f"DB store failed: {exc}")


def _db_history() -> pd.DataFrame:
    """
    Return a DataFrame of past extractions from DB for the History tab.
    Columns: Company, FY, Sector, KPIs Found, Extracted At
    """
    try:
        from core.database import get_db
        from models.db_models import Company, Report, KPIRecord
        from sqlalchemy import func
        with get_db() as db:
            rows = (
                db.query(
                    Company.name,
                    Company.sector,
                    Report.report_year,
                    Report.report_type,
                    func.count(KPIRecord.id).label("kpi_count"),
                    func.max(KPIRecord.extracted_at).label("last_extracted"),
                )
                .join(Report, Report.company_id == Company.id)
                .join(KPIRecord, KPIRecord.report_id == Report.id)
                .filter(KPIRecord.normalized_value.isnot(None))
                .group_by(Company.name, Company.sector, Report.report_year, Report.report_type)
                .order_by(func.max(KPIRecord.extracted_at).desc())
                .limit(100)
                .all()
            )
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame([
                {
                    "Company": r[0],
                    "Sector":  r[1] or "—",
                    "FY":      r[2],
                    "Type":    r[3],
                    "KPIs in DB": r[4],
                    "Last Extracted": str(r[5])[:16] if r[5] else "—",
                }
                for r in rows
            ])
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Extraction pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline_for_company(
    company_name: str,
    fy: int,
    sector: str,
    pdf_bytes: Optional[bytes],
    db_online: bool,
) -> dict:
    """
    DB-first pipeline for one company. Returns dict ready for build_company_profile().

    Steps:
      1. DB lookup (skip if force or DB offline)
      2. Identify missing KPIs
      3. Extract missing from PDF (if uploaded)
      4. Derive total_ghg from scope_1 + scope_2
      5. Store new records to DB
      6. Return merged kpi_records + revenue_result + page_texts
    """
    # from agents.extraction_agent import extract_from_pdf
    from run_benchmark import _derive_total_ghg
    from services.revenue_extractor import extract_revenue

    cached_kpis: dict = {}
    cached_rev = None
    log: list[str] = []

    # Step 1: DB lookup
    if db_online:
        db_data = _db_lookup(company_name, fy)
        cached_kpis = db_data["kpis"]
        cached_rev  = db_data["revenue"]
        if cached_kpis:
            log.append(f"DB cache: {len(cached_kpis)} KPIs loaded")
        if cached_rev:
            log.append(f"DB cache: revenue loaded")

    # Step 2: What's missing?
    extractable = [k for k in TARGET_KPI_NAMES if k != "total_ghg_emissions"]
    missing = [k for k in extractable if k not in cached_kpis]
    need_rev = cached_rev is None

    new_kpi_recs: dict = {}
    new_revenue = None
    page_texts: list[str] = []

    # Step 3: Extract from PDF if needed
    if pdf_bytes and (missing or need_rev):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = Path(tmp.name)

        try:
            # if missing or force:
                # extraction = extract_from_pdf(tmp_path, company_name, fy)
                # for kpi_name, raw in extraction.kpis.items():
                #     if force or kpi_name in missing:
                #         new_kpi_recs[kpi_name] = {
                #             "value": raw.value, "unit": raw.unit,
                #             "method": raw.method, "confidence": raw.confidence,
                #         }
                #         log.append(f"Extracted: {kpi_name} = {raw.value:,.2f} {raw.unit} [{raw.method}]")

            if need_rev:
                new_revenue = extract_revenue(tmp_path, fiscal_year_hint=fy)
                if new_revenue:
                    log.append(f"Revenue: {new_revenue.value_cr:,.0f} Cr [{new_revenue.pattern_name}]")

            import fitz
            doc = fitz.open(str(tmp_path))
            page_texts = [p.get_text() for p in doc]
            doc.close()
        finally:
            tmp_path.unlink(missing_ok=True)
    elif not pdf_bytes and missing:
        log.append(f"No PDF uploaded — {len(missing)} KPIs unavailable from DB")

    # Step 4: Merge + derive total_ghg
    merged = {**cached_kpis, **new_kpi_recs}
    ghg = _derive_total_ghg(merged)
    if ghg:
        merged["total_ghg_emissions"] = ghg
        log.append(f"Derived total_ghg = {ghg['value']:,.2f} tCO2e")

    # Step 5: Upsert company/report rows then store KPIs
    if db_online and pdf_bytes and (new_kpi_recs or new_revenue):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp2:
            tmp2.write(pdf_bytes)
            tmp2_path = Path(tmp2.name)
        try:
            _db_upsert_company_report(company_name, sector, fy, tmp2_path)
        finally:
            tmp2_path.unlink(missing_ok=True)
        to_store = dict(new_kpi_recs)
        if "total_ghg_emissions" not in cached_kpis and "total_ghg_emissions" in merged:
            to_store["total_ghg_emissions"] = merged["total_ghg_emissions"]
        _db_store_kpis(company_name, fy, to_store, new_revenue)
        log.append("Saved to DB")

    revenue_result = cached_rev or new_revenue
    return {
        "company_name":   company_name,
        "fy":             fy,
        "kpi_records":    merged,
        "revenue_result": revenue_result,
        "page_texts":     page_texts,
        "log":            log,
    }


def _build_profiles_and_report(data1: dict, data2: dict, sector: str) -> dict:
    """Build CompanyProfile objects and BenchmarkReport from pipeline output."""
    from services.benchmark import build_company_profile, compare_profiles
    from services.summary_generator import generate_summary
    from services.llm_service import LLMService
    from core.config import get_settings

    profiles = []
    for d in [data1, data2]:
        rev = d["revenue_result"]
        profile = build_company_profile(
            kpi_records=d["kpi_records"],
            revenue_cr=rev.value_cr if rev else 315322.0,
            revenue_source=rev.source if rev else "default",
            company_name=d["company_name"],
            fiscal_year=d["fy"],
            page_texts=d["page_texts"],
        )
        profiles.append(profile)

    report = compare_profiles(profiles)
    filtered = _filter_comparable(report.comparisons)

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


# ─────────────────────────────────────────────────────────────────────────────
# Sanity filter
# ─────────────────────────────────────────────────────────────────────────────

def _filter_comparable(comparisons) -> list:
    out = []
    for comp in comparisons:
        meta = KPI_META.get(comp.kpi_name, {})
        mx = meta.get("max_ratio")
        ok = all(
            (not mx or v <= mx) and v > 0
            for _, v, _ in comp.entries
        )
        if ok:
            out.append(comp)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────
_FONT = dict(family="Inter, Arial, sans-serif", size=12, color=C["text"])

def _hex_rgba(h: str, a: float) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[:2],16), int(h[2:4],16), int(h[4:],16)
    return f"rgba({r},{g},{b},{a})"


def _radar(filtered, la, lb):
    cats, sa, sb = [], [], []
    for comp in filtered:
        vals = {l: v for l, v, _ in comp.entries}
        va, vb = vals.get(la), vals.get(lb)
        if va is None or vb is None:
            continue
        cats.append(KPI_META.get(comp.kpi_name, {}).get("label", comp.display_name))
        total = va + vb
        sa.append(round(100*(1-va/total),1) if total else 50)
        sb.append(round(100*(1-vb/total),1) if total else 50)
    if len(cats) < 2:
        return None
    cats_c = cats + [cats[0]]; sa_c = sa + [sa[0]]; sb_c = sb + [sb[0]]
    fig = go.Figure()
    for name, scores, color in [(la.split(" FY")[0], sa_c, C["ca"]),
                                 (lb.split(" FY")[0], sb_c, C["cb"])]:
        fig.add_trace(go.Scatterpolar(
            r=scores, theta=cats_c, fill="toself", name=name,
            line=dict(color=color, width=2.5),
            fillcolor=_hex_rgba(color, 0.1),
            hovertemplate="<b>%{theta}</b><br>Score: %{r:.0f}<extra></extra>",
        ))
    fig.update_layout(
        polar=dict(bgcolor=C["bg"],
                   radialaxis=dict(visible=True, range=[0,100],
                                   tickfont=dict(size=9,color=C["sub"]),
                                   gridcolor=C["grid"], linecolor=C["border"]),
                   angularaxis=dict(tickfont=dict(size=11,color=C["text"]),
                                    gridcolor=C["grid"], linecolor=C["border"])),
        paper_bgcolor=C["bg"], height=360,
        margin=dict(l=40,r=40,t=40,b=20), font=_FONT,
        legend=dict(orientation="h", y=-0.08, font=dict(size=12)),
    )
    return fig


def _donut(filtered, la, lb):
    na, nb = la.split(" FY")[0], lb.split(" FY")[0]
    wa = sum(1 for c in filtered if c.winner == la)
    wb = sum(1 for c in filtered if c.winner == lb)
    fig = go.Figure(go.Pie(
        labels=[na, nb], values=[wa, wb], hole=0.62,
        marker=dict(colors=[C["ca"], C["cb"]], line=dict(color=C["bg"], width=3)),
        textinfo="label+percent", textfont=dict(size=12),
        hovertemplate="<b>%{label}</b><br>%{value} wins<extra></extra>",
    ))
    fig.update_layout(paper_bgcolor=C["bg"], height=260,
                      margin=dict(l=0,r=0,t=10,b=0), showlegend=False,
                      annotations=[dict(text=f"{wa+wb}<br><span style='font-size:10px'>KPIs</span>",
                                        x=0.5,y=0.5,font=dict(size=20,color=C["text"]),showarrow=False)],
                      font=_FONT)
    return fig


def _gap_bars(filtered, la, lb):
    labels_list, va_list, vb_list = [], [], []
    for comp in filtered:
        vals = {l: v for l, v, _ in comp.entries}
        va, vb = vals.get(la), vals.get(lb)
        if va is None or vb is None:
            continue
        labels_list.append(KPI_META.get(comp.kpi_name,{}).get("label", comp.display_name))
        va_list.append(va); vb_list.append(vb)
    if not labels_list:
        return None
    fig = go.Figure()
    for name, vals, color, pat in [(la.split(" FY")[0], va_list, C["ca"], ""),
                                    (lb.split(" FY")[0], vb_list, C["cb"], "/")]:
        fig.add_trace(go.Bar(
            name=name, x=vals, y=labels_list, orientation="h",
            marker=dict(color=color, line=dict(color=color, width=1), pattern_shape=pat),
            text=[f"{v:.3g}" for v in vals], textposition="outside",
            textfont=dict(size=10),
            hovertemplate="<b>%{y}</b><br>%{x:.4e}<extra></extra>",
        ))
    fig.update_layout(
        barmode="group", height=300,
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"], font=_FONT,
        xaxis=dict(title="Intensity per  Crore",showgrid=True,gridcolor=C["grid"],zeroline=False),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=160,r=60,t=20,b=40),
    )
    return fig


def _mini_bar(comp, la, lb):
    vals = {l: v for l, v, _ in comp.entries}
    va, vb = vals.get(la, 0), vals.get(lb, 0)
    meta = KPI_META.get(comp.kpi_name, {})
    na, nb = la.split(" FY")[0], lb.split(" FY")[0]
    fig = go.Figure()
    for name, val, color, pat in [(na, va, C["ca"],""), (nb, vb, C["cb"],"/")]:
        fig.add_trace(go.Bar(
            name=name, x=[val], y=[meta.get("label","")], orientation="h",
            marker=dict(color=color, line=dict(color=color,width=1.5), pattern_shape=pat),
            text=[f"{val:.3g}"], textposition="outside", textfont=dict(size=11,color=C["text"]),
            hovertemplate=f"<b>{name}</b><br>{val:.4e} {meta.get('unit','')}<extra></extra>",
        ))
    fig.update_layout(
        barmode="group", height=110, showlegend=False,
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"], font=_FONT,
        yaxis=dict(visible=False),
        xaxis=dict(showgrid=True,gridcolor=C["grid"],zeroline=False,showticklabels=False),
        margin=dict(l=10,r=60,t=8,b=8),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PDF export
# ─────────────────────────────────────────────────────────────────────────────

def _export_pdf(profiles, filtered, summary: str, sector: str) -> bytes:
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
    BLUE  = colors.HexColor("#3B82F6")
    GREEN = colors.HexColor("#10B981")
    GRAY  = colors.HexColor("#64748B")
    LIGHT = colors.HexColor("#F8F9FB")
    BDR   = colors.HexColor("#E2E8F0")
    BLK   = colors.HexColor("#1A202C")
    ss    = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=ss["Title"], fontSize=22, textColor=BLK,
                            spaceAfter=4, leading=28, fontName="Helvetica-Bold")
    sub   = ParagraphStyle("s", parent=ss["Normal"], fontSize=11, textColor=GRAY,
                            spaceAfter=14, leading=16)
    h2    = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=13, textColor=BLK,
                            spaceBefore=16, spaceAfter=8, fontName="Helvetica-Bold")
    body  = ParagraphStyle("b", parent=ss["Normal"], fontSize=10, textColor=BLK,
                            leading=16, spaceAfter=8)
    note  = ParagraphStyle("n", parent=ss["Normal"], fontSize=9, textColor=GRAY, leading=14)

    labels = [f"{p.company_name} FY{p.fiscal_year}" for p in profiles]
    story  = []
    story.append(Paragraph("ESG Competitive Intelligence Report", title))
    story.append(Paragraph(f"{labels[0]} vs {labels[1]}", sub))
    story.append(Paragraph(f"Sector: {sector}", note))
    story.append(HRFlowable(width="100%", thickness=1, color=BDR, spaceAfter=14))

    story.append(Paragraph("Methodology", h2))
    story.append(Paragraph(
        "All metrics are intensity ratios (KPI ÷ 1 Crore revenue). "
        "KPIs with unrealistic ratios are excluded automatically. "
        "No absolute values are shown.", body))
    story.append(Spacer(1, 8))

    story.append(Paragraph("KPI Intensity Comparison", h2))
    tdata = [["Metric", "Unit", labels[0], labels[1], "Gap", "Leader"]]
    for comp in filtered:
        meta = KPI_META.get(comp.kpi_name, {})
        vals = {l: v for l, v, _ in comp.entries}
        v0, v1 = vals.get(labels[0]), vals.get(labels[1])
        def fmt(v): return f"{v:.2e}" if (v and v < 0.001) else (f"{v:.4f}" if v else "N/A")
        tdata.append([
            f"{meta.get('icon','')} {meta.get('label', comp.display_name)}",
            meta.get("unit", comp.unit),
            fmt(v0), fmt(v1),
            f"{comp.pct_gap:.1f}%",
            comp.winner.split(" FY")[0],
        ])
    cw = [5.2*cm, 2.3*cm, 2.8*cm, 2.8*cm, 1.6*cm, 2.5*cm]
    tbl = Table(tdata, colWidths=cw, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),BLUE), ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,0),9),
        ("BOTTOMPADDING",(0,0),(-1,0),8), ("TOPPADDING",(0,0),(-1,0),8),
        ("FONTSIZE",(0,1),(-1,-1),9), ("TOPPADDING",(0,1),(-1,-1),6),
        ("BOTTOMPADDING",(0,1),(-1,-1),6),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LIGHT]),
        ("GRID",(0,0),(-1,-1),0.5,BDR),
        ("ALIGN",(2,0),(-1,-1),"CENTER"), ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TEXTCOLOR",(5,1),(5,-1),GREEN), ("FONTNAME",(5,1),(5,-1),"Helvetica-Bold"),
    ]))
    story.append(tbl); story.append(Spacer(1,16))

    story.append(Paragraph("AI-Generated Narrative Summary", h2))
    for para in summary.split("\n\n"):
        para = para.strip()
        if para:
            story.append(Paragraph(para, body))

    story.append(Spacer(1,12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BDR))
    story.append(Spacer(1,6))
    story.append(Paragraph(
        "Generated by ESG Competitive Intelligence Pipeline. "
        "Data extracted from uploaded BRSR/Annual reports.", note))
    doc.build(story)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
db_online = _check_db()
known_companies = _get_companies_from_db() if db_online else []

if db_online:
    _db_ensure_schema()

with st.sidebar:
    st.markdown("""
    <div style="padding:0 0 14px">
        <div style="font-size:22px;font-weight:800;color:#1A202C;letter-spacing:-0.5px">
            🌿 ESG Intel
        </div>
        <div style="font-size:12px;color:#64748B;margin-top:2px">
            Competitive Benchmarking
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    sector = st.selectbox("Sector", SECTORS, key="sector")

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # Company 1
    st.markdown(f'<div class="label" style="color:{C["ca"]}">Company 1</div>',
                unsafe_allow_html=True)
    company1 = st.text_input("c1name", placeholder="Company name",
                              label_visibility="collapsed", key="c1_name")

    fy1 = st.number_input("FY1", min_value=2015, max_value=2030, value=2025,
                           label_visibility="collapsed", key="c1_fy",
                           help="Fiscal year of Company 1 report")
    pdf1 = st.file_uploader("Upload PDF — Company 1", type=["pdf"], key="pdf1")

    # DB status for company 1
    if db_online and company1:
        db1 = _db_lookup(company1, int(fy1))
        n1 = len(db1["kpis"])
        rev1_ok = db1["revenue"] is not None
        if n1 > 0:
            st.caption(f"✅ {n1} KPIs + {'revenue' if rev1_ok else 'no revenue'} in DB")
        else:
            st.caption("⚪ Not in DB — upload PDF to extract")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Company 2
    st.markdown(f'<div class="label" style="color:{C["cb"]}">Company 2</div>',
                unsafe_allow_html=True)
    company2 = st.text_input("c2name", placeholder="Company name",
                              label_visibility="collapsed", key="c2_name")
    fy2 = st.number_input("FY2", min_value=2015, max_value=2030, value=2024,
                           label_visibility="collapsed", key="c2_fy",
                           help="Fiscal year of Company 2 report")
    pdf2 = st.file_uploader("Upload PDF — Company 2", type=["pdf"], key="pdf2")

    if db_online and company2:
        db2 = _db_lookup(company2, int(fy2))
        n2 = len(db2["kpis"])
        rev2_ok = db2["revenue"] is not None
        if n2 > 0:
            st.caption(f"✅ {n2} KPIs + {'revenue' if rev2_ok else 'no revenue'} in DB")
        else:
            st.caption("⚪ Not in DB — upload PDF to extract")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # force = st.checkbox("Force re-extract (skip DB cache)", value=False)

    ready = bool(company1 and company2)
    # Can run if both companies in DB OR both PDFs uploaded
    c1_in_db = db_online and bool(known_companies and any(
        company1.lower() in c.lower() for c in known_companies))
    c2_in_db = db_online and bool(known_companies and any(
        company2.lower() in c.lower() for c in known_companies))
    has_data1 = bool(pdf1) or c1_in_db
    has_data2 = bool(pdf2) or c2_in_db

    compare_btn = st.button(
        "⚡  Compare",
        disabled=not (ready and has_data1 and has_data2),
        use_container_width=True,
    )
    if ready and not (has_data1 and has_data2):
        if not has_data1:
            st.caption(f"⚠ {company1}: not in DB — upload PDF")
        if not has_data2:
            st.caption(f"⚠ {company2}: not in DB — upload PDF")

    st.markdown("---")
    st.caption("Ratios: KPI ÷  Crore revenue · Lower = better")


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_compare, = st.tabs(["📊 Comparison"])   

# ── Comparison tab ────────────────────────────────────────────────────────────
with tab_compare:

    # Landing / idle state
    if "result" not in st.session_state and not compare_btn:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;padding:80px 40px;text-align:center">
            <div style="font-size:52px;margin-bottom:16px">🌿</div>
            <div style="font-size:28px;font-weight:800;color:#1A202C;
                        letter-spacing:-0.5px;margin-bottom:8px">
                ESG Competitive Intelligence
            </div>
            <div style="font-size:14px;color:#64748B;max-width:460px;
                        line-height:1.7;margin-bottom:36px">
                DB-first benchmarking: if a company's ESG data is already in the
                database, no PDF upload is needed. Upload only when extracting new data.
            </div>
        </div>""", unsafe_allow_html=True)


    # Trigger
    if compare_btn and ready:
        with st.spinner(f"Running pipeline for {company1} and {company2}…"):
            try:
                pdf1_bytes = pdf1.read() if pdf1 else None
                pdf2_bytes = pdf2.read() if pdf2 else None

                data1 = _run_pipeline_for_company(
                    company1, int(fy1), sector, pdf1_bytes, db_online)
                data2 = _run_pipeline_for_company(
                    company2, int(fy2), sector, pdf2_bytes, db_online)

                result = _build_profiles_and_report(data1, data2, sector)
                result["log1"] = data1["log"]
                result["log2"] = data2["log"]
                result["sector"] = sector

                st.session_state["result"]   = result
                st.session_state["company1"] = company1
                st.session_state["company2"] = company2
                st.session_state["fy1"]      = fy1
                st.session_state["fy2"]      = fy2
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())

    # Results
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

        # Header row
        hc1, hc2 = st.columns([3, 1])
        with hc1:
            st.markdown(f"""
            <div style="margin-bottom:4px">
                <span style="font-size:24px;font-weight:800;color:#1A202C">{c1n}</span>
                <span style="font-size:16px;color:#64748B;margin:0 10px">vs</span>
                <span style="font-size:24px;font-weight:800;color:#1A202C">{c2n}</span>
            </div>
            <div style="font-size:13px;color:#64748B">
                <span class="badge-blue">{_sector}</span>
                &nbsp;&nbsp;FY{fy1v} · FY{fy2v}
                &nbsp;·&nbsp; Intensity ratios (KPI /  Crore)
            </div>""", unsafe_allow_html=True)
        with hc2:
            pdf_bytes_out = _export_pdf(profiles, filtered, summary, _sector)
            st.download_button(
                "⬇ Download PDF",
                data=pdf_bytes_out,
                file_name=f"ESG_{c1n}_vs_{c2n}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        if skipped:
            skipped_labels = [KPI_META.get(k,{}).get("label",k) for k in skipped]
            st.info(f"ℹ️ Excluded (unrealistic ratio): {', '.join(skipped_labels)}")

        if not filtered:
            st.warning("No comparable KPIs after sanity filtering.")
            st.stop()

        st.markdown("---")

        # Scorecard
        st.markdown('<div class="sec">📌 Overview</div>', unsafe_allow_html=True)
        wins_a = sum(1 for c in filtered if c.winner == label_a)
        wins_b = sum(1 for c in filtered if c.winner == label_b)
        leader = c1n if wins_a >= wins_b else c2n

        sc = st.columns(4)
        for col, (bcolor, blabel, bval, bsub) in zip(sc, [
            (C["blue"],  "KPIs Compared",          str(len(filtered)),   f"of {len(report.comparisons)} extracted"),
            (C["ca"],    f"{c1n} FY{fy1v}",         str(wins_a),          "KPI wins"),
            (C["cb"],    f"{c2n} FY{fy2v}",         str(wins_b),          "KPI wins"),
            (C["green"], "Higher performance across compared KPIs", leader, "More KPI wins"),
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

        # Radar + donut
        st.markdown('<div class="sec">📊 Performance Overview</div>', unsafe_allow_html=True)
        vc1, vc2 = st.columns([2, 1])
        with vc1:
            st.markdown("**Normalised Score Radar** — Higher = better")
            fig = _radar(filtered, label_a, label_b)
            if fig:
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})
        with vc2:
            st.markdown("**Win Distribution**")
            st.plotly_chart(_donut(filtered, label_a, label_b),
                            use_container_width=True, config={"displayModeBar": False})
            st.markdown(f"""
            <div style="display:flex;gap:12px;justify-content:center;margin-top:6px">
                <div style="display:flex;align-items:center;gap:6px">
                    <div style="width:12px;height:12px;border-radius:3px;
                                background:{C['ca']}"></div>
                    <span style="font-size:12px">{c1n}</span>
                </div>
                <div style="display:flex;align-items:center;gap:6px">
                    <div style="width:12px;height:12px;border-radius:3px;
                                background:{C['cb']}"></div>
                    <span style="font-size:12px">{c2n}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Gap bars
        st.markdown('<div class="sec">📉 Intensity Comparison</div>', unsafe_allow_html=True)
        st.caption("All KPIs normalised by  Crore revenue. Lower bar = better environmental performance.")
        gap = _gap_bars(filtered, label_a, label_b)
        if gap:
            st.plotly_chart(gap, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Per-KPI cards
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
                            {meta.get('icon','')} {meta.get('label', comp.display_name)}
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

            st.plotly_chart(_mini_bar(comp, label_a, label_b),
                            use_container_width=True, config={"displayModeBar": False})

            mc1, mc2 = st.columns(2)
            for col, label_full, val, color, is_win, name in [
                (mc1, label_a, va, C["ca"], wa,  c1n),
                (mc2, label_b, vb, C["cb"], not wa, c2n),
            ]:
                val_str = f"{val:.4e}" if (val and val < 0.001) else (f"{val:.4f}" if val else "N/A")
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

        # AI Summary
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec">Summary</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="summary-box">{summary.replace(chr(10),"<br>")}</div>',
            unsafe_allow_html=True)

        # Methodology
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        with st.expander("ℹ️ Methodology & Data Provenance"):
            st.markdown("""
**Intensity ratios** — Every KPI is divided by annual revenue ( Crore) before comparison,
removing scale bias between companies of different sizes.

**Unit normalisation** — All values converted to canonical units: energy → GJ,
emissions → tCO₂e, water → KL, waste → MT.

**Sanity filtering** — KPIs where either company's ratio exceeds a plausible ceiling
(energy >1,000 GJ/Cr; GHG >500 tCO₂e/Cr; water >5,000 KL/Cr; waste >200 MT/Cr)
are excluded. This catches unit errors in source PDFs.

**DB-first** — KPI records already in the database are used without re-extracting.
New extractions are stored immediately for future use.

**Winner selection** — Lower intensity = better for all five environmental KPIs.
Overall leader = company with more individual KPI wins.

**AI summary** — Gemini narrates from pre-computed verified ratios only.
Rule-based fallback when no API key is configured.
            """)

        # Extraction log
        log1 = result.get("log1", [])
        log2 = result.get("log2", [])
        if log1 or log2:
            with st.expander("🔎 Extraction log"):
                if log1:
                    st.markdown(f"**{c1n} FY{fy1v}**")
                    for line in log1:
                        st.code(line, language=None)
                if log2:
                    st.markdown(f"**{c2n} FY{fy2v}**")
                    for line in log2:
                        st.code(line, language=None)


# # # ── History tab ───────────────────────────────────────────────────────────────
# # with tab_history:
# #     st.markdown('<div class="sec">🗂 Extraction History</div>', unsafe_allow_html=True)

# #     if not db_online:
# #         st.warning("Database is offline — history unavailable.")
# #     else:
# #         df = _db_history()
# #         if df.empty:
# #             st.info("No extractions in database yet. Run a comparison to populate.")
# #         else:
# #             st.caption(f"{len(df)} company-year records in database")

# #             # Summary metrics
# #             hm = st.columns(3)
# #             hm[0].metric("Total Companies", df["Company"].nunique())
# #             hm[1].metric("Total Reports",   len(df))
# #             hm[2].metric("Avg KPIs / Report", f"{df['KPIs in DB'].mean():.1f}")

#             st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

#             # Searchable table
#             search = st.text_input("🔍 Filter by company name", placeholder="Type to filter…",
#                                    label_visibility="collapsed")
#             if search:
#                 df = df[df["Company"].str.contains(search, case=False, na=False)]

#             st.dataframe(
#                 df.style.set_properties(**{
#                     "font-family": "Inter, sans-serif",
#                     "font-size": "13px",
#                 }),
#                 use_container_width=True,
#                 hide_index=True,
#             )

#             # KPIs per company bar
#             kpi_agg = df.groupby("Company")["KPIs in DB"].max().reset_index()
#             if len(kpi_agg) > 1:
#                 fig_hist = go.Figure(go.Bar(
#                     x=kpi_agg["Company"],
#                     y=kpi_agg["KPIs in DB"],
#                     marker_color=C["ca"],
#                     text=kpi_agg["KPIs in DB"],
#                     textposition="outside",
#                     textfont=dict(size=11),
#                 ))
#                 fig_hist.update_layout(
#                     title="KPI Records per Company",
#                     paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
#                     height=280, font=_FONT,
#                     xaxis=dict(showgrid=False),
#                     yaxis=dict(gridcolor=C["grid"], zeroline=False),
#                     margin=dict(l=20, r=20, t=36, b=20),
#                 )
#                 st.plotly_chart(fig_hist, use_container_width=True,
#                                 config={"displayModeBar": False})