"""
esg_bench/run_benchmark.py

Terminal CLI — DB-first ESG benchmarking pipeline.

Flow per company:
  1. DB lookup   — check kpi_records + reports.revenue_cr
  2. PDF extract — only for what is missing (block-regex → LLM)
  3. DB store    — persist new results (append-only, skip duplicates)
  4. Normalise   — canonical units via services/normalizer.py
  5. Ratios      — KPI / revenue (reported ratio from PDF if available)
  6. Compare     — services/benchmark.py
  7. Summary     — services/summary_generator.py (LLM or rule-based)

Usage:
    python -m esg_bench.run_benchmark \
        --pdf1 /path/infosys.pdf  --company1 "Infosys"  --fy1 2025 \
        --pdf2 /path/tcs.pdf      --company2 "TCS"       --fy2 2024 \
        [--force]   # skip DB cache, re-extract from PDF
        [--no-llm]  # disable LLM calls entirely

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPATIBILITY FIXES applied over the original version
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[FIX 1] TARGET_KPI_NAMES was referenced but never defined.
        Added as a module-level constant listing all 9 active KPI names.

[FIX 2] extract_from_pdf() was called but never existed.
        Added the function. It bridges the PDF path → DB pipeline by:
          a) Ensuring company + report rows exist in the DB (upsert).
          b) Running ParseOrchestrator if no parse cache exists.
          c) Calling ExtractionAgent.extract_all(report_id, db, kpi_names)
             which is the actual extraction entry point.
          d) Returning an ExtractionResult wrapping a dict of KPIResult objects.

[FIX 3] KPIResult namedtuple was referenced (.value, .unit, .method,
        .confidence, .page, .raw_str) but ExtractedKPI uses different
        field names (.normalized_value, .extraction_method, .source_chunk_id).
        Added KPIResult dataclass that maps ExtractedKPI fields correctly.

[FIX 4] ExtractionResult wrapper added (.kpis dict) so process_company's
        `extraction.kpis.items()` loop works without change.

[FIX 5] Module-level `agent = ExtractionAgent()` was a singleton that was
        never actually called (extract_from_pdf didn't exist).  Moved agent
        instantiation inside extract_from_pdf so it is created on demand and
        the module imports cleanly even without a DB connection.

[FIX 6] report.revenue_cr is accessed in _db_store but does not exist on the
        Report model.  ensure_revenue_columns() is called at startup (already
        present) and _db_store now guards the attribute access with getattr.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.logging_config import get_logger
from services.normalizer import normalize, NormalizationError, CANONICAL
from services.benchmark import build_company_profile, compare_profiles, print_report
from services.summary_generator import generate_summary
from services.revenue_extractor import (
    extract_revenue, store_revenue, get_cached_revenue,
    ensure_revenue_columns, RevenueResult,
)

logger = get_logger(__name__)
W = 72  # terminal ruler width

# ─────────────────────────────────────────────────────────────────────────────
# FIX 1 — TARGET_KPI_NAMES was referenced but never defined
# ─────────────────────────────────────────────────────────────────────────────

TARGET_KPI_NAMES: list[str] = [
    "scope_1_emissions",
    "scope_2_emissions",
    "total_ghg_emissions",
    # "energy_consumption",
    # "renewable_energy_percentage",
    # "water_consumption",
    "waste_generated",
    # "employee_count",
    # "women_in_workforce_percentage",
    # "csr_spend",
]

# ─────────────────────────────────────────────────────────────────────────────
# FIX 3 — KPIResult dataclass bridges ExtractedKPI field names to what
#          process_company expects (.value, .unit, .method, .confidence,
#          .page, .raw_str)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KPIResult:
    """
    Thin adapter over ExtractedKPI.

    process_company accesses:  raw.value / raw.unit / raw.method /
                               raw.confidence / raw.page / raw.raw_str

    ExtractedKPI provides:     .normalized_value / .unit / .extraction_method /
                               .confidence / .source_chunk_id / .raw_value

    This dataclass normalises the field names so process_company needs
    no changes.
    """
    value:      float
    unit:       str
    method:     str          # "regex" | "llm" | "not_found"
    confidence: float
    page:       Optional[int]   # best-effort; None when not available
    raw_str:    str             # the raw string as found in the PDF


# ─────────────────────────────────────────────────────────────────────────────
# FIX 4 — ExtractionResult wrapper so extraction.kpis.items() works
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    """Return type of extract_from_pdf()."""
    kpis: dict[str, KPIResult] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# FIX 2 — extract_from_pdf() bridges PDF path → DB pipeline → ExtractionAgent
# ─────────────────────────────────────────────────────────────────────────────

def extract_from_pdf(
    pdf_path: Path,
    company_name: str,
    fy: int,
    kpi_names: Optional[list[str]] = None,
) -> ExtractionResult:
    """
    Full extraction bridge:

      1. Ensure company row exists in DB (get-or-create).
      2. Ensure report row exists for (company, fy) and points to pdf_path.
         If the report was never downloaded through the normal pipeline,
         we register it manually with status="downloaded".
      3. Run ParseOrchestrator if no parse cache exists for this report.
      4. Call ExtractionAgent.extract_all(report_id, db, kpi_names).
      5. Convert list[ExtractedKPI] → ExtractionResult(.kpis dict of KPIResult).

    Why this function is necessary
    ───────────────────────────────
    ExtractionAgent.extract_all() requires:
      - A Report row with status in ("downloaded", "parsed", "extracted")
      - A ParsedDocument cache row with DocumentChunk rows
      - An active SQLAlchemy Session

    run_benchmark receives a raw PDF file path.  This function creates the
    minimal DB scaffolding so the extraction agent can run, without requiring
    the user to manually run `python main.py ingest / parse` first.
    """
    from core.database import get_db
    from models.db_models import Company, Report, ParsedDocument
    from models.schemas import CompanyCreate
    from services.parse_orchestrator import ParseOrchestrator
    from agents.extraction_agent import ExtractionAgent  # FIX 5: lazy import

    result = ExtractionResult()

    with get_db() as db:
        # ── Step A: ensure company ────────────────────────────────────────
        company = (
            db.query(Company)
            .filter(Company.name.ilike(f"%{company_name}%"))
            .first()
        )
        if not company:
            company = Company(name=company_name, country="India")
            db.add(company)
            db.flush()
            logger.info("extract_from_pdf.company_created", name=company_name)

        # ── Step B: ensure report row ─────────────────────────────────────
        report = (
            db.query(Report)
            .filter(
                Report.company_id == company.id,
                Report.report_year == fy,
            )
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
            logger.info("extract_from_pdf.report_registered",
                        company=company_name, fy=fy, path=str(pdf_path))
        elif not report.file_path:
            # Report row existed but had no file — update it
            report.file_path = str(pdf_path.resolve())
            if report.status not in ("downloaded", "parsed", "extracted"):
                report.status = "downloaded"
            db.flush()

        report_id = report.id

        # ── Step C: parse if not already cached ───────────────────────────
        has_cache = (
            db.query(ParsedDocument.id)
            .filter(ParsedDocument.report_id == report_id)
            .first()
        ) is not None

        if not has_cache:
            logger.info("extract_from_pdf.parsing", report_id=str(report_id))
            try:
                orchestrator = ParseOrchestrator()
                orchestrator.run(report_id=report_id)
            except Exception as exc:
                logger.error("extract_from_pdf.parse_failed", error=str(exc))
                return result

        # ── Step D: extract KPIs ──────────────────────────────────────────
        agent = ExtractionAgent()
        extracted_list = agent.extract_all(
            report_id=report_id,
            db=db,
            kpi_names=kpi_names or TARGET_KPI_NAMES,
        )

    # ── Step E: convert list[ExtractedKPI] → ExtractionResult ────────────
    # FIX 3: map ExtractedKPI field names → KPIResult field names
    for ext in extracted_list:
        if ext.normalized_value is None:
            continue  # skip not-found KPIs; process_company handles the gap

        result.kpis[ext.kpi_name] = KPIResult(
            value=ext.normalized_value,          # ExtractedKPI.normalized_value → KPIResult.value
            unit=ext.unit or "",
            method=ext.extraction_method,        # ExtractedKPI.extraction_method → KPIResult.method
            confidence=ext.confidence or 0.0,
            page=None,                           # source_chunk_id is a UUID, not a page; set None
            raw_str=ext.raw_value or str(ext.normalized_value),  # ExtractedKPI.raw_value → KPIResult.raw_str
        )

    logger.info(
        "extract_from_pdf.complete",
        company=company_name, fy=fy,
        found=len(result.kpis),
        missing=[k for k in TARGET_KPI_NAMES if k not in result.kpis],
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers — all wrapped so pipeline runs even when DB is down
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_revenue_cols() -> None:
    """Run ALTER TABLE ... ADD COLUMN IF NOT EXISTS for revenue columns."""
    try:
        from core.database import get_db
        with get_db() as db:
            ensure_revenue_columns(db)
    except Exception as exc:
        logger.warning("run_benchmark.migration_failed", error=str(exc))


def _db_lookup(company_name: str, fy: int) -> dict:
    """
    Check DB for cached KPI records and revenue.
    Returns {"kpis": {name: {...}}, "revenue": RevenueResult|None}
    """
    empty: dict = {"kpis": {}, "revenue": None}
    try:
        from core.database import get_db
        from models.db_models import Company, Report, KPIRecord, KPIDefinition
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
                .filter(
                    Report.company_id == company.id,
                    Report.report_year == fy,
                )
                .order_by(Report.created_at.desc())
                .first()
            )
            if not report:
                return empty

            cached_rev = get_cached_revenue(report)

            kpis: dict = {}
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
                        "value":      rec.normalized_value,
                        "unit":       rec.unit or kdef.expected_unit,
                        "method":     rec.extraction_method,
                        "confidence": rec.confidence or 0.9,
                    }
            return {"kpis": kpis, "revenue": cached_rev}

    except Exception as exc:
        logger.warning("run_benchmark.db_lookup_failed", error=str(exc))
        return empty


def _db_store(
    company_name: str,
    fy: int,
    new_kpis: dict,
    revenue_result: Optional[RevenueResult],
) -> None:
    """Persist extracted KPIs + revenue. Skips already-stored records."""
    try:
        from core.database import get_db
        from models.db_models import Company, Report, KPIRecord, KPIDefinition
        with get_db() as db:
            company = (
                db.query(Company)
                .filter(Company.name.ilike(f"%{company_name}%"))
                .first()
            )
            if not company:
                return

            report = (
                db.query(Report)
                .filter(
                    Report.company_id == company.id,
                    Report.report_year == fy,
                )
                .order_by(Report.created_at.desc())
                .first()
            )
            if not report:
                return

            # FIX 6: revenue_cr may not exist on the model yet — guard with getattr
            if revenue_result and getattr(report, "revenue_cr", None) is None:
                store_revenue(report, revenue_result, db)

            # KPI records — append-only, skip exact duplicates
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
                    validation_notes="extracted by esg_bench",
                ))

    except Exception as exc:
        logger.warning("run_benchmark.db_store_failed", error=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Per-company pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_company(
    pdf_path: Path,
    company_name: str,
    fy: int,
    force: bool,
    llm_service=None,
):
    """
    Full pipeline for one company. Returns CompanyProfile for benchmarking.
    """
    from services.benchmark import CompanyProfile

    print(f"\n{'─' * W}")
    print(f"  {company_name}  FY{fy}")
    print(f"{'─' * W}")

    # ── Step 1: DB lookup ──────────────────────────────────────────────────────
    cached_kpis: dict = {}
    cached_rev:  Optional[RevenueResult] = None

    if not force:
        print("  [1/4] Checking DB cache...")
        db_data     = _db_lookup(company_name, fy)
        cached_kpis = db_data["kpis"]
        cached_rev  = db_data["revenue"]
        if cached_kpis:
            print(f"        {len(cached_kpis)} KPIs in DB: {list(cached_kpis)}")
        else:
            print("        No cached KPIs")
        if cached_rev:
            print(f"        Revenue in DB: ₹{cached_rev.value_cr:,.0f} Cr")
        else:
            print("        No cached revenue")
    else:
        print("  [1/4] --force: skipping DB")

    # ── Step 2: PDF extraction for missing data ────────────────────────────────
    missing_kpis = [k for k in TARGET_KPI_NAMES if k not in cached_kpis]
    need_revenue = cached_rev is None

    new_kpi_recs: dict = {}
    new_revenue:  Optional[RevenueResult] = None

    if missing_kpis or need_revenue:
        needs = []
        if missing_kpis: needs.append(f"KPIs {missing_kpis}")
        if need_revenue:  needs.append("revenue")
        print(f"  [2/4] PDF extraction: {' + '.join(needs)}")
        print(f"        {pdf_path}")

        # KPI extraction — only request the missing ones to save API cost
        if missing_kpis:
            extraction = extract_from_pdf(pdf_path, company_name, fy,
                                          kpi_names=missing_kpis)
            for kpi_name, raw in extraction.kpis.items():
                if kpi_name not in cached_kpis:
                    new_kpi_recs[kpi_name] = {
                        "value":      raw.value,
                        "unit":       raw.unit,
                        "method":     raw.method,
                        "confidence": raw.confidence,
                    }
                    tag = "✓"
                    print(f"        {tag} {kpi_name}: {raw.raw_str} {raw.unit} "
                          f"[{raw.method} {raw.confidence:.2f}]")
            for k in missing_kpis:
                if k not in new_kpi_recs and k not in cached_kpis:
                    print(f"        ✗ {k}: not found")

        # Revenue extraction: regex → LLM → validation
        if need_revenue:
            new_revenue = extract_revenue(
                pdf_path=pdf_path,
                fiscal_year_hint=fy,
                llm_service=llm_service,
            )
            if new_revenue:
                print(f"        ✓ revenue: ₹{new_revenue.value_cr:,.0f} Cr  "
                      f"p{new_revenue.page_number} [{new_revenue.pattern_name} "
                      f"conf={new_revenue.confidence:.2f}]")
            else:
                print("        ✗ revenue: not found")
    else:
        print("  [2/4] All data in DB — skipping PDF")

    # ── Step 3: Store new results ──────────────────────────────────────────────
    if new_kpi_recs or new_revenue:
        print("  [3/4] Storing to DB...")
        _db_store(company_name, fy, new_kpi_recs, new_revenue)
        print("        Done.")
    else:
        print("  [3/4] Nothing new to store")

    # ── Step 4: Merge + normalise + build profile ─────────────────────────────
    merged_kpis    = {**cached_kpis, **new_kpi_recs}
    revenue_result = cached_rev or new_revenue

    print(f"\n  [4/4] Normalised KPIs — {company_name} FY{fy}")
    print(f"  {'KPI':<32} {'Value':>18}  Unit")
    print("  " + "─" * 56)

    for kpi_name in TARGET_KPI_NAMES:
        rec = merged_kpis.get(kpi_name)
        if not rec:
            print(f"  {'  ' + kpi_name:<32} {'NOT FOUND':>18}")
            continue
        try:
            n = normalize(kpi_name=kpi_name, value=float(rec["value"]), unit=str(rec["unit"]))
            conv = ""
            if n.conversion_factor != 1.0:
                conv = f"  (from {rec['value']:,.2f} {rec['unit']})"
            print(f"  {'  ' + kpi_name:<32} {n.normalized_value:>16,.2f}  {n.normalized_unit}{conv}")
        except NormalizationError as e:
            print(f"  {'  ' + kpi_name:<32} {rec['value']:>16,.2f}  "
                  f"{rec['unit']}  [WARN: {e}]")

    if revenue_result:
        print(f"  {'  revenue':<32} {revenue_result.value_cr:>16,.0f}  INR_Crore")
    else:
        print(f"  {'  revenue':<32} {'NOT FOUND':>18}")

    # Use page_texts from extraction (or reload) for reported-ratio detection
    page_texts: list[str] = []
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        for pg in doc:
            page_texts.append(pg.get_text())
        doc.close()
    except Exception:
        pass

    rev_cr  = revenue_result.value_cr if revenue_result else None
    rev_src = revenue_result.source   if revenue_result else "unavailable"

    if rev_cr is None:
        print(f"\n  [WARN] No revenue — intensity ratios will be incorrect")
        rev_cr  = 1.0
        rev_src = "unavailable"

    profile = build_company_profile(
        kpi_records=merged_kpis,
        revenue_cr=rev_cr,
        revenue_source=rev_src,
        company_name=company_name,
        fiscal_year=fy,
        page_texts=page_texts,
    )

    print(f"\n  Intensity ratios ({company_name} FY{fy}):")
    for kpi_name, ratio in profile.ratios.items():
        src = "[reported]" if "reported" in ratio.ratio_source else "[computed]"
        v   = f"{ratio.ratio_value:.4e}" if ratio.ratio_value < 0.001 else f"{ratio.ratio_value:.4f}"
        print(f"  {'  ' + kpi_name:<32} {v:>14}  {ratio.ratio_unit}  {src}")

    return profile


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ESG Competitive Benchmarking — terminal pipeline"
    )
    parser.add_argument("--pdf1",     required=True, type=Path)
    parser.add_argument("--company1", required=True)
    parser.add_argument("--fy1",      required=True, type=int)
    parser.add_argument("--pdf2",     required=True, type=Path)
    parser.add_argument("--company2", required=True)
    parser.add_argument("--fy2",      required=True, type=int)
    parser.add_argument("--force",   action="store_true",
                        help="Skip DB cache and re-extract from PDF")
    parser.add_argument("--no-llm",  action="store_true",
                        help="Skip all LLM calls (rule-based summary)")
    args = parser.parse_args()

    print("\n" + "═" * W)
    print("  ESG COMPETITIVE BENCHMARKING PIPELINE")
    print(f"  {args.company1} FY{args.fy1}  vs  {args.company2} FY{args.fy2}")
    print("═" * W)

    # Ensure revenue columns exist on every run
    print("\n  Ensuring DB schema...")
    _ensure_revenue_cols()

    # Shared LLM service
    llm_service = None
    if not args.no_llm:
        try:
            from services.llm_service import LLMService
            from core.config import get_settings
            if get_settings().llm_api_key:
                llm_service = LLMService()
                print("  LLM: enabled")
            else:
                print("  LLM: no API key — rule-based fallback")
        except Exception as exc:
            print(f"  LLM: unavailable ({exc})")

    profile1 = process_company(args.pdf1, args.company1, args.fy1, args.force, llm_service)
    profile2 = process_company(args.pdf2, args.company2, args.fy2, args.force, llm_service)

    report = compare_profiles([profile1, profile2])
    print_report(report)

    print("\n" + "═" * W)
    print("  NARRATIVE SUMMARY")
    print("═" * W + "\n")
    print(generate_summary([profile1, profile2], report, llm=llm_service))
    print()


if __name__ == "__main__":
    main()