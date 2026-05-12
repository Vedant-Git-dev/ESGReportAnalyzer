"""
api/routers/upload.py

Handles PDF upload — verbatim logic from ui.py run_upload_pipeline.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, UploadFile, File

from api.pipeline import (
    cache_load, cache_load_revenue, db_store_kpis, ensure_schema,
    get_llm_service, step_extract_missing, step_parse, ReportInfo,
)
from api.schemas import ALL_KPI_NAMES, KPIRecord, RevenueInfo, UploadResponse

router = APIRouter(tags=["upload"])

MAX_PDF_BYTES = 50 * 1024 * 1024   # 50 MB


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file:        UploadFile    = File(...),
    company:     str           = Form(...),
    fy:          int           = Form(...),
    sector:      str           = Form("Information Technology"),
    report_type: str           = Form("BRSR"),
) -> UploadResponse:
    """
    Accept a PDF upload, ingest it, parse it, and extract ESG KPIs.
    Mirrors ui.py run_upload_pipeline exactly.
    """
    ensure_schema()

    # Size guard
    content = await file.read()
    if len(content) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 50 MB limit.")

    # PDF magic check
    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF.")

    log: list[str] = []
    llm = get_llm_service()

    # Write to temp file
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, prefix="esg_upload_") as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not save file: {exc}")

    try:
        from agents.ingestion_agent import IngestionAgent
        result = IngestionAgent().ingest_uploaded_pdf(
            source_path=tmp_path, company_name=company,
            year=fy, sector=sector, report_type=report_type,
        )
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)

    company_obj = result["company"]
    report_obj  = result["report"]
    company_id  = company_obj.id
    report_id   = report_obj.id

    # Cache check
    cached_kpis    = cache_load(company_id, fy)
    cached_revenue = cache_load_revenue(company_id, fy)
    missing        = [k for k in ALL_KPI_NAMES if k not in cached_kpis]
    need_rev       = cached_revenue is None

    if not missing and not need_rev:
        log.append(f"All data loaded from cache ({len(cached_kpis)} metrics).")
        return _build_upload_response(
            success=True, company_id=str(company_id), report_id=str(report_id),
            company=company, fy=fy, kpi_records=cached_kpis,
            revenue=cached_revenue, log=log,
            message=f"All {len(cached_kpis)} metrics loaded from cache.",
        )

    # Parse
    step_parse(report_id, report_type, log)

    # Extract
    ext = step_extract_missing(
        report_id=report_id, report_type=report_type, fy=fy,
        log=log, llm_service=llm,
        missing_kpi_names=missing, need_revenue=need_rev,
    )
    new_kpis    = ext["kpis"]
    new_revenue = ext["revenue"]

    merged      = {**new_kpis, **cached_kpis}
    final_rev   = cached_revenue or new_revenue

    if company_id and (new_kpis or new_revenue):
        db_store_kpis(company_id, report_id, fy, new_kpis, new_revenue)
        log.append(f"Saved {len(new_kpis)} metric(s) to database.")

    msg = (
        f"Done — {len(merged)} ESG metric(s) extracted."
        if merged else
        "No metrics could be extracted from this report."
    )
    return _build_upload_response(
        success=True, company_id=str(company_id), report_id=str(report_id),
        company=company, fy=fy, kpi_records=merged,
        revenue=final_rev, log=log, message=msg,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _build_upload_response(
    success: bool, company_id, report_id, company: str, fy: int,
    kpi_records: dict, revenue, log: list[str], message: str,
) -> UploadResponse:
    kpi_schema: dict[str, KPIRecord] = {}
    for name, rec in kpi_records.items():
        try:
            kpi_schema[name] = KPIRecord(
                value=float(rec["value"]),
                unit=str(rec.get("unit") or ""),
                method=str(rec.get("method") or ""),
                confidence=float(rec.get("confidence") or 0.5),
                report_type=rec.get("report_type"),
            )
        except Exception:
            pass

    rev_schema = None
    if revenue is not None:
        try:
            rev_schema = RevenueInfo(
                value_cr=float(revenue.value_cr),
                source=str(revenue.source),
                pattern_name=str(revenue.pattern_name),
                confidence=float(revenue.confidence),
            )
        except Exception:
            pass

    return UploadResponse(
        success=success,
        company_id=company_id,
        report_id=report_id,
        company_name=company,
        fy=fy,
        kpi_records=kpi_schema,
        revenue=rev_schema,
        log=log,
        message=message,
    )
