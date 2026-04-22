"""
api/routers/compare.py

Two endpoints:
  POST /api/compare        — synchronous; returns full CompareResponse JSON
  GET  /api/compare/stream — SSE; streams progress then final result as events

Both call the exact same pipeline.run_company_pipeline() that ui.py used.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.pipeline import (
    build_benchmark, ensure_schema, get_llm_service, run_company_pipeline,
)
from api.schemas import (
    CompanyResult, CompareRequest, CompareResponse, ComparisonEntry,
    KPI_GROUPS, KPIComparison, KPIRecord, RevenueInfo,
)

router = APIRouter(tags=["compare"])


# ---------------------------------------------------------------------------
# Helpers — convert internal objects to response schemas
# ---------------------------------------------------------------------------

def _kpi_records_to_schema(kpi_records: dict) -> dict[str, KPIRecord]:
    out: dict[str, KPIRecord] = {}
    for name, rec in kpi_records.items():
        try:
            out[name] = KPIRecord(
                value=float(rec["value"]),
                unit=str(rec.get("unit") or ""),
                method=str(rec.get("method") or ""),
                confidence=float(rec.get("confidence") or 0.5),
                report_type=rec.get("report_type"),
            )
        except Exception:
            pass
    return out


def _revenue_to_schema(rev) -> RevenueInfo | None:
    if rev is None:
        return None
    try:
        return RevenueInfo(
            value_cr=float(rev.value_cr),
            source=str(rev.source),
            pattern_name=str(rev.pattern_name),
            confidence=float(rev.confidence),
        )
    except Exception:
        return None


def _build_response(
    data1, data2, benchmark: dict,
) -> CompareResponse:
    label_a = f"{data1.company_name} FY{data1.fy}"
    label_b = f"{data2.company_name} FY{data2.fy}"

    comparisons: list[KPIComparison] = []
    for comp in benchmark["filtered"]:
        entries = [
            ComparisonEntry(company_label=lbl, value=v, source=src)
            for lbl, v, src in comp.entries
        ]
        comparisons.append(KPIComparison(
            kpi_name=comp.kpi_name,
            display_name=comp.display_name,
            group=comp.group,
            unit=comp.unit,
            entries=entries,
            winner=comp.winner,
            pct_gap=comp.pct_gap,
            meta=KPI_GROUPS.get(comp.kpi_name, {}),
        ))

    return CompareResponse(
        company1=CompanyResult(
            company_name=data1.company_name, fy=data1.fy,
            kpi_records=_kpi_records_to_schema(data1.kpi_records),
            revenue=_revenue_to_schema(data1.revenue_result),
            log=data1.log,
        ),
        company2=CompanyResult(
            company_name=data2.company_name, fy=data2.fy,
            kpi_records=_kpi_records_to_schema(data2.kpi_records),
            revenue=_revenue_to_schema(data2.revenue_result),
            log=data2.log,
        ),
        comparisons=comparisons,
        summary=benchmark["summary"],
        label_a=label_a,
        label_b=label_b,
    )


# ---------------------------------------------------------------------------
# Synchronous endpoint
# ---------------------------------------------------------------------------

@router.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest) -> CompareResponse:
    """
    Run the full two-company ESG pipeline and return the benchmark result.
    This is a synchronous blocking call — for progress streaming use /compare/stream.
    """
    ensure_schema()
    llm = get_llm_service()

    data1 = run_company_pipeline(req.company1, req.fy1, req.sector, llm)
    data2 = run_company_pipeline(req.company2, req.fy2, req.sector, llm)

    benchmark = build_benchmark(data1, data2, req.sector)
    return _build_response(data1, data2, benchmark)


# ---------------------------------------------------------------------------
# SSE streaming endpoint
# ---------------------------------------------------------------------------

@router.get("/compare/stream")
async def compare_stream(
    company1: str,
    fy1: int,
    company2: str,
    fy2: int,
    sector: str = "Information Technology",
) -> StreamingResponse:
    """
    Stream pipeline progress as Server-Sent Events.

    Event types emitted:
      progress  — {"company": str, "message": str}
      result    — full CompareResponse JSON
      error     — {"message": str}
      done      — {}
    """
    async def event_generator() -> AsyncGenerator[str, None]:

        def _sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        ensure_schema()
        llm = get_llm_service()

        loop     = asyncio.get_event_loop()
        data1    = None
        data2    = None

        # Company 1
        def _emit1(msg: str) -> None:
            pass  # captured in run_in_executor — can't yield from sync callback

        log1: list[str] = []
        log2: list[str] = []

        # We run the blocking pipeline in a thread executor and collect events.
        # Progress messages are captured in the log and emitted after each company.

        try:
            yield _sse("progress", {"company": company1, "message": "Starting pipeline..."})
            data1 = await loop.run_in_executor(
                None,
                lambda: run_company_pipeline(company1, fy1, sector, llm),
            )
            for msg in data1.log:
                yield _sse("progress", {"company": company1, "message": msg})
                await asyncio.sleep(0)   # yield control

            yield _sse("progress", {"company": company2, "message": "Starting pipeline..."})
            data2 = await loop.run_in_executor(
                None,
                lambda: run_company_pipeline(company2, fy2, sector, llm),
            )
            for msg in data2.log:
                yield _sse("progress", {"company": company2, "message": msg})
                await asyncio.sleep(0)

            yield _sse("progress", {"company": "benchmark", "message": "Building comparison..."})
            benchmark = await loop.run_in_executor(
                None,
                lambda: build_benchmark(data1, data2, sector),
            )

            response  = _build_response(data1, data2, benchmark)
            yield _sse("result", response.model_dump())
            yield _sse("done", {})

        except Exception as exc:
            yield _sse("error", {"message": str(exc)})
            yield _sse("done", {})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )
