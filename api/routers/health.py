"""api/routers/health.py"""
from __future__ import annotations

from fastapi import APIRouter

from api.pipeline import check_db, get_llm_service
from api.schemas import HealthResponse, MetadataResponse, SECTORS, UPLOAD_REPORT_TYPE_OPTIONS, KPI_GROUPS, ALL_KPI_NAMES

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    db_ok  = check_db()
    llm    = get_llm_service()
    status = "ok" if db_ok else "db_offline"
    return HealthResponse(db_online=db_ok, llm_ready=llm is not None, status=status)


@router.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    """Return all static configuration the frontend needs at startup."""
    return MetadataResponse(
        sectors=SECTORS,
        report_types=UPLOAD_REPORT_TYPE_OPTIONS,
        kpi_groups=KPI_GROUPS,
        kpi_names=ALL_KPI_NAMES,
    )
