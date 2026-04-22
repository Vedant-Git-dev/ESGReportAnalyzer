"""api/routers/companies.py"""
from __future__ import annotations

from fastapi import APIRouter

from api.pipeline import list_companies
from api.schemas import CompanyListItem

router = APIRouter(tags=["companies"])


@router.get("/companies", response_model=list[CompanyListItem])
def get_companies() -> list[CompanyListItem]:
    """Return all active companies stored in the database."""
    rows = list_companies()
    return [CompanyListItem(**r) for r in rows]
