"""
api/routers/companies.py

GET /api/companies          — all active companies (no filter)
GET /api/companies?sector=X — companies in a specific sector

Used by the React sidebar to populate the company dropdown when a sector is selected.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from api.pipeline import list_companies, list_companies_by_sector
from api.schemas import CompanyListItem

router = APIRouter(tags=["companies"])


@router.get("/companies", response_model=list[CompanyListItem])
def get_companies(
    sector: Optional[str] = Query(
        default=None,
        description=(
            "Filter by sector (case-insensitive exact match). "
            "Omit to return all companies."
        ),
    ),
) -> list[CompanyListItem]:
    """
    Return active companies from the database.

    When sector is supplied the list is filtered to that sector only.
    When sector is omitted all companies are returned.

    The React sidebar calls this endpoint whenever the sector dropdown changes.
    """
    if sector:
        rows = list_companies_by_sector(sector)
    else:
        rows = list_companies()
    return [CompanyListItem(**r) for r in rows]