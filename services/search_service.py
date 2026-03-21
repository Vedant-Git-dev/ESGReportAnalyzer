"""
services/search_service.py

Tavily-powered ESG report discovery.
Generates 5 query variants, calls Tavily, filters PDF URLs, deduplicates.
"""
from __future__ import annotations

import re
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import get_settings
from core.logging_config import get_logger
from models.schemas import DiscoveredReport, SearchResult

logger = get_logger(__name__)

_PDF_RE = re.compile(r"\.pdf(\?.*)?$", re.IGNORECASE)
_TAVILY_URL = "https://api.tavily.com/search"


def _is_pdf_url(url: str) -> bool:
    """Heuristic: URL ends with .pdf (with optional query string)."""
    return bool(_PDF_RE.search(url))


def _build_queries(company: str, year: int) -> list[str]:
    """
    5 query variants for maximum recall.
    Keeps queries short — Tavily performs best with concise natural-language queries.
    """
    return [
        f"{company} sustainability report {year} filetype:pdf",
        f"{company} ESG report {year} PDF",
        f"{company} BRSR report {year}",
        f"{company} annual report ESG {year} download",
        f"{company} environmental social governance report {year}",
    ]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _call_tavily(query: str, api_key: str, max_results: int = 10) -> list[dict]:
    """
    Single Tavily search call. Returns raw result list.
    Retries up to 3 times with exponential backoff.
    """
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }
    with httpx.Client(timeout=30) as client:
        response = client.post(_TAVILY_URL, json=payload)
        response.raise_for_status()
        return response.json().get("results", [])


def search_reports(
    company_name: str,
    year: int,
    max_results_per_query: int = 10,
    pdf_only: bool = True,
) -> SearchResult:
    """
    Public interface for the ingestion agent.

    1. Generates 5 query variants
    2. Calls Tavily for each
    3. Filters PDF links (when pdf_only=True)
    4. Deduplicates by URL
    5. Returns SearchResult with ranked DiscoveredReport list

    Args:
        company_name: Full company name (e.g. "Infosys")
        year: Report year (e.g. 2023)
        max_results_per_query: Tavily max_results per call
        pdf_only: If True, drop non-PDF URLs

    Returns:
        SearchResult with all discovered PDF links
    """
    settings = get_settings()

    if not settings.tavily_api_key:
        logger.warning("search_service.no_api_key", note="Tavily key not set — returning empty results")
        return SearchResult(company_name=company_name, year=year)

    queries = _build_queries(company_name, year)
    logger.info("search_service.start", company=company_name, year=year, queries=len(queries))

    seen_urls: set[str] = set()
    discovered: list[DiscoveredReport] = []

    for query in queries:
        try:
            raw_results = _call_tavily(query, settings.tavily_api_key, max_results_per_query)
        except Exception as exc:
            logger.error("search_service.query_failed", query=query, error=str(exc))
            continue

        for item in raw_results:
            url: str = item.get("url", "").strip()
            if not url:
                continue
            if pdf_only and not _is_pdf_url(url):
                continue
            if url in seen_urls:
                continue

            seen_urls.add(url)
            discovered.append(
                DiscoveredReport(
                    url=url,
                    title=item.get("title"),
                    snippet=item.get("content", "")[:300],
                    score=float(item.get("score", 0.0)),
                    query_source=query,
                )
            )

        logger.debug("search_service.query_done", query=query, new_results=len(discovered))

    # Sort by Tavily relevance score descending
    discovered.sort(key=lambda r: r.score, reverse=True)

    result = SearchResult(
        company_name=company_name,
        year=year,
        discovered=discovered,
        total_found=len(discovered),
        queries_run=len(queries),
    )

    logger.info(
        "search_service.complete",
        company=company_name,
        year=year,
        pdf_links=len(discovered),
    )
    return result