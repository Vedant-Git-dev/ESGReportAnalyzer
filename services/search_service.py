"""
services/search_service.py

Tavily-powered ESG report discovery.
Supported report types (default: BRSR): BRSR | ESG | Sustainability | Annual
"""
from __future__ import annotations
import re
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from core.config import get_settings
from core.logging_config import get_logger
from models.schemas import DiscoveredReport, SearchResult

logger = get_logger(__name__)
_PDF_RE = re.compile(r"\.pdf(\?.*)?$", re.IGNORECASE)
_TAVILY_URL = "https://api.tavily.com/search"

REPORT_QUERY_TEMPLATES: dict[str, list[str]] = {
    "BRSR": [
        # "site:nsearchives.nseindia.com Business Responsibility and Sustainability Report {past_year}-{year} filetype:pdf",
        "{company} business responsibility sustainability report {past_year}-{year} nsearchive filetype:pdf",
    ],
    "Annual": [
        "{company} annual report {past_year}-{year} filetype:pdf",
        "{company} annual report {past_year}-{year} download",
    ],
}

DEFAULT_REPORT_TYPE = "BRSR"
VALID_REPORT_TYPES = list(REPORT_QUERY_TEMPLATES.keys())


def _normalise_type(report_type: str) -> str:
    for key in REPORT_QUERY_TEMPLATES:
        if key.lower() == report_type.lower():
            return key
    logger.warning("search_service.unknown_report_type", given=report_type, fallback=DEFAULT_REPORT_TYPE)
    return DEFAULT_REPORT_TYPE


def _build_queries(company: str, year: int, report_type: str = DEFAULT_REPORT_TYPE) -> list[str]:
    key = _normalise_type(report_type)
    return [t.format(company=company, year=year, past_year = year - 1) for t in REPORT_QUERY_TEMPLATES[key]]


def _is_pdf_url(url: str) -> bool:
    return bool(_PDF_RE.search(url))


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _call_tavily(query: str, api_key: str, max_results: int = 10) -> list[dict]:
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
    report_type: str = DEFAULT_REPORT_TYPE,
    max_results_per_query: int = 1,
    pdf_only: bool = True,
) -> SearchResult:
    """
    Discover PDF reports for a company via Tavily.

    Args:
        company_name:  e.g. "Infosys"
        year:          e.g. 2024
        report_type:   BRSR | ESG | Sustainability | Annual  (default: BRSR)
        max_results_per_query: Tavily max_results per call
        pdf_only:      Drop non-PDF URLs when True
    """
    settings = get_settings()
    if not settings.tavily_api_key:
        logger.warning("search_service.no_api_key")
        return SearchResult(company_name=company_name, year=year, report_type=report_type)

    queries = _build_queries(company_name, year, report_type)
    print(queries)
    logger.info("search_service.start", company=company_name, year=year, report_type=report_type)

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
            if not url or url in seen_urls:
                continue
            if pdf_only and not _is_pdf_url(url):
                continue
            seen_urls.add(url)
            discovered.append(DiscoveredReport(
                url=url,
                title=item.get("title"),
                snippet=item.get("content", "")[:300],
                score=float(item.get("score", 0.0)),
                query_source=query,
            ))

    discovered.sort(key=lambda r: r.score, reverse=True)
    result = SearchResult(
        company_name=company_name,
        year=year,
        report_type=report_type,
        discovered=discovered,
        total_found=len(discovered),
        queries_run=len(queries),
    )
    logger.info("search_service.complete", company=company_name, pdf_links=len(discovered))
    return result