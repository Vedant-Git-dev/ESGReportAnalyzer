"""
services/search_service.py

Tavily-powered ESG report discovery with URL-keyword classification.

Algorithm
---------
Step 1 — Collect
    Run every query template for every report type against Tavily.
    All raw results (URL + title + score + source query) are pooled into
    one flat list. No filtering is done at this stage.

Step 2 — Deduplicate
    Remove exact duplicate URLs globally across all types and queries.
    Only the first occurrence (highest Tavily score) is kept.

Step 3 — Classify
    Inspect the URL string of each unique result against per-type keyword
    sets. A URL "matches" a type when its lowercased string contains ALL
    required keywords for that type.

    Keyword sets:
        BRSR        : ("brsr",) OR ("business", "sustainability", "responsibility")
        ESG         : ("sustainability",) OR ("esg",)
        Integrated  : ("integrated",) AND ("annual" OR "report")

    A URL may match multiple types (e.g. a combined BRSR + Sustainability
    report). In that case the priority order decides:
        BRSR > ESG > Integrated

    A URL that matches no type keyword set is discarded — it is not a
    relevant document.

Step 4 — Assign
    Each type collects only the URLs assigned to it (sorted by score desc).
    A URL is assigned to exactly one type: the highest-priority type it
    matched. This prevents the same URL from being downloaded multiple times.

Step 5 — Return
    Returns dict[str, SearchResult] with one entry per type.
    Each SearchResult.discovered contains only URLs classified for that type,
    sorted by Tavily score descending.
    If no URLs were found for a type, .discovered is empty and .total_found is 0.

Filename contract (enforced in ingestion_agent.py)
---------------------------------------------------
    {year}_{TYPE}_{company_slug}_{id8}.pdf
    e.g. 2025_BRSR_infosys_limited_a3f19c2b.pdf

Public API
----------
collect_and_classify(company_name, year) -> dict[str, SearchResult]
    Main entry point. Returns per-type SearchResults after full
    collect-deduplicate-classify-assign pipeline.

search_reports(company_name, year, report_type) -> SearchResult
    Search a single type (kept for backward compatibility).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import get_settings
from core.logging_config import get_logger
from models.schemas import DiscoveredReport, SearchResult

logger = get_logger(__name__)

_TAVILY_ENDPOINT = "https://api.tavily.com/search"

# ---------------------------------------------------------------------------
# Canonical types and priority order
# ---------------------------------------------------------------------------

# PRIORITY_ORDER defines which type wins when a URL matches multiple types.
# Index 0 = highest priority.
PRIORITY_ORDER: list[str] = ["BRSR", "ESG", "Integrated"]

ALL_REPORT_TYPES: list[str] = PRIORITY_ORDER  # alias used by other modules
DEFAULT_REPORT_TYPE = "BRSR"

# ---------------------------------------------------------------------------
# Query templates
#
# Substitution keys: {company}, {year}, {past_year}
#
# Design rules:
# - First template per type = highest precision (most specific phrase).
# - Later templates = progressively broader fallbacks.
# - "filetype:pdf" hint is included wherever it helps Tavily return direct
#   PDF links vs. investor-relations landing pages.
# - Site-specific hints (nsearchive, bseindia) target known filing portals.
#
# All templates for all types are run regardless of type — every URL returned
# by every query is pooled before classification. The type groupings here are
# used only to determine which queries to run; classification is done by URL
# keyword matching, not by which query returned the URL.
# ---------------------------------------------------------------------------

_QUERY_TEMPLATES: dict[str, list[str]] = {

    # BRSR — India-mandated under SEBI LODR. Filed on NSE/BSE exchanges or
    # embedded in the Annual Report as a standalone chapter.
    "BRSR": [
        "{company} BRSR {past_year}-{year} filetype:pdf",
        # "{company} business responsibility sustainability report {past_year}-{year} filetype:pdf",
        "{company} BRSR {year} site:nsearchive.nseindia.com",
        "{company} business responsibility report {year} site:bseindia.com filetype:pdf",
        # "{company} BRSR {year} filetype:pdf",
        # "{company} business responsibility and sustainability report {year} pdf",
    ],

    # ESG / Sustainability — voluntarily published by most large-cap Indian
    # companies. Naming is highly inconsistent; cover full alias spectrum.
    "ESG": [
        "{company} ESG report {past_year}-{year} filetype:pdf",
        "{company} sustainability report {past_year}-{year} filetype:pdf",
        # "{company} sustainability ESG report {year} filetype:pdf",
        # "{company} environmental social governance report {year} filetype:pdf",
        # "{company} corporate responsibility report {year} filetype:pdf",
        # "{company} responsible business report {year} filetype:pdf",
        # "{company} environmental report {year} filetype:pdf",
    ],

    # Integrated — follows IIRC <IR> framework. Combines financial and
    # non-financial information. Often replaces the standalone Annual Report.
    "Integrated": [
        # "{company} integrated report {year} filetype:pdf",
        "{company} integrated annual report {past_year}-{year} filetype:pdf",
        # "{company} integrated value creation report {year} filetype:pdf",
        # "{company} integrated annual report {year} pdf",
        # "{company} integrated report IR {year} filetype:pdf",
    ],
}

# ---------------------------------------------------------------------------
# URL keyword classification rules
#
# Each type maps to a list of "match rules". A URL matches a type when ANY
# one of its match rules is satisfied.
#
# A match rule is a tuple of keyword strings. The URL (lowercased) must
# contain ALL keywords in the tuple for the rule to be satisfied.
#
# Examples:
#   ("brsr",)                                 — URL contains "brsr"
#   ("business", "sustainability")            — URL contains both words
#   ("integrated", "annual")                  — URL contains both words
#
# Keywords are matched against the full URL string (scheme + host + path),
# so path segments like "/brsr-report-2025.pdf" are matched correctly.
# ---------------------------------------------------------------------------

_URL_KEYWORD_RULES: dict[str, list[tuple[str, ...]]] = {
    "BRSR": [
        # Exact abbreviation in URL path (most common on NSE/BSE portals)
        # ("brsr",),
        # Full phrase components in URL — companies that spell it out
        ("business", "sustainability"),
        ("business", "responsibility"),
    ],
    "ESG": [
        # ESG acronym in URL
        ("esg",),
        # Sustainability without BRSR context — a standalone sustainability report
        ("sustainability",),
        # Responsibility without "business" prefix is typically CSR/ESG
        # ("responsibility",),
        # Environmental or corporate responsibility naming
        ("environmental",),
    ],
    "Integrated": [
        # Integrated report or integrated annual report
        ("integrated", "report"),
        ("integrated", "annual"),
        ("integrated",),   # fallback: "integrated" alone in URL
    ],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_all_queries(company: str, year: int) -> list[tuple[str, str]]:
    """
    Expand every template for every type into concrete query strings.

    Returns a list of (query_string, report_type) pairs.
    The report_type label here is only for logging — it does NOT determine
    how the URL will be classified. Classification happens by URL keywords.
    """
    pairs: list[tuple[str, str]] = []
    for report_type, templates in _QUERY_TEMPLATES.items():
        for template in templates:
            query = template.format(
                company=company,
                year=year,
                past_year=year - 1,
            )
            pairs.append((query, report_type))
    return pairs


def _classify_url(url: str) -> Optional[str]:
    """
    Classify a URL into exactly one report type using keyword matching.

    Checks each type's keyword rules in PRIORITY_ORDER (BRSR first, then
    ESG, then Integrated). Returns the first type whose rules match, or
    None if no type matches.

    Args:
        url: Full URL string (will be lowercased internally).

    Returns:
        Type string ("BRSR", "ESG", or "Integrated") or None.
    """
    url_lower = url.lower()

    for report_type in PRIORITY_ORDER:
        rules = _URL_KEYWORD_RULES[report_type]
        for keyword_tuple in rules:
            # A rule is satisfied when ALL keywords in the tuple appear in the URL
            if all(kw in url_lower for kw in keyword_tuple):
                return report_type

    # URL does not match any type — not a relevant document
    return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def _call_tavily(query: str, api_key: str, max_results: int) -> list[dict]:
    """
    Make one Tavily search API call and return the raw results list.

    Retries up to 3 times with exponential backoff on transient failures.
    Raises the final exception if all retries are exhausted.
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
        response = client.post(_TAVILY_ENDPOINT, json=payload)
        response.raise_for_status()
        return response.json().get("results", [])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collect_and_classify(
    company_name: str,
    year: int,
    max_results_per_query: int = 10,
) -> dict[str, SearchResult]:
    """
    Main entry point. Run all queries, pool results, classify by URL keywords.

    Steps performed:
    1. Run every query template for every type against Tavily.
    2. Pool all raw results into one flat list.
    3. Globally deduplicate by URL (keep highest-score occurrence).
    4. Classify each unique URL by keyword matching against _URL_KEYWORD_RULES.
       Priority order: BRSR > ESG > Integrated.
       A URL is assigned to at most one type.
       URLs that match no type are discarded (not relevant).
    5. Build one SearchResult per type from the classified URLs.

    Args:
        company_name:          Company name string, e.g. "Infosys".
        year:                  Fiscal year end integer, e.g. 2025.
        max_results_per_query: Tavily max_results per individual API call.
                               3 gives good recall without burning quota.

    Returns:
        {
            "BRSR":       SearchResult(discovered=[...], total_found=N, ...),
            "ESG":        SearchResult(discovered=[...], total_found=N, ...),
            "Integrated": SearchResult(discovered=[...], total_found=N, ...),
        }
        All three keys are always present.
        .discovered is empty when no classified URLs were found for a type.

    Logging:
        Logs per-type counts and any types with zero URLs found.
    """
    settings = get_settings()

    # Return empty results immediately if Tavily is not configured.
    if not settings.tavily_api_key:
        logger.warning("search_service.no_api_key")
        return _empty_results(company_name, year)

    all_queries = _build_all_queries(company_name, year)
    logger.info(
        "search_service.collect_start",
        company=company_name,
        year=year,
        total_queries=len(all_queries),
    )

    # ------------------------------------------------------------------
    # Step 1: Run all queries and pool results
    # ------------------------------------------------------------------

    # raw_pool: list of (url, title, snippet, score, source_query)
    # Multiple queries may return the same URL with different scores.
    raw_pool: list[dict] = []

    for query, query_type_hint in all_queries:
        try:
            results = _call_tavily(query, settings.tavily_api_key, max_results_per_query)
        except Exception as exc:
            # A single query failure should not abort the whole search.
            logger.error(
                "search_service.query_failed",
                query=query,
                query_type_hint=query_type_hint,
                error=str(exc),
            )
            continue

        for item in results:
            url = item.get("url", "").strip()
            if not url:
                continue
            raw_pool.append({
                "url":          url,
                "title":        item.get("title"),
                "snippet":      item.get("content", "")[:300],
                "score":        float(item.get("score", 0.0)),
                "query_source": query,
            })

    logger.info(
        "search_service.raw_pool_size",
        company=company_name,
        raw_count=len(raw_pool),
    )

    # ------------------------------------------------------------------
    # Step 2: Global URL deduplication — keep highest score per URL
    # ------------------------------------------------------------------

    # best_by_url: url -> the dict with the highest score seen so far
    best_by_url: dict[str, dict] = {}
    for item in raw_pool:
        url = item["url"]
        if url not in best_by_url or item["score"] > best_by_url[url]["score"]:
            best_by_url[url] = item

    unique_items = list(best_by_url.values())
    logger.info(
        "search_service.after_dedup",
        company=company_name,
        unique_count=len(unique_items),
    )

    # ------------------------------------------------------------------
    # Step 3 & 4: Classify each URL and assign to exactly one type
    # ------------------------------------------------------------------

    # classified: report_type -> list of DiscoveredReport
    classified: dict[str, list[DiscoveredReport]] = {t: [] for t in PRIORITY_ORDER}
    discarded_count = 0

    for item in unique_items:
        url        = item["url"]
        assigned   = _classify_url(url)

        if assigned is None:
            # URL contains no keywords matching any known report type.
            # Common causes: investor-relations landing pages, press releases,
            # index pages that link to the actual PDF.
            logger.debug(
                "search_service.url_unclassified",
                url=url[:80],
            )
            discarded_count += 1
            continue

        classified[assigned].append(DiscoveredReport(
            url=url,
            title=item["title"],
            snippet=item["snippet"],
            score=item["score"],
            query_source=item["query_source"],
        ))

    logger.info(
        "search_service.classification_complete",
        company=company_name,
        discarded_unclassified=discarded_count,
        brsr_count=len(classified["BRSR"]),
        esg_count=len(classified["ESG"]),
        integrated_count=len(classified["Integrated"]),
    )

    # ------------------------------------------------------------------
    # Step 5: Build SearchResult per type, sort by score desc
    # ------------------------------------------------------------------

    results: dict[str, SearchResult] = {}

    for report_type in PRIORITY_ORDER:
        urls_for_type = classified[report_type]
        urls_for_type.sort(key=lambda r: r.score, reverse=True)

        results[report_type] = SearchResult(
            company_name=company_name,
            year=year,
            report_type=report_type,
            discovered=urls_for_type,
            total_found=len(urls_for_type),
            queries_run=len(all_queries),
        )

        if not urls_for_type:
            logger.warning(
                "search_service.type_not_found",
                company=company_name,
                year=year,
                report_type=report_type,
                message=(
                    f"No URLs classified as {report_type} for {company_name} FY{year}. "
                    f"The company may not publish a separate {report_type} report, "
                    f"or Tavily did not return relevant results."
                ),
            )
        else:
            logger.info(
                "search_service.type_found",
                company=company_name,
                year=year,
                report_type=report_type,
                count=len(urls_for_type),
                top_url=urls_for_type[0].url[:80],
            )

    return results


def _empty_results(company_name: str, year: int) -> dict[str, SearchResult]:
    """Return an empty SearchResult for every type (used when API key missing)."""
    return {
        rtype: SearchResult(
            company_name=company_name,
            year=year,
            report_type=rtype,
            discovered=[],
            total_found=0,
            queries_run=0,
        )
        for rtype in PRIORITY_ORDER
    }


def search_reports(
    company_name: str,
    year: int,
    report_type: str = DEFAULT_REPORT_TYPE,
    max_results_per_query: int = 3,
) -> SearchResult:
    """
    Search for a single report type.

    Kept for backward compatibility with the CLI and single-type flows.
    Internally calls collect_and_classify() and returns the slice for the
    requested type.

    Note: even when requesting a single type, ALL query templates for all
    types are run so that cross-type URL classification works correctly.
    If you need only one type and want to minimise API calls, use this
    function. If you need all three types, use collect_and_classify() directly
    to avoid running all queries three times.
    """
    # Normalise input
    canonical_type = next(
        (t for t in PRIORITY_ORDER if t.lower() == report_type.lower()),
        DEFAULT_REPORT_TYPE,
    )

    all_results = collect_and_classify(
        company_name=company_name,
        year=year,
        max_results_per_query=max_results_per_query,
    )
    return all_results[canonical_type]


# Alias for callers that import search_all_report_types by name
def search_all_report_types(
    company_name: str,
    year: int,
    max_results_per_query: int = 3,
) -> dict[str, SearchResult]:
    """Alias for collect_and_classify(). Kept for backward compatibility."""
    return collect_and_classify(
        company_name=company_name,
        year=year,
        max_results_per_query=max_results_per_query,
    )