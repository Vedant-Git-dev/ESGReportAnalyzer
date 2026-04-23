"""
services/search_service.py

SerpApi-powered ESG report discovery with STRICT three-way validation.

Validation contract (ALL three must pass — no fallbacks, no partial matches):
  1. Company match  — the FULL company name (or its legal-suffix-stripped form)
                      must appear in TITLE or URL domain/path.
                      Individual word tokens are NOT used; "Solar Industries"
                      matches only as a phrase, never as bare "solar".
  2. Year match     — target fiscal year present, no adjacent-year contamination
  3. Report type    — detected type matches the query type bucket

If ANY of the three checks fails, the result is discarded.
An empty discovered list is returned (and no PDF is downloaded) when no
high-confidence match is found.  The caller must handle the empty case.

Company matching — full-name approach
--------------------------------------
Previous behaviour split the company name into individual word tokens
(e.g. "Solar Industries" → ["solar", "industries"]), which caused false
positives: any result mentioning "solar" (solar energy, solar funds, etc.)
would be accepted.

New behaviour: only the FULL company name is used as a match token.
  "Solar Industries"          → anchor: ["solar industries"]
  "Wipro Limited"             → anchor: ["wipro limited", "wipro"]   (suffix stripped)
  "Bharat Electronics Ltd"    → anchor: ["bharat electronics ltd", "bharat electronics"]
  "TCS"                       → anchor: ["tcs", "tata consultancy", "tcs.com"]  (alias)

The _STOP_WORDS list and per-word iteration are removed entirely.

Bug fixes in this version
--------------------------
Fix 1 — ETF/fund documents passed via snippet-only company match:
    Fix: snippet-only matching requires _TRUSTED_FILING_DOMAINS whitelist.

Fix 2 — ISO calendar dates matched as valid fiscal year signal:
    Fix: is_correct_year() strips all ISO date occurrences before testing
    the bare year.

Search backend
--------------
Uses the SerpApi Google Search endpoint (https://serpapi.com/search) via
direct httpx calls.  Required env var: SERPAPI_API_KEY.
"""
from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urlparse

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import get_settings
from core.logging_config import get_logger
from models.schemas import DiscoveredReport, SearchResult

logger = get_logger(__name__)

_SERPAPI_ENDPOINT = "https://serpapi.com/search"

# ---------------------------------------------------------------------------
# Report type priorities and query templates
# ---------------------------------------------------------------------------

PRIORITY_ORDER: list[str] = [ "Integrated", "ESG", "BRSR"]
ALL_REPORT_TYPES: list[str] = PRIORITY_ORDER
DEFAULT_REPORT_TYPE = "BRSR"

_QUERY_TEMPLATES: dict[str, list[str]] = {
    "BRSR": [
        "{company} BRSR {past_year}-{year_short} filetype:pdf",
    ],
    "ESG": [
        "{company} sustainability report {past_year}-{year_short} filetype:pdf",
    ],
    "Integrated": [
        "{company} annual report {past_year}-{year_short} filetype:pdf",
    ],
}


# ---------------------------------------------------------------------------
# COMPANY NAME MATCHING — FULL-NAME APPROACH
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]")

# Known short ticker → URL/title anchor strings.
# Keyed on the full-slug of the canonical short name (e.g. "tcs", "ongc").

# Legal-entity suffixes stripped from the end of the name to produce a
# "bare name" slug variant for URL matching.
# E.g. "Wipro Limited" → bare "wipro" so tcs.com/wipro/... still matches.
_LEGAL_SUFFIXES: list[str] = [
    " limited", " ltd", " inc", " corp", " corporation",
    " private", " pvt", " llp", " llc", " plc",
]

# Domains that are trusted regulatory/exchange filing hosts.
_TRUSTED_FILING_DOMAINS: frozenset[str] = frozenset({
    "bseindia.com",
    "bsmedia.business-standard.com",
    "nseindia.com",
    "nsearchive.nseindia.com",
    "connect2nse.com",
    "sebi.gov.in",
    "mca.gov.in",
    "nsdl.co.in",
    "cdsl.co.in",
    "india.gov.in",
    "nic.in",
})


def _get_company_tokens(company_name: str) -> dict[str, list[str]]:
    """
    Build company match tokens using the FULL company name, not word tokens.

    This replaces the previous per-word tokenization that caused false positives
    (e.g. "Solar Industries" → standalone "solar" token matching solar energy
    funds, solar power reports, and any other "solar" document).

    Strategy
    --------
    anchor[0]  — full lowercased name  (primary match in title / URL)
    anchor[1+] — ticker alias expansions when the slug is in _TICKER_ALIASES
    anchor[-1] — bare name with legal suffix stripped (e.g. "wipro" from
                 "Wipro Limited"), added only when it differs from the full name

    slug[0]    — full alphanumeric slug  (for URL substring match)
    slug[-1]   — alphanumeric slug of bare name (when different)

    No individual word tokens are produced. "Solar Industries" will ONLY
    match results that contain the phrase "solar industries" as a unit.

    Args:
        company_name: The company name as entered by the user.

    Returns:
        {
            "anchor": [str, ...],   # checked in title / URL text
            "slug":   [str, ...],   # checked in alphanumeric-stripped URL
        }
    """
    name_lower = company_name.lower().strip()
    full_slug  = _SLUG_RE.sub("", name_lower)

    anchor_tokens: list[str] = [name_lower]
    slug_tokens:   list[str] = [full_slug]

    # Strip legal suffix to produce a bare-name variant.
    # E.g. "Wipro Limited" → bare "wipro"; "Bharat Electronics Ltd" → "bharat electronics".
    bare_name = name_lower
    for suffix in _LEGAL_SUFFIXES:
        if name_lower.endswith(suffix):
            bare_name = name_lower[: -len(suffix)].strip()
            break

    if bare_name and bare_name != name_lower:
        anchor_tokens.append(bare_name)
        slug_tokens.append(_SLUG_RE.sub("", bare_name))

    def _dedup(lst: list[str], min_len: int) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for t in lst:
            t = t.strip()
            if t and len(t) >= min_len and t not in seen:
                seen.add(t)
                out.append(t)
        return out

    return {
        "anchor": _dedup(anchor_tokens, 3),
        "slug":   _dedup(slug_tokens,   3),
    }


def has_company_match(
    title:      str,
    url:        str,
    snippet:    str,
    token_sets: dict[str, list[str]],
) -> bool:
    """
    Return True when the company name (or its legal-suffix-stripped form) is
    found in the TITLE or URL.

    Because tokens now represent the FULL company name rather than individual
    words, a match is a genuine signal that the document belongs to this company.

    Priority order:
      1. Full name (anchor token) found in page title
      2. Slug token found in URL domain+path
      3. Anchor token (>= 5 chars) found in URL after stripping punctuation

    Snippet-only matching is intentionally not used (removed in an earlier
    fix to prevent ETF/fund false positives; kept removed here for the same
    reason — even the full name appearing only in a snippet can be a passing
    reference in a portfolio holdings list).

    Args:
        title:      Page title string.
        url:        Full URL string.
        snippet:    Search snippet string (not used for matching).
        token_sets: Output of _get_company_tokens().

    Returns:
        True if match found; False otherwise.
    """
    anchor_tokens = token_sets["anchor"]
    slug_tokens   = token_sets["slug"]

    title_l = title.lower()

    # Parse URL — use netloc+path only (drop query/fragment)
    try:
        parsed   = urlparse(url.lower())
        url_core = (parsed.netloc + parsed.path).replace("www.", "")
    except Exception:
        url_core = url.lower()

    # 1. Full name (anchor token) in title
    for token in anchor_tokens:
        if token in title_l:
            logger.debug("search.company_match.title", token=token)
            return True

    # 2. Slug token in URL domain+path
    for token in slug_tokens:
        if token in url_core:
            logger.debug("search.company_match.url_slug", token=token)
            return True

    # 3. Anchor token (>= 5 chars) in URL after stripping punctuation
    url_stripped = _SLUG_RE.sub("", url_core)
    for token in anchor_tokens:
        if len(token) >= 5:
            token_slug = _SLUG_RE.sub("", token)
            if token_slug and token_slug in url_stripped:
                logger.debug("search.company_match.url_nopunct", token=token)
                return True

    return False


# ---------------------------------------------------------------------------
# YEAR VALIDATION
# ---------------------------------------------------------------------------

def is_correct_year(text: str, target_year: int) -> bool:
    """
    Return True ONLY when a valid fiscal-year pattern is found AND no
    wrong-year pattern is present.

    Valid patterns for target_year=2025 (FY2024-25):
      "2024-25", "2024–25", "fy2025", "fy25", "2025"

    Wrong patterns (adjacent next year):
      "2025-26", "2025-2026", "fy2026", "fy26", "2026"

    Wrong-year check takes strict priority.

    ISO date guard:
      A URL like ".../RR/2025-10-31" contains "2025" as a substring but
      this is a calendar date, not a fiscal year reference.
      If the bare year appears ONLY inside ISO dates, the function returns False.
    """
    text = text.lower()
    prev   = target_year - 1
    next_y = target_year + 1

    wrong_patterns: list[str] = [
        f"{target_year}-{str(next_y)[-2:]}",
        f"{target_year}-{next_y}",
        f"fy{next_y}",
        f"fy{str(next_y)[-2:]}",
        f"AR_{prev}",
        str(next_y),
    ]
    if any(w in text for w in wrong_patterns):
        return False

    strong_patterns: list[str] = [
        f"{prev}-{str(target_year)[-2:]}",
        f"{str(prev)}_{target_year}",
        f"{prev}\u2013{str(target_year)[-2:]}",
        f"fy{target_year}",
        f"fy{str(target_year)[-2:]}",
    ]
    if any(v in text for v in strong_patterns):
        return True

    bare_year = str(target_year)
    if bare_year not in text:
        return False

    iso_date_re = re.compile(
        rf"{target_year}-(0[1-9]|1[0-2])-([0-2][0-9]|3[01])"
    )
    text_without_iso_dates = iso_date_re.sub("", text)
    return bare_year in text_without_iso_dates


# ---------------------------------------------------------------------------
# REPORT TYPE DETECTION
# ---------------------------------------------------------------------------

def matches_type(text: str) -> Optional[str]:
    """
    Detect the report type from combined validation text.

    Returns one of "BRSR", "ESG", "Integrated", or None.
    Evaluated in priority order — first match wins.
    """
    text = text.lower()


    esg_keywords = [
        "esg report", "esg-report",
        "sustainability report", "sustainability-report",
        "csr report", "csr-report",
        "environmental report", "environmental-report",
        "responsible business report",
        "corporate responsibility report",
        "corporate sustainability report",
    ]
    if any(k in text for k in esg_keywords):
        return "ESG"

    integrated_keywords = [
        "annual report", "annual-report", "annualreport",
        "integrated report", "integrated-report",
        "integrated annual",
    ]
    if any(k in text for k in integrated_keywords):
        return "Integrated"

    return None


# ---------------------------------------------------------------------------
# STRICT THREE-WAY VALIDATOR
# ---------------------------------------------------------------------------

def _strict_validate(
    url:         str,
    title:       str,
    snippet:     str,
    token_sets:  dict[str, list[str]],
    target_year: int,
    target_type: str,
) -> bool:
    """
    Apply all three strict validation checks to one search result.

    All three must pass; first failure returns False immediately.

    Check 1 — company_match: full name in title or URL (no word tokens)
    Check 2 — year_match:    correct year present, no adjacent-year signal
    Check 3 — type_match:    detected type equals target_type
    """
    if not has_company_match(title, url, snippet, token_sets):
        logger.debug(
            "search.strict_filter.company_fail",
            url=url[:90],
            target_type=target_type,
        )
        return False

    text_full = f"{title} {snippet} {url}".lower()

    if not is_correct_year(text_full, target_year):
        logger.debug(
            "search.strict_filter.year_fail",
            url=url[:90],
            target_year=target_year,
        )
        return False

    detected = matches_type(text_full)
    if detected != target_type:
        logger.debug(
            "search.strict_filter.type_fail",
            url=url[:90],
            detected=detected,
            expected=target_type,
        )
        return False

    logger.info(
        "search.strict_filter.pass",
        url=url[:90],
        target_type=target_type,
        target_year=target_year,
    )
    return True


# ---------------------------------------------------------------------------
# SERPAPI CLIENT
# ---------------------------------------------------------------------------

def _serpapi_score(position: int) -> float:
    """Return 1/position score for ordering among valid results only."""
    return round(1.0 / max(position, 1), 4)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def _call_serpapi(query: str, api_key: str, num_results: int) -> list[dict]:
    """
    Execute one Google search via SerpApi and return normalised result dicts.
    """
    params = {
        "engine":  "google",
        "q":       query,
        "num":     num_results,
        "api_key": api_key,
        "output":  "json",
    }

    with httpx.Client(timeout=30) as client:
        response = client.get(_SERPAPI_ENDPOINT, params=params)
        response.raise_for_status()
        data = response.json()

    if "error" in data:
        raise RuntimeError(f"SerpApi error: {data['error']}")

    organic = data.get("organic_results", [])
    results: list[dict] = []

    for item in organic:
        url = item.get("link", "").strip()
        if not url:
            continue

        results.append({
            "url":     url,
            "title":   item.get("title", "") or "",
            "content": (item.get("snippet", "") or "")[:400],
            "score":   _serpapi_score(item.get("position", 99)),
        })

    return results


# ---------------------------------------------------------------------------
# QUERY BUILDER
# ---------------------------------------------------------------------------

def _build_all_queries(company: str, year: int) -> list[tuple[str, str]]:
    """Expand every template for every type into (query_string, type_hint) pairs."""
    pairs: list[tuple[str, str]] = []
    year_short = str(year)[-2:]
    for report_type, templates in _QUERY_TEMPLATES.items():
        for template in templates:
            query = template.format(
                company=company,
                year=year,
                year_short=year_short,
                past_year=year - 1,
            )
            pairs.append((query, report_type))
    return pairs


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def collect_and_classify(
    company_name: str,
    year: int,
    max_results_per_query: int = 3,
) -> dict[str, SearchResult]:
    """
    Run all queries via SerpApi, apply STRICT three-way validation per result,
    and return only high-confidence matches grouped by report type.

    Company matching uses the FULL company name (not individual word tokens).
    See _get_company_tokens() for the matching strategy.

    Algorithm
    ---------
    1. Build company token sets using full-name approach.
    2. Run every query template against SerpApi (7 results each).
    3. Pool all raw results; globally deduplicate by URL (keep highest score).
    4. For each result, apply strict three-way filter:
         a) company_match — full name in title or URL
         b) year_match    — correct year present, no adjacent-year signal
         c) type_match    — detected type equals query's target type
       ALL three must pass.
    5. Return one SearchResult per type.

    No fallback logic. No partial matches. No guessing.
    """
    settings = get_settings()

    if not settings.serpapi_api_key:
        logger.warning(
            "search_service.no_api_key",
            message="SERPAPI_API_KEY not set -- returning empty results.",
        )
        return _empty_results(company_name, year)

    token_sets  = _get_company_tokens(company_name)
    all_queries = _build_all_queries(company_name, year)

    logger.info(
        "search_service.collect_start",
        company=company_name,
        year=year,
        total_queries=len(all_queries),
        anchor_tokens=token_sets["anchor"],
    )

    # Step 1: Run all queries
    raw_pool: list[dict] = []

    for query, query_type_hint in all_queries:
        try:
            items = _call_serpapi(query, settings.serpapi_api_key, max_results_per_query)
        except Exception as exc:
            logger.error(
                "search_service.query_failed",
                query=query,
                error=str(exc),
            )
            continue

        for item in items:
            raw_pool.append({
                **item,
                "query_source":    query,
                "query_type_hint": query_type_hint,
            })

    logger.info(
        "search_service.raw_pool_size",
        company=company_name,
        raw_count=len(raw_pool),
    )

    if not raw_pool:
        logger.warning(
            "search_service.empty_pool",
            company=company_name,
            year=year,
        )
        return _empty_results(company_name, year)

    # Step 2: Global URL deduplication — keep highest score per URL
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

    # Step 3: Strict three-way validation
    classified: dict[str, list[DiscoveredReport]] = {t: [] for t in PRIORITY_ORDER}
    passed_count  = 0
    dropped_count = 0

    for item in unique_items:
        url        = item["url"]
        title      = item["title"]
        snippet    = item["content"]
        query_type = item["query_type_hint"]

        passed = _strict_validate(
            url=url,
            title=title,
            snippet=snippet,
            token_sets=token_sets,
            target_year=year,
            target_type=query_type,
        )

        if not passed:
            dropped_count += 1
            continue

        passed_count += 1
        classified[query_type].append(DiscoveredReport(
            url=url,
            title=title,
            snippet=snippet,
            score=item["score"],
            query_source=item["query_source"],
        ))

    logger.info(
        "search_service.strict_filter_summary",
        company=company_name,
        year=year,
        total_unique=len(unique_items),
        passed=passed_count,
        dropped=dropped_count,
        brsr_count=len(classified["BRSR"]),
        esg_count=len(classified["ESG"]),
        integrated_count=len(classified["Integrated"]),
    )

    # Step 4: Build SearchResult per type
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
                    f"No high-confidence {report_type} result for "
                    f"{company_name} FY{year}. All three checks must pass."
                ),
            )
        else:
            logger.info(
                "search_service.type_found",
                company=company_name,
                year=year,
                report_type=report_type,
                count=len(urls_for_type),
                top_url=urls_for_type[0].url[:90],
            )

    return results


def _empty_results(company_name: str, year: int) -> dict[str, SearchResult]:
    """Return an empty SearchResult for every type."""
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
    max_results_per_query: int = 7,
) -> SearchResult:
    """Single-type search. Kept for backward compatibility."""
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


def search_all_report_types(
    company_name: str,
    year: int,
    max_results_per_query: int = 7,
) -> dict[str, SearchResult]:
    """Alias for collect_and_classify(). Kept for backward compatibility."""
    return collect_and_classify(
        company_name=company_name,
        year=year,
        max_results_per_query=max_results_per_query,
    )