"""
services/search_service.py

SerpApi-powered ESG report discovery with STRICT three-way validation.

Validation contract (ALL three must pass — no fallbacks, no partial matches):
  1. Company match  — company token must appear in TITLE or URL domain/path
                      (snippet alone is not sufficient — it can contain any
                      company name in passing context)
  2. Year match     — target fiscal year present, no adjacent-year contamination
  3. Report type    — detected type matches the query type bucket

If ANY of the three checks fails, the result is discarded.
An empty discovered list is returned (and no PDF is downloaded) when no
high-confidence match is found.  The caller must handle the empty case.

Bug fixes in this version
--------------------------
Fix 1 — ETF/fund documents passed via snippet-only company match:
    Symptom (seen in log): "dokumenty.analizy.pl/.../2025-10-31" was accepted
    as a Kennametal Integrated report.  The document is an iShares ETF fund
    report — Kennametal appeared only in its "Top Holdings" snippet line.

    Root cause: has_company_match() case 4 allowed snippet-only matches on
    any domain that was not in _AGGREGATOR_DOMAINS.  The ETF site was not in
    that list, so it passed.

    Fix: case 4 now checks a whitelist (_TRUSTED_FILING_DOMAINS) instead of
    a blacklist (_AGGREGATOR_DOMAINS).  Snippet-only matching is permitted ONLY
    on known exchange/regulatory filing domains (BSE, NSE, SEBI, MCA).  Every
    other domain requires the company name in the title or URL.

Fix 2 — ISO calendar dates matched as valid fiscal year signal:
    Symptom: URL path "...RR/2025-10-31" contains "2025" as a substring.
    is_correct_year() treated "2025" as a valid bare-year hit for FY2025.

    Root cause: The bare-year pattern ("2025") matched any substring occurrence
    including calendar dates like 2025-10-31 (Oct 31 2025).

    Fix: is_correct_year() strips all ISO date occurrences (YYYY-MM-DD) before
    testing the bare year.  If "2025" only appears inside ISO dates and no
    proper fiscal-year pattern (2024-25, fy2025, fy25) is present, the function
    returns False.  Documents with a proper FY pattern always have at least one
    of the strong patterns in title or snippet.

Search backend
--------------
Uses the SerpApi Google Search endpoint (https://serpapi.com/search) via
direct httpx calls.  Required env var: SERPAPI_API_KEY.

SerpApi request parameters used
---------------------------------
  engine  = "google"
  q       = <query string>
  num     = 7              (results per query)
  api_key = <key>
  output  = "json"

Score synthesis
---------------
score = 1.0 / position  (ordering only — not used for filtering)

STRICT validation pipeline (all three required)
-----------------------------------------------
text_full   = (title + url + snippet).lower()
text_anchor = (title + url).lower()         ← stricter surface for company

Step 1 — company_match (STRICT):
  At least one company token must appear in the ANCHOR text (title + url).
  Snippet matching is allowed ONLY on trusted exchange/regulatory filing
  domains (BSE, NSE, SEBI, MCA) where the snippet IS the official filing
  announcement, not a passing mention in a portfolio or news article.

Step 2 — year_match:
  Valid   : "2024-25", "2024–25", "fy2025", "fy25", "2025"
  Wrong   : "2025-26", "2025-2026", "fy2026", "fy26", "2026"
  ISO date guard: bare "2025" inside "2025-10-31" does NOT count.
  Wrong-year check takes priority.

Step 3 — type_match:
  BRSR        : brsr, business responsibility
  ESG         : "esg report", "sustainability report", "csr report",
                "environmental report" (must have "report" qualifier)
  Integrated  : "annual report", "integrated report", "integrated annual",
                "annual-report"

  The detected type must exactly equal the query's target type bucket.

No fallback. No partial matches. No score-based override.
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
# COMPANY TOKEN BUILDER
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]")

_STOP_WORDS: frozenset[str] = frozenset({
    "the", "and", "of", "for", "from", "limited", "ltd", "inc",
    "corp", "corporation", "private", "pvt", "services", "solutions",
    "technologies", "technology", "india", "group", "holdings",
    "international", "global", "company", "enterprises", "industries",
})

# Known short ticker → URL/title anchor strings
_TICKER_ALIASES: dict[str, list[str]] = {
    "tcs":         ["tata consultancy", "tcs.com"],
    "infosys":     ["infosys"],
    "wipro":       ["wipro"],
    "hcl":         ["hcltech", "hcl technologies"],
    "bpcl":        ["bharat petroleum"],
    "hpcl":        ["hindustan petroleum"],
    "iocl":        ["indianoil", "indian oil"],
    "ongc":        ["ongcindia"],
    "ntpc":        ["ntpclimited"],
    "sail":        ["steelauthority"],
    "gail":        ["gailonline"],
    "sbi":         ["statebankofin"],
    "itc":         ["itcportal"],
    "ltim":        ["ltimindtree"],
    "kennametal":  ["kennametal"],
}

# Domains that aggregate many companies — snippet matches on these are rejected
_AGGREGATOR_DOMAINS: frozenset[str] = frozenset({
    "ishares.com", "vanguard.com", "blackrock.com", "morningstar.com",
    "bloomberg.com", "reuters.com", "wsj.com", "ft.com", "economist.com",
    "nordstrom.com", "cision.com", "prnewswire.com", "businesswire.com",
    "globenewswire.com", "sec.gov", "edgar.gov", "cityof",
    "moneycontrol.com", "economictimes.indiatimes.com", "livemint.com",
    "thehindu.com", "ndtv.com", "screener.in", "tickertape.in",
    "q4cdn.com", "annualreport.co", "annualreports.com",
    "globalreporting.org", "sustainalytics.com", "msci.com",
    "s&p.com", "spglobal.com", "refinitiv.com",
})

# Domains that are trusted regulatory/exchange filing hosts.
# On these domains, a company name appearing ONLY in the snippet is still
# a reliable signal — the snippet is the filing announcement text, not a
# passing mention in a fund portfolio or news article.
# All other (unknown) domains require the company to appear in title or URL.
_TRUSTED_FILING_DOMAINS: frozenset[str] = frozenset({
    "bseindia.com",
    "bsmedia.business-standard.com",  # BSE regulatory filings mirror
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
    Build company token sets for two validation surfaces:
      "anchor" — tokens checked in title or URL domain+path (primary check)
      "slug"   — alphanumeric slug forms for URL matching

    Returns:
        {
            "anchor": [str, ...],
            "slug":   [str, ...],
        }
    """
    name_lower = company_name.lower().strip()
    slug = _SLUG_RE.sub("", name_lower)

    anchor_tokens: str = [name_lower]
    slug_tokens:   list[str] = [slug]

    # Alias table
    if slug in _TICKER_ALIASES:
        anchor_tokens.extend(_TICKER_ALIASES[slug])

    # Meaningful words (>= 5 chars, not stop words)
    words = [
        w.strip(".,()&-")
        for w in re.split(r"[\s&.,/\(\)\-]+", name_lower)
        if w.strip(".,()&-") and w.strip(".,()&-") not in _STOP_WORDS
    ]

    for word in words:
        if len(word) >= 5:
            anchor_tokens.append(word)
            slug_tokens.append(_SLUG_RE.sub("", word))

    # Two-word phrases
    for i in range(len(words) - 1):
        phrase = f"{words[i]} {words[i + 1]}"
        if len(phrase) >= 6:
            anchor_tokens.append(phrase)

    # Deduplicate preserving order
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
    Return True when at least one company token is found in TITLE or URL.

    Priority order:
      1. Anchor token in title (case-insensitive substring)
      2. Slug token in URL domain+path (no query string)
      3. Anchor token (>= 5 chars) in URL domain+path after stripping punctuation
      4. Full company name (anchor[0], >= 6 chars) verbatim in snippet,
         BUT only when URL is not an aggregator domain

    Snippet-only matches are rejected unless condition 4 applies.

    Args:
        title:      Page title string.
        url:        Full URL string.
        snippet:    Search snippet string.
        token_sets: Output of _get_company_tokens().

    Returns:
        True if match found; False otherwise.
    """
    anchor_tokens = token_sets["anchor"]
    slug_tokens   = token_sets["slug"]

    title_l   = title.lower()
    snippet_l = snippet.lower()

    # Parse URL — use netloc+path only (drop query/fragment)
    try:
        parsed   = urlparse(url.lower())
        url_core = (parsed.netloc + parsed.path).replace("www.", "")
    except Exception:
        url_core = url.lower()

    # 1. Anchor token in title
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

    # 4. Snippet match — ONLY on trusted regulatory/exchange filing domains.
    #
    #    Why this restriction:
    #      An ETF, fund, or portfolio document may mention any company name in
    #      its "Top Holdings" section.  A news article or sector report may
    #      mention a company in passing.  Accepting a snippet-only match on
    #      unknown domains lets these through Gates 2 and 3.
    #
    #    Example of a WRONG pass (before this fix):
    #      URL:     dokumenty.analizy.pl/etf/E_ISHII045_A_USD/RR/2025-10-31
    #      Snippet: "... Kennametal Inc. (KMT) 0.42% ..."  ← holding line
    #      The snippet mentioned "kennametal" but the document is an ETF report.
    #
    #    On trusted exchange filing domains (BSE, NSE, SEBI, MCA), the snippet
    #    is the official filing announcement text — reliable evidence of the
    #    company's own document, not a passing mention.
    # if anchor_tokens and len(anchor_tokens[0]) >= 6:
    #     domain = parsed.netloc.lower().replace("www.", "") if parsed else ""
    #     is_trusted_filer = any(d in domain for d in _TRUSTED_FILING_DOMAINS)
    #     if is_trusted_filer and anchor_tokens[0] in snippet_l:
    #         logger.debug("search.company_match.snippet_trusted", token=anchor_tokens[0])
    #         return True

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

    ISO date guard (added to fix ETF/fund document false positives):
      A URL like ".../RR/2025-10-31" contains "2025" as a substring but
      this is a calendar date (Oct 31 2025), not a fiscal year reference.
      If the bare year appears ONLY as part of an ISO date (YYYY-MM-DD)
      and no proper fiscal-year pattern is also present, the function
      returns False.  A document that genuinely covers FY2025 will always
      have at least one of: "2024-25", "fy2025", or "fy25" somewhere in
      its title or snippet — the bare year alone is not accepted when it
      is embedded in an ISO date string.

    Args:
        text:        Combined validation text, lowercased.
        target_year: Fiscal year end integer.

    Returns:
        True only if correct-year fiscal signal present and no wrong-year signal.
    """
    text = text.lower()
    prev   = target_year - 1
    next_y = target_year + 1

    wrong_patterns: list[str] = [
        f"{target_year}-{str(next_y)[-2:]}",      # "2025-26"
        f"{target_year}-{next_y}",                # "2025-2026"
        f"fy{next_y}",                            # "fy2026"
        f"fy{str(next_y)[-2:]}",   
        f"AR_{prev}",               # "fy26"
        str(next_y),                              # "2026"
    ]
    if any(w in text for w in wrong_patterns):
        return False

    # Strong fiscal-year patterns (unambiguous — never part of a date string)
    strong_patterns: list[str] = [
        f"{prev}-{str(target_year)[-2:]}", 
        f"{str(prev)}_{target_year}",       # "2024-25"
        f"{prev}\u2013{str(target_year)[-2:]}",   # "2024–25" (en-dash)
        f"fy{target_year}",                        # "fy2025"
        f"fy{str(target_year)[-2:]}",             # "fy25"
    ]
    if any(v in text for v in strong_patterns):
        return True

    # Bare year ("2025") — accepted ONLY when NOT embedded in an ISO calendar date.
    #
    # ISO date pattern: YYYY-MM-DD where MM is 01-12 and DD is 01-31.
    # Example: "2025-10-31" in a URL path is a fund document date, not FY2025.
    # If every occurrence of the bare year is inside an ISO date, reject.
    bare_year = str(target_year)
    if bare_year not in text:
        return False

    # Check whether the bare year occurs outside of any ISO date context.
    # We remove all ISO-date occurrences from the text and see if the year remains.
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

    Note: bare "sustainability" or "esg" without "report" qualifier is
    intentionally excluded to avoid matching index fund documents, news
    articles, and sector analysis that mention sustainability in passing.

    Args:
        text: Combined validation text (title + snippet + url), lowercased.

    Returns:
        Detected type string or None.
    """
    text = text.lower()

    # BRSR — highest priority; very specific keyword
    if any(k in text for k in ["brsr", "business responsibility"]):
        return "BRSR"

    # ESG — requires "report" qualifier
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

    # Integrated / Annual
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

    Check 1 — company_match:
      Token in title or URL domain+path (snippet-only rejected).

    Check 2 — year_match:
      Correct year present, no adjacent-year contamination.

    Check 3 — type_match:
      Detected type equals target_type.

    Args:
        url:         Result URL string.
        title:       Page title (may be empty).
        snippet:     Search snippet (may be empty).
        token_sets:  Output of _get_company_tokens().
        target_year: Fiscal year end integer.
        target_type: One of "BRSR", "ESG", "Integrated".

    Returns:
        True if and only if all three checks pass.
    """
    # Check 1: company — must appear in title or URL
    if not has_company_match(title, url, snippet, token_sets):
        logger.debug(
            "search.strict_filter.company_fail",
            url=url[:90],
            target_type=target_type,
        )
        return False

    # Build combined text for year + type checks
    text_full = f"{title} {snippet} {url}".lower()

    # Check 2: year
    if not is_correct_year(text_full, target_year):
        logger.debug(
            "search.strict_filter.year_fail",
            url=url[:90],
            target_year=target_year,
        )
        return False

    # Check 3: type
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

    Each returned dict has:
        url          str   — the page URL (SerpApi "link" field)
        title        str   — the page title
        content      str   — snippet, truncated to 400 chars
        score        float — synthesised from position (1/position)
        query_source str   — set by caller

    Raises:
        RuntimeError           on SerpApi-level errors.
        httpx.HTTPStatusError  on non-2xx responses after retries.
        httpx.TimeoutException on network timeout.
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
    max_results_per_query: int = 7,
) -> dict[str, SearchResult]:
    """
    Run all queries via SerpApi, apply STRICT three-way validation per result,
    and return only high-confidence matches grouped by report type.

    Algorithm
    ---------
    1. Build company token sets once (anchor tokens for title/URL matching).
    2. Run every query template against SerpApi (7 results each).
    3. Pool all raw results; globally deduplicate by URL (keep highest score).
    4. For each result, apply strict three-way filter:
         a) company_match — token in title or URL (not snippet-only)
         b) year_match    — correct year present, no adjacent-year signal
         c) type_match    — detected type equals query's target type
       ALL three must pass.  Any failure = result silently discarded.
    5. Return one SearchResult per type.  Empty discovered list when no
       high-confidence result found.

    No fallback logic. No partial matches. No guessing.
    No PDF download unless all three checks pass.

    Args:
        company_name:          Company name, e.g. "Kennametal".
        year:                  Fiscal year end integer, e.g. 2025.
        max_results_per_query: SerpApi num= parameter (7 recommended).

    Returns:
        {
            "BRSR":       SearchResult(discovered=[...] or []),
            "ESG":        SearchResult(discovered=[...] or []),
            "Integrated": SearchResult(discovered=[...] or []),
        }
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
        anchor_tokens=token_sets["anchor"][:6],
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