"""
services/search_service.py

SerpApi-powered ESG report discovery with URL-keyword classification.

Search backend
--------------
Uses the SerpApi Google Search endpoint (https://serpapi.com/search) via
direct httpx calls — no serpapi Python package required.  Only httpx and
tenacity (already in the project's dependencies) are needed.

Required env var:  SERPAPI_API_KEY
Config key:        settings.serpapi_api_key

SerpApi request parameters used
---------------------------------
  engine  = "google"          Google web search
  q       = <query string>    The search query
  num     = <max_results>     Results per page (max 100; we use 10)
  api_key = <key>             Authentication
  output  = "json"            Response format

SerpApi response fields consumed
----------------------------------
  organic_results[].link      The result URL  (NOTE: "link", not "url")
  organic_results[].title     Page title
  organic_results[].snippet   Short description
  organic_results[].position  1-based rank (used to synthesise a score)

Score synthesis
---------------
SerpApi does not return a relevance float the way Tavily does.
We synthesise one from position:   score = 1.0 / position
  position 1  -> score 1.00   (top result)
  position 5  -> score 0.20
  position 10 -> score 0.10

This score plays the same role as Tavily's score field throughout the
rest of the pipeline (deduplication keeps highest score, re-ranking
uses it as the "original weight" component).

Company-awareness layer
------------------------
The pipeline runs in this order:

  1. Search        — query SerpApi Google Search
  2. Deduplicate   — global URL dedup (keep highest score per URL)
  3. Entity score  — score each result against the target company
  4. Domain-aware filter — three-tier domain categorisation:
       TRUSTED  -> always KEEP  (official filing portals; bypass entity filter)
       NEGATIVE -> always DROP  (domain belongs to a different company)
       UNKNOWN  -> apply entity_score > 0 gate
                   AND drop if explicit wrong-year signal found with no
                   target-year signal (year-aware hard filter)
  5. Year score    — boost results matching the requested fiscal year;
                     penalise and hard-drop results for adjacent years
  5. Re-rank       — combined (entity + search) score
  6. Classify      — type assignment by URL keywords
  7. Assign        — BRSR > ESG > Integrated priority

Nothing changes for callers: signatures, return types, and the
classify/assign contract are identical to the original.

Domain-aware filter rules (step 4)
------------------------------------
Evaluated in strict order — first matching rule wins:

  Rule 1: domain is TRUSTED  -> KEEP immediately, skip entity scoring.
           Trusted = official regulatory/filing portals: nseindia.com,
           bseindia.com, msei.in, sebi.gov.in, etc.  These portals host
           filings for ALL companies so entity scoring is irrelevant; what
           matters is that the URL was returned by a query for the right
           company.  We trust SerpApi query intent here.

  Rule 2: domain is NEGATIVE -> DROP immediately.
           Negative = domain is definitively owned by a DIFFERENT company
           (e.g. tatasteel.com while searching TCS).

  Rule 3: domain is UNKNOWN  -> apply entity scoring.
           Keep only if entity_score > 0  (at least one alias appears
           somewhere in title / URL / snippet).
           entity_score == 0  means zero textual evidence of the company
           -> DROP (hard negative gate).

Fallback safety net
--------------------
If every result is dropped after filtering, the pipeline reverts to
all non-negative-domain results in original SerpApi order.  This prevents
returning an empty result for genuinely obscure / newly listed companies.

How a link passes the filter (examples)
-----------------------------------------
  URL: https://archives.nseindia.com/annual_reports/wipro-brsr-2024.pdf
    domain_tier = TRUSTED  -> Rule 1 fires -> KEEP
    (entity scoring is skipped entirely)

  URL: https://www.wipro.com/content/dam/wipro/documents/brsr-2024.pdf
    Searching for: Wipro
    domain_tier = UNKNOWN (wipro.com is own domain, not a rival)
    entity_score("wipro" in title/url/snippet) > 0 -> KEEP (Rule 3)

  URL: https://www.tatasteel.com/sustainability-report-2024.pdf
    Searching for: TCS
    domain_tier = NEGATIVE (tatasteel.com -> rival) -> Rule 2 -> DROP

  URL: https://www.someanalysisblog.com/tcs-esg-report-2024
    domain_tier = UNKNOWN
    entity_score: "tcs" in URL path -> score > 0 -> KEEP (Rule 3)

  URL: https://www.someanalysisblog.com/india-esg-trends-2024
    domain_tier = UNKNOWN
    entity_score: no alias found anywhere -> score == 0 -> DROP (Rule 3)
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

# SerpApi Google Search endpoint — accessed directly via httpx (no serpapi package needed)
_SERPAPI_ENDPOINT = "https://serpapi.com/search"

# ---------------------------------------------------------------------------
# Report type priorities and query templates
# ---------------------------------------------------------------------------

PRIORITY_ORDER: list[str] = ["BRSR", "ESG", "Integrated"]
ALL_REPORT_TYPES: list[str] = PRIORITY_ORDER
DEFAULT_REPORT_TYPE = "BRSR"

_QUERY_TEMPLATES: dict[str, list[str]] = {
    "BRSR": [
        "{company} BRSR {past_year}-{year} nseindia filetype:pdf",
    ],
    "ESG": [
        "{company} ESG report {past_year}-{year} filetype:pdf",
        "{company} sustainability report {past_year}-{year} filetype:pdf",
    ],
    "Integrated": [
        "{company} annual report {past_year}-{year} filetype:pdf",
    ],
}

_URL_KEYWORD_RULES: dict[str, list[tuple[str, ...]]] = {
    "BRSR": [
        ("business", "sustainability"),
        ("business", "responsibility"),
        ("brsr"),
    ],
    "ESG": [
        ("esg",),
        ("sustainability",),
        ("environmental",),
    ],
    "Integrated": [
        ("integrated", "report"),
        ("integrated", "annual"),
        ("integrated",),
        ("annual"),
    ],
}

# =============================================================================
# DOMAIN TIER DEFINITIONS
# =============================================================================

# ---------------------------------------------------------------------------
# TRUSTED DOMAINS
# Official regulatory/filing portals that host filings for ALL companies.
# Results from these domains bypass entity filtering entirely (Rule 1).
# The query intent (company name in the search query) is sufficient signal.
#
# Matching: a URL is TRUSTED if its netloc ends with any suffix in this set.
# Suffix matching handles subdomains automatically:
#   "nseindia.com" matches archives.nseindia.com, www.nseindia.com, etc.
# ---------------------------------------------------------------------------
_TRUSTED_DOMAIN_SUFFIXES: frozenset[str] = frozenset({
    # Stock exchanges (India)
    "nseindia.com",
    "nsearchive.nseindia.com",   # explicit archive subdomain
    "bseindia.com",
    "msei.in",
    # Regulator
    "sebi.gov.in",
    # Ministry of Corporate Affairs
    "mca.gov.in",
    # NSE/BSE filing portals
    "connect2nse.com",
    "listing.bseindia.com",
    # NSDL / CDSL (used for some filings)
    "nsdl.co.in",
    "cdsl.co.in",
    # Government portals that host annual reports
    "india.gov.in",
    "nic.in",
})


def _domain_tier_trusted(domain: str) -> bool:
    """
    Return True if the domain is a TRUSTED filing/regulatory portal.

    Checks whether the domain ends with (or equals) any suffix in the
    trusted set.  This handles subdomains automatically.

    Examples:
        archives.nseindia.com  -> ends with nseindia.com  -> True
        www.bseindia.com       -> ends with bseindia.com  -> True
        infosys.com            -> no suffix match          -> False
    """
    domain = domain.lower().lstrip("www.")
    for suffix in _TRUSTED_DOMAIN_SUFFIXES:
        if domain == suffix or domain.endswith("." + suffix):
            return True
    return False


# ---------------------------------------------------------------------------
# ENTITY ALIAS TABLE
# ---------------------------------------------------------------------------
_ENTITY_ALIAS_TABLE: dict[str, list[str]] = {
    # IT
    "tcs":            ["tata consultancy services"],
    "infosys":        ["infosys limited", "infy"],
    "wipro":          ["wipro limited"],
    "hcl":            ["hcl technologies", "hcltech"],
    "ltim":           ["ltimindtree", "larsen toubro infotech", "mindtree"],
    "ltimindtree":    ["ltimindtree", "larsen toubro infotech"],
    "techm":          ["tech mahindra"],
    "techmahindra":   ["tech mahindra"],
    "mphasis":        ["mphasis"],
    "hexaware":       ["hexaware"],
    "persistent":     ["persistent systems"],
    "coforge":        ["coforge", "niit technologies"],
    "cyient":         ["cyient"],
    # Oil & Gas
    "bpcl":           ["bharat petroleum", "bharat petroleum corporation"],
    "hpcl":           ["hindustan petroleum", "hindustan petroleum corporation"],
    "iocl":           ["indian oil", "indian oil corporation"],
    "indianoil":      ["indian oil", "indian oil corporation"],
    "ongc":           ["oil and natural gas", "oil and natural gas corporation"],
    "oilindia":       ["oil india", "oil india limited"],
    "gail":           ["gas authority of india", "gail india"],
    "mrpl":           ["mangalore refinery"],
    "cpcl":           ["chennai petroleum"],
    "nrl":            ["numaligarh refinery"],
    # Power & Utilities
    "ntpc":           ["national thermal power", "ntpc limited"],
    "powergrid":      ["power grid corporation", "pgcil"],
    "nhpc":           ["nhpc limited", "national hydroelectric"],
    "torrentpower":   ["torrent power"],
    "adanigreen":     ["adani green energy"],
    "adanipower":     ["adani power"],
    # Metals & Mining
    "tatasteel":      ["tata steel"],
    "jsw":            ["jsw steel", "jsw energy"],
    "jswsteel":       ["jsw steel"],
    "sail":           ["steel authority of india"],
    "hindalco":       ["hindalco industries", "aditya birla"],
    "vedanta":        ["vedanta limited", "vedanta resources"],
    "nalco":          ["national aluminium", "nalco"],
    "moil":           ["manganese ore india"],
    # Pharma
    "sunpharma":      ["sun pharmaceutical", "sun pharma"],
    "drreddy":        ["dr reddy", "dr reddys laboratories"],
    "cipla":          ["cipla limited"],
    "lupin":          ["lupin limited"],
    "aurobindo":      ["aurobindo pharma"],
    # Auto
    "tatamotors":     ["tata motors"],
    "maruti":         ["maruti suzuki", "msil"],
    "marutisuzuki":   ["maruti suzuki"],
    "mahindra":       ["mahindra and mahindra", "m&m"],
    "bajaj":          ["bajaj auto"],
    "heromoto":       ["hero motocorp"],
    # Finance
    "hdfc":           ["hdfc bank", "hdfc limited"],
    "hdfcbank":       ["hdfc bank"],
    "icici":          ["icici bank"],
    "icicbank":       ["icici bank"],
    "sbi":            ["state bank of india"],
    "kotak":          ["kotak mahindra bank"],
    "axisbank":       ["axis bank"],
    # FMCG / Consumer
    "itc":            ["itc limited"],
    "hindustan":      ["hindustan unilever", "hul"],
    "hul":            ["hindustan unilever"],
    "nestle":         ["nestle india"],
    "dabur":          ["dabur india"],
    "britannia":      ["britannia industries"],
    "godrej":         ["godrej consumer", "godrej industries"],
    # Conglomerates / Others
    "reliance":       ["reliance industries", "ril"],
    "ril":            ["reliance industries"],
    "adani":          ["adani group", "adani enterprises"],
    "lt":             ["larsen toubro", "l&t"],
    "larsentoubro":   ["larsen toubro", "l&t"],
    "bajajfinance":   ["bajaj finance"],
    "ultracemco":     ["ultratech cement"],
    "acc":            ["acc limited", "acc cement"],
    "ambujacement":   ["ambuja cement"],
    "asianpaint":     ["asian paints"],
    "pidilite":       ["pidilite industries"],
    "divi":           ["divis laboratories"],
    "torrent":        ["torrent pharma", "torrent pharmaceuticals"],
}

# Domains definitively owned by a specific company slug.
# Key = domain suffix; Value = owner slug.
# Used only for NEGATIVE detection — "is this domain owned by a RIVAL?".
_DOMAIN_OWNERS: dict[str, str] = {
    # IT
    "tcs.com":                  "tcs",
    "infosys.com":              "infosys",
    "wipro.com":                "wipro",
    "hcltech.com":              "hcl",
    "ltimindtree.com":          "ltimindtree",
    "techmahindra.com":         "techmahindra",
    "mphasis.com":              "mphasis",
    "hexaware.com":             "hexaware",
    "persistent.com":           "persistent",
    "coforge.com":              "coforge",
    # Oil & Gas
    "bpcl.in":                  "bpcl",
    "bharatpetroleum.com":      "bpcl",
    "hindpetro.com":            "hpcl",
    "iocl.com":                 "iocl",
    "indianoil.in":             "iocl",
    "ongcindia.com":            "ongc",
    "ongc.co.in":               "ongc",
    "oilindia.in":              "oilindia",
    "oil-india.com":            "oilindia",
    "gailonline.com":           "gail",
    "gail.co.in":               "gail",
    # Power
    "ntpc.co.in":               "ntpc",
    "ntpclimited.com":          "ntpc",
    "powergridindia.com":       "powergrid",
    "nhpcindia.com":            "nhpc",
    "adanigreen.com":           "adanigreen",
    "adanipower.com":           "adanipower",
    # Metals
    "tatasteel.com":            "tatasteel",
    "jswsteel.com":             "jswsteel",
    "sail.co.in":               "sail",
    "hindalco.com":             "hindalco",
    "vedanta.com":              "vedanta",
    "nalcoindia.com":           "nalco",
    # Tata group — individual entities
    "tata.com":                 "tata",
    "tatamotors.com":           "tatamotors",
    "tatapower.com":            "tatapower",
    "tatacommunications.com":   "tatacommunications",
    "tataconsumer.com":         "tataconsumer",
    "tatachemicals.com":        "tatachemicals",
    # Pharma
    "sunpharma.com":            "sunpharma",
    "drreddys.com":             "drreddy",
    "cipla.com":                "cipla",
    "lupin.com":                "lupin",
    # Finance
    "hdfcbank.com":             "hdfcbank",
    "icicibank.com":            "icicibank",
    "sbi.co.in":                "sbi",
    "kotak.com":                "kotak",
    "axisbank.com":             "axisbank",
    # FMCG
    "itcportal.com":            "itc",
    "hul.co.in":                "hul",
    # Others
    "ril.com":                  "ril",
    "adanienterprises.com":     "adani",
    "adanigroup.com":           "adani",
}

_SLUG_RE = re.compile(r"[^a-z0-9]")


def _company_slug(name: str) -> str:
    """Lowercase, strip non-alphanumeric characters."""
    return _SLUG_RE.sub("", name.lower().strip())


def expand_company_aliases(company_name: str) -> frozenset[str]:
    """
    Expand a company name into a set of lowercase alias strings.

    Steps:
      1. Add the raw lowercased name and its alphanumeric slug.
      2. Look up the slug in the hand-curated alias table.
      3. Split on common separators and add non-trivial parts.
      4. Add each word in multi-word names (if word length >= 3).

    Returns a frozenset so callers can cache it if needed.
    """
    name_lower = company_name.lower().strip()
    slug       = _company_slug(name_lower)
    aliases: set[str] = set()

    aliases.add(name_lower)
    aliases.add(slug)

    if slug in _ENTITY_ALIAS_TABLE:
        aliases.update(_ENTITY_ALIAS_TABLE[slug])
    else:
        for key, expansions in _ENTITY_ALIAS_TABLE.items():
            if slug.startswith(key) or key.startswith(slug):
                aliases.update(expansions)
                break

    parts = re.split(r"[\s&.,/\(\)\-]+", name_lower)
    for part in parts:
        part = part.strip()
        if len(part) >= 2:
            aliases.add(part)
            aliases.add(_company_slug(part))

    for word in name_lower.split():
        w = word.strip(".,()&-")
        if len(w) >= 3:
            aliases.add(w)

    aliases.discard("")
    aliases.discard(" ")
    return frozenset(aliases)


# =============================================================================
# DOMAIN TIER RESOLUTION
# =============================================================================

# A URL's domain falls into exactly one of three tiers, checked in this order:
#   TRUSTED  -> Rule 1: KEEP immediately
#   NEGATIVE -> Rule 2: DROP immediately
#   UNKNOWN  -> Rule 3: apply entity_score gate

_TIER_TRUSTED  = "trusted"
_TIER_NEGATIVE = "negative"
_TIER_UNKNOWN  = "unknown"


def _extract_domain(url: str) -> str:
    """Return the netloc of the URL, lowercased, www. stripped."""
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def _domain_owner_slug(domain: str) -> Optional[str]:
    """
    Return the canonical company slug that owns *domain*, or None.
    Matches by checking if any known domain suffix appears in the netloc.
    """
    for known_domain, owner_slug in _DOMAIN_OWNERS.items():
        if known_domain in domain:
            return owner_slug
    return None


def _resolve_domain_tier(domain: str, company_aliases: frozenset[str]) -> str:
    """
    Classify a domain into one of three tiers.

    Evaluation order (first match wins):
      1. TRUSTED   -- domain matches a known regulatory / filing portal suffix.
      2. NEGATIVE  -- domain is owned by a DIFFERENT company (rival domain).
      3. UNKNOWN   -- everything else; entity scoring decides.

    Args:
        domain:           Cleaned netloc string (no www., lowercase).
        company_aliases:  Expanded alias set for the target company.

    Returns:
        One of _TIER_TRUSTED, _TIER_NEGATIVE, _TIER_UNKNOWN.

    NEGATIVE detection note:
        Only marked NEGATIVE when _DOMAIN_OWNERS has a positive mapping AND
        the owner is demonstrably NOT the target company.  Unknown domains
        are never marked NEGATIVE.

        False-positive avoidance:
            Searching "TCS": tcs.com -> owner slug "tcs" -> aliases contain
            "tcs" -> NOT negative -> falls through to UNKNOWN, where
            entity scoring scores > 0 and keeps it.
    """
    # Rule 1: TRUSTED -- official filing portals bypass all entity scoring
    if _domain_tier_trusted(domain):
        return _TIER_TRUSTED

    # Rule 2: NEGATIVE -- rival company domain
    owner = _domain_owner_slug(domain)
    if owner is not None:
        owner_matches_target = any(
            owner in alias or alias in owner
            for alias in company_aliases
            if len(alias) >= 2
        )
        if not owner_matches_target:
            return _TIER_NEGATIVE
        # Own domain -- falls through to UNKNOWN; entity scoring will confirm

    # Rule 3: UNKNOWN -- apply entity scoring
    return _TIER_UNKNOWN


# =============================================================================
# ENTITY RELEVANCE SCORING
# =============================================================================

def score_entity_relevance(
    url: str,
    title: str,
    snippet: str,
    company_aliases: frozenset[str],
) -> float:
    """
    Return a [0, 1] score representing how strongly this search result
    belongs to the target company.

    Signal weights (additive, capped at 1.0):
      - Title match:    0.50  per alias hit  (strong -- title is curated)
      - Domain match:   0.30  per alias hit  (medium -- domain is authoritative)
      - URL path match: 0.20  per alias hit  (medium -- path often has ticker)
      - Snippet match:  0.10  per alias hit  (weak   -- snippet can be noisy)

    Short aliases (< 3 chars) are skipped to avoid false positives from
    common abbreviations like "of", "in", "lt".
    """
    url_l   = url.lower()
    title_l = (title   or "").lower()
    snip_l  = (snippet or "").lower()
    domain  = _extract_domain(url)

    score = 0.0
    for alias in company_aliases:
        if len(alias) < 2:
            continue

        if alias in title_l:
            score += 0.50

        alias_slug = _company_slug(alias)
        if alias_slug and (alias_slug in domain.replace(".", "") or alias in domain):
            score += 0.30
        elif alias in url_l:
            score += 0.20

        if alias in snip_l:
            score += 0.10

    return min(score, 1.0)


# =============================================================================
# DOMAIN-AWARE FILTER + RERANK
# =============================================================================

# Weight applied to entity relevance when combining with the search score.
# Combined score = entity_weight * entity + year_weight * year + orig_weight * search
# Year signal is strong: wrong-year results should never beat right-year ones.
_ENTITY_WEIGHT   = 0.35
_YEAR_WEIGHT     = 0.30
_ORIGINAL_WEIGHT = 0.35


def _year_patterns(year: int) -> tuple[list[str], list[str]]:
    """
    Return (target_patterns, adjacent_patterns) for a fiscal year end integer.

    Indian fiscal years run Apr–Mar, so year=2024 means FY2023-24.
    We generate all common representations:
        - "2023-24", "2023-2024"       (short and long hyphen forms)
        - "fy2024", "fy24", "fy 2024"  (FY prefix forms)
        - bare "2024"                  (year-only, weakest signal)

    Adjacent years are year-1 and year+1 in the same representations.
    """
    py = year - 1   # past year  (2023 for year=2024)

    target: list[str] = [
        f"FY{py}-{str(year)[-2:]}",   # "2023-24"
        f"FY{py}-{year}",             # "2023-2024"
        f"fy{year}",                # "fy2024"
        f"fy{str(year)[-2:]}",      # "fy24"
        f"fy {year}",               # "fy 2024"
        str(year),                  # "2024"  (weak – last resort)
    ]

    # Adjacent year -1: e.g. FY2022-23 when searching for FY2023-24
    py1, y1 = year - 2, year - 1
    # Adjacent year +1: e.g. FY2024-25 when searching for FY2023-24
    py2, y2 = year,     year + 1

    adjacent: list[str] = [
        f"{py1}-{str(y1)[-2:]}", f"{py1}-{y1}", f"fy{y1}", f"fy{str(y1)[-2:]}",
        f"{py2}-{str(y2)[-2:]}", f"{py2}-{y2}", f"fy{y2}", f"fy{str(y2)[-2:]}",
    ]

    return target, adjacent


def score_year_relevance(
    url: str,
    title: str,
    snippet: str,
    year: int,
) -> tuple[float, bool]:
    """
    Return (year_score, is_wrong_year) for a search result.

    year_score:
        1.0  -- strong target-year signal (hyphenated form or FY prefix found)
        0.7  -- weak target-year signal (bare year digit only)
        0.5  -- no year signal at all (neutral; don't penalise, don't boost)
        0.0  -- explicit adjacent-year signal with NO target-year signal

    is_wrong_year:
        True when an adjacent-year pattern appears AND no target-year pattern
        appears anywhere in url+title+snippet.  Used as a hard-drop signal
        for non-trusted domains.

    Matching is case-insensitive.  We check url+title+snippet concatenated so
    a match in any field counts.
    """
    haystack = " ".join([url, title or "", snippet or ""]).lower()

    target_pats, adjacent_pats = _year_patterns(year)

    # Strong target patterns: all except the bare year digit
    strong_targets = target_pats[:-1]   # everything except str(year)
    weak_target    = target_pats[-1]    # bare year digit, e.g. "2024"

    has_strong   = any(p in haystack for p in strong_targets)
    has_adjacent = any(p in haystack for p in adjacent_pats)

    # Weak match: bare year is present BUT not merely embedded inside an
    # adjacent-year pattern.  E.g. "2024-25" contains "2024" as a substring —
    # that shouldn't count as a target-year hit.
    haystack_stripped = haystack
    for adj in adjacent_pats:
        haystack_stripped = haystack_stripped.replace(adj, " ")
    has_weak = weak_target in haystack_stripped

    if has_strong:
        # Explicit target year — best case
        return (1.0, False)
    if has_weak and not has_adjacent:
        # Bare year present, no competing adjacent year
        return (0.7, False)
    if has_adjacent and not has_strong and not has_weak:
        # Only adjacent year found — definitely wrong year
        return (0.0, True)
    if has_adjacent and has_weak:
        # Both bare years present (e.g. "report 2024 covers 2024-25"):
        # treat as wrong-year but don't hard-drop (weak signal)
        return (0.2, False)
    # No year signal at all — neutral
    return (0.5, False)


def _filter_and_rerank(
    items: list[dict],
    company_name: str,
    year: int = 0,
) -> list[dict]:
    """
    Apply domain-aware, company-aware, and year-aware scoring; return re-ranked results.

    Each input dict is expected to have: url, title, content (snippet), score.

    Algorithm
    ---------
    1. Expand company name into alias set.
    2. For each result, resolve its domain tier (TRUSTED / NEGATIVE / UNKNOWN).
    3. Apply filtering rules in strict order:
         Rule 1 (TRUSTED)  -> KEEP always; year scoring used for ranking only.
         Rule 2 (NEGATIVE) -> DROP immediately.
         Rule 3 (UNKNOWN)  -> compute entity_score; DROP if score == 0.
                              Also DROP if is_wrong_year=True (explicit adjacent-year
                              signal found, no target-year signal found).
    4. Fallback: if all results dropped, restore all non-NEGATIVE results
                 (year hard-filter relaxed so we never return empty).
    5. Absolute fallback: if still empty, return original list unchanged.
    6. Re-rank kept results by combined score:
         combined = entity_weight * entity_score
                  + year_weight   * year_score
                  + orig_weight   * search_score

    Returns
    -------
    Filtered and re-ranked list with the same dict shape as input.
    """
    if not items:
        return items

    aliases = expand_company_aliases(company_name)

    # Tuples: (item, entity_score, year_score, tier, keep)
    evaluated: list[tuple[dict, float, float, str, bool]] = []

    for item in items:
        url     = item.get("url", "")
        title   = item.get("title", "") or ""
        snippet = item.get("content", "") or ""
        domain  = _extract_domain(url)

        tier = _resolve_domain_tier(domain, aliases)

        yr_score, is_wrong_year = (
            score_year_relevance(url, title, snippet, year) if year
            else (0.5, False)
        )

        if tier == _TIER_TRUSTED:
            # Rule 1: trusted portal -- always keep; year signal used for ranking.
            entity_score = 1.0
            keep         = True
            logger.debug(
                "search.domain_filter",
                url=url[:80],
                tier=tier,
                entity_score=entity_score,
                year_score=round(yr_score, 2),
                decision="KEEP (trusted portal)",
            )

        elif tier == _TIER_NEGATIVE:
            # Rule 2: rival company domain -- always drop.
            entity_score = 0.0
            keep         = False
            logger.debug(
                "search.domain_filter",
                url=url[:80],
                tier=tier,
                entity_score=entity_score,
                year_score=round(yr_score, 2),
                decision="DROP (rival domain)",
            )

        else:
            # Rule 3: unknown domain -- entity score + year signal decide.
            entity_score = score_entity_relevance(url, title, snippet, aliases)
            if entity_score == 0.0:
                keep     = False
                decision = "DROP (zero entity evidence)"
            elif is_wrong_year:
                keep     = False
                decision = "DROP (explicit wrong-year signal)"
            else:
                keep     = True
                decision = "KEEP"
            logger.debug(
                "search.domain_filter",
                url=url[:80],
                tier=tier,
                entity_score=round(entity_score, 3),
                year_score=round(yr_score, 2),
                is_wrong_year=is_wrong_year,
                decision=decision,
            )

        evaluated.append((item, entity_score, yr_score, tier, keep))

    # Apply decisions
    kept = [(item, es, ys, tier) for item, es, ys, tier, keep in evaluated if keep]

    logger.info(
        "search.domain_filter_summary",
        company=company_name,
        year=year,
        total=len(items),
        kept=len(kept),
        dropped=len(items) - len(kept),
        trusted=sum(1 for _, _, _, t, k in evaluated if t == _TIER_TRUSTED  and k),
        negative=sum(1 for _, _, _, t, _ in evaluated if t == _TIER_NEGATIVE),
        wrong_year_dropped=sum(
            1 for _, es, ys, t, k in evaluated
            if not k and t == _TIER_UNKNOWN and ys == 0.0
        ),
        unknown_kept=sum(1 for _, _, _, t, k in evaluated if t == _TIER_UNKNOWN and k),
        unknown_dropped=sum(1 for _, _, _, t, k in evaluated if t == _TIER_UNKNOWN and not k),
    )

    if not kept:
        # Soft fallback: restore non-NEGATIVE results, relax year hard-filter.
        logger.warning(
            "search.domain_filter_fallback",
            company=company_name,
            message=(
                "All results dropped by domain/year filter. "
                "Reverting to non-NEGATIVE results in original order."
            ),
        )
        kept = [
            (
                item,
                score_entity_relevance(
                    item.get("url", ""), item.get("title", ""),
                    item.get("content", ""), aliases,
                ),
                (score_year_relevance(
                    item.get("url", ""), item.get("title", ""),
                    item.get("content", ""), year,
                )[0] if year else 0.5),
                tier,
            )
            for item, _, _, tier, _ in evaluated
            if tier != _TIER_NEGATIVE
        ]

    if not kept:
        # Absolute fallback: all results were from rival domains -- return as-is
        logger.warning(
            "search.domain_filter_absolute_fallback",
            company=company_name,
            message=(
                "All results from rival domains. "
                "Returning original list to avoid empty result."
            ),
        )
        return items

    # Re-rank: combined = entity_weight * entity + year_weight * year + orig_weight * search
    kept.sort(
        key=lambda x: (
            _ENTITY_WEIGHT   * x[1] +
            _YEAR_WEIGHT     * x[2] +
            _ORIGINAL_WEIGHT * x[0].get("score", 0.0)
        ),
        reverse=True,
    )

    return [item for item, _, _, _ in kept]


# =============================================================================
# SERPAPI CLIENT
# =============================================================================

def _serpapi_score(position: int) -> float:
    """
    Synthesise a [0, 1] relevance score from a 1-based search position.

    SerpApi does not return a relevance float.  We use the inverse of
    position so that the first result scores highest and later results
    score progressively lower.

      position 1  -> 1.0000
      position 2  -> 0.5000
      position 5  -> 0.2000
      position 10 -> 0.1000

    This score is used identically to Tavily's score field throughout the
    rest of the pipeline (dedup keeps highest score per URL; re-ranking
    weights it at _ORIGINAL_WEIGHT = 0.55).
    """
    return round(1.0 / max(position, 1), 4)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def _call_serpapi(query: str, api_key: str, num_results: int) -> list[dict]:
    """
    Execute one Google search via SerpApi and return normalised result dicts.

    Each returned dict has the same internal shape used throughout the rest
    of the pipeline:
        url          str   -- the page URL  (from SerpApi's "link" field)
        title        str   -- the page title
        content      str   -- snippet, truncated to 300 chars
        score        float -- synthesised from position (1/position)
        query_source str   -- the query that produced this result (set by caller)

    SerpApi request parameters:
        engine  = "google"    Google web search
        q       = query       the search query string
        num     = num_results results per page (max 100; caller passes 10)
        api_key = api_key     authentication key
        output  = "json"      response format

    Error handling:
        HTTP 4xx/5xx  -> raises httpx.HTTPStatusError; tenacity retries on 5xx
                         (4xx are terminal after the retry budget).
        API-level error field (e.g. invalid key, quota exceeded)
                      -> raises RuntimeError with the error message.
        Missing "link" on a result -> that result is silently skipped.
        Empty organic_results -> returns [] (not an error; handled upstream).

    Args:
        query:       Search query string.
        api_key:     SerpApi API key (from settings.serpapi_api_key).
        num_results: Number of results to request (num= parameter).

    Returns:
        List of normalised result dicts.  Empty list on no organic results.

    Raises:
        RuntimeError           on SerpApi-level errors (bad key, quota, etc.)
        httpx.HTTPStatusError  on non-2xx responses after retries exhausted.
        httpx.TimeoutException on network timeout after retries exhausted.
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

    # Surface API-level errors (e.g. invalid key, quota exceeded).
    # SerpApi returns HTTP 200 with {"error": "..."} for these cases.
    if "error" in data:
        raise RuntimeError(f"SerpApi error: {data['error']}")

    organic = data.get("organic_results", [])
    results: list[dict] = []

    for item in organic:
        # SerpApi uses "link" for the URL field, unlike most search APIs.
        url = item.get("link", "").strip()
        if not url:
            continue    # skip results with no usable URL

        results.append({
            "url":     url,
            "title":   item.get("title", "") or "",
            "content": (item.get("snippet", "") or "")[:300],
            "score":   _serpapi_score(item.get("position", 99)),
            # query_source is injected by the caller (collect_and_classify)
        })

    return results


# =============================================================================
# QUERY BUILDER + URL CLASSIFIER
# =============================================================================

def _build_all_queries(company: str, year: int) -> list[tuple[str, str]]:
    """Expand every template for every type into (query_string, type_hint) pairs."""
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
    Checks types in PRIORITY_ORDER; returns the first match or None.
    """
    url_lower = url.lower()
    for report_type in PRIORITY_ORDER:
        rules = _URL_KEYWORD_RULES[report_type]
        for keyword_tuple in rules:
            if all(kw in url_lower for kw in keyword_tuple):
                return report_type
    return None


# =============================================================================
# PUBLIC API
# =============================================================================

def collect_and_classify(
    company_name: str,
    year: int,
    max_results_per_query: int = 5,
) -> dict[str, SearchResult]:
    """
    Run all queries via SerpApi, pool + deduplicate results, apply domain-aware
    company filtering and re-ranking, then classify into report-type buckets.

    Steps:
      1. Run every query template against SerpApi Google Search.
      2. Pool + globally deduplicate by URL (keep highest score per URL).
      3. Domain-aware filter + re-rank:
           TRUSTED domains  -> always kept (regulatory portals, Rule 1)
           NEGATIVE domains -> always dropped (rival company sites, Rule 2)
           UNKNOWN domains  -> kept only if entity_score > 0 (Rule 3)
      4. Classify each URL into one report type by URL keyword matching.
      5. Assign each URL to exactly one type (BRSR > ESG > Integrated).
      6. Return one SearchResult per type.

    Args:
        company_name:          Company name string, e.g. "Infosys".
        year:                  Fiscal year end integer, e.g. 2025.
        max_results_per_query: SerpApi num= parameter per query (max 100).
                               10 gives good recall without burning quota.

    Returns:
        {
            "BRSR":       SearchResult(discovered=[...], total_found=N, ...),
            "ESG":        SearchResult(discovered=[...], total_found=N, ...),
            "Integrated": SearchResult(discovered=[...], total_found=N, ...),
        }
        All three keys always present.  .discovered is empty when no
        classified URLs were found for a type.
    """
    settings = get_settings()

    if not settings.serpapi_api_key:
        logger.warning(
            "search_service.no_api_key",
            message="SERPAPI_API_KEY not set -- returning empty results.",
        )
        return _empty_results(company_name, year)

    all_queries = _build_all_queries(company_name, year)
    logger.info(
        "search_service.collect_start",
        company=company_name,
        year=year,
        total_queries=len(all_queries),
    )

    # ── Step 1: Run all queries ───────────────────────────────────────────────
    raw_pool: list[dict] = []

    for query, query_type_hint in all_queries:
        try:
            items = _call_serpapi(query, settings.serpapi_api_key, max_results_per_query)
        except Exception as exc:
            logger.error(
                "search_service.query_failed",
                query=query,
                query_type_hint=query_type_hint,
                error=str(exc),
            )
            continue

        for item in items:
            # Inject query_source here so _call_serpapi stays pure
            raw_pool.append({**item, "query_source": query})

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
            message="All queries returned no results. Check API key and quota.",
        )
        return _empty_results(company_name, year)

    # ── Step 2: Global URL deduplication (keep highest score per URL) ─────────
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

    # ── Step 3: Domain-aware filter + re-rank ────────────────────────────────
    unique_items = _filter_and_rerank(unique_items, company_name, year=year)

    # ── Steps 4 & 5: Classify + assign ───────────────────────────────────────
    classified: dict[str, list[DiscoveredReport]] = {t: [] for t in PRIORITY_ORDER}
    discarded_count = 0

    for item in unique_items:
        url      = item["url"]
        assigned = _classify_url(url)
        if assigned is None:
            logger.debug("search_service.url_unclassified", url=url[:80])
            discarded_count += 1
            continue
        classified[assigned].append(DiscoveredReport(
            url=url,
            title=item["title"],
            snippet=item["content"],
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

    # ── Step 6: Build SearchResult per type ──────────────────────────────────
    results: dict[str, SearchResult] = {}

    for report_type in PRIORITY_ORDER:
        urls_for_type = classified[report_type]
        # Combined score ordering applied by _filter_and_rerank;
        # secondary stable sort by raw search score within each bucket.
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
    """Return an empty SearchResult for every type (used on missing API key or quota)."""
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
    max_results_per_query: int = 3,
) -> dict[str, SearchResult]:
    """Alias for collect_and_classify(). Kept for backward compatibility."""
    return collect_and_classify(
        company_name=company_name,
        year=year,
        max_results_per_query=max_results_per_query,
    )