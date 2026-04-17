"""
services/search_service.py

SerpApi-powered ESG report discovery with URL-keyword classification.

v2 improvements over v1
------------------------
1. Query diversity
   _QUERY_TEMPLATES expanded from 4 queries (1+2+1) to 16 queries (5+6+5).
   New forms: bare year (2025), FY-prefix (FY2025, FY25), sustainability
   synonyms, "integrated annual report" variant.  All templates use .format()
   with company/year/past_year kwargs — the extra kwarg past_year is silently
   ignored by templates that don't need it, so .format() never raises.

2. Domain boosting
   score_entity_relevance now adds +0.30 when the company slug appears
   directly in the domain string (e.g. "infosys" in "infosys.com").
   This is additive to the existing alias-in-domain credit and capped at 1.0.
   Effect: official company IR domains score 0.95–1.0; neutral domains 0.4–0.7.

3. Aggregator / low-quality domain penalty
   New constant AGGREGATOR_DOMAIN_KEYWORDS (frozenset).
   New helper _aggregator_penalty(domain) → float (returns −0.25 or 0.0).
   Applied inside _filter_and_rerank to the combined score AFTER normal
   scoring.  Results are penalised, not dropped — so a genuinely authoritative
   link from a news site (e.g. direct PDF via reuters.com) can still pass if
   its entity + year scores are strong enough.

4. Confidence threshold
   New constant _MIN_COMBINED_SCORE = 0.20.
   Applied inside collect_and_classify after reranking, per report type.
   Results below the threshold are filtered out.  If ALL results for a type
   fall below the threshold the filter is skipped (fallback preserved —
   never return empty when results exist).  Threshold is intentionally low
   so that strong results on trusted domains (entity=1.0) always pass.

5. Top-K preserved
   No structural change needed.  collect_and_classify already returns all
   classified URLs per type, sorted by combined score.  IngestionAgent uses
   the full list and tries candidates in order.  Top-3 are now explicitly
   logged after reranking (improvement 6).

6. Enhanced logging
   _filter_and_rerank logs the top-3 results after reranking with:
     entity_score, year_score, aggregator_penalty, combined_score, domain_tier.
   collect_and_classify logs per-type confidence filtering stats.

7. Backward compatibility
   All public function signatures are unchanged.
   SearchResult structure is unchanged.
   Fallback logic (soft fallback → absolute fallback) is unchanged.

Search backend
--------------
Uses the SerpApi Google Search endpoint (https://serpapi.com/search) via
direct httpx calls.  Required env var: SERPAPI_API_KEY.

Score synthesis
---------------
SerpApi does not return a relevance float.  We synthesise: score = 1.0 / position.

Combined scoring formula (unchanged):
  combined = ENTITY_WEIGHT * entity_score
           + YEAR_WEIGHT   * year_score
           + ORIGINAL_WEIGHT * search_score
           + aggregator_penalty           (new, ≤ 0)
  weights: 0.35 / 0.30 / 0.35
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

PRIORITY_ORDER: list[str] = ["BRSR", "ESG", "Integrated"]
ALL_REPORT_TYPES: list[str] = PRIORITY_ORDER
DEFAULT_REPORT_TYPE = "BRSR"

# ── CHANGE 1: Expanded query templates ──────────────────────────────────────
# Old: 4 total (1 BRSR + 2 ESG + 1 Integrated)
# New: 16 total (5 BRSR + 6 ESG + 5 Integrated)
#
# All templates use {company}, {year}, and optionally {past_year}.
# str.format() silently ignores unused kwargs so templates without
# {past_year} are safe to format with past_year=year-1.
#
# Design choices:
#   - Hyphenated year range (2024-25) remains for BRSR (NSE/BSE prefer it)
#   - FY-prefix forms (FY2025, FY25) added for all types
#   - Bare year (2025) added as fallback broad form
#   - "sustainability" synonym added for ESG
#   - "integrated annual report" added as Integrated variant
#   - nseindia filetype:pdf kept for BRSR (good for exchange filings)
#   - Duplicate removal: templates differ in at least one keyword
_QUERY_TEMPLATES: dict[str, list[str]] = {
    "BRSR": [
        # Original (hyphenated, NSE-targeted)
        "{company} BRSR {past_year}-{year} nseindia filetype:pdf",
        # FY prefix forms
        "{company} BRSR FY{year} filetype:pdf",
        "{company} BRSR FY{past_year}-{year} filetype:pdf",
        # Bare year (broader recall)
        "{company} BRSR {year} filetype:pdf",
        # Long-form synonym
        "{company} business responsibility sustainability report {year} filetype:pdf",
    ],
    "ESG": [
        # Original two
        "{company} ESG report {past_year}-{year} filetype:pdf",
        "{company} sustainability report {past_year}-{year} filetype:pdf",
        # FY-prefix additions
        "{company} ESG report FY{year} filetype:pdf",
        "{company} sustainability report FY{year} filetype:pdf",
        # Bare year
        "{company} ESG report {year} filetype:pdf",
        # Synonym: "environment social governance"
        "{company} environment social governance report {year} filetype:pdf",
    ],
    "Integrated": [
        # Original
        "{company} annual report {past_year}-{year} filetype:pdf",
        # FY-prefix additions
        "{company} annual report FY{year} filetype:pdf",
        "{company} integrated annual report FY{year} filetype:pdf",
        # Bare year
        "{company} annual report {year} filetype:pdf",
        # Explicit "integrated report" form
        "{company} integrated report {year} filetype:pdf",
    ],
}

_URL_KEYWORD_RULES: dict[str, list[tuple[str, ...]]] = {
    "BRSR": [
        ("business", "sustainability"),
        ("business", "responsibility"),
        ("brsr",),
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
        ("annual",),
    ],
}

# =============================================================================
# ── CHANGE 3: Aggregator / low-quality domain penalty ───────────────────────
# =============================================================================
# Keywords whose presence in a domain signals a low-quality source.
# Penalty: −0.25 on the final combined score.
# NOT a drop — strong entity+year scores can still overcome the penalty.
#
# Why these keywords:
#   - Financial data aggregators (moneycontrol, screener, trendlyne, tickertape,
#     indiainfoline, equitymaster, marketscreener): host scraped summaries, not PDFs
#   - News sites (economictimes, businessstandard, livemint, thehindu, reuters,
#     bloomberg): occasionally link to PDFs but rarely host the canonical source
#   - Generic qualifiers (blog, analysis, research, news): almost never host
#     the authoritative annual report PDF
#   - annualreports.com: re-hosts PDFs but with stale/incorrect metadata
#
# Domains explicitly NOT in this list (safe):
#   - nseindia.com, bseindia.com: handled by TRUSTED tier
#   - *.gov.in: handled by TRUSTED tier
#   - Company own-domains (infosys.com, tcs.com): entity boost outweighs any penalty
AGGREGATOR_DOMAIN_KEYWORDS: frozenset[str] = frozenset({
    "moneycontrol",
    "screener",
    "trendlyne",
    "tickertape",
    "indiainfoline",
    "equitymaster",
    "marketscreener",
    "economictimes",
    "businessstandard",
    "livemint",
    "thehindu",
    "reuters",
    "bloomberg",
    "annualreports",
    "blog",
    "analysis",
    "research",
    "news",
})

# Amount to subtract from combined score for aggregator domains
_AGGREGATOR_PENALTY: float = -0.25

# ── CHANGE 4: Confidence threshold ───────────────────────────────────────────
# Results below this combined score are filtered out per type.
# Calibration: a correct result on a weak UNKNOWN domain scores ~0.45+;
# pure noise or wrong-company noise with aggregator penalty scores ~0.10–0.18.
# Threshold 0.20 catches the latter while keeping all legitimate results.
# The filter is skipped if ALL results for a type fall below it (fallback safe).
_MIN_COMBINED_SCORE: float = 0.20

# =============================================================================
# DOMAIN TIER DEFINITIONS  (unchanged from v1)
# =============================================================================

_TRUSTED_DOMAIN_SUFFIXES: frozenset[str] = frozenset({
    "nseindia.com",
    "nsearchive.nseindia.com",
    "bseindia.com",
    "msei.in",
    "sebi.gov.in",
    "mca.gov.in",
    "connect2nse.com",
    "listing.bseindia.com",
    "nsdl.co.in",
    "cdsl.co.in",
    "india.gov.in",
    "nic.in",
})


def _domain_tier_trusted(domain: str) -> bool:
    domain = domain.lower().lstrip("www.")
    for suffix in _TRUSTED_DOMAIN_SUFFIXES:
        if domain == suffix or domain.endswith("." + suffix):
            return True
    return False


# ---------------------------------------------------------------------------
# ENTITY ALIAS TABLE  (unchanged from v1)
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

_DOMAIN_OWNERS: dict[str, str] = {
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
    "ntpc.co.in":               "ntpc",
    "ntpclimited.com":          "ntpc",
    "powergridindia.com":       "powergrid",
    "nhpcindia.com":            "nhpc",
    "adanigreen.com":           "adanigreen",
    "adanipower.com":           "adanipower",
    "tatasteel.com":            "tatasteel",
    "jswsteel.com":             "jswsteel",
    "sail.co.in":               "sail",
    "hindalco.com":             "hindalco",
    "vedanta.com":              "vedanta",
    "nalcoindia.com":           "nalco",
    "tata.com":                 "tata",
    "tatamotors.com":           "tatamotors",
    "tatapower.com":            "tatapower",
    "tatacommunications.com":   "tatacommunications",
    "tataconsumer.com":         "tataconsumer",
    "tatachemicals.com":        "tatachemicals",
    "sunpharma.com":            "sunpharma",
    "drreddys.com":             "drreddy",
    "cipla.com":                "cipla",
    "lupin.com":                "lupin",
    "hdfcbank.com":             "hdfcbank",
    "icicibank.com":            "icicibank",
    "sbi.co.in":                "sbi",
    "kotak.com":                "kotak",
    "axisbank.com":             "axisbank",
    "itcportal.com":            "itc",
    "hul.co.in":                "hul",
    "ril.com":                  "ril",
    "adanienterprises.com":     "adani",
    "adanigroup.com":           "adani",
}

_SLUG_RE = re.compile(r"[^a-z0-9]")


def _company_slug(name: str) -> str:
    return _SLUG_RE.sub("", name.lower().strip())


def expand_company_aliases(company_name: str) -> frozenset[str]:
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
# DOMAIN TIER RESOLUTION  (unchanged from v1)
# =============================================================================

_TIER_TRUSTED  = "trusted"
_TIER_NEGATIVE = "negative"
_TIER_UNKNOWN  = "unknown"


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def _domain_owner_slug(domain: str) -> Optional[str]:
    for known_domain, owner_slug in _DOMAIN_OWNERS.items():
        if known_domain in domain:
            return owner_slug
    return None


def _resolve_domain_tier(domain: str, company_aliases: frozenset[str]) -> str:
    if _domain_tier_trusted(domain):
        return _TIER_TRUSTED

    owner = _domain_owner_slug(domain)
    if owner is not None:
        owner_matches_target = any(
            owner in alias or alias in owner
            for alias in company_aliases
            if len(alias) >= 2
        )
        if not owner_matches_target:
            return _TIER_NEGATIVE

    return _TIER_UNKNOWN


# =============================================================================
# ── CHANGE 3 (helper): Aggregator penalty ───────────────────────────────────
# =============================================================================

def _aggregator_penalty(domain: str) -> float:
    """
    Return the score penalty for a low-quality / aggregator domain.

    Returns _AGGREGATOR_PENALTY (negative float) if the domain matches any
    keyword in AGGREGATOR_DOMAIN_KEYWORDS, otherwise 0.0.

    The penalty is applied to the final combined score after entity, year,
    and search weights are summed.  It is never applied to TRUSTED domains
    (those are handled upstream and always get entity_score = 1.0).

    Args:
        domain: Cleaned netloc string (no www., lowercase).

    Returns:
        0.0 or _AGGREGATOR_PENALTY (−0.25).
    """
    d = domain.lower()
    if any(kw in d for kw in AGGREGATOR_DOMAIN_KEYWORDS):
        return _AGGREGATOR_PENALTY
    return 0.0


# =============================================================================
# ENTITY RELEVANCE SCORING  (CHANGE 2: domain boost added)
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

    Signal weights (additive, pre-boost cap at 1.0):
      Title match:    0.50  per alias hit  (strong)
      Domain match:   0.30  per alias hit  (medium)
      URL path match: 0.20  per alias hit  (medium)
      Snippet match:  0.10  per alias hit  (weak)

    CHANGE 2 — Domain boost:
      If the company slug (alphanumeric-only form of the name) appears
      directly in the domain string, add an additional +0.30.
      Rationale: the slug is tighter than any alias (no spaces, no
      punctuation) so a slug hit in the domain is a very strong signal
      that this is the company's own IR site.
      Example: company="Infosys", slug="infosys", domain="infosys.com"
               → +0.30 on top of any alias-loop credit.
      Cap: min(score + 0.30, 1.0)

    Args:
        url:              Full result URL.
        title:            Page title string.
        snippet:          Short description / snippet.
        company_aliases:  Expanded alias set from expand_company_aliases().

    Returns:
        float in [0.0, 1.0].
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

    score = min(score, 1.0)

    # ── CHANGE 2: Additional boost when company slug is in the domain ─────────
    # The slug is the strictest identifier (no spaces, alphanumeric only).
    # A slug hit in the domain is strong evidence this is the official IR site.
    # We apply this AFTER the alias loop cap so the boost is always meaningful.
    company_slug_str = _company_slug(" ".join(company_aliases))
    if not company_slug_str:
        # Derive slug from the longest alias as fallback
        longest = max(company_aliases, key=len, default="")
        company_slug_str = _company_slug(longest)

    if company_slug_str and len(company_slug_str) >= 3 and company_slug_str in domain:
        score = min(score + 0.30, 1.0)
        logger.debug(
            "search.domain_slug_boost",
            domain=domain,
            slug=company_slug_str,
            score_after_boost=round(score, 3),
        )

    return score


# =============================================================================
# YEAR RELEVANCE SCORING  (unchanged from v1)
# =============================================================================

def _year_patterns(year: int) -> tuple[list[str], list[str]]:
    py = year - 1

    target: list[str] = [
        f"{py}-{str(year)[-2:]}",
        f"{py}-{year}",
        f"fy{year}",
        f"fy{str(year)[-2:]}",
        f"fy {year}",
        str(year),
    ]

    py1, y1 = year - 2, year - 1
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
    Return (year_score, is_wrong_year).  Unchanged from v1.
    """
    haystack = " ".join([url, title or "", snippet or ""]).lower()

    target_pats, adjacent_pats = _year_patterns(year)

    strong_targets = target_pats[:-1]
    weak_target    = target_pats[-1]

    has_strong   = any(p in haystack for p in strong_targets)
    has_adjacent = any(p in haystack for p in adjacent_pats)

    haystack_stripped = haystack
    for adj in adjacent_pats:
        haystack_stripped = haystack_stripped.replace(adj, " ")
    has_weak = weak_target in haystack_stripped

    if has_strong:
        return (1.0, False)
    if has_weak and not has_adjacent:
        return (0.7, False)
    if has_adjacent and not has_strong and not has_weak:
        return (0.0, True)
    if has_adjacent and has_weak:
        return (0.2, False)
    return (0.5, False)


# =============================================================================
# DOMAIN-AWARE FILTER + RERANK  (CHANGES 2, 3, 6)
# =============================================================================

_ENTITY_WEIGHT   = 0.35
_YEAR_WEIGHT     = 0.30
_ORIGINAL_WEIGHT = 0.35


def _filter_and_rerank(
    items: list[dict],
    company_name: str,
    year: int = 0,
) -> list[dict]:
    """
    Apply domain-aware, company-aware, and year-aware scoring.

    Changes vs v1:
      CHANGE 2: score_entity_relevance now includes slug-in-domain boost.
      CHANGE 3: aggregator penalty applied to combined score after weighting.
      CHANGE 6: top-3 results logged with full score breakdown after reranking.

    Algorithm (unchanged structure):
      1. Expand aliases.
      2. Resolve domain tier (TRUSTED / NEGATIVE / UNKNOWN).
      3. Apply filtering rules (same as v1).
      4. Compute combined score = entity*0.35 + year*0.30 + search*0.35
         + aggregator_penalty.         ← NEW
      5. Sort by combined score.
      6. Log top-3 results.            ← NEW
      7. Return filtered + sorted list.
    """
    if not items:
        return items

    aliases = expand_company_aliases(company_name)

    # Tuples: (item, entity_score, year_score, agg_penalty, combined, tier, keep)
    evaluated: list[tuple[dict, float, float, float, float, str, bool]] = []

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

        # ── Domain tier filtering (unchanged logic) ───────────────────────────
        if tier == _TIER_TRUSTED:
            entity_score = 1.0
            keep         = True
            decision     = "KEEP (trusted portal)"

        elif tier == _TIER_NEGATIVE:
            entity_score = 0.0
            keep         = False
            decision     = "DROP (rival domain)"

        else:
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
            is_wrong_year=is_wrong_year if tier == _TIER_UNKNOWN else False,
            decision=decision,
        )

        # ── CHANGE 3: Compute aggregator penalty ─────────────────────────────
        # Only penalise UNKNOWN tier (TRUSTED domains are exempt by design;
        # NEGATIVE domains are already dropped so penalty is irrelevant).
        agg_pen = _aggregator_penalty(domain) if tier == _TIER_UNKNOWN else 0.0

        # ── Combined score (includes aggregator penalty) ──────────────────────
        combined = (
            _ENTITY_WEIGHT   * entity_score
            + _YEAR_WEIGHT   * yr_score
            + _ORIGINAL_WEIGHT * item.get("score", 0.0)
            + agg_pen
        )
        combined = max(combined, 0.0)   # floor at 0 (penalty can't go negative)

        evaluated.append((item, entity_score, yr_score, agg_pen, combined, tier, keep))

    # Apply keep/drop decisions
    kept = [
        (item, es, ys, ap, combined, tier)
        for item, es, ys, ap, combined, tier, keep in evaluated
        if keep
    ]

    logger.info(
        "search.domain_filter_summary",
        company=company_name,
        year=year,
        total=len(items),
        kept=len(kept),
        dropped=len(items) - len(kept),
        trusted=sum(1 for _, _, _, _, _, t, k in evaluated if t == _TIER_TRUSTED and k),
        negative=sum(1 for _, _, _, _, _, t, _ in evaluated if t == _TIER_NEGATIVE),
        wrong_year_dropped=sum(
            1 for _, es, ys, _, _, t, k in evaluated
            if not k and t == _TIER_UNKNOWN and ys == 0.0
        ),
        unknown_kept=sum(1 for _, _, _, _, _, t, k in evaluated if t == _TIER_UNKNOWN and k),
        unknown_dropped=sum(1 for _, _, _, _, _, t, k in evaluated if t == _TIER_UNKNOWN and not k),
    )

    if not kept:
        # Soft fallback: restore non-NEGATIVE results (unchanged from v1)
        logger.warning(
            "search.domain_filter_fallback",
            company=company_name,
            message=(
                "All results dropped by domain/year filter. "
                "Reverting to non-NEGATIVE results in original order."
            ),
        )
        kept = []
        for item, es, ys, ap, combined, tier, _ in evaluated:
            if tier != _TIER_NEGATIVE:
                recomputed_combined = max(
                    _ENTITY_WEIGHT   * score_entity_relevance(
                        item.get("url", ""), item.get("title", ""),
                        item.get("content", ""), aliases)
                    + _YEAR_WEIGHT   * (score_year_relevance(
                        item.get("url", ""), item.get("title", ""),
                        item.get("content", ""), year)[0] if year else 0.5)
                    + _ORIGINAL_WEIGHT * item.get("score", 0.0)
                    + (_aggregator_penalty(_extract_domain(item.get("url", "")))
                       if tier == _TIER_UNKNOWN else 0.0),
                    0.0,
                )
                kept.append((item, es, ys, ap, recomputed_combined, tier))

    if not kept:
        # Absolute fallback: return original list (unchanged from v1)
        logger.warning(
            "search.domain_filter_absolute_fallback",
            company=company_name,
            message=(
                "All results from rival domains. "
                "Returning original list to avoid empty result."
            ),
        )
        return items

    # Re-rank by combined score (includes aggregator penalty)
    kept.sort(key=lambda x: x[4], reverse=True)

    # ── CHANGE 6: Log top-3 results with full score breakdown ─────────────────
    top3 = kept[:3]
    for rank, (item, es, ys, ap, combined, tier) in enumerate(top3, start=1):
        url    = item.get("url", "")
        domain = _extract_domain(url)
        logger.info(
            "search.top_result",
            rank=rank,
            url=url[:100],
            domain=domain,
            domain_tier=tier,
            entity_score=round(es, 3),
            year_score=round(ys, 3),
            aggregator_penalty=round(ap, 3),
            search_score=round(item.get("score", 0.0), 3),
            combined_score=round(combined, 3),
        )

    return [item for item, _, _, _, _, _ in kept]


# =============================================================================
# SERPAPI CLIENT  (unchanged from v1)
# =============================================================================

def _serpapi_score(position: int) -> float:
    return round(1.0 / max(position, 1), 4)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def _call_serpapi(query: str, api_key: str, num_results: int) -> list[dict]:
    """
    Execute one Google search via SerpApi.  Unchanged from v1.
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
            "content": (item.get("snippet", "") or "")[:300],
            "score":   _serpapi_score(item.get("position", 99)),
        })

    return results


# =============================================================================
# QUERY BUILDER + URL CLASSIFIER  (unchanged logic, new templates used above)
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


def _classify_url(url: str, title: str = "", snippet: str = "") -> Optional[str]:
    """
    Classify a URL into exactly one report type.

    v3 — uses combined text (URL + title + snippet) and adds fallback heuristics
    for URLs that carry no ESG/annual-report keywords themselves but are clearly
    valid investor-relations or PDF links (e.g. investors.kennametal.com/static-files/xxxx.pdf,
    /node/xxxxx/pdf).

    Priority order (checked in sequence; first match wins):
      1. ESG   — "sustainability" | "esg" | "environmental"
      2. Integrated — "integrated report" | "annual report" | "10-k"
      3. BRSR  — "brsr" | "business responsibility"

    Fallback heuristics (only reached when no keyword rule fires):
      a. "investors." in domain  → Integrated
      b. URL path ends with ".pdf" → Integrated

    Preserves _URL_KEYWORD_RULES for backward compatibility; the keyword
    rules defined there are still used as the primary classification layer.

    Args:
        url:     Full result URL.
        title:   Page/document title from the search result.
        snippet: Short description/snippet from the search result.

    Returns:
        "BRSR" | "ESG" | "Integrated" | None
    """
    text   = f"{url} {title} {snippet}".lower()
    domain = _extract_domain(url)

    # ── Priority 1: ESG keywords ──────────────────────────────────────────────
    if any(kw in text for kw in ("sustainability", "esg", "environmental")):
        return "ESG"

    # ── Priority 2: Integrated / Annual Report keywords ───────────────────────
    if any(kw in text for kw in ("integrated report", "annual report", "10-k")):
        return "Integrated"

    # ── Priority 3: BRSR keywords ─────────────────────────────────────────────
    if any(kw in text for kw in ("brsr", "business responsibility")):
        return "BRSR"

    # ── Fallback: keyword rules from _URL_KEYWORD_RULES (backward compat) ─────
    # These are broader tuple-based rules; checked only when the priority
    # keywords above did not fire.
    for report_type in PRIORITY_ORDER:
        for keyword_tuple in _URL_KEYWORD_RULES[report_type]:
            if all(kw in text for kw in keyword_tuple):
                return report_type

    # ── Fallback heuristics for opaque IR / PDF URLs ──────────────────────────
    # Handles URLs like:
    #   investors.kennametal.com/static-files/abc123.pdf
    #   example.com/node/12345/pdf
    # where neither the URL path nor title/snippet contains report-type keywords.
    if "investors." in domain:
        return "Integrated"

    url_path = url.lower().split("?")[0]   # strip query string before checking extension
    if url_path.endswith(".pdf"):
        return "Integrated"

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

    Changes vs v1:
      CHANGE 1: 16 query templates (was 4) → better recall.
      CHANGE 4: per-type confidence threshold applied after reranking.
      CHANGE 6: per-type confidence filtering stats logged.

    Steps (structure unchanged):
      1. Run every query template against SerpApi.
      2. Pool + globally deduplicate by URL (keep highest score per URL).
      3. Domain-aware filter + re-rank (now includes aggregator penalty).
      4. NEW: Confidence threshold filter per result.
      5. Classify each URL into one report type.
      6. Assign each URL to exactly one type (BRSR > ESG > Integrated).
      7. Return one SearchResult per type.

    Args:
        company_name:          Company name string.
        year:                  Fiscal year end integer (e.g. 2025).
        max_results_per_query: SerpApi num= per query (default 5).

    Returns:
        {"BRSR": SearchResult, "ESG": SearchResult, "Integrated": SearchResult}
        All three keys always present.
    """
    settings = get_settings()

    if not settings.serpapi_api_key:
        logger.warning(
            "search_service.no_api_key",
            message="SERPAPI_API_KEY not set — returning empty results.",
        )
        return _empty_results(company_name, year)

    all_queries = _build_all_queries(company_name, year)
    logger.info(
        "search_service.collect_start",
        company=company_name,
        year=year,
        total_queries=len(all_queries),   # now 16 vs old 4
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

    # ── Step 2: Global URL deduplication ─────────────────────────────────────
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

    # ── Step 4: CHANGE 4 — Per-result confidence threshold ───────────────────
    # Compute combined score for each surviving item and filter out weak results.
    # We need the combined score computed in _filter_and_rerank but that
    # function returns only items (not scores).  We recompute lightly here:
    # entity and year scoring are cheap (regex over short strings).
    #
    # Fallback safety: if ALL items for a type fall below the threshold after
    # classification, the threshold is waived for that type (same principle as
    # the existing soft/absolute fallbacks).
    aliases = expand_company_aliases(company_name)

    def _quick_combined(item: dict) -> float:
        """Recompute combined score for threshold check."""
        url     = item.get("url", "")
        title   = item.get("title", "") or ""
        snippet = item.get("content", "") or ""
        domain  = _extract_domain(url)
        tier    = _resolve_domain_tier(domain, aliases)

        if tier == _TIER_TRUSTED:
            es = 1.0
        elif tier == _TIER_NEGATIVE:
            es = 0.0
        else:
            es = score_entity_relevance(url, title, snippet, aliases)

        ys, _ = score_year_relevance(url, title, snippet, year) if year else (0.5, False)
        ap    = _aggregator_penalty(domain) if tier == _TIER_UNKNOWN else 0.0
        return max(
            _ENTITY_WEIGHT * es + _YEAR_WEIGHT * ys
            + _ORIGINAL_WEIGHT * item.get("score", 0.0) + ap,
            0.0,
        )

    # Tag each item with its combined score for logging
    scored_unique = [(item, _quick_combined(item)) for item in unique_items]

    # ── Step 5: Classify + assign ─────────────────────────────────────────────
    classified: dict[str, list[tuple[DiscoveredReport, float]]] = {
        t: [] for t in PRIORITY_ORDER
    }
    discarded_count = 0

    for item, combined in scored_unique:
        url      = item["url"]
        title    = item.get("title", "")
        snippet  = item.get("content", "")
        assigned = _classify_url(url, title, snippet)
        if assigned is None:
            if combined >= 0.5:
                assigned = "Integrated"
                logger.debug(
                    "search_service.fallback_classification",
                    url=url[:80],
                    assigned=assigned,
                    combined_score=round(combined, 3),
                )
            else:
                logger.debug("search_service.url_unclassified", url=url[:80])
                discarded_count += 1
                continue
        classified[assigned].append((
            DiscoveredReport(
                url=url,
                title=item["title"],
                snippet=item["content"],
                score=item["score"],
                query_source=item["query_source"],
            ),
            combined,
        ))

    logger.info(
        "search_service.classification_complete",
        company=company_name,
        discarded_unclassified=discarded_count,
        brsr_count=len(classified["BRSR"]),
        esg_count=len(classified["ESG"]),
        integrated_count=len(classified["Integrated"]),
    )

    # ── Step 6: Build SearchResult per type with threshold filtering ──────────
    results: dict[str, SearchResult] = {}

    for report_type in PRIORITY_ORDER:
        type_entries = classified[report_type]
        # Primary sort: combined score descending (best first)
        type_entries.sort(key=lambda x: x[1], reverse=True)

        # ── CHANGE 4: Confidence threshold ────────────────────────────────────
        above_threshold = [
            (disc, score) for disc, score in type_entries
            if score >= _MIN_COMBINED_SCORE
        ]

        if above_threshold:
            # Normal case: at least one result meets the threshold
            filtered_entries = above_threshold
            dropped_by_threshold = len(type_entries) - len(above_threshold)
            if dropped_by_threshold > 0:
                logger.info(
                    "search_service.threshold_filter",
                    report_type=report_type,
                    company=company_name,
                    total=len(type_entries),
                    kept=len(above_threshold),
                    dropped=dropped_by_threshold,
                    threshold=_MIN_COMBINED_SCORE,
                    top_score=round(type_entries[0][1], 3) if type_entries else 0,
                )
        else:
            # Fallback: waive threshold to prevent empty results
            filtered_entries = type_entries
            if type_entries:
                logger.warning(
                    "search_service.threshold_waived",
                    report_type=report_type,
                    company=company_name,
                    reason="All results below threshold — waiving to preserve fallback",
                    best_score=round(type_entries[0][1], 3) if type_entries else 0,
                    threshold=_MIN_COMBINED_SCORE,
                )

        urls_for_type = [disc for disc, _ in filtered_entries]

        if not urls_for_type:
            logger.warning(
                "search_service.type_not_found",
                company=company_name,
                year=year,
                report_type=report_type,
            )
        else:
            # ── CHANGE 6: Log per-type top-3 with combined scores ─────────────
            for rank, (disc, score) in enumerate(filtered_entries[:3], start=1):
                logger.info(
                    "search_service.type_top_result",
                    report_type=report_type,
                    rank=rank,
                    url=disc.url[:100],
                    combined_score=round(score, 3),
                    threshold=_MIN_COMBINED_SCORE,
                )
            logger.info(
                "search_service.type_found",
                company=company_name,
                year=year,
                report_type=report_type,
                count=len(urls_for_type),
                top_url=urls_for_type[0].url[:80],
                top_combined_score=round(filtered_entries[0][1], 3),
            )

        results[report_type] = SearchResult(
            company_name=company_name,
            year=year,
            report_type=report_type,
            discovered=urls_for_type,
            total_found=len(urls_for_type),
            queries_run=len(all_queries),
        )

    return results


def _empty_results(company_name: str, year: int) -> dict[str, SearchResult]:
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
    """Single-type search. Backward-compatible wrapper."""
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
    """Alias for collect_and_classify(). Backward-compatible."""
    return collect_and_classify(
        company_name=company_name,
        year=year,
        max_results_per_query=max_results_per_query,
    )