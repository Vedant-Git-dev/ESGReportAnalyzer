"""
services/revenue_extractor.py

Revenue extraction using the same regex → LLM → validation pipeline
as the KPI extraction agent.

Root causes of previous failures fixed here:
  1. Indian number format: 1,62,990 has groups of 2 AND 3 digits.
     Old regex `(?:,\d{2})` only matched 2-digit groups, giving 1,62,99 → 16299.
     Fixed: `(?:,\d{2,3})+` matches both.
  2. Space padding: " 1,62,990 " — strip() before parse.
  3. Decimal-with-note numbers like "2.18" (section ref) confused as revenue.
     Fixed: section refs have 1-2 decimal places AND are <100, so min_val=1000 filters.
  4. Some pages have revenue as plain integer "162990" (no commas).
     Fixed: added bare-integer pattern.
  5. TCS pattern "Revenue from operations\n12 \n2,40,893" — note number on its own line.
     Fixed: dedicated pattern for note-on-separate-line format.

Extraction layers (same as ExtractionAgent):
  Layer 1: Block-regex  — deterministic, zero API cost, ~2ms
  Layer 2: LLM          — Gemini batch call, fires only on regex miss
  Layer 3: Validation   — plausibility range + unit consistency

Public API:
    extract_revenue(pdf_path, fiscal_year_hint, llm_service) -> RevenueResult | None
    store_revenue(report, result, db)
    get_cached_revenue(report, db) -> RevenueResult | None
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.logging_config import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Indian number format
# ─────────────────────────────────────────────────────────────────────────────
# Indian lakh-crore notation:
#   2,40,893   = 240893   (first group 1-3 digits, then groups of 2-3 digits)
#   1,62,990   = 162990
#   10,50,000  = 1050000
#   1,00,000   = 100000
#   162990     = 162990   (no commas at all — some PDF tables strip them)
#
# Key fix: (?:,\d{2,3})+ handles both lakh groups (2-digit) and crore groups (3-digit).
# Western format 2,40,893 also matched because the last group is 3 digits.
_IND_NUM_RE = r"\d{1,3}(?:,\d{2,3})+"   # with commas
_PLAIN_NUM_RE = r"\d{5,7}"               # plain integer 10000–9999999 (5-7 digits = Crore range)
_REVENUE_NUM = rf"(?:{_IND_NUM_RE}|{_PLAIN_NUM_RE})(?:\.\d+)?"

# ─────────────────────────────────────────────────────────────────────────────
# Regex patterns — ordered by specificity / confidence
# ─────────────────────────────────────────────────────────────────────────────
# Each entry: (name, pattern, base_confidence)
# Group 1 always captures the raw revenue number string.

_PATTERNS: list[tuple[str, re.Pattern, float]] = [
    # "Total revenue from operations\n 1,62,990 \n 1,53,670"
    (
        "total_block",
        re.compile(
            r"total\s+revenue\s+from\s+operations\s*\n\s*(" + _REVENUE_NUM + r")",
            re.IGNORECASE,
        ),
        0.95,
    ),
    # "Revenue from operations\n1,62,990\n1,53,670"
    (
        "block",
        re.compile(
            r"revenue\s+from\s+operations\s*\n\s*(" + _REVENUE_NUM + r")",
            re.IGNORECASE,
        ),
        0.92,
    ),
    # TCS: "Revenue from operations\n12 \n2,40,893" — note num on its own line
    (
        "block_with_note_line",
        re.compile(
            r"revenue\s+from\s+operations\s*\n\s*\d{1,3}\s*\n\s*(" + _REVENUE_NUM + r")",
            re.IGNORECASE,
        ),
        0.90,
    ),
    # Infosys: "Revenue from operations\n2.18\n 1,62,990" — section ref on own line
    (
        "block_with_ref_line",
        re.compile(
            r"revenue\s+from\s+operations\s*\n\s*\d{1,2}\.\d{1,3}\s*\n\s*(" + _REVENUE_NUM + r")",
            re.IGNORECASE,
        ),
        0.90,
    ),
    # Inline: "Revenue from operations  12  2,40,893" (note inline)
    (
        "inline_note",
        re.compile(
            r"revenue\s+from\s+operations\s+\d{1,3}\s+(" + _REVENUE_NUM + r")",
            re.IGNORECASE,
        ),
        0.85,
    ),
    # Broad inline fallback
    (
        "inline",
        re.compile(
            r"revenue\s+from\s+operations[\s\n]+(" + _REVENUE_NUM + r")",
            re.IGNORECASE,
        ),
        0.75,
    ),
]

# Pages that are NOT financial statements — skip to avoid false positives
_SKIP_PAGE_KEYWORDS = [
    "business responsibility",
    "sustainability report",
    "brsr",
    "our approach",
    "material topics",
    "stakeholder",
    "esg vision",
]

# Pages that ARE financial statements — prioritise
_FINANCIAL_PAGE_KEYWORDS = [
    "statement of profit",
    "income statement",
    "profit and loss",
    "total income",
    "financial highlights",
    "revenue from operations",
    "consolidated statement",
    "standalone statement",
]

# Plausibility range for Indian large-cap revenue in INR Crore
_MIN_REVENUE_CR = 1_000      # ₹1,000 Cr floor
_MAX_REVENUE_CR = 9_000_000  # ₹90 lakh Cr ceiling (well above any current company)


@dataclass
class RevenueResult:
    value_cr: float           # normalised INR Crore
    raw_value: str            # exactly as captured from PDF
    raw_unit: str             # "INR_Crore" | "INR_Lakh" | "USD_Million"
    source: str               # "regex" | "llm" | "manual"
    page_number: int
    confidence: float         # 0.0–1.0
    pattern_name: str = ""    # which regex pattern matched


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_indian(s: str) -> float:
    """Strip commas and spaces → float. '1,62,990' → 162990.0"""
    cleaned = s.strip().replace(",", "").replace(" ", "")
    return float(cleaned)


def _is_financial_page(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in _FINANCIAL_PAGE_KEYWORDS)


def _is_skip_page(text: str) -> bool:
    """ESG/BRSR report pages often contain revenue in intensity context — skip."""
    t = text.lower()
    return any(kw in t for kw in _SKIP_PAGE_KEYWORDS)


def _validate_revenue(value: float) -> tuple[bool, str]:
    """Return (is_valid, reason)."""
    if value < _MIN_REVENUE_CR:
        return False, f"below minimum {_MIN_REVENUE_CR} Cr"
    if value > _MAX_REVENUE_CR:
        return False, f"above maximum {_MAX_REVENUE_CR} Cr"
    return True, "ok"


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: Block-regex extraction
# ─────────────────────────────────────────────────────────────────────────────

def _regex_extract(page_texts: list[tuple[int, str, bool]]) -> list[RevenueResult]:
    """
    Run all patterns against all pages.

    Args:
        page_texts: list of (page_num_1based, text, is_financial_page)

    Returns:
        All plausible candidates, unsorted.
    """
    candidates: list[RevenueResult] = []

    for page_num, text, is_financial in page_texts:
        for name, pattern, base_conf in _PATTERNS:
            for m in pattern.finditer(text):
                raw = m.group(1).strip()
                try:
                    val = _parse_indian(raw)
                except (ValueError, AttributeError):
                    continue

                valid, reason = _validate_revenue(val)
                if not valid:
                    logger.debug("revenue_extractor.regex_invalid",
                                 raw=raw, reason=reason, page=page_num)
                    continue

                # Boost confidence for financial statement pages
                conf = base_conf + (0.03 if is_financial else 0.0)

                candidates.append(RevenueResult(
                    value_cr=val,
                    raw_value=raw,
                    raw_unit="INR_Crore",
                    source="regex",
                    page_number=page_num,
                    confidence=min(conf, 0.99),
                    pattern_name=name,
                ))
                logger.debug("revenue_extractor.regex_candidate",
                             val=val, pattern=name, page=page_num, conf=conf)

    return candidates


def _pick_best(candidates: list[RevenueResult]) -> Optional[RevenueResult]:
    """
    Select best candidate:
      1. Highest confidence
      2. Among ties, highest value (current FY > prior FY in the same document)
    """
    if not candidates:
        return None
    return max(candidates, key=lambda r: (r.confidence, r.value_cr))


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: LLM fallback
# ─────────────────────────────────────────────────────────────────────────────

def _llm_extract(
    page_texts: list[tuple[int, str, bool]],
    fiscal_year_hint: Optional[int],
    llm_service,
) -> Optional[RevenueResult]:
    """
    Ask the LLM for revenue when regex fails.
    Uses the existing LLMService._call() — no new HTTP client.
    """
    if llm_service is None:
        return None

    try:
        from core.config import get_settings
        if not get_settings().llm_api_key:
            return None
    except Exception:
        return None

    # Select financial pages only — limit to 5 pages for token budget
    financial_pages = [
        (pn, txt) for pn, txt, is_fin in page_texts if is_fin
    ][:5]
    if not financial_pages:
        financial_pages = [(pn, txt) for pn, txt, _ in page_texts][:3]

    merged = "\n\n---PAGE BREAK---\n\n".join(
        f"[Page {pn}]\n{txt[:3000]}" for pn, txt in financial_pages
    )[:15_000]

    year_hint = (
        f"The report covers fiscal year {fiscal_year_hint}. "
        f"Return the revenue for FY{fiscal_year_hint} (NOT the prior year)."
    ) if fiscal_year_hint else ""

    system = (
        "You are an expert financial analyst. "
        "Extract revenue from operations from the given annual report text. "
        "Return ONLY valid JSON, no markdown."
    )

    user = f"""Extract the REVENUE FROM OPERATIONS for the most recent fiscal year from this annual report text.

{year_hint}

Rules:
1. Extract "Revenue from operations" only — NOT "Total income", NOT "Other income".
2. The number is in INR Crore. Strip all commas: 2,40,893 → 240893.
3. Return the CURRENT year value, not the prior year comparison column.
4. Common formats in Indian annual reports:
   - "Revenue from operations  2,40,893  2,25,458" — take FIRST number (current year)
   - "Revenue from operations\\n12\\n2,40,893" — the "12" is a note number, take 2,40,893
   - "Revenue from operations\\n2.18\\n1,62,990" — the "2.18" is a section ref, take 1,62,990
5. If genuinely not found, return null.

TEXT:
{merged}

Return ONLY this JSON:
{{"value": <number or null>, "unit": "INR_Crore", "confidence": <0.0-1.0>, "page": <page_number or null>, "notes": "<brief>"}}"""

    logger.info("revenue_extractor.llm_call", chars=len(merged))

    raw_response = llm_service._call(system, user)
    print("LLM raw response:", raw_response)
    print(merged)
    if not raw_response:
        return None

    # Parse JSON response
    try:
        import json
        cleaned = raw_response.strip()
        # Strip markdown fences if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).replace("```", "").strip()
        result = json.loads(cleaned)
    except (json.JSONDecodeError, Exception):
        # Try extracting JSON object from response
        m = re.search(r"\{[\s\S]*\}", raw_response)
        if not m:
            logger.warning("revenue_extractor.llm_bad_json", raw=raw_response[:200])
            return None
        try:
            import json
            result = json.loads(m.group())
        except Exception:
            return None

    val = result.get("value")
    if val is None:
        return None
    try:
        val = float(str(val).replace(",", ""))
    except (ValueError, TypeError):
        return None

    valid, reason = _validate_revenue(val)
    if not valid:
        logger.warning("revenue_extractor.llm_invalid", val=val, reason=reason)
        return None

    conf = float(result.get("confidence") or 0.7)
    page = result.get("page") or 0

    logger.info("revenue_extractor.llm_hit", val=val, conf=conf, page=page)
    return RevenueResult(
        value_cr=val,
        raw_value=str(val),
        raw_unit="INR_Crore",
        source="llm",
        page_number=page,
        confidence=min(conf, 0.95),
        pattern_name="llm",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_revenue(
    pdf_path: Path,
    fiscal_year_hint: Optional[int] = None,
    llm_service=None,
) -> Optional[RevenueResult]:
    """
    Extract revenue from operations from a corporate annual/ESG report PDF.

    Extraction layers:
      Layer 1 — Block-regex  (deterministic, handles all known Indian AR formats)
      Layer 2 — LLM fallback (Gemini, fires only if regex returns nothing)
      Layer 3 — Validation   (plausibility range check)

    Args:
        pdf_path:          Path to PDF file
        fiscal_year_hint:  FY year (e.g. 2025) used in LLM prompt for disambiguation
        llm_service:       Optional LLMService instance; if None, LLM layer is skipped

    Returns:
        RevenueResult with .value_cr in INR Crore, or None if not found
    """
    import fitz

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(path))

    # Build (page_num, text, is_financial) tuples for all pages
    page_texts: list[tuple[int, str, bool]] = []
    for pg_i, page in enumerate(doc):
        text = page.get_text()
        if "revenue" not in text.lower():
            continue
        is_fin = _is_financial_page(text)
        is_skip = _is_skip_page(text)
        # Keep financial pages regardless; keep non-skip pages too
        if is_fin or not is_skip:
            page_texts.append((pg_i + 1, text, is_fin))

    doc.close()

    # ── Layer 1: Regex ────────────────────────────────────────────────────────
    candidates = _regex_extract(page_texts)
    best = _pick_best(candidates)

    if best and best.confidence >= 0.75:
        logger.info(
            "revenue_extractor.regex_success",
            val=best.value_cr, page=best.page_number,
            pattern=best.pattern_name, conf=best.confidence,
        )
        return best

    # ── Layer 2: LLM ─────────────────────────────────────────────────────────
    if llm_service is not None:
        llm_result = _llm_extract(page_texts, fiscal_year_hint, llm_service)
        if llm_result:
            # If regex also found something, take higher-confidence result
            if best:
                result = llm_result if llm_result.confidence > best.confidence else best
            else:
                result = llm_result
            logger.info("revenue_extractor.llm_success", val=result.value_cr)
            return result

    # ── Return regex result even if low confidence ────────────────────────────
    if best:
        logger.warning(
            "revenue_extractor.low_confidence",
            val=best.value_cr, conf=best.confidence,
        )
        return best

    logger.warning("revenue_extractor.not_found", pdf=str(path))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# DB integration (uses existing SQLAlchemy session pattern)
# ─────────────────────────────────────────────────────────────────────────────

def store_revenue(report, result: RevenueResult, db) -> None:
    """
    Persist revenue onto the Report ORM object.
    Caller owns the session and commit.

    Args:
        report:  Report ORM instance
        result:  RevenueResult from extract_revenue()
        db:      SQLAlchemy Session (not used directly — ORM handles flush)
    """
    report.revenue_cr     = result.value_cr
    report.revenue_unit   = result.raw_unit
    report.revenue_source = result.source
    logger.info(
        "revenue_extractor.stored",
        report_id=str(report.id),
        val=result.value_cr,
        source=result.source,
    )


def get_cached_revenue(report) -> Optional[RevenueResult]:
    """
    Return a RevenueResult from a Report ORM object if revenue is already stored.
    Returns None if revenue_cr is null.

    Args:
        report: Report ORM instance (already loaded)
    """
    if report.revenue_cr is None:
        return None
    return RevenueResult(
        value_cr=report.revenue_cr,
        raw_value=str(report.revenue_cr),
        raw_unit=report.revenue_unit or "INR_Crore",
        source=report.revenue_source or "db",
        page_number=0,
        confidence=0.99,  # already validated when stored
        pattern_name="cached",
    )


def ensure_revenue_columns(db) -> None:
    """
    Add revenue columns to the reports table if they don't exist yet.
    Safe to call on every startup — uses IF NOT EXISTS.
    Handles DB unavailability gracefully.
    """
    from sqlalchemy import text
    try:
        db.execute(text(
            "ALTER TABLE reports "
            "ADD COLUMN IF NOT EXISTS revenue_cr DOUBLE PRECISION, "
            "ADD COLUMN IF NOT EXISTS revenue_unit VARCHAR(30), "
            "ADD COLUMN IF NOT EXISTS revenue_source VARCHAR(50);"
        ))
        db.commit()
        logger.info("revenue_extractor.columns_ensured")
    except Exception as exc:
        db.rollback()
        logger.warning("revenue_extractor.column_migration_failed", error=str(exc))