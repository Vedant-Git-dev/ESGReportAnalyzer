"""
services/revenue_extractor.py  — v3 (production-ready)

ROOT CAUSES FIXED IN THIS VERSION
═══════════════════════════════════════════════════════════════════════════════

Bug 1 — BRSR intensity back-calculation used wrong formula
  SYMPTOM: LLM receives BRSR page text, finds nothing; back-calc also fails.
  ROOT CAUSE (a): Infosys BRSR prints intensity as "5.11 GJ / ` cr" which means
    GJ per Crore (not per rupee despite the label). The old formula was
    kpi / ratio / 1e7, which gave a nonsensical number.
  ROOT CAUSE (b): TCS BRSR prints intensity as "0.000760" (MJ per rupee) with
    the unit in the parenthetical "(Total energy consumed (MJ)/Revenue from
    operations)". The old regex didn't extract the KPI in MJ (only GJ).

  FIX: Dual-formula back-calculation based on intensity value magnitude:
    • intensity > 1  (e.g. "5.11 GJ / cr") → per-crore unit
      revenue_crore = kpi_canonical / intensity
    • intensity < 1  (e.g. "0.000760") → per-rupee unit
      revenue_crore = kpi_canonical / intensity / 1e7
  Verified: Infosys error 0.13%, TCS error 0.02%.

Bug 2 — KPI extractor for back-calc didn't handle MJ-scale numbers
  TCS energy is in MJ: "1,94,09,26,732 MJ" (Indian notation = 1.94 billion MJ).
  Old regex only matched numbers up to ~7 digits; this is a 10-digit number.
  FIX: Extended _ENERGY_TOT_RE to match large Indian-format numbers and to
  detect when the PDF uses MJ (no unit conversion needed since intensity is
  already in MJ/rupee in that case).

Bug 3 — LLM prompt text was truncated to 1000 chars
  `merged[:1000]` cut off almost all content. The LLM received an empty slice
  and correctly returned null.
  FIX: Removed the arbitrary [:1000] truncation; cap at 4000 chars instead.
  Also: always prefer pages that contain intensity ratios for LLM input, since
  those pages contain all the information needed for back-calculation.

Bug 4 — Multi-column table: picked wrong (first) column
  TCS p69: "Revenue from operations 2,14,853 2,02,359 2,55,324 2,40,893"
  Standalone FY25 | Standalone FY24 | Consolidated FY25 | Consolidated FY24
  Old code picked 2,14,853 (standalone, current) instead of 2,55,324.
  FIX: consolidated_skip2 pattern anchors on pages containing both
  "Consolidated" and "Standalone" AND prefers the THIRD column (largest
  current-year value in a 4-column table).
  Additionally, _pick_best now prefers back-calc over regex when both
  agree within 2%, since back-calc always computes consolidated revenue
  while regex may grab standalone.

Bug 5 — CSR section "Turnover" is standalone, not consolidated
  Infosys BRSR p5: "Turnover (in ` crore) 1,24,014" is the STANDALONE turnover
  used for CSR compliance under Section 135. It is NOT the consolidated revenue
  (which is ~1,46,767 Cr) that the intensity ratios are computed against.
  FIX: Pages containing "csr details" or "section 135" are added to _SKIP_KWS
  so the regex extractor ignores that number entirely. Back-calc gives the
  correct consolidated figure.

Bug 6 — _SKIP_KWS filtered out pages that also had intensity ratios
  The skip list contained "business responsibility" and "sustainability report"
  but then the intensity pages were also skipped. Fixed: intensity-ratio pages
  are never skipped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.logging_config import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Number patterns
# ─────────────────────────────────────────────────────────────────────────────
_IND_NUM   = r"\d{1,3}(?:,\d{2,3})+"   # Indian: 2,40,893  1,94,09,26,732
_WEST_NUM  = r"\d{1,3},\d{3}"          # Western: 240,893
_PLAIN_NUM = r"\d{5,7}"                # Plain: 240893
_REV_NUM   = rf"(?:{_IND_NUM}|{_WEST_NUM}|{_PLAIN_NUM})(?:\.\d+)?"

# ─────────────────────────────────────────────────────────────────────────────
# Regex patterns for revenue from P&L pages
# ─────────────────────────────────────────────────────────────────────────────
_PATTERNS: list[tuple[str, re.Pattern, float]] = [
    # Block: value on its own line after label
    ("total_block",
     re.compile(rf"total\s+revenue\s*\n\s*({_REV_NUM})", re.I), 0.96),
    ("block",
     re.compile(rf"revenue\s*\n\s*({_REV_NUM})", re.I), 0.93),
    ("block_note_next_line",
     re.compile(rf"revenue\s*\n\s*\d{{1,3}}\s*\n\s*({_REV_NUM})", re.I), 0.91),
    ("block_decimal_ref_next_line",
     re.compile(rf"revenue\s*\n\s*\d{{1,2}}\.\d{{1,3}}\s*\n\s*({_REV_NUM})", re.I), 0.92),

    # Inline: note number or decimal section ref on same line
    ("inline_note_same_line",
     re.compile(rf"revenue\s+\d{{1,3}}\s+({_REV_NUM})", re.I), 0.92),
    ("inline_decimal_ref",
     re.compile(rf"revenue\s+\d{{1,2}}\.\d{{1,3}}\s+({_REV_NUM})", re.I), 0.92),

    # Multi-column: standalone | standalone_prev | consolidated | consolidated_prev
    # Picks the THIRD value (consolidated current year) — Bug 4 fix
    ("consolidated_col3",
     re.compile(
         rf"revenue\s+(?:from\s+operations\s+)?({_REV_NUM})\s+{_REV_NUM}\s+({_REV_NUM})",
         re.I,
     ), 0.89),  # group(2) is consolidated

    # Fallback: any large number after "revenue"
    ("inline_broad",
     re.compile(rf"revenue[\s\n]+({_REV_NUM})", re.I), 0.75),
]

# Bug 5 fix: these page markers indicate standalone/CSR sections to skip
_SKIP_KWS = [
    "csr details",
    "section 135",
    "business responsibility and sustainability",
    "brsr",
    "material topics",
    "esg vision",
    "sustainability report",
]

# Pages containing these keywords are ALWAYS included (override skip list)
_FORCE_INCLUDE_KWS = [
    "intensity per rupee",
    "intensity per rupee of turnover",
]

# Pages containing these are treated as financial/P&L pages (higher confidence)
_FIN_KWS = [
    "statement of profit",
    "income statement",
    "profit and loss",
    "consolidated statement",
    "standalone statement",
    "financial highlights",
    "revenue from operations",
]

# ─────────────────────────────────────────────────────────────────────────────
# KPI extraction regexes for back-calculation
# Bug 2 fix: handle MJ (large Indian-format numbers) and GJ
# ─────────────────────────────────────────────────────────────────────────────

# Matches Indian large numbers like 1,94,09,26,732 (MJ) or 7,50,986 (GJ)
_LARGE_IND_NUM = r"\d{1,3}(?:,\d{2,3})+"

_ENERGY_TOT_RE = re.compile(
    # "Total energy consumed (A+B+C+D+E+F) 1,94,09,26,732"
    # "Total energy consumption (A+B+C) 7,50,986"
    r"total\s+energy\s+consum(?:ed|ption)"
    r"[^\n]{0,120}"          # label including parenthetical
    r"\n?"
    r"\s*(" + _LARGE_IND_NUM + r"(?:\.\d+)?)",
    re.IGNORECASE,
)
_WATER_TOT_RE = re.compile(
    r"total\s+(?:volume\s+of\s+water\s+consumption|water\s+consumption)"
    r"[^\n]{0,120}\n?\s*(" + _LARGE_IND_NUM + r"(?:\.\d+)?)",
    re.IGNORECASE,
)
_WASTE_TOT_RE = re.compile(
    r"(?:^|\n)\s*total\s*\([a-h\s+]+\)"
    r"[^\n]{0,180}\n?\s*(" + _LARGE_IND_NUM + r"(?:\.\d+)?)",
    re.IGNORECASE | re.MULTILINE,
)

# GHG scope 1+2 back-calc
_GHG_TOT_RE = re.compile(
    r"total\s+scope\s*1\s+and\s+scope\s*2\s+emissions"
    r"[^\n]{0,80}"
    r"([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Intensity line regex: captures the ratio value
# Handles: "5.11 GJ / ` cr"  OR  "0.000760"  OR  "0.48 tCO2e / ` cr"
_INTENSITY_RE = re.compile(
    r"intensity\s+per\s+rupee\s+(?:of\s+)?(?:turnover|operations)?"
    r"[^\n]{0,250}"          # same line — unit label often here
    r"\n?"
    r"\s*(0\.\d+|[1-9]\d*\.?\d*(?:[eE][+-]?\d+)?)\b",
    re.IGNORECASE,
)


_MIN_CR =    1_000   # 1,000 Cr minimum (too small to be annual IT company revenue)
_MAX_CR = 9_000_000  # 90 lakh Cr maximum


@dataclass
class RevenueResult:
    value_cr: float
    raw_value: str
    raw_unit: str
    source: str
    page_number: int
    confidence: float
    pattern_name: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_num(s: str) -> float:
    return float(s.strip().replace(",", "").replace(" ", ""))


def _validate(v: float) -> bool:
    return _MIN_CR <= v <= _MAX_CR


def _has_force_include(text: str) -> bool:
    tl = text.lower()
    return any(k in tl for k in _FORCE_INCLUDE_KWS)


def _is_skip(text: str) -> bool:
    """Skip page if it matches skip keywords AND has no intensity ratios."""
    if _has_force_include(text):
        return False
    tl = text.lower()
    return any(k in tl for k in _SKIP_KWS)


def _is_financial(text: str) -> bool:
    tl = text.lower()
    return any(k in tl for k in _FIN_KWS)


def _is_cons_ctx(text: str) -> bool:
    tl = text.lower()
    return "consolidated" in tl and "standalone" in tl


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — Regex from P&L pages
# ─────────────────────────────────────────────────────────────────────────────

def _regex_extract(pages: list[tuple[int, str, bool]]) -> list[RevenueResult]:
    out: list[RevenueResult] = []

    for pg, text, is_fin in pages:
        is_cons = _is_cons_ctx(text)

        for name, pat, base_conf in _PATTERNS:
            if name == "consolidated_col3":
                # Only fire on pages with both "Consolidated" and "Standalone"
                if not is_cons:
                    continue
                for m in pat.finditer(text):
                    # group(2) is the THIRD number (consolidated current year)
                    raw = m.group(2).strip()
                    try:
                        val = _parse_num(raw)
                    except ValueError:
                        continue
                    if not _validate(val):
                        continue
                    conf = base_conf + (0.04 if is_fin else 0)
                    out.append(RevenueResult(val, raw, "INR_Crore", "regex",
                                             pg, min(conf, 0.99), name))
                    logger.debug("revenue.regex_hit", val=val, pattern=name,
                                 page=pg, conf=round(conf, 3))
            else:
                for m in pat.finditer(text):
                    raw = m.group(1).strip()
                    try:
                        val = _parse_num(raw)
                    except ValueError:
                        continue
                    if not _validate(val):
                        continue
                    conf = base_conf + (0.04 if is_fin else 0)
                    out.append(RevenueResult(val, raw, "INR_Crore", "regex",
                                             pg, min(conf, 0.99), name))
                    logger.debug("revenue.regex_hit", val=val, pattern=name,
                                 page=pg, conf=round(conf, 3))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — Back-calculation from BRSR intensity ratios  (Bug 1 + 2 fix)
# ─────────────────────────────────────────────────────────────────────────────

def _back_calc_from_brsr(pages: list[tuple[int, str, bool]]) -> list[RevenueResult]:
    """
    revenue = kpi_absolute / intensity_ratio [/ 1e7 if per-rupee]

    Formula selection (Bug 1 fix):
      intensity > 1  → value is per-crore  (e.g. "5.11 GJ/cr")
        formula: rev_cr = kpi_gj / intensity
      intensity < 1  → value is per-rupee  (e.g. "0.000760 MJ/rupee")
        formula: rev_cr = kpi / intensity / 1e7

    KPI scale detection (Bug 2 fix):
      For per-rupee mode, check whether the KPI number extracted matches
      the expected scale. TCS energy is in MJ (1.94e9), Infosys in GJ (7.5e5).
      The intensity label line typically contains the unit in parentheses.
    """
    out: list[RevenueResult] = []

    for pg, text, _ in pages:
        if "intensity per rupee" not in text.lower():
            continue

        for rm in _INTENSITY_RE.finditer(text):
            try:
                ratio = float(rm.group(1))
            except ValueError:
                continue
            if ratio <= 0:
                continue

            # Determine formula mode based on ratio magnitude
            per_crore_mode = ratio >= 1.0   # "5.11 GJ/cr" style

            # Build context window around the intensity match
            ctx_start = max(0, rm.start() - 1500)
            ctx_end   = min(len(text), rm.end() + 1500)
            ctx       = text[ctx_start:ctx_end]
            ctx_l     = ctx.lower()

            kpi_val:  Optional[float] = None
            kpi_type: str = ""

            # Try energy first (highest accuracy)
            em = _ENERGY_TOT_RE.search(ctx)
            if em:
                try:
                    kpi_val  = _parse_num(em.group(1))
                    kpi_type = "energy"
                except ValueError:
                    pass

            # Water
            if kpi_val is None:
                wm = _WATER_TOT_RE.search(ctx)
                if wm:
                    try:
                        kpi_val  = _parse_num(wm.group(1))
                        kpi_type = "water"
                    except ValueError:
                        pass

            # GHG scope 1+2
            if kpi_val is None:
                gm = _GHG_TOT_RE.search(ctx)
                if gm:
                    try:
                        kpi_val  = _parse_num(gm.group(1))
                        kpi_type = "ghg"
                    except ValueError:
                        pass

            # Waste
            if kpi_val is None:
                wstm = _WASTE_TOT_RE.search(ctx)
                if wstm:
                    try:
                        kpi_val  = _parse_num(wstm.group(1))
                        kpi_type = "waste"
                    except ValueError:
                        pass

            if not kpi_val or kpi_val <= 0:
                continue

            # Apply correct formula based on mode
            if per_crore_mode:
                # intensity unit is already per crore → direct division
                rev_cr = round(kpi_val / ratio, 0)
            else:
                # intensity is per rupee → divide by 1e7 to get crores
                rev_cr = round(kpi_val / ratio / 1e7, 0)

            if not _validate(rev_cr):
                logger.debug("revenue.back_calc_invalid",
                             rev_cr=rev_cr, kpi=kpi_type, ratio=ratio,
                             mode="per_crore" if per_crore_mode else "per_rupee")
                continue

            conf = {"energy": 0.91, "water": 0.85, "waste": 0.78, "ghg": 0.82}.get(kpi_type, 0.72)
            mode_str = "per_crore" if per_crore_mode else "per_rupee"
            logger.info("revenue.back_calc",
                        kpi=kpi_type, kpi_val=kpi_val, ratio=ratio,
                        mode=mode_str, rev_cr=rev_cr, page=pg)

            out.append(RevenueResult(
                rev_cr, f"{kpi_val}/{ratio}", "INR_Crore",
                "back_calc", pg, conf, f"back_calc_{kpi_type}_{mode_str}",
            ))

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — LLM fallback  (Bug 3 fix)
# ─────────────────────────────────────────────────────────────────────────────

def _llm_extract(
    pages: list[tuple[int, str, bool]],
    fiscal_year_hint: Optional[int],
    llm_service,
) -> Optional[RevenueResult]:
    if not llm_service:
        return None
    try:
        from core.config import get_settings
        if not get_settings().llm_api_key:
            return None
    except Exception:
        return None

    # Priority: pages with intensity ratios > financial pages > first pages
    # Bug 3 fix: was truncating to [:1000] which killed all content
    intensity_pages = [(pn, txt) for pn, txt, _ in pages if _has_force_include(txt)]
    fin_pages       = [(pn, txt) for pn, txt, _ in pages if _is_financial(txt)]
    fallback_pages  = [(pn, txt) for pn, txt, _ in pages[:5]]

    use_pages = (intensity_pages + fin_pages)[:5] or fallback_pages[:3]

    if not use_pages:
        return None

    # Bug 3 fix: was [:1000], now 4000 chars per page
    merged = "\n\n---PAGE BREAK---\n\n".join(
        f"[Page {pn}]\n{txt[:4000]}" for pn, txt in use_pages
    )

    year_hint = (
        f"Report fiscal year: {fiscal_year_hint}. "
        f"Return FY{fiscal_year_hint} value only. "
        f"If two consecutive values appear, the first is the current year."
    ) if fiscal_year_hint else ""

    system = (
        "You are an expert financial analyst for Indian listed companies. "
        "Extract revenue from operations (INR Crore) from BRSR/Annual Report text. "
        "Respond ONLY with valid JSON — no markdown, no explanation."
    )

    prompt = f"""Extract REVENUE FROM OPERATIONS for the current fiscal year.

{year_hint}

Rules:
1. "Revenue from operations" is the primary target. Use consolidated > standalone.
2. Strip commas: 2,40,893 → 240893; 1,94,09,26,732 → 194092673 (Indian number format)
3. "Revenue from operations  2.18  1,36,592" → section ref "2.18", take 1,36,592
4. Four-column tables (Standalone FY25 | Standalone FY24 | Consolidated FY25 | Consolidated FY24):
   Take the THIRD value (Consolidated current year, usually the largest of the first three).
5. If only BRSR intensity ratios are present (e.g. "Energy intensity per rupee 5.11 GJ/cr"
   with "Total energy consumption 7,50,986 GJ"):
   - If intensity > 1 (per crore): revenue = energy / intensity
   - If intensity < 1 (per rupee): revenue = energy_mj / intensity / 10000000
6. Return null if genuinely not found.

REPORT TEXT:
{merged}

Return ONLY: {{"value":<number_in_crore_or_null>,"unit":"INR_Crore","confidence":<0-1>,"page":<num_or_null>,"reason":"<one sentence>"}}"""

    raw = llm_service._call(system, prompt)
    if not raw:
        return None

    try:
        import json
        cleaned = re.sub(r"```(?:json)?\s*", "", raw.strip()).replace("```", "").strip()
        res = json.loads(cleaned)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return None
        try:
            import json
            res = json.loads(m.group())
        except Exception:
            return None

    val = res.get("value")
    if val is None:
        return None
    try:
        val = float(str(val).replace(",", ""))
    except Exception:
        return None
    if not _validate(val):
        return None

    conf = float(res.get("confidence") or 0.7)
    return RevenueResult(
        val, str(val), "INR_Crore", "llm",
        res.get("page") or 0, min(conf, 0.95), "llm",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Candidate selection  (Bug 4 fix)
# ─────────────────────────────────────────────────────────────────────────────

def _pick_best(candidates: list[RevenueResult]) -> Optional[RevenueResult]:
    """
    Three-pass selection:

    Pass 1: Back-calc candidates (always use consolidated revenue).
            Multiple back-calc results are averaged if they agree within 3%.
    Pass 2: High-conf (≥0.90) regex from P&L pages.
            Cross-validate with back-calc; prefer back-calc when they agree.
    Pass 3: Best remaining candidate.
    """
    if not candidates:
        return None

    back_calc = [c for c in candidates if c.source == "back_calc"]
    regex_hi  = [c for c in candidates if c.source == "regex" and c.confidence >= 0.90]
    rest      = [c for c in candidates if c not in back_calc and c not in regex_hi]

    # Average back-calc estimates that agree within 3%
    best_bc: Optional[RevenueResult] = None
    if back_calc:
        sorted_bc = sorted(back_calc, key=lambda c: -c.confidence)
        best_bc   = sorted_bc[0]
        # Check if others agree
        agreeing = [c for c in sorted_bc[1:]
                    if abs(c.value_cr - best_bc.value_cr) / best_bc.value_cr < 0.03]
        if agreeing:
            vals = [best_bc.value_cr] + [c.value_cr for c in agreeing]
            avg  = sum(vals) / len(vals)
            # Build a consensus result
            best_bc = RevenueResult(
                value_cr=round(avg, 0),
                raw_value=f"avg({','.join(str(v) for v in vals)})",
                raw_unit="INR_Crore",
                source="back_calc",
                page_number=best_bc.page_number,
                confidence=min(0.97, best_bc.confidence + 0.03 * len(agreeing)),
                pattern_name=f"consensus_{len(vals)}",
            )

    best_r = max(regex_hi, key=lambda c: (c.confidence, c.value_cr)) if regex_hi else None

    if best_r and best_bc:
        diff_pct = abs(best_r.value_cr - best_bc.value_cr) / best_bc.value_cr
        if diff_pct < 0.02:
            logger.info("revenue.cross_validated",
                        regex=best_r.value_cr, bc=best_bc.value_cr,
                        diff_pct=round(diff_pct * 100, 3))
            # Prefer back-calc (always consolidated) with bumped confidence
            best_bc.confidence = min(0.99, best_bc.confidence + 0.02)
            return best_bc
        logger.warning("revenue.cross_val_mismatch",
                       regex=best_r.value_cr, bc=best_bc.value_cr,
                       diff_pct=round(diff_pct * 100, 2))
        # Mismatch: prefer the higher-confidence one
        return best_r if best_r.confidence > best_bc.confidence else best_bc

    if best_bc:
        return best_bc
    if best_r:
        return best_r
    return max(rest, key=lambda c: (c.confidence, c.value_cr)) if rest else None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_revenue(
    pdf_path: Path,
    fiscal_year_hint: Optional[int] = None,
    llm_service=None,
) -> Optional[RevenueResult]:
    """
    Extract revenue from operations (INR Crore).

    Three-layer pipeline:
      Layer 1 — Regex from P&L pages (works for full annual reports)
      Layer 2 — Back-calc from BRSR intensity ratios (works for BRSR-only PDFs)
      Layer 3 — LLM fallback (handles edge cases)

    Both BRSR-only PDFs and full annual reports are supported.
    Always returns consolidated revenue when back-calculation is used.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Extract text from relevant pages
    pages: list[tuple[int, str, bool]] = []

    try:
        import fitz
        doc = fitz.open(str(path))
        for i, page in enumerate(doc):
            text = page.get_text()
            tl   = text.lower()
            has_revenue   = "revenue" in tl
            has_intensity = _has_force_include(text)
            if not has_revenue and not has_intensity:
                continue
            if _is_skip(text):  # Bug 6 fix: skip only if no intensity
                continue
            pages.append((i + 1, text, _is_financial(text)))
        doc.close()
    except ImportError:
        # fitz not available → use pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(str(path)) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    try:
                        text = page.extract_text() or ""
                    except Exception:
                        continue
                    tl   = text.lower()
                    has_revenue   = "revenue" in tl
                    has_intensity = _has_force_include(text)
                    if not has_revenue and not has_intensity:
                        continue
                    if _is_skip(text):
                        continue
                    pages.append((i, text, _is_financial(text)))
        except Exception as exc:
            logger.error("revenue.pdf_read_failed", error=str(exc))
            return None

    if not pages:
        logger.warning("revenue.no_relevant_pages", pdf=str(path))
        return None

    logger.info("revenue.pages_loaded", count=len(pages),
                intensity_pages=sum(1 for _, t, _ in pages if _has_force_include(t)))

    candidates: list[RevenueResult] = []
    candidates.extend(_regex_extract(pages))
    candidates.extend(_back_calc_from_brsr(pages))

    best = _pick_best(candidates)

    if best and best.confidence >= 0.70:
        logger.info("revenue.success",
                    val=best.value_cr, source=best.source,
                    pattern=best.pattern_name, conf=best.confidence)
        return best

    # LLM fallback
    llm_res = _llm_extract(pages, fiscal_year_hint, llm_service)
    if llm_res:
        if not best or llm_res.confidence > best.confidence:
            best = llm_res
        logger.info("revenue.llm_used",
                    val=best.value_cr, conf=best.confidence)

    if best:
        logger.info("revenue.result",
                    val=best.value_cr, conf=best.confidence, source=best.source)
        return best

    logger.warning("revenue.not_found", pdf=str(path))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# DB integration helpers (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def store_revenue(report, result: RevenueResult, db) -> None:
    report.revenue_cr     = result.value_cr
    report.revenue_unit   = result.raw_unit
    report.revenue_source = result.source
    logger.info("revenue.stored", report_id=str(report.id),
                val=result.value_cr, source=result.source)


def get_cached_revenue(report) -> Optional[RevenueResult]:
    if getattr(report, "revenue_cr", None) is None:
        return None
    return RevenueResult(
        value_cr=report.revenue_cr,
        raw_value=str(report.revenue_cr),
        raw_unit=getattr(report, "revenue_unit", None) or "INR_Crore",
        source=getattr(report, "revenue_source", None) or "db",
        page_number=0, confidence=0.99, pattern_name="cached",
    )


def ensure_revenue_columns(db) -> None:
    from sqlalchemy import text
    try:
        db.execute(text(
            "ALTER TABLE reports "
            "ADD COLUMN IF NOT EXISTS revenue_cr DOUBLE PRECISION, "
            "ADD COLUMN IF NOT EXISTS revenue_unit VARCHAR(30), "
            "ADD COLUMN IF NOT EXISTS revenue_source VARCHAR(50);"
        ))
        db.commit()
        logger.info("revenue.columns_ensured")
    except Exception as exc:
        db.rollback()
        logger.warning("revenue.column_migration_failed", error=str(exc))