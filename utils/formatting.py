"""
utils/formatting.py
-------------------
Value-display and gap-row formatting helpers.

Display rules
-------------
  None       → "—"   (missing / not extracted)
  0.0        → "N/A" (zero value — treat as not reported)
  float>1e9  → ₹ with comma grouping
  float      → context-appropriate decimal places

Gap language rules (Fix 5 — no raw percentage advantage/disadvantage claims)
------------------------------------------------------------------------------
Raw gap percentages like "+99%" or "-87% advantage" are statistically
misleading with small peer groups and can exaggerate differences.

Instead use calibrated qualitative descriptors:

  |gap_pct| ≥ 50%  → "substantially" (not "99% advantage")
  |gap_pct| ≥ 25%  → "significantly"
  |gap_pct| ≥ 10%  → "moderately"
  |gap_pct| < 10%  → "marginally"

Examples:
  BEFORE: "Company has a 99% advantage in GHG per employee"
  AFTER:  "GHG intensity is substantially lower than the industry median"

  BEFORE: "87% below target"
  AFTER:  "Energy intensity is significantly above the industry median — reduce"

The raw gap value (absolute units) IS shown so readers can verify, but
percentage claims are replaced with these qualitative phrases.

ESG direction rules
-------------------
  LOWER_IS_BETTER KPIs:
    Above benchmark → emitting/consuming LESS → better efficiency
    Below benchmark → emitting/consuming MORE → needs reduction

  HIGHER_IS_BETTER KPIs:
    Above benchmark → outperforming
    Below benchmark → below median — improve
"""

from __future__ import annotations
from config.constants import LOWER_IS_BETTER, ESG_EFFICIENCY_METRICS, SAFETY_METRICS


# ── Qualitative magnitude descriptors ────────────────────────────────────────

def _magnitude(gap_pct: float | None) -> str:
    """Return a qualitative descriptor for the size of a gap — no raw % claim."""
    if gap_pct is None:
        return "marginally"
    abs_pct = abs(gap_pct)
    if abs_pct >= 50: return "substantially"
    if abs_pct >= 25: return "significantly"
    if abs_pct >= 10: return "moderately"
    return "marginally"


# ── Value formatter ───────────────────────────────────────────────────────────

def fmt_value(val, col_w: int = 28, kpi: str = "") -> str:
    """Format a metric value for tabular display."""
    if val is None:
        return f"{'—':<{col_w}}"
    if isinstance(val, (int, float)) and val == 0.0:
        return f"{'N/A':<{col_w}}"
    if isinstance(val, float) and val > 1_000_000_000:
        return f"{'₹{:,.0f}'.format(val):<{col_w}}"
    if isinstance(val, float):
        return f"{val:<{col_w}.4f}"
    return f"{int(val):<{col_w},}"


# ── Gap row formatter ─────────────────────────────────────────────────────────

def fmt_gap_row(
    kpi: str, val: float, benchmark: float,
    gap: float, gap_pct: float | None, status: str,
    col_label_w: int = 46,
) -> str:
    """
    One line of gap-analysis output.

    Uses qualitative magnitude language instead of raw percentage claims.
    Shows absolute gap value (in metric units) so the reader can verify.
    """
    lower   = kpi in LOWER_IS_BETTER
    mag     = _magnitude(gap_pct)
    val_str = "N/A" if val == 0.0 else f"{val:.4f}"
    ref_str = "N/A" if benchmark == 0.0 else f"{benchmark:.4f}"
    gap_str = f"{gap:+.4f}"

    if lower:
        if status == "Above":
            icon = "✓"
            sent = f"{mag} lower than median → {_lower_better_good_label(kpi)}"
        elif status == "Below":
            icon = "✗"
            sent = f"{mag} higher than median → {_lower_better_bad_label(kpi)}"
        else:
            icon = "~"
            sent = "at industry median"
    else:
        if status == "Above":
            icon = "✓"
            sent = f"{mag} above median — outperforming"
        elif status == "Below":
            icon = "✗"
            sent = f"{mag} below median — needs improvement"
        else:
            icon = "~"
            sent = "at industry median"

    return (
        f"    {icon}  {kpi:<{col_label_w}} "
        f"val={val_str}  median={ref_str}  "
        f"Δ={gap_str}  [{sent}]"
    )


# ── Comparison section language helpers ───────────────────────────────────────

def fmt_comparison_row(
    kpi: str, val: float, benchmark: float,
    gap_pct: float | None, status: str,
) -> str:
    """
    Compact single-line comparison (for Section 4 weaknesses / strengths).
    No percentage claims — qualitative magnitude only.
    """
    lower = kpi in LOWER_IS_BETTER
    mag   = _magnitude(gap_pct)
    val_d = "N/A" if val == 0.0 else f"{val:.4f}"
    ref_d = "N/A" if benchmark == 0.0 else f"{benchmark:.4f}"

    if lower:
        direction = (
            f"{mag} lower than median (more efficient)"
            if status == "Above" else
            f"{mag} higher than median (less efficient)"
        )
    else:
        direction = (
            f"{mag} above median"
            if status == "Above" else
            f"{mag} below median"
        )
    return f"val={val_d}  median={ref_d}  [{direction}]"


# ── Private sentiment label helpers ──────────────────────────────────────────

def _lower_better_good_label(kpi: str) -> str:
    if kpi in ESG_EFFICIENCY_METRICS:
        return "lower intensity = better environmental efficiency"
    if kpi in SAFETY_METRICS:
        return "lower count = better safety record"
    return "lower = better"


def _lower_better_bad_label(kpi: str) -> str:
    if kpi in ESG_EFFICIENCY_METRICS:
        return "higher intensity — reduce per-employee or per-revenue consumption"
    if kpi in SAFETY_METRICS:
        return "higher incidence — improve safety practices"
    return "exceeds median — needs reduction"


def rating_label(status: str, kpi: str) -> str:
    if status == "Aligned": return "Average"
    if status == "Above":   return "Good"
    return "Below Average"