"""
utils/formatting.py
-------------------
Value-display and gap-row formatting helpers.

Display rules
-------------
  None       → "—"   (missing / not extracted)
  0.0        → "N/A" (zero value — treat as not reported)
  float>1e9  → ₹ with comma grouping
  float      → 2 decimal places

Gap sentiment rules (ESG-correct language)
------------------------------------------
  LOWER_IS_BETTER KPIs:
    Status=Above  → val < benchmark → company emits/uses LESS → "✓ More efficient"
    Status=Below  → val > benchmark → company emits/uses MORE → "✗ Exceeds target — reduce"

  HIGHER_IS_BETTER KPIs:
    Status=Above  → val > benchmark → "✓ Outperforming"
    Status=Below  → val < benchmark → "✗ Below target — improve"
"""

from __future__ import annotations
from config.constants import LOWER_IS_BETTER, ESG_EFFICIENCY_METRICS, SAFETY_METRICS


def fmt_value(val, col_w: int = 28, kpi: str = "") -> str:
    """Format a metric value for tabular display."""
    if val is None:
        return f"{'—':<{col_w}}"
    if isinstance(val, (int, float)) and val == 0.0:
        return f"{'N/A':<{col_w}}"
    if isinstance(val, float) and val > 1_000_000_000:
        return f"{'₹{:,.0f}'.format(val):<{col_w}}"
    if isinstance(val, float):
        return f"{val:<{col_w}.2f}"
    return f"{int(val):<{col_w},}"


def fmt_gap_row(
    kpi: str, val: float, benchmark: float,
    gap: float, gap_pct: float | None, status: str,
    col_label_w: int = 46,
) -> str:
    """
    One line of gap-analysis output.
    Generates ESG-correct language for lower-is-better KPIs.
    """
    lower   = kpi in LOWER_IS_BETTER
    pct_str = f"({gap_pct:+.1f}%)" if gap_pct is not None else ""
    val_str = "N/A" if val == 0.0 else f"{val:.3f}"
    ref_str = "N/A" if benchmark == 0.0 else f"{benchmark:.3f}"

    if lower:
        if status == "Above":
            icon  = "✓"
            sent  = _lower_better_good_label(kpi)
        elif status == "Below":
            icon  = "✗"
            sent  = _lower_better_bad_label(kpi)
        else:
            icon, sent = "~", "on target"
    else:
        if status == "Above":
            icon, sent = "✓", "outperforming"
        elif status == "Below":
            icon, sent = "✗", "below target — improve"
        else:
            icon, sent = "~", "on target"

    return (
        f"    {icon}  {kpi:<{col_label_w}} "
        f"val={val_str}  ref={ref_str}  "
        f"gap={gap:+.3f} {pct_str}  [{sent}]"
    )


def _lower_better_good_label(kpi: str) -> str:
    """ESG-correct 'this is good' language for lower-is-better KPIs."""
    if kpi in ESG_EFFICIENCY_METRICS:
        return "lower intensity → better environmental efficiency"
    if kpi in SAFETY_METRICS:
        return "lower count → better safety record"
    return "lower = better — efficient"


def _lower_better_bad_label(kpi: str) -> str:
    """ESG-correct 'this needs improvement' language for lower-is-better KPIs."""
    if kpi in ESG_EFFICIENCY_METRICS:
        return "higher intensity → reduce emissions/consumption per employee"
    if kpi in SAFETY_METRICS:
        return "higher count → improve safety practices"
    return "exceeds target — needs reduction"


def rating_label(status: str, kpi: str) -> str:
    """Plain-language rating accounting for lower-is-better inversion."""
    lower = kpi in LOWER_IS_BETTER
    if status == "Aligned":
        return "Average"
    if status == "Above":
        return "Good"
    return "Below Average"