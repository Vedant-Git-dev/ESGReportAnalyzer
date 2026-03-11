"""
agents/gap_analysis.py
----------------------
Agent 5 — GapAnalysisAgent

Classifies each company-KPI pair vs its benchmark.

Scoring logic (in priority order)
----------------------------------
1. Percentile-based (preferred, used when ≥4 peers exist):
     For LOWER_IS_BETTER:  rank 1 (lowest value) = 100th percentile (best)
     For HIGHER_IS_BETTER: rank 1 (lowest value) = 0th percentile (worst)
     Percentile ≥ PERCENTILE_GOOD (75) → Good
     Percentile ≤ PERCENTILE_BAD  (25) → Below Average
     Otherwise → Average

2. Gap-threshold fallback (used when < 4 peers or static benchmark):
     GOOD         : value is ≥10% BETTER than benchmark (direction-aware)
     AVERAGE      : within ±10%
     BELOW AVERAGE: value is ≥10% WORSE than benchmark (direction-aware)

Critical rules (Fixes 2, 3, 4)
--------------
• OPERATIONAL_SCALE_METRICS NEVER appear in gap analysis.
• None values are EXCLUDED — they are never ranked (Fix 4).
• _UNRELIABLE ratios are excluded.
• Direction is strictly from LOWER_IS_BETTER — strength = top quartile
  considering direction (Fix 2 + 3).
• Direction column added to each row for downstream audit.

Output columns
--------------
Company, KPI, KPI_Type, Value, Benchmark, Q1, Q3,
Gap, Gap_Pct, Status, Rating, Percentile, Direction
"""

from __future__ import annotations

import pandas as pd

from config.constants import (
    LOWER_IS_BETTER, RANKABLE_KPIS, OPERATIONAL_SCALE_METRICS,
    GOOD_THRESHOLD, BAD_THRESHOLD, PERCENTILE_GOOD, PERCENTILE_BAD,
    ESG_EFFICIENCY_METRICS, ESG_POLICY_METRICS,
    SOCIAL_METRICS, SAFETY_METRICS, GOVERNANCE_METRICS,
)


def _kpi_type(kpi: str) -> str:
    if kpi in ESG_EFFICIENCY_METRICS: return "ESG Efficiency"
    if kpi in ESG_POLICY_METRICS:     return "ESG Policy"
    if kpi in SOCIAL_METRICS:         return "Social"
    if kpi in SAFETY_METRICS:         return "Safety"
    if kpi in GOVERNANCE_METRICS:     return "Governance"
    return "Other"


def _rating_from_percentile(percentile: float) -> str:
    """Quartile banding: top 25% = Good, bottom 25% = Below Average."""
    if percentile >= PERCENTILE_GOOD:
        return "Good"
    if percentile <= PERCENTILE_BAD:
        return "Below Average"
    return "Average"


def _rating_from_gap(gap_pct: float | None, lower: bool) -> str:
    """
    Direction-aware ±10% threshold fallback (< 4 peers).

    perf_pct > 0 means outperforming benchmark (direction-corrected):
      lower_better: gap_pct < 0  → value < benchmark → outperforming
      higher_better: gap_pct > 0 → value > benchmark → outperforming
    """
    if gap_pct is None:
        return "Average"
    perf_pct = -gap_pct if lower else gap_pct
    if perf_pct >= GOOD_THRESHOLD * 100:
        return "Good"
    if perf_pct <= -BAD_THRESHOLD * 100:
        return "Below Average"
    return "Average"


def _is_valid_for_ranking(val, kpi: str, metrics_dict: dict) -> bool:
    """
    Fix Issue 4: N/A values must never be ranked.
    Returns True only when value is a genuine, usable numeric observation.
    """
    if val is None:
        return False
    if not isinstance(val, (int, float)):
        return False
    if metrics_dict.get(kpi + "_UNRELIABLE"):
        return False
    return True


class GapAnalysisAgent:

    def analyze(
        self,
        companies: dict,
        benchmarks: dict,
        quartiles: dict | None = None,
        peer_counts: dict | None = None,
        rankings_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        quartiles    : {kpi: (Q1, Q3)} from BenchmarkAgent
        peer_counts  : {kpi: n} — number of companies used for benchmark
        rankings_df  : rankings DataFrame (for percentile lookup)
        """
        quartiles   = quartiles   or {}
        peer_counts = peer_counts or {}

        # Build percentile lookup: {(company, kpi): percentile}
        pctile_map: dict[tuple, float] = {}
        if rankings_df is not None and not rankings_df.empty:
            for _, r in rankings_df.iterrows():
                pctile_map[(r["Company"], r["KPI"])] = r["Percentile"]

        rows = []

        for company, metrics in companies.items():
            for kpi, benchmark in benchmarks.items():
                if kpi in OPERATIONAL_SCALE_METRICS:
                    continue
                if kpi not in RANKABLE_KPIS:
                    continue

                val = metrics.get(kpi)

                # Fix Issue 4: exclude N/A — never rank missing values
                if not _is_valid_for_ranking(val, kpi, metrics):
                    continue

                if benchmark is None or not isinstance(benchmark, (int, float)):
                    continue

                lower   = kpi in LOWER_IS_BETTER
                gap     = round(val - benchmark, 6)
                pct_gap = round(gap / benchmark * 100, 1) if benchmark != 0 else None

                # Status: which side of benchmark (direction-aware label)
                if abs(gap) < 1e-9:
                    status = "Aligned"
                elif (gap > 0 and not lower) or (gap < 0 and lower):
                    status = "Above"    # better than benchmark
                else:
                    status = "Below"    # worse than benchmark

                # Fix Issues 2+3: strictly direction-aware rating
                percentile = pctile_map.get((company, kpi))
                n          = peer_counts.get(kpi, 0)

                if percentile is not None and n >= 4:
                    rating = _rating_from_percentile(percentile)
                else:
                    rating = _rating_from_gap(pct_gap, lower)

                q1_val, q3_val = quartiles.get(kpi, (None, None))

                rows.append(dict(
                    Company=company,
                    KPI=kpi,
                    KPI_Type=_kpi_type(kpi),
                    Value=round(val, 6),
                    Benchmark=round(benchmark, 6),
                    Q1=round(q1_val, 4) if q1_val is not None else None,
                    Q3=round(q3_val, 4) if q3_val is not None else None,
                    Gap=round(gap, 6),
                    Gap_Pct=pct_gap,
                    Status=status,
                    Rating=rating,
                    Percentile=percentile,
                    Direction="lower_better" if lower else "higher_better",
                ))

        df = pd.DataFrame(rows)
        print(
            f"  [GapAnalysisAgent] → {len(df)} gap records "
            f"(operational scale excluded, median benchmark)"
        )
        return df