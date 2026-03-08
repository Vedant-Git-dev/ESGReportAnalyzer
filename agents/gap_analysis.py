"""
agents/gap_analysis.py
----------------------
Agent 5 — GapAnalysisAgent

Classifies each company-KPI pair vs its benchmark.

Critical rules
--------------
• Gap analysis runs ONLY on RANKABLE_KPIS (efficiency intensities, policy
  rates, social ratios, safety, governance).
• OPERATIONAL_SCALE_METRICS (TotalGHG, TotalEnergy, etc.) are NEVER
  included in gap analysis — their absolute magnitude reflects company
  size, not ESG performance.
• None values (missing data) are excluded entirely.
• Metrics flagged as "_UNRELIABLE" (small denominator) are excluded.
• For LOWER_IS_BETTER KPIs the semantics are inverted:
    val < benchmark → Status = 'Above'  (performing better — emitting less)
    val > benchmark → Status = 'Below'  (performing worse  — emitting more)
• ±10% band determines Good / Average / Below Average in the executive summary.

Output columns
--------------
Company, KPI, KPI_Type, Value, Benchmark, Gap, Gap_Pct, Status, Rating
"""

from __future__ import annotations

import pandas as pd

from config.constants import (
    LOWER_IS_BETTER,
    RANKABLE_KPIS,
    OPERATIONAL_SCALE_METRICS,
    GOOD_THRESHOLD,
    BAD_THRESHOLD,
    ESG_EFFICIENCY_METRICS,
    ESG_POLICY_METRICS,
    SOCIAL_METRICS,
    SAFETY_METRICS,
    GOVERNANCE_METRICS,
)


def _kpi_type(kpi: str) -> str:
    if kpi in ESG_EFFICIENCY_METRICS: return "ESG Efficiency"
    if kpi in ESG_POLICY_METRICS:     return "ESG Policy"
    if kpi in SOCIAL_METRICS:         return "Social"
    if kpi in SAFETY_METRICS:         return "Safety"
    if kpi in GOVERNANCE_METRICS:     return "Governance"
    return "Other"


def _rating(gap_pct: float | None, lower: bool) -> str:
    """
    GOOD         : ≥ +10% better than benchmark
    AVERAGE      : within ±10%
    BELOW AVERAGE: ≥ 10% worse than benchmark
    For lower-is-better KPIs the sign is inverted before comparison.
    """
    if gap_pct is None:
        return "Average"
    # For lower-is-better, negative gap_pct means we're below benchmark = better
    perf_pct = -gap_pct if lower else gap_pct
    if perf_pct >= GOOD_THRESHOLD * 100:
        return "Good"
    if perf_pct <= -BAD_THRESHOLD * 100:
        return "Below Average"
    return "Average"


class GapAnalysisAgent:

    def analyze(self, companies: dict, benchmarks: dict) -> pd.DataFrame:
        rows = []

        for company, metrics in companies.items():
            for kpi, benchmark in benchmarks.items():

                # Hard guard: never analyse operational scale metrics
                if kpi in OPERATIONAL_SCALE_METRICS:
                    continue

                # Only analyse rankable KPIs
                if kpi not in RANKABLE_KPIS:
                    continue

                val = metrics.get(kpi)

                # Skip missing data
                if val is None:
                    continue
                if not isinstance(val, (int, float)):
                    continue

                # Skip unreliable ratios (small denominator)
                if metrics.get(kpi + "_UNRELIABLE"):
                    continue

                # Skip zero benchmarks that would cause division errors
                if benchmark is None or not isinstance(benchmark, (int, float)):
                    continue

                lower   = kpi in LOWER_IS_BETTER
                gap     = round(val - benchmark, 6)
                pct_gap = round(gap / benchmark * 100, 1) if benchmark != 0 else None

                if abs(gap) < 1e-9:
                    status = "Aligned"
                elif (gap > 0 and not lower) or (gap < 0 and lower):
                    status = "Above"   # better than benchmark
                else:
                    status = "Below"   # worse than benchmark

                rating = _rating(pct_gap, lower)

                rows.append(dict(
                    Company=company,
                    KPI=kpi,
                    KPI_Type=_kpi_type(kpi),
                    Value=round(val, 4),
                    Benchmark=round(benchmark, 4),
                    Gap=round(gap, 4),
                    Gap_Pct=pct_gap,
                    Status=status,
                    Rating=rating,
                ))

        df = pd.DataFrame(rows)
        print(f"  [GapAnalysisAgent] → {len(df)} gap records "
              f"(operational scale metrics excluded)")
        return df