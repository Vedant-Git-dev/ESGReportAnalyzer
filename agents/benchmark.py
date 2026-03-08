"""
agents/benchmark.py
-------------------
Agent 4 — BenchmarkAgent

Computes benchmarks and KPI rankings.

Critical rules
--------------
• OPERATIONAL_SCALE_METRICS are NEVER ranked or benchmarked.
  They are absolute totals (TotalGHG, TotalEnergy, etc.) whose magnitude
  reflects company size, not ESG performance.
• Only RANKABLE_KPIS (efficiency intensities, policy rates, social ratios,
  safety, governance) participate in rankings and gap analysis.
• Metrics flagged as "_UNRELIABLE" (small-denominator ratios) are excluded
  from rankings with a warning.
• None values are treated as missing and excluded from ranking entries.
• True zero (0.0) is treated as a valid data point and included.

Modes
-----
  'industry' → benchmark = mean across all loaded companies for each KPI
  'peer'     → benchmark = one specific peer company's metrics
"""

from __future__ import annotations
from typing import Optional

import pandas as pd

from config.constants import (
    LOWER_IS_BETTER,
    RANKABLE_KPIS,
    OPERATIONAL_SCALE_METRICS,
    STATIC_BENCHMARKS,
)


class BenchmarkAgent:

    def compute(
        self,
        companies: dict,
        mode: str = "industry",
        focal: Optional[str] = None,
        peer: Optional[str] = None,
    ) -> dict:
        print(
            f"  [BenchmarkAgent] mode={mode}"
            + (f" | focal={focal} | peer={peer}" if mode == "peer" else "")
        )

        # ── Benchmarks ────────────────────────────────────────────────────────
        if mode == "industry":
            benchmarks: dict[str, float] = {}
            # Only compute benchmarks for RANKABLE_KPIS
            for kpi in RANKABLE_KPIS:
                vals = [
                    m[kpi]
                    for m in companies.values()
                    if m.get(kpi) is not None           # None = missing — skip
                    and not m.get(kpi + "_UNRELIABLE")  # unreliable ratio — skip
                    and isinstance(m.get(kpi), (int, float))
                ]
                if vals:
                    benchmarks[kpi] = round(sum(vals) / len(vals), 6)
            # Fill in static industry benchmarks for KPIs not computed from data
            for k, v in STATIC_BENCHMARKS.items():
                benchmarks.setdefault(k, v)
            mode_info = "Industry average benchmark"

        elif mode == "peer":
            if not focal or not peer:
                raise ValueError("focal and peer are required for peer mode")
            peer_m = companies.get(peer, {})
            benchmarks = {
                k: v
                for k, v in peer_m.items()
                if k in RANKABLE_KPIS
                and v is not None
                and not peer_m.get(k + "_UNRELIABLE")
                and isinstance(v, (int, float))
            }
            mode_info = f"Peer benchmark: {focal} vs {peer}"

        else:
            raise ValueError(f"Unknown benchmark mode: {mode!r}")

        # ── KPI Rankings ──────────────────────────────────────────────────────
        rows = []
        unreliable_warnings = []

        for kpi in sorted(RANKABLE_KPIS):
            # Build valid entries: exclude None, exclude unreliable flags
            entries = []
            for name, m in companies.items():
                val = m.get(kpi)
                if val is None:
                    continue  # missing data
                if not isinstance(val, (int, float)):
                    continue
                if m.get(kpi + "_UNRELIABLE"):
                    unreliable_warnings.append(
                        f"    ⚠  {kpi} excluded from ranking for {name} "
                        f"(denominator < {30})"
                    )
                    continue
                entries.append((name, val))

            if len(entries) < 2:
                continue  # need at least 2 companies to rank

            reverse = kpi not in LOWER_IS_BETTER
            ranked  = sorted(entries, key=lambda x: x[1], reverse=reverse)
            for rank, (name, val) in enumerate(ranked, 1):
                rows.append({
                    "KPI":              kpi,
                    "Rank":             rank,
                    "Company":          name,
                    "Value":            val,
                    "Higher_is_Better": reverse,
                })

        for w in unreliable_warnings:
            print(w)

        rankings_df = pd.DataFrame(rows)
        print(
            f"    → benchmarks: {len(benchmarks)} KPIs | "
            f"rankings: {len(rankings_df)} rows"
        )
        return dict(
            benchmarks=benchmarks,
            rankings=rankings_df,
            mode_info=mode_info,
        )