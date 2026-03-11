"""
agents/benchmark.py
-------------------
Agent 4 — BenchmarkAgent

Computes benchmarks and KPI rankings.

Benchmarking method: MEDIAN (robust to outliers in small peer groups)
----------------------------------------------------------------------
For each KPI the benchmark is the median of all valid peer values.
Also computes Q1 (25th percentile) and Q3 (75th percentile) for
quartile-based performance banding in the executive summary.

With only 1 company the median = that company's own value, so
gap analysis will always show "Aligned" — this is correct and expected.

Scale context detection
-----------------------
If max(Turnover_INR) / min(Turnover_INR) > SCALE_RATIO_THRESHOLD (10×),
a scale_warning is added to the result for the report to display.

Ranking validation
------------------
A KPI is ranked ONLY if ≥2 companies have non-None, non-unreliable values.
If all values are missing, no ranking row is generated and a message is shown.

Critical rules
--------------
• OPERATIONAL_SCALE_METRICS never appear in benchmarks or rankings.
• None = missing (skipped); 0.0 = genuine zero (included).
• Revenue-normalised metrics (perRevCr) are ranked alongside per-employee.
"""

from __future__ import annotations
from typing import Optional

import statistics
import pandas as pd

from config.constants import (
    LOWER_IS_BETTER, RANKABLE_KPIS, OPERATIONAL_SCALE_METRICS,
    STATIC_BENCHMARKS, BENCHMARK_METHOD, SCALE_RATIO_THRESHOLD,
)


def _median_and_quartiles(vals: list[float]) -> tuple[float, float | None, float | None]:
    """Return (median, Q1, Q3). Q1/Q3 are None when n < 4."""
    med = statistics.median(vals)
    if len(vals) >= 4:
        sorted_v = sorted(vals)
        n = len(sorted_v)
        q1 = statistics.median(sorted_v[:n // 2])
        q3 = statistics.median(sorted_v[(n + 1) // 2:])
        return med, q1, q3
    return med, None, None


class BenchmarkAgent:

    def compute(
        self,
        companies: dict,
        mode: str = "industry",
        focal: Optional[str] = None,
        peer: Optional[str] = None,
    ) -> dict:
        print(
            f"  [BenchmarkAgent] mode={mode}, method={BENCHMARK_METHOD}"
            + (f" | focal={focal} | peer={peer}" if mode == "peer" else "")
        )

        benchmarks:  dict[str, float]          = {}
        quartiles:   dict[str, tuple]           = {}   # kpi → (Q1, Q3) or (None, None)
        peer_counts: dict[str, int]             = {}   # kpi → n companies used

        # ── Benchmarks ────────────────────────────────────────────────────────
        if mode == "industry":
            for kpi in RANKABLE_KPIS:
                vals = [
                    m[kpi]
                    for m in companies.values()
                    if m.get(kpi) is not None
                    and not m.get(kpi + "_UNRELIABLE")
                    and isinstance(m.get(kpi), (int, float))
                ]
                if vals:
                    med, q1, q3 = _median_and_quartiles(vals)
                    benchmarks[kpi]  = round(med, 6)
                    quartiles[kpi]   = (q1, q3)
                    peer_counts[kpi] = len(vals)

            for k, v in STATIC_BENCHMARKS.items():
                benchmarks.setdefault(k, v)
                quartiles.setdefault(k, (None, None))
                peer_counts.setdefault(k, 0)

            mode_info = f"Industry median benchmark ({BENCHMARK_METHOD})"

        elif mode == "peer":
            if not focal or not peer:
                raise ValueError("focal and peer are required for peer mode")
            peer_m = companies.get(peer, {})
            for k, v in peer_m.items():
                if k not in RANKABLE_KPIS:
                    continue
                if v is None or peer_m.get(k + "_UNRELIABLE"):
                    continue
                if not isinstance(v, (int, float)):
                    continue
                benchmarks[k]  = v
                quartiles[k]   = (None, None)
                peer_counts[k] = 1
            mode_info = f"Peer benchmark: {focal} vs {peer}"

        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        # ── Scale context detection ───────────────────────────────────────────
        scale_warning: str | None = None
        turnovers = [
            (name, m["Turnover_INR"])
            for name, m in companies.items()
            if isinstance(m.get("Turnover_INR"), (int, float))
            and m["Turnover_INR"] > 0
        ]
        if len(turnovers) >= 2:
            max_rev = max(t[1] for t in turnovers)
            min_rev = min(t[1] for t in turnovers)
            ratio   = max_rev / min_rev if min_rev else float("inf")
            if ratio >= SCALE_RATIO_THRESHOLD:
                big   = next(n for n, v in turnovers if v == max_rev)
                small = next(n for n, v in turnovers if v == min_rev)
                scale_warning = (
                    f"⚠  SCALE CONTEXT WARNING\n"
                    f"   Companies differ significantly in operational scale.\n"
                    f"   {big} revenue is {ratio:.0f}× larger than {small}.\n"
                    f"   Absolute environmental totals (TotalGHG, TotalEnergy, etc.) reflect\n"
                    f"   this size difference and are NOT ESG performance indicators.\n"
                    f"   Use per-employee and per-revenue intensity metrics for ESG comparison."
                )
                print(f"  [BenchmarkAgent] {scale_warning.splitlines()[0]}")

        # ── KPI Rankings ──────────────────────────────────────────────────────
        rows = []
        skipped_no_data      = []
        skipped_insufficient = []   # Issue 7: safety/governance with <2 valid values

        for kpi in sorted(RANKABLE_KPIS):
            entries = []
            for name, m in companies.items():
                val = m.get(kpi)
                if val is None:
                    continue
                if not isinstance(val, (int, float)):
                    continue
                if m.get(kpi + "_UNRELIABLE"):
                    continue
                entries.append((name, val))

            if len(entries) == 0:
                skipped_no_data.append(kpi)
                continue
            if len(entries) < 2:
                # Issue 7: if it's a safety/governance metric, log as insufficient
                from config.constants import SAFETY_METRICS, GOVERNANCE_METRICS
                if kpi in SAFETY_METRICS or kpi in GOVERNANCE_METRICS:
                    skipped_insufficient.append(kpi)
                continue   # need ≥2 to rank

            reverse = kpi not in LOWER_IS_BETTER
            ranked  = sorted(entries, key=lambda x: x[1], reverse=reverse)
            all_vals = [v for _, v in entries]
            _, q1, q3 = _median_and_quartiles(all_vals)

            for rank, (name, val) in enumerate(ranked, 1):
                # Percentile band based on ranking position
                n = len(ranked)
                pctile = 100 * (n - rank) / max(n - 1, 1)   # 0..100
                if not reverse:
                    pctile = 100 - pctile   # lower-is-better: rank 1 = 100th pctile

                rows.append({
                    "KPI":              kpi,
                    "Rank":             rank,
                    "Company":          name,
                    "Value":            val,
                    "Higher_is_Better": reverse,
                    "Percentile":       round(pctile, 1),
                    "Q1":               q1,
                    "Q3":               q3,
                })

        if skipped_no_data:
            print(f"    ⚠  No data for ranking (all values missing): "
                  f"{', '.join(skipped_no_data)}")
        if skipped_insufficient:
            print(f"    ⚠  Insufficient data for benchmarking (needs ≥2 companies): "
                  f"{', '.join(skipped_insufficient)}")

        rankings_df = pd.DataFrame(rows)
        print(
            f"    → benchmarks: {len(benchmarks)} KPIs | "
            f"rankings: {len(rankings_df)} rows"
        )
        return dict(
            benchmarks=benchmarks,
            quartiles=quartiles,
            peer_counts=peer_counts,
            rankings=rankings_df,
            scale_warning=scale_warning,
            mode_info=mode_info,
            insufficient_safety=skipped_insufficient,
        )