"""
agents/report_builder.py
------------------------
ReportBuilder — six-section ESG benchmarking report.

Sections
--------
  SECTION 0 — Executive Summary     (Good / Average / Below Average, ±10% threshold)
  SECTION 1 — Company KPI Summary   (side-by-side table, clearly typed)
  SECTION 2 — ESG Efficiency Rankings (intensity metrics only; lower = better)
  SECTION 3 — Gap Analysis          (ESG-correct language; scale metrics excluded)
  SECTION 4 — Company Comparison    (peer head-to-head or industry leaders)
  SECTION 5 — Key ESG Insights      (LLM narrative)

Display rules
-------------
  • None / missing            → "—"
  • 0.0 (genuine or missing)  → "N/A"  (treat as not-reported)
  • Lower-is-better KPIs      → ↓better label; correct gap language
  • Operational scale metrics → [scale] tag; never ranked
  • Unreliable ratios         → ⚠ [unreliable — small sample]

Scoring thresholds (GOOD_THRESHOLD / BAD_THRESHOLD from constants)
  GOOD         : ≥ +10% better than benchmark
  AVERAGE      : within ±10%
  BELOW AVERAGE: ≥ 10% worse than benchmark
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from config.constants import (
    LOWER_IS_BETTER, DISPLAY_GROUPS,
    OPERATIONAL_SCALE_METRICS,
    ESG_EFFICIENCY_METRICS, ESG_POLICY_METRICS,
    SOCIAL_METRICS, SAFETY_METRICS, GOVERNANCE_METRICS,
    GOOD_THRESHOLD, BAD_THRESHOLD, RANKABLE_KPIS,
)
from utils.formatting import fmt_value, fmt_gap_row


class ReportBuilder:

    MEDALS = ["🥇", "🥈", "🥉"]

    # ESG category → KPIs to score (only RANKABLE, ESG-correct KPIs)
    SUMMARY_GROUPS: list[tuple[str, list[str]]] = [
        ("Gender Diversity — Employees", [
            "Female_Ratio_PermanentEmp",
            "Female_Ratio_AllEmployees",
        ]),
        ("Gender Diversity — Workers", [
            "Female_Ratio_PermanentWorkers",
            "Female_Ratio_ContractWorkers",
            "Female_Ratio_AllWorkers",
        ]),
        ("Renewable Energy", [
            "RenewableEnergyShare_Pct",
        ]),
        ("GHG Emissions Intensity", [
            "GHG_tCO2e_perEmployee",      # lower = better
        ]),
        ("Energy Efficiency", [
            "Energy_GJ_perEmployee",       # lower = better
        ]),
        ("Water Efficiency", [
            "Water_KL_perEmployee",        # lower = better
        ]),
        ("Waste Management", [
            "WasteRecoveryRate_Pct",       # higher = better
            "Waste_MT_perEmployee",        # lower = better
        ]),
        ("Workplace Safety", [
            "Fatalities",
            "LTIFR",
            "HighConsequenceInjuries",
        ]),
        ("Training & Development", [
            "Pct_Emp_HS_Training",
            "Pct_Wkr_HS_Training",
        ]),
        ("Governance", [
            "Complaints_Filed",
            "Complaints_Pending",
        ]),
    ]

    # ── public entry point ────────────────────────────────────────────────────

    def build(
        self,
        companies: dict,
        gap_df: pd.DataFrame,
        rankings_df: pd.DataFrame,
        benchmark_result: dict,
        insights: str,
        output_dir: str,
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)

        names = list(companies.keys())
        COL_W = 30
        W     = 52 + COL_W * len(names)
        def hr(ch="━"): return ch * W

        lines: list[str] = []

        lines += [
            "=" * W,
            "  ESG BENCHMARKING REPORT — SEBI BRSR XBRL  |  FY 2024-25",
            f"  Mode     : {benchmark_result['mode_info']}",
            f"  Companies: {' | '.join(names)}",
            "=" * W, "",
        ]

        lines += self._section_summary(companies, gap_df, names, W, COL_W)
        lines += self._section_kpi_table(companies, names, W, COL_W)
        lines += self._section_rankings(rankings_df, companies, W)
        lines += self._section_gap(gap_df, names, W)
        lines += self._section_comparison(
            companies, gap_df, rankings_df, names, W, COL_W,
            benchmark_result["mode_info"])
        lines += [hr(), "  SECTION 5 — KEY ESG INSIGHTS", hr(), "", insights, ""]
        lines += ["=" * W, "  END OF REPORT", "=" * W]

        report = "\n".join(lines)

        with open(os.path.join(output_dir, "esg_report.txt"), "w", encoding="utf-8") as f:
            f.write(report)
        gap_df.to_csv(os.path.join(output_dir, "gap_analysis.csv"), index=False)
        rankings_df.to_csv(os.path.join(output_dir, "kpi_rankings.csv"), index=False)
        pd.DataFrame(companies).T.rename_axis("Company").to_csv(
            os.path.join(output_dir, "normalized_metrics.csv"))

        print(f"  [ReportBuilder] Saved → {output_dir}/esg_report.txt")
        return report

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 0 — Executive Summary
    # ══════════════════════════════════════════════════════════════════════════

    def _section_summary(
        self, companies: dict, gap_df: pd.DataFrame,
        names: list, W: int, COL_W: int,
    ) -> list[str]:
        def hr(ch="━"): return ch * W

        lines: list[str] = [
            hr(), "  SECTION 0 — EXECUTIVE SUMMARY", hr(), "",
            "  ESG Performance Rating per Company",
            "  ✓ GOOD (≥+10% vs benchmark)  |  ~ AVERAGE (±10%)  "
            "|  ✗ BELOW AVERAGE (≤−10%)  |  — N/A", "",
        ]

        # ── Build rating matrix ───────────────────────────────────────────────
        # Use the Rating column from gap_df (pre-computed with ±10% thresholds)
        matrix: dict[str, dict[str, str]] = {c: {} for c in names}

        for group_name, kpis in self.SUMMARY_GROUPS:
            for company in names:
                if gap_df.empty:
                    matrix[company][group_name] = "N/A"
                    continue
                sub = gap_df[
                    (gap_df["Company"] == company) &
                    (gap_df["KPI"].isin(kpis))
                ]
                if sub.empty:
                    matrix[company][group_name] = "N/A"
                    continue
                ratings = sub["Rating"].tolist()
                # Aggregate: any Below Average → Below Average;
                # all Good → Good; else Average
                if "Below Average" in ratings:
                    agg = "Below Average"
                elif all(r == "Good" for r in ratings):
                    agg = "Good"
                else:
                    agg = "Average"
                matrix[company][group_name] = agg

        ICON = {"Good": "✓", "Average": "~", "Below Average": "✗", "N/A": "—"}

        # ── Matrix table ──────────────────────────────────────────────────────
        hdr = f"  {'Category':<38}" + "".join(f"{n[:COL_W-2]:<{COL_W}}" for n in names)
        lines.append(hdr)
        lines.append("  " + "─" * (38 + COL_W * len(names)))

        for group_name, _ in self.SUMMARY_GROUPS:
            row = f"  {group_name:<38}"
            for company in names:
                rating = matrix[company].get(group_name, "N/A")
                icon   = ICON.get(rating, "—")
                row   += f"{icon + ' ' + rating:<{COL_W}}"
            lines.append(row)

        lines.append("")

        # ── Per-company narrative ─────────────────────────────────────────────
        lines.append("  Company Snapshot")
        lines.append("  " + "─" * 60)
        for company in names:
            rm   = matrix[company]
            good = [g for g, r in rm.items() if r == "Good"]
            avg  = [g for g, r in rm.items() if r == "Average"]
            bad  = [g for g, r in rm.items() if r == "Below Average"]
            na   = [g for g, r in rm.items() if r == "N/A"]

            lines.append(f"\n  {company}")
            if good:
                lines.append(f"    ✓ GOOD          : {', '.join(good)}")
            if avg:
                lines.append(f"    ~ AVERAGE       : {', '.join(avg)}")
            if bad:
                lines.append(f"    ✗ BELOW AVERAGE : {', '.join(bad)}")
            if na:
                lines.append(f"    — DATA MISSING  : {', '.join(na)}")

        # Unreliable flags
        for company in names:
            flags = [k.replace("_UNRELIABLE","") for k, v
                     in companies[company].items() if k.endswith("_UNRELIABLE") and v]
            if flags:
                lines.append(
                    f"\n  ⚠  {company}: the following ratios have small denominators "
                    f"(< 30) and are excluded from rankings:\n"
                    + "\n".join(f"       • {f}" for f in flags))

        lines.append("")
        return lines

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — KPI Summary Table
    # ══════════════════════════════════════════════════════════════════════════

    def _section_kpi_table(
        self, companies: dict, names: list, W: int, COL_W: int,
    ) -> list[str]:
        def hr(ch="━"): return ch * W

        lines: list[str] = [hr(), "  SECTION 1 — COMPANY KPI SUMMARY", hr(), ""]
        LABEL_W = 52

        hdr = f"  {'Metric':<{LABEL_W}}" + "".join(f"{n[:COL_W-2]:<{COL_W}}" for n in names)
        lines.append(hdr)
        lines.append("  " + "─" * (LABEL_W + COL_W * len(names)))

        for group_name, kpi_pairs in DISPLAY_GROUPS:
            visible = [
                (key, label) for key, label in kpi_pairs
                if any(companies[c].get(key) is not None for c in names)
            ]
            if not visible:
                continue
            lines.append(f"\n  [{group_name}]")
            for key, label in visible:
                # Suffix for metric type
                if key in LOWER_IS_BETTER:
                    suffix = "  ↓better"
                elif key in OPERATIONAL_SCALE_METRICS:
                    suffix = "  [scale]"
                else:
                    suffix = ""
                row = f"  {label + suffix:<{LABEL_W}}"
                for c in names:
                    val  = companies[c].get(key)
                    flag = companies[c].get(key + "_UNRELIABLE")
                    if flag:
                        row += f"{'⚠ ' + str(round(val,2)) + '%':<{COL_W}}"
                    else:
                        row += fmt_value(val, COL_W, key)
                lines.append(row)

        lines.append("")
        return lines

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — ESG Efficiency Rankings
    # ══════════════════════════════════════════════════════════════════════════

    def _section_rankings(
        self, rankings_df: pd.DataFrame, companies: dict, W: int,
    ) -> list[str]:
        def hr(ch="━"): return ch * W

        lines: list[str] = [
            hr(),
            "  SECTION 2 — ESG EFFICIENCY & POLICY RANKINGS",
            "  (Only normalised intensity metrics, policy rates, social ratios,",
            "   safety and governance KPIs — operational scale metrics excluded)",
            hr(), "",
        ]

        if rankings_df.empty:
            lines.append(
                "  Rankings require ≥2 companies with overlapping rankable KPIs.")
            lines.append("")
            return lines

        # Group by metric type for clarity
        TYPE_ORDER = [
            ("ESG Efficiency — lower = better ↓", ESG_EFFICIENCY_METRICS),
            ("ESG Policy — higher = better ↑",    ESG_POLICY_METRICS),
            ("Social — higher = better ↑",         SOCIAL_METRICS),
            ("Safety — lower = better ↓",          SAFETY_METRICS),
            ("Governance — lower = better ↓",      GOVERNANCE_METRICS),
        ]

        for type_label, kpi_set in TYPE_ORDER:
            type_kpis = sorted(
                k for k in rankings_df["KPI"].unique() if k in kpi_set)
            if not type_kpis:
                continue
            lines.append(f"  ── {type_label} ──")
            for kpi in type_kpis:
                sub   = rankings_df[rankings_df["KPI"] == kpi].sort_values("Rank")
                lower = kpi in LOWER_IS_BETTER
                direction = "↓ lowest = best" if lower else "↑ highest = best"
                lines.append(f"\n  {kpi}  [{direction}]")

                for _, row in sub.iterrows():
                    rank     = int(row["Rank"])
                    medal    = self.MEDALS[rank-1] if rank <= 3 else f"   #{rank}"
                    val_d    = "N/A" if row["Value"] == 0.0 else f"{row['Value']:.4f}"
                    n        = len(sub)
                    if lower:
                        perf = ("✓ Most Efficient (lowest intensity)"
                                if rank == 1 else
                                "✗ Least Efficient (highest intensity)"
                                if rank == n else "")
                    else:
                        perf = ("✓ Leader"  if rank == 1 else
                                "✗ Laggard" if rank == n else "")
                    lines.append(
                        f"    {medal}  {row['Company']:<46} {val_d:<14} {perf}")
            lines.append("")

        return lines

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Gap Analysis
    # ══════════════════════════════════════════════════════════════════════════

    def _section_gap(self, gap_df: pd.DataFrame, names: list, W: int) -> list[str]:
        def hr(ch="━"): return ch * W

        lines: list[str] = [
            hr(),
            "  SECTION 3 — GAP ANALYSIS",
            "  (ESG efficiency, policy, social, safety, governance metrics only)",
            "  (Operational scale metrics — TotalGHG, TotalEnergy etc. — excluded)",
            hr(), "",
        ]

        if gap_df.empty:
            lines.append("  No gap data available.")
            lines.append("")
            return lines

        for company in names:
            sub     = gap_df[gap_df["Company"] == company]
            good    = sub[sub["Rating"] == "Good"]
            avg     = sub[sub["Rating"] == "Average"]
            bad     = sub[sub["Rating"] == "Below Average"]

            lines.append(f"  {company}")
            lines.append("  " + "─" * 68)
            lines.append(
                f"  ✓ Good (≥+10%): {len(good):2d}  |  "
                f"~ Average (±10%): {len(avg):2d}  |  "
                f"✗ Below Average (≤−10%): {len(bad):2d}"
            )

            if not bad.empty:
                lines.append("\n  Areas needing improvement:")
                for _, r in bad.sort_values("Gap_Pct").iterrows():
                    lines.append(fmt_gap_row(
                        r["KPI"], r["Value"], r["Benchmark"],
                        r["Gap"], r["Gap_Pct"], r["Status"]))

            if not good.empty:
                lines.append("\n  Areas of strength:")
                for _, r in good.sort_values("Gap_Pct", ascending=False).iterrows():
                    lines.append(fmt_gap_row(
                        r["KPI"], r["Value"], r["Benchmark"],
                        r["Gap"], r["Gap_Pct"], r["Status"]))

            if not avg.empty:
                lines.append("\n  On-target KPIs:")
                for _, r in avg.iterrows():
                    val_d = "N/A" if r["Value"] == 0.0 else f"{r['Value']:.3f}"
                    ref_d = "N/A" if r["Benchmark"] == 0.0 else f"{r['Benchmark']:.3f}"
                    lines.append(
                        f"    ~  {r['KPI']:<46} "
                        f"val={val_d}  ref={ref_d}  [within ±10%]")

            lines.append("")

        return lines

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Company Comparison
    # ══════════════════════════════════════════════════════════════════════════

    def _section_comparison(
        self, companies: dict, gap_df: pd.DataFrame, rankings_df: pd.DataFrame,
        names: list, W: int, COL_W: int, mode_info: str = "",
    ) -> list[str]:
        is_peer = "peer" in mode_info.lower() and len(names) == 2
        title   = (
            f"  SECTION 4 — PEER COMPARISON: {names[0]}  vs  {names[1]}"
            if is_peer else "  SECTION 4 — COMPANY COMPARISON"
        )
        def hr(ch="━"): return ch * W

        lines: list[str] = [hr(), title, hr(), ""]
        lines.append("  " + "  |  ".join(names) + "\n")

        # ── 4A: Head-to-head (peer) or KPI leaders (industry) ────────────────
        if is_peer:
            lines += self._peer_head_to_head(companies, names, COL_W)
        else:
            lines += self._industry_kpi_leaders(rankings_df)

        # ── 4B: Strengths ─────────────────────────────────────────────────────
        lines.append("  ESG Strengths (Good rating — ≥+10% vs benchmark)")
        lines.append("  " + "─" * 68)
        for company in names:
            sub  = gap_df[gap_df["Company"] == company] if not gap_df.empty \
                   else pd.DataFrame()
            good = sub[sub["Rating"] == "Good"] if not sub.empty else pd.DataFrame()
            lines.append(f"\n  {company}")
            if not good.empty:
                for _, r in good.sort_values("Gap_Pct", ascending=False).iterrows():
                    lower = r["KPI"] in LOWER_IS_BETTER
                    val_d = "N/A" if r["Value"] == 0.0 else f"{r['Value']:.3f}"
                    ref_d = r["Benchmark"]
                    note  = (
                        f"lower intensity = better environmental efficiency"
                        if lower and r["KPI"] in ESG_EFFICIENCY_METRICS else
                        "lower = safer" if lower and r["KPI"] in SAFETY_METRICS else
                        "lower = better" if lower else "outperforming"
                    )
                    lines.append(
                        f"    ✓ {r['KPI']:<46} "
                        f"{val_d} vs {ref_d:.3f}  [{note}]"
                    )
            else:
                lines.append("    — No KPIs rated Good vs benchmark")
        lines.append("")

        # ── 4C: Weaknesses ────────────────────────────────────────────────────
        lines.append("  ESG Weaknesses (Below Average — ≤−10% vs benchmark)")
        lines.append("  " + "─" * 68)
        any_bad = False
        for company in names:
            sub = gap_df[gap_df["Company"] == company] if not gap_df.empty \
                  else pd.DataFrame()
            bad = sub[sub["Rating"] == "Below Average"] if not sub.empty \
                  else pd.DataFrame()
            if not bad.empty:
                any_bad = True
                lines.append(f"\n  {company}")
                for _, r in bad.sort_values("Gap_Pct").iterrows():
                    lower = r["KPI"] in LOWER_IS_BETTER
                    val_d = "N/A" if r["Value"] == 0.0 else f"{r['Value']:.3f}"
                    action = (
                        "reduce emissions intensity" if r["KPI"] == "GHG_tCO2e_perEmployee" else
                        "improve energy efficiency" if r["KPI"] == "Energy_GJ_perEmployee" else
                        "reduce water consumption per employee"
                            if r["KPI"] == "Water_KL_perEmployee" else
                        "reduce waste generation" if r["KPI"] == "Waste_MT_perEmployee" else
                        "increase renewable energy share"
                            if r["KPI"] == "RenewableEnergyShare_Pct" else
                        "improve waste recovery" if r["KPI"] == "WasteRecoveryRate_Pct" else
                        "increase female representation"
                            if "Female" in r["KPI"] else
                        "improve safety record" if r["KPI"] in SAFETY_METRICS else
                        "resolve complaints" if r["KPI"] in GOVERNANCE_METRICS else
                        "improve"
                    )
                    lines.append(
                        f"    ✗ {r['KPI']:<46} "
                        f"{val_d} vs {r['Benchmark']:.3f}  "
                        f"gap={r['Gap_Pct']:+.1f}%  → {action}"
                    )
        if not any_bad:
            lines.append("  All companies at or above benchmark on every ranked KPI.")
        lines.append("")

        # ── 4D: Operational Scale (separate, clearly labelled) ────────────────
        lines.append(
            "  Operational Scale & Environmental Impact  "
            "[context only — not used for ESG performance ranking]"
        )
        lines.append("  " + "─" * 68)
        SCALE_ROWS = [
            ("perm_emp_total",    "Permanent Employees"),
            ("total_wkr_total",   "Total Workers"),
            ("Turnover_INR",      "Annual Turnover (INR)"),
            ("NetWorth_INR",      "Net Worth (INR)"),
            ("plants_national",   "National Plants"),
            ("TotalGHG_tCO2e",   "Total GHG Scope1+2 (tCO2e)  ← size-dependent"),
            ("TotalEnergy_GJ",   "Total Energy (GJ)  ← size-dependent"),
            ("WaterConsumption_KL","Water Consumption (KL)  ← size-dependent"),
            ("WasteGenerated_MT", "Waste Generated (MT)  ← size-dependent"),
        ]
        LABEL_W = 48
        lines.append(
            f"  {'Metric':<{LABEL_W}}"
            + "".join(f"{c[:COL_W-2]:<{COL_W}}" for c in names))
        lines.append("  " + "─" * (LABEL_W + COL_W * len(names)))
        for key, label in SCALE_ROWS:
            if not any(companies[c].get(key) is not None for c in names):
                continue
            row = f"  {label:<{LABEL_W}}"
            for c in names:
                row += fmt_value(companies[c].get(key), COL_W, key)
            lines.append(row)

        # Scale comparison narrative
        if len(names) >= 2:
            lines.append("")
            for key, label in [
                ("perm_emp_total", "Permanent Employees"),
                ("Turnover_INR",   "Turnover"),
                ("TotalGHG_tCO2e","Total GHG"),
            ]:
                vals = [(c, companies[c][key]) for c in names
                        if companies[c].get(key) and companies[c][key] > 0]
                if len(vals) >= 2:
                    hi    = max(vals, key=lambda x: x[1])
                    lo    = min(vals, key=lambda x: x[1])
                    ratio = hi[1] / lo[1]
                    extra = (
                        " — higher total GHG is expected for a larger company; "
                        "compare GHG_tCO2e_perEmployee for ESG performance"
                        if "GHG" in key else ""
                    )
                    lines.append(
                        f"  {label}: {hi[0]} is {ratio:.1f}× larger "
                        f"than {lo[0]}{extra}"
                    )
        lines.append("")

        # ── 4E: Environmental efficiency snapshot ─────────────────────────────
        EFF_KEYS = [
            ("GHG_tCO2e_perEmployee",  "GHG / Employee (tCO2e)  ↓ lower = better"),
            ("Energy_GJ_perEmployee",  "Energy / Employee (GJ)  ↓ lower = better"),
            ("Water_KL_perEmployee",   "Water / Employee (KL)  ↓ lower = better"),
            ("Waste_MT_perEmployee",   "Waste / Employee (MT)  ↓ lower = better"),
            ("RenewableEnergyShare_Pct","Renewable Energy Share (%)  ↑ higher = better"),
            ("WasteRecoveryRate_Pct",  "Waste Recovery Rate (%)  ↑ higher = better"),
        ]
        if any(companies[c].get(k) for k, _ in EFF_KEYS for c in names):
            lines.append(
                "  ESG Efficiency Snapshot  "
                "[these are the metrics used for ESG performance comparison]"
            )
            lines.append("  " + "─" * 68)
            ELABEL_W = 46
            lines.append(
                f"  {'Metric':<{ELABEL_W}}"
                + "".join(f"{c[:COL_W-2]:<{COL_W}}" for c in names))
            lines.append("  " + "─" * (ELABEL_W + COL_W * len(names)))
            for key, label in EFF_KEYS:
                if not any(companies[c].get(key) for c in names):
                    continue
                row = f"  {label:<{ELABEL_W}}"
                for c in names:
                    row += fmt_value(companies[c].get(key), COL_W, key)
                lines.append(row)
            lines.append("")

        # ── 4F: Peer delta table ───────────────────────────────────────────────
        if is_peer:
            lines += self._peer_env_delta(companies, names, EFF_KEYS)

        return lines

    # ── peer helpers ──────────────────────────────────────────────────────────

    def _peer_head_to_head(
        self, companies: dict, names: list, COL_W: int,
    ) -> list[str]:
        a, b = names[0], names[1]
        shared = sorted(
            k for k in RANKABLE_KPIS
            if companies[a].get(k) is not None
            and companies[b].get(k) is not None
            and not companies[a].get(k + "_UNRELIABLE")
            and not companies[b].get(k + "_UNRELIABLE")
        )
        if not shared:
            return ["  No shared rankable KPIs found.\n"]

        lines = ["  Head-to-Head KPI Comparison  (ESG metrics only)", "  " + "─" * 68]
        LABEL_W = 42
        lines.append(
            f"  {'KPI':<{LABEL_W}}"
            f"{a[:COL_W-2]:<{COL_W}}"
            f"{b[:COL_W-2]:<{COL_W}}"
            f"{'Winner':<20}")
        lines.append("  " + "─" * (LABEL_W + COL_W * 2 + 20))

        a_wins = b_wins = ties = 0
        for kpi in shared:
            va    = companies[a][kpi]
            vb    = companies[b][kpi]
            lower = kpi in LOWER_IS_BETTER
            va_d  = "N/A" if va == 0.0 else f"{va:.3f}"
            vb_d  = "N/A" if vb == 0.0 else f"{vb:.3f}"
            if va == 0.0 and vb == 0.0:
                winner, ties = "Tie (N/A)", ties + 1
            elif va == 0.0:
                winner, b_wins = f"→ {b[:14]}", b_wins + 1
            elif vb == 0.0:
                winner, a_wins = f"→ {a[:14]}", a_wins + 1
            else:
                best = (a if (lower and va < vb) or (not lower and va > vb)
                        else b if (lower and vb < va) or (not lower and vb > va)
                        else None)
                if best is None:
                    winner, ties = "Tie", ties + 1
                elif best == a:
                    winner, a_wins = f"→ {a[:14]}", a_wins + 1
                else:
                    winner, b_wins = f"→ {b[:14]}", b_wins + 1
            lines.append(
                f"  {kpi:<{LABEL_W}}"
                f"{va_d:<{COL_W}}{vb_d:<{COL_W}}{winner:<20}")

        lines.append("  " + "─" * (LABEL_W + COL_W * 2 + 20))
        lines.append(
            f"  {'KPI Wins':<{LABEL_W}}"
            f"{str(a_wins):<{COL_W}}{str(b_wins):<{COL_W}}"
            f"{'Ties: ' + str(ties):<20}")
        lines.append("")
        return lines

    def _industry_kpi_leaders(self, rankings_df: pd.DataFrame) -> list[str]:
        if rankings_df.empty:
            return []
        lines = ["  KPI Leaders per Metric", "  " + "─" * 68]
        for kpi in sorted(rankings_df["KPI"].unique()):
            sub   = rankings_df[rankings_df["KPI"] == kpi].sort_values("Rank")
            best  = sub.iloc[0]
            lower = kpi in LOWER_IS_BETTER
            label = "Most Efficient (lowest intensity)" if lower and kpi in ESG_EFFICIENCY_METRICS \
                    else "Most Efficient" if lower else "Leader"
            b_val = "N/A" if best["Value"] == 0.0 else f"{best['Value']:.3f}"
            lines.append(f"  {kpi:<50} {label}: {best['Company']} ({b_val})")
        lines.append("")
        return lines

    def _peer_env_delta(
        self, companies: dict, names: list, eff_keys: list,
    ) -> list[str]:
        a, b = names[0], names[1]
        rows = []
        for key, label in eff_keys:
            va = companies[a].get(key)
            vb = companies[b].get(key)
            if not va or not vb or va == 0.0 or vb == 0.0:
                continue
            delta     = va - vb
            delta_pct = delta / vb * 100
            lower     = key in LOWER_IS_BETTER
            winner    = a if (lower and va < vb) or (not lower and va > vb) else b
            rows.append((label.split("  ")[0], va, vb, delta, delta_pct, winner))

        if not rows:
            return []

        lines = [
            "  ESG Efficiency Delta  (A vs B — positive delta means A is higher)",
            "  " + "─" * 96,
            f"  {'Metric':<44} {'A':>10} {'B':>10} {'Δ':>10} {'Δ%':>8}  Better",
            "  " + "─" * 96,
        ]
        for label, va, vb, d, dp, winner in rows:
            lines.append(
                f"  {label:<44} {va:>10.2f} {vb:>10.2f} "
                f"{d:>+10.2f} {dp:>+7.1f}%  {winner}")
        lines.append("")
        return lines