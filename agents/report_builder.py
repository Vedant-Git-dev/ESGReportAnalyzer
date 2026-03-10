"""
agents/report_builder.py
------------------------
ReportBuilder — seven-section ESG benchmarking report.

SECTION 0 — Executive Summary  (percentile-based scoring, ESG composite grade)
SECTION 1 — Company KPI Summary (clearly typed groups, both normalisation types)
SECTION 2 — ESG Rankings        (intensity + policy; missing-data guard)
SECTION 3 — Gap Analysis        (median benchmark, quartile bands shown)
SECTION 4 — Company Comparison  (peer head-to-head or industry leaders)
SECTION 5 — Scale Context       (operational totals with size-difference warnings)
SECTION 6 — Key ESG Insights    (LLM narrative)

Key changes from v6
-------------------
• Revenue-normalised metrics shown alongside per-employee (Fix 1)
• Benchmark shown as MEDIAN with Q1/Q3 quartile band (Fix 2)
• Ranking validation: skip KPI if all values missing (Fix 3)
• Scale ratio warning at top of report if companies differ >10× (Fix 4)
• Percentile-based executive scoring (top 25% / middle 50% / bottom 25%) (Fix 5)
• ESG composite weighted scores (E=40%, S=30%, G=30%) with grades (Fix 6)
• Scale metrics in dedicated section with correct language (Fix 8)
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from config.constants import (
    LOWER_IS_BETTER, DISPLAY_GROUPS,
    OPERATIONAL_SCALE_METRICS,
    ESG_EFFICIENCY_EMP_METRICS, ESG_EFFICIENCY_REV_METRICS,
    ESG_EFFICIENCY_METRICS, ESG_POLICY_METRICS,
    SOCIAL_METRICS, SAFETY_METRICS, GOVERNANCE_METRICS,
    GOOD_THRESHOLD, BAD_THRESHOLD, RANKABLE_KPIS,
    BENCHMARK_METHOD,
)
from utils.formatting import fmt_value, fmt_gap_row, fmt_comparison_row


class ReportBuilder:

    MEDALS = ["🥇", "🥈", "🥉"]

    # ESG summary category → KPIs
    SUMMARY_GROUPS: list[tuple[str, list[str]]] = [
        ("Gender Diversity — Employees", [
            "Female_Ratio_PermanentEmp", "Female_Ratio_AllEmployees"]),
        ("Gender Diversity — Workers", [
            "Female_Ratio_PermanentWorkers", "Female_Ratio_ContractWorkers",
            "Female_Ratio_AllWorkers"]),
        ("Renewable Energy", ["RenewableEnergyShare_Pct"]),
        ("GHG Intensity", [
            "GHG_tCO2e_perEmployee", "GHG_tCO2e_perRevCr"]),
        ("Energy Efficiency", [
            "Energy_GJ_perEmployee", "Energy_GJ_perRevCr"]),
        ("Water Efficiency", [
            "Water_KL_perEmployee", "Water_KL_perRevCr"]),
        ("Waste Management", [
            "WasteRecoveryRate_Pct", "Waste_MT_perEmployee", "Waste_MT_perRevCr"]),
        ("Workplace Safety", ["Fatalities", "LTIFR", "HighConsequenceInjuries"]),
        ("Training & Development", ["Pct_Emp_HS_Training", "Pct_Wkr_HS_Training"]),
        ("Governance", ["Complaints_Filed", "Complaints_Pending"]),
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
        esg_scores: Optional[dict] = None,
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)

        names  = list(companies.keys())
        COL_W  = 30
        W      = 54 + COL_W * len(names)
        def hr(ch="━"): return ch * W

        scale_warning = benchmark_result.get("scale_warning")

        lines: list[str] = []
        lines += [
            "=" * W,
            "  ESG BENCHMARKING REPORT — SEBI BRSR XBRL  |  FY 2024-25",
            f"  Benchmark : {benchmark_result['mode_info']}",
            f"  Method    : {BENCHMARK_METHOD.upper()} (robust to outliers)",
            f"  Companies : {' | '.join(names)}",
            "=" * W, "",
        ]

        # Scale warning banner — surfaced immediately if triggered
        if scale_warning:
            lines += [
                "━" * W,
                "  ⚠  " + scale_warning.replace("\n", "\n     "),
                "━" * W, "",
            ]

        lines += self._section_summary(
            companies, gap_df, names, W, COL_W, esg_scores)
        lines += self._section_kpi_table(companies, names, W, COL_W)
        lines += self._section_rankings(rankings_df, companies, W)
        lines += self._section_gap(gap_df, benchmark_result, names, W)
        lines += self._section_comparison(
            companies, gap_df, rankings_df, names, W, COL_W,
            benchmark_result["mode_info"])
        lines += self._section_scale(companies, names, W, COL_W)
        lines += [hr(), "  SECTION 6 — KEY ESG INSIGHTS", hr(), "", insights, ""]
        lines += ["=" * W, "  END OF REPORT", "=" * W]

        report = "\n".join(lines)

        with open(os.path.join(output_dir, "esg_report.txt"), "w", encoding="utf-8") as f:
            f.write(report)
        gap_df.to_csv(os.path.join(output_dir, "gap_analysis.csv"), index=False)
        rankings_df.to_csv(os.path.join(output_dir, "kpi_rankings.csv"), index=False)
        pd.DataFrame(companies).T.rename_axis("Company").to_csv(
            os.path.join(output_dir, "normalized_metrics.csv"))
        if esg_scores:
            pd.DataFrame(esg_scores).T.rename_axis("Company").to_csv(
                os.path.join(output_dir, "esg_scores.csv"))

        print(f"  [ReportBuilder] Saved → {output_dir}/esg_report.txt")
        return report

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 0 — Executive Summary
    # ══════════════════════════════════════════════════════════════════════════

    def _section_summary(
        self, companies: dict, gap_df: pd.DataFrame,
        names: list, W: int, COL_W: int,
        esg_scores: Optional[dict] = None,
    ) -> list[str]:
        def hr(ch="━"): return ch * W

        lines: list[str] = [
            hr(), "  SECTION 0 — EXECUTIVE SUMMARY", hr(), "",
            "  ESG Performance Rating per Company",
            f"  Scoring: percentile-based (top 25% = GOOD, middle 50% = AVERAGE, "
            f"bottom 25% = BELOW AVERAGE)",
            f"  Benchmark: industry {BENCHMARK_METHOD}",
            "  ✓ GOOD  |  ~ AVERAGE  |  ✗ BELOW AVERAGE  |  — N/A", "",
        ]

        # ── Rating matrix ─────────────────────────────────────────────────────
        matrix: dict[str, dict[str, str]] = {c: {} for c in names}
        for group_name, kpis in self.SUMMARY_GROUPS:
            for company in names:
                if gap_df.empty:
                    matrix[company][group_name] = "N/A"; continue
                sub = gap_df[
                    (gap_df["Company"] == company) &
                    (gap_df["KPI"].isin(kpis))
                ]
                if sub.empty:
                    matrix[company][group_name] = "N/A"; continue
                ratings = sub["Rating"].tolist()
                if "Below Average" in ratings:
                    agg = "Below Average"
                elif all(r == "Good" for r in ratings):
                    agg = "Good"
                else:
                    agg = "Average"
                matrix[company][group_name] = agg

        ICON = {"Good": "✓", "Average": "~", "Below Average": "✗", "N/A": "—"}

        hdr = f"  {'Category':<38}" + "".join(f"{n[:COL_W-2]:<{COL_W}}" for n in names)
        lines.append(hdr)
        lines.append("  " + "─" * (38 + COL_W * len(names)))
        for group_name, _ in self.SUMMARY_GROUPS:
            row = f"  {group_name:<38}"
            for company in names:
                rating = matrix[company].get(group_name, "N/A")
                row   += f"{ICON.get(rating,'—') + ' ' + rating:<{COL_W}}"
            lines.append(row)

        # ── ESG Composite Scores ──────────────────────────────────────────────
        if esg_scores:
            # Build effective-weight note (may differ from nominal if pillars missing)
            # Use the first company's effective weights as a representative display
            first_sc    = next(iter(esg_scores.values()), {})
            eff_w       = first_sc.get("Effective_Weights", {})
            miss_p      = first_sc.get("Missing_Pillars", [])
            weight_note = "  ".join(
                f"{p[0]}={round(w*100):.0f}%"
                for p, w in eff_w.items()
            ) if eff_w else "E=40% · S=30% · G=30%"
            miss_note   = (f"  ⚠ Excluded (no data): {', '.join(miss_p)}"
                           if miss_p else "")
            lines += ["", f"  ESG Composite Scores  ({weight_note}){miss_note}"]
            lines.append("  " + "─" * (38 + COL_W * len(names)))

            for pillar in ["Environment", "Social", "Governance", "Composite"]:
                row = f"  {pillar + ' score (0–100)':<38}"
                for company in names:
                    sc = esg_scores.get(company, {})
                    v  = sc.get(pillar)
                    mp = sc.get("Missing_Pillars", [])
                    if pillar in mp:
                        row += f"{'N/A (excluded)':<{COL_W}}"
                    else:
                        row += f"{str(round(v, 1)) + '/100' if v is not None else '—':<{COL_W}}"
                lines.append(row)

            # Grade row
            row = f"  {'ESG Grade (A≥70 B≥50 C≥30 D<30)':<38}"
            for company in names:
                grade = esg_scores.get(company, {}).get("Grade", "—")
                row  += f"{grade:<{COL_W}}"
            lines.append(row)

            # Confidence row
            row = f"  {'Confidence (High≥10 Medium≥5 Low<5)':<38}"
            for company in names:
                sc   = esg_scores.get(company, {})
                conf = sc.get("Confidence", "—")
                n    = sc.get("Valid_KPI_Count", 0)
                row += f"{conf + ' (' + str(n) + ' KPIs)':<{COL_W}}"
            lines.append(row)

        lines.append("")

        # ── Per-company snapshot ──────────────────────────────────────────────
        lines.append("  Company Snapshot")
        lines.append("  " + "─" * 60)
        for company in names:
            rm   = matrix[company]
            good = [g for g, r in rm.items() if r == "Good"]
            avg  = [g for g, r in rm.items() if r == "Average"]
            bad  = [g for g, r in rm.items() if r == "Below Average"]
            na   = [g for g, r in rm.items() if r == "N/A"]

            lines.append(f"\n  {company}")
            if esg_scores and company in esg_scores:
                sc   = esg_scores[company]
                comp = sc.get("Composite")
                conf = sc.get("Confidence", "—")
                n    = sc.get("Valid_KPI_Count", 0)
                mp   = sc.get("Missing_Pillars", [])
                lines.append(
                    f"    ESG Grade: {sc.get('Grade','—')}  "
                    f"Score: {round(comp,1) if comp is not None else 'N/A'}/100  "
                    f"Confidence: {conf} ({n} KPIs)"
                )
                lines.append(
                    f"    E:{sc.get('Environment','N/A')}  "
                    f"S:{sc.get('Social','N/A')}  "
                    f"G:{sc.get('Governance','N/A') if 'Governance' not in mp else 'N/A (excluded)'}"
                )
                if mp:
                    eff_w = sc.get("Effective_Weights", {})
                    ew_str = "  ".join(f"{p[0]}={round(w*100):.0f}%" for p, w in eff_w.items())
                    lines.append(
                        f"    ⚠ Pillar(s) excluded (no data): {', '.join(mp)}"
                        f"  →  Effective weights: {ew_str}"
                    )
            if good: lines.append(f"    ✓ GOOD          : {', '.join(good)}")
            if avg:  lines.append(f"    ~ AVERAGE       : {', '.join(avg)}")
            if bad:  lines.append(f"    ✗ BELOW AVERAGE : {', '.join(bad)}")
            if na:   lines.append(f"    — DATA MISSING  : {', '.join(na)}")

            flags = [k.replace("_UNRELIABLE","") for k, v
                     in companies[company].items() if k.endswith("_UNRELIABLE") and v]
            if flags:
                lines.append(
                    f"    ⚠ Excluded (small denominator < 30): {', '.join(flags)}")

        lines.append("")
        return lines

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — KPI Summary Table
    # ══════════════════════════════════════════════════════════════════════════

    def _section_kpi_table(
        self, companies: dict, names: list, W: int, COL_W: int,
    ) -> list[str]:
        def hr(ch="━"): return ch * W
        lines = [hr(), "  SECTION 1 — COMPANY KPI SUMMARY", hr(), ""]
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
                    if flag and val is not None:
                        row += f"{'⚠ ' + str(round(val,2)) + '%':<{COL_W}}"
                    else:
                        row += fmt_value(val, COL_W, key)
                lines.append(row)

        lines.append("")
        return lines

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — ESG Rankings
    # ══════════════════════════════════════════════════════════════════════════

    def _section_rankings(
        self, rankings_df: pd.DataFrame, companies: dict, W: int,
    ) -> list[str]:
        def hr(ch="━"): return ch * W
        lines = [
            hr(),
            "  SECTION 2 — ESG EFFICIENCY & POLICY RANKINGS",
            f"  (Benchmark: industry {BENCHMARK_METHOD}  |  "
            "Operational scale metrics excluded)",
            hr(), "",
        ]

        if rankings_df.empty:
            lines.append(
                "  Rankings require ≥2 companies with overlapping rankable KPIs.")
            lines.append("")
            return lines

        TYPE_ORDER = [
            ("ESG Efficiency per Employee — lower = better ↓",
             ESG_EFFICIENCY_EMP_METRICS),
            ("ESG Efficiency per ₹Cr Revenue — lower = better ↓",
             ESG_EFFICIENCY_REV_METRICS),
            ("ESG Policy — higher = better ↑",   ESG_POLICY_METRICS),
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

                # Fix 3: check for all-missing guard (shouldn't reach here but belt+braces)
                real_vals = [r for _, r in sub.iterrows() if r["Value"] != 0.0]
                if not real_vals:
                    lines.append(f"\n  {kpi}  [No data available for ranking]")
                    continue

                direction = "↓ lowest = best" if lower else "↑ highest = best"
                q1 = sub.iloc[0].get("Q1")
                q3 = sub.iloc[0].get("Q3")
                quartile_note = ""
                if q1 is not None and q3 is not None:
                    quartile_note = (
                        f"  Q1={q1:.3f}  median  Q3={q3:.3f}")
                lines.append(f"\n  {kpi}  [{direction}]{quartile_note}")

                for _, row in sub.iterrows():
                    rank    = int(row["Rank"])
                    medal   = self.MEDALS[rank-1] if rank <= 3 else f"   #{rank}"
                    val_d   = "N/A" if row["Value"] == 0.0 else f"{row['Value']:.5f}"
                    pctile  = row.get("Percentile")
                    pct_str = f"  [pctile={pctile:.0f}%]" if pctile is not None else ""
                    n       = len(sub)
                    if lower:
                        perf = ("✓ Most Efficient"   if rank == 1 else
                                "✗ Least Efficient"  if rank == n else "")
                    else:
                        perf = ("✓ Leader"  if rank == 1 else
                                "✗ Laggard" if rank == n else "")
                    lines.append(
                        f"    {medal}  {row['Company']:<46} "
                        f"{val_d:<16} {perf}{pct_str}")
            lines.append("")

        return lines

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Gap Analysis
    # ══════════════════════════════════════════════════════════════════════════

    def _section_gap(
        self, gap_df: pd.DataFrame,
        benchmark_result: dict, names: list, W: int,
    ) -> list[str]:
        def hr(ch="━"): return ch * W
        lines = [
            hr(),
            "  SECTION 3 — GAP ANALYSIS",
            f"  Benchmark: industry {BENCHMARK_METHOD}  "
            "|  Scoring: percentile-based (top/bottom 25%)",
            "  Operational scale metrics excluded",
            hr(), "",
        ]

        if gap_df.empty:
            lines.append("  No gap data available.")
            lines.append("")
            return lines

        for company in names:
            sub  = gap_df[gap_df["Company"] == company]
            good = sub[sub["Rating"] == "Good"]
            avg  = sub[sub["Rating"] == "Average"]
            bad  = sub[sub["Rating"] == "Below Average"]

            lines.append(f"  {company}")
            lines.append("  " + "─" * 70)
            lines.append(
                f"  ✓ Good: {len(good):2d}  |  ~ Average: {len(avg):2d}  "
                f"|  ✗ Below Average: {len(bad):2d}"
            )

            if not bad.empty:
                lines.append("\n  Areas needing improvement:")
                for _, r in bad.sort_values("Gap_Pct").iterrows():
                    pctile = r.get("Percentile")
                    pct_str = f"  [pctile={pctile:.0f}%]" if pctile is not None else ""
                    lines.append(
                        fmt_gap_row(r["KPI"], r["Value"], r["Benchmark"],
                                    r["Gap"], r["Gap_Pct"], r["Status"])
                        + pct_str
                    )

            if not good.empty:
                lines.append("\n  Areas of strength:")
                for _, r in good.sort_values("Gap_Pct", ascending=False).iterrows():
                    pctile = r.get("Percentile")
                    pct_str = f"  [pctile={pctile:.0f}%]" if pctile is not None else ""
                    lines.append(
                        fmt_gap_row(r["KPI"], r["Value"], r["Benchmark"],
                                    r["Gap"], r["Gap_Pct"], r["Status"])
                        + pct_str
                    )

            if not avg.empty:
                lines.append("\n  On-target KPIs (within benchmark band):")
                for _, r in avg.iterrows():
                    val_d = "N/A" if r["Value"] == 0.0 else f"{r['Value']:.4f}"
                    ref_d = f"{r['Benchmark']:.4f}"
                    q1v   = r.get("Q1")
                    q3v   = r.get("Q3")
                    band_str = ""
                    if q1v is not None and q3v is not None:
                        band_str = f"  [Q1={q1v:.3f}  Q3={q3v:.3f}]"
                    lines.append(
                        f"    ~  {r['KPI']:<46} "
                        f"val={val_d}  median={ref_d}{band_str}")

            lines.append("")

        return lines

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Company Comparison
    # ══════════════════════════════════════════════════════════════════════════

    def _section_comparison(
        self, companies: dict, gap_df: pd.DataFrame,
        rankings_df: pd.DataFrame, names: list,
        W: int, COL_W: int, mode_info: str = "",
    ) -> list[str]:
        is_peer = "peer" in mode_info.lower() and len(names) == 2
        title   = (
            f"  SECTION 4 — PEER COMPARISON: {names[0]}  vs  {names[1]}"
            if is_peer else "  SECTION 4 — COMPANY COMPARISON"
        )
        def hr(ch="━"): return ch * W
        lines = [hr(), title, hr(), ""]
        lines.append("  " + "  |  ".join(names) + "\n")

        if is_peer:
            lines += self._peer_head_to_head(companies, names, COL_W)
        else:
            lines += self._industry_kpi_leaders(rankings_df)

        # Strengths
        lines.append("  ESG Strengths  (Good — top quartile vs benchmark)")
        lines.append("  " + "─" * 70)
        for company in names:
            sub  = gap_df[gap_df["Company"] == company] if not gap_df.empty else pd.DataFrame()
            good = sub[sub["Rating"] == "Good"] if not sub.empty else pd.DataFrame()
            lines.append(f"\n  {company}")
            if not good.empty:
                for _, r in good.sort_values("Gap_Pct", ascending=False).iterrows():
                    detail = fmt_comparison_row(
                        r["KPI"], r["Value"], r["Benchmark"],
                        r["Gap_Pct"], r["Status"])
                    lines.append(f"    ✓ {r['KPI']:<48} {detail}")
            else:
                lines.append("    — No KPIs in top quartile vs benchmark")
        lines.append("")

        # Weaknesses
        lines.append("  ESG Weaknesses  (Below Average — bottom quartile vs benchmark)")
        lines.append("  " + "─" * 70)
        any_bad = False
        for company in names:
            sub = gap_df[gap_df["Company"] == company] if not gap_df.empty else pd.DataFrame()
            bad = sub[sub["Rating"] == "Below Average"] if not sub.empty else pd.DataFrame()
            if not bad.empty:
                any_bad = True
                lines.append(f"\n  {company}")
                for _, r in bad.sort_values("Gap_Pct").iterrows():
                    action = (
                        "reduce GHG intensity" if "GHG" in r["KPI"] else
                        "improve energy efficiency" if "Energy" in r["KPI"] else
                        "reduce water intensity" if "Water" in r["KPI"] else
                        "reduce waste generation" if "Waste_MT" in r["KPI"] else
                        "increase renewables" if "Renewable" in r["KPI"] else
                        "improve waste recovery" if "Recovery" in r["KPI"] else
                        "increase female representation" if "Female" in r["KPI"] else
                        "improve safety" if r["KPI"] in SAFETY_METRICS else
                        "resolve complaints" if r["KPI"] in GOVERNANCE_METRICS else "improve"
                    )
                    detail = fmt_comparison_row(
                        r["KPI"], r["Value"], r["Benchmark"],
                        r["Gap_Pct"], r["Status"])
                    lines.append(
                        f"    ✗ {r['KPI']:<48} {detail}  → {action}")
        if not any_bad:
            lines.append("  All companies at or above industry median on every ranked KPI.")
        lines.append("")

        # Efficiency snapshot (both normalisation types)
        EFF_ROWS = [
            ("GHG_tCO2e_perEmployee",  "GHG / Employee (tCO2e)  ↓"),
            ("GHG_tCO2e_perRevCr",     "GHG / ₹Cr Revenue (tCO2e)  ↓"),
            ("Energy_GJ_perEmployee",  "Energy / Employee (GJ)  ↓"),
            ("Energy_GJ_perRevCr",     "Energy / ₹Cr Revenue (GJ)  ↓"),
            ("Water_KL_perEmployee",   "Water / Employee (KL)  ↓"),
            ("Water_KL_perRevCr",      "Water / ₹Cr Revenue (KL)  ↓"),
            ("Waste_MT_perEmployee",   "Waste / Employee (MT)  ↓"),
            ("Waste_MT_perRevCr",      "Waste / ₹Cr Revenue (MT)  ↓"),
            ("RenewableEnergyShare_Pct","Renewable Energy Share (%)  ↑"),
            ("WasteRecoveryRate_Pct",  "Waste Recovery Rate (%)  ↑"),
        ]
        has_eff = any(companies[c].get(k) for k, _ in EFF_ROWS for c in names)
        if has_eff:
            lines.append(
                "  ESG Efficiency Metrics  "
                "[these are the metrics used for ESG comparison]")
            lines.append("  " + "─" * 70)
            ELABEL_W = 46
            lines.append(
                f"  {'Metric':<{ELABEL_W}}"
                + "".join(f"{c[:COL_W-2]:<{COL_W}}" for c in names))
            lines.append("  " + "─" * (ELABEL_W + COL_W * len(names)))
            for key, label in EFF_ROWS:
                if not any(companies[c].get(key) for c in names):
                    continue
                row = f"  {label:<{ELABEL_W}}"
                for c in names:
                    row += fmt_value(companies[c].get(key), COL_W, key)
                lines.append(row)
            lines.append("")

        if is_peer:
            lines += self._peer_eff_delta(companies, names, EFF_ROWS)

        return lines

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — Operational Scale Context
    # ══════════════════════════════════════════════════════════════════════════

    def _section_scale(
        self, companies: dict, names: list, W: int, COL_W: int,
    ) -> list[str]:
        def hr(ch="━"): return ch * W
        lines = [
            hr(),
            "  SECTION 5 — OPERATIONAL SCALE & ENVIRONMENTAL IMPACT",
            "  ⚠  These are SIZE indicators, NOT ESG performance metrics.",
            "     A larger company will naturally have higher absolute totals.",
            "     Compare per-employee and per-revenue intensity metrics for ESG.",
            hr(), "",
        ]

        SCALE_ROWS = [
            ("perm_emp_total",     "Permanent Employees"),
            ("total_wkr_total",    "Total Workers"),
            ("Turnover_INR",       "Annual Turnover (INR)"),
            ("NetWorth_INR",       "Net Worth (INR)"),
            ("plants_national",    "National Plants"),
            ("TotalEnergy_GJ",     "Total Energy (GJ)  ← scales with operations"),
            ("TotalGHG_tCO2e",    "Total GHG Scope1+2 (tCO2e)  ← scales with ops"),
            ("WaterConsumption_KL","Water Consumption (KL)  ← scales with ops"),
            ("WasteGenerated_MT",  "Waste Generated (MT)  ← scales with ops"),
        ]

        LABEL_W = 50
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

        # Scale ratio narrative
        if len(names) >= 2:
            lines.append("")
            lines.append("  Scale comparison (for context):")
            for key, label in [
                ("Turnover_INR",  "Revenue"),
                ("perm_emp_total","Headcount"),
                ("TotalGHG_tCO2e","Total GHG"),
            ]:
                vals = [
                    (c, companies[c][key]) for c in names
                    if companies[c].get(key) and companies[c][key] > 0
                ]
                if len(vals) < 2:
                    continue
                hi    = max(vals, key=lambda x: x[1])
                lo    = min(vals, key=lambda x: x[1])
                ratio = hi[1] / lo[1]
                ghg_note = (
                    " — higher GHG is expected for the larger company; "
                    "see GHG/employee and GHG/₹Cr for ESG performance comparison"
                    if "GHG" in key else ""
                )
                lines.append(
                    f"    {label}: {hi[0]} is {ratio:.1f}× larger "
                    f"than {lo[0]}{ghg_note}")

        lines.append("")
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

        LABEL_W = 42
        lines = [
            "  Head-to-Head KPI Comparison  (ESG metrics only, both normalisation types)",
            "  " + "─" * 70,
        ]
        lines.append(
            f"  {'KPI':<{LABEL_W}}"
            f"{a[:COL_W-2]:<{COL_W}}"
            f"{b[:COL_W-2]:<{COL_W}}"
            f"{'Winner':<22}")
        lines.append("  " + "─" * (LABEL_W + COL_W * 2 + 22))

        a_wins = b_wins = ties = 0
        for kpi in shared:
            va    = companies[a][kpi]
            vb    = companies[b][kpi]
            lower = kpi in LOWER_IS_BETTER
            va_d  = "N/A" if va == 0.0 else f"{va:.4f}"
            vb_d  = "N/A" if vb == 0.0 else f"{vb:.4f}"
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
                f"{va_d:<{COL_W}}{vb_d:<{COL_W}}{winner:<22}")

        lines.append("  " + "─" * (LABEL_W + COL_W * 2 + 22))
        lines.append(
            f"  {'KPI Wins':<{LABEL_W}}"
            f"{str(a_wins):<{COL_W}}{str(b_wins):<{COL_W}}"
            f"{'Ties: ' + str(ties):<22}")
        lines.append("")
        return lines

    def _industry_kpi_leaders(self, rankings_df: pd.DataFrame) -> list[str]:
        if rankings_df.empty:
            return []
        lines = ["  KPI Leaders per Metric", "  " + "─" * 70]
        for kpi in sorted(rankings_df["KPI"].unique()):
            sub   = rankings_df[rankings_df["KPI"] == kpi].sort_values("Rank")
            best  = sub.iloc[0]
            lower = kpi in LOWER_IS_BETTER
            label = "Most Efficient (lowest intensity)" \
                    if lower and kpi in ESG_EFFICIENCY_METRICS else \
                    "Most Efficient" if lower else "Leader"
            b_val = "N/A" if best["Value"] == 0.0 else f"{best['Value']:.5f}"
            lines.append(f"  {kpi:<52} {label}: {best['Company']} ({b_val})")
        lines.append("")
        return lines

    def _peer_eff_delta(
        self, companies: dict, names: list, eff_keys: list,
    ) -> list[str]:
        a, b = names[0], names[1]
        rows = []
        for key, label in eff_keys:
            va = companies[a].get(key)
            vb = companies[b].get(key)
            if not va or not vb or va == 0.0 or vb == 0.0:
                continue
            delta  = va - vb
            dp     = delta / vb * 100
            lower  = key in LOWER_IS_BETTER
            winner = a if (lower and va < vb) or (not lower and va > vb) else b
            rows.append((label.split("  ")[0], va, vb, delta, dp, winner))
        if not rows:
            return []
        lines = [
            "  ESG Efficiency Delta  (positive Δ means A is higher)",
            "  " + "─" * 100,
            f"  {'Metric':<46} {'A':>10} {'B':>10} {'Δ':>10} {'Δ%':>8}  Better",
            "  " + "─" * 100,
        ]
        for label, va, vb, d, dp, winner in rows:
            lines.append(
                f"  {label:<46} {va:>10.4f} {vb:>10.4f} "
                f"{d:>+10.4f} {dp:>+7.1f}%  {winner}")
        lines.append("")
        return lines