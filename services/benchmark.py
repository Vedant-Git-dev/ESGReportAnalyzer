"""
services/benchmark.py

Converts extracted KPI values → intensity ratios (KPI / revenue).

Design principles:
- No hardcoded company data
- Checks for pre-reported intensity ratios first (more accurate, PPP-adjusted)
- Falls back to computing absolute / revenue
- Emits clear provenance for every ratio (reported vs computed)

Public API:
    build_company_profile(kpi_records, revenue_result, company_name, fy) -> CompanyProfile
    compare_profiles(profiles) -> BenchmarkReport
    print_report(report)        -> terminal output
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from services.normalizer import NormalizedKPI, normalize, NormalizationError

from core.logging_config import get_logger

logger = get_logger(__name__)

# ── KPIs benchmarked in this module ──────────────────────────────────────────
BENCHMARK_KPI_NAMES = [
    "energy_consumption",
    "scope_1_emissions",
    "scope_2_emissions",
    "total_ghg_emissions",
    "water_consumption",
    "waste_generated",
]

# Lower intensity is better for all environmental KPIs
LOWER_IS_BETTER = {
    "energy_consumption", "scope_1_emissions", "scope_2_emissions",
    "total_ghg_emissions", "water_consumption", "waste_generated",
}

# Canonical units for display
CANONICAL_UNITS = {
    "energy_consumption":  "GJ",
    "scope_1_emissions":   "tCO2e",
    "scope_2_emissions":   "tCO2e",
    "total_ghg_emissions": "tCO2e",
    "water_consumption":   "KL",
    "waste_generated":     "MT",
}

# Hard ceilings on intensity ratio (per INR Crore).
# Any computed ratio above the ceiling is an extraction/unit error and
# must be excluded from comparisons rather than shown to the user.
# Calibrated for Indian large-cap companies across sectors.
RATIO_CEILINGS: dict[str, float] = {
    "energy_consumption":  1_000,   # GJ / Cr   — typical IT ≈ 5, heavy industry ≈ 500
    "scope_1_emissions":   10,      # tCO2e / Cr
    "scope_2_emissions":   10,
    "total_ghg_emissions": 20,
    "water_consumption":   500,     # KL / Cr
    "waste_generated":     5,       # MT / Cr
}

# Pre-reported intensity ratio patterns found in BRSR/ESG reports
# Matches: "intensity per rupee of turnover\n0.000000522"
_INTENSITY_RE = re.compile(
    r"intensity\s+per\s+rupee\s+(?:of\s+)?(?:turnover|operations)[^\n]{0,120}\n(0\.\d{5,})",
    re.IGNORECASE,
)

# Category-aware intensity labels
_KPI_INTENSITY_LABELS: dict[str, list[str]] = {
    "energy_consumption": [
        "energy intensity per rupee",
        "energy intensity per rupee of turnover",
        "energy intensity per rupee turnover",
    ],
    "scope_1_emissions": [
        "scope 1 and scope 2 emission intensity per rupee",
        "scope 1 and 2 emission intensity per rupee",
        "total scope 1 and scope 2 emissions per rupee",
    ],
    "scope_2_emissions": [
        "scope 1 and scope 2 emission intensity per rupee",  # same block as scope 1
    ],
    "water_consumption": [
        "water intensity per rupee",
        "water intensity per rupee of turnover",
    ],
    "waste_generated": [
        "waste intensity per rupee",
        "waste intensity per rupee of turnover",
    ],
}


@dataclass
class RatioResult:
    kpi_name: str
    display_name: str
    normalized_value: float        # absolute in canonical unit
    normalized_unit: str
    revenue_cr: float
    ratio_value: float             # normalized_value / revenue_cr
    ratio_unit: str                # e.g. "GJ / INR_Crore"
    ratio_source: str              # "reported_ratio" | "computed"
    reported_ratio_raw: Optional[float] = None  # raw ratio from PDF if available


@dataclass
class CompanyProfile:
    company_name: str
    fiscal_year: int
    revenue_cr: float
    revenue_source: str            # "financial_statement" | "back_calculated"
    ratios: dict[str, RatioResult] = field(default_factory=dict)
    raw_kpis: dict[str, NormalizedKPI] = field(default_factory=dict)


@dataclass
class KPIComparison:
    kpi_name: str
    display_name: str
    unit: str
    entries: list[tuple[str, float, str]]  # (company_label, ratio_value, source)
    winner: str                             # company_label with best (lowest) ratio
    pct_gap: float                          # % gap between best and worst


@dataclass
class BenchmarkReport:
    comparisons: list[KPIComparison]
    company_labels: list[str]
    overall_winner: Optional[str]


_KPI_DISPLAY_NAMES = {
    "energy_consumption": "Total Energy Consumption",
    "scope_1_emissions":  "Scope 1 GHG Emissions",
    "scope_2_emissions":  "Scope 2 GHG Emissions",
    "water_consumption":  "Total Water Consumption",
    "waste_generated":    "Total Waste Generated",
}


def _find_reported_ratio(
    kpi_name: str,
    page_texts: list[str],
) -> Optional[float]:
    """
    Search page texts for a pre-reported intensity ratio (KPI / revenue).
    Returns the ratio value or None.

    Corporate BRSR reports publish these as:
      "Energy intensity per rupee of turnover (Total energy consumed / revenue from operations)
       0.000000522   0.000000546"
    The first number is the current year.
    """
    labels = _KPI_INTENSITY_LABELS.get(kpi_name, [])
    if not labels:
        return None

    for text in page_texts:
        text_lower = text.lower()
        for label in labels:
            idx = text_lower.find(label)
            if idx < 0:
                continue
            # Grab the text after the label — the ratio number should be nearby
            snippet = text[idx:idx+400]
            # Find the first very small decimal (intensity ratios are ~1e-7 to 1e-4)
            m = re.search(r"\b(0\.0{3,}\d+)\b", snippet)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
    return None


def build_company_profile(
    kpi_records: dict[str, dict],
    revenue_cr: float,
    revenue_source: str,
    company_name: str,
    fiscal_year: int,
    page_texts: Optional[list[str]] = None,
) -> CompanyProfile:
    """
    Build a CompanyProfile with KPI/revenue intensity ratios.

    Args:
        kpi_records:    {kpi_name: {"value": float, "unit": str}} — already extracted
        revenue_cr:     Revenue in INR Crore
        revenue_source: How revenue was obtained
        company_name:   Display name
        fiscal_year:    FY year
        page_texts:     Optional list of page text strings to search for pre-reported ratios

    Returns:
        CompanyProfile with .ratios populated
    """
    profile = CompanyProfile(
        company_name=company_name,
        fiscal_year=fiscal_year,
        revenue_cr=revenue_cr,
        revenue_source=revenue_source,
    )

    for kpi_name in BENCHMARK_KPI_NAMES:
        rec = kpi_records.get(kpi_name)
        if not rec or rec.get("value") is None:
            continue

        # Normalise the raw KPI value
        try:
            norm = normalize(
                kpi_name=kpi_name,
                value=float(rec["value"]),
                unit=str(rec.get("unit") or CANONICAL_UNITS.get(kpi_name, "")),
            )
        except NormalizationError as e:
            print(f"  [WARN] Normalisation failed for {kpi_name}: {e}")
            continue

        profile.raw_kpis[kpi_name] = norm

        # Check for pre-reported ratio in PDF pages
        reported_ratio = None
        if page_texts:
            reported_ratio = _find_reported_ratio(kpi_name, page_texts)

        if reported_ratio is not None:
            # Convert per-rupee ratio → per-Crore ratio (1 Crore = 1e7 rupees)
            ratio_per_cr = reported_ratio * 1e7
            source = "reported_ratio"
        else:
            # Compute from absolute / revenue
            ratio_per_cr = norm.normalized_value / revenue_cr
            source = "computed"

        # Guard: drop ratios that exceed plausibility ceiling.
        # This catches unit errors (MJ not converted → 1000× inflation)
        # and LLM hallucinations that produced absurd absolute values.
        ceiling = RATIO_CEILINGS.get(kpi_name)
        if ceiling and (ratio_per_cr <= 0 or ratio_per_cr > ceiling):
            logger.warning(
                "benchmark.ratio_exceeds_ceiling",
                kpi=kpi_name,
                ratio=ratio_per_cr,
                ceiling=ceiling,
                company=company_name,
            )
            continue  # exclude this KPI from the profile entirely

        display = _KPI_DISPLAY_NAMES.get(kpi_name, kpi_name)
        canonical = CANONICAL_UNITS.get(kpi_name, norm.normalized_unit)

        profile.ratios[kpi_name] = RatioResult(
            kpi_name=kpi_name,
            display_name=display,
            normalized_value=norm.normalized_value,
            normalized_unit=canonical,
            revenue_cr=revenue_cr,
            ratio_value=ratio_per_cr,
            ratio_unit=f"{canonical} / INR_Crore",
            ratio_source=source,
            reported_ratio_raw=reported_ratio,
        )

    return profile


def compare_profiles(profiles: list[CompanyProfile]) -> BenchmarkReport:
    """
    Compare N company profiles across all benchmark KPIs.
    Returns a BenchmarkReport with per-KPI comparisons and overall winner.
    """
    if len(profiles) < 2:
        raise ValueError("Need at least 2 profiles to compare")

    labels = [f"{p.company_name} FY{p.fiscal_year}" for p in profiles]
    comparisons: list[KPIComparison] = []

    for kpi_name in BENCHMARK_KPI_NAMES:
        display = _KPI_DISPLAY_NAMES.get(kpi_name, kpi_name)
        canonical = CANONICAL_UNITS.get(kpi_name, "")

        # Collect ratio for each company
        entries: list[tuple[str, float, str]] = []
        for label, profile in zip(labels, profiles):
            ratio = profile.ratios.get(kpi_name)
            if ratio:
                entries.append((label, ratio.ratio_value, ratio.ratio_source))

        if len(entries) < 2:
            continue

        # Lower is better for all 4 environmental KPIs
        lower = kpi_name in LOWER_IS_BETTER
        best_entry = min(entries, key=lambda e: e[1]) if lower else max(entries, key=lambda e: e[1])
        worst_entry = max(entries, key=lambda e: e[1]) if lower else min(entries, key=lambda e: e[1])

        if worst_entry[1] > 0:
            pct_gap = abs(best_entry[1] - worst_entry[1]) / worst_entry[1] * 100
        else:
            pct_gap = 0.0

        comparisons.append(KPIComparison(
            kpi_name=kpi_name,
            display_name=display,
            unit=f"{canonical}/Cr",
            entries=entries,
            winner=best_entry[0],
            pct_gap=pct_gap,
        ))

    # Overall winner: most KPI wins
    win_counts: dict[str, int] = {label: 0 for label in labels}
    for comp in comparisons:
        win_counts[comp.winner] = win_counts.get(comp.winner, 0) + 1
    overall_winner = max(win_counts, key=win_counts.get) if win_counts else None

    return BenchmarkReport(
        comparisons=comparisons,
        company_labels=labels,
        overall_winner=overall_winner,
    )


def print_report(report: BenchmarkReport) -> None:
    """
    Print benchmarking results to terminal.
    Clean, structured output — no UI framework required.
    """
    W = 80
    print("\n" + "=" * W)
    print("  ESG COMPETITIVE BENCHMARKING REPORT")
    print("  Comparison basis: KPI intensity per INR Crore revenue")
    print("=" * W)

    print(f"\n  Companies compared: {' vs '.join(report.company_labels)}")
    print(f"  Overall winner:     {report.overall_winner or 'N/A'}")

    print("\n" + "-" * W)
    print(f"  {'KPI':<32} {'Metric':<18} {'Value':>16} {'Source':<18} {'Winner'}")
    print("-" * W)

    for comp in report.comparisons:
        winner_label = comp.winner.split(" FY")[0]  # short name
        print(f"\n  ── {comp.display_name} ──")
        for company_label, ratio_val, source in comp.entries:
            is_winner = company_label == comp.winner
            marker = "★" if is_winner else " "
            # Scientific notation for tiny ratios
            val_str = f"{ratio_val:.4e}" if ratio_val < 0.001 else f"{ratio_val:.4f}"
            src_short = "reported" if "reported" in source else "computed"
            print(f"  {marker} {company_label:<35} {val_str:>16} {comp.unit:<18} [{src_short}]")
        print(f"    Gap: {comp.pct_gap:.1f}%  |  Winner: {winner_label}")

    print("\n" + "=" * W)
    print("  SUMMARY — WINS PER COMPANY")
    print("-" * W)
    win_counts: dict[str, int] = {label: 0 for label in report.company_labels}
    for comp in report.comparisons:
        win_counts[comp.winner] = win_counts.get(comp.winner, 0) + 1
    for label, wins in sorted(win_counts.items(), key=lambda x: -x[1]):
        bar = "█" * wins
        print(f"  {label:<40} {wins} wins  {bar}")
    print("=" * W + "\n")