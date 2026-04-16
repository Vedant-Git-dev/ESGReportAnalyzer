"""
services/benchmark.py

Converts extracted KPI values → intensity ratios (KPI / revenue or / FTE).

v2 changes
----------
KPI groups with distinct normalization:

  Environmental (divide by revenue INR Crore):
    scope_1_emissions, scope_2_emissions, scope_3_emissions,
    energy_consumption, waste_generated, water_consumption

  Social (divide by revenue INR Crore):
    employee_count, women_in_workforce_percentage

  Governance (NO normalization — absolute values):
    complaints_filed, complaints_pending

  Removed: total_ghg_emissions

Lower is better for all environmental metrics, complaints, and most social.
Higher is better for renewable_energy_percentage, women_in_workforce_percentage.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from services.normalizer import NormalizedKPI, normalize, NormalizationError
from core.logging_config import get_logger

logger = get_logger(__name__)

# ── KPI catalogue ─────────────────────────────────────────────────────────────

# Environmental — ratio = KPI / revenue_cr
ENVIRONMENTAL_KPI_NAMES = [
    "scope_1_emissions",
    "scope_2_emissions",
    "scope_3_emissions",
    "energy_consumption",
    "waste_generated",
    "water_consumption",
]

# Social — ratio = KPI / revenue_cr  (employee_count gives employees per Crore)
SOCIAL_KPI_NAMES = [
    "employee_count",
    "women_in_workforce_percentage",
]

# Governance — NO ratio; raw absolute value used directly
GOVERNANCE_KPI_NAMES = [
    "complaints_filed",
    "complaints_pending",
]

BENCHMARK_KPI_NAMES = (
    ENVIRONMENTAL_KPI_NAMES + SOCIAL_KPI_NAMES + GOVERNANCE_KPI_NAMES
)

# For these KPIs lower ratio = better performance
LOWER_IS_BETTER = {
    "scope_1_emissions", "scope_2_emissions", "scope_3_emissions",
    "energy_consumption", "waste_generated", "water_consumption",
    "employee_count",      # lower employees/Crore = higher productivity
    "complaints_filed",    # fewer complaints = better
    "complaints_pending",
}

# For these KPIs higher = better
HIGHER_IS_BETTER = {
    "women_in_workforce_percentage",
    "renewable_energy_percentage",
}

# Canonical display units
CANONICAL_UNITS = {
    "scope_1_emissions":             "tCO2e",
    "scope_2_emissions":             "tCO2e",
    "scope_3_emissions":             "tCO2e",
    "energy_consumption":            "GJ",
    "waste_generated":               "MT",
    "water_consumption":             "KL",
    "employee_count":                "count",
    "women_in_workforce_percentage": "%",
    "complaints_filed":              "count",
    "complaints_pending":            "count",
}

# Ratio unit labels (shown in UI)
RATIO_UNIT_LABELS = {
    "scope_1_emissions":             "tCO2e / INR_Crore",
    "scope_2_emissions":             "tCO2e / INR_Crore",
    "scope_3_emissions":             "tCO2e / INR_Crore",
    "energy_consumption":            "GJ / INR_Crore",
    "waste_generated":               "MT / INR_Crore",
    "water_consumption":             "KL / INR_Crore",
    "employee_count":                "count / INR_Crore",
    "women_in_workforce_percentage": "%",          # no ratio
    "complaints_filed":              "count",      # no ratio
    "complaints_pending":            "count",      # no ratio
}

# Hard ceilings on ratio values — anything above is likely an extraction error
RATIO_CEILINGS: dict[str, float] = {
    "scope_1_emissions":   10,
    "scope_2_emissions":   10,
    "scope_3_emissions":   50,
    "energy_consumption":  1_000,
    "waste_generated":     5,
    "water_consumption":   500,
    "employee_count":      5_000,     # employees per Crore
    "women_in_workforce_percentage": 100,
    "complaints_filed":    1_000_000,
    "complaints_pending":  100_000,
}

# Implied revenue sanity (for back-calculation validation)
_IMPLIED_REV_MIN_CR = 1_000
_IMPLIED_REV_MAX_CR = 9_000_000

# BRSR intensity label search (for pre-reported ratios)
_KPI_INTENSITY_LABELS: dict[str, list[str]] = {
    "energy_consumption": [
        "energy intensity per rupee of turnover",
        "energy intensity per rupee of turnover",
    ],
    "water_consumption": [
        "water intensity per rupee of turnover",
        "water intensity per rupee",
    ],
    "waste_generated": [
        "waste intensity per rupee of turnover",
        "waste intensity per rupee",
    ],
}

_DISPLAY_NAMES = {
    "scope_1_emissions":             "Scope 1 GHG Emissions",
    "scope_2_emissions":             "Scope 2 GHG Emissions",
    "scope_3_emissions":             "Scope 3 GHG Emissions",
    "energy_consumption":            "Total Energy Consumption",
    "waste_generated":               "Total Waste Generated",
    "water_consumption":             "Total Water Consumption",
    "employee_count":                "Total Employees",
    "women_in_workforce_percentage": "Women in Workforce (%)",
    "complaints_filed":              "Complaints Filed",
    "complaints_pending":            "Complaints Pending",
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RatioResult:
    kpi_name:           str
    display_name:       str
    group:              str          # "environmental" | "social" | "governance"
    normalized_value:   float        # absolute in canonical unit
    normalized_unit:    str
    revenue_cr:         float
    ratio_value:        float        # normalized_value / revenue_cr  OR absolute (governance)
    ratio_unit:         str
    ratio_source:       str          # "reported_ratio" | "computed" | "absolute"
    reported_ratio_raw: Optional[float] = None


@dataclass
class CompanyProfile:
    company_name:   str
    fiscal_year:    int
    revenue_cr:     float
    revenue_source: str
    ratios:         dict[str, RatioResult] = field(default_factory=dict)
    raw_kpis:       dict[str, NormalizedKPI] = field(default_factory=dict)


@dataclass
class KPIComparison:
    kpi_name:    str
    display_name: str
    group:       str
    unit:        str
    entries:     list[tuple[str, float, str]]  # (company_label, ratio_value, source)
    winner:      str
    pct_gap:     float


@dataclass
class BenchmarkReport:
    comparisons:    list[KPIComparison]
    company_labels: list[str]
    overall_winner: Optional[str]


# ── Pre-reported ratio search ─────────────────────────────────────────────────

def _find_reported_ratio(
    kpi_name: str,
    page_texts: list[str],
    abs_normalized_value: float,
) -> Optional[float]:
    """
    Search for pre-reported intensity ratio (per INR Crore) in PDF text.
    Only applies to environmental KPIs that appear in BRSR intensity tables.
    Returns per-Crore ratio or None.
    """
    labels = _KPI_INTENSITY_LABELS.get(kpi_name, [])
    if not labels:
        return None

    _INTENSITY_VALUE_RE = re.compile(r"\b(0\.0{6,}\d+)\b")

    for text in page_texts:
        text_lower = text.lower()
        for label in labels:
            idx = text_lower.find(label)
            if idx < 0:
                continue
            snippet = text[idx: idx + 400]
            m = _INTENSITY_VALUE_RE.search(snippet)
            if not m:
                continue
            try:
                ratio_per_rupee = float(m.group(1))
            except ValueError:
                continue
            if ratio_per_rupee <= 0:
                continue
            ratio_per_cr = ratio_per_rupee * 1e7
            if abs_normalized_value > 0:
                implied_rev = abs_normalized_value / ratio_per_cr
                if not (_IMPLIED_REV_MIN_CR <= implied_rev <= _IMPLIED_REV_MAX_CR):
                    continue
            return ratio_per_cr

    return None


# ── KPI group classifier ──────────────────────────────────────────────────────

def _kpi_group(kpi_name: str) -> str:
    if kpi_name in GOVERNANCE_KPI_NAMES:
        return "governance"
    if kpi_name in SOCIAL_KPI_NAMES:
        return "social"
    return "environmental"


# ── Profile builder ───────────────────────────────────────────────────────────

def build_company_profile(
    kpi_records:    dict[str, dict],
    revenue_cr:     float,
    revenue_source: str,
    company_name:   str,
    fiscal_year:    int,
    page_texts:     Optional[list[str]] = None,
) -> CompanyProfile:
    """
    Build a CompanyProfile with per-KPI ratios.

    Ratio logic by group
    --------------------
    Environmental / Social:
        ratio = normalized_value / revenue_cr
        (pre-reported ratio used when found in PDF pages)

    Governance (complaints_filed, complaints_pending):
        ratio = normalized_value  (absolute count — no denominator)
        ratio_source = "absolute"

    Women / Renewable percentage:
        ratio = normalized_value  (already a %)
        ratio_source = "absolute" (percentage is self-normalizing)
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

        group = _kpi_group(kpi_name)

        try:
            norm = normalize(
                kpi_name=kpi_name,
                value=float(rec["value"]),
                unit=str(rec.get("unit") or CANONICAL_UNITS.get(kpi_name, "")),
            )
        except NormalizationError as e:
            logger.warning("benchmark.normalisation_failed", kpi=kpi_name, error=str(e))
            continue

        profile.raw_kpis[kpi_name] = norm

        # ── Ratio calculation ─────────────────────────────────────────────────
        if group == "governance":
            # Complaints: use absolute count, no revenue denominator
            ratio_val    = norm.normalized_value
            ratio_source = "absolute"
            ratio_unit   = RATIO_UNIT_LABELS.get(kpi_name, "count")

        elif kpi_name in ("women_in_workforce_percentage", "renewable_energy_percentage"):
            # Percentages: already normalized, no denominator
            ratio_val    = norm.normalized_value
            ratio_source = "absolute"
            ratio_unit   = "%"

        else:
            # Environmental / Social: ratio = value / revenue
            reported = None
            if page_texts:
                reported = _find_reported_ratio(
                    kpi_name=kpi_name,
                    page_texts=page_texts,
                    abs_normalized_value=norm.normalized_value,
                )

            if reported is not None:
                ratio_val    = reported
                ratio_source = "reported_ratio"
            else:
                if revenue_cr <= 0:
                    logger.warning("benchmark.zero_revenue", kpi=kpi_name)
                    continue
                ratio_val    = norm.normalized_value / revenue_cr
                ratio_source = "computed"

            ratio_unit = RATIO_UNIT_LABELS.get(kpi_name, "")

        # ── Ceiling guard ─────────────────────────────────────────────────────
        ceiling = RATIO_CEILINGS.get(kpi_name)
        if ceiling and ratio_val > ceiling:
            logger.warning(
                "benchmark.ratio_exceeds_ceiling",
                kpi=kpi_name, ratio=ratio_val, ceiling=ceiling,
                company=company_name,
            )
            continue

        display = _DISPLAY_NAMES.get(kpi_name, kpi_name)

        profile.ratios[kpi_name] = RatioResult(
            kpi_name=kpi_name,
            display_name=display,
            group=group,
            normalized_value=norm.normalized_value,
            normalized_unit=CANONICAL_UNITS.get(kpi_name, norm.normalized_unit),
            revenue_cr=revenue_cr,
            ratio_value=ratio_val,
            ratio_unit=ratio_unit,
            ratio_source=ratio_source,
            reported_ratio_raw=(
                reported / 1e7 if ratio_source == "reported_ratio" else None
            ),
        )

    return profile


# ── Comparison ────────────────────────────────────────────────────────────────

def compare_profiles(profiles: list[CompanyProfile]) -> BenchmarkReport:
    """
    Compare N company profiles across all benchmark KPIs.
    """
    if len(profiles) < 2:
        raise ValueError("Need at least 2 profiles to compare")

    labels = [f"{p.company_name} FY{p.fiscal_year}" for p in profiles]
    comparisons: list[KPIComparison] = []

    for kpi_name in BENCHMARK_KPI_NAMES:
        display = _DISPLAY_NAMES.get(kpi_name, kpi_name)
        group   = _kpi_group(kpi_name)

        entries: list[tuple[str, float, str]] = []
        for label, profile in zip(labels, profiles):
            ratio = profile.ratios.get(kpi_name)
            if ratio:
                entries.append((label, ratio.ratio_value, ratio.ratio_source))

        if len(entries) < 2:
            continue

        lower = kpi_name in LOWER_IS_BETTER
        best  = min(entries, key=lambda e: e[1]) if lower else max(entries, key=lambda e: e[1])
        worst = max(entries, key=lambda e: e[1]) if lower else min(entries, key=lambda e: e[1])

        pct_gap = (
            abs(best[1] - worst[1]) / worst[1] * 100
            if worst[1] > 0 else 0.0
        )

        unit = RATIO_UNIT_LABELS.get(kpi_name, CANONICAL_UNITS.get(kpi_name, ""))

        comparisons.append(KPIComparison(
            kpi_name=kpi_name,
            display_name=display,
            group=group,
            unit=unit,
            entries=entries,
            winner=best[0],
            pct_gap=pct_gap,
        ))

    win_counts: dict[str, int] = {label: 0 for label in labels}
    for comp in comparisons:
        win_counts[comp.winner] = win_counts.get(comp.winner, 0) + 1
    overall_winner = max(win_counts, key=win_counts.get) if win_counts else None

    return BenchmarkReport(
        comparisons=comparisons,
        company_labels=labels,
        overall_winner=overall_winner,
    )


# ── Terminal output ───────────────────────────────────────────────────────────

def print_report(report: BenchmarkReport) -> None:
    W = 80
    print("\n" + "=" * W)
    print("  ESG COMPETITIVE BENCHMARKING REPORT")
    print("  Groups: Environmental | Social | Governance")
    print("=" * W)
    print(f"\n  Companies compared: {' vs '.join(report.company_labels)}")
    print(f"  Overall winner:     {report.overall_winner or 'N/A'}")

    current_group = None
    for comp in report.comparisons:
        if comp.group != current_group:
            current_group = comp.group
            print(f"\n  {'─'*70}")
            print(f"  {current_group.upper()}")
            print(f"  {'─'*70}")

        print(f"\n  ── {comp.display_name} ──")
        for company_label, ratio_val, source in comp.entries:
            is_winner = company_label == comp.winner
            marker    = "★" if is_winner else " "
            val_str   = (
                f"{ratio_val:.4e}" if 0 < ratio_val < 0.001
                else f"{ratio_val:.4f}" if ratio_val < 1
                else f"{ratio_val:,.2f}"
            )
            src_short = (
                "reported" if "reported" in source else
                "absolute" if source == "absolute" else "computed"
            )
            print(f"  {marker} {company_label:<35} {val_str:>16} {comp.unit:<25} [{src_short}]")
        print(f"    Gap: {comp.pct_gap:.1f}%  |  Winner: {comp.winner.split(' FY')[0]}")

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