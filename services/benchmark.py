"""
services/benchmark.py

Converts extracted KPI values → intensity ratios (KPI / revenue).

Design principles:
- No hardcoded company data
- Checks for pre-reported intensity ratios first (more accurate)
- Falls back to computing absolute / revenue
- Emits clear provenance for every ratio (reported vs computed)

Bug fixes in this version
--------------------------
Bug 1 — _find_reported_ratio returns wrong value (ratio=580 for scope_1)
  Root cause (a): The BRSR intensity search regex required only 3+ zeros
  (pattern 0.0{3,}\\d+), so values like 0.0000580 (only 4 zeros) matched.
  Real BRSR intensity ratios have 6-9 zeros (1e-6 to 1e-9 per rupee).
  Fix: tighten to 6+ zeros (pattern 0.0{6,}\\d+).

  Root cause (b): Even when the correct 6-zero value is found, BRSR uses a
  *combined* scope1+scope2 label for GHG intensity, so the same value gets
  applied to BOTH scope_1 and scope_2 individually — producing a 2× overcount.
  Fix: validate the implied revenue (abs_value / (ratio * 1e7)) is within the
  plausible corporate revenue range [1_000, 9_000_000] Cr. If not, discard
  the reported ratio and fall back to computed (absolute / revenue).

Bug 2 — waste/water reported_ratio also affected
  Same root-cause as Bug 1 — the regex matched any 3+ zero decimal.
  Both fixes apply equally to all KPI categories.

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

# ── Plausible corporate revenue range (INR Crore) ─────────────────────────────
# Used to validate back-computed revenue from reported intensity ratios.
# Too small → the ratio was picked up from a subsection (CSR, standalone)
# Too large → the ratio is for an entirely different metric
_IMPLIED_REV_MIN_CR = 1_000       # ₹1,000 Cr minimum
_IMPLIED_REV_MAX_CR = 9_000_000   # ₹90 lakh Cr maximum

# Category-aware intensity labels searched in BRSR PDF pages
_KPI_INTENSITY_LABELS: dict[str, list[str]] = {
    "energy_consumption": [
        "energy intensity per rupee",
        "energy intensity per rupee of turnover",
        "energy intensity per rupee turnover",
    ],
    # Note: BRSR only publishes a *combined* scope1+scope2 intensity.
    # We deliberately do NOT list scope_1 or scope_2 here so that the
    # combined label never gets applied to just one component.
    # Both are computed from absolute / revenue instead.
    "total_ghg_emissions": [
        "scope 1 and scope 2 emission intensity per rupee",
        "scope 1 and 2 emission intensity per rupee",
        "total scope 1 and scope 2 emissions per rupee",
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
    reported_ratio_raw: Optional[float] = None  # raw per-rupee ratio from PDF


@dataclass
class CompanyProfile:
    company_name: str
    fiscal_year: int
    revenue_cr: float
    revenue_source: str
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
    "energy_consumption":  "Total Energy Consumption",
    "scope_1_emissions":   "Scope 1 GHG Emissions",
    "scope_2_emissions":   "Scope 2 GHG Emissions",
    "total_ghg_emissions": "Total GHG Emissions (Scope 1+2)",
    "water_consumption":   "Total Water Consumption",
    "waste_generated":     "Total Waste Generated",
}


def _find_reported_ratio(
    kpi_name: str,
    page_texts: list[str],
    abs_normalized_value: float,
) -> Optional[float]:
    """
    Search page texts for a pre-reported intensity ratio (KPI per INR Crore).

    Returns the per-Crore ratio, or None if not found / implausible.

    Bug 1 fix (a) — tighter regex: require 6+ zeros after the decimal point.
    Real BRSR per-rupee intensities are in the range 1e-6 to 1e-9, so they
    always have 6+ leading zeros:
        energy:  ~5e-7  -> 0.000000522   (6 zeros)
        GHG:     ~1e-8  -> 0.000000015   (7 zeros)
        water:   ~3e-7  -> 0.00000030    (6 zeros)
        waste:   ~2e-9  -> 0.0000000023  (8 zeros)
    Values with only 4-5 zeros (e.g. 0.0000580) come from other sections
    of the BRSR and must be excluded.

    Bug 1 fix (b) — implied-revenue sanity check: after finding a ratio,
    compute the implied company revenue:
        implied_rev = abs_value / (ratio_per_rupee * 1e7)
    If this falls outside [1_000, 9_000_000] Crore, the ratio is wrong
    and we return None so the caller falls back to computed ratio.

    Args:
        kpi_name:             Name of the KPI (used to look up label list).
        page_texts:           List of page text strings from the PDF.
        abs_normalized_value: The already-normalised absolute KPI value
                              (tCO2e, GJ, KL, MT etc.). Used for validation.

    Returns:
        per-Crore ratio (float) or None.
    """
    labels = _KPI_INTENSITY_LABELS.get(kpi_name, [])
    if not labels:
        return None

    # Bug 1 fix (a): 6+ zeros = per-rupee intensity in the right range
    _INTENSITY_VALUE_RE = re.compile(r"\b(0\.0{6,}\d+)\b")

    for text in page_texts:
        text_lower = text.lower()
        for label in labels:
            idx = text_lower.find(label)
            if idx < 0:
                continue
            # Grab the text after the label — the ratio should appear nearby
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

            # Convert per-rupee → per-Crore (1 Crore = 1e7 rupees)
            ratio_per_cr = ratio_per_rupee * 1e7

            # Bug 1 fix (b): validate implied revenue
            if abs_normalized_value > 0:
                implied_rev = abs_normalized_value / ratio_per_cr
                if not (_IMPLIED_REV_MIN_CR <= implied_rev <= _IMPLIED_REV_MAX_CR):
                    logger.debug(
                        "benchmark.reported_ratio_implausible",
                        kpi=kpi_name,
                        ratio_per_rupee=ratio_per_rupee,
                        ratio_per_cr=ratio_per_cr,
                        abs_value=abs_normalized_value,
                        implied_rev=implied_rev,
                    )
                    continue  # discard — ratio leads to nonsensical revenue

            logger.debug(
                "benchmark.reported_ratio_accepted",
                kpi=kpi_name,
                ratio_per_rupee=ratio_per_rupee,
                ratio_per_cr=ratio_per_cr,
            )
            return ratio_per_cr

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
        page_texts:     Optional list of page text strings to search for
                        pre-reported intensity ratios

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

        # Normalise the raw KPI value to canonical unit
        try:
            norm = normalize(
                kpi_name=kpi_name,
                value=float(rec["value"]),
                unit=str(rec.get("unit") or CANONICAL_UNITS.get(kpi_name, "")),
            )
        except NormalizationError as e:
            logger.warning("benchmark.normalisation_failed", kpi=kpi_name, error=str(e))
            print(f"  [WARN] Normalisation failed for {kpi_name}: {e}")
            continue

        profile.raw_kpis[kpi_name] = norm

        # Try to find a pre-reported intensity ratio in PDF pages.
        # Pass the absolute value so the validator can sanity-check.
        reported_ratio_per_cr: Optional[float] = None
        if page_texts:
            reported_ratio_per_cr = _find_reported_ratio(
                kpi_name=kpi_name,
                page_texts=page_texts,
                abs_normalized_value=norm.normalized_value,
            )

        if reported_ratio_per_cr is not None:
            ratio_per_cr = reported_ratio_per_cr
            source = "reported_ratio"
        else:
            # Compute from absolute / revenue
            ratio_per_cr = norm.normalized_value / revenue_cr
            source = "computed"

        # Guard: drop ratios that exceed plausibility ceiling.
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
            reported_ratio_raw=(
                reported_ratio_per_cr / 1e7
                if reported_ratio_per_cr is not None else None
            ),
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

        lower = kpi_name in LOWER_IS_BETTER
        best_entry  = min(entries, key=lambda e: e[1]) if lower else max(entries, key=lambda e: e[1])
        worst_entry = max(entries, key=lambda e: e[1]) if lower else min(entries, key=lambda e: e[1])

        pct_gap = (
            abs(best_entry[1] - worst_entry[1]) / worst_entry[1] * 100
            if worst_entry[1] > 0 else 0.0
        )

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
    """Print benchmarking results to terminal."""
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
        winner_label = comp.winner.split(" FY")[0]
        print(f"\n  ── {comp.display_name} ──")
        for company_label, ratio_val, source in comp.entries:
            is_winner = company_label == comp.winner
            marker = "★" if is_winner else " "
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