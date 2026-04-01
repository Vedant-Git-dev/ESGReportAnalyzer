"""
services/summary_generator.py

LLM-powered narrative summary comparing two or more companies' ESG performance.

Design principles:
- Uses existing LLMService — no new HTTP client
- Structured prompt with factual data only — no hallucination possible
  because all numbers are pre-computed by benchmark.py before the LLM sees them
- Falls back to a rule-based template summary if LLM is unavailable
- Output is plain text for terminal display
"""
from __future__ import annotations

from typing import Optional

from services.benchmark import BenchmarkReport, CompanyProfile, RatioResult, CANONICAL_UNITS
from services.llm_service import LLMService
from core.config import get_settings
from core.logging_config import get_logger

logger = get_logger(__name__)

_MAX_PROMPT_CHARS = 8_000


def _build_data_block(
    profiles: list[CompanyProfile],
    report: BenchmarkReport,
) -> str:
    """
    Serialize benchmarking data into a compact, LLM-readable text block.
    The LLM only narrates — it never invents numbers.
    """
    lines: list[str] = []
    lines.append("=== COMPANY PROFILES ===")

    for p in profiles:
        lines.append(f"\n{p.company_name} (FY{p.fiscal_year})")
        lines.append(f"  Revenue: INR {p.revenue_cr:,.0f} Crore [{p.revenue_source}]")
        for kpi_name, ratio in p.ratios.items():
            val_abs = f"{ratio.normalized_value:,.2f} {ratio.normalized_unit}"
            if ratio.ratio_value < 0.001:
                ratio_str = f"{ratio.ratio_value:.4e} {ratio.ratio_unit}"
            else:
                ratio_str = f"{ratio.ratio_value:.4f} {ratio.ratio_unit}"
            src = "reported in PDF" if "reported" in ratio.ratio_source else "computed"
            lines.append(
                f"  {ratio.display_name}: "
                f"absolute={val_abs} | intensity={ratio_str} [{src}]"
            )

    lines.append("\n=== KPI COMPARISON ===")
    for comp in report.comparisons:
        lines.append(f"\n{comp.display_name} (lower = better, unit: {comp.unit})")
        for label, val, source in sorted(comp.entries, key=lambda e: e[1]):
            val_str = f"{val:.4e}" if val < 0.001 else f"{val:.4f}"
            lines.append(f"  {label}: {val_str}")
        lines.append(f"  Winner: {comp.winner} (gap: {comp.pct_gap:.1f}%)")

    return "\n".join(lines)


def _rule_based_summary(
    profiles: list[CompanyProfile],
    report: BenchmarkReport,
) -> str:
    """
    Deterministic summary when LLM is unavailable.
    Generates readable English from the benchmark data.
    """
    labels = [f"{p.company_name} (FY{p.fiscal_year})" for p in profiles]
    lines: list[str] = []

    lines.append(f"ESG Benchmarking Summary: {' vs '.join(labels)}")
    lines.append("=" * 70)
    lines.append(
        "\nThis comparison uses intensity ratios (KPI per INR Crore revenue) "
        "to ensure fair comparison across companies of different sizes.\n"
    )

    for comp in report.comparisons:
        winner_short = comp.winner.split(" FY")[0]
        loser_entry  = max(comp.entries, key=lambda e: e[1])
        winner_entry = min(comp.entries, key=lambda e: e[1])
        lines.append(f"▸ {comp.display_name}:")
        lines.append(
            f"  {winner_short} leads with {winner_entry[1]:.4e} {comp.unit} "
            f"({comp.pct_gap:.1f}% better than {loser_entry[0].split(' FY')[0]})."
        )
        for label, val, source in comp.entries:
            src = "as reported" if "reported" in source else "computed from absolutes"
            lines.append(f"    {label}: {val:.4e} {comp.unit} [{src}]")
        lines.append("")

    if report.overall_winner:
        winner_short = report.overall_winner.split(" FY")[0]
        wins = sum(1 for c in report.comparisons if c.winner == report.overall_winner)
        lines.append(
            f"Overall: {winner_short} has lower ESG intensity on {wins} of "
            f"{len(report.comparisons)} metrics benchmarked."
        )

    return "\n".join(lines)


def generate_summary(
    profiles: list[CompanyProfile],
    report: BenchmarkReport,
    llm: Optional[LLMService] = None,
) -> str:
    """
    Generate a narrative ESG comparison summary.

    Uses LLM if available and API key is configured.
    Falls back to rule-based summary otherwise.

    Args:
        profiles: List of CompanyProfile (from benchmark.build_company_profile)
        report:   BenchmarkReport (from benchmark.compare_profiles)
        llm:      Optional LLMService instance (created if not passed)

    Returns:
        Plain text summary string for terminal display
    """
    settings = get_settings()
    if llm is None:
        llm = LLMService()

    # Build factual data block — LLM only narrates, never invents
    data_block = _build_data_block(profiles, report)

    if not settings.llm_api_key:
        logger.info("summary_generator.llm_unavailable, using rule_based")
        return _rule_based_summary(profiles, report)

    company_names = " and ".join(p.company_name for p in profiles)
    fys = " / ".join(str(p.fiscal_year) for p in profiles)

    system_prompt = (
        "You are a senior ESG analyst writing a concise benchmarking report. "
        "You have been given pre-computed, verified data. "
        "Your job is to narrate the comparison clearly and fairly. "
        "Do not invent any numbers — only use what is given. "
        "Write in professional English. Be concise (4-6 paragraphs). "
        "Structure: overview → per-KPI findings → overall conclusion."
    )

    user_prompt = f"""Write a professional ESG benchmarking summary comparing {company_names} (FY {fys}).

Use ONLY the data below. Do not add external knowledge, estimates, or context not in the data.
Every claim must be supported by a specific number from the data.

{data_block[:_MAX_PROMPT_CHARS]}

Write the summary now. Plain text only — no markdown headers, no bullet points.
Start directly with the comparison narrative."""

    logger.info("summary_generator.llm_call", companies=company_names)
    raw = llm._call(system_prompt, user_prompt)

    if raw and len(raw.strip()) > 100:
        return raw.strip()

    # LLM failed — fall back
    logger.warning("summary_generator.llm_failed, using rule_based")
    return _rule_based_summary(profiles, report)