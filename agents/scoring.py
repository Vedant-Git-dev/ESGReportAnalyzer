"""
agents/scoring.py
-----------------
ESGScoringAgent — computes a weighted composite ESG score per company.

Methodology
-----------
Three pillars: Environment (40%), Social (30%), Governance (30%).

Per KPI:
  1. Collect valid values: None AND 0.0 are both treated as missing/not-reported.
     (0.0 for safety counts like Fatalities means "not reported", not "zero incidents")
  2. If ALL companies have missing values for a KPI → skip that KPI entirely.
  3. If only 1 company has a value → percentile = 50 (neutral, no peer context).
  4. Otherwise: rank-based percentile (0–100), inverted for lower-is-better KPIs.

Pillar scoring:
  • Pillar score = mean of valid KPI percentile scores within the pillar.
  • If NO valid KPIs exist for a pillar → pillar score = None (N/A).
  • A None pillar is EXCLUDED from the composite calculation.
  • Composite = weighted average of available pillars only (weights re-normalised).

Example: if Governance is fully missing:
  Available: Environment (40%), Social (30%) → re-normalised to 57% / 43%
  Composite = (E_score × 0.571) + (S_score × 0.429)

Confidence level:
  High   : ≥ 10 valid KPI scores used across all pillars
  Medium : 5–9 valid KPI scores
  Low    : < 5 valid KPI scores
  N/A    : No valid scores at all

Missing-data rules:
  • None   = genuinely missing — skip
  • 0.0    = treated as missing for scoring (cannot distinguish "zero incidents
             reported" from "data not filed") — skip, flag as "not reported"
  • _UNRELIABLE flag → skip (small-denominator ratio)

Output per company
------------------
{
  "Environment":        float | None,
  "Social":             float | None,
  "Governance":         float | None,
  "Composite":          float | None,
  "Effective_Weights":  {"Environment": float, ...},  # re-normalised
  "Grade":              "A"/"B"/"C"/"D"/"N/A",
  "Confidence":         "High"/"Medium"/"Low"/"N/A",
  "Valid_KPI_Count":    int,
  "Missing_Pillars":    [str],
  "KPI_scores":         {kpi: float},
}

Grade scale: A ≥ 70, B ≥ 50, C ≥ 30, D < 30
"""

from __future__ import annotations

import pandas as pd

from config.constants import (
    LOWER_IS_BETTER, ESG_WEIGHTS, ESG_PILLAR_KPIS, RANKABLE_KPIS,
)


def _grade(score: float | None) -> str:
    if score is None: return "N/A"
    if score >= 70:   return "A"
    if score >= 50:   return "B"
    if score >= 30:   return "C"
    return "D"


def _confidence(valid_kpi_count: int) -> str:
    if valid_kpi_count >= 10: return "High"
    if valid_kpi_count >= 5:  return "Medium"
    if valid_kpi_count >= 1:  return "Low"
    return "N/A"


def _is_valid(val) -> bool:
    """
    A metric value is valid for scoring if:
      - it is numeric
      - it is NOT None
      - it is NOT 0.0  (0.0 = not-reported / indistinguishable from missing)
    """
    return isinstance(val, (int, float)) and val != 0.0


class ESGScoringAgent:

    def score(self, companies: dict, gap_df: pd.DataFrame) -> dict:
        names  = list(companies.keys())
        result: dict = {}

        # ── Step 1: per-KPI percentile scores (valid values only) ─────────────
        kpi_scores: dict[str, dict[str, float]] = {c: {} for c in names}

        for kpi in RANKABLE_KPIS:
            # Only include companies with valid (non-None, non-zero) values
            entries = [
                (name, companies[name][kpi])
                for name in names
                if _is_valid(companies[name].get(kpi))
                and not companies[name].get(kpi + "_UNRELIABLE")
            ]

            if not entries:
                # All missing — skip this KPI entirely, no score assigned
                continue

            n = len(entries)
            if n == 1:
                # Single company — neutral score (no peer context)
                kpi_scores[entries[0][0]][kpi] = 50.0
                continue

            vals_sorted = sorted(entries, key=lambda x: x[1])
            for i, (name, _) in enumerate(vals_sorted):
                raw_pctile   = (i / (n - 1)) * 100
                final_pctile = (100 - raw_pctile) if kpi in LOWER_IS_BETTER \
                               else raw_pctile
                kpi_scores[name][kpi] = round(final_pctile, 1)

        # ── Step 2: pillar scores — skip pillar if no valid KPIs ──────────────
        for company in names:
            pillar_scores:   dict[str, float | None] = {}
            pillar_kpi_used: dict[str, list[str]]    = {}

            for pillar, kpis in ESG_PILLAR_KPIS.items():
                valid_scores = {
                    k: kpi_scores[company][k]
                    for k in kpis
                    if k in kpi_scores[company]
                }
                if valid_scores:
                    pillar_scores[pillar]   = round(
                        sum(valid_scores.values()) / len(valid_scores), 1)
                    pillar_kpi_used[pillar] = list(valid_scores.keys())
                else:
                    pillar_scores[pillar]   = None    # pillar fully missing
                    pillar_kpi_used[pillar] = []

            # ── Step 3: re-normalised composite (exclude missing pillars) ──────
            available = {
                p: w for p, w in ESG_WEIGHTS.items()
                if pillar_scores.get(p) is not None
            }
            missing_pillars = [
                p for p in ESG_WEIGHTS if pillar_scores.get(p) is None]

            if available:
                total_w  = sum(available.values())
                # Re-normalise weights to sum to 1.0
                eff_w    = {p: round(w / total_w, 4) for p, w in available.items()}
                composite = round(
                    sum(pillar_scores[p] * eff_w[p] for p in available), 1)
            else:
                eff_w     = {}
                composite = None

            # ── Step 4: confidence level ───────────────────────────────────────
            valid_kpi_count = len(kpi_scores[company])

            result[company] = {
                **{p: pillar_scores.get(p) for p in ESG_WEIGHTS},
                "Composite":       composite,
                "Effective_Weights": eff_w,
                "Grade":           _grade(composite),
                "Confidence":      _confidence(valid_kpi_count),
                "Valid_KPI_Count": valid_kpi_count,
                "Missing_Pillars": missing_pillars,
                "KPI_scores":      kpi_scores[company],
            }

        names_str = ", ".join(
            f"{c}: {result[c]['Composite']} ({result[c]['Confidence']} conf, "
            f"{result[c]['Valid_KPI_Count']} KPIs)"
            for c in names
        )
        print(f"  [ESGScoringAgent] {names_str}")
        return result