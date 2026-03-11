"""
agents/data_validator.py
------------------------
DataValidationAgent — post-normalisation sanity checks.

Runs after NormalizationAgent, before Benchmarking.

Checks performed
----------------
1. Negative environmental values  — physically impossible
2. Revenue scale sanity           — detect if Turnover_INR looks like crore/lakh scale
3. Unrealistic intensity metrics  — flag if perRevCr or perEmployee exceeds ceiling
4. Extreme outliers               — >10× the cross-company median (needs ≥2 companies)

Revenue heuristic
-----------------
A well-formed Turnover_INR for an Indian listed company should be > 1e8 (₹1 Cr minimum).
If the stored value is < 1e6, it is almost certainly in crores or lakhs.
We attempt to auto-correct by testing candidate scales:
  • value × 1e7  (if value looks like crores)
  • value × 1e5  (if value looks like lakhs)
  • value × 1e6  (if value looks like millions)
  • value × 1e9  (if value looks like billions)
The corrected scale that produces a plausible rev_cr is used.

Critical rule: corrections are flagged as DATA_QUALITY warnings and
stored in the audit. They do NOT silently change the metric — the
caller decides whether to apply them.

Output
------
{
  "warnings": [
    {
      "company":   str,
      "metric":    str,
      "value":     float,
      "check":     str,       # human-readable check name
      "severity":  "ERROR" | "WARNING",
      "detail":    str,
      "suggested_correction": float | None,
    }
  ],
  "revenue_corrections": {
    company: {
      "original_INR":    float,
      "corrected_INR":   float,
      "factor_applied":  float,
      "unit_assumed":    str,     # e.g. "crore", "lakh"
      "confidence":      str,     # "high" | "medium" | "low"
    }
  },
  "n_errors":   int,
  "n_warnings": int,
}
"""

from __future__ import annotations

import statistics
from typing import Optional

# ── Thresholds ────────────────────────────────────────────────────────────────

# Absolute minimum Turnover_INR for a listed Indian company (₹1 Crore = 1e7 INR)
REVENUE_FLOOR_INR: float = 1e7          # 1 crore

# If Turnover_INR > this, it's already in INR and no correction needed
REVENUE_LOOKS_FINE_ABOVE: float = 1e8  # 10 crores

# Intensity ceilings — any value above these is a normalization error
INTENSITY_CEILINGS: dict[str, float] = {
    "Energy_GJ_perRevCr":        100_000.0,   # 100k GJ per ₹Cr is physically absurd
    "GHG_tCO2e_perRevCr":         50_000.0,
    "Water_KL_perRevCr":         200_000.0,
    "Waste_MT_perRevCr":          20_000.0,
    "Energy_GJ_perEmployee":     500_000.0,
    "GHG_tCO2e_perEmployee":     100_000.0,
    "Water_KL_perEmployee":      200_000.0,
    "Waste_MT_perEmployee":       50_000.0,
}

# Metrics that cannot be negative
NON_NEGATIVE_METRICS: set[str] = {
    "TotalEnergy_GJ", "RenewableEnergy_GJ", "NonRenewableEnergy_GJ",
    "TotalGHG_tCO2e", "Scope1_tCO2e", "Scope2_tCO2e",
    "WaterWithdrawal_KL", "WaterConsumption_KL", "WasteGenerated_MT",
    "WasteRecovered_MT", "WasteDisposed_MT",
    "Energy_GJ_perEmployee", "GHG_tCO2e_perEmployee",
    "Water_KL_perEmployee", "Waste_MT_perEmployee",
    "Energy_GJ_perRevCr", "GHG_tCO2e_perRevCr",
    "Water_KL_perRevCr", "Waste_MT_perRevCr",
    "Turnover_INR", "NetWorth_INR",
}

# Revenue scale candidates: (factor, label, confidence)
# Ordered from most to least common for Indian XBRL filers
REVENUE_SCALE_CANDIDATES: list[tuple[float, str, str]] = [
    (1e7,  "crore",   "high"),
    (1e5,  "lakh",    "high"),
    (1e6,  "million", "medium"),
    (1e9,  "billion", "medium"),
    (1e3,  "thousand","low"),
]


class DataValidationAgent:
    """
    Validates normalised metrics for a set of companies.
    Call validate(companies_dict) after NormalizationAgent.normalize().
    """

    def validate(self, companies: dict) -> dict:
        """
        Parameters
        ----------
        companies : {company_name: metrics_dict}  (post-normalisation)

        Returns
        -------
        validation_result dict (see module docstring)
        """
        warnings:             list[dict]  = []
        revenue_corrections:  dict        = {}

        for company, metrics in companies.items():

            # 1. Negative environmental values
            for metric in NON_NEGATIVE_METRICS:
                val = metrics.get(metric)
                if val is not None and isinstance(val, (int, float)) and val < 0:
                    warnings.append(dict(
                        company=company, metric=metric, value=val,
                        check="negative_environmental_value",
                        severity="ERROR",
                        detail=(
                            f"{metric} = {val:.4f} — environmental/financial metrics "
                            f"cannot be negative. Likely data entry error."),
                        suggested_correction=None,
                    ))

            # 2. Revenue scale sanity
            rev = metrics.get("Turnover_INR")
            corr = self._check_revenue_scale(company, rev)
            if corr:
                revenue_corrections[company] = corr
                warnings.append(dict(
                    company=company, metric="Turnover_INR", value=rev,
                    check="revenue_scale_normalization",
                    severity="ERROR",
                    detail=(
                        f"Turnover_INR = {rev:,.0f} — too small for an Indian listed "
                        f"company. Likely reported in {corr['unit_assumed']}. "
                        f"Suggested correction: {corr['corrected_INR']:,.0f} INR "
                        f"(× {corr['factor_applied']:.0e}). "
                        f"All per-revenue intensities may be inflated."
                    ),
                    suggested_correction=corr["corrected_INR"],
                ))

            # 3. Unrealistic intensity metrics
            for metric, ceiling in INTENSITY_CEILINGS.items():
                val = metrics.get(metric)
                if val is None or not isinstance(val, (int, float)):
                    continue
                if val > ceiling:
                    # Attempt to infer the revenue error factor
                    hint = ""
                    if "perRevCr" in metric and rev:
                        expected_order = val / ceiling
                        hint = (f" This is {expected_order:,.0f}× above ceiling — "
                                f"revenue may be misreported by a factor of ~{expected_order:.0e}.")
                    warnings.append(dict(
                        company=company, metric=metric, value=val,
                        check="unrealistic_intensity",
                        severity="ERROR",
                        detail=(
                            f"{metric} = {val:,.2f} exceeds maximum plausible value "
                            f"of {ceiling:,.0f}.{hint} "
                            f"Verify source units and revenue scale."),
                        suggested_correction=None,
                    ))

        # 4. Extreme outliers (cross-company, requires ≥2 companies)
        if len(companies) >= 2:
            for metric in INTENSITY_CEILINGS:
                vals = [
                    (c, m.get(metric))
                    for c, m in companies.items()
                    if m.get(metric) is not None
                    and isinstance(m.get(metric), (int, float))
                    and m.get(metric) > 0
                ]
                if len(vals) < 2:
                    continue
                nums   = [v for _, v in vals]
                median = statistics.median(nums)
                if median == 0:
                    continue
                for company, val in vals:
                    if val > median * 10:
                        warnings.append(dict(
                            company=company, metric=metric, value=val,
                            check="extreme_outlier",
                            severity="WARNING",
                            detail=(
                                f"{metric} = {val:,.4f} is {val/median:.1f}× "
                                f"the cross-company median ({median:,.4f}). "
                                f"Likely a unit normalization error."),
                            suggested_correction=None,
                        ))

        n_errors   = sum(1 for w in warnings if w["severity"] == "ERROR")
        n_warnings = sum(1 for w in warnings if w["severity"] == "WARNING")

        if warnings:
            print(f"  [DataValidationAgent] {n_errors} ERRORS, {n_warnings} WARNINGS")
            for w in warnings:
                sev = "✗" if w["severity"] == "ERROR" else "⚠"
                print(f"    {sev} [{w['company']}] {w['check']}: {w['metric']} "
                      f"= {w.get('value')}")
        else:
            print(f"  [DataValidationAgent] ✓ All checks passed")

        return dict(
            warnings=warnings,
            revenue_corrections=revenue_corrections,
            n_errors=n_errors,
            n_warnings=n_warnings,
        )

    # ── Revenue scale detection ───────────────────────────────────────────────

    @staticmethod
    def _check_revenue_scale(
        company: str,
        rev: Optional[float],
    ) -> Optional[dict]:
        """
        Returns a correction dict if revenue looks mis-scaled, else None.
        Heuristic: if rev < REVENUE_LOOKS_FINE_ABOVE (10 crore = 1e8 INR),
        test scale candidates to find one that puts rev in the plausible range.
        """
        if rev is None or not isinstance(rev, (int, float)) or rev <= 0:
            return None
        if rev >= REVENUE_LOOKS_FINE_ABOVE:
            return None   # looks fine

        # rev < 1e8 — suspicious. Try candidates.
        for factor, unit_label, confidence in REVENUE_SCALE_CANDIDATES:
            corrected = rev * factor
            if corrected >= REVENUE_FLOOR_INR:
                return {
                    "original_INR":   rev,
                    "corrected_INR":  corrected,
                    "factor_applied": factor,
                    "unit_assumed":   unit_label,
                    "confidence":     confidence,
                }
        return None

    # ── Apply revenue corrections to a company metrics dict ──────────────────

    @staticmethod
    def apply_revenue_correction(
        metrics: dict,
        correction: dict,
    ) -> dict:
        """
        Returns a NEW metrics dict with Turnover_INR corrected and all
        perRevCr intensities recomputed from the corrected revenue.

        This is called ONLY when the engine decides to auto-correct.
        The original value is preserved as Turnover_INR_raw.
        """
        m = dict(metrics)
        old_rev    = m.get("Turnover_INR", 0)
        new_rev    = correction["corrected_INR"]
        factor     = correction["factor_applied"]

        m["Turnover_INR_raw"]     = old_rev
        m["Turnover_INR"]         = new_rev
        m["Turnover_unit_assumed"]= correction["unit_assumed"]
        m["Turnover_corr_factor"] = factor

        # Recompute rev_cr and all perRevCr metrics
        new_rev_cr = new_rev / 1e7
        if new_rev_cr > 0:
            for src, out in [
                ("TotalEnergy_GJ",      "Energy_GJ_perRevCr"),
                ("TotalGHG_tCO2e",      "GHG_tCO2e_perRevCr"),
                ("WasteGenerated_MT",   "Waste_MT_perRevCr"),
                ("WaterConsumption_KL", "Water_KL_perRevCr"),
            ]:
                if m.get(src) is not None:
                    m[out] = round(m[src] / new_rev_cr, 6)

            # Employee-revenue ratio
            emp = m.get("perm_emp_total", 0)
            if emp:
                m["EmpIntensity_per_CrTurnover"] = round(emp / new_rev_cr, 4)

        print(f"    [DataValidator] Revenue corrected: {old_rev:,.0f} → "
              f"{new_rev:,.0f} INR  (×{factor:.0e}, assumed {correction['unit_assumed']})")
        return m