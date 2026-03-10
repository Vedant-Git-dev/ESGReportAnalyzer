"""
agents/normalizer.py
--------------------
Agent 3 — NormalizationAgent

Computes derived ratios, shares, and intensity metrics.

Key rules
---------
• Gender ratios: denominator must be ≥ MIN_RATIO_DENOMINATOR (30).
  Small denominators are flagged as _UNRELIABLE and excluded from rankings.

• Per-employee intensities (ESG_EFFICIENCY_EMP_METRICS):
    GHG_tCO2e_perEmployee  = TotalGHG / total_headcount
    Energy_GJ_perEmployee  = TotalEnergy / total_headcount
    Water_KL_perEmployee   = WaterConsumption / total_headcount
    Waste_MT_perEmployee   = WasteGenerated / total_headcount
  Useful for comparing companies with similar business models.

• Per-revenue intensities (ESG_EFFICIENCY_REV_METRICS) — NEW:
    GHG_tCO2e_perRevCr  = TotalGHG / (Turnover_INR / 1e7)
    Energy_GJ_perRevCr  = TotalEnergy / (Turnover_INR / 1e7)
    Water_KL_perRevCr   = WaterConsumption / (Turnover_INR / 1e7)
    Waste_MT_perRevCr   = WasteGenerated / (Turnover_INR / 1e7)
  Units: per ₹Crore of revenue.
  Better for comparing capital-intensive companies with very different
  workforce sizes (refineries vs. lubricant manufacturers).
  Both normalisation types are retained in the output.

• None = missing data (never coerced to 0.0)
• 0.0 = genuine zero (valid data point)
"""

from __future__ import annotations
from config.constants import MIN_RATIO_DENOMINATOR


class NormalizationAgent:

    def normalize(self, metrics: dict) -> dict:
        m = dict(metrics)

        emp   = m.get("perm_emp_total") or 0
        wkr   = m.get("perm_wkr_total") or 0
        total = (emp + wkr) or m.get("total_emp_total") or 0
        rev   = m.get("Turnover_INR")
        rev_cr = (rev / 1e7) if rev and rev > 0 else None   # ₹ → crores

        # ── Gender ratios (with denominator validation) ────────────────────────
        def ratio(num_key: str, den_key: str, out_key: str) -> None:
            n = m.get(num_key)
            d = m.get(den_key)
            if n is None or d is None or d <= 0:
                return
            val = round(n / d * 100, 2)
            if d < MIN_RATIO_DENOMINATOR:
                m[out_key] = val
                m[out_key + "_UNRELIABLE"] = True
                print(f"    ⚠  {out_key}: denominator={int(d)} < "
                      f"{MIN_RATIO_DENOMINATOR} — flagged as unreliable")
            else:
                m[out_key] = val

        ratio("perm_emp_female",     "perm_emp_total",     "Female_Ratio_PermanentEmp")
        ratio("perm_wkr_female",     "perm_wkr_total",     "Female_Ratio_PermanentWorkers")
        ratio("contract_wkr_female", "contract_wkr_total", "Female_Ratio_ContractWorkers")
        ratio("total_emp_female",    "total_emp_total",    "Female_Ratio_AllEmployees")
        ratio("total_wkr_female",    "total_wkr_total",    "Female_Ratio_AllWorkers")
        ratio("nonperm_emp_female",  "nonperm_emp_total",  "Female_Ratio_NonPermEmployees")

        # ── Renewable energy share ─────────────────────────────────────────────
        ren = m.get("RenewableEnergy_GJ")
        tot = m.get("TotalEnergy_GJ")
        if ren is not None and tot and tot > 0:
            m["RenewableEnergyShare_Pct"] = round(ren / tot * 100, 2)

        # ── Combined GHG (operational scale — stored but not ranked) ──────────
        s1 = m.get("Scope1_tCO2e")
        s2 = m.get("Scope2_tCO2e")
        if s1 is not None and s2 is not None:
            m["TotalGHG_tCO2e"] = round(s1 + s2, 2)

        # ── Waste recovery rate ────────────────────────────────────────────────
        rec = m.get("WasteRecovered_MT")
        gen = m.get("WasteGenerated_MT")
        if rec is not None and gen and gen > 0:
            m["WasteRecoveryRate_Pct"] = round(rec / gen * 100, 2)

        # ── Per-employee ESG efficiency intensities ────────────────────────────
        # Best for companies with similar workforce structures
        if total > 0:
            for src, out in [
                ("TotalEnergy_GJ",      "Energy_GJ_perEmployee"),
                ("TotalGHG_tCO2e",      "GHG_tCO2e_perEmployee"),
                ("WasteGenerated_MT",   "Waste_MT_perEmployee"),
                ("WaterConsumption_KL", "Water_KL_perEmployee"),
            ]:
                if m.get(src) is not None:
                    m[out] = round(m[src] / total, 4)

        # ── Per-revenue ESG efficiency intensities (NEW) ───────────────────────
        # Better for capital-intensive industries with different workforce sizes
        # Units: per ₹Crore of annual revenue
        if rev_cr is not None and rev_cr > 0:
            for src, out in [
                ("TotalEnergy_GJ",      "Energy_GJ_perRevCr"),
                ("TotalGHG_tCO2e",      "GHG_tCO2e_perRevCr"),
                ("WasteGenerated_MT",   "Waste_MT_perRevCr"),
                ("WaterConsumption_KL", "Water_KL_perRevCr"),
            ]:
                if m.get(src) is not None:
                    m[out] = round(m[src] / rev_cr, 6)

        # ── Employee-revenue ratio ─────────────────────────────────────────────
        if rev_cr and rev_cr > 0 and emp:
            m["EmpIntensity_per_CrTurnover"] = round(emp / rev_cr, 4)

        # ── Training coverage ──────────────────────────────────────────────────
        if m.get("Emp_HS_Training") is not None and emp > 0:
            m["Pct_Emp_HS_Training"] = round(m["Emp_HS_Training"] / emp * 100, 2)
        if m.get("Wkr_HS_Training") is not None and wkr > 0:
            m["Pct_Wkr_HS_Training"] = round(m["Wkr_HS_Training"] / wkr * 100, 2)

        print(f"    → {len(m)} metrics after normalization")
        return m