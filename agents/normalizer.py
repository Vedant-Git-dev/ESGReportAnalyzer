"""
agents/normalizer.py
--------------------
Agent 3 — NormalizationAgent

Computes derived ratios, shares, and per-employee intensity metrics.

Key rules
---------
• Gender ratios require denominator ≥ MIN_RATIO_DENOMINATOR (default 30).
  If denominator is too small the ratio is stored as None and the metric
  is flagged with suffix "_UNRELIABLE" so downstream agents can warn.
• Per-employee intensities require total headcount > 0.
• Raw values of None stay None — never coerced to 0.

Input : raw metrics dict  (None = missing, 0.0 = genuine zero)
Output: extended metrics dict + optional "_UNRELIABLE" flag keys
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

        # ── Gender ratios (with denominator validation) ────────────────────────
        def ratio(num_key: str, den_key: str, out_key: str) -> None:
            n = m.get(num_key)
            d = m.get(den_key)
            if n is None or d is None or d <= 0:
                return  # missing data — leave as None
            if d < MIN_RATIO_DENOMINATOR:
                # flag as statistically unreliable — exclude from ranking
                m[out_key] = round(n / d * 100, 2)
                m[out_key + "_UNRELIABLE"] = True
                print(
                    f"    ⚠  {out_key}: denominator={int(d)} < {MIN_RATIO_DENOMINATOR}"
                    f" — flagged as unreliable"
                )
            else:
                m[out_key] = round(n / d * 100, 2)

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

        # ── Per-employee ESG efficiency intensities ───────────────────────────
        # These are the ONLY env metrics that get ranked and gap-analysed
        if total > 0:
            for src, out in [
                ("TotalEnergy_GJ",      "Energy_GJ_perEmployee"),
                ("TotalGHG_tCO2e",      "GHG_tCO2e_perEmployee"),
                ("WasteGenerated_MT",   "Waste_MT_perEmployee"),
                ("WaterConsumption_KL", "Water_KL_perEmployee"),
            ]:
                if m.get(src) is not None:
                    m[out] = round(m[src] / total, 2)

        # ── Revenue intensity (operational, not ranked) ────────────────────────
        if rev and rev > 0 and emp:
            rev_cr = rev / 1e7
            m["EmpIntensity_per_CrTurnover"] = round(emp / rev_cr, 4)

        # ── Training coverage ──────────────────────────────────────────────────
        if m.get("Emp_HS_Training") is not None and emp > 0:
            m["Pct_Emp_HS_Training"] = round(m["Emp_HS_Training"] / emp * 100, 2)
        if m.get("Wkr_HS_Training") is not None and wkr > 0:
            m["Pct_Wkr_HS_Training"] = round(m["Wkr_HS_Training"] / wkr * 100, 2)

        print(f"    → {len(m)} metrics after normalization")
        return m