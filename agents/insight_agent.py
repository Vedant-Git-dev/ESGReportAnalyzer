"""
agents/insight_agent.py
-----------------------
Agent 6 — InsightAgent (LLM-only, Groq)

Hallucination prevention
------------------------
The LLM receives a pre-computed, structured summary including:
  • kpi_leaders / kpi_laggards    — pre-ranked, direction-labelled
  • top_strengths / top_gaps      — per company, with advantage/deficit %
  • esg_scores                    — composite weighted scores (E/S/G + composite)
  • scale_warning                 — if companies differ massively in size
  • company_sizes                 — turnover, employees (for economic context)
  • benchmark_method              — "median" so LLM cites it correctly
  • operational_scale_context     — totals clearly labelled as size indicators
  • metric_type_guide             — so LLM uses correct direction language

The system prompt forbids re-computing metrics and contains explicit rules
for how to describe scale metrics vs ESG performance metrics.
"""

from __future__ import annotations

import json
import os
import textwrap
from typing import Optional

import pandas as pd

from config.constants import (
    GROQ_MODEL, LOWER_IS_BETTER, BENCHMARK_METHOD,
    ESG_EFFICIENCY_METRICS, ESG_POLICY_METRICS,
    SOCIAL_METRICS, SAFETY_METRICS, GOVERNANCE_METRICS,
    OPERATIONAL_SCALE_METRICS,
)
from utils.formatting import _magnitude

try:
    from groq import Groq as _GroqLib
    _GROQ_AVAILABLE = True
except ImportError:
    _GroqLib = None
    _GROQ_AVAILABLE = False


class InsightAgent:

    SYSTEM_PROMPT = textwrap.dedent("""
        You are a senior ESG analyst specialising in Indian energy sector companies
        filing SEBI BRSR XBRL reports.

        You receive a pre-computed, structured ESG performance summary.
        Your ONLY job is to write clear, accurate, observation-based narrative.

        ABSOLUTE RULES — never violate any of these:

        1. Do NOT re-compute, re-rank, or re-derive any metric.
           All rankings, scores, leaders, laggards, and gaps are pre-computed.

        2. SCALE METRICS ARE NOT PERFORMANCE INDICATORS:
           TotalGHG, TotalEnergy, WaterConsumption, WasteGenerated, Turnover
           are operational scale indicators, NOT ESG performance metrics.
           CORRECT: "Company X has larger operational GHG due to its larger scale."
           INCORRECT: "Company X performs worse on emissions."

        3. Use INTENSITY metrics for ESG performance:
           - GHG/employee, Energy/employee, GHG/₹Cr-revenue, etc.
           - Lower intensity = better environmental efficiency.

        4. Revenue-normalised intensities (perRevCr) are often more meaningful
           in capital-intensive sectors.

        5. Cite the benchmark method as "industry median" not "industry average".

        6. Cite ESG composite scores (E/S/G + composite) and grades (A/B/C/D).

        7. If a scale_warning is present, include it prominently.

        8. Do NOT fabricate data. If a value is null, say "data not available".

        9. STRICT TEMPLATE — every observation must follow this format:
           Observation: [What the data shows — one factual sentence]
           Benchmark comparison: [How it compares to industry median]
           Gap magnitude: [Quantified difference using magnitude words]

           Example:
           Observation: Energy intensity per ₹Cr revenue is 5.73 GJ/₹Cr.
           Benchmark comparison: The industry median is 8.20 GJ/₹Cr.
           Gap magnitude: Energy intensity is moderately lower than the industry median.

        10. NO CAUSAL SPECULATION — never suggest causes or attributions.
            WRONG: "This may be due to outdated infrastructure."
            WRONG: "This likely reflects the company's investment in solar."
            WRONG: "The high waste intensity could indicate inefficient processes."
            CORRECT: Report only what the data shows, nothing more.

        11. LANGUAGE RULE — never cite raw percentage gaps as performance claims:
           WRONG: "Company has a 99% advantage in GHG per employee"
           CORRECT: "GHG intensity is substantially lower than the industry median"
           Magnitude qualifiers:
             |gap| ≥ 50% → "substantially"
             |gap| ≥ 25% → "significantly"
             |gap| ≥ 10% → "moderately"
             |gap| < 10% → "marginally"

        12. DIRECTION: always confirm direction explicitly:
            lower_better metric: lower value = better performance
            higher_better metric: higher value = better performance

        13. If a pillar score is N/A, state the effective re-normalised weights.
            Cite the weight basis: "Fixed configuration weights (E=40% S=30% G=30%)"

        Structure your response with these exact headings:

        ## KEY INSIGHTS
        (3-5 bullets; lead with scale warning if present; compare on ESG intensity)

        ## STRENGTHS BY COMPANY
        (KPIs rated Good; for each: Observation / Benchmark comparison / Gap magnitude)

        ## GAPS & IMPROVEMENT AREAS
        (KPIs rated Below Average; for each: Observation / Benchmark comparison / Gap magnitude)

        ## ENVIRONMENTAL PERFORMANCE
        (Intensity metrics only for comparison; totals for scale context only)

        ## ESG COMPOSITE SCORES
        (Cite E/S/G pillar scores, composite score, grade, effective weights, weight basis)

        ## OVERALL ESG RANKING
        (Rank companies with justification; reference composite scores and grades)
    """).strip()

    QUERY_SYSTEM_PROMPT = textwrap.dedent("""
        You are a senior ESG analyst. Answer using ONLY pre-computed data provided.
        Do NOT re-compute. For lower-is-better KPIs, lower = better.
        Scale metrics (TotalGHG, TotalEnergy) reflect company size, not performance.
        Cite the benchmark method as "industry median".
        Keep answer under 200 words.
    """).strip()

    def _call_groq(self, system: str, user_prompt: str,
                   max_tokens: int = 1200) -> Optional[str]:
        if not _GROQ_AVAILABLE:
            return None
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            return None
        try:
            client = _GroqLib(api_key=api_key)
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                max_tokens=max_tokens,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"    [InsightAgent] Groq error: {e}")
            return None

    @staticmethod
    def _unavailable(reason: str) -> str:
        return (
            f"  ⚠  LLM insights unavailable: {reason}\n"
            "     To enable: pip install groq  &&  export GROQ_API_KEY=gsk_...\n"
        )

    # ── public interface ──────────────────────────────────────────────────────

    def generate(
        self,
        companies: dict,
        gap_df: pd.DataFrame,
        rankings_df: pd.DataFrame,
        mode_info: str,
        scale_warning: Optional[str] = None,
        esg_scores: Optional[dict] = None,
    ) -> str:
        if not _GROQ_AVAILABLE:
            return self._unavailable("groq library not installed")
        if not os.environ.get("GROQ_API_KEY", "").strip():
            return self._unavailable("GROQ_API_KEY not set")

        payload = self._build_payload(
            companies, gap_df, rankings_df, mode_info,
            scale_warning, esg_scores)
        user_prompt = (
            f"Benchmark mode: {mode_info}\n"
            f"Companies: {', '.join(companies.keys())}\n\n"
            "IMPORTANT: Use ONLY this pre-computed data. Do NOT recompute metrics.\n\n"
            f"{json.dumps(payload, indent=2)}\n\n"
            "Write the structured ESG narrative report using the section headings."
        )
        result = self._call_groq(self.SYSTEM_PROMPT, user_prompt, max_tokens=1600)
        return result or self._unavailable(
            "Groq API call failed — check key and network")

    def answer_query(
        self, query: str, companies: dict,
        gap_df: pd.DataFrame, rankings_df: pd.DataFrame,
    ) -> str:
        if not _GROQ_AVAILABLE:
            return self._unavailable("groq library not installed")
        if not os.environ.get("GROQ_API_KEY", "").strip():
            return self._unavailable("GROQ_API_KEY not set")

        payload = self._build_payload(companies, gap_df, rankings_df, "")
        user_prompt = (
            "Pre-computed ESG data:\n"
            f"{json.dumps(payload, indent=2)}\n\n"
            f"Question: {query}"
        )
        result = self._call_groq(self.QUERY_SYSTEM_PROMPT, user_prompt, max_tokens=400)
        return result or self._unavailable("Groq API call failed")

    # ── payload builder ───────────────────────────────────────────────────────

    def _build_payload(
        self,
        companies: dict,
        gap_df: pd.DataFrame,
        rankings_df: pd.DataFrame,
        mode_info: str,
        scale_warning: Optional[str] = None,
        esg_scores: Optional[dict] = None,
    ) -> dict:

        payload: dict = {
            "critical_rules": [
                "Scale metrics (TotalGHG, TotalEnergy, etc.) reflect company SIZE — "
                "they are NOT ESG performance indicators",
                "Use only intensity metrics (per employee, per revenue) for ESG comparison",
                "Lower intensity = better environmental efficiency",
                "Benchmark method = " + BENCHMARK_METHOD,
            ],
            "benchmark_mode":  mode_info,
            "benchmark_method": BENCHMARK_METHOD,
            "lower_is_better_kpis": sorted(LOWER_IS_BETTER),
            "metric_type_guide": {
                "ESG_Efficiency_per_employee (lower=better, ranked)":
                    sorted(set(ESG_EFFICIENCY_METRICS) & {
                        "GHG_tCO2e_perEmployee","Energy_GJ_perEmployee",
                        "Water_KL_perEmployee","Waste_MT_perEmployee",
                    }),
                "ESG_Efficiency_per_revenue (lower=better, ranked)":
                    sorted(set(ESG_EFFICIENCY_METRICS) & {
                        "GHG_tCO2e_perRevCr","Energy_GJ_perRevCr",
                        "Water_KL_perRevCr","Waste_MT_perRevCr",
                    }),
                "ESG_Policy (higher=better, ranked)": sorted(ESG_POLICY_METRICS),
                "Social (higher=better, ranked)":     sorted(SOCIAL_METRICS),
                "Safety (lower=better, ranked)":      sorted(SAFETY_METRICS),
                "Governance (lower=better, ranked)":  sorted(GOVERNANCE_METRICS),
                "Operational_Scale (context only — NOT for ESG performance)": [
                    "TotalGHG_tCO2e","TotalEnergy_GJ",
                    "WaterConsumption_KL","WasteGenerated_MT","Turnover_INR",
                ],
            },
        }

        if scale_warning:
            payload["SCALE_WARNING"] = scale_warning

        # ── Company sizes (for economic context) ──────────────────────────────
        payload["company_sizes"] = {}
        for company, m in companies.items():
            payload["company_sizes"][company] = {
                "turnover_INR":         m.get("Turnover_INR"),
                "permanent_employees":  m.get("perm_emp_total"),
                "total_workers":        m.get("total_wkr_total"),
            }

        # ── ESG composite scores ───────────────────────────────────────────────
        if esg_scores:
            payload["esg_composite_scores"] = {
                company: {
                    "Environment":        sc.get("Environment"),
                    "Social":             sc.get("Social"),
                    "Governance":         sc.get("Governance"),
                    "Composite":          sc.get("Composite"),
                    "Grade":              sc.get("Grade"),
                    "Confidence":         sc.get("Confidence"),
                    "Valid_KPI_Count":    sc.get("Valid_KPI_Count"),
                    "Missing_Pillars":    sc.get("Missing_Pillars", []),
                    "Effective_Weights":  sc.get("Effective_Weights", {}),
                    "note": (
                        f"Composite uses re-normalised weights because "
                        f"{sc.get('Missing_Pillars')} pillar(s) had no valid data"
                        if sc.get("Missing_Pillars")
                        else "All three pillars available"
                    ),
                }
                for company, sc in esg_scores.items()
            }

        # ── Pre-computed KPI leaders / laggards ───────────────────────────────
        if not rankings_df.empty:
            payload["kpi_leaders"]  = {}
            payload["kpi_laggards"] = {}
            for kpi in sorted(rankings_df["KPI"].unique()):
                sub   = rankings_df[rankings_df["KPI"] == kpi].sort_values("Rank")
                best  = sub.iloc[0]
                worst = sub.iloc[-1]
                lower = kpi in LOWER_IS_BETTER
                bv = None if best["Value"] == 0.0 else round(best["Value"], 6)
                wv = None if worst["Value"] == 0.0 else round(worst["Value"], 6)
                payload["kpi_leaders"][kpi] = {
                    "company": best["Company"], "value": bv,
                    "direction": "lower_is_better" if lower else "higher_is_better",
                    "note": (
                        "This company has LOWER intensity = more efficient"
                        if lower else "This company has HIGHER value = better"
                    ),
                }
                if len(sub) > 1:
                    payload["kpi_laggards"][kpi] = {
                        "company": worst["Company"], "value": wv,
                        "direction": "lower_is_better" if lower else "higher_is_better",
                    }

        # ── Per-company strengths and gaps ────────────────────────────────────
        payload["company_analysis"] = {}
        for company in companies:
            sub = gap_df[gap_df["Company"] == company] if not gap_df.empty \
                else pd.DataFrame()

            good = sub[sub["Rating"] == "Good"] if not sub.empty else pd.DataFrame()
            bad  = sub[sub["Rating"] == "Below Average"] if not sub.empty \
                else pd.DataFrame()

            top_strengths = []
            if not good.empty:
                for _, r in good.sort_values("Gap_Pct", ascending=False).head(6).iterrows():
                    lower = r["KPI"] in LOWER_IS_BETTER
                    gp    = r["Gap_Pct"]
                    mag   = _magnitude(gp)
                    top_strengths.append({
                        "kpi":          r["KPI"],
                        "kpi_type":     r["KPI_Type"],
                        "value":        None if r["Value"] == 0.0 else round(r["Value"], 4),
                        "median":       round(r["Benchmark"], 4),
                        "magnitude":    mag,
                        "description":  (
                            f"{mag} lower than the industry median "
                            f"(lower intensity = better efficiency)"
                            if lower else
                            f"{mag} above the industry median"
                        ),
                        "percentile":   r.get("Percentile"),
                    })

            top_gaps = []
            if not bad.empty:
                for _, r in bad.sort_values("Gap_Pct").head(6).iterrows():
                    lower = r["KPI"] in LOWER_IS_BETTER
                    gp    = r["Gap_Pct"]
                    mag   = _magnitude(gp)
                    top_gaps.append({
                        "kpi":         r["KPI"],
                        "kpi_type":    r["KPI_Type"],
                        "value":       None if r["Value"] == 0.0 else round(r["Value"], 4),
                        "median":      round(r["Benchmark"], 4),
                        "magnitude":   mag,
                        "description": (
                            f"{mag} higher than the industry median "
                            f"(higher intensity = less efficient — needs reduction)"
                            if lower else
                            f"{mag} below the industry median — needs improvement"
                        ),
                        "action":      (
                            "reduce_intensity" if lower else "increase_value"),
                        "percentile":  r.get("Percentile"),
                    })

            unreliable = [
                k.replace("_UNRELIABLE", "")
                for k, v in companies[company].items()
                if k.endswith("_UNRELIABLE") and v
            ]

            # operational scale — clearly labelled
            scale_ctx = {}
            for k, label in [
                ("TotalGHG_tCO2e",    "total_GHG_tCO2e_scale_indicator"),
                ("TotalEnergy_GJ",    "total_energy_GJ_scale_indicator"),
                ("WaterConsumption_KL","water_consumption_KL_scale_indicator"),
                ("WasteGenerated_MT", "waste_generated_MT_scale_indicator"),
                ("Turnover_INR",      "annual_turnover_INR"),
                ("perm_emp_total",    "permanent_employees"),
            ]:
                v = companies[company].get(k)
                if v is not None and v != 0.0:
                    scale_ctx[label] = v

            payload["company_analysis"][company] = {
                "top_strengths":            top_strengths,
                "top_gaps":                 top_gaps,
                "unreliable_metrics":       unreliable,
                "operational_scale_context": scale_ctx,
            }

        return payload