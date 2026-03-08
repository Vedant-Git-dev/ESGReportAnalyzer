"""
agents/insight_agent.py
-----------------------
Agent 6 — InsightAgent

LLM-only insights via Groq (official `groq` library).
No rule-based fallback.

Hallucination prevention
------------------------
The LLM is given a pre-computed, structured summary — NOT raw data.
It is explicitly told NOT to recompute or re-rank any metric.
The payload includes:
  • kpi_leaders       — who leads each KPI and by how much
  • kpi_laggards      — who lags each KPI and by how much
  • top_strengths     — per company: KPIs rated "Good" with ±% vs benchmark
  • top_gaps          — per company: KPIs rated "Below Average" with ±%
  • operational_scale — absolute totals for context only (not for ranking)
  • unreliable_flags  — metrics excluded due to small denominators

The system prompt strictly forbids re-computing or re-ranking.
"""

from __future__ import annotations

import json
import os
import textwrap
from typing import Optional

import pandas as pd

from config.constants import (
    GROQ_MODEL, LOWER_IS_BETTER,
    ESG_EFFICIENCY_METRICS, ESG_POLICY_METRICS,
    SOCIAL_METRICS, SAFETY_METRICS, GOVERNANCE_METRICS,
    OPERATIONAL_SCALE_METRICS,
)

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

        You will be given a pre-computed, structured ESG summary.
        Your ONLY job is to write clear narrative explanations of what the numbers mean.

        ABSOLUTE RULES — never violate these:
        1. Do NOT re-compute, re-rank, or re-derive any metric.
           The summary already contains the correct leaders, laggards, gaps, and strengths.
        2. Do NOT say a company "performs better" because its total emissions or total energy
           is larger. Absolute operational scale metrics (TotalGHG, TotalEnergy, etc.)
           reflect company size — they are NOT performance indicators.
        3. Only judge environmental performance using intensity metrics (per employee):
           GHG_tCO2e_perEmployee, Energy_GJ_perEmployee, Water_KL_perEmployee,
           Waste_MT_perEmployee.
           Lower intensity = better environmental efficiency.
        4. Do NOT fabricate data. If a metric is null or absent, say "data not available".
        5. For "lower is better" KPIs (efficiency intensities, LTIFR, Fatalities,
           Complaints): a LOWER value is the BETTER outcome. Always state this clearly.
        6. Any metric marked as "unreliable" (small sample) must be flagged, not praised.

        Structure your response with these exact headings:

        ## KEY INSIGHTS
        (3-5 data-driven bullets; compare companies on ESG efficiency, not scale)

        ## STRENGTHS BY COMPANY
        (For each company: list KPIs where it is rated GOOD vs benchmark, with %
         advantage; use ESG-correct language)

        ## GAPS & IMPROVEMENT AREAS
        (For each company: list KPIs rated BELOW AVERAGE; state the specific gap %;
         recommend concrete actions)

        ## ENVIRONMENTAL PERFORMANCE
        (Discuss only intensity metrics — GHG/employee, Energy/employee, etc.;
         mention absolute totals for context only, clearly labeled as scale indicators)

        ## OVERALL ESG RANKING
        (Rank companies; justify using only the pre-computed data provided)
    """).strip()

    QUERY_SYSTEM_PROMPT = textwrap.dedent("""
        You are a senior ESG analyst. Answer the question using ONLY the structured
        data provided. Do NOT re-compute metrics. Do NOT fabricate comparisons.
        For lower-is-better KPIs (efficiency intensities, LTIFR, Fatalities,
        Complaints), a lower value = better performance.
        Keep the answer under 200 words.
    """).strip()

    # ── Groq call ─────────────────────────────────────────────────────────────

    def _call_groq(self, system: str, user_prompt: str,
                   max_tokens: int = 1000) -> Optional[str]:
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
                temperature=0.2,   # low temperature = less hallucination
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

    def generate(self, companies: dict, gap_df: pd.DataFrame,
                 rankings_df: pd.DataFrame, mode_info: str) -> str:
        if not _GROQ_AVAILABLE:
            return self._unavailable("groq library not installed")
        if not os.environ.get("GROQ_API_KEY", "").strip():
            return self._unavailable("GROQ_API_KEY not set")

        payload = self._build_structured_payload(
            companies, gap_df, rankings_df, mode_info)
        user_prompt = (
            f"Benchmark mode: {mode_info}\n"
            f"Companies: {', '.join(companies.keys())}\n\n"
            "IMPORTANT: Use ONLY the pre-computed summary below.\n"
            "Do NOT recompute or rerank any metric.\n\n"
            f"{json.dumps(payload, indent=2)}\n\n"
            "Write the structured ESG narrative report using the section "
            "headings in your instructions."
        )
        result = self._call_groq(self.SYSTEM_PROMPT, user_prompt, max_tokens=1400)
        return result or self._unavailable(
            "Groq API call failed — check your key and network")

    def answer_query(self, query: str, companies: dict,
                     gap_df: pd.DataFrame, rankings_df: pd.DataFrame) -> str:
        if not _GROQ_AVAILABLE:
            return self._unavailable("groq library not installed")
        if not os.environ.get("GROQ_API_KEY", "").strip():
            return self._unavailable("GROQ_API_KEY not set")

        payload = self._build_structured_payload(companies, gap_df, rankings_df, "")
        user_prompt = (
            "Pre-computed ESG summary (use only this — do not recompute):\n"
            f"{json.dumps(payload, indent=2)}\n\n"
            f"Question: {query}"
        )
        result = self._call_groq(self.QUERY_SYSTEM_PROMPT, user_prompt, max_tokens=400)
        return result or self._unavailable(
            "Groq API call failed — check your key and network")

    # ── structured payload builder ────────────────────────────────────────────

    def _build_structured_payload(
        self,
        companies: dict,
        gap_df: pd.DataFrame,
        rankings_df: pd.DataFrame,
        mode_info: str,
    ) -> dict:
        """
        Build a pre-computed, structured payload that tells the LLM exactly:
          - who leads / lags each KPI (and by how much)
          - each company's top strengths and gaps (with % vs benchmark)
          - operational scale context (clearly labelled — not for ranking)
          - which metrics are unreliable

        The LLM only needs to translate this into narrative.
        """

        payload: dict = {
            "instructions_to_llm": (
                "Use ONLY this pre-computed data. Do NOT re-derive metrics. "
                "Do NOT use operational_scale_context for performance comparison."
            ),
            "benchmark_mode": mode_info,
            "lower_is_better_kpis": sorted(LOWER_IS_BETTER),
            "metric_type_guide": {
                "ESG Efficiency (lower=better, ranked)":
                    sorted(ESG_EFFICIENCY_METRICS),
                "ESG Policy (higher=better, ranked)":
                    sorted(ESG_POLICY_METRICS),
                "Social (higher=better, ranked)":
                    sorted(SOCIAL_METRICS),
                "Safety (lower=better, ranked)":
                    sorted(SAFETY_METRICS),
                "Governance (lower=better, ranked)":
                    sorted(GOVERNANCE_METRICS),
                "Operational Scale (context only — NOT for ranking)":
                    sorted(OPERATIONAL_SCALE_METRICS & {
                        "TotalGHG_tCO2e","TotalEnergy_GJ",
                        "WaterConsumption_KL","WasteGenerated_MT",
                        "Turnover_INR",
                    }),
            },
        }

        # ── KPI leaders & laggards (pre-computed from rankings_df) ────────────
        if not rankings_df.empty:
            payload["kpi_leaders"]  = {}
            payload["kpi_laggards"] = {}
            for kpi in sorted(rankings_df["KPI"].unique()):
                sub   = rankings_df[rankings_df["KPI"] == kpi].sort_values("Rank")
                best  = sub.iloc[0]
                worst = sub.iloc[-1]
                lower = kpi in LOWER_IS_BETTER
                perf_label = "lower_is_better" if lower else "higher_is_better"
                bv = None if best["Value"] == 0.0 else round(best["Value"], 4)
                wv = None if worst["Value"] == 0.0 else round(worst["Value"], 4)
                payload["kpi_leaders"][kpi] = {
                    "company": best["Company"], "value": bv,
                    "performance_direction": perf_label,
                }
                if len(sub) > 1:
                    payload["kpi_laggards"][kpi] = {
                        "company": worst["Company"], "value": wv,
                        "performance_direction": perf_label,
                    }

        # ── Per-company strengths and gaps (pre-computed from gap_df) ─────────
        payload["company_analysis"] = {}
        for company in companies:
            sub = gap_df[gap_df["Company"] == company] if not gap_df.empty \
                else pd.DataFrame()

            # top strengths (Good rating, sorted by gap_pct advantage)
            good = sub[sub["Rating"] == "Good"] if not sub.empty else pd.DataFrame()
            top_strengths = []
            if not good.empty:
                lower_mask = good["KPI"].isin(LOWER_IS_BETTER)
                # For lower-is-better, gap is negative when we're better
                good_sorted = good.copy()
                good_sorted["perf_pct"] = good_sorted.apply(
                    lambda r: -r["Gap_Pct"] if r["KPI"] in LOWER_IS_BETTER
                              else r["Gap_Pct"], axis=1)
                for _, r in good_sorted.sort_values(
                        "perf_pct", ascending=False).head(5).iterrows():
                    top_strengths.append({
                        "kpi":      r["KPI"],
                        "kpi_type": r["KPI_Type"],
                        "value":    None if r["Value"] == 0.0 else r["Value"],
                        "benchmark":r["Benchmark"],
                        "advantage_pct": abs(r["Gap_Pct"]) if r["Gap_Pct"] else None,
                        "note": (
                            "lower_value_is_better — this company emits/uses less"
                            if r["KPI"] in LOWER_IS_BETTER else "higher_value_is_better"
                        ),
                    })

            # top gaps (Below Average, sorted by worst first)
            bad = sub[sub["Rating"] == "Below Average"] \
                if not sub.empty else pd.DataFrame()
            top_gaps = []
            if not bad.empty:
                bad_sorted = bad.copy()
                bad_sorted["deficit_pct"] = bad_sorted.apply(
                    lambda r: r["Gap_Pct"] if r["KPI"] in LOWER_IS_BETTER
                              else -r["Gap_Pct"], axis=1)
                for _, r in bad_sorted.sort_values(
                        "deficit_pct", ascending=False).head(5).iterrows():
                    top_gaps.append({
                        "kpi":       r["KPI"],
                        "kpi_type":  r["KPI_Type"],
                        "value":     None if r["Value"] == 0.0 else r["Value"],
                        "benchmark": r["Benchmark"],
                        "deficit_pct": abs(r["Gap_Pct"]) if r["Gap_Pct"] else None,
                        "note": (
                            "lower_value_is_better — this company needs to reduce this"
                            if r["KPI"] in LOWER_IS_BETTER
                            else "higher_value_is_better — this company needs to improve this"
                        ),
                    })

            # unreliable metrics for this company
            unreliable = [
                k.replace("_UNRELIABLE", "")
                for k, v in companies[company].items()
                if k.endswith("_UNRELIABLE") and v
            ]

            # operational scale context (display-only, clearly labelled)
            scale_keys = {
                "TotalGHG_tCO2e":     "Total GHG Scope1+2 (tCO2e) — scale indicator only",
                "TotalEnergy_GJ":     "Total Energy (GJ) — scale indicator only",
                "WaterConsumption_KL":"Water Consumption (KL) — scale indicator only",
                "WasteGenerated_MT":  "Waste Generated (MT) — scale indicator only",
                "Turnover_INR":       "Turnover (INR) — scale indicator only",
                "perm_emp_total":     "Permanent Employees",
            }
            scale_ctx = {
                label: companies[company][k]
                for k, label in scale_keys.items()
                if companies[company].get(k) is not None
                and companies[company][k] != 0.0
            }

            payload["company_analysis"][company] = {
                "top_strengths":       top_strengths,
                "top_gaps":            top_gaps,
                "unreliable_metrics":  unreliable,
                "operational_scale_context": scale_ctx,
            }

        return payload