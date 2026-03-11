"""
engine.py
---------
ESGBenchmarkingEngine v8 — pipeline orchestrator.

Pipeline (v8 — with unit normalisation layer)
----------------------------------------------

  XMLParsingAgent          — parses XBRL → structured dict
        ↓
  MetricExtractionAgent    — extracts (metrics_dict, unit_ref_map)
        ↓
  UnitConverter            — deterministic conversion to canonical units
    • Step 1: XBRL unit_ref → XBRL_UNIT_TABLE (deterministic)
    • Step 2: label regex   → LABEL_PATTERNS   (deterministic)
    • Step 3: LLM detection → unit name only; arithmetic done here
    • Produces: ConversionRecord audit trail per metric
        ↓
  UnitConverter.check_anomalies()  — post-conversion sanity check
    • Flags intensity metrics that exceed plausible thresholds
    • Anomalous metrics excluded from ranking
        ↓
  NormalizationAgent       — ratios, shares, intensity metrics
    • Receives pre-converted metrics + anomaly_flags
    • Propagates anomaly suppression to derived intensities
        ↓
  BenchmarkAgent           — median benchmark, quartiles, scale warning
        ↓
  GapAnalysisAgent         — percentile-aware gap classification
        ↓
  ESGScoringAgent          — weighted composite ESG scores + confidence
        ↓
  InsightAgent             — Groq LLM narrative (unit-aware context)
        ↓
  ReportBuilder            — 7-section report + unit audit CSV

LLM safety contract
-------------------
The LLM (InsightAgent and UnitConverter._detect_unit_via_llm) is NEVER
given numeric values to convert. The LLM only names units from text labels.
All arithmetic is performed deterministically by UnitConverter.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import pandas as pd

from agents.benchmark import BenchmarkAgent
from agents.gap_analysis import GapAnalysisAgent
from agents.insight_agent import InsightAgent
from agents.xml_parser import XMLParsingAgent
from agents.report_builder import ReportBuilder
from agents.normalizer import NormalizationAgent
from agents.metric_extractor import MetricExtractionAgent
from agents.scoring import ESGScoringAgent
from agents.unit_converter import UnitConverter
from agents.data_validator import DataValidationAgent

try:
    from groq import Groq as _GroqLib
    _GROQ_AVAILABLE = True
except ImportError:
    _GroqLib = None
    _GROQ_AVAILABLE = False


class ESGBenchmarkingEngine:
    """
    Parameters
    ----------
    mode       : 'industry' | 'peer'
    focal      : company name for peer mode focal point
    peer       : company name for peer mode comparator
    output_dir : directory where report and CSVs are written
    """

    def __init__(
        self,
        mode: str = "industry",
        focal: Optional[str] = None,
        peer: Optional[str] = None,
        output_dir: str = "outputs",
    ) -> None:
        self.mode       = mode
        self.focal      = focal
        self.peer       = peer
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        self._parser         = XMLParsingAgent()
        self._extractor      = MetricExtractionAgent()
        self._normalizer     = NormalizationAgent()
        self._benchmarker    = BenchmarkAgent()
        self._gap_agent      = GapAnalysisAgent()
        self._scorer         = ESGScoringAgent()
        self._insight_agent  = InsightAgent()
        self._report_builder = ReportBuilder()

        # UnitConverter — initialised with Groq client if available
        groq_client = None
        api_key     = os.environ.get("GROQ_API_KEY", "").strip()
        if _GROQ_AVAILABLE and api_key:
            try:
                groq_client = _GroqLib(api_key=api_key)
            except Exception:
                pass

        from config.constants import GROQ_MODEL
        self._unit_converter = UnitConverter(
            use_llm     = bool(groq_client),
            groq_client = groq_client,
            model       = GROQ_MODEL,
        )
        self._data_validator = DataValidationAgent()

        self._companies:        dict         = {}
        self._gap_df:           pd.DataFrame = pd.DataFrame()
        self._rankings_df:      pd.DataFrame = pd.DataFrame()
        # {company: [ConversionRecord]}
        self._conversion_logs:  dict         = {}
        # {company: {metric: anomaly_info}}
        self._anomaly_logs:     dict         = {}
        # Full validation result from DataValidationAgent
        self._validation_result: dict        = {}

    # ── main pipeline ─────────────────────────────────────────────────────────

    def run(
        self,
        file_paths: list[str],
        nl_queries: Optional[list[str]] = None,
    ) -> str:
        print("\n" + "=" * 64)
        print("  ESG BENCHMARKING ENGINE v8  (hybrid unit normalisation)")
        print("=" * 64)

        self._companies       = {}
        self._conversion_logs = {}
        self._anomaly_logs    = {}

        for fp in file_paths:
            result = self._load_file(fp)
            if result:
                name, metrics, conv_recs, anomalies = result
                self._companies[name]       = metrics
                self._conversion_logs[name] = conv_recs
                self._anomaly_logs[name]    = anomalies

        if not self._companies:
            msg = (
                "  ✗  No companies loaded. Check file paths.\n"
                + "\n".join(f"    • {fp}" for fp in file_paths)
            )
            print(msg)
            return msg

        print(f"\n  Loaded {len(self._companies)} company/companies: "
              f"{', '.join(self._companies.keys())}")

        # ── Data Validation (Fix 5: post-normalisation checks) ────────────────
        self._validation_result = self._data_validator.validate(self._companies)
        # Auto-apply revenue corrections for confirmed errors
        for company, corr in self._validation_result["revenue_corrections"].items():
            if company in self._companies:
                self._companies[company] = DataValidationAgent.apply_revenue_correction(
                    self._companies[company], corr)

        # ── Benchmark ─────────────────────────────────────────────────────────
        bench_result = self._benchmarker.compute(
            self._companies, mode=self.mode,
            focal=self.focal, peer=self.peer,
        )
        self._rankings_df = bench_result["rankings"]

        # ── Gap analysis ──────────────────────────────────────────────────────
        self._gap_df = self._gap_agent.analyze(
            self._companies,
            bench_result["benchmarks"],
            quartiles   = bench_result.get("quartiles"),
            peer_counts = bench_result.get("peer_counts"),
            rankings_df = self._rankings_df,
        )

        # ── ESG composite scoring ─────────────────────────────────────────────
        esg_scores = self._scorer.score(self._companies, self._gap_df)

        # ── LLM insights ──────────────────────────────────────────────────────
        print("\n  [InsightAgent] Calling Groq LLM...")
        insights = self._insight_agent.generate(
            self._companies, self._gap_df,
            self._rankings_df, bench_result["mode_info"],
            scale_warning = bench_result.get("scale_warning"),
            esg_scores    = esg_scores,
        )

        # ── Build report ──────────────────────────────────────────────────────
        report = self._report_builder.build(
            self._companies, self._gap_df, self._rankings_df,
            bench_result, insights, self.output_dir,
            esg_scores       = esg_scores,
            conversion_logs  = self._conversion_logs,
            anomaly_logs     = self._anomaly_logs,
            validation_result= self._validation_result,
        )
        print(report)

        # ── Save unit audit CSV ───────────────────────────────────────────────
        self._save_unit_audit()

        # ── NL queries ────────────────────────────────────────────────────────
        if nl_queries:
            print("\n" + "═" * 64)
            print("  NATURAL LANGUAGE QUERIES")
            print("═" * 64)
            query_log = []
            for q in nl_queries:
                print(f"\n  Q: {q}")
                answer = self.query(q)
                print(f"  A:\n{answer}")
                query_log.append({"question": q, "answer": answer})
            with open(os.path.join(self.output_dir, "nl_queries.json"), "w") as f:
                json.dump(query_log, f, indent=2)

        return report

    def query(self, question: str) -> str:
        if not self._companies:
            return "No data loaded — call run() first."
        return self._insight_agent.answer_query(
            question, self._companies, self._gap_df, self._rankings_df
        )

    # ── file loading sub-pipeline ─────────────────────────────────────────────

    def _load_file(self, filepath: str) -> Optional[tuple]:
        """
        Returns (company_name, normalised_metrics, conversion_records, anomalies)
        or None on failure.
        """
        if not os.path.exists(filepath):
            print(f"  ✗  NOT FOUND — skipping: {filepath}")
            return None
        try:
            parsed = self._parser.parse(filepath)
        except Exception as e:
            print(f"  ✗  PARSE ERROR — skipping {filepath}: {e}")
            return None

        name = (
            parsed["company_info"].get("company_name")
            or os.path.splitext(os.path.basename(filepath))[0]
        )

        try:
            # Step 1 — extract (metrics, unit_ref_map)
            raw_metrics, unit_ref_map = self._extractor.extract(parsed)

            # Step 2 — unit detection + deterministic conversion
            print(f"  [UnitConverter] Converting {name}...")
            converted, conv_records = self._unit_converter.convert_all(
                metrics      = raw_metrics,
                unit_ref_map = unit_ref_map,
            )
            _summarise_conversion(conv_records, name)

            # Step 3 — post-conversion anomaly detection
            anomalies = UnitConverter.check_anomalies(converted)
            if anomalies:
                print(f"  [UnitConverter] ⚠  {len(anomalies)} anomaly/anomalies "
                      f"detected for {name}:")
                for metric, info in anomalies.items():
                    print(f"    ✗ {metric}: {info['reason']}")

            # Step 4 — normalisation (ratios, intensities) with anomaly propagation
            normed = self._normalizer.normalize(converted, anomaly_flags=anomalies)

        except Exception as e:
            import traceback
            print(f"  ✗  EXTRACTION ERROR — skipping {name}: {e}")
            traceback.print_exc()
            return None

        return name, normed, conv_records, anomalies

    # ── audit CSV ─────────────────────────────────────────────────────────────

    def _save_unit_audit(self) -> None:
        """Write unit_conversion_audit.csv with full conversion trail."""
        rows = []
        for company, records in self._conversion_logs.items():
            for r in records:
                anomaly_info = self._anomaly_logs.get(company, {}).get(r.metric)
                rows.append({
                    "Company":         company,
                    "Metric":          r.metric,
                    "Raw_Value":       r.raw_value,
                    "Raw_Unit":        r.raw_unit,
                    "Factor":          r.factor,
                    "Canonical_Value": r.converted_value,
                    "Canonical_Unit":  r.canonical_unit,
                    "Detection_Source": r.source,
                    "LLM_Confidence":  r.llm_confidence,
                    "Anomaly":         bool(anomaly_info),
                    "Anomaly_Reason":  anomaly_info["reason"] if anomaly_info else "",
                    "Excluded":        bool(anomaly_info),
                })
        if rows:
            import pandas as pd
            df = pd.DataFrame(rows)
            path = os.path.join(self.output_dir, "unit_conversion_audit.csv")
            df.to_csv(path, index=False)
            print(f"  [Engine] Unit audit saved → {path}")
            llm_rows = df[df["Detection_Source"] == "llm_detected"]
            if not llm_rows.empty:
                print(f"    LLM-detected units: {len(llm_rows)} "
                      f"(avg confidence: {llm_rows['LLM_Confidence'].mean():.2f})")


def _summarise_conversion(records, company: str) -> None:
    """Print a concise summary of conversion sources for one company."""
    from collections import Counter
    src_counts = Counter(r.source for r in records)
    total = len(records)
    parts = [f"{src}={n}" for src, n in sorted(src_counts.items())]
    print(f"    → {total} metrics converted  [{', '.join(parts)}]")
    # Print non-assumed conversions for transparency
    for r in records:
        if r.source != "xbrl_unit_ref" and r.factor not in (1.0, ""):
            print(f"    ⟳  {r}")