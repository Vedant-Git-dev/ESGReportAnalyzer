"""
engine.py
---------
ESGBenchmarkingEngine v7 — pipeline orchestrator.

Changes from v6
---------------
• Passes quartiles, peer_counts, rankings_df to GapAnalysisAgent for
  percentile-based scoring.
• Passes scale_warning and benchmark stats to InsightAgent and ReportBuilder.
• Passes esg_scores (composite weighted scores) to ReportBuilder.
• scale_warning is surfaced at the top of Section 0 when triggered.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import pandas as pd

from agents.benchmark import BenchmarkAgent
from agents.gap_analysis import GapAnalysisAgent
from agents.insight_agent import InsightAgent
from agents.metric_extractor import MetricExtractionAgent
from agents.normalizer import NormalizationAgent
from agents.xml_parser import XMLParsingAgent
from agents.report_builder import ReportBuilder
from agents.scoring import ESGScoringAgent


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

        self._companies:   dict         = {}
        self._gap_df:      pd.DataFrame = pd.DataFrame()
        self._rankings_df: pd.DataFrame = pd.DataFrame()

    # ── main pipeline ─────────────────────────────────────────────────────────

    def run(
        self,
        file_paths: list[str],
        nl_queries: Optional[list[str]] = None,
    ) -> str:
        print("\n" + "=" * 62)
        print("  ESG BENCHMARKING ENGINE v7")
        print("=" * 62)

        self._companies = {}
        for fp in file_paths:
            result = self._load_file(fp)
            if result:
                name, metrics = result
                self._companies[name] = metrics

        if not self._companies:
            msg = (
                "  ✗  No companies loaded. Check file paths.\n"
                + "\n".join(f"    • {fp}" for fp in file_paths)
            )
            print(msg)
            return msg

        print(f"\n  Loaded {len(self._companies)} company/companies: "
              f"{', '.join(self._companies.keys())}")

        # ── Benchmark (median + quartiles + scale warning) ────────────────────
        bench_result      = self._benchmarker.compute(
            self._companies, mode=self.mode,
            focal=self.focal, peer=self.peer,
        )

        self._rankings_df = bench_result["rankings"]

        # ── Gap analysis (percentile-aware) ───────────────────────────────────
        self._gap_df = self._gap_agent.analyze(
            self._companies,
            bench_result["benchmarks"],
            quartiles    = bench_result.get("quartiles"),
            peer_counts  = bench_result.get("peer_counts"),
            rankings_df  = self._rankings_df,
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
            esg_scores    = esg_scores,
        )
        print(report)

        # ── NL queries ────────────────────────────────────────────────────────
        if nl_queries:
            print("\n" + "═" * 62)
            print("  NATURAL LANGUAGE QUERIES")
            print("═" * 62)
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

    def _load_file(self, filepath: str) -> Optional[tuple[str, dict]]:
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
            raw    = self._extractor.extract(parsed)
            normed = self._normalizer.normalize(raw)
        except Exception as e:
            print(f"  ✗  EXTRACTION ERROR — skipping {name}: {e}")
            return None
        return name, normed