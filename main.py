"""
main.py
-------
Entry point for the ESG Benchmarking Analytics Engine.

Edit FILES and QUERIES below, then run:
    cd esg_engine
    python main.py

To enable Groq LLM insights, set the environment variable:
    export GROQ_API_KEY=gsk_...
    python main.py
"""

from engine import ESGBenchmarkingEngine

# ── Input XML files (add / remove as needed) ─────────────────────────────────
FILES = [
    "./data/castrol.xml",
    "./data/bpcl.xml",
    "./data/indianoil.xml"
]

# ── Natural-language queries (rule-based or Groq-powered) ────────────────────
QUERIES = [
    "Which company leads in gender diversity among permanent employees?"
]

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = ESGBenchmarkingEngine(
        mode="industry",
        output_dir="output",
    )
    engine.run(FILES, nl_queries=QUERIES)