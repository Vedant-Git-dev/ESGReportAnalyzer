"""

main.py

---

To enable Groq LLM insights, set the environment variable:
    export GROQ_API_KEY=gsk_...
    python main.py
"""

from engine import ESGBenchmarkingEngine

# ── Input XML files (add / remove as needed) ─────────────────────────────────
FILES = [
    "./data/bpcl.xml",
    "./data/hp.xml",
    "./data/indianoil.xml",
    # "./data/cpcl.xml",
    "./data/mrpl.xml",
    "./data/reliance.xml",
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