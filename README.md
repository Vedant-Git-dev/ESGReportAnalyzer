# ESG Competitive Intelligence Pipeline

Production-grade multi-agent pipeline for extracting and benchmarking ESG KPIs from PDF reports.

---

**Core principle:** Decouple parsing (expensive, once) from extraction (cheap, repeatable).
The parse cache is keyed on `(report_id, parser_version)` — bump the version to force re-parse.

---

## Project Structure (Plan)

```
esg_pipeline/
├── agents/
│   ├── ingestion_agent.py      # Phase 1 — discover + download
│   ├── parsing_agent.py        # Phase 2 — PDF → structured chunks
│   ├── chunking_agent.py       # Phase 3 — chunk + keyword index
│   ├── extraction_agent.py     # Phase 4 — regex → LLM → validate
│   ├── normalization_agent.py  # Phase 5 — unit conversion
│   └── benchmarking_agent.py   # Phase 6 — ranking + gap analysis
├── services/
│   ├── search_service.py       # Tavily multi-query search
│   ├── parse_cache.py          # Cache read/write
│   ├── retrieval_service.py    # Keyword + embedding chunk retrieval
│   ├── kpi_service.py          # KPI definition CRUD
│   └── llm_service.py          # Abstracted LLM client
├── core/
│   ├── config.py               # Settings (pydantic-settings)
│   ├── database.py             # Engine + session factory
│   └── logging_config.py       # structlog setup
├── models/
│   ├── db_models.py            # SQLAlchemy ORM
│   └── schemas.py              # Pydantic v2 schemas
├── api/
│   └── routes.py               # FastAPI routes
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── storage/pdfs/               # Downloaded PDFs (gitignored)
├── logs/                       # Log files (gitignored)
├── main.py                     # CLI entrypoint
├── requirements.txt
└── .env.example
```

---

## Setup

### 1. Prerequisites

```bash
python 3.11+
PostgreSQL 15+
```

### 2. Install dependencies

```bash
cd esg_pipeline
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env — set DATABASE_URL, TAVILY_API_KEY, LLM_API_KEY
```

### 4. Initialise database

```bash
python main.py init-db
python main.py seed-kpis
```

---

## Phase 1 Test Flow

```bash
# Add a company + discover + download
python main.py ingest --company "Infosys" --year 2023 --sector "Technology"

# Verify
python main.py list-companies
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Parse cache keyed on `(report_id, parser_version)` | Parse once, extract many times. Bump version to invalidate. |
| Config-driven `kpi_definitions` table | Add KPIs via DB insert — zero code changes |
| Regex-first extraction | 100% coverage for well-formatted PDFs, zero API cost |
| Top 3–7 chunks to LLM only | Hard limit — never send full document |
| Append-only `kpi_records` | Full audit trail of every extraction |
| No Celery / queues | Synchronous, debuggable, simple to deploy |

---

## Environment Variables

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `TAVILY_API_KEY` | Tavily search API key |
| `LLM_API_KEY` | LLM API key |
| `LLM_MODEL` | Model name (e.g. `llama3-70b-8192`) |
| `PARSER_VERSION` | Bump to invalidate parse cache |
| `PDF_STORAGE_PATH` | Where to store downloaded PDFs |
