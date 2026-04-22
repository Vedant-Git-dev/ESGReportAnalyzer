"""
api/main.py

FastAPI application entry point for the ESG Competitive Intelligence backend.

Replaces the Streamlit ui.py as the backend server.
Zero changes to any agent, service, model, or core module.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Or via convenience script:
    python run_api.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so existing imports work unchanged
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import health, companies, compare, upload, export

app = FastAPI(
    title="ESG Competitive Intelligence API",
    description=(
        "Backend API for the ESG benchmarking pipeline. "
        "Exposes ingestion, extraction, caching, and benchmarking operations "
        "previously embedded in the Streamlit dashboard."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# ---------------------------------------------------------------------------
# CORS — allow the React dev server (port 5173) and any production origin
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",   # CRA dev server (fallback)
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(health.router,    prefix="/api")
app.include_router(companies.router, prefix="/api")
app.include_router(compare.router,   prefix="/api")
app.include_router(upload.router,    prefix="/api")
app.include_router(export.router,    prefix="/api")


@app.get("/", include_in_schema=False)
def root():
    return {
        "service": "ESG Competitive Intelligence API",
        "version": "1.0.0",
        "docs": "/api/docs",
    }