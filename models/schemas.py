"""
models/schemas.py
Pydantic v2 schemas — used by API layer and as typed data contracts between agents.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


# ---------------------------------------------------------------------------
# Company
# ---------------------------------------------------------------------------
class CompanyCreate(BaseModel):
    name: str
    ticker: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: str = "India"


class CompanyRead(CompanyCreate):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    is_active: bool
    created_at: datetime


# ---------------------------------------------------------------------------
# Report discovery result (from Tavily)
# ---------------------------------------------------------------------------
class DiscoveredReport(BaseModel):
    """Intermediate result from the search service — not yet in DB."""
    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None
    score: float = 0.0
    query_source: str = ""  # which query variant found this


class SearchResult(BaseModel):
    company_name: str
    year: int
    discovered: list[DiscoveredReport] = []
    total_found: int = 0
    queries_run: int = 0


# ---------------------------------------------------------------------------
# Report (DB-backed)
# ---------------------------------------------------------------------------
class ReportCreate(BaseModel):
    company_id: uuid.UUID
    report_year: int
    report_type: str = "ESG"
    source_url: Optional[str] = None


class ReportRead(ReportCreate):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    status: str
    file_path: Optional[str] = None
    file_hash: Optional[str] = None
    file_size_bytes: Optional[int] = None
    error_message: Optional[str] = None
    discovered_at: datetime
    downloaded_at: Optional[datetime] = None


class ReportStatusUpdate(BaseModel):
    status: str
    error_message: Optional[str] = None
    file_path: Optional[str] = None
    file_hash: Optional[str] = None
    file_size_bytes: Optional[int] = None


# ---------------------------------------------------------------------------
# Parse cache
# ---------------------------------------------------------------------------
class ParsedDocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    report_id: uuid.UUID
    parser_version: str
    page_count: Optional[int]
    parsed_at: datetime
    meta: dict[str, Any]


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------
class ChunkRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    chunk_index: int
    chunk_type: str
    page_number: Optional[int]
    content: str
    token_count: Optional[int]


# ---------------------------------------------------------------------------
# KPI Definition
# ---------------------------------------------------------------------------
class KPIDefinitionCreate(BaseModel):
    name: str
    display_name: str
    category: str
    subcategory: Optional[str] = None
    expected_unit: str
    regex_patterns: list[str] = []
    retrieval_keywords: list[str] = []
    valid_min: Optional[float] = None
    valid_max: Optional[float] = None


class KPIDefinitionRead(KPIDefinitionCreate):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    is_active: bool
    created_at: datetime


# ---------------------------------------------------------------------------
# KPI Extraction result (inter-agent contract)
# ---------------------------------------------------------------------------
class ExtractedKPI(BaseModel):
    kpi_name: str
    raw_value: Optional[str] = None
    normalized_value: Optional[float] = None
    unit: Optional[str] = None
    year: Optional[int] = None
    extraction_method: str = "regex"  # regex | llm | manual
    confidence: float = 1.0
    source_chunk_id: Optional[uuid.UUID] = None
    validation_passed: bool = True
    validation_notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
class BenchmarkRequest(BaseModel):
    company_ids: list[uuid.UUID] = Field(min_length=1)
    kpi_names: list[str] = Field(min_length=1)
    year: int


class BenchmarkEntry(BaseModel):
    company_id: uuid.UUID
    company_name: str
    kpi_name: str
    normalized_value: Optional[float]
    unit: Optional[str]
    rank: Optional[int] = None
    percentile: Optional[float] = None
    gap_to_leader: Optional[float] = None


class BenchmarkResult(BaseModel):
    year: int
    kpi_names: list[str]
    entries: list[BenchmarkEntry]