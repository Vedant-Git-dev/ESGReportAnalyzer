"""
models/db_models.py

Full PostgreSQL schema for the ESG pipeline.

Tables
------
companies           — tracked companies
reports             — discovered / uploaded ESG reports  [revenue_cr added]
parsed_documents    — parse cache keyed on (report_id, parser_version)
document_chunks     — chunked text/table segments, retrieval unit
kpi_definitions     — config-driven KPI catalogue
kpi_records         — extracted KPI values, append-only audit log

Change log:
  2024-xx — Added revenue_cr, revenue_unit, revenue_source to reports table
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import relationship

from core.database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# companies
# ---------------------------------------------------------------------------
class Company(Base):
    __tablename__ = "companies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    ticker = Column(String(20), nullable=True)
    sector = Column(String(100), nullable=True)
    industry = Column(String(100), nullable=True)
    country = Column(String(100), nullable=True, default="India")
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_now, onupdate=_now)

    reports = relationship("Report", back_populates="company", cascade="all, delete-orphan")
    kpi_records = relationship("KPIRecord", back_populates="company")

    __table_args__ = (
        Index("ix_companies_name", "name"),
        Index("ix_companies_sector", "sector"),
    )

    def __repr__(self) -> str:
        return f"<Company {self.name} ({self.ticker})>"


# ---------------------------------------------------------------------------
# reports  (revenue columns added)
# ---------------------------------------------------------------------------
class Report(Base):
    __tablename__ = "reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    company_id = Column(
        UUID(as_uuid=True), ForeignKey("companies.id", ondelete="CASCADE"), nullable=False
    )
    report_year = Column(Integer, nullable=False)
    report_type = Column(String(50), nullable=False, default="ESG")
    source_url = Column(Text, nullable=True)
    file_path = Column(Text, nullable=True)
    file_hash = Column(String(64), nullable=True)
    file_size_bytes = Column(BigInteger, nullable=True)
    status = Column(String(30), nullable=False, default="discovered")
    error_message = Column(Text, nullable=True)
    discovered_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    downloaded_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_now, onupdate=_now)

    # ── Revenue columns (new) ─────────────────────────────────────────────────
    # Stored directly on the report so we don't need a separate KPI definition
    # for revenue, keeping it conceptually separate from ESG KPIs.
    revenue_cr = Column(Float, nullable=True)          # INR Crore
    revenue_unit = Column(String(30), nullable=True)   # "INR_Crore" | "USD_Million"
    revenue_source = Column(String(50), nullable=True) # "regex" | "llm" | "manual"

    company = relationship("Company", back_populates="reports")
    parsed_documents = relationship(
        "ParsedDocument", back_populates="report", cascade="all, delete-orphan"
    )
    kpi_records = relationship("KPIRecord", back_populates="report")

    __table_args__ = (
        UniqueConstraint("company_id", "source_url", name="uq_report_company_url"),
        Index("ix_reports_status", "status"),
        Index("ix_reports_company_id", "company_id"),
        Index("ix_reports_company_year", "company_id", "report_year"),
    )

    def __repr__(self) -> str:
        return f"<Report {self.report_type} {self.report_year} status={self.status}>"


# ---------------------------------------------------------------------------
# parsed_documents  (parse cache)
# ---------------------------------------------------------------------------
class ParsedDocument(Base):
    __tablename__ = "parsed_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_id = Column(
        UUID(as_uuid=True), ForeignKey("reports.id", ondelete="CASCADE"), nullable=False
    )
    parser_version = Column(String(20), nullable=False)
    page_count = Column(Integer, nullable=True)
    parsed_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    meta = Column(JSONB, nullable=False, default=dict)

    report = relationship("Report", back_populates="parsed_documents")
    chunks = relationship(
        "DocumentChunk", back_populates="parsed_document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("report_id", "parser_version", name="uq_parsed_report_version"),
        Index("ix_parsed_documents_report_id", "report_id"),
    )

    def __repr__(self) -> str:
        return f"<ParsedDocument report={self.report_id} v={self.parser_version}>"


# ---------------------------------------------------------------------------
# document_chunks
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 384


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    parsed_document_id = Column(
        UUID(as_uuid=True), ForeignKey("parsed_documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index = Column(Integer, nullable=False)
    chunk_type = Column(String(20), nullable=False, default="text")
    page_number = Column(Integer, nullable=True)
    content = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=True)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=True)
    keywords = Column(Text, nullable=True)
    is_embedded = Column(Boolean, nullable=False, default=False)

    parsed_document = relationship("ParsedDocument", back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_parsed_document_id", "parsed_document_id"),
        Index("ix_chunks_chunk_type", "chunk_type"),
    )

    def __repr__(self) -> str:
        return f"<Chunk {self.chunk_index} type={self.chunk_type} tokens={self.token_count}>"


# ---------------------------------------------------------------------------
# kpi_definitions
# ---------------------------------------------------------------------------
class KPIDefinition(Base):
    __tablename__ = "kpi_definitions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, unique=True)
    display_name = Column(String(150), nullable=False)
    category = Column(String(50), nullable=False)
    subcategory = Column(String(100), nullable=True)
    expected_unit = Column(String(50), nullable=False)
    regex_patterns = Column(JSONB, nullable=False, default=list)
    retrieval_keywords = Column(JSONB, nullable=False, default=list)
    valid_min = Column(Float, nullable=True)
    valid_max = Column(Float, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)

    kpi_records = relationship("KPIRecord", back_populates="kpi_definition")

    __table_args__ = (
        Index("ix_kpi_definitions_category", "category"),
        Index("ix_kpi_definitions_is_active", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<KPIDef {self.name} [{self.expected_unit}]>"


# ---------------------------------------------------------------------------
# kpi_records  (append-only)
# ---------------------------------------------------------------------------
class KPIRecord(Base):
    __tablename__ = "kpi_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    company_id = Column(
        UUID(as_uuid=True), ForeignKey("companies.id", ondelete="CASCADE"), nullable=False
    )
    report_id = Column(
        UUID(as_uuid=True), ForeignKey("reports.id", ondelete="CASCADE"), nullable=False
    )
    kpi_definition_id = Column(
        UUID(as_uuid=True), ForeignKey("kpi_definitions.id"), nullable=False
    )
    report_year = Column(Integer, nullable=False)
    raw_value = Column(Text, nullable=True)
    normalized_value = Column(Float, nullable=True)
    unit = Column(String(50), nullable=True)
    extraction_method = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=True)
    source_chunk_id = Column(
        UUID(as_uuid=True), ForeignKey("document_chunks.id"), nullable=True
    )
    is_validated = Column(Boolean, nullable=False, default=False)
    validation_notes = Column(Text, nullable=True)
    extracted_at = Column(DateTime(timezone=True), nullable=False, default=_now)

    company = relationship("Company", back_populates="kpi_records")
    report = relationship("Report", back_populates="kpi_records")
    kpi_definition = relationship("KPIDefinition", back_populates="kpi_records")

    __table_args__ = (
        Index("ix_kpi_records_company_kpi_year", "company_id", "kpi_definition_id", "report_year"),
        Index("ix_kpi_records_report_id", "report_id"),
        Index("ix_kpi_records_extraction_method", "extraction_method"),
    )

    def __repr__(self) -> str:
        return (
            f"<KPIRecord kpi={self.kpi_definition_id} "
            f"value={self.normalized_value} {self.unit}>"
        )