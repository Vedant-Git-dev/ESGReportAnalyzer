"""
services/kpi_service.py

KPI Service — manages KPIDefinition reads and KPIRecord writes.

Responsibilities:
  - Fetch active KPI definitions (with caching)
  - Store extracted KPI records (append-only)
  - Query existing records for a report
"""

from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy.orm import Session

from core.logging_config import get_logger
from models.db_models import KPIDefinition, KPIRecord
from models.schemas import ExtractedKPI

logger = get_logger(__name__)

class KPIService:

    def get_all_active(self, db: Session) -> list[KPIDefinition]:
        """Return all active KPI definitions, ordered by category."""
        return (
            db.query(KPIDefinition)
            .filter(KPIDefinition.is_active == True)
            .order_by(KPIDefinition.category, KPIDefinition.name)
            .all()
        )

    def get_by_name(self, name: str, db: Session) -> Optional[KPIDefinition]:
        return db.query(KPIDefinition).filter(KPIDefinition.name == name).first()

    def get_by_names(self, names: list[str], db: Session) -> list[KPIDefinition]:
        return (
            db.query(KPIDefinition)
            .filter(KPIDefinition.name.in_(names))
            .all()
        )

    def store_record(
        self,
        company_id: uuid.UUID,
        report_id: uuid.UUID,
        report_year: int,
        kpi_definition_id: uuid.UUID,
        extracted: ExtractedKPI,
        source_chunk_id: Optional[uuid.UUID],
        db: Session,
    ) -> KPIRecord:
        """
        Append-only insert of a KPI extraction result.
        Multiple records for the same KPI are allowed (full audit trail).
        """
        record = KPIRecord(
            company_id=company_id,
            report_id=report_id,
            kpi_definition_id=kpi_definition_id,
            report_year=report_year,
            raw_value=extracted.raw_value,
            normalized_value=extracted.normalized_value,
            unit=extracted.unit,
            extraction_method=extracted.extraction_method,
            confidence=extracted.confidence,
            source_chunk_id=source_chunk_id,
            is_validated=extracted.validation_passed,
            validation_notes=extracted.validation_notes,
        )
        db.add(record)
        db.flush()
        logger.info(
            "kpi_service.record_stored",
            kpi=extracted.kpi_name,
            value=extracted.normalized_value,
            unit=extracted.unit,
            method=extracted.extraction_method,
        )
        return record

    def get_records_for_report(
        self,
        report_id: uuid.UUID,
        db: Session,
    ) -> list[KPIRecord]:
        """Fetch all KPI records for a report, newest first."""
        return (
            db.query(KPIRecord)
            .filter(KPIRecord.report_id == report_id)
            .order_by(KPIRecord.extracted_at.desc())
            .all()
        )

    def get_latest_record(
        self,
        company_id: uuid.UUID,
        kpi_definition_id: uuid.UUID,
        report_year: int,
        db: Session,
    ) -> Optional[KPIRecord]:
        """Return the most recent validated record for a company/KPI/year."""
        return (
            db.query(KPIRecord)
            .filter(
                KPIRecord.company_id == company_id,
                KPIRecord.kpi_definition_id == kpi_definition_id,
                KPIRecord.report_year == report_year,
                KPIRecord.is_validated == True,
            )
            .order_by(KPIRecord.extracted_at.desc())
            .first()
        )