"""
agents/ingestion_agent.py

Phase 1 agent — owns the full ingestion lifecycle:

  1. Ensure company exists in DB
  2. Call SearchService to discover PDF report URLs
  3. Deduplicate against already-known reports
  4. Download PDFs (with size + hash guard)
  5. Persist report metadata + update status

Does NOT parse PDFs — that is the parsing agent's job.
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from sqlalchemy.orm import Session

from core.config import get_settings
from core.database import get_db
from core.logging_config import get_logger
from models.db_models import Company, Report
from models.schemas import (
    CompanyCreate,
    CompanyRead,
    DiscoveredReport,
    ReportRead,
    SearchResult,
)
from services.search_service import search_reports

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_pdf(url: str, dest: Path, timeout: int, max_mb: int) -> tuple[int, str]:
    """
    Stream-download a PDF to `dest`.
    Returns (file_size_bytes, sha256_hash).
    Raises ValueError if content-type is wrong or file exceeds size limit.
    """
    max_bytes = max_mb * 1024 * 1024

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            # lenient check — some servers return application/octet-stream
            if "html" in content_type:
                raise ValueError(f"Server returned HTML, not a PDF: {content_type}")

            dest.parent.mkdir(parents=True, exist_ok=True)
            size = 0
            with dest.open("wb") as f:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    size += len(chunk)
                    if size > max_bytes:
                        dest.unlink(missing_ok=True)
                        raise ValueError(f"PDF exceeds size limit ({max_mb} MB)")
                    f.write(chunk)

    sha = _sha256(dest)
    return size, sha


# ---------------------------------------------------------------------------
# Public Agent API
# ---------------------------------------------------------------------------

class IngestionAgent:
    """
    Stateless agent — all state lives in the DB and filesystem.
    Instantiate fresh per call or reuse across calls.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    # --- Company management ---------------------------------------------------

    def get_or_create_company(self, data: CompanyCreate) -> CompanyRead:
        """
        Idempotent: if a company with the same name (case-insensitive) exists,
        return it. Otherwise insert.
        """
        with get_db() as db:
            existing = (
                db.query(Company)
                .filter(Company.name.ilike(data.name))
                .first()
            )
            if existing:
                logger.info("ingestion.company_exists", name=existing.name, id=str(existing.id))
                return CompanyRead.model_validate(existing)

            company = Company(
                name=data.name,
                ticker=data.ticker,
                sector=data.sector,
                industry=data.industry,
                country=data.country,
            )
            db.add(company)
            db.flush()
            logger.info("ingestion.company_created", name=company.name, id=str(company.id))
            return CompanyRead.model_validate(company)

    # --- Discovery ------------------------------------------------------------

    def discover_reports(self, company_name: str, year: int, report_type: str = "BRSR") -> SearchResult:
        """
        Run Tavily multi-query search and return discovered PDF links.
        Does NOT persist anything — caller decides which to download.
        """
        logger.info("ingestion.discover_start", company=company_name, year=year, report_type=report_type)
        result = search_reports(company_name, year, report_type=report_type)
        logger.info("ingestion.discover_complete", company=company_name, year=year, found=result.total_found)
        return result

    # --- Persistence of discovered reports ------------------------------------

    def register_discovered_report(
        self,
        company_id: uuid.UUID,
        discovered: DiscoveredReport,
        year: int,
        report_type: str = "ESG",
    ) -> ReportRead:
        """
        Insert a discovered report into DB with status=discovered.
        Skips if a report with the same URL already exists for this company+year.
        """
        with get_db() as db:
            existing = (
                db.query(Report)
                .filter(
                    Report.company_id == company_id,
                    Report.source_url == discovered.url,
                )
                .first()
            )
            if existing:
                logger.debug("ingestion.report_already_known", url=discovered.url)
                return ReportRead.model_validate(existing)

            report = Report(
                company_id=company_id,
                report_year=year,
                report_type=report_type,
                source_url=discovered.url,
                status="discovered",
            )
            db.add(report)
            db.flush()
            logger.info("ingestion.report_registered", url=discovered.url, id=str(report.id))
            return ReportRead.model_validate(report)

    # --- Download -------------------------------------------------------------

    def download_report(self, report_id: uuid.UUID) -> ReportRead:
        """
        Download the PDF for a registered report.
        Updates report status to downloaded | failed.
        Skips if already downloaded (idempotent).
        """
        with get_db() as db:
            report = db.query(Report).filter(Report.id == report_id).first()
            if not report:
                raise ValueError(f"Report {report_id} not found")

            if report.status == "downloaded":
                logger.info("ingestion.already_downloaded", report_id=str(report_id))
                return ReportRead.model_validate(report)

            if not report.source_url:
                report.status = "failed"
                report.error_message = "No source URL"
                return ReportRead.model_validate(report)

            dest = (
                self.settings.pdf_storage_path
                / str(report.company_id)
                / f"{report.report_year}_{report.report_type}_{report.id}.pdf"
            )

            logger.info("ingestion.download_start", url=report.source_url, dest=str(dest))
            report.status = "downloading"
            db.flush()

            try:
                size, sha = _download_pdf(
                    url=report.source_url,
                    dest=dest,
                    timeout=self.settings.download_timeout_seconds,
                    max_mb=self.settings.max_download_size_mb,
                )
                report.status = "downloaded"
                report.file_path = str(dest)
                report.file_hash = sha
                report.file_size_bytes = size
                report.downloaded_at = datetime.now(timezone.utc)
                logger.info("ingestion.download_complete", size_mb=round(size / 1e6, 2), sha=sha[:8])

            except Exception as exc:
                report.status = "failed"
                report.error_message = str(exc)
                logger.error("ingestion.download_failed", error=str(exc), url=report.source_url)

            db.flush()
            return ReportRead.model_validate(report)

    # --- Orchestration -------------------------------------------------------

    def run(
        self,
        company_data: CompanyCreate,
        year: int,
        report_type: str = "ESG",
        auto_download: bool = True,
        max_downloads: int = 1,
    ) -> dict:
        """
        Full Phase 1 pipeline for one company + year:
          1. get_or_create_company
          2. discover_reports via Tavily
          3. register each discovered PDF
          4. download top-N (if auto_download=True)

        Returns summary dict with company, search result, and downloaded report IDs.
        """
        logger.info("ingestion.run_start", company=company_data.name, year=year)

        company = self.get_or_create_company(company_data)
        search_result = self.discover_reports(company_data.name, year, report_type=report_type)

        registered: list[ReportRead] = []
        for disc in search_result.discovered:
            r = self.register_discovered_report(company.id, disc, year, report_type)
            registered.append(r)

        downloaded: list[ReportRead] = []
        if auto_download and registered:
            for report in registered[:max_downloads]:
                dl = self.download_report(report.id)
                downloaded.append(dl)

        logger.info(
            "ingestion.run_complete",
            company=company_data.name,
            discovered=len(registered),
            downloaded=len(downloaded),
        )

        return {
            "company": company,
            "search_result": search_result,
            "registered_reports": registered,
            "downloaded_reports": downloaded,
        }