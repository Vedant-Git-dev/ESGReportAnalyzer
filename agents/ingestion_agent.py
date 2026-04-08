"""
agents/ingestion_agent.py

Phase 1 agent: company creation, report discovery, registration, download.

Deduplication contract
----------------------
Search-level deduplication (global across all queries and types):
    Handled in search_service.collect_and_classify(). Each URL appears in
    at most one type's result list.

Type-level deduplication (do not download the same type twice):
    Enforced in run_multi_report_types(). For each type, exactly one download
    is attempted. If the download succeeds, no further downloads are attempted
    for that type, even if multiple URLs were classified for it.

URL-level idempotency (do not re-download a URL already in DB):
    Enforced in download_report(). If a report row already has
    status='downloaded', it is returned immediately without re-downloading.

Content-level deduplication is NOT performed here. That is the parse layer's
responsibility (keyed on (report_id, parser_version) as it was originally).

Filename convention
-------------------
Every downloaded PDF is named:
    {year}_{TYPE}_{company_slug}_{first_8_chars_of_report_uuid}.pdf

The TYPE comes from the classification result in search_service — not from
which query template found the URL. So if a URL was classified as BRSR based
on its path keywords, the file will be named 2025_BRSR_infosys_a3f19c2b.pdf
regardless of which query template surfaced it.

Examples:
    2025_BRSR_infosys_limited_a3f19c2b.pdf
    2025_ESG_tata_consultancy_services_b7d04e1a.pdf
    2024_Integrated_wipro_c2e85f33.pdf

Public API
----------
IngestionAgent.get_or_create_company(data)
IngestionAgent.register_discovered_report(company_id, discovered, year, report_type)
IngestionAgent.download_report(report_id, company_name)
IngestionAgent.ingest_uploaded_pdf(source_path, company_name, year, ...)
IngestionAgent.run(company_data, year, report_type)           -- single type
IngestionAgent.run_multi_report_types(company_data, year)     -- all three types
"""
from __future__ import annotations

import hashlib
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

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
from services.search_service import (
    ALL_REPORT_TYPES,
    collect_and_classify,
    search_reports,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

# Replaces any run of characters that are not lowercase letters or digits
# with a single underscore. Used to build the company slug in filenames.
_NON_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _company_slug(name: str) -> str:
    """
    Convert a company name to a filesystem-safe lowercase slug.

    Examples:
        "Infosys Limited"           -> "infosys_limited"
        "Tata Consultancy Services" -> "tata_consultancy_services"
        "L&T Finance Holdings"      -> "l_t_finance_holdings"

    Capped at 40 characters to keep filenames readable.
    """
    slug = _NON_SLUG_RE.sub("_", name.lower()).strip("_")
    return slug[:40]


def _make_pdf_filename(
    year: int,
    report_type: str,
    company_name: str,
    report_id: uuid.UUID,
) -> str:
    """
    Build a human-readable, filesystem-safe PDF filename.

    Format:
        {year}_{TYPE}_{company_slug}_{first_8_uuid_chars}.pdf

    The report_type here is the CLASSIFIED type from search_service, not
    the query type. This ensures the filename reflects the actual content
    of the PDF as determined by URL keyword analysis.

    Examples:
        2025_BRSR_infosys_limited_a3f19c2b.pdf
        2024_ESG_wipro_b9e22d4f.pdf
        2025_Integrated_hcl_technologies_c1d03a11.pdf
    """
    slug       = _company_slug(company_name)
    type_label = report_type.strip().replace(" ", "_").replace("-", "_").upper()
    short_id   = str(report_id)[:8]
    return f"{year}_{type_label}_{slug}_{short_id}.pdf"


# ---------------------------------------------------------------------------
# SHA-256 helper
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    """
    Compute the SHA-256 hex digest of a file by reading it in 64 KB chunks.
    Memory-safe for large PDFs (some ESG reports are 80+ MB).
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# HTTP download
# ---------------------------------------------------------------------------

# Three browser User-Agent strings rotated on 4xx/5xx errors.
# Corporate IR portals (NSE, BSE, company websites) frequently reject
# requests without a realistic User-Agent header (returning 403 or 406).
_USER_AGENTS: list[str] = [
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) "
        "Gecko/20100101 Firefox/125.0"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
]

# Headers that mimic a real browser navigating to a PDF document.
_DOWNLOAD_HEADERS: dict[str, str] = {
    "Accept":                    "application/pdf,application/octet-stream,*/*;q=0.9",
    "Accept-Language":           "en-US,en;q=0.9",
    "Accept-Encoding":           "gzip, deflate, br",
    "Connection":                "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest":            "document",
    "Sec-Fetch-Mode":            "navigate",
    "Sec-Fetch-Site":            "none",
    "Cache-Control":             "no-cache",
}


def _stream_to_file(
    url: str,
    dest: Path,
    timeout_seconds: int,
    max_size_bytes: int,
) -> tuple[int, str]:
    """
    Stream-download a URL to dest with User-Agent rotation.

    Returns:
        (bytes_written, sha256_hex_digest)

    Raises:
        ValueError for hard failures that should not be retried:
            - HTTP 400/404/410 (resource gone or invalid)
            - Server returned HTML instead of PDF (login wall)
            - File exceeds size limit
            - Downloaded bytes are not a valid PDF (%PDF- magic check)

        The caller should catch ValueError, mark the report as 'failed',
        and try the next candidate URL.
    """
    last_exc: Exception = RuntimeError("No download attempts were made.")

    for ua in _USER_AGENTS:
        headers = {**_DOWNLOAD_HEADERS, "User-Agent": ua}

        try:
            with httpx.Client(
                timeout=timeout_seconds,
                follow_redirects=True,
                headers=headers,
            ) as client:
                with client.stream("GET", url) as response:
                    response.raise_for_status()

                    # Reject HTML responses — these are login walls or
                    # investor portal landing pages, not the PDF itself.
                    content_type = response.headers.get("content-type", "")
                    if "text/html" in content_type and "pdf" not in content_type:
                        raise ValueError(
                            f"Server returned HTML instead of PDF "
                            f"(Content-Type: {content_type!r}). "
                            f"The URL likely requires authentication. URL: {url}"
                        )

                    # Stream to disk in 64 KB chunks. Abort if the size
                    # limit is exceeded to avoid filling the disk.
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    bytes_written = 0
                    with dest.open("wb") as out_fh:
                        for chunk in response.iter_bytes(chunk_size=65536):
                            bytes_written += len(chunk)
                            if bytes_written > max_size_bytes:
                                dest.unlink(missing_ok=True)
                                raise ValueError(
                                    f"Download aborted: file size exceeded "
                                    f"{max_size_bytes // (1024 * 1024)} MB limit. "
                                    f"URL: {url}"
                                )
                            out_fh.write(chunk)

            # Validate PDF magic bytes after the download is complete.
            # %PDF- is the standard magic number for all PDF versions (1.0 through 2.0).
            with dest.open("rb") as fh:
                magic = fh.read(5)
            if magic != b"%PDF-":
                dest.unlink(missing_ok=True)
                raise ValueError(
                    f"Downloaded file is not a valid PDF "
                    f"(magic bytes: {magic!r}). URL: {url}"
                )

            sha256 = _sha256_file(dest)
            return bytes_written, sha256

        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            logger.warning(
                "ingestion.download_http_error",
                status=status,
                url=url[:80],
                user_agent_prefix=ua[:40],
            )
            # 400/404/410: resource is gone or the URL is malformed.
            # No point rotating UA for these — raise immediately.
            if status in (400, 404, 410):
                raise ValueError(f"HTTP {status} for URL: {url}") from exc
            # 403/406/429/5xx: may be UA-based; try next UA.
            last_exc = exc
            continue

        except ValueError:
            # Propagate validation errors (HTML response, size limit, bad magic).
            raise

        except Exception as exc:
            logger.warning(
                "ingestion.download_network_error",
                error=str(exc),
                url=url[:80],
            )
            last_exc = exc
            continue

    raise ValueError(
        f"Download failed after trying {len(_USER_AGENTS)} User-Agents. "
        f"Last error: {last_exc}. URL: {url}"
    )


# ---------------------------------------------------------------------------
# Public Agent
# ---------------------------------------------------------------------------

class IngestionAgent:
    """
    Stateless Phase 1 agent. All persistent state lives in the DB and
    on the filesystem. Safe to instantiate multiple times or reuse.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    # ------------------------------------------------------------------
    # Company
    # ------------------------------------------------------------------

    def get_or_create_company(self, data: CompanyCreate) -> CompanyRead:
        """
        Look up an existing company by name (case-insensitive) or create one.
        Idempotent: safe to call multiple times for the same company.
        """
        with get_db() as db:
            existing = (
                db.query(Company)
                .filter(Company.name.ilike(data.name))
                .first()
            )
            if existing:
                logger.info(
                    "ingestion.company_exists",
                    name=existing.name,
                    id=str(existing.id),
                )
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
            logger.info(
                "ingestion.company_created",
                name=company.name,
                id=str(company.id),
            )
            return CompanyRead.model_validate(company)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_discovered_report(
        self,
        company_id:  uuid.UUID,
        discovered:  DiscoveredReport,
        year:        int,
        report_type: str,
    ) -> ReportRead:
        """
        Insert one classified URL into the reports table with status='discovered'.

        Idempotent on (company_id, source_url): if the same URL was already
        registered (from a previous pipeline run), the existing row is returned
        unchanged. This prevents duplicate rows on re-runs.

        The report_type stored here is the CLASSIFIED type from search_service,
        not a guess based on which query returned the URL.
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
                logger.debug(
                    "ingestion.url_already_registered",
                    url=discovered.url[:80],
                    existing_type=existing.report_type,
                )
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
            logger.info(
                "ingestion.report_registered",
                report_type=report_type,
                url=discovered.url[:80],
                report_id=str(report.id)[:8],
            )
            return ReportRead.model_validate(report)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_report(
        self,
        report_id:    uuid.UUID,
        company_name: str,
    ) -> ReportRead:
        """
        Download the PDF for a registered report and write it to disk.

        Behavior:
        - Returns immediately (no network call) if status is already 'downloaded'.
        - Marks status as 'downloading' before the network call so that
          concurrent runs or crash recovery can detect interrupted state.
        - Writes file to: {storage_path}/{company_id}/{year}_{TYPE}_{slug}_{id8}.pdf
        - On success: sets status='downloaded', file_path, file_hash, file_size_bytes.
        - On failure: sets status='failed', error_message. Does NOT raise;
          the caller decides whether to try the next candidate URL.

        The file_hash stored is the SHA-256 of the downloaded file. This is
        used by the parse layer's (report_id, parser_version) cache but NOT
        for cross-file deduplication here.

        Args:
            report_id:    UUID of a Report row with status != 'downloaded'.
            company_name: Used only for building the filename. Does not need
                          to exactly match the DB company name.

        Returns:
            Updated ReportRead. Inspect .status to determine outcome.
        """
        with get_db() as db:
            report = db.query(Report).filter(Report.id == report_id).first()
            if not report:
                raise ValueError(f"Report not found: {report_id}")

            # Idempotency: already downloaded in a previous run.
            if report.status == "downloaded":
                logger.info(
                    "ingestion.already_downloaded",
                    report_id=str(report_id)[:8],
                    file=report.file_path,
                )
                return ReportRead.model_validate(report)

            if not report.source_url:
                report.status        = "failed"
                report.error_message = "No source_url set on this report row."
                db.flush()
                return ReportRead.model_validate(report)

            # Build deterministic destination path.
            # The TYPE in the filename comes from the DB row's report_type,
            # which was set by register_discovered_report() using the
            # classification result from search_service.
            filename = _make_pdf_filename(
                year=report.report_year,
                report_type=report.report_type,
                company_name=company_name,
                report_id=report.id,
            )
            dest_path = (
                self.settings.pdf_storage_path
                / str(report.company_id)
                / filename
            )

            # Signal that a download is in progress.
            report.status = "downloading"
            db.flush()

            try:
                bytes_written, sha256 = _stream_to_file(
                    url=report.source_url,
                    dest=dest_path,
                    timeout_seconds=self.settings.download_timeout_seconds,
                    max_size_bytes=self.settings.max_download_size_mb * 1024 * 1024,
                )
            except Exception as exc:
                report.status        = "failed"
                report.error_message = str(exc)
                db.flush()
                logger.error(
                    "ingestion.download_failed",
                    report_type=report.report_type,
                    url=report.source_url[:80],
                    error=str(exc),
                )
                return ReportRead.model_validate(report)

            report.status          = "downloaded"
            report.file_path       = str(dest_path)
            report.file_hash       = sha256
            report.file_size_bytes = bytes_written
            report.downloaded_at   = datetime.now(timezone.utc)
            db.flush()

            logger.info(
                "ingestion.download_complete",
                report_type=report.report_type,
                filename=filename,
                size_mb=round(bytes_written / (1024 * 1024), 2),
                sha256_prefix=sha256[:8],
            )
            return ReportRead.model_validate(report)

    # ------------------------------------------------------------------
    # Uploaded PDF ingestion
    # ------------------------------------------------------------------

    def ingest_uploaded_pdf(
        self,
        source_path:  Path,
        company_name: str,
        year:         int,
        sector:       str = "Other",
        country:      str = "India",
        report_type:  str = "uploaded",
    ) -> dict:
        """
        Register a locally saved PDF as a downloaded report in the DB.

        The file is treated identically to an auto-downloaded report from
        this point on: it can be parsed, chunked, embedded, and KPIs extracted.

        Steps:
        1. Validate PDF magic bytes.
        2. Compute SHA-256; check for duplicate upload (same company, same hash).
        3. get_or_create Company row.
        4. Copy file to permanent storage with standard filename.
        5. Insert Report row with status='downloaded' and no source_url.

        The caller is responsible for deleting source_path (the temp file)
        after this method returns.

        Args:
            source_path:  Path to temporary PDF on disk.
            company_name: Company name for DB and filename.
            year:         Fiscal year end integer.
            sector:       Sector tag stored on Company row.
            country:      Country stored on Company row.
            report_type:  Label stored in DB and used in filename
                          (e.g. "BRSR", "ESG", "uploaded").

        Returns:
            {
                "company":      CompanyRead,
                "report":       ReportRead,
                "is_duplicate": bool,
            }
        """
        if not source_path.exists():
            raise FileNotFoundError(f"Source PDF not found: {source_path}")

        # Validate PDF magic bytes before any DB interaction.
        with source_path.open("rb") as fh:
            magic = fh.read(5)
        if magic != b"%PDF-":
            raise ValueError(
                f"Uploaded file is not a valid PDF (magic bytes: {magic!r})."
            )

        file_size = source_path.stat().st_size
        sha256    = _sha256_file(source_path)

        # Ensure Company exists before dedup check (need company.id).
        company = self.get_or_create_company(
            CompanyCreate(name=company_name, sector=sector, country=country)
        )

        # Detect duplicate upload: same company, same file hash.
        with get_db() as db:
            existing = (
                db.query(Report)
                .filter(
                    Report.company_id == company.id,
                    Report.file_hash  == sha256,
                )
                .first()
            )
            if existing:
                logger.info(
                    "ingestion.uploaded_duplicate",
                    sha256_prefix=sha256[:8],
                    existing_report_id=str(existing.id)[:8],
                )
                return {
                    "company":      company,
                    "report":       ReportRead.model_validate(existing),
                    "is_duplicate": True,
                }

        # Build permanent path and copy.
        new_report_id = uuid.uuid4()
        filename      = _make_pdf_filename(
            year=year,
            report_type=report_type,
            company_name=company_name,
            report_id=new_report_id,
        )
        dest_path = (
            self.settings.pdf_storage_path
            / str(company.id)
            / filename
        )
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)

        logger.info(
            "ingestion.uploaded_pdf_saved",
            filename=filename,
            size_mb=round(file_size / (1024 * 1024), 2),
        )

        with get_db() as db:
            report = Report(
                id=new_report_id,
                company_id=company.id,
                report_year=year,
                report_type=report_type,
                source_url=None,
                file_path=str(dest_path),
                file_hash=sha256,
                file_size_bytes=file_size,
                status="downloaded",
                downloaded_at=datetime.now(timezone.utc),
            )
            db.add(report)
            db.flush()
            report_read = ReportRead.model_validate(report)

        return {
            "company":      company,
            "report":       report_read,
            "is_duplicate": False,
        }

    # ------------------------------------------------------------------
    # Single-type orchestration (backward-compatible)
    # ------------------------------------------------------------------

    def run(
        self,
        company_data:  CompanyCreate,
        year:          int,
        report_type:   str = "BRSR",
        auto_download: bool = True,
        max_downloads: int = 1,
    ) -> dict:
        """
        Full Phase 1 pipeline for one company and one report type.

        Note: internally this calls collect_and_classify() which runs ALL
        query templates for ALL types. Only the slice for the requested type
        is returned. Use run_multi_report_types() if you need all three types
        to avoid running the full query set multiple times.

        Returns:
            {
                "company":            CompanyRead,
                "search_result":      SearchResult,
                "registered_reports": list[ReportRead],
                "downloaded_reports": list[ReportRead],
            }
        """
        company       = self.get_or_create_company(company_data)
        search_result = search_reports(
            company_name=company_data.name,
            year=year,
            report_type=report_type,
        )

        registered: list[ReportRead] = []
        for disc in search_result.discovered:
            registered.append(
                self.register_discovered_report(
                    company_id=company.id,
                    discovered=disc,
                    year=year,
                    report_type=report_type,
                )
            )

        downloaded: list[ReportRead] = []
        if auto_download:
            for report_row in registered[:max_downloads]:
                downloaded.append(
                    self.download_report(
                        report_id=report_row.id,
                        company_name=company_data.name,
                    )
                )

        return {
            "company":            company,
            "search_result":      search_result,
            "registered_reports": registered,
            "downloaded_reports": downloaded,
        }

    # ------------------------------------------------------------------
    # Multi-type orchestration (main pipeline entry point)
    # ------------------------------------------------------------------

    def run_multi_report_types(
        self,
        company_data:  CompanyCreate,
        year:          int,
        auto_download: bool = True,
    ) -> dict:
        """
        Full Phase 1 pipeline for all three types: BRSR, ESG, Integrated.

        Algorithm:
        1. Call collect_and_classify() once. This runs all query templates
           for all three types in a single Tavily session, pools every URL
           returned, globally deduplicates by URL, classifies each URL into
           exactly one type by keyword matching, and returns per-type lists
           sorted by Tavily score.

        2. For each type, register every classified URL in the DB
           (one Report row per URL, idempotent on source_url).

        3. For each type, download exactly one PDF — the top-ranked URL
           assigned to that type. If the download fails, try the next
           URL in that type's list. Stop after first success.

           Key constraint: only one download per type. Once a type has a
           successful download, no further downloads are attempted for it.

        4. Log any types for which no URL was classified or all downloads failed.

        Returns:
            {
                "company":             CompanyRead,
                "results_by_type":     dict[str, SearchResult],
                "registered_reports":  list[ReportRead],   # all URL rows created
                "downloaded_reports":  list[ReportRead],   # successful downloads only
                "not_found_types":     list[str],          # types with no classified URLs
                "failed_types":        list[str],          # types where all downloads failed
            }
        """
        logger.info(
            "ingestion.multi_start",
            company=company_data.name,
            year=year,
        )

        company = self.get_or_create_company(company_data)

        # Step 1: Run all queries, pool results, classify by URL keywords.
        # This is one call that covers all three types.
        results_by_type = collect_and_classify(
            company_name=company_data.name,
            year=year,
        )

        all_registered:   list[ReportRead] = []
        all_downloaded:   list[ReportRead] = []
        not_found_types:  list[str]        = []  # zero URLs classified for this type
        failed_types:     list[str]        = []  # URLs found but all downloads failed

        # Steps 2 & 3: Per-type register + download.
        for report_type in ALL_REPORT_TYPES:
            search_result = results_by_type[report_type]

            # Log and track types with no classified URLs.
            if not search_result.discovered:
                logger.warning(
                    "ingestion.type_not_found",
                    report_type=report_type,
                    company=company_data.name,
                    year=year,
                    message=(
                        f"No URLs were classified as {report_type} for "
                        f"{company_data.name} FY{year}. The company may not "
                        f"publish a separate {report_type} report."
                    ),
                )
                not_found_types.append(report_type)
                continue

            # Register every classified URL for this type.
            # Sorted by score descending (already done by collect_and_classify).
            type_registered: list[ReportRead] = []
            for disc in search_result.discovered:
                row = self.register_discovered_report(
                    company_id=company.id,
                    discovered=disc,
                    year=year,
                    report_type=report_type,
                )
                type_registered.append(row)
                all_registered.append(row)

            if not auto_download:
                continue

            # Download exactly one PDF for this type.
            # Try candidates in score order (best first).
            type_downloaded = False

            for candidate in type_registered:
                # Idempotency: a previous pipeline run may have already
                # downloaded this exact row. Accept it without re-downloading.
                if candidate.status == "downloaded":
                    logger.info(
                        "ingestion.type_already_downloaded",
                        report_type=report_type,
                        report_id=str(candidate.id)[:8],
                    )
                    all_downloaded.append(candidate)
                    type_downloaded = True
                    break  # do not download any more URLs for this type

                # Attempt the download.
                dl_result = self.download_report(
                    report_id=candidate.id,
                    company_name=company_data.name,
                )

                if dl_result.status == "downloaded":
                    all_downloaded.append(dl_result)
                    type_downloaded = True
                    filename = (
                        Path(dl_result.file_path).name
                        if dl_result.file_path
                        else "unknown"
                    )
                    logger.info(
                        "ingestion.type_download_success",
                        report_type=report_type,
                        filename=filename,
                        size_mb=round((dl_result.file_size_bytes or 0) / (1024 * 1024), 2),
                    )
                    break  # one download per type — stop here

                # This candidate failed. Log and try the next one.
                logger.warning(
                    "ingestion.type_candidate_failed",
                    report_type=report_type,
                    url=(candidate.source_url or "")[:80],
                    error=dl_result.error_message,
                )

            if not type_downloaded:
                failed_types.append(report_type)
                logger.error(
                    "ingestion.type_all_downloads_failed",
                    report_type=report_type,
                    company=company_data.name,
                    candidates_tried=len(type_registered),
                    message=(
                        f"All {len(type_registered)} candidate URL(s) for "
                        f"{report_type} failed to download. "
                        f"Check network access and URL validity."
                    ),
                )

        logger.info(
            "ingestion.multi_complete",
            company=company_data.name,
            year=year,
            downloaded=len(all_downloaded),
            not_found_types=not_found_types,
            failed_types=failed_types,
        )

        return {
            "company":            company,
            "results_by_type":    results_by_type,
            "registered_reports": all_registered,
            "downloaded_reports": all_downloaded,
            "not_found_types":    not_found_types,
            "failed_types":       failed_types,
        }

    # ------------------------------------------------------------------
    # Fallback: retry all pending/failed URLs for a company+year
    # ------------------------------------------------------------------

    def download_next_available(
        self,
        company_id:   uuid.UUID,
        year:         int,
        company_name: str,
    ) -> Optional[ReportRead]:
        """
        Try all discovered/failed report rows for a company+year in
        registration order and return the first successful download.

        Used for manual retry flows (e.g. from the CLI's retry-failed command).
        Returns None if all candidates are exhausted.
        """
        with get_db() as db:
            pending = (
                db.query(Report)
                .filter(
                    Report.company_id == company_id,
                    Report.report_year == year,
                    Report.status.in_(["discovered", "failed"]),
                )
                .order_by(Report.discovered_at)
                .all()
            )
            pending_ids = [r.id for r in pending]

        for report_id in pending_ids:
            result = self.download_report(
                report_id=report_id,
                company_name=company_name,
            )
            if result.status == "downloaded":
                return result
            logger.warning(
                "ingestion.fallback_candidate_failed",
                report_id=str(report_id)[:8],
                error=result.error_message,
            )

        logger.error(
            "ingestion.all_fallbacks_exhausted",
            company_id=str(company_id),
            year=year,
        )
        return None