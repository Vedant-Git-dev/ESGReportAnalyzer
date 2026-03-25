"""
main.py
CLI entrypoint for the ESG pipeline.

Usage examples
--------------
# Initialise DB tables
python main.py init-db

# Run Phase 1 ingestion for a company
python main.py ingest --company "Infosys" --year 2023

# List all companies
python main.py list-companies
"""
from __future__ import annotations

import sys
import argparse

from core.config import get_settings
from core.logging_config import configure_logging


def cmd_init_db(_args) -> None:
    from core.database import create_all_tables, check_connection
    print("Checking DB connection...")
    if not check_connection():
        print("ERROR: Cannot connect to database. Check DATABASE_URL in .env")
        sys.exit(1)
    create_all_tables()
    print("✓ Database tables created / verified.")


def cmd_seed_kpis(_args) -> None:
    """Seed default KPI definitions into the database."""
    from core.database import get_db
    from models.db_models import KPIDefinition

    DEFAULT_KPIS = [
        {
            "name": "scope_1_emissions",
            "display_name": "Scope 1 GHG Emissions",
            "category": "Environmental",
            "subcategory": "Emissions",
            "expected_unit": "tCO2e",
            "regex_patterns": [
                r"scope[\s\-]*1[\s\S]{0,60}?([\d,]+\.?\d*)\s*(tco2e?|t\s*co2e?|mt\s*co2e?|kt\s*co2e?|tonnes?\s*co2e?)",
                r"direct\s+(?:ghg\s+)?emissions[\s\S]{0,60}?([\d,]+\.?\d*)\s*(tco2e?|t\s*co2e?|tonnes?\s*co2e?)",
                r"([\d,]+\.?\d*)\s*(tco2e?|t\s*co2e?)\s*[\s\S]{0,40}?scope[\s\-]*1",
            ],
            "retrieval_keywords": [
                "scope 1", "direct emissions", "direct ghg", "tCO2e",
                "stationary combustion", "fuel combustion", "fugitive", "owned vehicles",
            ],
            "valid_min": 0, "valid_max": 1e9,
        },
        {
            "name": "scope_2_emissions",
            "display_name": "Scope 2 GHG Emissions",
            "category": "Environmental",
            "subcategory": "Emissions",
            "expected_unit": "tCO2e",
            "regex_patterns": [
                r"scope[\s\-]*2[\s\S]{0,60}?([\d,]+\.?\d*)\s*(tco2e?|t\s*co2e?|mt\s*co2e?|kt\s*co2e?|tonnes?\s*co2e?)",
                r"indirect\s+(?:ghg\s+)?emissions[\s\S]{0,60}?([\d,]+\.?\d*)\s*(tco2e?|t\s*co2e?)",
                r"purchased\s+electricity[\s\S]{0,60}?([\d,]+\.?\d*)\s*(tco2e?|t\s*co2e?)",
            ],
            "retrieval_keywords": [
                "scope 2", "indirect emissions", "purchased electricity",
                "market-based", "location-based", "tCO2e", "electricity ghg",
            ],
            "valid_min": 0, "valid_max": 1e9,
        },
        {
            "name": "total_ghg_emissions",
            "display_name": "Total GHG Emissions",
            "category": "Environmental",
            "subcategory": "Emissions",
            "expected_unit": "tCO2e",
            "regex_patterns": [
                r"total\s+(?:ghg|greenhouse\s+gas)\s+emissions[\s\S]{0,60}?([\d,]+\.?\d*)\s*(tco2e?|t\s*co2e?|mt\s*co2e?)",
                r"(?:scope\s*1\s*(?:and|&|\+)\s*2|scope\s*1\s*\+\s*scope\s*2)[\s\S]{0,60}?([\d,]+\.?\d*)\s*(tco2e?|t\s*co2e?)",
                r"(?:carbon|ghg)\s+footprint[\s\S]{0,60}?([\d,]+\.?\d*)\s*(tco2e?|t\s*co2e?|mt\s*co2e?)",
                r"([\d,]+\.?\d*)\s*(tco2e?|t\s*co2e?)\s*[\s\S]{0,50}?(?:total|combined|overall)\s*(?:ghg|emissions)",
            ],
            "retrieval_keywords": [
                "total GHG", "total emissions", "scope 1 and 2", "carbon footprint",
                "tCO2e", "greenhouse gas", "CO2 equivalent", "carbon neutral", "ghg inventory",
            ],
            "valid_min": 0, "valid_max": 1e10,
        },
        {
            "name": "energy_consumption",
            "display_name": "Total Energy Consumption",
            "category": "Environmental",
            "subcategory": "Energy",
            "expected_unit": "GJ",
            "regex_patterns": [
                r"total\s+energy\s+(?:consumption|consumed|use|used)[\s\S]{0,60}?([\d,]+\.?\d*)\s*(gj|mwh|gwh|tj|pj)",
                r"energy\s+(?:consumption|consumed|use|used)[\s\S]{0,60}?([\d,]+\.?\d*)\s*(gj|mwh|gwh|tj)",
                r"([\d,]+\.?\d*)\s*(gj|gwh|mwh|tj)\s*[\s\S]{0,40}?(?:total\s+)?energy",
            ],
            "retrieval_keywords": [
                "energy consumption", "total energy", "energy consumed", "energy use",
                "GJ", "MWh", "GWh", "gigajoules", "megawatt", "fuel consumed",
            ],
            "valid_min": 0, "valid_max": 1e12,
        },
        {
            "name": "renewable_energy_percentage",
            "display_name": "Renewable Energy Percentage",
            "category": "Environmental",
            "subcategory": "Energy",
            "expected_unit": "%",
            "regex_patterns": [
                r"renewable\s+energy[\s\S]{0,80}?([\d]+\.?\d*)\s*%",
                r"([\d]+\.?\d*)\s*%[\s\S]{0,40}?(?:from\s+)?renewable",
                r"(?:solar|wind|clean|green)\s+energy[\s\S]{0,80}?([\d]+\.?\d*)\s*%",
                r"renewable[\s\S]{0,40}?([\d]+\.?\d*)\s*percent",
            ],
            "retrieval_keywords": [
                "renewable energy", "solar", "wind", "clean energy", "green energy",
                "non-fossil", "RE share", "renewable electricity", "percent renewable",
            ],
            "valid_min": 0, "valid_max": 100,
        },
        {
            "name": "water_consumption",
            "display_name": "Total Water Consumption",
            "category": "Environmental",
            "subcategory": "Water",
            "expected_unit": "KL",
            "regex_patterns": [
                r"water\s+(?:consumption|consumed|use|used|withdrawal|withdrawn)[\s\S]{0,60}?([\d,]+\.?\d*)\s*(kl|kilolitr\w*|m3|cubic\s+met\w*|million\s+litr\w*)",
                r"([\d,]+\.?\d*)\s*(kl|kilolitr\w*|m3)\s*[\s\S]{0,40}?water",
                r"freshwater[\s\S]{0,60}?([\d,]+\.?\d*)\s*(kl|kilolitr\w*|m3)",
            ],
            "retrieval_keywords": [
                "water consumption", "water withdrawal", "water use", "water consumed",
                "KL", "kilolitre", "freshwater", "water recycled", "water intensity",
            ],
            "valid_min": 0, "valid_max": 1e12,
        },
        {
            "name": "waste_generated",
            "display_name": "Total Waste Generated",
            "category": "Environmental",
            "subcategory": "Waste",
            "expected_unit": "MT",
            "regex_patterns": [
                r"(?:total\s+)?waste\s+(?:generated|produced|disposed)[\s\S]{0,60}?([\d,]+\.?\d*)\s*(mt|metric\s*tonn?\w*|tonne\w*|kg)",
                r"([\d,]+\.?\d*)\s*(mt|metric\s*tonn?\w*|tonne\w*)\s*[\s\S]{0,40}?waste",
                r"hazardous\s+waste[\s\S]{0,60}?([\d,]+\.?\d*)\s*(mt|metric\s*tonn?\w*|tonne\w*|kg)",
            ],
            "retrieval_keywords": [
                "waste generated", "total waste", "solid waste", "hazardous waste",
                "non-hazardous", "metric tonnes", "MT", "waste disposed", "waste diverted",
            ],
            "valid_min": 0, "valid_max": 1e9,
        },
        {
            "name": "employee_count",
            "display_name": "Total Employees",
            "category": "Social",
            "subcategory": "Workforce",
            "expected_unit": "headcount",
            "regex_patterns": [
                r"total\s+(?:number\s+of\s+)?employees[\s\S]{0,40}?([\d,]+)\s*(?:nos|number|headcount|employees|$|\n)",
                r"workforce\s+(?:size|strength|count|of)[\s\S]{0,40}?([\d,]+)",
                r"([\d,]+)\s+(?:permanent\s+)?employees\s+(?:as\s+of|globally|worldwide|in\s+india)",
                r"total\s+headcount[\s\S]{0,40}?([\d,]+)",
            ],
            "retrieval_keywords": [
                "total employees", "workforce", "headcount", "FTE", "full-time",
                "permanent employees", "employee strength", "number of employees",
            ],
            "valid_min": 1, "valid_max": 5e6,
        },
        {
            "name": "women_in_workforce_percentage",
            "display_name": "Women in Workforce (%)",
            "category": "Social",
            "subcategory": "Diversity",
            "expected_unit": "%",
            "regex_patterns": [
                r"women[\s\S]{0,60}?([\d]+\.?\d*)\s*%",
                r"female[\s\S]{0,60}?([\d]+\.?\d*)\s*%",
                r"([\d]+\.?\d*)\s*%[\s\S]{0,30}?(?:women|female)",
                r"gender\s+(?:ratio|diversity|split)[\s\S]{0,80}?women[\s\S]{0,30}?([\d]+\.?\d*)\s*%",
            ],
            "retrieval_keywords": [
                "women", "female", "gender diversity", "gender ratio",
                "women employees", "female workforce", "women in leadership",
            ],
            "valid_min": 0, "valid_max": 100,
        },
        {
            "name": "csr_spend",
            "display_name": "CSR / Social Investment Spend",
            "category": "Social",
            "subcategory": "Community",
            "expected_unit": "INR Crore",
            "regex_patterns": [
                r"csr\s+(?:expenditure|spend|investment|amount\s+spent)[\s\S]{0,60}?([\d,]+\.?\d*)\s*(?:crore|cr|lakh|inr)?",
                r"(?:amount\s+spent|expenditure)\s+on\s+csr[\s\S]{0,60}?([\d,]+\.?\d*)\s*(?:crore|cr)",
                r"([\d,]+\.?\d*)\s*(?:crore|cr)\s*[\s\S]{0,40}?csr",
            ],
            "retrieval_keywords": [
                "CSR expenditure", "CSR spend", "CSR investment", "social spend",
                "community investment", "corporate social responsibility", "crore",
            ],
            "valid_min": 0, "valid_max": 1e6,
        },
    ]

    with get_db() as db:
        added = 0
        updated = 0
        for kpi_data in DEFAULT_KPIS:
            existing = db.query(KPIDefinition).filter(KPIDefinition.name == kpi_data["name"]).first()
            if existing:
                # Always update patterns and keywords to latest version
                existing.regex_patterns = kpi_data["regex_patterns"]
                existing.retrieval_keywords = kpi_data["retrieval_keywords"]
                existing.valid_min = kpi_data.get("valid_min")
                existing.valid_max = kpi_data.get("valid_max")
                updated += 1
            else:
                kpi = KPIDefinition(**kpi_data)
                db.add(kpi)
                added += 1
        db.flush()
        print(f"✓ KPI definitions: {added} added, {updated} updated.")


def cmd_ingest(args) -> None:
    from agents.ingestion_agent import IngestionAgent
    from models.schemas import CompanyCreate

    agent = IngestionAgent()
    company_data = CompanyCreate(
        name=args.company,
        ticker=args.ticker,
        sector=args.sector,
    )
    result = agent.run(
        company_data=company_data,
        year=args.year,
        report_type=args.report_type,
        auto_download=not args.no_download,
        max_downloads=args.max_downloads,
    )

    print(f"\n=== Ingestion Complete ===")
    print(f"Company:    {result['company'].name} (id={result['company'].id})")
    print(f"Discovered: {result['search_result'].total_found} PDF(s)")
    for r in result["registered_reports"]:
        print(f"  [registered] {r.source_url[:80]}...")
    for r in result["downloaded_reports"]:
        status = "✓" if r.status == "downloaded" else "✗"
        print(f"  [{status} {r.status}] {r.file_path or r.error_message}")


def cmd_list_chunks(args) -> None:
    import uuid as _uuid
    from core.database import get_db
    from models.db_models import ParsedDocument, DocumentChunk

    report_id = _uuid.UUID(args.report_id)

    with get_db() as db:
        parsed_doc = (
            db.query(ParsedDocument)
            .filter(ParsedDocument.report_id == report_id)
            .order_by(ParsedDocument.parsed_at.desc())
            .first()
        )
        if not parsed_doc:
            print(f"No parse cache found for report {report_id}. Run: python main.py parse --report-id {report_id}")
            return

        chunks = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.parsed_document_id == parsed_doc.id)
            .order_by(DocumentChunk.chunk_index)
            .all()
        )

        print(f"\nParsed Document : {parsed_doc.id}")
        print(f"Parser version  : {parsed_doc.parser_version}")
        print(f"Total chunks    : {len(chunks)}")
        print(f"Tables          : {sum(1 for c in chunks if c.chunk_type == 'table')}")
        print(f"Text            : {sum(1 for c in chunks if c.chunk_type == 'text')}")
        print(f"Footnotes       : {sum(1 for c in chunks if c.chunk_type == 'footnote')}")
        print()

        # Filter by type if requested
        filtered = chunks
        if args.type:
            filtered = [c for c in chunks if c.chunk_type == args.type]

        # Limit display
        display = filtered[:args.limit]

        print(f"{'#':<5} {'Type':<10} {'Page':<6} {'Tokens':<8} Preview")
        print("-" * 100)
        for chunk in display:
            preview = chunk.content[:80].replace("\n", " ")
            print(f"{chunk.chunk_index:<5} {chunk.chunk_type:<10} {chunk.page_number or '?':<6} {chunk.token_count or '?':<8} {preview}")

        if len(filtered) > args.limit:
            print(f"\n... {len(filtered) - args.limit} more chunks (use --limit to see more)")

        # Show full content of a specific chunk
        if args.show is not None:
            match = next((c for c in chunks if c.chunk_index == args.show), None)
            if match:
                print(f"\n{'='*60}")
                print(f"Chunk #{match.chunk_index} | {match.chunk_type} | page {match.page_number} | {match.token_count} tokens")
                print(f"{'='*60}")
                print(match.content)
            else:
                print(f"Chunk #{args.show} not found")


    from core.database import get_db
    from models.db_models import Company

    with get_db() as db:
        companies = db.query(Company).order_by(Company.name).all()
        if not companies:
            print("No companies found.")
            return
        print(f"\n{'Name':<40} {'Ticker':<10} {'Sector':<25} ID")
        print("-" * 100)
        for c in companies:
            print(f"{c.name:<40} {(c.ticker or ''):<10} {(c.sector or ''):<25} {c.id}")


def cmd_list_reports(_args) -> None:
    from core.database import get_db
    from models.db_models import Report, Company

    with get_db() as db:
        rows = (
            db.query(Report, Company.name)
            .join(Company, Report.company_id == Company.id)
            .order_by(Company.name, Report.report_year.desc())
            .all()
        )
        if not rows:
            print("No reports found.")
            return
        print(f"\n{'Company':<30} {'Year':<6} {'Type':<15} {'Status':<12} {'ID'}")
        print("-" * 110)
        for report, company_name in rows:
            print(
                f"{company_name:<30} {report.report_year:<6} {report.report_type:<15} "
                f"{report.status:<12} {report.id}"
            )


# ---------------------------------------------------------------------------
# Phase 2 commands
# ---------------------------------------------------------------------------

def cmd_parse(args) -> None:
    """Parse a downloaded PDF and cache the result."""
    import uuid as _uuid
    from services.parse_orchestrator import ParseOrchestrator

    report_id = _uuid.UUID(args.report_id)
    orchestrator = ParseOrchestrator()

    print(f"Parsing report {report_id} (parser_version={get_settings().parser_version}) ...")
    result = orchestrator.run(report_id=report_id, force=args.force)

    print(f"\n=== Parse Complete ===")
    print(f"ParsedDocument ID : {result.id}")
    print(f"Parser version    : {result.parser_version}")
    print(f"Pages             : {result.page_count}")
    print(f"Chunks stored     : {result.meta.get('chunk_count', '?')}")
    print(f"Tables            : {result.meta.get('table_count', '?')}")
    print(f"OCR pages         : {result.meta.get('ocr_page_count', 0)}")
    print(f"Word count        : {result.meta.get('word_count', '?')}")


def cmd_parse_status(_args) -> None:
    """Show parse cache entries."""
    from core.database import get_db
    from models.db_models import ParsedDocument, Report, Company

    with get_db() as db:
        rows = (
            db.query(ParsedDocument, Report, Company.name)
            .join(Report, ParsedDocument.report_id == Report.id)
            .join(Company, Report.company_id == Company.id)
            .order_by(Company.name, Report.report_year.desc())
            .all()
        )
        if not rows:
            print("No parse cache entries found.")
            return
        print(f"\n{'Company':<28} {'Year':<6} {'Version':<10} {'Pages':<7} {'Chunks':<8} ParsedDoc ID")
        print("-" * 110)
        for pd, report, company_name in rows:
            chunks = pd.meta.get("chunk_count", "?")
            print(
                f"{company_name:<28} {report.report_year:<6} {pd.parser_version:<10} "
                f"{pd.page_count or '?':<7} {chunks:<8} {pd.id}"
            )


# ---------------------------------------------------------------------------
# Phase 3 commands
# ---------------------------------------------------------------------------

def cmd_search_chunks(args) -> None:
    """
    Ad-hoc keyword search across chunks of a parsed document.
    Use this to test retrieval before running full KPI extraction.
    """
    import uuid as _uuid
    from core.database import get_db
    from models.db_models import ParsedDocument
    from services.retrieval_service import RetrievalService

    report_id = _uuid.UUID(args.report_id)
    keywords = [k.strip() for k in args.keywords.split(",")]

    with get_db() as db:
        parsed_doc = (
            db.query(ParsedDocument)
            .filter(ParsedDocument.report_id == report_id)
            .order_by(ParsedDocument.parsed_at.desc())
            .first()
        )
        if not parsed_doc:
            print(f"No parse cache found for report {report_id}. Run: python main.py parse --report-id {report_id}")
            return

        service = RetrievalService()
        chunk_types = [args.type] if args.type else None
        results = service.retrieve_by_keywords(
            parsed_document_id=parsed_doc.id,
            keywords=keywords,
            db=db,
            top_k=args.top_k,
            chunk_types=chunk_types,
        )

        # Extract all data while session is still open
        rows = [
            {
                "score": sc.score,
                "matched_keywords": sc.matched_keywords,
                "chunk_index": sc.chunk.chunk_index,
                "chunk_type": sc.chunk.chunk_type,
                "page_number": sc.chunk.page_number,
                "token_count": sc.chunk.token_count,
                "content": sc.chunk.content,
            }
            for sc in results
        ]

    if not rows:
        print(f"No chunks matched keywords: {keywords}")
        return

    print(f"\n=== Retrieval Results ===")
    print(f"Keywords : {', '.join(keywords)}")
    print(f"Returned : {len(rows)} chunk(s)\n")
    print(f"{'Rank':<5} {'Score':<8} {'Type':<10} {'Page':<6} {'Tokens':<8} {'Keywords Hit':<30} Preview")
    print("-" * 120)
    for rank, row in enumerate(rows, 1):
        preview = row["content"][:70].replace("\n", " ")
        kw_hit = ", ".join(row["matched_keywords"][:4])
        print(
            f"{rank:<5} {row['score']:<8.3f} {row['chunk_type']:<10} "
            f"{row['page_number'] or '?':<6} {row['token_count'] or '?':<8} "
            f"{kw_hit:<30} {preview}"
        )

    if args.show_top:
        top = rows[0]
        print(f"\n{'='*60}")
        print(f"Top chunk #{top['chunk_index']} | {top['chunk_type']} | page {top['page_number']} | score {top['score']:.3f}")
        print(f"{'='*60}")
        print(top["content"])


def cmd_kpi_retrieve(args) -> None:
    """
    Retrieve top chunks for a specific KPI from a report's parse cache.
    Shows exactly what will be sent to the LLM in Phase 4.
    """
    import uuid as _uuid
    from core.database import get_db
    from models.db_models import KPIDefinition, ParsedDocument
    from services.retrieval_service import RetrievalService

    report_id = _uuid.UUID(args.report_id)

    with get_db() as db:
        kpi = db.query(KPIDefinition).filter(KPIDefinition.name == args.kpi).first()
        if not kpi:
            print(f"KPI '{args.kpi}' not found. Run: python main.py list-kpis")
            return

        parsed_doc = (
            db.query(ParsedDocument)
            .filter(ParsedDocument.report_id == report_id)
            .order_by(ParsedDocument.parsed_at.desc())
            .first()
        )
        if not parsed_doc:
            print(f"No parse cache found for report {report_id}")
            return

        service = RetrievalService()
        results = service.retrieve(
            parsed_document_id=parsed_doc.id,
            kpi=kpi,
            db=db,
            top_k=args.top_k,
        )

        # Extract while session is open
        kpi_display = kpi.display_name
        kpi_unit = kpi.expected_unit
        kpi_keywords = list(kpi.retrieval_keywords)
        rows = [
            {
                "score": sc.score,
                "chunk_index": sc.chunk.chunk_index,
                "chunk_type": sc.chunk.chunk_type,
                "page_number": sc.chunk.page_number,
                "content": sc.chunk.content,
            }
            for sc in results
        ]

    if not rows:
        print(f"No relevant chunks found for KPI: {args.kpi}")
        return

    print(f"\n=== KPI Retrieval: {kpi_display} ===")
    print(f"Expected unit : {kpi_unit}")
    print(f"Keywords used : {', '.join(kpi_keywords)}")
    print(f"Chunks found  : {len(rows)}\n")

    for rank, row in enumerate(rows, 1):
        print(f"--- Rank {rank} | {row['chunk_type']} | page {row['page_number']} | score {row['score']:.3f} ---")
        print(row["content"][:400])
        print()


def cmd_list_kpis(_args) -> None:
    """List all KPI definitions in the database."""
    from core.database import get_db
    from models.db_models import KPIDefinition

    with get_db() as db:
        kpis = db.query(KPIDefinition).filter(KPIDefinition.is_active == True).order_by(KPIDefinition.category, KPIDefinition.name).all()
        if not kpis:
            print("No KPI definitions found. Run: python main.py seed-kpis")
            return
        print(f"\n{'Name':<35} {'Category':<15} {'Unit':<15} {'Keywords'}")
        print("-" * 110)
        for kpi in kpis:
            kws = ", ".join(kpi.retrieval_keywords[:3])
            print(f"{kpi.name:<35} {kpi.category:<15} {kpi.expected_unit:<15} {kws}")


# ---------------------------------------------------------------------------
# Phase 4 commands
# ---------------------------------------------------------------------------

def cmd_extract(args) -> None:
    """Run KPI extraction (regex → LLM → validation) on a parsed report."""
    import uuid as _uuid
    from core.database import get_db
    from agents.extraction_agent import ExtractionAgent

    report_id = _uuid.UUID(args.report_id)
    kpi_names = [k.strip() for k in args.kpis.split(",")] if args.kpis else None

    agent = ExtractionAgent()

    print(f"Extracting KPIs for report {report_id} ...")
    if kpi_names:
        print(f"KPIs: {', '.join(kpi_names)}")
    else:
        print("KPIs: all active")

    with get_db() as db:
        results = agent.extract_all(
            report_id=report_id,
            db=db,
            kpi_names=kpi_names,
        )
        # Snapshot while session open
        rows = [
            {
                "kpi_name": r.kpi_name,
                "value": r.normalized_value,
                "unit": r.unit,
                "method": r.extraction_method,
                "confidence": r.confidence,
                "valid": r.validation_passed,
                "notes": r.validation_notes,
            }
            for r in results
        ]

    found = sum(1 for r in rows if r["value"] is not None)
    print(f"\n=== Extraction Complete: {found}/{len(rows)} KPIs found ===\n")
    print(f"{'KPI':<35} {'Value':<15} {'Unit':<12} {'Method':<8} {'Conf':<6} {'Valid':<6} Notes")
    print("-" * 120)
    for row in rows:
        val = f"{row['value']:,.2f}" if row["value"] is not None else "NOT FOUND"
        conf = f"{row['confidence']:.2f}" if row["confidence"] else "-"
        valid = "✓" if row["valid"] else "✗"
        notes = (row["notes"] or "")[:40]
        print(
            f"{row['kpi_name']:<35} {val:<15} {(row['unit'] or ''):<12} "
            f"{row['method']:<8} {conf:<6} {valid:<6} {notes}"
        )


def cmd_list_kpi_records(args) -> None:
    """Show extracted KPI records for a report."""
    import uuid as _uuid
    from core.database import get_db
    from models.db_models import KPIRecord, KPIDefinition, Report, Company

    report_id = _uuid.UUID(args.report_id)

    with get_db() as db:
        rows = (
            db.query(KPIRecord, KPIDefinition.name, KPIDefinition.expected_unit, Company.name)
            .join(KPIDefinition, KPIRecord.kpi_definition_id == KPIDefinition.id)
            .join(Company, KPIRecord.company_id == Company.id)
            .filter(KPIRecord.report_id == report_id)
            .order_by(KPIDefinition.category, KPIDefinition.name, KPIRecord.extracted_at.desc())
            .all()
        )

        if not rows:
            print(f"No KPI records found for report {report_id}")
            return

        data = [
            {
                "kpi": kpi_name,
                "value": r.normalized_value,
                "unit": r.unit or expected_unit,
                "method": r.extraction_method,
                "confidence": r.confidence,
                "valid": r.is_validated,
                "year": r.report_year,
                "company": company_name,
                "extracted_at": r.extracted_at.strftime("%Y-%m-%d %H:%M"),
            }
            for r, kpi_name, expected_unit, company_name in rows
        ]

    print(f"\n=== KPI Records for report {report_id} ===")
    print(f"{'KPI':<35} {'Value':>12} {'Unit':<12} {'Method':<8} {'Conf':<6} {'Valid'} {'Extracted'}")
    print("-" * 110)
    for d in data:
        val = f"{d['value']:,.2f}" if d["value"] is not None else "—"
        conf = f"{d['confidence']:.2f}" if d["confidence"] else "-"
        valid = "✓" if d["valid"] else "✗"
        print(
            f"{d['kpi']:<35} {val:>12} {d['unit']:<12} {d['method']:<8} "
            f"{conf:<6} {valid:<6} {d['extracted_at']}"
        )


def main():
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_file)

    parser = argparse.ArgumentParser(description="ESG Competitive Intelligence Pipeline")
    sub = parser.add_subparsers(dest="command")

    # init-db
    sub.add_parser("init-db", help="Create all database tables")

    # seed-kpis
    sub.add_parser("seed-kpis", help="Seed default KPI definitions")

    # ingest
    ingest_p = sub.add_parser("ingest", help="Discover and download ESG reports for a company")
    ingest_p.add_argument("--company", required=True, help="Company name")
    ingest_p.add_argument("--year", type=int, required=True, help="Report year")
    ingest_p.add_argument("--ticker", default=None)
    ingest_p.add_argument("--sector", default=None)
    ingest_p.add_argument("--report-type", default="BRSR",
                          choices=["BRSR", "ESG", "Sustainability", "Annual"],
                          help="Report type to search for (default: BRSR)")
    ingest_p.add_argument("--no-download", action="store_true", help="Register but don't download")
    ingest_p.add_argument("--max-downloads", type=int, default=1)

    # list-companies
    sub.add_parser("list-companies", help="List all companies in DB")

    # list-reports
    sub.add_parser("list-reports", help="List all reports and their status")

    # list-chunks
    chunks_p = sub.add_parser("list-chunks", help="Show parsed chunks for a report")
    chunks_p.add_argument("--report-id", required=True, help="UUID of the report")
    chunks_p.add_argument("--type", default=None, choices=["text", "table", "footnote"], help="Filter by chunk type")
    chunks_p.add_argument("--limit", type=int, default=20, help="Max chunks to display (default: 20)")
    chunks_p.add_argument("--show", type=int, default=None, help="Print full content of chunk #N")

    # Phase 2: parse
    parse_p = sub.add_parser("parse", help="Parse a downloaded PDF and cache result")
    parse_p.add_argument("--report-id", required=True, help="UUID of a downloaded report")
    parse_p.add_argument("--force", action="store_true", help="Re-parse even if cache exists")

    # Phase 2: parse-status
    sub.add_parser("parse-status", help="Show parse cache entries")

    # Phase 3: search-chunks
    sc_p = sub.add_parser("search-chunks", help="Ad-hoc keyword search across parsed chunks")
    sc_p.add_argument("--report-id", required=True, help="UUID of the report")
    sc_p.add_argument("--keywords", required=True, help="Comma-separated keywords e.g. 'scope 1,emissions,tCO2e'")
    sc_p.add_argument("--top-k", type=int, default=7, help="Max chunks to return (default: 7)")
    sc_p.add_argument("--type", default=None, choices=["text", "table", "footnote"], help="Filter chunk type")
    sc_p.add_argument("--show-top", action="store_true", help="Print full content of top result")

    # Phase 3: kpi-retrieve
    kr_p = sub.add_parser("kpi-retrieve", help="Retrieve top chunks for a KPI (preview of Phase 4 input)")
    kr_p.add_argument("--report-id", required=True, help="UUID of the report")
    kr_p.add_argument("--kpi", required=True, help="KPI name e.g. scope_1_emissions")
    kr_p.add_argument("--top-k", type=int, default=7)

    # list-kpis
    sub.add_parser("list-kpis", help="List all KPI definitions")

    # Phase 4: extract
    ex_p = sub.add_parser("extract", help="Extract KPIs from a parsed report (regex → LLM → validation)")
    ex_p.add_argument("--report-id", required=True, help="UUID of a parsed report")
    ex_p.add_argument("--kpis", default=None, help="Comma-separated KPI names to extract (default: all active)")

    # Phase 4: list-kpi-records
    rec_p = sub.add_parser("list-kpi-records", help="Show extracted KPI records for a report")
    rec_p.add_argument("--report-id", required=True, help="UUID of the report")

    args = parser.parse_args()

    dispatch = {
        "init-db": cmd_init_db,
        "seed-kpis": cmd_seed_kpis,
        "ingest": cmd_ingest,
        "list-reports": cmd_list_reports,
        "list-chunks": cmd_list_chunks,
        "parse": cmd_parse,
        "parse-status": cmd_parse_status,
        "search-chunks": cmd_search_chunks,
        "kpi-retrieve": cmd_kpi_retrieve,
        "list-kpis": cmd_list_kpis,
        "extract": cmd_extract,
        "list-kpi-records": cmd_list_kpi_records,
    }

    if args.command not in dispatch:
        parser.print_help()
        sys.exit(1)

    dispatch[args.command](args)


if __name__ == "__main__":
    main()