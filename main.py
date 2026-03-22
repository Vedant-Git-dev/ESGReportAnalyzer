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
                r"scope\s*1[^0-9]*?([\d,]+\.?\d*)\s*(tco2e|tco2|mt\s*co2|ktco2e)",
                r"direct\s+emissions[^0-9]*?([\d,]+\.?\d*)\s*(tco2e|tco2)",
            ],
            "retrieval_keywords": ["scope 1", "direct emissions", "tCO2e", "greenhouse gas"],
            "valid_min": 0,
            "valid_max": 1e9,
        },
        {
            "name": "scope_2_emissions",
            "display_name": "Scope 2 GHG Emissions",
            "category": "Environmental",
            "subcategory": "Emissions",
            "expected_unit": "tCO2e",
            "regex_patterns": [
                r"scope\s*2[^0-9]*?([\d,]+\.?\d*)\s*(tco2e|tco2|mt\s*co2|ktco2e)",
                r"indirect\s+emissions[^0-9]*?([\d,]+\.?\d*)\s*(tco2e|tco2)",
            ],
            "retrieval_keywords": ["scope 2", "indirect emissions", "purchased electricity", "tCO2e"],
            "valid_min": 0,
            "valid_max": 1e9,
        },
        {
            "name": "total_ghg_emissions",
            "display_name": "Total GHG Emissions",
            "category": "Environmental",
            "subcategory": "Emissions",
            "expected_unit": "tCO2e",
            "regex_patterns": [
                r"total\s+(?:ghg|greenhouse\s+gas)\s+emissions[^0-9]*?([\d,]+\.?\d*)\s*(tco2e|tco2)",
                r"(?:ghg|greenhouse\s+gas)\s+footprint[^0-9]*?([\d,]+\.?\d*)\s*(tco2e|tco2)",
            ],
            "retrieval_keywords": ["total GHG", "total emissions", "carbon footprint", "tCO2e"],
            "valid_min": 0,
            "valid_max": 1e10,
        },
        {
            "name": "energy_consumption",
            "display_name": "Total Energy Consumption",
            "category": "Environmental",
            "subcategory": "Energy",
            "expected_unit": "GJ",
            "regex_patterns": [
                r"total\s+energy\s+consumption[^0-9]*?([\d,]+\.?\d*)\s*(gj|mwh|gwh|tj)",
                r"energy\s+consumed[^0-9]*?([\d,]+\.?\d*)\s*(gj|mwh|gwh|tj)",
            ],
            "retrieval_keywords": ["energy consumption", "total energy", "GJ", "MWh", "GWh"],
            "valid_min": 0,
            "valid_max": 1e12,
        },
        {
            "name": "renewable_energy_percentage",
            "display_name": "Renewable Energy Percentage",
            "category": "Environmental",
            "subcategory": "Energy",
            "expected_unit": "%",
            "regex_patterns": [
                r"renewable\s+energy[^0-9]*?([\d]+\.?\d*)\s*%",
                r"([\d]+\.?\d*)\s*%\s+(?:of\s+)?(?:energy\s+from\s+)?renewable",
            ],
            "retrieval_keywords": ["renewable energy", "solar", "wind", "clean energy", "%"],
            "valid_min": 0,
            "valid_max": 100,
        },
        {
            "name": "water_consumption",
            "display_name": "Total Water Consumption",
            "category": "Environmental",
            "subcategory": "Water",
            "expected_unit": "KL",
            "regex_patterns": [
                r"water\s+consumption[^0-9]*?([\d,]+\.?\d*)\s*(kl|kilolitr|m3|cubic\s+meter|ml)",
                r"water\s+withdraw[^0-9]*?([\d,]+\.?\d*)\s*(kl|kilolitr|m3)",
            ],
            "retrieval_keywords": ["water consumption", "water withdrawal", "KL", "kilolitre"],
            "valid_min": 0,
            "valid_max": 1e12,
        },
        {
            "name": "waste_generated",
            "display_name": "Total Waste Generated",
            "category": "Environmental",
            "subcategory": "Waste",
            "expected_unit": "MT",
            "regex_patterns": [
                r"(?:total\s+)?waste\s+generated[^0-9]*?([\d,]+\.?\d*)\s*(mt|metric\s+ton|tonne|kg)",
                r"waste\s+produced[^0-9]*?([\d,]+\.?\d*)\s*(mt|tonne|kg)",
            ],
            "retrieval_keywords": ["waste generated", "waste produced", "metric tonnes", "MT"],
            "valid_min": 0,
            "valid_max": 1e9,
        },
        {
            "name": "employee_count",
            "display_name": "Total Employees",
            "category": "Social",
            "subcategory": "Workforce",
            "expected_unit": "headcount",
            "regex_patterns": [
                r"total\s+employees[^0-9]*?([\d,]+)",
                r"workforce\s+(?:size|strength)[^0-9]*?([\d,]+)",
                r"([\d,]+)\s+employees",
            ],
            "retrieval_keywords": ["total employees", "workforce", "headcount", "FTE"],
            "valid_min": 1,
            "valid_max": 5e6,
        },
        {
            "name": "women_in_workforce_percentage",
            "display_name": "Women in Workforce (%)",
            "category": "Social",
            "subcategory": "Diversity",
            "expected_unit": "%",
            "regex_patterns": [
                r"women[^0-9]*?([\d]+\.?\d*)\s*%",
                r"female\s+employees[^0-9]*?([\d]+\.?\d*)\s*%",
                r"([\d]+\.?\d*)\s*%\s+women",
            ],
            "retrieval_keywords": ["women", "female", "gender diversity", "diversity"],
            "valid_min": 0,
            "valid_max": 100,
        },
        {
            "name": "csr_spend",
            "display_name": "CSR / Social Investment Spend",
            "category": "Social",
            "subcategory": "Community",
            "expected_unit": "INR Crore",
            "regex_patterns": [
                r"csr\s+(?:expenditure|spend|investment)[^0-9]*?([\d,]+\.?\d*)\s*(?:crore|cr|lakh|inr)?",
                r"social\s+investment[^0-9]*?([\d,]+\.?\d*)\s*(?:crore|cr)",
            ],
            "retrieval_keywords": ["CSR", "corporate social responsibility", "social spend", "crore"],
            "valid_min": 0,
            "valid_max": 1e6,
        },
    ]

    with get_db() as db:
        added = 0
        for kpi_data in DEFAULT_KPIS:
            existing = db.query(KPIDefinition).filter(KPIDefinition.name == kpi_data["name"]).first()
            if existing:
                continue
            kpi = KPIDefinition(**kpi_data)
            db.add(kpi)
            added += 1
        db.flush()
        print(f"✓ Seeded {added} KPI definitions ({len(DEFAULT_KPIS) - added} already existed).")


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

    args = parser.parse_args()

    dispatch = {
        "init-db": cmd_init_db,
        "seed-kpis": cmd_seed_kpis,
        "ingest": cmd_ingest,
        "list-reports": cmd_list_reports,
        "list-chunks": cmd_list_chunks,
        "parse": cmd_parse,
        "parse-status": cmd_parse_status,
    }

    if args.command not in dispatch:
        parser.print_help()
        sys.exit(1)

    dispatch[args.command](args)


if __name__ == "__main__":
    main()