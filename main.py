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


def cmd_list_companies(_args) -> None:
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

    args = parser.parse_args()

    dispatch = {
        "init-db": cmd_init_db,
        "seed-kpis": cmd_seed_kpis,
        "ingest": cmd_ingest,
        "list-companies": cmd_list_companies,
    }

    if args.command not in dispatch:
        parser.print_help()
        sys.exit(1)

    dispatch[args.command](args)


if __name__ == "__main__":
    main()