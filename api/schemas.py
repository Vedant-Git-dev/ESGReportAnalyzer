"""
api/schemas.py

Pydantic request/response schemas and shared constants for the FastAPI layer.

All domain constants (KPI_GROUPS, SECTORS, plausibility ranges, etc.) are
copied verbatim from the original ui.py so the React frontend receives the
exact same data structures the Streamlit app used internally.

Nothing in this file touches agents, services, models, or core modules.
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Constants (verbatim from ui.py)
# ---------------------------------------------------------------------------

SECTORS: list[str] = [
    "Information Technology",
    "Banking & Financial Services",
    "Energy & Utilities",
    "Pharmaceuticals & Healthcare",
    "Automotive & Manufacturing",
    "Fast-Moving Consumer Goods (FMCG)",
    "Chemicals & Materials",
    "Telecommunications",
    "Infrastructure & Real Estate",
    "Metals & Mining",
    "Other",
]

UPLOAD_REPORT_TYPE_OPTIONS: list[str] = [
    "BRSR", "ESG", "Integrated", "Annual", "CSR", "Other",
]

KPI_GROUPS: dict[str, dict] = {
    "revenue_from_operations": {
        "label": "Revenue from Operations", "group": "Financial", "unit": "INR Crore",
        "ratio_unit": "INR_Crore", "ratio_denominator": "none",
        "max_ratio": 0, "higher_is_better": True,
        "desc": "Annual consolidated revenue from operations",
    },
    "revenue_per_employee": {
        "label": "Revenue per Employee", "group": "Financial", "unit": "INR Crore",
        "ratio_unit": "INR_Crore/employee", "ratio_denominator": "employee",
        "max_ratio": 0, "higher_is_better": True,
        "desc": "Revenue from operations divided by total employee count",
    },
    "scope_1_emissions": {
        "label": "Scope 1 GHG", "group": "Environmental", "unit": "tCO2e",
        "ratio_unit": "tCO2e/Cr", "ratio_denominator": "revenue",
        "max_ratio": 10, "higher_is_better": False,
        "desc": "Direct GHG emissions per INR Crore revenue",
    },
    "scope_2_emissions": {
        "label": "Scope 2 GHG", "group": "Environmental", "unit": "tCO2e",
        "ratio_unit": "tCO2e/Cr", "ratio_denominator": "revenue",
        "max_ratio": 10, "higher_is_better": False,
        "desc": "Indirect GHG emissions per INR Crore revenue",
    },
    "scope_3_emissions": {
        "label": "Scope 3 GHG", "group": "Environmental", "unit": "tCO2e",
        "ratio_unit": "tCO2e/Cr", "ratio_denominator": "revenue",
        "max_ratio": 50, "higher_is_better": False,
        "desc": "Value chain GHG emissions per INR Crore revenue",
    },
    "energy_consumption": {
        "label": "Energy Intensity", "group": "Environmental", "unit": "GJ",
        "ratio_unit": "GJ/Cr", "ratio_denominator": "revenue",
        "max_ratio": 1_000, "higher_is_better": False,
        "desc": "Total energy consumed per INR Crore revenue",
    },
    "water_consumption": {
        "label": "Water Intensity", "group": "Environmental", "unit": "KL",
        "ratio_unit": "KL/Cr", "ratio_denominator": "revenue",
        "max_ratio": 500, "higher_is_better": False,
        "desc": "Total water consumed per INR Crore revenue",
    },
    "waste_generated": {
        "label": "Waste Intensity", "group": "Environmental", "unit": "MT",
        "ratio_unit": "MT/Cr", "ratio_denominator": "revenue",
        "max_ratio": 5, "higher_is_better": False,
        "desc": "Waste generated per INR Crore revenue",
    },
    "renewable_energy_percentage": {
        "label": "Renewable Energy", "group": "Environmental", "unit": "%",
        "ratio_unit": "%", "ratio_denominator": "none",
        "max_ratio": 100, "higher_is_better": True,
        "desc": "Share of energy from renewable sources",
    },
    "employee_count": {
        "label": "Workforce", "group": "Social", "unit": "count",
        "ratio_unit": "employees/Cr", "ratio_denominator": "revenue",
        "max_ratio": 5_000, "higher_is_better": False,
        "desc": "Total employees per INR Crore revenue",
    },
    "women_in_workforce_percentage": {
        "label": "Women in Workforce", "group": "Social", "unit": "%",
        "ratio_unit": "%", "ratio_denominator": "none",
        "max_ratio": 100, "higher_is_better": True,
        "desc": "Percentage of women in workforce",
    },
    "complaints_filed": {
        "label": "Complaints Filed", "group": "Governance", "unit": "count",
        "ratio_unit": "count", "ratio_denominator": "none",
        "max_ratio": 1_000_000, "higher_is_better": False,
        "desc": "Total complaints filed during the year",
    },
    "complaints_pending": {
        "label": "Complaints Pending", "group": "Governance", "unit": "count",
        "ratio_unit": "count", "ratio_denominator": "none",
        "max_ratio": 100_000, "higher_is_better": False,
        "desc": "Complaints pending resolution at year end",
    },
}

ALL_KPI_NAMES: list[str] = list(KPI_GROUPS.keys())

KPI_PLAUSIBILITY: dict[str, tuple[float, float]] = {
    "revenue_from_operations":    (5_000,   500_000),
    "revenue_per_employee":       (0.01,         50),
    "scope_1_emissions":             (1,       5_000_000),
    "scope_2_emissions":             (1,       5_000_000),
    "scope_3_emissions":             (1,      10_000_000),
    "energy_consumption":            (100,   500_000_000),
    "water_consumption":             (100,   100_000_000),
    "waste_generated":               (0.1,       500_000),
    "renewable_energy_percentage":   (0,              100),
    "employee_count":                (1,        5_000_000),
    "women_in_workforce_percentage": (0,              100),
    "complaints_filed":              (0,        1_000_000),
    "complaints_pending":            (0,          100_000),
}

DEFAULT_REVENUE_CR = 315_322.0

REPORT_TYPE_PRIORITY: dict[str, int] = {
    "Integrated": 0,
    "BRSR":       1,
    "ESG":        2,
}


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class CompareRequest(BaseModel):
    company1:    str   = Field(..., description="First company name")
    fy1:         int   = Field(..., description="First company fiscal year end")
    company2:    str   = Field(..., description="Second company name")
    fy2:         int   = Field(..., description="Second company fiscal year end")
    sector:      str   = Field("Information Technology", description="Sector label")


class ExportPdfRequest(BaseModel):
    """
    Payload for PDF report export.
    The frontend sends back the comparison data it received from /api/compare.
    """
    company1:     str
    fy1:          int
    company2:     str
    fy2:          int
    sector:       str
    summary:      str
    comparisons:  list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    db_online:   bool
    llm_ready:   bool
    status:      str


class KPIRecord(BaseModel):
    value:       float
    unit:        str
    method:      str
    confidence:  float
    report_type: Optional[str] = None


class RevenueInfo(BaseModel):
    value_cr:     float
    source:       str
    pattern_name: str
    confidence:   float


class CompanyResult(BaseModel):
    company_name:  str
    fy:            int
    kpi_records:   dict[str, KPIRecord]
    revenue:       Optional[RevenueInfo]
    log:           list[str]


class ComparisonEntry(BaseModel):
    company_label: str
    value:         float
    source:        str


class KPIComparison(BaseModel):
    kpi_name:     str
    display_name: str
    group:        str
    unit:         str
    entries:      list[ComparisonEntry]
    winner:       str
    pct_gap:      float
    meta:         dict[str, Any]   # KPI_GROUPS metadata for the frontend


class CompareResponse(BaseModel):
    company1:       CompanyResult
    company2:       CompanyResult
    comparisons:    list[KPIComparison]
    summary:        str
    recommendation: str
    label_a:        str
    label_b:        str


class UploadResponse(BaseModel):
    success:       bool
    company_id:    Optional[str]
    report_id:     Optional[str]
    company_name:  str
    fy:            int
    kpi_records:   dict[str, KPIRecord]
    revenue:       Optional[RevenueInfo]
    log:           list[str]
    message:       str


class CompanyListItem(BaseModel):
    id:     str
    name:   str
    sector: Optional[str]


class MetadataResponse(BaseModel):
    sectors:       list[str]
    report_types:  list[str]
    kpi_groups:    dict[str, dict]
    kpi_names:     list[str]
