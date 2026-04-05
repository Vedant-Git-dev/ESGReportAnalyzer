"""
services/normalizer.py

Unit normalisation for ESG KPIs extracted from corporate reports.

Design principles:
- No company-specific logic — works on any KPI record with a unit field
- Canonical units: GJ (energy), tCO2e (GHG), KL (water), MT (waste), INR_Crore (revenue)
- Handles all unit spellings found in BRSR / GRI / CDP reports (Indian and international)
- Raises NormalizationError on ambiguous or unknown units rather than silently coercing
- Revenue normalisation handles both INR Crore and USD Million (common in Indian multinationals)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# ── Canonical output units ────────────────────────────────────────────────────
CANONICAL = {
    "energy":   "GJ",
    "ghg":      "tCO2e",
    "water":    "KL",
    "waste":    "MT",
    "revenue":  "INR_Crore",
}

# ── Conversion tables: {unit_key: factor_to_canonical} ───────────────────────
# All keys are lowercase and stripped.
# Factor = multiply raw_value by this to get canonical.

_ENERGY_TO_GJ: dict[str, float] = {
    # Joules family
    "j":  1e-9,
    "kj": 1e-6,
    "mj": 1e-3,
    "gj": 1.0,
    "tj": 1e3,
    "pj": 1e6,
    # Watt-hour family
    "wh":  3.6e-6,
    "kwh": 3.6e-3,
    "mwh": 3.6,
    "gwh": 3600.0,
    "twh": 3_600_000.0,
    # Common long-form spellings
    "megajoule": 1e-3,
    "megajoules": 1e-3,
    "gigajoule": 1.0,
    "gigajoules": 1.0,
    "terajoule": 1e3,
    "terajoules": 1e3,
    "megawatt hour": 3.6,
    "megawatt hours": 3.6,
    "gigawatt hour": 3600.0,
    "gigawatt hours": 3600.0,
    "kilowatt hour": 3.6e-3,
    "kilowatt hours": 3.6e-3,
    # BTU (for international reports)
    "btu":   1.0551e-6,
    "mmbtu": 1.0551,
}

_GHG_TO_TCO2E: dict[str, float] = {
    "tco2e": 1.0,
    "tco2":  1.0,
    "t co2e": 1.0,
    "t co2": 1.0,
    "tonne co2e": 1.0,
    "tonnes co2e": 1.0,
    "metric tonne co2e": 1.0,
    "metric tonnes co2e": 1.0,
    "metric tonnes of co2 equivalent": 1.0,
    "metric tonnes of co2equivalent": 1.0,
    # Kilo / Mega
    "ktco2e": 1_000.0,
    "kt co2e": 1_000.0,
    "ktco2":  1_000.0,
    "kilo tonne co2e": 1_000.0,
    "mtco2e": 1_000_000.0,
    "million tonnes coe": 1_000_000.0,
    "mt co2e": 1_000_000.0,
    # GHG without CO2e label (context must confirm GHG)
    "ghg": 1.0,
    "t ghg": 1.0,
}

_WATER_TO_KL: dict[str, float] = {
    "kl": 1.0,
    "kilolitre": 1.0,
    "kilolitres": 1.0,
    "kiloliter": 1.0,
    "kiloliters": 1.0,
    "kilo litre": 1.0,
    "kilo litres": 1.0,
    "m3": 1.0,          # 1 m³ = 1 KL
    "m³": 1.0,
    "cubic meter": 1.0,
    "cubic meters": 1.0,
    "cubic metre": 1.0,
    "cubic metres": 1.0,
    # Larger / smaller
    "ml": 0.001,
    "millilitre": 0.001,
    "litre": 0.001,
    "litres": 0.001,
    "liter": 0.001,
    "liters": 0.001,
    "l": 0.001,
    "ml3": 1_000.0,     # megalitres
    "megalitres": 1_000.0,
    "megaliters": 1_000.0,
    "million litre": 1_000.0,
    "million litres": 1_000.0,
    "million liter": 1_000.0,
    "million liters": 1_000.0,
}

_WASTE_TO_MT: dict[str, float] = {
    "mt": 1.0,
    "metric tonne": 1.0,
    "metric tonnes": 1.0,
    "metric ton": 1.0,
    "metric tons": 1.0,
    "tonne": 1.0,
    "tonnes": 1.0,
    "ton": 1.0,
    "tons": 1.0,
    "t": 1.0,           # ambiguous but commonly used in ESG reports for metric tonne
    # Smaller
    "kg": 0.001,
    "kilogram": 0.001,
    "kilograms": 0.001,
    # Larger
    "kt": 1_000.0,
    "kilotonne": 1_000.0,
    "kilotonnes": 1_000.0,
    "thousand tonnes": 1_000.0,
    "lakh tonnes": 100_000.0,
}

_REVENUE_TO_INR_CRORE: dict[str, float] = {
    # INR denominations
    "inr crore": 1.0,
    "crore": 1.0,
    "crores": 1.0,
    "cr": 1.0,
    "rs crore": 1.0,
    "rs. crore": 1.0,
    "inr lakh": 0.01,
    "lakh": 0.01,
    "lakhs": 0.01,
    "inr": 1e-7,        # raw rupees → Crore
    "rupee": 1e-7,
    "rupees": 1e-7,
    # USD (approximate; use for directional comparison only)
    "usd million": 83.0,    # 1 USD mn ≈ 83 Cr (FY2024 rate; rough)
    "million usd": 83.0,
    "$ million": 83.0,
    "usd billion": 83_000.0,
    "billion usd": 83_000.0,
}

# Map KPI category names → conversion table + canonical unit
_CATEGORY_MAP: dict[str, tuple[dict, str]] = {
    "energy":  (_ENERGY_TO_GJ,        "GJ"),
    "ghg":     (_GHG_TO_TCO2E,        "tCO2e"),
    "water":   (_WATER_TO_KL,         "KL"),
    "waste":   (_WASTE_TO_MT,         "MT"),
    "revenue": (_REVENUE_TO_INR_CRORE,"INR_Crore"),
}

# Mapping from KPI definition names → category
KPI_NAME_TO_CATEGORY: dict[str, str] = {
    "energy_consumption":        "energy",
    "total_energy_consumption":  "energy",
    "scope_1_emissions":         "ghg",
    "scope_2_emissions":         "ghg",
    "total_ghg_emissions":       "ghg",
    "ghg_emissions":             "ghg",
    "water_consumption":         "water",
    "total_water_consumption":   "water",
    "waste_generated":           "waste",
    "total_waste_generated":     "waste",
    "revenue":                   "revenue",
    "revenue_from_operations":   "revenue",
}


class NormalizationError(ValueError):
    """Raised when a unit cannot be normalised to the canonical unit."""
    pass


@dataclass
class NormalizedKPI:
    """Result of normalising a single KPI record."""
    kpi_name: str
    raw_value: float
    raw_unit: str
    normalized_value: float
    normalized_unit: str
    category: str
    conversion_factor: float


def _clean_unit(unit: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for table lookup."""
    u = unit.lower().strip()
    u = re.sub(r"[/\-_]", " ", u)
    u = re.sub(r"\s+", " ", u)
    return u


def infer_category(kpi_name: str) -> Optional[str]:
    """
    Infer the KPI category from its name.
    Returns None if unknown — caller should pass category explicitly.
    """
    name_lower = kpi_name.lower().strip()
    # Direct match
    if name_lower in KPI_NAME_TO_CATEGORY:
        return KPI_NAME_TO_CATEGORY[name_lower]
    # Substring match — order matters (more specific first)
    for kw, cat in [
        ("scope_1", "ghg"), ("scope_2", "ghg"), ("scope 1", "ghg"), ("scope 2", "ghg"),
        ("ghg", "ghg"), ("emission", "ghg"), ("co2", "ghg"),
        ("energy", "energy"),
        ("water", "water"),
        ("waste", "waste"),
        ("revenue", "revenue"), ("turnover", "revenue"),
    ]:
        if kw in name_lower:
            return cat
    return None


def normalize(
    kpi_name: str,
    value: float,
    unit: str,
    category: Optional[str] = None,
) -> NormalizedKPI:
    """
    Normalise a single KPI value to its canonical unit.

    Args:
        kpi_name: Name of the KPI (e.g. "scope_1_emissions")
        value:    Raw numeric value as extracted
        unit:     Unit string as extracted (e.g. "MJ", "tCO2e", "KL")
        category: Optional override ("energy"|"ghg"|"water"|"waste"|"revenue")

    Returns:
        NormalizedKPI with .normalized_value in canonical units

    Raises:
        NormalizationError if the unit is unknown for the inferred category
    """
    cat = category or infer_category(kpi_name)
    if cat is None:
        raise NormalizationError(
            f"Cannot infer category for KPI '{kpi_name}'. "
            f"Pass category= explicitly."
        )

    if cat not in _CATEGORY_MAP:
        raise NormalizationError(f"Unknown category '{cat}'")

    conv_table, canonical_unit = _CATEGORY_MAP[cat]
    cleaned = _clean_unit(unit)

    factor = conv_table.get(cleaned)
    if factor is None:
        # Try prefix stripping (e.g. "metric tonnes CO2e" → "tco2e")
        # Remove common prefixes before lookup
        for prefix in ("metric ", "total ", "absolute "):
            if cleaned.startswith(prefix):
                factor = conv_table.get(cleaned[len(prefix):])
                if factor:
                    break

    if factor is None:
        raise NormalizationError(
            f"Unknown unit '{unit}' (cleaned: '{cleaned}') "
            f"for category '{cat}'. "
            f"Canonical unit is '{canonical_unit}'."
        )

    return NormalizedKPI(
        kpi_name=kpi_name,
        raw_value=value,
        raw_unit=unit,
        normalized_value=value * factor,
        normalized_unit=canonical_unit,
        category=cat,
        conversion_factor=factor,
    )


def normalize_batch(
    records: list[dict],
) -> list[NormalizedKPI]:
    """
    Normalise a list of KPI dicts.

    Each dict must have: kpi_name, value (float), unit (str).
    Optional: category (str).

    Silently skips records where value is None.
    Raises NormalizationError for unknown units.
    """
    results = []
    for rec in records:
        val = rec.get("value")
        if val is None:
            continue
        results.append(normalize(
            kpi_name=rec["kpi_name"],
            value=float(val),
            unit=rec.get("unit") or "",
            category=rec.get("category"),
        ))
    return results