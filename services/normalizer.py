"""
services/normalizer.py

Unit normalisation for ESG KPIs extracted from corporate reports.

v2 additions
------------
- "count"      category: employee_count, complaints_filed, complaints_pending
- "percentage" category: women_in_workforce_percentage, renewable_energy_percentage
- KPI_NAME_TO_CATEGORY extended for all active KPIs

Design principles (unchanged)
- No company-specific logic
- Canonical units: GJ (energy), tCO2e (GHG), KL (water), MT (waste),
                   INR_Crore (revenue), count (headcount), % (percentage)
- Raises NormalizationError on unknown units rather than silently coercing
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# ── Canonical output units ────────────────────────────────────────────────────
CANONICAL = {
    "energy":     "GJ",
    "ghg":        "tCO2e",
    "water":      "KL",
    "waste":      "MT",
    "revenue":    "INR_Crore",
    "count":      "count",       # NEW – headcount, complaint counts
    "percentage": "%",           # NEW – women %, renewable %
}

# ── Conversion tables ─────────────────────────────────────────────────────────

_ENERGY_TO_GJ: dict[str, float] = {
    "j":  1e-9,
    "kj": 1e-6,
    "mj": 1e-3,          # Megajoules  — common in TCS BRSR
    "gj": 1.0,
    "tj": 1e3,
    "pj": 1e6,
    "wh":  3.6e-6,
    "kwh": 3.6e-3,
    "mwh": 3.6,
    "gwh": 3_600.0,
    "twh": 3_600_000.0,
    "megajoule":  1e-3,
    "megajoules": 1e-3,
    "gigajoule":  1.0,
    "gigajoules": 1.0,
    "terajoule":  1e3,
    "terajoules": 1e3,
    "megawatt hour":  3.6,
    "megawatt hours": 3.6,
    "gigawatt hour":  3_600.0,
    "gigawatt hours": 3_600.0,
    "kilowatt hour":  3.6e-3,
    "kilowatt hours": 3.6e-3,
    "btu":   1.0551e-6,
    "mmbtu": 1.0551,
}

_GHG_TO_TCO2E: dict[str, float] = {
    "tco2e": 1.0,
    "tco2":  1.0,
    "t co2e": 1.0,
    "t co2":  1.0,
    "tonne co2e":  1.0,
    "tonnes co2e": 1.0,
    "ton co2e":  1.0,
    "tons co2e": 1.0,
    "metric tonne co2e":  1.0,
    "metric tonnes co2e": 1.0,
    "metric ton co2e":  1.0,
    "metric tons co2e": 1.0,
    "metric tonnes of co2 equivalent":  1.0,
    "metric tonnes of co2equivalent":   1.0,
    "metric tonnes co2 equivalent":     1.0,
    "tonnes of co2 equivalent":  1.0,
    "tonnes co2 equivalent":     1.0,
    "tonne of co2 equivalent":   1.0,
    "co2 equivalent": 1.0,
    "co2e":           1.0,
    "ktco2e": 1_000.0,
    "kt co2e": 1_000.0,
    "ktco2":   1_000.0,
    "kt co2":  1_000.0,
    "kilo tonne co2e":     1_000.0,
    "kilotonne co2e":      1_000.0,
    "kilotonnes co2e":     1_000.0,
    "thousand tonnes co2e":        1_000.0,
    "thousand metric tonnes co2e": 1_000.0,
    "million tonnes co2e":             1_000_000.0,
    "million tonne co2e":              1_000_000.0,
    "million metric tonnes co2e":      1_000_000.0,
    "million metric tonne co2e":       1_000_000.0,
    "million tonnes of co2e":          1_000_000.0,
    "million tonnes of co2 equivalent":1_000_000.0,
    "million tonne co2 equivalent":    1_000_000.0,
    "million tons co2e":  1_000_000.0,
    "million ton co2e":   1_000_000.0,
    "mtco2e": 1_000_000.0,
    "mt co2e": 1_000_000.0,
    "mtco2":  1_000_000.0,
    "mt co2": 1_000_000.0,
    "million tonnes coe": 1_000_000.0,
    "million co2e":       1_000_000.0,
    "ghg": 1.0,
    "t ghg": 1.0,
}

_WATER_TO_KL: dict[str, float] = {
    "kl": 1.0,
    "kilolitre":  1.0,
    "kilolitres": 1.0,
    "kiloliter":  1.0,
    "kiloliters": 1.0,
    "kilo litre":  1.0,
    "kilo litres": 1.0,
    "m3": 1.0,
    "m\u00b3": 1.0,
    "cubic meter":  1.0,
    "cubic meters": 1.0,
    "cubic metre":  1.0,
    "cubic metres": 1.0,
    "ml": 0.001,
    "millilitre": 0.001,
    "litre":  0.001,
    "litres": 0.001,
    "liter":  0.001,
    "liters": 0.001,
    "l": 0.001,
    "ml3": 1_000.0,
    "megalitres": 1_000.0,
    "megaliters": 1_000.0,
    "million litre":  1_000.0,
    "million litres": 1_000.0,
    "million liter":  1_000.0,
    "million liters": 1_000.0,
}

_WASTE_TO_MT: dict[str, float] = {
    "mt": 1.0,
    "metric tonne":  1.0,
    "metric tonnes": 1.0,
    "metric ton":  1.0,
    "metric tons": 1.0,
    "tonne":  1.0,
    "tonnes": 1.0,
    "ton":  1.0,
    "tons": 1.0,
    "t": 1.0,
    "kg": 0.001,
    "kilogram":  0.001,
    "kilograms": 0.001,
    "kt": 1_000.0,
    "kilotonne":  1_000.0,
    "kilotonnes": 1_000.0,
    "thousand tonnes": 1_000.0,
    "lakh tonnes": 100_000.0,
}

_REVENUE_TO_INR_CRORE: dict[str, float] = {
    "inr crore": 1.0,
    "crore":  1.0,
    "crores": 1.0,
    "cr":     1.0,
    "rs crore":  1.0,
    "rs. crore": 1.0,
    "inr lakh": 0.01,
    "lakh":  0.01,
    "lakhs": 0.01,
    "inr":    1e-7,
    "rupee":  1e-7,
    "rupees": 1e-7,
    "usd million":  83.0,
    "million usd":  83.0,
    "$ million":    83.0,
    "usd billion":  83_000.0,
    "billion usd":  83_000.0,
}

# NEW – count category (factor always 1.0; unit is just a label)
_COUNT_TO_COUNT: dict[str, float] = {
    "count":     1.0,
    "headcount": 1.0,
    "number":    1.0,
    "nos":       1.0,
    "employees": 1.0,
    "persons":   1.0,
    "people":    1.0,
    "workers":   1.0,
    "complaints":  1.0,
    "grievances":  1.0,
    "cases":       1.0,
}

# NEW – percentage category (factor always 1.0; value is already in %)
_PCT_TO_PCT: dict[str, float] = {
    "%": 1.0,
    "percent":    1.0,
    "per cent":   1.0,
    "percentage": 1.0,
}

# Map category → (conversion_table, canonical_unit)
_CATEGORY_MAP: dict[str, tuple[dict, str]] = {
    "energy":     (_ENERGY_TO_GJ,        "GJ"),
    "ghg":        (_GHG_TO_TCO2E,        "tCO2e"),
    "water":      (_WATER_TO_KL,         "KL"),
    "waste":      (_WASTE_TO_MT,         "MT"),
    "revenue":    (_REVENUE_TO_INR_CRORE,"INR_Crore"),
    "count":      (_COUNT_TO_COUNT,      "count"),      # NEW
    "percentage": (_PCT_TO_PCT,          "%"),          # NEW
}

# Map KPI name → category
KPI_NAME_TO_CATEGORY: dict[str, str] = {
    # Energy
    "energy_consumption":       "energy",
    "total_energy_consumption": "energy",
    # GHG
    "scope_1_emissions":        "ghg",
    "scope_2_emissions":        "ghg",
    "scope_3_emissions":        "ghg",       # NEW
    "total_ghg_emissions":      "ghg",
    "ghg_emissions":            "ghg",
    # Water
    "water_consumption":        "water",     # NEW
    "total_water_consumption":  "water",
    # Waste
    "waste_generated":          "waste",
    "total_waste_generated":    "waste",
    # Revenue
    "revenue":                  "revenue",
    "revenue_from_operations":  "revenue",
    # Count — NEW
    "employee_count":                    "count",
    "complaints_filed":                  "count",
    "complaints_pending":                "count",
    # Percentage — NEW
    "women_in_workforce_percentage":     "percentage",
    "renewable_energy_percentage":       "percentage",
}


class NormalizationError(ValueError):
    """Raised when a unit cannot be normalised to the canonical unit."""
    pass


@dataclass
class NormalizedKPI:
    """Result of normalising a single KPI record."""
    kpi_name:         str
    raw_value:        float
    raw_unit:         str
    normalized_value: float
    normalized_unit:  str
    category:         str
    conversion_factor: float


def _clean_unit(unit: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for table lookup."""
    u = unit.lower().strip()
    u = re.sub(r"[/\-_]", " ", u)
    u = re.sub(r"\s+", " ", u)
    u = u.strip(".")
    return u


def infer_category(kpi_name: str) -> Optional[str]:
    """
    Infer the KPI category from its name.
    Returns None if unknown.
    """
    name_lower = kpi_name.lower().strip()
    if name_lower in KPI_NAME_TO_CATEGORY:
        return KPI_NAME_TO_CATEGORY[name_lower]
    # Substring match — ordered most-specific first
    for kw, cat in [
        ("scope_1", "ghg"), ("scope_2", "ghg"), ("scope_3", "ghg"),
        ("scope 1", "ghg"), ("scope 2", "ghg"), ("scope 3", "ghg"),
        ("ghg", "ghg"), ("emission", "ghg"), ("co2", "ghg"),
        ("energy", "energy"),
        ("water", "water"),
        ("waste", "waste"),
        ("revenue", "revenue"), ("turnover", "revenue"),
        ("employee", "count"), ("headcount", "count"),
        ("complaint", "count"), ("grievance", "count"),
        ("women", "percentage"), ("female", "percentage"),
        ("renewable", "percentage"), ("percent", "percentage"),
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

    For "count" and "percentage" categories the conversion factor is always
    1.0 — the value is already in canonical units.

    Raises NormalizationError if the unit is unknown for the inferred category.
    """
    cat = category or infer_category(kpi_name)
    if cat is None:
        raise NormalizationError(
            f"Cannot infer category for KPI '{kpi_name}'. Pass category= explicitly."
        )
    if cat not in _CATEGORY_MAP:
        raise NormalizationError(f"Unknown category '{cat}'")

    conv_table, canonical_unit = _CATEGORY_MAP[cat]

    # "count" and "percentage" — accept any reasonable label with factor 1.0
    if cat == "count":
        return NormalizedKPI(
            kpi_name=kpi_name, raw_value=value, raw_unit=unit,
            normalized_value=value, normalized_unit=canonical_unit,
            category=cat, conversion_factor=1.0,
        )
    if cat == "percentage":
        # Guard: percentage values must be 0-100
        if not (0 <= value <= 100):
            raise NormalizationError(
                f"Percentage value {value} out of range [0, 100] for '{kpi_name}'"
            )
        return NormalizedKPI(
            kpi_name=kpi_name, raw_value=value, raw_unit=unit,
            normalized_value=value, normalized_unit=canonical_unit,
            category=cat, conversion_factor=1.0,
        )

    cleaned = _clean_unit(unit)
    factor  = conv_table.get(cleaned)

    if factor is None:
        for prefix in ("metric ", "total ", "absolute "):
            if cleaned.startswith(prefix):
                stripped = cleaned[len(prefix):]
                factor = conv_table.get(stripped)
                if factor:
                    break

    if factor is None:
        for suffix in (" equivalent", " emissions", " emission"):
            if cleaned.endswith(suffix):
                stripped = cleaned[: -len(suffix)].strip()
                factor = conv_table.get(stripped)
                if factor:
                    break

    if factor is None:
        raise NormalizationError(
            f"Unknown unit '{unit}' (cleaned: '{cleaned}') "
            f"for category '{cat}'. Canonical unit is '{canonical_unit}'."
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


def normalize_batch(records: list[dict]) -> list[NormalizedKPI]:
    """
    Normalise a list of KPI dicts.
    Each dict must have: kpi_name, value (float), unit (str).
    Optional: category (str). Skips records where value is None.
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