"""
agents/extraction_agent.py

KPI Extraction Agent — Phase 4 (v3, verified against real PDF chunk text).

Every pattern was tested against the EXACT strings that pdfplumber layout=True
produces for the Infosys 2024 Integrated Annual Report before being committed.

Root causes fixed
-----------------
scope_3_emissions
    The Integrated report layout (pdfplumber layout=True) produces:
      LINE 1: "Total Scope 3 emissions (Break-up of the GHG into CO2, Metric tonnes of CO2"
      LINE 2: "                                         1,80,737   1,83,976 (2)"
      LINE 3: "CH4, N2O, HFCs, PFCs, SF6, NF3, if available)(1) equivalent"
    The value is on line 2; "equivalent" appears AFTER the value on line 3.
    Old patterns assumed "equivalent" preceded the value, so they all failed.
    New patterns A/B cross the newline explicitly with [^\n]*\\n\\s*.

renewable_energy_percentage
    The Integrated report says:
      "We get 67.52% of electricity for\\n     our India operations from renewable"
    The percentage and the renewable keyword are split across a line break.
    None of the old patterns allowed for a newline between them.
    New pattern N_crossline: r"(pct)%[^\n]*\\n[^\n]*(?:renew\\w*|met\\s+through)"

complaints_filed / complaints_pending  (new KPIs)
    BRSR Section 25 row format (pdfplumber layout=True):
      "Employees and workers HEAR@infosys.com, GRB@infosys.com 180 19"
    First number = filed, second = pending.
    Retrieval keywords include "employees and workers" so the chunk is found.

Invariants
----------
  All pre-existing KPI patterns are preserved without modification.
  New patterns are appended to their respective lists.
  No changes to retrieval, chunking, or DB layers.
"""
from __future__ import annotations

import re
import uuid
from typing import Optional

from sqlalchemy import or_
from sqlalchemy.orm import Session

from core.config import get_settings
from core.logging_config import get_logger
from models.db_models import DocumentChunk, KPIDefinition, ParsedDocument, Report
from models.schemas import ExtractedKPI
from services.kpi_service import KPIService
from services.llm_service import LLMService
from services.retrieval_service import HybridRetrievalService, RetrievalService, ScoredChunk

logger = get_logger(__name__)

_REGEX_HIGH_CONFIDENCE = 0.88
_LLM_CHUNK_CHAR_LIMIT = 6000

_DELTA_CONTEXT_RE = re.compile(
    r"\b("
    r"reduc(?:ed|es?|tion|ing)|"
    r"declin(?:ed|e|ing)|"
    r"decreas(?:ed|es?|e|ing)|"
    r"avoid(?:ed|ance|ing|s?)|"
    r"offset(?:ting|s)?|"
    r"sequester(?:ed|ing)?|"
    r"sav(?:ed|ing|ings)|"
    r"target(?:ed|ing|s?)?\s+(?:of|to|at)|"
    r"goal\s+(?:of|to)|"
    r"aim(?:s|ing)?\s+(?:to|for)|"
    r"commit(?:ted|ment)\s+(?:to|of)|"
    r"plan(?:ned|s|ning)?\s+(?:to|for)|"
    r"by\s+(?:fy|20)\d{2}|"
    r"per\s+(?:employee|fte|unit|sqm|sq\.?\s*m|revenue|capita|tonne|kwh)|"
    r"intensit(?:y|ies)|"
    r"baseline\s+(?:of|year|value)|"
    r"compared\s+to\s+(?:fy|20)\d{2}|"
    r"vs\.?\s+(?:fy|20)\d{2}|"
    r"from\s+(?:fy|20)\d{2}\s+(?:to|level)|"
    r"improvement\s+(?:of|in)|"
    r"increase[sd]?\s+by|"
    r"net\s+(?:zero|neutral)|"
    r"emission\s+factor"
    r")\b",
    re.IGNORECASE,
)

# Indian comma-separated number (e.g. 1,80,737 or 55,881)
_IND_NUM = r"[\d]{1,3}(?:,\d{2,3})+"


def _get_sentence_context(text: str, match_start: int, window: int = 120) -> str:
    start = max(0, match_start - window)
    end   = min(len(text), match_start + window)
    snippet = text[start:end]
    sentences = re.split(r"[.!?\n]", snippet)
    for sent in sentences:
        if str(match_start - start) and len(sent) > 5:
            return sent
    return snippet


def _is_delta_context(text: str, match_start: int) -> bool:
    context = _get_sentence_context(text, match_start, window=150)
    return bool(_DELTA_CONTEXT_RE.search(context))


# ---------------------------------------------------------------------------
# Unit synonym map
# ---------------------------------------------------------------------------
_UNIT_SYNONYMS: dict[str, str] = {
    "tco2e": "tCO2e", "tco2": "tCO2e", "t co2e": "tCO2e",
    "mtco2e": "tCO2e", "mt co2e": "tCO2e",
    "ktco2e": "tCO2e", "kt co2e": "tCO2e",
    "tonnes co2e": "tCO2e", "tons co2e": "tCO2e",
    "mwh": "MWh", "gwh": "GWh", "twh": "TWh",
    "gj": "GJ", "tj": "TJ", "pj": "PJ",
    "kwh": "kWh", "mj": "MJ",
    "kl": "KL", "kilolitre": "KL", "kiloliter": "KL", "kilo litre": "KL",
    "m3": "m\u00b3", "cubic meter": "m\u00b3", "cubic metre": "m\u00b3",
    "mt": "MT", "metric ton": "MT", "metric tonne": "MT",
    "kg": "kg", "tonne": "MT",
    "%": "%", "percent": "%", "per cent": "%",
    "inr crore": "INR Crore", "crore": "INR Crore", "cr": "INR Crore",
    "inr lakh": "INR Lakh", "lakh": "INR Lakh",
    "number": "count", "nos": "count", "headcount": "count",
}

# ---------------------------------------------------------------------------
# KPI aliases
# ---------------------------------------------------------------------------
_KPI_ALIASES: dict[str, list[str]] = {
    "scope_1_emissions": [
        "scope 1", "direct emissions", "direct ghg", "fuel combustion",
        "stationary combustion", "owned vehicles", "fugitive emissions",
    ],
    "scope_2_emissions": [
        "scope 2", "indirect emissions", "purchased electricity",
        "electricity consumption ghg", "market-based", "location-based",
    ],
    "total_ghg_emissions": [
        "total emissions", "total ghg", "scope 1 and 2", "scope 1+2",
        "combined emissions", "carbon footprint", "co2 equivalent",
        "greenhouse gas", "ghg inventory", "carbon neutral",
    ],
    "energy_consumption": [
        "total energy", "energy consumed", "energy use", "energy usage",
        "electricity consumed", "fuel consumed", "energy intensity",
        "gigajoules", "megawatt hours", "energy consumption",
        "total energy consumed", "total energy consumption",
        "energy used", "electricity consumption", "fuel consumption",
        "power consumption", "thermal energy", "renewable energy consumed",
        "non-renewable energy", "energy from grid",
    ],
    "renewable_energy_percentage": [
        "renewable energy", "solar energy", "wind energy", "clean energy",
        "green energy", "renewable electricity", "non-fossil", "re share",
        "percent renewable", "% renewable",
        # Covers Integrated-report phrasing
        "share of renewables", "renewables", "met through renewables",
        "from renewable", "electricity from renewable",
        "renewable sources", "solar pv", "green power",
    ],
    "water_consumption": [
        "water consumed", "water usage", "water use", "water withdrawal",
        "freshwater", "water recycled", "water intensity", "kilolitres",
        "water consumption", "total water", "water intake",
        "water used", "total water intake", "water withdrawn",
        "water abstraction", "water sourced", "ground water",
        "surface water", "municipal water", "third party water",
        "total freshwater consumption", "water discharge",
        "net water consumption", "water footprint",
    ],
    "waste_generated": [
        "waste generated", "total waste", "solid waste", "hazardous waste",
        "non-hazardous waste", "waste disposed", "waste diverted",
    ],
    "employee_count": [
        "total employees", "total workforce", "number of employees",
        "headcount", "fte", "full time", "permanent employees",
        "workforce strength", "employee base",
    ],
    "women_in_workforce_percentage": [
        "women", "female", "gender diversity", "women employees",
        "female employees", "gender ratio", "women in workforce",
    ],
    "scope_3_emissions": [
        "scope 3", "scope-3",
        "indirect value chain ghg", "supply chain emissions",
        "scope 3 category", "business travel emissions",
        "purchased goods and services", "use of sold products",
        "capital goods emissions", "transportation and distribution",
        "processing of sold products", "franchises",
        "upstream scope 3", "downstream scope 3",
        "total scope 3", "scope 3 tco2e",
        "other indirect emissions", "indirect ghg scope 3",
        "scope 3 carbon", "scope 3 footprint",
    ],
    "complaints_filed": [
        # These keywords land in the chunk keyword index so retrieval finds it
        "complaints filed", "complaints received",
        "number of complaints filed", "complaints filed during the year",
        "grievances filed", "grievances received", "complaints lodged",
        "filed during the year", "sexual harassment complaints",
        "stakeholder complaints", "working conditions complaints",
        "ngrbc complaints",
        # BRSR table terms that ARE in chunk keywords
        "employees and workers", "shareholders", "customers",
        "filed during", "pending resolution",
    ],
    "complaints_pending": [
        "complaints pending", "pending complaints",
        "pending resolution", "complaints pending resolution",
        "pending at close of year", "pending at end of year",
        "unresolved complaints", "grievances pending",
        "pending resolution at close",
        "employees and workers", "shareholders", "customers",
        "filed during", "close of the year",
    ],
}


def _normalise_unit(unit_str: str) -> str:
    if not unit_str:
        return unit_str
    key = unit_str.lower().strip()
    return _UNIT_SYNONYMS.get(key, unit_str.strip())


def _parse_number(s: str) -> Optional[float]:
    """Parse number strings including Indian number format (1,00,000)."""
    if not s:
        return None
    try:
        cleaned = s.replace(",", "").replace(" ", "").strip()
        return float(cleaned)
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Layer 1 — Regex: block patterns and broad patterns
# ---------------------------------------------------------------------------

_NUM      = r"([\d,]+(?:\.\d+)?)(?:\(\d+\))?"
_WS       = r"[\s:--|]*"
_UNIT_PAT = (
    r"(tco2e?|t\s*co2e?|mt\s*co2e?|kt\s*co2e?|tonnes?\s*co2e?|"
    r"mwh|gwh|gj|tj|kl|kilolitr\w*|m3|mt|metric\s*tonn?\w*|"
    r"%|percent|crore|lakh|number|nos|headcount)"
)

_BROAD_PATTERNS = [
    rf"(?:was|is|were|of|:)\s*{_NUM}\s*{_UNIT_PAT}",
    rf"{_NUM}\s*{_UNIT_PAT}\s*(?:in|for|during|fy|fiscal)",
    rf"\|\s*{_NUM}\s*\|\s*{_UNIT_PAT}",
    # rf"(?:total(?:led|s)?|amount(?:ed|s)?|reached|stood at|equat\w+)\s*{_WS}{_NUM}\s*{_UNIT_PAT}",
]


def _try_block_patterns(chunk, kpi: "KPIDefinition") -> Optional["ExtractedKPI"]:
    """
    Block-aware patterns.  Each list is ordered: most-specific first.
    Returns on first positive match.

    Key design decisions
    --------------------
    scope_3_emissions
        Patterns A/B use [^\n]*\\n\\s* to cross the newline explicitly.
        This is necessary because in pdfplumber layout=True output the value
        is on the line IMMEDIATELY AFTER the label, not after "equivalent".

    renewable_energy_percentage
        Pattern N_crossline allows a line break between the percentage and
        the renewable keyword, covering the Integrated-report layout.

    complaints_filed / complaints_pending
        Pattern P1/Q1 targets the BRSR Section 25 row directly.
        "Employees and workers HEAR@infosys.com, GRB@infosys.com 180 19"
        P1 picks the first number (180 = filed); Q1 picks the second (19 = pending).
    """
    text = chunk.content

    block_patterns_for_kpi: dict[str, list[str]] = {
        # ── Existing KPIs (unchanged) ────────────────────────────────────────
        "energy_consumption": [
            r"total\s+energy\s+consumed\s*\(A[^)]*D[^)]*F\)[^\n]*\n([\d,]+(?:\.\d+)?)(?:\(\d+\))?",
            r"total\s+energy\s+consumed\s*\((?:A[^)]*)\)[^\n]*\n([\d,]+(?:\.\d+)?)(?:\(\d+\))?",
            r"total\s+energy\s+consumption[^\n]{0,80}\n([\d,]+(?:\.\d+)?)(?:\(\d+\))?",
            r"total\s+energy\s+consumed[^\n]{0,80}\n([\d,]+(?:\.\d+)?)(?:\(\d+\))?",
        ],
        "water_consumption": [
            r"total\s+volume\s+of\s+water\s+consumption[^\n]*\n([\d,]+(?:\.\d+)?)(?:\(\d+\))?",
            r"total\s+water\s+consumption[^\n]*\n([\d,]+(?:\.\d+)?)(?:\(\d+\))?",
            r"total\s+water\s+withdrawn[^\n]{0,80}\n([\d,]+(?:\.\d+)?)(?:\(\d+\))?",
            r"total\s+water\s+(?:intake|sourced|used)[^\n]{0,80}\n([\d,]+(?:\.\d+)?)(?:\(\d+\))?",
        ],
        "waste_generated": [
            r"^total\s*\([a-h\s\+]+\)\s*\n([\d,]+(?:\.\d+)?)(?:\(\d+\))?",
            r"total\s+\(A\s*\+\s*B[^\n]*\)\s*\n([\d,]+(?:\.\d+)?)(?:\(\d+\))?",
        ],
        "scope_1_emissions": [
            r"total\s+scope\s*1\s+emissions.{0,200}?metric\s+tonnes\s+of\s+([\d,]+(?:\.\d+)?)(?:\(\d+\))?",
            r"total\s+scope\s*1\s+emissions[\s\S]{0,300}?equivalent\n\s*([\d,]+(?:\.\d+)?)(?:\(\d+\))?",
        ],
        "scope_2_emissions": [
            r"total\s+scope\s*2\s+emissions.{0,200}?metric\s+tonnes\s+of\s+([\d,]+(?:\.\d+)?)(?:\(\d+\))?",
            r"total\s+scope\s*2\s+emissions[\s\S]{0,300}?equivalent\n\s*([\d,]+(?:\.\d+)?)(?:\(\d+\))?",
        ],
        "total_ghg_emissions": [],

        # ── scope_3_emissions ────────────────────────────────────────────────
        "scope_3_emissions": [
            # ── Existing (unchanged) ─────────────────────────────────────────
            r"total\s+scope\s*3\s+emissions.{0,200}?metric\s+tonnes\s+of\s+([\d,]+(?:\.\d+)?)(?:\(\d+\))?",
            r"total\s+scope\s*3\s+emissions[\s\S]{0,300}?equivalent\n\s*([\d,]+(?:\.\d+)?)(?:\(\d+\))?",
            r"scope\s*3\s+(?:ghg\s+)?emissions[^\n]{0,80}\n\s*([\d,]+(?:\.\d+)?)(?:\(\d+\))?",
            r"total\s+value\s+chain\s+emissions[^\n]{0,80}\n\s*([\d,]+(?:\.\d+)?)(?:\(\d+\))?",
            r"other\s+indirect\s+\(scope\s*3\)[^\n]{0,80}\n\s*([\d,]+(?:\.\d+)?)(?:\(\d+\))?",

            # ── Pattern A — "total scope 3 emissions" then value on next line ─
            # Verified: pdfplumber layout=True page 167 of Infosys 2024 Integrated
            r"total\s+scope\s*3\s+emissions[^\n]*\n\s*(" + _IND_NUM + r"(?:\.\d+)?)",

            # ── Pattern B — any "scope 3 emissions" label then next line ─────
            r"scope\s*3\s+(?:ghg\s+)?emissions[^\n]*\n\s*(" + _IND_NUM + r"(?:\.\d+)?)",

            # ── Pattern C — "other indirect (scope 3)" then next line ────────
            r"other\s+indirect\s+\(?scope\s*3\)?[^\n]*\n\s*(" + _IND_NUM + r"(?:\.\d+)?)",

            # ── Pattern D — inline value with GHG unit on same line ──────────
            r"scope\s*3\s+(?:ghg\s+)?emissions[^\n]{0,100}?("
            + _IND_NUM + r"(?:\.\d+)?)\s*(?:tco2e?|metric\s*tonn\w*)",

            # ── Pattern E — dotall search for "equivalent" then newline+value ─
            r"total\s+scope\s*3\s+emissions[\s\S]{0,400}?equivalent[^\n]*\n\s*("
            + _IND_NUM + r"(?:\.\d+)?)",
        ],

        # ── renewable_energy_percentage ──────────────────────────────────────
        "renewable_energy_percentage": [
            # ── Existing (unchanged) ─────────────────────────────────────────
            r"renewable\s+energy[\s\S]{0,80}?([\d]+(?:\.\d+)?)\s*%",
            r"([\d]+(?:\.\d+)?)\s*%[\s\S]{0,40}?(?:from\s+)?renewable",
            r"renewable[\s\S]{0,40}?([\d]+(?:\.\d+)?)\s*percent",

            # ── N_crossline — % on line N, renewable keyword on line N+1 ────
            # Verified: pages 26 and 65 of Infosys 2024 Integrated
            r"([\d]+(?:\.\d+)?)\s*%[^\n]*\n[^\n]*(?:renew\w*|met\s+through)",

            # ── N_share — "Share of renewables ... X%" ───────────────────────
            # Verified: page 22 of Infosys 2024 Integrated
            r"share\s+of\s+renew\w*[^\n]{0,80}?([\d]+(?:\.\d+)?)\s*%",

            # ── N_elec — "X% of electricity ... renewable" (same line) ──────
            r"([\d]+(?:\.\d+)?)\s*%\s+of\s+(?:\w+\s+){0,5}electricity[^\n]{0,150}renew\w*",

            # ── N_ml — multiline "X% ... from renewable" (no "sources" req.) ─
            r"([\d]+(?:\.\d+)?)\s*%[\s\S]{0,200}?from\s+renew\w+",

            # ── N_rev — reverse "renewables ... X%" (2+ digit guard) ─────────
            r"renew(?:able\s+sources?|ables?|able\s+energy)\b[^\n]{0,120}([\d]{2,}(?:\.\d+)?)\s*%",

            # ── N_solar — solar/wind/hydro sub-type ──────────────────────────
            r"(?:solar|wind|hydro|geotherm\w+)\s+energy[^\n]{0,80}?([\d]+(?:\.\d+)?)\s*%",
        ],

        # ── complaints_filed ─────────────────────────────────────────────────
        "complaints_filed": [
            # ── P1: BRSR Sec 25 row — "Employees and workers ... 180 19" ────
            # Filed = first number; verified on page 135 Infosys 2024 Integrated
            r"employees?\s+and\s+workers?[^\n]{0,250}?(\b\d{2,}\b)\s+\d+",

            # ── P2: "filed during the year: N" ───────────────────────────────
            r"filed\s+during\s+(?:the\s+)?year[^\n]{0,60}?[:\s]+(\d[\d,]*)",

            # ── P3: "N complaints were filed" ────────────────────────────────
            r"(\d[\d,]+)\s+complaints?\s+(?:were\s+)?filed",

            # ── P4: "complaints received: N" ─────────────────────────────────
            r"complaints?\s+received[^\n]{0,60}?[:\s]+(\d[\d,]+)",

            # ── P5: "complaints filed: N" ─────────────────────────────────────
            r"complaints?\s+filed[^\n]{0,60}?[:\s]+(\d[\d,]+)",

            # ── P6: "Number of complaints filed ... N" ────────────────────────
            r"number\s+of\s+complaints?\s+filed[^\n]{0,100}?(\d[\d,]+)",

            # ── P7: "Total complaints filed/received N" ───────────────────────
            r"total\s+complaints?\s+(?:filed|received)[^\n]{0,80}?(\d[\d,]+)",

            # ── P8: "N complaints were lodged/raised/reported" ────────────────
            r"(\d[\d,]+)\s+complaints?\s+(?:were\s+)?(?:lodged|raised|reported|submitted)",

            # ── P9: H&S table — "Working conditions 16 0" ────────────────────
            r"(?:working\s+conditions?|health\s+and\s+safety)[^\n]{0,80}?(\d+)\s+\d+",

            # ── P10: "grievances filed/received: N" ───────────────────────────
            r"grievances?\s+(?:filed|received)[^\n]{0,80}?[:\s]+(\d[\d,]+)",
        ],

        # ── complaints_pending ───────────────────────────────────────────────
        "complaints_pending": [
            # ── Q1: BRSR Sec 25 row — "Employees and workers ... 180 19" ────
            # Pending = second number; verified on page 135 Infosys 2024 Integrated
            r"employees?\s+and\s+workers?[^\n]{0,250}?\d[\d,]+\s+(\b\d+\b)",

            # ── Q2: "pending resolution at close/end ... N" ───────────────────
            r"pending\s+resolution\s+at\s+(?:close|end)[^\n]{0,100}?(\d[\d,]*)",

            # ── Q3: "N complaints are pending" ────────────────────────────────
            r"(\d[\d,]*)\s+(?:complaints?\s+)?(?:are\s+)?pending\b",

            # ── Q4: "complaints pending: N" ────────────────────────────────────
            r"complaints?\s+pending\s+(?:resolution\s+)?[^\n]{0,60}?[:\s]+(\d[\d,]*)",

            # ── Q5: "pending: N" / "outstanding: N" ──────────────────────────
            r"(?:pending|outstanding)[^\n]{0,40}?[:\s]+(\d+)",

            # ── Q6: "close of the year\nN" (table-cell layout) ───────────────
            r"close\s+of\s+the\s+year[^\n]{0,20}\n\s*(\d+)",

            # ── Q7: "complaints pending ... N" ────────────────────────────────
            r"complaints?\s+pending[^\n]{0,80}?[:\s]+(\d+)",

            # ── Q8: "Total pending N" ──────────────────────────────────────────
            r"total\s+(?:complaints?\s+)?pending[^\n]{0,80}?(\d+)",

            # ── Q9: "N were pending at year/end" ──────────────────────────────
            r"(\d[\d,]*)\s+(?:complaints?\s+)?(?:were\s+)?pending\s+at\s+(?:year|the\s+(?:end|close))",

            # ── Q10: "grievances pending: N" ──────────────────────────────────
            r"grievances?\s+pending[^\n]{0,80}?[:\s]+(\d+)",
        ],
    }

    patterns = block_patterns_for_kpi.get(kpi.name, [])
    if not patterns:
        return None

    expected_unit = kpi.expected_unit

    for pat_str in patterns:
        try:
            m = re.search(pat_str, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if not m:
                continue

            raw_val = m.group(1).strip()
            value   = _parse_number(raw_val)
            if value is None or value <= 0:
                continue

            if kpi.name == "renewable_energy_percentage":
                if value > 100:
                    continue
                confidence = 0.90 if chunk.chunk_type == "table" else 0.88

            elif kpi.name in ("complaints_filed", "complaints_pending"):
                if value > 10_000_000:
                    continue
                confidence = 0.88 if chunk.chunk_type == "table" else 0.85

            else:
                if value < 0.1:
                    continue
                confidence = 0.93 if chunk.chunk_type == "table" else 0.88

            logger.info(
                "extraction.block_pattern_hit",
                kpi=kpi.name, value=value, unit=expected_unit,
                chunk_type=chunk.chunk_type, page=chunk.page_number,
            )

            return ExtractedKPI(
                kpi_name=kpi.name,
                raw_value=raw_val,
                normalized_value=value,
                unit=expected_unit,
                extraction_method="regex",
                confidence=confidence,
                source_chunk_id=chunk.id,
                validation_passed=True,
            )

        except re.error:
            continue

    return None


def _try_ghg_row_strategy(
    chunks: list,
    kpi: "KPIDefinition",
) -> Optional["ExtractedKPI"]:
    """
    Handles Infosys-style BRSR GHG table (scope 1, 2, 3).
    Two regex variants: original (CO 2 with space) and new (CO2, no space).
    """
    if kpi.name not in (
        "scope_1_emissions", "scope_2_emissions", "scope_3_emissions"
    ):
        return None

    target_scope = (
        1 if kpi.name == "scope_1_emissions" else
        2 if kpi.name == "scope_2_emissions" else
        3
    )

    _GHG_ROW_PATTERNS = [
        re.compile(
            r"ghg\s+into\s+co.{0,80}\s2\s+([\d,]+(?:\.\d+)?)(?:\(\d+\))?",
            re.IGNORECASE,
        ),
        # pdfplumber "CO2," (no space) — value on next line after label
        re.compile(
            r"break-?up\s+of\s+the\s+ghg[^\n]*\n\s*(" + _IND_NUM + r"(?:\.\d+)?)",
            re.IGNORECASE,
        ),
    ]

    for chunk in chunks:
        text = chunk.content
        tl   = text.lower()

        if "ghg into co" not in tl and "break-up of the ghg" not in tl:
            continue

        scope_context: Optional[int] = None
        for line in text.splitlines():
            ll = line.lower()
            if "total scope 1 emissions" in ll:
                scope_context = 1
            elif "total scope 2 emissions" in ll:
                scope_context = 2
            elif "total scope 3 emissions" in ll or "scope 3 ghg" in ll:
                scope_context = 3
            elif "total scope 4" in ll:
                scope_context = None

        if scope_context != target_scope:
            continue

        for ghg_re in _GHG_ROW_PATTERNS:
            m = ghg_re.search(text)
            if not m:
                continue
            raw   = m.group(1)
            value = _parse_number(raw)
            if value is None or value < 10:
                continue
            logger.info(
                "extraction.ghg_row_strategy_hit",
                kpi=kpi.name, value=value, page=chunk.page_number,
            )
            return ExtractedKPI(
                kpi_name=kpi.name,
                raw_value=raw,
                normalized_value=value,
                unit=kpi.expected_unit,
                extraction_method="regex",
                confidence=0.91 if chunk.chunk_type == "table" else 0.87,
                source_chunk_id=chunk.id,
                validation_passed=True,
            )

    return None


def _try_regex(
    chunks: list[DocumentChunk],
    kpi: KPIDefinition,
) -> Optional[ExtractedKPI]:
    """Block patterns → GHG-row → KPI-specific → broad fallback."""
    kpi_patterns   = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in (kpi.regex_patterns or [])]
    broad_patterns = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _BROAD_PATTERNS]

    sorted_chunks = sorted(chunks, key=lambda c: 0 if c.chunk_type == "table" else 1)
    best: Optional[ExtractedKPI] = None

    skip_delta_kpis = frozenset({
        "complaints_filed", "complaints_pending", "renewable_energy_percentage",
    })

    for chunk in sorted_chunks:
        text = chunk.content

        block_result = _try_block_patterns(chunk, kpi)
        if block_result:
            if kpi.name in skip_delta_kpis or not _is_delta_context(text, 0):
                if best is None or block_result.confidence > best.confidence:
                    best = block_result
                if block_result.confidence >= _REGEX_HIGH_CONFIDENCE:
                    return best

        ghg_row_result = _try_ghg_row_strategy([chunk], kpi)
        if ghg_row_result:
            if best is None or ghg_row_result.confidence > best.confidence:
                best = ghg_row_result
            if ghg_row_result.confidence >= _REGEX_HIGH_CONFIDENCE:
                return best

        for pattern in kpi_patterns:
            match = pattern.search(text)
            if not match or len(match.groups()) < 2:
                continue
            if kpi.name not in skip_delta_kpis and _is_delta_context(text, match.start()):
                logger.debug(
                    "extraction.regex_delta_rejected", kpi=kpi.name,
                    snippet=text[max(0, match.start()-60):match.start()+60],
                )
                continue
            value = _parse_number(match.group(1))
            if value is None:
                continue
            unit       = _normalise_unit(match.group(2))
            confidence = 0.92 if chunk.chunk_type == "table" else 0.80
            logger.info(
                "extraction.regex_hit", kpi=kpi.name, value=value, unit=unit,
                chunk_type=chunk.chunk_type, page=chunk.page_number, pattern="specific",
            )
            result = ExtractedKPI(
                kpi_name=kpi.name, raw_value=match.group(1),
                normalized_value=value, unit=unit,
                extraction_method="regex", confidence=confidence,
                source_chunk_id=chunk.id, validation_passed=True,
            )
            if best is None or result.confidence > best.confidence:
                best = result
            if confidence >= _REGEX_HIGH_CONFIDENCE:
                return best

        # Broad patterns — complaints excluded (too many false positives)
        if kpi.name in ("complaints_filed", "complaints_pending"):
            continue

        kpi_keywords = set(
            w.lower() for kw in (kpi.retrieval_keywords or []) for w in kw.split()
        )
        kpi_keywords.update(
            w.lower() for alias in _KPI_ALIASES.get(kpi.name, []) for w in alias.split()
        )
        text_lower = text.lower()
        if not any(kw in text_lower for kw in kpi_keywords):
            continue

        for pattern in broad_patterns:
            match = pattern.search(text)
            if not match or len(match.groups()) < 2:
                continue
            if kpi.name not in skip_delta_kpis and _is_delta_context(text, match.start()):
                continue
            value = _parse_number(match.group(1))
            if value is None:
                continue
            unit = _normalise_unit(match.group(2))
            if kpi.expected_unit and unit.lower() != kpi.expected_unit.lower():
                continue
            confidence = 0.65 if chunk.chunk_type == "table" else 0.55
            logger.info(
                "extraction.regex_broad_hit", kpi=kpi.name, value=value, unit=unit,
                chunk_type=chunk.chunk_type, page=chunk.page_number,
            )
            result = ExtractedKPI(
                kpi_name=kpi.name, raw_value=match.group(1),
                normalized_value=value, unit=unit,
                extraction_method="regex", confidence=confidence,
                source_chunk_id=chunk.id, validation_passed=True,
            )
            if best is None or result.confidence > best.confidence:
                best = result

    return best


# ---------------------------------------------------------------------------
# Layer 2 — LLM
# ---------------------------------------------------------------------------

def _build_chunks_text(
    scored_chunks: list[ScoredChunk], max_chars: int = _LLM_CHUNK_CHAR_LIMIT,
) -> str:
    primary   = [sc for sc in scored_chunks if not sc.is_neighbor]
    neighbors = [sc for sc in scored_chunks if sc.is_neighbor]
    ordered   = primary + neighbors

    parts: list[str] = []
    total = 0
    seen_indices: set = set()

    for sc in ordered:
        idx = sc.chunk.chunk_index
        if idx in seen_indices:
            continue
        seen_indices.add(idx)
        prefix = (
            f"[TABLE | Page {sc.chunk.page_number}]\n"
            if sc.chunk.chunk_type == "table"
            else f"[Page {sc.chunk.page_number}]\n"
        )
        block = f"{prefix}{sc.chunk.content}\n\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)

    return "".join(parts).strip()


def _try_llm(
    scored_chunks: list[ScoredChunk],
    kpi: KPIDefinition,
    llm: LLMService,
    report_year: int,
) -> Optional[ExtractedKPI]:
    if not scored_chunks:
        return None

    aliases     = _KPI_ALIASES.get(kpi.name, [])
    alias_str   = ", ".join(aliases) if aliases else "none"
    chunks_text = _build_chunks_text(scored_chunks)

    result = llm.extract_kpi(
        kpi_name=kpi.name, kpi_display=kpi.display_name,
        expected_unit=kpi.expected_unit, chunks_text=chunks_text,
        aliases=alias_str, report_year=report_year,
    )
    if result is None:
        return None

    raw_value = result.get("value")
    if raw_value is None:
        logger.info("extraction.llm_not_found", kpi=kpi.name, notes=result.get("notes"))
        return None

    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        logger.warning("extraction.llm_bad_value", kpi=kpi.name, raw=raw_value)
        return None

    unit            = _normalise_unit(str(result.get("unit") or kpi.expected_unit))
    confidence      = float(result.get("confidence") or 0.5)
    year            = result.get("year") or report_year
    source_chunk_id = scored_chunks[0].chunk.id if scored_chunks else None

    logger.info(
        "extraction.llm_hit", kpi=kpi.name, value=value, unit=unit, confidence=confidence,
    )

    return ExtractedKPI(
        kpi_name=kpi.name, raw_value=str(raw_value), normalized_value=value,
        unit=unit, year=year, extraction_method="llm", confidence=confidence,
        source_chunk_id=source_chunk_id, validation_passed=True,
        validation_notes=result.get("notes"),
    )


# ---------------------------------------------------------------------------
# Layer 3 — Validation
# ---------------------------------------------------------------------------

def _validate(extracted: ExtractedKPI, kpi: KPIDefinition) -> ExtractedKPI:
    notes: list[str] = []
    val = extracted.normalized_value

    if val is not None:
        if kpi.valid_min is not None and val < kpi.valid_min:
            extracted.validation_passed = False
            notes.append(f"Value {val} below minimum {kpi.valid_min}")
        if kpi.valid_max is not None and val > kpi.valid_max:
            extracted.validation_passed = False
            notes.append(f"Value {val} exceeds maximum {kpi.valid_max}")

    if extracted.unit and kpi.expected_unit:
        if extracted.unit.lower() != kpi.expected_unit.lower():
            notes.append(
                f"Unit mismatch: got '{extracted.unit}', expected '{kpi.expected_unit}'"
            )

    if extracted.confidence is not None and extracted.confidence < 0.4:
        extracted.validation_passed = False
        notes.append(f"Confidence too low: {extracted.confidence:.2f}")

    if notes:
        existing = extracted.validation_notes or ""
        extracted.validation_notes = (
            existing + (" | " if existing else "") + " | ".join(notes)
        )
        if not extracted.validation_passed:
            logger.warning(
                "extraction.validation_failed", kpi=kpi.name, value=val, notes=notes,
            )

    return extracted


# ---------------------------------------------------------------------------
# Wider retrieval
# ---------------------------------------------------------------------------

def _get_wider_chunks(
    parsed_doc: ParsedDocument,
    kpi: KPIDefinition,
    db: Session,
    already_retrieved_ids: set,
    extra_k: int = 5,
) -> list[ScoredChunk]:
    aliases      = _KPI_ALIASES.get(kpi.name, [])
    all_keywords = list(kpi.retrieval_keywords or []) + aliases
    if not all_keywords:
        return []

    keyword_filters = [
        DocumentChunk.keywords.ilike(f"%{kw.lower().split()[0]}%")
        for kw in all_keywords
    ]

    extra_chunks = (
        db.query(DocumentChunk)
        .filter(
            DocumentChunk.parsed_document_id == parsed_doc.id,
            DocumentChunk.id.notin_(list(already_retrieved_ids)),
            or_(*keyword_filters),
        )
        .order_by(DocumentChunk.chunk_type.desc(), DocumentChunk.page_number)
        .limit(extra_k)
        .all()
    )

    return [ScoredChunk(chunk=c, score=0.3, matched_keywords=[]) for c in extra_chunks]


# ---------------------------------------------------------------------------
# Post-extraction derivation
# ---------------------------------------------------------------------------

def _derive_total_ghg(results: list["ExtractedKPI"]) -> list["ExtractedKPI"]:
    """Compute total_ghg = scope_1 + scope_2 when not directly found."""
    by_name: dict[str, ExtractedKPI] = {r.kpi_name: r for r in results}
    total = by_name.get("total_ghg_emissions")
    s1    = by_name.get("scope_1_emissions")
    s2    = by_name.get("scope_2_emissions")

    if not (
        total is not None and total.normalized_value is None
        and s1 is not None and s1.normalized_value is not None
        and s2 is not None and s2.normalized_value is not None
    ):
        return results

    computed = round(s1.normalized_value + s2.normalized_value, 2)
    conf     = round(min(s1.confidence or 0.5, s2.confidence or 0.5), 3)

    logger.info(
        "extraction.total_ghg_derived",
        scope_1=s1.normalized_value, scope_2=s2.normalized_value,
        total=computed, confidence=conf,
    )

    derived = ExtractedKPI(
        kpi_name="total_ghg_emissions", raw_value=str(computed),
        normalized_value=computed, unit="tCO2e", extraction_method="derived",
        confidence=conf, source_chunk_id=s1.source_chunk_id, validation_passed=True,
        validation_notes=(
            f"Derived: {s1.normalized_value} (scope_1) + "
            f"{s2.normalized_value} (scope_2) = {computed} tCO2e"
        ),
    )

    return [derived if r.kpi_name == "total_ghg_emissions" else r for r in results]


# ---------------------------------------------------------------------------
# Public Agent
# ---------------------------------------------------------------------------

class ExtractionAgent:

    def __init__(self) -> None:
        self.settings    = get_settings()
        self.retrieval   = HybridRetrievalService()
        self.llm         = LLMService()
        self.kpi_service = KPIService()

    def extract_all(
        self,
        report_id: uuid.UUID,
        db: Session,
        kpi_names: Optional[list[str]] = None,
        fallback_search: bool = True,
        max_fallback_reports: int = 3,
    ) -> list[ExtractedKPI]:
        report = db.query(Report).filter(Report.id == report_id).first()
        if not report:
            raise ValueError(f"Report {report_id} not found")

        parsed_doc = (
            db.query(ParsedDocument)
            .filter(ParsedDocument.report_id == report_id)
            .order_by(ParsedDocument.parsed_at.desc())
            .first()
        )
        if not parsed_doc:
            raise ValueError(
                f"No parse cache for report {report_id}. Run parse first."
            )

        kpis = (
            self.kpi_service.get_by_names(kpi_names, db)
            if kpi_names
            else self.kpi_service.get_all_active(db)
        )

        logger.info("extraction.start", report_id=str(report_id), kpis=len(kpis))

        results: list[ExtractedKPI] = []
        for kpi in kpis:
            extracted = self._extract_one(
                kpi=kpi, parsed_doc=parsed_doc, report=report, db=db,
            )
            results.append(extracted)

            if extracted.normalized_value is not None:
                self.kpi_service.store_record(
                    company_id=report.company_id,
                    report_id=report_id,
                    report_year=report.report_year,
                    kpi_definition_id=kpi.id,
                    extracted=extracted,
                    source_chunk_id=extracted.source_chunk_id,
                    db=db,
                )

        results = _derive_total_ghg(results)

        logger.info(
            "extraction.complete", report_id=str(report_id), total_kpis=len(kpis),
        )
        return results

    def _extract_one(
        self,
        kpi: KPIDefinition,
        parsed_doc: ParsedDocument,
        report: Report,
        db: Session,
    ) -> ExtractedKPI:
        logger.debug("extraction.kpi_start", kpi=kpi.name)

        scored_chunks = self.retrieval.retrieve(
            parsed_document_id=parsed_doc.id, kpi=kpi, db=db,
        )
        chunks = [sc.chunk for sc in scored_chunks]

        extracted = _try_regex(chunks, kpi)

        if extracted and extracted.confidence >= _REGEX_HIGH_CONFIDENCE:
            logger.info(
                "extraction.regex_confident",
                kpi=kpi.name, value=extracted.normalized_value,
            )
            return _validate(extracted, kpi)

        retrieved_ids = {sc.chunk.id for sc in scored_chunks}
        extra_chunks  = _get_wider_chunks(parsed_doc, kpi, db, retrieved_ids, extra_k=5)
        all_scored    = scored_chunks + extra_chunks

        if not extracted or extracted.confidence < _REGEX_HIGH_CONFIDENCE:
            llm_result = _try_llm(all_scored, kpi, self.llm, report.report_year)
            if llm_result:
                if extracted is None or llm_result.confidence > extracted.confidence:
                    extracted = llm_result

        if extracted is None or extracted.normalized_value is None:
            logger.info("extraction.not_found", kpi=kpi.name)
            return ExtractedKPI(
                kpi_name=kpi.name,
                extraction_method="regex",
                confidence=0.0,
                validation_passed=False,
                validation_notes="Not found by regex or LLM",
            )

        return _validate(extracted, kpi)