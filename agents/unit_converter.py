"""
agents/unit_converter.py
------------------------
Deterministic Unit Conversion Engine + LLM-assisted Unit Detection

Architecture
------------
1. XBRL unit_ref is read directly from parsed data — this handles most cases.
2. For metrics with explicit XBRL unit_refs, conversion is purely deterministic.
3. For free-text labels (no unit_ref, or unit_ref = "pure" / ""), the
   LLM is invoked ONLY to extract the unit string from the label text.
   The LLM returns a unit string; all arithmetic is done by this module.
4. The LLM NEVER performs numeric conversion.

Canonical target units per category
-------------------------------------
  Energy   → GJ
  Water    → KL  (kilolitre)
  GHG      → tCO2e
  Waste    → MT  (metric tonne)
  Revenue  → INR (Indian Rupees)

XBRL unit_ref registry (from real castrol.xml inspection)
----------------------------------------------------------
  Gigajoule       → GJ            (factor 1.0)
  GigajoulePerINR → GJ/INR        (intensity, no conversion needed)
  MtCO2e          → tCO2e         (factor 1.0 — XBRL uses "Mt" loosely = metric tonne here)
  MtCO2ePerINR    → tCO2e/INR     (intensity)
  Kiloliters      → KL            (factor 1.0)
  KilolitersPerINR→ KL/INR        (intensity)
  Tonne           → MT            (factor 1.0 — metric tonne = MT)
  TonnePerINR     → MT/INR        (intensity)
  Kg              → MT            (factor 0.001)
  INR             → INR           (factor 1.0)
  pure            → dimensionless (no conversion)

Free-text / label unit patterns (regex-based, no LLM needed in most cases)
---------------------------------------------------------------------------
  Energy  : kWh → GJ (×0.0036), MWh → GJ (×3.6), TJ → GJ (×1000)
  GHG     : kgCO2e → tCO2e (×0.001), MtCO2e/ktCO2e etc.
  Revenue : ₹ crore / crore / cr → INR (×1e7)
             ₹ lakh / lakh / lac → INR (×1e5)
  Waste   : kg → MT (×0.001)
  Water   : m3 → KL (×1.0), ML → KL (×1000), litre/liter → KL (×0.001)

Anomaly detection thresholds (post-conversion)
-----------------------------------------------
  Energy_GJ_perEmployee    > 50,000 → ANOMALY
  Water_KL_perEmployee     > 20,000 → ANOMALY
  GHG_tCO2e_perEmployee    > 10,000 → ANOMALY
  Waste_MT_perEmployee     >  5,000 → ANOMALY

LLM safety contract
-------------------
  • The LLM receives: tag_name, label_text, raw_value
  • The LLM returns: {"unit_detected": str, "confidence": float}
  • This module performs all arithmetic — the LLM sees no numbers to convert
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Optional

# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConversionRecord:
    """Full audit trail for one metric's unit conversion."""
    metric:          str
    raw_value:       float
    raw_unit:        str             # as-seen in XBRL / label
    canonical_unit:  str             # target unit
    factor:          float           # multiplier applied
    converted_value: float
    source:          str             # 'xbrl_unit_ref' | 'label_regex' | 'llm_detected' | 'assumed'
    llm_confidence:  Optional[float] = None
    anomaly:         bool            = False
    anomaly_reason:  Optional[str]   = None
    excluded:        bool            = False   # True → excluded from ranking

    def __str__(self) -> str:
        flag = " ⚠ ANOMALY" if self.anomaly else ""
        excl = " [EXCLUDED]" if self.excluded else ""
        conf = f" (LLM conf={self.llm_confidence:.2f})" if self.llm_confidence else ""
        return (
            f"  {self.metric:<40} "
            f"{self.raw_value} {self.raw_unit} → "
            f"{self.converted_value} {self.canonical_unit} "
            f"[×{self.factor}  via {self.source}{conf}]{flag}{excl}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# DETERMINISTIC CONVERSION TABLES
# ══════════════════════════════════════════════════════════════════════════════

# XBRL unitRef → (canonical_unit, factor_to_canonical)
XBRL_UNIT_TABLE: dict[str, tuple[str, float]] = {
    # Energy
    "Gigajoule":           ("GJ",     1.0),
    "GigajoulePerINR":     ("GJ/INR", 1.0),    # intensity — no conversion
    "Terajoule":           ("GJ",     1000.0),
    "Megajoule":           ("GJ",     0.001),
    "KilojoulePerTonne":   ("GJ/MT",  0.000001),

    # GHG  — note: SEBI XBRL uses "MtCO2e" but contextually means metric tonne (tCO2e)
    "MtCO2e":              ("tCO2e",  1.0),     # metric tonne CO2e → tCO2e (same)
    "MtCO2ePerINR":        ("tCO2e/INR", 1.0),
    "KtCO2e":              ("tCO2e",  1000.0),
    "KgCO2e":              ("tCO2e",  0.001),

    # Water
    "Kiloliters":          ("KL",     1.0),
    "Kilolitre":           ("KL",     1.0),
    "KilolitersPerINR":    ("KL/INR", 1.0),
    "CubicMeter":          ("KL",     1.0),     # 1 m³ = 1 KL
    "CubicMetre":          ("KL",     1.0),
    "MegaLitre":           ("KL",     1000.0),
    "Litre":               ("KL",     0.001),

    # Waste
    "Tonne":               ("MT",     1.0),     # metric tonne = MT
    "TonnePerINR":         ("MT/INR", 1.0),
    "Kg":                  ("MT",     0.001),
    "Kilogram":            ("MT",     0.001),
    "KiloTonne":           ("MT",     1000.0),

    # Revenue
    "INR":                 ("INR",    1.0),
    "INRCrore":            ("INR",    1e7),
    "INRLakh":             ("INR",    1e5),
    "INRThousand":         ("INR",    1e3),

    # Dimensionless
    "pure":                ("pure",   1.0),
    "":                    ("unknown",""),
}

# Free-text label → (canonical_unit, factor) — applied when unit_ref is missing
LABEL_PATTERNS: list[tuple[re.Pattern, str, float]] = [
    # Energy
    (re.compile(r'\bkwh\b',    re.I), "GJ",    0.0036),
    (re.compile(r'\bmwh\b',    re.I), "GJ",    3.6),
    (re.compile(r'\bgwh\b',    re.I), "GJ",    3600.0),
    (re.compile(r'\btj\b',     re.I), "GJ",    1000.0),
    (re.compile(r'\bgj\b',     re.I), "GJ",    1.0),
    (re.compile(r'\bmj\b',     re.I), "GJ",    0.001),
    # GHG
    (re.compile(r'\bkgco2e?\b',re.I), "tCO2e", 0.001),
    (re.compile(r'\bmtco2e?\b',re.I), "tCO2e", 1.0),
    (re.compile(r'\bktco2e?\b',re.I), "tCO2e", 1000.0),
    (re.compile(r'\btco2e?\b', re.I), "tCO2e", 1.0),
    # Water
    (re.compile(r'\bm[\u00b3³3]\b|cubic\s*met(?:re|er)',re.I), "KL", 1.0),
    (re.compile(r'\bml\b|\bmegalit',re.I), "KL", 1000.0),
    (re.compile(r'\bkl\b|\bkilolit',re.I), "KL", 1.0),
    (re.compile(r'\blit(?:re|er)\b', re.I), "KL", 0.001),
    # Waste
    (re.compile(r'\bkg\b|\bkilogram',re.I), "MT", 0.001),
    (re.compile(r'\bkt\b|\bkilotonne',re.I), "MT", 1000.0),
    (re.compile(r'\bmt\b|\bm\.t\b|\bmetric\s*tonne',re.I), "MT", 1.0),
    (re.compile(r'\btonne\b|\bton\b', re.I), "MT", 1.0),
    # Revenue
    (re.compile(r'(?:₹|rs\.?|inr)\s*crore|crore',re.I), "INR", 1e7),
    (re.compile(r'(?:₹|rs\.?|inr)\s*lakh|lakh|lac',re.I), "INR", 1e5),
    (re.compile(r'(?:₹|rs\.?|inr)\s*thousand',      re.I), "INR", 1e3),
    (re.compile(r'\binr\b',    re.I), "INR", 1.0),
]

# Metrics whose XBRL unit_ref is known from the SEBI taxonomy
# Maps metric_name → expected canonical_unit (for validation)
EXPECTED_CANONICAL: dict[str, str] = {
    "TotalEnergy_GJ":        "GJ",
    "RenewableEnergy_GJ":    "GJ",
    "NonRenewableEnergy_GJ": "GJ",
    "Scope1_tCO2e":          "tCO2e",
    "Scope2_tCO2e":          "tCO2e",
    "WaterWithdrawal_KL":    "KL",
    "WaterConsumption_KL":   "KL",
    "WasteGenerated_MT":     "MT",
    "WasteRecovered_MT":     "MT",
    "WasteDisposed_MT":      "MT",
    "PlasticWaste_MT":       "MT",
    "EWaste_MT":             "MT",
    "HazardousWaste_MT":     "MT",
    "Turnover_INR":          "INR",
    "NetWorth_INR":          "INR",
    "PaidUpCapital_INR":     "INR",
}

# Anomaly detection thresholds (applied to normalised intensity metrics)
ANOMALY_THRESHOLDS: dict[str, float] = {
    "Energy_GJ_perEmployee":   50_000.0,
    "Water_KL_perEmployee":    20_000.0,
    "GHG_tCO2e_perEmployee":   10_000.0,
    "Waste_MT_perEmployee":     5_000.0,
    "Energy_GJ_perRevCr":      10_000.0,
    "Water_KL_perRevCr":       20_000.0,
    "GHG_tCO2e_perRevCr":      1_000.0,
    "Waste_MT_perRevCr":       1_000.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# LLM UNIT DETECTOR (label text → unit string only)
# ══════════════════════════════════════════════════════════════════════════════

def _detect_unit_via_llm(
    tag: str,
    label: str,
    groq_client,
    model: str,
) -> tuple[str, float]:
    """
    Asks the LLM to identify the unit embedded in a label string.

    Contract
    --------
    • The LLM receives only the tag name and label — NO numeric values.
    • The LLM returns JSON: {"unit_detected": str, "confidence": float}
    • This function returns (unit_string, confidence).
    • If the LLM call fails or returns low-confidence, returns ("unknown", 0.0).

    Safety: the LLM cannot perform arithmetic — it only names the unit.
    """
    system = textwrap.dedent("""
        You are a unit-detection assistant for ESG XBRL data.
        Given a metric tag name and its label text, identify the physical unit
        embedded in the label.

        RULES:
        1. Return ONLY valid JSON: {"unit_detected": "<unit>", "confidence": <0-1>}
        2. unit_detected must be one of:
           kWh, MWh, GWh, GJ, TJ, MJ,
           tCO2e, ktCO2e, kgCO2e, MtCO2e,
           KL, m3, ML, litre,
           MT, Tonne, kg, KT,
           INR, INR_crore, INR_lakh,
           pure, unknown
        3. Never include any numeric value in unit_detected.
        4. If truly ambiguous, return {"unit_detected": "unknown", "confidence": 0.0}
    """).strip()

    user = f'Tag: "{tag}"\nLabel: "{label}"'
    try:
        resp = groq_client.chat.completions.create(
            model=model,
            max_tokens=60,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        text = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        text = re.sub(r"```(?:json)?|```", "", text).strip()
        data = json.loads(text)
        return (str(data.get("unit_detected", "unknown")),
                float(data.get("confidence", 0.0)))
    except Exception:
        return ("unknown", 0.0)


# ── Revenue metric helper set ────────────────────────────────────────────────
_REVENUE_METRICS: frozenset[str] = frozenset({
    "Turnover_INR", "NetWorth_INR", "PaidUpCapital_INR",
})

# Indian listed company revenue plausibility bounds (in INR)
# Any value below 1e8 (₹1 Cr) is suspicious for a SEBI-listed entity
_REV_FLOOR_INR:    float = 1e8   # 10 crore minimum for listing eligibility
_REV_ALREADY_OK:   float = 1e8   # if >= this, assume already in INR

# Ordered by probability for SEBI XBRL filers:
# Many companies (esp PSUs) report in ₹ crore or ₹ lakh in their MDA,
# then some filers carry this scale into XBRL inadvertently.
_REV_CANDIDATES: list[tuple[float, str]] = [
    (1e7,  "crore"),    # most common — revenue "1000" means ₹1000 Cr = ₹10,000 Cr INR
    (1e5,  "lakh"),
    (1e6,  "million"),
    (1e9,  "billion"),
    (1e3,  "thousand"),
]


def _is_revenue_metric(metric: str) -> bool:
    return metric in _REVENUE_METRICS


def _revenue_magnitude_heuristic(
    value: float,
) -> tuple[float, float, str] | None:
    """
    If value is suspiciously small for a revenue-in-INR metric, return
    (corrected_value, factor, unit_label) of the first candidate that
    brings the value into a plausible range.

    Returns None if value is already plausible (>= _REV_ALREADY_OK).

    NOTE: This is called ONLY when the XBRL unit_ref already returned "INR"
    with factor=1.0 — meaning we accepted the value as INR but it may still
    be wrong in magnitude. The heuristic adds a second guard.
    """
    if value >= _REV_ALREADY_OK:
        return None   # value is already plausible as INR

    for factor, unit_label in _REV_CANDIDATES:
        corrected = round(value * factor, 2)
        if corrected >= _REV_FLOOR_INR:
            return (corrected, factor, unit_label)

    return None  # couldn't rescue — return None, caller handles gracefully


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONVERTER
# ══════════════════════════════════════════════════════════════════════════════

class UnitConverter:
    """
    Resolves the unit for each (metric, value, unit_ref, label) tuple and
    returns the value converted to the canonical unit.

    Call order:
      1. XBRL unit_ref → deterministic lookup in XBRL_UNIT_TABLE
      2. Label regex   → deterministic pattern matching on label text
      3. LLM detection → only if both above fail; LLM returns unit name,
                         this class performs the arithmetic
      4. Assumed       → use expected canonical with no conversion (factor=1)

    All paths return a ConversionRecord with a full audit trail.
    """

    def __init__(self, use_llm: bool = True, groq_client=None, model: str = ""):
        self.use_llm      = use_llm
        self.groq_client  = groq_client
        self.model        = model
        self._llm_calls   = 0
        self._llm_cache:  dict[str, tuple[str, float]] = {}

    # ── public: convert one metric value ─────────────────────────────────────

    def convert(
        self,
        metric:   str,
        value:    float,
        unit_ref: str,
        label:    str = "",
    ) -> ConversionRecord:
        """
        Returns a ConversionRecord. converted_value is in the canonical unit.
        If conversion is impossible, factor=1.0 and source='assumed'.
        """
        # ── Step 1: XBRL unit_ref lookup ─────────────────────────────────────
        if unit_ref and unit_ref != "unknown":
            entry = XBRL_UNIT_TABLE.get(unit_ref)
            if entry:
                canonical, factor = entry
                if canonical == "unknown":
                    pass   # fall through to label/LLM
                elif _is_revenue_metric(metric) and factor == 1.0 and canonical == "INR":
                    # XBRL says "INR" — value may still be in crores/lakhs.
                    # Check heuristic; if value is already plausible, return as-is.
                    heuristic = _revenue_magnitude_heuristic(value)
                    if heuristic is None:
                        # Value is already plausible — accept the XBRL unit
                        return ConversionRecord(
                            metric=metric, raw_value=value, raw_unit=unit_ref,
                            canonical_unit="INR", factor=1.0,
                            converted_value=value,
                            source="xbrl_unit_ref",
                        )
                    # Otherwise fall through to Step 1b to apply the heuristic
                else:
                    return ConversionRecord(
                        metric=metric, raw_value=value, raw_unit=unit_ref,
                        canonical_unit=canonical, factor=factor,
                        converted_value=round(value * factor, 8) if isinstance(factor, float) else value,
                        source="xbrl_unit_ref",
                    )

        # ── Step 1b: Revenue magnitude heuristic ─────────────────────────────
        # Applied AFTER XBRL lookup so it only triggers when:
        #   (a) unit_ref says "INR" (factor=1) — already returned above, OR
        #   (b) unit_ref is empty/unknown and label also gives no unit.
        # The heuristic checks whether a revenue metric value is suspiciously
        # small (< 1e8 INR ≈ ₹1 crore), indicating it was filed in crores/lakhs.
        # This is the root cause of the GAIL-type intensity explosion.
        if _is_revenue_metric(metric) and isinstance(value, (int, float)) and value > 0:
            heuristic = _revenue_magnitude_heuristic(value)
            if heuristic:
                corrected_val, factor_used, unit_label = heuristic
                return ConversionRecord(
                    metric=metric, raw_value=value,
                    raw_unit=f"heuristic:{unit_label}",
                    canonical_unit="INR",
                    factor=factor_used,
                    converted_value=corrected_val,
                    source="revenue_heuristic",
                )

        # ── Step 2: label regex ───────────────────────────────────────────────
        if label:
            for pattern, canonical, factor in LABEL_PATTERNS:
                if pattern.search(label):
                    return ConversionRecord(
                        metric=metric, raw_value=value, raw_unit=f"label:{label[:40]}",
                        canonical_unit=canonical, factor=factor,
                        converted_value=round(value * factor, 8),
                        source="label_regex",
                    )

        # ── Step 3: LLM unit detection ────────────────────────────────────────
        if self.use_llm and self.groq_client and label:
            cache_key = f"{metric}:{label}"
            if cache_key not in self._llm_cache:
                self._llm_calls += 1
                unit_str, conf = _detect_unit_via_llm(
                    metric, label, self.groq_client, self.model)
                self._llm_cache[cache_key] = (unit_str, conf)
            unit_str, conf = self._llm_cache[cache_key]

            if unit_str != "unknown" and conf >= 0.60:
                # Map LLM-returned unit string back through label patterns
                for pattern, canonical, factor in LABEL_PATTERNS:
                    if pattern.search(unit_str):
                        return ConversionRecord(
                            metric=metric, raw_value=value,
                            raw_unit=f"llm:{unit_str}",
                            canonical_unit=canonical, factor=factor,
                            converted_value=round(value * factor, 8),
                            source="llm_detected",
                            llm_confidence=conf,
                        )
                # LLM found something but it didn't match patterns — try XBRL table
                xbrl_key = unit_str.replace("_", "").replace(" ", "")
                entry = XBRL_UNIT_TABLE.get(xbrl_key)
                if entry:
                    canonical, factor = entry
                    return ConversionRecord(
                        metric=metric, raw_value=value,
                        raw_unit=f"llm:{unit_str}",
                        canonical_unit=canonical, factor=factor,
                        converted_value=round(value * factor, 8),
                        source="llm_detected",
                        llm_confidence=conf,
                    )

        # ── Step 4: fallback — assume canonical, no conversion ────────────────
        canonical = EXPECTED_CANONICAL.get(metric, "unknown")
        # For revenue metrics that reach here, flag clearly so audit can catch them
        src = "assumed_revenue_INR" if _is_revenue_metric(metric) else "assumed"
        return ConversionRecord(
            metric=metric, raw_value=value,
            raw_unit=unit_ref or "unknown",
            canonical_unit=canonical, factor=1.0,
            converted_value=value,
            source=src,
        )

    # ── public: convert a whole metrics dict with unit_ref map ────────────────

    def convert_all(
        self,
        metrics:      dict[str, float],
        unit_ref_map: dict[str, str],
        label_map:    dict[str, str] | None = None,
    ) -> tuple[dict[str, float], list[ConversionRecord]]:
        """
        Parameters
        ----------
        metrics      : {metric_name: raw_value}
        unit_ref_map : {metric_name: xbrl_unit_ref_string}
        label_map    : {metric_name: human-readable label from XBRL}

        Returns
        -------
        converted_metrics : {metric_name: value_in_canonical_unit}
        records           : list of ConversionRecord (audit trail)
        """
        label_map    = label_map or {}
        converted    = {}
        records      = []

        for metric, raw_val in metrics.items():
            if not isinstance(raw_val, (int, float)):
                converted[metric] = raw_val
                continue

            unit_ref = unit_ref_map.get(metric, "")
            label    = label_map.get(metric, "")

            rec = self.convert(metric, float(raw_val), unit_ref, label)
            records.append(rec)
            converted[metric] = rec.converted_value

        return converted, records

    # ── anomaly detection (called after normalization) ─────────────────────────

    @staticmethod
    def check_anomalies(
        metrics: dict[str, float],
    ) -> dict[str, dict]:
        """
        Checks intensity metrics against ANOMALY_THRESHOLDS.

        Returns {metric: {"value": v, "threshold": t, "reason": str}}
        for any anomalous metric. Caller should:
          • Mark metric as "needs verification"
          • Set metric to None (exclude from ranking) if extreme
        """
        anomalies = {}
        for metric, threshold in ANOMALY_THRESHOLDS.items():
            val = metrics.get(metric)
            if val is None or not isinstance(val, (int, float)):
                continue
            if val > threshold:
                anomalies[metric] = {
                    "value":     val,
                    "threshold": threshold,
                    "reason":    (
                        f"{metric} = {val:.2f} exceeds maximum plausible "
                        f"value of {threshold:,.0f} — likely unit error or "
                        f"data entry mistake; excluded from ranking"
                    ),
                }
        return anomalies

    @property
    def llm_call_count(self) -> int:
        return self._llm_calls