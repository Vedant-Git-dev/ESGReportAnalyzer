"""
services/llm_service.py

Abstracted LLM client — works with any OpenAI-compatible endpoint.
Enhanced prompt engineering for natural-language ESG PDFs.

Key fixes vs previous version:
  - Prompt explicitly forbids returning "X Million tonnes" — must return
    the exact number in tCO2e (e.g. 5,000,000 not 5 Million).
  - Prompt enforces canonical unit expectations per KPI.
  - JSON sanitisation handles Python repr (type=NoneType) Gemini quirk.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from core.config import get_settings
from core.logging_config import get_logger

logger = get_logger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _strip_json_fence(text: str) -> str:
    match = _JSON_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _sanitise_json_value(text: str) -> str:
    """
    Fix common LLM JSON output issues before parsing:
      1. Python repr null: type=NoneType  →  null
      2. Trailing commas before closing brace/bracket
      3. Unquoted None/True/False
    """
    # Replace Python None/True/False with JSON equivalents
    text = re.sub(r'\bNone\b', 'null', text)
    text = re.sub(r'\bTrue\b', 'true', text)
    text = re.sub(r'\bFalse\b', 'false', text)
    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return text


def _parse_json_response(text: str) -> Optional[dict]:
    cleaned = _strip_json_fence(text)
    cleaned = _sanitise_json_value(cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        obj_match = re.search(r"\{[\s\S]*\}", cleaned)
        if obj_match:
            try:
                return json.loads(_sanitise_json_value(obj_match.group()))
            except json.JSONDecodeError:
                pass
    logger.warning("llm_service.json_parse_failed", raw=text[:200])
    return None


class LLMService:

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=self.settings.llm_base_url,
                    api_key=self.settings.llm_api_key or "no-key",
                )
            except ImportError:
                raise RuntimeError("openai package not installed. Run: pip install openai")
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=False,
    )
    def _call(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.settings.llm_max_tokens,
                temperature=self.settings.llm_temperature,
            )
            return response.choices[0].message.content
        except Exception as exc:
            logger.warning("llm_service.call_failed", error=str(exc))
            raise

    def extract_kpi(
        self,
        kpi_name: str,
        kpi_display: str,
        expected_unit: str,
        chunks_text: str,
        aliases: str = "",
        report_year: int = 0,
    ) -> Optional[dict]:
        """
        Extract a single KPI value from pre-filtered chunk text.
        Returns strict JSON with: {value, unit, reasoning, confidence}

        Designed to handle:
        - Values in tables
        - Values in narrative text
        - Values next line from keyword
        - Multiple occurrences (prefer most recent year)
        """
        if not self.settings.llm_api_key:
            logger.warning("llm_service.no_api_key", kpi=kpi_name)
            return None

        # Build unit-specific guidance to prevent LLM from using wrong scale
        unit_guidance = _build_unit_guidance(kpi_name, expected_unit)

        system_prompt = (
            "You are an expert ESG data analyst. "
            "Your task is to find specific numeric metrics from sustainability report excerpts. "
            "The text may be from tables, narrative paragraphs, or data summaries. "
            "\n\nCRITICAL extraction rules:"
            "\n1. Search ALL text carefully — values may be in tables with separators or in narrative sentences"
            "\n2. Accept any phrasing: 'scope 1 was X', 'X tCO2e of direct emissions', 'emitted X tonnes', etc."
            "\n3. Do NOT invent values — only extract what is explicitly stated"
            "\n4. If multiple values appear, prefer the most recent year or highest reported absolute value"
            "\n5. ALWAYS return the raw number exactly as it appears — do NOT convert or rescale"
            "\n6. NEVER return values in 'Million tonnes' — convert: 5 Million tonnes = 5000000"
            "\n7. Return the unit EXACTLY as it appears in the source text"
            "\n8. Respond ONLY with valid JSON — no explanation, no markdown fences, no Python None"
        )

        alias_section = f"\nAlso known as: {aliases}" if aliases else ""
        year_hint = f"\nReport fiscal year: {report_year}" if report_year else ""

        user_prompt = f"""Find the numeric value for this ESG metric:

Metric: {kpi_display}
Expected unit: {expected_unit}
{unit_guidance}{alias_section}{year_hint}

Text to search:
{chunks_text}

Return ONLY this JSON (single line, no fences, no Python None — use null):
{{"value": <number or null>, "unit": "<unit found or null>", "reasoning": "<1 sentence>", "confidence": <0.0-1.0>}}

Examples of CORRECT responses:
{{"value": 21949, "unit": "tCO2e", "reasoning": "Scope 1 Emissions: 21,949 tCO2e in table", "confidence": 0.92}}
{{"value": 62352, "unit": "tCO2e", "reasoning": "Total Scope 2 emissions 62,352 tCO2e", "confidence": 0.91}}
{{"value": null, "unit": null, "reasoning": "Value not found in text", "confidence": 0.0}}

Examples of WRONG responses (DO NOT return these):
{{"value": 5, "unit": "Million tonnes CO2e"}}  <- WRONG: use 5000000, unit tCO2e
{{"value": None, ...}}  <- WRONG: use null not None"""

        logger.info("llm_service.extract_kpi_start", kpi=kpi_name, chunk_chars=len(chunks_text))

        raw = self._call(system_prompt, user_prompt)
        if raw is None:
            return None

        result = _parse_json_response(raw)
        if result is None:
            logger.warning("llm_service.json_parse_failed", kpi=kpi_name, raw=raw[:200])
            return None

        # Validate required fields
        if "value" not in result or "confidence" not in result:
            logger.warning("llm_service.missing_fields", kpi=kpi_name, result=result)
            return None

        # Post-process: if LLM returned a "million scale" unit, convert value
        result = _auto_convert_million_scale(result, kpi_name)

        logger.info(
            "llm_service.extracted",
            kpi=kpi_name,
            value=result.get("value"),
            unit=result.get("unit"),
            confidence=result.get("confidence"),
            reasoning=result.get("reasoning", "")[:80],
        )
        return result


def _build_unit_guidance(kpi_name: str, expected_unit: str) -> str:
    """
    Return KPI-specific unit extraction guidance to inject into the prompt.
    Prevents the LLM from returning rounded million-scale values.
    """
    if kpi_name in ("scope_1_emissions", "scope_2_emissions", "total_ghg_emissions"):
        return (
            "\nUNIT RULE: Return value in tCO2e (metric tonnes of CO2 equivalent). "
            "If the PDF shows '21,949' in a tCO2e column, return value=21949. "
            "If the PDF shows '5 Million tonnes CO2e', return value=5000000 unit='tCO2e'. "
            "NEVER return value=5 with unit='Million tonnes CO2e'.\n"
        )
    elif kpi_name == "energy_consumption":
        return (
            "\nUNIT RULE: Return value in the unit shown in the PDF (GJ, MJ, MWh, GWh, TJ). "
            "Do not convert between units — return exact number and exact unit label from PDF.\n"
        )
    elif kpi_name == "waste_generated":
        return (
            "\nUNIT RULE: Return value in metric tonnes (MT). "
            "Typical IT company total waste is 1,000 – 50,000 MT per year. "
            "If you see a value > 500,000 MT, it is likely a different company's data — skip it.\n"
        )
    elif kpi_name == "water_consumption":
        return (
            "\nUNIT RULE: Return value in KL (kilolitres) or the unit shown in the PDF.\n"
        )
    return f"\nUNIT RULE: Return value in {expected_unit} as shown in the PDF.\n"


def _auto_convert_million_scale(result: dict, kpi_name: str) -> dict:
    """
    If the LLM returns a value in 'million tonnes CO2e' or similar despite
    the prompt, convert it to tCO2e automatically.

    Example: value=5, unit="Million tonnes CO2e"  →  value=5000000, unit="tCO2e"
    """
    if result.get("value") is None:
        return result

    unit = str(result.get("unit") or "").lower().strip()

    # Detect million-scale GHG units
    ghg_kpis = {"scope_1_emissions", "scope_2_emissions", "total_ghg_emissions"}
    if kpi_name in ghg_kpis and (
        "million" in unit or
        "mtco2" in unit.replace(" ", "") or
        unit in ("mt co2e", "mt co2", "mtco2e", "mtco2")
    ):
        try:
            original_val = float(result["value"])
            # Only scale if value looks like it's in millions (< 10000 would be absurd for tCO2e)
            # A real IT company scope 1 is 5,000–200,000 tCO2e
            # If LLM returned "5" with unit "million tonnes CO2e", that means 5,000,000
            if original_val < 10_000:
                result["value"] = original_val * 1_000_000
                result["unit"] = "tCO2e"
                logger.info(
                    "llm_service.auto_converted_million_scale",
                    kpi=kpi_name,
                    original_value=original_val,
                    original_unit=unit,
                    converted_value=result["value"],
                )
        except (TypeError, ValueError):
            pass

    return result