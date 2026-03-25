"""
services/llm_service.py

Abstracted LLM client — works with any OpenAI-compatible endpoint.
Enhanced prompt engineering for natural-language ESG PDFs.
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


def _parse_json_response(text: str) -> Optional[dict]:
    cleaned = _strip_json_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        obj_match = re.search(r"\{[\s\S]*\}", cleaned)
        if obj_match:
            try:
                return json.loads(obj_match.group())
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
        Uses a rich, alias-aware prompt to handle naturally-worded PDFs.
        """
        if not self.settings.llm_api_key:
            logger.warning("llm_service.no_api_key", kpi=kpi_name)
            return None

        system_prompt = (
            "You are an expert ESG data analyst. "
            "Your task is to find specific numeric metrics from sustainability report excerpts. "
            "The text may be from tables, narrative paragraphs, or data summaries. "
            "Respond ONLY with valid JSON — no explanation, no markdown fences."
        )

        alias_section = f"\nAlso known as / may appear as: {aliases}" if aliases else ""
        year_hint = f"\nReport fiscal year: {report_year}. Prefer values for this year if multiple years appear." if report_year else ""

        user_prompt = f"""Find the value for this ESG metric in the report text below.

Metric: {kpi_display}
Expected unit: {expected_unit}{alias_section}{year_hint}

Extraction rules:
1. Search ALL text carefully — the value may appear in narrative sentences, not just tables
2. Accept any phrasing: "our scope 1 was X", "X tCO2e of direct emissions", "emitted X tonnes", etc.
3. If the unit differs from expected (e.g. MtCO2e vs tCO2e), still extract the value and report the actual unit
4. If you find multiple values for different years, return the most recent one
5. If the value is expressed as a range (e.g. "between X and Y"), return the midpoint
6. Do NOT invent or estimate values — only extract what is explicitly stated

Report text:
{chunks_text}

Return ONLY this JSON (no other text):
{{
  "value": <number or null if truly not found>,
  "unit": "<exact unit as written in text, or null>",
  "year": <fiscal year as 4-digit integer or null>,
  "confidence": <0.0 to 1.0 — how certain you are this is the right value>,
  "notes": "<one sentence: where exactly you found this, or why not found>"
}}"""

        logger.info("llm_service.extract_kpi", kpi=kpi_name, chunk_chars=len(chunks_text))

        raw = self._call(system_prompt, user_prompt)
        if raw is None:
            return None

        result = _parse_json_response(raw)
        if result is None:
            logger.warning("llm_service.bad_json", kpi=kpi_name, raw=raw[:300])
            return None

        logger.info(
            "llm_service.extracted",
            kpi=kpi_name,
            value=result.get("value"),
            unit=result.get("unit"),
            confidence=result.get("confidence"),
        )
        return result