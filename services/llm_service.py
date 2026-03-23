"""
services/llm_service.py

Abstracted LLM client — works with any OpenAI-compatible endpoint:
  Groq:       LLM_BASE_URL=https://api.groq.com/openai/v1
  OpenRouter: LLM_BASE_URL=https://openrouter.ai/api/v1
  OpenAI:     LLM_BASE_URL=https://api.openai.com/v1

Rules (strictly enforced):
  - NEVER send more than top_k chunks (enforced by caller)
  - Always request JSON-only output
  - Retry up to 3 times on transient failures
  - Return None on failure — never raise (extraction agent handles fallback)
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
    """Remove markdown code fences if present."""
    match = _JSON_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _parse_json_response(text: str) -> Optional[dict]:
    """Attempt to parse JSON from LLM response, stripping fences."""
    cleaned = _strip_json_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object anywhere in the response
        obj_match = re.search(r"\{[\s\S]*\}", cleaned)
        if obj_match:
            try:
                return json.loads(obj_match.group())
            except json.JSONDecodeError:
                pass
    logger.warning("llm_service.json_parse_failed", raw=text[:200])
    return None


class LLMService:
    """
    Thin wrapper around OpenAI-compatible /v1/chat/completions.
    Instantiate once per pipeline run.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = None

    def _get_client(self):
        """Lazy-init OpenAI client."""
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
        """Single chat completion call. Returns raw text or None."""
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
    ) -> Optional[dict]:
        """
        Ask the LLM to extract a single KPI value from pre-filtered chunk text.

        Args:
            kpi_name:      Internal name e.g. "scope_1_emissions"
            kpi_display:   Human name e.g. "Scope 1 GHG Emissions"
            expected_unit: e.g. "tCO2e"
            chunks_text:   Concatenated top-K chunks (NEVER full document)

        Returns:
            Dict with keys: value (float|null), unit (str), year (int|null),
                            confidence (float 0-1), notes (str)
            None if LLM call fails entirely.
        """
        if not self.settings.llm_api_key:
            logger.warning("llm_service.no_api_key", kpi=kpi_name)
            return None

        system_prompt = (
            "You are a precise ESG data extraction assistant. "
            "Extract numeric KPI values from report text. "
            "Respond ONLY with valid JSON — no explanation, no markdown."
        )

        user_prompt = f"""Extract the following KPI from the report text below.

KPI: {kpi_display}
Expected unit: {expected_unit}

Report text:
{chunks_text}

Return ONLY this JSON object:
{{
  "value": <number or null if not found>,
  "unit": "<unit string or null>",
  "year": <fiscal year as integer or null>,
  "confidence": <0.0 to 1.0>,
  "notes": "<brief note on where/how value was found, or why not found>"
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