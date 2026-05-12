"""
services/revenue_llm_service.py

Dedicated gemini-2.5-flash service for revenue search using google-genai SDK.
Uses Google Search grounding for accurate financial data extraction.

Pattern: from google import genai; client = genai.Client() (reads GEMINI_API_KEY from env)
Model: gemini-2.5-flash
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from core.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class RevenueSearchResult:
    """Structured result from revenue web search."""
    value_cr: float
    raw_value: str
    source_url: str
    source_domain: str
    confidence: float
    year: int
    company_name: str
    is_consolidated: bool = True


def _parse_json_response(text: str) -> Optional[dict]:
    """Parse JSON from LLM response."""
    # Try to find JSON block
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.I)
    if match:
        text = match.group(1)

    # Try direct JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to extract JSON-like content
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def search_revenue(
    company_name: str,
    fiscal_year: int,
    kpi: str = "revenue_from_operations",
    max_retries: int = 5,
    retry_delay: float = 15.0,
    model: str = "gemini-2.5-flash",  # Gemini-2.5-flash with Google Search
) -> Optional[RevenueSearchResult]:
    """
    Search for revenue using gemini-2.5-flash with web grounding.

    Pattern:
        from google import genai
        client = genai.Client()  # reads GEMINI_API_KEY from environment automatically
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="...",
            config=types.GenerateContentConfig(tools=[{"google_search":{}}])
        )

    Args:
        company_name: Company name (e.g. "TCS", "Infosys")
        fiscal_year: Target fiscal year (e.g. 2025 for FY25)
        kpi: Which KPI to search — "revenue_from_operations" or "net_revenue"
        max_retries: Number of retry attempts for API errors
        retry_delay: Delay between retries in seconds

    Returns:
        RevenueSearchResult or None
    """
    if kpi == "net_revenue":
        return _search_net_revenue(company_name, fiscal_year, max_retries, retry_delay, model)

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.warning("revenue_llm.google_genai_not_installed")
        return None

    last_error = None

    for attempt in range(max_retries):
        try:
            client = genai.Client()

            prompt = f"""Find the annual CONSOLIDATED "Revenue from Operations" for {company_name} for FY{fiscal_year} in INR Crore.

Return ONLY valid JSON (no markdown, no explanation):
{{"revenue_cr": <number in crore INR>, "is_consolidated": true/false, "confidence": 0.0-1.0, "reasoning": "<brief explanation>", "source_url": "<URL of source>"}}.

Rules:
1. ONLY extract "Revenue from Operations" (NOT Total Income, Net Sales, Other Income)
2. Prefer CONSOLIDATED figures over standalone
3. Indian companies report in Crore (1 Crore = 10 million)
4. FY{fiscal_year} = fiscal year ending March {fiscal_year}
5. Only use reliable sources (company filings, reputable news, financial databases). Avoid forums, social media, or unverified sites.

If you cannot find a reliable figure, return:
{{"revenue_cr": null, "is_consolidated": false, "confidence": 0.0, "reasoning": "Not found"}}"""

            logger.info("revenue_llm.search_start", company=company_name, year=fiscal_year, attempt=attempt, kpi=kpi)

            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}],
                    temperature=0.0,
                ),
            )

            raw_text = response.text or ""
            logger.debug("revenue_llm.raw_response", text=raw_text[:300])

            # Extract source URL from grounding metadata
            source_url = ""
            source_domain = ""
            if hasattr(response, 'candidates') and response.candidates:
                cand = response.candidates[0]
                if hasattr(cand, 'grounding_metadata') and cand.grounding_metadata:
                    chunks = getattr(cand.grounding_metadata, 'grounding_chunks', [])
                    if chunks:
                        first = chunks[0].web if hasattr(chunks[0], 'web') else None
                        if first and hasattr(first, 'uri'):
                            source_url = first.uri
                            source_domain = first.uri.split("/")[2] if "/" in first.uri else ""
                    logger.debug("revenue_llm.sources", url=source_url)

            if not raw_text.strip():
                logger.warning("revenue_llm.empty_response", company=company_name, year=fiscal_year)
                return None

            result = _parse_json_response(raw_text)

            if not result:
                logger.warning("revenue_llm.parse_failed", raw=raw_text[:200])
                return None

            revenue_cr = result.get("revenue_cr")

            # Validate
            if revenue_cr is None or revenue_cr <= 0:
                logger.info("revenue_llm.no_revenue_found", company=company_name, year=fiscal_year)
                return None

            logger.info(
                "revenue_llm.search_success",
                company=company_name,
                year=fiscal_year,
                revenue_cr=revenue_cr,
                source_url=source_url,
                kpi=kpi,
            )

            search_result = RevenueSearchResult(
                value_cr=revenue_cr,
                raw_value=str(revenue_cr),
                source_url=source_url,
                source_domain=source_domain,
                confidence=result.get("confidence", 0.90),
                year=fiscal_year,
                company_name=company_name,
                is_consolidated=result.get("is_consolidated", True),
            )

            _store_in_cache(search_result, kpi=kpi)
            return search_result

        except Exception as exc:
            last_error = exc
            error_str = str(exc)
            logger.warning("revenue_llm.search_error", company=company_name, year=fiscal_year, error=error_str, attempt=attempt)

            # Retry on 5xx errors or rate limiting
            if "500" in error_str or "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < max_retries - 1:
                    delay = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info("revenue_llm.retrying", delay=delay, attempt=attempt + 1)
                    time.sleep(delay)
                    continue

    logger.error("revenue_llm.all_retries_failed", company=company_name, year=fiscal_year, error=str(last_error))
    return None


def _search_net_revenue(
    company_name: str,
    fiscal_year: int,
    max_retries: int = 5,
    retry_delay: float = 15.0,
    model: str = "gemini-2.5-flash",
) -> Optional[RevenueSearchResult]:
    """Search for Net Revenue / Total Income using gemini-2.5-flash with web grounding."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.warning("revenue_llm.google_genai_not_installed")
        return None

    last_error = None

    for attempt in range(max_retries):
        try:
            client = genai.Client()

            prompt = f"""Find the annual CONSOLIDATED "Total Income" (also called "Net Revenue", "Revenue from Operations plus Other Income", or "Total Revenue") for {company_name} for FY{fiscal_year} in INR Crore.

Return ONLY valid JSON (no markdown, no explanation):
{{"revenue_cr": <number in crore INR>, "is_consolidated": true/false, "confidence": 0.0-1.0, "reasoning": "<brief explanation>", "source_url": "<URL of source>"}}.

Rules:
1. Extract "Total Income" or "Net Revenue" — this is Revenue from Operations PLUS Other Income (finance income, other income, etc.)
2. This is typically HIGHER than Revenue from Operations alone
3. Prefer CONSOLIDATED figures over standalone
4. Indian companies report in Crore (1 Crore = 10 million)
5. FY{fiscal_year} = fiscal year ending March {fiscal_year}
6. Only use reliable sources (company filings, reputable news, financial databases). Avoid forums, social media, or unverified sites.

If you cannot find a reliable figure, return:
{{"revenue_cr": null, "is_consolidated": false, "confidence": 0.0, "reasoning": "Not found"}}"""

            logger.info("revenue_llm.search_start.net_revenue", company=company_name, year=fiscal_year, attempt=attempt)

            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}],
                    temperature=0.0,
                ),
            )

            raw_text = response.text or ""
            logger.debug("revenue_llm.raw_response", text=raw_text[:300])

            # Extract source URL from grounding metadata
            source_url = ""
            source_domain = ""
            if hasattr(response, 'candidates') and response.candidates:
                cand = response.candidates[0]
                if hasattr(cand, 'grounding_metadata') and cand.grounding_metadata:
                    chunks = getattr(cand.grounding_metadata, 'grounding_chunks', [])
                    if chunks:
                        first = chunks[0].web if hasattr(chunks[0], 'web') else None
                        if first and hasattr(first, 'uri'):
                            source_url = first.uri
                            source_domain = first.uri.split("/")[2] if "/" in first.uri else ""
                    logger.debug("revenue_llm.sources", url=source_url)

            if not raw_text.strip():
                logger.warning("revenue_llm.empty_response", company=company_name, year=fiscal_year)
                return None

            result = _parse_json_response(raw_text)

            if not result:
                logger.warning("revenue_llm.parse_failed", raw=raw_text[:200])
                return None

            revenue_cr = result.get("revenue_cr")

            # Validate — net revenue should be >= revenue from operations
            if revenue_cr is None or revenue_cr <= 0:
                logger.info("revenue_llm.no_revenue_found", company=company_name, year=fiscal_year)
                return None

            logger.info(
                "revenue_llm.search_success.net_revenue",
                company=company_name,
                year=fiscal_year,
                revenue_cr=revenue_cr,
                source_url=source_url,
            )

            search_result = RevenueSearchResult(
                value_cr=revenue_cr,
                raw_value=str(revenue_cr),
                source_url=source_url,
                source_domain=source_domain,
                confidence=result.get("confidence", 0.90),
                year=fiscal_year,
                company_name=company_name,
                is_consolidated=result.get("is_consolidated", True),
            )

            _store_in_cache(search_result, kpi="net_revenue")
            return search_result

        except Exception as exc:
            last_error = exc
            error_str = str(exc)
            logger.warning("revenue_llm.search_error", company=company_name, year=fiscal_year, error=error_str, attempt=attempt)

            # Retry on 5xx errors or rate limiting
            if "500" in error_str or "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < max_retries - 1:
                    delay = retry_delay * (2 ** attempt)
                    logger.info("revenue_llm.retrying", delay=delay, attempt=attempt + 1)
                    time.sleep(delay)
                    continue

    logger.error("revenue_llm.all_retries_failed.net_revenue", company=company_name, year=fiscal_year, error=str(last_error))
    return None


def _store_in_cache(result: RevenueSearchResult, model_id=None, kpi: str = "revenue_from_operations") -> None:
    """Write a successful search result to the revenue_search_cache table."""
    try:
        from core.database import get_db
        from models.db_models import RevenueSearchCache

        with get_db() as db:
            existing = (
                db.query(RevenueSearchCache)
                .filter(
                    RevenueSearchCache.company_name == result.company_name,
                    RevenueSearchCache.fiscal_year  == result.year,
                    RevenueSearchCache.kpi          == kpi,
                )
                .first()
            )
            if existing:
                existing.revenue_cr      = result.value_cr
                existing.raw_value       = result.raw_value
                existing.source_url      = result.source_url
                existing.source_domain   = result.source_domain
                existing.is_consolidated = result.is_consolidated
                existing.confidence      = result.confidence
                existing.is_valid        = True
                existing.searched_at     = datetime.now(timezone.utc)
                print(f"Updated cache for {result.company_name} FY{result.year} [{kpi}]: {result.value_cr}")
            else:
                db.add(RevenueSearchCache(
                    company_name     = result.company_name,
                    fiscal_year      = result.year,
                    kpi              = kpi,
                    revenue_cr       = result.value_cr,
                    raw_value        = result.raw_value,
                    source_url       = result.source_url,
                    source_domain    = result.source_domain,
                    is_consolidated  = result.is_consolidated,
                    confidence       = result.confidence,
                    extraction_method = "llm_web_search",
                ))
                print(f"Added to cache for {result.company_name} FY{result.year} [{kpi}]: {result.value_cr}")
            db.commit()
            logger.info("revenue_llm.cache_stored", company=result.company_name,
                        year=result.year, value_cr=result.value_cr, kpi=kpi)
    except Exception as exc:
        logger.warning("revenue_llm.cache_store_failed", error=str(exc))


# Async wrapper
async def search_revenue_async(
    company_name: str,
    fiscal_year: int,
    kpi: str = "revenue_from_operations",
) -> Optional[RevenueSearchResult]:
    """Async wrapper for revenue search."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, search_revenue, company_name, fiscal_year, kpi)


def search_both_revenues(
    company_name: str,
    fiscal_year: int,
) -> tuple[Optional[RevenueSearchResult], Optional[RevenueSearchResult]]:
    """
    Search for both revenue_from_operations and net_revenue.
    Returns (revenue_from_ops, net_revenue) tuple.
    """
    rev_ops = search_revenue(company_name, fiscal_year, kpi="revenue_from_operations")
    net_rev = search_revenue(company_name, fiscal_year, kpi="net_revenue")
    return rev_ops, net_rev