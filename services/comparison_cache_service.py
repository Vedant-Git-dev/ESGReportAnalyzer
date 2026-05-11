"""
services/comparison_cache_service.py

Service for caching comparison summaries and recommendations.
Avoids redundant LLM calls for the same company/year comparisons.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import and_

from core.database import get_db
from core.logging_config import get_logger
from models.db_models import ComparisonCache

logger = get_logger(__name__)


@dataclass
class ComparisonCacheResult:
    """Result from comparison cache lookup."""
    summary: Optional[str]
    recommendation: Optional[str]
    created_at: datetime
    updated_at: datetime


class ComparisonCacheService:
    """
    Service for caching ESG comparison summaries and recommendations.

    Cache key is based on (company1, fy1, company2, fy2) tuple.
    The order of companies is normalized to ensure A vs B and B vs A
    share the same cache entry.
    """

    def get(
        self,
        company1: str,
        fy1: int,
        company2: str,
        fy2: int,
    ) -> Optional[ComparisonCacheResult]:
        """
        Retrieve cached comparison result if available.

        Args:
            company1: First company name
            fy1: First company fiscal year
            company2: Second company name
            fy2: Second company fiscal year

        Returns:
            ComparisonCacheResult if found, None otherwise
        """
        norm_c1, norm_c2, norm_fy1, norm_fy2 = self._normalize_pair(
            company1, fy1, company2, fy2
        )

        try:
            with get_db() as db:
                cache_entry = (
                    db.query(ComparisonCache)
                    .filter(
                        and_(
                            ComparisonCache.company1 == norm_c1,
                            ComparisonCache.fy1 == norm_fy1,
                            ComparisonCache.company2 == norm_c2,
                            ComparisonCache.fy2 == norm_fy2,
                        )
                    )
                    .first()
                )

                if cache_entry:
                    logger.info(
                        "comparison_cache.hit",
                        company1=norm_c1,
                        fy1=norm_fy1,
                        company2=norm_c2,
                        fy2=norm_fy2,
                    )
                    return ComparisonCacheResult(
                        summary=cache_entry.summary,
                        recommendation=cache_entry.recommendation,
                        created_at=cache_entry.created_at,
                        updated_at=cache_entry.updated_at,
                    )

                logger.info(
                    "comparison_cache.miss",
                    company1=norm_c1,
                    fy1=norm_fy1,
                    company2=norm_c2,
                    fy2=norm_fy2,
                )
                return None

        except Exception as e:
            logger.warning("comparison_cache.get_error", error=str(e))
            return None

    def store(
        self,
        company1: str,
        fy1: int,
        company2: str,
        fy2: int,
        sector: str,
        summary: str,
        recommendation: str,
    ) -> bool:
        """
        Store or update comparison result in cache.

        Args:
            company1: First company name
            fy1: First company fiscal year
            company2: Second company name
            fy2: Second company fiscal year
            sector: Sector string
            summary: Generated summary text
            recommendation: Generated recommendation text

        Returns:
            True if stored successfully, False otherwise
        """
        norm_c1, norm_c2, norm_fy1, norm_fy2 = self._normalize_pair(
            company1, fy1, company2, fy2
        )

        try:
            with get_db() as db:
                # Check if entry already exists
                existing = (
                    db.query(ComparisonCache)
                    .filter(
                        and_(
                            ComparisonCache.company1 == norm_c1,
                            ComparisonCache.fy1 == norm_fy1,
                            ComparisonCache.company2 == norm_c2,
                            ComparisonCache.fy2 == norm_fy2,
                        )
                    )
                    .first()
                )

                now = datetime.now(timezone.utc)

                if existing:
                    # Update existing entry
                    existing.summary = summary
                    existing.recommendation = recommendation
                    existing.sector = sector
                    existing.updated_at = now
                    logger.info(
                        "comparison_cache.updated",
                        company1=norm_c1,
                        fy1=norm_fy1,
                        company2=norm_c2,
                        fy2=norm_fy2,
                    )
                else:
                    # Create new entry
                    new_entry = ComparisonCache(
                        company1=norm_c1,
                        fy1=norm_fy1,
                        company2=norm_c2,
                        fy2=norm_fy2,
                        sector=sector,
                        summary=summary,
                        recommendation=recommendation,
                        created_at=now,
                        updated_at=now,
                    )
                    db.add(new_entry)
                    logger.info(
                        "comparison_cache.stored",
                        company1=norm_c1,
                        fy1=norm_fy1,
                        company2=norm_c2,
                        fy2=norm_fy2,
                    )

                db.flush()
                return True

        except Exception as e:
            logger.warning("comparison_cache.store_error", error=str(e))
            return False

    def _normalize_pair(
        self,
        company1: str,
        fy1: int,
        company2: str,
        fy2: int,
    ) -> tuple[str, str, int, int]:
        """
        Normalize company pair to ensure consistent cache keys.

        Normalizes by sorting company names alphabetically so that
        "TCS FY2024 vs Infosys FY2024" and "Infosys FY2024 vs TCS FY2024"
        produce the same cache key.

        Args:
            company1: First company name
            fy1: First company fiscal year
            company2: Second company name
            fy2: Second company fiscal year

        Returns:
            Tuple of (normalized_company1, normalized_company2, fy1, fy2)
        """
        c1_clean = company1.strip().lower()
        c2_clean = company2.strip().lower()

        # Sort by company name to ensure consistent ordering
        if (c1_clean, fy1) <= (c2_clean, fy2):
            return company1.strip(), company2.strip(), fy1, fy2
        else:
            return company2.strip(), company1.strip(), fy2, fy1
