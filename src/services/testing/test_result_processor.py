"""Test result processing and aggregation for the testing service."""

import logging
from datetime import UTC, datetime
from typing import Any

from models.testing_models import (
    TestRequest,
    TestResult,
)
from services.stubs import RedisCacheManager

logger = logging.getLogger(__name__)


class TestResultProcessor:
    """
    Processes and aggregates test results, and manages result caching.
    """

    def __init__(self, cache_manager: RedisCacheManager | None = None):
        self.cache_manager = cache_manager

    async def process_result(self, result: TestResult, request: TestRequest) -> TestResult:
        """Process and enhance test result."""
        # Cache successful results
        if self.cache_manager and result.success:
            await self._cache_result(request, result)

        return result

    async def _cache_result(self, request: TestRequest, result: TestResult) -> None:
        """Cache test result."""
        if not self.cache_manager:
            return

        cache_key = f"test_result:{result.test_id}"
        cache_data = result.model_dump(exclude={"test_id", "created_at"})

        # 1 hour TTL
        await self.cache_manager.set(cache_key, cache_data, ttl=3600)

    def convert_container_result(
        self,
        container_result: dict[str, Any],
        test_id: str,
        request: TestRequest,
    ) -> TestResult:
        """Convert container execution result to TestResult."""
        return TestResult(
            test_id=test_id,
            success=container_result.get("exit_code", 1) == 0,
            stdout=container_result.get("stdout", ""),
            stderr=container_result.get("stderr", ""),
            exit_code=container_result.get("exit_code", 1),
            execution_time=container_result.get("execution_time", 0.0),
            created_at=datetime.now(UTC),
            language=request.language,
            security_level=request.security_level,
            coverage_data=None,
            test_metadata=None,
            test_type=None,
            container_name=None,
            quality_report=None,
            error=None,
        )
