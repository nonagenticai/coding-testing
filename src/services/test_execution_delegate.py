"""
Test Execution Delegate - Handles test running logic for CodeExecutionService.

This delegate encapsulates all test execution functionality including:
- Test request execution via orchestrator
- Test result caching
- Language-specific test handling
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid7

from models.testing_models import (
    QualityRequest,
    TestRequest,
    TestResult,
    create_test_result,
)
from services.test_execution.test_orchestrator import (
    OrchestratorType,
    TestExecutionRequest,
    TestLanguage,
)

if TYPE_CHECKING:
    from services.stubs import RedisCacheManager
    from services.test_execution.test_orchestrator import TestOrchestrator

logger = logging.getLogger(__name__)


class QualityAnalyzerProtocol(Protocol):
    """Protocol for quality analysis capability."""

    async def analyze_quality(self, request: QualityRequest) -> Any:
        """Analyze code quality."""
        ...


class TestExecutionDelegate:
    """
    Delegate class handling test execution logic.

    Responsibilities:
    - Execute tests via orchestrator
    - Manage test result caching
    - Handle test-specific error handling
    """

    def __init__(
        self,
        test_orchestrator: TestOrchestrator,
        cache_manager: RedisCacheManager | None,
    ):
        """Initialize the TestExecutionDelegate.

        Args:
            test_orchestrator: Orchestrator for test execution
            cache_manager: Optional cache manager for result caching
        """
        self.test_orchestrator = test_orchestrator
        self.cache_manager = cache_manager
        logger.debug("TestExecutionDelegate initialized")

    def generate_cache_key(self, request: TestRequest) -> str:
        """Generate cache key for test request.

        Args:
            request: The test request to generate a key for

        Returns:
            A unique cache key string
        """
        cache_data = f"{request.code or ''}{request.test_code or ''}{request.language}{request.security_level.value}"
        # Non-cryptographic: MD5 is a fast digest for a cache key, not security.
        return f"test_{hashlib.md5(cache_data.encode(), usedforsecurity=False).hexdigest()}"

    async def get_cached_result(self, cache_key: str) -> TestResult | None:
        """Get cached test result.

        Args:
            cache_key: The cache key to look up

        Returns:
            Cached TestResult if found, None otherwise
        """
        try:
            if not self.cache_manager:
                return None
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data:
                logger.debug(f"Cache hit for test key: {cache_key}")
                try:
                    return TestResult(**cached_data)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to deserialize cached result: {e}")
            return None
        except (RuntimeError, OSError, ValueError) as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return None

    async def cache_result(self, cache_key: str, result: TestResult) -> None:
        """Cache test result.

        Args:
            cache_key: The cache key to store under
            result: The test result to cache
        """
        try:
            if not self.cache_manager:
                return
            await self.cache_manager.set(cache_key, result.model_dump(), ttl=3600)
            logger.debug(f"Cached test result: {cache_key}")
        except (RuntimeError, OSError, ValueError) as e:
            logger.warning(f"Cache storage failed: {e}")

    async def execute_with_orchestrator(
        self,
        request: TestRequest,
        quality_analyzer: QualityAnalyzerProtocol | None = None,
    ) -> TestResult:
        """Execute test using the test orchestrator for all languages.

        Args:
            request: The test request to execute
            quality_analyzer: Optional quality analyzer for post-test analysis

        Returns:
            TestResult with execution details
        """

        start_time = datetime.now(UTC)
        try:
            exec_request = TestExecutionRequest(
                source_code=request.code or "",
                test_files={"test.py": request.test_code} if request.test_code else {},
                language=TestLanguage(request.language),
                container_name=request.container_name,
                test_type=OrchestratorType(
                    request.test_type.value if request.test_type else "unit"
                ),
                timeout_seconds=request.timeout or 300,
                dependencies=request.dependencies or {},
                resource_limits=(
                    request.resource_limits.model_dump() if request.resource_limits else None
                ),
                security_level=request.security_level,
                project_name=request.container_name,
            )
            exec_result = await self.test_orchestrator.execute_tests(exec_request)

            quality_report = None
            if request.run_quality_check and exec_result.success and quality_analyzer:
                try:
                    quality_result = await quality_analyzer.analyze_quality(
                        QualityRequest(
                            container_name=request.container_name or "unknown",
                            code=request.code,
                            language=request.language,
                            tools=request.quality_tools,
                            code_content=request.code,
                            code_path=None,
                            custom_rules=None,
                        )
                    )
                    quality_report = quality_result
                except (TimeoutError, RuntimeError, OSError, ValueError) as e:
                    logger.warning(f"Quality analysis failed: {e}")

            return create_test_result(
                test_id=exec_result.test_id,
                success=exec_result.success,
                exit_code=exec_result.exit_code,
                stdout=exec_result.stdout,
                stderr=exec_result.stderr,
                execution_time=exec_result.execution_time_ms / 1000.0,
                language=request.language,
                test_type=request.test_type.value if request.test_type else "unit",
                security_level=request.security_level,
                created_at=exec_result.created_at,
                container_name=exec_result.container_name or "unknown",
                quality_report=quality_report,
                error=None if exec_result.success else exec_result.stderr,
            )
        except (TimeoutError, RuntimeError, OSError, ValueError) as e:
            logger.error(f"Test execution failed: {e}")
            execution_time = (datetime.now(UTC) - start_time).total_seconds()
            return create_test_result(
                test_id=str(uuid7()),
                success=False,
                container_name=request.container_name or "unknown",
                exit_code=1,
                stdout="",
                stderr=str(e),
                execution_time=execution_time,
                language=request.language,
                test_type=request.test_type.value if request.test_type else "unit",
                security_level=request.security_level,
                created_at=start_time,
                error=str(e),
            )
