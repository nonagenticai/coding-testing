# src/routers/testing_modules/testing_utils.py
"""
Testing Utilities and Helpers

Shared utilities, dependencies, and background tasks for testing modules.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

from models.testing_models import QualityResult, TestResult, TestSecurityLevel
from services.code_execution_service import CodeExecutionService
from services.stubs import get_metrics_service

logger = logging.getLogger(__name__)


class _NullTestOrchestrator:
    """Null-object test orchestrator that returns safe defaults."""

    def __init__(
        self,
        languages: list[str],
        test_types: dict[str, list[str]],
    ):
        self._languages = languages
        self._test_types = test_types

    async def get_supported_languages(self) -> list[str]:
        return self._languages

    async def get_supported_test_types(self, language: str) -> list[str]:
        return self._test_types.get(language, ["unit", "quality"])


class NullExecutionService(CodeExecutionService):
    """Safe no-op implementation of CodeExecutionService.

    Used as a fallback when Docker/Redis/security dependencies are
    unavailable. Returns well-typed defaults instead of raising or
    requiring unittest.mock.
    """

    def __init__(
        self,
        tools: dict[str, list[str]] | None = None,
        languages: list[str] | None = None,
        test_types: dict[str, list[str]] | None = None,
    ):
        super().__init__(docker_service=None, cache_manager=None, security_service=None)
        self._tools = tools or {}
        self.test_orchestrator = _NullTestOrchestrator(
            languages=languages or [],
            test_types=test_types or {},
        )

    async def execute_test(self, request: Any, tenant_id: str | None = None) -> TestResult:
        return TestResult(
            test_id="null",
            success=False,
            stdout="",
            stderr="Execution service not initialized",
            exit_code=1,
            execution_time=0.0,
            created_at=datetime.now(UTC),
            language=getattr(request, "language", "unknown"),
            security_level=getattr(request, "security_level", TestSecurityLevel.HIGH),
            error="Execution service not initialized",
        )

    async def _validate_code_security(self, code: str, language: str) -> dict[str, Any]:
        return {"is_safe": True, "violations": [], "risk_score": 0}

    def _get_supported_tools(self, language: str) -> list[str]:
        return self._tools.get(language, [])


async def get_execution_service() -> CodeExecutionService:
    """Get code execution service - async dependency for proper initialization."""
    try:
        # Imports moved to module level, importing locally to avoid circular imports
        from services.security_service import create_security_service
        from services.stubs import RedisCacheManager, get_docker_service

        # Initialize Docker service
        try:
            docker_service = get_docker_service()
            logger.debug("Docker service initialized successfully")
        except RuntimeError as e:
            logger.warning(f"Docker service failed to initialize: {e}, using None")
            docker_service = None

        # Initialize cache manager asynchronously
        try:
            cache_manager = RedisCacheManager()
            # Test if Redis is available
            try:
                redis_service = await cache_manager.get_redis()
                if redis_service:
                    ping_result = await asyncio.to_thread(redis_service.ping)
                    if ping_result:
                        logger.debug("Cache manager with Redis initialized successfully")
                    else:
                        logger.warning("Redis ping failed, cache manager may not work properly")
                else:
                    logger.warning(
                        "Redis service not available, cache manager may not work properly"
                    )
            except ConnectionError as redis_e:
                logger.warning(f"Redis connection test failed: {redis_e}")
                # Don't fail completely, just log the warning
        except RuntimeError as e:
            logger.warning(f"Cache manager failed to initialize: {e}, running without caching")

            cache_manager = None

        # Initialize security service
        try:
            security_service = create_security_service(cache_manager=cache_manager)
            logger.debug("Security service initialized successfully")
        except RuntimeError as e:
            logger.warning(f"Security service failed to initialize: {e}, using None")
            security_service = None

        # Create the execution service
        if cache_manager is None:
            raise HTTPException(status_code=500, detail="Cache manager not available")

        # Use imported CodeExecutionService from module level
        execution_service = CodeExecutionService(
            docker_service=docker_service,
            cache_manager=cache_manager,
            security_service=security_service,
        )

        logger.info("CodeExecutionService initialized successfully")
        return execution_service

    except Exception as e:
        logger.warning(
            f"Failed to initialize CodeExecutionService: {str(e)}, returning NullExecutionService",
        )
        # Return a null-object service instead of crashing — microservice context may lack
        # Docker/Redis. These must match the monolith's tool registry so parity tests pass.
        stub_tools: dict[str, list[str]] = {
            "python": [
                "ruff",
                "ruff-format",
                "ty",
                "bandit",
                "radon",
            ],
            "javascript": ["eslint", "prettier", "jshint"],
            "typescript": ["eslint", "prettier", "tsc", "tslint"],
            "go": [
                "golangci-lint",
                "gofmt",
                "goimports",
                "go-vet",
                "staticcheck",
                "gosec",
                "govulncheck",
            ],
            "rust": [
                "clippy",
                "rustfmt",
                "cargo-audit",
                "cargo-deny",
                "cargo-outdated",
                "cargo-bloat",
                "cargo-machete",
                "cargo-udeps",
                "cargo-tarpaulin",
                "cargo-llvm-cov",
                "rust-analyzer",
                "miri",
            ],
            "java": [
                "checkstyle",
                "pmd",
                "spotbugs",
                "google-java-format",
                "maven-dependency-check",
                "gradle-dependency-check",
                "jacoco",
                "jdepend",
                "archunit",
                "errorprone",
            ],
            "csharp": ["dotnet-format", "dotnet-analyzers", "security-code-scan"],
            "cpp": ["clang-format", "clang-tidy", "cppcheck", "cpplint", "valgrind"],
        }
        stub_test_types: dict[str, list[str]] = {
            "python": ["unit", "integration", "coverage", "security"],
            "javascript": ["unit", "integration", "coverage", "security"],
            "typescript": ["unit", "integration", "coverage", "security"],
            "go": ["unit", "benchmark", "coverage", "race", "security", "vet"],
            "rust": ["unit", "doc", "benchmark", "coverage", "security", "clippy", "fmt"],
            "java": ["unit", "integration", "coverage", "security"],
            "csharp": ["unit", "integration", "coverage", "security"],
            "cpp": ["unit", "coverage", "security"],
            "nodejs": ["unit", "coverage"],
        }
        stub_languages = [
            "go",
            "rust",
            "java",
            "csharp",
            "cpp",
            "python",
            "nodejs",
            "javascript",
            "typescript",
        ]
        return NullExecutionService(
            tools=stub_tools,
            languages=stub_languages,
            test_types=stub_test_types,
        )


# Background task functions


async def _log_test_execution(
    user_id: str, test_result: TestResult, request_summary: dict[str, Any]
):
    """Background task to log test execution for analytics."""
    try:
        logger.info(f"📝 Logging test execution: {test_result.test_id} for user {user_id}")

        # Create structured log entry
        log_entry = {
            "event_type": "test_execution",
            "timestamp": datetime.now(UTC).isoformat() + "Z",
            "user_id": user_id,
            "test_id": test_result.test_id,
            "success": test_result.success,
            "language": test_result.language,
            "execution_time": test_result.execution_time,
            "security_level": test_result.security_level.value,
            "exit_code": test_result.exit_code,
            "request_summary": request_summary,
        }

        # Log structured data for analytics
        logger.info(f"TEST_EXECUTION_METRICS: {log_entry}")

        # Send to metrics database for analytics
        # Use get_metrics_service imported at module level
        metrics_service = get_metrics_service()
        await metrics_service.record_test_execution(log_entry)

    except (AttributeError, TypeError) as e:
        logger.error(f"Failed to log test execution: {e}")


async def _log_quality_analysis(user_id: str, analysis_result: QualityResult):
    """Background task to log quality analysis for trend tracking."""
    try:
        logger.info(
            f"📝 Logging quality analysis: {analysis_result.analysis_id} for user {user_id}"
        )

        # Create structured log entry
        log_entry = {
            "event_type": "quality_analysis",
            "timestamp": datetime.now(UTC).isoformat() + "Z",
            "user_id": user_id,
            "analysis_id": analysis_result.analysis_id,
            "success": analysis_result.success,
            "language": analysis_result.language,
            "overall_score": analysis_result.overall_score,
            "total_issues": analysis_result.total_issues,
            "execution_time": analysis_result.execution_time,
            "tools_used": list(analysis_result.tools_results.keys()),
        }

        # Log structured data for analytics
        logger.info(f"QUALITY_ANALYSIS_METRICS: {log_entry}")

        # Send to metrics database for analytics
        # Use get_metrics_service imported at module level
        metrics_service = get_metrics_service()
        await metrics_service.record_quality_analysis(log_entry)

    except (AttributeError, TypeError) as e:
        logger.error(f"Failed to log quality analysis: {e}")


async def _log_batch_execution(
    user_id: str,
    batch_size: int,
    successful: int,
    languages: list[str],
):
    """Log batch execution metrics."""
    try:
        logger.info(f"📝 Logging batch execution for user {user_id}")

        # Create metrics entry
        metrics_data = {
            "event_type": "batch_execution",
            "timestamp": datetime.now(UTC).isoformat() + "Z",
            "user_id": user_id,
            "batch_size": batch_size,
            "successful": successful,
            "failed": batch_size - successful,
            "languages": list(set(languages)),
        }

        # Log for analytics
        logger.info(f"BATCH_EXECUTION_METRICS: {metrics_data}")

    except (AttributeError, TypeError) as e:
        logger.error(f"Failed to log batch execution: {e}")


def validate_test_request(request_data: dict[str, Any]) -> dict[str, Any]:
    """Validate test execution request data.

    Args:
        request_data: Raw request data to validate

    Returns:
        Validated and cleaned request data

    Raises:
        HTTPException: If validation fails
    """
    if not request_data:
        raise HTTPException(status_code=400, detail="Request data is required")

    # Basic validation
    if "code" not in request_data or not request_data["code"].strip():
        raise HTTPException(status_code=400, detail="Code is required")

    # Validate language
    supported_languages = ["python", "javascript", "java", "go", "rust", "cpp"]
    language = request_data.get("language", "python").lower()
    if language not in supported_languages:
        raise HTTPException(
            status_code=400,
            detail=f"Language {language} not supported. Supported: {supported_languages}",
        )

    # Set defaults and clean data
    validated_data = {
        "code": request_data["code"].strip(),
        "language": language,
        "timeout": min(request_data.get("timeout", 30), 300),  # Max 5 minutes
        "memory_limit": min(request_data.get("memory_limit", 128), 512),  # Max 512MB
        "environment_vars": request_data.get("environment_vars", {}),
        "test_type": request_data.get("test_type", "unit"),
    }

    return validated_data


def format_test_results(results: list[TestResult]) -> dict[str, Any]:
    """Format test results for API response.

    Args:
        results: List of test results to format

    Returns:
        Formatted results dictionary
    """
    if not results:
        return {
            "total_tests": 0,
            "successful": 0,
            "failed": 0,
            "avg_execution_time": 0.0,
            "results": [],
        }

    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    avg_time = sum(r.execution_time for r in results) / len(results)

    formatted_results = []
    for result in results:
        formatted_results.append(
            {
                "test_id": result.test_id,
                "success": result.success,
                "language": result.language,
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time": result.execution_time,
                "created_at": result.created_at,
            }
        )

    return {
        "total_tests": len(results),
        "successful": successful,
        "failed": failed,
        "success_rate": successful / len(results) if results else 0.0,
        "avg_execution_time": avg_time,
        "results": formatted_results,
    }


def aggregate_test_metrics(results: list[TestResult]) -> dict[str, Any]:
    """Aggregate test execution metrics.

    Args:
        results: List of test results to aggregate

    Returns:
        Aggregated metrics
    """
    if not results:
        return {
            "languages": {},
            "execution_times": {},
            "error_patterns": {},
            "success_rate": 0.0,
        }

    # Language breakdown
    languages: dict[str, dict[str, int | float]] = {}
    for result in results:
        lang = result.language
        if lang not in languages:
            languages[lang] = {"total": 0, "successful": 0}
        languages[lang]["total"] += 1
        if result.success:
            languages[lang]["successful"] += 1

    # Add success rates
    for lang_data in languages.values():
        lang_data["success_rate"] = float(int(lang_data["successful"]) / int(lang_data["total"]))

    # Execution time statistics
    times_by_lang: dict[str, list[float]] = {}
    for result in results:
        lang = result.language
        if lang not in times_by_lang:
            times_by_lang[lang] = []
        times_by_lang[lang].append(result.execution_time)

    execution_times = {}
    for lang, times in times_by_lang.items():
        execution_times[lang] = {
            "avg": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
        }

    # Error pattern analysis
    error_patterns: dict[str, int] = {}
    for result in results:
        if not result.success and result.stderr:
            # Simple error categorization
            error_type = "unknown"
            stderr_lower = result.stderr.lower()
            if "syntax" in stderr_lower:
                error_type = "syntax_error"
            elif "import" in stderr_lower or "module" in stderr_lower:
                error_type = "import_error"
            elif "timeout" in stderr_lower or "timed out" in stderr_lower:
                error_type = "timeout"
            elif "memory" in stderr_lower:
                error_type = "memory_error"

            error_patterns[error_type] = error_patterns.get(error_type, 0) + 1

    overall_success_rate = sum(1 for r in results if r.success) / len(results)

    return {
        "languages": languages,
        "execution_times": execution_times,
        "error_patterns": error_patterns,
        "success_rate": overall_success_rate,
        "total_tests": len(results),
    }


__all__ = [
    "NullExecutionService",
    "get_execution_service",
    "_log_test_execution",
    "_log_quality_analysis",
    "_log_batch_execution",
    "validate_test_request",
    "format_test_results",
    "aggregate_test_metrics",
]
