"""
Intelligent Test Integration Module
Comprehensive integration layer for PydanticAI-powered test orchestration.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from config.test_framework_config import TestConfig
from models.testing_models import SecurityLevel, TestRequest
from services.stubs import DockerService
from services.test_execution.intelligent_test_orchestrator import (
    IntelligentTestExecutionRequest,
    IntelligentTestExecutionResult,
    IntelligentTestOrchestrator,
    OrchestratorType,
)
from services.test_execution.pydantic_ai_agents import (
    ExecutionPattern,
    PydanticAIRecommendation,
)
from services.test_execution.test_orchestrator import TestLanguage

logger = logging.getLogger(__name__)


class TestExecutionSettings(BaseModel):
    """Configuration settings for intelligent test execution."""

    enable_ai_optimization: bool = Field(default=True)
    enable_pattern_analysis: bool = Field(default=True)
    enable_failure_analysis: bool = Field(default=True)
    enable_scheduling_optimization: bool = Field(default=True)
    ai_model: str = Field(default="gpt-4o-mini")
    max_concurrent_tests: int = Field(default=5, ge=1, le=20)
    ai_analysis_threshold: int = Field(default=10, ge=5, le=100)
    pattern_analysis_interval: int = Field(default=300, ge=60, le=3600)  # seconds
    resource_monitoring_interval: int = Field(default=30, ge=10, le=300)  # seconds
    queue_optimization_interval: int = Field(default=120, ge=60, le=600)  # seconds
    execution_timeout: int = Field(default=600, ge=60, le=3600)  # seconds
    retry_attempts: int = Field(default=3, ge=1, le=10)
    retry_delay: float = Field(default=1.0, ge=0.1, le=10.0)  # seconds

    @field_validator("max_concurrent_tests")
    @classmethod
    def validate_max_concurrent_tests(cls, v):
        """Validate max concurrent tests is reasonable."""
        if v < 1 or v > 20:
            raise ValueError("max_concurrent_tests must be between 1 and 20")
        return v

    @field_validator("ai_analysis_threshold")
    @classmethod
    def validate_ai_analysis_threshold(cls, v):
        """Validate AI analysis threshold is reasonable."""
        if v < 5 or v > 100:
            raise ValueError("ai_analysis_threshold must be between 5 and 100")
        return v


class TestExecutionError(Exception):
    """Custom exception for test execution errors."""

    def __init__(
        self,
        message: str,
        error_type: str = "general",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.details = details or {}
        self.timestamp = datetime.now(UTC)


class TestExecutionRetryError(TestExecutionError):
    """Exception for test execution retry failures."""

    def __init__(self, message: str, attempts: int, last_error: Exception):
        super().__init__(
            message,
            "retry_failure",
            {"attempts": attempts, "last_error": str(last_error)},
        )
        self.attempts = attempts
        self.last_error = last_error


class TestExecutionTimeoutError(TestExecutionError):
    """Exception for test execution timeouts."""

    def __init__(self, message: str, timeout_seconds: int):
        super().__init__(message, "timeout", {"timeout_seconds": timeout_seconds})
        self.timeout_seconds = timeout_seconds


class TestExecutionResourceError(TestExecutionError):
    """Exception for test execution resource errors."""

    def __init__(self, message: str, resource_type: str, resource_limit: str):
        super().__init__(
            message,
            "resource_limit",
            {"resource_type": resource_type, "resource_limit": resource_limit},
        )
        self.resource_type = resource_type
        self.resource_limit = resource_limit


class IntelligentTestExecutionService:
    """
    Comprehensive intelligent test execution service with error handling,
    retry logic, and performance monitoring.
    """

    def __init__(
        self,
        docker_service: DockerService,
        test_config: TestConfig,
        settings: TestExecutionSettings | None = None,
    ):
        """Initialize the intelligent test execution service."""
        self.docker_service = docker_service
        self.test_config = test_config
        self.settings = settings or TestExecutionSettings()

        # Initialize orchestrator
        self.orchestrator = IntelligentTestOrchestrator(docker_service, test_config)

        # Performance tracking
        self.performance_metrics: dict[str, float] = {}
        self.error_counts: dict[str, int] = {}
        self.last_health_check = datetime.now(UTC)

        # Status tracking
        self.is_running = False
        self.is_healthy = True
        self.startup_time = datetime.now(UTC)

        logger.info(
            f"Intelligent Test Execution Service initialized with settings: {self.settings.model_dump()}"
        )

    async def start(self) -> None:
        """Start the intelligent test execution service."""
        try:
            logger.info("Starting Intelligent Test Execution Service")

            # Start orchestrator
            await self.orchestrator.start()

            # Initialize performance tracking
            self.performance_metrics = {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "average_execution_time": 0.0,
                "ai_analysis_count": 0,
                "pattern_detections": 0,
                "recommendations_generated": 0,
            }

            self.error_counts = {
                "timeout_errors": 0,
                "resource_errors": 0,
                "retry_failures": 0,
                "ai_analysis_errors": 0,
                "general_errors": 0,
            }

            self.is_running = True
            self.is_healthy = True

            logger.info("Intelligent Test Execution Service started successfully")

        except Exception as e:
            logger.error(f"Failed to start Intelligent Test Execution Service: {e}")
            self.is_healthy = False
            raise TestExecutionError(f"Service startup failed: {e}", "startup_failure")

    async def stop(self) -> None:
        """Stop the intelligent test execution service."""
        try:
            logger.info("Stopping Intelligent Test Execution Service")

            # Stop orchestrator
            await self.orchestrator.stop()

            self.is_running = False

            logger.info("Intelligent Test Execution Service stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping Intelligent Test Execution Service: {e}")
            raise TestExecutionError(f"Service shutdown failed: {e}", "shutdown_failure")

    async def execute_test(
        self,
        request: TestRequest | IntelligentTestExecutionRequest,
        enable_retry: bool = True,
    ) -> IntelligentTestExecutionResult:
        """Execute a test with intelligent analysis and error handling."""
        if not self.is_running:
            raise TestExecutionError("Service is not running", "service_not_running")

        # Convert TestRequest to IntelligentTestExecutionRequest if needed
        if isinstance(request, TestRequest):
            request = await self._convert_test_request(request)

        # Validate request
        await self._validate_test_request(request)

        # Execute with retry logic
        if enable_retry:
            return await self._execute_with_retry(request)
        else:
            return await self._execute_single_attempt(request)

    async def execute_batch(
        self,
        requests: list[TestRequest | IntelligentTestExecutionRequest],
        enable_retry: bool = True,
    ) -> list[IntelligentTestExecutionResult]:
        """Execute a batch of tests with intelligent optimization."""
        if not self.is_running:
            raise TestExecutionError("Service is not running", "service_not_running")

        logger.info(f"Starting batch execution of {len(requests)} tests")

        # Convert requests
        converted_requests = []
        for req in requests:
            if isinstance(req, TestRequest):
                converted_requests.append(await self._convert_test_request(req))
            else:
                converted_requests.append(req)

        # Validate all requests
        for req in converted_requests:
            await self._validate_test_request(req)

        # Execute batch with error handling
        try:
            results = await self.orchestrator.batch_execute_tests(converted_requests)

            # Update performance metrics
            self._update_batch_metrics(results)

            return results

        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            self.error_counts["general_errors"] += 1
            raise TestExecutionError(f"Batch execution failed: {e}", "batch_execution_failure")

    async def get_execution_insights(self) -> dict[str, Any]:
        """Get comprehensive execution insights with error handling."""
        try:
            insights = await self.orchestrator.get_execution_insights()

            # Add service-level metrics
            insights["service_metrics"] = {
                "performance_metrics": self.performance_metrics,
                "error_counts": self.error_counts,
                "service_health": {
                    "is_running": self.is_running,
                    "is_healthy": self.is_healthy,
                    "startup_time": self.startup_time.isoformat(),
                    "last_health_check": self.last_health_check.isoformat(),
                },
            }

            return insights

        except Exception as e:
            logger.error(f"Failed to get execution insights: {e}")
            return {
                "error": "Failed to get execution insights",
                "details": str(e),
                "service_metrics": {
                    "performance_metrics": self.performance_metrics,
                    "error_counts": self.error_counts,
                },
            }

    async def get_recommendations(self) -> list[PydanticAIRecommendation]:
        """Get AI-generated recommendations with error handling."""
        try:
            recommendations = await self.orchestrator.get_recommendations()

            # Add service-level recommendations
            service_recommendations = await self._generate_service_recommendations()
            recommendations.extend(service_recommendations)

            return recommendations

        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            self.error_counts["ai_analysis_errors"] += 1
            return []

    async def get_patterns(self) -> list[ExecutionPattern]:
        """Get identified patterns with error handling."""
        try:
            patterns = await self.orchestrator.get_patterns()

            # Add service-level patterns
            service_patterns = await self._detect_service_patterns()
            patterns.extend(service_patterns)

            return patterns

        except Exception as e:
            logger.error(f"Failed to get patterns: {e}")
            self.error_counts["ai_analysis_errors"] += 1
            return []

    async def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status."""
        try:
            # Update health check timestamp
            self.last_health_check = datetime.now(UTC)

            # Get queue status
            queue_status = await self.orchestrator.get_queue_status()

            # Calculate health metrics
            total_executions = self.performance_metrics.get("total_executions", 0)
            success_rate = (
                self.performance_metrics.get("successful_executions", 0) / total_executions
                if total_executions > 0
                else 1.0
            )

            error_rate = sum(self.error_counts.values()) / max(total_executions, 1)

            # Determine health status
            is_healthy = (
                self.is_running
                and success_rate >= 0.8  # At least 80% success rate
                and error_rate <= 0.2  # At most 20% error rate
                and queue_status.get("pending", 0) < 100  # Not overwhelmed
            )

            self.is_healthy = is_healthy

            return {
                "service_status": {
                    "is_running": self.is_running,
                    "is_healthy": is_healthy,
                    "startup_time": self.startup_time.isoformat(),
                    "last_health_check": self.last_health_check.isoformat(),
                    "uptime_seconds": (datetime.now(UTC) - self.startup_time).total_seconds(),
                },
                "performance_metrics": self.performance_metrics,
                "error_counts": self.error_counts,
                "queue_status": queue_status,
                "health_scores": {
                    "success_rate": success_rate,
                    "error_rate": error_rate,
                    "overall_health": 1.0 if is_healthy else 0.0,
                },
                "settings": self.settings.model_dump(),
            }

        except Exception as e:
            logger.error(f"Failed to get health status: {e}")
            self.is_healthy = False
            return {
                "service_status": {
                    "is_running": self.is_running,
                    "is_healthy": False,
                    "error": str(e),
                },
                "performance_metrics": self.performance_metrics,
                "error_counts": self.error_counts,
            }

    async def _convert_test_request(self, request: TestRequest) -> IntelligentTestExecutionRequest:
        """Convert TestRequest to IntelligentTestExecutionRequest."""
        try:
            # Map language string to enum
            language_mapping = {
                "python": TestLanguage.PYTHON,
                "javascript": TestLanguage.JAVASCRIPT,
                "typescript": TestLanguage.TYPESCRIPT,
                "java": TestLanguage.JAVA,
                "go": TestLanguage.GO,
                "rust": TestLanguage.RUST,
                "csharp": TestLanguage.CSHARP,
                "cpp": TestLanguage.CPP,
                "nodejs": TestLanguage.NODEJS,
            }

            language = language_mapping.get(request.language.lower())
            if not language:
                raise TestExecutionError(
                    f"Unsupported language: {request.language}", "unsupported_language"
                )

            # Determine test type
            test_type = OrchestratorType.UNIT  # Default
            if hasattr(request, "test_type") and request.test_type:
                test_type_mapping = {
                    "unit": OrchestratorType.UNIT,
                    "integration": OrchestratorType.INTEGRATION,
                    "benchmark": OrchestratorType.BENCHMARK,
                    "coverage": OrchestratorType.COVERAGE,
                    "security": OrchestratorType.SECURITY,
                }
                test_type = test_type_mapping.get(request.test_type.lower(), OrchestratorType.UNIT)

            # Create test files dictionary
            test_files = {}
            if request.test_code:
                extension = self._get_file_extension(request.language)
                test_files[f"test_main{extension}"] = request.test_code

            # Create intelligent request
            intelligent_request = IntelligentTestExecutionRequest(
                language=language,
                test_type=test_type,
                source_code=request.code or "",
                container_name=request.container_name,
                test_files=test_files,
                timeout_seconds=request.timeout or 300,
                security_level=request.security_level or SecurityLevel.MEDIUM,
                project_name=f"test_{datetime.now(UTC).timestamp()}",
                enable_ai_optimization=self.settings.enable_ai_optimization,
                enable_pattern_analysis=self.settings.enable_pattern_analysis,
                enable_failure_analysis=self.settings.enable_failure_analysis,
                enable_scheduling_optimization=self.settings.enable_scheduling_optimization,
                priority_level=5,  # Default priority
                resource_weight=1.0,
            )

            return intelligent_request

        except Exception as e:
            logger.error(f"Failed to convert test request: {e}")
            raise TestExecutionError(
                f"Request conversion failed: {e}", "request_conversion_failure"
            )

    async def _validate_test_request(self, request: IntelligentTestExecutionRequest) -> None:
        """Validate test execution request."""
        try:
            # Basic validation
            if not request.source_code.strip():
                raise TestExecutionError("Source code cannot be empty", "invalid_request")

            if request.timeout_seconds > self.settings.execution_timeout:
                raise TestExecutionError(
                    f"Timeout exceeds maximum allowed ({self.settings.execution_timeout}s)",
                    "timeout_too_long",
                )

            if len(request.source_code) > 1000000:  # 1MB limit
                raise TestExecutionError("Source code exceeds size limit (1MB)", "code_too_large")

            # Security validation
            if (
                request.security_level == SecurityLevel.HIGH
                and not self.settings.enable_ai_optimization
            ):
                logger.warning("High security level requested but AI optimization is disabled")

        except TestExecutionError:
            raise
        except Exception as e:
            logger.error(f"Request validation failed: {e}")
            raise TestExecutionError(f"Request validation failed: {e}", "validation_failure")

    async def _execute_with_retry(
        self, request: IntelligentTestExecutionRequest
    ) -> IntelligentTestExecutionResult:
        """Execute test with retry logic."""
        last_error: Exception | None = None

        for attempt in range(self.settings.retry_attempts):
            try:
                logger.debug(f"Test execution attempt {attempt + 1}/{self.settings.retry_attempts}")

                # Execute test
                result = await self._execute_single_attempt(request)

                # Update metrics
                self._update_execution_metrics(result)

                return result

            except TestExecutionTimeoutError as e:
                logger.warning(f"Test execution timed out on attempt {attempt + 1}: {e}")
                last_error = e
                self.error_counts["timeout_errors"] += 1

                if attempt < self.settings.retry_attempts - 1:
                    await asyncio.sleep(self.settings.retry_delay * (attempt + 1))

            except TestExecutionResourceError as e:
                logger.warning(f"Resource error on attempt {attempt + 1}: {e}")
                last_error = e
                self.error_counts["resource_errors"] += 1

                if attempt < self.settings.retry_attempts - 1:
                    await asyncio.sleep(self.settings.retry_delay * (attempt + 1))

            except Exception as e:
                logger.error(f"Test execution failed on attempt {attempt + 1}: {e}")
                last_error = e
                self.error_counts["general_errors"] += 1

                if attempt < self.settings.retry_attempts - 1:
                    await asyncio.sleep(self.settings.retry_delay * (attempt + 1))

        # All attempts failed
        self.error_counts["retry_failures"] += 1
        if last_error is None:
            last_error = TestExecutionError("Test execution failed with unknown error")
        raise TestExecutionRetryError(
            f"Test execution failed after {self.settings.retry_attempts} attempts",
            self.settings.retry_attempts,
            last_error,
        )

    async def _execute_single_attempt(
        self, request: IntelligentTestExecutionRequest
    ) -> IntelligentTestExecutionResult:
        """Execute test in a single attempt with timeout."""
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self.orchestrator.execute_test(request),
                timeout=self.settings.execution_timeout,
            )

            return result

        except TimeoutError:
            raise TestExecutionTimeoutError(
                f"Test execution timed out after {self.settings.execution_timeout} seconds",
                self.settings.execution_timeout,
            )
        except Exception as e:
            # Check if it's a resource error
            if "memory" in str(e).lower() or "resource" in str(e).lower():
                raise TestExecutionResourceError(
                    f"Resource limit exceeded: {e}", "memory", "unknown"
                )

            # Re-raise as general execution error
            raise TestExecutionError(f"Test execution failed: {e}", "execution_failure")

    def _update_execution_metrics(self, result: IntelligentTestExecutionResult) -> None:
        """Update performance metrics based on execution result."""
        try:
            self.performance_metrics["total_executions"] += 1

            if result.success:
                self.performance_metrics["successful_executions"] += 1
            else:
                self.performance_metrics["failed_executions"] += 1

            # Update average execution time
            total_time = (
                self.performance_metrics.get("average_execution_time", 0)
                * (self.performance_metrics["total_executions"] - 1)
                + result.execution_time_ms
            )
            self.performance_metrics["average_execution_time"] = (
                total_time / self.performance_metrics["total_executions"]
            )

            # Update AI analysis metrics
            if result.ai_analysis:
                self.performance_metrics["ai_analysis_count"] += 1

                if result.ai_analysis.get("pattern_analysis"):
                    self.performance_metrics["pattern_detections"] += len(
                        result.ai_analysis["pattern_analysis"]
                    )

                if result.ai_analysis.get("recommendations"):
                    self.performance_metrics["recommendations_generated"] += len(
                        result.ai_analysis["recommendations"]
                    )

        except Exception as e:
            logger.error(f"Failed to update execution metrics: {e}")

    def _update_batch_metrics(self, results: list[IntelligentTestExecutionResult]) -> None:
        """Update metrics for batch execution."""
        try:
            for result in results:
                self._update_execution_metrics(result)

        except Exception as e:
            logger.error(f"Failed to update batch metrics: {e}")

    async def _generate_service_recommendations(
        self,
    ) -> list[PydanticAIRecommendation]:
        """Generate service-level recommendations."""
        recommendations = []

        try:
            # Analyze error patterns
            total_errors = sum(self.error_counts.values())
            total_executions = self.performance_metrics.get("total_executions", 0)

            if total_executions > 0:
                error_rate = total_errors / total_executions

                if error_rate > 0.2:  # More than 20% error rate
                    recommendations.append(
                        PydanticAIRecommendation(
                            recommendation_id=f"service-error-rate-{datetime.now(UTC).timestamp()}",
                            recommendation_type="configuration",
                            priority="high",
                            title="High error rate detected",
                            description=f"Service has {error_rate:.1%} error rate, which exceeds recommended threshold",
                            impact_assessment="High impact on test reliability",
                            implementation_steps=[
                                "Review error logs for common patterns",
                                "Increase resource limits if resource errors are common",
                                "Consider reducing concurrent test limit",
                                "Review test code quality",
                            ],
                            confidence_level=0.9,
                        )
                    )

                # Check for timeout issues
                if self.error_counts.get("timeout_errors", 0) > total_executions * 0.1:
                    recommendations.append(
                        PydanticAIRecommendation(
                            recommendation_id=f"service-timeout-{datetime.now(UTC).timestamp()}",
                            recommendation_type="configuration",
                            priority="medium",
                            title="Frequent timeout errors",
                            description="Tests are timing out frequently",
                            impact_assessment="Medium impact on test execution time",
                            implementation_steps=[
                                "Increase timeout limits",
                                "Optimize test performance",
                                "Review container resource allocation",
                            ],
                            confidence_level=0.8,
                        )
                    )

            return recommendations

        except Exception as e:
            logger.error(f"Failed to generate service recommendations: {e}")
            return []

    async def _detect_service_patterns(self) -> list[ExecutionPattern]:
        """Detect service-level patterns."""
        patterns: list[ExecutionPattern] = []

        try:
            # Pattern detection logic would go here
            # For now, return empty list
            return patterns

        except Exception as e:
            logger.error(f"Failed to detect service patterns: {e}")
            return []

    def _get_file_extension(self, language: str) -> str:
        """Get file extension for language."""
        extensions = {
            "python": ".py",
            "javascript": ".js",
            "typescript": ".ts",
            "java": ".java",
            "go": ".go",
            "rust": ".rs",
            "csharp": ".cs",
            "cpp": ".cpp",
            "nodejs": ".js",
        }
        return extensions.get(language.lower(), ".txt")

    @asynccontextmanager
    async def managed_execution(self):
        """Context manager for managed test execution service."""
        try:
            await self.start()
            yield self
        finally:
            await self.stop()


# Factory function for creating the service
def create_intelligent_test_service(
    docker_service: DockerService,
    test_config: TestConfig,
    settings: TestExecutionSettings | None = None,
) -> IntelligentTestExecutionService:
    """Factory function to create IntelligentTestExecutionService."""
    return IntelligentTestExecutionService(docker_service, test_config, settings)


# Convenience function for quick execution
async def execute_test_with_intelligence(
    code: str,
    language: str,
    test_code: str | None = None,
    timeout: int = 300,
    docker_service: DockerService | None = None,
    test_config: TestConfig | None = None,
    settings: TestExecutionSettings | None = None,
) -> IntelligentTestExecutionResult:
    """
    Convenience function for executing a test with intelligent analysis.

    Args:
        code: Source code to test
        language: Programming language
        test_code: Optional test code
        timeout: Execution timeout in seconds
        docker_service: Optional Docker service (will create if not provided)
        test_config: Optional test configuration
        settings: Optional execution settings

    Returns:
        Intelligent test execution result
    """
    # Create services if not provided
    if docker_service is None:
        docker_service = DockerService()

    if test_config is None:
        test_config = TestConfig()

    # Create test request
    test_request = TestRequest(
        code=code,
        language=language,
        test_code=test_code,
        timeout=timeout,
        container_name=f"quick-test-{datetime.now(UTC).timestamp()}",
    )

    # Create and use service
    service = create_intelligent_test_service(docker_service, test_config, settings)

    async with service.managed_execution():
        return await service.execute_test(test_request)
