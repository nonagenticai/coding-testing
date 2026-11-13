"""Service for executing tests within a container-based architecture."""

# TestExecutionService Implementation
# Integrates Python test suite components with container-based architecture using modular design

import logging
from typing import Any

from config.test_framework_config import TestConfig, TestFrameworkSecurityConfig
from models.testing_models import (
    TestRequest,
    TestResult,
)
from processing.logging_manager import LoggingManager
from processing.test_runner import TestRunner
from services.security_service import create_security_service
from services.stubs import DockerService, RedisCacheManager
from services.test_execution.intelligent_failure_analyzer import (
    FailureAnalysisConfig,
    IntelligentFailureAnalyzer,
    create_intelligent_failure_analyzer,
)
from services.testing.test_config_manager import TestConfigManager
from services.testing.test_container_manager import TestContainerManager

# Import modular components
from services.testing.test_executor import TestExecutor
from services.testing.test_monitor import TestMonitor
from services.testing.test_result_processor import TestResultProcessor

logger = logging.getLogger(__name__)


class TestExecutionService:
    """
    TestExecutionService that integrates Python test suite components
    with the container-based architecture for secure test execution.

    This service now uses a modular architecture with specialized components
    for different aspects of test execution.
    """

    def __init__(
        self,
        docker_service: DockerService,
        cache_manager: RedisCacheManager | None = None,
        config: Any | None = None,
        enable_intelligent_analysis: bool = True,
    ):
        self.config = config

        # Initialize TestPy components
        self.testpy_logger = LoggingManager()
        self.testpy_config = TestConfig()
        self.security_config = TestFrameworkSecurityConfig()
        self.security_service = create_security_service(cache_manager)
        self.test_runner = TestRunner(self.testpy_logger, self.testpy_config)

        # Initialize intelligent failure analyzer
        self.intelligent_analyzer: IntelligentFailureAnalyzer | None = None
        if enable_intelligent_analysis:
            try:
                # Provider/model selection now lives server-side in
                # inference-router; client only governs enablement, timeout,
                # and the heuristic-analysis toggles.
                analysis_config = FailureAnalysisConfig(
                    enabled=True,
                    timeout_seconds=30,
                    max_tokens=2000,
                    enable_code_fixes=True,
                    enable_performance_analysis=True,
                    enable_security_analysis=True,
                    cache_results=True,
                    cache_ttl_seconds=3600,
                )
                self.intelligent_analyzer = create_intelligent_failure_analyzer(
                    config=analysis_config,
                    cache_manager=cache_manager,
                )
                logger.info("Intelligent failure analyzer initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize intelligent failure analyzer: {e}")
                self.intelligent_analyzer = None

        # Initialize modular components
        self.config_manager = TestConfigManager(
            security_service=self.security_service,
            testpy_config=self.testpy_config,
            security_config=self.security_config,
        )

        self.result_processor = TestResultProcessor(cache_manager=cache_manager)

        self.container_manager = TestContainerManager(docker_service=docker_service)

        self.test_executor = TestExecutor(
            test_runner=self.test_runner,
            intelligent_analyzer=self.intelligent_analyzer,
        )

        self.monitor = TestMonitor(cache_manager=cache_manager)

    async def execute_test(self, request: TestRequest) -> TestResult:
        """Execute test using modular components."""
        return await self.test_executor.execute_test(
            request=request,
            container_executor=self.container_manager,
            config_manager=self.config_manager,
            result_processor=self.result_processor,
        )

    async def get_test_results(self, test_id: str) -> TestResult:
        """Get cached test results."""
        return await self.monitor.get_test_results(test_id)

    async def cleanup_test_environment(self, test_id: str) -> bool:
        """Cleanup test environment and cached results."""
        return await self.monitor.cleanup_test_environment(test_id, self.container_manager)

    async def health_check(self) -> dict[str, Any]:
        """Health check for the testing service."""
        return await self.monitor.health_check(
            container_manager=self.container_manager,
            config_manager=self.config_manager,
            result_processor=self.result_processor,
            intelligent_analyzer=self.intelligent_analyzer,
        )

    async def analyze_test_failure(
        self,
        test_result: TestResult,
        source_code: str | None = None,
        test_code: str | None = None,
        dependencies: dict[str, str] | None = None,
        environment: dict[str, str] | None = None,
        resource_usage: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Analyze a test failure using intelligent AI analysis."""
        return await self.test_executor.analyze_test_failure(
            test_result=test_result,
            source_code=source_code,
            test_code=test_code,
            dependencies=dependencies,
            environment=environment,
            resource_usage=resource_usage,
        )

    async def batch_analyze_failures(
        self,
        test_results: list[TestResult],
        max_concurrent: int = 5,
    ) -> list[dict[str, Any]]:
        """Analyze multiple test failures concurrently."""
        return await self.test_executor.batch_analyze_failures(
            test_results=test_results,
            max_concurrent=max_concurrent,
        )

    async def get_analysis_summary(self) -> dict[str, Any]:
        """Get summary of analysis service status and metrics."""
        return await self.test_executor.get_analysis_summary()

    # Additional monitoring and management methods
    def track_active_test(self, test_id: str, task) -> None:
        """Track an active test."""
        self.monitor.track_active_test(test_id, task)

    def untrack_test(self, test_id: str):
        """Stop tracking a test."""
        return self.monitor.untrack_test(test_id)

    def get_active_test_ids(self) -> list[str]:
        """Get list of active test IDs."""
        return self.monitor.get_active_test_ids()

    def get_active_test_count(self) -> int:
        """Get count of active tests."""
        return self.monitor.get_active_test_count()

    async def cancel_test(self, test_id: str) -> bool:
        """Cancel an active test."""
        return await self.monitor.cancel_test(test_id)

    async def cleanup_completed_tests(self) -> int:
        """Clean up completed test tasks."""
        return await self.monitor.cleanup_completed_tests()

    async def get_test_status(self, test_id: str) -> dict[str, Any]:
        """Get status information for a specific test."""
        return await self.monitor.get_test_status(test_id)

    async def get_system_metrics(self) -> dict[str, Any]:
        """Get system metrics for monitoring."""
        return await self.monitor.get_system_metrics()


# Factory function for dependency injection
def get_test_execution_service(
    docker_service: DockerService,
    cache_manager: RedisCacheManager | None = None,
    config: Any | None = None,
    enable_intelligent_analysis: bool = True,
) -> TestExecutionService:
    """Factory function to create TestExecutionService."""
    return TestExecutionService(
        docker_service=docker_service,
        cache_manager=cache_manager,
        config=config,
        enable_intelligent_analysis=enable_intelligent_analysis,
    )
