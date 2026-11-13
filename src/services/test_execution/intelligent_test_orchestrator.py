"""
Intelligent Test Orchestrator with PydanticAI Integration
Advanced test execution orchestration with AI-powered analysis and optimization.
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid7

from pydantic import ConfigDict, Field

from config.test_framework_config import TestConfig
from models.testing_models import (
    TestRequest,
    TestResult,
    TestSecurityLevel,
)
from services.stubs import DockerService
from services.test_execution.pydantic_ai_agents import (
    ExecutionMetrics,
    ExecutionPattern,
    FailureAnalysis,
    PydanticAIRecommendation,
    PydanticAITestCoordinator,
)
from services.test_execution.test_orchestrator import (
    OrchestratorType,
    TestExecutionRequest,
    TestExecutionResult,
    TestOrchestrator,
)

logger = logging.getLogger(__name__)


class IntelligentTestExecutionRequest(TestExecutionRequest):
    """Extended test execution request with AI optimization flags."""

    enable_ai_optimization: bool = Field(default=True)
    enable_pattern_analysis: bool = Field(default=True)
    enable_failure_analysis: bool = Field(default=True)
    enable_scheduling_optimization: bool = Field(default=True)
    priority_level: int = Field(default=5, ge=1, le=10)  # 1=lowest, 10=highest
    expected_execution_time_ms: float | None = None
    resource_weight: float = Field(default=1.0, ge=0.1, le=10.0)

    model_config = ConfigDict(extra="allow")


class IntelligentTestExecutionResult(TestExecutionResult):
    """Extended test execution result with AI analysis."""

    ai_analysis: dict[str, Any] | None = None
    pattern_matches: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    failure_analysis: FailureAnalysis | None = None
    optimization_applied: bool = False
    scheduling_score: float | None = None

    model_config = ConfigDict(extra="allow")


class TestTaskQueue:
    """Intelligent test execution queue with AI-powered scheduling."""

    __test__ = False  # Not a pytest test class

    def __init__(self):
        """Initialize the test execution queue."""
        self.pending_tests: list[IntelligentTestExecutionRequest] = []
        self.active_tests: dict[str, IntelligentTestExecutionRequest] = {}
        self.completed_tests: list[IntelligentTestExecutionResult] = []
        self.failed_tests: list[IntelligentTestExecutionResult] = []
        self._lock = asyncio.Lock()

    async def add_test(self, request: IntelligentTestExecutionRequest) -> None:
        """Add a test to the pending queue."""
        async with self._lock:
            self.pending_tests.append(request)
            logger.debug(f"Added test {request.project_name or 'unknown'} to queue")

    async def get_next_test(
        self, optimized_order: list[str] | None = None
    ) -> IntelligentTestExecutionRequest | None:
        """Get the next test to execute based on optimized order."""
        async with self._lock:
            if not self.pending_tests:
                return None

            if optimized_order:
                # Use AI-optimized order
                for test_id in optimized_order:
                    for i, test in enumerate(self.pending_tests):
                        if (hasattr(test, "test_id") and test.test_id == test_id) or (
                            test.project_name == test_id
                        ):
                            return self.pending_tests.pop(i)

            # Fallback to priority-based selection
            self.pending_tests.sort(key=lambda x: x.priority_level, reverse=True)
            return self.pending_tests.pop(0)

    async def mark_active(self, test_id: str, request: IntelligentTestExecutionRequest) -> None:
        """Mark a test as active."""
        async with self._lock:
            self.active_tests[test_id] = request

    async def mark_completed(self, test_id: str, result: IntelligentTestExecutionResult) -> None:
        """Mark a test as completed."""
        async with self._lock:
            if test_id in self.active_tests:
                del self.active_tests[test_id]

            if result.success:
                self.completed_tests.append(result)
            else:
                self.failed_tests.append(result)

    async def get_queue_status(self) -> dict[str, Any]:
        """Get current queue status."""
        async with self._lock:
            return {
                "pending": len(self.pending_tests),
                "active": len(self.active_tests),
                "completed": len(self.completed_tests),
                "failed": len(self.failed_tests),
                "total_processed": len(self.completed_tests) + len(self.failed_tests),
            }


class IntelligentTestOrchestrator:
    """
    Intelligent test orchestrator with PydanticAI integration for advanced
    test execution analysis, optimization, and coordination.
    """

    def __init__(self, docker_service: DockerService, config: TestConfig):
        """Initialize the intelligent test orchestrator."""
        self.docker_service = docker_service
        self.config = config

        # Initialize base orchestrator
        self.base_orchestrator = TestOrchestrator(docker_service, config)

        # Initialize AI coordinator
        self.ai_coordinator = PydanticAITestCoordinator()

        # Initialize execution queue
        self.execution_queue = TestTaskQueue()

        # Performance tracking
        self.execution_metrics: dict[str, ExecutionMetrics] = {}
        self.resource_usage: dict[str, float] = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "container_count": 0,
        }

        # Configuration
        self.max_concurrent_tests = getattr(config, "max_concurrent_tests", 5)
        self.ai_analysis_threshold = getattr(config, "ai_analysis_threshold", 10)
        self.pattern_analysis_interval = getattr(
            config, "pattern_analysis_interval", 300
        )  # 5 minutes

        # Background tasks
        self._background_tasks: list[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

        logger.info("Intelligent Test Orchestrator initialized with PydanticAI integration")

    async def start(self) -> None:
        """Start the intelligent test orchestrator."""
        logger.info("Starting Intelligent Test Orchestrator")

        # Start background tasks
        self._background_tasks = [
            asyncio.create_task(self._pattern_analysis_loop()),
            asyncio.create_task(self._resource_monitoring_loop()),
            asyncio.create_task(self._queue_optimization_loop()),
        ]

        logger.info("Background analysis tasks started")

    async def stop(self) -> None:
        """Stop the intelligent test orchestrator."""
        logger.info("Stopping Intelligent Test Orchestrator")

        # Signal shutdown
        self._shutdown_event.set()

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self._background_tasks, return_exceptions=True)

        logger.info("Intelligent Test Orchestrator stopped")

    async def execute_test(
        self, request: IntelligentTestExecutionRequest
    ) -> IntelligentTestExecutionResult:
        """Execute a test with intelligent analysis and optimization."""
        test_id = str(uuid7())
        start_time = datetime.now(UTC)

        try:
            logger.info(f"Starting intelligent test execution {test_id}")

            # Add to queue
            await self.execution_queue.add_test(request)

            # Wait for execution slot
            await self._wait_for_execution_slot()

            # Get optimized execution parameters
            optimized_request = await self._optimize_test_request(request)

            # Execute using base orchestrator
            base_result = await self.base_orchestrator.execute_tests(optimized_request)

            # Create execution metrics
            metrics = await self._create_execution_metrics(base_result, request)

            # Record metrics with AI coordinator
            await self.ai_coordinator.record_execution(metrics)

            # Perform AI analysis
            ai_analysis = await self._perform_ai_analysis(base_result, request)

            # Create intelligent result
            result = IntelligentTestExecutionResult(
                **base_result.model_dump(),
                ai_analysis=ai_analysis,
                optimization_applied=True,
                scheduling_score=await self._calculate_scheduling_score(request),
            )

            # Mark as completed
            await self.execution_queue.mark_completed(test_id, result)

            logger.info(f"Intelligent test execution {test_id} completed successfully")
            return result

        except Exception as e:
            logger.error(f"Intelligent test execution {test_id} failed: {e}")

            # Create failure result
            execution_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
            result = IntelligentTestExecutionResult(
                test_id=test_id,
                container_name=request.container_name,
                success=False,
                exit_code=1,
                stdout="",
                stderr=str(e),
                execution_time_ms=execution_time,
                language=request.language,
                test_type=request.test_type,
                created_at=start_time,
                ai_analysis={"error": "AI analysis failed due to execution error"},
            )

            # Mark as failed
            await self.execution_queue.mark_completed(test_id, result)

            return result

    async def batch_execute_tests(
        self, requests: list[IntelligentTestExecutionRequest]
    ) -> list[IntelligentTestExecutionResult]:
        """Execute multiple tests with intelligent batch optimization."""
        logger.info(f"Starting batch execution of {len(requests)} tests")

        # Add all requests to queue
        for request in requests:
            await self.execution_queue.add_test(request)

        # Get AI-optimized execution order
        await self._get_optimized_batch_order(requests)

        # Execute tests in optimized order
        results = []
        semaphore = asyncio.Semaphore(self.max_concurrent_tests)

        async def execute_with_semaphore(
            req: IntelligentTestExecutionRequest,
        ) -> IntelligentTestExecutionResult:
            async with semaphore:
                return await self.execute_test(req)

        # Create tasks for concurrent execution
        tasks = [execute_with_semaphore(req) for req in requests]

        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Test {i} failed with exception: {result}")
                # Create error result
                error_result = IntelligentTestExecutionResult(
                    test_id=f"error_{i}",
                    container_name=requests[i].container_name,
                    success=False,
                    exit_code=1,
                    stdout="",
                    stderr=str(result),
                    execution_time_ms=0,
                    language=requests[i].language,
                    test_type=requests[i].test_type,
                    created_at=datetime.now(UTC),
                    ai_analysis={"error": "Execution failed with exception"},
                )
                processed_results.append(error_result)
            elif isinstance(result, IntelligentTestExecutionResult):
                processed_results.append(result)

        logger.info(f"Batch execution completed: {len(processed_results)} results")
        return processed_results

    async def get_execution_insights(self) -> dict[str, Any]:
        """Get comprehensive execution insights from AI analysis."""
        return await self.ai_coordinator.get_execution_insights()

    async def get_recommendations(self) -> list[PydanticAIRecommendation]:
        """Get AI-generated recommendations for test optimization."""
        return await self.ai_coordinator.generate_recommendations()

    async def get_patterns(self) -> list[ExecutionPattern]:
        """Get identified test execution patterns."""
        return await self.ai_coordinator.analyze_patterns()

    async def get_queue_status(self) -> dict[str, Any]:
        """Get current execution queue status."""
        queue_status = await self.execution_queue.get_queue_status()

        # Add resource usage information
        queue_status.update(
            {
                "resource_usage": self.resource_usage,
                "ai_coordinator_status": {
                    "execution_history_size": len(self.ai_coordinator.execution_history),
                    "failure_history_size": len(self.ai_coordinator.failure_history),
                    "patterns_cached": len(self.ai_coordinator.patterns_cache),
                    "recommendations_cached": len(self.ai_coordinator.recommendations_cache),
                },
            }
        )

        return queue_status

    async def _wait_for_execution_slot(self) -> None:
        """Wait for an available execution slot."""
        while len(self.execution_queue.active_tests) >= self.max_concurrent_tests:
            await asyncio.sleep(0.1)

    async def _optimize_test_request(
        self, request: IntelligentTestExecutionRequest
    ) -> TestExecutionRequest:
        """Optimize test request based on AI analysis."""
        if not request.enable_ai_optimization:
            return TestExecutionRequest(**request.model_dump())

        # Get historical performance data
        similar_tests = [
            metrics
            for metrics in self.ai_coordinator.execution_history
            if metrics.language == request.language.value
            and metrics.test_type == request.test_type.value
        ]

        # Optimize timeout based on historical data
        if similar_tests and not request.expected_execution_time_ms:
            avg_time = sum(m.execution_time_ms for m in similar_tests) / len(similar_tests)
            # Add 50% buffer for safety
            optimized_timeout = max(1, int((avg_time * 1.5) / 1000))
            request.timeout_seconds = min(optimized_timeout, request.timeout_seconds)

        # Optimize resource limits based on patterns
        if request.resource_limits is None:
            request.resource_limits = {}

        # Apply resource optimization based on historical data
        if similar_tests:
            avg_memory = sum(m.memory_usage_mb or 512 for m in similar_tests) / len(similar_tests)
            request.resource_limits["memory"] = f"{int(avg_memory * 1.2)}m"

        return TestExecutionRequest(**request.model_dump())

    async def _create_execution_metrics(
        self, result: TestExecutionResult, request: IntelligentTestExecutionRequest
    ) -> ExecutionMetrics:
        """Create execution metrics from test result."""
        return ExecutionMetrics(
            test_id=result.test_id,
            language=result.language.value,
            test_type=result.test_type.value,
            execution_time_ms=result.execution_time_ms,
            memory_usage_mb=self._extract_memory_usage(result),
            cpu_usage_percent=self._extract_cpu_usage(result),
            exit_code=result.exit_code,
            success=result.success,
            error_count=self._count_errors(result.stderr),
            warning_count=self._count_warnings(result.stderr),
            test_count=len(result.test_results),
            passed_tests=sum(1 for r in result.test_results if r.get("status") == "passed"),
            failed_tests=sum(1 for r in result.test_results if r.get("status") == "failed"),
            skipped_tests=sum(1 for r in result.test_results if r.get("status") == "skipped"),
            coverage_percentage=self._extract_coverage(result.coverage_report),
            container_id=result.container_id,
            timestamp=result.created_at,
        )

    async def _perform_ai_analysis(
        self, result: TestExecutionResult, request: IntelligentTestExecutionRequest
    ) -> dict[str, Any]:
        """Perform AI analysis on test result."""
        analysis: dict[str, Any] = {
            "pattern_analysis": [],
            "recommendations": [],
            "failure_analysis": None,
            "optimization_suggestions": [],
        }

        if not result.success and request.enable_failure_analysis:
            # Create TestResult for failure analysis
            test_result = TestResult(
                test_id=result.test_id,
                success=result.success,
                exit_code=result.exit_code,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time=result.execution_time_ms / 1000.0,
                language=result.language.value,
                security_level=TestSecurityLevel.MEDIUM,
                created_at=result.created_at,
                container_name=result.container_name,
            )

            # Analyze failure
            failure_analysis = await self.ai_coordinator.analyze_failure(test_result)
            analysis["failure_analysis"] = failure_analysis.model_dump()

        if request.enable_pattern_analysis:
            # Get recent patterns
            patterns = await self.ai_coordinator.analyze_patterns()
            analysis["pattern_analysis"] = [p.model_dump() for p in patterns[-5:]]

        # Get recommendations
        recommendations = await self.ai_coordinator.generate_recommendations()
        analysis["recommendations"] = [r.model_dump() for r in recommendations[-3:]]

        return analysis

    async def _calculate_scheduling_score(self, request: IntelligentTestExecutionRequest) -> float:
        """Calculate scheduling score for the test."""
        score = 0.0

        # Base score from priority
        score += request.priority_level * 0.1

        # Bonus for AI optimization enabled
        if request.enable_ai_optimization:
            score += 0.2

        # Bonus for expected execution time
        if request.expected_execution_time_ms:
            # Favor shorter tests
            score += max(0, 1.0 - (request.expected_execution_time_ms / 300000))  # 5 minutes max

        # Resource weight factor
        score *= request.resource_weight

        return min(score, 10.0)

    async def _get_optimized_batch_order(
        self, requests: list[IntelligentTestExecutionRequest]
    ) -> list[str]:
        """Get AI-optimized execution order for batch requests."""
        try:
            # Convert to TestRequest objects for AI analysis
            test_requests = []
            for req in requests:
                test_req = TestRequest(
                    code=req.source_code,
                    test_code=("\n".join(req.test_files.values()) if req.test_files else None),
                    language=req.language.value,
                    container_name=req.container_name,
                    timeout=req.timeout_seconds,
                    security_level=req.security_level,
                )
                test_requests.append(test_req)

            # Get AI-optimized order
            return await self.ai_coordinator.optimize_schedule(
                pending_tests=test_requests,
                active_tests=list(self.execution_queue.active_tests.keys()),
                available_resources=self.resource_usage,
                current_load=self._calculate_current_load(),
            )

        except Exception as e:
            logger.error(f"Failed to get optimized batch order: {e}")
            # Fallback to priority-based order
            return [req.project_name or str(i) for i, req in enumerate(requests)]

    async def _pattern_analysis_loop(self) -> None:
        """Background task for periodic pattern analysis."""
        while not self._shutdown_event.is_set():
            try:
                if len(self.ai_coordinator.execution_history) >= self.ai_analysis_threshold:
                    await self.ai_coordinator.analyze_patterns(force_refresh=True)
                    await self.ai_coordinator.generate_recommendations(force_refresh=True)
                    logger.debug("Completed periodic pattern analysis")

                await asyncio.sleep(self.pattern_analysis_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Pattern analysis loop error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    async def _resource_monitoring_loop(self) -> None:
        """Background task for resource monitoring."""
        while not self._shutdown_event.is_set():
            try:
                # Update resource usage
                self.resource_usage.update(
                    {
                        "cpu_usage": await self._get_cpu_usage(),
                        "memory_usage": await self._get_memory_usage(),
                        "container_count": len(self.execution_queue.active_tests),
                    }
                )

                await asyncio.sleep(30)  # Update every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Resource monitoring loop error: {e}")
                await asyncio.sleep(60)

    async def _queue_optimization_loop(self) -> None:
        """Background task for queue optimization."""
        while not self._shutdown_event.is_set():
            try:
                # Periodically optimize pending test order
                if len(self.execution_queue.pending_tests) > 1:
                    await self._optimize_pending_queue()

                await asyncio.sleep(120)  # Optimize every 2 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue optimization loop error: {e}")
                await asyncio.sleep(60)

    async def _optimize_pending_queue(self) -> None:
        """Optimize the pending test queue order."""
        try:
            # Get optimized order
            optimized_order = await self._get_optimized_batch_order(
                self.execution_queue.pending_tests
            )

            # Reorder pending tests
            reordered_tests = []
            for test_id in optimized_order:
                for test in self.execution_queue.pending_tests:
                    if test.project_name == test_id:
                        reordered_tests.append(test)
                        break

            # Add any remaining tests
            for test in self.execution_queue.pending_tests:
                if test not in reordered_tests:
                    reordered_tests.append(test)

            # Update queue
            async with self.execution_queue._lock:
                self.execution_queue.pending_tests = reordered_tests

            logger.debug(f"Optimized pending queue with {len(reordered_tests)} tests")

        except Exception as e:
            logger.error(f"Failed to optimize pending queue: {e}")

    def _calculate_current_load(self) -> float:
        """Calculate current system load."""
        active_tests = len(self.execution_queue.active_tests)
        max_tests = self.max_concurrent_tests
        return min(active_tests / max_tests, 1.0)

    def _extract_memory_usage(self, result: TestExecutionResult) -> float | None:
        """Extract memory usage from test result."""
        # Implementation would parse container stats
        return None

    def _extract_cpu_usage(self, result: TestExecutionResult) -> float | None:
        """Extract CPU usage from test result."""
        # Implementation would parse container stats
        return None

    def _count_errors(self, stderr: str) -> int:
        """Count errors in stderr output."""
        if not stderr:
            return 0
        error_indicators = ["error:", "failed:"]
        return sum(stderr.lower().count(indicator) for indicator in error_indicators)

    def _count_warnings(self, stderr: str) -> int:
        """Count warnings in stderr output."""
        if not stderr:
            return 0
        warning_indicators = ["warning:", "warn:"]
        return sum(stderr.lower().count(indicator) for indicator in warning_indicators)

    def _extract_coverage(self, coverage_report: dict[str, Any] | None) -> float | None:
        """Extract coverage percentage from coverage report."""
        if not coverage_report:
            return None

        # Basic coverage extraction - would need more sophisticated parsing
        if "coverage" in coverage_report:
            return float(coverage_report["coverage"])

        return None

    async def _get_cpu_usage(self) -> float:
        """Get current CPU usage."""
        # Implementation would use system monitoring
        return 0.0

    async def _get_memory_usage(self) -> float:
        """Get current memory usage."""
        # Implementation would use system monitoring
        return 0.0


def create_intelligent_test_orchestrator(
    docker_service: DockerService, config: TestConfig
) -> IntelligentTestOrchestrator:
    """Factory function to create IntelligentTestOrchestrator."""
    return IntelligentTestOrchestrator(docker_service, config)


# Backward compatibility alias
TestType = OrchestratorType
