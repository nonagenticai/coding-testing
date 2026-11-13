"""Tests for src/services/testing/test_execution_service.py.

Verifies:
- Service construction wires all modular components with correct types
- Each public method routes through the correct component and returns its output
- Arguments are forwarded faithfully to delegates
- Active test tracking works with real state mutations
- Factory function produces a correctly configured service
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from models.testing_models import (
    TestRequest,
    TestResult,
    TestSecurityLevel,
)
from services.stubs import DockerService, RedisCacheManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def docker_service():
    return AsyncMock(spec=DockerService)


@pytest.fixture
def cache_manager():
    cm = AsyncMock(spec=RedisCacheManager)
    cm.get = AsyncMock(return_value=None)
    cm.set = AsyncMock()
    cm.delete = AsyncMock()
    return cm


@pytest.fixture
def execution_service(docker_service, cache_manager):
    """Create TestExecutionService with mocked dependencies."""
    from services.testing.test_execution_service import TestExecutionService

    return TestExecutionService(
        docker_service=docker_service,
        cache_manager=cache_manager,
        enable_intelligent_analysis=False,
    )


@pytest.fixture
def test_request():
    return TestRequest(
        code="def add(a,b): return a+b",
        test_code="def test_add(): assert add(1,2)==3",
        language="python",
    )


@pytest.fixture
def passing_result():
    return TestResult(
        test_id="t-pass",
        success=True,
        stdout="1 passed",
        stderr="",
        exit_code=0,
        execution_time=0.5,
        created_at=datetime.now(UTC),
        language="python",
        security_level=TestSecurityLevel.HIGH,
    )


@pytest.fixture
def failing_result():
    return TestResult(
        test_id="t-fail",
        success=False,
        stdout="FAIL",
        stderr="AssertionError",
        exit_code=1,
        execution_time=1.0,
        created_at=datetime.now(UTC),
        language="python",
        security_level=TestSecurityLevel.HIGH,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_all_components_wired(self, execution_service):
        """All modular components are created with correct concrete types."""
        from services.testing.test_config_manager import TestConfigManager
        from services.testing.test_container_manager import TestContainerManager
        from services.testing.test_executor import TestExecutor
        from services.testing.test_monitor import TestMonitor
        from services.testing.test_result_processor import TestResultProcessor

        assert isinstance(execution_service.config_manager, TestConfigManager)
        assert isinstance(execution_service.result_processor, TestResultProcessor)
        assert isinstance(execution_service.container_manager, TestContainerManager)
        assert isinstance(execution_service.test_executor, TestExecutor)
        assert isinstance(execution_service.monitor, TestMonitor)

    def test_intelligent_analysis_off(self, execution_service):
        """When disabled, intelligent_analyzer is None."""
        assert execution_service.intelligent_analyzer is None

    def test_intelligent_analysis_on(self, docker_service, cache_manager):
        """Enabling intelligent analysis initializes analyzer (or gracefully fails)."""
        from services.testing.test_execution_service import TestExecutionService

        svc = TestExecutionService(
            docker_service=docker_service,
            cache_manager=cache_manager,
            enable_intelligent_analysis=True,
        )
        # The analyzer may or may not initialize depending on env (API keys),
        # but the service itself must still be fully constructed with all components.
        from services.testing.test_config_manager import TestConfigManager
        from services.testing.test_executor import TestExecutor
        from services.testing.test_monitor import TestMonitor

        assert isinstance(svc.config_manager, TestConfigManager)
        assert isinstance(svc.test_executor, TestExecutor)
        assert isinstance(svc.monitor, TestMonitor)
        # intelligent_analyzer is either an IntelligentFailureAnalyzer or None (graceful failure)
        assert svc.intelligent_analyzer is None or hasattr(
            svc.intelligent_analyzer, "analyze_test_failure"
        )

    def test_security_service_created(self, execution_service):
        """Security service is wired through config_manager with correct type."""
        from services.security_service import SystemSecurityService

        assert isinstance(execution_service.security_service, SystemSecurityService)
        # Also verify it's the same instance used in config_manager
        assert (
            execution_service.config_manager.security_service is execution_service.security_service
        )

    def test_cache_manager_propagated(self, execution_service, cache_manager):
        """Cache manager is propagated to result_processor and monitor."""
        assert execution_service.result_processor.cache_manager is cache_manager
        assert execution_service.monitor.cache_manager is cache_manager

    def test_docker_service_propagated(self, execution_service, docker_service):
        """Docker service is propagated to container_manager."""
        assert execution_service.container_manager.docker_service is docker_service

    def test_test_runner_propagated(self, execution_service):
        """TestRunner is created and passed to test_executor."""
        from processing.test_runner import TestRunner

        assert isinstance(execution_service.test_runner, TestRunner)
        assert execution_service.test_executor.test_runner is execution_service.test_runner

    def test_config_stored(self, docker_service, cache_manager):
        """Config parameter is stored on the service."""
        from services.testing.test_execution_service import TestExecutionService

        custom_config = {"key": "value"}
        svc = TestExecutionService(
            docker_service=docker_service,
            cache_manager=cache_manager,
            config=custom_config,
            enable_intelligent_analysis=False,
        )
        assert svc.config == {"key": "value"}


# ---------------------------------------------------------------------------
# execute_test
# ---------------------------------------------------------------------------


class TestExecuteTest:
    @pytest.mark.asyncio
    async def test_returns_executor_result(self, execution_service, test_request, passing_result):
        """execute_test returns exactly what test_executor.execute_test returns."""
        execution_service.test_executor.execute_test = AsyncMock(return_value=passing_result)
        result = await execution_service.execute_test(test_request)
        assert result is passing_result
        assert result.success is True
        assert result.test_id == "t-pass"
        assert result.stdout == "1 passed"
        assert result.exit_code == 0
        assert result.execution_time == 0.5

    @pytest.mark.asyncio
    async def test_passes_request_and_components(
        self, execution_service, test_request, passing_result
    ):
        """The original request and all component references reach the executor."""
        execution_service.test_executor.execute_test = AsyncMock(return_value=passing_result)
        await execution_service.execute_test(test_request)
        call_kwargs = execution_service.test_executor.execute_test.call_args.kwargs
        assert call_kwargs["request"] is test_request
        assert call_kwargs["container_executor"] is execution_service.container_manager
        assert call_kwargs["config_manager"] is execution_service.config_manager
        assert call_kwargs["result_processor"] is execution_service.result_processor

    @pytest.mark.asyncio
    async def test_failing_result_propagated(self, execution_service, test_request, failing_result):
        """A failing TestResult is returned without modification."""
        execution_service.test_executor.execute_test = AsyncMock(return_value=failing_result)
        result = await execution_service.execute_test(test_request)
        assert result is failing_result
        assert result.success is False
        assert result.test_id == "t-fail"
        assert result.exit_code == 1
        assert result.stderr == "AssertionError"


# ---------------------------------------------------------------------------
# Delegation methods
# ---------------------------------------------------------------------------


class TestDelegation:
    """Verify delegation methods forward arguments and return delegate output exactly."""

    @pytest.mark.asyncio
    async def test_get_test_results(self, execution_service, passing_result):
        execution_service.monitor.get_test_results = AsyncMock(return_value=passing_result)
        result = await execution_service.get_test_results("t-pass")
        assert result is passing_result
        assert result.test_id == "t-pass"
        assert result.success is True
        assert result.stdout == "1 passed"
        execution_service.monitor.get_test_results.assert_called_once_with("t-pass")

    @pytest.mark.asyncio
    async def test_cleanup_test_environment(self, execution_service):
        execution_service.monitor.cleanup_test_environment = AsyncMock(return_value=True)
        result = await execution_service.cleanup_test_environment("t1")
        assert result is True
        # Verify container_manager is forwarded
        execution_service.monitor.cleanup_test_environment.assert_called_once_with(
            "t1", execution_service.container_manager
        )

    @pytest.mark.asyncio
    async def test_cleanup_test_environment_failure(self, execution_service):
        """Cleanup returning False is propagated."""
        execution_service.monitor.cleanup_test_environment = AsyncMock(return_value=False)
        result = await execution_service.cleanup_test_environment("t-bad")
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check(self, execution_service):
        expected = {
            "service": "test_execution",
            "status": "healthy",
            "checks": {"containers": "ok", "cache": "ok"},
        }
        execution_service.monitor.health_check = AsyncMock(return_value=expected)
        result = await execution_service.health_check()
        assert result == expected
        assert result["status"] == "healthy"
        assert result["service"] == "test_execution"
        # Verify all four components are forwarded
        call_kwargs = execution_service.monitor.health_check.call_args.kwargs
        assert call_kwargs["container_manager"] is execution_service.container_manager
        assert call_kwargs["config_manager"] is execution_service.config_manager
        assert call_kwargs["result_processor"] is execution_service.result_processor
        assert call_kwargs["intelligent_analyzer"] is execution_service.intelligent_analyzer

    @pytest.mark.asyncio
    async def test_analyze_test_failure(self, execution_service, failing_result):
        expected = {"root_cause": "logic error", "confidence": 0.9}
        execution_service.test_executor.analyze_test_failure = AsyncMock(return_value=expected)
        result = await execution_service.analyze_test_failure(failing_result)
        assert result == expected
        assert result["root_cause"] == "logic error"
        assert result["confidence"] == 0.9
        # Verify all kwargs are forwarded with defaults
        call_kwargs = execution_service.test_executor.analyze_test_failure.call_args.kwargs
        assert call_kwargs["test_result"] is failing_result
        assert call_kwargs["source_code"] is None
        assert call_kwargs["test_code"] is None
        assert call_kwargs["dependencies"] is None
        assert call_kwargs["environment"] is None
        assert call_kwargs["resource_usage"] is None

    @pytest.mark.asyncio
    async def test_analyze_test_failure_with_context(self, execution_service, failing_result):
        """Optional kwargs are forwarded when provided."""
        expected = {"root_cause": "import error", "confidence": 0.8}
        execution_service.test_executor.analyze_test_failure = AsyncMock(return_value=expected)
        result = await execution_service.analyze_test_failure(
            failing_result,
            source_code="def foo(): pass",
            test_code="def test_foo(): assert False",
            dependencies={"pytest": "8.0"},
            environment={"PYTHONPATH": "/app"},
            resource_usage={"memory_mb": 128},
        )
        assert result == expected
        call_kwargs = execution_service.test_executor.analyze_test_failure.call_args.kwargs
        assert call_kwargs["source_code"] == "def foo(): pass"
        assert call_kwargs["test_code"] == "def test_foo(): assert False"
        assert call_kwargs["dependencies"] == {"pytest": "8.0"}
        assert call_kwargs["environment"] == {"PYTHONPATH": "/app"}
        assert call_kwargs["resource_usage"] == {"memory_mb": 128}

    @pytest.mark.asyncio
    async def test_batch_analyze_failures(self, execution_service, failing_result):
        expected = [{"analysis": "a", "id": 1}, {"analysis": "b", "id": 2}]
        execution_service.test_executor.batch_analyze_failures = AsyncMock(return_value=expected)
        result = await execution_service.batch_analyze_failures([failing_result, failing_result])
        assert result == expected
        assert len(result) == 2
        assert result[0] == {"analysis": "a", "id": 1}
        assert result[1] == {"analysis": "b", "id": 2}
        # Verify forwarded kwargs
        call_kwargs = execution_service.test_executor.batch_analyze_failures.call_args.kwargs
        assert len(call_kwargs["test_results"]) == 2
        assert call_kwargs["test_results"][0] is failing_result
        assert call_kwargs["max_concurrent"] == 5

    @pytest.mark.asyncio
    async def test_batch_analyze_failures_custom_concurrency(
        self, execution_service, failing_result
    ):
        """Custom max_concurrent is forwarded."""
        execution_service.test_executor.batch_analyze_failures = AsyncMock(return_value=[])
        await execution_service.batch_analyze_failures([failing_result], max_concurrent=2)
        call_kwargs = execution_service.test_executor.batch_analyze_failures.call_args.kwargs
        assert call_kwargs["max_concurrent"] == 2

    @pytest.mark.asyncio
    async def test_get_analysis_summary(self, execution_service):
        expected = {"total_analyzed": 10, "avg_confidence": 0.85}
        execution_service.test_executor.get_analysis_summary = AsyncMock(return_value=expected)
        result = await execution_service.get_analysis_summary()
        assert result == expected
        assert result["total_analyzed"] == 10
        assert result["avg_confidence"] == 0.85
        execution_service.test_executor.get_analysis_summary.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_test(self, execution_service):
        execution_service.monitor.cancel_test = AsyncMock(return_value=True)
        result = await execution_service.cancel_test("t1")
        assert result is True
        execution_service.monitor.cancel_test.assert_called_once_with("t1")

    @pytest.mark.asyncio
    async def test_cancel_test_not_found(self, execution_service):
        """Cancelling a non-existent test returns False."""
        execution_service.monitor.cancel_test = AsyncMock(return_value=False)
        result = await execution_service.cancel_test("nonexistent")
        assert result is False
        execution_service.monitor.cancel_test.assert_called_once_with("nonexistent")

    @pytest.mark.asyncio
    async def test_cleanup_completed_tests(self, execution_service):
        execution_service.monitor.cleanup_completed_tests = AsyncMock(return_value=5)
        result = await execution_service.cleanup_completed_tests()
        assert result == 5
        execution_service.monitor.cleanup_completed_tests.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_completed_tests_zero(self, execution_service):
        """Zero completed tests returns 0."""
        execution_service.monitor.cleanup_completed_tests = AsyncMock(return_value=0)
        result = await execution_service.cleanup_completed_tests()
        assert result == 0

    @pytest.mark.asyncio
    async def test_get_test_status(self, execution_service):
        expected = {"status": "running", "test_id": "t1", "elapsed": 2.5}
        execution_service.monitor.get_test_status = AsyncMock(return_value=expected)
        result = await execution_service.get_test_status("t1")
        assert result == expected
        assert result["status"] == "running"
        assert result["test_id"] == "t1"
        assert result["elapsed"] == 2.5
        execution_service.monitor.get_test_status.assert_called_once_with("t1")

    @pytest.mark.asyncio
    async def test_get_system_metrics(self, execution_service):
        expected = {"active_tests": 3, "total_completed": 100}
        execution_service.monitor.get_system_metrics = AsyncMock(return_value=expected)
        result = await execution_service.get_system_metrics()
        assert result == expected
        assert result["active_tests"] == 3
        assert result["total_completed"] == 100
        execution_service.monitor.get_system_metrics.assert_called_once()


# ---------------------------------------------------------------------------
# Active test tracking (uses real state, no mocks)
# ---------------------------------------------------------------------------


class TestActiveTestTracking:
    def test_starts_empty(self, execution_service):
        assert execution_service.get_active_test_ids() == []
        assert execution_service.get_active_test_count() == 0

    def test_track_and_count(self, execution_service):
        task_a = MagicMock()
        task_b = MagicMock()
        execution_service.track_active_test("a", task_a)
        execution_service.track_active_test("b", task_b)
        assert execution_service.get_active_test_count() == 2
        assert set(execution_service.get_active_test_ids()) == {"a", "b"}

    def test_track_preserves_insertion_order(self, execution_service):
        """IDs are returned in insertion order (dict ordering guarantee)."""
        for label in ["x", "y", "z"]:
            execution_service.track_active_test(label, MagicMock())
        assert execution_service.get_active_test_ids() == ["x", "y", "z"]

    def test_untrack_returns_task(self, execution_service):
        task = MagicMock()
        execution_service.track_active_test("x", task)
        returned = execution_service.untrack_test("x")
        assert returned is task
        assert execution_service.get_active_test_count() == 0
        assert execution_service.get_active_test_ids() == []

    def test_untrack_nonexistent_returns_none(self, execution_service):
        assert execution_service.untrack_test("missing") is None

    def test_untrack_leaves_others(self, execution_service):
        """Untracking one test does not affect the others."""
        task_a = MagicMock()
        task_b = MagicMock()
        task_c = MagicMock()
        execution_service.track_active_test("a", task_a)
        execution_service.track_active_test("b", task_b)
        execution_service.track_active_test("c", task_c)
        execution_service.untrack_test("b")
        assert execution_service.get_active_test_count() == 2
        assert set(execution_service.get_active_test_ids()) == {"a", "c"}

    @pytest.mark.parametrize("count", [1, 5, 10])
    def test_track_multiple(self, execution_service, count):
        for i in range(count):
            execution_service.track_active_test(f"t{i}", MagicMock())
        assert execution_service.get_active_test_count() == count
        assert len(execution_service.get_active_test_ids()) == count
        # Verify each expected ID is present
        for i in range(count):
            assert f"t{i}" in execution_service.get_active_test_ids()

    def test_track_overwrites_same_id(self, execution_service):
        """Tracking the same ID again replaces the task."""
        task_old = MagicMock(name="old")
        task_new = MagicMock(name="new")
        execution_service.track_active_test("dup", task_old)
        execution_service.track_active_test("dup", task_new)
        assert execution_service.get_active_test_count() == 1
        returned = execution_service.untrack_test("dup")
        assert returned is task_new


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


class TestFactory:
    def test_creates_service_instance(self, docker_service):
        from services.testing.test_execution_service import (
            TestExecutionService,
            get_test_execution_service,
        )

        svc = get_test_execution_service(
            docker_service=docker_service,
            enable_intelligent_analysis=False,
        )
        assert isinstance(svc, TestExecutionService)

        from services.testing.test_config_manager import TestConfigManager
        from services.testing.test_executor import TestExecutor
        from services.testing.test_monitor import TestMonitor
        from services.testing.test_result_processor import TestResultProcessor

        assert isinstance(svc.config_manager, TestConfigManager)
        assert isinstance(svc.monitor, TestMonitor)
        assert isinstance(svc.test_executor, TestExecutor)
        assert isinstance(svc.result_processor, TestResultProcessor)
        # intelligent_analyzer should be None or valid (no API keys in test)
        assert svc.intelligent_analyzer is None or hasattr(
            svc.intelligent_analyzer, "analyze_test_failure"
        )

    def test_factory_passes_config(self, docker_service, cache_manager):
        from services.testing.test_execution_service import get_test_execution_service

        cfg = {"timeout": 60}
        svc = get_test_execution_service(
            docker_service=docker_service,
            cache_manager=cache_manager,
            config=cfg,
            enable_intelligent_analysis=False,
        )
        assert svc.config == {"timeout": 60}
        assert svc.result_processor.cache_manager is cache_manager
        assert svc.monitor.cache_manager is cache_manager

    def test_factory_default_intelligent_analysis(self, docker_service):
        """Default is enable_intelligent_analysis=True, service still builds."""
        from services.testing.test_execution_service import get_test_execution_service

        svc = get_test_execution_service(docker_service=docker_service)
        # Service must be fully constructed regardless of analyzer init success
        assert svc.get_active_test_count() == 0
        assert svc.get_active_test_ids() == []
