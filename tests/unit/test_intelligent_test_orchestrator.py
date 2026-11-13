"""Tests for IntelligentTestOrchestrator in intelligent_test_orchestrator.py."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from config.test_framework_config import TestConfig
from services.stubs import DockerService
from services.test_execution.intelligent_test_orchestrator import (
    IntelligentTestExecutionRequest,
    IntelligentTestExecutionResult,
    IntelligentTestOrchestrator,
    TestTaskQueue,
    create_intelligent_test_orchestrator,
)
from services.test_execution.pydantic_ai_agents import ExecutionMetrics
from services.test_execution.test_orchestrator import (
    OrchestratorType,
    TestExecutionRequest,
    TestLanguage,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def docker_service():
    svc = AsyncMock(spec=DockerService)
    svc.create_test_container = AsyncMock(return_value="container-123")
    svc.start_container = AsyncMock()
    svc.create_file_in_container = AsyncMock(return_value=True)
    svc.exec_in_container = AsyncMock(
        return_value={
            "exit_code": 0,
            "stdout": "ok",
            "stderr": "",
        }
    )
    svc.stop_container = AsyncMock()
    svc.remove_container_by_id = AsyncMock()
    return svc


@pytest.fixture
def config():
    return TestConfig()


@pytest.fixture
def orchestrator(docker_service, config):
    return IntelligentTestOrchestrator(docker_service, config)


@pytest.fixture
def sample_request():
    return IntelligentTestExecutionRequest(
        language=TestLanguage.PYTHON,
        test_type=OrchestratorType.UNIT,
        source_code="def add(a,b): return a+b",
        test_files={"test_main.py": "import pytest\ndef test_add(): assert add(1,2)==3"},
        timeout_seconds=60,
        enable_ai_optimization=True,
        priority_level=5,
    )


# ---------------------------------------------------------------------------
# TestTaskQueue
# ---------------------------------------------------------------------------


class TestTestTaskQueue:
    @pytest.mark.asyncio
    async def test_add_and_get_test(self):
        q = TestTaskQueue()
        req = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="x=1",
            timeout_seconds=60,
            priority_level=5,
        )
        await q.add_test(req)
        assert len(q.pending_tests) == 1
        result = await q.get_next_test()
        # Returned request must be the exact object we added
        assert result is req
        assert result.source_code == "x=1"
        assert result.language == TestLanguage.PYTHON
        assert result.test_type == OrchestratorType.UNIT
        assert result.priority_level == 5
        # Queue should be empty after popping
        assert len(q.pending_tests) == 0

    @pytest.mark.asyncio
    async def test_get_from_empty_queue(self):
        q = TestTaskQueue()
        assert len(q.pending_tests) == 0
        result = await q.get_next_test()
        assert result is None

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        q = TestTaskQueue()
        low = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="low",
            timeout_seconds=60,
            priority_level=1,
        )
        high = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="high",
            timeout_seconds=60,
            priority_level=10,
        )
        await q.add_test(low)
        await q.add_test(high)
        first = await q.get_next_test()
        assert first.source_code == "high"
        assert first.priority_level == 10
        second = await q.get_next_test()
        assert second.source_code == "low"
        assert second.priority_level == 1
        # Queue is now empty
        assert await q.get_next_test() is None

    @pytest.mark.asyncio
    async def test_mark_active(self):
        q = TestTaskQueue()
        req = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="x",
            timeout_seconds=60,
            priority_level=5,
        )
        assert len(q.active_tests) == 0
        await q.mark_active("t1", req)
        assert "t1" in q.active_tests
        assert q.active_tests["t1"] is req
        assert len(q.active_tests) == 1

    @pytest.mark.asyncio
    async def test_mark_completed_success(self):
        q = TestTaskQueue()
        req = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="x",
            timeout_seconds=60,
            priority_level=5,
        )
        await q.mark_active("t1", req)
        assert "t1" in q.active_tests

        result = IntelligentTestExecutionResult(
            test_id="t1",
            success=True,
            exit_code=0,
            stdout="ok",
            stderr="",
            execution_time_ms=100,
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            created_at=datetime.now(UTC),
        )
        await q.mark_completed("t1", result)
        assert "t1" not in q.active_tests
        assert len(q.completed_tests) == 1
        assert q.completed_tests[0].test_id == "t1"
        assert q.completed_tests[0].success is True
        assert len(q.failed_tests) == 0

    @pytest.mark.asyncio
    async def test_mark_completed_failure(self):
        q = TestTaskQueue()
        result = IntelligentTestExecutionResult(
            test_id="t2",
            success=False,
            exit_code=1,
            stdout="",
            stderr="error",
            execution_time_ms=100,
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            created_at=datetime.now(UTC),
        )
        await q.mark_completed("t2", result)
        assert len(q.failed_tests) == 1
        assert q.failed_tests[0].test_id == "t2"
        assert q.failed_tests[0].success is False
        assert len(q.completed_tests) == 0

    @pytest.mark.asyncio
    async def test_get_queue_status(self):
        q = TestTaskQueue()
        status = await q.get_queue_status()
        assert status == {
            "pending": 0,
            "active": 0,
            "completed": 0,
            "failed": 0,
            "total_processed": 0,
        }


# ---------------------------------------------------------------------------
# IntelligentTestOrchestrator init
# ---------------------------------------------------------------------------


class TestOrchestratorInit:
    def test_creates_base_orchestrator(self, orchestrator, docker_service, config):
        from services.test_execution.test_orchestrator import TestOrchestrator

        assert isinstance(orchestrator.base_orchestrator, TestOrchestrator)
        # base_orchestrator should share the same docker_service and config
        assert orchestrator.base_orchestrator.docker_service is docker_service
        assert orchestrator.base_orchestrator.config is config

    def test_creates_ai_coordinator(self, orchestrator):
        from services.test_execution.pydantic_ai_agents import PydanticAITestCoordinator

        assert isinstance(orchestrator.ai_coordinator, PydanticAITestCoordinator)
        assert orchestrator.ai_coordinator.execution_history == []
        assert orchestrator.ai_coordinator.failure_history == []
        assert orchestrator.ai_coordinator.patterns_cache == []
        assert orchestrator.ai_coordinator.recommendations_cache == []

    def test_creates_execution_queue(self, orchestrator):
        assert isinstance(orchestrator.execution_queue, TestTaskQueue)
        assert orchestrator.execution_queue.pending_tests == []
        assert orchestrator.execution_queue.active_tests == {}
        assert orchestrator.execution_queue.completed_tests == []
        assert orchestrator.execution_queue.failed_tests == []

    def test_initial_resource_usage(self, orchestrator):
        assert orchestrator.resource_usage == {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "container_count": 0,
        }
        assert orchestrator.max_concurrent_tests == 5
        assert orchestrator.execution_metrics == {}

    def test_factory_function(self, docker_service, config):
        orch = create_intelligent_test_orchestrator(docker_service, config)
        assert isinstance(orch, IntelligentTestOrchestrator)
        assert orch.docker_service is docker_service
        assert orch.config is config
        # Factory should create all internal components
        assert orch.execution_queue.pending_tests == []
        assert orch.ai_coordinator.execution_history == []


# ---------------------------------------------------------------------------
# execute_test
# ---------------------------------------------------------------------------


class TestExecuteTest:
    @pytest.mark.asyncio
    async def test_successful_execution(self, orchestrator, sample_request):
        result = await orchestrator.execute_test(sample_request)
        assert result.success is True
        assert result.exit_code == 0
        assert result.optimization_applied is True
        assert result.language == TestLanguage.PYTHON
        assert result.test_type == OrchestratorType.UNIT
        assert result.execution_time_ms >= 0
        assert result.test_id  # non-empty UUID
        assert result.created_at is not None
        # ai_analysis must have the four expected keys from _perform_ai_analysis
        ai = result.ai_analysis
        assert isinstance(ai, dict)
        assert isinstance(ai["pattern_analysis"], list)
        assert isinstance(ai["recommendations"], list)
        assert ai["optimization_suggestions"] == []
        # Successful test -> failure_analysis should be None
        assert ai["failure_analysis"] is None

    @pytest.mark.asyncio
    async def test_failed_execution_on_error(self, orchestrator, sample_request, docker_service):
        docker_service.create_test_container.side_effect = RuntimeError("container fail")
        result = await orchestrator.execute_test(sample_request)
        # base_orchestrator.execute_tests catches the exception internally
        # and returns a TestExecutionResult(success=False), so the outer
        # execute_test still follows the success path but with a failed result
        assert result.success is False
        assert result.exit_code == 1
        assert "container fail" in result.stderr
        assert result.language == TestLanguage.PYTHON
        assert result.test_type == OrchestratorType.UNIT
        assert result.optimization_applied is True
        # Since success=False and enable_failure_analysis=True,
        # _perform_ai_analysis produces a failure_analysis dict
        ai = result.ai_analysis
        assert isinstance(ai, dict)
        assert isinstance(ai["failure_analysis"], dict)
        # failure_analysis must contain the FailureAnalysis model fields
        fa = ai["failure_analysis"]
        assert "analysis_id" in fa
        assert "failure_type" in fa
        assert "error_message" in fa
        assert "suggested_fixes" in fa
        assert isinstance(fa["suggested_fixes"], list)
        assert len(fa["suggested_fixes"]) > 0
        # Pattern analysis and recommendations should still be present
        assert isinstance(ai["pattern_analysis"], list)
        assert isinstance(ai["recommendations"], list)

    @pytest.mark.asyncio
    async def test_scheduling_score_populated(self, orchestrator, sample_request):
        result = await orchestrator.execute_test(sample_request)
        # priority_level=5, ai_opt=True: score = 5*0.1 + 0.2 = 0.7
        assert result.scheduling_score == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# _calculate_scheduling_score
# ---------------------------------------------------------------------------


class TestCalculateSchedulingScore:
    @pytest.mark.asyncio
    async def test_base_score_from_priority(self, orchestrator):
        req = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="x=1",
            timeout_seconds=60,
            priority_level=10,
        )
        score = await orchestrator._calculate_scheduling_score(req)
        # priority=10, ai_opt=True(default): 10*0.1 + 0.2 = 1.2, * 1.0 resource_weight
        assert score == pytest.approx(1.2)

    @pytest.mark.asyncio
    async def test_ai_optimization_bonus(self, orchestrator):
        req_with = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="x=1",
            timeout_seconds=60,
            enable_ai_optimization=True,
            priority_level=5,
        )
        req_without = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="x=1",
            timeout_seconds=60,
            enable_ai_optimization=False,
            priority_level=5,
        )
        score_with = await orchestrator._calculate_scheduling_score(req_with)
        score_without = await orchestrator._calculate_scheduling_score(req_without)
        # With: 5*0.1 + 0.2 = 0.7, Without: 5*0.1 = 0.5
        assert score_with == pytest.approx(0.7)
        assert score_without == pytest.approx(0.5)
        assert score_with > score_without

    @pytest.mark.asyncio
    async def test_expected_time_bonus(self, orchestrator):
        req = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="x=1",
            timeout_seconds=60,
            priority_level=5,
            expected_execution_time_ms=1000,  # very short
        )
        score = await orchestrator._calculate_scheduling_score(req)
        # 5*0.1 + 0.2 (ai) + max(0, 1.0 - 1000/300000) = 0.5+0.2+0.9967 = 1.6967
        expected = (0.5 + 0.2 + (1.0 - 1000 / 300000)) * 1.0
        assert score == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_score_capped_at_10(self, orchestrator):
        req = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="x=1",
            timeout_seconds=60,
            priority_level=10,
            resource_weight=10.0,
            expected_execution_time_ms=1,
            enable_ai_optimization=True,
        )
        score = await orchestrator._calculate_scheduling_score(req)
        # Raw: (1.0 + 0.2 + ~1.0) * 10.0 = ~22.0, capped at 10.0
        assert score == 10.0


# ---------------------------------------------------------------------------
# _count_errors / _count_warnings
# ---------------------------------------------------------------------------


class TestErrorWarningCounting:
    def test_count_errors_empty(self, orchestrator):
        assert orchestrator._count_errors("") == 0

    def test_count_errors_multiple(self, orchestrator):
        stderr = "error: something\nERROR: again\nfailed: once"
        count = orchestrator._count_errors(stderr)
        # "error:" appears 2x (case-insensitive), "failed:" appears 1x
        assert count == 3

    def test_count_warnings_empty(self, orchestrator):
        assert orchestrator._count_warnings("") == 0

    def test_count_warnings_multiple(self, orchestrator):
        stderr = "warning: thing\nWARNING: another\nwarn: too"
        count = orchestrator._count_warnings(stderr)
        # "warning:" appears 2x (case-insensitive), "warn:" appears 1x
        assert count == 3


# ---------------------------------------------------------------------------
# _extract_coverage / _extract_memory_usage / _extract_cpu_usage
# ---------------------------------------------------------------------------


class TestExtractMethods:
    def test_extract_coverage_none(self, orchestrator):
        assert orchestrator._extract_coverage(None) is None

    def test_extract_coverage_with_key(self, orchestrator):
        assert orchestrator._extract_coverage({"coverage": 85.5}) == 85.5

    def test_extract_coverage_no_key(self, orchestrator):
        assert orchestrator._extract_coverage({"other": "data"}) is None

    def test_extract_coverage_integer_value(self, orchestrator):
        # Verify it converts int to float via float() call in source
        result = orchestrator._extract_coverage({"coverage": 90})
        assert result == 90.0
        # Confirm the return type is float, not int (source does float())
        assert type(result) is float

    def test_extract_memory_usage_returns_none(self, orchestrator):
        result = MagicMock()
        assert orchestrator._extract_memory_usage(result) is None

    def test_extract_cpu_usage_returns_none(self, orchestrator):
        result = MagicMock()
        assert orchestrator._extract_cpu_usage(result) is None


# ---------------------------------------------------------------------------
# _calculate_current_load
# ---------------------------------------------------------------------------


class TestCalculateCurrentLoad:
    def test_no_active_tests(self, orchestrator):
        assert orchestrator._calculate_current_load() == 0.0

    def test_at_capacity(self, orchestrator):
        # Fill up active tests to max_concurrent_tests (default 5)
        for i in range(orchestrator.max_concurrent_tests):
            orchestrator.execution_queue.active_tests[f"t{i}"] = MagicMock()
        assert orchestrator._calculate_current_load() == 1.0

    def test_half_capacity(self, orchestrator):
        # max_concurrent_tests defaults to 5
        max_tests = orchestrator.max_concurrent_tests
        half = max_tests // 2
        for i in range(half):
            orchestrator.execution_queue.active_tests[f"t{i}"] = MagicMock()
        expected = half / max_tests
        assert orchestrator._calculate_current_load() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _optimize_test_request
# ---------------------------------------------------------------------------


class TestOptimizeTestRequest:
    @pytest.mark.asyncio
    async def test_returns_base_request_when_disabled(self, orchestrator):
        req = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="x=1",
            timeout_seconds=60,
            enable_ai_optimization=False,
            priority_level=5,
        )
        result = await orchestrator._optimize_test_request(req)
        # Returns a plain TestExecutionRequest, preserving all original fields
        assert type(result) is TestExecutionRequest
        assert result.source_code == "x=1"
        assert result.language == TestLanguage.PYTHON
        assert result.timeout_seconds == 60
        assert result.test_type == OrchestratorType.UNIT

    @pytest.mark.asyncio
    async def test_optimizes_with_history(self, orchestrator):
        # Add some history with 5000ms avg execution time
        for _ in range(3):
            orchestrator.ai_coordinator.execution_history.append(
                ExecutionMetrics(
                    test_id="h1",
                    language="python",
                    test_type="unit",
                    execution_time_ms=5000,
                    exit_code=0,
                    success=True,
                )
            )
        req = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="x=1",
            timeout_seconds=300,
            enable_ai_optimization=True,
            priority_level=5,
        )
        result = await orchestrator._optimize_test_request(req)
        # avg_time=5000, optimized_timeout = int(5000*1.5/1000) = 7
        # min(7, 300) = 7
        assert result.timeout_seconds == 7


# ---------------------------------------------------------------------------
# get_queue_status
# ---------------------------------------------------------------------------


class TestGetQueueStatus:
    @pytest.mark.asyncio
    async def test_returns_status_with_resources(self, orchestrator):
        status = await orchestrator.get_queue_status()
        assert status["pending"] == 0
        assert status["active"] == 0
        assert status["completed"] == 0
        assert status["failed"] == 0
        assert status["total_processed"] == 0
        assert status["resource_usage"] == {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "container_count": 0,
        }
        ai_status = status["ai_coordinator_status"]
        assert ai_status["execution_history_size"] == 0
        assert ai_status["failure_history_size"] == 0
        assert ai_status["patterns_cached"] == 0
        assert ai_status["recommendations_cached"] == 0


# ---------------------------------------------------------------------------
# IntelligentTestExecutionRequest fields
# ---------------------------------------------------------------------------


class TestIntelligentTestExecutionRequest:
    def test_defaults(self):
        req = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="x=1",
            timeout_seconds=60,
        )
        assert req.enable_ai_optimization is True
        assert req.enable_pattern_analysis is True
        assert req.enable_failure_analysis is True
        assert req.enable_scheduling_optimization is True
        assert req.priority_level == 5
        assert req.resource_weight == 1.0
        assert req.expected_execution_time_ms is None
        assert req.resource_limits is None

    def test_priority_bounds(self):
        with pytest.raises(ValidationError, match="priority_level"):
            IntelligentTestExecutionRequest(
                language=TestLanguage.PYTHON,
                test_type=OrchestratorType.UNIT,
                source_code="x=1",
                timeout_seconds=60,
                priority_level=0,  # below min (ge=1)
            )

    def test_priority_upper_bound(self):
        with pytest.raises(ValidationError, match="priority_level"):
            IntelligentTestExecutionRequest(
                language=TestLanguage.PYTHON,
                test_type=OrchestratorType.UNIT,
                source_code="x=1",
                timeout_seconds=60,
                priority_level=11,  # above max (le=10)
            )

    def test_resource_weight_bounds(self):
        with pytest.raises(ValidationError, match="resource_weight"):
            IntelligentTestExecutionRequest(
                language=TestLanguage.PYTHON,
                test_type=OrchestratorType.UNIT,
                source_code="x=1",
                timeout_seconds=60,
                resource_weight=0.0,  # below min (ge=0.1)
            )


# ---------------------------------------------------------------------------
# start / stop
# ---------------------------------------------------------------------------


class TestStartStop:
    @pytest.mark.asyncio
    async def test_start_creates_background_tasks(self, orchestrator):
        await orchestrator.start()
        assert len(orchestrator._background_tasks) == 3
        for task in orchestrator._background_tasks:
            assert not task.done()
        # Clean up
        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_tasks(self, orchestrator):
        await orchestrator.start()
        assert not orchestrator._shutdown_event.is_set()
        await orchestrator.stop()
        assert orchestrator._shutdown_event.is_set()
        for task in orchestrator._background_tasks:
            assert task.done()


# ---------------------------------------------------------------------------
# batch_execute_tests
# ---------------------------------------------------------------------------


class TestBatchExecuteTests:
    @pytest.mark.asyncio
    async def test_batch_execute_basic(self, orchestrator):
        req1 = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="x=1",
            timeout_seconds=60,
            priority_level=5,
        )
        req2 = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="y=2",
            timeout_seconds=60,
            priority_level=5,
        )
        results = await orchestrator.batch_execute_tests([req1, req2])
        assert len(results) == 2
        for r in results:
            assert r.success is True
            assert r.exit_code == 0
            assert r.optimization_applied is True
            assert r.language == TestLanguage.PYTHON
            assert r.test_type == OrchestratorType.UNIT
            assert r.scheduling_score is not None
            assert 0.0 <= r.scheduling_score <= 10.0
            # Each result should have ai_analysis with expected keys
            ai = r.ai_analysis
            assert isinstance(ai, dict)
            assert "pattern_analysis" in ai
            assert "recommendations" in ai


# ---------------------------------------------------------------------------
# get_execution_insights / get_recommendations / get_patterns
# ---------------------------------------------------------------------------


class TestInsightsAndRecommendations:
    @pytest.mark.asyncio
    async def test_get_execution_insights(self, orchestrator):
        insights = await orchestrator.get_execution_insights()
        assert insights["summary"]["total_executions"] == 0
        assert insights["summary"]["success_rate"] == 0.0
        assert insights["summary"]["average_execution_time_ms"] == 0.0
        assert insights["summary"]["patterns_identified"] == 0
        assert insights["summary"]["recommendations_available"] == 0
        assert insights["summary"]["failure_analyses"] == 0
        assert insights["patterns"] == []
        assert insights["recommendations"] == []
        assert insights["recent_failures"] == []
        # generated_at should be a valid ISO datetime string
        generated_at = insights["generated_at"]
        assert isinstance(generated_at, str)
        datetime.fromisoformat(generated_at)  # raises ValueError if invalid

    @pytest.mark.asyncio
    async def test_get_recommendations(self, orchestrator):
        recs = await orchestrator.get_recommendations()
        # No history = no recommendations
        assert recs == []

    @pytest.mark.asyncio
    async def test_get_patterns(self, orchestrator):
        patterns = await orchestrator.get_patterns()
        # No history (< 10 entries) = no patterns
        assert patterns == []


# ---------------------------------------------------------------------------
# IntelligentTestExecutionResult fields
# ---------------------------------------------------------------------------


class TestIntelligentTestExecutionResult:
    def test_defaults(self):
        result = IntelligentTestExecutionResult(
            test_id="t1",
            success=True,
            exit_code=0,
            stdout="ok",
            stderr="",
            execution_time_ms=100,
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            created_at=datetime.now(UTC),
        )
        assert result.ai_analysis is None
        assert result.pattern_matches == []
        assert result.recommendations == []
        assert result.failure_analysis is None
        assert result.optimization_applied is False
        assert result.scheduling_score is None
        assert result.test_id == "t1"
        assert result.stdout == "ok"
        assert result.execution_time_ms == 100

    def test_with_ai_analysis(self):
        result = IntelligentTestExecutionResult(
            test_id="t1",
            success=True,
            exit_code=0,
            stdout="ok",
            stderr="",
            execution_time_ms=100,
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            created_at=datetime.now(UTC),
            ai_analysis={"key": "val"},
            optimization_applied=True,
            scheduling_score=7.5,
        )
        assert result.ai_analysis == {"key": "val"}
        assert result.optimization_applied is True
        assert result.scheduling_score == 7.5
        assert result.success is True
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# TestTaskQueue advanced
# ---------------------------------------------------------------------------


class TestTestTaskQueueAdvanced:
    @pytest.mark.asyncio
    async def test_get_next_with_optimized_order_by_project_name(self):
        q = TestTaskQueue()
        req1 = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="first",
            timeout_seconds=60,
            priority_level=1,
            project_name="proj_a",
        )
        req2 = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="second",
            timeout_seconds=60,
            priority_level=10,
            project_name="proj_b",
        )
        await q.add_test(req1)
        await q.add_test(req2)
        # Use optimized order to fetch proj_a first (even though lower priority)
        result = await q.get_next_test(optimized_order=["proj_a", "proj_b"])
        assert result.source_code == "first"
        assert result.project_name == "proj_a"
        # proj_b still in queue
        assert len(q.pending_tests) == 1
        assert q.pending_tests[0].project_name == "proj_b"

    @pytest.mark.asyncio
    async def test_get_queue_status_after_operations(self):
        q = TestTaskQueue()
        req = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="x",
            timeout_seconds=60,
            priority_level=5,
        )
        await q.add_test(req)
        await q.mark_active("t1", req)
        success_result = IntelligentTestExecutionResult(
            test_id="t1",
            success=True,
            exit_code=0,
            stdout="ok",
            stderr="",
            execution_time_ms=100,
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            created_at=datetime.now(UTC),
        )
        await q.mark_completed("t1", success_result)
        status = await q.get_queue_status()
        assert status["pending"] == 1  # add_test added one, never popped
        assert status["active"] == 0  # was removed by mark_completed
        assert status["completed"] == 1
        assert status["failed"] == 0
        assert status["total_processed"] == 1


# ---------------------------------------------------------------------------
# _count_errors / _count_warnings edge cases
# ---------------------------------------------------------------------------


class TestErrorWarningCountingAdvanced:
    def test_count_errors_none_string(self, orchestrator):
        assert orchestrator._count_errors("") == 0

    def test_count_errors_case_insensitive(self, orchestrator):
        stderr = "Error: something"
        count = orchestrator._count_errors(stderr)
        # "error:" matches once (case-insensitive)
        assert count == 1

    def test_count_warnings_single(self, orchestrator):
        stderr = "Warning: deprecation"
        count = orchestrator._count_warnings(stderr)
        # "warning:" matches once (case-insensitive)
        assert count == 1

    def test_count_errors_no_match(self, orchestrator):
        assert orchestrator._count_errors("all good here") == 0

    def test_count_warnings_no_match(self, orchestrator):
        assert orchestrator._count_warnings("all good here") == 0


# ---------------------------------------------------------------------------
# _optimize_test_request edge cases
# ---------------------------------------------------------------------------


class TestOptimizeTestRequestAdvanced:
    @pytest.mark.asyncio
    async def test_optimize_sets_resource_limits(self, orchestrator):
        # Add history to trigger resource optimization
        for _ in range(3):
            orchestrator.ai_coordinator.execution_history.append(
                ExecutionMetrics(
                    test_id="h1",
                    language="python",
                    test_type="unit",
                    execution_time_ms=5000,
                    exit_code=0,
                    success=True,
                    memory_usage_mb=256,
                )
            )
        req = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="x=1",
            timeout_seconds=300,
            enable_ai_optimization=True,
            priority_level=5,
        )
        result = await orchestrator._optimize_test_request(req)
        assert isinstance(result.resource_limits, dict)
        assert len(result.resource_limits) >= 1
        # avg_memory = 256, memory = int(256 * 1.2) = 307
        assert result.resource_limits["memory"] == "307m"
        # Timeout should also be optimized: int(5000*1.5/1000) = 7
        assert result.timeout_seconds == 7


# ---------------------------------------------------------------------------
# Backward compat alias
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_test_type_alias(self):
        from services.test_execution.intelligent_test_orchestrator import TestType

        assert TestType is OrchestratorType
