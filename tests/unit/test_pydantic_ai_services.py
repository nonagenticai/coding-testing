"""Tests for pydantic AI service modules: pydantic_ai_agent, pydantic_ai_agents,
kubernetes_test_executor."""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
    from services.test_execution.pydantic_ai_agents import ExecutionMetrics

from models.pydantic_ai_models import (
    AgentConfiguration,
    PydanticAITestFailureContext,
)
from models.testing_models import SecurityLevel, TestRequest, TestResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent_config(**overrides) -> AgentConfiguration:
    defaults = dict(
        agent_id="test-agent-1",
        ai_model_provider="openai",
        ai_model_name="gpt-4o-mini",
        enabled=True,
    )
    defaults.update(overrides)
    return AgentConfiguration(**defaults)


def _failure_context(**overrides) -> PydanticAITestFailureContext:
    defaults = dict(
        test_id="test-123",
        language="python",
        test_type="unit",
        exit_code=1,
        execution_time_ms=500.0,
        stdout="",
        stderr="AssertionError: expected 5 got 3",
    )
    defaults.update(overrides)
    return PydanticAITestFailureContext(**defaults)


def _test_result(**overrides) -> TestResult:
    defaults = dict(
        test_id="test-abc",
        success=False,
        stdout="test output",
        stderr="AssertionError: assert 1 == 2",
        exit_code=1,
        execution_time=2.5,
        created_at=datetime.now(UTC),
        language="python",
        security_level=SecurityLevel.HIGH,
        coverage_data=None,
        test_metadata=None,
        test_type="unit",
        container_name=None,
        quality_report=None,
        error="AssertionError: assert 1 == 2",
    )
    defaults.update(overrides)
    return TestResult(**defaults)


# pydantic_ai_agents.py — ExecutionPatternAnalyzer, SchedulingOptimizer,
# RecommendationEngine, ServiceTestFailureAnalyzer, PydanticAITestCoordinator
# ===========================================================================


def _execution_metrics(**overrides) -> ExecutionMetrics:
    from services.test_execution.pydantic_ai_agents import ExecutionMetrics

    defaults = dict(
        test_id="test-1",
        language="python",
        test_type="unit",
        execution_time_ms=1000.0,
        exit_code=0,
        success=True,
    )
    defaults.update(overrides)
    return ExecutionMetrics(**defaults)


class TestExecutionMetrics:
    def test_basic_creation(self):
        m = _execution_metrics()
        assert m.test_id == "test-1"
        assert m.success is True

    def test_serialize_datetime(self):
        m = _execution_metrics()
        serialized = m.serialize_datetime(m.timestamp)
        assert serialized == m.timestamp.isoformat()

    def test_serialize_datetime_none(self):
        m = _execution_metrics()
        assert m.serialize_datetime(None) is None


class TestExecutionPattern:
    def test_confidence_validation(self):
        from services.test_execution.pydantic_ai_agents import ExecutionPattern

        pattern = ExecutionPattern(
            pattern_id="p1",
            pattern_type="failure",
            frequency=5,
            severity="high",
            description="test",
            affected_languages=["python"],
            affected_test_types=["unit"],
            first_occurrence=datetime.now(UTC),
            last_occurrence=datetime.now(UTC),
            recommendations=["fix it"],
            confidence_score=0.5,
        )
        assert pattern.confidence_score == 0.5

    def test_invalid_confidence_raises(self):
        from services.test_execution.pydantic_ai_agents import ExecutionPattern

        with pytest.raises(ValueError):
            ExecutionPattern(
                pattern_id="p1",
                pattern_type="failure",
                frequency=5,
                severity="high",
                description="test",
                affected_languages=[],
                affected_test_types=[],
                first_occurrence=datetime.now(UTC),
                last_occurrence=datetime.now(UTC),
                recommendations=[],
                confidence_score=1.5,
            )


class TestPydanticAIRecommendation:
    def test_basic_creation(self):
        from services.test_execution.pydantic_ai_agents import PydanticAIRecommendation

        rec = PydanticAIRecommendation(
            recommendation_id="r1",
            recommendation_type="optimization",
            priority="high",
            title="Test",
            description="desc",
            impact_assessment="high",
            implementation_steps=["step1"],
            confidence_level=0.8,
        )
        assert rec.confidence_level == 0.8


class TestFailureAnalysisModel:
    def test_score_validators(self):
        from services.test_execution.pydantic_ai_agents import FailureAnalysis

        fa = FailureAnalysis(
            analysis_id="a1",
            test_id="t1",
            failure_type="assertion",
            error_message="fail",
            frequency_score=0.5,
            complexity_score=0.3,
            fix_confidence=0.9,
            language="python",
            test_type="unit",
        )
        assert fa.frequency_score == 0.5

    def test_invalid_score_raises(self):
        from services.test_execution.pydantic_ai_agents import FailureAnalysis

        with pytest.raises(ValueError):
            FailureAnalysis(
                analysis_id="a1",
                test_id="t1",
                failure_type="assertion",
                error_message="fail",
                frequency_score=2.0,
                complexity_score=0.3,
                fix_confidence=0.9,
                language="python",
                test_type="unit",
            )


class TestSchedulingContext:
    def test_valid_load(self):
        from services.test_execution.pydantic_ai_agents import SchedulingContext

        ctx = SchedulingContext(
            pending_tests=[],
            active_tests=[],
            available_resources={},
            historical_performance={},
            current_load=0.5,
        )
        assert ctx.current_load == 0.5

    def test_invalid_load_raises(self):
        from services.test_execution.pydantic_ai_agents import SchedulingContext

        with pytest.raises(ValueError):
            SchedulingContext(
                pending_tests=[],
                active_tests=[],
                available_resources={},
                historical_performance={},
                current_load=1.5,
            )


class TestExecutionPatternAnalyzer:
    def _make(self):
        from services.test_execution.pydantic_ai_agents import ExecutionPatternAnalyzer

        return ExecutionPatternAnalyzer()

    def test_calculate_time_range_empty(self):
        analyzer = self._make()
        assert analyzer._calculate_time_range([]) == "No data"

    def test_calculate_time_range_nonempty(self):
        analyzer = self._make()
        metrics = [_execution_metrics(), _execution_metrics()]
        result = analyzer._calculate_time_range(metrics)
        # Format: "<start_iso> to <end_iso> (<duration>)"
        assert " to " in result
        assert "(" in result and ")" in result
        # Both timestamps are similar, so start and end isoformat appear
        assert metrics[0].timestamp.isoformat() in result

    def test_calculate_success_rate_empty(self):
        analyzer = self._make()
        assert analyzer._calculate_success_rate([]) == 0.0

    def test_calculate_success_rate(self):
        analyzer = self._make()
        metrics = [
            _execution_metrics(success=True),
            _execution_metrics(success=False),
        ]
        assert analyzer._calculate_success_rate(metrics) == 0.5

    def test_calculate_average_execution_time_empty(self):
        analyzer = self._make()
        assert analyzer._calculate_average_execution_time([]) == 0.0

    def test_calculate_average_execution_time(self):
        analyzer = self._make()
        metrics = [
            _execution_metrics(execution_time_ms=1000),
            _execution_metrics(execution_time_ms=3000),
        ]
        assert analyzer._calculate_average_execution_time(metrics) == 2000.0

    def test_detect_patterns_empty(self):
        analyzer = self._make()
        assert analyzer._detect_patterns_rule_based([], {}) == []

    def test_detect_slow_tests_pattern(self):
        analyzer = self._make()
        # 5 out of 10 are slow (>5s), that's >20% threshold => "performance" pattern
        metrics = [_execution_metrics(execution_time_ms=100)] * 5 + [
            _execution_metrics(execution_time_ms=10000)
        ] * 5
        patterns = analyzer._detect_patterns_rule_based(metrics, {})
        perf_patterns = [p for p in patterns if p.pattern_type == "performance"]
        assert len(perf_patterns) == 1
        assert perf_patterns[0].frequency == 5
        # 5/10 = 50%, but severity "high" requires >50%, so this is "medium"
        assert perf_patterns[0].severity == "medium"
        assert perf_patterns[0].confidence_score == 0.8

    def test_detect_high_failure_rate(self):
        analyzer = self._make()
        # 3 out of 10 fail (30%) => exceeds 10% threshold => "failure" pattern
        metrics = [_execution_metrics(success=True)] * 7 + [_execution_metrics(success=False)] * 3
        patterns = analyzer._detect_patterns_rule_based(metrics, {})
        failure_patterns = [p for p in patterns if p.pattern_type == "failure"]
        assert len(failure_patterns) == 1
        assert failure_patterns[0].frequency == 3
        # 3/10 = 30%, but severity "high" requires >30%, so this is "medium"
        assert failure_patterns[0].severity == "medium"
        assert failure_patterns[0].confidence_score == 0.9

    def test_detect_high_memory_pattern(self):
        analyzer = self._make()
        # 3 out of 10 use high memory (>512MB), that's 30% > 15% threshold => "resource" pattern
        metrics = [_execution_metrics()] * 7 + [_execution_metrics(memory_usage_mb=1024)] * 3
        patterns = analyzer._detect_patterns_rule_based(metrics, {})
        resource_patterns = [p for p in patterns if p.pattern_type == "resource"]
        assert len(resource_patterns) == 1
        assert resource_patterns[0].frequency == 3
        assert resource_patterns[0].severity == "medium"
        assert resource_patterns[0].confidence_score == 0.7

    @pytest.mark.asyncio
    async def test_analyze_execution_patterns(self):
        analyzer = self._make()
        metrics = [_execution_metrics()] * 5 + [_execution_metrics(success=False)] * 5
        patterns = await analyzer.analyze_execution_patterns(metrics)
        # 5/10 failures (50%) triggers high-severity failure pattern
        types = [p.pattern_type for p in patterns]
        assert "failure" in types
        failure_pattern = next(p for p in patterns if p.pattern_type == "failure")
        assert failure_pattern.frequency == 5
        assert failure_pattern.severity == "high"

    @pytest.mark.asyncio
    async def test_analyze_insufficient_data(self):
        analyzer = self._make()
        metrics = [_execution_metrics()]
        patterns = await analyzer.analyze_execution_patterns(metrics)
        # 1 successful metric: 0% failure, 0% slow, 0% high-memory => no patterns
        assert patterns == []


class TestSchedulingOptimizer:
    def _make(self):
        from services.test_execution.pydantic_ai_agents import SchedulingOptimizer

        return SchedulingOptimizer()

    @pytest.mark.asyncio
    async def test_optimize_empty(self):
        from services.test_execution.pydantic_ai_agents import SchedulingContext

        optimizer = self._make()
        ctx = SchedulingContext(
            pending_tests=[],
            active_tests=[],
            available_resources={},
            historical_performance={},
            current_load=0.0,
        )
        result = await optimizer.optimize_test_schedule(ctx)
        assert result == []

    @pytest.mark.asyncio
    async def test_optimize_sorts_by_time(self):
        from services.test_execution.pydantic_ai_agents import SchedulingContext

        optimizer = self._make()

        # Create test requests with test_id
        req1 = TestRequest(
            code="a",
            test_code="b",
            language="python",
        )
        req1.__dict__["test_id"] = "slow"
        req2 = TestRequest(
            code="a",
            test_code="b",
            language="python",
        )
        req2.__dict__["test_id"] = "fast"

        hist = {
            "slow": _execution_metrics(execution_time_ms=10000),
            "fast": _execution_metrics(execution_time_ms=100),
        }

        ctx = SchedulingContext(
            pending_tests=[req1, req2],
            active_tests=[],
            available_resources={},
            historical_performance=hist,
            current_load=0.0,
        )
        result = await optimizer.optimize_test_schedule(ctx)
        assert result[0] == "fast"


class TestRecommendationEngine:
    def _make(self):
        from services.test_execution.pydantic_ai_agents import RecommendationEngine

        return RecommendationEngine()

    @pytest.mark.asyncio
    async def test_empty_patterns_and_metrics(self):
        engine = self._make()
        recs = await engine.generate_recommendations([], [])
        assert recs == []

    @pytest.mark.asyncio
    async def test_performance_pattern_recommendation(self):
        from services.test_execution.pydantic_ai_agents import ExecutionPattern

        engine = self._make()
        pattern = ExecutionPattern(
            pattern_id="p1",
            pattern_type="performance",
            frequency=10,
            severity="high",
            description="Slow tests",
            affected_languages=["python"],
            affected_test_types=["unit"],
            first_occurrence=datetime.now(UTC),
            last_occurrence=datetime.now(UTC),
            recommendations=["optimize"],
            confidence_score=0.9,
        )
        recs = await engine.generate_recommendations([pattern], [])
        assert len(recs) == 1
        assert recs[0].recommendation_type == "optimization"
        assert recs[0].title == "Address Performance Issues"
        assert recs[0].priority == "high"

    @pytest.mark.asyncio
    async def test_failure_pattern_recommendation(self):
        from services.test_execution.pydantic_ai_agents import ExecutionPattern

        engine = self._make()
        pattern = ExecutionPattern(
            pattern_id="p1",
            pattern_type="failure",
            frequency=10,
            severity="high",
            description="High failures",
            affected_languages=["python"],
            affected_test_types=["unit"],
            first_occurrence=datetime.now(UTC),
            last_occurrence=datetime.now(UTC),
            recommendations=["fix"],
            confidence_score=0.9,
        )
        recs = await engine.generate_recommendations([pattern], [])
        assert len(recs) == 1
        assert recs[0].recommendation_type == "reliability"
        assert recs[0].title == "Improve Test Reliability"
        assert recs[0].priority == "high"
        assert recs[0].confidence_level == 0.85

    @pytest.mark.asyncio
    async def test_resource_pattern_recommendation(self):
        from services.test_execution.pydantic_ai_agents import ExecutionPattern

        engine = self._make()
        pattern = ExecutionPattern(
            pattern_id="p1",
            pattern_type="resource",
            frequency=10,
            severity="medium",
            description="High memory",
            affected_languages=["python"],
            affected_test_types=["unit"],
            first_occurrence=datetime.now(UTC),
            last_occurrence=datetime.now(UTC),
            recommendations=["optimize"],
            confidence_score=0.7,
        )
        recs = await engine.generate_recommendations([pattern], [])
        assert len(recs) == 1
        assert recs[0].recommendation_type == "resource"
        assert recs[0].title == "Optimize Resource Usage"
        assert recs[0].priority == "medium"
        assert recs[0].confidence_level == 0.7

    @pytest.mark.asyncio
    async def test_metrics_based_recommendation(self):
        engine = self._make()
        metrics = [_execution_metrics(success=True)] * 7 + [_execution_metrics(success=False)] * 3
        recs = await engine.generate_recommendations([], metrics)
        # 3/10 = 30% failure rate > 10% threshold => 1 "configuration" recommendation
        assert len(recs) == 1
        assert recs[0].recommendation_type == "configuration"
        assert recs[0].title == "Improve Overall Test Reliability"
        assert "30.0%" in recs[0].description


class TestServiceTestFailureAnalyzer:
    def _make(self):
        from services.test_execution.pydantic_ai_agents import ServiceTestFailureAnalyzer

        return ServiceTestFailureAnalyzer()

    def test_classify_assertion(self):
        analyzer = self._make()
        assert analyzer._classify_failure_type("AssertionError: 1!=2", "") == "assertion"

    def test_classify_timeout(self):
        analyzer = self._make()
        assert analyzer._classify_failure_type("test timed out", "") == "timeout"

    def test_classify_compilation(self):
        analyzer = self._make()
        assert analyzer._classify_failure_type("SyntaxError", "") == "compilation"

    def test_classify_import(self):
        analyzer = self._make()
        assert analyzer._classify_failure_type("ImportError: no module", "") == "import"

    def test_classify_null(self):
        analyzer = self._make()
        assert analyzer._classify_failure_type("NullPointerException", "") == "null_pointer"

    def test_classify_memory(self):
        analyzer = self._make()
        assert analyzer._classify_failure_type("Out of memory error", "") == "memory"

    def test_classify_network(self):
        analyzer = self._make()
        assert analyzer._classify_failure_type("Connection refused", "") == "network"

    def test_classify_runtime(self):
        analyzer = self._make()
        assert analyzer._classify_failure_type("some random error", "") == "runtime"

    def test_generate_fixes(self):
        analyzer = self._make()
        expected = {
            "assertion": [
                "Review test assertions for correctness",
                "Check expected vs actual values",
                "Verify test data setup",
            ],
            "timeout": [
                "Increase timeout limit",
                "Optimize test performance",
                "Check for blocking operations",
            ],
            "compilation": [
                "Fix syntax errors",
                "Check import statements",
                "Verify language version compatibility",
            ],
            "import": [
                "Check module installation",
                "Verify import paths",
                "Check virtual environment setup",
            ],
            "null_pointer": [
                "Add null checks",
                "Initialize variables properly",
                "Check object lifecycle",
            ],
            "memory": [
                "Increase memory limits",
                "Optimize memory usage",
                "Check for memory leaks",
            ],
            "network": [
                "Check network connectivity",
                "Verify service availability",
                "Add retry logic",
            ],
            "runtime": [
                "Review stack trace for clues",
                "Check runtime dependencies",
                "Verify environment setup",
            ],
        }
        for failure_type, expected_fixes in expected.items():
            fixes = analyzer._generate_fixes_for_failure_type(failure_type, "")
            assert fixes == expected_fixes, f"Fixes mismatch for {failure_type}"

    def test_generate_fixes_unknown(self):
        analyzer = self._make()
        fixes = analyzer._generate_fixes_for_failure_type("unknown_type", "")
        assert fixes == ["Manual investigation required"]

    def test_determine_root_cause(self):
        analyzer = self._make()
        expected_causes = {
            "assertion": "Test expectation mismatch",
            "timeout": "Test execution exceeds time limit",
            "compilation": "Code compilation error",
            "import": "Module import failure",
            "null_pointer": "Null reference error",
            "memory": "Memory resource exhaustion",
            "network": "Network connectivity issue",
            "runtime": "Runtime execution error",
        }
        for ft, expected_cause in expected_causes.items():
            cause = analyzer._determine_root_cause(ft, "", "")
            assert cause == expected_cause, f"Root cause mismatch for {ft}"

    def test_calculate_frequency_no_history(self):
        analyzer = self._make()
        result = _test_result()
        assert analyzer._calculate_frequency_score(result, None) == 0.0

    def test_calculate_frequency_with_history(self):
        from services.test_execution.pydantic_ai_agents import FailureAnalysis

        analyzer = self._make()
        result = _test_result()
        history = [
            FailureAnalysis(
                analysis_id="a1",
                test_id="t1",
                failure_type="assertion",
                error_message="fail",
                frequency_score=0.0,
                complexity_score=0.5,
                fix_confidence=0.9,
                language="python",
                test_type="unit",
            )
        ] * 5
        score = analyzer._calculate_frequency_score(result, history)
        assert score == 0.5

    def test_calculate_complexity(self):
        analyzer = self._make()
        expected_scores = {
            "assertion": 0.2,
            "timeout": 0.4,
            "memory": 0.7,
            "runtime": 0.5,
        }
        for ft, expected_score in expected_scores.items():
            score = analyzer._calculate_complexity_score(ft, "")
            assert score == expected_score, f"Complexity score mismatch for {ft}"

    def test_calculate_fix_confidence(self):
        analyzer = self._make()
        expected_scores = {
            "assertion": 0.9,
            "compilation": 0.95,
            "memory": 0.6,
        }
        for ft, expected_score in expected_scores.items():
            score = analyzer._calculate_fix_confidence(ft, "")
            assert score == expected_score, f"Fix confidence mismatch for {ft}"

    def test_find_similar_no_history(self):
        analyzer = self._make()
        result = _test_result()
        assert analyzer._find_similar_failures(result, None) == []

    @pytest.mark.asyncio
    async def test_analyze_test_failure(self):
        analyzer = self._make()
        result = _test_result()
        analysis = await analyzer.analyze_test_failure(result)
        assert analysis.test_id == result.test_id
        assert analysis.language == "python"
        assert analysis.test_type == "unit"
        # stderr = "AssertionError: assert 1 == 2" => failure_type = "assertion"
        assert analysis.failure_type == "assertion"
        assert analysis.root_cause == "Test expectation mismatch"
        assert analysis.suggested_fixes == [
            "Review test assertions for correctness",
            "Check expected vs actual values",
            "Verify test data setup",
        ]
        assert analysis.complexity_score == 0.2
        assert analysis.fix_confidence == 0.9
        # No historical failures => frequency_score = 0.0
        assert analysis.frequency_score == 0.0


class TestPydanticAITestCoordinator:
    def _make(self):
        from services.test_execution.pydantic_ai_agents import PydanticAITestCoordinator

        return PydanticAITestCoordinator()

    @pytest.mark.asyncio
    async def test_record_execution(self):
        coord = self._make()
        await coord.record_execution(_execution_metrics())
        assert len(coord.execution_history) == 1

    @pytest.mark.asyncio
    async def test_record_execution_rolling_window(self):
        coord = self._make()
        for i in range(1100):
            await coord.record_execution(_execution_metrics(test_id=f"t-{i}"))
        assert len(coord.execution_history) == 1000

    @pytest.mark.asyncio
    async def test_analyze_patterns_insufficient_data(self):
        coord = self._make()
        result = await coord.analyze_patterns()
        assert result == []

    @pytest.mark.asyncio
    async def test_analyze_patterns_cached(self):
        coord = self._make()
        coord.patterns_cache = [MagicMock()]
        result = await coord.analyze_patterns()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_analyze_patterns_force_refresh(self):
        coord = self._make()
        for i in range(15):
            await coord.record_execution(_execution_metrics(test_id=f"t-{i}"))
        patterns = await coord.analyze_patterns(force_refresh=True)
        # All 15 metrics are success=True, execution_time_ms=1000, no memory
        # No patterns triggered: 0% slow (>5s), 0% failures, 0% high-memory
        assert patterns == []

    @pytest.mark.asyncio
    async def test_analyze_failure(self):
        coord = self._make()
        result = _test_result()
        analysis = await coord.analyze_failure(result)
        assert analysis.test_id == result.test_id
        assert len(coord.failure_history) == 1

    @pytest.mark.asyncio
    async def test_analyze_failure_rolling_window(self):
        coord = self._make()
        for i in range(550):
            result = _test_result(test_id=f"t-{i}")
            await coord.analyze_failure(result)
        assert len(coord.failure_history) == 500

    @pytest.mark.asyncio
    async def test_generate_recommendations_cached(self):
        coord = self._make()
        coord.recommendations_cache = [MagicMock()]
        recs = await coord.generate_recommendations()
        assert len(recs) == 1

    @pytest.mark.asyncio
    async def test_get_execution_insights(self):
        coord = self._make()
        for i in range(15):
            await coord.record_execution(_execution_metrics(test_id=f"t-{i}", success=(i % 2 == 0)))
        insights = await coord.get_execution_insights()
        summary = insights["summary"]
        assert summary["total_executions"] == 15
        # 8 even indices (0,2,4,6,8,10,12,14) succeed out of 15
        assert summary["success_rate"] == 8 / 15
        assert summary["average_execution_time_ms"] == 1000.0
        assert summary["failure_analyses"] == 0
        assert "patterns" in insights
        assert "recommendations" in insights
        assert "generated_at" in insights

    @pytest.mark.asyncio
    async def test_get_execution_insights_empty(self):
        coord = self._make()
        insights = await coord.get_execution_insights()
        assert insights["summary"]["total_executions"] == 0


class TestCoordinatorExceptionPaths:
    """Cover exception handlers in PydanticAITestCoordinator."""

    def _make(self):
        from services.test_execution.pydantic_ai_agents import PydanticAITestCoordinator

        return PydanticAITestCoordinator()

    @pytest.mark.asyncio
    async def test_analyze_patterns_exception_returns_empty(self):
        """Covers lines 849-851: exception in analyze_patterns."""
        coord = self._make()
        # Fill enough history
        for i in range(15):
            await coord.record_execution(_execution_metrics(test_id=f"t-{i}"))
        # Force exception in pattern_analyzer
        coord.pattern_analyzer.analyze_execution_patterns = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        result = await coord.analyze_patterns(force_refresh=True)
        assert result == []

    @pytest.mark.asyncio
    async def test_optimize_schedule_exception_returns_fifo(self):
        """Covers lines 880-883: exception in optimize_schedule."""
        coord = self._make()
        coord.scheduling_optimizer.optimize_test_schedule = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        req = TestRequest(code="x=1", test_code="pass", language="python")
        req.__dict__["test_id"] = "t-1"
        result = await coord.optimize_schedule(
            pending_tests=[req],
            active_tests=[],
            available_resources={},
            current_load=0.0,
        )
        assert result == ["t-1"]

    @pytest.mark.asyncio
    async def test_optimize_schedule_builds_historical_performance(self):
        """Covers line 865: building historical_performance dict."""
        coord = self._make()
        m = _execution_metrics(test_id="t-rec")
        await coord.record_execution(m)
        coord.scheduling_optimizer.optimize_test_schedule = AsyncMock(return_value=["t-rec"])
        result = await coord.optimize_schedule(
            pending_tests=[], active_tests=[], available_resources={}, current_load=0.0
        )
        assert result == ["t-rec"]

    @pytest.mark.asyncio
    async def test_generate_recommendations_exception_returns_empty(self):
        """Covers lines 904-906: exception in generate_recommendations."""
        coord = self._make()
        coord.recommendation_engine.generate_recommendations = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        # Need enough patterns for analyze_patterns to work
        for i in range(15):
            await coord.record_execution(_execution_metrics(test_id=f"t-{i}"))
        result = await coord.generate_recommendations(force_refresh=True)
        assert result == []

    @pytest.mark.asyncio
    async def test_analyze_failure_exception_returns_fallback(self):
        """Covers lines 933-936: exception in analyze_failure."""
        coord = self._make()
        coord.failure_analyzer.analyze_test_failure = AsyncMock(side_effect=RuntimeError("boom"))
        result = _test_result()
        analysis = await coord.analyze_failure(result)
        assert analysis.failure_type == "system_error"
        assert analysis.root_cause == "AI analysis failed"
        assert analysis.test_id == result.test_id
        assert analysis.language == "python"
        assert analysis.suggested_fixes == ["Check system logs", "Retry analysis"]
        assert analysis.frequency_score == 0.0
        assert analysis.fix_confidence == 0.0

    @pytest.mark.asyncio
    async def test_get_execution_insights_exception_returns_error(self):
        """Covers lines 985-987: exception in get_execution_insights."""
        coord = self._make()
        for i in range(15):
            await coord.record_execution(_execution_metrics(test_id=f"t-{i}"))
        # Force exception in analyze_patterns
        coord.analyze_patterns = AsyncMock(side_effect=RuntimeError("boom"))
        insights = await coord.get_execution_insights()
        assert insights["summary"] == {"error": "Failed to generate insights"}
        assert insights["patterns"] == []
        assert insights["recommendations"] == []
        assert insights["recent_failures"] == []


class TestPatternAnalyzerExceptionPaths:
    """Cover exception handlers in ExecutionPatternAnalyzer."""

    @pytest.mark.asyncio
    async def test_analyze_execution_patterns_exception(self):
        """Covers lines 206-208: exception in analyze_execution_patterns."""
        from services.test_execution.pydantic_ai_agents import ExecutionPatternAnalyzer

        analyzer = ExecutionPatternAnalyzer()
        # Force internal exception
        analyzer._detect_patterns_rule_based = MagicMock(side_effect=RuntimeError("boom"))
        result = await analyzer.analyze_execution_patterns([_execution_metrics()] * 10)
        assert result == []


class TestSchedulingOptimizerExceptionPaths:
    """Cover exception handlers in SchedulingOptimizer."""

    @pytest.mark.asyncio
    async def test_optimize_exception_returns_fifo(self):
        """Covers lines 353-356: exception in optimize_test_schedule."""
        from services.test_execution.pydantic_ai_agents import (
            SchedulingContext,
            SchedulingOptimizer,
        )

        optimizer = SchedulingOptimizer()
        optimizer._optimize_schedule_rule_based = MagicMock(side_effect=RuntimeError("boom"))
        req = TestRequest(code="x=1", test_code="pass", language="python")
        req.__dict__["test_id"] = "t-1"
        ctx = SchedulingContext(
            pending_tests=[req],
            active_tests=[],
            available_resources={},
            historical_performance={},
            current_load=0.0,
        )
        result = await optimizer.optimize_test_schedule(ctx)
        assert result == ["t-1"]


class TestRecommendationEngineExceptionPaths:
    """Cover exception handlers in RecommendationEngine."""

    @pytest.mark.asyncio
    async def test_generate_recommendations_exception(self):
        """Covers lines 430-432: exception in generate_recommendations."""
        from services.test_execution.pydantic_ai_agents import RecommendationEngine

        engine = RecommendationEngine()
        engine._generate_recommendations_rule_based = MagicMock(side_effect=RuntimeError("boom"))
        result = await engine.generate_recommendations([], [])
        assert result == []


class TestServiceTestFailureAnalyzerExceptionPaths:
    """Cover exception handlers in ServiceTestFailureAnalyzer."""

    @pytest.mark.asyncio
    async def test_analyze_failure_exception_returns_fallback(self):
        """Covers lines 586-589: exception in analyze_test_failure."""
        from services.test_execution.pydantic_ai_agents import ServiceTestFailureAnalyzer

        analyzer = ServiceTestFailureAnalyzer()
        analyzer._analyze_failure_rule_based = MagicMock(side_effect=RuntimeError("boom"))
        result = _test_result()
        analysis = await analyzer.analyze_test_failure(result)
        assert analysis.failure_type == "unknown"
        assert analysis.root_cause == "Analysis failed due to system error"
        assert analysis.test_id == result.test_id
        assert analysis.language == "python"
        assert analysis.error_message == "AssertionError: assert 1 == 2"
        assert analysis.suggested_fixes == ["Manual investigation required"]
        assert analysis.frequency_score == 0.0
        assert analysis.fix_confidence == 0.0
        assert analysis.complexity_score == 0.5


class TestValidatorEdgeCases:
    """Cover field_validator rejection paths for pydantic models."""

    def test_execution_pattern_confidence_too_low(self):
        """Covers line 84: confidence_score < 0."""
        from services.test_execution.pydantic_ai_agents import ExecutionPattern

        with pytest.raises(ValueError):
            ExecutionPattern(
                pattern_id="p1",
                pattern_type="test",
                frequency=1,
                severity="low",
                description="x",
                affected_languages=[],
                affected_test_types=[],
                first_occurrence=datetime.now(UTC),
                last_occurrence=datetime.now(UTC),
                recommendations=[],
                confidence_score=-0.1,
            )

    def test_scheduling_context_load_negative(self):
        """Covers line 104: current_load < 0."""
        from services.test_execution.pydantic_ai_agents import SchedulingContext

        with pytest.raises(ValueError):
            SchedulingContext(
                pending_tests=[],
                active_tests=[],
                available_resources={},
                historical_performance={},
                current_load=-0.1,
            )

    def test_recommendation_confidence_too_high(self):
        """Covers line 129: confidence_level > 1."""
        from services.test_execution.pydantic_ai_agents import PydanticAIRecommendation

        with pytest.raises(ValueError):
            PydanticAIRecommendation(
                recommendation_id="r1",
                recommendation_type="optimization",
                priority="high",
                title="T",
                description="d",
                impact_assessment="h",
                implementation_steps=["s"],
                confidence_level=1.5,
            )

    def test_failure_analysis_score_too_high(self):
        """Covers line 156: score > 1."""
        from services.test_execution.pydantic_ai_agents import FailureAnalysis

        with pytest.raises(ValueError):
            FailureAnalysis(
                analysis_id="a1",
                test_id="t1",
                failure_type="assertion",
                error_message="fail",
                frequency_score=0.5,
                complexity_score=1.5,
                fix_confidence=0.5,
                language="python",
                test_type="unit",
            )


class TestTestRecommendationAlias:
    def test_alias(self):
        from services.test_execution.pydantic_ai_agents import (
            PydanticAIRecommendation,
            TestRecommendation,
        )

        assert TestRecommendation is PydanticAIRecommendation


# ===========================================================================
# KubernetesTestExecutor
# ===========================================================================


class TestKubernetesTestExecutor:
    """Tests for KubernetesTestExecutor — kubernetes package fully stubbed."""

    @pytest.fixture(autouse=True)
    def _install_k8s_stubs(self):
        """Install kubernetes module stubs for import."""
        import sys
        import types

        # Create kubernetes module hierarchy
        k8s = types.ModuleType("kubernetes")
        k8s_client = types.ModuleType("kubernetes.client")
        k8s_config = types.ModuleType("kubernetes.config")

        k8s_config.load_kube_config = MagicMock()
        k8s_config.load_incluster_config = MagicMock()

        # API classes
        k8s_client.BatchV1Api = MagicMock(return_value=MagicMock())
        k8s_client.CoreV1Api = MagicMock(return_value=MagicMock())
        k8s_client.ApiException = type("ApiException", (Exception,), {"status": 404})

        # V1 model stubs
        for cls_name in [
            "V1Job",
            "V1ObjectMeta",
            "V1JobSpec",
            "V1PodTemplateSpec",
            "V1PodSpec",
            "V1Container",
            "V1VolumeMount",
            "V1Volume",
            "V1EmptyDirVolumeSource",
            "V1EnvVar",
            "V1ResourceRequirements",
            "V1DeleteOptions",
        ]:
            setattr(k8s_client, cls_name, MagicMock())

        k8s.client = k8s_client
        k8s.config = k8s_config

        saved = {}
        for mod_name in ["kubernetes", "kubernetes.client", "kubernetes.config"]:
            saved[mod_name] = sys.modules.get(mod_name)

        sys.modules["kubernetes"] = k8s
        sys.modules["kubernetes.client"] = k8s_client
        sys.modules["kubernetes.config"] = k8s_config

        # Force reimport
        if "services.test_execution.kubernetes_test_executor" in sys.modules:
            del sys.modules["services.test_execution.kubernetes_test_executor"]

        yield

        # Restore
        for mod_name, original in saved.items():
            if original is None:
                sys.modules.pop(mod_name, None)
            else:
                sys.modules[mod_name] = original
        sys.modules.pop("services.test_execution.kubernetes_test_executor", None)

    def _make(self, **kw):
        from services.test_execution.kubernetes_test_executor import KubernetesTestExecutor

        return KubernetesTestExecutor(**kw)

    def test_init_defaults(self):
        executor = self._make()
        assert executor.namespace == "code-graph-helper"

    def test_init_custom_namespace(self):
        executor = self._make(namespace="test-ns")
        assert executor.namespace == "test-ns"

    def test_init_in_cluster(self):
        executor = self._make(in_cluster=True)
        assert executor.namespace == "code-graph-helper"

    # -- _extension -----------------------------------------------------

    def test_extension_go(self):
        executor = self._make()
        assert executor._extension("go") == ".go"

    def test_extension_rust(self):
        executor = self._make()
        assert executor._extension("rust") == ".rs"

    def test_extension_python(self):
        executor = self._make()
        assert executor._extension("python") == ".py"

    def test_extension_java(self):
        executor = self._make()
        assert executor._extension("java") == ".java"

    def test_extension_csharp(self):
        executor = self._make()
        assert executor._extension("csharp") == ".cs"

    def test_extension_cpp(self):
        executor = self._make()
        assert executor._extension("cpp") == ".cpp"

    def test_extension_javascript(self):
        executor = self._make()
        assert executor._extension("javascript") == ".js"

    def test_extension_typescript(self):
        executor = self._make()
        assert executor._extension("typescript") == ".ts"

    def test_extension_unknown(self):
        executor = self._make()
        assert executor._extension("brainfuck") == ".txt"

    # -- _write_file_cmd ------------------------------------------------

    def test_write_file_cmd(self):
        executor = self._make()
        cmd = executor._write_file_cmd("test.py", "print('hello')")
        assert "base64" in cmd
        assert "test.py" in cmd
        encoded = base64.b64encode(b"print('hello')").decode()
        assert encoded in cmd

    # -- _build_init_script ---------------------------------------------

    def test_build_init_script(self):
        executor = self._make()
        request = MagicMock()
        request.source_code = "def main(): pass"
        request.language.value = "python"
        request.test_files = {"test_main.py": "def test(): pass"}
        with patch.object(executor, "_language_config_files", return_value=[]):
            script = executor._build_init_script(request)
        assert "set -e" in script
        assert "base64" in script

    def test_build_init_script_no_source(self):
        executor = self._make()
        request = MagicMock()
        request.source_code = None
        request.test_files = {"test.py": "pass"}
        with patch.object(executor, "_language_config_files", return_value=[]):
            script = executor._build_init_script(request)
        assert "set -e" in script

    # -- _language_config_files -----------------------------------------

    def test_language_config_go(self):
        executor = self._make()
        request = MagicMock()
        request.language.value = "go"
        request.dependencies = {"github.com/test/pkg": "v1.0.0"}
        files = executor._language_config_files(request)
        assert len(files) == 1
        assert files[0][0] == "go.mod"
        assert "module test-module" in files[0][1]

    def test_language_config_rust(self):
        executor = self._make()
        request = MagicMock()
        request.language.value = "rust"
        request.dependencies = {"serde": "1.0"}
        files = executor._language_config_files(request)
        assert len(files) == 1
        assert files[0][0] == "Cargo.toml"
        assert "[package]" in files[0][1]

    def test_language_config_javascript(self):
        executor = self._make()
        request = MagicMock()
        request.language.value = "javascript"
        request.dependencies = {}
        files = executor._language_config_files(request)
        assert len(files) == 1
        assert files[0][0] == "package.json"

    def test_language_config_typescript(self):
        executor = self._make()
        request = MagicMock()
        request.language.value = "typescript"
        request.dependencies = {}
        files = executor._language_config_files(request)
        assert len(files) == 1
        assert files[0][0] == "package.json"

    def test_language_config_python_no_files(self):
        executor = self._make()
        request = MagicMock()
        request.language.value = "python"
        files = executor._language_config_files(request)
        assert files == []

    # -- _format_go_dependencies ----------------------------------------

    def test_format_go_dependencies_empty(self):
        executor = self._make()
        assert executor._format_go_dependencies({}) == ""

    def test_format_go_dependencies(self):
        executor = self._make()
        deps = {"github.com/pkg/errors": "v0.9.1"}
        result = executor._format_go_dependencies(deps)
        assert "require" in result
        assert "github.com/pkg/errors" in result

    # -- _format_rust_dependencies --------------------------------------

    def test_format_rust_dependencies_empty(self):
        executor = self._make()
        assert executor._format_rust_dependencies({}) == ""

    def test_format_rust_dependencies(self):
        executor = self._make()
        deps = {"serde": "1.0", "tokio": "1.35"}
        result = executor._format_rust_dependencies(deps)
        assert 'serde = "1.0"' in result
        assert 'tokio = "1.35"' in result

    # -- _go_mod_content / _cargo_toml_content --------------------------

    def test_go_mod_content(self):
        executor = self._make()
        content = executor._go_mod_content({"github.com/x/y": "v1.0"})
        assert "module test-module" in content
        assert "go 1.21" in content

    def test_cargo_toml_content(self):
        executor = self._make()
        content = executor._cargo_toml_content({"serde": "1.0"})
        assert "[package]" in content
        assert 'edition = "2021"' in content

    # -- _package_json_content ------------------------------------------

    def test_package_json_javascript(self):
        import json

        executor = self._make()
        request = MagicMock()
        request.language.value = "javascript"
        request.dependencies = {}
        content = executor._package_json_content(request)
        data = json.loads(content)
        assert data["scripts"]["test"] == "jest"

    def test_package_json_typescript(self):
        import json

        executor = self._make()
        request = MagicMock()
        request.language.value = "typescript"
        request.dependencies = {}
        content = executor._package_json_content(request)
        data = json.loads(content)
        assert "ts-jest" in data["scripts"]["test"]
        assert "ts-jest" in data["devDependencies"]
