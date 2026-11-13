"""Tests for analysis_core.py — core analysis functions."""

from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass
class MockResult:
    execution_time_ms: float | None = None
    resource_limited: bool = False
    language: str = "python"
    success: bool = True


class TestIdentifyPerformanceBottlenecks:
    @pytest.fixture
    def _import(self):
        from routers.intelligent_testing_modules.analysis_core import (
            _identify_performance_bottlenecks,
        )

        return _identify_performance_bottlenecks

    async def test_empty_results(self, _import):
        result = await _import([])
        assert result == []

    async def test_no_execution_times(self, _import):
        results = [MockResult(execution_time_ms=None)]
        assert await _import(results) == []

    async def test_slow_tests_detected(self, _import):
        # Need the p95 threshold value to differ from the max so some tests
        # exceed it. 95 fast + 4 medium + 1 very slow:
        # sorted: [100]*95 + [5000]*4 + [50000] → threshold_index=95 → value=5000
        # slow_tests: execution_time_ms > 5000 → the single 50000ms test.
        results = [MockResult(execution_time_ms=100) for _ in range(95)]
        results += [MockResult(execution_time_ms=5000) for _ in range(4)]
        results.append(MockResult(execution_time_ms=50000))
        bottlenecks = await _import(results)
        assert len(bottlenecks) == 1
        entry = bottlenecks[0]
        assert entry["type"] == "slow_tests"
        assert entry["affected_tests"] == 1
        assert entry["threshold"] == 5000
        # 1 slow test out of 100 is < 10%, so severity = "medium"
        assert entry["severity"] == "medium"
        assert "1 tests slower than 5000ms" in entry["description"]

    async def test_slow_tests_high_severity_unreachable(self, _import):
        """The 'high' severity path for slow_tests is unreachable (source bug).

        The threshold is the value at index floor(0.95*N) of the sorted list,
        so at most ~5% of tests can strictly exceed it. But severity='high'
        requires >= 10% of tests to exceed it. This is mathematically impossible.
        We document the bug here and verify the code still runs without error.
        """
        # Best attempt: 91 fast + 5 medium + 4 very slow = 100 total
        # threshold_index=95 → value=5000. slow_tests: >5000 → 4 = 4% < 10% = medium.
        results = [MockResult(execution_time_ms=100) for _ in range(91)]
        results += [MockResult(execution_time_ms=5000) for _ in range(5)]
        results += [MockResult(execution_time_ms=50000) for _ in range(4)]
        bottlenecks = await _import(results)
        assert len(bottlenecks) == 1
        assert bottlenecks[0]["severity"] == "medium"  # "high" is unreachable

    async def test_no_slow_tests_when_all_equal(self, _import):
        """When all execution times are equal, no slow tests should be detected."""
        results = [MockResult(execution_time_ms=500) for _ in range(20)]
        bottlenecks = await _import(results)
        assert bottlenecks == []

    async def test_resource_limited(self, _import):
        results = [MockResult(execution_time_ms=100, resource_limited=True)]
        bottlenecks = await _import(results)
        assert len(bottlenecks) == 1
        entry = bottlenecks[0]
        assert entry["type"] == "resource_limitation"
        assert entry["severity"] == "high"
        assert entry["affected_tests"] == 1
        assert "1 tests limited by resources" in entry["description"]

    async def test_both_slow_and_resource_limited(self, _import):
        """When both slow tests and resource-limited tests exist, both bottlenecks appear."""
        results = [MockResult(execution_time_ms=100) for _ in range(95)]
        results += [MockResult(execution_time_ms=5000) for _ in range(4)]
        results.append(MockResult(execution_time_ms=50000, resource_limited=True))
        # sorted times: [100]*95 + [5000]*4 + [50000] → threshold_index=95, threshold=5000
        # slow_tests: >5000 → 1 (the 50000ms one). resource_limited: 1.
        bottlenecks = await _import(results)
        assert len(bottlenecks) == 2
        types = {b["type"] for b in bottlenecks}
        assert types == {"slow_tests", "resource_limitation"}


class TestGeneratePerformanceRecommendations:
    @pytest.fixture
    def _import(self):
        from routers.intelligent_testing_modules.analysis_core import (
            _generate_performance_recommendations,
        )

        return _generate_performance_recommendations

    async def test_low_success_rate(self, _import):
        recs = await _import({"success_rate": 80}, [])
        assert len(recs) == 1
        rec = recs[0]
        assert rec["type"] == "reliability"
        assert rec["priority"] == "high"
        assert "80.0%" in rec["description"]
        assert rec["action"] == "Review and fix failing tests"

    async def test_success_rate_at_boundary(self, _import):
        """95% is not < 95, so no recommendation."""
        recs = await _import({"success_rate": 95}, [])
        assert recs == []

    async def test_high_avg_time(self, _import):
        recs = await _import({"average_execution_time": 15000}, [])
        assert len(recs) == 1
        rec = recs[0]
        assert rec["type"] == "performance"
        assert rec["priority"] == "medium"
        assert "15000ms" in rec["description"]
        assert "Optimize" in rec["action"]

    async def test_avg_time_at_boundary(self, _import):
        """10000ms is not > 10000, so no recommendation."""
        recs = await _import({"average_execution_time": 10000}, [])
        assert recs == []

    async def test_slow_bottleneck(self, _import):
        bottlenecks = [{"type": "slow_tests", "severity": "medium"}]
        recs = await _import({}, bottlenecks)
        assert len(recs) == 1
        rec = recs[0]
        assert rec["type"] == "optimization"
        assert rec["priority"] == "medium"
        assert rec["description"] == "Optimize slow test execution"

    async def test_slow_bottleneck_high_severity(self, _import):
        bottlenecks = [{"type": "slow_tests", "severity": "high"}]
        recs = await _import({}, bottlenecks)
        assert len(recs) == 1
        assert recs[0]["priority"] == "high"

    async def test_resource_bottleneck(self, _import):
        bottlenecks = [{"type": "resource_limitation", "severity": "high"}]
        recs = await _import({}, bottlenecks)
        assert len(recs) == 1
        rec = recs[0]
        assert rec["type"] == "scaling"
        assert rec["priority"] == "high"
        assert rec["description"] == "Increase resource allocation"
        assert rec["action"] == "Scale up resources or optimize resource usage"

    async def test_empty_input(self, _import):
        recs = await _import({}, [])
        assert recs == []

    async def test_combined_recommendations(self, _import):
        """Low success rate + high avg time + bottleneck yields 3 recommendations."""
        bottlenecks = [{"type": "slow_tests", "severity": "medium"}]
        recs = await _import({"success_rate": 50, "average_execution_time": 20000}, bottlenecks)
        assert len(recs) == 3
        types = [r["type"] for r in recs]
        assert types == ["reliability", "performance", "optimization"]


class TestIdentifyFailurePatterns:
    @pytest.fixture
    def _import(self):
        from routers.intelligent_testing_modules.analysis_core import (
            _identify_failure_patterns,
        )

        return _identify_failure_patterns

    async def test_empty_failures(self, _import):
        patterns = await _import([])
        # Always adds a temporal pattern even with no failures
        assert len(patterns) == 1
        assert patterns[0]["type"] == "temporal"
        assert patterns[0]["pattern"] == "failure_clustering"
        assert patterns[0]["severity"] == "low"

    async def test_language_concentration(self, _import):
        # 5 python failures out of 5 total → 100% > 30% threshold
        results = [MockResult(language="python") for _ in range(5)]
        patterns = await _import(results)
        # Should have language_specific + temporal
        assert len(patterns) == 2
        lang_pattern = next(p for p in patterns if p["type"] == "language_specific")
        assert lang_pattern["occurrences"] == 5
        assert "python" in lang_pattern["pattern"]
        assert lang_pattern["severity"] == "medium"
        temporal_pattern = next(p for p in patterns if p["type"] == "temporal")
        assert temporal_pattern["pattern"] == "failure_clustering"

    async def test_multiple_languages_below_threshold(self, _import):
        """When no single language exceeds 30%, only temporal pattern appears."""
        results = [
            MockResult(language="python"),
            MockResult(language="java"),
            MockResult(language="rust"),
            MockResult(language="go"),
        ]
        # Each language is 1/4 = 25% < 30%
        patterns = await _import(results)
        assert len(patterns) == 1
        assert patterns[0]["type"] == "temporal"

    async def test_multiple_languages_some_above_threshold(self, _import):
        """Two languages above 30% both appear as patterns."""
        results = [MockResult(language="python") for _ in range(4)]
        results += [MockResult(language="java") for _ in range(4)]
        results += [MockResult(language="rust")]
        # python: 4/9=44%, java: 4/9=44%, rust: 1/9=11%
        patterns = await _import(results)
        types = [p["type"] for p in patterns]
        assert types.count("language_specific") == 2
        assert types.count("temporal") == 1


class TestGenerateFailureRecommendations:
    @pytest.fixture
    def _import(self):
        from routers.intelligent_testing_modules.analysis_core import (
            _generate_failure_recommendations,
        )

        return _generate_failure_recommendations

    async def test_timeout_errors(self, _import):
        errors = [{"error_type": "timeout", "occurrences": 5}]
        recs = await _import(errors)
        assert len(recs) == 1
        rec = recs[0]
        assert rec["type"] == "timeout_optimization"
        assert rec["priority"] == "high"
        assert "5 times" in rec["description"]
        assert "timeout" in rec["action"].lower()

    async def test_resource_errors(self, _import):
        errors = [{"error_type": "resource_error", "occurrences": 3}]
        recs = await _import(errors)
        assert len(recs) == 1
        rec = recs[0]
        assert rec["type"] == "resource_management"
        assert rec["priority"] == "high"
        assert "3 times" in rec["description"]

    async def test_dependency_errors(self, _import):
        errors = [{"error_type": "dependency_error", "occurrences": 2}]
        recs = await _import(errors)
        assert len(recs) == 1
        rec = recs[0]
        assert rec["type"] == "dependency_management"
        assert rec["priority"] == "medium"
        assert "2 times" in rec["description"]

    async def test_other_errors(self, _import):
        errors = [{"error_type": "assertion_error", "occurrences": 4}]
        recs = await _import(errors)
        assert len(recs) == 1
        rec = recs[0]
        assert rec["type"] == "general_reliability"
        assert rec["priority"] == "medium"
        assert "assertion_error" in rec["description"]
        assert "4 times" in rec["description"]
        assert "assertion_error" in rec["action"]

    async def test_empty_errors(self, _import):
        recs = await _import([])
        assert recs == []

    async def test_multiple_error_types(self, _import):
        errors = [
            {"error_type": "timeout", "occurrences": 3},
            {"error_type": "resource_error", "occurrences": 2},
            {"error_type": "syntax_error", "occurrences": 1},
        ]
        recs = await _import(errors)
        assert len(recs) == 3
        assert recs[0]["type"] == "timeout_optimization"
        assert recs[1]["type"] == "resource_management"
        assert recs[2]["type"] == "general_reliability"


class TestGenerateTrendRecommendations:
    @pytest.fixture
    def _import(self):
        from routers.intelligent_testing_modules.analysis_core import (
            _generate_trend_recommendations,
        )

        return _generate_trend_recommendations

    async def test_high_volume(self, _import):
        data = {"execution_trends": {"total_executions": 2000}}
        recs = await _import(data)
        assert len(recs) == 1
        rec = recs[0]
        assert rec["type"] == "scaling"
        assert rec["priority"] == "medium"
        assert rec["description"] == "High test execution volume detected"

    async def test_volume_at_boundary(self, _import):
        """1000 is not > 1000, so no recommendation."""
        data = {"execution_trends": {"total_executions": 1000}}
        recs = await _import(data)
        assert recs == []

    async def test_peak_hours(self, _import):
        data = {"peak_hours": [9, 10, 14]}
        recs = await _import(data)
        assert len(recs) == 1
        rec = recs[0]
        assert rec["type"] == "scheduling"
        assert rec["priority"] == "low"
        assert "[9, 10, 14]" in rec["description"]

    async def test_empty_peak_hours(self, _import):
        """Empty peak_hours list should not generate scheduling recommendation."""
        data = {"peak_hours": []}
        recs = await _import(data)
        assert recs == []

    async def test_empty_data(self, _import):
        recs = await _import({})
        assert recs == []

    async def test_combined_volume_and_peaks(self, _import):
        data = {
            "execution_trends": {"total_executions": 5000},
            "peak_hours": [10, 11],
        }
        recs = await _import(data)
        assert len(recs) == 2
        assert recs[0]["type"] == "scaling"
        assert recs[1]["type"] == "scheduling"


class TestExceptionHandlers:
    """Cover exception handler paths (lines 73-74, 131-132, 175-176, 229-230, 266-267)."""

    async def test_bottleneck_exception_returns_empty(self):
        """Covers lines 73-74: exception in _identify_performance_bottlenecks."""
        from routers.intelligent_testing_modules.analysis_core import (
            _identify_performance_bottlenecks,
        )

        # A result where execution_time_ms is not comparable (string) triggers exception
        bad_result = MockResult(execution_time_ms="not_a_number")  # type: ignore[arg-type]
        result = await _identify_performance_bottlenecks([bad_result])
        assert result == []

    async def test_perf_recommendations_exception_returns_empty(self):
        """Covers lines 131-132: exception in _generate_performance_recommendations."""
        from routers.intelligent_testing_modules.analysis_core import (
            _generate_performance_recommendations,
        )

        # success_rate="bad" causes TypeError on `"bad" < 95`
        recs = await _generate_performance_recommendations(
            {"success_rate": "bad"},  # type: ignore[dict-item]
            [{"type": "slow_tests", "severity": "medium"}],
        )
        assert recs == []

    async def test_failure_patterns_exception_returns_empty(self):
        """Covers lines 175-176: exception in _identify_failure_patterns."""
        from routers.intelligent_testing_modules.analysis_core import (
            _identify_failure_patterns,
        )

        class BadResult:
            @property
            def language(self):
                raise RuntimeError("boom")

        result = await _identify_failure_patterns([BadResult()])  # type: ignore[arg-type]
        assert result == []

    async def test_failure_recommendations_exception_returns_empty(self):
        """Covers lines 229-230: exception in _generate_failure_recommendations."""
        from routers.intelligent_testing_modules.analysis_core import (
            _generate_failure_recommendations,
        )

        # Missing "error_type" key triggers KeyError
        recs = await _generate_failure_recommendations([{"missing_error_type": True}])
        assert recs == []

    async def test_trend_recommendations_exception_returns_empty(self):
        """Covers lines 266-267: exception in _generate_trend_recommendations."""
        from routers.intelligent_testing_modules.analysis_core import (
            _generate_trend_recommendations,
        )

        # "not_a_dict" has no .get() method → AttributeError → caught by except
        recs = await _generate_trend_recommendations({"execution_trends": "not_a_dict"})
        assert recs == []
