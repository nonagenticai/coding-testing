"""Unit tests for services.ai_test_insights.pattern_recognition module."""

from __future__ import annotations

from unittest.mock import MagicMock

from services.ai_test_insights.config import AITestInsightsConfig, TestPatternType
from services.ai_test_insights.pattern_recognition import (
    TestPatternDetector,
    create_test_pattern_detector,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(**overrides) -> dict:
    defaults = {
        "execution_times": [1.0, 2.0, 3.0],
        "avg_execution_time": 2.0,
        "median_execution_time": 2.0,
        "p95_execution_time": 3.0,
        "p99_execution_time": 3.0,
        "total_tests": 10,
        "overall_success_rate": 0.9,
        "error_messages": [],
        "results_by_language": {},
        "quality_issues": [],
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreatePatternDetector:
    def test_default(self):
        d = create_test_pattern_detector()
        assert isinstance(d, TestPatternDetector)

    def test_custom_config(self):
        cfg = AITestInsightsConfig(slow_test_threshold=1.0)
        d = create_test_pattern_detector(config=cfg)
        assert d.config.slow_test_threshold == 1.0


# ---------------------------------------------------------------------------
# detect_patterns — aggregator
# ---------------------------------------------------------------------------


class TestDetectPatterns:
    async def test_returns_list(self):
        detector = TestPatternDetector()
        data = _make_data()
        result = await detector.detect_patterns(data)
        assert isinstance(result, list)

    async def test_no_patterns_for_healthy_data(self):
        detector = TestPatternDetector()
        data = _make_data(
            avg_execution_time=0.5,
            overall_success_rate=0.95,
            total_tests=5,
        )
        result = await detector.detect_patterns(data)
        # Healthy data — no patterns expected
        assert len(result) == 0


# ---------------------------------------------------------------------------
# _detect_performance_patterns
# ---------------------------------------------------------------------------


class TestPerformancePatterns:
    async def test_slow_execution_pattern(self):
        detector = TestPatternDetector(AITestInsightsConfig(slow_test_threshold=1.0))
        data = _make_data(
            avg_execution_time=5.0,
            execution_times=[5.0, 6.0, 7.0],
        )
        patterns = await detector._detect_performance_patterns(data)
        assert any(p.pattern_type == TestPatternType.PERFORMANCE_PATTERN for p in patterns)
        # Check affected tests populated
        perf = [p for p in patterns if "Slow" in p.description][0]
        assert len(perf.affected_tests) > 0

    async def test_no_slow_pattern_below_threshold(self):
        detector = TestPatternDetector(AITestInsightsConfig(slow_test_threshold=100.0))
        data = _make_data(avg_execution_time=1.0)
        patterns = await detector._detect_performance_patterns(data)
        slow_patterns = [p for p in patterns if "Slow" in p.description]
        assert len(slow_patterns) == 0

    async def test_high_memory_pattern(self):
        detector = TestPatternDetector(AITestInsightsConfig(memory_threshold=256))
        data = _make_data(peak_memory_usage=512)
        patterns = await detector._detect_performance_patterns(data)
        mem_patterns = [p for p in patterns if "memory" in p.description.lower()]
        assert len(mem_patterns) == 1

    async def test_no_memory_pattern_below_threshold(self):
        detector = TestPatternDetector(AITestInsightsConfig(memory_threshold=1024))
        data = _make_data(peak_memory_usage=256)
        patterns = await detector._detect_performance_patterns(data)
        mem_patterns = [p for p in patterns if "memory" in p.description.lower()]
        assert len(mem_patterns) == 0

    async def test_parallel_execution_pattern(self):
        detector = TestPatternDetector()
        mock_result = MagicMock()
        data = _make_data(
            results_by_language={"python": [mock_result], "go": [mock_result]},
            total_tests=100,
        )
        patterns = await detector._detect_performance_patterns(data)
        parallel_patterns = [p for p in patterns if "parallel" in p.description.lower()]
        assert len(parallel_patterns) == 1

    async def test_no_parallel_pattern_single_language(self):
        detector = TestPatternDetector()
        data = _make_data(
            results_by_language={"python": [MagicMock()]},
            total_tests=100,
        )
        patterns = await detector._detect_performance_patterns(data)
        parallel_patterns = [p for p in patterns if "parallel" in p.description.lower()]
        assert len(parallel_patterns) == 0


# ---------------------------------------------------------------------------
# _detect_reliability_patterns
# ---------------------------------------------------------------------------


class TestReliabilityPatterns:
    async def test_flaky_pattern_detected(self):
        detector = TestPatternDetector(AITestInsightsConfig(flaky_test_threshold=0.9))
        data = _make_data(overall_success_rate=0.7, total_tests=50)
        patterns = await detector._detect_reliability_patterns(data)
        flaky = [p for p in patterns if p.pattern_type == TestPatternType.FLAKY_PATTERN]
        assert len(flaky) == 1
        assert flaky[0].impact_score == 8.0

    async def test_no_flaky_pattern_above_threshold(self):
        detector = TestPatternDetector(AITestInsightsConfig(flaky_test_threshold=0.8))
        data = _make_data(overall_success_rate=0.95)
        patterns = await detector._detect_reliability_patterns(data)
        flaky = [p for p in patterns if p.pattern_type == TestPatternType.FLAKY_PATTERN]
        assert len(flaky) == 0

    async def test_error_concentration_pattern(self):
        detector = TestPatternDetector()
        errors = ["Timeout"] * 5 + ["AssertionError"]
        data = _make_data(error_messages=errors)
        patterns = await detector._detect_reliability_patterns(data)
        concentration = [
            p for p in patterns if p.pattern_type == TestPatternType.RELIABILITY_PATTERN
        ]
        assert len(concentration) == 1
        assert "timeout" in concentration[0].description.lower()

    async def test_no_error_concentration_uniform_errors(self):
        detector = TestPatternDetector()
        # 3 errors, none exceeds 30% of 3 = 0.9 → each is 1 which > 0.9
        # Actually each is 1 and 1 > 0.9, so it will trigger.
        # Use 10 errors with 3 types equally: max=4 but 4 > 0.3*10 = 3 ⇒ triggers
        # For NO trigger we need max <= 30% of total
        errors = ["Timeout", "AssertionError", "Connection error", "Other1"]
        data = _make_data(error_messages=errors)
        patterns = await detector._detect_reliability_patterns(data)
        concentration = [
            p for p in patterns if p.pattern_type == TestPatternType.RELIABILITY_PATTERN
        ]
        assert len(concentration) == 0

    async def test_flaky_test_names(self):
        detector = TestPatternDetector(AITestInsightsConfig(flaky_test_threshold=0.9))
        data = _make_data(overall_success_rate=0.5, total_tests=100)
        patterns = await detector._detect_reliability_patterns(data)
        flaky = [p for p in patterns if p.pattern_type == TestPatternType.FLAKY_PATTERN]
        assert len(flaky) == 1
        # count = max(0, int(100 * 0.5 * 0.3)) = 15 → capped at 5
        assert len(flaky[0].affected_tests) == 5


# ---------------------------------------------------------------------------
# _detect_coverage_patterns
# ---------------------------------------------------------------------------


class TestCoveragePatterns:
    async def test_low_coverage_pattern(self):
        detector = TestPatternDetector()
        issues = [MagicMock() for _ in range(15)]
        data = _make_data(quality_issues=issues)
        patterns = await detector._detect_coverage_patterns(data)
        cov = [p for p in patterns if p.pattern_type == TestPatternType.COVERAGE_PATTERN]
        assert len(cov) >= 1

    async def test_no_low_coverage_pattern_few_issues(self):
        detector = TestPatternDetector()
        data = _make_data(quality_issues=[MagicMock()] * 5)
        patterns = await detector._detect_coverage_patterns(data)
        cov = [p for p in patterns if "Low coverage" in p.description]
        assert len(cov) == 0

    async def test_language_imbalance_pattern(self):
        detector = TestPatternDetector()
        data = _make_data(
            results_by_language={
                "python": [MagicMock()] * 30,
                "go": [MagicMock()] * 5,
            },
        )
        patterns = await detector._detect_coverage_patterns(data)
        imbalance = [p for p in patterns if "Uneven" in p.description]
        assert len(imbalance) == 1

    async def test_no_language_imbalance_balanced(self):
        detector = TestPatternDetector()
        data = _make_data(
            results_by_language={
                "python": [MagicMock()] * 10,
                "go": [MagicMock()] * 8,
            },
        )
        patterns = await detector._detect_coverage_patterns(data)
        imbalance = [p for p in patterns if "Uneven" in p.description]
        assert len(imbalance) == 0


# ---------------------------------------------------------------------------
# _detect_security_patterns
# ---------------------------------------------------------------------------


class TestSecurityPatterns:
    async def test_security_pattern_detected(self):
        detector = TestPatternDetector()
        data = _make_data(
            error_messages=["auth token invalid", "Permission denied"],
            total_tests=30,
        )
        patterns = await detector._detect_security_patterns(data)
        sec = [p for p in patterns if p.pattern_type == TestPatternType.SECURITY_PATTERN]
        assert len(sec) == 1

    async def test_no_security_pattern_no_security_errors(self):
        detector = TestPatternDetector()
        data = _make_data(error_messages=["some random error"], total_tests=30)
        patterns = await detector._detect_security_patterns(data)
        sec = [p for p in patterns if p.pattern_type == TestPatternType.SECURITY_PATTERN]
        assert len(sec) == 0

    async def test_no_security_pattern_few_tests(self):
        detector = TestPatternDetector()
        data = _make_data(
            error_messages=["auth failure"],
            total_tests=10,
        )
        patterns = await detector._detect_security_patterns(data)
        sec = [p for p in patterns if p.pattern_type == TestPatternType.SECURITY_PATTERN]
        assert len(sec) == 0


# ---------------------------------------------------------------------------
# _detect_unit_test_patterns
# ---------------------------------------------------------------------------


class TestUnitTestPatterns:
    async def test_high_variance_pattern(self):
        detector = TestPatternDetector()
        data = _make_data(
            execution_times=[0.1, 0.2, 0.1, 0.1, 0.1, 0.1],
            p95_execution_time=100.0,
            median_execution_time=1.0,
        )
        patterns = await detector._detect_unit_test_patterns(data)
        unit = [p for p in patterns if p.pattern_type == TestPatternType.UNIT_PATTERN]
        assert len(unit) == 1

    async def test_no_variance_pattern_low_ratio(self):
        detector = TestPatternDetector()
        data = _make_data(
            execution_times=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            p95_execution_time=5.5,
            median_execution_time=3.5,
        )
        patterns = await detector._detect_unit_test_patterns(data)
        unit = [p for p in patterns if p.pattern_type == TestPatternType.UNIT_PATTERN]
        assert len(unit) == 0

    async def test_no_pattern_too_few_execution_times(self):
        detector = TestPatternDetector()
        data = _make_data(execution_times=[1.0, 2.0])
        patterns = await detector._detect_unit_test_patterns(data)
        assert len(patterns) == 0

    async def test_no_pattern_zero_median(self):
        detector = TestPatternDetector()
        data = _make_data(
            execution_times=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            p95_execution_time=0.0,
            median_execution_time=0.0,
        )
        patterns = await detector._detect_unit_test_patterns(data)
        assert len(patterns) == 0


# ---------------------------------------------------------------------------
# _detect_integration_patterns
# ---------------------------------------------------------------------------


class TestIntegrationPatterns:
    async def test_high_slow_test_ratio(self):
        detector = TestPatternDetector(AITestInsightsConfig(slow_test_threshold=1.0))
        data = _make_data(
            execution_times=[5.0] * 8 + [0.5] * 2,
            total_tests=10,
        )
        patterns = await detector._detect_integration_patterns(data)
        integ = [p for p in patterns if p.pattern_type == TestPatternType.INTEGRATION_PATTERN]
        assert len(integ) == 1

    async def test_no_pattern_low_slow_ratio(self):
        detector = TestPatternDetector(AITestInsightsConfig(slow_test_threshold=10.0))
        data = _make_data(
            execution_times=[1.0] * 10,
            total_tests=10,
        )
        patterns = await detector._detect_integration_patterns(data)
        integ = [p for p in patterns if p.pattern_type == TestPatternType.INTEGRATION_PATTERN]
        assert len(integ) == 0

    async def test_zero_tests(self):
        detector = TestPatternDetector()
        data = _make_data(execution_times=[], total_tests=0)
        patterns = await detector._detect_integration_patterns(data)
        assert len(patterns) == 0


# ---------------------------------------------------------------------------
# _get_slow_tests / _get_flaky_tests
# ---------------------------------------------------------------------------


class TestHelperMethods:
    def test_get_slow_tests(self):
        detector = TestPatternDetector(AITestInsightsConfig(slow_test_threshold=2.0))
        data = _make_data(execution_times=[1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.5])
        result = detector._get_slow_tests(data)
        # 5 slow tests (3,4,5,6,7) → capped at 5
        assert len(result) == 5
        assert all(r.startswith("test_") for r in result)

    def test_get_slow_tests_less_than_5(self):
        detector = TestPatternDetector(AITestInsightsConfig(slow_test_threshold=2.0))
        data = _make_data(execution_times=[1.0, 3.0])
        result = detector._get_slow_tests(data)
        assert len(result) == 1

    def test_get_flaky_tests(self):
        detector = TestPatternDetector()
        result = detector._get_flaky_tests({}, 3)
        assert len(result) == 3
        assert result[0] == "flaky_test_0"

    def test_get_flaky_tests_capped_at_5(self):
        detector = TestPatternDetector()
        result = detector._get_flaky_tests({}, 20)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# _analyze_error_patterns
# ---------------------------------------------------------------------------


class TestAnalyzeErrorPatterns:
    def test_timeout_errors(self):
        detector = TestPatternDetector()
        result = detector._analyze_error_patterns(["Request timeout", "Timeout exceeded"])
        assert result["timeout"] == 2

    def test_assertion_errors(self):
        detector = TestPatternDetector()
        result = detector._analyze_error_patterns(["AssertionError", "assert failed"])
        assert result["assertion"] == 2

    def test_connection_errors(self):
        detector = TestPatternDetector()
        result = detector._analyze_error_patterns(["Connection refused", "Network error"])
        assert result["connection"] == 2

    def test_permission_errors(self):
        detector = TestPatternDetector()
        result = detector._analyze_error_patterns(["Permission denied", "Auth failed"])
        assert result["permission"] == 2

    def test_memory_errors(self):
        detector = TestPatternDetector()
        result = detector._analyze_error_patterns(["Memory error", "OOM killed"])
        assert result["memory"] == 2

    def test_other_errors(self):
        detector = TestPatternDetector()
        result = detector._analyze_error_patterns(["something weird"])
        assert result["other"] == 1

    def test_empty_errors_skipped(self):
        detector = TestPatternDetector()
        result = detector._analyze_error_patterns(["", ""])
        assert len(result) == 0

    def test_mixed_errors(self):
        detector = TestPatternDetector()
        result = detector._analyze_error_patterns(
            ["Timeout", "AssertionError", "Connection error", "blah"]
        )
        assert result["timeout"] == 1
        assert result["assertion"] == 1
        assert result["connection"] == 1
        assert result["other"] == 1
