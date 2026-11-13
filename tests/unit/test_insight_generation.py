"""Unit tests for services.ai_test_insights.insight_generation module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from models.testing_models import QualityIssue
from services.ai_test_insights.config import AITestInsightsConfig, TestInsightType
from services.ai_test_insights.insight_generation import (
    TestInsightGenerator,
    create_test_insight_generator,
)
from services.ai_test_insights.models import (
    TestCoverageInsights,
    TestPatternInsight,
    TestPerformanceMetrics,
    TestQualityInsights,
    TestReliabilityMetrics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(**overrides) -> dict:
    """Build a minimal analysis data dict with sensible defaults."""
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
        "quality_scores": [],
        "quality_issues": [],
    }
    defaults.update(overrides)
    return defaults


def _make_quality_issue(severity="warning", category=None) -> QualityIssue:
    kwargs = {
        "tool": "test_tool",
        "severity": severity,
        "message": "issue message",
        "file_path": "file.py",
    }
    return QualityIssue(**kwargs)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateInsightGenerator:
    def test_default_config(self):
        gen = create_test_insight_generator()
        assert gen.config.slow_test_threshold == 10.0
        assert gen.config.coverage_threshold == 80
        assert gen.config.enable_ai_features is True

    def test_custom_config(self):
        cfg = AITestInsightsConfig(slow_test_threshold=5.0)
        gen = create_test_insight_generator(config=cfg)
        assert gen.config.slow_test_threshold == 5.0


# ---------------------------------------------------------------------------
# analyze_performance
# ---------------------------------------------------------------------------


class TestAnalyzePerformance:
    async def test_basic_performance(self):
        gen = TestInsightGenerator()
        data = _make_data()
        result = await gen.analyze_performance(data)
        assert result.average_execution_time == 2.0
        assert result.median_execution_time == 2.0
        assert result.p95_execution_time == 3.0
        assert result.p99_execution_time == 3.0
        assert result.slow_tests_count == 0  # all below default 10.0 threshold
        assert result.fast_tests_count == 10
        assert result.performance_trend == "improving"  # 2.0 < 10.0 * 0.5

    async def test_slow_tests_counted(self):
        gen = TestInsightGenerator(AITestInsightsConfig(slow_test_threshold=1.5))
        data = _make_data(execution_times=[1.0, 2.0, 3.0])
        result = await gen.analyze_performance(data)
        # 2.0 and 3.0 are above threshold 1.5
        assert result.slow_tests_count == 2
        assert result.fast_tests_count == 10 - 2

    async def test_no_slow_tests(self):
        gen = TestInsightGenerator(AITestInsightsConfig(slow_test_threshold=100.0))
        data = _make_data()
        result = await gen.analyze_performance(data)
        assert result.slow_tests_count == 0

    async def test_performance_trend_degrading(self):
        gen = TestInsightGenerator(AITestInsightsConfig(slow_test_threshold=5.0))
        data = _make_data(avg_execution_time=15.0)  # > 5*2
        result = await gen.analyze_performance(data)
        assert result.performance_trend == "degrading"

    async def test_performance_trend_improving(self):
        gen = TestInsightGenerator(AITestInsightsConfig(slow_test_threshold=10.0))
        data = _make_data(avg_execution_time=3.0)  # < 10*0.5
        result = await gen.analyze_performance(data)
        assert result.performance_trend == "improving"

    async def test_performance_trend_stable(self):
        gen = TestInsightGenerator(AITestInsightsConfig(slow_test_threshold=10.0))
        data = _make_data(avg_execution_time=7.0)  # between 5 and 20
        result = await gen.analyze_performance(data)
        assert result.performance_trend == "stable"

    async def test_bottleneck_execution_time(self):
        gen = TestInsightGenerator(AITestInsightsConfig(slow_test_threshold=1.0))
        data = _make_data(avg_execution_time=5.0)
        result = await gen.analyze_performance(data)
        assert result.bottleneck_categories[0] == "execution_time"

    async def test_bottleneck_multi_language(self):
        gen = TestInsightGenerator()
        mock_result = MagicMock()
        data = _make_data(
            results_by_language={"python": [mock_result], "go": [mock_result]},
            avg_execution_time=0.1,
        )
        result = await gen.analyze_performance(data)
        assert "multi_language_overhead" in result.bottleneck_categories

    async def test_bottleneck_inconsistent_performance(self):
        gen = TestInsightGenerator()
        # Variance >> mean ⇒ inconsistent
        data = _make_data(
            execution_times=[0.1, 100.0],
            avg_execution_time=50.0,
        )
        result = await gen.analyze_performance(data)
        assert "inconsistent_performance" in result.bottleneck_categories

    async def test_memory_and_cpu_estimates(self):
        gen = TestInsightGenerator()
        data = _make_data(total_tests=5, avg_execution_time=2.0)
        result = await gen.analyze_performance(data)
        # 128 + 5*2 = 138
        assert result.average_memory_usage == 138.0
        assert result.peak_memory_usage == 138.0 * 1.5
        # cpu: 25 + min(2*5,50) = 35
        assert result.average_cpu_usage == 35.0


# ---------------------------------------------------------------------------
# analyze_reliability
# ---------------------------------------------------------------------------


class TestAnalyzeReliability:
    async def test_basic_reliability(self):
        gen = TestInsightGenerator()
        data = _make_data(overall_success_rate=0.95)
        result = await gen.analyze_reliability(data)
        assert result.overall_success_rate == 0.95
        assert result.flaky_tests_count == 0  # int(10 * 0.05 * 0.3) = int(0.15) = 0
        assert result.reliability_trend == "stable"  # 0.95 >= 0.95
        assert result.error_categories == {}
        assert result.common_failure_patterns == []

    async def test_flaky_tests_estimated(self):
        gen = TestInsightGenerator()
        data = _make_data(overall_success_rate=0.5, total_tests=100)
        result = await gen.analyze_reliability(data)
        # 100 * 0.5 * 0.3 = 15
        assert result.flaky_tests_count == 15

    async def test_no_flaky_tests(self):
        gen = TestInsightGenerator()
        data = _make_data(overall_success_rate=1.0, total_tests=10)
        result = await gen.analyze_reliability(data)
        assert result.flaky_tests_count == 0

    async def test_error_categorization_timeout(self):
        gen = TestInsightGenerator()
        data = _make_data(error_messages=["Timeout waiting for response", "timeout error"])
        result = await gen.analyze_reliability(data)
        assert "timeout" in result.error_categories

    async def test_error_categorization_assertion(self):
        gen = TestInsightGenerator()
        data = _make_data(error_messages=["AssertionError: 1 != 2"])
        result = await gen.analyze_reliability(data)
        assert "assertion" in result.error_categories

    async def test_error_categorization_connection(self):
        gen = TestInsightGenerator()
        data = _make_data(error_messages=["Connection refused"])
        result = await gen.analyze_reliability(data)
        assert "connection" in result.error_categories

    async def test_error_categorization_permission(self):
        gen = TestInsightGenerator()
        data = _make_data(error_messages=["Permission denied", "Auth failure"])
        result = await gen.analyze_reliability(data)
        assert "permission" in result.error_categories

    async def test_error_categorization_memory(self):
        gen = TestInsightGenerator()
        data = _make_data(error_messages=["OOM killed"])
        result = await gen.analyze_reliability(data)
        assert "memory" in result.error_categories

    async def test_error_categorization_other(self):
        gen = TestInsightGenerator()
        data = _make_data(error_messages=["some random error"])
        result = await gen.analyze_reliability(data)
        assert "other" in result.error_categories

    async def test_empty_errors_skipped(self):
        gen = TestInsightGenerator()
        data = _make_data(error_messages=["", "Timeout"])
        result = await gen.analyze_reliability(data)
        assert result.error_categories.get("timeout", 0) == 1

    async def test_reliability_trend_stable(self):
        gen = TestInsightGenerator()
        data = _make_data(overall_success_rate=0.96)
        result = await gen.analyze_reliability(data)
        assert result.reliability_trend == "stable"

    async def test_reliability_trend_declining(self):
        gen = TestInsightGenerator()
        data = _make_data(overall_success_rate=0.90)
        result = await gen.analyze_reliability(data)
        assert result.reliability_trend == "declining"

    async def test_reliability_trend_degrading(self):
        gen = TestInsightGenerator()
        data = _make_data(overall_success_rate=0.50)
        result = await gen.analyze_reliability(data)
        assert result.reliability_trend == "degrading"

    async def test_stability_score_perfect(self):
        gen = TestInsightGenerator()
        data = _make_data(overall_success_rate=1.0, error_messages=[])
        result = await gen.analyze_reliability(data)
        assert result.stability_score == 100.0

    async def test_stability_score_with_errors(self):
        gen = TestInsightGenerator()
        data = _make_data(
            overall_success_rate=0.8,
            error_messages=["Timeout", "Timeout", "Connection error"],
        )
        result = await gen.analyze_reliability(data)
        # base = 80, penalty = (2/3)*10 = 6.67 -> ~73.3
        assert 70.0 < result.stability_score < 80.0

    async def test_success_rates_by_type(self):
        gen = TestInsightGenerator()
        mock_pass = MagicMock(success=True)
        mock_fail = MagicMock(success=False)
        data = _make_data(
            results_by_language={
                "python": [mock_pass, mock_pass, mock_fail],
                "go": [mock_pass],
            },
        )
        result = await gen.analyze_reliability(data)
        assert result.success_rate_by_type["python"] == pytest.approx(2 / 3)
        assert result.success_rate_by_type["go"] == 1.0


# ---------------------------------------------------------------------------
# analyze_coverage
# ---------------------------------------------------------------------------


class TestAnalyzeCoverage:
    async def test_basic_coverage(self):
        gen = TestInsightGenerator()
        data = _make_data(overall_success_rate=1.0, total_tests=10)
        result = await gen.analyze_coverage(data)
        # coverage = 60 + (1.0 * 30) = 90.0; no bonus (total_tests <= 50)
        assert result.line_coverage == 90.0
        assert result.branch_coverage == 90.0 * 0.9  # 81.0
        assert result.function_coverage == 90.0 * 0.95  # 85.5

    async def test_coverage_bonus_for_many_tests(self):
        gen = TestInsightGenerator()
        data_few = _make_data(overall_success_rate=0.8, total_tests=10)
        data_many = _make_data(overall_success_rate=0.8, total_tests=100)
        r_few = await gen.analyze_coverage(data_few)
        r_many = await gen.analyze_coverage(data_many)
        assert r_many.line_coverage > r_few.line_coverage

    async def test_coverage_capped_at_100(self):
        gen = TestInsightGenerator()
        data = _make_data(overall_success_rate=1.0, total_tests=1000)
        result = await gen.analyze_coverage(data)
        assert result.line_coverage <= 100.0

    async def test_coverage_gaps_below_threshold(self):
        gen = TestInsightGenerator(AITestInsightsConfig(coverage_threshold=100))
        data = _make_data(overall_success_rate=0.5)
        result = await gen.analyze_coverage(data)
        assert "Overall coverage below threshold" in result.coverage_gaps

    async def test_coverage_gaps_assertion_errors(self):
        gen = TestInsightGenerator()
        data = _make_data(error_messages=["AssertionError"])
        result = await gen.analyze_coverage(data)
        assert "Business logic validation coverage" in result.coverage_gaps

    async def test_coverage_gaps_connection_errors(self):
        gen = TestInsightGenerator()
        data = _make_data(error_messages=["Connection refused"])
        result = await gen.analyze_coverage(data)
        assert "External service integration coverage" in result.coverage_gaps

    async def test_coverage_gaps_permission_errors(self):
        gen = TestInsightGenerator()
        data = _make_data(error_messages=["Permission denied"])
        result = await gen.analyze_coverage(data)
        assert "Security and authorization coverage" in result.coverage_gaps

    async def test_uncovered_critical_paths_low_coverage(self):
        gen = TestInsightGenerator()
        data = _make_data(overall_success_rate=0.0, total_tests=1)
        result = await gen.analyze_coverage(data)
        # coverage ~ 60 → below 70 → should have all paths
        assert "Main business logic flows" in result.uncovered_critical_paths
        assert "Error handling paths" in result.uncovered_critical_paths

    async def test_uncovered_critical_paths_medium_coverage(self):
        gen = TestInsightGenerator()
        data = _make_data(overall_success_rate=0.4, total_tests=5)
        result = await gen.analyze_coverage(data)
        # coverage ~ 72 → below 80 but above 70
        assert "Error handling paths" in result.uncovered_critical_paths
        # should NOT include "Main business logic flows"
        assert "Main business logic flows" not in result.uncovered_critical_paths

    async def test_redundant_tests_detection(self):
        gen = TestInsightGenerator()
        # 21 times (> 20), 10 at 1.0s and 11 at 2.0s — both groups > 5
        times = [1.0] * 10 + [2.0] * 11
        data = _make_data(execution_times=times, total_tests=21)
        result = await gen.analyze_coverage(data)
        assert len(result.redundant_tests) == 2
        assert "Tests with 1.0s execution time pattern" in result.redundant_tests
        assert "Tests with 2.0s execution time pattern" in result.redundant_tests

    async def test_no_redundant_tests_with_few_tests(self):
        gen = TestInsightGenerator()
        data = _make_data(execution_times=[1.0, 2.0, 3.0], total_tests=3)
        result = await gen.analyze_coverage(data)
        assert result.redundant_tests == []

    async def test_recommended_tests_low_coverage(self):
        gen = TestInsightGenerator()
        data = _make_data(overall_success_rate=0.3, total_tests=5)
        result = await gen.analyze_coverage(data)
        assert "Add unit tests for core business logic" in result.recommended_tests

    async def test_recommended_tests_connection_errors(self):
        gen = TestInsightGenerator()
        data = _make_data(error_messages=["Connection refused"])
        result = await gen.analyze_coverage(data)
        assert "Add tests for external service failures" in result.recommended_tests

    async def test_recommended_tests_permission_errors(self):
        gen = TestInsightGenerator()
        data = _make_data(error_messages=["Permission denied"])
        result = await gen.analyze_coverage(data)
        assert "Add comprehensive authorization tests" in result.recommended_tests


# ---------------------------------------------------------------------------
# analyze_quality
# ---------------------------------------------------------------------------


class TestAnalyzeQuality:
    async def test_default_quality_score(self):
        gen = TestInsightGenerator()
        data = _make_data(quality_scores=[])
        result = await gen.analyze_quality(data)
        assert result.test_quality_score == 75.0
        assert result.code_quality_score == 75.0  # same as test_quality_score
        assert result.critical_issues == []
        assert result.quality_debt == 0.0

    async def test_custom_quality_score(self):
        gen = TestInsightGenerator()
        data = _make_data(quality_scores=[60.0, 80.0])
        result = await gen.analyze_quality(data)
        assert result.test_quality_score == 70.0

    async def test_critical_issues_filtered(self):
        gen = TestInsightGenerator()
        critical = _make_quality_issue(severity="critical")
        warning = _make_quality_issue(severity="warning")
        error = _make_quality_issue(severity="error")
        data = _make_data(quality_issues=[critical, warning, error])
        result = await gen.analyze_quality(data)
        # critical and error
        assert len(result.critical_issues) == 2

    async def test_quality_debt_calculation(self):
        gen = TestInsightGenerator()
        issues = [
            _make_quality_issue(severity="critical"),
            _make_quality_issue(severity="error"),
            _make_quality_issue(severity="warning"),
            _make_quality_issue(severity="info"),
        ]
        data = _make_data(quality_issues=issues)
        result = await gen.analyze_quality(data)
        # 4.0 + 2.0 + 0.5 + 0.25 = 6.75
        assert result.quality_debt == pytest.approx(6.75)

    async def test_quality_recommendations_low_score(self):
        gen = TestInsightGenerator()
        data = _make_data(quality_scores=[50.0])
        result = await gen.analyze_quality(data)
        assert "Address critical quality issues first" in result.quality_recommendations

    async def test_quality_recommendations_high_score(self):
        gen = TestInsightGenerator()
        data = _make_data(quality_scores=[90.0])
        result = await gen.analyze_quality(data)
        assert "Address critical quality issues first" not in result.quality_recommendations

    async def test_automated_fixes_always_present(self):
        gen = TestInsightGenerator()
        data = _make_data()
        result = await gen.analyze_quality(data)
        assert result.automated_fixes == [
            "Code formatting issues",
            "Import organization",
            "Basic type annotations",
            "Unused variable removal",
        ]


# ---------------------------------------------------------------------------
# generate_recommendations
# ---------------------------------------------------------------------------


class TestGenerateRecommendations:
    def _make_perf(self, slow=0, avg=1.0) -> TestPerformanceMetrics:
        return TestPerformanceMetrics(
            average_execution_time=avg,
            median_execution_time=avg,
            p95_execution_time=avg * 2,
            p99_execution_time=avg * 3,
            average_memory_usage=128.0,
            peak_memory_usage=192.0,
            average_cpu_usage=30.0,
            slow_tests_count=slow,
            fast_tests_count=10 - slow,
            performance_trend="stable",
            bottleneck_categories=[],
        )

    def _make_rel(self, flaky=0, success=1.0) -> TestReliabilityMetrics:
        return TestReliabilityMetrics(
            overall_success_rate=success,
            success_rate_by_type={},
            common_failure_patterns=[],
            flaky_tests_count=flaky,
            stability_score=90.0,
            reliability_trend="stable",
            error_categories={},
        )

    def _make_cov(self, line=90.0) -> TestCoverageInsights:
        return TestCoverageInsights(
            line_coverage=line,
            branch_coverage=line * 0.9,
            function_coverage=line * 0.95,
            uncovered_critical_paths=[],
            redundant_tests=[],
            coverage_gaps=[],
            recommended_tests=[],
        )

    def _make_qual(self, critical_issues=None) -> TestQualityInsights:
        return TestQualityInsights(
            test_quality_score=80.0,
            code_quality_score=80.0,
            critical_issues=critical_issues or [],
            quality_debt=0.0,
            quality_recommendations=[],
            automated_fixes=[],
        )

    async def test_perf_recommendation_when_slow_tests(self):
        gen = TestInsightGenerator()
        recs = await gen.generate_recommendations(
            self._make_perf(slow=5),
            self._make_rel(),
            self._make_cov(),
            self._make_qual(),
            [],
        )
        perf_recs = [r for r in recs if r.type == TestInsightType.PERFORMANCE]
        assert len(perf_recs) == 1
        assert perf_recs[0].recommendation_id == "perf_001"
        assert perf_recs[0].priority == "high"
        assert perf_recs[0].confidence == 0.9
        assert perf_recs[0].title == "Optimize Slow Tests"
        assert perf_recs[0].supporting_metrics["slow_tests_count"] == 5

    async def test_no_perf_recommendation_when_no_slow_tests(self):
        gen = TestInsightGenerator()
        recs = await gen.generate_recommendations(
            self._make_perf(slow=0),
            self._make_rel(),
            self._make_cov(),
            self._make_qual(),
            [],
        )
        perf_recs = [r for r in recs if r.type == TestInsightType.PERFORMANCE]
        assert len(perf_recs) == 0

    async def test_reliability_recommendation_when_flaky(self):
        gen = TestInsightGenerator()
        recs = await gen.generate_recommendations(
            self._make_perf(),
            self._make_rel(flaky=3, success=0.8),
            self._make_cov(),
            self._make_qual(),
            [],
        )
        rel_recs = [r for r in recs if r.type == TestInsightType.RELIABILITY]
        assert len(rel_recs) == 1
        assert rel_recs[0].recommendation_id == "rel_001"
        assert rel_recs[0].priority == "high"
        assert rel_recs[0].supporting_metrics["flaky_tests_count"] == 3
        assert rel_recs[0].supporting_metrics["success_rate"] == 0.8

    async def test_coverage_recommendation_when_low(self):
        gen = TestInsightGenerator()
        recs = await gen.generate_recommendations(
            self._make_perf(),
            self._make_rel(),
            self._make_cov(line=50.0),
            self._make_qual(),
            [],
        )
        cov_recs = [r for r in recs if r.type == TestInsightType.COVERAGE]
        assert len(cov_recs) == 1
        assert cov_recs[0].recommendation_id == "cov_001"
        assert cov_recs[0].priority == "medium"
        assert cov_recs[0].supporting_metrics["line_coverage"] == 50.0
        assert cov_recs[0].supporting_metrics["branch_coverage"] == 45.0

    async def test_quality_recommendation_when_critical_issues(self):
        gen = TestInsightGenerator()
        issue = _make_quality_issue(severity="critical")
        recs = await gen.generate_recommendations(
            self._make_perf(),
            self._make_rel(),
            self._make_cov(),
            self._make_qual(critical_issues=[issue]),
            [],
        )
        qual_recs = [r for r in recs if r.type == TestInsightType.QUALITY]
        assert len(qual_recs) == 1
        assert qual_recs[0].recommendation_id == "qual_001"
        assert qual_recs[0].priority == "high"
        assert qual_recs[0].supporting_metrics["critical_issues_count"] == 1

    async def test_pattern_recommendation_for_high_impact(self):
        gen = TestInsightGenerator()
        pattern = TestPatternInsight(
            pattern_type="performance_pattern",
            confidence=0.9,
            description="test",
            affected_tests=[],
            recommendations=[],
            impact_score=8.0,
        )
        recs = await gen.generate_recommendations(
            self._make_perf(),
            self._make_rel(),
            self._make_cov(),
            self._make_qual(),
            [pattern],
        )
        pattern_recs = [r for r in recs if r.type == TestInsightType.PATTERNS]
        assert len(pattern_recs) == 1
        assert pattern_recs[0].recommendation_id == "pattern_001"
        assert pattern_recs[0].priority == "high"
        assert pattern_recs[0].supporting_metrics["high_impact_patterns"] == 1

    async def test_recommendations_sorted_by_priority_and_confidence(self):
        gen = TestInsightGenerator()
        recs = await gen.generate_recommendations(
            self._make_perf(slow=3),
            self._make_rel(flaky=5, success=0.7),
            self._make_cov(line=50.0),
            self._make_qual(),
            [],
        )
        # perf_001 (high, 0.9), rel_001 (high, 0.85), cov_001 (medium, 0.8)
        assert len(recs) == 3
        assert recs[0].recommendation_id == "perf_001"
        assert recs[0].priority == "high"
        assert recs[0].confidence == 0.9
        assert recs[1].recommendation_id == "rel_001"
        assert recs[1].priority == "high"
        assert recs[1].confidence == 0.85
        assert recs[2].recommendation_id == "cov_001"
        assert recs[2].priority == "medium"


# ---------------------------------------------------------------------------
# _calculate_variance
# ---------------------------------------------------------------------------


class TestCalculateVariance:
    def test_empty_list(self):
        gen = TestInsightGenerator()
        assert gen._calculate_variance([]) == 0.0

    def test_single_value(self):
        gen = TestInsightGenerator()
        assert gen._calculate_variance([5.0]) == 0.0

    def test_known_variance(self):
        gen = TestInsightGenerator()
        # [2, 4, 4, 4, 5, 5, 7, 9] → mean=5, variance=4.0
        v = gen._calculate_variance([2, 4, 4, 4, 5, 5, 7, 9])
        assert v == pytest.approx(4.0)
