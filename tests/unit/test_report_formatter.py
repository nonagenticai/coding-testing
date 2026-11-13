"""Unit tests for services.ai_test_insights.report_formatter module."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from models.testing_models import QualityIssue, QualityResult, TestResult, TestSecurityLevel
from services.ai_test_insights.config import AITestInsightsConfig, TestInsightType, TestPatternType
from services.ai_test_insights.insight_generation import TestInsightGenerator
from services.ai_test_insights.models import (
    TestCoverageInsights,
    TestInsightAnalysis,
    TestInsightRecommendation,
    TestPatternInsight,
    TestPerformanceMetrics,
    TestQualityInsights,
    TestReliabilityMetrics,
)
from services.ai_test_insights.pattern_recognition import TestPatternDetector
from services.ai_test_insights.report_formatter import (
    TestReportFormatter,
    create_test_report_formatter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_result(success=True, execution_time=1.0, language="python", stderr="") -> TestResult:
    return TestResult(
        test_id="test-1",
        success=success,
        stdout="ok",
        stderr=stderr,
        exit_code=0 if success else 1,
        execution_time=execution_time,
        created_at=datetime.now(UTC),
        language=language,
        security_level=TestSecurityLevel.HIGH,
    )


def _make_quality_result(score=80.0, issues=None) -> QualityResult:
    return QualityResult(
        analysis_id="qa-1",
        success=True,
        overall_score=score,
        total_issues=len(issues) if issues else 0,
        issues=issues or [],
        tools_results={},
        execution_time=0.5,
        created_at=datetime.now(UTC),
        language="python",
    )


def _make_analysis(
    health_score=80.0,
    improvement=20.0,
    slow=0,
    flaky=0,
    critical_issues=None,
    patterns=None,
    recommendations=None,
) -> TestInsightAnalysis:
    return TestInsightAnalysis(
        analysis_id="analysis-1",
        analyzed_period="recent",
        performance_metrics=TestPerformanceMetrics(
            average_execution_time=2.0,
            median_execution_time=1.5,
            p95_execution_time=5.0,
            p99_execution_time=8.0,
            average_memory_usage=200.0,
            peak_memory_usage=300.0,
            average_cpu_usage=40.0,
            slow_tests_count=slow,
            fast_tests_count=10,
            performance_trend="stable",
            bottleneck_categories=[],
        ),
        reliability_metrics=TestReliabilityMetrics(
            overall_success_rate=0.9,
            success_rate_by_type={},
            common_failure_patterns=[],
            flaky_tests_count=flaky,
            stability_score=85.0,
            reliability_trend="stable",
            error_categories={},
        ),
        coverage_insights=TestCoverageInsights(
            line_coverage=80.0,
            branch_coverage=72.0,
            function_coverage=76.0,
            uncovered_critical_paths=[],
            redundant_tests=[],
            coverage_gaps=[],
            recommended_tests=[],
        ),
        quality_insights=TestQualityInsights(
            test_quality_score=85.0,
            code_quality_score=85.0,
            critical_issues=critical_issues or [],
            quality_debt=2.0,
            quality_recommendations=[],
            automated_fixes=[],
        ),
        detected_patterns=patterns or [],
        recommendations=recommendations or [],
        overall_health_score=health_score,
        improvement_potential=improvement,
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateReportFormatter:
    def test_default(self):
        f = create_test_report_formatter()
        assert f.cache_manager is None
        assert f.config.slow_test_threshold == 10.0  # DEFAULT_CONFIG value
        assert f.config.cache_ttl == 86400
        assert isinstance(f.insight_generator, TestInsightGenerator)
        assert isinstance(f.pattern_detector, TestPatternDetector)
        assert f.insight_generator.config is f.config
        assert f.pattern_detector.config is f.config

    def test_with_cache_manager(self):
        cm = AsyncMock()
        f = create_test_report_formatter(cache_manager=cm)
        assert f.cache_manager is cm

    def test_with_custom_config(self):
        cfg = AITestInsightsConfig(slow_test_threshold=5.0)
        f = create_test_report_formatter(config=cfg)
        assert f.config.slow_test_threshold == 5.0


# ---------------------------------------------------------------------------
# _prepare_analysis_data
# ---------------------------------------------------------------------------


class TestPrepareAnalysisData:
    async def test_basic_data_preparation(self):
        f = TestReportFormatter()
        tr = [_make_test_result(execution_time=2.0), _make_test_result(execution_time=4.0)]
        qr = [_make_quality_result(score=90.0)]
        data = await f._prepare_analysis_data(tr, qr)

        assert data["total_tests"] == 2
        assert data["avg_execution_time"] == 3.0
        assert data["overall_success_rate"] == 1.0
        assert data["total_quality_analyses"] == 1
        assert 90.0 in data["quality_scores"]

    async def test_empty_inputs(self):
        f = TestReportFormatter()
        data = await f._prepare_analysis_data([], [])
        assert data["total_tests"] == 0
        assert data["avg_execution_time"] == 0
        assert data["overall_success_rate"] == 0

    async def test_error_messages_extracted(self):
        f = TestReportFormatter()
        tr = [_make_test_result(stderr="timeout error")]
        data = await f._prepare_analysis_data(tr, [])
        assert "timeout error" in data["error_messages"]

    async def test_results_grouped_by_language(self):
        f = TestReportFormatter()
        tr = [
            _make_test_result(language="python"),
            _make_test_result(language="go"),
            _make_test_result(language="python"),
        ]
        data = await f._prepare_analysis_data(tr, [])
        assert len(data["results_by_language"]["python"]) == 2
        assert len(data["results_by_language"]["go"]) == 1


# ---------------------------------------------------------------------------
# _handle_result
# ---------------------------------------------------------------------------


class TestHandleResult:
    def test_returns_value_when_no_exception(self):
        f = TestReportFormatter()
        assert f._handle_result("ok", "test", str) == "ok"

    def test_returns_exact_value_passthrough(self):
        f = TestReportFormatter()
        data = {"key": [1, 2, 3]}
        assert f._handle_result(data, "test", dict) is data

    def test_returns_empty_list_for_list_type(self):
        f = TestReportFormatter()
        result = f._handle_result(ValueError("boom"), "patterns", list)
        assert result == []

    def test_returns_default_performance_metrics_on_error(self):
        f = TestReportFormatter()
        result = f._handle_result(ValueError("boom"), "performance", TestPerformanceMetrics)
        assert result.average_execution_time == 0.0
        assert result.median_execution_time == 0.0
        assert result.p95_execution_time == 0.0
        assert result.p99_execution_time == 0.0
        assert result.average_memory_usage == 0.0
        assert result.peak_memory_usage == 0.0
        assert result.average_cpu_usage == 0.0
        assert result.slow_tests_count == 0
        assert result.fast_tests_count == 0
        assert result.performance_trend == "unknown"
        assert result.bottleneck_categories == []


# ---------------------------------------------------------------------------
# _generate_summary
# ---------------------------------------------------------------------------


class TestGenerateSummary:
    def test_basic_summary(self):
        f = TestReportFormatter()
        analysis = _make_analysis()
        summary = f._generate_summary(analysis, {}, None)
        assert summary["overall_health_score"] == 80.0
        assert summary["health_grade"] == "B"
        assert summary["improvement_potential"] == 20.0
        # Verify core_metrics are computed from the analysis model
        perf_score = analysis.performance_metrics.performance_score
        reliability_score = analysis.reliability_metrics.reliability_score
        coverage_score = analysis.coverage_insights.overall_coverage_score
        quality_score = analysis.quality_insights.combined_quality_score
        assert summary["core_metrics"]["performance_score"] == perf_score
        assert summary["core_metrics"]["reliability_score"] == reliability_score
        assert summary["core_metrics"]["coverage_score"] == coverage_score
        assert summary["core_metrics"]["quality_score"] == quality_score
        # Verify critical_counts reflect the fixture values
        assert summary["critical_counts"]["slow_tests"] == 0
        assert summary["critical_counts"]["flaky_tests"] == 0
        assert summary["critical_counts"]["critical_quality_issues"] == 0
        assert summary["critical_counts"]["high_impact_patterns"] == 0
        # No comprehensive recommendations
        assert summary["recommendations_summary"]["total_core_recommendations"] == 0
        assert summary["recommendations_summary"]["has_comprehensive_recommendations"] is False

    def test_summary_with_slow_and_flaky_counts(self):
        f = TestReportFormatter()
        analysis = _make_analysis(slow=5, flaky=3)
        summary = f._generate_summary(analysis, {}, None)
        assert summary["critical_counts"]["slow_tests"] == 5
        assert summary["critical_counts"]["flaky_tests"] == 3

    def test_summary_with_specialized_performance(self):
        f = TestReportFormatter()
        analysis = _make_analysis()
        specialized = {"performance_analysis": {"performance_score": 95.0}}
        summary = f._generate_summary(analysis, specialized, None)
        assert summary["specialized_performance_score"] == 95.0

    def test_summary_with_specialized_quality(self):
        f = TestReportFormatter()
        analysis = _make_analysis()
        specialized = {"quality_analysis": {"overall_quality_score": 88.0}}
        summary = f._generate_summary(analysis, specialized, None)
        assert summary["specialized_quality_score"] == 88.0

    def test_summary_with_pattern_analysis(self):
        f = TestReportFormatter()
        analysis = _make_analysis()
        specialized = {"pattern_analysis": {"overall_pattern_health": 70.0}}
        summary = f._generate_summary(analysis, specialized, None)
        assert summary["pattern_health"] == 70.0

    def test_summary_with_failure_predictions(self):
        f = TestReportFormatter()
        analysis = _make_analysis()
        specialized = {"failure_predictions": {"overall_risk_level": "low"}}
        summary = f._generate_summary(analysis, specialized, None)
        assert summary["failure_risk"] == "low"

    def test_summary_omits_specialized_keys_when_empty(self):
        f = TestReportFormatter()
        analysis = _make_analysis()
        summary = f._generate_summary(analysis, {}, None)
        assert "specialized_performance_score" not in summary
        assert "specialized_quality_score" not in summary
        assert "pattern_health" not in summary
        assert "failure_risk" not in summary
        assert "total_comprehensive_recommendations" not in summary

    def test_summary_with_comprehensive_recommendations(self):
        f = TestReportFormatter()
        analysis = _make_analysis()
        mock_recs = MagicMock()
        mock_recs.total_recommendations = 5
        mock_recs.critical_recommendations = ["a", "b"]
        summary = f._generate_summary(analysis, {}, mock_recs)
        assert summary["total_comprehensive_recommendations"] == 5
        assert summary["critical_comprehensive_recommendations"] == 2


# ---------------------------------------------------------------------------
# format_analysis_summary
# ---------------------------------------------------------------------------


class TestFormatAnalysisSummary:
    async def test_summary_structure(self):
        f = TestReportFormatter()
        analysis = _make_analysis(health_score=85.0, improvement=15.0)
        summary = await f.format_analysis_summary(analysis)

        assert summary["analysis_id"] == "analysis-1"
        assert summary["health_overview"]["overall_score"] == 85.0
        assert summary["health_overview"]["health_grade"] == "B"
        assert summary["health_overview"]["improvement_potential"] == 15.0
        # key_metrics should contain computed scores from the analysis model
        assert (
            summary["key_metrics"]["performance_score"]
            == analysis.performance_metrics.performance_score
        )
        assert (
            summary["key_metrics"]["reliability_score"]
            == analysis.reliability_metrics.reliability_score
        )
        assert (
            summary["key_metrics"]["coverage_score"]
            == analysis.coverage_insights.overall_coverage_score
        )
        assert (
            summary["key_metrics"]["quality_score"]
            == analysis.quality_insights.combined_quality_score
        )
        # critical_issues reflect fixture defaults (0 slow, 0 flaky, 0 quality, 0 high-impact)
        assert summary["critical_issues"]["slow_tests"] == 0
        assert summary["critical_issues"]["flaky_tests"] == 0
        assert summary["critical_issues"]["critical_quality_issues"] == 0
        assert summary["critical_issues"]["high_impact_patterns"] == 0
        # analysis_metadata
        assert summary["analysis_metadata"]["analyzed_period"] == "recent"
        assert summary["analysis_metadata"]["total_patterns_detected"] == 0
        assert summary["analysis_metadata"]["total_recommendations"] == 0

    async def test_summary_with_recommendations(self):
        rec = TestInsightRecommendation(
            recommendation_id="r1",
            type=TestInsightType.PERFORMANCE,
            priority="high",
            confidence=0.9,
            title="Fix slow tests",
            description="desc",
            rationale="because",
            implementation_steps=["step1"],
            estimated_effort="medium",
            expected_impact="30% faster",
            supporting_metrics={},
            affected_components=[],
        )
        analysis = _make_analysis(recommendations=[rec])
        f = TestReportFormatter()
        summary = await f.format_analysis_summary(analysis)
        assert len(summary["top_recommendations"]) == 1
        top_rec = summary["top_recommendations"][0]
        assert top_rec["id"] == "r1"
        assert top_rec["title"] == "Fix slow tests"
        assert top_rec["priority"] == "high"
        assert top_rec["confidence"] == 0.9
        assert top_rec["expected_impact"] == "30% faster"
        assert summary["analysis_metadata"]["total_recommendations"] == 1

    async def test_summary_caps_recommendations_at_5(self):
        recs = [
            TestInsightRecommendation(
                recommendation_id=f"r{i}",
                type=TestInsightType.PERFORMANCE,
                priority="medium",
                confidence=0.5,
                title=f"Rec {i}",
                description="d",
                rationale="r",
                implementation_steps=[],
                estimated_effort="low",
                expected_impact="impact",
                supporting_metrics={},
                affected_components=[],
            )
            for i in range(10)
        ]
        analysis = _make_analysis(recommendations=recs)
        f = TestReportFormatter()
        summary = await f.format_analysis_summary(analysis)
        assert len(summary["top_recommendations"]) == 5
        # Verify first 5 were selected (top_recommendations[:5])
        for i in range(5):
            assert summary["top_recommendations"][i]["id"] == f"r{i}"
            assert summary["top_recommendations"][i]["title"] == f"Rec {i}"
        # Total recommendations metadata still reports all 10
        assert summary["analysis_metadata"]["total_recommendations"] == 10


# ---------------------------------------------------------------------------
# generate_markdown_report
# ---------------------------------------------------------------------------


class TestGenerateMarkdownReport:
    async def test_markdown_has_header(self):
        f = TestReportFormatter()
        analysis = _make_analysis()
        md = await f.generate_markdown_report(analysis)
        assert "# Test Insights Analysis Report" in md
        assert "analysis-1" in md

    async def test_markdown_has_executive_summary(self):
        f = TestReportFormatter()
        analysis = _make_analysis(health_score=85.0, improvement=15.0)
        md = await f.generate_markdown_report(analysis)
        assert "Executive Summary" in md
        assert "85.0/100" in md

    async def test_markdown_has_key_metrics_table(self):
        f = TestReportFormatter()
        analysis = _make_analysis()
        md = await f.generate_markdown_report(analysis)
        assert "Key Metrics" in md
        assert "| Performance |" in md
        assert "| Reliability |" in md

    async def test_markdown_no_critical_issues(self):
        f = TestReportFormatter()
        analysis = _make_analysis(slow=0, flaky=0)
        md = await f.generate_markdown_report(analysis)
        assert "No critical issues detected" in md

    async def test_markdown_with_slow_and_flaky(self):
        f = TestReportFormatter()
        analysis = _make_analysis(slow=3, flaky=2)
        md = await f.generate_markdown_report(analysis)
        assert "3 slow tests" in md
        assert "2 flaky tests" in md

    async def test_markdown_with_critical_quality_issues(self):
        f = TestReportFormatter()
        issue = QualityIssue(tool="ruff", severity="critical", message="bad", file_path="f.py")
        analysis = _make_analysis(critical_issues=[issue])
        md = await f.generate_markdown_report(analysis)
        assert "1 critical quality issues" in md

    async def test_markdown_with_recommendations(self):
        rec = TestInsightRecommendation(
            recommendation_id="r1",
            type=TestInsightType.PERFORMANCE,
            priority="high",
            confidence=0.9,
            title="Fix Tests",
            description="desc",
            rationale="because",
            implementation_steps=[],
            estimated_effort="medium",
            expected_impact="big",
            supporting_metrics={},
            affected_components=[],
        )
        analysis = _make_analysis(recommendations=[rec])
        f = TestReportFormatter()
        md = await f.generate_markdown_report(analysis)
        assert "Fix Tests" in md
        assert "High" in md

    async def test_markdown_with_patterns(self):
        pattern = TestPatternInsight(
            pattern_type=TestPatternType.PERFORMANCE_PATTERN,
            confidence=0.85,
            description="Slow test execution",
            affected_tests=[],
            recommendations=[],
            impact_score=7.5,
        )
        analysis = _make_analysis(patterns=[pattern])
        f = TestReportFormatter()
        md = await f.generate_markdown_report(analysis)
        assert "Detected Patterns" in md
        assert "Slow test execution" in md

    async def test_markdown_detailed_metrics_section(self):
        f = TestReportFormatter()
        analysis = _make_analysis()
        md = await f.generate_markdown_report(analysis)
        assert "<details>" in md
        assert "Performance Metrics" in md
        assert "Reliability Metrics" in md
        assert "Coverage Insights" in md
        assert "Quality Insights" in md

    async def test_markdown_high_impact_patterns(self):
        pattern = TestPatternInsight(
            pattern_type=TestPatternType.FLAKY_PATTERN,
            confidence=0.9,
            description="Flaky tests",
            affected_tests=[],
            recommendations=[],
            impact_score=9.0,
        )
        analysis = _make_analysis(patterns=[pattern])
        f = TestReportFormatter()
        md = await f.generate_markdown_report(analysis)
        assert "1 high-impact patterns" in md


# ---------------------------------------------------------------------------
# create_comprehensive_analysis
# ---------------------------------------------------------------------------


class TestCreateComprehensiveAnalysis:
    async def test_creates_analysis(self):
        f = TestReportFormatter()
        tr = [
            _make_test_result(execution_time=1.0),
            _make_test_result(execution_time=2.0, success=False, stderr="timeout error"),
        ]
        qr = [_make_quality_result(score=70.0)]
        analysis = await f.create_comprehensive_analysis(tr, qr)
        assert analysis.analyzed_period == "recent"
        # Health score is bounded 0-100
        assert 0 <= analysis.overall_health_score <= 100
        # Improvement potential is bounded 0-100
        assert 0 <= analysis.improvement_potential <= 100
        # Performance metrics were computed from the 2 test results
        assert isinstance(analysis.performance_metrics, TestPerformanceMetrics)
        assert analysis.performance_metrics.average_execution_time == 1.5  # mean(1.0, 2.0)
        assert isinstance(analysis.reliability_metrics, TestReliabilityMetrics)
        assert analysis.reliability_metrics.overall_success_rate == 0.5  # 1 of 2 succeeded
        assert isinstance(analysis.coverage_insights, TestCoverageInsights)
        assert isinstance(analysis.quality_insights, TestQualityInsights)
        # health_grade is derived from overall_health_score
        expected_grade = (
            "A"
            if analysis.overall_health_score >= 90
            else "B"
            if analysis.overall_health_score >= 80
            else "C"
            if analysis.overall_health_score >= 70
            else "D"
            if analysis.overall_health_score >= 60
            else "F"
        )
        assert analysis.health_grade == expected_grade

    async def test_creates_analysis_with_custom_period(self):
        f = TestReportFormatter()
        tr = [_make_test_result()]
        qr = [_make_quality_result()]
        analysis = await f.create_comprehensive_analysis(tr, qr, time_period="last_week")
        assert analysis.analyzed_period == "last_week"

    async def test_caches_when_cache_manager_present(self):
        cm = AsyncMock()
        f = TestReportFormatter(cache_manager=cm)
        tr = [_make_test_result()]
        qr = [_make_quality_result()]
        analysis = await f.create_comprehensive_analysis(tr, qr)
        cm.set.assert_called_once()
        # Verify the cache key format and TTL
        call_args = cm.set.call_args
        cache_key = call_args.args[0] if call_args.args else call_args[0][0]
        assert cache_key.startswith("test_insights:")
        assert cache_key == f"test_insights:{analysis.analysis_id}"
        # TTL should match config default
        assert call_args.kwargs.get("ttl") == f.config.cache_ttl

    async def test_does_not_cache_without_cache_manager(self):
        f = TestReportFormatter()
        assert f.cache_manager is None
        tr = [_make_test_result()]
        qr = [_make_quality_result()]
        # Should complete without error and not attempt to cache
        analysis = await f.create_comprehensive_analysis(tr, qr)
        assert analysis.analyzed_period == "recent"


# ---------------------------------------------------------------------------
# Cache methods
# ---------------------------------------------------------------------------


class TestCacheMethods:
    async def test_get_cached_analysis_no_cache_manager(self):
        f = TestReportFormatter()
        result = await f.get_cached_analysis("id-1")
        assert result is None

    async def test_get_cached_analysis_miss(self):
        cm = AsyncMock()
        cm.get = AsyncMock(return_value=None)
        f = TestReportFormatter(cache_manager=cm)
        result = await f.get_cached_analysis("id-1")
        assert result is None
        cm.get.assert_called_once_with("test_insights:id-1")

    async def test_get_cached_analysis_hit(self):
        analysis = _make_analysis()
        cm = AsyncMock()
        cm.get = AsyncMock(return_value=analysis.model_dump())
        f = TestReportFormatter(cache_manager=cm)
        result = await f.get_cached_analysis("id-1")
        assert result is not None
        assert result.analysis_id == "analysis-1"
        assert result.overall_health_score == 80.0
        assert result.analyzed_period == "recent"
        cm.get.assert_called_once_with("test_insights:id-1")

    async def test_get_cached_comprehensive_no_cache_manager(self):
        f = TestReportFormatter()
        result = await f.get_cached_comprehensive_analysis("id-1")
        assert result is None

    async def test_get_cached_comprehensive_miss(self):
        cm = AsyncMock()
        cm.get = AsyncMock(return_value=None)
        f = TestReportFormatter(cache_manager=cm)
        result = await f.get_cached_comprehensive_analysis("id-1")
        assert result is None
        cm.get.assert_called_once_with("comprehensive_test_analysis:id-1")

    async def test_get_cached_comprehensive_hit(self):
        report_data = {"analysis_id": "comp-1", "analysis_type": "comprehensive"}
        cm = AsyncMock()
        cm.get = AsyncMock(return_value=report_data)
        f = TestReportFormatter(cache_manager=cm)
        result = await f.get_cached_comprehensive_analysis("comp-1")
        assert result == {"analysis_id": "comp-1", "analysis_type": "comprehensive"}
        cm.get.assert_called_once_with("comprehensive_test_analysis:comp-1")

    async def test_cache_analysis_results_no_cache_manager(self):
        f = TestReportFormatter()
        # Should not raise — early return when no cache manager
        await f._cache_analysis_results("id-1", _make_analysis())

    async def test_cache_analysis_results_calls_set(self):
        cm = AsyncMock()
        f = TestReportFormatter(cache_manager=cm)
        analysis = _make_analysis()
        await f._cache_analysis_results("id-1", analysis)
        cm.set.assert_called_once()
        call_args = cm.set.call_args
        assert call_args.args[0] == "test_insights:id-1"
        assert call_args.kwargs["ttl"] == f.config.cache_ttl

    async def test_cache_comprehensive_no_cache_manager(self):
        f = TestReportFormatter()
        await f._cache_comprehensive_analysis("id-1", {})

    async def test_cache_comprehensive_calls_set(self):
        cm = AsyncMock()
        f = TestReportFormatter(cache_manager=cm)
        report = {"key": "value"}
        await f._cache_comprehensive_analysis("id-1", report)
        cm.set.assert_called_once()
        call_args = cm.set.call_args
        assert call_args.args[0] == "comprehensive_test_analysis:id-1"
        assert call_args.args[1] == report
        assert call_args.kwargs["ttl"] == f.config.comprehensive_cache_ttl

    async def test_cache_analysis_exception_logged(self):
        cm = AsyncMock()
        cm.set = AsyncMock(side_effect=Exception("cache error"))
        f = TestReportFormatter(cache_manager=cm)
        # Should not raise — exception is caught and logged
        await f._cache_analysis_results("id-1", _make_analysis())
        cm.set.assert_called_once()

    async def test_cache_comprehensive_exception_logged(self):
        cm = AsyncMock()
        cm.set = AsyncMock(side_effect=Exception("cache error"))
        f = TestReportFormatter(cache_manager=cm)
        await f._cache_comprehensive_analysis("id-1", {})
        cm.set.assert_called_once()

    async def test_get_cached_analysis_exception_returns_none(self):
        cm = AsyncMock()
        cm.get = AsyncMock(side_effect=Exception("boom"))
        f = TestReportFormatter(cache_manager=cm)
        result = await f.get_cached_analysis("id-1")
        assert result is None
        cm.get.assert_called_once()

    async def test_get_cached_comprehensive_exception_returns_none(self):
        cm = AsyncMock()
        cm.get = AsyncMock(side_effect=Exception("boom"))
        f = TestReportFormatter(cache_manager=cm)
        result = await f.get_cached_comprehensive_analysis("id-1")
        assert result is None
        cm.get.assert_called_once()
