"""Unit tests for services.ai_test_insights.core_service module."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from models.testing_models import QualityResult, TestResult, TestSecurityLevel
from services.ai_test_insights.config import AITestInsightsConfig, TestInsightType
from services.ai_test_insights.core_service import (
    AITestInsightsService,
    create_ai_test_insights_service,
)
from services.ai_test_insights.insight_generation import TestInsightGenerator
from services.ai_test_insights.models import (
    TestCoverageInsights,
    TestInsightAnalysis,
    TestPatternInsight,
    TestPerformanceMetrics,
    TestQualityInsights,
    TestReliabilityMetrics,
)
from services.ai_test_insights.pattern_recognition import TestPatternDetector
from services.ai_test_insights.report_formatter import TestReportFormatter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_result(
    success=True, execution_time=1.0, language="python", stderr="", test_type=None
) -> TestResult:
    return TestResult(
        test_id="t-1",
        success=success,
        stdout="ok",
        stderr=stderr,
        exit_code=0 if success else 1,
        execution_time=execution_time,
        created_at=datetime.now(UTC),
        language=language,
        security_level=TestSecurityLevel.HIGH,
        test_type=test_type,
    )


def _make_quality_result(score=80.0) -> QualityResult:
    return QualityResult(
        analysis_id="qa-1",
        success=True,
        overall_score=score,
        total_issues=0,
        issues=[],
        tools_results={},
        execution_time=0.5,
        created_at=datetime.now(UTC),
        language="python",
    )


def _make_analysis(health_score=80.0) -> TestInsightAnalysis:
    return TestInsightAnalysis(
        analysis_id="a-1",
        analyzed_period="recent",
        performance_metrics=TestPerformanceMetrics(
            average_execution_time=1.0,
            median_execution_time=1.0,
            p95_execution_time=2.0,
            p99_execution_time=3.0,
            average_memory_usage=128.0,
            peak_memory_usage=192.0,
            average_cpu_usage=30.0,
            slow_tests_count=0,
            fast_tests_count=10,
            performance_trend="stable",
            bottleneck_categories=[],
        ),
        reliability_metrics=TestReliabilityMetrics(
            overall_success_rate=0.95,
            success_rate_by_type={},
            common_failure_patterns=[],
            flaky_tests_count=0,
            stability_score=90.0,
            reliability_trend="stable",
            error_categories={},
        ),
        coverage_insights=TestCoverageInsights(
            line_coverage=85.0,
            branch_coverage=76.5,
            function_coverage=80.75,
            uncovered_critical_paths=[],
            redundant_tests=[],
            coverage_gaps=[],
            recommended_tests=[],
        ),
        quality_insights=TestQualityInsights(
            test_quality_score=85.0,
            code_quality_score=85.0,
            critical_issues=[],
            quality_debt=1.0,
            quality_recommendations=[],
            automated_fixes=[],
        ),
        detected_patterns=[],
        recommendations=[],
        overall_health_score=health_score,
        improvement_potential=20.0,
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_create_default(self):
        svc = create_ai_test_insights_service()
        assert isinstance(svc, AITestInsightsService)
        assert svc.cache_manager is None
        assert svc.ai_insights_engine is None

    def test_create_with_cache_manager(self):
        cm = AsyncMock()
        svc = create_ai_test_insights_service(cache_manager=cm)
        assert svc.cache_manager is cm

    def test_create_with_config(self):
        cfg = AITestInsightsConfig(slow_test_threshold=5.0)
        svc = create_ai_test_insights_service(config=cfg)
        assert svc.config.slow_test_threshold == 5.0
        assert svc.config is cfg


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestServiceInit:
    def test_components_initialized(self):
        svc = AITestInsightsService(config=AITestInsightsConfig(enable_ai_features=False))
        assert isinstance(svc.insight_generator, TestInsightGenerator)
        assert isinstance(svc.pattern_detector, TestPatternDetector)
        assert isinstance(svc.report_formatter, TestReportFormatter)

    def test_ai_agents_disabled_when_feature_off(self):
        svc = AITestInsightsService(config=AITestInsightsConfig(enable_ai_features=False))
        assert svc.performance_agent is None
        assert svc.reliability_agent is None
        assert svc.coverage_agent is None
        assert svc.quality_agent is None
        assert svc.pattern_agent is None
        assert svc.recommendation_agent is None

    def test_get_config(self):
        cfg = AITestInsightsConfig(slow_test_threshold=7.0)
        svc = AITestInsightsService(config=cfg)
        returned_cfg = svc.get_config()
        assert returned_cfg.slow_test_threshold == 7.0
        assert returned_cfg is cfg

    def test_update_config(self):
        svc = AITestInsightsService(config=AITestInsightsConfig(enable_ai_features=False))
        svc.update_config(slow_test_threshold=3.0)
        assert svc.config.slow_test_threshold == 3.0
        # Components should be re-initialized with the new config
        assert svc.insight_generator.config.slow_test_threshold == 3.0
        assert isinstance(svc.insight_generator, TestInsightGenerator)
        assert isinstance(svc.pattern_detector, TestPatternDetector)

    def test_update_config_unknown_key_ignored(self):
        svc = AITestInsightsService(config=AITestInsightsConfig(enable_ai_features=False))
        original_threshold = svc.config.slow_test_threshold
        svc.update_config(nonexistent_param=42)
        # Config should be unchanged for known keys
        assert svc.config.slow_test_threshold == original_threshold
        assert not hasattr(svc.config, "nonexistent_param")


# ---------------------------------------------------------------------------
# analyze_test_insights
# ---------------------------------------------------------------------------


class TestAnalyzeTestInsights:
    async def test_returns_analysis(self):
        svc = AITestInsightsService(config=AITestInsightsConfig(enable_ai_features=False))
        tr = [_make_test_result(), _make_test_result(execution_time=3.0)]
        qr = [_make_quality_result()]
        result = await svc.analyze_test_insights(tr, qr)
        assert isinstance(result, TestInsightAnalysis)
        assert result.analyzed_period == "recent"
        assert isinstance(result.performance_metrics, TestPerformanceMetrics)
        assert isinstance(result.reliability_metrics, TestReliabilityMetrics)

    async def test_passes_time_period(self):
        svc = AITestInsightsService(config=AITestInsightsConfig(enable_ai_features=False))
        tr = [_make_test_result()]
        qr = [_make_quality_result()]
        result = await svc.analyze_test_insights(tr, qr, time_period="weekly")
        assert result.analyzed_period == "weekly"


# ---------------------------------------------------------------------------
# predict_test_failures
# ---------------------------------------------------------------------------


class TestPredictFailures:
    async def test_all_passing(self):
        svc = AITestInsightsService(config=AITestInsightsConfig(enable_ai_features=False))
        tr = [_make_test_result(language="python", test_type="unit")] * 5
        preds = await svc.predict_test_failures(tr)
        assert preds == {"python_unit": 0.0}

    async def test_all_failing(self):
        svc = AITestInsightsService(config=AITestInsightsConfig(enable_ai_features=False))
        tr = [_make_test_result(success=False, language="go", test_type="unit")] * 5
        preds = await svc.predict_test_failures(tr)
        assert preds["go_unit"] == pytest.approx(1.0)

    async def test_mixed_results(self):
        svc = AITestInsightsService(config=AITestInsightsConfig(enable_ai_features=False))
        tr = [
            _make_test_result(success=True, language="python", test_type="unit"),
            _make_test_result(success=False, language="python", test_type="unit"),
        ]
        preds = await svc.predict_test_failures(tr)
        assert 0.0 < preds["python_unit"] < 1.0
        # 50% failure rate: overall = 0.5, recent (last 5 = both) = 0.5
        # weighted: 0.7 * 0.5 + 0.3 * 0.5 = 0.5
        assert preds["python_unit"] == pytest.approx(0.5)

    async def test_unknown_test_type(self):
        svc = AITestInsightsService(config=AITestInsightsConfig(enable_ai_features=False))
        tr = [_make_test_result(language="rust")]
        preds = await svc.predict_test_failures(tr)
        assert "rust_unknown" in preds
        # All passing -> 0.0
        assert preds["rust_unknown"] == 0.0

    async def test_bounded_0_to_1(self):
        svc = AITestInsightsService(config=AITestInsightsConfig(enable_ai_features=False))
        tr = [_make_test_result(success=False)] * 10
        preds = await svc.predict_test_failures(tr)
        for v in preds.values():
            assert 0.0 <= v <= 1.0

    async def test_multiple_groups(self):
        svc = AITestInsightsService(config=AITestInsightsConfig(enable_ai_features=False))
        tr = [
            _make_test_result(language="python", test_type="unit"),
            _make_test_result(language="go", test_type="integration", success=False),
        ]
        preds = await svc.predict_test_failures(tr)
        assert "python_unit" in preds
        assert "go_integration" in preds
        assert preds["python_unit"] == 0.0
        assert preds["go_integration"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# generate_test_recommendations
# ---------------------------------------------------------------------------


class TestGenerateTestRecommendations:
    async def test_low_health_triggers_overhaul(self):
        svc = AITestInsightsService(config=AITestInsightsConfig(enable_ai_features=False))
        analysis = _make_analysis(health_score=50.0)
        recs = await svc.generate_test_recommendations(analysis)
        overhaul = [r for r in recs if "Overhaul" in r.title]
        assert len(overhaul) == 1
        assert overhaul[0].priority == "high"
        assert overhaul[0].type == TestInsightType.RECOMMENDATIONS
        assert overhaul[0].confidence == 0.85
        assert "50.0" in overhaul[0].rationale

    async def test_high_health_no_overhaul(self):
        svc = AITestInsightsService(config=AITestInsightsConfig(enable_ai_features=False))
        analysis = _make_analysis(health_score=90.0)
        recs = await svc.generate_test_recommendations(analysis)
        overhaul = [r for r in recs if "Overhaul" in r.title]
        assert len(overhaul) == 0

    async def test_high_impact_patterns_trigger_recommendation(self):
        svc = AITestInsightsService(config=AITestInsightsConfig(enable_ai_features=False))
        pattern = TestPatternInsight(
            pattern_type="flaky_pattern",
            confidence=0.9,
            description="Flaky",
            affected_tests=[],
            recommendations=[],
            impact_score=9.0,
        )
        analysis = _make_analysis(health_score=90.0)
        analysis.detected_patterns = [pattern]
        recs = await svc.generate_test_recommendations(analysis)
        pattern_recs = [r for r in recs if r.type == TestInsightType.PATTERNS]
        assert len(pattern_recs) == 1
        assert pattern_recs[0].priority == "high"
        assert pattern_recs[0].title == "Address High-Impact Test Patterns"
        assert "1 high-impact patterns" in pattern_recs[0].rationale

    async def test_no_patterns_no_pattern_recommendation(self):
        svc = AITestInsightsService(config=AITestInsightsConfig(enable_ai_features=False))
        analysis = _make_analysis(health_score=90.0)
        recs = await svc.generate_test_recommendations(analysis)
        pattern_recs = [r for r in recs if r.type == TestInsightType.PATTERNS]
        assert len(pattern_recs) == 0

    async def test_low_impact_patterns_not_triggered(self):
        svc = AITestInsightsService(config=AITestInsightsConfig(enable_ai_features=False))
        pattern = TestPatternInsight(
            pattern_type="unit_pattern",
            confidence=0.5,
            description="Minor",
            affected_tests=[],
            recommendations=[],
            impact_score=3.0,  # Below 7 threshold
        )
        analysis = _make_analysis(health_score=90.0)
        analysis.detected_patterns = [pattern]
        recs = await svc.generate_test_recommendations(analysis)
        pattern_recs = [r for r in recs if r.type == TestInsightType.PATTERNS]
        assert len(pattern_recs) == 0


# ---------------------------------------------------------------------------
# Cache delegation
# ---------------------------------------------------------------------------


class TestCacheDelegation:
    async def test_get_cached_analysis_delegates(self):
        svc = AITestInsightsService(config=AITestInsightsConfig(enable_ai_features=False))
        # No cache manager -> should return None
        result = await svc.get_cached_analysis("id-1")
        assert result is None

    async def test_get_cached_comprehensive_delegates(self):
        svc = AITestInsightsService(config=AITestInsightsConfig(enable_ai_features=False))
        result = await svc.get_cached_comprehensive_analysis("id-1")
        assert result is None


# ---------------------------------------------------------------------------
# _analyze_with_llm
# ---------------------------------------------------------------------------


class TestAnalyzeWithLLM:
    async def test_returns_none_when_disabled(self):
        svc = AITestInsightsService(
            config=AITestInsightsConfig(enable_ai_features=False, use_ml_backend=False)
        )
        result = await svc._analyze_with_llm("content", "prompt")
        assert result is None

    async def test_returns_none_when_ai_disabled(self):
        svc = AITestInsightsService(
            config=AITestInsightsConfig(enable_ai_features=False, use_ml_backend=True)
        )
        result = await svc._analyze_with_llm("content", "prompt")
        assert result is None

    async def test_returns_none_when_ml_disabled_ai_enabled(self):
        """When use_ml_backend is False but enable_ai_features is True, still returns None."""
        svc = AITestInsightsService(
            config=AITestInsightsConfig(enable_ai_features=True, use_ml_backend=False)
        )
        result = await svc._analyze_with_llm("content", "prompt")
        assert result is None
