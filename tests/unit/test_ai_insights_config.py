"""Unit tests for services.ai_test_insights.config module."""

from __future__ import annotations

import logging

import pytest

from services.ai_test_insights.config import (
    AITestInsightsConfig,
    AnalysisResultHandler,
    TestInsightType,
    TestPatternType,
    calculate_health_score,
    calculate_improvement_potential,
    get_health_grade,
)

# ---------------------------------------------------------------------------
# TestPatternType enum
# ---------------------------------------------------------------------------


class TestPatternTypeEnum:
    """Tests for TestPatternType StrEnum."""

    def test_all_values_exist(self):
        assert TestPatternType.UNIT_PATTERN == "unit_pattern"
        assert TestPatternType.INTEGRATION_PATTERN == "integration_pattern"
        assert TestPatternType.PERFORMANCE_PATTERN == "performance_pattern"
        assert TestPatternType.FLAKY_PATTERN == "flaky_pattern"
        assert TestPatternType.COVERAGE_PATTERN == "coverage_pattern"
        assert TestPatternType.SECURITY_PATTERN == "security_pattern"
        assert TestPatternType.RELIABILITY_PATTERN == "reliability_pattern"

    def test_is_string(self):
        assert isinstance(TestPatternType.UNIT_PATTERN, str)


# ---------------------------------------------------------------------------
# TestInsightType enum
# ---------------------------------------------------------------------------


class TestInsightTypeEnum:
    """Tests for TestInsightType StrEnum."""

    def test_all_values_exist(self):
        assert TestInsightType.PERFORMANCE == "performance"
        assert TestInsightType.RELIABILITY == "reliability"
        assert TestInsightType.COVERAGE == "coverage"
        assert TestInsightType.QUALITY == "quality"
        assert TestInsightType.PATTERNS == "patterns"
        assert TestInsightType.RECOMMENDATIONS == "recommendations"
        assert TestInsightType.PREDICTIONS == "predictions"


# ---------------------------------------------------------------------------
# AITestInsightsConfig
# ---------------------------------------------------------------------------


class TestAITestInsightsConfig:
    """Tests for AITestInsightsConfig."""

    def test_defaults(self):
        cfg = AITestInsightsConfig()
        assert cfg.slow_test_threshold == 10.0
        assert cfg.flaky_test_threshold == 0.8
        assert cfg.memory_threshold == 512
        assert cfg.cpu_threshold == 80
        assert cfg.coverage_threshold == 80
        assert cfg.cache_ttl == 86400
        assert cfg.comprehensive_cache_ttl == 172800
        assert cfg.enable_ai_features is True
        assert cfg.use_ml_backend is True
        assert cfg.analysis_weights == {
            "performance": 0.25,
            "reliability": 0.35,
            "coverage": 0.20,
            "quality": 0.20,
        }

    def test_custom_values(self):
        cfg = AITestInsightsConfig(
            slow_test_threshold=5.0,
            flaky_test_threshold=0.9,
            memory_threshold=1024,
            cpu_threshold=50,
            coverage_threshold=90,
            cache_ttl=3600,
            comprehensive_cache_ttl=7200,
            enable_ai_features=False,
            use_ml_backend=False,
            analysis_weights={
                "performance": 1.0,
                "reliability": 0.0,
                "coverage": 0.0,
                "quality": 0.0,
            },
        )
        assert cfg.slow_test_threshold == 5.0
        assert cfg.flaky_test_threshold == 0.9
        assert cfg.memory_threshold == 1024
        assert cfg.enable_ai_features is False

    def test_to_dict(self):
        cfg = AITestInsightsConfig()
        d = cfg.to_dict()
        assert d["slow_test_threshold"] == 10.0
        assert d["flaky_test_threshold"] == 0.8
        assert "analysis_weights" in d
        assert d["enable_ai_features"] is True

    def test_from_dict(self):
        d = {
            "slow_test_threshold": 20.0,
            "flaky_test_threshold": 0.5,
            "memory_threshold": 256,
            "cpu_threshold": 60,
            "coverage_threshold": 70,
            "cache_ttl": 1000,
            "comprehensive_cache_ttl": 2000,
            "enable_ai_features": False,
            "use_ml_backend": False,
        }
        cfg = AITestInsightsConfig.from_dict(d)
        assert cfg.slow_test_threshold == 20.0
        assert cfg.enable_ai_features is False

    def test_roundtrip_to_dict_from_dict(self):
        original = AITestInsightsConfig(slow_test_threshold=7.5, cache_ttl=999)
        restored = AITestInsightsConfig.from_dict(original.to_dict())
        assert restored.slow_test_threshold == 7.5
        assert restored.cache_ttl == 999


# ---------------------------------------------------------------------------
# calculate_health_score
# ---------------------------------------------------------------------------


class TestCalculateHealthScore:
    """Tests for calculate_health_score utility."""

    def test_perfect_scores(self):
        score = calculate_health_score(100, 100, 100, 100)
        assert score == 100.0

    def test_zero_scores(self):
        score = calculate_health_score(0, 0, 0, 0)
        assert score == 0.0

    def test_default_weights(self):
        # performance=0.25, reliability=0.35, coverage=0.20, quality=0.20
        score = calculate_health_score(80, 80, 80, 80)
        assert score == pytest.approx(80.0)

    def test_custom_weights(self):
        weights = {"performance": 1.0, "reliability": 0.0, "coverage": 0.0, "quality": 0.0}
        score = calculate_health_score(50, 100, 100, 100, weights=weights)
        assert score == pytest.approx(50.0)

    def test_clamped_to_100(self):
        # Weights sum to 1.0 and all scores 100, so max is 100
        score = calculate_health_score(100, 100, 100, 100)
        assert score <= 100.0

    def test_clamped_to_0(self):
        score = calculate_health_score(0, 0, 0, 0)
        assert score >= 0.0


# ---------------------------------------------------------------------------
# calculate_improvement_potential
# ---------------------------------------------------------------------------


class TestCalculateImprovementPotential:
    """Tests for calculate_improvement_potential utility."""

    def test_perfect_scores_low_potential(self):
        potential = calculate_improvement_potential(95, 95, 90, 95, 0.0)
        # All close to or above defaults — low potential
        assert potential < 20.0

    def test_low_scores_high_potential(self):
        potential = calculate_improvement_potential(20, 20, 20, 20, 10.0)
        assert potential > 30.0

    def test_pattern_impact_capped(self):
        # pattern_impact is capped at 30
        p1 = calculate_improvement_potential(50, 50, 50, 50, 30.0)
        p2 = calculate_improvement_potential(50, 50, 50, 50, 100.0)
        assert p1 == p2

    def test_custom_targets(self):
        targets = {"performance": 100, "reliability": 100, "coverage": 100, "quality": 100}
        potential = calculate_improvement_potential(0, 0, 0, 0, 0, target_scores=targets)
        assert potential > 0

    def test_result_bounded(self):
        potential = calculate_improvement_potential(0, 0, 0, 0, 30)
        assert 0 <= potential <= 100


# ---------------------------------------------------------------------------
# get_health_grade
# ---------------------------------------------------------------------------


class TestGetHealthGrade:
    """Tests for get_health_grade utility."""

    def test_grade_a(self):
        assert get_health_grade(90) == "A"
        assert get_health_grade(100) == "A"

    def test_grade_b(self):
        assert get_health_grade(80) == "B"
        assert get_health_grade(89.9) == "B"

    def test_grade_c(self):
        assert get_health_grade(70) == "C"
        assert get_health_grade(79) == "C"

    def test_grade_d(self):
        assert get_health_grade(60) == "D"
        assert get_health_grade(69) == "D"

    def test_grade_f(self):
        assert get_health_grade(0) == "F"
        assert get_health_grade(59) == "F"


# ---------------------------------------------------------------------------
# AnalysisResultHandler
# ---------------------------------------------------------------------------


class TestAnalysisResultHandler:
    """Tests for AnalysisResultHandler."""

    def test_returns_result_when_not_exception(self):
        result = AnalysisResultHandler.handle_analysis_result(
            "ok_value", "test_type", str, logging.getLogger()
        )
        assert result == "ok_value"

    def test_returns_default_performance_on_exception(self):
        from services.ai_test_insights.models import TestPerformanceMetrics

        result = AnalysisResultHandler.handle_analysis_result(
            ValueError("boom"), "performance", TestPerformanceMetrics, logging.getLogger()
        )
        assert isinstance(result, TestPerformanceMetrics)
        assert result.average_execution_time == 0.0

    def test_returns_default_reliability_on_exception(self):
        from services.ai_test_insights.models import TestReliabilityMetrics

        result = AnalysisResultHandler.handle_analysis_result(
            RuntimeError("fail"), "reliability", TestReliabilityMetrics, logging.getLogger()
        )
        assert isinstance(result, TestReliabilityMetrics)
        assert result.overall_success_rate == 0.0

    def test_returns_default_coverage_on_exception(self):
        from services.ai_test_insights.models import TestCoverageInsights

        result = AnalysisResultHandler.handle_analysis_result(
            Exception("err"), "coverage", TestCoverageInsights, logging.getLogger()
        )
        assert isinstance(result, TestCoverageInsights)
        assert result.line_coverage == 0.0

    def test_returns_default_quality_on_exception(self):
        from services.ai_test_insights.models import TestQualityInsights

        result = AnalysisResultHandler.handle_analysis_result(
            Exception("err"), "quality", TestQualityInsights, logging.getLogger()
        )
        assert isinstance(result, TestQualityInsights)
        assert result.test_quality_score == 0.0

    def test_returns_empty_list_for_unknown_type(self):
        result = AnalysisResultHandler.handle_analysis_result(
            Exception("err"), "unknown", dict, logging.getLogger()
        )
        assert result == []
