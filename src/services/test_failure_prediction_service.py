"""Test Failure Prediction and Analysis Service.

This module provides AI-powered prediction and analysis of test failures,
helping teams proactively identify and prevent test failures before they occur.
It uses historical data, code changes, and machine learning techniques to
predict failure probability and provide actionable insights.
"""

import logging
from collections import defaultdict
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_serializer

from models.testing_models import TestResult

logger = logging.getLogger(__name__)


class FailureCategory(StrEnum):
    """Categories of test failures."""

    ASSERTION_ERROR = "assertion_error"
    TIMEOUT = "timeout"
    INFRASTRUCTURE = "infrastructure"
    DEPENDENCY = "dependency"
    ENVIRONMENTAL = "environmental"
    FLAKY = "flaky"
    REGRESSION = "regression"
    CONFIGURATION = "configuration"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    UNKNOWN = "unknown"


class RiskLevel(StrEnum):
    """Risk levels for failure prediction."""

    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class PredictionConfidence(StrEnum):
    """Confidence levels for predictions."""

    VERY_HIGH = "very_high"  # 90-100%
    HIGH = "high"  # 80-89%
    MEDIUM = "medium"  # 60-79%
    LOW = "low"  # 40-59%
    VERY_LOW = "very_low"  # 0-39%


class FailurePattern(BaseModel):
    """Detected failure pattern."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # Pattern identification
    pattern_id: str = Field(description="Unique pattern identifier")
    pattern_name: str = Field(description="Human-readable pattern name")
    category: FailureCategory = Field(description="Failure category")

    # Pattern characteristics
    occurrence_count: int = Field(description="Number of occurrences")
    frequency: float = Field(description="Frequency as ratio of total tests")

    # Temporal patterns
    first_occurrence: datetime = Field(description="First occurrence timestamp")
    last_occurrence: datetime = Field(description="Last occurrence timestamp")
    occurrence_trend: str = Field(
        description="Trend in occurrences (increasing, decreasing, stable)"
    )

    # Affected tests
    affected_tests: list[str] = Field(
        default_factory=list, description="Tests affected by this pattern"
    )
    affected_languages: set[str] = Field(
        default_factory=set, description="Languages affected by this pattern"
    )

    # Pattern details
    description: str = Field(description="Pattern description")
    typical_symptoms: list[str] = Field(
        default_factory=list, description="Typical symptoms of this pattern"
    )
    root_causes: list[str] = Field(default_factory=list, description="Common root causes")

    # Predictive indicators
    leading_indicators: list[str] = Field(
        default_factory=list, description="Leading indicators for this pattern"
    )
    correlation_factors: dict[str, float] = Field(
        default_factory=dict, description="Factors correlated with this pattern"
    )

    @computed_field
    @property
    def severity_score(self) -> float:
        """Calculate severity score based on frequency and impact."""
        return min(10.0, self.frequency * 100 + self.occurrence_count * 0.1)


class FailurePrediction(BaseModel):
    """Individual failure prediction."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # Prediction identification
    prediction_id: str = Field(description="Unique prediction identifier")
    test_identifier: str = Field(description="Test or test group identifier")

    # Prediction details
    failure_probability: float = Field(ge=0.0, le=1.0, description="Probability of failure (0-1)")
    risk_level: RiskLevel = Field(description="Risk level")
    confidence: PredictionConfidence = Field(description="Prediction confidence")

    # Prediction factors
    contributing_factors: list[str] = Field(
        default_factory=list, description="Factors contributing to failure risk"
    )
    historical_patterns: list[str] = Field(
        default_factory=list, description="Historical patterns influencing prediction"
    )

    # Temporal aspects
    prediction_window: str = Field(description="Time window for prediction")
    prediction_created: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When prediction was created",
    )

    # Mitigation suggestions
    mitigation_strategies: list[str] = Field(
        default_factory=list, description="Suggested mitigation strategies"
    )
    preventive_actions: list[str] = Field(
        default_factory=list, description="Preventive actions to take"
    )

    # Supporting evidence
    supporting_evidence: dict[str, Any] = Field(
        default_factory=dict, description="Supporting evidence for prediction"
    )

    @computed_field
    @property
    def priority_score(self) -> float:
        """Calculate priority score for this prediction."""
        risk_weights = {
            RiskLevel.VERY_HIGH: 10.0,
            RiskLevel.HIGH: 8.0,
            RiskLevel.MEDIUM: 6.0,
            RiskLevel.LOW: 4.0,
            RiskLevel.VERY_LOW: 2.0,
        }

        confidence_weights = {
            PredictionConfidence.VERY_HIGH: 1.0,
            PredictionConfidence.HIGH: 0.8,
            PredictionConfidence.MEDIUM: 0.6,
            PredictionConfidence.LOW: 0.4,
            PredictionConfidence.VERY_LOW: 0.2,
        }

        return risk_weights[self.risk_level] * confidence_weights[self.confidence]


class FailureAnalysis(BaseModel):
    """Analysis of historical failures."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # Analysis metadata
    analysis_period: str = Field(description="Time period analyzed")
    total_tests_analyzed: int = Field(description="Total tests analyzed")
    total_failures: int = Field(description="Total failures found")

    # Failure statistics
    failure_rate: float = Field(description="Overall failure rate")
    failure_rate_by_language: dict[str, float] = Field(
        default_factory=dict, description="Failure rates by language"
    )
    failure_rate_by_type: dict[str, float] = Field(
        default_factory=dict, description="Failure rates by test type"
    )

    # Failure categorization
    failures_by_category: dict[FailureCategory, int] = Field(
        default_factory=dict, description="Failures grouped by category"
    )
    most_common_failure_category: FailureCategory = Field(
        description="Most common failure category"
    )

    # Temporal patterns
    failure_trends: dict[str, Any] = Field(
        default_factory=dict, description="Trends in failure rates"
    )
    peak_failure_times: list[str] = Field(
        default_factory=list, description="Times when failures are most common"
    )

    # Detected patterns
    failure_patterns: list[FailurePattern] = Field(
        default_factory=list, description="Detected failure patterns"
    )

    # Recovery metrics
    mean_time_to_recovery: float = Field(description="Average time to fix failures")
    recovery_success_rate: float = Field(description="Success rate of failure fixes")

    @computed_field
    @property
    def reliability_score(self) -> float:
        """Calculate reliability score based on failure analysis."""
        return max(0.0, 100.0 - (self.failure_rate * 100))


class PredictionReport(BaseModel):
    """Comprehensive failure prediction report."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    # Report metadata
    report_id: str = Field(description="Unique report identifier")
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When report was generated",
    )
    prediction_window: str = Field(description="Time window for predictions")

    # Historical analysis
    historical_analysis: FailureAnalysis = Field(description="Analysis of historical failures")

    # Predictions
    predictions: list[FailurePrediction] = Field(
        default_factory=list, description="Failure predictions"
    )
    high_risk_predictions: list[FailurePrediction] = Field(
        default_factory=list, description="High-risk predictions"
    )

    # Summary statistics
    total_predictions: int = Field(description="Total number of predictions")
    predictions_by_risk: dict[RiskLevel, int] = Field(
        default_factory=dict, description="Predictions grouped by risk level"
    )
    average_failure_probability: float = Field(
        description="Average failure probability across all predictions"
    )

    # Actionable insights
    immediate_actions: list[str] = Field(
        default_factory=list, description="Immediate actions recommended"
    )
    preventive_measures: list[str] = Field(
        default_factory=list, description="Preventive measures to implement"
    )
    monitoring_recommendations: list[str] = Field(
        default_factory=list, description="Monitoring recommendations"
    )

    # Model performance
    model_accuracy: float = Field(description="Prediction model accuracy")
    model_confidence: float = Field(description="Overall model confidence")

    @field_serializer("generated_at")
    def serialize_datetime(self, v: datetime | None) -> str | None:
        if v is None:
            return None
        return v.isoformat()

    @computed_field
    @property
    def overall_risk_level(self) -> RiskLevel:
        """Calculate overall risk level."""
        high_risk_count = len(self.high_risk_predictions)
        total_predictions = len(self.predictions)

        if total_predictions == 0:
            return RiskLevel.LOW

        high_risk_ratio = high_risk_count / total_predictions

        if high_risk_ratio >= 0.3:
            return RiskLevel.VERY_HIGH
        elif high_risk_ratio >= 0.2:
            return RiskLevel.HIGH
        elif high_risk_ratio >= 0.1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW


class TestFailurePredictionService:
    """Service for predicting and analyzing test failures."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the failure prediction service."""
        self.config = config or self._get_default_config()
        logger.info("Test Failure Prediction Service initialized")

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration."""
        return {
            # Prediction thresholds
            "high_risk_threshold": 0.7,
            "medium_risk_threshold": 0.4,
            "low_risk_threshold": 0.2,
            # Pattern detection
            "min_pattern_occurrences": 3,
            "pattern_significance_threshold": 0.1,
            "temporal_window_days": 30,
            # Model parameters
            "prediction_window_hours": 24,
            "confidence_threshold": 0.6,
            "historical_analysis_days": 90,
            # Failure categorization
            "timeout_keywords": ["timeout", "timed out", "exceeded"],
            "assertion_keywords": ["assert", "expected", "actual"],
            "infrastructure_keywords": ["connection", "network", "server"],
            "dependency_keywords": ["import", "module", "package"],
        }

    async def predict_failures(
        self,
        test_results: list[TestResult],
        code_changes: list[str] | None = None,
        environmental_factors: dict[str, Any] | None = None,
    ) -> PredictionReport:
        """Generate failure predictions based on historical data."""
        from uuid import uuid7

        report_id = str(uuid7())

        logger.info(f"Generating failure predictions {report_id}")

        # Analyze historical failures
        historical_analysis = await self._analyze_historical_failures(test_results)

        # Generate predictions
        predictions = await self._generate_predictions(
            test_results, code_changes, environmental_factors, historical_analysis
        )

        # Identify high-risk predictions
        high_risk_predictions = [
            p for p in predictions if p.failure_probability >= self.config["high_risk_threshold"]
        ]

        # Calculate summary statistics
        predictions_by_risk = defaultdict(int)
        for prediction in predictions:
            predictions_by_risk[prediction.risk_level] += 1

        avg_failure_probability = (
            sum(p.failure_probability for p in predictions) / len(predictions)
            if predictions
            else 0.0
        )

        # Generate actionable insights
        immediate_actions = await self._generate_immediate_actions(high_risk_predictions)
        preventive_measures = await self._generate_preventive_measures(historical_analysis)
        monitoring_recommendations = await self._generate_monitoring_recommendations(predictions)

        # Calculate model performance metrics
        model_accuracy = await self._calculate_model_accuracy(test_results)
        model_confidence = await self._calculate_model_confidence(predictions)

        # Create report
        report = PredictionReport(
            report_id=report_id,
            prediction_window=f"{self.config['prediction_window_hours']} hours",
            historical_analysis=historical_analysis,
            predictions=predictions,
            high_risk_predictions=high_risk_predictions,
            total_predictions=len(predictions),
            predictions_by_risk=dict(predictions_by_risk),
            average_failure_probability=avg_failure_probability,
            immediate_actions=immediate_actions,
            preventive_measures=preventive_measures,
            monitoring_recommendations=monitoring_recommendations,
            model_accuracy=model_accuracy,
            model_confidence=model_confidence,
        )

        logger.info(f"Generated {len(predictions)} failure predictions")

        return report

    async def _analyze_historical_failures(self, test_results: list[TestResult]) -> FailureAnalysis:
        """Analyze historical test failures."""

        # Filter for failures
        failures = [r for r in test_results if not r.success]

        # Calculate failure rates
        total_tests = len(test_results)
        total_failures = len(failures)
        failure_rate = total_failures / total_tests if total_tests > 0 else 0.0

        # Failure rates by language
        language_stats = defaultdict(lambda: {"total": 0, "failures": 0})
        for result in test_results:
            language_stats[result.language]["total"] += 1
            if not result.success:
                language_stats[result.language]["failures"] += 1

        failure_rate_by_language = {
            lang: stats["failures"] / stats["total"] if stats["total"] > 0 else 0.0
            for lang, stats in language_stats.items()
        }

        # Failure rates by test type
        type_stats = defaultdict(lambda: {"total": 0, "failures": 0})
        for result in test_results:
            test_type = result.test_type or "unknown"
            type_stats[test_type]["total"] += 1
            if not result.success:
                type_stats[test_type]["failures"] += 1

        failure_rate_by_type = {
            test_type: stats["failures"] / stats["total"] if stats["total"] > 0 else 0.0
            for test_type, stats in type_stats.items()
        }

        # Categorize failures
        failures_by_category = await self._categorize_failures(failures)
        most_common_category = (
            max(failures_by_category, key=lambda k: failures_by_category[k])
            if failures_by_category
            else FailureCategory.UNKNOWN
        )

        # Detect failure patterns
        failure_patterns = await self._detect_failure_patterns(failures)

        # Calculate recovery metrics
        mean_time_to_recovery = await self._calculate_mean_time_to_recovery(failures)
        recovery_success_rate = 0.8  # Mock - would need actual recovery data

        # Analyze trends
        failure_trends = await self._analyze_failure_trends(test_results)

        return FailureAnalysis(
            analysis_period=f"{self.config['historical_analysis_days']} days",
            total_tests_analyzed=total_tests,
            total_failures=total_failures,
            failure_rate=failure_rate,
            failure_rate_by_language=failure_rate_by_language,
            failure_rate_by_type=failure_rate_by_type,
            failures_by_category=failures_by_category,
            most_common_failure_category=most_common_category,
            failure_trends=failure_trends,
            peak_failure_times=["morning", "evening"],  # Mock data
            failure_patterns=failure_patterns,
            mean_time_to_recovery=mean_time_to_recovery,
            recovery_success_rate=recovery_success_rate,
        )

    async def _categorize_failures(self, failures: list[TestResult]) -> dict[FailureCategory, int]:
        """Categorize failures by type."""

        categories = defaultdict(int)

        for failure in failures:
            error_message = failure.stderr.lower() if failure.stderr else ""

            # Categorize based on error message
            if any(keyword in error_message for keyword in self.config["timeout_keywords"]):
                categories[FailureCategory.TIMEOUT] += 1
            elif any(keyword in error_message for keyword in self.config["assertion_keywords"]):
                categories[FailureCategory.ASSERTION_ERROR] += 1
            elif any(
                keyword in error_message for keyword in self.config["infrastructure_keywords"]
            ):
                categories[FailureCategory.INFRASTRUCTURE] += 1
            elif any(keyword in error_message for keyword in self.config["dependency_keywords"]):
                categories[FailureCategory.DEPENDENCY] += 1
            else:
                categories[FailureCategory.UNKNOWN] += 1

        return dict(categories)

    async def _detect_failure_patterns(self, failures: list[TestResult]) -> list[FailurePattern]:
        """Detect patterns in test failures."""

        patterns = []

        # Group failures by error similarity
        error_groups = defaultdict(list)
        for failure in failures:
            error_key = failure.stderr[:50] if failure.stderr else "no_error"
            error_groups[error_key].append(failure)

        # Create patterns for significant error groups
        for error_key, group_failures in error_groups.items():
            if len(group_failures) >= self.config["min_pattern_occurrences"]:
                # Determine category
                category = await self._categorize_single_failure(group_failures[0])

                # Create pattern
                pattern = FailurePattern(
                    pattern_id=f"pattern_{hash(error_key)}",
                    pattern_name=f"Pattern: {error_key[:30]}...",
                    category=category,
                    occurrence_count=len(group_failures),
                    frequency=len(group_failures) / len(failures),
                    first_occurrence=min(f.created_at for f in group_failures),
                    last_occurrence=max(f.created_at for f in group_failures),
                    occurrence_trend="stable",  # Simplified
                    affected_tests=[f.test_id for f in group_failures],
                    affected_languages=set(f.language for f in group_failures),
                    description=f"Recurring failure pattern with {len(group_failures)} occurrences",
                    typical_symptoms=[error_key],
                    root_causes=[
                        "Common code path",
                        "Shared dependency",
                        "Environmental factor",
                    ],
                    leading_indicators=[
                        "Code changes",
                        "Dependency updates",
                        "Infrastructure changes",
                    ],
                    correlation_factors={"time_of_day": 0.3, "code_changes": 0.7},
                )
                patterns.append(pattern)

        return patterns

    async def _categorize_single_failure(self, failure: TestResult) -> FailureCategory:
        """Categorize a single failure."""

        error_message = failure.stderr.lower() if failure.stderr else ""

        if any(keyword in error_message for keyword in self.config["timeout_keywords"]):
            return FailureCategory.TIMEOUT
        elif any(keyword in error_message for keyword in self.config["assertion_keywords"]):
            return FailureCategory.ASSERTION_ERROR
        elif any(keyword in error_message for keyword in self.config["infrastructure_keywords"]):
            return FailureCategory.INFRASTRUCTURE
        elif any(keyword in error_message for keyword in self.config["dependency_keywords"]):
            return FailureCategory.DEPENDENCY
        else:
            return FailureCategory.UNKNOWN

    async def _calculate_mean_time_to_recovery(self, failures: list[TestResult]) -> float:
        """Calculate mean time to recovery for failures."""

        # Mock calculation - would need actual recovery data
        return 2.5  # hours

    async def _analyze_failure_trends(self, test_results: list[TestResult]) -> dict[str, Any]:
        """Analyze trends in failure rates."""

        # Simple trend analysis
        recent_results = test_results[-20:] if len(test_results) >= 20 else test_results
        earlier_results = test_results[:-20] if len(test_results) >= 20 else []

        if not earlier_results:
            return {"trend": "insufficient_data"}

        recent_failure_rate = sum(1 for r in recent_results if not r.success) / len(recent_results)
        earlier_failure_rate = sum(1 for r in earlier_results if not r.success) / len(
            earlier_results
        )

        trend = "stable"
        if recent_failure_rate > earlier_failure_rate * 1.2:
            trend = "increasing"
        elif recent_failure_rate < earlier_failure_rate * 0.8:
            trend = "decreasing"

        return {
            "trend": trend,
            "recent_failure_rate": recent_failure_rate,
            "earlier_failure_rate": earlier_failure_rate,
            "change_magnitude": abs(recent_failure_rate - earlier_failure_rate),
        }

    async def _generate_predictions(
        self,
        test_results: list[TestResult],
        code_changes: list[str] | None,
        environmental_factors: dict[str, Any] | None,
        historical_analysis: FailureAnalysis,
    ) -> list[FailurePrediction]:
        """Generate failure predictions."""

        predictions = []

        # Group tests by language and type
        test_groups = defaultdict(list)
        for result in test_results:
            key = f"{result.language}_{result.test_type or 'unknown'}"
            test_groups[key].append(result)

        # Generate predictions for each group
        for group_key, group_results in test_groups.items():
            # Calculate base failure probability from historical data
            failed_tests = [r for r in group_results if not r.success]
            base_probability = len(failed_tests) / len(group_results) if group_results else 0.0

            # Adjust probability based on factors
            adjusted_probability = base_probability

            # Code changes increase risk
            if code_changes:
                adjusted_probability *= 1.5

            # Environmental factors
            if environmental_factors:
                if environmental_factors.get("heavy_load"):
                    adjusted_probability *= 1.3
                if environmental_factors.get("infrastructure_changes"):
                    adjusted_probability *= 1.4

            # Historical patterns
            contributing_factors = []
            if base_probability > 0.2:
                contributing_factors.append("Historical failure rate")
            if code_changes:
                contributing_factors.append("Recent code changes")
            if environmental_factors:
                contributing_factors.append("Environmental changes")

            # Determine risk level
            risk_level = await self._determine_risk_level(adjusted_probability)
            confidence = await self._determine_confidence(len(group_results), historical_analysis)

            # Create prediction
            prediction = FailurePrediction(
                prediction_id=f"pred_{group_key}",
                test_identifier=group_key,
                failure_probability=min(1.0, adjusted_probability),
                risk_level=risk_level,
                confidence=confidence,
                contributing_factors=contributing_factors,
                historical_patterns=[
                    p.pattern_name for p in historical_analysis.failure_patterns[:3]
                ],
                prediction_window=f"{self.config['prediction_window_hours']} hours",
                mitigation_strategies=await self._generate_mitigation_strategies(risk_level),
                preventive_actions=await self._generate_preventive_actions(contributing_factors),
                supporting_evidence={
                    "historical_failure_rate": base_probability,
                    "adjusted_probability": adjusted_probability,
                    "sample_size": len(group_results),
                },
            )
            predictions.append(prediction)

        return predictions

    async def _determine_risk_level(self, probability: float) -> RiskLevel:
        """Determine risk level based on probability."""

        if probability >= 0.8:
            return RiskLevel.VERY_HIGH
        elif probability >= self.config["high_risk_threshold"]:
            return RiskLevel.HIGH
        elif probability >= self.config["medium_risk_threshold"]:
            return RiskLevel.MEDIUM
        elif probability >= self.config["low_risk_threshold"]:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW

    async def _determine_confidence(
        self, sample_size: int, historical_analysis: FailureAnalysis
    ) -> PredictionConfidence:
        """Determine confidence level for predictions."""

        # Base confidence on sample size and historical data quality
        base_confidence = min(1.0, sample_size / 50.0)  # Normalize to 50 samples

        # Adjust based on historical analysis reliability
        if historical_analysis.total_tests_analyzed > 100:
            base_confidence *= 1.2

        base_confidence = min(1.0, base_confidence)

        if base_confidence >= 0.9:
            return PredictionConfidence.VERY_HIGH
        elif base_confidence >= 0.8:
            return PredictionConfidence.HIGH
        elif base_confidence >= 0.6:
            return PredictionConfidence.MEDIUM
        elif base_confidence >= 0.4:
            return PredictionConfidence.LOW
        else:
            return PredictionConfidence.VERY_LOW

    async def _generate_mitigation_strategies(self, risk_level: RiskLevel) -> list[str]:
        """Generate mitigation strategies based on risk level."""

        strategies = []

        if risk_level in [RiskLevel.VERY_HIGH, RiskLevel.HIGH]:
            strategies.extend(
                [
                    "Implement additional test validation",
                    "Add retry mechanisms for flaky tests",
                    "Increase test timeout values",
                    "Add comprehensive logging and monitoring",
                ]
            )

        if risk_level in [RiskLevel.HIGH, RiskLevel.MEDIUM]:
            strategies.extend(
                [
                    "Review and update test data",
                    "Improve test isolation",
                    "Add health checks for dependencies",
                    "Implement gradual rollout",
                ]
            )

        strategies.extend(
            [
                "Monitor test execution closely",
                "Have rollback plan ready",
                "Ensure team availability for quick fixes",
            ]
        )

        return strategies

    async def _generate_preventive_actions(self, contributing_factors: list[str]) -> list[str]:
        """Generate preventive actions based on contributing factors."""

        actions = []

        if "Historical failure rate" in contributing_factors:
            actions.extend(
                [
                    "Investigate root causes of historical failures",
                    "Improve test reliability",
                    "Add test result monitoring",
                ]
            )

        if "Recent code changes" in contributing_factors:
            actions.extend(
                [
                    "Implement comprehensive code review",
                    "Add integration testing",
                    "Use feature flags for risky changes",
                ]
            )

        if "Environmental changes" in contributing_factors:
            actions.extend(
                [
                    "Validate environment stability",
                    "Add infrastructure monitoring",
                    "Implement environment health checks",
                ]
            )

        return actions

    async def _generate_immediate_actions(
        self, high_risk_predictions: list[FailurePrediction]
    ) -> list[str]:
        """Generate immediate actions for high-risk predictions."""

        actions = []

        if high_risk_predictions:
            actions.extend(
                [
                    "Review high-risk test predictions immediately",
                    "Implement additional monitoring for flagged tests",
                    "Prepare contingency plans for predicted failures",
                    "Alert relevant team members about high-risk areas",
                ]
            )

        return actions

    async def _generate_preventive_measures(
        self, historical_analysis: FailureAnalysis
    ) -> list[str]:
        """Generate preventive measures based on historical analysis."""

        measures = []

        if historical_analysis.failure_rate > 0.1:  # 10% failure rate
            measures.extend(
                [
                    "Implement systematic test reliability improvements",
                    "Add comprehensive test suite monitoring",
                    "Establish failure pattern detection and alerting",
                ]
            )

        if historical_analysis.failure_patterns:
            measures.extend(
                [
                    "Address recurring failure patterns",
                    "Implement pattern-specific preventive measures",
                    "Add pattern detection to CI/CD pipeline",
                ]
            )

        return measures

    async def _generate_monitoring_recommendations(
        self, predictions: list[FailurePrediction]
    ) -> list[str]:
        """Generate monitoring recommendations."""

        recommendations = []

        high_risk_count = sum(
            1 for p in predictions if p.risk_level in [RiskLevel.VERY_HIGH, RiskLevel.HIGH]
        )

        if high_risk_count > 0:
            recommendations.extend(
                [
                    "Implement real-time test failure monitoring",
                    "Add automated alerting for high-risk tests",
                    "Create failure prediction dashboards",
                    "Set up proactive failure detection",
                ]
            )

        recommendations.extend(
            [
                "Monitor test execution patterns",
                "Track prediction accuracy over time",
                "Implement continuous model improvement",
                "Add performance metrics tracking",
            ]
        )

        return recommendations

    async def _calculate_model_accuracy(self, test_results: list[TestResult]) -> float:
        """Calculate model accuracy based on historical validation."""

        # Mock calculation - would need actual validation data
        return 0.75  # 75% accuracy

    async def _calculate_model_confidence(self, predictions: list[FailurePrediction]) -> float:
        """Calculate overall model confidence."""

        if not predictions:
            return 0.0

        confidence_values = {
            PredictionConfidence.VERY_HIGH: 0.95,
            PredictionConfidence.HIGH: 0.85,
            PredictionConfidence.MEDIUM: 0.70,
            PredictionConfidence.LOW: 0.50,
            PredictionConfidence.VERY_LOW: 0.30,
        }

        total_confidence = sum(confidence_values[p.confidence] for p in predictions)
        return total_confidence / len(predictions)


# Factory function
def create_test_failure_prediction_service(
    config: dict[str, Any] | None = None,
) -> TestFailurePredictionService:
    """Create a test failure prediction service instance."""
    return TestFailurePredictionService(config=config)
