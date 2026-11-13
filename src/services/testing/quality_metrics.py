# /src/app/services/testing/quality_metrics.py
"""
Quality Metric Calculation Module
Handles calculation of quality scores and metrics for code analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from models.testing_models import QualityIssue

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics data structure.

    Attributes:
        overall_score (float): Overall quality score (0-10).
        total_issues (int): Total number of issues found.
        severity_breakdown (Dict[str, int]): Issues grouped by severity.
        tool_breakdown (Dict[str, int]): Issues grouped by tool.
        file_breakdown (Dict[str, int]): Issues grouped by file.
        quality_grade (str): Letter grade (A-F) based on score.
        improvement_potential (float): Potential score improvement percentage.
    """

    overall_score: float
    total_issues: int
    severity_breakdown: dict[str, int]
    tool_breakdown: dict[str, int]
    file_breakdown: dict[str, int]
    quality_grade: str
    improvement_potential: float


class QualityScoreCalculator:
    """Calculator for quality scores and metrics."""

    def __init__(self):
        self.severity_weights = {
            "error": 2.0,
            "warning": 1.0,
            "info": 0.3,
            "style": 0.2,
            "convention": 0.1,
        }

        self.tool_weights = {
            "ruff": 1.0,
            "ty": 1.5,
            "bandit": 2.0,
            "ruff-format": 0.5,
            "eslint": 1.0,
            "prettier": 0.3,
            "tsc": 1.5,
        }

    def calculate_quality_score(
        self, tools_results: dict[str, Any], issues: list[QualityIssue]
    ) -> float:
        """Calculate overall quality score based on tools results and issues.

        Args:
            tools_results: Results from quality tools execution.
            issues: List of quality issues found.

        Returns:
            Quality score between 0 and 10.
        """
        if not tools_results:
            return 0.0

        # Start with perfect score
        base_score = 10.0

        # Calculate weighted penalty for issues
        total_penalty = 0.0

        for issue in issues:
            severity_weight = self.severity_weights.get(issue.severity.lower(), 0.5)
            tool_weight = self.tool_weights.get(issue.tool.lower(), 1.0)

            # Apply compound penalty with diminishing returns
            penalty = severity_weight * tool_weight
            total_penalty += penalty

        # Apply logarithmic scaling to prevent excessive penalties
        import math

        scaled_penalty = math.log1p(total_penalty) * 0.8

        # Calculate final score with minimum threshold
        final_score = max(0.0, base_score - scaled_penalty)

        return round(final_score, 2)

    def calculate_comprehensive_metrics(
        self, tools_results: dict[str, Any], issues: list[QualityIssue]
    ) -> QualityMetrics:
        """Calculate comprehensive quality metrics.

        Args:
            tools_results: Results from quality tools execution.
            issues: List of quality issues found.

        Returns:
            Comprehensive quality metrics.
        """
        overall_score = self.calculate_quality_score(tools_results, issues)

        # Calculate severity breakdown
        severity_breakdown = {}
        for issue in issues:
            severity = issue.severity.lower()
            severity_breakdown[severity] = severity_breakdown.get(severity, 0) + 1

        # Calculate tool breakdown
        tool_breakdown = {}
        for issue in issues:
            tool = issue.tool
            tool_breakdown[tool] = tool_breakdown.get(tool, 0) + 1

        # Calculate file breakdown
        file_breakdown = {}
        for issue in issues:
            file_path = issue.file_path or "unknown"
            file_breakdown[file_path] = file_breakdown.get(file_path, 0) + 1

        # Calculate quality grade
        quality_grade = self._calculate_grade(overall_score)

        # Calculate improvement potential
        improvement_potential = self._calculate_improvement_potential(overall_score, issues)

        return QualityMetrics(
            overall_score=overall_score,
            total_issues=len(issues),
            severity_breakdown=severity_breakdown,
            tool_breakdown=tool_breakdown,
            file_breakdown=file_breakdown,
            quality_grade=quality_grade,
            improvement_potential=improvement_potential,
        )

    def _calculate_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 9.0:
            return "A+"
        elif score >= 8.5:
            return "A"
        elif score >= 8.0:
            return "A-"
        elif score >= 7.5:
            return "B+"
        elif score >= 7.0:
            return "B"
        elif score >= 6.5:
            return "B-"
        elif score >= 6.0:
            return "C+"
        elif score >= 5.5:
            return "C"
        elif score >= 5.0:
            return "C-"
        elif score >= 4.0:
            return "D"
        else:
            return "F"

    def _calculate_improvement_potential(
        self, current_score: float, issues: list[QualityIssue]
    ) -> float:
        """Calculate potential improvement percentage."""
        if current_score >= 9.5:
            return 0.0

        # Count fixable issues
        fixable_issues = [
            issue for issue in issues if issue.tool in ["ruff", "ruff-format", "prettier"]
        ]

        if not fixable_issues:
            return min(10.0, (10.0 - current_score) * 0.3)

        # Estimate improvement from fixable issues
        fixable_penalty = sum(
            self.severity_weights.get(issue.severity.lower(), 0.5) for issue in fixable_issues
        )

        potential_improvement = min(fixable_penalty * 0.8, 10.0 - current_score)

        return round((potential_improvement / (10.0 - current_score)) * 100, 1)


class QualityTrendAnalyzer:
    """Analyzer for quality trends over time."""

    def __init__(self):
        self.trend_window_days = 30

    def calculate_trend_metrics(self, historical_scores: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate quality trend metrics.

        Args:
            historical_scores: List of historical quality scores with timestamps.

        Returns:
            Trend analysis metrics.
        """
        if len(historical_scores) < 2:
            return {
                "trend_direction": "insufficient_data",
                "trend_percentage": 0.0,
                "volatility": 0.0,
                "stability_score": 0.0,
            }

        scores = [entry["score"] for entry in historical_scores]

        # Calculate trend direction
        trend_percentage = self._calculate_trend_percentage(scores)

        # Calculate volatility (standard deviation)
        volatility = self._calculate_volatility(scores)

        # Calculate stability score
        stability_score = self._calculate_stability_score(scores, volatility)

        trend_direction = (
            "improving"
            if trend_percentage > 1.0
            else "declining"
            if trend_percentage < -1.0
            else "stable"
        )

        return {
            "trend_direction": trend_direction,
            "trend_percentage": trend_percentage,
            "volatility": volatility,
            "stability_score": stability_score,
            "average_score": sum(scores) / len(scores),
            "best_score": max(scores),
            "worst_score": min(scores),
        }

    def _calculate_trend_percentage(self, scores: list[float]) -> float:
        """Calculate trend percentage using linear regression."""
        n = len(scores)
        if n < 2:
            return 0.0

        # Simple linear regression
        x_values = list(range(n))
        x_mean = sum(x_values) / n
        y_mean = sum(scores) / n

        numerator = sum((x_values[i] - x_mean) * (scores[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        slope = numerator / denominator

        # Convert slope to percentage change
        if y_mean > 0:
            return (slope * (n - 1) / y_mean) * 100
        else:
            return 0.0

    def _calculate_volatility(self, scores: list[float]) -> float:
        """Calculate score volatility."""
        if len(scores) < 2:
            return 0.0

        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)

        import math

        return math.sqrt(variance)

    def _calculate_stability_score(self, scores: list[float], volatility: float) -> float:
        """Calculate stability score (0-10)."""
        if not scores:
            return 0.0

        max_possible_volatility = 10.0  # Maximum score range
        stability = max(0.0, 10.0 - (volatility / max_possible_volatility) * 10.0)

        return round(stability, 2)


class QualityBenchmarkComparator:
    """Compares quality metrics against benchmarks."""

    def __init__(self):
        self.language_benchmarks = {
            "python": {
                "excellent": 8.5,
                "good": 7.0,
                "acceptable": 5.5,
                "needs_improvement": 4.0,
            },
            "javascript": {
                "excellent": 8.0,
                "good": 6.5,
                "acceptable": 5.0,
                "needs_improvement": 3.5,
            },
            "typescript": {
                "excellent": 8.5,
                "good": 7.0,
                "acceptable": 5.5,
                "needs_improvement": 4.0,
            },
        }

    def compare_to_benchmarks(self, score: float, language: str) -> dict[str, Any]:
        """Compare score to language-specific benchmarks.

        Args:
            score: Quality score to compare.
            language: Programming language.

        Returns:
            Benchmark comparison results.
        """
        benchmarks = self.language_benchmarks.get(
            language.lower(), self.language_benchmarks["python"]
        )

        # Determine quality level
        if score >= benchmarks["excellent"]:
            level = "excellent"
            percentile = 95
        elif score >= benchmarks["good"]:
            level = "good"
            percentile = 75
        elif score >= benchmarks["acceptable"]:
            level = "acceptable"
            percentile = 50
        elif score >= benchmarks["needs_improvement"]:
            level = "needs_improvement"
            percentile = 25
        else:
            level = "poor"
            percentile = 10

        # Calculate gap to next level
        next_level_score = None
        for level_name, threshold in benchmarks.items():
            if score < threshold:
                next_level_score = threshold
                break

        gap_to_next_level = (next_level_score - score) if next_level_score else 0.0

        return {
            "quality_level": level,
            "estimated_percentile": percentile,
            "benchmarks": benchmarks,
            "gap_to_next_level": gap_to_next_level,
            "next_level_score": next_level_score,
        }


# Factory function for easy access
def get_quality_metrics_calculator() -> QualityScoreCalculator:
    """Get a quality metrics calculator instance."""
    return QualityScoreCalculator()


def get_quality_trend_analyzer() -> QualityTrendAnalyzer:
    """Get a quality trend analyzer instance."""
    return QualityTrendAnalyzer()


def get_quality_benchmark_comparator() -> QualityBenchmarkComparator:
    """Get a quality benchmark comparator instance."""
    return QualityBenchmarkComparator()
