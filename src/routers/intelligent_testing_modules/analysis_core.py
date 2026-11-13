"""
Core analysis functionality for intelligent testing.
Contains core analysis functions separated for better modularity.
"""

from __future__ import annotations

import logging
from typing import Any

from services.test_execution.intelligent_test_orchestrator import (
    IntelligentTestExecutionResult,
)

logger = logging.getLogger(__name__)


async def _identify_performance_bottlenecks(
    results: list[IntelligentTestExecutionResult],
) -> list[dict[str, Any]]:
    """Identify performance bottlenecks in test execution results.

    Args:
        results: List of test execution results

    Returns:
        List of identified bottlenecks
    """
    bottlenecks: list[dict[str, Any]] = []

    try:
        execution_times = [r.execution_time_ms for r in results if r.execution_time_ms]
        if not execution_times:
            return bottlenecks

        # Calculate threshold for slow tests (95th percentile)
        execution_times.sort()
        threshold_index = int(len(execution_times) * 0.95)
        slow_threshold = (
            execution_times[threshold_index]
            if threshold_index < len(execution_times)
            else execution_times[-1]
        )

        # Identify slow tests
        slow_tests = [
            r for r in results if r.execution_time_ms and r.execution_time_ms > slow_threshold
        ]

        if slow_tests:
            bottlenecks.append(
                {
                    "type": "slow_tests",
                    "description": f"Found {len(slow_tests)} tests slower than {slow_threshold}ms",
                    "severity": ("medium" if len(slow_tests) < len(results) * 0.1 else "high"),
                    "affected_tests": len(slow_tests),
                    "threshold": slow_threshold,
                }
            )

        # Check for resource bottlenecks
        resource_limited_tests = [r for r in results if getattr(r, "resource_limited", False)]
        if resource_limited_tests:
            bottlenecks.append(
                {
                    "type": "resource_limitation",
                    "description": f"Found {len(resource_limited_tests)} tests limited by resources",
                    "severity": "high",
                    "affected_tests": len(resource_limited_tests),
                }
            )

    except Exception as e:
        logger.error(f"Failed to identify performance bottlenecks: {e}")

    return bottlenecks


async def _generate_performance_recommendations(
    performance_summary: dict[str, Any], bottlenecks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Generate performance improvement recommendations."""
    recommendations = []

    try:
        # Check success rate
        success_rate = performance_summary.get("success_rate", 100)
        if success_rate < 95:
            recommendations.append(
                {
                    "type": "reliability",
                    "priority": "high",
                    "description": f"Success rate is {success_rate:.1f}%, consider improving test reliability",
                    "action": "Review and fix failing tests",
                }
            )

        # Check average execution time
        avg_time = performance_summary.get("average_execution_time", 0)
        if avg_time > 10000:  # 10 seconds
            recommendations.append(
                {
                    "type": "performance",
                    "priority": "medium",
                    "description": f"Average execution time is {avg_time:.0f}ms, consider optimization",
                    "action": "Optimize slow tests or increase parallelization",
                }
            )

        # Recommendations based on bottlenecks
        for bottleneck in bottlenecks:
            if bottleneck["type"] == "slow_tests":
                recommendations.append(
                    {
                        "type": "optimization",
                        "priority": bottleneck["severity"],
                        "description": "Optimize slow test execution",
                        "action": "Review and optimize tests exceeding performance thresholds",
                    }
                )
            elif bottleneck["type"] == "resource_limitation":
                recommendations.append(
                    {
                        "type": "scaling",
                        "priority": "high",
                        "description": "Increase resource allocation",
                        "action": "Scale up resources or optimize resource usage",
                    }
                )

    except Exception as e:
        logger.error(f"Failed to generate performance recommendations: {e}")

    return recommendations


async def _identify_failure_patterns(
    failed_results: list[IntelligentTestExecutionResult],
) -> list[dict[str, Any]]:
    """Identify patterns in test failures."""
    patterns = []

    try:
        # Group by language
        language_failures = {}
        for result in failed_results:
            language = getattr(result, "language", "unknown")
            if language not in language_failures:
                language_failures[language] = 0
            language_failures[language] += 1

        # Identify language-specific patterns
        for language, count in language_failures.items():
            if count > len(failed_results) * 0.3:  # More than 30% of failures
                patterns.append(
                    {
                        "type": "language_specific",
                        "pattern": f"High failure rate in {language}",
                        "severity": "medium",
                        "occurrences": count,
                        "description": f"{language} tests show higher failure rates",
                    }
                )

        # Time-based patterns (simplified - would need timestamps)
        patterns.append(
            {
                "type": "temporal",
                "pattern": "failure_clustering",
                "severity": "low",
                "description": "Analyze temporal clustering of failures for insights",
            }
        )

    except Exception as e:
        logger.error(f"Failed to identify failure patterns: {e}")

    return patterns


async def _generate_failure_recommendations(
    common_errors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate recommendations based on failure analysis."""
    recommendations = []

    try:
        for error in common_errors:
            error_type = error["error_type"]
            occurrences = error["occurrences"]

            if error_type == "timeout":
                recommendations.append(
                    {
                        "type": "timeout_optimization",
                        "priority": "high",
                        "description": f"Multiple timeout failures detected ({occurrences} times)",
                        "action": "Increase timeout values or optimize test performance",
                    }
                )
            elif error_type == "resource_error":
                recommendations.append(
                    {
                        "type": "resource_management",
                        "priority": "high",
                        "description": f"Resource errors detected ({occurrences} times)",
                        "action": "Increase resource limits or optimize resource usage",
                    }
                )
            elif error_type == "dependency_error":
                recommendations.append(
                    {
                        "type": "dependency_management",
                        "priority": "medium",
                        "description": f"Dependency errors detected ({occurrences} times)",
                        "action": "Review and update test dependencies",
                    }
                )
            else:
                recommendations.append(
                    {
                        "type": "general_reliability",
                        "priority": "medium",
                        "description": f"Recurring {error_type} errors ({occurrences} times)",
                        "action": f"Investigate and fix {error_type} error patterns",
                    }
                )

    except Exception as e:
        logger.error(f"Failed to generate failure recommendations: {e}")

    return recommendations


async def _generate_trend_recommendations(
    trend_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate recommendations based on trend analysis."""
    recommendations = []

    try:
        execution_trends = trend_data.get("execution_trends", {})
        total_executions = execution_trends.get("total_executions", 0)

        if total_executions > 1000:  # High volume
            recommendations.append(
                {
                    "type": "scaling",
                    "priority": "medium",
                    "description": "High test execution volume detected",
                    "action": "Consider scaling infrastructure for better performance",
                }
            )

        peak_hours = trend_data.get("peak_hours", [])
        if len(peak_hours) > 0:
            recommendations.append(
                {
                    "type": "scheduling",
                    "priority": "low",
                    "description": f"Peak usage detected during hours: {peak_hours}",
                    "action": "Consider load balancing or off-peak scheduling",
                }
            )

    except Exception as e:
        logger.error(f"Failed to generate trend recommendations: {e}")

    return recommendations
