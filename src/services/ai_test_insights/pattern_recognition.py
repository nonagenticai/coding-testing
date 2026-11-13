"""Test pattern recognition module for AI Test Insights Service.

This module contains logic for detecting various test patterns and anti-patterns
to help improve test quality and maintainability.
"""

import logging
from typing import Any

from services.ai_test_insights.config import (
    DEFAULT_CONFIG,
    AITestInsightsConfig,
    TestPatternType,
)
from services.ai_test_insights.models import TestPatternInsight

logger = logging.getLogger(__name__)


class TestPatternDetector:
    """Detects various test patterns and anti-patterns."""

    def __init__(self, config: AITestInsightsConfig = DEFAULT_CONFIG):
        """Initialize the pattern detector.

        Args:
            config: Configuration for the AI test insights service
        """
        self.config = config

    async def detect_patterns(self, data: dict[str, Any]) -> list[TestPatternInsight]:
        """Detect test patterns using analysis data.

        Args:
            data: Prepared analysis data containing test results and metrics

        Returns:
            List of detected test pattern insights
        """
        patterns = []

        # Detect performance patterns
        performance_patterns = await self._detect_performance_patterns(data)
        patterns.extend(performance_patterns)

        # Detect reliability patterns
        reliability_patterns = await self._detect_reliability_patterns(data)
        patterns.extend(reliability_patterns)

        # Detect coverage patterns
        coverage_patterns = await self._detect_coverage_patterns(data)
        patterns.extend(coverage_patterns)

        # Detect security patterns
        security_patterns = await self._detect_security_patterns(data)
        patterns.extend(security_patterns)

        # Detect unit test patterns
        unit_patterns = await self._detect_unit_test_patterns(data)
        patterns.extend(unit_patterns)

        # Detect integration patterns
        integration_patterns = await self._detect_integration_patterns(data)
        patterns.extend(integration_patterns)

        logger.info(f"Detected {len(patterns)} test patterns")

        return patterns

    async def _detect_performance_patterns(self, data: dict[str, Any]) -> list[TestPatternInsight]:
        """Detect performance-related test patterns."""
        patterns = []

        # Slow test execution pattern
        if data["avg_execution_time"] > self.config.slow_test_threshold:
            patterns.append(
                TestPatternInsight(
                    pattern_type=TestPatternType.PERFORMANCE_PATTERN,
                    confidence=0.8,
                    description="Slow test execution pattern detected",
                    affected_tests=self._get_slow_tests(data),
                    recommendations=[
                        "Profile slow tests to identify bottlenecks",
                        "Consider parallel test execution",
                        "Optimize test setup and teardown",
                        "Use test data fixtures instead of creating data in tests",
                        "Mock external dependencies to reduce I/O operations",
                    ],
                    impact_score=7.0,
                )
            )

        # Memory usage pattern
        if data.get("peak_memory_usage", 0) > self.config.memory_threshold:
            patterns.append(
                TestPatternInsight(
                    pattern_type=TestPatternType.PERFORMANCE_PATTERN,
                    confidence=0.7,
                    description="High memory usage pattern in tests",
                    affected_tests=[],
                    recommendations=[
                        "Profile memory usage in tests",
                        "Clean up test data after each test",
                        "Use smaller test datasets",
                        "Implement proper test isolation",
                    ],
                    impact_score=6.0,
                )
            )

        # Sequential test execution pattern
        results_by_language = data.get("results_by_language", {})
        if len(results_by_language) > 1 and data["total_tests"] > 50:
            patterns.append(
                TestPatternInsight(
                    pattern_type=TestPatternType.PERFORMANCE_PATTERN,
                    confidence=0.6,
                    description="Multiple language tests could benefit from parallel execution",
                    affected_tests=[],
                    recommendations=[
                        "Implement parallel test execution by language",
                        "Use container-based test isolation",
                        "Configure CI pipeline for parallel test jobs",
                    ],
                    impact_score=5.0,
                )
            )

        return patterns

    async def _detect_reliability_patterns(self, data: dict[str, Any]) -> list[TestPatternInsight]:
        """Detect reliability-related test patterns."""
        patterns = []

        # Flaky test pattern
        if data["overall_success_rate"] < self.config.flaky_test_threshold:
            flaky_count = max(
                0, int(data["total_tests"] * (1 - data["overall_success_rate"]) * 0.3)
            )
            patterns.append(
                TestPatternInsight(
                    pattern_type=TestPatternType.FLAKY_PATTERN,
                    confidence=0.7,
                    description="Flaky test pattern detected",
                    affected_tests=self._get_flaky_tests(data, flaky_count),
                    recommendations=[
                        "Identify and fix flaky tests",
                        "Add retry mechanisms for unstable tests",
                        "Improve test isolation and cleanup",
                        "Use deterministic test data",
                        "Add explicit waits for asynchronous operations",
                    ],
                    impact_score=8.0,
                )
            )

        # Error concentration pattern
        error_categories = self._analyze_error_patterns(data["error_messages"])
        if error_categories and max(error_categories.values()) > len(data["error_messages"]) * 0.3:
            dominant_error = max(error_categories.keys(), key=lambda k: error_categories[k])
            patterns.append(
                TestPatternInsight(
                    pattern_type=TestPatternType.RELIABILITY_PATTERN,
                    confidence=0.8,
                    description=f"High concentration of {dominant_error} errors detected",
                    affected_tests=[],
                    recommendations=[
                        f"Address systematic {dominant_error} issues",
                        "Review and improve error handling",
                        "Add proper test environment setup",
                        "Implement better test data management",
                    ],
                    impact_score=7.5,
                )
            )

        return patterns

    async def _detect_coverage_patterns(self, data: dict[str, Any]) -> list[TestPatternInsight]:
        """Detect coverage-related test patterns."""
        patterns = []

        # Low coverage pattern
        quality_issues = data.get("quality_issues", [])
        if len(quality_issues) > 10:  # Using quality issues as coverage proxy
            patterns.append(
                TestPatternInsight(
                    pattern_type=TestPatternType.COVERAGE_PATTERN,
                    confidence=0.6,
                    description="Low coverage pattern detected",
                    affected_tests=[],
                    recommendations=[
                        "Add tests for uncovered code paths",
                        "Implement coverage reporting in CI",
                        "Set coverage thresholds and gates",
                        "Focus on critical path coverage",
                        "Add integration tests for user flows",
                    ],
                    impact_score=6.0,
                )
            )

        # Language coverage imbalance
        results_by_language = data.get("results_by_language", {})
        if len(results_by_language) > 1:
            test_counts = [len(results) for results in results_by_language.values()]
            if max(test_counts) > min(test_counts) * 3:  # Significant imbalance
                patterns.append(
                    TestPatternInsight(
                        pattern_type=TestPatternType.COVERAGE_PATTERN,
                        confidence=0.7,
                        description="Uneven test coverage across languages",
                        affected_tests=[],
                        recommendations=[
                            "Balance test coverage across all languages",
                            "Add more tests for under-tested languages",
                            "Ensure critical functionality is tested in all languages",
                        ],
                        impact_score=5.0,
                    )
                )

        return patterns

    async def _detect_security_patterns(self, data: dict[str, Any]) -> list[TestPatternInsight]:
        """Detect security-related test patterns."""
        patterns = []

        # Missing security tests pattern
        security_related_errors = [
            error
            for error in data["error_messages"]
            if any(
                keyword in error.lower() for keyword in ["auth", "permission", "security", "token"]
            )
        ]

        if len(security_related_errors) > 0 and data["total_tests"] > 20:
            patterns.append(
                TestPatternInsight(
                    pattern_type=TestPatternType.SECURITY_PATTERN,
                    confidence=0.6,
                    description="Security-related test failures indicate missing security tests",
                    affected_tests=[],
                    recommendations=[
                        "Add comprehensive security tests",
                        "Test authentication and authorization flows",
                        "Add input validation tests",
                        "Test for common security vulnerabilities",
                        "Implement security regression tests",
                    ],
                    impact_score=7.0,
                )
            )

        return patterns

    async def _detect_unit_test_patterns(self, data: dict[str, Any]) -> list[TestPatternInsight]:
        """Detect unit test specific patterns."""
        patterns = []

        # Large test execution time variance
        execution_times = data.get("execution_times", [])
        if execution_times and len(execution_times) > 5:
            p95 = data.get("p95_execution_time", 0)
            median = data.get("median_execution_time", 0)

            if median > 0 and p95 / median > 10:  # High variance
                patterns.append(
                    TestPatternInsight(
                        pattern_type=TestPatternType.UNIT_PATTERN,
                        confidence=0.7,
                        description="High execution time variance suggests mixed unit and integration tests",
                        affected_tests=[],
                        recommendations=[
                            "Separate unit tests from integration tests",
                            "Create different test suites for different test types",
                            "Optimize slow unit tests",
                            "Use proper test categorization",
                        ],
                        impact_score=6.0,
                    )
                )

        return patterns

    async def _detect_integration_patterns(self, data: dict[str, Any]) -> list[TestPatternInsight]:
        """Detect integration test specific patterns."""
        patterns = []

        # Many slow tests pattern (suggests integration tests)
        slow_test_ratio = 0
        if data["total_tests"] > 0:
            slow_tests = len(
                [t for t in data["execution_times"] if t > self.config.slow_test_threshold]
            )
            slow_test_ratio = slow_tests / data["total_tests"]

        if slow_test_ratio > 0.3:  # More than 30% slow tests
            patterns.append(
                TestPatternInsight(
                    pattern_type=TestPatternType.INTEGRATION_PATTERN,
                    confidence=0.8,
                    description="High ratio of slow tests suggests integration test optimization opportunities",
                    affected_tests=[],
                    recommendations=[
                        "Optimize integration test setup",
                        "Use test containers for isolation",
                        "Implement test data fixtures",
                        "Consider mocking external dependencies",
                        "Run integration tests in parallel",
                    ],
                    impact_score=7.0,
                )
            )

        return patterns

    def _get_slow_tests(self, data: dict[str, Any]) -> list[str]:
        """Get list of slow test identifiers."""
        slow_tests = []
        execution_times = data.get("execution_times", [])

        for i, time in enumerate(execution_times):
            if time > self.config.slow_test_threshold:
                slow_tests.append(f"test_{i}")  # Mock test identifier

        return slow_tests[:5]  # Return first 5 for brevity

    def _get_flaky_tests(self, data: dict[str, Any], count: int) -> list[str]:
        """Get list of flaky test identifiers."""
        return [f"flaky_test_{i}" for i in range(min(count, 5))]

    def _analyze_error_patterns(self, error_messages: list[str]) -> dict[str, int]:
        """Analyze error messages to find patterns."""
        error_patterns = {}

        for error in error_messages:
            if not error:
                continue

            error_lower = error.lower()
            if "timeout" in error_lower:
                error_patterns["timeout"] = error_patterns.get("timeout", 0) + 1
            elif "assertion" in error_lower or "assert" in error_lower:
                error_patterns["assertion"] = error_patterns.get("assertion", 0) + 1
            elif "connection" in error_lower or "network" in error_lower:
                error_patterns["connection"] = error_patterns.get("connection", 0) + 1
            elif "permission" in error_lower or "auth" in error_lower:
                error_patterns["permission"] = error_patterns.get("permission", 0) + 1
            elif "memory" in error_lower or "oom" in error_lower:
                error_patterns["memory"] = error_patterns.get("memory", 0) + 1
            else:
                error_patterns["other"] = error_patterns.get("other", 0) + 1

        return error_patterns


# Factory function
def create_test_pattern_detector(
    config: AITestInsightsConfig = DEFAULT_CONFIG,
) -> TestPatternDetector:
    """Create a test pattern detector instance.

    Args:
        config: Configuration for the detector

    Returns:
        TestPatternDetector instance
    """
    return TestPatternDetector(config=config)
