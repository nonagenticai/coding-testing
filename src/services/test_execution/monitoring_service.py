"""
Test Execution Monitoring Service
Provides comprehensive monitoring and logging for multi-language test execution.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from models.logging_models import LogLevel
from services.test_execution.test_orchestrator import TestExecutionResult

logger = logging.getLogger(__name__)


class MetricType(StrEnum):
    """Types of metrics that can be collected."""

    EXECUTION_TIME = "execution_time"
    SUCCESS_RATE = "success_rate"
    CONTAINER_USAGE = "container_usage"
    LANGUAGE_USAGE = "language_usage"
    TEST_TYPE_USAGE = "test_type_usage"
    SECURITY_VIOLATIONS = "security_violations"
    RESOURCE_USAGE = "resource_usage"
    ERROR_RATE = "error_rate"


@dataclass
class TestExecutionMetric:
    """Represents a single test execution metric."""

    timestamp: datetime
    test_id: str
    language: str
    test_type: str
    metric_type: MetricType
    value: float
    unit: str
    metadata: dict[str, Any]


@dataclass
class TestExecutionLogEntry:
    """Represents a test execution log entry."""

    timestamp: datetime
    test_id: str
    level: LogLevel
    message: str
    component: str
    language: str | None = None
    test_type: str | None = None
    container_name: str | None = None
    metadata: dict[str, Any] | None = None


class TestExecutionStats(BaseModel):
    """Statistics for test execution."""

    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time_ms: float = 0.0
    total_execution_time_ms: float = 0.0
    language_breakdown: dict[str, int] = Field(default_factory=dict)
    test_type_breakdown: dict[str, int] = Field(default_factory=dict)
    security_level_breakdown: dict[str, int] = Field(default_factory=dict)
    hourly_executions: list[int] = Field(default_factory=list)
    error_breakdown: dict[str, int] = Field(default_factory=dict)

    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_executions == 0:
            return 0.0
        return (self.successful_executions / self.total_executions) * 100.0


class ContainerMetrics(BaseModel):
    """Container usage metrics."""

    active_containers: int = 0
    total_containers_created: int = 0
    total_containers_destroyed: int = 0
    container_creation_time_ms: float = 0.0
    container_cleanup_time_ms: float = 0.0
    container_failures: int = 0
    language_container_usage: dict[str, int] = Field(default_factory=dict)
    resource_usage: dict[str, float] = Field(default_factory=dict)


class SecurityMetrics(BaseModel):
    """Security-related metrics."""

    security_violations: int = 0
    blocked_executions: int = 0
    dangerous_patterns_detected: int = 0
    unauthorized_imports: int = 0
    violation_types: dict[str, int] = Field(default_factory=dict)
    security_level_distribution: dict[str, int] = Field(default_factory=dict)


class TestExecutionMonitor:
    """
    Comprehensive monitoring service for test execution.

    Collects metrics, logs events, and provides analytics for the test
    orchestration system.
    """

    def __init__(self, max_metrics_history: int = 10000, max_log_history: int = 5000):
        """Initialize the monitoring service."""
        self.max_metrics_history = max_metrics_history
        self.max_log_history = max_log_history

        # Metrics storage
        self.metrics: deque[TestExecutionMetric] = deque(maxlen=max_metrics_history)
        self.logs: deque[TestExecutionLogEntry] = deque(maxlen=max_log_history)

        # Statistics
        self.stats = TestExecutionStats()
        self.container_metrics = ContainerMetrics()
        self.security_metrics = SecurityMetrics()

        # Active test tracking
        self.active_tests: dict[str, dict[str, Any]] = {}
        self.test_start_times: dict[str, datetime] = {}

        # Performance tracking
        self.performance_history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=100))

        logger.info("TestExecutionMonitor initialized")

    def log_test_started(
        self,
        test_id: str,
        language: str,
        test_type: str,
        container_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log test execution start."""
        self.test_start_times[test_id] = datetime.now(UTC)
        self.active_tests[test_id] = {
            "language": language,
            "test_type": test_type,
            "container_name": container_name,
            "start_time": self.test_start_times[test_id],
            "metadata": metadata or {},
        }

        self._add_log_entry(
            test_id=test_id,
            level=LogLevel.INFO,
            message=f"Test execution started for {language} {test_type}",
            component="orchestrator",
            language=language,
            test_type=test_type,
            container_name=container_name,
            metadata=metadata,
        )

        logger.info("Test %s started: %s %s", test_id, language, test_type)

    def log_test_completed(self, result: TestExecutionResult) -> None:
        """Log test execution completion and update metrics."""
        test_id = result.test_id

        # Update statistics
        self.stats.total_executions += 1

        if result.success:
            self.stats.successful_executions += 1
        else:
            self.stats.failed_executions += 1

            # Track error type
            if result.stderr:
                error_type = self._classify_error(result.stderr)
                error_breakdown = dict(self.stats.error_breakdown)
                error_breakdown[error_type] = error_breakdown.get(error_type, 0) + 1
                self.stats.error_breakdown = error_breakdown

        # Update execution time metrics
        execution_time = result.execution_time_ms
        self.stats.total_execution_time_ms += execution_time
        self.stats.average_execution_time_ms = (
            self.stats.total_execution_time_ms / self.stats.total_executions
        )

        # Update breakdowns
        language = result.language.value
        test_type = result.test_type.value

        language_breakdown = dict(self.stats.language_breakdown)
        language_breakdown[language] = language_breakdown.get(language, 0) + 1
        self.stats.language_breakdown = language_breakdown

        test_type_breakdown = dict(self.stats.test_type_breakdown)
        test_type_breakdown[test_type] = test_type_breakdown.get(test_type, 0) + 1
        self.stats.test_type_breakdown = test_type_breakdown

        # Update performance history
        self.performance_history[language].append(execution_time)
        self.performance_history[f"{language}_{test_type}"].append(execution_time)

        # Add metrics
        self._add_metric(
            test_id=test_id,
            language=language,
            test_type=test_type,
            metric_type=MetricType.EXECUTION_TIME,
            value=execution_time,
            unit="ms",
            metadata={
                "success": result.success,
                "exit_code": result.exit_code,
            },
        )

        # Log completion
        log_level = LogLevel.INFO if result.success else LogLevel.ERROR
        message = f"Test execution {'completed' if result.success else 'failed'} in {execution_time:.2f}ms"

        self._add_log_entry(
            test_id=test_id,
            level=log_level,
            message=message,
            component="orchestrator",
            language=language,
            test_type=test_type,
            container_name=result.container_name,
            metadata={
                "success": result.success,
                "exit_code": result.exit_code,
                "execution_time_ms": execution_time,
            },
        )

        # Clean up tracking
        if test_id in self.active_tests:
            del self.active_tests[test_id]
        if test_id in self.test_start_times:
            del self.test_start_times[test_id]

        logger.info(
            "Test %s completed: %s %s - %s",
            test_id,
            language,
            test_type,
            "SUCCESS" if result.success else "FAILED",
        )

    def log_container_created(
        self, container_name: str, language: str, creation_time_ms: float
    ) -> None:
        """Log container creation."""
        self.container_metrics.total_containers_created += 1
        self.container_metrics.active_containers += 1
        self.container_metrics.container_creation_time_ms += creation_time_ms

        lang_usage = dict(self.container_metrics.language_container_usage)
        lang_usage[language] = lang_usage.get(language, 0) + 1
        self.container_metrics.language_container_usage = lang_usage

        self._add_log_entry(
            test_id="",
            level=LogLevel.INFO,
            message=f"Container created for {language} in {creation_time_ms:.2f}ms",
            component="container_manager",
            language=language,
            container_name=container_name,
            metadata={"creation_time_ms": creation_time_ms},
        )

        logger.debug("Container %s created for %s", container_name, language)

    def log_container_destroyed(
        self, container_name: str, language: str, cleanup_time_ms: float
    ) -> None:
        """Log container destruction."""
        self.container_metrics.total_containers_destroyed += 1
        self.container_metrics.active_containers = max(
            0, self.container_metrics.active_containers - 1
        )
        self.container_metrics.container_cleanup_time_ms += cleanup_time_ms

        self._add_log_entry(
            test_id="",
            level=LogLevel.INFO,
            message=f"Container destroyed for {language} in {cleanup_time_ms:.2f}ms",
            component="container_manager",
            language=language,
            container_name=container_name,
            metadata={"cleanup_time_ms": cleanup_time_ms},
        )

        logger.debug("Container %s destroyed for %s", container_name, language)

    def log_container_failure(self, container_name: str, language: str, error: str) -> None:
        """Log container failure."""
        self.container_metrics.container_failures += 1

        self._add_log_entry(
            test_id="",
            level=LogLevel.ERROR,
            message=f"Container failure for {language}: {error}",
            component="container_manager",
            language=language,
            container_name=container_name,
            metadata={"error": error},
        )

        logger.error("Container %s failed for %s: %s", container_name, language, error)

    def log_security_violation(
        self, test_id: str, language: str, violation_type: str, details: str
    ) -> None:
        """Log security violation."""
        self.security_metrics.security_violations += 1
        self.security_metrics.blocked_executions += 1

        violation_types = dict(self.security_metrics.violation_types)
        violation_types[violation_type] = violation_types.get(violation_type, 0) + 1
        self.security_metrics.violation_types = violation_types

        self._add_log_entry(
            test_id=test_id,
            level=LogLevel.WARNING,
            message=f"Security violation detected: {violation_type}",
            component="security_manager",
            language=language,
            metadata={"violation_type": violation_type, "details": details},
        )

        logger.warning("Security violation in test %s: %s - %s", test_id, violation_type, details)

    def log_resource_usage(
        self,
        container_name: str,
        language: str,
        cpu_usage: float,
        memory_usage: float,
    ) -> None:
        """Log resource usage."""
        resource_usage = dict(self.container_metrics.resource_usage)
        resource_usage[f"{container_name}_cpu"] = cpu_usage
        resource_usage[f"{container_name}_memory"] = memory_usage
        self.container_metrics.resource_usage = resource_usage

        self._add_metric(
            test_id="",
            language=language,
            test_type="",
            metric_type=MetricType.RESOURCE_USAGE,
            value=cpu_usage,
            unit="percent",
            metadata={"container_name": container_name, "resource_type": "cpu"},
        )

        self._add_metric(
            test_id="",
            language=language,
            test_type="",
            metric_type=MetricType.RESOURCE_USAGE,
            value=memory_usage,
            unit="mb",
            metadata={"container_name": container_name, "resource_type": "memory"},
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "execution_stats": self.stats.model_dump(),
            "container_metrics": self.container_metrics.model_dump(),
            "security_metrics": self.security_metrics.model_dump(),
            "active_tests": len(self.active_tests),
            "metrics_count": len(self.metrics),
            "logs_count": len(self.logs),
            "performance_trends": self._get_performance_trends(),
            "health_status": self._get_health_status(),
        }

    def get_metrics_by_type(
        self, metric_type: MetricType, hours_back: int = 24
    ) -> list[TestExecutionMetric]:
        """Get metrics of a specific type within time range."""
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours_back)

        return [
            metric
            for metric in self.metrics
            if metric.metric_type == metric_type and metric.timestamp >= cutoff_time
        ]

    def get_logs_by_level(
        self, level: LogLevel, hours_back: int = 24
    ) -> list[TestExecutionLogEntry]:
        """Get logs of a specific level within time range."""
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours_back)

        return [log for log in self.logs if log.level == level and log.timestamp >= cutoff_time]

    def get_language_performance(self, language: str) -> dict[str, float]:
        """Get performance metrics for a specific language."""
        if language not in self.performance_history:
            return {
                "avg_execution_time": 0.0,
                "min_execution_time": 0.0,
                "max_execution_time": 0.0,
            }

        times = list(self.performance_history[language])
        if not times:
            return {
                "avg_execution_time": 0.0,
                "min_execution_time": 0.0,
                "max_execution_time": 0.0,
            }

        return {
            "avg_execution_time": sum(times) / len(times),
            "min_execution_time": min(times),
            "max_execution_time": max(times),
            "execution_count": len(times),
        }

    def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in specified format."""
        data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "statistics": self.get_statistics(),
            "recent_metrics": [
                {
                    "timestamp": metric.timestamp.isoformat(),
                    "test_id": metric.test_id,
                    "language": metric.language,
                    "test_type": metric.test_type,
                    "metric_type": metric.metric_type.value,
                    "value": metric.value,
                    "unit": metric.unit,
                    "metadata": metric.metadata,
                }
                for metric in list(self.metrics)[-100:]  # Last 100 metrics
            ],
            "recent_logs": [
                {
                    "timestamp": log.timestamp.isoformat(),
                    "test_id": log.test_id,
                    "level": log.level.value,
                    "message": log.message,
                    "component": log.component,
                    "language": log.language,
                    "test_type": log.test_type,
                    "container_name": log.container_name,
                    "metadata": log.metadata,
                }
                for log in list(self.logs)[-100:]  # Last 100 logs
            ],
        }

        if format_type.lower() == "json":
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

    def _add_metric(
        self,
        test_id: str,
        language: str,
        test_type: str,
        metric_type: MetricType,
        value: float,
        unit: str,
        metadata: dict[str, Any],
    ) -> None:
        """Add a metric to the collection."""
        metric = TestExecutionMetric(
            timestamp=datetime.now(UTC),
            test_id=test_id,
            language=language,
            test_type=test_type,
            metric_type=metric_type,
            value=value,
            unit=unit,
            metadata=metadata,
        )
        self.metrics.append(metric)

    def _add_log_entry(
        self,
        test_id: str,
        level: LogLevel,
        message: str,
        component: str,
        language: str | None = None,
        test_type: str | None = None,
        container_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a log entry to the collection."""
        log_entry = TestExecutionLogEntry(
            timestamp=datetime.now(UTC),
            test_id=test_id,
            level=level,
            message=message,
            component=component,
            language=language,
            test_type=test_type,
            container_name=container_name,
            metadata=metadata,
        )
        self.logs.append(log_entry)

    def _classify_error(self, error_message: str) -> str:
        """Classify error message into category."""
        error_lower = error_message.lower()

        if "timeout" in error_lower:
            return "timeout"
        elif "memory" in error_lower:
            return "memory"
        elif "permission" in error_lower or "access" in error_lower:
            return "permission"
        elif "network" in error_lower:
            return "network"
        elif "container" in error_lower:
            return "container"
        elif "security" in error_lower:
            return "security"
        elif "compilation" in error_lower or "compile" in error_lower:
            return "compilation"
        elif "syntax" in error_lower:
            return "syntax"
        else:
            return "other"

    def _get_performance_trends(self) -> dict[str, Any]:
        """Get performance trend analysis."""
        trends = {}

        for key, times in self.performance_history.items():
            if len(times) >= 2:
                recent_avg = sum(list(times)[-10:]) / min(10, len(times))
                overall_avg = sum(times) / len(times)

                trends[key] = {
                    "recent_avg": recent_avg,
                    "overall_avg": overall_avg,
                    "trend": (
                        "improving"
                        if recent_avg < overall_avg
                        else ("degrading" if recent_avg > overall_avg else "stable")
                    ),
                }

        return trends

    def _get_health_status(self) -> dict[str, Any]:
        """Get system health status."""
        success_rate = self.stats.success_rate()

        # Determine health status
        if success_rate >= 95:
            health = "excellent"
        elif success_rate >= 85:
            health = "good"
        elif success_rate >= 70:
            health = "fair"
        else:
            health = "poor"

        return {
            "overall_health": health,
            "success_rate": success_rate,
            "active_containers": self.container_metrics.active_containers,
            "security_violations": self.security_metrics.security_violations,
            "container_failures": self.container_metrics.container_failures,
            "recommendations": self._get_recommendations(),
        }

    def _get_recommendations(self) -> list[str]:
        """Get system recommendations based on metrics."""
        recommendations = []

        success_rate = self.stats.success_rate()
        if success_rate < 85:
            recommendations.append("Consider investigating frequent test failures")

        if self.container_metrics.container_failures > 10:
            recommendations.append(
                "High container failure rate detected - check Docker configuration"
            )

        if self.security_metrics.security_violations > 5:
            recommendations.append(
                "Multiple security violations detected - review security policies"
            )

        if self.stats.average_execution_time_ms > 30000:  # 30 seconds
            recommendations.append(
                "High average execution time - consider optimizing test containers"
            )

        return recommendations


def create_test_execution_monitor(
    max_metrics_history: int = 10000, max_log_history: int = 5000
) -> TestExecutionMonitor:
    """Factory function to create TestExecutionMonitor."""
    return TestExecutionMonitor(max_metrics_history, max_log_history)
