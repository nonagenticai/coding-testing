"""
Performance metrics tracking for language-specific test executors.

This module provides comprehensive performance tracking for each language executor,
including execution time analysis, resource usage profiling, and performance benchmarking.
"""

import json
import statistics
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any


class PerformanceMetricType(StrEnum):
    """Types of performance metrics."""

    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    CONTAINER_STARTUP = "container_startup"
    TEST_COMPILATION = "test_compilation"
    TEST_EXECUTION = "test_execution"
    CLEANUP_TIME = "cleanup_time"


class PerformanceCategory(StrEnum):
    """Categories for performance analysis."""

    THROUGHPUT = "throughput"
    LATENCY = "latency"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    SCALABILITY = "scalability"
    RELIABILITY = "reliability"


@dataclass
class PerformanceDataPoint:
    """Individual performance measurement."""

    timestamp: datetime
    metric_type: PerformanceMetricType
    category: PerformanceCategory
    value: float
    unit: str
    language: str
    test_type: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "metric_type": self.metric_type.value,
            "category": self.category.value,
            "value": self.value,
            "unit": self.unit,
            "language": self.language,
            "test_type": self.test_type,
            "context": self.context,
        }


@dataclass
class PerformanceBenchmark:
    """Performance benchmark for comparison."""

    language: str
    test_type: str
    metric_type: PerformanceMetricType
    target_value: float
    unit: str
    tolerance_percent: float = 10.0
    baseline_timestamp: datetime = field(default_factory=datetime.now)

    def is_within_tolerance(self, value: float) -> bool:
        """Check if value is within acceptable tolerance."""
        deviation = abs(value - self.target_value) / self.target_value * 100
        return deviation <= self.tolerance_percent


@dataclass
class PerformanceStatistics:
    """Statistical analysis of performance metrics."""

    metric_type: PerformanceMetricType
    language: str
    test_type: str
    count: int
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    percentile_95: float
    percentile_99: float
    unit: str
    time_period: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metric_type": self.metric_type.value,
            "language": self.language,
            "test_type": self.test_type,
            "count": self.count,
            "mean": self.mean,
            "median": self.median,
            "std_dev": self.std_dev,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "percentile_95": self.percentile_95,
            "percentile_99": self.percentile_99,
            "unit": self.unit,
            "time_period": self.time_period,
        }


class PerformanceTimer:
    """High-precision timer for performance measurements."""

    def __init__(self, name: str = "timer"):
        """Initialize the PerformanceTimer service.

        This initializes the PerformanceTimer with required dependencies
        and configuration for service operations.

        Args:
            name (str): Name
        """
        self.name = name
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.duration: float | None = None

    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer and return duration."""
        if self.start_time is None:
            raise ValueError("Timer not started")
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        return self.duration

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


class LanguagePerformanceTracker:
    """Performance tracker for specific language executors."""

    def __init__(self, language: str):
        """Initialize the LanguagePerformanceTracker service.

        This initializes the LanguagePerformanceTracker with required dependencies
        and configuration for service operations.

        Args:
            language (str): Language
        """
        self.language = language
        self.data_points: list[PerformanceDataPoint] = []
        self.benchmarks: dict[str, PerformanceBenchmark] = {}
        self.lock = threading.Lock()
        self.active_timers: dict[str, PerformanceTimer] = {}
        self._load_benchmarks()

    def _load_benchmarks(self) -> None:
        """Load performance benchmarks for the language."""
        language_benchmarks = {
            "go": {
                "container_startup": PerformanceBenchmark(
                    language="go",
                    test_type="unit",
                    metric_type=PerformanceMetricType.CONTAINER_STARTUP,
                    target_value=2.0,
                    unit="seconds",
                    tolerance_percent=20.0,
                ),
                "test_execution": PerformanceBenchmark(
                    language="go",
                    test_type="unit",
                    metric_type=PerformanceMetricType.TEST_EXECUTION,
                    target_value=10.0,
                    unit="seconds",
                    tolerance_percent=30.0,
                ),
                "memory_usage": PerformanceBenchmark(
                    language="go",
                    test_type="unit",
                    metric_type=PerformanceMetricType.MEMORY_USAGE,
                    target_value=256.0,
                    unit="MB",
                    tolerance_percent=25.0,
                ),
            },
            "rust": {
                "container_startup": PerformanceBenchmark(
                    language="rust",
                    test_type="unit",
                    metric_type=PerformanceMetricType.CONTAINER_STARTUP,
                    target_value=3.0,
                    unit="seconds",
                    tolerance_percent=25.0,
                ),
                "test_compilation": PerformanceBenchmark(
                    language="rust",
                    test_type="unit",
                    metric_type=PerformanceMetricType.TEST_COMPILATION,
                    target_value=30.0,
                    unit="seconds",
                    tolerance_percent=40.0,
                ),
                "memory_usage": PerformanceBenchmark(
                    language="rust",
                    test_type="unit",
                    metric_type=PerformanceMetricType.MEMORY_USAGE,
                    target_value=512.0,
                    unit="MB",
                    tolerance_percent=30.0,
                ),
            },
            "java": {
                "container_startup": PerformanceBenchmark(
                    language="java",
                    test_type="unit",
                    metric_type=PerformanceMetricType.CONTAINER_STARTUP,
                    target_value=4.0,
                    unit="seconds",
                    tolerance_percent=30.0,
                ),
                "memory_usage": PerformanceBenchmark(
                    language="java",
                    test_type="unit",
                    metric_type=PerformanceMetricType.MEMORY_USAGE,
                    target_value=768.0,
                    unit="MB",
                    tolerance_percent=25.0,
                ),
            },
            "csharp": {
                "container_startup": PerformanceBenchmark(
                    language="csharp",
                    test_type="unit",
                    metric_type=PerformanceMetricType.CONTAINER_STARTUP,
                    target_value=3.5,
                    unit="seconds",
                    tolerance_percent=25.0,
                ),
                "memory_usage": PerformanceBenchmark(
                    language="csharp",
                    test_type="unit",
                    metric_type=PerformanceMetricType.MEMORY_USAGE,
                    target_value=512.0,
                    unit="MB",
                    tolerance_percent=30.0,
                ),
            },
            "cpp": {
                "container_startup": PerformanceBenchmark(
                    language="cpp",
                    test_type="unit",
                    metric_type=PerformanceMetricType.CONTAINER_STARTUP,
                    target_value=2.5,
                    unit="seconds",
                    tolerance_percent=20.0,
                ),
                "test_compilation": PerformanceBenchmark(
                    language="cpp",
                    test_type="unit",
                    metric_type=PerformanceMetricType.TEST_COMPILATION,
                    target_value=15.0,
                    unit="seconds",
                    tolerance_percent=35.0,
                ),
                "memory_usage": PerformanceBenchmark(
                    language="cpp",
                    test_type="unit",
                    metric_type=PerformanceMetricType.MEMORY_USAGE,
                    target_value=256.0,
                    unit="MB",
                    tolerance_percent=25.0,
                ),
            },
        }
        self.benchmarks = language_benchmarks.get(self.language, {})

    def start_timer(self, name: str) -> PerformanceTimer:
        """Start a named timer."""
        with self.lock:
            timer = PerformanceTimer(name)
            self.active_timers[name] = timer
            timer.start()
            return timer

    def stop_timer(self, name: str) -> float | None:
        """Stop a named timer and return duration."""
        with self.lock:
            timer = self.active_timers.pop(name, None)
            if timer:
                return timer.stop()
            return None

    @contextmanager
    def measure_performance(
        self,
        metric_type: PerformanceMetricType,
        category: PerformanceCategory,
        test_type: str,
        unit: str = "seconds",
        **context,
    ) -> Generator[PerformanceTimer]:
        """Context manager for measuring performance."""
        timer = PerformanceTimer()
        timer.start()
        try:
            yield timer
        finally:
            duration = timer.stop()
            self.record_metric(
                metric_type=metric_type,
                category=category,
                value=duration,
                unit=unit,
                test_type=test_type,
                context=context,
            )

    def record_metric(
        self,
        metric_type: PerformanceMetricType,
        category: PerformanceCategory,
        value: float,
        unit: str,
        test_type: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record a performance metric."""
        data_point = PerformanceDataPoint(
            timestamp=datetime.now(UTC),
            metric_type=metric_type,
            category=category,
            value=value,
            unit=unit,
            language=self.language,
            test_type=test_type,
            context=context or {},
        )
        with self.lock:
            self.data_points.append(data_point)

    def get_statistics(
        self, metric_type: PerformanceMetricType, test_type: str, hours: int = 24
    ) -> PerformanceStatistics | None:
        """Get statistical analysis for a metric."""
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours)
        with self.lock:
            filtered_data = [
                dp
                for dp in self.data_points
                if dp.metric_type == metric_type
                and dp.test_type == test_type
                and (dp.timestamp >= cutoff_time)
            ]
        if not filtered_data:
            return None
        values = [dp.value for dp in filtered_data]
        return PerformanceStatistics(
            metric_type=metric_type,
            language=self.language,
            test_type=test_type,
            count=len(values),
            mean=statistics.mean(values),
            median=statistics.median(values),
            std_dev=statistics.stdev(values) if len(values) > 1 else 0.0,
            min_value=min(values),
            max_value=max(values),
            percentile_95=(
                statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values)
            ),
            percentile_99=(
                statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values)
            ),
            unit=filtered_data[0].unit,
            time_period=f"{hours} hours",
        )

    def check_performance_regression(
        self, metric_type: PerformanceMetricType, value: float, test_type: str
    ) -> dict[str, Any]:
        """Check for performance regression."""
        benchmark_key = f"{metric_type.value}"
        benchmark = self.benchmarks.get(benchmark_key)
        if not benchmark:
            return {
                "has_regression": False,
                "message": f"No benchmark defined for {metric_type.value}",
                "severity": "info",
            }
        if benchmark.is_within_tolerance(value):
            return {
                "has_regression": False,
                "message": "Performance within acceptable range",
                "severity": "info",
                "deviation_percent": abs(value - benchmark.target_value)
                / benchmark.target_value
                * 100,
            }
        deviation = abs(value - benchmark.target_value) / benchmark.target_value * 100
        severity = "warning" if deviation < 50 else "critical"
        return {
            "has_regression": True,
            "message": f"Performance regression detected: {deviation:.1f}% deviation from target",
            "severity": severity,
            "deviation_percent": deviation,
            "target_value": benchmark.target_value,
            "actual_value": value,
            "tolerance_percent": benchmark.tolerance_percent,
        }


class PerformanceTrackerService:
    """Central performance tracking service for all language executors."""

    def __init__(self):
        """Initialize the PerformanceTrackerService service.

        This initializes the PerformanceTrackerService with required dependencies
        and configuration for service operations.
        """
        self.language_trackers: dict[str, LanguagePerformanceTracker] = {}
        self.lock = threading.Lock()

    def get_tracker(self, language: str) -> LanguagePerformanceTracker:
        """Get or create a language-specific tracker."""
        with self.lock:
            if language not in self.language_trackers:
                self.language_trackers[language] = LanguagePerformanceTracker(language)
            return self.language_trackers[language]

    def record_execution_metrics(
        self,
        language: str,
        test_type: str,
        container_startup_time: float,
        test_execution_time: float,
        cleanup_time: float,
        peak_memory_mb: float,
        avg_cpu_percent: float,
        **context,
    ) -> None:
        """Record comprehensive execution metrics."""
        tracker = self.get_tracker(language)
        tracker.record_metric(
            PerformanceMetricType.CONTAINER_STARTUP,
            PerformanceCategory.LATENCY,
            container_startup_time,
            "seconds",
            test_type,
            context,
        )
        tracker.record_metric(
            PerformanceMetricType.TEST_EXECUTION,
            PerformanceCategory.THROUGHPUT,
            test_execution_time,
            "seconds",
            test_type,
            context,
        )
        tracker.record_metric(
            PerformanceMetricType.CLEANUP_TIME,
            PerformanceCategory.RESOURCE_EFFICIENCY,
            cleanup_time,
            "seconds",
            test_type,
            context,
        )
        tracker.record_metric(
            PerformanceMetricType.MEMORY_USAGE,
            PerformanceCategory.RESOURCE_EFFICIENCY,
            peak_memory_mb,
            "MB",
            test_type,
            context,
        )
        tracker.record_metric(
            PerformanceMetricType.CPU_USAGE,
            PerformanceCategory.RESOURCE_EFFICIENCY,
            avg_cpu_percent,
            "percent",
            test_type,
            context,
        )

    def get_performance_report(self, language: str, hours: int = 24) -> dict[str, Any]:
        """Get comprehensive performance report for a language."""
        tracker = self.get_tracker(language)
        metrics = [
            PerformanceMetricType.CONTAINER_STARTUP,
            PerformanceMetricType.TEST_EXECUTION,
            PerformanceMetricType.MEMORY_USAGE,
            PerformanceMetricType.CPU_USAGE,
        ]
        test_types = ["unit", "integration", "benchmark"]
        report: dict[str, Any] = {
            "language": language,
            "time_period": f"{hours} hours",
            "statistics": {},
            "regression_alerts": [],
        }
        for metric in metrics:
            for test_type in test_types:
                stats = tracker.get_statistics(metric, test_type, hours)
                if stats:
                    key = f"{metric.value}_{test_type}"
                    report["statistics"][key] = stats.to_dict()
                    regression = tracker.check_performance_regression(metric, stats.mean, test_type)
                    if regression["has_regression"]:
                        report["regression_alerts"].append(
                            {
                                "metric": metric.value,
                                "test_type": test_type,
                                **regression,
                            }
                        )
        return report

    def get_cross_language_comparison(self, hours: int = 24) -> dict[str, Any]:
        """Get performance comparison across languages."""
        comparison: dict[str, Any] = {
            "time_period": f"{hours} hours",
            "languages": {},
            "rankings": {},
        }
        for language, tracker in self.language_trackers.items():
            lang_data = {}
            for metric in [
                PerformanceMetricType.CONTAINER_STARTUP,
                PerformanceMetricType.TEST_EXECUTION,
            ]:
                stats = tracker.get_statistics(metric, "unit", hours)
                if stats:
                    lang_data[metric.value] = {
                        "mean": stats.mean,
                        "median": stats.median,
                        "count": stats.count,
                    }
            if lang_data:
                comparison["languages"][language] = lang_data
        for metric_name in ["container_startup", "test_execution"]:
            rankings: list[dict[str, Any]] = []
            for lang, data in comparison["languages"].items():
                if metric_name in data:
                    rankings.append(
                        {
                            "language": lang,
                            "mean": data[metric_name]["mean"],
                            "median": data[metric_name]["median"],
                            "count": data[metric_name]["count"],
                        }
                    )
            rankings.sort(key=lambda x: x["mean"])
            comparison["rankings"][metric_name] = rankings
        return comparison

    def export_metrics(self, language: str | None = None, hours: int = 24) -> str:
        """Export performance metrics as JSON."""
        if language:
            tracker = self.get_tracker(language)
            cutoff_time = datetime.now(UTC) - timedelta(hours=hours)
            with tracker.lock:
                filtered_data = [
                    dp.to_dict() for dp in tracker.data_points if dp.timestamp >= cutoff_time
                ]
            return json.dumps(
                {
                    "language": language,
                    "time_period": f"{hours} hours",
                    "data_points": filtered_data,
                },
                indent=2,
            )
        else:
            all_data = {}
            for lang, tracker in self.language_trackers.items():
                cutoff_time = datetime.now(UTC) - timedelta(hours=hours)
                with tracker.lock:
                    filtered_data = [
                        dp.to_dict() for dp in tracker.data_points if dp.timestamp >= cutoff_time
                    ]
                all_data[lang] = filtered_data
            return json.dumps({"time_period": f"{hours} hours", "languages": all_data}, indent=2)


_performance_tracker = PerformanceTrackerService()


def get_performance_tracker() -> PerformanceTrackerService:
    """Get the global performance tracker instance."""
    return _performance_tracker
