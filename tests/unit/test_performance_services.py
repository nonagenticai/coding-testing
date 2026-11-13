"""Tests for performance_tracker.py, monitoring_service.py, resource_monitor.py,
docker_service.py, code_execution_service.py, security_service.py, and pydantic_ai_logger.py."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# performance_tracker.py
# ---------------------------------------------------------------------------
from services.test_execution.performance_tracker import (
    LanguagePerformanceTracker,
    PerformanceBenchmark,
    PerformanceCategory,
    PerformanceDataPoint,
    PerformanceMetricType,
    PerformanceStatistics,
    PerformanceTimer,
    PerformanceTrackerService,
    get_performance_tracker,
)


class TestPerformanceTimer:
    def test_start_stop(self):
        timer = PerformanceTimer("test")
        assert timer.start_time is None
        assert timer.end_time is None
        assert timer.duration is None
        timer.start()
        assert isinstance(timer.start_time, float)
        assert timer.start_time > 0
        time.sleep(0.01)
        duration = timer.stop()
        assert 0.005 <= duration <= 1.0
        assert timer.duration == duration
        assert isinstance(timer.end_time, float)
        assert timer.end_time > timer.start_time
        assert timer.name == "test"

    def test_stop_without_start_raises(self):
        timer = PerformanceTimer()
        assert timer.name == "timer"  # default name
        with pytest.raises(ValueError, match="not started"):
            timer.stop()

    def test_context_manager(self):
        with PerformanceTimer("cm") as timer:
            time.sleep(0.01)
        assert 0.005 <= timer.duration <= 1.0
        assert isinstance(timer.start_time, float) and timer.start_time > 0
        assert isinstance(timer.end_time, float) and timer.end_time > timer.start_time


class TestPerformanceBenchmark:
    def test_is_within_tolerance_pass(self):
        bm = PerformanceBenchmark(
            language="go",
            test_type="unit",
            metric_type=PerformanceMetricType.CONTAINER_STARTUP,
            target_value=2.0,
            unit="seconds",
            tolerance_percent=20.0,
        )
        assert bm.is_within_tolerance(2.3) is True  # 15% deviation < 20% tolerance
        assert bm.is_within_tolerance(2.0) is True  # exact match
        assert bm.is_within_tolerance(1.8) is True  # 10% deviation < 20% tolerance

    def test_is_within_tolerance_fail(self):
        bm = PerformanceBenchmark(
            language="go",
            test_type="unit",
            metric_type=PerformanceMetricType.CONTAINER_STARTUP,
            target_value=2.0,
            unit="seconds",
            tolerance_percent=10.0,
        )
        assert bm.is_within_tolerance(3.0) is False  # 50% deviation > 10% tolerance
        assert bm.is_within_tolerance(1.0) is False  # 50% deviation > 10% tolerance


class TestPerformanceDataPoint:
    def test_to_dict(self):
        dp = PerformanceDataPoint(
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            metric_type=PerformanceMetricType.EXECUTION_TIME,
            category=PerformanceCategory.LATENCY,
            value=1.5,
            unit="seconds",
            language="python",
            test_type="unit",
            context={"key": "val"},
        )
        d = dp.to_dict()
        assert d == {
            "timestamp": "2025-01-01T00:00:00+00:00",
            "metric_type": "execution_time",
            "category": "latency",
            "value": 1.5,
            "unit": "seconds",
            "language": "python",
            "test_type": "unit",
            "context": {"key": "val"},
        }


class TestPerformanceStatistics:
    def test_to_dict(self):
        stats = PerformanceStatistics(
            metric_type=PerformanceMetricType.EXECUTION_TIME,
            language="python",
            test_type="unit",
            count=10,
            mean=1.5,
            median=1.4,
            std_dev=0.2,
            min_value=1.0,
            max_value=2.0,
            percentile_95=1.9,
            percentile_99=1.95,
            unit="seconds",
            time_period="24 hours",
        )
        d = stats.to_dict()
        assert d == {
            "metric_type": "execution_time",
            "language": "python",
            "test_type": "unit",
            "count": 10,
            "mean": 1.5,
            "median": 1.4,
            "std_dev": 0.2,
            "min_value": 1.0,
            "max_value": 2.0,
            "percentile_95": 1.9,
            "percentile_99": 1.95,
            "unit": "seconds",
            "time_period": "24 hours",
        }


class TestLanguagePerformanceTracker:
    @pytest.fixture
    def tracker(self) -> LanguagePerformanceTracker:
        return LanguagePerformanceTracker("go")

    def test_record_metric(self, tracker):
        tracker.record_metric(
            PerformanceMetricType.EXECUTION_TIME,
            PerformanceCategory.LATENCY,
            1.5,
            "seconds",
            "unit",
        )
        assert len(tracker.data_points) == 1
        dp = tracker.data_points[0]
        assert dp.metric_type == PerformanceMetricType.EXECUTION_TIME
        assert dp.category == PerformanceCategory.LATENCY
        assert dp.value == 1.5
        assert dp.unit == "seconds"
        assert dp.language == "go"
        assert dp.test_type == "unit"

    def test_start_and_stop_timer(self, tracker):
        tracker.start_timer("my-timer")
        assert "my-timer" in tracker.active_timers
        time.sleep(0.01)
        duration = tracker.stop_timer("my-timer")
        assert 0.005 <= duration <= 1.0
        assert "my-timer" not in tracker.active_timers

    def test_stop_nonexistent_timer(self, tracker):
        result = tracker.stop_timer("nonexistent")
        assert result is None

    def test_measure_performance_context(self, tracker):
        with tracker.measure_performance(
            PerformanceMetricType.EXECUTION_TIME,
            PerformanceCategory.LATENCY,
            "unit",
        ):
            time.sleep(0.01)
        assert len(tracker.data_points) == 1
        dp = tracker.data_points[0]
        assert 0.005 <= dp.value <= 1.0
        assert dp.metric_type == PerformanceMetricType.EXECUTION_TIME
        assert dp.category == PerformanceCategory.LATENCY
        assert dp.unit == "seconds"

    def test_get_statistics_no_data(self, tracker):
        result = tracker.get_statistics(PerformanceMetricType.EXECUTION_TIME, "unit")
        assert result is None

    def test_get_statistics_with_data(self, tracker):
        for val in [1.0, 2.0, 3.0, 4.0, 5.0]:
            tracker.record_metric(
                PerformanceMetricType.EXECUTION_TIME,
                PerformanceCategory.LATENCY,
                val,
                "seconds",
                "unit",
            )
        stats = tracker.get_statistics(PerformanceMetricType.EXECUTION_TIME, "unit")
        assert stats.count == 5
        assert stats.mean == 3.0
        assert stats.median == 3.0
        assert stats.min_value == 1.0
        assert stats.max_value == 5.0
        assert stats.unit == "seconds"
        assert stats.language == "go"
        assert stats.test_type == "unit"
        assert stats.time_period == "24 hours"
        # With only 5 data points, percentile_95 and percentile_99 fall back to max
        assert stats.percentile_95 == 5.0
        assert stats.percentile_99 == 5.0

    def test_check_performance_regression_no_benchmark(self, tracker):
        result = tracker.check_performance_regression(PerformanceMetricType.DISK_IO, 1.0, "unit")
        assert result == {
            "has_regression": False,
            "message": "No benchmark defined for disk_io",
            "severity": "info",
        }

    def test_check_performance_regression_within_tolerance(self):
        tracker = LanguagePerformanceTracker("go")
        # Go container_startup benchmark: target=2.0, tolerance=20%
        result = tracker.check_performance_regression(
            PerformanceMetricType.CONTAINER_STARTUP, 2.2, "unit"
        )
        assert result["has_regression"] is False
        assert result["severity"] == "info"
        assert result["message"] == "Performance within acceptable range"
        assert result["deviation_percent"] == pytest.approx(10.0)

    def test_check_performance_regression_detected(self):
        tracker = LanguagePerformanceTracker("go")
        # Go container_startup benchmark: target=2.0, tolerance=20%
        # value 10.0 is 400% deviation which is > 50%, so severity should be "critical"
        result = tracker.check_performance_regression(
            PerformanceMetricType.CONTAINER_STARTUP, 10.0, "unit"
        )
        assert result["has_regression"] is True
        assert result["deviation_percent"] == pytest.approx(400.0)
        assert result["severity"] == "critical"
        assert result["target_value"] == 2.0
        assert result["actual_value"] == 10.0
        assert result["tolerance_percent"] == 20.0
        assert "400.0% deviation" in result["message"]

    def test_benchmarks_loaded_for_known_language(self):
        tracker = LanguagePerformanceTracker("rust")
        assert len(tracker.benchmarks) == 3
        assert "container_startup" in tracker.benchmarks
        assert "test_compilation" in tracker.benchmarks
        assert "memory_usage" in tracker.benchmarks

    def test_benchmarks_empty_for_unknown_language(self):
        tracker = LanguagePerformanceTracker("brainfuck")
        assert tracker.benchmarks == {}


class TestPerformanceTrackerService:
    @pytest.fixture
    def service(self) -> PerformanceTrackerService:
        return PerformanceTrackerService()

    def test_get_tracker_creates_new(self, service):
        tracker = service.get_tracker("python")
        assert tracker.language == "python"
        assert "python" in service.language_trackers
        assert service.language_trackers["python"] is tracker

    def test_get_tracker_returns_same(self, service):
        t1 = service.get_tracker("python")
        t2 = service.get_tracker("python")
        assert t1 is t2

    def test_record_execution_metrics(self, service):
        service.record_execution_metrics(
            language="python",
            test_type="unit",
            container_startup_time=1.0,
            test_execution_time=5.0,
            cleanup_time=0.5,
            peak_memory_mb=256.0,
            avg_cpu_percent=45.0,
        )
        tracker = service.get_tracker("python")
        assert len(tracker.data_points) == 5  # 5 metrics recorded
        metric_types = [dp.metric_type for dp in tracker.data_points]
        assert PerformanceMetricType.CONTAINER_STARTUP in metric_types
        assert PerformanceMetricType.TEST_EXECUTION in metric_types
        assert PerformanceMetricType.CLEANUP_TIME in metric_types
        assert PerformanceMetricType.MEMORY_USAGE in metric_types
        assert PerformanceMetricType.CPU_USAGE in metric_types
        # Verify values
        values_by_type = {dp.metric_type: dp.value for dp in tracker.data_points}
        assert values_by_type[PerformanceMetricType.CONTAINER_STARTUP] == 1.0
        assert values_by_type[PerformanceMetricType.TEST_EXECUTION] == 5.0
        assert values_by_type[PerformanceMetricType.CLEANUP_TIME] == 0.5
        assert values_by_type[PerformanceMetricType.MEMORY_USAGE] == 256.0
        assert values_by_type[PerformanceMetricType.CPU_USAGE] == 45.0

    def test_get_performance_report(self, service):
        report = service.get_performance_report("python")
        assert report == {
            "language": "python",
            "time_period": "24 hours",
            "statistics": {},
            "regression_alerts": [],
        }

    def test_get_cross_language_comparison(self, service):
        comparison = service.get_cross_language_comparison()
        assert comparison == {
            "time_period": "24 hours",
            "languages": {},
            "rankings": {"container_startup": [], "test_execution": []},
        }

    def test_export_metrics_single_language(self, service):
        service.record_execution_metrics(
            language="go",
            test_type="unit",
            container_startup_time=1.0,
            test_execution_time=5.0,
            cleanup_time=0.5,
            peak_memory_mb=256.0,
            avg_cpu_percent=45.0,
        )
        exported = service.export_metrics(language="go")
        data = json.loads(exported)
        assert data["language"] == "go"
        assert data["time_period"] == "24 hours"
        assert len(data["data_points"]) == 5

    def test_export_metrics_all_languages(self, service):
        service.record_execution_metrics(
            language="go",
            test_type="unit",
            container_startup_time=1.0,
            test_execution_time=5.0,
            cleanup_time=0.5,
            peak_memory_mb=256.0,
            avg_cpu_percent=45.0,
        )
        exported = service.export_metrics()
        data = json.loads(exported)
        assert data["time_period"] == "24 hours"
        assert "go" in data["languages"]
        assert len(data["languages"]["go"]) == 5

    def test_get_global_performance_tracker(self):
        tracker = get_performance_tracker()
        assert isinstance(tracker.language_trackers, dict)
        # Verify it always returns the same module-level singleton
        assert get_performance_tracker() is tracker

    def test_get_performance_report_with_data(self, service):
        """Report with actual data populates statistics and alerts (lines 531-535)."""
        # Record enough data to generate statistics
        for _ in range(5):
            service.record_execution_metrics(
                language="python",
                test_type="unit",
                container_startup_time=1.0,
                test_execution_time=5.0,
                cleanup_time=0.5,
                peak_memory_mb=256.0,
                avg_cpu_percent=45.0,
            )
        report = service.get_performance_report("python", hours=1)
        assert report["language"] == "python"
        assert report["time_period"] == "1 hours"
        # Should have statistics for the 4 metric types x "unit" test_type
        assert len(report["statistics"]) == 4
        assert "container_startup_unit" in report["statistics"]
        assert "test_execution_unit" in report["statistics"]
        assert "memory_usage_unit" in report["statistics"]
        assert "cpu_usage_unit" in report["statistics"]
        # Verify one statistic in detail
        cs_stats = report["statistics"]["container_startup_unit"]
        assert cs_stats["count"] == 5
        assert cs_stats["mean"] == 1.0
        assert cs_stats["min_value"] == 1.0
        assert cs_stats["max_value"] == 1.0

    def test_get_cross_language_comparison_with_data(self, service):
        """Cross-language comparison with data populates languages and rankings (lines 552-570)."""
        for lang in ["python", "javascript"]:
            for _ in range(5):
                service.record_execution_metrics(
                    language=lang,
                    test_type="unit",
                    container_startup_time=1.0 if lang == "python" else 2.0,
                    test_execution_time=5.0,
                    cleanup_time=0.5,
                    peak_memory_mb=256.0,
                    avg_cpu_percent=45.0,
                )
        comparison = service.get_cross_language_comparison(hours=1)
        assert len(comparison["languages"]) == 2
        assert "python" in comparison["languages"]
        assert "javascript" in comparison["languages"]
        # Python container_startup mean should be 1.0, JS should be 2.0
        assert comparison["languages"]["python"]["container_startup"]["mean"] == 1.0
        assert comparison["languages"]["javascript"]["container_startup"]["mean"] == 2.0
        # Both should have same test_execution mean
        assert comparison["languages"]["python"]["test_execution"]["mean"] == 5.0
        assert comparison["languages"]["javascript"]["test_execution"]["mean"] == 5.0
        # Rankings should be sorted by mean (ascending)
        cs_rankings = comparison["rankings"]["container_startup"]
        assert len(cs_rankings) == 2
        assert cs_rankings[0]["language"] == "python"
        assert cs_rankings[0]["mean"] == 1.0
        assert cs_rankings[1]["language"] == "javascript"
        assert cs_rankings[1]["mean"] == 2.0


# ---------------------------------------------------------------------------
# monitoring_service.py
# ---------------------------------------------------------------------------

from models.logging_models import LogLevel  # noqa: E402
from services.test_execution.monitoring_service import (  # noqa: E402
    MetricType,
    TestExecutionMonitor,
    TestExecutionStats,
    create_test_execution_monitor,
)


class TestTestExecutionStats:
    def test_success_rate_zero(self):
        stats = TestExecutionStats()
        assert stats.success_rate() == 0.0
        assert stats.total_executions == 0
        assert stats.successful_executions == 0
        assert stats.failed_executions == 0

    def test_success_rate_calculated(self):
        stats = TestExecutionStats(
            total_executions=10,
            successful_executions=8,
            failed_executions=2,
        )
        assert stats.success_rate() == 80.0


class TestTestExecutionMonitor:
    @pytest.fixture
    def monitor(self) -> TestExecutionMonitor:
        return TestExecutionMonitor(max_metrics_history=100, max_log_history=100)

    def test_log_test_started(self, monitor):
        monitor.log_test_started("t1", "python", "unit", "container-1")
        assert "t1" in monitor.active_tests
        assert monitor.active_tests["t1"]["language"] == "python"
        assert monitor.active_tests["t1"]["test_type"] == "unit"
        assert monitor.active_tests["t1"]["container_name"] == "container-1"
        assert "t1" in monitor.test_start_times
        assert len(monitor.logs) == 1
        assert monitor.logs[0].level == LogLevel.INFO
        assert "python" in monitor.logs[0].message
        assert monitor.logs[0].test_id == "t1"

    def test_log_container_created(self, monitor):
        monitor.log_container_created("container-1", "python", 150.0)
        assert monitor.container_metrics.total_containers_created == 1
        assert monitor.container_metrics.active_containers == 1
        assert monitor.container_metrics.container_creation_time_ms == 150.0
        assert monitor.container_metrics.language_container_usage["python"] == 1

    def test_log_container_destroyed(self, monitor):
        monitor.log_container_created("container-1", "python", 150.0)
        monitor.log_container_destroyed("container-1", "python", 50.0)
        assert monitor.container_metrics.active_containers == 0
        assert monitor.container_metrics.total_containers_destroyed == 1
        assert monitor.container_metrics.container_cleanup_time_ms == 50.0

    def test_log_container_failure(self, monitor):
        monitor.log_container_failure("container-1", "python", "OOM killed")
        assert monitor.container_metrics.container_failures == 1
        assert len(monitor.logs) == 1
        assert monitor.logs[0].level == LogLevel.ERROR
        assert "OOM killed" in monitor.logs[0].message

    def test_log_security_violation(self, monitor):
        monitor.log_security_violation("t1", "python", "import_banned", "os.system used")
        assert monitor.security_metrics.security_violations == 1
        assert monitor.security_metrics.blocked_executions == 1
        assert monitor.security_metrics.violation_types == {"import_banned": 1}
        assert len(monitor.logs) == 1
        assert monitor.logs[0].level == LogLevel.WARNING

    def test_log_resource_usage(self, monitor):
        monitor.log_resource_usage("container-1", "python", 45.0, 256.0)
        assert len(monitor.metrics) == 2  # cpu + memory metrics
        cpu_metric = monitor.metrics[0]
        mem_metric = monitor.metrics[1]
        assert cpu_metric.metric_type == MetricType.RESOURCE_USAGE
        assert cpu_metric.value == 45.0
        assert cpu_metric.unit == "percent"
        assert mem_metric.metric_type == MetricType.RESOURCE_USAGE
        assert mem_metric.value == 256.0
        assert mem_metric.unit == "mb"
        assert monitor.container_metrics.resource_usage["container-1_cpu"] == 45.0
        assert monitor.container_metrics.resource_usage["container-1_memory"] == 256.0

    def test_get_statistics(self, monitor):
        stats = monitor.get_statistics()
        assert stats["execution_stats"]["total_executions"] == 0
        assert stats["execution_stats"]["successful_executions"] == 0
        assert stats["container_metrics"]["active_containers"] == 0
        assert stats["security_metrics"]["security_violations"] == 0
        assert stats["active_tests"] == 0
        assert stats["metrics_count"] == 0
        assert stats["logs_count"] == 0
        assert stats["health_status"]["overall_health"] == "poor"  # 0% success rate < 70

    def test_get_metrics_by_type(self, monitor):
        monitor.log_resource_usage("container-1", "python", 45.0, 256.0)
        metrics = monitor.get_metrics_by_type(MetricType.RESOURCE_USAGE)
        assert len(metrics) == 2
        assert all(m.metric_type == MetricType.RESOURCE_USAGE for m in metrics)

    def test_get_logs_by_level(self, monitor):
        monitor.log_test_started("t1", "python", "unit")
        logs = monitor.get_logs_by_level(LogLevel.INFO)
        assert len(logs) == 1
        assert logs[0].level == LogLevel.INFO
        assert logs[0].test_id == "t1"

    def test_get_language_performance_no_data(self, monitor):
        perf = monitor.get_language_performance("python")
        assert perf == {
            "avg_execution_time": 0.0,
            "min_execution_time": 0.0,
            "max_execution_time": 0.0,
        }

    def test_get_language_performance_with_data(self, monitor):
        monitor.performance_history["python"].append(100.0)
        monitor.performance_history["python"].append(200.0)
        perf = monitor.get_language_performance("python")
        assert perf == {
            "avg_execution_time": 150.0,
            "min_execution_time": 100.0,
            "max_execution_time": 200.0,
            "execution_count": 2,
        }

    def test_export_metrics_json(self, monitor):
        exported = monitor.export_metrics(format_type="json")
        data = json.loads(exported)
        assert data["statistics"]["execution_stats"]["total_executions"] == 0
        assert data["recent_metrics"] == []
        assert data["recent_logs"] == []
        assert "timestamp" in data

    def test_export_metrics_unsupported_format(self, monitor):
        with pytest.raises(ValueError, match="Unsupported format"):
            monitor.export_metrics(format_type="xml")

    def test_classify_error(self, monitor):
        assert monitor._classify_error("timeout after 30s") == "timeout"
        assert monitor._classify_error("memory limit exceeded") == "memory"
        assert monitor._classify_error("permission denied") == "permission"
        assert monitor._classify_error("network error") == "network"
        assert monitor._classify_error("container crash") == "container"
        assert monitor._classify_error("security violation") == "security"
        assert monitor._classify_error("compilation failed") == "compilation"
        assert monitor._classify_error("syntax error") == "syntax"
        assert monitor._classify_error("something else") == "other"

    def test_performance_trends(self, monitor):
        monitor.performance_history["python"].extend([100.0, 200.0])
        trends = monitor._get_performance_trends()
        assert "python" in trends
        assert trends["python"]["overall_avg"] == 150.0
        assert trends["python"]["recent_avg"] == 150.0
        assert trends["python"]["trend"] == "stable"

    def test_health_status_excellent(self, monitor):
        monitor.stats.total_executions = 100
        monitor.stats.successful_executions = 98
        health = monitor._get_health_status()
        assert health["overall_health"] == "excellent"
        assert health["success_rate"] == 98.0
        assert health["active_containers"] == 0
        assert health["security_violations"] == 0

    def test_health_status_poor(self, monitor):
        monitor.stats.total_executions = 100
        monitor.stats.successful_executions = 50
        health = monitor._get_health_status()
        assert health["overall_health"] == "poor"
        assert health["success_rate"] == 50.0

    def test_recommendations(self, monitor):
        monitor.stats.total_executions = 100
        monitor.stats.successful_executions = 70
        recs = monitor._get_recommendations()
        assert len(recs) == 1
        assert recs[0] == "Consider investigating frequent test failures"

    def test_container_failure_recommendations(self, monitor):
        monitor.container_metrics.container_failures = 15
        recs = monitor._get_recommendations()
        # With 0 total executions, success_rate is 0% which triggers both recommendations
        assert len(recs) == 2
        assert recs[0] == "Consider investigating frequent test failures"
        assert recs[1] == "High container failure rate detected - check Docker configuration"

    def test_factory_function(self):
        monitor = create_test_execution_monitor(100, 100)
        assert monitor.max_metrics_history == 100
        assert monitor.max_log_history == 100
        assert monitor.stats.total_executions == 0


# ---------------------------------------------------------------------------
# resource_monitor.py
# ---------------------------------------------------------------------------

from services.test_execution.resource_monitor import (  # noqa: E402
    AlertAction,
    AlertLevel,
    AlertManager,
    ResourceAlert,
    ResourceMonitor,
    ResourceThreshold,
    ResourceType,
    ResourceUsage,
)


class TestResourceThreshold:
    def test_get_alert_level_emergency(self):
        th = ResourceThreshold(
            resource_type=ResourceType.CPU,
            warning_threshold=70,
            critical_threshold=85,
            emergency_threshold=95,
            unit="percent",
        )
        assert th.get_alert_level(96) == AlertLevel.EMERGENCY
        assert th.get_alert_level(95) == AlertLevel.EMERGENCY

    def test_get_alert_level_critical(self):
        th = ResourceThreshold(
            resource_type=ResourceType.CPU,
            warning_threshold=70,
            critical_threshold=85,
            emergency_threshold=95,
            unit="percent",
        )
        assert th.get_alert_level(90) == AlertLevel.CRITICAL
        assert th.get_alert_level(85) == AlertLevel.CRITICAL

    def test_get_alert_level_warning(self):
        th = ResourceThreshold(
            resource_type=ResourceType.CPU,
            warning_threshold=70,
            critical_threshold=85,
            emergency_threshold=95,
            unit="percent",
        )
        assert th.get_alert_level(75) == AlertLevel.WARNING
        assert th.get_alert_level(70) == AlertLevel.WARNING

    def test_get_alert_level_none(self):
        th = ResourceThreshold(
            resource_type=ResourceType.CPU,
            warning_threshold=70,
            critical_threshold=85,
            emergency_threshold=95,
            unit="percent",
        )
        assert th.get_alert_level(50) is None
        assert th.get_alert_level(69.9) is None


class TestResourceUsage:
    def test_to_dict(self):
        usage = ResourceUsage(
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            resource_type=ResourceType.MEMORY,
            current_value=8.0,
            max_value=16.0,
            usage_percent=50.0,
            unit="GB",
            metadata={"available_gb": 8.0},
        )
        d = usage.to_dict()
        assert d == {
            "timestamp": "2025-01-01T00:00:00+00:00",
            "resource_type": "memory",
            "current_value": 8.0,
            "max_value": 16.0,
            "usage_percent": 50.0,
            "unit": "GB",
            "metadata": {"available_gb": 8.0},
        }


class TestResourceAlert:
    def test_to_dict(self):
        alert = ResourceAlert(
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            resource_type=ResourceType.CPU,
            alert_level=AlertLevel.WARNING,
            current_value=75.0,
            threshold_value=70.0,
            usage_percent=75.0,
            message="CPU high",
        )
        d = alert.to_dict()
        assert d == {
            "timestamp": "2025-01-01T00:00:00+00:00",
            "resource_type": "cpu",
            "alert_level": "warning",
            "current_value": 75.0,
            "threshold_value": 70.0,
            "usage_percent": 75.0,
            "message": "CPU high",
            "metadata": {},
            "acknowledged": False,
        }


class TestAlertManager:
    @pytest.fixture
    def manager(self) -> AlertManager:
        return AlertManager()

    def test_create_alert(self, manager):
        usage = ResourceUsage(
            timestamp=datetime.now(UTC),
            resource_type=ResourceType.CPU,
            current_value=85.0,
            max_value=100.0,
            usage_percent=85.0,
            unit="percent",
        )
        threshold = ResourceThreshold(
            resource_type=ResourceType.CPU,
            warning_threshold=70,
            critical_threshold=85,
            emergency_threshold=95,
            unit="percent",
        )
        alert = manager.create_alert(usage, threshold, AlertLevel.CRITICAL)
        assert alert.alert_level == AlertLevel.CRITICAL
        assert alert.resource_type == ResourceType.CPU
        assert alert.current_value == 85.0
        assert alert.threshold_value == 85  # critical_threshold
        assert alert.usage_percent == 85.0
        assert "CPU" in alert.message
        assert "CRITICAL" in alert.message
        assert alert.acknowledged is False

    def test_process_alert(self, manager):
        alert = ResourceAlert(
            timestamp=datetime.now(UTC),
            resource_type=ResourceType.CPU,
            alert_level=AlertLevel.WARNING,
            current_value=75.0,
            threshold_value=70.0,
            usage_percent=75.0,
            message="CPU high",
        )
        handler = MagicMock()
        manager.register_alert_handler(AlertAction.LOG, handler)
        manager.process_alert(alert)
        assert "cpu_warning" in manager.active_alerts
        assert manager.active_alerts["cpu_warning"] is alert
        assert len(manager.alert_history) == 1
        handler.assert_called_once_with(alert)

    def test_suppress_alert(self, manager):
        manager.suppress_alert("cpu_warning", duration_minutes=60)
        assert "cpu_warning" in manager.suppressed_alerts
        # Suppressed time should be ~60 minutes in the future
        suppressed_until = manager.suppressed_alerts["cpu_warning"]
        assert suppressed_until > datetime.now(UTC)

    def test_process_suppressed_alert(self, manager):
        manager.suppress_alert("cpu_warning", duration_minutes=60)
        alert = ResourceAlert(
            timestamp=datetime.now(UTC),
            resource_type=ResourceType.CPU,
            alert_level=AlertLevel.WARNING,
            current_value=75.0,
            threshold_value=70.0,
            usage_percent=75.0,
            message="CPU high",
        )
        handler = MagicMock()
        manager.register_alert_handler(AlertAction.LOG, handler)
        manager.process_alert(alert)
        handler.assert_not_called()
        assert "cpu_warning" not in manager.active_alerts

    def test_acknowledge_alert(self, manager):
        alert = ResourceAlert(
            timestamp=datetime.now(UTC),
            resource_type=ResourceType.CPU,
            alert_level=AlertLevel.WARNING,
            current_value=75.0,
            threshold_value=70.0,
            usage_percent=75.0,
            message="CPU high",
        )
        manager.process_alert(alert)
        assert manager.acknowledge_alert("cpu_warning") is True
        assert manager.active_alerts["cpu_warning"].acknowledged is True

    def test_acknowledge_nonexistent_alert(self, manager):
        assert manager.acknowledge_alert("nonexistent") is False

    def test_get_active_alerts(self, manager):
        assert manager.get_active_alerts() == []

    def test_get_alert_history(self, manager):
        assert manager.get_alert_history() == []

    def test_alert_history_limit(self, manager):
        for i in range(1100):
            alert = ResourceAlert(
                timestamp=datetime.now(UTC),
                resource_type=ResourceType.CPU,
                alert_level=AlertLevel.WARNING,
                current_value=75.0,
                threshold_value=70.0,
                usage_percent=75.0,
                message=f"alert {i}",
            )
            manager.process_alert(alert)
        assert len(manager.alert_history) == 1000

    def test_get_actions_for_warning(self, manager):
        alert = ResourceAlert(
            timestamp=datetime.now(UTC),
            resource_type=ResourceType.CPU,
            alert_level=AlertLevel.WARNING,
            current_value=75.0,
            threshold_value=70.0,
            usage_percent=75.0,
            message="CPU high",
        )
        actions = manager._get_actions_for_alert(alert)
        assert actions == [AlertAction.LOG, AlertAction.EMAIL]

    def test_get_actions_for_emergency_cpu(self, manager):
        alert = ResourceAlert(
            timestamp=datetime.now(UTC),
            resource_type=ResourceType.CPU,
            alert_level=AlertLevel.EMERGENCY,
            current_value=98.0,
            threshold_value=95.0,
            usage_percent=98.0,
            message="CPU emergency",
        )
        actions = manager._get_actions_for_alert(alert)
        assert actions == [
            AlertAction.LOG,
            AlertAction.EMAIL,
            AlertAction.SLACK,
            AlertAction.WEBHOOK,
            AlertAction.SCALE_DOWN,
            AlertAction.TERMINATE_CONTAINERS,
        ]


class TestResourceMonitor:
    @pytest.fixture
    def monitor(self) -> ResourceMonitor:
        return ResourceMonitor(monitoring_interval=30)

    def test_default_thresholds(self, monitor):
        assert ResourceType.CPU in monitor.thresholds
        assert ResourceType.MEMORY in monitor.thresholds
        assert ResourceType.DISK in monitor.thresholds
        assert ResourceType.CONTAINER_COUNT in monitor.thresholds
        # Verify CPU threshold values
        cpu_th = monitor.thresholds[ResourceType.CPU]
        assert cpu_th.warning_threshold == 70.0
        assert cpu_th.critical_threshold == 85.0
        assert cpu_th.emergency_threshold == 95.0

    def test_set_threshold(self, monitor):
        new_threshold = ResourceThreshold(
            resource_type=ResourceType.CPU,
            warning_threshold=50,
            critical_threshold=70,
            emergency_threshold=90,
            unit="percent",
        )
        monitor.set_threshold(ResourceType.CPU, new_threshold)
        assert monitor.thresholds[ResourceType.CPU].warning_threshold == 50
        assert monitor.thresholds[ResourceType.CPU].critical_threshold == 70
        assert monitor.thresholds[ResourceType.CPU].emergency_threshold == 90

    def test_get_resource_history_empty(self, monitor):
        history = monitor.get_resource_history(hours=1)
        assert history == []

    def test_export_data(self, monitor):
        exported = monitor.export_data(hours=1)
        data = json.loads(exported)
        assert data["resource_history"] == []
        assert data["alert_history"] == []
        assert data["active_alerts"] == []
        assert ResourceType.CPU.value in data["thresholds"]
        assert ResourceType.MEMORY.value in data["thresholds"]

    def test_log_alert_handler(self, monitor):
        alert = ResourceAlert(
            timestamp=datetime.now(UTC),
            resource_type=ResourceType.CPU,
            alert_level=AlertLevel.WARNING,
            current_value=75.0,
            threshold_value=70.0,
            usage_percent=75.0,
            message="CPU high",
        )
        # Should not raise
        monitor._log_alert_handler(alert)

    def test_scale_down_handler(self, monitor):
        alert = ResourceAlert(
            timestamp=datetime.now(UTC),
            resource_type=ResourceType.CPU,
            alert_level=AlertLevel.EMERGENCY,
            current_value=98.0,
            threshold_value=95.0,
            usage_percent=98.0,
            message="CPU emergency",
        )
        # Should not raise
        monitor._scale_down_handler(alert)


# ---------------------------------------------------------------------------
# docker_service.py
# ---------------------------------------------------------------------------

from services.docker_service import create_docker_service  # noqa: E402


class TestDockerServiceModule:
    def test_create_docker_service(self):
        svc = create_docker_service()
        # create_docker_service returns a DockerService stub singleton
        assert svc._get_supported_tools if hasattr(svc, "_get_supported_tools") else True
        # Verify it's a DockerService with expected stub attributes
        assert hasattr(svc, "find_container_by_name")
        assert hasattr(svc, "get_language_container_image")
        assert svc.get_language_container_image("python") == "python:3.13-slim"
        assert svc.get_language_container_image("go") == "golang:1.23-alpine"


# ---------------------------------------------------------------------------
# code_execution_service.py
# ---------------------------------------------------------------------------

from services.code_execution_service import (  # noqa: E402
    CodeExecutionService,
    get_code_execution_service,
)


class TestCodeExecutionService:
    def test_init(self):
        svc = CodeExecutionService()
        assert svc.docker_service is None
        assert svc.cache_manager is None
        assert svc.security_service is None
        assert svc.test_orchestrator is None
        assert svc.settings is None

    @pytest.mark.asyncio
    async def test_execute_code_returns_stub(self):
        svc = CodeExecutionService()
        result = await svc.execute_code("print(1)", "python")
        assert result["exit_code"] == 1
        assert result["execution_time"] == 0.0
        assert "unreachable" in result["stderr"].lower() or "executor" in result["stderr"].lower()

    @pytest.mark.asyncio
    async def test_execute_tests_returns_stub(self):
        svc = CodeExecutionService()
        result = await svc.execute_tests("code", "test", "python")
        assert result["exit_code"] == 1
        assert result["tests_passed"] == 0
        assert result["tests_failed"] == 0

    @pytest.mark.asyncio
    async def test_execute_test_raises(self):
        from fastapi import HTTPException as _HTTPException

        svc = CodeExecutionService()
        # execute_test now forwards to executor; with no language attr it raises HTTPException(400)
        with pytest.raises(_HTTPException) as exc_info:
            await svc.execute_test(None)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_analyze_quality_calls_service(self):
        svc = CodeExecutionService()
        # Unreachable coding-analysis is "not computed" (success False + error),
        # never a raise that surfaces as an opaque 500 and never a 0-issue pass.
        from types import SimpleNamespace

        result = await svc.analyze_quality(SimpleNamespace(container_name="ws1"))
        assert result.success is False
        assert result.error and "not computed" in result.error

    @pytest.mark.asyncio
    async def test_analyze_container_quality_calls_service(self):
        svc = CodeExecutionService()
        # Unreachable coding-analysis is "not computed" (success False + error),
        # never a raise that surfaces as an opaque 500 and never a 0-issue pass.
        result = await svc.analyze_container_quality(None)
        assert result.success is False
        assert result.error and "not computed" in result.error

    @pytest.mark.asyncio
    async def test_validate_code_security_raises(self):
        from unittest.mock import AsyncMock, patch

        import httpx

        svc = CodeExecutionService()
        # Unreachable coding-security is "not computed" (is_safe None), never "safe".
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("unreachable"))
        with patch(
            "services.code_execution_service._get_security_http_client", return_value=mock_client
        ):
            result = await svc._validate_code_security("code", "python")
        assert isinstance(result, dict)
        assert result["is_safe"] is None
        assert result["computed"] is False
        assert "unreachable" in result["reason"].lower()

    def test_get_supported_tools(self):
        svc = CodeExecutionService()
        assert "ruff" in svc._get_supported_tools("python")
        assert "golint" in svc._get_supported_tools("go")
        assert svc._get_supported_tools("unknown") == []

    @pytest.mark.asyncio
    async def test_get_service_health(self):
        svc = CodeExecutionService()
        health = await svc.get_service_health()
        assert health == {"status": "stub", "message": "CodeExecutionService stub"}

    def test_singleton(self):
        # Reset global first
        import services.code_execution_service as mod

        mod._code_execution_service = None
        a = get_code_execution_service()
        b = get_code_execution_service()
        assert a is b
        # Cleanup
        mod._code_execution_service = None

    def test_init_with_services(self):
        """Test init with docker_service, cache_manager, security_service."""
        docker = MagicMock()
        cache = MagicMock()
        security = MagicMock()
        svc = CodeExecutionService(
            docker_service=docker,
            cache_manager=cache,
            security_service=security,
        )
        assert svc.docker_service is docker
        assert svc.cache_manager is cache
        assert svc.security_service is security
        assert svc.test_orchestrator is None
        assert svc.settings is None

    @pytest.mark.asyncio
    async def test_execute_code_custom_timeout(self):
        svc = CodeExecutionService()
        result = await svc.execute_code("print(1)", "python", timeout=10)
        assert result["exit_code"] == 1
        assert result["stdout"] == ""
        assert result["execution_time"] == 0.0

    @pytest.mark.asyncio
    async def test_execute_tests_custom_timeout(self):
        svc = CodeExecutionService()
        result = await svc.execute_tests("code", "test", "javascript", timeout=15)
        assert result["exit_code"] == 1
        assert result["tests_failed"] == 0
        assert result["tests_passed"] == 0

    @pytest.mark.asyncio
    async def test_get_service_health_has_message(self):
        svc = CodeExecutionService()
        health = await svc.get_service_health()
        assert health["message"] == "CodeExecutionService stub"
        assert health["status"] == "stub"


# ---------------------------------------------------------------------------
# security_service.py
# ---------------------------------------------------------------------------

from models.security_models import SecurityLevel  # noqa: E402
from services.security_service import (  # noqa: E402
    SystemSecurityService,
    create_security_service,
    get_security_service,
)


class TestSystemSecurityService:
    @pytest.fixture
    def svc(self) -> SystemSecurityService:
        return SystemSecurityService()

    @pytest.mark.asyncio
    async def test_validate_code_safe(self, svc):
        result = await svc.validate_code("print(1)", "python")
        assert result.is_safe is True
        assert result.violations == []
        assert result.risk_score == 0.0
        # Sandbox config should be MEDIUM default
        assert result.sandbox_config["network_access"] is False
        assert result.sandbox_config["file_system"] == "read_only"
        assert result.sandbox_config["max_execution_time"] == 120

    def test_get_sandbox_config_low(self, svc):
        cfg = svc.get_sandbox_config(SecurityLevel.LOW)
        assert cfg == {
            "network_access": True,
            "file_system": "read_write",
            "max_execution_time": 300,
        }

    def test_get_sandbox_config_high(self, svc):
        cfg = svc.get_sandbox_config(SecurityLevel.HIGH)
        assert cfg == {
            "network_access": False,
            "file_system": "none",
            "max_execution_time": 60,
        }

    def test_get_sandbox_config_critical(self, svc):
        cfg = svc.get_sandbox_config(SecurityLevel.CRITICAL)
        assert cfg == {
            "network_access": False,
            "file_system": "none",
            "max_execution_time": 30,
        }

    @pytest.mark.asyncio
    async def test_validate_code_security_with_string(self, svc):
        result = await svc.validate_code_security("print(1)", "python")
        assert result.is_safe is True
        assert result.risk_score == 0.0
        assert result.violations == []

    @pytest.mark.asyncio
    async def test_validate_code_security_with_request(self, svc):
        from models.security_models import SecurityValidationRequest

        request = SecurityValidationRequest(code="print(1)", language="python")
        result = await svc.validate_code_security(request)
        assert result.is_safe is True
        assert result.risk_score == 0.0

    @pytest.mark.asyncio
    async def test_validate_imports(self, svc):
        result = await svc.validate_imports("import os", "python")
        assert result.is_safe is True
        assert result.violations == []
        assert result.risk_score == 0.0
        assert result.sandbox_config is None

    def test_generate_sandbox_config(self, svc):
        cfg = svc.generate_sandbox_config(security_level=SecurityLevel.MEDIUM)
        assert cfg == {
            "network_access": False,
            "file_system": "read_only",
            "max_execution_time": 120,
        }

    def test_generate_sandbox_config_non_enum(self, svc):
        cfg = svc.generate_sandbox_config(security_level="not-an-enum")
        # Falls back to MEDIUM
        assert cfg == {
            "network_access": False,
            "file_system": "read_only",
            "max_execution_time": 120,
        }

    @pytest.mark.asyncio
    async def test_scan_container(self, svc):
        result = await svc.scan_container("test-container")
        assert result == {
            "container_name": "test-container",
            "vulnerabilities": [],
            "scan_type": "full",
        }

    def test_init_with_dict_config(self):
        svc = SystemSecurityService(config={"key": "val"})
        assert svc.config == {"key": "val"}

    def test_init_with_non_dict_config(self):
        svc = SystemSecurityService(config="not-a-dict")
        assert svc.config == {}

    def test_singleton(self):
        import services.security_service as mod

        mod._security_service = None
        a = create_security_service()
        b = get_security_service()
        assert a is b
        # Cleanup
        mod._security_service = None


# ---------------------------------------------------------------------------
# pydantic_ai_logger.py
# ---------------------------------------------------------------------------

from models.pydantic_ai_models import (  # noqa: E402
    AgentConfiguration,
    AIAnalysisRequest,
    AIAnalysisResponse,
    PydanticAIAnalysisType,
    PydanticAITestFailureContext,
)
from services.test_execution.pydantic_ai_logger import (  # noqa: E402
    PydanticAILogEntry,
    PydanticAILogger,
    create_pydantic_ai_logger,
    get_pydantic_ai_logger,
)


class TestPydanticAILogEntry:
    def test_creation(self):
        entry = PydanticAILogEntry(
            level="INFO",
            operation="test_op",
            message="test message",
        )
        assert entry.level == "INFO"
        assert entry.operation == "test_op"
        assert entry.message == "test message"
        assert isinstance(entry.timestamp, datetime)
        assert entry.timestamp.tzinfo is not None  # timezone-aware
        assert entry.agent_id is None
        assert entry.request_id is None
        assert entry.test_id is None
        assert entry.duration_ms is None
        assert entry.success is None
        assert entry.error_details is None
        assert entry.ai_model_info is None
        assert entry.usage_stats is None
        assert entry.metadata == {}


class TestPydanticAILogger:
    @pytest.fixture
    def ai_logger(self) -> PydanticAILogger:
        return PydanticAILogger("test-ai-logger")

    def test_log_agent_initialization_success(self, ai_logger):
        config = AgentConfiguration(
            agent_id="agent-1",
            ai_model_provider="openai",
            ai_model_name="gpt-4",
        )
        ai_logger.log_agent_initialization(config, success=True, duration_ms=100.0)
        assert len(ai_logger.entries) == 1
        entry = ai_logger.entries[0]
        assert entry.success is True
        assert entry.level == "INFO"
        assert entry.operation == "agent_initialization"
        assert entry.agent_id == "agent-1"
        assert entry.duration_ms == 100.0
        assert entry.ai_model_info["provider"] == "openai"
        assert entry.ai_model_info["model"] == "gpt-4"
        assert "succeeded" in entry.message

    def test_log_agent_initialization_failure(self, ai_logger):
        config = AgentConfiguration(
            agent_id="agent-1",
            ai_model_provider="openai",
            ai_model_name="gpt-4",
        )
        ai_logger.log_agent_initialization(
            config,
            success=False,
            error_details={"error": "API key invalid"},
        )
        assert len(ai_logger.entries) == 1
        entry = ai_logger.entries[0]
        assert entry.success is False
        assert entry.level == "ERROR"
        assert entry.error_details == {"error": "API key invalid"}
        assert "failed" in entry.message
        assert "API key invalid" in entry.message

    def test_log_analysis_request(self, ai_logger):
        ctx = PydanticAITestFailureContext(
            test_id="t1",
            language="python",
            test_type="unit",
            exit_code=1,
            execution_time_ms=100.0,
        )
        request = AIAnalysisRequest(
            context=ctx,
            analysis_types=[PydanticAIAnalysisType.FAILURE_ANALYSIS],
        )
        ai_logger.log_analysis_request(request, "req-1")
        assert len(ai_logger.entries) == 1
        entry = ai_logger.entries[0]
        assert entry.request_id == "req-1"
        assert entry.test_id == "t1"
        assert entry.operation == "analysis_request"
        assert entry.level == "INFO"
        assert entry.metadata["language"] == "python"
        assert entry.metadata["test_type"] == "unit"
        assert entry.metadata["exit_code"] == 1

    def test_log_analysis_response_success(self, ai_logger):
        response = AIAnalysisResponse(
            request_id="req-1",
            success=True,
            total_analysis_time_ms=200.0,
        )
        ai_logger.log_analysis_response(response, duration_ms=200.0, agent_id="agent-1")
        assert len(ai_logger.entries) == 1
        entry = ai_logger.entries[0]
        assert entry.success is True
        assert entry.level == "INFO"
        assert entry.operation == "analysis_response"
        assert entry.agent_id == "agent-1"
        assert entry.request_id == "req-1"
        assert entry.duration_ms == 200.0
        assert "completed" in entry.message

    def test_log_analysis_response_failure(self, ai_logger):
        response = AIAnalysisResponse(
            request_id="req-1",
            success=False,
            total_analysis_time_ms=200.0,
            error_message="Model failed",
        )
        ai_logger.log_analysis_response(response, duration_ms=200.0)
        entry = ai_logger.entries[0]
        assert entry.success is False
        assert entry.level == "ERROR"
        assert entry.error_details == {"error_message": "Model failed"}
        assert "failed" in entry.message

    def test_log_model_interaction_success(self, ai_logger):
        ai_logger.log_model_interaction(
            agent_id="agent-1",
            ai_model_provider="openai",
            ai_model_name="gpt-4",
            operation="completion",
            success=True,
            duration_ms=150.0,
            tokens_used=500,
            cost_usd=0.01,
        )
        assert len(ai_logger.entries) == 1
        entry = ai_logger.entries[0]
        assert entry.success is True
        assert entry.operation == "model_completion"
        assert entry.agent_id == "agent-1"
        assert entry.duration_ms == 150.0
        assert entry.usage_stats == {"tokens_used": 500, "cost_usd": 0.01}
        assert entry.ai_model_info["provider"] == "openai"
        assert entry.ai_model_info["model"] == "gpt-4"

    def test_log_model_interaction_failure(self, ai_logger):
        ai_logger.log_model_interaction(
            agent_id="agent-1",
            ai_model_provider="openai",
            ai_model_name="gpt-4",
            operation="completion",
            success=False,
            duration_ms=150.0,
            error_details={"error": "rate limit"},
        )
        entry = ai_logger.entries[0]
        assert entry.success is False
        assert entry.level == "WARNING"
        assert entry.error_details == {"error": "rate limit"}

    def test_log_error(self, ai_logger):
        try:
            raise ValueError("bad value")
        except ValueError as e:
            ai_logger.log_error(
                "test_op",
                e,
                agent_id="agent-1",
                request_id="req-1",
                test_id="t1",
                duration_ms=50.0,
                additional_context={"extra": "data"},
            )
        assert len(ai_logger.entries) == 1
        entry = ai_logger.entries[0]
        assert entry.error_details["error_type"] == "ValueError"
        assert entry.error_details["error_message"] == "bad value"
        assert entry.error_details["extra"] == "data"
        assert entry.success is False
        assert entry.level == "ERROR"
        assert entry.agent_id == "agent-1"
        assert entry.request_id == "req-1"
        assert entry.test_id == "t1"
        assert entry.duration_ms == 50.0
        assert entry.operation == "test_op"

    def test_log_performance_metric(self, ai_logger):
        ai_logger.log_performance_metric(
            "latency",
            42.0,
            agent_id="agent-1",
            operation="inference",
            metadata={"key": "val"},
        )
        assert len(ai_logger.entries) == 1
        entry = ai_logger.entries[0]
        assert entry.operation == "inference"
        assert entry.agent_id == "agent-1"
        assert entry.metadata["metric_name"] == "latency"
        assert entry.metadata["metric_value"] == 42.0
        assert entry.metadata["key"] == "val"
        assert entry.level == "DEBUG"

    def test_log_health_check_pass(self, ai_logger):
        ai_logger.log_health_check("agent-1", healthy=True, response_time_ms=5.0)
        entry = ai_logger.entries[0]
        assert entry.success is True
        assert entry.level == "INFO"
        assert entry.operation == "health_check"
        assert entry.agent_id == "agent-1"
        assert entry.duration_ms == 5.0
        assert "passed" in entry.message

    def test_log_health_check_fail(self, ai_logger):
        ai_logger.log_health_check(
            "agent-1",
            healthy=False,
            error_details={"error": "timeout"},
        )
        entry = ai_logger.entries[0]
        assert entry.success is False
        assert entry.level == "WARNING"
        assert entry.error_details == {"error": "timeout"}
        assert "failed" in entry.message

    def test_log_batch_analysis(self, ai_logger):
        ai_logger.log_batch_analysis(
            batch_size=10,
            successful_count=8,
            failed_count=2,
            total_duration_ms=1000.0,
            average_duration_ms=100.0,
        )
        assert len(ai_logger.entries) == 1
        entry = ai_logger.entries[0]
        assert entry.operation == "batch_analysis"
        assert entry.duration_ms == 1000.0
        assert entry.success is False  # failed_count > 0
        assert entry.metadata["batch_size"] == 10
        assert entry.metadata["successful_count"] == 8
        assert entry.metadata["failed_count"] == 2
        assert entry.metadata["success_rate"] == 0.8
        assert entry.metadata["average_duration_ms"] == 100.0

    def test_get_recent_entries(self, ai_logger):
        for i in range(5):
            ai_logger.log_performance_metric(f"metric_{i}", float(i))
        entries = ai_logger.get_recent_entries(limit=3)
        assert len(entries) == 3
        # Should be the last 3 entries
        assert entries[0].metadata["metric_name"] == "metric_2"
        assert entries[1].metadata["metric_name"] == "metric_3"
        assert entries[2].metadata["metric_name"] == "metric_4"

    def test_get_entries_by_operation(self, ai_logger):
        ai_logger.log_performance_metric("m1", 1.0)
        ai_logger.log_health_check("agent-1", True)
        entries = ai_logger.get_entries_by_operation("health_check")
        assert len(entries) == 1
        assert entries[0].operation == "health_check"

    def test_get_entries_by_agent(self, ai_logger):
        ai_logger.log_health_check("agent-1", True)
        ai_logger.log_health_check("agent-2", True)
        entries = ai_logger.get_entries_by_agent("agent-1")
        assert len(entries) == 1
        assert entries[0].agent_id == "agent-1"

    def test_get_error_entries(self, ai_logger):
        ai_logger.log_performance_metric("m1", 1.0)
        try:
            raise ValueError("oops")
        except ValueError as e:
            ai_logger.log_error("op", e)
        errors = ai_logger.get_error_entries()
        assert len(errors) == 1
        assert errors[0].level == "ERROR"
        assert errors[0].error_details["error_type"] == "ValueError"

    def test_get_performance_summary_empty(self, ai_logger):
        summary = ai_logger.get_performance_summary()
        assert summary == {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "average_duration_ms": 0.0,
            "min_duration_ms": 0.0,
            "max_duration_ms": 0.0,
            "success_rate": 0.0,
        }

    def test_get_performance_summary_with_data(self, ai_logger):
        for i in range(3):
            response = AIAnalysisResponse(
                request_id=f"req-{i}",
                success=True,
                total_analysis_time_ms=100.0 * (i + 1),
            )
            ai_logger.log_analysis_response(response, duration_ms=100.0 * (i + 1))
        summary = ai_logger.get_performance_summary()
        assert summary["total_analyses"] == 3
        assert summary["successful_analyses"] == 3
        assert summary["failed_analyses"] == 0
        assert summary["success_rate"] == 1.0
        assert summary["min_duration_ms"] == 100.0
        assert summary["max_duration_ms"] == 300.0
        assert summary["average_duration_ms"] == 200.0

    def test_export_logs_json(self, ai_logger):
        ai_logger.log_performance_metric("m1", 1.0)
        exported = ai_logger.export_logs(format="json", limit=10)
        data = json.loads(exported)
        assert len(data) == 1
        assert data[0]["operation"] == "performance_metric"
        assert data[0]["level"] == "DEBUG"

    def test_export_logs_csv(self, ai_logger):
        ai_logger.log_performance_metric("m1", 1.0)
        exported = ai_logger.export_logs(format="csv")
        lines = exported.strip().split("\n")
        assert len(lines) == 2  # header + 1 data row
        header = lines[0]
        assert "operation" in header
        assert "level" in header
        assert "message" in header

    def test_export_logs_unsupported(self, ai_logger):
        with pytest.raises(ValueError, match="Unsupported format"):
            ai_logger.export_logs(format="xml")

    def test_clear_logs(self, ai_logger):
        ai_logger.log_performance_metric("m1", 1.0)
        assert len(ai_logger.entries) == 1
        ai_logger.clear_logs()
        assert len(ai_logger.entries) == 0

    def test_max_entries_limit(self, ai_logger):
        ai_logger.max_entries = 5
        for i in range(10):
            ai_logger.log_performance_metric(f"m{i}", float(i))
        assert len(ai_logger.entries) == 5
        # Should keep the last 5 entries
        assert ai_logger.entries[0].metadata["metric_name"] == "m5"
        assert ai_logger.entries[4].metadata["metric_name"] == "m9"

    def test_global_instance(self):
        import services.test_execution.pydantic_ai_logger as mod

        mod._pydantic_ai_logger = None
        logger = get_pydantic_ai_logger()
        assert logger.max_entries == 1000
        assert logger.entries == []
        # Verify singleton behavior
        assert get_pydantic_ai_logger() is logger
        # Cleanup
        mod._pydantic_ai_logger = None

    def test_create_logger(self):
        logger = create_pydantic_ai_logger("custom-name")
        assert logger.logger.name == "custom-name"
        assert logger.max_entries == 1000
        assert logger.entries == []
