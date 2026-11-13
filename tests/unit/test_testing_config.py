"""Tests for testing configuration models."""

from datetime import datetime

import pytest

from models.testing_config import (
    BackgroundTaskConfig,
    DependencyScanConfig,
    QualityScanConfig,
    TaskExecutionResult,
    TestingConfiguration,
    TestingTaskQueueConfig,
    get_default_background_task_config,
    get_default_dependency_scan_config,
    get_default_quality_scan_config,
    get_default_testing_config,
)


class TestTestingConfiguration:
    def test_defaults(self):
        cfg = TestingConfiguration()
        assert cfg.default_timeout == 30
        assert cfg.max_memory_mb == 512
        assert cfg.network_isolation is True
        assert cfg.python_version == "3.11"
        assert "ruff-format" in cfg.enabled_python_tools
        assert cfg.quality_score_threshold == 7.0
        assert cfg.cache_ttl_seconds == 3600
        assert cfg.cleanup_containers is True
        assert "python" in cfg.base_images

    def test_override(self):
        cfg = TestingConfiguration(default_timeout=60, max_memory_mb=1024, python_version="3.12")
        assert cfg.default_timeout == 60
        assert cfg.max_memory_mb == 1024
        assert cfg.python_version == "3.12"

    def test_bounds(self):
        with pytest.raises(Exception):
            TestingConfiguration(default_timeout=0)
        with pytest.raises(Exception):
            TestingConfiguration(default_timeout=301)
        with pytest.raises(Exception):
            TestingConfiguration(max_memory_mb=64)


class TestDependencyScanConfig:
    def test_defaults(self):
        cfg = DependencyScanConfig()
        assert cfg.enabled is True
        assert cfg.scan_dev_dependencies is False
        assert cfg.max_scan_time_seconds == 300
        assert cfg.cache_ttl_hours == 24

    def test_override(self):
        cfg = DependencyScanConfig(scan_dev_dependencies=True, cache_ttl_hours=48)
        assert cfg.scan_dev_dependencies is True
        assert cfg.cache_ttl_hours == 48


class TestQualityScanConfig:
    def test_defaults(self):
        cfg = QualityScanConfig()
        assert cfg.parallel_tool_execution is True
        assert cfg.max_parallel_tools == 4
        assert cfg.fail_fast is False
        assert cfg.cache_ttl_hours == 6

    def test_bounds(self):
        with pytest.raises(Exception):
            QualityScanConfig(max_parallel_tools=0)
        with pytest.raises(Exception):
            QualityScanConfig(max_parallel_tools=9)


class TestBackgroundTaskConfig:
    def test_defaults(self):
        cfg = BackgroundTaskConfig()
        assert cfg.max_concurrent_tasks == 5
        assert cfg.retry_failed_tasks is True
        assert cfg.max_retries == 3
        assert cfg.cleanup_delay_hours == 24


class TestTaskExecutionResult:
    def test_creation(self):
        now = datetime.now()
        result = TaskExecutionResult(
            task_id="t1",
            task_type="test",
            success=True,
            start_time=now,
            execution_time_seconds=1.5,
        )
        assert result.task_id == "t1"
        assert result.success is True
        assert result.end_time is None
        assert result.retry_count == 0

    def test_failed(self):
        result = TaskExecutionResult(
            task_id="t2",
            task_type="scan",
            success=False,
            start_time=datetime.now(),
            error_message="timeout",
            retry_count=2,
        )
        assert result.success is False
        assert result.error_message == "timeout"
        assert result.retry_count == 2


class TestTestingTaskQueueConfig:
    def test_creation(self):
        cfg = TestingTaskQueueConfig(queue_name="test-queue")
        assert cfg.queue_name == "test-queue"
        assert cfg.max_queue_size == 1000
        assert cfg.worker_count == 3
        assert cfg.priority_enabled is True
        assert cfg.dead_letter_queue is True


class TestFactoryFunctions:
    def test_get_default_testing_config(self):
        cfg = get_default_testing_config()
        assert isinstance(cfg, TestingConfiguration)

    def test_get_default_dependency_scan_config(self):
        cfg = get_default_dependency_scan_config()
        assert isinstance(cfg, DependencyScanConfig)

    def test_get_default_quality_scan_config(self):
        cfg = get_default_quality_scan_config()
        assert isinstance(cfg, QualityScanConfig)

    def test_get_default_background_task_config(self):
        cfg = get_default_background_task_config()
        assert isinstance(cfg, BackgroundTaskConfig)
