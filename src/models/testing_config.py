"""Testing configuration models for background tasks and quality analysis."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TestingConfiguration(BaseModel):
    """Configuration for testing and quality analysis."""

    # Basic settings
    default_timeout: int = Field(
        default=30, ge=1, le=300, description="Default test timeout in seconds"
    )
    max_memory_mb: int = Field(
        default=512,
        ge=128,
        le=2048,
        description="Maximum memory per container in MB",
    )
    max_cpu_quota: int = Field(
        default=50000,
        ge=10000,
        le=100000,
        description="Maximum CPU quota (100000 = 1 CPU)",
    )

    # Security settings
    network_isolation: bool = Field(
        default=True, description="Enable network isolation for containers"
    )
    readonly_filesystem: bool = Field(
        default=True, description="Use read-only filesystem for containers"
    )
    enable_security_scanning: bool = Field(
        default=True, description="Enable security scanning of code"
    )

    # Language-specific settings
    python_version: str = Field(default="3.11", description="Python version to use")
    node_version: str = Field(default="18", description="Node.js version to use")

    # Quality tool settings
    enabled_python_tools: list[str] = Field(
        default=["ruff", "ruff-format", "ty", "bandit"],
        description="Enabled Python quality tools",
    )
    enabled_js_tools: list[str] = Field(
        default=["eslint", "prettier"],
        description="Enabled JavaScript quality tools",
    )
    enabled_ts_tools: list[str] = Field(
        default=["eslint", "prettier", "tsc"],
        description="Enabled TypeScript quality tools",
    )

    # Analysis settings
    quality_score_threshold: float = Field(
        default=7.0,
        ge=0.0,
        le=10.0,
        description="Minimum quality score threshold",
    )
    auto_fix_enabled: bool = Field(
        default=False, description="Enable automatic resolving of issues"
    )
    fail_on_security_issues: bool = Field(
        default=True, description="Fail analysis on security issues"
    )

    # Caching settings
    cache_test_results: bool = Field(default=True, description="Cache test execution results")
    cache_quality_results: bool = Field(default=True, description="Cache quality analysis results")
    cache_ttl_seconds: int = Field(
        default=3600, ge=300, le=86400, description="Cache TTL in seconds"
    )

    # Container settings
    cleanup_containers: bool = Field(
        default=True,
        description="Automatically cleanup containers after execution",
    )
    container_registry: str = Field(default="docker.io", description="Container registry to use")
    base_images: dict[str, str] = Field(
        default={
            "python": "code_graph_helper-api:latest",
            "javascript": "node:18-slim",
            "typescript": "node:18-slim",
        },
        description="Base container images for each language",
    )


class DependencyScanConfig(BaseModel):
    """Configuration for dependency scanning tasks."""

    enabled: bool = Field(default=True, description="Enable dependency scanning")
    scan_dev_dependencies: bool = Field(
        default=False, description="Include dev dependencies in scan"
    )
    security_advisory_check: bool = Field(default=True, description="Check for security advisories")
    outdated_package_check: bool = Field(default=True, description="Check for outdated packages")
    max_scan_time_seconds: int = Field(
        default=300, ge=30, le=1800, description="Maximum scan time in seconds"
    )
    cache_scan_results: bool = Field(default=True, description="Cache dependency scan results")
    cache_ttl_hours: int = Field(default=24, ge=1, le=168, description="Cache TTL in hours")


class QualityScanConfig(BaseModel):
    """Configuration for quality scanning tasks."""

    enabled: bool = Field(default=True, description="Enable quality scanning")
    parallel_tool_execution: bool = Field(default=True, description="Run quality tools in parallel")
    max_parallel_tools: int = Field(
        default=4, ge=1, le=8, description="Maximum parallel tool executions"
    )
    fail_fast: bool = Field(default=False, description="Stop on first tool failure")
    generate_reports: bool = Field(default=True, description="Generate quality reports")
    include_suggestions: bool = Field(default=True, description="Include improvement suggestions")
    max_scan_time_seconds: int = Field(
        default=600, ge=60, le=3600, description="Maximum scan time in seconds"
    )
    cache_quality_results: bool = Field(default=True, description="Cache quality scan results")
    cache_ttl_hours: int = Field(default=6, ge=1, le=72, description="Cache TTL in hours")


class BackgroundTaskConfig(BaseModel):
    """Configuration for background task execution."""

    max_concurrent_tasks: int = Field(
        default=5, ge=1, le=20, description="Maximum concurrent background tasks"
    )
    task_timeout_seconds: int = Field(
        default=1800, ge=300, le=7200, description="Task timeout in seconds"
    )
    retry_failed_tasks: bool = Field(default=True, description="Retry failed tasks")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum task retry attempts")
    retry_delay_seconds: int = Field(
        default=60, ge=10, le=300, description="Delay between retries in seconds"
    )
    cleanup_completed_tasks: bool = Field(default=True, description="Cleanup completed task data")
    cleanup_delay_hours: int = Field(
        default=24, ge=1, le=168, description="Delay before cleanup in hours"
    )


class TaskExecutionResult(BaseModel):
    """Result of a background task execution."""

    task_id: str = Field(description="Unique task identifier")
    task_type: str = Field(description="Type of task executed")
    success: bool = Field(description="Whether task completed successfully")
    start_time: datetime = Field(description="Task start timestamp")
    end_time: datetime | None = Field(None, description="Task completion timestamp")
    execution_time_seconds: float | None = Field(None, description="Task execution duration")
    result_data: dict[str, Any] | None = Field(None, description="Task result data")
    error_message: str | None = Field(None, description="Error message if failed")
    retry_count: int = Field(default=0, description="Number of retry attempts")
    metadata: dict[str, Any] | None = Field(None, description="Additional task metadata")


class TestingTaskQueueConfig(BaseModel):
    """Configuration for task queue management."""

    queue_name: str = Field(description="Name of the task queue")
    max_queue_size: int = Field(default=1000, ge=10, le=10000, description="Maximum queue size")
    worker_count: int = Field(default=3, ge=1, le=10, description="Number of worker processes")
    batch_size: int = Field(default=5, ge=1, le=50, description="Tasks processed per batch")
    poll_interval_seconds: int = Field(
        default=10, ge=1, le=60, description="Queue polling interval"
    )
    priority_enabled: bool = Field(default=True, description="Enable task priority handling")
    dead_letter_queue: bool = Field(
        default=True, description="Enable dead letter queue for failed tasks"
    )


# Factory function for creating default configurations
def get_default_testing_config() -> TestingConfiguration:
    """Get default testing configuration."""
    return TestingConfiguration()


def get_default_dependency_scan_config() -> DependencyScanConfig:
    """Get default dependency scan configuration."""
    return DependencyScanConfig()


def get_default_quality_scan_config() -> QualityScanConfig:
    """Get default quality scan configuration."""
    return QualityScanConfig()


def get_default_background_task_config() -> BackgroundTaskConfig:
    """Get default background task configuration."""
    return BackgroundTaskConfig()
