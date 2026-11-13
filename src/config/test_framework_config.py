"""Test framework configuration stub."""

import logging
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TestFrameworkSecurityConfig:
    max_execution_time: int = 30
    max_memory_mb: int = 1000
    max_file_size_mb: int = 10
    max_code_length: int = 10000


@dataclass
class TestConfig:
    timeout: int = 30
    parallel: bool = True
    max_workers: int = 4


@dataclass
class TestPySecurityConfig:
    """Security configuration for test execution."""

    max_file_size_mb: int = 10
    max_execution_time: int = 30
    max_memory_mb: int = 1000


@dataclass
class TestPyCacheConfig:
    """Cache configuration for quality analysis."""

    ttl_seconds: int = 300
    max_entries: int = 1000


@dataclass
class TestPyContainerConfig:
    """Container configuration for quality analysis."""

    image_map: dict[str, str] = field(
        default_factory=lambda: {
            "python": "code-graph-test-python:latest",
            "javascript": "code-graph-test-nodejs:latest",
            "typescript": "code-graph-test-nodejs:latest",
        }
    )


@dataclass
class TestPyConfig:
    """Configuration for the quality management / TestPy service."""

    security: TestPySecurityConfig = field(default_factory=TestPySecurityConfig)
    cache: TestPyCacheConfig = field(default_factory=TestPyCacheConfig)
    container: TestPyContainerConfig = field(default_factory=TestPyContainerConfig)


@dataclass
class TestRunner:
    """Stub TestRunner for test execution service integration.

    Wraps logging and config for the test execution pipeline.
    """

    logger: Any = field(default_factory=lambda: logging.getLogger("test_runner"))
    config: Any = None

    def run(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Run tests (stub)."""
        return {"success": True, "results": []}

    async def run_async(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Run tests asynchronously (stub)."""
        return {"success": True, "results": []}
