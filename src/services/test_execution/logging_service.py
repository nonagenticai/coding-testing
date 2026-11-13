"""
Structured logging framework for test execution infrastructure.

This module provides comprehensive logging capabilities for containerized test execution,
including structured log formatting, log aggregation, and contextual logging.
"""

import json
import logging
import os
import queue
import sys
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from models.logging_models import LogLevel

logger = logging.getLogger(__name__)


class LogCategory(StrEnum):
    """Log categories for organization."""

    CONTAINER = "container"
    SECURITY = "security"
    PERFORMANCE = "performance"
    EXECUTION = "execution"
    RESOURCE = "resource"
    SYSTEM = "system"
    AUDIT = "audit"


@dataclass
class LogContext:
    """Context information for structured logging."""

    container_id: str | None = None
    language: str | None = None
    test_type: str | None = None
    execution_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    request_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class StructuredLogEntry:
    """Structured log entry with comprehensive metadata."""

    timestamp: datetime
    level: LogLevel
    category: LogCategory
    message: str
    context: LogContext
    metadata: dict[str, Any] = field(default_factory=dict)
    source_file: str | None = None
    source_line: int | None = None
    thread_id: str | None = None
    process_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "category": self.category.value,
            "message": self.message,
            "context": self.context.to_dict(),
            "metadata": self.metadata,
            "source_file": self.source_file,
            "source_line": self.source_line,
            "thread_id": self.thread_id,
            "process_id": self.process_id,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class LogStorage:
    """Abstract base class for log storage implementations."""

    def store_log(self, log_entry: StructuredLogEntry) -> None:
        """Store a log entry."""
        raise NotImplementedError

    def query_logs(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        level: LogLevel | None = None,
        category: LogCategory | None = None,
        context_filters: dict[str, str] | None = None,
        limit: int = 100,
    ) -> list[StructuredLogEntry]:
        """Query stored logs with filters."""
        raise NotImplementedError


class FileLogStorage(LogStorage):
    """File-based log storage implementation."""

    def __init__(self, log_directory: Path):
        """Initialize the FileLogStorage service.

        This initializes the FileLogStorage with required dependencies
        and configuration for service operations.

        Args:
            log_directory (Path): Log directory
        """
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self.current_log_file = None
        self.lock = threading.Lock()

    def store_log(self, log_entry: StructuredLogEntry) -> None:
        """Store log entry to file."""
        with self.lock:
            log_file = self._get_log_file(log_entry.timestamp)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_entry.to_json() + "\n")

    def _get_log_file(self, timestamp: datetime) -> Path:
        """Get log file path for timestamp."""
        date_str = timestamp.strftime("%Y-%m-%d")
        return self.log_directory / f"test_execution_{date_str}.jsonl"

    def query_logs(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        level: LogLevel | None = None,
        category: LogCategory | None = None,
        context_filters: dict[str, str] | None = None,
        limit: int = 100,
    ) -> list[StructuredLogEntry]:
        """Query logs from files (basic implementation)."""
        results = []
        for log_file in self.log_directory.glob("test_execution_*.jsonl"):
            try:
                with open(log_file, encoding="utf-8") as f:
                    for line in f:
                        try:
                            log_data = json.loads(line.strip())
                            if (
                                start_time
                                and datetime.fromisoformat(log_data["timestamp"]) < start_time
                            ):
                                continue
                            if (
                                end_time
                                and datetime.fromisoformat(log_data["timestamp"]) > end_time
                            ):
                                continue
                            if level and log_data["level"] != level.value:
                                continue
                            if category and log_data["category"] != category.value:
                                continue
                            if context_filters:
                                match = True
                                for key, value in context_filters.items():
                                    if log_data.get("context", {}).get(key) != value:
                                        match = False
                                        break
                                if not match:
                                    continue
                            results.append(log_data)
                            if len(results) >= limit:
                                break
                        except json.JSONDecodeError:
                            continue
                if len(results) >= limit:
                    break
            except Exception as e:
                logger.error(f"Error reading log file {log_file}: {e}")
        return results[:limit]


class LogAggregator:
    """Aggregates and processes log entries."""

    def __init__(self, storage: LogStorage):
        """Initialize the LogAggregator service.

        This initializes the LogAggregator with required dependencies
        and configuration for service operations.

        Args:
            storage (LogStorage): Storage
        """
        self.storage = storage
        self.log_queue: queue.Queue[StructuredLogEntry] = queue.Queue()
        self.processing_thread: threading.Thread | None = None
        self.shutdown_event = threading.Event()

    def start(self) -> None:
        """Start the log aggregator."""
        thread = threading.Thread(target=self._process_logs)
        thread.daemon = True
        thread.start()
        self.processing_thread = thread

    def stop(self) -> None:
        """Stop the log aggregator."""
        self.shutdown_event.set()
        if self.processing_thread:
            self.processing_thread.join()

    def add_log(self, log_entry: StructuredLogEntry) -> None:
        """Add a log entry to the processing queue."""
        self.log_queue.put(log_entry)

    def _process_logs(self) -> None:
        """Process logs from the queue."""
        while not self.shutdown_event.is_set():
            try:
                log_entry = self.log_queue.get(timeout=1.0)
                self.storage.store_log(log_entry)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing log entry: {e}")


class TestExecutionLogger:
    """Main logger for test execution infrastructure."""

    def __init__(self, storage: LogStorage):
        """Initialize the TestExecutionLogger service.

        This initializes the TestExecutionLogger with required dependencies
        and configuration for service operations.

        Args:
            storage (LogStorage): Storage
        """
        self.storage = storage
        self.aggregator = LogAggregator(storage)
        self.context_stack = threading.local()
        self.aggregator.start()

    def __del__(self) -> None:
        """Cleanup on destruction."""
        if hasattr(self, "aggregator"):
            self.aggregator.stop()

    @contextmanager
    def log_context(self, **context_kwargs) -> Generator[LogContext]:
        """Context manager for setting log context."""
        if not hasattr(self.context_stack, "contexts"):
            self.context_stack.contexts = []
        current_context = self._get_current_context()
        new_context = LogContext(**{**current_context.to_dict(), **context_kwargs})
        self.context_stack.contexts.append(new_context)
        try:
            yield new_context
        finally:
            self.context_stack.contexts.pop()

    def _get_current_context(self) -> LogContext:
        """Get current log context."""
        if not hasattr(self.context_stack, "contexts"):
            self.context_stack.contexts = []
        if self.context_stack.contexts:
            context = self.context_stack.contexts[-1]
            assert isinstance(context, LogContext), "Context must be LogContext instance"
            return context
        return LogContext()

    def _log(self, level: LogLevel, category: LogCategory, message: str, **metadata) -> None:
        """Internal logging method."""
        frame = sys._getframe(2)
        source_file = frame.f_code.co_filename
        source_line = frame.f_lineno
        log_entry = StructuredLogEntry(
            timestamp=datetime.now(UTC),
            level=level,
            category=category,
            message=message,
            context=self._get_current_context(),
            metadata=metadata,
            source_file=source_file,
            source_line=source_line,
            thread_id=threading.current_thread().name,
            process_id=os.getpid() if "os" in globals() else None,
        )
        self.aggregator.add_log(log_entry)
        std_logger = logging.getLogger(__name__)
        std_logger.log(
            getattr(logging, level.value),
            f"[{category.value}] {message} | Context: {log_entry.context.to_dict()}",
        )

    def debug(self, category: LogCategory, message: str, **metadata) -> None:
        """Log debug message."""
        self._log(LogLevel.DEBUG, category, message, **metadata)

    def info(self, category: LogCategory, message: str, **metadata) -> None:
        """Log info message."""
        self._log(LogLevel.INFO, category, message, **metadata)

    def warning(self, category: LogCategory, message: str, **metadata) -> None:
        """Log warning message."""
        self._log(LogLevel.WARNING, category, message, **metadata)

    def error(self, category: LogCategory, message: str, **metadata) -> None:
        """Log error message."""
        self._log(LogLevel.ERROR, category, message, **metadata)

    def critical(self, category: LogCategory, message: str, **metadata) -> None:
        """Log critical message."""
        self._log(LogLevel.CRITICAL, category, message, **metadata)

    def log_container_event(self, level: LogLevel, message: str, **metadata) -> None:
        """Log container-related event."""
        self._log(level, LogCategory.CONTAINER, message, **metadata)

    def log_security_event(self, level: LogLevel, message: str, **metadata) -> None:
        """Log security-related event."""
        self._log(level, LogCategory.SECURITY, message, **metadata)

    def log_performance_event(self, level: LogLevel, message: str, **metadata) -> None:
        """Log performance-related event."""
        self._log(level, LogCategory.PERFORMANCE, message, **metadata)

    def log_execution_event(self, level: LogLevel, message: str, **metadata) -> None:
        """Log execution-related event."""
        self._log(level, LogCategory.EXECUTION, message, **metadata)

    def log_resource_event(self, level: LogLevel, message: str, **metadata) -> None:
        """Log resource-related event."""
        self._log(level, LogCategory.RESOURCE, message, **metadata)

    def log_audit_event(self, level: LogLevel, message: str, **metadata) -> None:
        """Log audit-related event."""
        self._log(level, LogCategory.AUDIT, message, **metadata)

    def query_logs(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        level: LogLevel | None = None,
        category: LogCategory | None = None,
        context_filters: dict[str, str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query stored logs."""
        raw_logs = self.storage.query_logs(
            start_time=start_time,
            end_time=end_time,
            level=level,
            category=category,
            context_filters=context_filters,
            limit=limit,
        )
        result: list[dict[str, Any]] = []
        for log in raw_logs:
            if isinstance(log, StructuredLogEntry):
                result.append(log.to_dict())
            elif isinstance(log, dict):
                result.append(log)
        return result


_logger_instance = None


def get_logger() -> TestExecutionLogger:
    """Get the global logger instance."""
    global _logger_instance
    if _logger_instance is None:
        import os

        log_dir = Path(os.getenv("TEST_EXECUTION_LOG_DIR", "/tmp/test_execution_logs"))
        storage = FileLogStorage(log_dir)
        _logger_instance = TestExecutionLogger(storage)
    return _logger_instance


def setup_logging(log_directory: Path | None = None) -> TestExecutionLogger:
    """Setup logging with custom directory."""
    global _logger_instance
    if log_directory is None:
        import os

        log_directory = Path(os.getenv("TEST_EXECUTION_LOG_DIR", "/tmp/test_execution_logs"))
    storage = FileLogStorage(log_directory)
    _logger_instance = TestExecutionLogger(storage)
    return _logger_instance
