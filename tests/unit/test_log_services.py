"""Tests for logging_service.py, log_aggregation.py, and pydantic_ai_logger.py."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from models.logging_models import LogLevel
from services.test_execution.logging_service import (
    FileLogStorage,
    LogAggregator,
    LogCategory,
    LogContext,
    StructuredLogEntry,
    TestExecutionLogger,
)

# ---------------------------------------------------------------------------
# LogContext
# ---------------------------------------------------------------------------


class TestLogContext:
    def test_to_dict_excludes_none(self):
        ctx = LogContext(container_id="c1", language="python")
        d = ctx.to_dict()
        assert d == {"container_id": "c1", "language": "python"}
        assert "user_id" not in d
        assert "test_type" not in d
        assert "execution_id" not in d

    def test_to_dict_empty(self):
        ctx = LogContext()
        assert ctx.to_dict() == {}

    def test_to_dict_all_fields(self):
        ctx = LogContext(
            container_id="c1",
            language="python",
            test_type="unit",
            execution_id="ex-1",
            user_id="u-1",
            session_id="s-1",
            request_id="r-1",
        )
        d = ctx.to_dict()
        assert d == {
            "container_id": "c1",
            "language": "python",
            "test_type": "unit",
            "execution_id": "ex-1",
            "user_id": "u-1",
            "session_id": "s-1",
            "request_id": "r-1",
        }


# ---------------------------------------------------------------------------
# StructuredLogEntry
# ---------------------------------------------------------------------------


class TestStructuredLogEntry:
    def _make_entry(self, **overrides) -> StructuredLogEntry:
        defaults = dict(
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            level=LogLevel.INFO,
            category=LogCategory.EXECUTION,
            message="test message",
            context=LogContext(),
        )
        defaults.update(overrides)
        return StructuredLogEntry(**defaults)

    def test_to_dict(self):
        entry = self._make_entry(
            metadata={"key": "val"},
            source_file="test.py",
            source_line=42,
            thread_id="main",
            process_id=1234,
        )
        d = entry.to_dict()
        assert d["level"] == "INFO"
        assert d["category"] == "execution"
        assert d["message"] == "test message"
        assert d["source_file"] == "test.py"
        assert d["source_line"] == 42
        assert d["metadata"] == {"key": "val"}
        assert d["thread_id"] == "main"
        assert d["process_id"] == 1234
        assert d["timestamp"] == "2025-01-01T00:00:00+00:00"
        assert d["context"] == {}

    def test_to_dict_with_context(self):
        entry = self._make_entry(context=LogContext(language="go", container_id="c1"))
        d = entry.to_dict()
        assert d["context"] == {"language": "go", "container_id": "c1"}

    def test_to_json(self):
        entry = self._make_entry()
        j = entry.to_json()
        data = json.loads(j)
        assert data["message"] == "test message"
        assert data["level"] == "INFO"
        assert data["category"] == "execution"

    def test_to_dict_none_metadata_is_default_dict(self):
        entry = self._make_entry()
        d = entry.to_dict()
        assert d["metadata"] == {}
        assert d["source_file"] is None
        assert d["source_line"] is None


# ---------------------------------------------------------------------------
# FileLogStorage (logging_service version)
# ---------------------------------------------------------------------------


class TestFileLogStorage:
    @pytest.fixture
    def storage(self, tmp_path: Path) -> FileLogStorage:
        return FileLogStorage(tmp_path / "logs")

    def _make_entry(self, **overrides) -> StructuredLogEntry:
        defaults = dict(
            timestamp=datetime.now(UTC),
            level=LogLevel.INFO,
            category=LogCategory.EXECUTION,
            message="test message",
            context=LogContext(),
        )
        defaults.update(overrides)
        return StructuredLogEntry(**defaults)

    def test_store_and_query(self, storage):
        entry = self._make_entry(message="stored msg")
        storage.store_log(entry)
        results = storage.query_logs(limit=10)
        assert len(results) == 1
        assert results[0]["message"] == "stored msg"
        assert results[0]["level"] == "INFO"
        assert results[0]["category"] == "execution"

    def test_query_filter_by_level(self, storage):
        storage.store_log(self._make_entry(level=LogLevel.INFO, message="info msg"))
        storage.store_log(self._make_entry(level=LogLevel.ERROR, message="error msg"))
        results = storage.query_logs(level=LogLevel.ERROR, limit=10)
        assert len(results) == 1
        assert results[0]["message"] == "error msg"
        assert results[0]["level"] == "ERROR"

    def test_query_filter_by_category(self, storage):
        storage.store_log(self._make_entry(category=LogCategory.EXECUTION, message="exec msg"))
        storage.store_log(self._make_entry(category=LogCategory.SECURITY, message="security event"))
        results = storage.query_logs(category=LogCategory.SECURITY, limit=10)
        assert len(results) == 1
        assert results[0]["message"] == "security event"
        assert results[0]["category"] == "security"

    def test_query_filter_by_context(self, storage):
        storage.store_log(self._make_entry(context=LogContext(language="python"), message="py msg"))
        storage.store_log(self._make_entry(context=LogContext(language="go"), message="go msg"))
        results = storage.query_logs(context_filters={"language": "python"}, limit=10)
        assert len(results) == 1
        assert results[0]["message"] == "py msg"

    def test_query_limit(self, storage):
        for i in range(5):
            storage.store_log(self._make_entry(message=f"msg {i}"))
        results = storage.query_logs(limit=2)
        assert len(results) == 2

    def test_query_time_range(self, storage):
        now = datetime.now(UTC)
        storage.store_log(self._make_entry(timestamp=now, message="in range"))
        results = storage.query_logs(
            start_time=now - timedelta(minutes=1),
            end_time=now + timedelta(minutes=1),
            limit=10,
        )
        assert len(results) == 1
        assert results[0]["message"] == "in range"

    def test_creates_directory(self, tmp_path: Path):
        log_dir = tmp_path / "nested" / "logs"
        storage = FileLogStorage(log_dir)
        assert log_dir.exists()
        assert log_dir.is_dir()
        # Store and query to verify it works
        storage.store_log(self._make_entry(message="dir test"))
        results = storage.query_logs(limit=10)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# LogAggregator
# ---------------------------------------------------------------------------


class TestLogAggregator:
    def test_start_and_stop(self, tmp_path: Path):
        storage = FileLogStorage(tmp_path / "logs")
        aggregator = LogAggregator(storage)
        aggregator.start()
        assert aggregator.processing_thread is not None
        assert aggregator.processing_thread.is_alive()
        aggregator.stop()
        assert aggregator.shutdown_event.is_set()
        assert not aggregator.processing_thread.is_alive()

    def test_add_log_enqueues_entry(self, tmp_path: Path):
        """add_log puts entry on the queue and aggregator processes it to storage."""
        storage = FileLogStorage(tmp_path / "logs")
        aggregator = LogAggregator(storage)
        # Don't start the processing thread, so we can check the queue directly
        entry = StructuredLogEntry(
            timestamp=datetime.now(UTC),
            level=LogLevel.INFO,
            category=LogCategory.EXECUTION,
            message="test",
            context=LogContext(),
        )
        aggregator.add_log(entry)
        assert aggregator.log_queue.qsize() == 1
        # Verify the item in the queue is the correct entry
        queued = aggregator.log_queue.get_nowait()
        assert queued.message == "test"
        assert queued.level == LogLevel.INFO

    def test_aggregator_processes_to_storage(self, tmp_path: Path):
        """Started aggregator actually writes entries to storage."""
        storage = FileLogStorage(tmp_path / "logs")
        aggregator = LogAggregator(storage)
        aggregator.start()
        entry = StructuredLogEntry(
            timestamp=datetime.now(UTC),
            level=LogLevel.INFO,
            category=LogCategory.EXECUTION,
            message="processed entry",
            context=LogContext(),
        )
        aggregator.add_log(entry)
        # Give the processing thread time to consume
        time.sleep(0.2)
        aggregator.stop()
        results = storage.query_logs(limit=10)
        assert len(results) == 1
        assert results[0]["message"] == "processed entry"


# ---------------------------------------------------------------------------
# TestExecutionLogger
# ---------------------------------------------------------------------------


class TestExecLogger:
    """Tests for TestExecutionLogger. Named TestExecLogger to avoid pytest collection warning."""

    @pytest.fixture
    def exec_logger(self, tmp_path: Path) -> TestExecutionLogger:
        storage = FileLogStorage(tmp_path / "logs")
        return TestExecutionLogger(storage)

    def _wait_and_query(self, exec_logger, count=1):
        """Wait for the aggregator to process and return stored logs."""
        time.sleep(0.2)
        return exec_logger.storage.query_logs(limit=count + 5)

    def test_info_stores_to_storage(self, exec_logger):
        """info() routes message through aggregator to storage."""
        exec_logger.info(LogCategory.EXECUTION, "test message")
        results = self._wait_and_query(exec_logger)
        assert len(results) == 1
        assert results[0]["message"] == "test message"
        assert results[0]["category"] == "execution"
        assert results[0]["level"] == "INFO"

    def test_debug_stores_to_storage(self, exec_logger):
        """debug() routes message with DEBUG level."""
        exec_logger.debug(LogCategory.CONTAINER, "debug msg")
        results = self._wait_and_query(exec_logger)
        assert len(results) == 1
        assert results[0]["level"] == "DEBUG"
        assert results[0]["category"] == "container"

    def test_warning_stores_to_storage(self, exec_logger):
        """warning() routes message with WARNING level."""
        exec_logger.warning(LogCategory.SECURITY, "warning msg")
        results = self._wait_and_query(exec_logger)
        assert len(results) == 1
        assert results[0]["level"] == "WARNING"
        assert results[0]["category"] == "security"

    def test_error_stores_to_storage(self, exec_logger):
        """error() routes message with ERROR level."""
        exec_logger.error(LogCategory.RESOURCE, "error msg")
        results = self._wait_and_query(exec_logger)
        assert len(results) == 1
        assert results[0]["level"] == "ERROR"
        assert results[0]["category"] == "resource"

    def test_critical_stores_to_storage(self, exec_logger):
        """critical() routes message with CRITICAL level."""
        exec_logger.critical(LogCategory.SYSTEM, "critical msg")
        results = self._wait_and_query(exec_logger)
        assert len(results) == 1
        assert results[0]["level"] == "CRITICAL"
        assert results[0]["category"] == "system"

    def test_log_container_event(self, exec_logger):
        """log_container_event() stores with CONTAINER category."""
        exec_logger.log_container_event(LogLevel.INFO, "container started")
        results = self._wait_and_query(exec_logger)
        assert len(results) == 1
        assert results[0]["category"] == "container"
        assert results[0]["message"] == "container started"

    def test_log_security_event(self, exec_logger):
        """log_security_event() stores with SECURITY category."""
        exec_logger.log_security_event(LogLevel.WARNING, "violation detected")
        results = self._wait_and_query(exec_logger)
        assert len(results) == 1
        assert results[0]["category"] == "security"
        assert results[0]["message"] == "violation detected"

    def test_log_performance_event(self, exec_logger):
        """log_performance_event() stores with PERFORMANCE category."""
        exec_logger.log_performance_event(LogLevel.INFO, "perf recorded")
        results = self._wait_and_query(exec_logger)
        assert len(results) == 1
        assert results[0]["category"] == "performance"
        assert results[0]["message"] == "perf recorded"

    def test_log_execution_event(self, exec_logger):
        """log_execution_event() stores with EXECUTION category."""
        exec_logger.log_execution_event(LogLevel.INFO, "execution started")
        results = self._wait_and_query(exec_logger)
        assert len(results) == 1
        assert results[0]["category"] == "execution"

    def test_log_resource_event(self, exec_logger):
        """log_resource_event() stores with RESOURCE category."""
        exec_logger.log_resource_event(LogLevel.INFO, "resource usage")
        results = self._wait_and_query(exec_logger)
        assert len(results) == 1
        assert results[0]["category"] == "resource"

    def test_log_audit_event(self, exec_logger):
        """log_audit_event() stores with AUDIT category."""
        exec_logger.log_audit_event(LogLevel.INFO, "audit event")
        results = self._wait_and_query(exec_logger)
        assert len(results) == 1
        assert results[0]["category"] == "audit"

    def test_log_context_manager(self, exec_logger):
        with exec_logger.log_context(user_id="alice", container_id="c1") as ctx:
            assert ctx.user_id == "alice"
            assert ctx.container_id == "c1"
            assert ctx.language is None

    def test_nested_log_context(self, exec_logger):
        with exec_logger.log_context(user_id="alice"):
            with exec_logger.log_context(container_id="c1") as inner:
                assert inner.user_id == "alice"
                assert inner.container_id == "c1"

    def test_log_context_pops_on_exit(self, exec_logger):
        """Context is popped when exiting the context manager."""
        with exec_logger.log_context(user_id="alice"):
            pass
        # After exiting, current context should be empty
        ctx = exec_logger._get_current_context()
        assert ctx.user_id is None

    def test_query_logs_returns_empty_list(self, exec_logger):
        """query_logs returns an empty list when no logs are stored."""
        result = exec_logger.query_logs(limit=10)
        assert result == []


# ---------------------------------------------------------------------------
# log_aggregation.py — LogQuery and LogQueryResult
# ---------------------------------------------------------------------------

from services.test_execution.log_aggregation import (  # noqa: E402
    CompressionType,
    DatabaseLogStorage,
    LogAggregationService,
    LogQuery,
    LogQueryResult,
    StorageBackend,
)


class TestLogQuery:
    def test_to_dict_defaults(self):
        q = LogQuery()
        d = q.to_dict()
        assert d["limit"] == 100
        assert d["offset"] == 0
        assert d["sort_order"] == "desc"
        assert d["levels"] == []
        assert d["categories"] == []
        assert d["languages"] == []
        assert d["container_ids"] == []
        assert d["search_text"] is None
        assert d["start_time"] is None
        assert d["end_time"] is None

    def test_to_dict_populated(self):
        q = LogQuery(
            levels=[LogLevel.ERROR],
            categories=[LogCategory.SECURITY],
            languages=["python"],
            container_ids=["c1"],
            search_text="error",
            limit=50,
        )
        d = q.to_dict()
        assert d["levels"] == ["ERROR"]
        assert d["categories"] == ["security"]
        assert d["languages"] == ["python"]
        assert d["container_ids"] == ["c1"]
        assert d["search_text"] == "error"
        assert d["limit"] == 50


class TestLogQueryResult:
    def test_to_dict(self):
        result = LogQueryResult(
            logs=[{"msg": "test"}],
            total_count=1,
            query_time_ms=5.0,
            has_more=False,
        )
        d = result.to_dict()
        assert d["total_count"] == 1
        assert d["has_more"] is False
        assert d["query_time_ms"] == 5.0
        assert d["logs"] == [{"msg": "test"}]


# ---------------------------------------------------------------------------
# DatabaseLogStorage
# ---------------------------------------------------------------------------


class TestDatabaseLogStorage:
    @pytest.fixture
    def db_storage(self, tmp_path: Path) -> DatabaseLogStorage:
        return DatabaseLogStorage(tmp_path / "test.db")

    @pytest.mark.asyncio
    async def test_store_and_query(self, db_storage):
        entry = StructuredLogEntry(
            timestamp=datetime.now(UTC),
            level=LogLevel.INFO,
            category=LogCategory.EXECUTION,
            message="test message",
            context=LogContext(language="python"),
        )
        await db_storage.store_log(entry)
        result = await db_storage.query_logs(LogQuery(limit=10))
        assert result.total_count == 1
        assert len(result.logs) == 1
        assert result.logs[0]["message"] == "test message"
        assert result.logs[0]["level"] == "INFO"
        assert result.logs[0]["category"] == "execution"
        assert result.logs[0]["language"] == "python"

    @pytest.mark.asyncio
    async def test_query_filter_by_level(self, db_storage):
        entry_info = StructuredLogEntry(
            timestamp=datetime.now(UTC),
            level=LogLevel.INFO,
            category=LogCategory.EXECUTION,
            message="info msg",
            context=LogContext(),
        )
        entry_error = StructuredLogEntry(
            timestamp=datetime.now(UTC),
            level=LogLevel.ERROR,
            category=LogCategory.EXECUTION,
            message="error msg",
            context=LogContext(),
        )
        await db_storage.store_logs_batch([entry_info, entry_error])
        result = await db_storage.query_logs(LogQuery(levels=[LogLevel.ERROR]))
        assert result.total_count == 1
        assert result.logs[0]["message"] == "error msg"
        assert result.logs[0]["level"] == "ERROR"

    @pytest.mark.asyncio
    async def test_query_search_text(self, db_storage):
        entry = StructuredLogEntry(
            timestamp=datetime.now(UTC),
            level=LogLevel.INFO,
            category=LogCategory.EXECUTION,
            message="container crashed badly",
            context=LogContext(),
        )
        await db_storage.store_log(entry)
        result = await db_storage.query_logs(LogQuery(search_text="crashed"))
        assert result.total_count == 1
        assert result.logs[0]["message"] == "container crashed badly"

    @pytest.mark.asyncio
    async def test_query_pagination(self, db_storage):
        for i in range(5):
            entry = StructuredLogEntry(
                timestamp=datetime.now(UTC),
                level=LogLevel.INFO,
                category=LogCategory.EXECUTION,
                message=f"msg {i}",
                context=LogContext(),
            )
            await db_storage.store_log(entry)
        result = await db_storage.query_logs(LogQuery(limit=2, offset=0))
        assert len(result.logs) == 2
        assert result.has_more is True
        assert result.total_count == 5

    @pytest.mark.asyncio
    async def test_delete_old_logs(self, db_storage):
        old_entry = StructuredLogEntry(
            timestamp=datetime.now(UTC) - timedelta(days=60),
            level=LogLevel.INFO,
            category=LogCategory.EXECUTION,
            message="old msg",
            context=LogContext(),
        )
        await db_storage.store_log(old_entry)
        deleted = await db_storage.delete_old_logs(retention_days=30)
        assert deleted == 1
        # Verify it's actually gone
        result = await db_storage.query_logs(LogQuery(limit=10))
        assert result.total_count == 0

    @pytest.mark.asyncio
    async def test_get_storage_stats(self, db_storage):
        entry = StructuredLogEntry(
            timestamp=datetime.now(UTC),
            level=LogLevel.INFO,
            category=LogCategory.EXECUTION,
            message="msg",
            context=LogContext(),
        )
        await db_storage.store_log(entry)
        stats = await db_storage.get_storage_stats()
        assert stats["total_logs"] == 1
        assert stats["database_size_bytes"] > 0
        assert stats["database_size_mb"] == stats["database_size_bytes"] / (1024 * 1024)
        # With only one log, oldest and newest should be equal
        assert stats["oldest_log"] == stats["newest_log"]
        assert isinstance(stats["oldest_log"], str)
        assert len(stats["oldest_log"]) > 0

    @pytest.mark.asyncio
    async def test_query_by_language(self, db_storage):
        entry = StructuredLogEntry(
            timestamp=datetime.now(UTC),
            level=LogLevel.INFO,
            category=LogCategory.EXECUTION,
            message="go msg",
            context=LogContext(language="go"),
        )
        await db_storage.store_log(entry)
        result = await db_storage.query_logs(LogQuery(languages=["go"]))
        assert result.total_count == 1
        assert result.logs[0]["language"] == "go"

    @pytest.mark.asyncio
    async def test_query_by_container(self, db_storage):
        entry = StructuredLogEntry(
            timestamp=datetime.now(UTC),
            level=LogLevel.INFO,
            category=LogCategory.EXECUTION,
            message="container msg",
            context=LogContext(container_id="c1"),
        )
        await db_storage.store_log(entry)
        result = await db_storage.query_logs(LogQuery(container_ids=["c1"]))
        assert result.total_count == 1
        assert result.logs[0]["container_id"] == "c1"

    @pytest.mark.asyncio
    async def test_query_sort_order(self, db_storage):
        for i in range(3):
            entry = StructuredLogEntry(
                timestamp=datetime.now(UTC) + timedelta(seconds=i),
                level=LogLevel.INFO,
                category=LogCategory.EXECUTION,
                message=f"msg {i}",
                context=LogContext(),
            )
            await db_storage.store_log(entry)
        result_desc = await db_storage.query_logs(LogQuery(sort_order="desc"))
        result_asc = await db_storage.query_logs(LogQuery(sort_order="asc"))
        assert result_desc.logs[0]["message"] == "msg 2"
        assert result_desc.logs[2]["message"] == "msg 0"
        assert result_asc.logs[0]["message"] == "msg 0"
        assert result_asc.logs[2]["message"] == "msg 2"


# ---------------------------------------------------------------------------
# LogAggregationService
# ---------------------------------------------------------------------------


class TestLogAggregationService:
    @pytest.fixture
    def db_storage(self, tmp_path: Path) -> DatabaseLogStorage:
        return DatabaseLogStorage(tmp_path / "agg.db")

    @pytest.mark.asyncio
    async def test_query_logs(self, db_storage):
        service = LogAggregationService(db_storage, batch_size=10, flush_interval=1)
        result = await service.query_logs(LogQuery(limit=10))
        assert result.total_count == 0
        assert result.logs == []
        assert result.has_more is False

    @pytest.mark.asyncio
    async def test_get_storage_stats(self, db_storage):
        service = LogAggregationService(db_storage)
        stats = await service.get_storage_stats()
        assert stats["total_logs"] == 0
        assert stats["database_size_bytes"] > 0


# ---------------------------------------------------------------------------
# log_aggregation.py — LogAggregationServiceHolder / helpers
# ---------------------------------------------------------------------------

from services.test_execution.log_aggregation import (  # noqa: E402
    LogAggregationServiceHolder,
    LogStorageInterface,
    get_aggregation_service,
    reset_aggregation_service,
    setup_aggregation_service,
)


class TestStorageBackendEnum:
    def test_values(self):
        assert StorageBackend.FILE == "file"
        assert StorageBackend.DATABASE == "database"
        assert StorageBackend.ELASTICSEARCH == "elasticsearch"
        assert StorageBackend.LOKI == "loki"


class TestCompressionTypeEnum:
    def test_values(self):
        assert CompressionType.NONE == "none"
        assert CompressionType.GZIP == "gzip"
        assert CompressionType.BROTLI == "brotli"


class TestLogStorageInterface:
    """Ensure abstract base raises NotImplementedError."""

    @pytest.mark.asyncio
    async def test_store_log_not_implemented(self):
        iface = LogStorageInterface()
        with pytest.raises(NotImplementedError):
            await iface.store_log(None)

    @pytest.mark.asyncio
    async def test_store_logs_batch_not_implemented(self):
        iface = LogStorageInterface()
        with pytest.raises(NotImplementedError):
            await iface.store_logs_batch([])

    @pytest.mark.asyncio
    async def test_query_logs_not_implemented(self):
        iface = LogStorageInterface()
        with pytest.raises(NotImplementedError):
            await iface.query_logs(LogQuery())

    @pytest.mark.asyncio
    async def test_delete_old_logs_not_implemented(self):
        iface = LogStorageInterface()
        with pytest.raises(NotImplementedError):
            await iface.delete_old_logs(30)

    @pytest.mark.asyncio
    async def test_get_storage_stats_not_implemented(self):
        iface = LogStorageInterface()
        with pytest.raises(NotImplementedError):
            await iface.get_storage_stats()


class TestLogQueryAdvanced:
    def test_to_dict_with_times(self):
        now = datetime.now(UTC)
        q = LogQuery(start_time=now, end_time=now)
        d = q.to_dict()
        assert d["start_time"] == now.isoformat()
        assert d["end_time"] == now.isoformat()


class TestDatabaseLogStorageAdvanced:
    @pytest.fixture
    def db(self, tmp_path):
        return DatabaseLogStorage(tmp_path / "adv.db")

    @pytest.mark.asyncio
    async def test_query_by_time_range(self, db):
        now = datetime.now(UTC)
        entry = StructuredLogEntry(
            timestamp=now,
            level=LogLevel.INFO,
            category=LogCategory.EXECUTION,
            message="within range",
            context=LogContext(),
        )
        await db.store_log(entry)
        result = await db.query_logs(
            LogQuery(
                start_time=now - timedelta(minutes=1),
                end_time=now + timedelta(minutes=1),
            )
        )
        assert result.total_count == 1
        assert result.logs[0]["message"] == "within range"

    @pytest.mark.asyncio
    async def test_query_by_category(self, db):
        entry = StructuredLogEntry(
            timestamp=datetime.now(UTC),
            level=LogLevel.INFO,
            category=LogCategory.SECURITY,
            message="security event",
            context=LogContext(),
        )
        await db.store_log(entry)
        result = await db.query_logs(LogQuery(categories=[LogCategory.SECURITY]))
        assert result.total_count == 1
        assert result.logs[0]["message"] == "security event"
        assert result.logs[0]["category"] == "security"

    @pytest.mark.asyncio
    async def test_query_no_metadata(self, db):
        """Entry with metadata=None should be handled and stored."""
        entry = StructuredLogEntry(
            timestamp=datetime.now(UTC),
            level=LogLevel.INFO,
            category=LogCategory.EXECUTION,
            message="no meta",
            context=LogContext(),
            metadata=None,
        )
        await db.store_log(entry)
        result = await db.query_logs(LogQuery(limit=10))
        assert result.total_count == 1
        assert result.logs[0]["message"] == "no meta"

    @pytest.mark.asyncio
    async def test_double_initialize_is_idempotent(self, db):
        await db._initialize()
        await db._initialize()
        # Should not raise, and db should be functional
        result = await db.query_logs(LogQuery(limit=10))
        assert result.total_count == 0
        assert result.logs == []


class TestLogAggregationServiceAdvanced:
    @pytest.fixture
    def db(self, tmp_path):
        return DatabaseLogStorage(tmp_path / "svc.db")

    @pytest.mark.asyncio
    async def test_start_and_stop(self, db):
        service = LogAggregationService(db, batch_size=10, flush_interval=1)
        await service.start()
        assert isinstance(service.processing_task, asyncio.Task)
        assert isinstance(service.cleanup_task, asyncio.Task)
        assert not service.processing_task.done()
        assert not service.cleanup_task.done()
        await service.stop()
        assert service.shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_add_log(self, db):
        service = LogAggregationService(db, batch_size=10, flush_interval=1)
        entry = StructuredLogEntry(
            timestamp=datetime.now(UTC),
            level=LogLevel.INFO,
            category=LogCategory.EXECUTION,
            message="queued",
            context=LogContext(),
        )
        await service.add_log(entry)
        assert service.log_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_stop_flushes_remaining(self, db):
        service = LogAggregationService(db, batch_size=100, flush_interval=999)
        entry = StructuredLogEntry(
            timestamp=datetime.now(UTC),
            level=LogLevel.INFO,
            category=LogCategory.EXECUTION,
            message="remaining",
            context=LogContext(),
        )
        service.log_batch.append(entry)
        await service.stop()
        # After stop, the remaining batch should have been flushed
        result = await db.query_logs(LogQuery(limit=10))
        assert result.total_count == 1
        assert result.logs[0]["message"] == "remaining"


class TestLogAggregationServiceHolder:
    @pytest.mark.asyncio
    async def test_reset_instance(self):
        await LogAggregationServiceHolder.reset_instance()
        assert LogAggregationServiceHolder._instance is None
        assert LogAggregationServiceHolder._initialized is False

    @pytest.mark.asyncio
    async def test_set_instance(self, tmp_path):
        await LogAggregationServiceHolder.reset_instance()
        db = DatabaseLogStorage(tmp_path / "holder.db")
        instance = await LogAggregationServiceHolder.set_instance(db)
        assert isinstance(instance, LogAggregationService)
        assert instance.storage is db
        assert LogAggregationServiceHolder._initialized is True
        await LogAggregationServiceHolder.reset_instance()

    @pytest.mark.asyncio
    async def test_get_instance_creates_singleton(self, tmp_path):
        await LogAggregationServiceHolder.reset_instance()
        import os

        os.environ["LOG_STORAGE_DIR"] = str(tmp_path / "singleton_logs")
        try:
            instance = await LogAggregationServiceHolder.get_instance()
            assert isinstance(instance, LogAggregationService)
            assert LogAggregationServiceHolder._initialized is True
            # Getting again returns the same instance
            instance2 = await LogAggregationServiceHolder.get_instance()
            assert instance is instance2
        finally:
            await LogAggregationServiceHolder.reset_instance()
            os.environ.pop("LOG_STORAGE_DIR", None)

    @pytest.mark.asyncio
    async def test_helper_functions(self, tmp_path):
        await reset_aggregation_service()
        db = DatabaseLogStorage(tmp_path / "helpers.db")
        svc = await setup_aggregation_service(db)
        assert isinstance(svc, LogAggregationService)
        svc2 = await get_aggregation_service()
        # Should be the same instance
        assert svc is svc2
        await reset_aggregation_service()
        assert LogAggregationServiceHolder._instance is None
