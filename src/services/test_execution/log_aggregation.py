"""
Log aggregation and storage system for test execution infrastructure.

This module provides centralized log aggregation, storage, and querying capabilities
for distributed test execution across multiple containers and languages.
"""

import asyncio
import gzip
import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

# Try to import aiofiles, but make it optional
try:
    import aiofiles
    import aiofiles.os

    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False
    aiofiles: Any = None

from models.logging_models import LogLevel
from services.test_execution.logging_service import (
    LogCategory,
    StructuredLogEntry,
)

logger = logging.getLogger(__name__)


class StorageBackend(StrEnum):
    """Available storage backends for log aggregation."""

    FILE = "file"
    DATABASE = "database"
    ELASTICSEARCH = "elasticsearch"
    LOKI = "loki"


class CompressionType(StrEnum):
    """Log compression types."""

    NONE = "none"
    GZIP = "gzip"
    BROTLI = "brotli"


@dataclass
class LogQuery:
    """Query parameters for log search."""

    start_time: datetime | None = None
    end_time: datetime | None = None
    levels: list[LogLevel] = field(default_factory=list)
    categories: list[LogCategory] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    container_ids: list[str] = field(default_factory=list)
    search_text: str | None = None
    limit: int = 100
    offset: int = 0
    sort_order: str = "desc"  # desc or asc

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API serialization."""
        return {
            "start_time": (self.start_time.isoformat() if self.start_time else None),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "levels": [level.value for level in self.levels],
            "categories": [cat.value for cat in self.categories],
            "languages": self.languages,
            "container_ids": self.container_ids,
            "search_text": self.search_text,
            "limit": self.limit,
            "offset": self.offset,
            "sort_order": self.sort_order,
        }


@dataclass
class LogQueryResult:
    """Result of log query operation."""

    logs: list[dict[str, Any]]
    total_count: int
    query_time_ms: float
    has_more: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API serialization."""
        return {
            "logs": self.logs,
            "total_count": self.total_count,
            "query_time_ms": self.query_time_ms,
            "has_more": self.has_more,
        }


class LogStorageInterface:
    """Abstract interface for log storage backends."""

    async def store_log(self, log_entry: StructuredLogEntry) -> None:
        """Store a single log entry."""
        raise NotImplementedError

    async def store_logs_batch(self, log_entries: list[StructuredLogEntry]) -> None:
        """Store multiple log entries in a batch."""
        raise NotImplementedError

    async def query_logs(self, query: LogQuery) -> LogQueryResult:
        """Query logs with specified criteria."""
        raise NotImplementedError

    async def delete_old_logs(self, retention_days: int) -> int:
        """Delete logs older than retention period."""
        raise NotImplementedError

    async def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        raise NotImplementedError


class FileLogStorage(LogStorageInterface):
    """File-based log storage with rotation and compression."""

    def __init__(
        self,
        storage_directory: Path,
        compression: CompressionType = CompressionType.GZIP,
        max_file_size_mb: int = 100,
        rotation_interval_hours: int = 24,
    ):
        """Initialize the FileLogStorage service.

        This initializes the FileLogStorage with required dependencies
        and configuration for service operations.

        Args:
            storage_directory (Path): Storage directory
            compression (CompressionType): Compression
            max_file_size_mb (int): Max file size mb
            rotation_interval_hours (int): Rotation interval hours
        """
        self.storage_directory = Path(storage_directory)
        self.compression = compression
        self.max_file_size_mb = max_file_size_mb
        self.rotation_interval_hours = rotation_interval_hours
        self.current_file_handle: Any | None = None
        self.current_file_path: Path | None = None
        self.lock = asyncio.Lock()

        # Create storage directory
        self.storage_directory.mkdir(parents=True, exist_ok=True)

    async def store_log(self, log_entry: StructuredLogEntry) -> None:
        """Store a single log entry."""
        await self.store_logs_batch([log_entry])

    async def store_logs_batch(self, log_entries: list[StructuredLogEntry]) -> None:
        """Store multiple log entries in a batch."""
        async with self.lock:
            file_handle = await self._get_current_file_handle()

            for entry in log_entries:
                log_line = entry.to_json() + "\n"

                if self.compression == CompressionType.GZIP:
                    log_data = gzip.compress(log_line.encode("utf-8"))
                    await file_handle.write(log_data)
                else:
                    await file_handle.write(log_line.encode("utf-8"))

            await file_handle.flush()

    async def _get_current_file_handle(self):
        """Get or create current log file handle."""
        now = datetime.now(UTC)
        expected_file_path = self._get_log_file_path(now)

        # Check if we need to rotate
        if (
            self.current_file_path != expected_file_path
            or self.current_file_handle is None
            or await self._should_rotate_file()
        ):
            # Close current file
            if self.current_file_handle:
                await self.current_file_handle.close()

            # Open new file
            self.current_file_path = expected_file_path
            if self.compression == CompressionType.GZIP:
                self.current_file_handle = await aiofiles.open(  # type: ignore[union-attr]
                    str(self.current_file_path), mode="ab"
                )
            else:
                self.current_file_handle = await aiofiles.open(  # type: ignore[union-attr]
                    str(self.current_file_path), mode="a"
                )

        return self.current_file_handle

    def _get_log_file_path(self, timestamp: datetime) -> Path:
        """Generate log file path based on timestamp."""
        date_str = timestamp.strftime("%Y-%m-%d_%H")
        extension = ".jsonl.gz" if self.compression == CompressionType.GZIP else ".jsonl"
        return self.storage_directory / f"test_execution_{date_str}{extension}"

    async def _should_rotate_file(self) -> bool:
        """Check if current file should be rotated."""
        if not self.current_file_path or not await aiofiles.os.path.exists(self.current_file_path):  # type: ignore[possibly-missing-attribute]
            return False

        # Check file size
        stat = await aiofiles.os.stat(self.current_file_path)  # type: ignore[union-attr]
        size_mb = stat.st_size / (1024 * 1024)

        return size_mb >= self.max_file_size_mb

    async def query_logs(self, query: LogQuery) -> LogQueryResult:
        """Query logs from files."""
        start_time = datetime.now(UTC)
        matching_logs = []
        total_count = 0

        # Get relevant log files
        log_files = await self._get_log_files_for_query(query)

        for log_file in log_files:
            try:
                if log_file.suffix == ".gz":
                    # Read compressed file
                    with gzip.open(log_file, "rt", encoding="utf-8") as f:
                        file_logs = await self._process_file_logs(f, query)
                else:
                    # Read uncompressed file
                    async with aiofiles.open(log_file, "r", encoding="utf-8") as f:  # type: ignore[possibly-missing-attribute]
                        file_logs = await self._process_file_logs(f, query)

                matching_logs.extend(file_logs)
                total_count += len(file_logs)

                # Stop if we have enough results
                if len(matching_logs) >= query.limit + query.offset:
                    break

            except Exception as e:
                logger.error(f"Error reading log file {log_file}: {e}")

        # Sort and paginate
        matching_logs.sort(
            key=lambda x: x.get("timestamp", ""),
            reverse=(query.sort_order == "desc"),
        )

        paginated_logs = matching_logs[query.offset : query.offset + query.limit]
        has_more = len(matching_logs) > query.offset + query.limit

        query_time_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000

        return LogQueryResult(
            logs=paginated_logs,
            total_count=total_count,
            query_time_ms=query_time_ms,
            has_more=has_more,
        )

    async def _get_log_files_for_query(self, query: LogQuery) -> list[Path]:
        """Get log files relevant to the query time range."""
        all_files = []

        for file_path in self.storage_directory.glob("test_execution_*.jsonl*"):
            all_files.append(file_path)

        # Filter by time range if specified
        if query.start_time or query.end_time:
            filtered_files = []
            for file_path in all_files:
                if await self._file_matches_time_range(file_path, query.start_time, query.end_time):
                    filtered_files.append(file_path)
            return sorted(filtered_files)

        return sorted(all_files)

    async def _file_matches_time_range(
        self,
        file_path: Path,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> bool:
        """Check if file might contain logs in the specified time range."""
        try:
            # Extract date from filename
            filename = file_path.name
            # test_execution_YYYY-MM-DD_HH
            date_part = filename.split("_")[2:4]
            if len(date_part) >= 2:
                date_str = f"{date_part[0]}_{date_part[1]}"
                file_time = datetime.strptime(date_str, "%Y-%m-%d_%H")

                if start_time and file_time < start_time - timedelta(hours=1):
                    return False
                if end_time and file_time > end_time + timedelta(hours=1):
                    return False

                return True
        except (ValueError, IndexError):  # fmt: skip
            pass

        return True  # Include file if we can't determine time range

    async def _process_file_logs(self, file_handle, query: LogQuery) -> list[dict[str, Any]]:
        """Process logs from a file handle."""
        matching_logs = []

        async for line in file_handle:
            try:
                log_data = json.loads(line.strip())

                # Apply filters
                if await self._log_matches_query(log_data, query):
                    matching_logs.append(log_data)

            except json.JSONDecodeError:
                continue

        return matching_logs

    async def _log_matches_query(self, log_data: dict[str, Any], query: LogQuery) -> bool:
        """Check if log entry matches query criteria."""
        # Time range filter
        if query.start_time or query.end_time:
            try:
                log_time = datetime.fromisoformat(log_data["timestamp"])
                if query.start_time and log_time < query.start_time:
                    return False
                if query.end_time and log_time > query.end_time:
                    return False
            except (KeyError, ValueError):  # fmt: skip
                pass

        # Level filter
        if query.levels and log_data.get("level") not in [level.value for level in query.levels]:
            return False

        # Category filter
        if query.categories and log_data.get("category") not in [
            cat.value for cat in query.categories
        ]:
            return False

        # Language filter
        if query.languages:
            context_language = log_data.get("context", {}).get("language")
            if context_language not in query.languages:
                return False

        # Container ID filter
        if query.container_ids:
            context_container = log_data.get("context", {}).get("container_id")
            if context_container not in query.container_ids:
                return False

        # Text search
        if query.search_text:
            searchable_text = (
                f"{log_data.get('message', '')} {json.dumps(log_data.get('metadata', {}))}"
            )
            if query.search_text.lower() not in searchable_text.lower():
                return False

        return True

    async def delete_old_logs(self, retention_days: int) -> int:
        """Delete logs older than retention period."""
        cutoff_time = datetime.now(UTC) - timedelta(days=retention_days)
        deleted_count = 0

        for log_file in self.storage_directory.glob("test_execution_*.jsonl*"):
            try:
                stat = await aiofiles.os.stat(log_file)  # type: ignore[union-attr]
                file_time = datetime.fromtimestamp(stat.st_mtime, tz=UTC)

                if file_time < cutoff_time:
                    await aiofiles.os.remove(log_file)  # type: ignore[union-attr]
                    deleted_count += 1
                    logger.info(f"Deleted old log file: {log_file}")

            except Exception as e:
                logger.error(f"Error deleting log file {log_file}: {e}")

        return deleted_count

    async def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        total_size = 0
        file_count = 0

        for log_file in self.storage_directory.glob("test_execution_*.jsonl*"):
            try:
                stat = await aiofiles.os.stat(log_file)  # type: ignore[union-attr]
                total_size += stat.st_size
                file_count += 1
            except Exception:
                continue

        return {
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "file_count": file_count,
            "storage_directory": str(self.storage_directory),
            "compression": self.compression.value,
        }


class DatabaseLogStorage(LogStorageInterface):
    """SQLite-based log storage for structured queries."""

    def __init__(self, database_path: Path):
        """Initialize the DatabaseLogStorage service.

        This initializes the DatabaseLogStorage with required dependencies
        and configuration for service operations.

        Args:
            database_path (Path): Data payload for database path
        """
        self.database_path = database_path
        self.lock = asyncio.Lock()
        self._initialized = False

    async def _initialize(self) -> None:
        """Initialize database schema."""
        if self._initialized:
            return

        async with self.lock:
            if self._initialized:
                return

            # Create tables
            with sqlite3.connect(self.database_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        level TEXT NOT NULL,
                        category TEXT NOT NULL,
                        message TEXT NOT NULL,
                        language TEXT,
                        container_id TEXT,
                        execution_id TEXT,
                        metadata TEXT,
                        source_file TEXT,
                        source_line INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                # Create indexes for performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON logs(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_level ON logs(level)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON logs(category)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_language ON logs(language)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_container ON logs(container_id)")

                conn.commit()

            self._initialized = True

    async def store_log(self, log_entry: StructuredLogEntry) -> None:
        """Store a single log entry."""
        await self.store_logs_batch([log_entry])

    async def store_logs_batch(self, log_entries: list[StructuredLogEntry]) -> None:
        """Store multiple log entries in a batch."""
        await self._initialize()

        async with self.lock:
            with sqlite3.connect(self.database_path) as conn:
                for entry in log_entries:
                    conn.execute(
                        """
                        INSERT INTO logs (
                            timestamp, level, category, message, language,
                            container_id, execution_id, metadata, source_file, source_line
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            entry.timestamp.isoformat(),
                            entry.level.value,
                            entry.category.value,
                            entry.message,
                            entry.context.language,
                            entry.context.container_id,
                            entry.context.execution_id,
                            json.dumps(entry.metadata),
                            entry.source_file,
                            entry.source_line,
                        ),
                    )

                conn.commit()

    async def query_logs(self, query: LogQuery) -> LogQueryResult:
        """Query logs from database."""
        await self._initialize()

        start_time = datetime.now(UTC)

        # Build SQL query
        where_conditions = []
        params = []

        if query.start_time:
            where_conditions.append("timestamp >= ?")
            params.append(query.start_time.isoformat())

        if query.end_time:
            where_conditions.append("timestamp <= ?")
            params.append(query.end_time.isoformat())

        if query.levels:
            placeholders = ",".join(["?" for _ in query.levels])
            where_conditions.append(f"level IN ({placeholders})")
            params.extend([level.value for level in query.levels])

        if query.categories:
            placeholders = ",".join(["?" for _ in query.categories])
            where_conditions.append(f"category IN ({placeholders})")
            params.extend([cat.value for cat in query.categories])

        if query.languages:
            placeholders = ",".join(["?" for _ in query.languages])
            where_conditions.append(f"language IN ({placeholders})")
            params.extend(query.languages)

        if query.container_ids:
            placeholders = ",".join(["?" for _ in query.container_ids])
            where_conditions.append(f"container_id IN ({placeholders})")
            params.extend(query.container_ids)

        if query.search_text:
            where_conditions.append("message LIKE ?")
            params.append(f"%{query.search_text}%")

        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        order_clause = f"ORDER BY timestamp {'DESC' if query.sort_order == 'desc' else 'ASC'}"

        # Count query
        count_sql = f"SELECT COUNT(*) FROM logs WHERE {where_clause}"

        # Data query
        data_sql = f"""
            SELECT timestamp, level, category, message, language, container_id,
                   execution_id, metadata, source_file, source_line
            FROM logs
            WHERE {where_clause}
            {order_clause}
            LIMIT ? OFFSET ?
        """

        async with self.lock:
            with sqlite3.connect(self.database_path) as conn:
                conn.row_factory = sqlite3.Row

                # Get total count
                total_count = conn.execute(count_sql, params).fetchone()[0]

                # Get data
                data_params = params + [query.limit, query.offset]
                rows = conn.execute(data_sql, data_params).fetchall()

                logs = []
                for row in rows:
                    log_dict = dict(row)
                    if log_dict["metadata"]:
                        log_dict["metadata"] = json.loads(log_dict["metadata"])
                    logs.append(log_dict)

        query_time_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
        has_more = query.offset + len(logs) < total_count

        return LogQueryResult(
            logs=logs,
            total_count=total_count,
            query_time_ms=query_time_ms,
            has_more=has_more,
        )

    async def delete_old_logs(self, retention_days: int) -> int:
        """Delete logs older than retention period."""
        await self._initialize()

        cutoff_time = datetime.now(UTC) - timedelta(days=retention_days)

        async with self.lock:
            with sqlite3.connect(self.database_path) as conn:
                result = conn.execute(
                    "DELETE FROM logs WHERE timestamp < ?",
                    (cutoff_time.isoformat(),),
                )
                deleted_count = result.rowcount
                conn.commit()

        return deleted_count

    async def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        await self._initialize()

        async with self.lock:
            with sqlite3.connect(self.database_path) as conn:
                # Get total count
                total_count = conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]

                # Get oldest and newest timestamps
                oldest = conn.execute("SELECT MIN(timestamp) FROM logs").fetchone()[0]
                newest = conn.execute("SELECT MAX(timestamp) FROM logs").fetchone()[0]

                # Get file size
                file_size = os.path.getsize(self.database_path)

        return {
            "total_logs": total_count,
            "oldest_log": oldest,
            "newest_log": newest,
            "database_size_bytes": file_size,
            "database_size_mb": file_size / (1024 * 1024),
            "database_path": str(self.database_path),
        }


class LogAggregationService:
    """Central log aggregation service."""

    def __init__(
        self,
        storage_backend: LogStorageInterface,
        batch_size: int = 100,
        flush_interval: int = 5,
        retention_days: int = 30,
    ):
        """Initialize the LogAggregationService service.

        This initializes the LogAggregationService with required dependencies
        and configuration for service operations.

        Args:
            storage_backend (LogStorageInterface): Storage backend
            batch_size (int): Batch size
            flush_interval (int): Flush interval
            retention_days (int): Retention days
        """
        self.storage = storage_backend
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.retention_days = retention_days

        self.log_queue: asyncio.Queue[StructuredLogEntry] = asyncio.Queue()
        self.log_batch: list[StructuredLogEntry] = []
        self.processing_task: asyncio.Task | None = None
        self.cleanup_task: asyncio.Task | None = None
        self.shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the log aggregation service."""
        self.processing_task = asyncio.create_task(self._process_logs())
        self.cleanup_task = asyncio.create_task(self._cleanup_old_logs())
        logger.info("Log aggregation service started")

    async def stop(self) -> None:
        """Stop the log aggregation service."""
        self.shutdown_event.set()

        if self.processing_task:
            await self.processing_task

        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        # Flush remaining logs
        if self.log_batch:
            await self.storage.store_logs_batch(list(self.log_batch))
            self.log_batch.clear()

        logger.info("Log aggregation service stopped")

    async def add_log(self, log_entry: StructuredLogEntry) -> None:
        """Add a log entry to the aggregation queue."""
        await self.log_queue.put(log_entry)

    async def _process_logs(self) -> None:
        """Process logs from the queue."""
        last_flush_time = datetime.now(UTC)

        while not self.shutdown_event.is_set():
            try:
                # Get log entry with timeout
                log_entry = await asyncio.wait_for(self.log_queue.get(), timeout=1.0)

                self.log_batch.append(log_entry)

                # Check if we should flush
                should_flush = (
                    len(self.log_batch) >= self.batch_size
                    or (datetime.now(UTC) - last_flush_time).total_seconds() >= self.flush_interval
                )

                if should_flush:
                    await self.storage.store_logs_batch(list(self.log_batch))
                    self.log_batch.clear()
                    last_flush_time = datetime.now(UTC)

            except TimeoutError:
                # Check if we should flush due to time
                if (datetime.now(UTC) - last_flush_time).total_seconds() >= self.flush_interval:
                    if self.log_batch:
                        await self.storage.store_logs_batch(list(self.log_batch))
                        self.log_batch.clear()
                        last_flush_time = datetime.now(UTC)

            except Exception as e:
                logger.error(f"Error processing log batch: {e}")

    async def _cleanup_old_logs(self) -> None:
        """Periodic cleanup of old logs."""
        while not self.shutdown_event.is_set():
            try:
                deleted_count = await self.storage.delete_old_logs(self.retention_days)
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old log entries")

                # Sleep for 24 hours
                await asyncio.sleep(24 * 60 * 60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error during log cleanup: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying

    async def query_logs(self, query: LogQuery) -> LogQueryResult:
        """Query logs using the storage backend."""
        return await self.storage.query_logs(query)

    async def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        return await self.storage.get_storage_stats()


class LogAggregationServiceHolder:
    """Singleton holder for log aggregation service with class-level instance management."""

    _instance: LogAggregationService | None = None
    _lock = __import__("threading").Lock()
    _initialized = False

    @classmethod
    async def get_instance(cls) -> LogAggregationService:
        """Get or create the singleton instance (thread-safe)."""
        if cls._instance is None or not cls._initialized:
            with cls._lock:
                if cls._instance is None:
                    # Default to file storage
                    storage_dir = Path(os.getenv("LOG_STORAGE_DIR", "/tmp/test_execution_logs"))
                    storage = FileLogStorage(storage_dir)
                    cls._instance = LogAggregationService(storage)
                    await cls._instance.start()
                    cls._initialized = True
                    logger.info("Log aggregation service singleton created")
        assert cls._instance is not None
        return cls._instance

    @classmethod
    async def set_instance(
        cls, storage_backend: LogStorageInterface, **kwargs: Any
    ) -> LogAggregationService:
        """Set a custom instance with specific storage backend."""
        with cls._lock:
            if cls._instance and cls._initialized:
                await cls._instance.stop()

            cls._instance = LogAggregationService(storage_backend, **kwargs)
            await cls._instance.start()
            cls._initialized = True
            logger.info("Log aggregation service singleton replaced")
            return cls._instance

    @classmethod
    async def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            if cls._instance and cls._initialized:
                await cls._instance.stop()
            cls._instance = None
            cls._initialized = False
            logger.debug("Log aggregation service singleton reset")


async def get_aggregation_service() -> LogAggregationService:
    """Get the global log aggregation service instance."""
    return await LogAggregationServiceHolder.get_instance()


async def setup_aggregation_service(
    storage_backend: LogStorageInterface, **kwargs: Any
) -> LogAggregationService:
    """Setup log aggregation service with custom storage backend."""
    return await LogAggregationServiceHolder.set_instance(storage_backend, **kwargs)


async def reset_aggregation_service() -> None:
    """Reset the log aggregation service (useful for testing)."""
    await LogAggregationServiceHolder.reset_instance()
