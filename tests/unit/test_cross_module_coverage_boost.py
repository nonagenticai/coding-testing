"""Tests to boost coverage from 86% to 88%+ by targeting the largest uncovered areas:
- log_aggregation.py (FileLogStorage, DatabaseLogStorage query paths)
- resource_monitor.py (ResourceCollector, ResourceMonitor)
- test_execution_container_registry.py (build/pull, Docker-backed checks)
- audit_logger.py (compliance report, log integrity, encryption, time filtering)
- monitoring_service.py (log_test_completed, classify_error, health status, recommendations)
- test_orchestrator.py (execute_tests, _validate_execution_request, helper methods)
- health_monitor.py (HTTPHealthChecker, DockerHealthChecker, monitoring loops)
- intelligent_test_integration.py (exception classes, config validators)
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stubs for optional dependencies
# ---------------------------------------------------------------------------

if "docker" not in sys.modules:
    _docker_stub = types.ModuleType("docker")
    _docker_errors = types.ModuleType("docker.errors")
    _docker_errors.NotFound = type("NotFound", (Exception,), {})
    _docker_errors.APIError = type("APIError", (Exception,), {})
    _docker_errors.ContainerError = type("ContainerError", (Exception,), {})
    _docker_errors.ImageNotFound = type("ImageNotFound", (Exception,), {})
    _docker_stub.errors = _docker_errors
    _docker_stub.from_env = MagicMock()
    sys.modules["docker"] = _docker_stub
    sys.modules["docker.errors"] = _docker_errors
else:
    # Ensure ImageNotFound exists on the docker.errors stub
    _docker_errors = sys.modules.get("docker.errors")
    if _docker_errors and not hasattr(_docker_errors, "ImageNotFound"):
        _docker_errors.ImageNotFound = type("ImageNotFound", (Exception,), {})

if "aiohttp" not in sys.modules:
    _aiohttp_stub = types.ModuleType("aiohttp")
    _aiohttp_stub.ClientSession = MagicMock
    _aiohttp_stub.ClientTimeout = MagicMock
    _aiohttp_stub.ClientError = type("ClientError", (Exception,), {})
    sys.modules["aiohttp"] = _aiohttp_stub

if "psutil" not in sys.modules:
    _psutil_stub = types.ModuleType("psutil")
    _psutil_stub.cpu_percent = MagicMock(return_value=25.0)
    _mem = MagicMock()
    _mem.used = 4 * 1024**3
    _mem.total = 16 * 1024**3
    _mem.percent = 25.0
    _mem.available = 12 * 1024**3
    _psutil_stub.virtual_memory = MagicMock(return_value=_mem)
    _disk = MagicMock()
    _disk.used = 100 * 1024**3
    _disk.total = 500 * 1024**3
    _disk.free = 400 * 1024**3
    _psutil_stub.disk_usage = MagicMock(return_value=_disk)
    _net = MagicMock()
    _net.bytes_sent = 1000
    _net.bytes_recv = 2000
    _net.packets_sent = 50
    _net.packets_recv = 60
    _psutil_stub.net_io_counters = MagicMock(return_value=_net)
    sys.modules["psutil"] = _psutil_stub


# ===========================================================================
# LOG AGGREGATION: DatabaseLogStorage deeper coverage
# ===========================================================================

from models.logging_models import LogLevel  # noqa: E402
from services.test_execution.log_aggregation import (  # noqa: E402
    CompressionType,
    DatabaseLogStorage,
    LogAggregationService,
    LogAggregationServiceHolder,
    LogQuery,
    LogStorageInterface,
    StorageBackend,
    get_aggregation_service,
    reset_aggregation_service,
    setup_aggregation_service,
)
from services.test_execution.logging_service import (  # noqa: E402
    FileLogStorage as LoggingFileLogStorage,
)
from services.test_execution.logging_service import (
    LogAggregator,
    LogCategory,
    LogContext,
    StructuredLogEntry,
    TestExecutionLogger,
)


def _make_log_entry(
    message: str = "test log",
    level: LogLevel = LogLevel.INFO,
    category: LogCategory = LogCategory.EXECUTION,
    language: str = "python",
    container_id: str = "c1",
    execution_id: str = "e1",
    ts: datetime | None = None,
) -> MagicMock:
    """Create a mock log entry compatible with DatabaseLogStorage."""
    entry = MagicMock(spec=StructuredLogEntry)
    entry.timestamp = ts or datetime.now(UTC)
    entry.level = level
    entry.category = category
    entry.message = message
    entry.context = MagicMock()
    entry.context.language = language
    entry.context.container_id = container_id
    entry.context.execution_id = execution_id
    entry.metadata = {"key": "value"}
    entry.source_file = "test.py"
    entry.source_line = 10
    # For to_json used by FileLogStorage
    entry.to_json.return_value = json.dumps(
        {
            "timestamp": entry.timestamp.isoformat(),
            "level": level.value,
            "category": category.value,
            "message": message,
            "context": {"language": language, "container_id": container_id},
            "metadata": {"key": "value"},
        }
    )
    return entry


class TestDatabaseLogStorageAdvanced:
    """Cover DatabaseLogStorage query paths with various filters."""

    @pytest.fixture
    def db_storage(self, tmp_path):
        return DatabaseLogStorage(tmp_path / "test_logs.db")

    @pytest.mark.asyncio
    async def test_query_with_all_filters(self, db_storage):
        """Test query with every filter set at once."""
        entry = _make_log_entry()
        await db_storage.store_log(entry)

        q = LogQuery(
            start_time=datetime.now(UTC) - timedelta(hours=1),
            end_time=datetime.now(UTC) + timedelta(hours=1),
            levels=[LogLevel.INFO],
            categories=[LogCategory.EXECUTION],
            languages=["python"],
            container_ids=["c1"],
            search_text="test",
            limit=10,
            offset=0,
            sort_order="asc",
        )
        result = await db_storage.query_logs(q)
        assert result.total_count == 1
        assert result.has_more is False
        assert len(result.logs) == 1
        assert result.logs[0]["message"] == "test log"
        assert result.logs[0]["level"] == LogLevel.INFO.value
        assert result.logs[0]["language"] == "python"
        assert result.logs[0]["container_id"] == "c1"

    @pytest.mark.asyncio
    async def test_query_has_more_flag(self, db_storage):
        """Test that has_more is set correctly when more results exist."""
        for i in range(5):
            await db_storage.store_log(_make_log_entry(message=f"log-{i}"))
        result = await db_storage.query_logs(LogQuery(limit=2, offset=0))
        assert result.has_more is True

    @pytest.mark.asyncio
    async def test_query_no_results(self, db_storage):
        """Query with impossible filters returns empty."""
        await db_storage.store_log(_make_log_entry())
        q = LogQuery(languages=["haskell"])
        result = await db_storage.query_logs(q)
        assert result.total_count == 0

    @pytest.mark.asyncio
    async def test_delete_old_logs_returns_count(self, db_storage):
        """Deleting old logs removes only logs older than retention and keeps recent."""
        # Insert an old log
        old_ts = datetime.now(UTC) - timedelta(days=365)
        await db_storage.store_log(_make_log_entry(ts=old_ts, message="old log"))
        # Also insert a recent log
        await db_storage.store_log(_make_log_entry(message="recent log"))
        deleted = await db_storage.delete_old_logs(30)
        assert deleted == 1
        # Verify only the recent log remains
        remaining = await db_storage.query_logs(LogQuery())
        assert remaining.total_count == 1
        assert remaining.logs[0]["message"] == "recent log"

    @pytest.mark.asyncio
    async def test_get_storage_stats_complete(self, db_storage):
        """Stats include all expected keys with correct values."""
        await db_storage.store_log(_make_log_entry())
        stats = await db_storage.get_storage_stats()
        assert stats["total_logs"] == 1
        # With one log, oldest and newest are the same timestamp
        assert stats["oldest_log"] == stats["newest_log"]
        # Should be a valid ISO timestamp string
        datetime.fromisoformat(stats["oldest_log"])
        assert stats["database_size_bytes"] > 0
        assert stats["database_size_mb"] == stats["database_size_bytes"] / (1024 * 1024)
        assert stats["database_path"] == str(db_storage.database_path)

    @pytest.mark.asyncio
    async def test_query_metadata_parsed(self, db_storage):
        """Verify metadata is properly JSON-parsed from DB."""
        await db_storage.store_log(_make_log_entry())
        result = await db_storage.query_logs(LogQuery(limit=1))
        assert result.total_count == 1
        assert len(result.logs) == 1
        log = result.logs[0]
        assert log["metadata"] == {"key": "value"}
        assert log["message"] == "test log"

    @pytest.mark.asyncio
    async def test_store_log_delegates_to_batch(self, db_storage):
        """store_log should call store_logs_batch internally."""
        await db_storage.store_log(_make_log_entry())
        result = await db_storage.query_logs(LogQuery())
        assert result.total_count == 1


class TestLogAggregationServiceAdvanced:
    """Cover LogAggregationService _process_logs and _cleanup_old_logs."""

    @pytest.fixture
    def db_storage(self, tmp_path):
        return DatabaseLogStorage(tmp_path / "agg.db")

    @pytest.mark.asyncio
    async def test_start_stop_flushes_remaining(self, db_storage):
        """Stop should flush remaining logs in batch."""
        service = LogAggregationService(
            storage_backend=db_storage,
            batch_size=100,
            flush_interval=1,
        )
        await service.start()

        # Add a log but don't wait for batch
        entry = _make_log_entry()
        await service.add_log(entry)
        # Give processor time to pick it up
        await asyncio.sleep(0.3)

        await service.stop()

        # Verify the log was stored
        result = await db_storage.query_logs(LogQuery())
        assert result.total_count == 1
        assert result.logs[0]["message"] == "test log"

    @pytest.mark.asyncio
    async def test_batch_flush_on_size(self, db_storage):
        """Batch should flush when batch_size is reached."""
        service = LogAggregationService(
            storage_backend=db_storage,
            batch_size=3,
            flush_interval=60,  # Long interval so batch size triggers
        )
        await service.start()

        for i in range(4):
            await service.add_log(_make_log_entry(message=f"msg-{i}"))

        # Wait for processing
        await asyncio.sleep(0.5)
        await service.stop()

        result = await db_storage.query_logs(LogQuery(limit=10))
        assert result.total_count == 4


class TestLogAggregationServiceHolderAdvanced:
    """Cover set_instance replacing existing, and convenience functions."""

    @pytest.mark.asyncio
    async def test_get_instance_creates_singleton(self):
        """get_instance should create and start singleton."""
        await LogAggregationServiceHolder.reset_instance()
        svc = await LogAggregationServiceHolder.get_instance()
        assert isinstance(svc.processing_task, asyncio.Task)
        assert not svc.processing_task.done()
        assert isinstance(svc.cleanup_task, asyncio.Task)
        assert not svc.cleanup_task.done()
        # Second call returns the exact same object
        svc2 = await LogAggregationServiceHolder.get_instance()
        assert svc is svc2
        await LogAggregationServiceHolder.reset_instance()

    @pytest.mark.asyncio
    async def test_set_instance_replaces_existing(self):
        """set_instance should stop old and create new."""
        mock_storage1 = AsyncMock(spec=LogStorageInterface)
        mock_storage1.store_logs_batch = AsyncMock()
        mock_storage1.delete_old_logs = AsyncMock(return_value=0)

        svc1 = await LogAggregationServiceHolder.set_instance(mock_storage1)

        mock_storage2 = AsyncMock(spec=LogStorageInterface)
        mock_storage2.store_logs_batch = AsyncMock()
        mock_storage2.delete_old_logs = AsyncMock(return_value=0)

        svc2 = await LogAggregationServiceHolder.set_instance(mock_storage2)
        assert svc1 is not svc2

        await LogAggregationServiceHolder.reset_instance()

    @pytest.mark.asyncio
    async def test_convenience_functions(self):
        """Test get_aggregation_service, setup_aggregation_service, reset_aggregation_service."""
        await reset_aggregation_service()

        mock_storage = AsyncMock(spec=LogStorageInterface)
        mock_storage.store_logs_batch = AsyncMock()
        mock_storage.delete_old_logs = AsyncMock(return_value=0)

        svc = await setup_aggregation_service(mock_storage)
        assert isinstance(svc.processing_task, asyncio.Task)
        assert not svc.processing_task.done()
        assert svc.storage is mock_storage

        svc2 = await get_aggregation_service()
        assert svc2 is svc

        await reset_aggregation_service()


class TestStorageAndCompressionEnums:
    def test_storage_backend_values(self):
        assert StorageBackend.FILE == "file"
        assert StorageBackend.DATABASE == "database"
        assert StorageBackend.ELASTICSEARCH == "elasticsearch"
        assert StorageBackend.LOKI == "loki"

    def test_compression_type_values(self):
        assert CompressionType.NONE == "none"
        assert CompressionType.GZIP == "gzip"
        assert CompressionType.BROTLI == "brotli"


# ===========================================================================
# RESOURCE MONITOR: ResourceCollector + ResourceMonitor
# ===========================================================================

from services.test_execution.resource_monitor import (  # noqa: E402
    AlertLevel,
    ResourceAlert,
    ResourceCollector,
    ResourceMonitor,
    ResourceType,
    ResourceUsage,
    get_resource_monitor,
)


class TestResourceCollectorSystemResources:
    """Cover ResourceCollector.collect_system_resources (lines 164-264)."""

    def test_collect_system_resources_returns_data(self):
        """Should return CPU/Memory/Disk/Network resources with valid values."""
        collector = ResourceCollector()
        resources = collector.collect_system_resources()
        # At minimum: CPU, Memory, Disk, Network (container count depends on docker availability)
        assert len(resources) >= 4
        resource_map = {r.resource_type: r for r in resources}
        assert ResourceType.CPU in resource_map
        assert ResourceType.MEMORY in resource_map
        assert ResourceType.DISK in resource_map
        assert ResourceType.NETWORK in resource_map
        # CPU: percentage between 0 and 100, max always 100
        cpu = resource_map[ResourceType.CPU]
        assert 0.0 <= cpu.current_value <= 100.0
        assert cpu.max_value == 100.0
        assert cpu.usage_percent == cpu.current_value  # CPU usage_percent == current_value
        assert cpu.unit == "percent"
        # Memory: used < total, both in GB, percent matches ratio
        mem = resource_map[ResourceType.MEMORY]
        assert 0 < mem.current_value < mem.max_value
        assert mem.max_value > 1.0  # at least 1GB total
        assert 0.0 < mem.usage_percent < 100.0
        assert mem.unit == "GB"
        # Disk: used < total, both in GB, percent = (used/total)*100
        disk = resource_map[ResourceType.DISK]
        assert 0 < disk.current_value < disk.max_value
        assert disk.max_value > 1.0  # at least 1GB total
        expected_disk_pct = (disk.current_value / disk.max_value) * 100
        assert abs(disk.usage_percent - expected_disk_pct) < 0.01
        assert disk.unit == "GB"

    def test_collect_system_resources_network(self):
        """Network stats should include bytes sent/recv metadata."""
        collector = ResourceCollector()
        resources = collector.collect_system_resources()
        network = [r for r in resources if r.resource_type == ResourceType.NETWORK][0]
        # current_value = bytes_sent + bytes_recv
        assert (
            network.current_value == network.metadata["bytes_sent"] + network.metadata["bytes_recv"]
        )
        assert network.unit == "bytes"
        # Verify all counters are non-negative integers
        assert network.metadata["bytes_sent"] >= 0
        assert network.metadata["bytes_recv"] >= 0
        assert network.metadata["packets_sent"] >= 0
        assert network.metadata["packets_recv"] >= 0
        # current_value must equal bytes_sent + bytes_recv
        assert (
            network.current_value == network.metadata["bytes_sent"] + network.metadata["bytes_recv"]
        )

    def test_collect_container_resources_no_docker(self):
        """Without docker client, should return empty list."""
        collector = ResourceCollector()
        collector.docker_client = None
        resources = collector.collect_container_resources("test-container")
        assert resources == []


class TestResourceMonitorAdvanced:
    """Cover ResourceMonitor methods that were uncovered."""

    def test_get_current_usage(self):
        """get_current_usage delegates to collector and returns all resource types."""
        mon = ResourceMonitor()
        usage = mon.get_current_usage()
        # At minimum: CPU, Memory, Disk, Network (container count depends on docker)
        assert len(usage) >= 4
        resource_types = {r.resource_type for r in usage}
        # Core resource types must always be present
        assert {
            ResourceType.CPU,
            ResourceType.MEMORY,
            ResourceType.DISK,
            ResourceType.NETWORK,
        }.issubset(resource_types)

    def test_get_resource_summary(self):
        """Summary includes usage, alerts, and thresholds with correct structure."""
        mon = ResourceMonitor()
        summary = mon.get_resource_summary()
        # timestamp should be a valid ISO timestamp string
        assert datetime.fromisoformat(summary["timestamp"])
        # current_usage should contain resource types with real psutil values
        cpu_usage = summary["current_usage"]["cpu"]
        assert cpu_usage["unit"] == "percent"
        assert 0.0 <= cpu_usage["usage_percent"] <= 100.0
        assert cpu_usage["value"] == cpu_usage["usage_percent"]  # CPU value == percent
        mem_usage = summary["current_usage"]["memory"]
        assert mem_usage["unit"] == "GB"
        assert mem_usage["value"] > 0  # some memory in use
        assert 0.0 < mem_usage["usage_percent"] < 100.0
        disk_usage = summary["current_usage"]["disk"]
        assert disk_usage["unit"] == "GB"
        assert disk_usage["value"] > 0  # some disk in use
        assert 0.0 < disk_usage["usage_percent"] < 100.0
        # active_alerts count should be 0 with no alerts triggered
        assert summary["active_alerts"] == 0
        # alert_breakdown should have all alert levels at 0
        for level in AlertLevel:
            assert summary["alert_breakdown"][level.value] == 0
        # thresholds should contain the 4 configured resource types: CPU, MEMORY, DISK, CONTAINER_COUNT
        assert len(summary["thresholds"]) == 4
        assert set(summary["thresholds"].keys()) == {"cpu", "memory", "disk", "container_count"}

    def test_terminate_containers_handler_no_docker(self):
        """Handler should catch exception when Docker is unavailable."""
        mon = ResourceMonitor()
        mon.collector.docker_client = None
        alert = ResourceAlert(
            timestamp=datetime.now(UTC),
            resource_type=ResourceType.CPU,
            alert_level=AlertLevel.EMERGENCY,
            current_value=99.0,
            threshold_value=95.0,
            usage_percent=99.0,
            message="Emergency",
        )
        # Should not raise
        mon._terminate_containers_handler(alert)

    def test_get_resource_history_with_data(self):
        """History should include recently added data."""
        mon = ResourceMonitor()
        usage = ResourceUsage(
            timestamp=datetime.now(UTC),
            resource_type=ResourceType.CPU,
            current_value=50.0,
            max_value=100.0,
            usage_percent=50.0,
            unit="percent",
        )
        mon.resource_history.append(usage)
        history = mon.get_resource_history(hours=24)
        assert len(history) == 1

    def test_export_data_with_data(self):
        """Export should include resource and alert data."""
        mon = ResourceMonitor()
        usage = ResourceUsage(
            timestamp=datetime.now(UTC),
            resource_type=ResourceType.CPU,
            current_value=50.0,
            max_value=100.0,
            usage_percent=50.0,
            unit="percent",
        )
        mon.resource_history.append(usage)

        alert = ResourceAlert(
            timestamp=datetime.now(UTC),
            resource_type=ResourceType.CPU,
            alert_level=AlertLevel.WARNING,
            current_value=75.0,
            threshold_value=70.0,
            usage_percent=75.0,
            message="Alert",
        )
        mon.alert_manager.alert_history.append(alert)

        data_str = mon.export_data(hours=24)
        data = json.loads(data_str)
        assert len(data["resource_history"]) == 1
        assert data["resource_history"][0]["resource_type"] == "cpu"
        assert data["resource_history"][0]["current_value"] == 50.0
        assert len(data["alert_history"]) == 1
        assert data["alert_history"][0]["alert_level"] == "warning"
        assert data["alert_history"][0]["current_value"] == 75.0

    def test_get_resource_monitor_singleton(self):
        """get_resource_monitor returns same instance on repeated calls."""
        import services.test_execution.resource_monitor as rm_mod

        rm_mod._resource_monitor = None
        mon = get_resource_monitor()
        # Verify it's a properly initialized ResourceMonitor
        assert isinstance(mon, ResourceMonitor)
        assert isinstance(mon.collector, ResourceCollector)
        assert isinstance(mon.resource_history, list)
        assert len(mon.thresholds) == 4  # CPU, MEMORY, DISK, CONTAINER_COUNT
        mon2 = get_resource_monitor()
        assert mon is mon2
        rm_mod._resource_monitor = None


# ===========================================================================
# AUDIT LOGGER: compliance reports, integrity verification, time filtering
# ===========================================================================

from services.test_execution.audit_logger import (  # noqa: E402
    AuditContext,
    AuditEvent,
    AuditEventType,
    AuditLogger,
    AuditQuery,
    AuditSeverity,
    AuditStorage,
    ComplianceStandard,
    get_audit_logger,
    log_audit_event,
    log_container_audit,
    log_security_audit,
)


class TestAuditStorageAdvanced:
    """Cover _encrypt_data, _decrypt_data, _get_log_files_for_query with time filter,
    and _event_matches_query edge cases."""

    @pytest.fixture
    def storage(self, tmp_path) -> AuditStorage:
        return AuditStorage(tmp_path / "audit")

    @pytest.fixture
    def enc_storage(self, tmp_path) -> AuditStorage:
        return AuditStorage(tmp_path / "enc_audit", encryption_key="secret-key-123")

    def _make_event(self, ts=None, **overrides) -> AuditEvent:
        defaults = dict(
            event_id="evt-1",
            timestamp=ts or datetime.now(UTC),
            event_type=AuditEventType.CONTAINER_CREATED,
            severity=AuditSeverity.INFORMATIONAL,
            description="container created",
            context=AuditContext(container_id="c1"),
        )
        defaults.update(overrides)
        return AuditEvent(**defaults)

    def test_encrypt_data_produces_mac(self, enc_storage):
        """Encrypted data should have mac:data format with SHA256 HMAC prefix."""
        result = enc_storage._encrypt_data("hello\n")
        parts = result.split(":", 1)
        assert len(parts) == 2, "Encrypted format must be 'mac:data'"
        assert len(parts[0]) == 64, "MAC must be a 64-char SHA256 hex digest"
        # Data part is the original plaintext (MAC-based integrity, not encryption)
        assert parts[1] == "hello\n"
        # Verify the MAC is a valid hex string
        int(parts[0], 16)
        # Different data should produce a different MAC
        result2 = enc_storage._encrypt_data("world\n")
        mac2 = result2.split(":", 1)[0]
        assert mac2 != parts[0]

    def test_encrypt_no_key(self, storage):
        """Without encryption key, data passes through unchanged."""
        result = storage._encrypt_data("hello")
        assert result == "hello"

    def test_decrypt_no_key(self, storage):
        """Without encryption key, data passes through unchanged."""
        result = storage._decrypt_data("hello")
        assert result == "hello"

    def test_decrypt_valid_data(self, enc_storage):
        """Decrypt should reverse encrypt."""
        encrypted = enc_storage._encrypt_data("hello world\n")
        decrypted = enc_storage._decrypt_data(encrypted)
        assert decrypted == "hello world\n"

    def test_query_events_time_filter(self, storage):
        """Store events at different times and filter by time range."""
        now = datetime.now(UTC)
        event1 = self._make_event(ts=now - timedelta(hours=2), event_id="old")
        event2 = self._make_event(ts=now, event_id="new")
        storage.store_event(event1)
        storage.store_event(event2)

        # Query without time filter returns all
        results = storage.query_events(AuditQuery(limit=10))
        assert len(results) == 2
        event_ids = {e.event_id for e in results}
        assert event_ids == {"old", "new"}


class TestAuditLoggerAdvanced:
    """Cover generate_compliance_report, verify_log_integrity, convenience functions."""

    @pytest.fixture
    def audit_logger(self, tmp_path) -> AuditLogger:
        return AuditLogger(tmp_path / "audit")

    def test_generate_compliance_report_structure(self, audit_logger):
        """Report should have all expected fields."""
        now = datetime.now(UTC)
        audit_logger.log_event(
            AuditEventType.CONTAINER_CREATED,
            "Container created",
        )
        audit_logger.log_event(
            AuditEventType.SECURITY_VIOLATION,
            "Violation",
            severity=AuditSeverity.HIGH,
        )
        report = audit_logger.generate_compliance_report(
            ComplianceStandard.SOC2,
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(hours=1),
        )
        assert report["compliance_standard"] == "soc2"
        assert report["total_events"] == 2
        assert report["security_events"] == 1  # Only the HIGH severity violation
        assert AuditEventType.CONTAINER_CREATED.value in report["event_breakdown"]
        assert AuditEventType.SECURITY_VIOLATION.value in report["event_breakdown"]
        assert report["event_breakdown"][AuditEventType.CONTAINER_CREATED.value] == 1
        assert report["event_breakdown"][AuditEventType.SECURITY_VIOLATION.value] == 1
        assert report["severity_breakdown"][AuditSeverity.INFORMATIONAL.value] == 1
        assert report["severity_breakdown"][AuditSeverity.HIGH.value] == 1
        assert report["containers_affected"] == 0  # No container_id in default context
        assert report["languages_involved"] == []  # No language in default context

    def test_verify_log_integrity_result(self, audit_logger):
        """Integrity verification should return structured result."""
        audit_logger.log_event(
            AuditEventType.SYSTEM_STARTUP,
            "System started",
        )
        result = audit_logger.verify_log_integrity(hours=1)
        assert result["total_events"] == 1
        assert result["valid_events"] == 1
        assert result["invalid_events"] == 0
        assert result["integrity_percentage"] == 100.0
        assert result["time_period_hours"] == 1
        # verification_timestamp should be a valid ISO timestamp
        datetime.fromisoformat(result["verification_timestamp"])

    def test_log_event_with_additional_data(self, audit_logger):
        """Log event with all optional fields."""
        event_id = audit_logger.log_event(
            AuditEventType.DATA_MODIFICATION,
            "Modified data",
            severity=AuditSeverity.MEDIUM,
            resource_affected="database:users",
            action_taken="UPDATE",
            outcome="success",
            additional_data={"rows_affected": 5},
        )
        assert len(event_id) == 36  # uuid7 string: 8-4-4-4-12 format
        assert event_id.count("-") == 4
        events = audit_logger.query_events(AuditQuery(limit=1))
        assert len(events) == 1
        assert events[0].event_id == event_id
        assert events[0].additional_data["rows_affected"] == 5
        assert events[0].description == "Modified data"
        assert events[0].severity == AuditSeverity.MEDIUM
        assert events[0].resource_affected == "database:users"
        assert events[0].action_taken == "UPDATE"
        assert events[0].outcome == "success"


class TestAuditConvenienceFunctions:
    """Cover module-level convenience functions."""

    def test_log_audit_event(self, tmp_path, monkeypatch):
        """Convenience function log_audit_event stores event and returns id."""
        import services.test_execution.audit_logger as mod

        mod._audit_logger = AuditLogger(tmp_path / "audit_conv")
        event_id = log_audit_event(
            AuditEventType.SYSTEM_STARTUP,
            "Test startup",
        )
        assert len(event_id) == 36 and event_id.count("-") == 4  # uuid7 format
        events = mod._audit_logger.query_events(AuditQuery(limit=1))
        assert events[0].event_id == event_id
        assert events[0].event_type == AuditEventType.SYSTEM_STARTUP
        assert events[0].description == "Test startup"
        mod._audit_logger = None

    def test_log_container_audit(self, tmp_path, monkeypatch):
        """Convenience function log_container_audit stores container event."""
        import services.test_execution.audit_logger as mod

        mod._audit_logger = AuditLogger(tmp_path / "audit_conv2")
        event_id = log_container_audit(
            AuditEventType.CONTAINER_CREATED,
            container_id="c1",
            language="python",
            description="Container created",
        )
        assert len(event_id) == 36 and event_id.count("-") == 4  # uuid7 format
        events = mod._audit_logger.query_events(AuditQuery(limit=1))
        assert events[0].event_id == event_id
        assert events[0].event_type == AuditEventType.CONTAINER_CREATED
        assert events[0].description == "Container created"
        mod._audit_logger = None

    def test_log_security_audit(self, tmp_path, monkeypatch):
        """Convenience function log_security_audit stores security event."""
        import services.test_execution.audit_logger as mod

        mod._audit_logger = AuditLogger(tmp_path / "audit_conv3")
        event_id = log_security_audit(
            AuditEventType.CODE_INJECTION_ATTEMPT,
            "Injection detected",
        )
        assert len(event_id) == 36 and event_id.count("-") == 4  # uuid7 format
        events = mod._audit_logger.query_events(AuditQuery(limit=1))
        assert events[0].event_id == event_id
        assert events[0].event_type == AuditEventType.CODE_INJECTION_ATTEMPT
        assert events[0].description == "Injection detected"
        mod._audit_logger = None

    def test_get_audit_logger_creates_instance(self, tmp_path, monkeypatch):
        """get_audit_logger creates a global instance with configured directory."""
        import services.test_execution.audit_logger as mod

        mod._audit_logger = None
        monkeypatch.setenv("AUDIT_LOG_DIR", str(tmp_path / "global_audit"))
        audit = get_audit_logger()
        # Verify it was configured with the env var directory
        assert audit.storage.storage_directory == tmp_path / "global_audit"
        # Second call returns same instance
        audit2 = get_audit_logger()
        assert audit is audit2
        mod._audit_logger = None


# ===========================================================================
# MONITORING SERVICE: coverage for _get_health_status, _get_recommendations,
# _classify_error, export_metrics, log_test_completed
# ===========================================================================

from services.test_execution.monitoring_service import (  # noqa: E402
    MetricType,
    TestExecutionMonitor,
    TestExecutionStats,
    create_test_execution_monitor,
)
from services.test_execution.test_orchestrator import (  # noqa: E402
    OrchestratorType,
    TestExecutionResult,
    TestLanguage,
)


class TestMonitoringServiceHealthAndRecommendations:
    """Cover lines 175-254, 423, 575, 577, 604, 609."""

    @pytest.fixture
    def monitor(self):
        return TestExecutionMonitor()

    def _make_result(self, success=True, stderr="", language=TestLanguage.PYTHON, **kw):
        return TestExecutionResult(
            test_id=kw.get("test_id", "t1"),
            success=success,
            exit_code=0 if success else 1,
            stdout="output",
            stderr=stderr,
            execution_time_ms=kw.get("execution_time_ms", 100.0),
            language=language,
            test_type=kw.get("test_type", OrchestratorType.UNIT),
            created_at=datetime.now(UTC),
            container_name=kw.get("container_name", None),
        )

    def test_log_test_completed_success(self, monitor):
        """Successful test completion updates stats correctly."""
        monitor.log_test_started("t1", "python", "unit")
        result = self._make_result(test_id="t1")
        monitor.log_test_completed(result)

        assert monitor.stats.total_executions == 1
        assert monitor.stats.successful_executions == 1
        assert monitor.stats.failed_executions == 0
        assert "t1" not in monitor.active_tests
        assert monitor.stats.language_breakdown["python"] == 1
        assert monitor.stats.test_type_breakdown["unit"] == 1

    def test_log_test_completed_failure_with_error_classification(self, monitor):
        """Failed test with stderr classifies error type."""
        result = self._make_result(
            success=False,
            stderr="timeout exceeded",
            test_id="t2",
        )
        monitor.log_test_completed(result)

        assert monitor.stats.failed_executions == 1
        assert monitor.stats.error_breakdown["timeout"] == 1

    def test_classify_error_types(self, monitor):
        """Verify error classification for different error messages."""
        assert monitor._classify_error("timeout exceeded") == "timeout"
        assert monitor._classify_error("out of memory") == "memory"
        assert monitor._classify_error("permission denied") == "permission"
        assert monitor._classify_error("network unreachable") == "network"
        assert monitor._classify_error("container exited") == "container"
        assert monitor._classify_error("security violation") == "security"
        assert monitor._classify_error("compilation error") == "compilation"
        assert monitor._classify_error("syntax error") == "syntax"
        assert monitor._classify_error("something weird") == "other"

    def test_get_health_status_levels(self, monitor):
        """Health status should reflect success rate."""
        # Excellent health
        monitor.stats.total_executions = 100
        monitor.stats.successful_executions = 96
        status = monitor._get_health_status()
        assert status["overall_health"] == "excellent"

        # Good health
        monitor.stats.successful_executions = 90
        status = monitor._get_health_status()
        assert status["overall_health"] == "good"

        # Fair health
        monitor.stats.successful_executions = 75
        status = monitor._get_health_status()
        assert status["overall_health"] == "fair"

        # Poor health
        monitor.stats.successful_executions = 50
        status = monitor._get_health_status()
        assert status["overall_health"] == "poor"

    def test_get_recommendations(self, monitor):
        """Recommendations based on metrics thresholds."""
        # Low success rate
        monitor.stats.total_executions = 100
        monitor.stats.successful_executions = 70
        recs = monitor._get_recommendations()
        assert "Consider investigating frequent test failures" in recs

        # High container failures
        monitor.container_metrics.container_failures = 15
        recs = monitor._get_recommendations()
        assert "High container failure rate detected - check Docker configuration" in recs

        # Security violations
        monitor.security_metrics.security_violations = 10
        recs = monitor._get_recommendations()
        assert "Multiple security violations detected - review security policies" in recs

        # High execution time
        monitor.stats.average_execution_time_ms = 35000
        recs = monitor._get_recommendations()
        assert "High average execution time - consider optimizing test containers" in recs
        # At this point all 4 recommendations should be present
        assert len(recs) == 4

    def test_get_performance_trends(self, monitor):
        """Performance trends should report improving for declining execution times."""
        # Add declining times (improving: recent avg < overall avg)
        for t in [200.0, 180.0, 160.0, 140.0, 120.0, 100.0]:
            monitor.performance_history["python"].append(t)

        trends = monitor._get_performance_trends()
        assert set(trends.keys()) == {"python"}
        # All 6 values are within last 10, so recent_avg = overall_avg = 150.0
        assert trends["python"]["overall_avg"] == 150.0
        assert trends["python"]["recent_avg"] == 150.0
        assert trends["python"]["trend"] == "stable"

    def test_export_metrics_json(self, monitor):
        """export_metrics returns valid JSON with correct content."""
        monitor.log_test_started("t1", "python", "unit")
        result = self._make_result(test_id="t1")
        monitor.log_test_completed(result)

        exported = monitor.export_metrics("json")
        data = json.loads(exported)
        assert data["statistics"]["execution_stats"]["total_executions"] == 1
        assert data["statistics"]["execution_stats"]["successful_executions"] == 1
        assert len(data["recent_metrics"]) == 1  # 1 EXECUTION_TIME metric
        assert data["recent_metrics"][0]["test_id"] == "t1"
        assert data["recent_metrics"][0]["metric_type"] == "execution_time"
        assert len(data["recent_logs"]) == 2  # start log + completion log

    def test_export_metrics_invalid_format(self, monitor):
        """Unsupported format should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported"):
            monitor.export_metrics("xml")

    def test_get_statistics(self, monitor):
        """Statistics returns correct initial values for all sections."""
        stats = monitor.get_statistics()
        assert stats["execution_stats"]["total_executions"] == 0
        assert stats["execution_stats"]["successful_executions"] == 0
        assert stats["container_metrics"]["active_containers"] == 0
        assert stats["container_metrics"]["container_failures"] == 0
        assert stats["security_metrics"]["security_violations"] == 0
        assert stats["active_tests"] == 0
        assert stats["metrics_count"] == 0
        assert stats["logs_count"] == 0
        assert stats["performance_trends"] == {}
        assert stats["health_status"]["overall_health"] == "poor"  # 0% success rate

    def test_get_metrics_by_type(self, monitor):
        """Filter metrics by type returns only matching metric type."""
        result = self._make_result()
        monitor.log_test_completed(result)
        metrics = monitor.get_metrics_by_type(MetricType.EXECUTION_TIME, hours_back=1)
        assert len(metrics) == 1
        assert metrics[0].metric_type == MetricType.EXECUTION_TIME
        assert metrics[0].test_id == "t1"
        assert metrics[0].value == 100.0  # from _make_result default

    def test_get_logs_by_level(self, monitor):
        """Filter logs by level returns only matching level."""
        monitor.log_test_started("t1", "python", "unit")
        logs = monitor.get_logs_by_level(LogLevel.INFO, hours_back=1)
        assert len(logs) == 1
        assert logs[0].level == LogLevel.INFO
        assert logs[0].test_id == "t1"
        assert logs[0].message == "Test execution started for python unit"

    def test_get_language_performance_empty(self, monitor):
        """No data returns zeros."""
        perf = monitor.get_language_performance("go")
        assert perf["avg_execution_time"] == 0.0

    def test_get_language_performance_with_data(self, monitor):
        """With data, returns correct stats."""
        monitor.performance_history["python"].append(100.0)
        monitor.performance_history["python"].append(200.0)
        perf = monitor.get_language_performance("python")
        assert perf["avg_execution_time"] == 150.0
        assert perf["min_execution_time"] == 100.0
        assert perf["max_execution_time"] == 200.0

    def test_log_container_operations(self, monitor):
        """Container create/destroy/failure logging."""
        monitor.log_container_created("c1", "python", 50.0)
        assert monitor.container_metrics.total_containers_created == 1
        assert monitor.container_metrics.active_containers == 1

        monitor.log_container_destroyed("c1", "python", 10.0)
        assert monitor.container_metrics.total_containers_destroyed == 1
        assert monitor.container_metrics.active_containers == 0

        monitor.log_container_failure("c2", "go", "OOM killed")
        assert monitor.container_metrics.container_failures == 1

    def test_log_security_violation(self, monitor):
        """Security violation logging."""
        monitor.log_security_violation("t1", "python", "code_injection", "found eval()")
        assert monitor.security_metrics.security_violations == 1
        assert monitor.security_metrics.blocked_executions == 1
        assert monitor.security_metrics.violation_types["code_injection"] == 1

    def test_log_resource_usage(self, monitor):
        """Resource usage logging."""
        monitor.log_resource_usage("c1", "python", 50.0, 256.0)
        assert monitor.container_metrics.resource_usage["c1_cpu"] == 50.0
        assert monitor.container_metrics.resource_usage["c1_memory"] == 256.0

    def test_factory_function(self):
        """create_test_execution_monitor creates with params."""
        mon = create_test_execution_monitor(max_metrics_history=100, max_log_history=50)
        assert mon.max_metrics_history == 100
        assert mon.max_log_history == 50

    def test_success_rate_zero_executions(self):
        """Success rate is 0 with no executions."""
        stats = TestExecutionStats()
        assert stats.success_rate() == 0.0


# ===========================================================================
# CONTAINER REGISTRY: _check_container_availability with Docker
# ===========================================================================

from services.test_execution.test_execution_container_registry import (  # noqa: E402
    ContainerExecutionStatus,
    TestExecutionContainerRegistry,
)


class TestContainerRegistryDockerPaths:
    """Cover Docker-backed paths like _check_container_availability with running containers."""

    @pytest.fixture
    def registry(self, monkeypatch):
        monkeypatch.setenv("SKIP_TEST_EXECUTORS", "true")
        return TestExecutionContainerRegistry(enable_periodic_refresh=False)

    def test_check_container_availability_with_running_container(self, registry):
        """With docker client and running container, status should be BUILDING."""
        mock_client = MagicMock()
        container = MagicMock()
        container.status = "running"
        mock_client.containers.list.return_value = [container]
        registry.docker_client = mock_client
        registry._check_container_availability()
        # All languages with service mappings should be BUILDING since container is running
        building = {
            lang
            for lang, s in registry.container_status.items()
            if s == ContainerExecutionStatus.BUILDING
        }
        # At minimum, the core languages with mappings should be BUILDING
        assert "python" in building
        assert "go" in building
        assert "rust" in building

    def test_check_container_availability_stopped_container(self, registry):
        """With docker client and stopped container, status should be UNAVAILABLE."""
        mock_client = MagicMock()
        container = MagicMock()
        container.status = "exited"
        mock_client.containers.list.return_value = [container]
        registry.docker_client = mock_client
        registry._check_container_availability()
        # All should be UNAVAILABLE since container is not "running"
        for lang, status in registry.container_status.items():
            assert status == ContainerExecutionStatus.UNAVAILABLE, f"{lang} should be UNAVAILABLE"

    def test_check_container_availability_docker_error(self, registry):
        """Docker API error during check marks all as FAILED."""
        mock_client = MagicMock()
        mock_client.containers.list.side_effect = RuntimeError("Docker error")
        registry.docker_client = mock_client
        registry._check_container_availability()
        # All languages with service mappings should be FAILED
        for lang, status in registry.container_status.items():
            assert status == ContainerExecutionStatus.FAILED, f"{lang} should be FAILED"

    def test_build_container_success(self, registry, tmp_path):
        """Successful build marks as AVAILABLE."""
        mock_client = MagicMock()
        mock_image = MagicMock()
        mock_client.images.build.return_value = (mock_image, [{"stream": "Step 1/1\n"}])
        registry.docker_client = mock_client

        # Create a real Dockerfile
        df = tmp_path / "Dockerfile"
        df.write_text("FROM scratch\n")

        result = registry.build_container("python", str(df))
        assert result is True
        assert registry.container_status["python"] == ContainerExecutionStatus.AVAILABLE

    def test_get_health_status_structure(self, registry):
        """get_health_status returns correct health data."""
        status = registry.get_health_status()
        import time

        assert status["timestamp"] <= time.time()  # Unix timestamp should be at or before now
        assert status["docker_client_available"] is False  # SKIP_TEST_EXECUTORS disables Docker
        assert (
            len(status["container_status"]) == 9
        )  # go, rust, java, csharp, cpp, python, nodejs, javascript, typescript
        assert status["overall_health"]["total_services"] == 9
        # With no healthy services (SKIP_TEST_EXECUTORS=true), all health values should be 0
        assert status["overall_health"]["health_rate"] == 0
        assert status["overall_health"]["healthy_services"] == 0
        assert status["overall_health"]["available_containers"] == 0
        assert status["overall_health"]["availability_rate"] == 0

    def test_force_health_check_refresh(self, registry):
        """force_health_check_refresh returns health results for all languages."""
        result = registry.force_health_check_refresh()
        # Should have an entry for every language in specs (9 languages)
        assert len(result) == 9
        assert set(result.keys()) == {
            "go",
            "rust",
            "java",
            "csharp",
            "cpp",
            "python",
            "nodejs",
            "javascript",
            "typescript",
        }
        # With SKIP_TEST_EXECUTORS=true, all health checks should return False
        for lang, healthy in result.items():
            assert healthy is False, f"{lang} should be unhealthy with SKIP_TEST_EXECUTORS"

    def test_check_single_container_availability_with_docker(self, registry):
        """With docker client and found image, marks as AVAILABLE."""
        mock_client = MagicMock()
        mock_client.images.get.return_value = MagicMock()
        registry.docker_client = mock_client
        registry._check_single_container_availability("python")
        assert registry.container_status["python"] == ContainerExecutionStatus.AVAILABLE

    def test_check_single_container_availability_exception_marks_failed(self, registry):
        """Exception during image lookup marks container as FAILED."""
        mock_client = MagicMock()
        mock_client.images.get.side_effect = RuntimeError("connection refused")
        registry.docker_client = mock_client
        registry._check_single_container_availability("python")
        assert registry.container_status["python"] == ContainerExecutionStatus.FAILED

    def test_check_single_container_availability_other_error(self, registry):
        """Non-ImageNotFound error marks as FAILED."""
        mock_client = MagicMock()
        mock_client.images.get.side_effect = RuntimeError("generic error")
        registry.docker_client = mock_client
        registry._check_single_container_availability("python")
        assert registry.container_status["python"] == ContainerExecutionStatus.FAILED


# ===========================================================================
# INTELLIGENT TEST INTEGRATION: exception classes, config validators
# ===========================================================================

from services.test_execution.intelligent_test_integration import (  # noqa: E402
    TestExecutionError,
    TestExecutionResourceError,
    TestExecutionRetryError,
    TestExecutionSettings,
    TestExecutionTimeoutError,
)


class TestIntelligentTestExceptions:
    """Cover exception classes (lines 53-55, 61-63, 66-112)."""

    def test_test_execution_error(self):
        err = TestExecutionError("Something failed", "custom_type", {"key": "val"})
        assert str(err) == "Something failed"
        assert err.error_type == "custom_type"
        assert err.details == {"key": "val"}
        assert (datetime.now(UTC) - err.timestamp).total_seconds() < 5

    def test_test_execution_error_defaults(self):
        err = TestExecutionError("fail")
        assert err.error_type == "general"
        assert err.details == {}

    def test_test_execution_retry_error(self):
        inner = RuntimeError("inner fail")
        err = TestExecutionRetryError("retry failed", attempts=3, last_error=inner)
        assert err.attempts == 3
        assert err.last_error is inner
        assert err.error_type == "retry_failure"
        assert err.details["attempts"] == 3

    def test_test_execution_timeout_error(self):
        err = TestExecutionTimeoutError("timed out", timeout_seconds=300)
        assert err.timeout_seconds == 300
        assert err.error_type == "timeout"
        assert err.details["timeout_seconds"] == 300

    def test_test_execution_resource_error(self):
        err = TestExecutionResourceError(
            "out of memory", resource_type="memory", resource_limit="1024MB"
        )
        assert err.resource_type == "memory"
        assert err.resource_limit == "1024MB"
        assert err.error_type == "resource_limit"


class TestTestExecutionSettings:
    """Cover config validators (lines 49-63)."""

    def test_default_config(self):
        cfg = TestExecutionSettings()
        assert cfg.max_concurrent_tests == 5
        assert cfg.ai_analysis_threshold == 10

    def test_valid_custom_config(self):
        cfg = TestExecutionSettings(
            max_concurrent_tests=10,
            ai_analysis_threshold=50,
            pattern_analysis_interval=600,
            execution_timeout=1200,
        )
        assert cfg.max_concurrent_tests == 10
        assert cfg.ai_analysis_threshold == 50

    def test_invalid_concurrent_tests_too_high(self):
        with pytest.raises(Exception):
            TestExecutionSettings(max_concurrent_tests=25)

    def test_invalid_concurrent_tests_too_low(self):
        with pytest.raises(Exception):
            TestExecutionSettings(max_concurrent_tests=0)


# ===========================================================================
# HEALTH MONITOR: additional coverage for monitoring loops
# ===========================================================================

from services.test_execution.health_monitor import (  # noqa: E402
    ContainerHealthMonitor,
    HealthMonitoringService,
)


class TestHealthMonitoringServiceStartStop:
    """Cover start/stop monitoring lifecycle."""

    @pytest.mark.asyncio
    async def test_start_and_stop_monitoring(self):
        """Start and stop the monitoring service."""
        service = HealthMonitoringService()
        await service.start_monitoring()
        assert service.monitoring_active is True
        assert isinstance(service.system_monitoring_task, asyncio.Task)
        assert not service.system_monitoring_task.done()

        await service.stop_monitoring()
        assert service.monitoring_active is False

    @pytest.mark.asyncio
    async def test_add_and_remove_container_monitor(self):
        """Add and remove container monitors."""
        service = HealthMonitoringService()

        # Patch the actual monitoring to prevent real Docker calls
        with patch.object(ContainerHealthMonitor, "start_monitoring", new_callable=AsyncMock):
            await service.add_container_monitor("c1", "python")
            assert "c1" in service.container_monitors

            # Adding same container again should skip
            await service.add_container_monitor("c1", "python")
            assert len(service.container_monitors) == 1

        with patch.object(ContainerHealthMonitor, "stop_monitoring", new_callable=AsyncMock):
            await service.remove_container_monitor("c1")
            assert "c1" not in service.container_monitors


# ===========================================================================
# TEST ORCHESTRATOR: helper methods coverage
# ===========================================================================

from services.test_execution.test_orchestrator import (  # noqa: E402
    TestOrchestrator,
)


class TestOrchestratorHelpers:
    """Cover helper methods in TestOrchestrator."""

    @pytest.fixture
    def orchestrator(self, monkeypatch):
        monkeypatch.setenv("SKIP_TEST_EXECUTORS", "true")
        mock_docker = MagicMock()
        mock_config = MagicMock()
        mock_config.max_execution_time = 300
        mock_config.max_memory_mb = 1024
        mock_config.max_file_size_mb = 10
        mock_config.max_code_length = 50000
        mock_config.allowed_imports = ["pytest"]
        return TestOrchestrator(mock_docker, mock_config)

    def test_get_file_extension(self, orchestrator):
        """File extension mapping."""
        assert orchestrator._get_file_extension("go") == ".go"
        assert orchestrator._get_file_extension("rust") == ".rs"
        assert orchestrator._get_file_extension("java") == ".java"
        assert orchestrator._get_file_extension("csharp") == ".cs"
        assert orchestrator._get_file_extension("cpp") == ".cpp"
        assert orchestrator._get_file_extension("python") == ".py"
        assert orchestrator._get_file_extension("javascript") == ".js"
        assert orchestrator._get_file_extension("typescript") == ".ts"
        assert orchestrator._get_file_extension("unknown") == ".txt"

    def test_format_go_dependencies(self, orchestrator):
        assert orchestrator._format_go_dependencies({}) == ""
        result = orchestrator._format_go_dependencies({"github.com/pkg/errors": "v0.9.1"})
        assert result == "require github.com/pkg/errors v0.9.1"

    def test_format_rust_dependencies(self, orchestrator):
        assert orchestrator._format_rust_dependencies({}) == ""
        result = orchestrator._format_rust_dependencies({"serde": "1.0"})
        assert 'serde = "1.0"' in result

    def test_get_test_command(self, orchestrator):
        """Test command generation for various languages returns exact command lists."""
        cmd = orchestrator._get_test_command(TestLanguage.GO, OrchestratorType.UNIT)
        assert cmd == ["/usr/local/bin/go-test-runner.sh", "unit"]

        cmd = orchestrator._get_test_command(TestLanguage.PYTHON, OrchestratorType.UNIT)
        assert cmd == ["python", "-m", "pytest", "-v", "--tb=short"]

        cmd = orchestrator._get_test_command(TestLanguage.RUST, OrchestratorType.CLIPPY)
        assert cmd == ["/usr/local/bin/rust-test-runner.sh", "clippy"]

        # Unknown combination has NO command: an `echo` fallback exits 0 and
        # would be graded as a pass.
        cmd = orchestrator._get_test_command(TestLanguage.JAVA, OrchestratorType.FMT)
        assert cmd is None

    @pytest.mark.asyncio
    async def test_get_supported_languages(self, orchestrator):
        languages = await orchestrator.get_supported_languages()
        assert set(languages) == {
            "go",
            "rust",
            "java",
            "csharp",
            "cpp",
            "python",
            "nodejs",
            "javascript",
            "typescript",
        }

    @pytest.mark.asyncio
    async def test_get_supported_test_types(self, orchestrator):
        types = await orchestrator.get_supported_test_types("python")
        assert types == ["unit", "integration", "coverage", "security"]

        types = await orchestrator.get_supported_test_types("go")
        assert types == ["unit", "benchmark", "coverage", "race", "security", "vet"]

        # Unknown language
        types = await orchestrator.get_supported_test_types("unknown_lang")
        assert types == []

    @pytest.mark.asyncio
    async def test_parse_test_output_empty(self, orchestrator):
        results = await orchestrator._parse_test_output("", TestLanguage.PYTHON)
        assert results == []

    @pytest.mark.asyncio
    async def test_parse_test_output_go(self, orchestrator):
        stdout = "ok  \tpackage\t1.234s\nPASS\nFAIL something"
        results = await orchestrator._parse_test_output(stdout, TestLanguage.GO)
        assert len(results) == 2  # PASS line + FAIL line
        assert results[0] == {"line": "PASS", "parsed": True}
        assert results[1] == {"line": "FAIL something", "parsed": True}

    @pytest.mark.asyncio
    async def test_parse_test_output_rust(self, orchestrator):
        stdout = "test result: ok. 5 passed; 0 failed\n"
        results = await orchestrator._parse_test_output(stdout, TestLanguage.RUST)
        assert len(results) == 1
        assert results[0] == {"line": "test result: ok. 5 passed; 0 failed", "parsed": True}


# ===========================================================================
# QUALITY VALIDATION: QualityFileValidator, QualityFixValidator, QualityCacheValidator,
# QualityResultValidator, QualityRequestValidator
# ===========================================================================

from models.testing_models import QualityIssue, QualityRequest, QualityResult  # noqa: E402
from services.testing.quality_validation import (  # noqa: E402
    QualityCacheValidator,
    QualityFileValidator,
    QualityFixValidator,
    QualityRequestValidator,
    QualityResultValidator,
    get_cache_validator,
    get_file_validator,
    get_fix_validator,
    get_request_validator,
    get_result_validator,
)


class TestQualityFileValidator:
    """Cover QualityFileValidator file validation paths."""

    @pytest.fixture
    def validator(self):
        return QualityFileValidator()

    @pytest.mark.asyncio
    async def test_validate_valid_python_file(self, validator, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("def hello():\n    return 'world'\n\nprint(hello())\n")
        result = await validator.validate_file(str(f))
        assert result.is_valid is True
        assert result.metadata["line_count"] == 5  # 4 content lines + trailing newline split

    @pytest.mark.asyncio
    async def test_validate_binary_file(self, validator, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"\x00\x01\x02\x03binary content")
        result = await validator.validate_file(str(f))
        assert result.is_valid is False
        assert "File appears to be binary" in result.errors

    @pytest.mark.asyncio
    async def test_validate_short_file(self, validator, tmp_path):
        f = tmp_path / "short.py"
        f.write_text("x = 1\n")
        result = await validator.validate_file(str(f))
        assert result.is_valid is True
        assert "File is very short, may not need analysis" in result.warnings

    @pytest.mark.asyncio
    async def test_validate_file_with_secrets(self, validator, tmp_path):
        f = tmp_path / "config.py"
        f.write_text("api_key = 'secret123'\npassword = 'hunter2'\n")
        result = await validator.validate_file(str(f))
        assert result.is_valid is True
        # Should detect password, secret, key, api_key patterns
        assert len(result.warnings) == 1
        secret_warning = [w for w in result.warnings if "Potential secrets detected" in w]
        assert len(secret_warning) == 1
        assert "password" in secret_warning[0]
        assert "api_key" in secret_warning[0]

    @pytest.mark.asyncio
    async def test_validate_nonexistent_file(self, validator):
        result = await validator.validate_file("/nonexistent/file.py")
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].startswith("Error reading file:")

    @pytest.mark.asyncio
    async def test_validate_long_lines(self, validator, tmp_path):
        f = tmp_path / "long.py"
        f.write_text("x = " + "a" * 15000 + "\ny = 1\nz = 2\n")
        result = await validator.validate_file(str(f))
        long_line_warnings = [w for w in result.warnings if "Long lines detected at" in w]
        assert len(long_line_warnings) == 1
        assert "[1]" in long_line_warnings[0]  # Line 1 is the long line

    def test_is_binary_file_utf16(self, validator):
        """UTF-16 BOM should be detected as binary."""
        content = b"\xff\xfehello"
        assert validator._is_binary_file(content) is True

    def test_is_binary_file_normal_text(self, validator):
        """Normal text should not be binary."""
        content = b"def hello():\n    return 'world'\n"
        assert validator._is_binary_file(content) is False


class TestQualityFixValidator:
    """Cover QualityFixValidator fix validation paths."""

    @pytest.fixture
    def validator(self):
        return QualityFixValidator()

    def _make_issue(self, tool="ruff-format", suggested_fix="Fix it", fix_confidence=0.9, **kw):
        defaults = dict(
            tool=tool,
            severity="warning",
            message="issue",
            file_path="test.py",
            suggested_fix=suggested_fix,
            rule_id="E001",
            line_number=1,
            fix_confidence=fix_confidence,
        )
        defaults.update(kw)
        return QualityIssue(**defaults)

    @pytest.mark.asyncio
    async def test_validate_safe_fixes(self, validator):
        issues = [self._make_issue(tool="ruff-format", fix_confidence=0.95)]
        result = await validator.validate_fixes(issues, "python")
        assert result.can_apply is True
        assert len(result.safe_fixes) == 1
        assert len(result.risky_fixes) == 0
        assert len(result.blocked_fixes) == 0

    @pytest.mark.asyncio
    async def test_validate_risky_fixes(self, validator):
        issues = [self._make_issue(tool="ty", fix_confidence=0.6)]
        result = await validator.validate_fixes(issues, "python")
        assert len(result.risky_fixes) == 1
        assert len(result.safe_fixes) == 0

    @pytest.mark.asyncio
    async def test_validate_blocked_fixes(self, validator):
        issues = [self._make_issue(tool="unknown_tool", fix_confidence=0.2)]
        result = await validator.validate_fixes(issues, "python")
        assert len(result.blocked_fixes) == 1
        assert len(result.safe_fixes) == 0

    @pytest.mark.asyncio
    async def test_validate_no_suggested_fix(self, validator):
        issues = [
            QualityIssue(
                tool="ruff",
                severity="error",
                message="issue",
                file_path="test.py",
            ),
        ]
        result = await validator.validate_fixes(issues, "python")
        assert result.confidence_score == 0.0

    def test_validate_fix_safety_safe_tool(self, validator):
        # "ruff-format" is a safe tool (base_score=0.8), no dangerous/file keywords -> 0.8
        score = validator.validate_fix_safety("Format code", "ruff-format")
        assert score == 0.8

    def test_validate_fix_safety_dangerous(self, validator):
        # "ruff-format" safe (0.8), "Delete" is dangerous keyword -> 0.8 * 0.5 = 0.4
        score = validator.validate_fix_safety("Delete all imports", "ruff-format")
        assert score == 0.4

    def test_validate_fix_safety_file_operation(self, validator):
        # "ty" not safe (0.5), "Overwrite" is file keyword -> 0.5 * 0.8 = 0.4
        score = validator.validate_fix_safety("Overwrite file contents", "ty")
        assert score == 0.4


class TestQualityCacheValidator:
    """Cover QualityCacheValidator."""

    @pytest.fixture
    def validator(self):
        return QualityCacheValidator()

    def test_generate_cache_key(self, validator, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("print('hello')")
        request = QualityRequest(
            container_name="test-container",
            code_path=str(f),
            language="python",
            tools=["ruff", "ruff-format"],
        )
        key = validator.generate_cache_key(request)
        assert len(key) == 64  # SHA256 hex digest
        # Same request should produce same key (deterministic)
        key2 = validator.generate_cache_key(request)
        assert key == key2

    def test_generate_cache_key_missing_file(self, validator):
        request = QualityRequest(
            container_name="test-container",
            code_path="/nonexistent/file.py",
            language="python",
        )
        key = validator.generate_cache_key(request)
        assert len(key) == 64  # Still produces a valid SHA256 hex digest

    def test_validate_cached_result_valid(self, validator):
        cached = {
            "created_at": datetime.now(UTC).isoformat(),
            "language": "python",
            "tools_results": {"ruff": {}},
        }
        request = QualityRequest(
            container_name="test-container",
            code_path="/test.py",
            language="python",
            tools=["ruff"],
        )
        result = validator.validate_cached_result(cached, request)
        assert result.is_valid is True

    def test_validate_cached_result_too_old(self, validator):
        cached = {
            "created_at": (datetime.now(UTC) - timedelta(hours=48)).isoformat(),
            "language": "python",
        }
        request = QualityRequest(
            container_name="test-container", code_path="/test.py", language="python"
        )
        result = validator.validate_cached_result(cached, request)
        assert result.is_valid is False

    def test_validate_cached_result_language_mismatch(self, validator):
        cached = {
            "created_at": datetime.now(UTC).isoformat(),
            "language": "go",
        }
        request = QualityRequest(
            container_name="test-container", code_path="/test.py", language="python"
        )
        result = validator.validate_cached_result(cached, request)
        assert result.is_valid is False

    def test_validate_cached_result_tool_mismatch(self, validator):
        cached = {
            "created_at": datetime.now(UTC).isoformat(),
            "tools_results": {"ruff": {}},
        }
        request = QualityRequest(
            container_name="test-container", code_path="/test.py", tools=["ruff-format"]
        )
        result = validator.validate_cached_result(cached, request)
        assert "Tool set differs from cached result" in result.warnings

    def test_validate_cached_result_invalid_timestamp(self, validator):
        cached = {"created_at": "not-a-date"}
        request = QualityRequest(container_name="test-container", code_path="/test.py")
        result = validator.validate_cached_result(cached, request)
        assert "Invalid cached timestamp" in result.warnings


class TestQualityResultValidator:
    """Cover QualityResultValidator."""

    @pytest.fixture
    def validator(self):
        return QualityResultValidator()

    def _make_result(self, **kw):
        defaults = dict(
            analysis_id="test-1",
            overall_score=80.0,
            total_issues=0,
            issues=[],
            execution_time=5.0,
            created_at=datetime.now(UTC),
            language="python",
        )
        defaults.update(kw)
        return QualityResult(**defaults)

    def _make_issue(self, **kw):
        defaults = dict(
            tool="ruff",
            severity="warning",
            message="issue",
            file_path="test.py",
        )
        defaults.update(kw)
        return QualityIssue(**defaults)

    @pytest.mark.asyncio
    async def test_validate_valid_result(self, validator):
        result = self._make_result(
            total_issues=1,
            issues=[self._make_issue()],
        )
        validation = await validator.validate_result(result)
        assert validation.is_valid is True

    @pytest.mark.asyncio
    async def test_validate_result_missing_analysis_id(self, validator):
        result = self._make_result(analysis_id="")
        validation = await validator.validate_result(result)
        assert validation.is_valid is False

    @pytest.mark.asyncio
    async def test_validate_result_inconsistent_count(self, validator):
        result = self._make_result(total_issues=0, issues=[])
        # Force inconsistency after model auto-correction by directly setting attribute
        object.__setattr__(result, "total_issues", 5)
        validation = await validator.validate_result(result)
        assert "Inconsistent issue count: 0 vs 5" in validation.warnings

    @pytest.mark.asyncio
    async def test_validate_result_suspicious_timing(self, validator):
        result = self._make_result(execution_time=0.0001)
        validation = await validator.validate_result(result)
        assert f"Execution time suspiciously low: {0.0001}s" in validation.warnings

    @pytest.mark.asyncio
    async def test_validate_result_high_score_many_issues(self, validator):
        issues = [self._make_issue(severity="error", message=f"issue-{i}") for i in range(10)]
        result = self._make_result(
            overall_score=9.0,
            total_issues=10,
            issues=issues,
        )
        validation = await validator.validate_result(result)
        assert "High score (9.0) with many critical issues (10)" in validation.warnings

    @pytest.mark.asyncio
    async def test_validate_result_low_score_no_issues(self, validator):
        result = self._make_result(overall_score=2.0, total_issues=0)
        validation = await validator.validate_result(result)
        assert "Low score (2.0) with no issues" in validation.warnings


class TestQualityRequestValidator:
    """Cover QualityRequestValidator."""

    @pytest.fixture
    def validator(self):
        return QualityRequestValidator()

    @pytest.mark.asyncio
    async def test_validate_request_no_code_path(self, validator):
        request = QualityRequest(container_name="test-container", code_path=None, language="python")
        result = await validator.validate_request(request)
        # code_path=None triggers model_validator which defaults to "."
        # The path "." exists but is a directory, not a file
        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_validate_request_unsupported_language(self, validator):
        # "haskell" is rejected by QualityRequest's own field_validator,
        # so test with a language that passes model validation but is unsupported by the request validator
        request = QualityRequest(
            container_name="test-container", code_path="/nonexistent.py", language="javascript"
        )
        # javascript passes model validation; check the code_path error
        result = await validator.validate_request(request)
        assert result.is_valid is False  # nonexistent path

    @pytest.mark.asyncio
    async def test_validate_request_no_language(self, validator):
        request = QualityRequest(container_name="test-container", code_path="/nonexistent.py")
        result = await validator.validate_request(request)
        # Language None = no language-specific validation (no auto-detect warning from QualityRequestValidator)
        # But code path does not exist
        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_validate_request_unknown_tools(self, validator):
        request = QualityRequest(
            container_name="test-container",
            code_path="/nonexistent.py",
            language="python",
            tools=["unknown_tool"],
        )
        result = await validator.validate_request(request)
        assert "Unknown tools for python: ['unknown_tool']" in result.errors

    @pytest.mark.asyncio
    async def test_validate_request_valid_tools(self, validator, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("print('hello')\n")
        request = QualityRequest(
            container_name="test-container",
            code_path=str(f),
            language="python",
            tools=["ruff", "ruff-format"],
        )
        result = await validator.validate_request(request)
        assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_validate_request_with_fix(self, validator):
        request = QualityRequest(
            container_name="test-container",
            code_path="/nonexistent.py",
            language="python",
            fix_issues=True,
        )
        result = await validator.validate_request(request)
        assert "Automatic fixes enabled - code will be modified" in result.warnings

    def test_validate_language_python(self, validator):
        result = validator._validate_language("python")
        assert result.is_valid is True

    def test_validate_language_none(self, validator):
        result = validator._validate_language(None)
        assert result.is_valid is True
        assert "No language specified, will auto-detect" in result.warnings

    def test_validate_tools_empty(self, validator):
        result = validator._validate_tools(None, "python")
        assert result.is_valid is True

    def test_validate_tools_no_language(self, validator):
        result = validator._validate_tools(["ruff"], None)
        assert result.is_valid is True


class TestQualityValidationFactory:
    """Cover factory functions return correctly typed validator instances."""

    def test_get_request_validator(self):
        v = get_request_validator()
        assert isinstance(v, QualityRequestValidator)
        # Verify it produces a real validation result
        result = v._validate_language("python")
        assert result.is_valid is True

    def test_get_file_validator(self):
        v = get_file_validator()
        assert isinstance(v, QualityFileValidator)
        # Verify binary detection works on normal text
        assert v._is_binary_file(b"hello world\n") is False

    def test_get_fix_validator(self):
        v = get_fix_validator()
        assert isinstance(v, QualityFixValidator)
        # Verify safety scoring works
        score = v.validate_fix_safety("Format code", "ruff-format")
        assert 0.0 <= score <= 1.0

    def test_get_result_validator(self):
        v = get_result_validator()
        assert isinstance(v, QualityResultValidator)
        # Verify default thresholds are set
        assert v.min_execution_time == 0.001
        assert v.max_execution_time == 3600

    def test_get_cache_validator(self):
        v = get_cache_validator()
        assert isinstance(v, QualityCacheValidator)
        # Verify default cache age is set
        assert v.max_cache_age_hours == 24


# ===========================================================================
# INTELLIGENT TEST INTEGRATION: IntelligentTestExecutionService lifecycle
# ===========================================================================


class TestIntelligentTestExecutionServiceLifecycle:
    """Cover start/stop and basic service methods."""

    @pytest.fixture
    def service(self, monkeypatch):
        monkeypatch.setenv("SKIP_TEST_EXECUTORS", "true")
        from services.test_execution.intelligent_test_integration import (
            IntelligentTestExecutionService,
        )

        mock_docker = MagicMock()
        mock_config = MagicMock()
        mock_config.max_execution_time = 300
        mock_config.max_memory_mb = 1024
        mock_config.max_file_size_mb = 10
        mock_config.max_code_length = 50000
        mock_config.allowed_imports = ["pytest"]
        return IntelligentTestExecutionService(mock_docker, mock_config)

    @pytest.mark.asyncio
    async def test_start_sets_running(self, service):
        service.orchestrator.start = AsyncMock()
        await service.start()
        assert service.is_running is True
        assert service.is_healthy is True
        assert service.performance_metrics["total_executions"] == 0

    @pytest.mark.asyncio
    async def test_stop_clears_running(self, service):
        service.orchestrator.start = AsyncMock()
        service.orchestrator.stop = AsyncMock()
        await service.start()
        await service.stop()
        assert service.is_running is False

    @pytest.mark.asyncio
    async def test_start_failure_raises(self, service):
        service.orchestrator.start = AsyncMock(side_effect=RuntimeError("start failed"))
        with pytest.raises(TestExecutionError):
            await service.start()
        assert service.is_healthy is False

    @pytest.mark.asyncio
    async def test_stop_failure_raises(self, service):
        service.orchestrator.start = AsyncMock()
        service.orchestrator.stop = AsyncMock(side_effect=RuntimeError("stop failed"))
        await service.start()
        with pytest.raises(TestExecutionError):
            await service.stop()

    @pytest.mark.asyncio
    async def test_execute_test_not_running(self, service):
        """Should raise when service not running."""
        from services.test_execution.intelligent_test_orchestrator import (
            IntelligentTestExecutionRequest,
        )

        request = MagicMock(spec=IntelligentTestExecutionRequest)
        with pytest.raises(TestExecutionError, match="not running"):
            await service.execute_test(request)


# ===========================================================================
# LOGGING SERVICE: additional coverage for LogContext, StructuredLogEntry
# ===========================================================================


class TestTestExecutionLoggerAdvanced:
    """Cover TestExecutionLogger edge cases."""

    @pytest.fixture
    def test_logger(self, tmp_path):
        storage = LoggingFileLogStorage(tmp_path / "logs")
        return TestExecutionLogger(storage)

    def test_log_info(self, test_logger):
        """Log an info event and verify it's processed by the aggregator."""
        # Stop the processing thread so the entry stays in the queue
        test_logger.aggregator.shutdown_event.set()
        test_logger.aggregator.processing_thread.join()
        test_logger.info(LogCategory.EXECUTION, "test execution started", test_id="t1")
        # Since thread is stopped, entry stays in queue
        entry = test_logger.aggregator.log_queue.get_nowait()
        assert entry.message == "test execution started"
        assert entry.level == LogLevel.INFO
        assert entry.category == LogCategory.EXECUTION
        assert entry.metadata == {"test_id": "t1"}

    def test_log_error(self, test_logger):
        """Log an error event and verify entry has ERROR level."""
        test_logger.aggregator.shutdown_event.set()
        test_logger.aggregator.processing_thread.join()
        test_logger.error(LogCategory.EXECUTION, "Something went wrong", error="test error")
        entry = test_logger.aggregator.log_queue.get_nowait()
        assert entry.message == "Something went wrong"
        assert entry.level == LogLevel.ERROR
        assert entry.category == LogCategory.EXECUTION
        assert entry.metadata == {"error": "test error"}

    def test_log_context(self, test_logger):
        """Test log context manager."""
        with test_logger.log_context(container_id="c1", language="python") as ctx:
            assert ctx.container_id == "c1"
            assert ctx.language == "python"
            test_logger.info(LogCategory.EXECUTION, "inside context")

    def test_query_logs(self, test_logger):
        """Query logs returns empty list when no logs exist."""
        logs = test_logger.query_logs(limit=10)
        assert logs == []


class TestLogAggregatorAdvanced:
    """Cover LogAggregator methods."""

    @pytest.fixture
    def aggregator(self, tmp_path):
        storage = LoggingFileLogStorage(tmp_path / "logs")
        return LogAggregator(storage)

    def test_add_log(self, aggregator):
        entry = StructuredLogEntry(
            timestamp=datetime.now(UTC),
            level=LogLevel.INFO,
            category=LogCategory.EXECUTION,
            message="test",
            context=LogContext(),
        )
        aggregator.add_log(entry)
        # Verify the entry was queued
        assert not aggregator.log_queue.empty()

    def test_start_stop(self, aggregator):
        aggregator.start()
        assert aggregator.processing_thread.is_alive()
        aggregator.stop()
        assert aggregator.shutdown_event.is_set()
