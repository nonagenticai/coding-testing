"""Tests for remaining low-coverage modules:
- log_aggregation.py (60%)
- kubernetes_test_executor.py (62%)
- resource_monitor.py (62%)
- test_monitor.py (67%)
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
# Stubs for docker/kubernetes/psutil if missing
# ---------------------------------------------------------------------------

if "docker" not in sys.modules:
    _docker_stub = types.ModuleType("docker")
    _docker_errors = types.ModuleType("docker.errors")
    _docker_errors.NotFound = type("NotFound", (Exception,), {})
    _docker_errors.APIError = type("APIError", (Exception,), {})
    _docker_errors.ContainerError = type("ContainerError", (Exception,), {})
    _docker_stub.errors = _docker_errors
    _docker_stub.from_env = MagicMock()
    sys.modules["docker"] = _docker_stub
    sys.modules["docker.errors"] = _docker_errors

# Stub kubernetes if missing
if "kubernetes" not in sys.modules:
    _k8s = types.ModuleType("kubernetes")
    _k8s_client = types.ModuleType("kubernetes.client")

    # Create necessary classes as MagicMocks
    for cls_name in [
        "BatchV1Api",
        "CoreV1Api",
        "V1Job",
        "V1ObjectMeta",
        "V1JobSpec",
        "V1PodTemplateSpec",
        "V1PodSpec",
        "V1Container",
        "V1VolumeMount",
        "V1Volume",
        "V1EmptyDirVolumeSource",
        "V1EnvVar",
        "V1ResourceRequirements",
        "V1DeleteOptions",
    ]:
        setattr(_k8s_client, cls_name, MagicMock)

    _k8s_client.ApiException = type("ApiException", (Exception,), {"status": 404})
    _k8s.client = _k8s_client

    _k8s_config = types.ModuleType("kubernetes.config")
    _k8s_config.load_incluster_config = MagicMock()
    _k8s_config.load_kube_config = MagicMock()
    _k8s.config = _k8s_config

    sys.modules["kubernetes"] = _k8s
    sys.modules["kubernetes.client"] = _k8s_client
    sys.modules["kubernetes.config"] = _k8s_config


# ===========================================================================
# LOG AGGREGATION TESTS
# ===========================================================================

from models.logging_models import LogLevel  # noqa: E402
from services.test_execution.log_aggregation import (  # noqa: E402
    DatabaseLogStorage,
    LogAggregationService,
    LogAggregationServiceHolder,
    LogQuery,
    LogQueryResult,
    LogStorageInterface,
)
from services.test_execution.logging_service import LogCategory  # noqa: E402


class TestLogQueryDataclass:
    def test_defaults(self):
        q = LogQuery()
        assert q.limit == 100
        assert q.offset == 0
        assert q.sort_order == "desc"
        assert q.levels == []
        assert q.categories == []
        assert q.languages == []
        assert q.container_ids == []
        assert q.search_text is None
        assert q.start_time is None
        assert q.end_time is None

    def test_to_dict(self):
        q = LogQuery(
            start_time=datetime(2025, 1, 1, tzinfo=UTC),
            levels=[LogLevel.ERROR],
            categories=[LogCategory.EXECUTION],
            languages=["python"],
            container_ids=["c1"],
            search_text="fail",
        )
        d = q.to_dict()
        assert d["start_time"] == "2025-01-01T00:00:00+00:00"
        assert d["end_time"] is None
        assert d["levels"] == [LogLevel.ERROR.value]
        assert d["categories"] == [LogCategory.EXECUTION.value]
        assert d["languages"] == ["python"]
        assert d["container_ids"] == ["c1"]
        assert d["search_text"] == "fail"
        assert d["limit"] == 100
        assert d["offset"] == 0
        assert d["sort_order"] == "desc"


class TestLogQueryResult:
    def test_to_dict(self):
        r = LogQueryResult(
            logs=[{"msg": "test"}],
            total_count=1,
            query_time_ms=5.0,
            has_more=False,
        )
        d = r.to_dict()
        assert d == {
            "logs": [{"msg": "test"}],
            "total_count": 1,
            "query_time_ms": 5.0,
            "has_more": False,
        }


class TestLogStorageInterface:
    @pytest.mark.asyncio
    async def test_store_log_raises(self):
        storage = LogStorageInterface()
        with pytest.raises(NotImplementedError):
            await storage.store_log(MagicMock())

    @pytest.mark.asyncio
    async def test_store_logs_batch_raises(self):
        storage = LogStorageInterface()
        with pytest.raises(NotImplementedError):
            await storage.store_logs_batch([])

    @pytest.mark.asyncio
    async def test_query_logs_raises(self):
        storage = LogStorageInterface()
        with pytest.raises(NotImplementedError):
            await storage.query_logs(LogQuery())

    @pytest.mark.asyncio
    async def test_delete_old_logs_raises(self):
        storage = LogStorageInterface()
        with pytest.raises(NotImplementedError):
            await storage.delete_old_logs(30)

    @pytest.mark.asyncio
    async def test_get_storage_stats_raises(self):
        storage = LogStorageInterface()
        with pytest.raises(NotImplementedError):
            await storage.get_storage_stats()


class TestDatabaseLogStorage:
    @pytest.fixture
    def db_storage(self, tmp_path):
        db_path = tmp_path / "test_logs.db"
        return DatabaseLogStorage(db_path)

    @pytest.fixture
    def mock_log_entry(self):
        entry = MagicMock()
        entry.timestamp = datetime.now(UTC)
        entry.level = LogLevel.INFO
        entry.category = LogCategory.EXECUTION
        entry.message = "test log"
        entry.context = MagicMock()
        entry.context.language = "python"
        entry.context.container_id = "c1"
        entry.context.execution_id = "e1"
        entry.metadata = {"key": "value"}
        entry.source_file = "test.py"
        entry.source_line = 10
        return entry

    @pytest.mark.asyncio
    async def test_store_and_query(self, db_storage, mock_log_entry):
        await db_storage.store_log(mock_log_entry)
        result = await db_storage.query_logs(LogQuery())
        assert result.total_count == 1
        assert len(result.logs) == 1
        assert result.logs[0]["message"] == "test log"
        assert result.logs[0]["level"] == LogLevel.INFO.value
        assert result.logs[0]["category"] == LogCategory.EXECUTION.value
        assert result.logs[0]["language"] == "python"
        assert result.logs[0]["container_id"] == "c1"
        assert result.logs[0]["execution_id"] == "e1"
        assert result.has_more is False

    @pytest.mark.asyncio
    async def test_store_batch(self, db_storage, mock_log_entry):
        await db_storage.store_logs_batch([mock_log_entry, mock_log_entry])
        result = await db_storage.query_logs(LogQuery())
        assert result.total_count == 2

    @pytest.mark.asyncio
    async def test_query_with_filters(self, db_storage, mock_log_entry):
        await db_storage.store_logs_batch([mock_log_entry])
        q = LogQuery(
            levels=[LogLevel.INFO],
            categories=[LogCategory.EXECUTION],
            languages=["python"],
            container_ids=["c1"],
            search_text="test",
        )
        result = await db_storage.query_logs(q)
        assert result.total_count == 1
        assert result.logs[0]["message"] == "test log"
        assert result.logs[0]["language"] == "python"
        assert result.logs[0]["container_id"] == "c1"

    @pytest.mark.asyncio
    async def test_query_with_time_range(self, db_storage, mock_log_entry):
        await db_storage.store_logs_batch([mock_log_entry])
        q = LogQuery(
            start_time=datetime.now(UTC) - timedelta(hours=1),
            end_time=datetime.now(UTC) + timedelta(hours=1),
        )
        result = await db_storage.query_logs(q)
        assert result.total_count == 1
        assert result.logs[0]["message"] == "test log"
        assert result.has_more is False

    @pytest.mark.asyncio
    async def test_query_asc_order(self, db_storage):
        """Verify ascending sort returns logs in chronological order."""
        # Create entries with different timestamps
        entry1 = MagicMock()
        entry1.timestamp = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
        entry1.level = LogLevel.INFO
        entry1.category = LogCategory.EXECUTION
        entry1.message = "first"
        entry1.context = MagicMock()
        entry1.context.language = "python"
        entry1.context.container_id = "c1"
        entry1.context.execution_id = "e1"
        entry1.metadata = {}
        entry1.source_file = "a.py"
        entry1.source_line = 1

        entry2 = MagicMock()
        entry2.timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        entry2.level = LogLevel.ERROR
        entry2.category = LogCategory.EXECUTION
        entry2.message = "second"
        entry2.context = MagicMock()
        entry2.context.language = "python"
        entry2.context.container_id = "c1"
        entry2.context.execution_id = "e2"
        entry2.metadata = {}
        entry2.source_file = "b.py"
        entry2.source_line = 2

        await db_storage.store_logs_batch([entry2, entry1])  # Insert out of order

        q_asc = LogQuery(sort_order="asc")
        result = await db_storage.query_logs(q_asc)
        assert result.total_count == 2
        assert result.logs[0]["message"] == "first"
        assert result.logs[1]["message"] == "second"

        # Verify desc order is opposite
        q_desc = LogQuery(sort_order="desc")
        result_desc = await db_storage.query_logs(q_desc)
        assert result_desc.logs[0]["message"] == "second"
        assert result_desc.logs[1]["message"] == "first"

    @pytest.mark.asyncio
    async def test_delete_old_logs(self, db_storage):
        """Verify delete_old_logs removes exactly the logs older than retention_days."""
        # Create an old entry (30 days ago)
        old_entry = MagicMock()
        old_entry.timestamp = datetime.now(UTC) - timedelta(days=30)
        old_entry.level = LogLevel.INFO
        old_entry.category = LogCategory.EXECUTION
        old_entry.message = "old log"
        old_entry.context = MagicMock()
        old_entry.context.language = "python"
        old_entry.context.container_id = "c1"
        old_entry.context.execution_id = "e1"
        old_entry.metadata = {}
        old_entry.source_file = "test.py"
        old_entry.source_line = 1

        # Create a recent entry
        new_entry = MagicMock()
        new_entry.timestamp = datetime.now(UTC)
        new_entry.level = LogLevel.ERROR
        new_entry.category = LogCategory.EXECUTION
        new_entry.message = "new log"
        new_entry.context = MagicMock()
        new_entry.context.language = "go"
        new_entry.context.container_id = "c2"
        new_entry.context.execution_id = "e2"
        new_entry.metadata = {}
        new_entry.source_file = "test.go"
        new_entry.source_line = 2

        await db_storage.store_logs_batch([old_entry, new_entry])

        # Delete logs older than 7 days — should remove old_entry only
        deleted = await db_storage.delete_old_logs(7)
        assert deleted == 1

        # Verify only the new log remains
        result = await db_storage.query_logs(LogQuery())
        assert result.total_count == 1
        assert result.logs[0]["message"] == "new log"

    @pytest.mark.asyncio
    async def test_get_storage_stats(self, db_storage, mock_log_entry):
        await db_storage.store_logs_batch([mock_log_entry])
        stats = await db_storage.get_storage_stats()
        assert stats["total_logs"] == 1
        assert stats["database_size_bytes"] > 0
        assert stats["database_size_mb"] > 0
        assert "database_path" in stats

        # Validate oldest/newest are valid ISO timestamps matching the stored entry
        stored_ts_iso = mock_log_entry.timestamp.isoformat()
        assert stats["oldest_log"] == stored_ts_iso
        assert stats["newest_log"] == stored_ts_iso
        # With a single entry, oldest == newest
        assert stats["oldest_log"] == stats["newest_log"]

        # Store a second entry with a later timestamp and verify oldest != newest
        entry2 = MagicMock()
        entry2.timestamp = mock_log_entry.timestamp + timedelta(seconds=60)
        entry2.level = LogLevel.ERROR
        entry2.category = LogCategory.EXECUTION
        entry2.message = "second log"
        entry2.context = MagicMock()
        entry2.context.language = "go"
        entry2.context.container_id = "c2"
        entry2.context.execution_id = "e2"
        entry2.metadata = {}
        entry2.source_file = "test.go"
        entry2.source_line = 5
        await db_storage.store_log(entry2)

        stats2 = await db_storage.get_storage_stats()
        assert stats2["total_logs"] == 2
        # oldest should be the original entry, newest should be entry2
        assert stats2["oldest_log"] == stored_ts_iso
        assert stats2["newest_log"] == entry2.timestamp.isoformat()
        # Verify chronological ordering: oldest < newest
        oldest_dt = datetime.fromisoformat(stats2["oldest_log"])
        newest_dt = datetime.fromisoformat(stats2["newest_log"])
        assert oldest_dt < newest_dt

    @pytest.mark.asyncio
    async def test_double_init(self, db_storage):
        """Initialize should be idempotent."""
        await db_storage._initialize()
        await db_storage._initialize()
        assert db_storage._initialized is True


class TestLogAggregationService:
    @pytest.fixture
    def db_storage(self, tmp_path):
        """Use real DatabaseLogStorage for integration-style tests."""
        db_path = tmp_path / "agg_test.db"
        return DatabaseLogStorage(db_path)

    @pytest.fixture
    def service_with_real_storage(self, db_storage):
        return LogAggregationService(
            storage_backend=db_storage,
            batch_size=5,
            flush_interval=1,
            retention_days=7,
        )

    @pytest.fixture
    def mock_log_entry(self):
        entry = MagicMock()
        entry.timestamp = datetime.now(UTC)
        entry.level = LogLevel.INFO
        entry.category = LogCategory.EXECUTION
        entry.message = "aggregation test log"
        entry.context = MagicMock()
        entry.context.language = "python"
        entry.context.container_id = "agg-c1"
        entry.context.execution_id = "agg-e1"
        entry.metadata = {"key": "value"}
        entry.source_file = "test.py"
        entry.source_line = 42
        return entry

    @pytest.mark.asyncio
    async def test_add_log_queues_entry(self, service_with_real_storage, mock_log_entry):
        """Test that add_log actually enqueues the entry."""
        await service_with_real_storage.add_log(mock_log_entry)
        assert service_with_real_storage.log_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_query_logs_returns_real_data(
        self, service_with_real_storage, db_storage, mock_log_entry
    ):
        """Test that query_logs returns data stored via the real storage backend."""
        # Store directly into the real storage
        await db_storage.store_log(mock_log_entry)
        # Query through the aggregation service
        result = await service_with_real_storage.query_logs(LogQuery())
        assert result.total_count == 1
        assert len(result.logs) == 1
        assert result.logs[0]["message"] == "aggregation test log"
        assert result.logs[0]["language"] == "python"
        assert result.logs[0]["container_id"] == "agg-c1"

    @pytest.mark.asyncio
    async def test_get_storage_stats_returns_real_stats(
        self, service_with_real_storage, db_storage, mock_log_entry
    ):
        """Test that storage stats reflect actual data via the aggregation service."""
        await db_storage.store_log(mock_log_entry)
        stats = await service_with_real_storage.get_storage_stats()
        assert stats["total_logs"] == 1
        assert stats["database_size_bytes"] > 0

        # Validate that oldest/newest timestamps are valid ISO strings matching the entry
        stored_ts_iso = mock_log_entry.timestamp.isoformat()
        assert stats["oldest_log"] == stored_ts_iso
        assert stats["newest_log"] == stored_ts_iso

        # Verify they parse back to the correct datetime
        parsed_oldest = datetime.fromisoformat(stats["oldest_log"])
        assert parsed_oldest == mock_log_entry.timestamp

    @pytest.mark.asyncio
    async def test_start_and_stop(self, service_with_real_storage, mock_log_entry):
        """Test service lifecycle: start creates running tasks, logs get flushed, stop shuts down."""
        # Before start, tasks should be None
        assert service_with_real_storage.processing_task is None
        assert service_with_real_storage.cleanup_task is None
        assert not service_with_real_storage.shutdown_event.is_set()

        await service_with_real_storage.start()

        # After start, tasks must be actual asyncio.Task instances that are running
        assert isinstance(service_with_real_storage.processing_task, asyncio.Task)
        assert isinstance(service_with_real_storage.cleanup_task, asyncio.Task)
        assert not service_with_real_storage.processing_task.done()
        assert not service_with_real_storage.cleanup_task.done()

        # Add a log and verify it actually gets flushed to storage
        await service_with_real_storage.add_log(mock_log_entry)
        # flush_interval=1s for this fixture, wait enough for a flush cycle
        await asyncio.sleep(1.5)

        # The log should have been flushed from the queue to the real storage
        result = await service_with_real_storage.query_logs(LogQuery())
        assert result.total_count == 1
        assert result.logs[0]["message"] == "aggregation test log"

        await service_with_real_storage.stop()
        assert service_with_real_storage.shutdown_event.is_set()
        # After stop, the processing task should have finished
        assert service_with_real_storage.processing_task.done()


class TestLogAggregationServiceHolder:
    @pytest.mark.asyncio
    async def test_reset_instance(self):
        # Ensure clean state
        await LogAggregationServiceHolder.reset_instance()
        assert LogAggregationServiceHolder._instance is None
        assert LogAggregationServiceHolder._initialized is False

    @pytest.mark.asyncio
    async def test_set_instance(self):
        mock_storage = AsyncMock(spec=LogStorageInterface)
        mock_storage.store_logs_batch = AsyncMock()
        mock_storage.delete_old_logs = AsyncMock(return_value=0)
        mock_storage.query_logs = AsyncMock(
            return_value=LogQueryResult(logs=[], total_count=0, query_time_ms=0.0, has_more=False)
        )

        svc = await LogAggregationServiceHolder.set_instance(mock_storage)
        assert svc.storage is mock_storage
        assert LogAggregationServiceHolder._initialized is True
        assert LogAggregationServiceHolder._instance is svc

        # Verify the service is actually started and functional (tasks running)
        assert isinstance(svc.processing_task, asyncio.Task)
        assert not svc.processing_task.done()

        # Verify functional behavior: add a log entry and query
        mock_entry = MagicMock()
        mock_entry.timestamp = datetime.now(UTC)
        mock_entry.level = LogLevel.INFO
        mock_entry.category = LogCategory.EXECUTION
        mock_entry.message = "holder test"
        mock_entry.context = MagicMock()
        mock_entry.context.language = "python"
        mock_entry.context.container_id = "h1"
        mock_entry.context.execution_id = "he1"
        mock_entry.metadata = {}
        mock_entry.source_file = "test.py"
        mock_entry.source_line = 1

        await svc.add_log(mock_entry)
        assert svc.log_queue.qsize() == 1

        # Query delegates to storage backend
        result = await svc.query_logs(LogQuery())
        mock_storage.query_logs.assert_called_once()
        assert result.total_count == 0  # mock returns empty

        # Clean up
        await LogAggregationServiceHolder.reset_instance()


# ===========================================================================
# KUBERNETES TEST EXECUTOR TESTS
# ===========================================================================

from services.test_execution.kubernetes_test_executor import (  # noqa: E402
    KubernetesTestExecutor,
)


class TestKubernetesTestExecutor:
    @pytest.fixture
    def executor(self):
        with patch("services.test_execution.kubernetes_test_executor.config") as mock_config:
            mock_config.load_kube_config = MagicMock()
            executor = KubernetesTestExecutor(namespace="test-ns", in_cluster=False)
            executor.batch = MagicMock()
            executor.core = MagicMock()
            return executor

    @pytest.mark.parametrize(
        "language, expected_ext",
        [
            ("python", ".py"),
            ("go", ".go"),
            ("rust", ".rs"),
            ("java", ".java"),
            ("csharp", ".cs"),
            ("cpp", ".cpp"),
            ("javascript", ".js"),
            ("typescript", ".ts"),
            ("haskell", ".txt"),  # unknown language falls back to .txt
        ],
    )
    def test_extension(self, executor, language, expected_ext):
        assert executor._extension(language) == expected_ext

    def test_write_file_cmd(self, executor):
        import base64

        cmd = executor._write_file_cmd("test.py", "print('hello')")
        expected_b64 = base64.b64encode(b"print('hello')").decode()
        assert cmd == f"echo {expected_b64} | base64 -d > /workspace/test.py"

    def test_build_init_script(self, executor):
        import base64

        request = MagicMock()
        request.source_code = "def add(a,b): return a+b"
        request.test_files = {"test_add.py": "assert True"}
        request.language = MagicMock()
        request.language.value = "python"
        request.dependencies = {}

        script = executor._build_init_script(request)
        lines = script.split("\n")
        assert lines[0] == "set -e"
        # Should write source code as main.py (python -> .py extension)
        source_b64 = base64.b64encode(b"def add(a,b): return a+b").decode()
        assert f"echo {source_b64} | base64 -d > /workspace/main.py" in lines[1]
        # Should write test file
        test_b64 = base64.b64encode(b"assert True").decode()
        assert f"echo {test_b64} | base64 -d > /workspace/test_add.py" in lines[2]

    def test_build_init_script_no_source(self, executor):
        import base64

        request = MagicMock()
        request.source_code = None
        request.test_files = {"test.py": "assert True"}
        request.language = MagicMock()
        request.language.value = "python"
        request.dependencies = {}

        script = executor._build_init_script(request)
        lines = script.split("\n")
        assert lines[0] == "set -e"
        # No source code, so should only have set -e + the test file
        assert len(lines) == 2
        test_b64 = base64.b64encode(b"assert True").decode()
        assert f"echo {test_b64} | base64 -d > /workspace/test.py" in lines[1]

    def test_format_go_dependencies_empty(self, executor):
        assert executor._format_go_dependencies({}) == ""

    def test_format_go_dependencies(self, executor):
        result = executor._format_go_dependencies({"github.com/pkg/errors": "v0.9.1"})
        assert result == "require github.com/pkg/errors v0.9.1"

    def test_format_rust_dependencies_empty(self, executor):
        assert executor._format_rust_dependencies({}) == ""

    def test_format_rust_dependencies(self, executor):
        result = executor._format_rust_dependencies({"serde": "1.0"})
        assert 'serde = "1.0"' in result

    def test_go_mod_content(self, executor):
        result = executor._go_mod_content({"github.com/pkg/errors": "v0.9.1"})
        assert "module test-module" in result
        assert "go 1.21" in result

    def test_cargo_toml_content(self, executor):
        result = executor._cargo_toml_content({"serde": "1.0"})
        assert "[package]" in result
        assert 'name = "test-crate"' in result

    def test_package_json_content_javascript(self, executor):
        request = MagicMock()
        request.language = MagicMock()
        request.language.value = "javascript"
        request.dependencies = {}
        result = executor._package_json_content(request)
        data = json.loads(result)
        assert data["scripts"]["test"] == "jest"

    def test_package_json_content_typescript(self, executor):
        request = MagicMock()
        request.language = MagicMock()
        request.language.value = "typescript"
        request.dependencies = {}
        result = executor._package_json_content(request)
        data = json.loads(result)
        assert "ts-jest" in data["devDependencies"]

    def test_language_config_files_go(self, executor):
        request = MagicMock()
        request.language = MagicMock()
        request.language.value = "go"
        request.dependencies = {}
        files = executor._language_config_files(request)
        assert any("go.mod" in f[0] for f in files)

    def test_language_config_files_rust(self, executor):
        request = MagicMock()
        request.language = MagicMock()
        request.language.value = "rust"
        request.dependencies = {}
        files = executor._language_config_files(request)
        assert any("Cargo.toml" in f[0] for f in files)

    def test_language_config_files_js(self, executor):
        request = MagicMock()
        request.language = MagicMock()
        request.language.value = "javascript"
        request.dependencies = {}
        files = executor._language_config_files(request)
        assert any("package.json" in f[0] for f in files)

    def test_language_config_files_python(self, executor):
        request = MagicMock()
        request.language = MagicMock()
        request.language.value = "python"
        request.dependencies = {}
        files = executor._language_config_files(request)
        assert files == []

    @pytest.mark.asyncio
    async def test_wait_for_pod_timeout(self, executor):
        executor.pod_start_timeout = 0.1
        executor.core.list_namespaced_pod = MagicMock(return_value=MagicMock(items=[]))

        with pytest.raises(TimeoutError):
            await executor._wait_for_pod("job-1")

    @pytest.mark.asyncio
    async def test_wait_for_pod_success(self, executor):
        executor.pod_start_timeout = 5
        pod = MagicMock()
        pod.metadata.name = "pod-1"
        executor.core.list_namespaced_pod = MagicMock(return_value=MagicMock(items=[pod]))

        result = await executor._wait_for_pod("job-1")
        assert result == "pod-1"

    @pytest.mark.asyncio
    async def test_wait_for_completion_timeout(self, executor):
        executor.pod_completion_timeout = 0.1
        pod_status = MagicMock()
        pod_status.status.container_statuses = [MagicMock()]
        pod_status.status.container_statuses[0].state.terminated = None
        executor.core.read_namespaced_pod_status = MagicMock(return_value=pod_status)

        with pytest.raises(TimeoutError):
            await executor._wait_for_completion("pod-1")

    @pytest.mark.asyncio
    async def test_wait_for_completion_success(self, executor):
        executor.pod_completion_timeout = 5
        pod_status = MagicMock()
        terminated = MagicMock()
        terminated.exit_code = 42  # non-zero so `or 1` doesn't kick in
        pod_status.status.container_statuses = [MagicMock()]
        pod_status.status.container_statuses[0].state.terminated = terminated
        executor.core.read_namespaced_pod_status = MagicMock(return_value=pod_status)

        result = await executor._wait_for_completion("pod-1")
        assert result == 42

    @pytest.mark.asyncio
    async def test_wait_for_completion_zero_exit_code(self, executor):
        """When exit_code is 0, it should return 0 (successful exit)."""
        executor.pod_completion_timeout = 5
        pod_status = MagicMock()
        terminated = MagicMock()
        terminated.exit_code = 0
        pod_status.status.container_statuses = [MagicMock()]
        pod_status.status.container_statuses[0].state.terminated = terminated
        executor.core.read_namespaced_pod_status = MagicMock(return_value=pod_status)

        result = await executor._wait_for_completion("pod-1")
        assert result == 0

    @pytest.mark.asyncio
    async def test_run_full_lifecycle(self, executor):
        """Test the full run() method end-to-end with mocked K8s APIs."""
        request = MagicMock()
        request.source_code = "def add(a,b): return a+b"
        request.test_files = {"test_add.py": "assert True"}
        request.language = MagicMock()
        request.language.value = "python"
        request.dependencies = {}

        container_spec = MagicMock(spec=[])
        container_spec.image = "python:3.13-slim"
        container_spec.environment = {"PYTHONPATH": "/workspace"}
        container_spec.cpu_quota = "500m"
        container_spec.memory_limit = "256Mi"

        test_command = ["python", "-m", "pytest", "test_add.py"]

        # Mock K8s API responses
        executor.batch.create_namespaced_job = MagicMock()

        # Mock pod listing
        pod_mock = MagicMock()
        pod_mock.metadata.name = "test-pod-123"
        executor.core.list_namespaced_pod = MagicMock(return_value=MagicMock(items=[pod_mock]))

        # Mock pod completion
        terminated = MagicMock()
        terminated.exit_code = 0
        pod_status = MagicMock()
        pod_status.status.container_statuses = [MagicMock()]
        pod_status.status.container_statuses[0].state.terminated = terminated
        executor.core.read_namespaced_pod_status = MagicMock(return_value=pod_status)

        # Mock log retrieval
        executor.core.read_namespaced_pod_log = MagicMock(return_value="1 passed")

        # Mock job deletion
        executor.batch.delete_namespaced_job = MagicMock()

        executor.pod_start_timeout = 5
        executor.pod_completion_timeout = 5

        # Patch the K8s client classes to avoid spec validation issues
        with patch("services.test_execution.kubernetes_test_executor.client") as mock_k8s_client:
            mock_k8s_client.V1Job = MagicMock()
            mock_k8s_client.V1ObjectMeta = MagicMock()
            mock_k8s_client.V1JobSpec = MagicMock()
            mock_k8s_client.V1PodTemplateSpec = MagicMock()
            mock_k8s_client.V1PodSpec = MagicMock()
            mock_k8s_client.V1Container = MagicMock()
            mock_k8s_client.V1VolumeMount = MagicMock()
            mock_k8s_client.V1EnvVar = MagicMock()
            mock_k8s_client.V1ResourceRequirements = MagicMock()
            mock_k8s_client.V1Volume = MagicMock()
            mock_k8s_client.V1EmptyDirVolumeSource = MagicMock()
            mock_k8s_client.V1DeleteOptions = MagicMock()

            result = await executor.run(request, container_spec, test_command)

        assert result["exit_code"] == 0
        assert result["stdout"] == "1 passed"
        assert result["stderr"] == ""
        assert result["pod_name"] == "test-pod-123"
        assert result["job_name"].startswith("test-exec-python-")

    @pytest.mark.asyncio
    async def test_run_cleanup_on_404(self, executor):
        """Job cleanup ignores 404 errors."""
        from kubernetes.client import ApiException

        request = MagicMock()
        request.source_code = "x = 1"
        request.test_files = {}
        request.language = MagicMock()
        request.language.value = "python"
        request.dependencies = {}

        container_spec = MagicMock(spec=[])
        container_spec.image = "python:3.13-slim"
        container_spec.environment = {}
        container_spec.cpu_quota = "500m"
        container_spec.memory_limit = "256Mi"

        executor.batch.create_namespaced_job = MagicMock()

        pod_mock = MagicMock()
        pod_mock.metadata.name = "test-pod-456"
        executor.core.list_namespaced_pod = MagicMock(return_value=MagicMock(items=[pod_mock]))

        terminated = MagicMock()
        terminated.exit_code = 1
        pod_status = MagicMock()
        pod_status.status.container_statuses = [MagicMock()]
        pod_status.status.container_statuses[0].state.terminated = terminated
        executor.core.read_namespaced_pod_status = MagicMock(return_value=pod_status)
        executor.core.read_namespaced_pod_log = MagicMock(return_value="FAILED")

        # Simulate 404 on delete — should be silently ignored
        api_exc = ApiException("Not Found")
        api_exc.status = 404
        executor.batch.delete_namespaced_job = MagicMock(side_effect=api_exc)

        executor.pod_start_timeout = 5
        executor.pod_completion_timeout = 5

        with patch("services.test_execution.kubernetes_test_executor.client") as mock_k8s_client:
            mock_k8s_client.V1Job = MagicMock()
            mock_k8s_client.V1ObjectMeta = MagicMock()
            mock_k8s_client.V1JobSpec = MagicMock()
            mock_k8s_client.V1PodTemplateSpec = MagicMock()
            mock_k8s_client.V1PodSpec = MagicMock()
            mock_k8s_client.V1Container = MagicMock()
            mock_k8s_client.V1VolumeMount = MagicMock()
            mock_k8s_client.V1EnvVar = MagicMock()
            mock_k8s_client.V1ResourceRequirements = MagicMock()
            mock_k8s_client.V1Volume = MagicMock()
            mock_k8s_client.V1EmptyDirVolumeSource = MagicMock()
            mock_k8s_client.V1DeleteOptions = MagicMock()
            mock_k8s_client.ApiException = ApiException

            result = await executor.run(request, container_spec, ["python", "-m", "pytest"])

        assert result["exit_code"] == 1
        assert result["stderr"] == "FAILED"

    @pytest.mark.asyncio
    async def test_run_cleanup_non_404_logs_warning(self, executor):
        """Job cleanup with non-404 ApiException logs warning."""
        from kubernetes.client import ApiException

        request = MagicMock()
        request.source_code = "x = 1"
        request.test_files = {}
        request.language = MagicMock()
        request.language.value = "python"
        request.dependencies = {}

        container_spec = MagicMock(spec=[])
        container_spec.image = "python:3.13-slim"
        container_spec.environment = {}
        container_spec.cpu_quota = "500m"
        container_spec.memory_limit = "256Mi"

        executor.batch.create_namespaced_job = MagicMock()

        pod_mock = MagicMock()
        pod_mock.metadata.name = "test-pod-789"
        executor.core.list_namespaced_pod = MagicMock(return_value=MagicMock(items=[pod_mock]))

        terminated = MagicMock()
        terminated.exit_code = 0
        pod_status = MagicMock()
        pod_status.status.container_statuses = [MagicMock()]
        pod_status.status.container_statuses[0].state.terminated = terminated
        executor.core.read_namespaced_pod_status = MagicMock(return_value=pod_status)
        executor.core.read_namespaced_pod_log = MagicMock(return_value="passed")

        # Simulate 500 on delete — should log warning but not crash
        api_exc = ApiException("Server Error")
        api_exc.status = 500
        executor.batch.delete_namespaced_job = MagicMock(side_effect=api_exc)

        executor.pod_start_timeout = 5
        executor.pod_completion_timeout = 5

        with patch("services.test_execution.kubernetes_test_executor.client") as mock_k8s_client:
            mock_k8s_client.V1Job = MagicMock()
            mock_k8s_client.V1ObjectMeta = MagicMock()
            mock_k8s_client.V1JobSpec = MagicMock()
            mock_k8s_client.V1PodTemplateSpec = MagicMock()
            mock_k8s_client.V1PodSpec = MagicMock()
            mock_k8s_client.V1Container = MagicMock()
            mock_k8s_client.V1VolumeMount = MagicMock()
            mock_k8s_client.V1EnvVar = MagicMock()
            mock_k8s_client.V1ResourceRequirements = MagicMock()
            mock_k8s_client.V1Volume = MagicMock()
            mock_k8s_client.V1EmptyDirVolumeSource = MagicMock()
            mock_k8s_client.V1DeleteOptions = MagicMock()
            mock_k8s_client.ApiException = ApiException

            result = await executor.run(request, container_spec, ["python", "-m", "pytest"])

        assert result["exit_code"] == 0
        assert result["stdout"] == "passed"

    def test_init_failure(self):
        """When K8s config loading fails, exception propagates (lines 52-54)."""
        with patch("services.test_execution.kubernetes_test_executor.config") as mock_config:
            mock_config.load_kube_config = MagicMock(side_effect=RuntimeError("no kubeconfig"))
            with pytest.raises(RuntimeError, match="no kubeconfig"):
                KubernetesTestExecutor(namespace="test-ns", in_cluster=False)

    def test_init_in_cluster(self):
        """In-cluster config loads successfully."""
        with patch("services.test_execution.kubernetes_test_executor.config") as mock_config:
            mock_config.load_incluster_config = MagicMock()
            executor = KubernetesTestExecutor(namespace="test-ns", in_cluster=True)
            mock_config.load_incluster_config.assert_called_once()
            assert executor.namespace == "test-ns"

    @pytest.mark.asyncio
    async def test_wait_for_completion_none_exit_code(self, executor):
        """When exit_code is None, returns 1."""
        executor.pod_completion_timeout = 5
        pod_status = MagicMock()
        terminated = MagicMock()
        terminated.exit_code = None
        pod_status.status.container_statuses = [MagicMock()]
        pod_status.status.container_statuses[0].state.terminated = terminated
        executor.core.read_namespaced_pod_status = MagicMock(return_value=pod_status)

        result = await executor._wait_for_completion("pod-1")
        assert result == 1


# ===========================================================================
# RESOURCE MONITOR TESTS
# ===========================================================================

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
        t = ResourceThreshold(
            resource_type=ResourceType.CPU,
            warning_threshold=70,
            critical_threshold=85,
            emergency_threshold=95,
            unit="percent",
        )
        assert t.get_alert_level(96) == AlertLevel.EMERGENCY

    def test_get_alert_level_critical(self):
        t = ResourceThreshold(
            resource_type=ResourceType.CPU,
            warning_threshold=70,
            critical_threshold=85,
            emergency_threshold=95,
            unit="percent",
        )
        assert t.get_alert_level(90) == AlertLevel.CRITICAL

    def test_get_alert_level_warning(self):
        t = ResourceThreshold(
            resource_type=ResourceType.CPU,
            warning_threshold=70,
            critical_threshold=85,
            emergency_threshold=95,
            unit="percent",
        )
        assert t.get_alert_level(75) == AlertLevel.WARNING

    def test_get_alert_level_none(self):
        t = ResourceThreshold(
            resource_type=ResourceType.CPU,
            warning_threshold=70,
            critical_threshold=85,
            emergency_threshold=95,
            unit="percent",
        )
        assert t.get_alert_level(50) is None


class TestResourceUsage:
    def test_to_dict(self):
        ts = datetime.now(UTC)
        ru = ResourceUsage(
            timestamp=ts,
            resource_type=ResourceType.CPU,
            current_value=50.0,
            max_value=100.0,
            usage_percent=50.0,
            unit="percent",
        )
        d = ru.to_dict()
        assert d["timestamp"] == ts.isoformat()
        assert d["resource_type"] == "cpu"
        assert d["current_value"] == 50.0
        assert d["max_value"] == 100.0
        assert d["usage_percent"] == 50.0
        assert d["unit"] == "percent"
        assert d["metadata"] == {}


class TestResourceAlert:
    def test_to_dict(self):
        ts = datetime.now(UTC)
        ra = ResourceAlert(
            timestamp=ts,
            resource_type=ResourceType.MEMORY,
            alert_level=AlertLevel.WARNING,
            current_value=85.0,
            threshold_value=80.0,
            usage_percent=85.0,
            message="High memory",
        )
        d = ra.to_dict()
        assert d["timestamp"] == ts.isoformat()
        assert d["resource_type"] == "memory"
        assert d["alert_level"] == "warning"
        assert d["current_value"] == 85.0
        assert d["threshold_value"] == 80.0
        assert d["usage_percent"] == 85.0
        assert d["message"] == "High memory"
        assert d["metadata"] == {}
        assert d["acknowledged"] is False


class TestAlertManager:
    @pytest.fixture
    def manager(self):
        return AlertManager()

    def test_register_handler(self, manager):
        handler = MagicMock()
        manager.register_alert_handler(AlertAction.LOG, handler)
        assert manager.alert_handlers[AlertAction.LOG] is handler

    def test_create_alert(self, manager):
        usage = ResourceUsage(
            timestamp=datetime.now(UTC),
            resource_type=ResourceType.CPU,
            current_value=90.0,
            max_value=100.0,
            usage_percent=90.0,
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
        assert alert.current_value == 90.0
        assert alert.threshold_value == 85  # critical threshold
        assert alert.usage_percent == 90.0
        assert alert.acknowledged is False
        assert "CPU usage CRITICAL" in alert.message
        assert "90.0 percent" in alert.message
        assert "90.0%" in alert.message

    def test_process_alert(self, manager):
        handler = MagicMock()
        manager.register_alert_handler(AlertAction.LOG, handler)

        alert = ResourceAlert(
            timestamp=datetime.now(UTC),
            resource_type=ResourceType.CPU,
            alert_level=AlertLevel.WARNING,
            current_value=75.0,
            threshold_value=70.0,
            usage_percent=75.0,
            message="High CPU",
        )
        manager.process_alert(alert)
        handler.assert_called_once_with(alert)
        # Verify alert was stored in active_alerts and history
        assert "cpu_warning" in manager.active_alerts
        assert manager.active_alerts["cpu_warning"] is alert
        assert len(manager.alert_history) == 1
        assert manager.alert_history[0] is alert

    def test_process_alert_suppressed(self, manager):
        handler = MagicMock()
        manager.register_alert_handler(AlertAction.LOG, handler)

        alert = ResourceAlert(
            timestamp=datetime.now(UTC),
            resource_type=ResourceType.CPU,
            alert_level=AlertLevel.WARNING,
            current_value=75.0,
            threshold_value=70.0,
            usage_percent=75.0,
            message="High CPU",
        )
        manager.suppress_alert("cpu_warning", duration_minutes=60)
        manager.process_alert(alert)
        handler.assert_not_called()

    def test_acknowledge_alert(self, manager):
        alert = ResourceAlert(
            timestamp=datetime.now(UTC),
            resource_type=ResourceType.CPU,
            alert_level=AlertLevel.WARNING,
            current_value=75.0,
            threshold_value=70.0,
            usage_percent=75.0,
            message="High CPU",
        )
        manager.active_alerts["cpu_warning"] = alert
        assert manager.acknowledge_alert("cpu_warning") is True
        assert alert.acknowledged is True

    def test_acknowledge_nonexistent(self, manager):
        assert manager.acknowledge_alert("nonexistent") is False

    def test_get_active_alerts(self, manager):
        alert = ResourceAlert(
            timestamp=datetime.now(UTC),
            resource_type=ResourceType.CPU,
            alert_level=AlertLevel.WARNING,
            current_value=75.0,
            threshold_value=70.0,
            usage_percent=75.0,
            message="High CPU",
        )
        manager.active_alerts["cpu_warning"] = alert
        assert len(manager.get_active_alerts()) == 1

    def test_get_alert_history(self, manager):
        alert = ResourceAlert(
            timestamp=datetime.now(UTC),
            resource_type=ResourceType.CPU,
            alert_level=AlertLevel.WARNING,
            current_value=75.0,
            threshold_value=70.0,
            usage_percent=75.0,
            message="High CPU",
        )
        manager.alert_history.append(alert)
        history = manager.get_alert_history(hours=24)
        assert len(history) == 1

    def test_get_alert_history_filtered(self, manager):
        old_alert = ResourceAlert(
            timestamp=datetime.now(UTC) - timedelta(hours=48),
            resource_type=ResourceType.CPU,
            alert_level=AlertLevel.WARNING,
            current_value=75.0,
            threshold_value=70.0,
            usage_percent=75.0,
            message="Old",
        )
        manager.alert_history.append(old_alert)
        history = manager.get_alert_history(hours=24)
        assert len(history) == 0

    def test_get_actions_for_warning(self, manager):
        alert = ResourceAlert(
            timestamp=datetime.now(UTC),
            resource_type=ResourceType.CPU,
            alert_level=AlertLevel.WARNING,
            current_value=75.0,
            threshold_value=70.0,
            usage_percent=75.0,
            message="Warning",
        )
        actions = manager._get_actions_for_alert(alert)
        assert actions == [AlertAction.LOG, AlertAction.EMAIL]

    def test_get_actions_for_critical(self, manager):
        alert = ResourceAlert(
            timestamp=datetime.now(UTC),
            resource_type=ResourceType.CPU,
            alert_level=AlertLevel.CRITICAL,
            current_value=90.0,
            threshold_value=85.0,
            usage_percent=90.0,
            message="Critical",
        )
        actions = manager._get_actions_for_alert(alert)
        assert actions == [AlertAction.LOG, AlertAction.EMAIL, AlertAction.SLACK]

    def test_get_actions_for_emergency_cpu(self, manager):
        alert = ResourceAlert(
            timestamp=datetime.now(UTC),
            resource_type=ResourceType.CPU,
            alert_level=AlertLevel.EMERGENCY,
            current_value=98.0,
            threshold_value=95.0,
            usage_percent=98.0,
            message="Emergency",
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

    def test_get_actions_for_emergency_disk(self, manager):
        alert = ResourceAlert(
            timestamp=datetime.now(UTC),
            resource_type=ResourceType.DISK,
            alert_level=AlertLevel.EMERGENCY,
            current_value=98.0,
            threshold_value=95.0,
            usage_percent=98.0,
            message="Emergency",
        )
        actions = manager._get_actions_for_alert(alert)
        # Disk emergency does NOT include TERMINATE_CONTAINERS (only CPU/MEMORY)
        assert actions == [
            AlertAction.LOG,
            AlertAction.EMAIL,
            AlertAction.SLACK,
            AlertAction.WEBHOOK,
            AlertAction.SCALE_DOWN,
        ]

    def test_alert_history_limit(self, manager):
        for i in range(1100):
            alert = ResourceAlert(
                timestamp=datetime.now(UTC),
                resource_type=ResourceType.CPU,
                alert_level=AlertLevel.WARNING,
                current_value=75.0,
                threshold_value=70.0,
                usage_percent=75.0,
                message=f"Alert {i}",
            )
            manager.process_alert(alert)
        assert len(manager.alert_history) <= 1000

    def test_handler_exception_caught(self, manager):
        handler = MagicMock(side_effect=RuntimeError("fail"))
        manager.register_alert_handler(AlertAction.LOG, handler)

        alert = ResourceAlert(
            timestamp=datetime.now(UTC),
            resource_type=ResourceType.CPU,
            alert_level=AlertLevel.WARNING,
            current_value=75.0,
            threshold_value=70.0,
            usage_percent=75.0,
            message="Test",
        )
        # Should not raise
        manager.process_alert(alert)


class TestResourceMonitor:
    def test_default_thresholds(self):
        mon = ResourceMonitor()
        assert ResourceType.CPU in mon.thresholds
        assert ResourceType.MEMORY in mon.thresholds
        assert ResourceType.DISK in mon.thresholds
        assert ResourceType.CONTAINER_COUNT in mon.thresholds
        # Verify actual threshold values for CPU
        cpu_t = mon.thresholds[ResourceType.CPU]
        assert cpu_t.warning_threshold == 70.0
        assert cpu_t.critical_threshold == 85.0
        assert cpu_t.emergency_threshold == 95.0
        assert cpu_t.unit == "percent"

    def test_set_threshold(self):
        mon = ResourceMonitor()
        new_threshold = ResourceThreshold(
            resource_type=ResourceType.CPU,
            warning_threshold=60,
            critical_threshold=75,
            emergency_threshold=90,
            unit="percent",
        )
        mon.set_threshold(ResourceType.CPU, new_threshold)
        assert mon.thresholds[ResourceType.CPU].warning_threshold == 60

    def test_log_alert_handler(self):
        mon = ResourceMonitor()
        alert = ResourceAlert(
            timestamp=datetime.now(UTC),
            resource_type=ResourceType.CPU,
            alert_level=AlertLevel.WARNING,
            current_value=75.0,
            threshold_value=70.0,
            usage_percent=75.0,
            message="Test alert",
        )
        # Should not raise
        mon._log_alert_handler(alert)

    def test_scale_down_handler(self):
        mon = ResourceMonitor()
        alert = ResourceAlert(
            timestamp=datetime.now(UTC),
            resource_type=ResourceType.CPU,
            alert_level=AlertLevel.EMERGENCY,
            current_value=98.0,
            threshold_value=95.0,
            usage_percent=98.0,
            message="Emergency",
        )
        # Should not raise
        mon._scale_down_handler(alert)

    def test_get_resource_history_empty(self):
        mon = ResourceMonitor()
        history = mon.get_resource_history(hours=24)
        assert history == []

    def test_export_data(self):
        mon = ResourceMonitor()
        data_str = mon.export_data(hours=24)
        data = json.loads(data_str)
        assert data["resource_history"] == []
        assert data["alert_history"] == []
        assert data["active_alerts"] == []
        assert ResourceType.CPU.value in data["thresholds"]
        assert ResourceType.MEMORY.value in data["thresholds"]
        assert ResourceType.DISK.value in data["thresholds"]


# ===========================================================================
# TEST MONITOR TESTS
# ===========================================================================

from services.testing.test_monitor import TestMonitor  # noqa: E402


class TestTestMonitorHealthCheck:
    @pytest.mark.asyncio
    async def test_all_healthy(self):
        monitor = TestMonitor()
        container_mgr = AsyncMock()
        container_mgr.health_check = AsyncMock(return_value={"status": "healthy"})
        config_mgr = MagicMock()
        config_mgr.validate_configuration.return_value = {
            "valid": True,
            "details": {"all": "ok"},
        }
        result_proc = MagicMock()
        result = await monitor.health_check(container_mgr, config_mgr, result_proc)
        assert result["status"] == "healthy"
        assert result["service"] == "test_execution"
        assert result["checks"]["containers"] == {"status": "healthy"}
        assert result["checks"]["cache"] == {"status": "disabled"}
        assert result["checks"]["config"]["status"] == "healthy"
        assert result["checks"]["config"]["details"] == {"all": "ok"}
        assert result["checks"]["intelligent_analyzer"] == {"status": "disabled"}
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_container_unhealthy(self):
        monitor = TestMonitor()
        container_mgr = AsyncMock()
        container_mgr.health_check = AsyncMock(side_effect=RuntimeError("docker down"))
        config_mgr = MagicMock()
        config_mgr.validate_configuration.return_value = {
            "valid": True,
            "details": {},
        }
        result_proc = MagicMock()
        result = await monitor.health_check(container_mgr, config_mgr, result_proc)
        assert result["checks"]["containers"]["status"] == "unhealthy"
        assert result["checks"]["containers"]["error"] == "docker down"
        # Only containers unhealthy (1 out of 4 checks) -> degraded
        assert result["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_with_cache(self):
        cache = AsyncMock()
        cache.get = AsyncMock(return_value=None)
        monitor = TestMonitor(cache_manager=cache)
        container_mgr = AsyncMock()
        container_mgr.health_check = AsyncMock(return_value={"status": "healthy"})
        config_mgr = MagicMock()
        config_mgr.validate_configuration.return_value = {
            "valid": True,
            "details": {},
        }
        result = await monitor.health_check(container_mgr, config_mgr, MagicMock())
        assert result["checks"]["cache"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_cache_unhealthy(self):
        cache = AsyncMock()
        cache.get = AsyncMock(side_effect=RuntimeError("redis down"))
        monitor = TestMonitor(cache_manager=cache)
        container_mgr = AsyncMock()
        container_mgr.health_check = AsyncMock(return_value={"status": "healthy"})
        config_mgr = MagicMock()
        config_mgr.validate_configuration.return_value = {
            "valid": True,
            "details": {},
        }
        result = await monitor.health_check(container_mgr, config_mgr, MagicMock())
        assert result["checks"]["cache"]["status"] == "unhealthy"
        assert result["checks"]["cache"]["error"] == "redis down"
        assert result["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_config_unhealthy(self):
        monitor = TestMonitor()
        container_mgr = AsyncMock()
        container_mgr.health_check = AsyncMock(return_value={"status": "healthy"})
        config_mgr = MagicMock()
        config_mgr.validate_configuration.side_effect = RuntimeError("bad config")
        result = await monitor.health_check(container_mgr, config_mgr, MagicMock())
        assert result["checks"]["config"]["status"] == "unhealthy"
        assert result["checks"]["config"]["error"] == "bad config"
        assert result["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_with_analyzer(self):
        monitor = TestMonitor()
        container_mgr = AsyncMock()
        container_mgr.health_check = AsyncMock(return_value={"status": "healthy"})
        config_mgr = MagicMock()
        config_mgr.validate_configuration.return_value = {
            "valid": True,
            "details": {},
        }
        analyzer = AsyncMock()
        analyzer.health_check = AsyncMock(return_value={"status": "healthy"})
        result = await monitor.health_check(
            container_mgr, config_mgr, MagicMock(), intelligent_analyzer=analyzer
        )
        assert result["checks"]["intelligent_analyzer"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_analyzer_unhealthy(self):
        monitor = TestMonitor()
        container_mgr = AsyncMock()
        container_mgr.health_check = AsyncMock(return_value={"status": "healthy"})
        config_mgr = MagicMock()
        config_mgr.validate_configuration.return_value = {
            "valid": True,
            "details": {},
        }
        analyzer = AsyncMock()
        analyzer.health_check = AsyncMock(side_effect=RuntimeError("ai down"))
        result = await monitor.health_check(
            container_mgr, config_mgr, MagicMock(), intelligent_analyzer=analyzer
        )
        assert result["checks"]["intelligent_analyzer"]["status"] == "unhealthy"
        assert result["checks"]["intelligent_analyzer"]["error"] == "ai down"
        assert result["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_degraded_status(self):
        monitor = TestMonitor()
        container_mgr = AsyncMock()
        container_mgr.health_check = AsyncMock(return_value={"status": "healthy"})
        config_mgr = MagicMock()
        config_mgr.validate_configuration.return_value = {
            "valid": False,
            "details": {"issue": "bad"},
        }
        result = await monitor.health_check(container_mgr, config_mgr, MagicMock())
        # Only config is degraded (not "healthy" or "disabled"), containers are healthy,
        # cache is disabled, analyzer is disabled -> 1 unhealthy out of 4 -> "degraded"
        assert result["status"] == "degraded"
        assert result["checks"]["config"]["status"] == "degraded"


class TestTestMonitorGetTestResults:
    @pytest.mark.asyncio
    async def test_from_cache(self):
        cache = AsyncMock()
        cache.get = AsyncMock(
            return_value={
                "test_id": "t1",
                "success": True,
                "stdout": "OK",
                "stderr": "",
                "exit_code": 0,
                "execution_time": 1.0,
                "created_at": datetime.now(UTC).isoformat(),
                "language": "python",
                "security_level": "high",
            }
        )
        monitor = TestMonitor(cache_manager=cache)
        result = await monitor.get_test_results("t1")
        assert result.test_id == "t1"
        assert result.success is True
        assert result.stdout == "OK"
        assert result.exit_code == 0
        assert result.execution_time == 1.0
        assert result.language == "python"
        cache.get.assert_called_once_with("test_result:t1")

    @pytest.mark.asyncio
    async def test_from_active_done_task(self):
        monitor = TestMonitor()
        expected = MagicMock()
        expected.success = True

        async def done_coroutine():
            return expected

        task = asyncio.ensure_future(done_coroutine())
        await task  # let it finish
        monitor.active_tests["t1"] = task
        result = await monitor.get_test_results("t1")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_from_active_running_task(self):
        monitor = TestMonitor()

        async def running_coroutine():
            await asyncio.sleep(100)

        task = asyncio.ensure_future(running_coroutine())
        monitor.active_tests["t1"] = task
        with pytest.raises(ValueError, match="still running"):
            await monitor.get_test_results("t1")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_not_found(self):
        monitor = TestMonitor()
        with pytest.raises(ValueError, match="not found"):
            await monitor.get_test_results("nonexistent")


class TestTestMonitorCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_running_task(self):
        monitor = TestMonitor()

        async def running_coroutine():
            await asyncio.sleep(100)

        task = asyncio.ensure_future(running_coroutine())
        monitor.active_tests["t1"] = task
        container_mgr = AsyncMock()
        container_mgr.cleanup_test_containers = AsyncMock()

        result = await monitor.cleanup_test_environment("t1", container_mgr)
        assert result is True
        assert "t1" not in monitor.active_tests

    @pytest.mark.asyncio
    async def test_cleanup_with_cache(self):
        cache = AsyncMock()
        cache.delete = AsyncMock()
        monitor = TestMonitor(cache_manager=cache)
        container_mgr = AsyncMock()
        container_mgr.cleanup_test_containers = AsyncMock()

        result = await monitor.cleanup_test_environment("t1", container_mgr)
        assert result is True
        cache.delete.assert_called_once_with("test_result:t1")
        container_mgr.cleanup_test_containers.assert_called_once_with("t1")

    @pytest.mark.asyncio
    async def test_cleanup_exception(self):
        monitor = TestMonitor()
        container_mgr = AsyncMock()
        container_mgr.cleanup_test_containers = AsyncMock(side_effect=RuntimeError("fail"))
        result = await monitor.cleanup_test_environment("t1", container_mgr)
        # Still returns False because container cleanup failed
        assert result is False


class TestTestMonitorActiveTracking:
    def test_track_and_get_ids(self):
        monitor = TestMonitor()
        task = MagicMock()
        monitor.track_active_test("t1", task)
        assert monitor.get_active_test_ids() == ["t1"]
        assert monitor.get_active_test_count() == 1

    def test_untrack(self):
        monitor = TestMonitor()
        task = MagicMock()
        monitor.track_active_test("t1", task)
        returned = monitor.untrack_test("t1")
        assert returned == task
        assert monitor.get_active_test_count() == 0

    def test_untrack_nonexistent(self):
        monitor = TestMonitor()
        assert monitor.untrack_test("nonexistent") is None


class TestTestMonitorCancelTest:
    @pytest.mark.asyncio
    async def test_cancel_nonexistent(self):
        monitor = TestMonitor()
        result = await monitor.cancel_test("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_done_task(self):
        monitor = TestMonitor()

        async def done():
            return MagicMock()

        task = asyncio.ensure_future(done())
        await task
        monitor.active_tests["t1"] = task
        result = await monitor.cancel_test("t1")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_running_task(self):
        monitor = TestMonitor()

        async def running():
            await asyncio.sleep(100)

        task = asyncio.ensure_future(running())
        monitor.active_tests["t1"] = task
        result = await monitor.cancel_test("t1")
        assert result is True


class TestTestMonitorCleanupCompleted:
    @pytest.mark.asyncio
    async def test_cleanup_completed(self):
        monitor = TestMonitor()

        async def done():
            return MagicMock()

        task = asyncio.ensure_future(done())
        await task
        monitor.active_tests["t1"] = task

        count = await monitor.cleanup_completed_tests()
        assert count == 1
        assert "t1" not in monitor.active_tests


class TestTestMonitorGetTestStatus:
    @pytest.mark.asyncio
    async def test_not_found(self):
        monitor = TestMonitor()
        status = await monitor.get_test_status("nonexistent")
        assert status["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_completed(self):
        monitor = TestMonitor()

        async def done():
            result = MagicMock()
            result.success = True
            result.execution_time = 1.0
            return result

        task = asyncio.ensure_future(done())
        await task
        monitor.active_tests["t1"] = task
        status = await monitor.get_test_status("t1")
        assert status["status"] == "completed"

    @pytest.mark.asyncio
    async def test_failed(self):
        monitor = TestMonitor()

        async def failed():
            raise RuntimeError("oops")

        task = asyncio.ensure_future(failed())
        try:
            await task
        except RuntimeError:
            pass
        monitor.active_tests["t1"] = task
        status = await monitor.get_test_status("t1")
        assert status["status"] == "failed"

    @pytest.mark.asyncio
    async def test_running(self):
        monitor = TestMonitor()

        async def running():
            await asyncio.sleep(100)

        task = asyncio.ensure_future(running())
        monitor.active_tests["t1"] = task
        status = await monitor.get_test_status("t1")
        assert status["status"] == "running"
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


class TestTestMonitorGetSystemMetrics:
    @pytest.mark.asyncio
    async def test_returns_metrics(self):
        monitor = TestMonitor()
        metrics = await monitor.get_system_metrics()
        assert "active_tests" in metrics
        assert metrics["cache_status"] == "disabled"

    @pytest.mark.asyncio
    async def test_with_cache(self):
        monitor = TestMonitor(cache_manager=AsyncMock())
        metrics = await monitor.get_system_metrics()
        assert metrics["cache_status"] == "enabled"
