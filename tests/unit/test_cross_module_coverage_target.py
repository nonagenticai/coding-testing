"""Targeted tests to push coverage above 88%.

Covers uncovered lines in:
- test_execution_container_registry.py (74%): health checks, docker init, periodic refresh
- resource_monitor.py (77%): collect_container_resources, monitoring loop, start/stop
- test_orchestrator.py (82%): k8s_executor, collect_coverage/perf/security, container_name validation
- container_manager.py (83%): cleanup_old_containers, _create_test_container, _detect_container_language
- pydantic_ai_agent.py (81%): analyze_test_failure error paths
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

# ===========================================================================
# CONTAINER REGISTRY — check_service_health with aiohttp, docker init paths,
# periodic refresh workers, get_health_status, force_health_check_refresh
# ===========================================================================
from services.test_execution.test_execution_container_registry import (
    ContainerExecutionStatus,
    TestExecutionContainerRegistry,
    TestExecutionSpec,
)


@pytest.fixture
def skip_registry(monkeypatch):
    """Registry with SKIP_TEST_EXECUTORS=true."""
    monkeypatch.setenv("SKIP_TEST_EXECUTORS", "true")
    return TestExecutionContainerRegistry(enable_periodic_refresh=False)


class TestContainerRegistryCheckServiceHealth:
    """Cover check_service_health aiohttp paths (lines 472-506)."""

    @pytest.mark.asyncio
    async def test_health_check_healthy_response(self, skip_registry, monkeypatch):
        """Test health check with a 200 response — marks container AVAILABLE."""
        import services.test_execution.test_execution_container_registry as mod

        monkeypatch.setattr(mod, "HAS_AIOHTTP", True)

        # Container starts as non-AVAILABLE; health check should promote it
        assert skip_registry.container_status["python"] != ContainerExecutionStatus.AVAILABLE

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        mock_aiohttp.ClientTimeout = MagicMock()
        monkeypatch.setattr(mod, "aiohttp", mock_aiohttp)

        result = await skip_registry.check_service_health("python")
        # 200 response means healthy — method returns True
        assert result is True, "200 status should indicate healthy service"
        # Container status should transition to AVAILABLE after successful health check
        assert skip_registry.container_status["python"] == ContainerExecutionStatus.AVAILABLE
        # Verify the health check URL was constructed correctly for python
        mock_session.get.assert_called_once_with("http://python-test-executor:8087/health")

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_response(self, skip_registry, monkeypatch):
        """Test health check with a non-200 response — marks container BUILDING."""
        import services.test_execution.test_execution_container_registry as mod

        monkeypatch.setattr(mod, "HAS_AIOHTTP", True)

        mock_response = AsyncMock()
        mock_response.status = 503
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        mock_aiohttp.ClientTimeout = MagicMock()
        monkeypatch.setattr(mod, "aiohttp", mock_aiohttp)

        result = await skip_registry.check_service_health("python")
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_connection_error(self, skip_registry, monkeypatch):
        """Test health check with connection error — logs at INFO level."""
        import services.test_execution.test_execution_container_registry as mod

        monkeypatch.setattr(mod, "HAS_AIOHTTP", True)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=ConnectionError("connection refused"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        mock_aiohttp.ClientTimeout = MagicMock()
        monkeypatch.setattr(mod, "aiohttp", mock_aiohttp)

        result = await skip_registry.check_service_health("python")
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_non_connection_error(self, skip_registry, monkeypatch):
        """Test health check with non-connection error — logs at WARNING level."""
        import services.test_execution.test_execution_container_registry as mod

        monkeypatch.setattr(mod, "HAS_AIOHTTP", True)

        # Set status to AVAILABLE first to test the downgrade path
        skip_registry.container_status["python"] = ContainerExecutionStatus.AVAILABLE

        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=RuntimeError("unexpected error"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        mock_aiohttp.ClientTimeout = MagicMock()
        monkeypatch.setattr(mod, "aiohttp", mock_aiohttp)

        result = await skip_registry.check_service_health("python")
        assert result is False
        # Should be downgraded from AVAILABLE to BUILDING
        assert skip_registry.container_status["python"] == ContainerExecutionStatus.BUILDING

    @pytest.mark.asyncio
    async def test_health_check_go_timeout(self, skip_registry, monkeypatch):
        """Go language gets longer timeout (line 464-465)."""
        import services.test_execution.test_execution_container_registry as mod

        monkeypatch.setattr(mod, "HAS_AIOHTTP", True)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        mock_aiohttp.ClientTimeout = MagicMock()
        monkeypatch.setattr(mod, "aiohttp", mock_aiohttp)

        result = await skip_registry.check_service_health("go")
        assert result is True


class TestContainerRegistryDockerInit:
    """Cover Docker init path (lines 128-161) and _check_container_availability with docker."""

    def test_init_with_docker_available(self, monkeypatch):
        """Init path where HAS_DOCKER=True but docker fails to connect."""
        import services.test_execution.test_execution_container_registry as mod

        monkeypatch.delenv("SKIP_TEST_EXECUTORS", raising=False)
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("KUBERNETES_IN_CLUSTER", raising=False)

        mock_docker = MagicMock()
        mock_docker.from_env = MagicMock(side_effect=RuntimeError("Docker not running"))
        monkeypatch.setattr(mod, "HAS_DOCKER", True)
        monkeypatch.setattr(mod, "docker", mock_docker)

        reg = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        assert reg.docker_client is None
        assert reg._docker_available is False

    def test_init_with_docker_not_installed(self, monkeypatch):
        """Init path where HAS_DOCKER=False (line 140-141)."""
        import services.test_execution.test_execution_container_registry as mod

        monkeypatch.delenv("SKIP_TEST_EXECUTORS", raising=False)
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("KUBERNETES_IN_CLUSTER", raising=False)

        monkeypatch.setattr(mod, "HAS_DOCKER", False)

        reg = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        assert reg.docker_client is None
        assert reg._docker_available is False

    def test_check_availability_with_docker_running_container(self, skip_registry):
        """Test _check_container_availability with docker client and running container."""
        mock_container = MagicMock()
        mock_container.status = "running"

        mock_docker = MagicMock()
        mock_docker.containers.list = MagicMock(return_value=[mock_container])
        skip_registry.docker_client = mock_docker

        skip_registry._check_container_availability()
        # At least python should be BUILDING (running but not health checked)
        assert skip_registry.container_status["python"] == ContainerExecutionStatus.BUILDING

    def test_check_availability_with_docker_no_container(self, skip_registry):
        """Test _check_container_availability with docker but no matching containers."""
        mock_docker = MagicMock()
        mock_docker.containers.list = MagicMock(return_value=[])
        skip_registry.docker_client = mock_docker

        skip_registry._check_container_availability()
        assert skip_registry.container_status["python"] == ContainerExecutionStatus.UNAVAILABLE

    def test_check_availability_docker_exception(self, skip_registry):
        """Test _check_container_availability with docker raising exception."""
        mock_docker = MagicMock()
        mock_docker.containers.list = MagicMock(side_effect=RuntimeError("docker fail"))
        skip_registry.docker_client = mock_docker

        skip_registry._check_container_availability()
        assert skip_registry.container_status["python"] == ContainerExecutionStatus.FAILED


class TestContainerRegistryHealthStatus:
    """Cover get_health_status (lines 932-970)."""

    def test_get_health_status(self, skip_registry):
        status = skip_registry.get_health_status()
        assert isinstance(status["timestamp"], float)
        assert status["timestamp"] > 0
        assert status["last_health_check"] == 0  # Never checked yet
        assert status["health_check_interval"] == skip_registry.health_check_interval
        assert status["docker_client_available"] is False  # skip_registry has no docker
        assert status["overall_health"]["total_services"] == 9
        # All services should be unhealthy since SKIP_TEST_EXECUTORS=true
        assert status["overall_health"]["healthy_services"] == 0
        assert status["overall_health"]["health_rate"] == 0
        # Container status should have all 9 languages
        assert len(status["container_status"]) == 9
        assert len(status["health_results"]) == 9

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_force_health_check_refresh(self, skip_registry):
        result = skip_registry.force_health_check_refresh()
        assert len(result) == 9
        expected_langs = {
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
        assert set(result.keys()) == expected_langs
        # All services should be unhealthy (False) in skip mode
        for lang, healthy in result.items():
            assert healthy is False, f"{lang} should be unhealthy in skip mode"


class TestContainerRegistryBuildWithDocker:
    """Cover build_container with valid Dockerfile (lines 537-564)."""

    def test_build_success(self, skip_registry, tmp_path):
        mock_docker = MagicMock()
        mock_image = MagicMock()
        mock_logs = [{"stream": "Step 1: FROM scratch\n"}]
        mock_docker.images.build = MagicMock(return_value=(mock_image, mock_logs))
        skip_registry.docker_client = mock_docker

        df = tmp_path / "Dockerfile"
        df.write_text("FROM scratch\n")

        result = skip_registry.build_container("python", str(df))
        assert result is True
        assert skip_registry.container_status["python"] == ContainerExecutionStatus.AVAILABLE

    def test_build_log_non_stream_entry(self, skip_registry, tmp_path):
        """Build log entries without 'stream' key are skipped."""
        mock_docker = MagicMock()
        mock_image = MagicMock()
        # Mix of stream and non-stream log entries
        mock_logs = [{"stream": "Step 1\n"}, {"error": "warning"}, {"stream": 123}]
        mock_docker.images.build = MagicMock(return_value=(mock_image, mock_logs))
        skip_registry.docker_client = mock_docker

        df = tmp_path / "Dockerfile"
        df.write_text("FROM scratch\n")
        result = skip_registry.build_container("python", str(df))
        assert result is True


class TestContainerRegistrySingleAvailability:
    """Cover _check_single_container_availability with docker (lines 1022-1033)."""

    def test_image_found(self, skip_registry):
        mock_docker = MagicMock()
        mock_docker.images.get = MagicMock(return_value=MagicMock())
        skip_registry.docker_client = mock_docker

        skip_registry._check_single_container_availability("python")
        assert skip_registry.container_status["python"] == ContainerExecutionStatus.AVAILABLE

    def test_image_not_found(self, skip_registry, monkeypatch):
        """ImageNotFound exception marks container UNAVAILABLE (lines 1026-1031)."""
        import services.test_execution.test_execution_container_registry as mod

        monkeypatch.setattr(mod, "HAS_DOCKER", True)

        # Create a mock docker module with errors.ImageNotFound
        mock_docker_mod = MagicMock()
        mock_docker_mod.errors = MagicMock()
        mock_docker_mod.errors.ImageNotFound = type("ImageNotFound", (Exception,), {})
        monkeypatch.setattr(mod, "docker", mock_docker_mod)

        mock_client = MagicMock()
        mock_client.images.get = MagicMock(
            side_effect=mock_docker_mod.errors.ImageNotFound("not found")
        )
        skip_registry.docker_client = mock_client

        skip_registry._check_single_container_availability("python")
        assert skip_registry.container_status["python"] == ContainerExecutionStatus.UNAVAILABLE

    def test_other_docker_error(self, skip_registry, monkeypatch):
        """Non-ImageNotFound exception marks container FAILED (line 1033)."""
        import services.test_execution.test_execution_container_registry as mod

        monkeypatch.setattr(mod, "HAS_DOCKER", False)

        mock_client = MagicMock()
        mock_client.images.get = MagicMock(side_effect=RuntimeError("connection error"))
        skip_registry.docker_client = mock_client

        skip_registry._check_single_container_availability("python")
        assert skip_registry.container_status["python"] == ContainerExecutionStatus.FAILED


class TestSafeHealthCheck:
    """Cover _safe_health_check (lines 909-930)."""

    @pytest.mark.asyncio
    async def test_safe_health_check_timeout(self, skip_registry, monkeypatch):
        """Timeout returns False and logs at INFO (lines 915-917)."""

        async def slow_check(lang, timeout=5.0):
            await asyncio.sleep(10)

        skip_registry.check_service_health = slow_check
        result = await skip_registry._safe_health_check("python")
        assert result is False

    @pytest.mark.asyncio
    async def test_safe_health_check_connection_error(self, skip_registry):
        """Connection error returns False (lines 921-927)."""

        async def conn_error(lang, timeout=5.0):
            raise ConnectionError("connection refused")

        skip_registry.check_service_health = conn_error
        result = await skip_registry._safe_health_check("python")
        assert result is False

    @pytest.mark.asyncio
    async def test_safe_health_check_other_error(self, skip_registry):
        """Non-connection error returns False (lines 928-930)."""

        async def other_error(lang, timeout=5.0):
            raise ValueError("unexpected")

        skip_registry.check_service_health = other_error
        result = await skip_registry._safe_health_check("python")
        assert result is False


# ===========================================================================
# RESOURCE MONITOR — collect_container_resources, start/stop, monitoring loop
# ===========================================================================

from services.test_execution.resource_monitor import (  # noqa: E402
    AlertLevel,
    ResourceCollector,
    ResourceMonitor,
    ResourceType,
    ResourceUsage,
    get_resource_monitor,
)


class TestResourceCollectorContainerResources:
    """Cover collect_container_resources (lines 266-329)."""

    def test_collect_container_no_docker(self):
        collector = ResourceCollector()
        collector.docker_client = None
        result = collector.collect_container_resources("container-123")
        assert result == []

    def test_collect_container_with_mock_docker(self):
        collector = ResourceCollector()
        mock_container = MagicMock()
        mock_container.stats.return_value = {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 200000, "percpu_usage": [100000, 100000]},
                "system_cpu_usage": 1000000,
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": 100000},
                "system_cpu_usage": 900000,
            },
            "memory_stats": {
                "usage": 100 * 1024 * 1024,  # 100MB
                "limit": 512 * 1024 * 1024,  # 512MB
                "cache": 10 * 1024 * 1024,
            },
        }
        mock_docker = MagicMock()
        mock_docker.containers.get.return_value = mock_container
        collector.docker_client = mock_docker

        result = collector.collect_container_resources("container-123")
        assert len(result) == 2  # CPU and Memory
        cpu_resource = next(r for r in result if r.resource_type == ResourceType.CPU)
        mem_resource = next(r for r in result if r.resource_type == ResourceType.MEMORY)
        # CPU: (100000/100000) * 2 CPUs * 100 = 200.0%
        assert cpu_resource.usage_percent == 200.0
        assert cpu_resource.current_value == 200.0
        assert cpu_resource.max_value == 100.0
        assert cpu_resource.unit == "percent"
        # Memory: 100MB used / 512MB limit
        assert mem_resource.current_value == 100.0  # 100 MB
        assert mem_resource.max_value == 512.0  # 512 MB
        assert mem_resource.usage_percent == pytest.approx(19.53125)
        assert mem_resource.unit == "MB"

    def test_collect_container_docker_exception(self):
        collector = ResourceCollector()
        mock_docker = MagicMock()
        mock_docker.containers.get.side_effect = RuntimeError("container not found")
        collector.docker_client = mock_docker

        result = collector.collect_container_resources("nonexistent")
        assert result == []

    def test_collect_container_zero_system_delta(self):
        """When system_delta is 0, cpu_percent should be 0."""
        collector = ResourceCollector()
        mock_container = MagicMock()
        mock_container.stats.return_value = {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 100000, "percpu_usage": [100000]},
                "system_cpu_usage": 1000000,
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": 100000},
                "system_cpu_usage": 1000000,  # Same, so delta = 0
            },
            "memory_stats": {
                "usage": 50 * 1024 * 1024,
                "limit": 256 * 1024 * 1024,
            },
        }
        mock_docker = MagicMock()
        mock_docker.containers.get.return_value = mock_container
        collector.docker_client = mock_docker

        result = collector.collect_container_resources("container-zero")
        cpu_resource = next(r for r in result if r.resource_type == ResourceType.CPU)
        assert cpu_resource.usage_percent == 0.0


class TestResourceMonitorStartStop:
    """Cover start_monitoring, stop_monitoring, _monitoring_loop (lines 574-618)."""

    @pytest.mark.asyncio
    async def test_start_and_stop_monitoring(self):
        mon = ResourceMonitor(monitoring_interval=1)
        # Mock the collector to avoid real system calls
        mon.collector.collect_system_resources = MagicMock(return_value=[])

        await mon.start_monitoring()
        assert mon.monitoring_active is True
        assert isinstance(mon.monitoring_task, asyncio.Task)
        assert not mon.monitoring_task.done()

        # Let it run briefly
        await asyncio.sleep(0.1)

        await mon.stop_monitoring()
        assert mon.monitoring_active is False
        assert mon.monitoring_task.done()

    @pytest.mark.asyncio
    async def test_monitoring_loop_collects_resources(self):
        """Monitoring loop processes resources and checks thresholds."""
        mon = ResourceMonitor(monitoring_interval=0)

        now = datetime.now(UTC)
        fake_resources = [
            ResourceUsage(
                timestamp=now,
                resource_type=ResourceType.CPU,
                current_value=95.0,
                max_value=100.0,
                usage_percent=95.0,
                unit="percent",
            )
        ]
        mon.collector.collect_system_resources = MagicMock(return_value=fake_resources)

        await mon.start_monitoring()
        await asyncio.sleep(0.15)
        await mon.stop_monitoring()

        # Should have collected the fake CPU resource at least once
        assert len(mon.resource_history) >= 1
        assert mon.resource_history[0].resource_type == ResourceType.CPU
        assert mon.resource_history[0].usage_percent == 95.0
        assert mon.resource_history[0].current_value == 95.0

    @pytest.mark.asyncio
    async def test_monitoring_loop_handles_exception(self):
        """Monitoring loop handles exceptions without crashing."""
        mon = ResourceMonitor(monitoring_interval=0)
        mon.collector.collect_system_resources = MagicMock(
            side_effect=RuntimeError("collection failed")
        )

        await mon.start_monitoring()
        await asyncio.sleep(0.15)
        await mon.stop_monitoring()
        # Should not crash


class TestResourceMonitorTerminateContainers:
    """Cover _terminate_containers_handler (lines 541-567)."""

    def test_terminate_handler_no_docker(self):
        mon = ResourceMonitor()
        mon.collector.docker_client = None
        alert = MagicMock()
        # Should not raise
        mon._terminate_containers_handler(alert)

    def test_terminate_handler_with_containers(self):
        mon = ResourceMonitor()
        mock_container = MagicMock()
        mock_container.name = "test-executor-1"
        mock_container.attrs = {"Created": "2024-01-01T00:00:00Z"}
        mock_container.stop = MagicMock()

        mock_docker = MagicMock()
        mock_docker.containers.list.return_value = [mock_container]
        mon.collector.docker_client = mock_docker

        from services.test_execution.resource_monitor import ResourceAlert

        alert = ResourceAlert(
            timestamp=datetime.now(UTC),
            resource_type=ResourceType.CPU,
            alert_level=AlertLevel.EMERGENCY,
            current_value=99.0,
            threshold_value=95.0,
            usage_percent=99.0,
            message="Emergency",
        )
        mon._terminate_containers_handler(alert)
        mock_container.stop.assert_called_once_with(timeout=10)

    def test_terminate_handler_stop_fails(self):
        mon = ResourceMonitor()
        mock_container = MagicMock()
        mock_container.name = "test-executor-2"
        mock_container.attrs = {"Created": "2024-01-01T00:00:00Z"}
        mock_container.stop.side_effect = RuntimeError("stop failed")

        mock_docker = MagicMock()
        mock_docker.containers.list.return_value = [mock_container]
        mon.collector.docker_client = mock_docker

        from services.test_execution.resource_monitor import ResourceAlert

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


class TestGetResourceMonitor:
    """Cover get_resource_monitor factory (lines 682-687)."""

    def test_get_global_instance(self, monkeypatch):
        import services.test_execution.resource_monitor as mod

        monkeypatch.setattr(mod, "_resource_monitor", None)
        mon = get_resource_monitor()
        # Verify it's a properly initialized ResourceMonitor
        assert mon.monitoring_active is False
        assert mon.resource_history == []
        assert mon.monitoring_interval == 30  # default interval
        # Second call returns same instance (singleton)
        mon2 = get_resource_monitor()
        assert mon is mon2


# ===========================================================================
# TEST ORCHESTRATOR — k8s_executor, _collect_coverage_report (Rust),
# _collect_performance_metrics, _collect_security_issues (Rust),
# container_name validation, _parse_test_output exception
# ===========================================================================

from services.test_execution.test_orchestrator import (  # noqa: E402
    OrchestratorType,
    TestExecutionRequest,
    TestLanguage,
    TestOrchestrator,
)


@pytest.fixture
def orch_docker_service():
    svc = AsyncMock()
    svc.create_test_container = AsyncMock(return_value="container-123")
    svc.start_container = AsyncMock()
    svc.create_file_in_container = AsyncMock(return_value=True)
    svc.exec_in_container = AsyncMock(return_value={"exit_code": 0, "stdout": "", "stderr": ""})
    svc.stop_container = AsyncMock()
    svc.remove_container_by_id = AsyncMock()
    return svc


@pytest.fixture
def orch(orch_docker_service):
    from config.test_framework_config import TestConfig

    return TestOrchestrator(orch_docker_service, TestConfig())


class TestOrchestratorK8sExecutor:
    """Cover k8s_executor property (lines 157-168)."""

    def test_k8s_executor_not_k8s_mode(self, orch):
        orch.use_k8s = False
        assert orch.k8s_executor is None

    def test_k8s_executor_import_error(self, orch, monkeypatch):
        orch.use_k8s = True
        orch._k8s_executor = None
        # Force the import to fail by patching builtins.__import__
        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def fail_k8s_import(name, *args, **kwargs):
            if "kubernetes_test_executor" in name:
                raise ImportError("mocked import failure")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fail_k8s_import)
        result = orch.k8s_executor
        assert result is None


class TestOrchestratorCollectCoverageReport:
    """Cover _collect_coverage_report Rust branch (lines 709-720)."""

    @pytest.mark.asyncio
    async def test_collect_coverage_go(self, orch, orch_docker_service):
        orch_docker_service.exec_in_container = AsyncMock(
            return_value={"exit_code": 0, "stdout": "coverage data"}
        )
        result = await orch._collect_coverage_report("c1", TestLanguage.GO)
        assert result == {"format": "go_coverage", "data": "coverage data"}

    @pytest.mark.asyncio
    async def test_collect_coverage_rust(self, orch, orch_docker_service):
        orch_docker_service.exec_in_container = AsyncMock(
            return_value={"exit_code": 0, "stdout": '{"coverage": 85}'}
        )
        result = await orch._collect_coverage_report("c1", TestLanguage.RUST)
        assert result == {"format": "tarpaulin_json", "data": '{"coverage": 85}'}

    @pytest.mark.asyncio
    async def test_collect_coverage_rust_failed(self, orch, orch_docker_service):
        orch_docker_service.exec_in_container = AsyncMock(
            return_value={"exit_code": 1, "stdout": ""}
        )
        result = await orch._collect_coverage_report("c1", TestLanguage.RUST)
        assert result is None

    @pytest.mark.asyncio
    async def test_collect_coverage_exception(self, orch, orch_docker_service):
        orch_docker_service.exec_in_container = AsyncMock(side_effect=RuntimeError("fail"))
        result = await orch._collect_coverage_report("c1", TestLanguage.GO)
        assert result is None

    @pytest.mark.asyncio
    async def test_collect_coverage_unsupported_lang(self, orch, orch_docker_service):
        result = await orch._collect_coverage_report("c1", TestLanguage.PYTHON)
        assert result is None


class TestOrchestratorCollectPerformanceMetrics:
    """Cover _collect_performance_metrics (lines 727-744)."""

    @pytest.mark.asyncio
    async def test_collect_perf_go(self, orch, orch_docker_service):
        stdout_data = '{"ns_per_op": 123}'
        orch_docker_service.exec_in_container = AsyncMock(
            return_value={"exit_code": 0, "stdout": stdout_data}
        )
        result = await orch._collect_performance_metrics("c1", TestLanguage.GO)
        assert result == {"benchmark_data": len(stdout_data)}

    @pytest.mark.asyncio
    async def test_collect_perf_go_failed(self, orch, orch_docker_service):
        orch_docker_service.exec_in_container = AsyncMock(
            return_value={"exit_code": 1, "stdout": ""}
        )
        result = await orch._collect_performance_metrics("c1", TestLanguage.GO)
        assert result is None

    @pytest.mark.asyncio
    async def test_collect_perf_exception(self, orch, orch_docker_service):
        orch_docker_service.exec_in_container = AsyncMock(side_effect=RuntimeError("fail"))
        result = await orch._collect_performance_metrics("c1", TestLanguage.GO)
        assert result is None

    @pytest.mark.asyncio
    async def test_collect_perf_unsupported_lang(self, orch, orch_docker_service):
        result = await orch._collect_performance_metrics("c1", TestLanguage.PYTHON)
        assert result is None


class TestOrchestratorCollectSecurityIssues:
    """Cover _collect_security_issues Rust branch (lines 746-773)."""

    @pytest.mark.asyncio
    async def test_collect_security_go(self, orch, orch_docker_service):
        orch_docker_service.exec_in_container = AsyncMock(
            return_value={"exit_code": 0, "stdout": "security report data"}
        )
        result = await orch._collect_security_issues("c1", TestLanguage.GO)
        assert result == ["golangci-lint: 20 bytes of output"]

    @pytest.mark.asyncio
    async def test_collect_security_rust(self, orch, orch_docker_service):
        orch_docker_service.exec_in_container = AsyncMock(
            return_value={"exit_code": 0, "stdout": "audit data"}
        )
        result = await orch._collect_security_issues("c1", TestLanguage.RUST)
        assert result == ["cargo-audit: 10 bytes of output"]

    @pytest.mark.asyncio
    async def test_collect_security_exception(self, orch, orch_docker_service):
        orch_docker_service.exec_in_container = AsyncMock(side_effect=RuntimeError("fail"))
        result = await orch._collect_security_issues("c1", TestLanguage.GO)
        assert result == []

    @pytest.mark.asyncio
    async def test_collect_security_unsupported_lang(self, orch, orch_docker_service):
        result = await orch._collect_security_issues("c1", TestLanguage.PYTHON)
        assert result == []


class TestOrchestratorContainerNameValidation:
    """Cover execute_tests container_name validation path (lines 186-201)."""

    @pytest.mark.asyncio
    async def test_execute_with_container_name(self, orch, orch_docker_service, monkeypatch):
        # Mock container validation to avoid real HTTP calls
        from shared import container_validation

        mock_container_info = MagicMock()
        mock_container_info.container_name = "my-container"
        monkeypatch.setattr(
            container_validation,
            "validate_and_get_container_for_tenant",
            AsyncMock(return_value=mock_container_info),
        )

        req = TestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="x = 1",
            container_name="my-container",
            timeout_seconds=60,
        )
        result = await orch.execute_tests(req)
        assert result.success is True
        assert result.exit_code == 0
        assert result.language == TestLanguage.PYTHON
        assert result.test_type == OrchestratorType.UNIT
        assert result.container_name == "my-container"
        # Verify container validation was called
        container_validation.validate_and_get_container_for_tenant.assert_called_once_with(
            "my-container", tenant_id=None
        )


class TestOrchestratorK8sPath:
    """Cover execute_tests K8s path (lines 214-226)."""

    @pytest.mark.asyncio
    async def test_k8s_path(self, orch, orch_docker_service):
        orch.use_k8s = True
        mock_executor = AsyncMock()
        mock_executor.run = AsyncMock(return_value={"exit_code": 0, "stdout": "PASS", "stderr": ""})
        orch._k8s_executor = mock_executor

        req = TestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="x = 1",
            timeout_seconds=60,
        )
        result = await orch.execute_tests(req)
        assert result.success is True
        mock_executor.run.assert_called_once()


# ===========================================================================
# CONTAINER MANAGER — cleanup_old_containers, _create_test_container errors,
# _detect_container_language, _container_supports_language
# ===========================================================================

from services.testing.container_manager import UnifiedContainerManager  # noqa: E402


@pytest.fixture
def cm_docker():
    svc = AsyncMock()
    svc.list_containers = AsyncMock(return_value=[])
    svc.find_container_by_name = AsyncMock(return_value=None)
    svc.create_test_container = AsyncMock(return_value="docker-id-1")
    svc.start_container = AsyncMock()
    svc.remove_container = AsyncMock()
    return svc


@pytest.fixture
def container_manager(cm_docker):
    return UnifiedContainerManager(cm_docker)


class TestUnifiedContainerManagerCleanup:
    """Cover cleanup_old_containers (lines 259-289)."""

    @pytest.mark.asyncio
    async def test_cleanup_old_containers(self, container_manager, cm_docker):
        old_time = (datetime.now(UTC) - timedelta(hours=48)).isoformat()
        cm_docker.list_containers.return_value = [
            {
                "container_name": "test-python-abc",
                "status": "running",
                "api_endpoint": "",
                "created_at": old_time,
            }
        ]

        cleaned = await container_manager.cleanup_old_containers(max_age_hours=24)
        assert cleaned == 1
        cm_docker.remove_container.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_no_created_at(self, container_manager, cm_docker):
        cm_docker.list_containers.return_value = [
            {
                "container_name": "test-python-abc",
                "status": "running",
                "api_endpoint": "",
                "created_at": None,
            }
        ]
        cleaned = await container_manager.cleanup_old_containers(max_age_hours=24)
        assert cleaned == 0

    @pytest.mark.asyncio
    async def test_cleanup_remove_fails(self, container_manager, cm_docker):
        old_time = (datetime.now(UTC) - timedelta(hours=48)).isoformat()
        cm_docker.list_containers.return_value = [
            {
                "container_name": "test-python-abc",
                "status": "running",
                "api_endpoint": "",
                "created_at": old_time,
            }
        ]
        cm_docker.remove_container.side_effect = RuntimeError("remove failed")
        cleaned = await container_manager.cleanup_old_containers(max_age_hours=24)
        assert cleaned == 0


class TestUnifiedContainerManagerCreateContainer:
    """Cover _create_test_container error path (lines 363-374)."""

    @pytest.mark.asyncio
    async def test_create_container_docker_error(self, container_manager, cm_docker):
        cm_docker.create_test_container.side_effect = RuntimeError("creation failed")

        spec = TestExecutionSpec(
            image="test:latest",
            version="1",
            memory_limit="256m",
            cpu_quota=25000,
            timeout_seconds=120,
            environment={},
            volumes={},
        )

        with pytest.raises(RuntimeError, match="Container creation failed"):
            await container_manager._create_test_container("test-cont", spec)


class TestUnifiedContainerManagerDetectLanguage:
    """Cover _detect_container_language (lines 385-406)."""

    def test_detect_python(self, container_manager):
        container = MagicMock()
        container.container_name = "codegraph-python-test"
        assert container_manager._detect_container_language(container) == "python"

    def test_detect_javascript(self, container_manager):
        container = MagicMock()
        container.container_name = "js-executor-123"
        assert container_manager._detect_container_language(container) == "javascript"

    def test_detect_go(self, container_manager):
        container = MagicMock()
        container.container_name = "golang-runner"
        assert container_manager._detect_container_language(container) == "go"

    def test_detect_unknown(self, container_manager):
        container = MagicMock()
        container.container_name = "my-service"
        assert container_manager._detect_container_language(container) is None


class TestUnifiedContainerManagerSupportsLanguage:
    """Cover _container_supports_language (lines 408-423)."""

    def test_direct_match(self, container_manager):
        info = {"language": "python"}
        assert container_manager._container_supports_language(info, "python") is True

    def test_nodejs_javascript_match(self, container_manager):
        info = {"language": "javascript"}
        assert container_manager._container_supports_language(info, "nodejs") is True

    def test_no_match(self, container_manager):
        info = {"language": "python"}
        assert container_manager._container_supports_language(info, "rust") is False

    def test_no_language_key(self, container_manager):
        info = {}
        assert container_manager._container_supports_language(info, "python") is False
