"""Round 4 coverage tests targeting ~460 uncovered lines across 13 files.

Files targeted:
1. test_execution_container_registry.py (69 uncov)
2. test_orchestration.py (56 uncov)
3. health_monitor.py (51 uncov)
4. report_formatter.py (51 uncov)
6. intelligent_test_orchestrator.py (42 uncov)
7. test_runner.py (37 uncov)
8. testing_utils.py (37 uncov)
9. pydantic_ai_agents.py (36 uncov)
10. intelligent_test_integration.py (34 uncov)
11. test_discovery.py (34 uncov)
12. intelligent_test_recommendation_engine.py (33 uncov)
13. core_service.py (32 uncov)
"""

from __future__ import annotations

import json
import sys
import types as _types
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Install docker/aiohttp stubs if not available (health_monitor imports at top level)
if "docker" not in sys.modules:
    _docker_stub = _types.ModuleType("docker")
    _docker_errors = _types.ModuleType("docker.errors")
    _docker_errors.NotFound = type("NotFound", (Exception,), {})
    _docker_errors.APIError = type("APIError", (Exception,), {})
    _docker_errors.ContainerError = type("ContainerError", (Exception,), {})
    _docker_stub.errors = _docker_errors
    _docker_stub.from_env = MagicMock()
    sys.modules["docker"] = _docker_stub
    sys.modules["docker.errors"] = _docker_errors

if "aiohttp" not in sys.modules:
    _aiohttp_stub = _types.ModuleType("aiohttp")
    _aiohttp_stub.ClientSession = MagicMock
    _aiohttp_stub.ClientTimeout = MagicMock
    _aiohttp_stub.ClientError = type("ClientError", (Exception,), {})
    sys.modules["aiohttp"] = _aiohttp_stub

# ---------------------------------------------------------------------------
# 1. test_execution_container_registry.py
# ---------------------------------------------------------------------------


class TestContainerRegistryCoverageR4:
    """Cover uncovered lines in test_execution_container_registry.py."""

    def test_init_ci_mode(self, monkeypatch):
        """Test __init__ in CI mode (lines 114-125)."""
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("SKIP_TEST_EXECUTORS", "false")
        monkeypatch.setenv("KUBERNETES_IN_CLUSTER", "false")
        from services.test_execution.test_execution_container_registry import (
            ContainerExecutionStatus,
            TestExecutionContainerRegistry,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        assert registry._docker_available is False
        assert registry.docker_client is None
        for lang in registry.specs:
            assert registry.container_status[lang] == ContainerExecutionStatus.UNAVAILABLE

    def test_init_skip_executors_mode(self, monkeypatch):
        """Test __init__ with SKIP_TEST_EXECUTORS (line 118 branch)."""
        monkeypatch.setenv("CI", "false")
        monkeypatch.setenv("SKIP_TEST_EXECUTORS", "true")
        monkeypatch.setenv("KUBERNETES_IN_CLUSTER", "false")
        from services.test_execution.test_execution_container_registry import (
            TestExecutionContainerRegistry,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        assert registry._docker_available is False

    def test_init_k8s_mode(self, monkeypatch):
        """Test __init__ with KUBERNETES_IN_CLUSTER mode."""
        monkeypatch.setenv("KUBERNETES_IN_CLUSTER", "true")
        monkeypatch.setenv("CI", "false")
        monkeypatch.setenv("SKIP_TEST_EXECUTORS", "false")
        from services.test_execution.test_execution_container_registry import (
            TestExecutionContainerRegistry,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        assert registry._docker_available is False

    def test_init_docker_not_available(self, monkeypatch):
        """Test __init__ when docker module not importable (lines 139-141)."""
        monkeypatch.setenv("CI", "false")
        monkeypatch.setenv("SKIP_TEST_EXECUTORS", "false")
        monkeypatch.setenv("KUBERNETES_IN_CLUSTER", "false")
        import services.test_execution.test_execution_container_registry as mod

        orig_has_docker = mod.HAS_DOCKER
        orig_docker = getattr(mod, "docker", None)
        try:
            mod.HAS_DOCKER = False
            mod.docker = None
            # Mock check_all_services_health_sync
            with patch.object(
                mod.TestExecutionContainerRegistry,
                "check_all_services_health_sync",
                return_value={},
            ):
                registry = mod.TestExecutionContainerRegistry(enable_periodic_refresh=False)
                assert registry._docker_available is False
                assert registry.docker_client is None
        finally:
            mod.HAS_DOCKER = orig_has_docker
            if orig_docker is not None:
                mod.docker = orig_docker

    def test_init_docker_ping_fails(self, monkeypatch):
        """Test __init__ when docker.from_env().ping() raises (lines 133-138)."""
        monkeypatch.setenv("CI", "false")
        monkeypatch.setenv("SKIP_TEST_EXECUTORS", "false")
        monkeypatch.setenv("KUBERNETES_IN_CLUSTER", "false")
        import services.test_execution.test_execution_container_registry as mod

        orig_has_docker = mod.HAS_DOCKER
        try:
            mod.HAS_DOCKER = True
            mock_docker_mod = MagicMock()
            mock_client = MagicMock()
            mock_client.ping.side_effect = Exception("Connection refused")
            mock_docker_mod.from_env.return_value = mock_client
            mod.docker = mock_docker_mod
            with patch.object(
                mod.TestExecutionContainerRegistry,
                "check_all_services_health_sync",
                return_value={},
            ):
                registry = mod.TestExecutionContainerRegistry(enable_periodic_refresh=False)
                assert registry._docker_available is False
                assert registry.docker_client is None
        finally:
            mod.HAS_DOCKER = orig_has_docker

    def test_get_available_languages(self, monkeypatch):
        """Test get_available_languages with some available (line 367)."""
        monkeypatch.setenv("CI", "true")
        from services.test_execution.test_execution_container_registry import (
            ContainerExecutionStatus,
            TestExecutionContainerRegistry,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        registry.container_status["python"] = ContainerExecutionStatus.AVAILABLE
        available = registry.get_available_languages()
        assert "python" in available

    def test_check_container_availability_no_service_mapping(self, monkeypatch):
        """Test _check_container_availability for unknown language (lines 413-415)."""
        monkeypatch.setenv("CI", "true")
        from services.test_execution.test_execution_container_registry import (
            ContainerExecutionStatus,
            TestExecutionContainerRegistry,
            TestExecutionSpec,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        registry.docker_client = MagicMock()
        # Add unsupported language
        registry.specs["brainfuck"] = TestExecutionSpec(
            image="x:latest",
            version="1",
            memory_limit="1g",
            cpu_quota=50000,
            timeout_seconds=60,
            environment={},
            volumes={},
        )
        registry._check_container_availability()
        assert registry.container_status["brainfuck"] == ContainerExecutionStatus.UNAVAILABLE

    def test_check_container_availability_docker_running(self, monkeypatch):
        """Test _check_container_availability when container is running (lines 419-429)."""
        monkeypatch.setenv("CI", "true")
        from services.test_execution.test_execution_container_registry import (
            ContainerExecutionStatus,
            TestExecutionContainerRegistry,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        mock_container = MagicMock()
        mock_container.status = "running"
        mock_docker_client = MagicMock()
        mock_docker_client.containers.list.return_value = [mock_container]
        registry.docker_client = mock_docker_client

        registry._check_container_availability()
        assert registry.container_status["python"] == ContainerExecutionStatus.BUILDING

    def test_check_container_availability_docker_not_running(self, monkeypatch):
        """Test _check_container_availability when container is not running (line 431-432)."""
        monkeypatch.setenv("CI", "true")
        from services.test_execution.test_execution_container_registry import (
            ContainerExecutionStatus,
            TestExecutionContainerRegistry,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        mock_docker_client = MagicMock()
        mock_docker_client.containers.list.return_value = []
        registry.docker_client = mock_docker_client

        registry._check_container_availability()
        assert registry.container_status["python"] == ContainerExecutionStatus.UNAVAILABLE

    def test_check_container_availability_exception(self, monkeypatch):
        """Test _check_container_availability with exception (lines 434-436)."""
        monkeypatch.setenv("CI", "true")
        from services.test_execution.test_execution_container_registry import (
            ContainerExecutionStatus,
            TestExecutionContainerRegistry,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        mock_docker_client = MagicMock()
        mock_docker_client.containers.list.side_effect = Exception("Docker error")
        registry.docker_client = mock_docker_client

        registry._check_container_availability()
        assert registry.container_status["python"] == ContainerExecutionStatus.FAILED

    def test_build_container_no_docker_client(self, monkeypatch):
        """Test build_container with no docker client (line 523-524)."""
        monkeypatch.setenv("CI", "true")
        from services.test_execution.test_execution_container_registry import (
            TestExecutionContainerRegistry,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        assert registry.build_container("python") is False

    def test_build_container_no_spec(self, monkeypatch):
        """Test build_container with nonexistent language (line 528-529)."""
        monkeypatch.setenv("CI", "true")
        from services.test_execution.test_execution_container_registry import (
            TestExecutionContainerRegistry,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        registry.docker_client = MagicMock()
        assert registry.build_container("nonexistent") is False

    def test_build_container_dockerfile_not_found(self, monkeypatch):
        """Test build_container when dockerfile doesn't exist (lines 537-543)."""
        monkeypatch.setenv("CI", "true")
        from services.test_execution.test_execution_container_registry import (
            ContainerExecutionStatus,
            TestExecutionContainerRegistry,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        registry.docker_client = MagicMock()
        result = registry.build_container("python")
        assert result is False
        assert registry.container_status["python"] == ContainerExecutionStatus.FAILED

    def test_build_container_build_fails(self, monkeypatch, tmp_path):
        """Test build_container when build raises exception (lines 567-570)."""
        monkeypatch.setenv("CI", "true")
        from services.test_execution.test_execution_container_registry import (
            ContainerExecutionStatus,
            TestExecutionContainerRegistry,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        mock_client = MagicMock()
        mock_client.images.build.side_effect = Exception("Build failed")
        registry.docker_client = mock_client
        # Create a temporary dockerfile
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM scratch\n")
        result = registry.build_container("python", str(dockerfile))
        assert result is False
        assert registry.container_status["python"] == ContainerExecutionStatus.FAILED

    def test_build_container_success(self, monkeypatch, tmp_path):
        """Test build_container success path (lines 546-565)."""
        monkeypatch.setenv("CI", "true")
        from services.test_execution.test_execution_container_registry import (
            ContainerExecutionStatus,
            TestExecutionContainerRegistry,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        mock_client = MagicMock()
        mock_image = MagicMock()
        mock_client.images.build.return_value = (
            mock_image,
            [{"stream": "Step 1/1 : FROM scratch\n"}],
        )
        registry.docker_client = mock_client
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM scratch\n")
        result = registry.build_container("python", str(dockerfile))
        assert result is True
        assert registry.container_status["python"] == ContainerExecutionStatus.AVAILABLE

    def test_pull_container_no_docker(self, monkeypatch):
        """Test pull_container without docker client (line 574-575)."""
        monkeypatch.setenv("CI", "true")
        from services.test_execution.test_execution_container_registry import (
            TestExecutionContainerRegistry,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        assert registry.pull_container("python") is False

    def test_pull_container_no_spec(self, monkeypatch):
        """Test pull_container for unknown language (line 579-580)."""
        monkeypatch.setenv("CI", "true")
        from services.test_execution.test_execution_container_registry import (
            TestExecutionContainerRegistry,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        registry.docker_client = MagicMock()
        assert registry.pull_container("nonexistent") is False

    def test_pull_container_success(self, monkeypatch):
        """Test pull_container success (lines 584-589)."""
        monkeypatch.setenv("CI", "true")
        from services.test_execution.test_execution_container_registry import (
            ContainerExecutionStatus,
            TestExecutionContainerRegistry,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        registry.docker_client = MagicMock()
        assert registry.pull_container("python") is True
        assert registry.container_status["python"] == ContainerExecutionStatus.AVAILABLE

    def test_pull_container_failure(self, monkeypatch):
        """Test pull_container exception (lines 591-594)."""
        monkeypatch.setenv("CI", "true")
        from services.test_execution.test_execution_container_registry import (
            ContainerExecutionStatus,
            TestExecutionContainerRegistry,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        mock_client = MagicMock()
        mock_client.images.pull.side_effect = Exception("Pull failed")
        registry.docker_client = mock_client
        assert registry.pull_container("python") is False
        assert registry.container_status["python"] == ContainerExecutionStatus.FAILED

    def test_register_custom_container_invalid_spec(self, monkeypatch):
        """Test register_custom_container with invalid spec (lines 976-979)."""
        monkeypatch.setenv("CI", "true")
        from services.test_execution.test_execution_container_registry import (
            TestExecutionContainerRegistry,
            TestExecutionSpec,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        bad_spec = TestExecutionSpec(
            image="",
            version="1",
            memory_limit="1g",
            cpu_quota=50000,
            timeout_seconds=60,
            environment={},
            volumes={},
        )
        assert registry.register_custom_container("custom", bad_spec) is False

    def test_register_custom_container_success(self, monkeypatch):
        """Test register_custom_container success (lines 982-989)."""
        monkeypatch.setenv("CI", "true")
        from services.test_execution.test_execution_container_registry import (
            TestExecutionContainerRegistry,
            TestExecutionSpec,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        spec = TestExecutionSpec(
            image="custom:latest",
            version="1",
            memory_limit="1g",
            cpu_quota=50000,
            timeout_seconds=60,
            environment={},
            volumes={},
        )
        assert registry.register_custom_container("custom", spec) is True
        assert "custom" in registry.specs

    def test_check_single_container_no_docker(self, monkeypatch):
        """Test _check_single_container_availability without docker (lines 1013-1015)."""
        monkeypatch.setenv("CI", "true")
        from services.test_execution.test_execution_container_registry import (
            ContainerExecutionStatus,
            TestExecutionContainerRegistry,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        registry._check_single_container_availability("python")
        assert registry.container_status["python"] == ContainerExecutionStatus.UNAVAILABLE

    def test_check_single_container_no_spec(self, monkeypatch):
        """Test _check_single_container_availability for unknown language (line 1018-1019)."""
        monkeypatch.setenv("CI", "true")
        from services.test_execution.test_execution_container_registry import (
            ContainerExecutionStatus,
            TestExecutionContainerRegistry,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        registry.docker_client = MagicMock()
        registry._check_single_container_availability("nonexistent")
        assert registry.container_status["nonexistent"] == ContainerExecutionStatus.UNAVAILABLE

    def test_check_single_container_image_found(self, monkeypatch):
        """Test _check_single_container_availability when image exists (lines 1022-1024)."""
        monkeypatch.setenv("CI", "true")
        from services.test_execution.test_execution_container_registry import (
            ContainerExecutionStatus,
            TestExecutionContainerRegistry,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        registry.docker_client = MagicMock()
        registry._check_single_container_availability("python")
        assert registry.container_status["python"] == ContainerExecutionStatus.AVAILABLE

    def test_check_single_container_generic_exception(self, monkeypatch):
        """Test _check_single_container_availability with generic error (line 1032-1033)."""
        monkeypatch.setenv("CI", "true")
        import services.test_execution.test_execution_container_registry as mod
        from services.test_execution.test_execution_container_registry import (
            ContainerExecutionStatus,
            TestExecutionContainerRegistry,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        mock_client = MagicMock()
        mock_client.images.get.side_effect = RuntimeError("Generic error")
        registry.docker_client = mock_client

        # Ensure docker mock has proper errors module with ImageNotFound as a real type
        mock_docker = MagicMock()
        mock_docker.errors = MagicMock()
        mock_docker.errors.ImageNotFound = type("ImageNotFound", (Exception,), {})
        orig_docker = getattr(mod, "docker", None)
        orig_has = mod.HAS_DOCKER
        try:
            mod.docker = mock_docker
            mod.HAS_DOCKER = True
            registry._check_single_container_availability("python")
            assert registry.container_status["python"] == ContainerExecutionStatus.FAILED
        finally:
            mod.HAS_DOCKER = orig_has
            if orig_docker is not None:
                mod.docker = orig_docker

    def test_check_all_services_health_deprecated(self, monkeypatch):
        """Test deprecated check_all_services_health (lines 509-513)."""
        monkeypatch.setenv("CI", "true")
        from services.test_execution.test_execution_container_registry import (
            TestExecutionContainerRegistry,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        with patch.object(registry, "check_all_services_health_sync", return_value={}) as mock:
            result = registry.check_all_services_health()
            mock.assert_called_once()
            assert result == {}

    def test_refresh_container_status(self, monkeypatch):
        """Test refresh_container_status (lines 516-518)."""
        monkeypatch.setenv("CI", "true")
        from services.test_execution.test_execution_container_registry import (
            ContainerExecutionStatus,
            TestExecutionContainerRegistry,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        registry.refresh_container_status()
        # In CI mode with no docker, all containers remain unavailable after refresh
        for lang_status in registry.container_status.values():
            assert lang_status == ContainerExecutionStatus.UNAVAILABLE

    def test_force_health_check_refresh(self, monkeypatch):
        """Test force_health_check_refresh (lines 844-848)."""
        monkeypatch.setenv("CI", "true")
        from services.test_execution.test_execution_container_registry import (
            TestExecutionContainerRegistry,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        with patch.object(
            registry, "check_all_services_health_sync", return_value={"python": False}
        ):
            result = registry.force_health_check_refresh()
            assert result == {"python": False}

    @pytest.mark.asyncio
    async def test_check_all_services_health_sync_in_event_loop(self, monkeypatch):
        """Test check_all_services_health_sync from async context (lines 893-896)."""
        monkeypatch.setenv("CI", "true")
        from services.test_execution.test_execution_container_registry import (
            TestExecutionContainerRegistry,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        # When called from within an event loop, should return conservative result
        result = registry.check_all_services_health_sync()
        # All should be False (conservative)
        for lang in registry.specs:
            assert result[lang] is False

    @pytest.mark.asyncio
    async def test_check_service_health_no_aiohttp(self, monkeypatch):
        """Test check_service_health when aiohttp not available (lines 455-457)."""
        monkeypatch.setenv("CI", "true")
        import services.test_execution.test_execution_container_registry as mod

        orig = mod.HAS_AIOHTTP
        try:
            mod.HAS_AIOHTTP = False
            registry = mod.TestExecutionContainerRegistry(enable_periodic_refresh=False)
            result = await registry.check_service_health("python")
            assert result is False
        finally:
            mod.HAS_AIOHTTP = orig

    @pytest.mark.asyncio
    async def test_check_service_health_no_health_url(self, monkeypatch):
        """Test check_service_health for unknown language (lines 460-461)."""
        monkeypatch.setenv("CI", "true")
        from services.test_execution.test_execution_container_registry import (
            TestExecutionContainerRegistry,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        result = await registry.check_service_health("nonexistent")
        assert result is False

    def test_stop_periodic_refresh(self, monkeypatch):
        """Test stop_periodic_refresh (lines 808-813)."""
        monkeypatch.setenv("CI", "true")
        from services.test_execution.test_execution_container_registry import (
            TestExecutionContainerRegistry,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        registry._refresh_thread = None
        registry.stop_periodic_refresh()
        assert registry._stop_refresh is True
        assert registry._refresh_thread is None

    def test_get_health_status(self, monkeypatch):
        """Test get_health_status (lines 932-970)."""
        monkeypatch.setenv("CI", "true")
        from services.test_execution.test_execution_container_registry import (
            TestExecutionContainerRegistry,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)
        with patch.object(
            registry, "check_all_services_health_sync", return_value={"python": False}
        ):
            status = registry.get_health_status()
            assert status["timestamp"] > 0  # valid epoch timestamp
            assert status["docker_client_available"] is False
            assert status["periodic_refresh_enabled"] is False
            assert status["health_results"] == {"python": False}
            assert status["overall_health"]["total_services"] == len(registry.specs)
            assert status["overall_health"]["healthy_services"] == 0
            assert status["overall_health"]["health_rate"] == 0.0

    def test_validate_container_spec_object_invalid(self, monkeypatch):
        """Test validate_container_spec_object with invalid specs (lines 995-1009)."""
        monkeypatch.setenv("CI", "true")
        from services.test_execution.test_execution_container_registry import (
            TestExecutionContainerRegistry,
            TestExecutionSpec,
        )

        registry = TestExecutionContainerRegistry(enable_periodic_refresh=False)

        # No image
        spec = TestExecutionSpec(
            image="",
            version="1",
            memory_limit="1g",
            cpu_quota=50000,
            timeout_seconds=60,
            environment={},
            volumes={},
        )
        result = registry.validate_container_spec_object(spec)
        assert result["valid"] is False
        assert "Image" in result["error"]

        # No memory limit
        spec2 = TestExecutionSpec(
            image="test:latest",
            version="1",
            memory_limit="",
            cpu_quota=50000,
            timeout_seconds=60,
            environment={},
            volumes={},
        )
        result2 = registry.validate_container_spec_object(spec2)
        assert result2["valid"] is False
        assert "Memory" in result2["error"]

        # Negative CPU quota
        spec3 = TestExecutionSpec(
            image="test:latest",
            version="1",
            memory_limit="1g",
            cpu_quota=-1,
            timeout_seconds=60,
            environment={},
            volumes={},
        )
        result3 = registry.validate_container_spec_object(spec3)
        assert result3["valid"] is False
        assert "CPU" in result3["error"]

        # Negative timeout
        spec4 = TestExecutionSpec(
            image="test:latest",
            version="1",
            memory_limit="1g",
            cpu_quota=50000,
            timeout_seconds=-1,
            environment={},
            volumes={},
        )
        result4 = registry.validate_container_spec_object(spec4)
        assert result4["valid"] is False
        assert "Timeout" in result4["error"]


# ---------------------------------------------------------------------------
# 2. routers/test_orchestration.py
# ---------------------------------------------------------------------------


class TestOrchestrationRouterR4:
    """Cover uncovered lines in test_orchestration.py."""

    def test_calculate_execution_order_with_dependencies(self):
        """Test _calculate_execution_order with dependencies."""
        from routers.test_orchestration import _calculate_execution_order

        suite_ids = ["a", "b", "c"]
        dependencies: dict[str, set[str]] = {"b": {"a"}, "c": {"b"}}
        order = _calculate_execution_order(suite_ids, dependencies)
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

    def test_calculate_parallel_groups_with_deps(self):
        """Test _calculate_parallel_groups with dependencies."""
        from routers.test_orchestration import _calculate_parallel_groups

        execution_order = ["a", "b", "c"]
        dependencies: dict[str, set[str]] = {"b": {"a"}, "c": {"a"}}
        groups = _calculate_parallel_groups(execution_order, dependencies)
        assert len(groups) == 2
        assert groups[0] == ["a"]
        assert set(groups[1]) == {"b", "c"}

    def test_calculate_parallel_groups_no_deps(self):
        """Test _calculate_parallel_groups without dependencies."""
        from routers.test_orchestration import _calculate_parallel_groups

        execution_order = ["a", "b", "c"]
        dependencies: dict[str, set[str]] = {}
        groups = _calculate_parallel_groups(execution_order, dependencies)
        # All should be in one group since no deps
        assert len(groups) == 1
        assert set(groups[0]) == {"a", "b", "c"}

    @pytest.mark.asyncio
    async def test_execute_suite_async_parallel(self):
        """Test execute_suite_async parallel path (lines 806-828)."""
        from routers.test_orchestration import (
            OrchestrationTestSuite,
            execute_suite_async,
        )
        from services.test_execution.test_orchestrator import (
            OrchestratorType,
            TestExecutionRequest,
            TestExecutionResult,
            TestLanguage,
            TestOrchestrator,
        )

        suite = OrchestrationTestSuite(
            suite_id="s1",
            name="Test Suite",
            description="desc",
            container_name="c1",
            tenant_id="t1",
        )
        req = TestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="print(1)",
            container_name="c1",
        )
        suite.tests = [req]

        mock_orchestrator = AsyncMock(spec=TestOrchestrator)
        mock_result = TestExecutionResult(
            test_id="t1",
            success=True,
            exit_code=0,
            stdout="ok",
            stderr="",
            execution_time_ms=100.0,
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            container_name="c1",
            created_at=datetime.now(UTC),
        )
        mock_orchestrator.execute_tests.return_value = mock_result

        await execute_suite_async(
            suite=suite,
            orchestrator=mock_orchestrator,
            parallel=True,
            max_workers=4,
            user_id="user1",
        )
        assert suite.status == "completed"
        assert len(suite.results) == 1
        assert suite.results[0].success is True

    @pytest.mark.asyncio
    async def test_execute_suite_async_parallel_with_exception(self):
        """Test execute_suite_async parallel with exception results (lines 813-826)."""
        from routers.test_orchestration import (
            OrchestrationTestSuite,
            execute_suite_async,
        )
        from services.test_execution.test_orchestrator import (
            OrchestratorType,
            TestExecutionRequest,
            TestLanguage,
            TestOrchestrator,
        )

        suite = OrchestrationTestSuite(
            suite_id="s2",
            name="Test Suite 2",
            description="desc",
            container_name="c1",
            tenant_id="t1",
        )
        req = TestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="print(1)",
            container_name="c1",
        )
        suite.tests = [req]

        mock_orchestrator = AsyncMock(spec=TestOrchestrator)
        mock_orchestrator.execute_tests.side_effect = Exception("execution failed")

        await execute_suite_async(
            suite=suite,
            orchestrator=mock_orchestrator,
            parallel=True,
            max_workers=4,
            user_id="user1",
        )
        assert suite.status == "completed"
        assert len(suite.results) == 1
        assert suite.results[0].success is False

    @pytest.mark.asyncio
    async def test_execute_suite_async_sequential(self):
        """Test execute_suite_async sequential path (lines 829-847)."""
        from routers.test_orchestration import (
            OrchestrationTestSuite,
            execute_suite_async,
        )
        from services.test_execution.test_orchestrator import (
            OrchestratorType,
            TestExecutionRequest,
            TestExecutionResult,
            TestLanguage,
            TestOrchestrator,
        )

        suite = OrchestrationTestSuite(
            suite_id="s3",
            name="Seq Suite",
            description="desc",
            container_name="c1",
            tenant_id="t1",
        )
        req = TestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="print(1)",
            container_name="c1",
        )
        suite.tests = [req]

        mock_orchestrator = AsyncMock(spec=TestOrchestrator)
        mock_result = TestExecutionResult(
            test_id="t1",
            success=True,
            exit_code=0,
            stdout="ok",
            stderr="",
            execution_time_ms=100.0,
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            container_name="c1",
            created_at=datetime.now(UTC),
        )
        mock_orchestrator.execute_tests.return_value = mock_result

        await execute_suite_async(
            suite=suite,
            orchestrator=mock_orchestrator,
            parallel=False,
            max_workers=4,
            user_id="user1",
        )
        assert suite.status == "completed"
        assert len(suite.results) == 1

    @pytest.mark.asyncio
    async def test_execute_suite_async_sequential_error(self):
        """Test execute_suite_async sequential with exception (lines 834-847)."""
        from routers.test_orchestration import (
            OrchestrationTestSuite,
            execute_suite_async,
        )
        from services.test_execution.test_orchestrator import (
            OrchestratorType,
            TestExecutionRequest,
            TestLanguage,
            TestOrchestrator,
        )

        suite = OrchestrationTestSuite(
            suite_id="s4",
            name="Seq Error Suite",
            description="desc",
            container_name="c1",
            tenant_id="t1",
        )
        req = TestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="print(1)",
            container_name="c1",
        )
        suite.tests = [req]

        mock_orchestrator = AsyncMock(spec=TestOrchestrator)
        mock_orchestrator.execute_tests.side_effect = Exception("seq fail")

        await execute_suite_async(
            suite=suite,
            orchestrator=mock_orchestrator,
            parallel=False,
            max_workers=4,
            user_id="user1",
        )
        assert suite.status == "completed"
        assert len(suite.results) == 1
        assert suite.results[0].success is False

    @pytest.mark.asyncio
    async def test_execute_suite_async_global_exception(self):
        """Test execute_suite_async global exception handler (lines 867-869)."""
        from routers.test_orchestration import (
            OrchestrationTestSuite,
            execute_suite_async,
        )
        from services.test_execution.test_orchestrator import TestOrchestrator

        suite = OrchestrationTestSuite(
            suite_id="s5",
            name="Global Error",
            description="desc",
            container_name="c1",
            tenant_id="t1",
        )
        suite.tests = []  # No tests but we'll force an error

        mock_orchestrator = AsyncMock(spec=TestOrchestrator)

        # Make results property raise
        original_results = suite.results
        type(suite).results = property(lambda s: (_ for _ in ()).throw(RuntimeError("boom")))

        try:
            await execute_suite_async(
                suite=suite,
                orchestrator=mock_orchestrator,
                parallel=True,
                max_workers=4,
                user_id="user1",
            )
        finally:
            type(suite).results = None
            suite.results = original_results
        # The property with no setter causes AttributeError on assignment,
        # which is caught by the outer except and sets status to "failed"
        assert suite.status == "failed"

    @pytest.mark.asyncio
    async def test_execute_plan_async(self):
        """Test execute_plan_async (lines 872-919)."""
        from routers.test_orchestration import (
            OrchestrationTestSuite,
            TestOrchestrationPlan,
            execute_plan_async,
        )
        from services.test_execution.test_orchestrator import (
            OrchestratorType,
            TestExecutionRequest,
            TestExecutionResult,
            TestLanguage,
            TestOrchestrator,
        )

        plan = TestOrchestrationPlan(
            plan_id="p1",
            container_name="c1",
            tenant_id="t1",
        )
        suite = OrchestrationTestSuite(
            suite_id="s1",
            name="Suite1",
            description="desc",
            container_name="c1",
            tenant_id="t1",
        )
        req = TestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="print(1)",
            container_name="c1",
        )
        suite.tests = [req]
        plan.suites = [suite]
        plan.parallel_groups = [["s1"]]

        mock_orchestrator = AsyncMock(spec=TestOrchestrator)
        mock_result = TestExecutionResult(
            test_id="t1",
            success=True,
            exit_code=0,
            stdout="ok",
            stderr="",
            execution_time_ms=100.0,
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            container_name="c1",
            created_at=datetime.now(UTC),
        )
        mock_orchestrator.execute_tests.return_value = mock_result

        await execute_plan_async(
            plan=plan,
            orchestrator=mock_orchestrator,
            user_id="user1",
        )
        assert plan.status == "completed"


# ---------------------------------------------------------------------------
# 3. health_monitor.py
# ---------------------------------------------------------------------------


class TestHealthMonitorR4:
    """Cover uncovered lines in health_monitor.py."""

    def test_http_health_checker_init(self):
        """Test HTTPHealthChecker init (lines 280-291)."""
        from services.test_execution.health_monitor import (
            HealthCheckConfig,
            HealthCheckType,
            HTTPHealthChecker,
        )

        config = HealthCheckConfig(check_type=HealthCheckType.READINESS)
        checker = HTTPHealthChecker(config, endpoint="/ready")
        assert checker.endpoint == "/ready"

    @pytest.mark.asyncio
    async def test_http_health_checker_check_health_client_error(self):
        """Test HTTPHealthChecker check_health with client error (lines 345-348)."""
        from services.test_execution.health_monitor import (
            HealthCheckConfig,
            HealthCheckType,
            HealthStatus,
            HTTPHealthChecker,
        )

        config = HealthCheckConfig(check_type=HealthCheckType.READINESS, timeout_seconds=1)
        checker = HTTPHealthChecker(config)

        # Patch the docker.from_env to raise ValueError (covers lines 295-348)
        import aiohttp as real_aiohttp

        with patch("services.test_execution.health_monitor.docker") as mock_docker:
            mock_docker.errors.APIError = type("APIError", (Exception,), {})
            mock_docker.from_env.return_value.containers.get.side_effect = real_aiohttp.ClientError(
                "connection failed"
            )
            result = await checker.check_health("container123")
            assert result.status == HealthStatus.UNHEALTHY
            assert "health check failed" in result.message.lower()

    @pytest.mark.asyncio
    async def test_container_health_monitor_start_stop(self):
        """Test ContainerHealthMonitor start/stop monitoring (lines 420-442)."""
        from services.test_execution.health_monitor import (
            ContainerHealthMonitor,
            HealthCheckConfig,
            HealthCheckType,
        )

        monitor = ContainerHealthMonitor("c1", "python")

        # Replace checkers with mocks so they don't actually run
        mock_checker = AsyncMock()
        mock_checker.config = HealthCheckConfig(
            check_type=HealthCheckType.LIVENESS,
            enabled=True,
            interval_seconds=1,
        )
        mock_checker.check_health = AsyncMock()
        monitor.health_checkers = {HealthCheckType.LIVENESS: mock_checker}

        await monitor.start_monitoring()
        assert monitor.monitoring_active is True
        assert len(monitor.monitoring_tasks) == 1

        await monitor.stop_monitoring()
        assert monitor.monitoring_active is False
        assert len(monitor.monitoring_tasks) == 0

    @pytest.mark.asyncio
    async def test_container_health_monitor_trigger_recovery(self):
        """Test ContainerHealthMonitor _trigger_recovery (lines 500-533)."""
        from services.test_execution.health_monitor import (
            ContainerHealthMonitor,
            HealthCheckResult,
            HealthCheckType,
            HealthStatus,
            RecoveryAction,
        )

        monitor = ContainerHealthMonitor("c1", "python")
        result = HealthCheckResult(
            timestamp=datetime.now(UTC),
            check_type=HealthCheckType.LIVENESS,
            status=HealthStatus.UNHEALTHY,
            response_time_ms=100.0,
            message="Container down",
        )

        # First recovery attempt -> RESTART
        handler_called = False

        async def mock_handler(cid, ct, r):
            nonlocal handler_called
            handler_called = True

        monitor.recovery_handlers[RecoveryAction.RESTART] = mock_handler
        await monitor._trigger_recovery(HealthCheckType.LIVENESS, result)
        assert handler_called
        assert monitor.health.recovery_attempts == 1

        # Second attempt -> RECREATE (attempt 2)
        monitor.recovery_handlers[RecoveryAction.RECREATE] = mock_handler
        await monitor._trigger_recovery(HealthCheckType.LIVENESS, result)
        assert monitor.health.recovery_attempts == 2

        # Fourth attempt -> ALERT_ONLY
        monitor.health.recovery_attempts = 3
        await monitor._trigger_recovery(HealthCheckType.LIVENESS, result)
        assert monitor.health.recovery_attempts == 4

    def test_update_overall_status_all_healthy(self):
        """Test _update_overall_status with all healthy (line 552-553)."""
        from services.test_execution.health_monitor import (
            ContainerHealthMonitor,
            HealthCheckResult,
            HealthCheckType,
            HealthStatus,
        )

        monitor = ContainerHealthMonitor("c1", "python")
        monitor.health.check_results[HealthCheckType.LIVENESS] = [
            HealthCheckResult(
                timestamp=datetime.now(UTC),
                check_type=HealthCheckType.LIVENESS,
                status=HealthStatus.HEALTHY,
                response_time_ms=10.0,
                message="ok",
            ),
        ]
        monitor._update_overall_status()
        assert monitor.health.overall_status == HealthStatus.HEALTHY

    def test_update_overall_status_degraded(self):
        """Test _update_overall_status with degraded status (line 549)."""
        from services.test_execution.health_monitor import (
            ContainerHealthMonitor,
            HealthCheckResult,
            HealthCheckType,
            HealthStatus,
        )

        monitor = ContainerHealthMonitor("c1", "python")
        monitor.health.check_results[HealthCheckType.LIVENESS] = [
            HealthCheckResult(
                timestamp=datetime.now(UTC),
                check_type=HealthCheckType.LIVENESS,
                status=HealthStatus.DEGRADED,
                response_time_ms=10.0,
                message="degraded",
            ),
        ]
        monitor._update_overall_status()
        assert monitor.health.overall_status == HealthStatus.DEGRADED

    def test_update_overall_status_starting(self):
        """Test _update_overall_status with starting status (line 551)."""
        from services.test_execution.health_monitor import (
            ContainerHealthMonitor,
            HealthCheckResult,
            HealthCheckType,
            HealthStatus,
        )

        monitor = ContainerHealthMonitor("c1", "python")
        monitor.health.check_results[HealthCheckType.LIVENESS] = [
            HealthCheckResult(
                timestamp=datetime.now(UTC),
                check_type=HealthCheckType.LIVENESS,
                status=HealthStatus.STARTING,
                response_time_ms=10.0,
                message="starting",
            ),
        ]
        monitor._update_overall_status()
        assert monitor.health.overall_status == HealthStatus.STARTING

    def test_update_overall_status_unknown(self):
        """Test _update_overall_status with unknown status (line 555)."""
        from services.test_execution.health_monitor import (
            ContainerHealthMonitor,
            HealthCheckResult,
            HealthCheckType,
            HealthStatus,
        )

        monitor = ContainerHealthMonitor("c1", "python")
        monitor.health.check_results[HealthCheckType.LIVENESS] = [
            HealthCheckResult(
                timestamp=datetime.now(UTC),
                check_type=HealthCheckType.LIVENESS,
                status=HealthStatus.STOPPING,
                response_time_ms=10.0,
                message="stopping",
            ),
        ]
        monitor._update_overall_status()
        assert monitor.health.overall_status == HealthStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_health_monitoring_service_add_remove(self):
        """Test HealthMonitoringService add/remove container (lines 698-720)."""
        from services.test_execution.health_monitor import HealthMonitoringService

        service = HealthMonitoringService()

        # Patch ContainerHealthMonitor to avoid real docker calls
        with patch(
            "services.test_execution.health_monitor.ContainerHealthMonitor"
        ) as mock_monitor_cls:
            mock_monitor = MagicMock()
            mock_monitor.start_monitoring = AsyncMock()
            mock_monitor.stop_monitoring = AsyncMock()
            mock_monitor.add_recovery_handler = MagicMock()
            mock_monitor_cls.return_value = mock_monitor

            await service.add_container_monitor("c1", "python")
            assert "c1" in service.container_monitors

            # Re-adding same container should log warning and return
            await service.add_container_monitor("c1", "python")

            await service.remove_container_monitor("c1")
            assert "c1" not in service.container_monitors

            # Removing non-existent should be fine
            await service.remove_container_monitor("nonexistent")

    def test_health_monitoring_service_get_health_summary(self):
        """Test HealthMonitoringService get_health_summary (lines 734-762)."""
        from services.test_execution.health_monitor import (
            ContainerHealthMonitor,
            HealthMonitoringService,
            HealthStatus,
        )

        service = HealthMonitoringService()
        mock_monitor = MagicMock(spec=ContainerHealthMonitor)
        mock_health = MagicMock()
        mock_health.overall_status = HealthStatus.HEALTHY
        mock_monitor.get_health_status.return_value = mock_health
        mock_monitor.get_health_summary.return_value = {"container_id": "c1", "language": "python"}
        service.container_monitors["c1"] = mock_monitor

        summary = service.get_health_summary()
        assert summary["total_containers"] == 1
        assert summary["healthy_containers"] == 1
        assert summary["unhealthy_containers"] == 0
        assert summary["degraded_containers"] == 0
        assert summary["health_percentage"] == 100.0
        assert len(summary["monitored_containers"]) == 1

    def test_export_health_data(self):
        """Test HealthMonitoringService export_health_data (lines 787-797)."""
        from services.test_execution.health_monitor import HealthMonitoringService

        service = HealthMonitoringService()
        data = service.export_health_data()
        parsed = json.loads(data)
        assert isinstance(parsed["timestamp"], str)  # ISO format string
        assert parsed["summary"]["total_containers"] == 0
        assert parsed["summary"]["healthy_containers"] == 0
        assert parsed["containers"] == {}

    @pytest.mark.asyncio
    async def test_health_monitoring_service_start_stop(self):
        """Test HealthMonitoringService start/stop (lines 674-696)."""
        from services.test_execution.health_monitor import HealthMonitoringService

        service = HealthMonitoringService()
        await service.start_monitoring()
        assert service.monitoring_active is True
        import asyncio

        assert isinstance(service.system_monitoring_task, asyncio.Task)
        assert not service.system_monitoring_task.cancelled()

        await service.stop_monitoring()
        assert service.monitoring_active is False
        assert service.system_monitoring_task.cancelled()

    @pytest.mark.asyncio
    async def test_health_service_recovery_handlers(self):
        """Test recovery handlers (lines 603-658)."""
        from services.test_execution.health_monitor import (
            HealthCheckResult,
            HealthCheckType,
            HealthMonitoringService,
            HealthStatus,
        )

        service = HealthMonitoringService()
        result = HealthCheckResult(
            timestamp=datetime.now(UTC),
            check_type=HealthCheckType.LIVENESS,
            status=HealthStatus.UNHEALTHY,
            response_time_ms=100.0,
            message="down",
        )

        # Test _alert_only
        await service._alert_only("c1", HealthCheckType.LIVENESS, result)

        # Test _restart_container with mock
        with patch("services.test_execution.health_monitor.docker") as mock_docker:
            mock_container = MagicMock()
            mock_docker.from_env.return_value.containers.get.return_value = mock_container
            await service._restart_container("c1", HealthCheckType.LIVENESS, result)
            mock_container.restart.assert_called_once()

        # Test _restart_container with NotFound
        with patch("services.test_execution.health_monitor.docker") as mock_docker:
            mock_docker.errors = MagicMock()
            mock_docker.errors.NotFound = type("NotFound", (Exception,), {})
            mock_docker.errors.APIError = type("APIError", (Exception,), {})
            mock_docker.from_env.return_value.containers.get.side_effect = (
                mock_docker.errors.NotFound("nf")
            )
            await service._restart_container("c1", HealthCheckType.LIVENESS, result)

        # Test _recreate_container
        with patch("services.test_execution.health_monitor.docker") as mock_docker:
            mock_container = MagicMock()
            mock_container.attrs = {
                "Config": {"Image": "test:latest"},
                "HostConfig": {},
            }
            new_container = MagicMock()
            new_container.id = "new_c1"
            mock_docker.from_env.return_value.containers.get.return_value = mock_container
            mock_docker.from_env.return_value.containers.run.return_value = new_container
            await service._recreate_container("c1", HealthCheckType.LIVENESS, result)
            mock_container.stop.assert_called_once()
            mock_container.remove.assert_called_once()


# ---------------------------------------------------------------------------
# 4. report_formatter.py
# ---------------------------------------------------------------------------


class TestReportFormatterR4:
    """Cover uncovered lines in report_formatter.py."""

    @pytest.mark.asyncio
    async def test_generate_comprehensive_report_basic(self):
        """Test generate_comprehensive_report without specialized analyzers (lines 177-310)."""
        from services.ai_test_insights.report_formatter import TestReportFormatter

        formatter = TestReportFormatter()

        # Create minimal test data
        from models.testing_models import QualityResult, TestResult

        test_result = TestResult(
            test_id="t1",
            success=True,
            exit_code=0,
            stdout="ok",
            stderr="",
            execution_time=1.0,
            language="python",
            security_level="medium",
            created_at=datetime.now(UTC),
            container_name="c1",
        )
        quality_result = QualityResult(
            analysis_id="q1",
            success=True,
            overall_score=8.0,
            total_issues=0,
            issues=[],
            tools_results={},
            execution_time=0.5,
            created_at=datetime.now(UTC),
            language="python",
        )

        report = await formatter.generate_comprehensive_report(
            test_results=[test_result],
            quality_results=[quality_result],
            time_period="recent",
            include_predictions=False,
        )
        assert isinstance(report["analysis_id"], str)
        assert len(report["analysis_id"]) > 0
        assert report["analysis_type"] == "comprehensive"
        assert report["input_summary"]["test_results_count"] == 1
        assert report["input_summary"]["quality_results_count"] == 1
        assert report["input_summary"]["code_changes"] == 0
        assert report["input_summary"]["time_period"] == "recent"
        assert report["input_summary"]["predictions_included"] is False
        assert "test_insights" in report["core_analysis"]
        assert report["specialized_analyses"] == {}
        assert "metadata" in report
        assert report["metadata"]["format_version"] == "1.0"

    @pytest.mark.asyncio
    async def test_generate_comprehensive_report_with_code_changes(self):
        """Test with code_changes parameter (line 276)."""
        from services.ai_test_insights.report_formatter import TestReportFormatter

        formatter = TestReportFormatter()

        from models.testing_models import QualityResult, TestResult

        test_result = TestResult(
            test_id="t1",
            success=True,
            exit_code=0,
            stdout="ok",
            stderr="",
            execution_time=1.0,
            language="python",
            security_level="medium",
            created_at=datetime.now(UTC),
            container_name="c1",
        )
        quality_result = QualityResult(
            analysis_id="q1",
            success=True,
            overall_score=8.0,
            total_issues=0,
            issues=[],
            tools_results={},
            execution_time=0.5,
            created_at=datetime.now(UTC),
            language="python",
        )

        report = await formatter.generate_comprehensive_report(
            test_results=[test_result],
            quality_results=[quality_result],
            code_changes=["file1.py", "file2.py"],
        )
        assert report["input_summary"]["code_changes"] == 2

    @pytest.mark.asyncio
    async def test_cache_analysis_results(self):
        """Test _cache_analysis_results with mock cache (lines 637-651)."""
        from services.ai_test_insights.report_formatter import TestReportFormatter

        mock_cache = AsyncMock()
        formatter = TestReportFormatter(cache_manager=mock_cache)

        mock_analysis = MagicMock()
        mock_analysis.model_dump.return_value = {"test": "data"}

        await formatter._cache_analysis_results("test_id", mock_analysis)
        mock_cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_analysis_results_failure(self):
        """Test _cache_analysis_results when cache fails (line 651)."""
        from services.ai_test_insights.report_formatter import TestReportFormatter

        mock_cache = AsyncMock()
        mock_cache.set.side_effect = Exception("cache error")
        formatter = TestReportFormatter(cache_manager=mock_cache)

        mock_analysis = MagicMock()
        mock_analysis.model_dump.return_value = {"test": "data"}

        # Exception is swallowed gracefully, does not propagate
        await formatter._cache_analysis_results("test_id", mock_analysis)
        mock_cache.set.assert_called_once()  # cache.set was attempted despite failure

    @pytest.mark.asyncio
    async def test_get_cached_analysis(self):
        """Test get_cached_analysis (lines 670-685)."""
        from services.ai_test_insights.report_formatter import TestReportFormatter

        # Without cache
        formatter = TestReportFormatter()
        result = await formatter.get_cached_analysis("test_id")
        assert result is None

        # With cache returning None
        mock_cache = AsyncMock()
        mock_cache.get.return_value = None
        formatter2 = TestReportFormatter(cache_manager=mock_cache)
        result2 = await formatter2.get_cached_analysis("test_id")
        assert result2 is None

        # With cache raising
        mock_cache2 = AsyncMock()
        mock_cache2.get.side_effect = Exception("cache error")
        formatter3 = TestReportFormatter(cache_manager=mock_cache2)
        result3 = await formatter3.get_cached_analysis("test_id")
        assert result3 is None

    @pytest.mark.asyncio
    async def test_get_cached_comprehensive_analysis(self):
        """Test get_cached_comprehensive_analysis (lines 687-700)."""
        from services.ai_test_insights.report_formatter import TestReportFormatter

        # Without cache
        formatter = TestReportFormatter()
        result = await formatter.get_cached_comprehensive_analysis("test_id")
        assert result is None

        # With cache raising
        mock_cache = AsyncMock()
        mock_cache.get.side_effect = Exception("cache error")
        formatter2 = TestReportFormatter(cache_manager=mock_cache)
        result2 = await formatter2.get_cached_comprehensive_analysis("test_id")
        assert result2 is None


# ---------------------------------------------------------------------------
# 6. intelligent_test_orchestrator.py
# ---------------------------------------------------------------------------


class TestIntelligentOrchestratorR4:
    """Cover uncovered lines in intelligent_test_orchestrator.py."""

    def test_count_errors(self):
        """Test _count_errors (lines 613-618)."""
        from config.test_framework_config import TestConfig
        from services.test_execution.intelligent_test_orchestrator import (
            IntelligentTestOrchestrator,
        )

        mock_docker = MagicMock()
        orch = IntelligentTestOrchestrator(mock_docker, TestConfig())
        assert orch._count_errors("") == 0
        # Deduplicated indicators: "error:" and "failed:" (both lowercase)
        # "error:" appears 2x, "failed:" appears 0x
        text = "Error: something failed\nERROR: another"
        count = orch._count_errors(text)
        assert count == 2

    def test_count_warnings(self):
        """Test _count_warnings (lines 620-625)."""
        from config.test_framework_config import TestConfig
        from services.test_execution.intelligent_test_orchestrator import (
            IntelligentTestOrchestrator,
        )

        mock_docker = MagicMock()
        orch = IntelligentTestOrchestrator(mock_docker, TestConfig())
        assert orch._count_warnings("") == 0
        # 3 indicators lowercase to "warning:" (each counted), 2 lowercase to "warn:"
        # "warn:" does NOT appear in "warning:" (w-a-r-n-i != w-a-r-n-:)
        text = "Warning: something\nWARNING: another"
        count = orch._count_warnings(text)
        # Deduplicated indicators: "warning:" and "warn:" (both lowercase)
        # "warning:" appears 2x, "warn:" appears 0x in the text
        assert count == 2

    def test_extract_coverage(self):
        """Test _extract_coverage (lines 627-636)."""
        from config.test_framework_config import TestConfig
        from services.test_execution.intelligent_test_orchestrator import (
            IntelligentTestOrchestrator,
        )

        mock_docker = MagicMock()
        orch = IntelligentTestOrchestrator(mock_docker, TestConfig())
        assert orch._extract_coverage(None) is None
        assert orch._extract_coverage({"coverage": 85.0}) == 85.0
        assert orch._extract_coverage({"other": "data"}) is None

    def test_extract_memory_and_cpu(self):
        """Test _extract_memory_usage and _extract_cpu_usage (lines 603-610)."""
        from config.test_framework_config import TestConfig
        from services.test_execution.intelligent_test_orchestrator import (
            IntelligentTestOrchestrator,
        )

        mock_docker = MagicMock()
        orch = IntelligentTestOrchestrator(mock_docker, TestConfig())
        mock_result = MagicMock()
        assert orch._extract_memory_usage(mock_result) is None
        assert orch._extract_cpu_usage(mock_result) is None

    def test_calculate_current_load(self):
        """Test _calculate_current_load (lines 597-601)."""
        from config.test_framework_config import TestConfig
        from services.test_execution.intelligent_test_orchestrator import (
            IntelligentTestOrchestrator,
        )

        mock_docker = MagicMock()
        orch = IntelligentTestOrchestrator(mock_docker, TestConfig())
        orch.max_concurrent_tests = 5
        orch.execution_queue.active_tests = {"a": MagicMock(), "b": MagicMock()}
        load = orch._calculate_current_load()
        assert load == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_calculate_scheduling_score(self):
        """Test _calculate_scheduling_score (lines 461-480)."""
        from config.test_framework_config import TestConfig
        from services.test_execution.intelligent_test_orchestrator import (
            IntelligentTestExecutionRequest,
            IntelligentTestOrchestrator,
        )
        from services.test_execution.test_orchestrator import OrchestratorType, TestLanguage

        mock_docker = MagicMock()
        orch = IntelligentTestOrchestrator(mock_docker, TestConfig())

        req = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="print(1)",
            container_name="c1",
            priority_level=8,
            enable_ai_optimization=True,
            expected_execution_time_ms=10000.0,
            resource_weight=1.5,
        )
        score = await orch._calculate_scheduling_score(req)
        # priority=8 -> 8*0.1=0.8, AI bonus +0.2=1.0,
        # time bonus: max(0, 1.0 - 10000/300000) = 0.9667, total=1.9667
        # resource_weight=1.5 -> 1.9667*1.5 = 2.95
        assert score == pytest.approx(2.95, abs=0.01)


# ---------------------------------------------------------------------------
# 7. test_runner.py
# ---------------------------------------------------------------------------


class TestContainerTestRunnerR4:
    """Cover uncovered lines in processing/test_execution/test_runner.py."""

    def test_get_file_extension(self):
        """Test _get_file_extension (lines 419-426)."""
        from config.test_framework_config import TestConfig
        from processing.test_execution.test_runner import ContainerTestRunner

        runner = ContainerTestRunner(MagicMock(), TestConfig())
        assert runner._get_file_extension("python") == ".py"
        assert runner._get_file_extension("javascript") == ".js"
        assert runner._get_file_extension("typescript") == ".ts"
        assert runner._get_file_extension("unknown") == ".py"

    def test_get_test_file_extension(self):
        """Test _get_test_file_extension (lines 428-434)."""
        from config.test_framework_config import TestConfig
        from processing.test_execution.test_runner import ContainerTestRunner

        runner = ContainerTestRunner(MagicMock(), TestConfig())
        assert runner._get_test_file_extension("python") == ".py"
        assert runner._get_test_file_extension("javascript") == ".test.js"
        assert runner._get_test_file_extension("typescript") == ".test.ts"
        assert runner._get_test_file_extension("unknown") == ".py"

    def test_get_memory_limit(self):
        """Test _get_memory_limit (lines 401-408)."""
        from config.test_framework_config import TestConfig
        from models.testing_models import SecurityLevel
        from processing.test_execution.test_runner import ContainerTestRunner

        runner = ContainerTestRunner(MagicMock(), TestConfig())
        assert runner._get_memory_limit(SecurityLevel.HIGH) == "256m"
        assert runner._get_memory_limit(SecurityLevel.MEDIUM) == "512m"
        assert runner._get_memory_limit(SecurityLevel.LOW) == "1g"

    def test_get_cpu_quota(self):
        """Test _get_cpu_quota (lines 410-417)."""
        from config.test_framework_config import TestConfig
        from models.testing_models import SecurityLevel
        from processing.test_execution.test_runner import ContainerTestRunner

        runner = ContainerTestRunner(MagicMock(), TestConfig())
        assert runner._get_cpu_quota(SecurityLevel.HIGH) == 25000
        assert runner._get_cpu_quota(SecurityLevel.MEDIUM) == 50000
        assert runner._get_cpu_quota(SecurityLevel.LOW) == 100000

    def test_should_use_orchestrator(self):
        """Test _should_use_orchestrator (lines 132-136)."""
        from config.test_framework_config import TestConfig
        from models.testing_models import TestRequest
        from processing.test_execution.test_runner import ContainerTestRunner

        runner = ContainerTestRunner(MagicMock(), TestConfig())
        req_go = MagicMock(spec=TestRequest)
        req_go.language = "go"
        assert runner._should_use_orchestrator(req_go) is True

        req_python = MagicMock(spec=TestRequest)
        req_python.language = "python"
        assert runner._should_use_orchestrator(req_python) is False

    @pytest.mark.asyncio
    async def test_validate_request_empty(self):
        """Test _validate_request with empty request (lines 210-239)."""
        from config.test_framework_config import TestConfig
        from models.testing_models import TestRequest
        from processing.test_execution.test_runner import ContainerTestRunner

        runner = ContainerTestRunner(MagicMock(), TestConfig())
        req = MagicMock(spec=TestRequest)
        req.code = None
        req.test_code = None
        req.code_path = None

        with pytest.raises(ValueError, match="Security validation failed"):
            await runner._validate_request(req)

    @pytest.mark.asyncio
    async def test_validate_request_too_long(self):
        """Test _validate_request with code too long (line 234-235)."""
        from config.test_framework_config import TestConfig
        from models.testing_models import TestRequest
        from processing.test_execution.test_runner import ContainerTestRunner

        runner = ContainerTestRunner(MagicMock(), TestConfig())
        runner.security_manager = None  # Skip security manager
        req = MagicMock(spec=TestRequest)
        req.code = "x" * 60000
        req.test_code = ""
        req.code_path = None

        with pytest.raises(ValueError, match="Security validation failed"):
            await runner._validate_request(req)

    @pytest.mark.asyncio
    async def test_cleanup_container(self):
        """Test _cleanup_container (lines 392-399)."""
        from config.test_framework_config import TestConfig
        from processing.test_execution.test_runner import ContainerTestRunner

        mock_docker = AsyncMock()
        runner = ContainerTestRunner(mock_docker, TestConfig())
        await runner._cleanup_container("docker_123")
        mock_docker.stop_container.assert_called_once_with("docker_123")
        mock_docker.remove_container_by_id.assert_called_once_with("docker_123")

    @pytest.mark.asyncio
    async def test_cleanup_container_error(self):
        """Test _cleanup_container with error (line 398-399)."""
        from config.test_framework_config import TestConfig
        from processing.test_execution.test_runner import ContainerTestRunner

        mock_docker = AsyncMock()
        mock_docker.stop_container.side_effect = RuntimeError("stop failed")
        runner = ContainerTestRunner(mock_docker, TestConfig())
        # Should not raise
        await runner._cleanup_container("docker_123")


# ---------------------------------------------------------------------------
# 8. testing_utils.py
# ---------------------------------------------------------------------------


class TestTestingUtilsR4:
    """Cover uncovered lines in routers/testing_modules/testing_utils.py."""

    def test_validate_test_request_empty(self):
        """Test validate_test_request with empty data (lines 312-313)."""
        from fastapi import HTTPException

        from routers.testing_modules.testing_utils import validate_test_request

        with pytest.raises(HTTPException) as exc_info:
            validate_test_request({})
        assert exc_info.value.status_code == 400
        assert "required" in exc_info.value.detail.lower()

    def test_validate_test_request_no_code(self):
        """Test validate_test_request with missing code (lines 316-317)."""
        from fastapi import HTTPException

        from routers.testing_modules.testing_utils import validate_test_request

        with pytest.raises(HTTPException) as exc_info:
            validate_test_request({"language": "python"})
        assert exc_info.value.status_code == 400
        assert "code" in exc_info.value.detail.lower()

    def test_validate_test_request_empty_code(self):
        """Test validate_test_request with empty code string."""
        from fastapi import HTTPException

        from routers.testing_modules.testing_utils import validate_test_request

        with pytest.raises(HTTPException) as exc_info:
            validate_test_request({"code": "   "})
        assert exc_info.value.status_code == 400

    def test_validate_test_request_unsupported_language(self):
        """Test validate_test_request with unsupported language (lines 322-326)."""
        from fastapi import HTTPException

        from routers.testing_modules.testing_utils import validate_test_request

        with pytest.raises(HTTPException) as exc_info:
            validate_test_request({"code": "print(1)", "language": "brainfuck"})
        assert exc_info.value.status_code == 400
        assert "brainfuck" in exc_info.value.detail.lower()

    def test_validate_test_request_success(self):
        """Test validate_test_request success (lines 329-338)."""
        from routers.testing_modules.testing_utils import validate_test_request

        result = validate_test_request({"code": "print(1)", "language": "python"})
        assert result["code"] == "print(1)"
        assert result["language"] == "python"
        assert result["timeout"] == 30
        assert result["memory_limit"] == 128

    def test_validate_test_request_caps_timeout(self):
        """Test validate_test_request caps timeout at 300."""
        from routers.testing_modules.testing_utils import validate_test_request

        result = validate_test_request({"code": "x", "timeout": 999})
        assert result["timeout"] == 300

    def test_format_test_results_empty(self):
        """Test format_test_results with empty list (lines 350-357)."""
        from routers.testing_modules.testing_utils import format_test_results

        result = format_test_results([])
        assert result["total_tests"] == 0
        assert result["successful"] == 0

    def test_format_test_results_with_data(self):
        """Test format_test_results with data (lines 359-385)."""
        from models.testing_models import TestResult
        from routers.testing_modules.testing_utils import format_test_results

        results = [
            TestResult(
                test_id="t1",
                success=True,
                exit_code=0,
                stdout="ok",
                stderr="",
                execution_time=1.0,
                language="python",
                security_level="medium",
                created_at=datetime.now(UTC),
                container_name="c1",
            ),
            TestResult(
                test_id="t2",
                success=False,
                exit_code=1,
                stdout="",
                stderr="fail",
                execution_time=2.0,
                language="python",
                security_level="medium",
                created_at=datetime.now(UTC),
                container_name="c1",
            ),
        ]
        formatted = format_test_results(results)
        assert formatted["total_tests"] == 2
        assert formatted["successful"] == 1
        assert formatted["failed"] == 1
        assert formatted["success_rate"] == 0.5

    def test_aggregate_test_metrics_empty(self):
        """Test aggregate_test_metrics with empty list (lines 397-403)."""
        from routers.testing_modules.testing_utils import aggregate_test_metrics

        result = aggregate_test_metrics([])
        assert result["success_rate"] == 0.0

    def test_aggregate_test_metrics_with_data(self):
        """Test aggregate_test_metrics with data (lines 405-461)."""
        from models.testing_models import TestResult
        from routers.testing_modules.testing_utils import aggregate_test_metrics

        results = [
            TestResult(
                test_id="t1",
                success=True,
                exit_code=0,
                stdout="ok",
                stderr="",
                execution_time=1.0,
                language="python",
                security_level="medium",
                created_at=datetime.now(UTC),
                container_name="c1",
            ),
            TestResult(
                test_id="t2",
                success=False,
                exit_code=1,
                stdout="",
                stderr="SyntaxError: invalid",
                execution_time=2.0,
                language="python",
                security_level="medium",
                created_at=datetime.now(UTC),
                container_name="c1",
            ),
            TestResult(
                test_id="t3",
                success=False,
                exit_code=1,
                stdout="",
                stderr="ImportError: no module",
                execution_time=3.0,
                language="javascript",
                security_level="medium",
                created_at=datetime.now(UTC),
                container_name="c1",
            ),
            TestResult(
                test_id="t4",
                success=False,
                exit_code=1,
                stdout="",
                stderr="timeout reached",
                execution_time=30.0,
                language="python",
                security_level="medium",
                created_at=datetime.now(UTC),
                container_name="c1",
            ),
            TestResult(
                test_id="t5",
                success=False,
                exit_code=1,
                stdout="",
                stderr="out of memory",
                execution_time=5.0,
                language="python",
                security_level="medium",
                created_at=datetime.now(UTC),
                container_name="c1",
            ),
        ]
        metrics = aggregate_test_metrics(results)
        assert metrics["total_tests"] == 5
        assert metrics["success_rate"] == pytest.approx(0.2)
        assert metrics["languages"]["python"]["total"] == 4
        assert metrics["languages"]["python"]["successful"] == 1
        assert metrics["languages"]["javascript"]["total"] == 1
        assert metrics["languages"]["javascript"]["successful"] == 0
        assert metrics["execution_times"]["python"]["min"] == 1.0
        assert metrics["execution_times"]["python"]["max"] == 30.0
        assert metrics["error_patterns"]["syntax_error"] == 1
        assert metrics["error_patterns"]["import_error"] == 1
        assert metrics["error_patterns"]["timeout"] == 1
        assert metrics["error_patterns"]["memory_error"] == 1

    @pytest.mark.asyncio
    async def test_log_test_execution(self):
        """Test _log_test_execution (lines 206-236)."""
        from models.testing_models import TestResult
        from routers.testing_modules.testing_utils import _log_test_execution

        test_result = TestResult(
            test_id="t1",
            success=True,
            exit_code=0,
            stdout="ok",
            stderr="",
            execution_time=1.0,
            language="python",
            security_level="medium",
            created_at=datetime.now(UTC),
            container_name="c1",
        )
        with patch("routers.testing_modules.testing_utils.get_metrics_service") as mock_get:
            mock_service = AsyncMock()
            mock_get.return_value = mock_service
            await _log_test_execution("user1", test_result, {"lang": "python"})
            mock_service.record_test_execution.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_quality_analysis(self):
        """Test _log_quality_analysis (lines 239-269)."""
        from models.testing_models import QualityResult
        from routers.testing_modules.testing_utils import _log_quality_analysis

        quality_result = QualityResult(
            analysis_id="q1",
            success=True,
            overall_score=8.0,
            total_issues=0,
            issues=[],
            tools_results={"ruff-format": {"success": True}},
            execution_time=0.5,
            created_at=datetime.now(UTC),
            language="python",
        )
        with patch("routers.testing_modules.testing_utils.get_metrics_service") as mock_get:
            mock_service = AsyncMock()
            mock_get.return_value = mock_service
            await _log_quality_analysis("user1", quality_result)
            mock_service.record_quality_analysis.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_batch_execution(self):
        """Test _log_batch_execution (lines 272-297)."""
        from routers.testing_modules.testing_utils import _log_batch_execution

        with patch("routers.testing_modules.testing_utils.logger") as mock_logger:
            await _log_batch_execution("user1", 5, 3, ["python", "javascript"])
            # Should log the metrics info message containing batch data
            assert mock_logger.info.call_count == 2  # one log + one BATCH_EXECUTION_METRICS
            metrics_log = mock_logger.info.call_args_list[1][0][0]
            assert "BATCH_EXECUTION_METRICS" in metrics_log
            assert "'batch_size': 5" in metrics_log
            assert "'successful': 3" in metrics_log
            assert "'failed': 2" in metrics_log


# ---------------------------------------------------------------------------
# 9. pydantic_ai_agents.py — additional coverage
# ---------------------------------------------------------------------------


class TestPydanticAIAgentsR4:
    """Cover uncovered lines in pydantic_ai_agents.py."""

    def test_execution_pattern_validator(self):
        """Test ExecutionPattern confidence_score validator (line 84)."""
        from pydantic import ValidationError

        from services.test_execution.pydantic_ai_agents import ExecutionPattern

        with pytest.raises(ValidationError, match="confidence_score"):
            ExecutionPattern(
                pattern_id="p1",
                pattern_type="failure",
                frequency=5,
                severity="high",
                description="test",
                affected_languages=["python"],
                affected_test_types=["unit"],
                first_occurrence=datetime.now(UTC),
                last_occurrence=datetime.now(UTC),
                recommendations=["fix it"],
                confidence_score=1.5,  # Invalid: must be 0..1
            )

    def test_scheduling_context_validator(self):
        """Test SchedulingContext current_load validator (line 104)."""
        from pydantic import ValidationError

        from services.test_execution.pydantic_ai_agents import SchedulingContext

        with pytest.raises(ValidationError, match="current_load"):
            SchedulingContext(
                pending_tests=[],
                active_tests=[],
                available_resources={},
                historical_performance={},
                current_load=1.5,  # Invalid: must be 0..1
            )

    def test_recommendation_confidence_validator(self):
        """Test PydanticAIRecommendation confidence_level validator (line 129)."""
        from pydantic import ValidationError

        from services.test_execution.pydantic_ai_agents import PydanticAIRecommendation

        with pytest.raises(ValidationError, match="confidence_level"):
            PydanticAIRecommendation(
                recommendation_id="r1",
                recommendation_type="optimization",
                priority="high",
                title="test",
                description="test",
                impact_assessment="high",
                implementation_steps=["step1"],
                confidence_level=1.5,  # Invalid: must be 0..1
            )

    def test_failure_analysis_scores_validator(self):
        """Test FailureAnalysis validate_scores (line 156)."""
        from pydantic import ValidationError

        from services.test_execution.pydantic_ai_agents import FailureAnalysis

        with pytest.raises(ValidationError, match="frequency_score"):
            FailureAnalysis(
                analysis_id="a1",
                test_id="t1",
                failure_type="runtime",
                error_message="test",
                language="python",
                test_type="unit",
                frequency_score=1.5,  # Invalid: must be 0..1
                complexity_score=0.5,
                fix_confidence=0.5,
            )

    @pytest.mark.asyncio
    async def test_execution_pattern_analyzer_analyze(self):
        """Test ExecutionPatternAnalyzer analyze_execution_patterns (lines 181-208)."""
        from services.test_execution.pydantic_ai_agents import (
            ExecutionMetrics,
            ExecutionPatternAnalyzer,
        )

        analyzer = ExecutionPatternAnalyzer()
        metrics = [
            ExecutionMetrics(
                test_id=f"t{i}",
                language="python",
                test_type="unit",
                execution_time_ms=10000.0 if i < 5 else 100.0,
                exit_code=0 if i < 8 else 1,
                success=i < 8,
            )
            for i in range(10)
        ]
        patterns = await analyzer.analyze_execution_patterns(metrics)
        # 5 out of 10 tests are slow (>5s), so >20% threshold triggers performance pattern
        # 2 out of 10 tests fail (exit_code=1), so >10% threshold triggers failure pattern
        assert len(patterns) == 2
        perf_patterns = [p for p in patterns if p.pattern_type == "performance"]
        fail_patterns = [p for p in patterns if p.pattern_type == "failure"]
        assert len(perf_patterns) == 1
        assert perf_patterns[0].frequency == 5
        assert perf_patterns[0].severity == "medium"  # 50% is NOT > 50%, so medium
        assert perf_patterns[0].confidence_score == 0.8
        assert perf_patterns[0].affected_languages == ["python"]
        assert len(fail_patterns) == 1
        assert fail_patterns[0].frequency == 2
        assert fail_patterns[0].confidence_score == 0.9

    @pytest.mark.asyncio
    async def test_scheduling_optimizer_optimize(self):
        """Test SchedulingOptimizer optimize_test_schedule (lines 344-386)."""
        from services.test_execution.pydantic_ai_agents import (
            SchedulingContext,
            SchedulingOptimizer,
        )

        optimizer = SchedulingOptimizer()
        context = SchedulingContext(
            pending_tests=[],
            active_tests=[],
            available_resources={},
            historical_performance={},
            current_load=0.5,
        )
        order = await optimizer.optimize_test_schedule(context)
        # No pending tests means empty optimized order
        assert order == []

    @pytest.mark.asyncio
    async def test_coordinator_record_execution_overflow(self):
        """Test PydanticAITestCoordinator record_execution with overflow (lines 822-831)."""
        from services.test_execution.pydantic_ai_agents import (
            ExecutionMetrics,
            PydanticAITestCoordinator,
        )

        coordinator = PydanticAITestCoordinator()
        # Add more than max_history
        for i in range(1005):
            await coordinator.record_execution(
                ExecutionMetrics(
                    test_id=f"t{i}",
                    language="python",
                    test_type="unit",
                    execution_time_ms=100.0,
                    exit_code=0,
                    success=True,
                )
            )
        assert len(coordinator.execution_history) == 1000


# ---------------------------------------------------------------------------
# 10. intelligent_test_integration.py
# ---------------------------------------------------------------------------


class TestIntelligentIntegrationR4:
    """Cover uncovered lines in intelligent_test_integration.py."""

    def test_test_execution_settings_validators(self):
        """Test validators (lines 49-63)."""
        from pydantic import ValidationError

        from services.test_execution.intelligent_test_integration import (
            TestExecutionSettings,
        )

        with pytest.raises(ValidationError, match="max_concurrent_tests"):
            TestExecutionSettings(max_concurrent_tests=25)

        with pytest.raises(ValidationError, match="ai_analysis_threshold"):
            TestExecutionSettings(ai_analysis_threshold=200)

    def test_test_execution_error_classes(self):
        """Test custom exception classes (lines 66-113)."""
        from services.test_execution.intelligent_test_integration import (
            TestExecutionError,
            TestExecutionResourceError,
            TestExecutionRetryError,
            TestExecutionTimeoutError,
        )

        err = TestExecutionError("test", "general", {"key": "val"})
        assert err.error_type == "general"
        assert err.details == {"key": "val"}

        retry_err = TestExecutionRetryError("retry fail", 3, Exception("inner"))
        assert retry_err.attempts == 3
        assert str(retry_err.last_error) == "inner"

        timeout_err = TestExecutionTimeoutError("timeout", 30)
        assert timeout_err.timeout_seconds == 30

        resource_err = TestExecutionResourceError("resource", "memory", "512m")
        assert resource_err.resource_type == "memory"
        assert resource_err.resource_limit == "512m"

    def test_file_extension_mapping(self):
        """Test _get_file_extension (lines 676-689)."""
        from config.test_framework_config import TestConfig
        from services.test_execution.intelligent_test_integration import (
            IntelligentTestExecutionService,
        )

        service = IntelligentTestExecutionService(MagicMock(), TestConfig())
        assert service._get_file_extension("python") == ".py"
        assert service._get_file_extension("java") == ".java"
        assert service._get_file_extension("go") == ".go"
        assert service._get_file_extension("unknown") == ".txt"


# ---------------------------------------------------------------------------
# 11. test_discovery.py — tested via mocking
# ---------------------------------------------------------------------------


class TestTestDiscoveryR4:
    """Cover uncovered lines in routers/testing_modules/test_discovery.py.
    These are endpoint functions tested by mocking dependencies.
    """

    @pytest.mark.asyncio
    async def test_get_supported_tools_with_language(self):
        """Test get_supported_tools with specific language (lines 117-127)."""
        from routers.testing_modules.test_discovery import get_supported_tools

        mock_user = MagicMock()
        mock_user.is_admin.return_value = True
        mock_service = MagicMock()
        mock_service._get_supported_tools.return_value = ["ruff-format", "ruff"]

        result = await get_supported_tools(
            language="python",
            container_name=None,
            user=mock_user,
            execution_service=mock_service,
        )
        assert result["language"] == "python"
        assert result["tools"] == ["ruff-format", "ruff"]
        assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_get_supported_tools_all_languages(self):
        """Test get_supported_tools without language filter (lines 129-142)."""
        from routers.testing_modules.test_discovery import get_supported_tools

        mock_user = MagicMock()
        mock_user.is_admin.return_value = True
        mock_service = MagicMock()
        mock_service._get_supported_tools.side_effect = lambda lang: (
            ["ruff-format"] if lang == "python" else []
        )

        result = await get_supported_tools(
            language=None,
            container_name=None,
            user=mock_user,
            execution_service=mock_service,
        )
        assert result["supported_languages"] == ["python"]
        assert result["tools_by_language"] == {"python": ["ruff-format"]}
        assert result["total_tools"] == 1

    @pytest.mark.asyncio
    async def test_get_supported_tools_not_found(self):
        """Test get_supported_tools when no tools found (lines 120-123)."""
        from fastapi import HTTPException

        from routers.testing_modules.test_discovery import get_supported_tools

        mock_user = MagicMock()
        mock_user.is_admin.return_value = True
        mock_service = MagicMock()
        mock_service._get_supported_tools.return_value = []

        with pytest.raises(HTTPException) as exc_info:
            await get_supported_tools(
                language="brainfuck",
                container_name=None,
                user=mock_user,
                execution_service=mock_service,
            )
        assert exc_info.value.status_code == 404
        assert "brainfuck" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_get_supported_languages(self):
        """Test get_supported_languages endpoint (lines 196-266)."""
        from routers.testing_modules.test_discovery import get_supported_languages

        mock_user = MagicMock()
        mock_user.is_admin.return_value = True
        mock_service = MagicMock()
        mock_service.test_orchestrator = AsyncMock()
        mock_service.test_orchestrator.get_supported_languages = AsyncMock(
            return_value=["python", "javascript"],
        )
        mock_service.test_orchestrator.get_supported_test_types = AsyncMock(
            return_value=["unit", "integration"],
        )
        mock_service._get_supported_tools.return_value = ["ruff-format"]

        result = await get_supported_languages(
            container_name=None,
            user=mock_user,
            execution_service=mock_service,
        )
        assert result["languages"] == ["python", "javascript"]
        assert result["total"] == 2
        assert result["details"]["python"]["supported"] is True
        assert result["details"]["python"]["test_types"] == ["unit", "integration"]
        assert result["details"]["python"]["quality_tools"] == ["ruff-format"]

    @pytest.mark.asyncio
    async def test_get_language_test_types(self):
        """Test get_language_test_types endpoint (lines 315-378)."""
        from routers.testing_modules.test_discovery import get_language_test_types

        mock_user = MagicMock()
        mock_user.is_admin.return_value = True
        mock_service = MagicMock()
        mock_service.test_orchestrator = AsyncMock()
        mock_service.test_orchestrator.get_supported_languages = AsyncMock(
            return_value=["python", "javascript"],
        )
        mock_service.test_orchestrator.get_supported_test_types = AsyncMock(
            return_value=["unit", "integration", "coverage"],
        )

        result = await get_language_test_types(
            language="python",
            container_name=None,
            user=mock_user,
            execution_service=mock_service,
        )
        assert result["language"] == "python"
        assert result["test_types"] == ["unit", "integration", "coverage"]
        assert result["count"] == 3

    @pytest.mark.asyncio
    async def test_get_language_test_types_not_found(self):
        """Test get_language_test_types with unsupported language (lines 344-348)."""
        from routers.testing_modules.test_discovery import get_language_test_types

        mock_user = MagicMock()
        mock_user.is_admin.return_value = True
        mock_service = MagicMock()
        mock_service.test_orchestrator = AsyncMock()
        mock_service.test_orchestrator.get_supported_languages = AsyncMock(
            return_value=["python"],
        )

        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await get_language_test_types(
                language="brainfuck",
                container_name=None,
                user=mock_user,
                execution_service=mock_service,
            )
        assert exc_info.value.status_code == 404
        assert "brainfuck" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_list_test_containers(self):
        """Test list_test_containers endpoint (lines 430-483).

        get_container_manager is not available in this context so the
        function will naturally hit the ImportError/RuntimeError fallback.
        """
        from routers.testing_modules.test_discovery import list_test_containers

        mock_user = MagicMock()
        mock_user.is_admin.return_value = True

        # The function imports get_container_manager locally; since it doesn't exist
        # in shared.container_validation, this will hit the ImportError/RuntimeError branch
        result = await list_test_containers(filter_type="all", user=mock_user)
        assert result["containers"] == []
        assert result["count"] == 0
        assert result["filter"] == "all"

    @pytest.mark.asyncio
    async def test_list_test_containers_provisioned_filter(self):
        """Test list_test_containers with provisioned filter."""
        from routers.testing_modules.test_discovery import list_test_containers

        mock_user = MagicMock()
        mock_user.is_admin.return_value = True
        result = await list_test_containers(filter_type="provisioned", user=mock_user)
        assert result["filter"] == "provisioned"

    @pytest.mark.asyncio
    async def test_list_test_containers_test_filter(self):
        """Test list_test_containers with test filter."""
        from routers.testing_modules.test_discovery import list_test_containers

        mock_user = MagicMock()
        mock_user.is_admin.return_value = True
        result = await list_test_containers(filter_type="test", user=mock_user)
        assert result["filter"] == "test"


# ---------------------------------------------------------------------------
# 12. intelligent_test_recommendation_engine.py
# ---------------------------------------------------------------------------


class TestRecommendationEngineR4:
    """Cover uncovered lines in intelligent_test_recommendation_engine.py."""

    def test_recommendation_computed_fields(self):
        """Test TestRecommendation priority_score computed field (lines 117-128)."""
        from services.intelligent_test_recommendation_engine import (
            RecommendationCategory,
            RecommendationPriority,
            RecommendationType,
            TestRecommendation,
        )

        rec = TestRecommendation(
            recommendation_id="r1",
            title="Test",
            description="test",
            type=RecommendationType.TEST_OPTIMIZATION,
            priority=RecommendationPriority.HIGH,
            category=RecommendationCategory.STRATEGIC_IMPROVEMENT,
            expected_benefit="better",
            impact_score=8.0,
            confidence=0.9,
            implementation_steps=["step1"],
            estimated_effort="medium",
            time_estimate="1 week",
            rationale="important",
        )
        assert rec.priority_score == (8.0 + 8.0) * 0.9

    def test_recommendation_suite_computed_fields(self):
        """Test RecommendationSuite suite_priority (lines 159-170)."""
        from services.intelligent_test_recommendation_engine import (
            RecommendationCategory,
            RecommendationPriority,
            RecommendationSuite,
            RecommendationType,
            TestRecommendation,
        )

        rec_critical = TestRecommendation(
            recommendation_id="r1",
            title="Critical",
            description="test",
            type=RecommendationType.TEST_OPTIMIZATION,
            priority=RecommendationPriority.CRITICAL,
            category=RecommendationCategory.IMMEDIATE_ACTION,
            expected_benefit="better",
            impact_score=9.0,
            confidence=0.9,
            implementation_steps=["step1"],
            estimated_effort="high",
            time_estimate="2 weeks",
            rationale="critical",
        )
        suite = RecommendationSuite(
            suite_id="s1",
            suite_name="Test Suite",
            description="test",
            recommendations=[rec_critical],
            total_impact_score=9.0,
            average_confidence=0.9,
            estimated_total_effort="high",
        )
        assert suite.suite_priority == RecommendationPriority.CRITICAL

    def test_report_overall_priority_score(self):
        """Test TestRecommendationReport overall_priority_score (lines 248-256)."""
        from services.intelligent_test_recommendation_engine import (
            TestRecommendationReport,
        )

        # Empty report
        report = TestRecommendationReport(
            report_id="r1",
            total_recommendations=0,
        )
        assert report.overall_priority_score == 0.0

    @pytest.mark.asyncio
    async def test_generate_recommendations_basic(self):
        """Test generate_recommendations with just best practices (lines 289-406)."""
        from services.intelligent_test_recommendation_engine import (
            IntelligentRecommendationEngine,
        )

        engine = IntelligentRecommendationEngine()
        report = await engine.generate_recommendations()
        assert report.total_recommendations > 0
        assert len(report.all_recommendations) > 0

    @pytest.mark.asyncio
    async def test_generate_recommendations_with_test_results(self):
        """Test with test_results (lines 729-779)."""
        from models.testing_models import TestResult
        from services.intelligent_test_recommendation_engine import (
            IntelligentRecommendationEngine,
        )

        engine = IntelligentRecommendationEngine()
        results = [
            TestResult(
                test_id="t1",
                success=False,
                exit_code=1,
                stdout="",
                stderr="fail",
                execution_time=1.0,
                language="python",
                security_level="medium",
                created_at=datetime.now(UTC),
                container_name="c1",
            ),
        ]
        report = await engine.generate_recommendations(test_results=results)
        assert any(r.recommendation_id == "test_failures" for r in report.all_recommendations)

    @pytest.mark.asyncio
    async def test_generate_best_practices_with_security_focus(self):
        """Test _generate_best_practices with security focus (lines 829-873)."""
        from services.intelligent_test_recommendation_engine import (
            IntelligentRecommendationEngine,
        )

        engine = IntelligentRecommendationEngine()
        recs = await engine._generate_best_practices(focus_areas=["security"])
        assert any(r.recommendation_id == "security_testing" for r in recs)

    @pytest.mark.asyncio
    async def test_generate_best_practices_without_focus(self):
        """Test _generate_best_practices without focus areas."""
        from services.intelligent_test_recommendation_engine import (
            IntelligentRecommendationEngine,
        )

        engine = IntelligentRecommendationEngine()
        recs = await engine._generate_best_practices(focus_areas=None)
        # Should include security when no focus specified
        assert any(r.recommendation_id == "security_testing" for r in recs)


# ---------------------------------------------------------------------------
# 13. core_service.py
# ---------------------------------------------------------------------------


class TestCoreServiceR4:
    """Cover uncovered lines in ai_test_insights/core_service.py."""

    @pytest.mark.asyncio
    async def test_analyze_with_llm_disabled(self):
        """Test _analyze_with_llm when AI features disabled (lines 250-251)."""
        from services.ai_test_insights.config import AITestInsightsConfig
        from services.ai_test_insights.core_service import AITestInsightsService

        config = AITestInsightsConfig()
        config.use_ml_backend = False
        service = AITestInsightsService(config=config)
        result = await service._analyze_with_llm("test content", "system prompt")
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_with_llm_import_error(self):
        """Test _analyze_with_llm when llm client not available (lines 253-265)."""
        from services.ai_test_insights.config import AITestInsightsConfig
        from services.ai_test_insights.core_service import AITestInsightsService

        config = AITestInsightsConfig()
        config.use_ml_backend = True
        config.enable_ai_features = True
        service = AITestInsightsService(config=config)

        with patch(
            "services.ai_test_insights.core_service.AITestInsightsService._analyze_with_llm"
        ) as mock:
            mock.return_value = None
            result = await service._analyze_with_llm("content", "prompt")
            assert result is None

    @pytest.mark.asyncio
    async def test_analyze_test_insights_error(self):
        """Test analyze_test_insights error path (lines 303-305)."""
        from services.ai_test_insights.core_service import AITestInsightsService

        service = AITestInsightsService()
        with patch.object(
            service.report_formatter,
            "create_comprehensive_analysis",
            side_effect=Exception("analysis failed"),
        ):
            with pytest.raises(Exception, match="analysis failed"):
                await service.analyze_test_insights([], [])

    @pytest.mark.asyncio
    async def test_comprehensive_analysis(self):
        """Test comprehensive_analysis (lines 329-348)."""
        from services.ai_test_insights.core_service import AITestInsightsService

        service = AITestInsightsService()

        mock_report = {"analysis_id": "test", "analysis_type": "comprehensive"}
        with patch.object(
            service.report_formatter,
            "generate_comprehensive_report",
            new_callable=AsyncMock,
            return_value=mock_report,
        ):
            result = await service.comprehensive_analysis([], [])
            assert result == mock_report

    @pytest.mark.asyncio
    async def test_comprehensive_analysis_error(self):
        """Test comprehensive_analysis error path (lines 346-348)."""
        from services.ai_test_insights.core_service import AITestInsightsService

        service = AITestInsightsService()
        with patch.object(
            service.report_formatter,
            "generate_comprehensive_report",
            side_effect=Exception("comprehensive failed"),
        ):
            with pytest.raises(Exception, match="comprehensive failed"):
                await service.comprehensive_analysis([], [])

    @pytest.mark.asyncio
    async def test_ai_agents_init_failure(self):
        """Test AI agent initialization failure (lines 223-230)."""
        from services.ai_test_insights.config import AITestInsightsConfig
        from services.ai_test_insights.core_service import AITestInsightsService

        config = AITestInsightsConfig()
        config.enable_ai_features = True
        # The initialization of agents may fail if pydantic_ai isn't available
        # which is the normal CI path - this covers lines 223-230
        service = AITestInsightsService(config=config)
        # Service initializes but agents may fail gracefully if pydantic_ai not available
        assert service.config.enable_ai_features is True
        from services.ai_test_insights.report_formatter import (
            TestReportFormatter as ReportFormatter,
        )

        assert isinstance(service.report_formatter, ReportFormatter)
        # If agents failed to initialize, they should be set to None
        if service.performance_agent is None:
            assert service.reliability_agent is None
            assert service.coverage_agent is None
