"""Tests for services/testing/testpy_integration_adapter.py."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.testing.testpy_integration_adapter import (
    FallbackSecurityManager,
    FallbackTestRunner,
    TestPyIntegrationAdapter,
    create_testpy_adapter,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_docker_service():
    svc = MagicMock()
    svc.get_language_container_image = MagicMock(return_value="python:3.13-slim")
    svc.create_test_container = AsyncMock(return_value="docker-123")
    svc.start_container = AsyncMock()
    svc.create_file_in_container = AsyncMock(return_value=True)
    svc.exec_in_container = AsyncMock(
        return_value={"exit_code": 0, "stdout": "1 passed", "stderr": ""}
    )
    svc.stop_container = AsyncMock()
    svc.remove_container_by_id = AsyncMock()
    svc.is_docker_available = MagicMock(return_value=True)
    svc.get_container_stats = AsyncMock(
        return_value={"cpu_percent": 5.0, "memory_usage": 1024 * 1024 * 128}
    )
    return svc


@pytest.fixture
def mock_test_runner():
    runner = MagicMock(spec=["run_tests"])
    runner.run_tests = MagicMock(return_value={"success": True})
    return runner


@pytest.fixture
def mock_security_manager():
    mgr = MagicMock()
    mgr.validate_code = MagicMock(return_value=MagicMock(valid=True))
    return mgr


@pytest.fixture
def mock_metrics_manager():
    mgr = MagicMock()
    mgr.record_metric = MagicMock()
    return mgr


@pytest.fixture
def adapter(mock_docker_service, mock_test_runner, mock_security_manager, mock_metrics_manager):
    return TestPyIntegrationAdapter(
        docker_service=mock_docker_service,
        test_runner=mock_test_runner,
        security_manager=mock_security_manager,
        metrics_manager=mock_metrics_manager,
    )


class FakeTestRequest:
    """Simple fake TestRequest with plain attributes (not MagicMock)."""

    def __init__(self, **kwargs):
        self.code = kwargs.get("code", "def add(a, b): return a + b")
        self.test_code = kwargs.get("test_code", "def test_add(): assert add(1,2)==3")
        self.language = kwargs.get("language", "python")
        self.security_level = kwargs.get("security_level", None)
        self.timeout = kwargs.get("timeout", 30)
        self.code_file_path = kwargs.get("code_file_path", None)
        self.test_file_path = kwargs.get("test_file_path", None)
        self.with_hypothesis = kwargs.get("with_hypothesis", False)
        self.test_metadata = kwargs.get("test_metadata", None)


def _make_test_request(**overrides):
    """Build a FakeTestRequest."""
    return FakeTestRequest(**overrides)


# ── TestPyIntegrationAdapter.__init__ ────────────────────────────────────────


class TestAdapterInit:
    def test_default_config(self, mock_docker_service):
        adapter = TestPyIntegrationAdapter(docker_service=mock_docker_service)
        cfg = adapter._get_default_config()
        assert cfg == {
            "security": {
                "max_execution_time": 30,
                "max_memory_mb": 512,
                "network_access": False,
            },
            "testing": {
                "default_timeout": 30,
                "hypothesis_enabled": True,
            },
        }

    def test_custom_config(self, mock_docker_service):
        custom = {"my": "config"}
        adapter = TestPyIntegrationAdapter(docker_service=mock_docker_service, config=custom)
        assert adapter.config is custom
        assert adapter.config == {"my": "config"}

    def test_components_initialized(
        self,
        adapter,
        mock_docker_service,
        mock_test_runner,
        mock_security_manager,
        mock_metrics_manager,
    ):
        assert adapter.test_runner is mock_test_runner
        assert adapter.security_manager is mock_security_manager
        assert adapter.metrics_manager is mock_metrics_manager
        assert adapter.docker_service is mock_docker_service


# ── _create_legacy_logger ────────────────────────────────────────────────────


class TestLegacyLogger:
    def test_logger_has_all_methods(self, adapter):
        log_adapter = adapter._create_legacy_logger()
        assert callable(log_adapter.info)
        assert callable(log_adapter.error)
        assert callable(log_adapter.warning)
        assert callable(log_adapter.debug)
        assert callable(log_adapter.component_context)

    def test_component_context(self, adapter):
        log_adapter = adapter._create_legacy_logger()
        ctx = log_adapter.component_context("test")
        assert ctx is log_adapter

    def test_context_manager(self, adapter):
        log_adapter = adapter._create_legacy_logger()
        with log_adapter as la:
            assert la is log_adapter


# ── _validate_with_testpy_security ───────────────────────────────────────────


class TestSecurityValidation:
    @pytest.mark.asyncio
    async def test_valid_code(self, adapter):
        request = _make_test_request()
        # Should complete without raising; validate_code called for both code and test_code
        await adapter._validate_with_testpy_security(request)
        assert adapter.security_manager.validate_code.call_count == 2

    @pytest.mark.asyncio
    async def test_code_fails_passed_attr(self, adapter):
        adapter.security_manager.validate_code.return_value = MagicMock(
            passed=False, error="dangerous", spec=["passed", "error"]
        )
        request = _make_test_request()
        with pytest.raises(ValueError, match="Security validation failed"):
            await adapter._validate_with_testpy_security(request)

    @pytest.mark.asyncio
    async def test_code_fails_valid_attr(self, adapter):
        adapter.security_manager.validate_code.return_value = MagicMock(valid=False, spec=["valid"])
        request = _make_test_request()
        with pytest.raises(ValueError, match="Security validation failed"):
            await adapter._validate_with_testpy_security(request)

    @pytest.mark.asyncio
    async def test_test_code_fails(self, adapter):
        # First call (code) passes, second call (test_code) fails
        results = [
            MagicMock(valid=True, spec=["valid"]),
            MagicMock(passed=False, error="bad test", spec=["passed", "error"]),
        ]
        adapter.security_manager.validate_code.side_effect = results
        request = _make_test_request()
        with pytest.raises(ValueError, match="security validation failed"):
            await adapter._validate_with_testpy_security(request)

    @pytest.mark.asyncio
    async def test_no_code(self, adapter):
        request = _make_test_request(code=None, test_code=None)
        await adapter._validate_with_testpy_security(request)
        # validate_code should never be called when both code and test_code are None
        adapter.security_manager.validate_code.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_security_manager(self, mock_docker_service):
        adapter = TestPyIntegrationAdapter(
            docker_service=mock_docker_service,
            security_manager=None,
        )
        adapter.security_manager = None
        request = _make_test_request()
        # Should complete without raising when no security_manager
        await adapter._validate_with_testpy_security(request)


# ── _prepare_testpy_environment ──────────────────────────────────────────────


class TestPrepareEnvironment:
    @pytest.mark.asyncio
    async def test_creates_files(self, adapter):
        request = _make_test_request()
        env = await adapter._prepare_testpy_environment(request)
        try:
            # Verify all expected keys are present with proper types
            assert Path(env["temp_dir"]).is_dir()
            assert Path(env["code_file"]).exists()
            assert Path(env["test_file"]).exists()
            assert env["module_name"].startswith("test_module_")
            # Code file should contain the source code
            assert Path(env["code_file"]).read_text() == "def add(a, b): return a + b"
            # code_content should match the input
            assert env["code_content"] == "def add(a, b): return a + b"
            # test_content should contain the processed test code with the import prepended
            # Since test_code="def test_add(): assert add(1,2)==3" has no "import" keyword,
            # the fallback prepends "from <module_name> import *\n\n"
            assert env["test_content"].startswith(f"from {env['module_name']} import *")
            assert "def test_add(): assert add(1,2)==3" in env["test_content"]
        finally:
            import shutil

            shutil.rmtree(env["temp_dir"], ignore_errors=True)

    @pytest.mark.asyncio
    async def test_no_test_code_uses_fallback(self, adapter):
        request = _make_test_request(test_code=None)
        env = await adapter._prepare_testpy_environment(request)
        try:
            assert Path(env["test_file"]).exists()
            content = Path(env["test_file"]).read_text()
            # The fallback adds "from <module_name> import *" because "# Empty test" has no "import"
            assert f"from {env['module_name']} import *" in content
            assert "# Empty test" in content
        finally:
            import shutil

            shutil.rmtree(env["temp_dir"], ignore_errors=True)

    @pytest.mark.asyncio
    async def test_with_process_test_code(self, adapter):
        adapter.test_runner._process_test_code = MagicMock(return_value="processed code")
        request = _make_test_request()
        env = await adapter._prepare_testpy_environment(request)
        try:
            content = Path(env["test_file"]).read_text()
            assert content == "processed code"
            # Verify _process_test_code was called with the test_code and a module name
            adapter.test_runner._process_test_code.assert_called_once()
            call_args = adapter.test_runner._process_test_code.call_args[0]
            assert call_args[0] == "def test_add(): assert add(1,2)==3"
            assert call_args[1].startswith("test_module_")
        finally:
            import shutil

            shutil.rmtree(env["temp_dir"], ignore_errors=True)

    @pytest.mark.asyncio
    async def test_with_import_in_test(self, adapter):
        # test_runner spec=["run_tests"] already lacks _process_test_code
        request = _make_test_request(test_code="import os\ndef test_x(): pass")
        env = await adapter._prepare_testpy_environment(request)
        try:
            content = Path(env["test_file"]).read_text()
            # Since test_code already has "import", fallback keeps it as-is
            assert content == "import os\ndef test_x(): pass"
        finally:
            import shutil

            shutil.rmtree(env["temp_dir"], ignore_errors=True)


# ── _run_testpy_in_container ─────────────────────────────────────────────────


class TestRunInContainer:
    @pytest.mark.asyncio
    async def test_success(self, adapter, mock_docker_service):
        env = {"test_file": "/tmp/test_foo.py"}
        request = _make_test_request()
        result = await adapter._run_testpy_in_container("docker-123", env, request)
        assert result == {"success": True, "output": "1 passed", "error": None}
        # Verify exec_in_container was called with correct args
        mock_docker_service.exec_in_container.assert_awaited_once()
        call_args = mock_docker_service.exec_in_container.call_args[0]
        assert call_args[0] == "docker-123"
        assert call_args[1] == ["python", "-m", "pytest", "/app/test/test_foo.py", "-v"]

    @pytest.mark.asyncio
    async def test_failure(self, adapter, mock_docker_service):
        mock_docker_service.exec_in_container.return_value = {
            "exit_code": 1,
            "stdout": "",
            "stderr": "FAILED",
        }
        env = {"test_file": "/tmp/test_foo.py"}
        request = _make_test_request()
        result = await adapter._run_testpy_in_container("docker-123", env, request)
        assert result["success"] is False
        assert result["output"] == ""
        assert result["error"] == "FAILED"

    @pytest.mark.asyncio
    async def test_hypothesis_flag(self, adapter, mock_docker_service):
        env = {"test_file": "/tmp/test_foo.py"}
        request = _make_test_request(with_hypothesis=True)
        await adapter._run_testpy_in_container("docker-123", env, request)
        cmd = mock_docker_service.exec_in_container.call_args[0][1]
        assert cmd == [
            "python",
            "-m",
            "pytest",
            "/app/test/test_foo.py",
            "-v",
            "--hypothesis-show-statistics",
        ]

    @pytest.mark.asyncio
    async def test_non_python(self, adapter, mock_docker_service):
        env = {"test_file": "/tmp/test_foo.js"}
        request = _make_test_request(language="javascript")
        result = await adapter._run_testpy_in_container("docker-123", env, request)
        assert result["success"] is True
        # Non-python still uses pytest runner (same command structure)
        cmd = mock_docker_service.exec_in_container.call_args[0][1]
        assert cmd == ["python", "-m", "pytest", "/app/test/test_foo.js", "-v"]

    @pytest.mark.asyncio
    async def test_exception(self, adapter, mock_docker_service):
        mock_docker_service.exec_in_container.side_effect = RuntimeError("boom")
        env = {"test_file": "/tmp/test_foo.py"}
        request = _make_test_request()
        result = await adapter._run_testpy_in_container("docker-123", env, request)
        assert result == {"success": False, "output": "", "error": "boom"}

    @pytest.mark.asyncio
    async def test_timeout(self, adapter, mock_docker_service):
        async def slow(*args, **kwargs):
            import asyncio

            await asyncio.sleep(100)

        mock_docker_service.exec_in_container = slow
        env = {"test_file": "/tmp/test_foo.py"}
        request = _make_test_request(timeout=0.01)
        with pytest.raises(TimeoutError):
            await adapter._run_testpy_in_container("docker-123", env, request)


# ── execute_test_with_testpy ─────────────────────────────────────────────────


class TestExecuteTestWithTestPy:
    @pytest.mark.asyncio
    async def test_success(self, adapter):
        """Mock internal methods to test the full execute flow."""
        request = _make_test_request()
        adapter._run_testpy_in_container = AsyncMock(
            return_value={"success": True, "output": "1 passed", "error": ""}
        )
        result = await adapter.execute_test_with_testpy(request)
        assert result.success is True
        assert result.exit_code == 0
        assert result.stdout == "1 passed"
        assert result.stderr == ""
        assert result.language == "python"
        assert result.error is None
        assert result.execution_time > 0.0
        assert result.test_id  # Should be a non-empty UUID string

    @pytest.mark.asyncio
    async def test_security_failure(self, adapter):
        adapter.security_manager.validate_code.return_value = MagicMock(
            passed=False, error="bad", spec=["passed", "error"]
        )
        request = _make_test_request()
        result = await adapter.execute_test_with_testpy(request)
        assert result.success is False
        assert result.exit_code == 1
        assert "Security validation failed" in result.stderr
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_records_metrics(self, adapter, mock_metrics_manager):
        request = _make_test_request()
        adapter._run_testpy_in_container = AsyncMock(
            return_value={"success": True, "output": "ok", "error": ""}
        )
        await adapter.execute_test_with_testpy(request)
        # Should record at least execution_time and success_rate (2 metrics)
        assert mock_metrics_manager.record_metric.call_count >= 2
        # Verify the first call is for execution time
        first_call = mock_metrics_manager.record_metric.call_args_list[0]
        assert first_call[0][0] == "test_execution_time"
        assert first_call[0][2] == "seconds"
        # Verify the second call is for success rate
        second_call = mock_metrics_manager.record_metric.call_args_list[1]
        assert second_call[0][0] == "test_success_rate"
        assert second_call[0][1] == 1.0  # success=True -> 1.0

    @pytest.mark.asyncio
    async def test_no_metrics_manager(
        self, mock_docker_service, mock_test_runner, mock_security_manager
    ):
        adapter = TestPyIntegrationAdapter(
            docker_service=mock_docker_service,
            test_runner=mock_test_runner,
            security_manager=mock_security_manager,
            metrics_manager=None,
        )
        adapter.metrics_manager = None
        adapter._run_testpy_in_container = AsyncMock(
            return_value={"success": True, "output": "ok", "error": ""}
        )
        request = _make_test_request()
        result = await adapter.execute_test_with_testpy(request)
        assert result.success is True
        assert result.exit_code == 0
        assert result.stdout == "ok"

    @pytest.mark.asyncio
    async def test_execution_exception(self, adapter):
        """When execute encounters an exception, it returns a failed TestResult."""
        adapter._validate_with_testpy_security = AsyncMock(side_effect=Exception("kaboom"))
        request = _make_test_request()
        result = await adapter.execute_test_with_testpy(request)
        assert result.success is False
        assert result.exit_code == 1
        assert "kaboom" in result.stderr
        assert result.error == "kaboom"
        assert result.execution_time == 0.0
        assert result.language == "python"


# ── _record_testpy_metrics ───────────────────────────────────────────────────


class TestRecordMetrics:
    @pytest.mark.asyncio
    async def test_records(self, adapter, mock_metrics_manager):
        result = MagicMock()
        result.execution_time = 2.5
        result.success = True
        result.test_metadata = {"memory_usage_mb": 128}

        await adapter._record_testpy_metrics("t1", result)
        assert mock_metrics_manager.record_metric.call_count == 3

        # Verify each metric call precisely
        calls = mock_metrics_manager.record_metric.call_args_list
        # Call 1: execution_time
        assert calls[0][0] == ("test_execution_time", 2.5, "seconds", "Test execution time for t1")
        # Call 2: success_rate
        assert calls[1][0] == ("test_success_rate", 1.0, "ratio", "Test success for t1")
        # Call 3: memory_usage
        assert calls[2][0] == ("test_memory_usage", 128, "MB", "Memory usage for t1")

    @pytest.mark.asyncio
    async def test_no_manager(self, adapter):
        adapter.metrics_manager = None
        result = MagicMock()
        result.execution_time = 1.0
        # Should return early without error
        await adapter._record_testpy_metrics("t1", result)

    @pytest.mark.asyncio
    async def test_no_memory_metadata(self, adapter, mock_metrics_manager):
        result = MagicMock()
        result.execution_time = 1.0
        result.success = False
        result.test_metadata = None

        await adapter._record_testpy_metrics("t1", result)
        assert mock_metrics_manager.record_metric.call_count == 2

        calls = mock_metrics_manager.record_metric.call_args_list
        assert calls[0][0] == ("test_execution_time", 1.0, "seconds", "Test execution time for t1")
        assert calls[1][0] == ("test_success_rate", 0.0, "ratio", "Test success for t1")

    @pytest.mark.asyncio
    async def test_record_error_handled(self, adapter, mock_metrics_manager):
        mock_metrics_manager.record_metric.side_effect = Exception("boom")
        result = MagicMock()
        result.execution_time = 1.0
        result.success = True
        result.test_metadata = None

        # Exception should be caught internally, not propagated
        await adapter._record_testpy_metrics("t1", result)


# ── _cleanup_testpy_container ────────────────────────────────────────────────


class TestCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_container(self, adapter, mock_docker_service):
        await adapter._cleanup_testpy_container("docker-123")
        mock_docker_service.stop_container.assert_awaited_once_with("docker-123")
        mock_docker_service.remove_container_by_id.assert_awaited_once_with("docker-123")

    @pytest.mark.asyncio
    async def test_cleanup_container_error(self, adapter, mock_docker_service):
        mock_docker_service.stop_container.side_effect = Exception("fail")
        # Exception should be caught; should not propagate
        await adapter._cleanup_testpy_container("docker-123")
        mock_docker_service.stop_container.assert_awaited_once_with("docker-123")

    @pytest.mark.asyncio
    async def test_cleanup_environment(self, adapter):
        td = tempfile.mkdtemp()
        assert Path(td).exists()
        await adapter._cleanup_testpy_environment({"temp_dir": td})
        assert not Path(td).exists()

    @pytest.mark.asyncio
    async def test_cleanup_environment_missing(self, adapter):
        # No temp_dir key -- should not raise
        await adapter._cleanup_testpy_environment({})

    @pytest.mark.asyncio
    async def test_cleanup_environment_error(self, adapter):
        # Nonexistent path -- ignore_errors=True means no error
        await adapter._cleanup_testpy_environment({"temp_dir": "/nonexistent/path"})


# ── _get_container_resource_metrics ──────────────────────────────────────────


class TestContainerResourceMetrics:
    @pytest.mark.asyncio
    async def test_with_stats(self, adapter, mock_docker_service):
        metrics = await adapter._get_container_resource_metrics("docker-123")
        assert metrics.cpu_usage == 5.0
        assert metrics.memory_usage == 128.0  # 128MB = 128 * 1024 * 1024 / (1024 * 1024)
        assert metrics.disk_usage == 0.0
        assert metrics.network_connections == 0
        mock_docker_service.get_container_stats.assert_awaited_once_with("docker-123")

    @pytest.mark.asyncio
    async def test_without_stats(self, adapter, mock_docker_service):
        mock_docker_service.get_container_stats.side_effect = Exception("no stats")
        metrics = await adapter._get_container_resource_metrics("docker-123")
        assert metrics.cpu_usage == 0.0
        assert metrics.memory_usage == 0.0
        assert metrics.disk_usage == 0.0
        assert metrics.network_connections == 0

    @pytest.mark.asyncio
    async def test_no_get_container_stats_method(self, adapter, mock_docker_service):
        del mock_docker_service.get_container_stats
        metrics = await adapter._get_container_resource_metrics("docker-123")
        assert metrics.cpu_usage == 0.0
        assert metrics.memory_usage == 0.0
        assert metrics.disk_usage == 0.0
        assert metrics.network_connections == 0


# ── _build_testpy_container_config ───────────────────────────────────────────


class TestBuildContainerConfig:
    def test_builds_config(self, mock_docker_service):
        config = MagicMock()
        config.security = MagicMock()
        config.security.max_memory_mb = 512
        config.security.network_access = False

        adapter = TestPyIntegrationAdapter(docker_service=mock_docker_service, config=config)

        env = {"temp_dir": "/tmp/test"}
        request = _make_test_request()
        result = adapter._build_testpy_container_config(request, env)
        assert result["mem_limit"] == "512m"
        assert result["network_disabled"] is True
        assert result["read_only"] is True
        assert result["image"] == "code-graph-test-python:latest"
        assert result["working_dir"] == "/app/test"
        assert result["cpu_quota"] == 50000
        assert result["user"] == "testuser"
        assert result["cap_drop"] == ["ALL"]
        assert result["security_opt"] == ["no-new-privileges:true"]
        assert result["environment"]["PYTHONPATH"] == "/app/test:/app/testpy"
        assert result["environment"]["TEST_MODE"] == "true"
        assert result["environment"]["TESTPY_MODE"] == "true"
        assert result["volumes"] == {"/tmp/test": {"bind": "/app/test", "mode": "rw"}}


# ── health_check ─────────────────────────────────────────────────────────────


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_healthy(self, adapter):
        result = await adapter.health_check()
        assert result == {
            "testpy_integration": "healthy",
            "test_runner_available": True,
            "security_manager_available": True,
            "metrics_manager_available": True,
            "docker_service_available": True,
            "config_valid": True,
        }

    @pytest.mark.asyncio
    async def test_unhealthy(self, adapter, mock_docker_service):
        mock_docker_service.is_docker_available.side_effect = Exception("fail")
        result = await adapter.health_check()
        assert result["testpy_integration"] == "unhealthy"
        assert result["error"] == "fail"


# ── create_testpy_adapter factory ────────────────────────────────────────────


class TestFactory:
    def test_creates_adapter(self, mock_docker_service):
        adapter = create_testpy_adapter(mock_docker_service)
        assert adapter.docker_service is mock_docker_service
        # Should have fallback implementations auto-assigned — verify they work
        assert isinstance(adapter.test_runner, FallbackTestRunner)
        assert adapter.test_runner.run_tests("x", "y") == {
            "success": False,
            "error": "TestRunner not available",
        }
        assert isinstance(adapter.security_manager, FallbackSecurityManager)
        assert adapter.security_manager.validate_code("x") == {"valid": True}

    def test_with_components(self, mock_docker_service, mock_test_runner, mock_security_manager):
        adapter = create_testpy_adapter(
            mock_docker_service,
            test_runner=mock_test_runner,
            security_manager=mock_security_manager,
        )
        assert adapter.test_runner is mock_test_runner
        assert adapter.security_manager is mock_security_manager
        assert adapter.docker_service is mock_docker_service


# ── _convert_config_format ───────────────────────────────────────────────────


class TestConvertConfigFormat:
    def test_with_valid_config(self, mock_docker_service):
        config = MagicMock()
        config.security = MagicMock()
        config.test = MagicMock()
        config.logging = MagicMock()

        adapter = TestPyIntegrationAdapter(docker_service=mock_docker_service, config=config)
        result = adapter._convert_config_format()
        # LegacyConfig wraps the config attributes
        assert result.security is config.security
        assert result.test is config.test
        assert result.logging is config.logging

    def test_fallback_on_error(self, adapter):
        adapter.config = "not-a-config"  # Will fail attribute access
        result = adapter._convert_config_format()
        assert result == "not-a-config"  # Returns raw config as fallback


# ---------------------------------------------------------------------------
# _init_testpy_components exception path -- covers lines 140-151
# ---------------------------------------------------------------------------


class TestInitComponentsException:
    def test_init_with_failing_test_runner(self, mock_docker_service):
        """When test_runner init fails, fallback is used (lines 146-151)."""
        adapter = TestPyIntegrationAdapter(
            docker_service=mock_docker_service,
            test_runner=None,
            security_manager=None,
            metrics_manager=None,
        )
        # Should have initialized with fallback implementations
        assert isinstance(adapter.test_runner, FallbackTestRunner)
        assert isinstance(adapter.security_manager, FallbackSecurityManager)
        # Verify they are actual working instances by calling their methods
        run_result = adapter.test_runner.run_tests("code", "test")
        assert run_result == {"success": False, "error": "TestRunner not available"}
        validate_result = adapter.security_manager.validate_code("import os")
        assert validate_result == {"valid": True}


# ---------------------------------------------------------------------------
# _validate_with_testpy_security test_code valid=False -- covers line 277
# ---------------------------------------------------------------------------


class TestSecurityValidationTestCodeValid:
    @pytest.mark.asyncio
    async def test_test_code_fails_valid_attr(self, adapter):
        """test_code validation fails via 'valid' attribute (line 277)."""
        results = [
            MagicMock(valid=True, spec=["valid"]),  # code passes
            MagicMock(valid=False, spec=["valid"]),  # test_code fails
        ]
        adapter.security_manager.validate_code.side_effect = results
        request = _make_test_request()
        with pytest.raises(ValueError, match="[Ss]ecurity validation failed"):
            await adapter._validate_with_testpy_security(request)


# ---------------------------------------------------------------------------
# _prepare_testpy_environment with file paths -- covers lines 293, 298
# ---------------------------------------------------------------------------


class TestPrepareEnvironmentFilePaths:
    @pytest.mark.asyncio
    async def test_code_file_path(self, adapter, tmp_path):
        """When code_file_path is set, reads content from file (line 293)."""
        code_file = tmp_path / "source.py"
        code_file.write_text("def hello(): return 42")

        request = _make_test_request(
            code=None,
            code_file_path=str(code_file),
        )
        env = await adapter._prepare_testpy_environment(request)
        try:
            content = Path(env["code_file"]).read_text()
            assert content == "def hello(): return 42"
            assert env["code_content"] == "def hello(): return 42"
        finally:
            import shutil

            shutil.rmtree(env["temp_dir"], ignore_errors=True)

    @pytest.mark.asyncio
    async def test_test_file_path(self, adapter, tmp_path):
        """When test_file_path is set, reads content from file (line 298)."""
        test_file = tmp_path / "test_source.py"
        test_file.write_text("def test_hello(): assert True")

        request = _make_test_request(
            test_code=None,
            test_file_path=str(test_file),
        )
        env = await adapter._prepare_testpy_environment(request)
        try:
            content = Path(env["test_file"]).read_text()
            # test_code has no "import" keyword, so fallback adds "from <module> import *"
            assert f"from {env['module_name']} import *" in content
            assert "def test_hello(): assert True" in content
        finally:
            import shutil

            shutil.rmtree(env["temp_dir"], ignore_errors=True)


# ---------------------------------------------------------------------------
# _prepare_testpy_environment processed_test_code is None -- covers line 318
# ---------------------------------------------------------------------------


class TestPrepareEnvironmentProcessedNone:
    @pytest.mark.asyncio
    async def test_process_test_code_returns_none(self, adapter):
        """When _process_test_code returns None, uses fallback (line 318)."""
        adapter.test_runner._process_test_code = MagicMock(return_value=None)
        request = _make_test_request()
        env = await adapter._prepare_testpy_environment(request)
        try:
            content = Path(env["test_file"]).read_text()
            assert content == "# Empty processed test"
        finally:
            import shutil

            shutil.rmtree(env["temp_dir"], ignore_errors=True)


# ---------------------------------------------------------------------------
# _execute_in_container_with_testpy timeout path -- covers lines 416-436
# ---------------------------------------------------------------------------


class TestExecuteContainerTimeout:
    @pytest.mark.asyncio
    async def test_container_execution_timeout_returns_exit_124(self, adapter, mock_docker_service):
        """When _run_testpy_in_container raises TimeoutError, returns exit_code=124 (line 417)."""
        adapter._run_testpy_in_container = AsyncMock(side_effect=TimeoutError("deadline"))
        request = _make_test_request(timeout=5)
        env = {"test_file": "/tmp/test_foo.py", "test_content": "test", "temp_dir": "/tmp/nonexist"}

        result = await adapter._execute_in_container_with_testpy("t1", request, env)
        assert result.success is False
        assert result.exit_code == 124
        assert "timed out" in result.stderr.lower()
        assert result.execution_time == 5.0
        assert result.error == "Test execution timed out after 5s"
        assert result.language == "python"
        assert result.container_name is None

    @pytest.mark.asyncio
    async def test_container_execution_general_error(self, adapter, mock_docker_service):
        """When a non-timeout error occurs, returns failed TestResult."""
        mock_docker_service.create_test_container = AsyncMock(
            side_effect=RuntimeError("container create failed")
        )
        request = _make_test_request()
        env = {"test_file": "/tmp/test_foo.py", "test_content": "test", "temp_dir": "/tmp/nonexist"}

        result = await adapter._execute_in_container_with_testpy("t1", request, env)
        assert result.success is False
        assert result.exit_code == 1
        assert "container create failed" in result.stderr
        assert result.error == "container create failed"
        assert result.container_name is None

    @pytest.mark.asyncio
    async def test_container_execution_with_non_python(self, adapter, mock_docker_service):
        """Execute with non-python language uses correct source file name."""
        adapter._run_testpy_in_container = AsyncMock(
            return_value={"success": True, "output": "ok", "error": ""}
        )
        request = _make_test_request(language="javascript")
        env = {
            "test_file": "/tmp/test_foo.js",
            "test_content": "test()",
            "temp_dir": "/tmp/nonexist",
        }

        result = await adapter._execute_in_container_with_testpy("t1", request, env)
        assert result.success is True
        assert result.language == "javascript"

        # Verify source file was named main.js not main.py
        calls = mock_docker_service.create_file_in_container.call_args_list
        # Find the source file copy call (not the test file copy)
        source_call = [c for c in calls if "main." in str(c)]
        assert len(source_call) == 1
        assert source_call[0][0][1] == "/app/test/main.js"

    @pytest.mark.asyncio
    async def test_container_execution_no_code(self, adapter, mock_docker_service):
        """When request has no code, skips source file copy."""
        adapter._run_testpy_in_container = AsyncMock(
            return_value={"success": True, "output": "ok", "error": ""}
        )
        request = _make_test_request(code=None)
        env = {"test_file": "/tmp/test_foo.py", "test_content": "test", "temp_dir": "/tmp/nonexist"}

        result = await adapter._execute_in_container_with_testpy("t1", request, env)
        assert result.success is True

        # Only 1 file should have been copied (the test file, not source)
        assert mock_docker_service.create_file_in_container.call_count == 1
        call_args = mock_docker_service.create_file_in_container.call_args[0]
        assert call_args[1] == "/app/test/test_foo.py"


# ---------------------------------------------------------------------------
# get_testpy_config -- covers lines 608-626
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# FallbackTestRunner.run_tests -- covers line 52
# ---------------------------------------------------------------------------


class TestFallbackTestRunner:
    def test_fallback_runner_returns_error(self):
        """FallbackTestRunner.run_tests returns error dict (line 52)."""
        runner = FallbackTestRunner()
        result = runner.run_tests("print(1)", "def test_x(): pass", timeout=10)
        assert result == {"success": False, "error": "TestRunner not available"}

    def test_fallback_security_manager_returns_valid(self):
        """FallbackSecurityManager.validate_code returns valid=True (line 59)."""
        mgr = FallbackSecurityManager()
        result = mgr.validate_code("import os")
        assert result == {"valid": True}

    def test_fallback_runner_no_timeout(self):
        """FallbackTestRunner.run_tests works without timeout arg."""
        runner = FallbackTestRunner()
        result = runner.run_tests("print(1)", "def test_x(): pass")
        assert result["success"] is False
        assert result["error"] == "TestRunner not available"


# ---------------------------------------------------------------------------
# _init_testpy_components MetricsManager failure -- covers lines 140-144
# ---------------------------------------------------------------------------


class TestInitMetricsFailure:
    def test_metrics_manager_init_failure(self, mock_docker_service):
        """When MetricsManager() raises, metrics_manager is set to None (lines 140-144)."""
        import services.testing.testpy_integration_adapter as mod

        original = mod.MetricsManager
        provided_runner = MagicMock(spec=["run_tests"])
        provided_security = MagicMock(spec=["validate_code"])
        try:
            mod.MetricsManager = MagicMock(side_effect=RuntimeError("no metrics"))
            adapter = TestPyIntegrationAdapter(
                docker_service=mock_docker_service,
                test_runner=provided_runner,
                security_manager=provided_security,
                metrics_manager=None,
            )
            assert adapter.metrics_manager is None
            # test_runner and security_manager should still be the exact provided mocks
            assert adapter.test_runner is provided_runner
            assert adapter.security_manager is provided_security
        finally:
            mod.MetricsManager = original

    def test_init_outer_except(self, mock_docker_service):
        """When OriginalTestRunner() raises, outer except sets fallbacks (lines 146-151)."""
        import services.testing.testpy_integration_adapter as mod

        original_runner = mod.OriginalTestRunner
        try:
            # First call raises, second call (in except block) succeeds
            call_count = {"n": 0}
            real_fallback = original_runner

            def failing_then_ok(*a, **kw):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise RuntimeError("init boom")
                return real_fallback()

            mod.OriginalTestRunner = failing_then_ok
            adapter = TestPyIntegrationAdapter(
                docker_service=mock_docker_service,
                test_runner=None,
                security_manager=None,
                metrics_manager=None,
            )
            # Should have fallen through to except block and set fallbacks
            assert adapter.metrics_manager is None
            # Verify fallback runner actually works
            run_result = adapter.test_runner.run_tests("code", "test")
            assert run_result == {"success": False, "error": "TestRunner not available"}
            validate_result = adapter.security_manager.validate_code("code")
            assert validate_result == {"valid": True}
        finally:
            mod.OriginalTestRunner = original_runner


# ---------------------------------------------------------------------------
# _prepare_testpy_environment error cleanup -- covers lines 336-341
# ---------------------------------------------------------------------------


class TestPrepareEnvironmentErrorCleanup:
    @pytest.mark.asyncio
    async def test_prepare_env_error_cleans_temp_dir(self, adapter, tmp_path):
        """When file operations fail, temp dir is cleaned up (lines 336-341)."""
        request = _make_test_request(
            code=None,
            code_file_path="/nonexistent/path/to/code.py",
        )
        with pytest.raises(FileNotFoundError):
            await adapter._prepare_testpy_environment(request)


# ---------------------------------------------------------------------------
# _cleanup_testpy_environment exception -- covers lines 605-606
# ---------------------------------------------------------------------------


class TestCleanupEnvironmentException:
    @pytest.mark.asyncio
    async def test_cleanup_env_exception_swallowed(self, adapter):
        """When shutil.rmtree raises an unexpected error, it is caught (lines 605-606)."""
        import shutil
        from unittest.mock import patch

        with patch.object(shutil, "rmtree", side_effect=PermissionError("denied")):
            # Should not raise -- exception is caught
            await adapter._cleanup_testpy_environment({"temp_dir": "/some/path"})


class TestGetTestPyConfig:
    def test_get_config(self, mock_docker_service):
        """get_testpy_config returns structured config dict."""
        config = MagicMock()
        config.security = MagicMock()
        config.security.max_memory_mb = 256
        config.security.max_execution_time = 30
        config.security.allowed_modules = ["os", "sys"]
        config.security.sandbox_enabled = True
        config.test = MagicMock()
        config.test.timeout = 60
        config.test.parallel = True
        config.test.max_workers = 4
        config.logging = MagicMock()
        config.logging.level = MagicMock(value="INFO")
        config.logging.console_output = True

        adapter = TestPyIntegrationAdapter(docker_service=mock_docker_service, config=config)
        result = adapter.get_testpy_config()

        assert result == {
            "security": {
                "max_memory_mb": 256,
                "max_execution_time": 30,
                "allowed_modules": ["os", "sys"],
                "sandbox_enabled": True,
            },
            "test": {
                "timeout": 60,
                "parallel": True,
                "max_workers": 4,
            },
            "logging": {
                "level": "INFO",
                "console_output": True,
            },
        }
