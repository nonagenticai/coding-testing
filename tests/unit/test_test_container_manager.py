"""Tests for src/services/testing/test_container_manager.py — 22% coverage target."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from models.testing_models import TestRequest, TestResult, TestSecurityLevel
from services.stubs import DockerService
from services.testing.test_container_manager import TestContainerManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def docker_service():
    """Mock DockerService with all async methods."""
    svc = AsyncMock(spec=DockerService)
    svc.create_test_container = AsyncMock(return_value="container-abc")
    svc.start_container = AsyncMock()
    svc.create_file_in_container = AsyncMock()
    svc.exec_in_container = AsyncMock(return_value={"exit_code": 0, "stdout": "OK", "stderr": ""})
    svc.get_container_stats = AsyncMock(return_value={"memory_usage_mb": 128, "cpu_percent": 5.0})
    svc.stop_container = AsyncMock()
    svc.remove_container_by_id = AsyncMock()
    return svc


@pytest.fixture
def manager(docker_service):
    return TestContainerManager(docker_service=docker_service)


@pytest.fixture
def test_request():
    return TestRequest(
        code="def add(a,b): return a+b",
        test_code="def test_add(): assert add(1,2)==3",
        language="python",
        timeout=30,
    )


@pytest.fixture
def config_manager():
    cm = MagicMock()
    cm.get_container_image.return_value = "python:3.13-slim"
    cm.create_container_config.return_value = {"image": "python:3.13-slim"}
    cm.prepare_test_files.return_value = {"test_add.py": "assert True"}
    cm.get_test_command.return_value = ["pytest", "test_add.py"]
    return cm


@pytest.fixture
def result_processor():
    rp = MagicMock()
    rp.convert_container_result.return_value = TestResult(
        test_id="t1",
        success=True,
        stdout="OK",
        stderr="",
        exit_code=0,
        execution_time=1.0,
        created_at=datetime.now(UTC),
        language="python",
        security_level="high",
    )
    return rp


# ---------------------------------------------------------------------------
# execute_in_container
# ---------------------------------------------------------------------------


class TestExecuteInContainer:
    @pytest.mark.asyncio
    async def test_success_path(
        self, manager, test_request, config_manager, result_processor, docker_service
    ):
        result = await manager.execute_in_container(
            test_request, "t1", config_manager, result_processor
        )
        # Verify the returned TestResult has the values from result_processor
        assert result.success is True
        assert result.test_id == "t1"
        assert result.stdout == "OK"
        assert result.stderr == ""
        assert result.exit_code == 0
        assert result.language == "python"

        # Verify the full container lifecycle was executed in order
        docker_service.create_test_container.assert_called_once_with({"image": "python:3.13-slim"})
        docker_service.start_container.assert_called_once_with("container-abc")
        docker_service.create_file_in_container.assert_called_once_with(
            "container-abc", "/app/test/test_add.py", "assert True"
        )
        docker_service.exec_in_container.assert_called_once_with(
            docker_id="container-abc",
            command=["pytest", "test_add.py"],
            timeout=30,
            user="testuser",
        )
        docker_service.get_container_stats.assert_called_once_with("container-abc")

        # Verify result_processor was called with the correct container_result dict
        rp_call_args = result_processor.convert_container_result.call_args
        container_result = rp_call_args[0][0]
        assert container_result["exit_code"] == 0
        assert container_result["stdout"] == "OK"
        assert container_result["stderr"] == ""
        assert container_result["memory_usage"] == 128
        assert container_result["cpu_usage"] == 5.0
        assert container_result["execution_time"] > 0
        # Verify test_id and request were passed
        assert rp_call_args[0][1] == "t1"
        assert rp_call_args[0][2] is test_request

        # Cleanup must happen even on success
        docker_service.stop_container.assert_called_once_with("container-abc")
        docker_service.remove_container_by_id.assert_called_once_with("container-abc")

    @pytest.mark.asyncio
    async def test_multiple_test_files(
        self, manager, test_request, config_manager, result_processor, docker_service
    ):
        config_manager.prepare_test_files.return_value = {
            "test_a.py": "assert True",
            "test_b.py": "assert False",
        }
        await manager.execute_in_container(test_request, "t2", config_manager, result_processor)
        assert docker_service.create_file_in_container.call_count == 2
        # Verify each file was created with the correct path and content
        calls = docker_service.create_file_in_container.call_args_list
        assert calls[0] == call("container-abc", "/app/test/test_a.py", "assert True")
        assert calls[1] == call("container-abc", "/app/test/test_b.py", "assert False")

    @pytest.mark.asyncio
    async def test_error_returns_error_result(
        self, manager, test_request, config_manager, result_processor, docker_service
    ):
        docker_service.exec_in_container.side_effect = RuntimeError("exec failed")
        result = await manager.execute_in_container(
            test_request, "t3", config_manager, result_processor
        )
        # Verify the error result has all the right fields
        assert result.success is False
        assert result.error == "exec failed"
        assert result.test_id == "t3"
        assert result.exit_code == 1
        assert result.language == "python"
        assert result.security_level == TestSecurityLevel.HIGH
        assert result.stdout == ""
        assert result.stderr == "exec failed"
        assert result.execution_time > 0
        # result_processor should NOT have been called (error path bypasses it)
        result_processor.convert_container_result.assert_not_called()
        # Cleanup still happens because container_id was set
        docker_service.stop_container.assert_called_once_with("container-abc")
        docker_service.remove_container_by_id.assert_called_once_with("container-abc")

    @pytest.mark.asyncio
    async def test_cleanup_failure_does_not_raise(
        self, manager, test_request, config_manager, result_processor, docker_service
    ):
        docker_service.stop_container.side_effect = RuntimeError("stop failed")
        # Should not raise despite cleanup error
        result = await manager.execute_in_container(
            test_request, "t4", config_manager, result_processor
        )
        # The main execution succeeded, so result comes from result_processor
        assert result.success is True
        assert result.test_id == "t1"
        assert result.exit_code == 0
        # stop_container was called (and failed), but remove was NOT called since
        # _cleanup_container catches the exception at stop and skips remove
        docker_service.stop_container.assert_called_once_with("container-abc")
        # remove_container_by_id is NOT called because the exception in stop_container
        # is caught by the try/except in _cleanup_container before remove is reached
        docker_service.remove_container_by_id.assert_not_called()

    @pytest.mark.asyncio
    async def test_container_creation_failure(
        self, manager, test_request, config_manager, result_processor, docker_service
    ):
        docker_service.create_test_container.side_effect = RuntimeError("no docker")
        result = await manager.execute_in_container(
            test_request, "t5", config_manager, result_processor
        )
        assert result.success is False
        assert result.error == "no docker"
        assert result.test_id == "t5"
        assert result.exit_code == 1
        assert result.language == "python"
        assert result.stderr == "no docker"
        # No container was created, so NO cleanup should happen
        docker_service.stop_container.assert_not_called()
        docker_service.remove_container_by_id.assert_not_called()
        # start_container should not have been called either
        docker_service.start_container.assert_not_called()

    @pytest.mark.asyncio
    async def test_uses_request_timeout(
        self, manager, config_manager, result_processor, docker_service
    ):
        """Verify the request's timeout value is passed through to exec_in_container."""
        req = TestRequest(
            code="x = 1",
            test_code="assert x",
            language="python",
            timeout=120,
        )
        await manager.execute_in_container(req, "t6", config_manager, result_processor)
        docker_service.exec_in_container.assert_called_once_with(
            docker_id="container-abc",
            command=["pytest", "test_add.py"],
            timeout=120,
            user="testuser",
        )

    @pytest.mark.asyncio
    async def test_default_timeout_when_none(
        self, manager, config_manager, result_processor, docker_service
    ):
        """When request.timeout is None, the default of 30 should be used."""
        req = TestRequest(
            code="x = 1",
            test_code="assert x",
            language="python",
            timeout=None,
        )
        await manager.execute_in_container(req, "t7", config_manager, result_processor)
        _, kwargs = docker_service.exec_in_container.call_args
        assert kwargs["timeout"] == 30


# ---------------------------------------------------------------------------
# _cleanup_container
# ---------------------------------------------------------------------------


class TestCleanupContainer:
    @pytest.mark.asyncio
    async def test_successful_cleanup(self, manager, docker_service):
        await manager._cleanup_container("c1")
        docker_service.stop_container.assert_called_once_with("c1")
        docker_service.remove_container_by_id.assert_called_once_with("c1")

    @pytest.mark.asyncio
    async def test_cleanup_error_logged_not_raised(self, manager, docker_service):
        docker_service.stop_container.side_effect = Exception("stop err")
        # Should not raise -- the exception is caught and logged
        await manager._cleanup_container("c2")
        # stop was called (and raised)
        docker_service.stop_container.assert_called_once_with("c2")
        # remove was NOT called because stop raised before we got there
        docker_service.remove_container_by_id.assert_not_called()

    @pytest.mark.asyncio
    async def test_remove_error_logged_not_raised(self, manager, docker_service):
        """If remove fails (but stop succeeds), the error is caught."""
        docker_service.remove_container_by_id.side_effect = Exception("rm err")
        await manager._cleanup_container("c3")
        docker_service.stop_container.assert_called_once_with("c3")
        docker_service.remove_container_by_id.assert_called_once_with("c3")


# ---------------------------------------------------------------------------
# _create_error_result
# ---------------------------------------------------------------------------


class TestCreateErrorResult:
    def test_creates_failed_result_with_all_fields(self, manager):
        start = datetime(2026, 3, 23, 12, 0, 0, tzinfo=UTC)
        req = TestRequest(code="x", test_code="t", language="python")
        result = manager._create_error_result("t1", start, req, "boom")

        # Core failure fields
        assert result.success is False
        assert result.error == "boom"
        assert result.exit_code == 1
        assert result.test_id == "t1"

        # Request context is preserved
        assert result.language == "python"
        assert result.security_level == TestSecurityLevel.HIGH  # default for TestRequest

        # Error is also written to stderr
        assert result.stderr == "boom"
        assert result.stdout == ""

        # Timing: execution_time should be positive (time since start)
        assert result.execution_time > 0
        assert result.created_at == start

        # Optional fields are explicitly None
        assert result.coverage_data is None
        assert result.test_metadata is None
        assert result.test_type is None
        assert result.container_name is None
        assert result.quality_report is None

    def test_preserves_custom_security_level(self, manager):
        start = datetime.now(UTC)
        req = TestRequest(code="x", test_code="t", language="python", security_level="low")
        result = manager._create_error_result("t2", start, req, "fail")
        assert result.security_level == TestSecurityLevel.LOW


# ---------------------------------------------------------------------------
# cleanup_test_containers
# ---------------------------------------------------------------------------


class TestCleanupTestContainers:
    @pytest.mark.asyncio
    async def test_no_op_does_not_call_docker(self, manager, docker_service):
        """cleanup_test_containers is a no-op stub; it should not touch docker_service."""
        await manager.cleanup_test_containers("test-xyz")
        # No docker methods should have been called
        docker_service.stop_container.assert_not_called()
        docker_service.remove_container_by_id.assert_not_called()
        docker_service.create_test_container.assert_not_called()


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_healthy_with_client(self, manager, docker_service):
        docker_service.client = MagicMock()
        result = await manager.health_check()
        assert result["status"] == "healthy"
        assert result["docker_service"] is True
        assert "timestamp" in result
        # Timestamp should be a valid ISO format string
        datetime.fromisoformat(result["timestamp"])

    @pytest.mark.asyncio
    async def test_unhealthy_without_client(self, manager, docker_service):
        # Remove client attribute so hasattr returns False
        if hasattr(docker_service, "client"):
            delattr(docker_service, "client")
        result = await manager.health_check()
        assert result["status"] == "unhealthy"
        assert result["docker_service"] is False
        assert "timestamp" in result
        datetime.fromisoformat(result["timestamp"])

    @pytest.mark.asyncio
    async def test_health_check_exception(self, manager, docker_service):
        # Make hasattr raise
        type(docker_service).__getattr__ = MagicMock(side_effect=RuntimeError("fail"))
        result = await manager.health_check()
        assert result["status"] == "unhealthy"
        assert result["error"] == "fail"
        assert "timestamp" in result
        datetime.fromisoformat(result["timestamp"])


# ---------------------------------------------------------------------------
# get_container_stats
# ---------------------------------------------------------------------------


class TestGetContainerStats:
    @pytest.mark.asyncio
    async def test_success_returns_full_stats(self, manager, docker_service):
        result = await manager.get_container_stats("c1")
        assert result == {"memory_usage_mb": 128, "cpu_percent": 5.0}
        docker_service.get_container_stats.assert_called_once_with("c1")

    @pytest.mark.asyncio
    async def test_failure_returns_empty_dict(self, manager, docker_service):
        docker_service.get_container_stats.side_effect = Exception("fail")
        result = await manager.get_container_stats("c1")
        assert result == {}
        docker_service.get_container_stats.assert_called_once_with("c1")


# ---------------------------------------------------------------------------
# create_test_file_in_container
# ---------------------------------------------------------------------------


class TestCreateTestFileInContainer:
    @pytest.mark.asyncio
    async def test_success_delegates_to_docker_service(self, manager, docker_service):
        await manager.create_test_file_in_container("c1", "/app/test.py", "code")
        docker_service.create_file_in_container.assert_called_once_with(
            "c1", "/app/test.py", "code"
        )

    @pytest.mark.asyncio
    async def test_raises_on_error_with_original_message(self, manager, docker_service):
        docker_service.create_file_in_container.side_effect = RuntimeError("write fail")
        with pytest.raises(RuntimeError, match="write fail"):
            await manager.create_test_file_in_container("c1", "/app/test.py", "code")
        # Verify the call was still made
        docker_service.create_file_in_container.assert_called_once_with(
            "c1", "/app/test.py", "code"
        )


# ---------------------------------------------------------------------------
# execute_command_in_container
# ---------------------------------------------------------------------------


class TestExecuteCommandInContainer:
    @pytest.mark.asyncio
    async def test_success_returns_full_result(self, manager, docker_service):
        result = await manager.execute_command_in_container("c1", ["echo", "hi"])
        assert result == {"exit_code": 0, "stdout": "OK", "stderr": ""}
        docker_service.exec_in_container.assert_called_once_with(
            docker_id="c1", command=["echo", "hi"], timeout=30, user="testuser"
        )

    @pytest.mark.asyncio
    async def test_custom_timeout(self, manager, docker_service):
        result = await manager.execute_command_in_container("c1", ["ls"], timeout=60)
        assert result == {"exit_code": 0, "stdout": "OK", "stderr": ""}
        docker_service.exec_in_container.assert_called_once_with(
            docker_id="c1", command=["ls"], timeout=60, user="testuser"
        )

    @pytest.mark.asyncio
    async def test_raises_on_error_with_original_message(self, manager, docker_service):
        docker_service.exec_in_container.side_effect = RuntimeError("exec fail")
        with pytest.raises(RuntimeError, match="exec fail"):
            await manager.execute_command_in_container("c1", ["bad"])
        docker_service.exec_in_container.assert_called_once_with(
            docker_id="c1", command=["bad"], timeout=30, user="testuser"
        )


# ---------------------------------------------------------------------------
# stop_container
# ---------------------------------------------------------------------------


class TestStopContainer:
    @pytest.mark.asyncio
    async def test_success_delegates_to_docker_service(self, manager, docker_service):
        await manager.stop_container("c1")
        docker_service.stop_container.assert_called_once_with("c1")

    @pytest.mark.asyncio
    async def test_raises_on_error_with_original_message(self, manager, docker_service):
        docker_service.stop_container.side_effect = RuntimeError("stop fail")
        with pytest.raises(RuntimeError, match="stop fail"):
            await manager.stop_container("c1")
        docker_service.stop_container.assert_called_once_with("c1")


# ---------------------------------------------------------------------------
# remove_container
# ---------------------------------------------------------------------------


class TestRemoveContainer:
    @pytest.mark.asyncio
    async def test_success_delegates_to_docker_service(self, manager, docker_service):
        await manager.remove_container("c1")
        docker_service.remove_container_by_id.assert_called_once_with("c1")

    @pytest.mark.asyncio
    async def test_raises_on_error_with_original_message(self, manager, docker_service):
        docker_service.remove_container_by_id.side_effect = RuntimeError("rm fail")
        with pytest.raises(RuntimeError, match="rm fail"):
            await manager.remove_container("c1")
        docker_service.remove_container_by_id.assert_called_once_with("c1")


# ---------------------------------------------------------------------------
# create_container
# ---------------------------------------------------------------------------


class TestCreateContainer:
    @pytest.mark.asyncio
    async def test_success_returns_container_id(self, manager, docker_service):
        cid = await manager.create_container({"image": "python:3.13"})
        assert cid == "container-abc"
        docker_service.create_test_container.assert_called_once_with({"image": "python:3.13"})

    @pytest.mark.asyncio
    async def test_raises_on_error_with_original_message(self, manager, docker_service):
        docker_service.create_test_container.side_effect = RuntimeError("create fail")
        with pytest.raises(RuntimeError, match="create fail"):
            await manager.create_container({"image": "python:3.13"})
        docker_service.create_test_container.assert_called_once_with({"image": "python:3.13"})


# ---------------------------------------------------------------------------
# start_container
# ---------------------------------------------------------------------------


class TestStartContainer:
    @pytest.mark.asyncio
    async def test_success_delegates_to_docker_service(self, manager, docker_service):
        await manager.start_container("c1")
        docker_service.start_container.assert_called_once_with("c1")

    @pytest.mark.asyncio
    async def test_raises_on_error_with_original_message(self, manager, docker_service):
        docker_service.start_container.side_effect = RuntimeError("start fail")
        with pytest.raises(RuntimeError, match="start fail"):
            await manager.start_container("c1")
        docker_service.start_container.assert_called_once_with("c1")
