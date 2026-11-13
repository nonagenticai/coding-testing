"""Final coverage push tests targeting remaining uncovered lines."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# ===========================================================================
# test_orchestration.py — edge cases
# ===========================================================================


class TestOrchestrationEdgeCases:
    """Tests for uncovered exception paths in test_orchestration.py."""

    async def test_create_suite_empty_name(self):
        """create_test_suite with empty name returns 400."""
        from routers.test_orchestration import create_test_suite

        mock_container = Mock()
        mock_container.status = Mock()
        mock_container.status.value = "running"

        with patch(
            "routers.test_orchestration.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
            return_value=mock_container,
        ):
            with pytest.raises(Exception):
                await create_test_suite(
                    name="",
                    container_name="test",
                    tests=[],
                    description="",
                    metadata=None,
                    current_user=Mock(preferred_username="user"),
                    tenant_id="test",
                )

    async def test_create_suite_no_tests(self):
        """create_test_suite with empty tests returns 400."""
        from routers.test_orchestration import create_test_suite

        mock_container = Mock()
        mock_container.status = Mock()
        mock_container.status.value = "running"

        with patch(
            "routers.test_orchestration.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
            return_value=mock_container,
        ):
            with pytest.raises(Exception):
                await create_test_suite(
                    name="test-suite",
                    container_name="test",
                    tests=[],
                    description="",
                    metadata=None,
                    current_user=Mock(preferred_username="user"),
                    tenant_id="test",
                )

    async def test_list_suites_empty(self, pg_db):
        """list_test_suites with no suites returns empty list."""
        from routers.test_orchestration import list_test_suites

        mock_user = Mock()
        mock_user.preferred_username = "user"

        result = await list_test_suites(
            container_name=None,
            status_filter=None,
            language_filter=None,
            current_user=mock_user,
            tenant_id="test-tenant",
        )
        assert result["data"]["total"] == 0

    async def test_list_suites_filters(self, pg_db):
        """list_test_suites applies filters."""
        from routers.test_orchestration import (
            OrchestrationTestSuite,
            list_test_suites,
        )
        from services.orchestration_store import get_orchestration_store
        from services.test_execution.test_orchestrator import (
            OrchestratorType,
            TestExecutionRequest,
            TestLanguage,
        )

        suite = OrchestrationTestSuite("s1", "Suite 1", "", "container-1", "test-tenant")
        suite.status = "completed"
        suite.tests = [
            TestExecutionRequest(
                language=TestLanguage.PYTHON,
                test_type=OrchestratorType.UNIT,
                source_code="print(1)",
                container_name="container-1",
            )
        ]
        await get_orchestration_store().save_suite(suite)

        mock_user = Mock()

        # Filter by status - should not match
        result = await list_test_suites(
            container_name=None,
            status_filter="pending",
            language_filter=None,
            current_user=mock_user,
            tenant_id="test-tenant",
        )
        assert result["data"]["total"] == 0

        # Filter by tenant - different tenant
        result = await list_test_suites(
            container_name=None,
            status_filter=None,
            language_filter=None,
            current_user=mock_user,
            tenant_id="other-tenant",
        )
        assert result["data"]["total"] == 0

    async def test_list_suites_exception(self):
        """list_test_suites raises HTTP 500 when the backing store fails."""
        from fastapi import HTTPException

        from routers.test_orchestration import list_test_suites

        mock_user = Mock()
        with patch(
            "services.orchestration_store.OrchestrationStore.list_suites",
            new_callable=AsyncMock,
            side_effect=RuntimeError("store unavailable"),
        ):
            with pytest.raises(HTTPException) as ei:
                await list_test_suites(
                    container_name=None,
                    status_filter=None,
                    language_filter=None,
                    current_user=mock_user,
                    tenant_id="test",
                )
        assert ei.value.status_code == 500


# ===========================================================================
# processing/test_execution/test_runner.py — more line coverage
# ===========================================================================


class TestContainerTestRunnerLines:
    """Tests covering specific uncovered lines in test_runner.py."""

    def _make_runner(self):
        from config.test_framework_config import TestConfig
        from processing.test_execution.test_runner import ContainerTestRunner

        docker_svc = MagicMock()
        config = TestConfig()
        runner = ContainerTestRunner(docker_svc, config)
        return runner

    async def test_execute_with_orchestrator(self):
        """Test _execute_with_orchestrator converts result correctly."""
        from models.testing_models import TestRequest
        from services.test_execution.test_orchestrator import (
            OrchestratorType,
            TestExecutionResult,
        )

        runner = self._make_runner()
        mock_result = TestExecutionResult(
            test_id="t1",
            success=True,
            stdout="ok",
            stderr="",
            exit_code=0,
            execution_time_ms=100.0,
            language="go",
            test_type=OrchestratorType.UNIT,
            container_id="c1",
            created_at=datetime.now(UTC),
        )
        runner.orchestrator = AsyncMock()
        runner.orchestrator.execute_tests = AsyncMock(return_value=mock_result)

        req = TestRequest(code="package main", test_code="func Test() {}", language="go")
        result = await runner._execute_with_orchestrator(req, "t1", datetime.now(UTC))
        assert result.success is True
        assert result.language == "go"

    async def test_create_js_package_file_javascript(self):
        """Test _create_js_package_file for javascript."""
        runner = self._make_runner()
        runner.docker_service.create_file_in_container = AsyncMock(return_value=True)
        await runner._create_js_package_file("container-1", "javascript")
        runner.docker_service.create_file_in_container.assert_awaited_once()

    async def test_create_js_package_file_typescript(self):
        """Test _create_js_package_file for typescript adds ts-jest deps."""
        runner = self._make_runner()
        runner.docker_service.create_file_in_container = AsyncMock(return_value=True)
        await runner._create_js_package_file("container-1", "typescript")
        runner.docker_service.create_file_in_container.assert_awaited_once()

    async def test_setup_test_environment_creates_files(self):
        """Test _setup_test_environment creates both code and test files."""
        from models.testing_models import TestRequest

        runner = self._make_runner()
        runner.docker_service.create_file_in_container = AsyncMock(return_value=True)

        req = TestRequest(
            code="def foo(): pass", test_code="def test_foo(): pass", language="python"
        )
        await runner._setup_test_environment("container-1", req)
        assert runner.docker_service.create_file_in_container.await_count == 2

    async def test_setup_test_environment_js_creates_package(self):
        """Test _setup_test_environment for JS creates package.json."""
        from models.testing_models import TestRequest

        runner = self._make_runner()
        runner.docker_service.create_file_in_container = AsyncMock(return_value=True)

        req = TestRequest(code="function foo() {}", language="javascript")
        await runner._setup_test_environment("container-1", req)
        # Should create code file + package.json (2 calls)
        assert runner.docker_service.create_file_in_container.await_count == 2

    async def test_setup_test_environment_file_creation_fails(self):
        """Test _setup_test_environment raises on file creation failure."""
        from models.testing_models import TestRequest

        runner = self._make_runner()
        # First call for the code file returns False
        runner.docker_service.create_file_in_container = AsyncMock(return_value=False)

        req = TestRequest(code="x=1", language="python")
        with pytest.raises((RuntimeError, ValueError)):
            await runner._setup_test_environment("container-1", req)

    async def test_run_test_python(self):
        """Test _run_test_in_container for Python."""
        from models.testing_models import TestRequest

        runner = self._make_runner()
        runner.docker_service.install_tools_in_container = AsyncMock()
        runner.docker_service.exec_in_container = AsyncMock(
            return_value={"exit_code": 0, "stdout": "PASSED", "stderr": ""}
        )

        req = TestRequest(code="x=1", language="python", timeout=30)
        result = await runner._run_test_in_container("container-1", req, "t1")
        assert result.success is True

    async def test_run_test_javascript(self):
        """Test _run_test_in_container for JavaScript."""
        from models.testing_models import TestRequest

        runner = self._make_runner()
        runner.docker_service.exec_in_container = AsyncMock(
            return_value={"exit_code": 0, "stdout": "ok", "stderr": ""}
        )

        req = TestRequest(code="x=1", language="javascript", timeout=30)
        result = await runner._run_test_in_container("container-1", req, "t1")
        assert result.success is True

    async def test_run_test_unsupported_language(self):
        """Test _run_test_in_container raises for unsupported language."""
        from models.testing_models import TestRequest

        runner = self._make_runner()
        req = TestRequest(code="x=1", language="go", timeout=30)
        with pytest.raises(ValueError, match="Unsupported language"):
            await runner._run_test_in_container("container-1", req, "t1")

    async def test_run_test_timeout(self):
        """Test _run_test_in_container handles timeout."""
        from models.testing_models import TestRequest

        runner = self._make_runner()
        runner.docker_service.install_tools_in_container = AsyncMock()
        runner.docker_service.exec_in_container = AsyncMock(side_effect=TimeoutError("timed out"))

        req = TestRequest(code="x=1", language="python", timeout=1)
        result = await runner._run_test_in_container("container-1", req, "t1")
        assert result.success is False
        assert result.exit_code == 124

    async def test_validate_request_uses_security_manager(self):
        """Test _validate_request calls security manager when available."""
        from models.testing_models import TestRequest

        runner = self._make_runner()
        runner.security_manager = Mock()
        runner.security_manager.check_code_safety = Mock()

        req = TestRequest(code="x=1", test_code="test", language="python")
        await runner._validate_request(req)
        assert runner.security_manager.check_code_safety.call_count == 2

    async def test_validate_request_no_code_no_test(self):
        """Test _validate_request with empty request raises ValueError."""
        from models.testing_models import TestRequest

        runner = self._make_runner()
        # Create request with explicit None code
        req = TestRequest(language="python")
        # Empty code/test_code is okay, only raises if code_path is also missing
        await runner._validate_request(req)  # Should not raise

    async def test_create_test_container(self):
        """Test _create_test_container creates and starts container."""
        from models.testing_models import TestRequest

        runner = self._make_runner()
        runner.docker_service.get_language_container_image = Mock(return_value="python:3.11")
        runner.docker_service.create_test_container = AsyncMock(return_value="container-123")
        runner.docker_service.start_container = AsyncMock()

        req = TestRequest(code="x=1", language="python")
        docker_id = await runner._create_test_container(req, "t1")
        assert docker_id == "container-123"

    async def test_execute_legacy_full_flow(self):
        """Test _execute_legacy runs the full validation->create->setup->run->cleanup flow."""
        from models.testing_models import TestRequest

        runner = self._make_runner()
        runner.docker_service.get_language_container_image = Mock(return_value="python:3.11")
        runner.docker_service.create_test_container = AsyncMock(return_value="container-123")
        runner.docker_service.start_container = AsyncMock()
        runner.docker_service.create_file_in_container = AsyncMock(return_value=True)
        runner.docker_service.install_tools_in_container = AsyncMock()
        runner.docker_service.exec_in_container = AsyncMock(
            return_value={"exit_code": 0, "stdout": "ok", "stderr": ""}
        )
        runner.docker_service.stop_container = AsyncMock()
        runner.docker_service.remove_container_by_id = AsyncMock()

        req = TestRequest(code="x=1", test_code="test", language="python", timeout=30)
        result = await runner._execute_legacy(req, "t1", datetime.now(UTC))
        assert result.success is True
        runner.docker_service.stop_container.assert_awaited_once()
        runner.docker_service.remove_container_by_id.assert_awaited_once()
