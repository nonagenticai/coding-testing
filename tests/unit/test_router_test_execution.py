"""Unit tests for test execution router."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


class TestGetTestOrchestrator:
    """Test the get_test_orchestrator factory."""

    async def test_creates_orchestrator_on_first_call(self):
        """Test orchestrator is created lazily."""
        import routers.test_execution as mod

        # Reset global state
        mod._test_orchestrator = None
        mock_docker = MagicMock()
        mock_orchestrator = AsyncMock()

        with (
            patch("routers.test_execution.get_docker_service", return_value=mock_docker),
            patch(
                "routers.test_execution.create_test_orchestrator",
                return_value=mock_orchestrator,
            ),
        ):
            result = await mod.get_test_orchestrator()
            assert result is mock_orchestrator

        # Cleanup
        mod._test_orchestrator = None

    async def test_returns_cached_orchestrator(self):
        """Test orchestrator is cached after first creation."""
        import routers.test_execution as mod

        mock_orchestrator = AsyncMock()
        mod._test_orchestrator = mock_orchestrator

        result = await mod.get_test_orchestrator()
        assert result is mock_orchestrator

        # Cleanup
        mod._test_orchestrator = None


class TestGetContainerRegistry:
    """Test the get_container_registry factory."""

    async def test_creates_registry_on_first_call(self):
        """Test container registry is created lazily."""
        import routers.test_execution as mod
        from services.test_execution.test_execution_container_registry import (
            TestExecutionContainerRegistry,
        )

        mod._container_registry = None
        result = await mod.get_container_registry()
        assert isinstance(result, TestExecutionContainerRegistry)
        # Verify it was cached in the module global
        assert mod._container_registry is result

        # Cleanup
        mod._container_registry = None

    async def test_returns_cached_registry(self):
        """Test registry is cached."""
        import routers.test_execution as mod

        mock_registry = MagicMock()
        mod._container_registry = mock_registry

        result = await mod.get_container_registry()
        assert result is mock_registry

        # Cleanup
        mod._container_registry = None


class TestLogTestExecutionMetrics:
    """Test background metrics logging."""

    async def test_log_metrics_handles_error_gracefully(self):
        """Test metrics logging doesn't raise on error."""
        from routers.test_execution import log_test_execution_metrics

        mock_result = MagicMock()
        mock_result.test_id = "test-123"
        mock_result.language.value = "python"
        mock_result.test_type.value = "unit"
        mock_result.success = True
        mock_result.execution_time_ms = 100.0
        mock_result.exit_code = 0

        mock_request = MagicMock()
        mock_request.security_level.value = "medium"
        mock_request.timeout_seconds = 30

        with patch(
            "routers.test_execution.get_metrics_service",
            side_effect=Exception("metrics unavailable"),
        ):
            # Should not raise
            await log_test_execution_metrics("user1", mock_result, mock_request)

    async def test_log_batch_metrics_handles_error(self):
        """Test batch metrics logging doesn't raise on error."""
        from routers.test_execution import log_batch_execution_metrics

        with patch(
            "routers.test_execution.get_metrics_service",
            side_effect=Exception("metrics unavailable"),
        ):
            # Should not raise
            await log_batch_execution_metrics("user1", 10, 8)

    async def test_log_metrics_success(self):
        """Test metrics logging succeeds when service is available."""
        from routers.test_execution import log_test_execution_metrics

        mock_result = MagicMock()
        mock_result.test_id = "test-456"
        mock_result.language.value = "python"
        mock_result.test_type.value = "unit"
        mock_result.success = True
        mock_result.execution_time_ms = 50.0
        mock_result.exit_code = 0

        mock_request = MagicMock()
        mock_request.security_level.value = "standard"
        mock_request.timeout_seconds = 60

        mock_metrics = AsyncMock()
        mock_metrics.record_test_execution = AsyncMock()

        with patch(
            "routers.test_execution.get_metrics_service",
            return_value=mock_metrics,
        ):
            await log_test_execution_metrics("user1", mock_result, mock_request)
            mock_metrics.record_test_execution.assert_called_once()
            call_args = mock_metrics.record_test_execution.call_args[0][0]
            assert call_args["event_type"] == "test_execution"
            assert call_args["user_id"] == "user1"
            assert call_args["test_id"] == "test-456"
            assert call_args["language"] == "python"
            assert call_args["test_type"] == "unit"
            assert call_args["success"] is True
            assert call_args["execution_time_ms"] == 50.0
            assert call_args["exit_code"] == 0
            assert call_args["security_level"] == "standard"
            assert call_args["timeout_seconds"] == 60

    async def test_log_batch_metrics_success(self):
        """Test batch metrics logging succeeds."""
        from routers.test_execution import log_batch_execution_metrics

        mock_metrics = AsyncMock()
        mock_metrics.record_test_execution = AsyncMock()

        with patch(
            "routers.test_execution.get_metrics_service",
            return_value=mock_metrics,
        ):
            await log_batch_execution_metrics("user1", 5, 4)
            mock_metrics.record_test_execution.assert_called_once()
            call_args = mock_metrics.record_test_execution.call_args[0][0]
            assert call_args["event_type"] == "batch_test_execution"
            assert call_args["user_id"] == "user1"
            assert call_args["total_tests"] == 5
            assert call_args["successful_tests"] == 4
            assert call_args["failed_tests"] == 1
            assert call_args["success_rate"] == 0.8


class TestGetSupportedLanguages:
    """Test the get_supported_languages endpoint."""

    async def test_get_languages_success(self):
        """Test successful retrieval of supported languages."""
        from routers.test_execution import get_supported_languages

        mock_orchestrator = AsyncMock()
        mock_orchestrator.get_supported_languages = AsyncMock(
            return_value=["python", "javascript", "go"]
        )

        mock_spec = MagicMock()
        mock_spec.image = "python:3.11-slim"
        mock_spec.version = "3.11"

        mock_registry = MagicMock()
        mock_registry.get_container_spec = MagicMock(return_value=mock_spec)

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        with (
            patch(
                "routers.test_execution.get_test_orchestrator",
                return_value=mock_orchestrator,
            ),
            patch(
                "routers.test_execution.get_container_registry",
                return_value=mock_registry,
            ),
        ):
            result = await get_supported_languages(current_user=mock_user, tenant_id="n8n-dev")

        assert result["total_languages"] == 3
        assert result["supported_languages"] == ["python", "javascript", "go"]
        assert len(result["available_containers"]) == 3
        for container in result["available_containers"]:
            assert container["image"] == "python:3.11-slim"
            assert container["version"] == "3.11"
            assert container["available"] is True
            assert container["language"] in ["python", "javascript", "go"]

    async def test_get_languages_empty(self):
        """Test 404 when no languages available."""
        from routers.test_execution import get_supported_languages

        mock_orchestrator = AsyncMock()
        mock_orchestrator.get_supported_languages = AsyncMock(return_value=[])

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        with (
            patch(
                "routers.test_execution.get_test_orchestrator",
                return_value=mock_orchestrator,
            ),
            patch("routers.test_execution.get_container_registry", return_value=MagicMock()),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_supported_languages(current_user=mock_user, tenant_id="n8n-dev")
            assert exc_info.value.status_code == 404
            assert "Supported languages" in str(exc_info.value.detail)

    async def test_get_languages_exception(self):
        """Test 500 on unexpected error."""
        from routers.test_execution import get_supported_languages

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        with patch(
            "routers.test_execution.get_test_orchestrator",
            side_effect=RuntimeError("connection failed"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_supported_languages(current_user=mock_user, tenant_id="n8n-dev")
            assert exc_info.value.status_code == 500
            assert "connection failed" in str(exc_info.value.detail)


class TestGetTestTypesForLanguage:
    """Test the get_test_types_for_language endpoint."""

    async def test_get_test_types_supported(self):
        """Test getting test types for a supported language."""
        from routers.test_execution import get_test_types_for_language

        mock_orchestrator = AsyncMock()
        mock_orchestrator.get_supported_languages = AsyncMock(return_value=["python", "javascript"])
        mock_orchestrator.get_supported_test_types = AsyncMock(
            return_value=["unit", "integration", "coverage"]
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        with patch(
            "routers.test_execution.get_test_orchestrator",
            return_value=mock_orchestrator,
        ):
            result = await get_test_types_for_language(
                language="python",
                current_user=mock_user,
                tenant_id="n8n-dev",
            )

        assert result["status"] == "success"
        assert result["data"]["language"] == "python"
        assert result["data"]["count"] == 3
        assert result["data"]["test_types"] == ["unit", "integration", "coverage"]

    async def test_get_test_types_unsupported_language(self):
        """Test 404 for unsupported language."""
        from routers.test_execution import get_test_types_for_language

        mock_orchestrator = AsyncMock()
        mock_orchestrator.get_supported_languages = AsyncMock(return_value=["python", "javascript"])

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        with patch(
            "routers.test_execution.get_test_orchestrator",
            return_value=mock_orchestrator,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_test_types_for_language(
                    language="cobol",
                    current_user=mock_user,
                    tenant_id="n8n-dev",
                )
            assert exc_info.value.status_code == 404
            assert "cobol" in str(exc_info.value.detail).lower()

    async def test_get_test_types_exception_returns_defaults(self):
        """Test that exception returns default test types instead of failing."""
        from routers.test_execution import get_test_types_for_language

        mock_orchestrator = AsyncMock()
        mock_orchestrator.get_supported_languages = AsyncMock(return_value=["python"])
        mock_orchestrator.get_supported_test_types = AsyncMock(side_effect=RuntimeError("failed"))

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        with patch(
            "routers.test_execution.get_test_orchestrator",
            return_value=mock_orchestrator,
        ):
            result = await get_test_types_for_language(
                language="python",
                current_user=mock_user,
                tenant_id="n8n-dev",
            )

        # Should return default test types instead of raising
        assert result["status"] == "success"
        assert result["data"]["language"] == "python"
        assert result["data"]["count"] == 3
        assert result["data"]["test_types"] == ["unit", "integration", "coverage"]


class TestExecuteBatchTests:
    """Test the execute_batch_tests endpoint."""

    async def test_execute_batch_success(self):
        """Test successful batch execution."""
        from routers.test_execution import execute_batch_tests
        from services.test_execution.test_orchestrator import (
            OrchestratorType,
            TestExecutionResult,
            TestLanguage,
        )

        result1 = TestExecutionResult(
            test_id="t1",
            success=True,
            exit_code=0,
            stdout="ok",
            stderr="",
            execution_time_ms=10.0,
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            created_at=datetime.now(UTC),
            container_name="c1",
        )
        result2 = TestExecutionResult(
            test_id="t2",
            success=True,
            exit_code=0,
            stdout="ok",
            stderr="",
            execution_time_ms=20.0,
            language=TestLanguage.JAVASCRIPT,
            test_type=OrchestratorType.UNIT,
            created_at=datetime.now(UTC),
            container_name="c1",
        )

        mock_orchestrator = AsyncMock()
        mock_orchestrator.execute_tests = AsyncMock(side_effect=[result1, result2])

        mock_req1 = MagicMock()
        mock_req1.container_name = None
        mock_req1.language = TestLanguage.PYTHON
        mock_req1.test_type = OrchestratorType.UNIT

        mock_req2 = MagicMock()
        mock_req2.container_name = None
        mock_req2.language = TestLanguage.JAVASCRIPT
        mock_req2.test_type = OrchestratorType.UNIT

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_bg = MagicMock()

        with patch(
            "routers.test_execution.get_test_orchestrator",
            return_value=mock_orchestrator,
        ):
            results = await execute_batch_tests(
                requests=[mock_req1, mock_req2],
                background_tasks=mock_bg,
                max_parallel=4,
                current_user=mock_user,
                tenant_id="n8n-dev",
            )

        assert len(results) == 2
        assert results[0].test_id == "t1"
        assert results[0].success is True
        assert results[0].exit_code == 0
        assert results[0].stdout == "ok"
        assert results[0].execution_time_ms == 10.0
        assert results[1].test_id == "t2"
        assert results[1].success is True
        assert results[1].exit_code == 0
        assert results[1].execution_time_ms == 20.0
        # Verify background task for batch metrics was added
        mock_bg.add_task.assert_called_once()

    async def test_execute_batch_partial_failure(self):
        """Test batch execution with some failures."""
        from routers.test_execution import execute_batch_tests
        from services.test_execution.test_orchestrator import (
            OrchestratorType,
            TestExecutionResult,
            TestLanguage,
        )

        result1 = TestExecutionResult(
            test_id="t1",
            success=True,
            exit_code=0,
            stdout="ok",
            stderr="",
            execution_time_ms=10.0,
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            created_at=datetime.now(UTC),
            container_name="c1",
        )

        mock_orchestrator = AsyncMock()
        # First succeeds, second raises
        mock_orchestrator.execute_tests = AsyncMock(
            side_effect=[result1, RuntimeError("Docker crashed")]
        )

        mock_req1 = MagicMock()
        mock_req1.container_name = None
        mock_req1.language = TestLanguage.PYTHON
        mock_req1.test_type = OrchestratorType.UNIT

        mock_req2 = MagicMock()
        mock_req2.container_name = None
        mock_req2.language = TestLanguage.JAVASCRIPT
        mock_req2.test_type = OrchestratorType.UNIT

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_bg = MagicMock()

        with patch(
            "routers.test_execution.get_test_orchestrator",
            return_value=mock_orchestrator,
        ):
            results = await execute_batch_tests(
                requests=[mock_req1, mock_req2],
                background_tasks=mock_bg,
                max_parallel=4,
                current_user=mock_user,
                tenant_id="n8n-dev",
            )

        assert len(results) == 2
        assert results[0].test_id == "t1"
        assert results[0].success is True
        assert results[0].exit_code == 0
        assert results[1].success is False
        assert results[1].exit_code == 1
        assert results[1].stdout == ""
        assert "Docker crashed" in results[1].stderr
        assert results[1].execution_time_ms == 0.0

    async def test_batch_container_validation_failure(self):
        """Batch execute with failing container validation (line 371)."""
        from routers.test_execution import execute_batch_tests

        mock_req = MagicMock()
        mock_req.container_name = "nonexistent"

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_bg = MagicMock()

        with patch(
            "routers.test_execution.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
            side_effect=HTTPException(status_code=404, detail="Not found"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await execute_batch_tests(
                    requests=[mock_req],
                    background_tasks=mock_bg,
                    max_parallel=4,
                    current_user=mock_user,
                    tenant_id="n8n-dev",
                )
            assert exc_info.value.status_code == 500
            assert "Batch execution failed" in str(exc_info.value.detail)

    async def test_batch_general_exception(self):
        """Batch execute with general exception (lines 408-410)."""
        from routers.test_execution import execute_batch_tests

        mock_req = MagicMock()
        mock_req.container_name = None

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_bg = MagicMock()

        with patch(
            "routers.test_execution.get_test_orchestrator",
            side_effect=RuntimeError("orchestrator failed"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await execute_batch_tests(
                    requests=[mock_req],
                    background_tasks=mock_bg,
                    max_parallel=4,
                    current_user=mock_user,
                    tenant_id="n8n-dev",
                )
            assert exc_info.value.status_code == 500
            assert "orchestrator failed" in str(exc_info.value.detail)


class TestGetActiveExecutions:
    """Test the get_active_executions endpoint."""

    async def test_get_active_executions_with_results(self):
        """Test getting active executions returns them."""
        from routers.test_execution import get_active_executions

        mock_orchestrator = AsyncMock()
        mock_orchestrator.active_containers = {
            "test-1": "container-abc",
            "test-2": "container-def",
        }
        mock_orchestrator.active_container_tenants = {
            "test-1": "n8n-dev",
            "test-2": "n8n-dev",
        }

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        with patch(
            "routers.test_execution.get_test_orchestrator",
            return_value=mock_orchestrator,
        ):
            result = await get_active_executions(current_user=mock_user, tenant_id="n8n-dev")

        assert result["status"] == "success"
        assert result["data"]["active_count"] == 2
        assert len(result["data"]["active_executions"]) == 2
        # Verify each execution entry has the expected structure and values
        executions = result["data"]["active_executions"]
        test_ids = {e["test_id"] for e in executions}
        assert test_ids == {"test-1", "test-2"}
        for execution in executions:
            assert execution["status"] == "running"
            assert "container_id" in execution
        # Verify correct container_id mapping
        by_id = {e["test_id"]: e for e in executions}
        assert by_id["test-1"]["container_id"] == "container-abc"
        assert by_id["test-2"]["container_id"] == "container-def"

    async def test_get_active_executions_empty(self):
        """Test getting active executions when none exist."""
        from routers.test_execution import get_active_executions

        mock_orchestrator = AsyncMock()
        mock_orchestrator.active_containers = {}

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        with patch(
            "routers.test_execution.get_test_orchestrator",
            return_value=mock_orchestrator,
        ):
            result = await get_active_executions(current_user=mock_user, tenant_id="n8n-dev")

        assert result["status"] == "success"
        assert result["data"]["active_count"] == 0
        assert result["data"]["active_executions"] == []

    async def test_get_active_executions_exception_returns_empty(self):
        """Test that exception returns empty result instead of 500."""
        from routers.test_execution import get_active_executions

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        with patch(
            "routers.test_execution.get_test_orchestrator",
            side_effect=RuntimeError("service down"),
        ):
            result = await get_active_executions(current_user=mock_user, tenant_id="n8n-dev")

        assert result["status"] == "success"
        assert result["data"]["active_count"] == 0
        assert result["data"]["active_executions"] == []

    async def test_get_active_executions_only_returns_caller_tenant(self):
        """S8 authz: /active must NOT leak other tenants' executions."""
        from routers.test_execution import get_active_executions

        mock_orchestrator = AsyncMock()
        # Two tenants have active executions on this replica.
        mock_orchestrator.active_containers = {
            "test-mine": "container-mine",
            "test-theirs": "container-theirs",
        }
        mock_orchestrator.active_container_tenants = {
            "test-mine": "tenant-a",
            "test-theirs": "tenant-b",
        }

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        with patch(
            "routers.test_execution.get_test_orchestrator",
            return_value=mock_orchestrator,
        ):
            result = await get_active_executions(current_user=mock_user, tenant_id="tenant-a")

        # Only tenant-a's execution is visible; tenant-b's is filtered out.
        assert result["data"]["active_count"] == 1
        executions = result["data"]["active_executions"]
        assert len(executions) == 1
        assert executions[0]["test_id"] == "test-mine"
        assert executions[0]["container_id"] == "container-mine"
        test_ids = {e["test_id"] for e in executions}
        assert "test-theirs" not in test_ids


class TestCleanupTestExecution:
    """Test the cleanup_test_execution endpoint."""

    async def test_cleanup_success(self):
        """Test successful cleanup of a test execution."""
        from routers.test_execution import cleanup_test_execution

        mock_orchestrator = AsyncMock()
        mock_orchestrator.active_containers = {"test-1": "container-abc"}
        mock_orchestrator.active_container_tenants = {"test-1": "n8n-dev"}
        mock_orchestrator._cleanup_container = AsyncMock()

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        with patch(
            "routers.test_execution.get_test_orchestrator",
            return_value=mock_orchestrator,
        ):
            result = await cleanup_test_execution(
                test_id="test-1",
                current_user=mock_user,
                tenant_id="n8n-dev",
            )

        assert result["test_id"] == "test-1"
        assert result["status"] == "cleaned"
        assert result["message"] == "Test execution resources cleaned up successfully"
        mock_orchestrator._cleanup_container.assert_called_once_with("test-1", "container-abc")

    async def test_cleanup_not_found(self):
        """Test cleanup raises 404 for nonexistent test."""
        from routers.test_execution import cleanup_test_execution

        mock_orchestrator = AsyncMock()
        mock_orchestrator.active_containers = {}

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        with patch(
            "routers.test_execution.get_test_orchestrator",
            return_value=mock_orchestrator,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await cleanup_test_execution(
                    test_id="nonexistent",
                    current_user=mock_user,
                    tenant_id="n8n-dev",
                )
            assert exc_info.value.status_code == 404
            assert "nonexistent" in str(exc_info.value.detail).lower()

    async def test_cleanup_general_exception(self):
        """Test cleanup handles general exception (lines 531-533)."""
        from routers.test_execution import cleanup_test_execution

        mock_orchestrator = AsyncMock()
        mock_orchestrator.active_containers = {"test-1": "container-abc"}
        mock_orchestrator.active_container_tenants = {"test-1": "n8n-dev"}
        mock_orchestrator._cleanup_container = AsyncMock(side_effect=RuntimeError("cleanup failed"))

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        with patch(
            "routers.test_execution.get_test_orchestrator",
            return_value=mock_orchestrator,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await cleanup_test_execution(
                    test_id="test-1",
                    current_user=mock_user,
                    tenant_id="n8n-dev",
                )
            assert exc_info.value.status_code == 500
            assert "cleanup failed" in str(exc_info.value.detail)

    async def test_cleanup_other_tenant_not_destroyed(self):
        """S8 authz: cleaning another tenant's test_id must 404 and NOT destroy it."""
        from routers.test_execution import cleanup_test_execution

        mock_orchestrator = AsyncMock()
        # test-1 belongs to tenant-b; caller is tenant-a.
        mock_orchestrator.active_containers = {"test-1": "container-abc"}
        mock_orchestrator.active_container_tenants = {"test-1": "tenant-b"}
        mock_orchestrator._cleanup_container = AsyncMock()

        mock_user = MagicMock()
        mock_user.preferred_username = "attacker"

        with patch(
            "routers.test_execution.get_test_orchestrator",
            return_value=mock_orchestrator,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await cleanup_test_execution(
                    test_id="test-1",
                    current_user=mock_user,
                    tenant_id="tenant-a",
                )

        # Indistinguishable from not-found; the container must NOT be destroyed.
        assert exc_info.value.status_code == 404
        mock_orchestrator._cleanup_container.assert_not_called()


class TestGetContainerSpecifications:
    """Test the get_container_specifications endpoint."""

    async def test_get_specs_all_languages(self):
        """Test getting specs for all languages."""
        from routers.test_execution import get_container_specifications

        mock_spec = MagicMock()
        mock_spec.image = "python:3.11-slim"
        mock_spec.version = "3.11"

        mock_registry = MagicMock()
        mock_registry.get_container_spec = MagicMock(return_value=mock_spec)

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        with patch(
            "routers.test_execution.get_container_registry",
            return_value=mock_registry,
        ):
            result = await get_container_specifications(
                language=None,
                current_user=mock_user,
                tenant_id="n8n-dev",
            )

        # The source iterates 8 hardcoded languages; mock returns a spec for each
        expected_languages = {
            "python",
            "go",
            "rust",
            "java",
            "csharp",
            "cpp",
            "javascript",
            "typescript",
        }
        assert set(result["specifications"].keys()) == expected_languages
        assert result["count"] == 8
        for lang, spec_data in result["specifications"].items():
            assert spec_data["image"] == "python:3.11-slim"
            assert spec_data["version"] == "3.11"
            assert spec_data["available"] is True

    async def test_get_specs_specific_language(self):
        """Test getting spec for a specific language."""
        from routers.test_execution import get_container_specifications

        mock_spec = MagicMock()
        mock_spec.image = "python:3.11-slim"
        mock_spec.version = "3.11"
        mock_spec.memory_limit = "512m"
        mock_spec.cpu_quota = 100000
        mock_spec.environment = {"PYTHONUNBUFFERED": "1"}
        mock_spec.mounts = []
        mock_spec.security_opts = ["no-new-privileges:true"]

        mock_registry = MagicMock()
        mock_registry.get_container_spec = MagicMock(return_value=mock_spec)

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        with patch(
            "routers.test_execution.get_container_registry",
            return_value=mock_registry,
        ):
            result = await get_container_specifications(
                language="python",
                current_user=mock_user,
                tenant_id="n8n-dev",
            )

        assert result["language"] == "python"
        assert result["specification"]["image"] == "python:3.11-slim"
        assert result["specification"]["version"] == "3.11"
        assert result["specification"]["memory_limit"] == "512m"
        assert result["specification"]["cpu_quota"] == 100000
        assert result["specification"]["environment"] == {"PYTHONUNBUFFERED": "1"}
        assert result["specification"]["mounts"] == []
        assert result["specification"]["security_opts"] == ["no-new-privileges:true"]

    async def test_get_specs_unknown_language(self):
        """Test 404 for unknown language."""
        from routers.test_execution import get_container_specifications

        mock_registry = MagicMock()
        mock_registry.get_container_spec = MagicMock(return_value=None)

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        with patch(
            "routers.test_execution.get_container_registry",
            return_value=mock_registry,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_container_specifications(
                    language="fortran",
                    current_user=mock_user,
                    tenant_id="n8n-dev",
                )
            assert exc_info.value.status_code == 404
            assert "fortran" in str(exc_info.value.detail).lower()

    async def test_get_specs_general_exception(self):
        """Test general exception in get_container_specifications (lines 648-650)."""
        from routers.test_execution import get_container_specifications

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        with patch(
            "routers.test_execution.get_container_registry",
            side_effect=RuntimeError("registry broke"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_container_specifications(
                    language=None,
                    current_user=mock_user,
                    tenant_id="n8n-dev",
                )
            assert exc_info.value.status_code == 500
            assert "registry broke" in str(exc_info.value.detail)


class TestHealthEndpoint:
    """Test the test_execution_health endpoint."""

    async def test_health_healthy(self):
        """Test health endpoint when all components are healthy."""
        from routers.test_execution import test_execution_health

        mock_orchestrator = AsyncMock()
        mock_orchestrator.active_containers = {}

        mock_registry = MagicMock()
        mock_registry.get_supported_languages = MagicMock(return_value=["python", "javascript"])

        mock_docker = MagicMock()

        with (
            patch(
                "routers.test_execution.get_test_orchestrator",
                return_value=mock_orchestrator,
            ),
            patch(
                "routers.test_execution.get_container_registry",
                return_value=mock_registry,
            ),
            patch(
                "routers.test_execution.get_docker_service",
                return_value=mock_docker,
            ),
        ):
            result = await test_execution_health(tenant_id="n8n-dev")

        assert result["status"] == "success"
        assert result["data"]["status"] == "healthy"
        assert result["data"]["service"] == "test_execution"
        assert "timestamp" in result["data"]
        assert result["data"]["components"]["orchestrator"]["status"] == "healthy"
        assert result["data"]["components"]["orchestrator"]["active_tests"] == 0
        assert result["data"]["components"]["registry"]["status"] == "healthy"
        assert result["data"]["components"]["registry"]["supported_languages"] == 2
        assert result["data"]["components"]["docker"]["status"] == "healthy"

    async def test_health_degraded(self):
        """Test health endpoint when some components fail."""
        from routers.test_execution import test_execution_health

        mock_orchestrator = AsyncMock()
        mock_orchestrator.active_containers = {}

        with (
            patch(
                "routers.test_execution.get_test_orchestrator",
                return_value=mock_orchestrator,
            ),
            patch(
                "routers.test_execution.get_container_registry",
                side_effect=RuntimeError("Registry init failed"),
            ),
            patch(
                "routers.test_execution.get_docker_service",
                return_value=MagicMock(),
            ),
        ):
            result = await test_execution_health(tenant_id="n8n-dev")

        assert result["data"]["status"] == "degraded"
        assert result["data"]["service"] == "test_execution"
        assert result["data"]["components"]["orchestrator"]["status"] == "healthy"
        assert result["data"]["components"]["orchestrator"]["active_tests"] == 0
        assert result["data"]["components"]["registry"]["status"] == "unhealthy"
        assert "Registry init failed" in result["data"]["components"]["registry"]["error"]
        assert result["data"]["components"]["docker"]["status"] == "healthy"

    async def test_health_orchestrator_error(self):
        """Health endpoint with orchestrator failure (lines 737-739)."""
        from routers.test_execution import test_execution_health

        mock_registry = MagicMock()
        mock_registry.get_supported_languages = MagicMock(return_value=["python"])

        with (
            patch(
                "routers.test_execution.get_test_orchestrator",
                side_effect=RuntimeError("orchestrator down"),
            ),
            patch(
                "routers.test_execution.get_container_registry",
                return_value=mock_registry,
            ),
            patch(
                "routers.test_execution.get_docker_service",
                return_value=MagicMock(),
            ),
        ):
            result = await test_execution_health(tenant_id="n8n-dev")

        assert result["data"]["status"] == "degraded"
        assert result["data"]["components"]["orchestrator"]["status"] == "unhealthy"
        assert "orchestrator down" in result["data"]["components"]["orchestrator"]["error"]
        assert result["data"]["components"]["registry"]["status"] == "healthy"
        assert result["data"]["components"]["registry"]["supported_languages"] == 1
        assert result["data"]["components"]["docker"]["status"] == "healthy"

    async def test_health_docker_error(self):
        """Health endpoint with docker failure (lines 755-757)."""
        from routers.test_execution import test_execution_health

        mock_orchestrator = AsyncMock()
        mock_orchestrator.active_containers = {}

        mock_registry = MagicMock()
        mock_registry.get_supported_languages = MagicMock(return_value=["python"])

        with (
            patch(
                "routers.test_execution.get_test_orchestrator",
                return_value=mock_orchestrator,
            ),
            patch(
                "routers.test_execution.get_container_registry",
                return_value=mock_registry,
            ),
            patch(
                "routers.test_execution.get_docker_service",
                side_effect=RuntimeError("docker down"),
            ),
        ):
            result = await test_execution_health(tenant_id="n8n-dev")

        assert result["data"]["status"] == "degraded"
        assert result["data"]["components"]["orchestrator"]["status"] == "healthy"
        assert result["data"]["components"]["orchestrator"]["active_tests"] == 0
        assert result["data"]["components"]["registry"]["status"] == "healthy"
        assert result["data"]["components"]["docker"]["status"] == "unhealthy"
        assert "docker down" in result["data"]["components"]["docker"]["error"]

    async def test_health_total_failure(self):
        """Health endpoint with top-level exception (lines 767-769)."""
        from routers.test_execution import test_execution_health

        with (
            patch(
                "routers.test_execution.get_test_orchestrator",
                side_effect=RuntimeError("crash"),
            ),
            patch(
                "routers.test_execution.get_container_registry",
                side_effect=RuntimeError("crash"),
            ),
            patch(
                "routers.test_execution.get_docker_service",
                side_effect=RuntimeError("crash"),
            ),
            patch(
                "routers.test_execution.success_response",
                side_effect=[
                    RuntimeError("total fail"),
                    {
                        "data": {
                            "status": "unhealthy",
                            "components": {},
                            "timestamp": "",
                            "error": "total fail",
                        }
                    },
                ],
            ),
        ):
            # The first call to success_response fails, triggering the outer except
            # But the outer except also calls success_response, which returns the dict
            result = await test_execution_health(tenant_id="n8n-dev")
            assert result["data"]["status"] == "unhealthy"
            assert result["data"]["error"] == "total fail"
            assert result["data"]["components"] == {}
