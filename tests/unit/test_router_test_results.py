"""Unit tests for test results data router."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request


class TestHealthCheck:
    """Test the health check endpoint."""

    async def test_health_check_returns_healthy(self):
        """Test health check returns healthy status."""
        from routers.test_results_data import health_check

        result = await health_check()

        assert result["status"] == "healthy"
        assert result["service"] == "test_results_data_api"
        assert len(result["endpoints"]) == 4
        assert "timestamp" in result

    async def test_health_check_endpoints_listed(self):
        """Test health check lists all endpoints."""
        from routers.test_results_data import health_check

        result = await health_check()

        endpoints = result["endpoints"]
        assert "GET /results/{container_name}/{test_id}" in endpoints
        assert "GET /history/{container_name}" in endpoints
        assert "POST /results/{container_name}/analyzed" in endpoints
        assert "POST /results/{container_name}/computed" in endpoints


class TestGetRawTestResult:
    """Test the get_raw_test_result endpoint."""

    async def test_get_result_success(self):
        """Test successful retrieval of a raw test result."""
        from routers.test_results_data import get_raw_test_result

        mock_storage = AsyncMock()
        mock_storage.get_test_result = AsyncMock(
            return_value={
                "test_id": "test-123",
                "container_name": "my-container",
                "container_id": "abc123",
                "test_name": "test_add",
                "language": "python",
                "test_type": "unit",
                "exit_code": 0,
                "stdout": "ok",
                "stderr": "",
                "execution_time_ms": 100.0,
                "success": True,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        mock_container = MagicMock()

        with (
            patch(
                "routers.test_results_data.validate_and_get_container_for_tenant",
                return_value=mock_container,
            ),
            patch(
                "routers.test_results_data.get_test_results_storage_service",
                return_value=mock_storage,
            ),
        ):
            result = await get_raw_test_result(
                container_name="my-container",
                test_id="test-123",
                current_user=mock_user,
                tenant_id="n8n-dev",
            )

        assert result.test_id == "test-123"
        assert result.container_name == "my-container"
        assert result.success is True

    async def test_get_result_not_found(self):
        """Test 404 when test result not found."""
        from routers.test_results_data import get_raw_test_result

        mock_storage = AsyncMock()
        mock_storage.get_test_result = AsyncMock(return_value=None)

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        mock_container = MagicMock()

        with (
            patch(
                "routers.test_results_data.validate_and_get_container_for_tenant",
                return_value=mock_container,
            ),
            patch(
                "routers.test_results_data.get_test_results_storage_service",
                return_value=mock_storage,
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_raw_test_result(
                    container_name="my-container",
                    test_id="nonexistent",
                    current_user=mock_user,
                    tenant_id="n8n-dev",
                )
            assert exc_info.value.status_code == 404

    async def test_get_result_container_not_found(self):
        """Test 404 when container not found."""
        from routers.test_results_data import get_raw_test_result

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        with patch(
            "routers.test_results_data.validate_and_get_container_for_tenant",
            side_effect=HTTPException(status_code=404, detail="Container not found"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_raw_test_result(
                    container_name="nonexistent",
                    test_id="test-123",
                    current_user=mock_user,
                    tenant_id="n8n-dev",
                )
            assert exc_info.value.status_code == 404

    async def test_get_result_internal_error(self):
        """Test 500 on storage error."""
        from routers.test_results_data import get_raw_test_result

        mock_storage = AsyncMock()
        mock_storage.get_test_result = AsyncMock(side_effect=RuntimeError("DB connection failed"))

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        mock_container = MagicMock()

        with (
            patch(
                "routers.test_results_data.validate_and_get_container_for_tenant",
                return_value=mock_container,
            ),
            patch(
                "routers.test_results_data.get_test_results_storage_service",
                return_value=mock_storage,
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_raw_test_result(
                    container_name="my-container",
                    test_id="test-123",
                    current_user=mock_user,
                    tenant_id="n8n-dev",
                )
            assert exc_info.value.status_code == 500


class TestGetTestHistory:
    """Test the get_test_history endpoint."""

    async def test_get_history_success(self):
        """Test successful history retrieval."""
        from routers.test_results_data import get_test_history

        mock_storage = AsyncMock()
        mock_storage.get_test_history = AsyncMock(
            return_value={
                "container_name": "my-container",
                "total_tests": 10,
                "returned_tests": 5,
                "test_results": [
                    {"test_id": "t1", "success": True},
                    {"test_id": "t2", "success": False},
                ],
                "filters_applied": {"language": "python"},
                "query_timestamp": datetime.now(UTC),
            }
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        mock_container = MagicMock()

        with (
            patch(
                "routers.test_results_data.validate_and_get_container_for_tenant",
                return_value=mock_container,
            ),
            patch(
                "routers.test_results_data.get_test_results_storage_service",
                return_value=mock_storage,
            ),
        ):
            result = await get_test_history(
                container_name="my-container",
                test_name=None,
                language="python",
                success=None,
                limit=100,
                offset=0,
                current_user=mock_user,
                tenant_id="n8n-dev",
            )

        assert result.container_name == "my-container"
        assert result.total_tests == 10
        assert len(result.test_results) == 2

    async def test_get_history_container_not_found(self):
        """Test 404 when container not found."""
        from routers.test_results_data import get_test_history

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        with patch(
            "routers.test_results_data.validate_and_get_container_for_tenant",
            side_effect=HTTPException(status_code=404, detail="Not found"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_test_history(
                    container_name="nonexistent",
                    current_user=mock_user,
                    tenant_id="n8n-dev",
                )
            assert exc_info.value.status_code == 404

    async def test_get_history_internal_error(self):
        """Test 500 on storage error."""
        from routers.test_results_data import get_test_history

        mock_storage = AsyncMock()
        mock_storage.get_test_history = AsyncMock(side_effect=RuntimeError("DB error"))

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        mock_container = MagicMock()

        with (
            patch(
                "routers.test_results_data.validate_and_get_container_for_tenant",
                return_value=mock_container,
            ),
            patch(
                "routers.test_results_data.get_test_results_storage_service",
                return_value=mock_storage,
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_test_history(
                    container_name="my-container",
                    current_user=mock_user,
                    tenant_id="n8n-dev",
                )
            assert exc_info.value.status_code == 500


class TestStoreAnalyzedResult:
    """Test the store_analyzed_result endpoint."""

    async def test_store_success(self):
        """Test successful storage of analyzed result."""
        from routers.test_results_data import (
            AnalyzedTestResultRequest,
            store_analyzed_result,
        )

        request = AnalyzedTestResultRequest(
            test_id="test-123",
            analysis_id="analysis-456",
            root_causes=[{"category": "semantic", "description": "Missing return"}],
            fix_suggestions=[{"title": "Add return", "description": "Add return stmt"}],
            detected_patterns=[],
            analysis_metadata={"model": "deepseek-chat"},
        )

        mock_storage = AsyncMock()
        mock_storage.store_analysis_result = AsyncMock()

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        mock_container = MagicMock()

        with (
            patch(
                "routers.test_results_data.validate_and_get_container_for_tenant",
                return_value=mock_container,
            ),
            patch(
                "routers.test_results_data.get_test_results_storage_service",
                return_value=mock_storage,
            ),
        ):
            result = await store_analyzed_result(
                request=request,
                container_name="my-container",
                current_user=mock_user,
                tenant_id="n8n-dev",
            )

        assert result.success is True
        assert result.test_id == "test-123"
        assert result.analysis_id == "analysis-456"
        assert result.container_name == "my-container"

    async def test_store_test_not_found(self):
        """Test 404 when test not found during storage."""
        from routers.test_results_data import (
            AnalyzedTestResultRequest,
            store_analyzed_result,
        )

        request = AnalyzedTestResultRequest(
            test_id="nonexistent",
            analysis_id="a1",
            root_causes=[],
            fix_suggestions=[],
            detected_patterns=[],
            analysis_metadata={"model": "test"},
        )

        mock_storage = AsyncMock()
        mock_storage.store_analysis_result = AsyncMock(side_effect=ValueError("Test not found"))

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        mock_container = MagicMock()

        with (
            patch(
                "routers.test_results_data.validate_and_get_container_for_tenant",
                return_value=mock_container,
            ),
            patch(
                "routers.test_results_data.get_test_results_storage_service",
                return_value=mock_storage,
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await store_analyzed_result(
                    request=request,
                    container_name="my-container",
                    current_user=mock_user,
                    tenant_id="n8n-dev",
                )
            assert exc_info.value.status_code == 404

    async def test_store_internal_error(self):
        """Test 500 on unexpected storage error."""
        from routers.test_results_data import (
            AnalyzedTestResultRequest,
            store_analyzed_result,
        )

        request = AnalyzedTestResultRequest(
            test_id="test-123",
            analysis_id="a1",
            root_causes=[],
            fix_suggestions=[],
            detected_patterns=[],
            analysis_metadata={"model": "test"},
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        with patch(
            "routers.test_results_data.validate_and_get_container_for_tenant",
            side_effect=RuntimeError("connection error"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await store_analyzed_result(
                    request=request,
                    container_name="my-container",
                    current_user=mock_user,
                    tenant_id="n8n-dev",
                )
            assert exc_info.value.status_code == 500


class TestStoreAnalyzedResultAuthz:
    """S8 authz: store_analyzed_result is a WRITE and must require write permission.

    Exercised through the endpoint's ACTUAL wired dependency (not a hardcoded
    reference), so a regression back to ``requires_read_permission`` — which would
    let a read-only ``viewer`` persist forged analysis results — fails the suite.
    """

    def _wired_permission_dependency(self):
        import inspect

        from routers.test_results_data import store_analyzed_result

        sig = inspect.signature(store_analyzed_result)
        depends = sig.parameters["current_user"].default
        # FastAPI Depends wraps the callable on `.dependency`.
        return depends.dependency

    def test_endpoint_is_gated_by_write_permission(self):
        """The endpoint must be wired to requires_write_permission, not read."""
        from shared.dependencies import requires_write_permission

        assert self._wired_permission_dependency() is requires_write_permission

    async def test_viewer_is_rejected(self):
        """A read-only viewer is rejected (403) by the wired dependency."""
        from coding_client.auth.models import User

        from shared.dependencies import PermissionException

        dependency = self._wired_permission_dependency()
        viewer = User(sub="v1", preferred_username="v1", realm_access={"roles": ["viewer"]})

        with pytest.raises(PermissionException) as exc_info:
            await dependency(request=Request({"type": "http", "headers": []}), user=viewer)
        assert exc_info.value.status_code == 403
        assert exc_info.value.error_code == "WRITE_PERMISSION_REQUIRED"

    async def test_developer_is_allowed(self):
        """A developer passes the wired dependency."""
        from coding_client.auth.models import User

        dependency = self._wired_permission_dependency()
        developer = User(sub="d1", preferred_username="d1", realm_access={"roles": ["developer"]})

        result = await dependency(request=Request({"type": "http", "headers": []}), user=developer)
        assert result is developer


class TestResponseModels:
    """Test Pydantic response models."""

    def test_raw_test_result_response(self):
        """Test RawTestResultResponse model creation."""
        from routers.test_results_data import RawTestResultResponse

        result = RawTestResultResponse(
            test_id="test-1",
            container_name="c1",
            container_id="abc",
            test_name="test_add",
            language="python",
            test_type="unit",
            exit_code=0,
            stdout="PASSED",
            stderr="",
            execution_time_ms=50.0,
            success=True,
            timestamp=datetime.now(UTC),
        )

        assert result.test_id == "test-1"
        assert result.success is True
        assert result.source_code is None

    def test_test_history_response(self):
        """Test TestHistoryResponse model creation."""
        from routers.test_results_data import TestHistoryResponse

        result = TestHistoryResponse(
            container_name="c1",
            total_tests=10,
            returned_tests=5,
            test_results=[{"test_id": "t1", "success": True}],
            filters_applied={"language": "python"},
            query_timestamp=datetime.now(UTC),
        )

        assert result.total_tests == 10
        assert result.returned_tests == 5
        assert len(result.test_results) == 1

    def test_analyzed_test_result_request(self):
        """Test AnalyzedTestResultRequest model creation."""
        from routers.test_results_data import AnalyzedTestResultRequest

        request = AnalyzedTestResultRequest(
            test_id="t1",
            analysis_id="a1",
            root_causes=[],
            fix_suggestions=[],
            detected_patterns=[],
            analysis_metadata={"model": "test"},
        )

        assert request.test_id == "t1"
        assert request.analysis_id == "a1"
        assert request.root_causes == []

    def test_analyzed_test_result_response(self):
        """Test AnalyzedTestResultResponse model creation."""
        from routers.test_results_data import AnalyzedTestResultResponse

        response = AnalyzedTestResultResponse(
            success=True,
            test_id="t1",
            analysis_id="a1",
            container_name="c1",
            stored_at=datetime.now(UTC),
            message="Stored",
        )

        assert response.success is True
        assert response.message == "Stored"
