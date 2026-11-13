"""Executors are orchestrator-only and reject requests lacking an X-Tenant-Id
header. These tests prove the coding-testing call paths always send it (the
caller's tenant when known, DEFAULT_TENANT_ID otherwise) so the internal path
is never rejected once the executors enforce the guard.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from auth.tenant import TENANT_ID_HEADER
from services.code_execution_service import CodeExecutionService, _executor_headers


def _mock_client(json_data: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(
        return_value=json_data
        or {"success": True, "stdout": "", "stderr": "", "exit_code": 0, "execution_time": 0.0}
    )
    client = MagicMock()
    client.post = AsyncMock(return_value=resp)
    return client


def _posted_headers(client: MagicMock) -> dict:
    return client.post.call_args.kwargs["headers"]


def test_executor_headers_uses_configured_default_tenant():
    """With DEFAULT_TENANT_ID configured, a caller with no tenant falls back to it."""
    with patch("services.code_execution_service.DEFAULT_TENANT_ID", "tenant-default"):
        assert _executor_headers(None)[TENANT_ID_HEADER] == "tenant-default"


def test_executor_headers_no_tenant_configured_raises():
    """Since coding-client 1.9.0 the default is None when unset — fail loud rather
    than send an empty X-Tenant-Id (which would silently mis-scope or be rejected)."""
    with patch("services.code_execution_service.DEFAULT_TENANT_ID", None):
        with pytest.raises(RuntimeError, match="DEFAULT_TENANT_ID is unset"):
            _executor_headers(None)


def test_executor_headers_uses_provided_tenant():
    assert _executor_headers("tenant-9")[TENANT_ID_HEADER] == "tenant-9"


async def test_execute_code_sends_tenant_header():
    client = _mock_client()
    with (
        patch("services.code_execution_service._get_executor_http_client", return_value=client),
        patch("services.code_execution_service._get_executor_url", return_value="http://exec"),
    ):
        await CodeExecutionService().execute_code("print(1)", "python", tenant_id="tenant-1")
    assert _posted_headers(client)[TENANT_ID_HEADER] == "tenant-1"


async def test_execute_tests_defaults_tenant_header():
    client = _mock_client()
    with (
        patch("services.code_execution_service._get_executor_http_client", return_value=client),
        patch("services.code_execution_service._get_executor_url", return_value="http://exec"),
        patch("services.code_execution_service.DEFAULT_TENANT_ID", "tenant-default"),
    ):
        await CodeExecutionService().execute_tests("code", "test", "python")
    assert _posted_headers(client)[TENANT_ID_HEADER] == "tenant-default"


async def test_execute_test_reads_tenant_from_request():
    client = _mock_client()
    request = SimpleNamespace(language="python", code="x", test_code="y", tenant_id="tenant-req")
    with (
        patch("services.code_execution_service._get_executor_http_client", return_value=client),
        patch("services.code_execution_service._get_executor_url", return_value="http://exec"),
    ):
        await CodeExecutionService().execute_test(request)
    assert _posted_headers(client)[TENANT_ID_HEADER] == "tenant-req"


async def test_execute_test_explicit_tenant_overrides_request():
    client = _mock_client()
    request = SimpleNamespace(language="python", code="x", test_code="y", tenant_id="from-req")
    with (
        patch("services.code_execution_service._get_executor_http_client", return_value=client),
        patch("services.code_execution_service._get_executor_url", return_value="http://exec"),
    ):
        await CodeExecutionService().execute_test(request, tenant_id="explicit")
    assert _posted_headers(client)[TENANT_ID_HEADER] == "explicit"
