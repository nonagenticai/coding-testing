"""Integration tests for test orchestration endpoints."""

from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def mock_container_validation():
    """Mock container validation to always succeed."""
    from shared.container_validation import ContainerInfo, ContainerStatus

    container = ContainerInfo(
        container_name="test-container",
        status=ContainerStatus.RUNNING,
        tenant_id="n8n-dev",
    )
    with patch(
        "routers.test_orchestration.validate_and_get_container_for_tenant",
        new_callable=AsyncMock,
        return_value=container,
    ) as mock:
        yield mock


class TestListSuites:
    """Test GET /api/tests/orchestration/suites."""

    async def test_list_suites_empty(self, pg_db, integration_client, mock_container_validation):
        response = await integration_client.get(
            "/api/tests/orchestration/suites",
            headers={"X-Tenant-Id": "n8n-dev"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["total"] == 0
        assert data["data"]["suites"] == []


class TestExecutionHistory:
    """Test GET /api/tests/orchestration/execution-history."""

    async def test_execution_history_empty(
        self, pg_db, integration_client, mock_container_validation
    ):
        response = await integration_client.get(
            "/api/tests/orchestration/execution-history",
            headers={"X-Tenant-Id": "n8n-dev"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["count"] == 0

    async def test_execution_history_with_container_filter(
        self, pg_db, integration_client, mock_container_validation
    ):
        response = await integration_client.get(
            "/api/tests/orchestration/execution-history",
            params={"container_name": "test-container"},
            headers={"X-Tenant-Id": "n8n-dev"},
        )
        assert response.status_code == 200


class TestOrchestrationMetrics:
    """Test GET /api/tests/orchestration/metrics/summary."""

    async def test_metrics_summary_empty(
        self, pg_db, integration_client, mock_container_validation
    ):
        response = await integration_client.get(
            "/api/tests/orchestration/metrics/summary",
            headers={"X-Tenant-Id": "n8n-dev"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["total_executions"] == 0
        assert data["data"]["success_rate"] == 0.0
