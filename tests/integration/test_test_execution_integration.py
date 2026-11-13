"""Integration tests for test execution endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def mock_orchestrator():
    """Mock the test orchestrator."""
    orchestrator = AsyncMock()
    orchestrator.get_supported_languages = AsyncMock(
        return_value=["python", "javascript", "go", "rust", "java", "csharp", "cpp", "typescript"]
    )
    orchestrator.get_supported_test_types = AsyncMock(
        return_value=["unit", "integration", "coverage"]
    )
    orchestrator.active_containers = {}
    return orchestrator


@pytest.fixture
def mock_container_registry():
    """Mock the container registry."""
    registry = MagicMock()
    registry.get_supported_languages = MagicMock(
        return_value=["python", "javascript", "go", "rust"]
    )
    spec = MagicMock()
    spec.image = "python:3.11-slim"
    spec.version = "3.11"
    spec.memory_limit = "512m"
    spec.cpu_quota = 100000
    spec.environment = {}
    spec.mounts = []
    spec.security_opts = []
    registry.get_container_spec = MagicMock(return_value=spec)
    return registry


class TestLanguagesEndpoint:
    """Test /api/tests/execute/languages endpoint."""

    async def test_get_supported_languages(
        self, integration_client, mock_orchestrator, mock_container_registry
    ):
        with (
            patch(
                "routers.test_execution.get_test_orchestrator",
                return_value=mock_orchestrator,
            ),
            patch(
                "routers.test_execution.get_container_registry",
                return_value=mock_container_registry,
            ),
        ):
            response = await integration_client.get(
                "/api/tests/execute/languages",
                headers={"X-Tenant-Id": "n8n-dev"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "supported_languages" in data
            assert "total_languages" in data
            assert data["total_languages"] == 8


class TestActiveExecutions:
    """Test /api/tests/execute/active endpoint."""

    async def test_get_active_executions_empty(self, integration_client, mock_orchestrator):
        with patch(
            "routers.test_execution.get_test_orchestrator",
            return_value=mock_orchestrator,
        ):
            response = await integration_client.get(
                "/api/tests/execute/active",
                headers={"X-Tenant-Id": "n8n-dev"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["data"]["active_count"] == 0


class TestExecutionHealth:
    """Test /api/tests/execute/health endpoint."""

    async def test_execution_health(
        self, integration_client, mock_orchestrator, mock_container_registry
    ):
        with (
            patch(
                "routers.test_execution.get_test_orchestrator",
                return_value=mock_orchestrator,
            ),
            patch(
                "routers.test_execution.get_container_registry",
                return_value=mock_container_registry,
            ),
        ):
            response = await integration_client.get(
                "/api/tests/execute/health",
                headers={"X-Tenant-Id": "n8n-dev"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["data"]["service"] == "test_execution"
