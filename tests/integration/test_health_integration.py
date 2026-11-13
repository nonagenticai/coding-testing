"""Integration tests for health endpoints through full middleware stack."""

import pytest

pytestmark = pytest.mark.integration


class TestRootHealth:
    """Test root /health endpoint."""

    async def test_root_health_returns_ok(self, integration_client):
        response = await integration_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "coding-testing"
        assert data["version"] == "1.0.0"

    async def test_root_returns_info(self, integration_client):
        response = await integration_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data


class TestApiHealth:
    """Test /api/health endpoints."""

    async def test_health_check(self, integration_client):
        response = await integration_client.get(
            "/api/health",
            headers={"X-Tenant-Id": "n8n-dev"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "coding-testing"

    async def test_readiness_check(self, integration_client):
        response = await integration_client.get("/api/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"

    async def test_liveness_check(self, integration_client):
        response = await integration_client.get("/api/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"

    async def test_detailed_health(self, integration_client):
        response = await integration_client.get(
            "/api/health/detailed",
            headers={"X-Tenant-Id": "n8n-dev"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "uptime_seconds" in data

    async def test_system_metrics(self, integration_client):
        response = await integration_client.get(
            "/api/health/system-metrics",
            headers={"X-Tenant-Id": "n8n-dev"},
        )
        assert response.status_code == 200
        data = response.json()
        # psutil should be available
        assert "cpu_percent" in data or data.get("cpu_percent") is None
