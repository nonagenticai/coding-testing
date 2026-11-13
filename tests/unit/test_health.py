"""Tests for health router endpoints."""


async def test_readiness_check(client):
    """GET /api/health/ready returns ready status."""
    resp = await client.get("/api/health/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"


async def test_liveness_check(client):
    """GET /api/health/live returns alive status."""
    resp = await client.get("/api/health/live")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "alive"


async def test_health_check_with_tenant(client):
    """GET /api/health returns health with SKIP_AUTH=true."""
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "uptime_seconds" in data


async def test_detailed_health(client):
    """GET /api/health/detailed returns detailed info."""
    resp = await client.get("/api/health/detailed")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "environment" in data
    assert "uptime_seconds" in data
