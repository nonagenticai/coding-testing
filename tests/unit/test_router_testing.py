"""Integration tests for the /api/tests router endpoints.

Sends real HTTP requests through the ASGI transport to verify:
- Response status codes and content types
- Response body structure with exact expected values
- Endpoint availability and routing

In test mode (SKIP_AUTH=true, ENABLE_AUTH=false), the real CodeExecutionService
is used but with no Docker/Redis.  Its _get_supported_tools returns [] and
test_orchestrator is None, so endpoints fall back to hardcoded defaults.
"""

from __future__ import annotations

import pytest

from tests.route_paths import route_paths as _route_paths

# ── Expected values from endpoint fallback paths ──────────────────────────
# When test_orchestrator is None the endpoint falls back to these lists.
FALLBACK_LANGUAGES = ["python", "javascript", "typescript"]
FALLBACK_PYTHON_TEST_TYPES = ["unit", "integration", "quality"]
FALLBACK_OTHER_TEST_TYPES = ["unit", "quality"]  # JS, TS


class TestTestingRouterDiscovery:
    """Test discovery endpoints: languages, tools, containers."""

    @pytest.mark.asyncio
    async def test_get_supported_languages(self, client):
        """GET /api/tests/languages returns the fallback language list."""
        resp = await client.get("/api/tests/languages")
        assert resp.status_code == 200
        data = resp.json()
        assert data["languages"] == FALLBACK_LANGUAGES
        assert data["total"] == 3

    @pytest.mark.asyncio
    async def test_get_supported_languages_has_details(self, client):
        """GET /api/tests/languages includes per-language detail with fallback values."""
        resp = await client.get("/api/tests/languages")
        data = resp.json()

        # Exactly the three fallback languages have detail entries
        assert set(data["details"].keys()) == set(FALLBACK_LANGUAGES)

        # Python detail
        py = data["details"]["python"]
        assert py["supported"] is True
        assert py["test_types"] == FALLBACK_PYTHON_TEST_TYPES
        # quality_tools come from executor discovery or _get_supported_tools
        assert isinstance(py["quality_tools"], list)

        # JavaScript detail
        js = data["details"]["javascript"]
        assert js["supported"] is True
        assert js["test_types"] == FALLBACK_OTHER_TEST_TYPES
        assert isinstance(js["quality_tools"], list)

        # TypeScript detail
        ts = data["details"]["typescript"]
        assert ts["supported"] is True
        assert ts["test_types"] == FALLBACK_OTHER_TEST_TYPES
        assert isinstance(ts["quality_tools"], list)

    @pytest.mark.asyncio
    async def test_get_language_test_types_python(self, client):
        """GET /api/tests/languages/python/test-types returns exact fallback types."""
        resp = await client.get("/api/tests/languages/python/test-types")
        assert resp.status_code == 200
        data = resp.json()
        assert data["language"] == "python"
        assert data["test_types"] == FALLBACK_PYTHON_TEST_TYPES
        assert data["count"] == 3

    @pytest.mark.asyncio
    async def test_get_language_test_types_javascript(self, client):
        """GET /api/tests/languages/javascript/test-types returns fallback types."""
        resp = await client.get("/api/tests/languages/javascript/test-types")
        assert resp.status_code == 200
        data = resp.json()
        assert data["language"] == "javascript"
        assert data["test_types"] == FALLBACK_OTHER_TEST_TYPES
        assert data["count"] == 2

    @pytest.mark.asyncio
    async def test_get_language_test_types_unsupported(self, client):
        """GET /api/tests/languages/brainfuck/test-types returns 404 with language in message."""
        resp = await client.get("/api/tests/languages/brainfuck/test-types")
        assert resp.status_code == 404
        body = resp.json()
        # Error envelope uses {"error": {"message": ...}}
        assert "brainfuck" in body["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_get_supported_tools_all(self, client):
        """GET /api/tests/supported-tools returns tools from the quality tool registry."""
        resp = await client.get("/api/tests/supported-tools")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["supported_languages"]) > 0
        assert data["total_tools"] > 0
        assert "python" in data["tools_by_language"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("language", ["python", "javascript"])
    async def test_get_supported_tools_by_language(self, client, language):
        """GET /api/tests/supported-tools?language=X returns tools for that language."""
        resp = await client.get(f"/api/tests/supported-tools?language={language}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["language"] == language
        assert len(data["tools"]) > 0
        assert data["count"] > 0

    @pytest.mark.asyncio
    async def test_list_test_containers(self, client):
        """GET /api/tests/containers returns empty list in test env (no container manager)."""
        resp = await client.get("/api/tests/containers")
        assert resp.status_code == 200
        data = resp.json()
        assert data["containers"] == []
        assert data["count"] == 0
        assert data["filter"] == "all"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("filter_type", ["all", "test", "provisioned"])
    async def test_list_test_containers_filters(self, client, filter_type):
        """GET /api/tests/containers?filter_type=X echoes the filter and returns empty list."""
        resp = await client.get(f"/api/tests/containers?filter_type={filter_type}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["filter"] == filter_type
        assert data["containers"] == []
        assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_list_test_containers_invalid_filter(self, client):
        """GET /api/tests/containers?filter_type=invalid returns 422 validation error."""
        resp = await client.get("/api/tests/containers?filter_type=invalid")
        assert resp.status_code == 422


class TestTestingRouterHealth:
    """Test the health and metrics endpoints."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        """GET /api/tests/health returns degraded status with ISO timestamp."""
        resp = await client.get("/api/tests/health")
        assert resp.status_code == 200
        data = resp.json()
        # Without Docker the overall status is degraded (execution_service up,
        # docker down, cache up)
        assert data["status"] == "degraded"
        assert "T" in data["timestamp"]  # ISO-8601 separator
        assert "Z" in data["timestamp"]  # UTC marker

    @pytest.mark.asyncio
    async def test_health_includes_services(self, client, monkeypatch):
        """GET /api/tests/health includes detailed service breakdown."""
        # Test asserts no executor pods reachable; clear the URL env vars set
        # by the autouse conftest fixture so health-check sees 0 total executors.
        for var in (
            "PYTHON_TEST_SERVICE_URL",
            "JAVASCRIPT_TEST_SERVICE_URL",
            "JAVA_TEST_SERVICE_URL",
            "GO_TEST_SERVICE_URL",
            "RUST_TEST_SERVICE_URL",
            "CSHARP_TEST_SERVICE_URL",
            "CPP_TEST_SERVICE_URL",
        ):
            monkeypatch.delenv(var, raising=False)
        resp = await client.get("/api/tests/health")
        data = resp.json()
        services = data["services"]

        # Execution service is healthy (instantiation succeeds)
        assert services["execution_service"]["status"] == "healthy"
        assert services["execution_service"]["available"] is True
        assert services["execution_service"]["error"] is None

        # Executors: no executor pods in test env → 0 healthy / 0 total
        assert services["executors"]["status"] == "unhealthy"
        assert services["executors"]["healthy_count"] == 0
        assert services["executors"]["total_count"] == 0

        # Cache is healthy (RedisCacheManager instantiated)
        assert services["cache_service"]["status"] == "healthy"
        assert services["cache_service"]["available"] is True
        assert services["cache_service"]["error"] is None

        # Quality tools — registry returns known tools per language
        qt = services["quality_tools"]
        assert qt["python_tools_count"] > 0
        assert qt["javascript_tools_count"] > 0
        assert qt["total_tools"] > 0
        assert qt["error"] is None

        # Capabilities
        caps = data["capabilities"]
        assert caps["test_execution"] is False  # no executors
        assert caps["quality_analysis"] is False  # no executors + no tools
        assert caps["security_validation"] is True  # always True
        assert caps["result_caching"] is True  # cache manager present


class TestTestingRouterStructure:
    """Test the aggregator router structure itself."""

    def test_testing_router_has_routes(self, app):
        """The testing router is mounted at /api/tests with discovery sub-routes."""
        routes = _route_paths(app.routes)
        for expected in (
            "/api/tests/languages",
            "/api/tests/health",
            "/api/tests/supported-tools",
            "/api/tests/containers",
        ):
            assert expected in routes, f"missing route {expected}"

    def test_testing_module_imports(self):
        """The testing module exports a router aggregating all four sub-routers."""
        from routers.testing import router

        route_paths = _route_paths(router.routes)
        # Every sub-router contributes known paths (relative, no /api/tests prefix)
        expected = {
            "/health",
            "/languages",
            "/languages/{language}/test-types",
            "/containers",
            "/supported-tools",
            "/execute",
            "/batch",
            "/validate-security",
            "/quality-check",
            "/quality-scan",
            "/metrics/summary",
        }
        missing = expected - route_paths
        assert not missing, f"Missing routes: {missing}"

    def test_router_includes_all_sub_routers(self):
        """The router includes all four sub-routers with their endpoint paths."""
        from routers.testing import router

        route_paths = _route_paths(router.routes)
        groups = {
            "discovery": {"/languages", "/containers", "/supported-tools"},
            "health": {"/health", "/metrics/summary"},
            "execution": {"/execute", "/batch", "/validate-security"},
            "quality": {"/quality-check", "/quality-scan"},
        }
        for group_name, paths in groups.items():
            missing = paths - route_paths
            assert not missing, f"{group_name} sub-router missing routes: {missing}"
