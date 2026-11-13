"""Tests for testing_modules endpoints through the full HTTP stack.

Covers uncovered lines in:
- health_and_metrics.py (lines 124-205, 329)
- test_discovery.py (lines 114-115, 146-148, 212-213, etc.)
- test_execution.py (lines 142-143, 166, 258, 275, 316-318, 443-445)
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from coding_client.auth.models import User
from fastapi import HTTPException

_test_user = User(
    sub="test-user",
    preferred_username="test-user",
    realm_access={"roles": ["user", "developer", "admin"]},
    tenant_id="n8n-dev",
)


def _make_stub_execution_service():
    """Create a stub CodeExecutionService with full tool registry."""
    from unittest.mock import AsyncMock, Mock

    from services.code_execution_service import CodeExecutionService

    stub_tools = {
        "python": ["ruff", "ruff-format", "ty", "bandit"],
        "javascript": ["eslint", "prettier", "jshint"],
        "typescript": ["eslint", "prettier", "tsc"],
    }
    stub_test_types = {
        "python": ["unit", "integration", "coverage", "security"],
        "javascript": ["unit", "integration", "coverage"],
        "typescript": ["unit", "integration", "coverage"],
    }
    stub_languages = ["python", "javascript", "typescript"]
    stub = Mock(spec=CodeExecutionService)
    stub._get_supported_tools = Mock(side_effect=lambda lang: stub_tools.get(lang, []))
    stub.test_orchestrator = Mock()
    stub.test_orchestrator.get_supported_languages = AsyncMock(return_value=stub_languages)
    stub.test_orchestrator.get_supported_test_types = AsyncMock(
        side_effect=lambda lang: stub_test_types.get(lang, ["unit", "quality"])
    )
    stub._validate_code_security = AsyncMock(side_effect=ValueError("Not available in test"))
    return stub


@pytest.fixture
def auth_client(app):
    """Client with proper auth overrides for endpoints requiring permissions."""
    from coding_client.auth.dependencies import get_current_user

    from auth.tenant import get_current_user_with_tenant
    from routers.testing_modules.testing_utils import get_execution_service
    from shared.dependencies import requires_read_permission, requires_write_permission

    stub_svc = _make_stub_execution_service()

    app.dependency_overrides[get_current_user] = lambda: _test_user
    app.dependency_overrides[requires_read_permission] = lambda: _test_user
    app.dependency_overrides[requires_write_permission] = lambda: _test_user
    app.dependency_overrides[get_current_user_with_tenant] = lambda: _test_user
    app.dependency_overrides[get_execution_service] = lambda: stub_svc
    return app


@pytest.fixture
async def aclient(auth_client):
    """Async HTTP client with auth overrides."""
    import httpx

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=auth_client),
        base_url="http://testserver",
    ) as c:
        yield c


# ===========================================================================
# health_and_metrics.py — /api/tests/health, /api/tests/metrics/summary
# ===========================================================================


class TestTestingHealth:
    """Tests for /api/tests/health endpoint (health_and_metrics.py)."""

    async def test_health_returns_response(self, client):
        """GET /api/tests/health returns a health status dict with expected structure."""
        resp = await client.get("/api/tests/health")
        assert resp.status_code == 200
        data = resp.json()
        # The endpoint is unauthenticated and will return degraded/unhealthy
        # since Docker/Redis aren't available in test, but the structure is fixed
        assert data["status"] in ("healthy", "degraded", "unhealthy")
        # Timestamp must be an ISO 8601 string ending with Z
        assert data["timestamp"].endswith("Z")
        assert "T" in data["timestamp"]

    async def test_health_shows_services_and_capabilities(self, client):
        """GET /api/tests/health includes services and capabilities with correct structure."""
        resp = await client.get("/api/tests/health")
        assert resp.status_code == 200
        data = resp.json()
        # The handler always builds a response with "services" and "capabilities" dicts
        # unless it hits the outer except (ValueError, RuntimeError) path
        if "services" in data:
            services = data["services"]
            # Verify all four service sections exist
            assert "execution_service" in services
            assert "executors" in services
            assert "cache_service" in services
            assert "quality_tools" in services
            # execution_service and cache_service have available/status/error
            for svc_key in ["execution_service", "cache_service"]:
                assert isinstance(services[svc_key]["available"], bool)
                assert services[svc_key]["status"] in ("healthy", "unhealthy")
            # executors has status/healthy_count/total_count/details
            assert services["executors"]["status"] in ("healthy", "unhealthy")
            assert isinstance(services["executors"]["healthy_count"], int)
            assert isinstance(services["executors"]["total_count"], int)
        if "capabilities" in data:
            caps = data["capabilities"]
            # All four capabilities are booleans
            for cap_key in [
                "test_execution",
                "quality_analysis",
                "security_validation",
                "result_caching",
            ]:
                assert isinstance(caps[cap_key], bool)


class TestTestingHealthDirect:
    """Direct tests for testing_health_check handler to cover internal logic."""

    async def test_health_check_service_init_error(self):
        """Test health check when get_execution_service raises RuntimeError."""
        from routers.testing_modules.health_and_metrics import testing_health_check

        with (
            patch(
                "routers.testing_modules.health_and_metrics.get_execution_service",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Docker not available"),
            ),
            patch.dict(
                "os.environ",
                {
                    "PYTHON_TEST_SERVICE_URL": "",
                    "JAVASCRIPT_TEST_SERVICE_URL": "",
                    "JAVA_TEST_SERVICE_URL": "",
                    "GO_TEST_SERVICE_URL": "",
                    "RUST_TEST_SERVICE_URL": "",
                    "CSHARP_TEST_SERVICE_URL": "",
                    "CPP_TEST_SERVICE_URL": "",
                },
            ),
        ):
            result = await testing_health_check(tenant_id="test")

        assert result["status"] == "unhealthy"
        assert result["services"]["execution_service"]["error"] == "Docker not available"
        assert result["services"]["execution_service"]["available"] is False
        assert result["services"]["execution_service"]["status"] == "unhealthy"
        # Executors: no env vars set → 0 healthy / 0 total
        assert result["services"]["executors"]["status"] == "unhealthy"
        assert result["services"]["executors"]["healthy_count"] == 0
        assert result["services"]["executors"]["total_count"] == 0
        # Cache should also be unavailable since svc_instance is None
        assert result["services"]["cache_service"]["available"] is False
        assert result["services"]["cache_service"]["error"] == "Cache manager not available"
        # Quality tools should all be zero
        assert result["services"]["quality_tools"]["python_tools_count"] == 0
        assert result["services"]["quality_tools"]["javascript_tools_count"] == 0
        assert result["services"]["quality_tools"]["total_tools"] == 0
        # Capabilities: all false except security_validation (always True)
        assert result["capabilities"]["test_execution"] is False
        assert result["capabilities"]["quality_analysis"] is False
        assert result["capabilities"]["security_validation"] is True
        assert result["capabilities"]["result_caching"] is False

    async def test_health_check_service_available(self):
        """Test health check when execution service is available and executors respond."""
        from routers.testing_modules.health_and_metrics import testing_health_check

        mock_svc = Mock()
        mock_svc.cache_manager = Mock()
        mock_svc._get_supported_tools = Mock(
            side_effect=lambda lang: ["tool1"] if lang == "python" else []
        )

        # Mock a healthy executor responding at the env-var URL
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "routers.testing_modules.health_and_metrics.get_execution_service",
                new_callable=AsyncMock,
                return_value=mock_svc,
            ),
            patch.dict(
                "os.environ",
                {
                    "PYTHON_TEST_SERVICE_URL": "http://localhost:8087",
                    "JAVASCRIPT_TEST_SERVICE_URL": "",
                    "JAVA_TEST_SERVICE_URL": "",
                    "GO_TEST_SERVICE_URL": "",
                    "RUST_TEST_SERVICE_URL": "",
                    "CSHARP_TEST_SERVICE_URL": "",
                    "CPP_TEST_SERVICE_URL": "",
                },
            ),
            patch(
                "routers.testing_modules.health_and_metrics._get_executor_client",
                return_value=mock_client,
            ),
        ):
            result = await testing_health_check(tenant_id="test")

        # With executor healthy, cache available, service available => healthy
        assert result["status"] == "healthy"
        assert result["services"]["execution_service"]["available"] is True
        assert result["services"]["execution_service"]["status"] == "healthy"
        assert result["services"]["executors"]["status"] == "healthy"
        assert result["services"]["executors"]["healthy_count"] == 1
        assert result["services"]["executors"]["total_count"] == 1
        assert result["services"]["cache_service"]["available"] is True
        assert result["services"]["cache_service"]["status"] == "healthy"
        assert result["services"]["cache_service"]["error"] is None
        assert result["services"]["quality_tools"]["python_tools_count"] == 1
        assert result["services"]["quality_tools"]["javascript_tools_count"] == 0
        assert result["services"]["quality_tools"]["total_tools"] == 1
        assert result["capabilities"]["test_execution"] is True
        assert result["capabilities"]["quality_analysis"] is True
        assert result["capabilities"]["security_validation"] is True
        assert result["capabilities"]["result_caching"] is True

    async def test_health_check_executor_unreachable(self):
        """Test health check when executor pod is unreachable."""
        from routers.testing_modules.health_and_metrics import testing_health_check

        mock_svc = Mock()
        mock_svc.cache_manager = None
        mock_svc._get_supported_tools = Mock(return_value=[])

        # Mock an executor URL that raises a connection error
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=ConnectionError("unreachable"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "routers.testing_modules.health_and_metrics.get_execution_service",
                new_callable=AsyncMock,
                return_value=mock_svc,
            ),
            patch.dict(
                "os.environ",
                {
                    "PYTHON_TEST_SERVICE_URL": "http://localhost:8087",
                    "JAVASCRIPT_TEST_SERVICE_URL": "",
                    "JAVA_TEST_SERVICE_URL": "",
                    "GO_TEST_SERVICE_URL": "",
                    "RUST_TEST_SERVICE_URL": "",
                    "CSHARP_TEST_SERVICE_URL": "",
                    "CPP_TEST_SERVICE_URL": "",
                },
            ),
            patch("httpx.AsyncClient", return_value=mock_client),
        ):
            result = await testing_health_check(tenant_id="test")

        # svc_instance exists but executor unreachable and cache is None => degraded
        assert result["status"] == "degraded"
        assert result["services"]["executors"]["status"] == "unhealthy"
        assert result["services"]["executors"]["healthy_count"] == 0
        assert result["services"]["executors"]["total_count"] == 1
        assert result["services"]["executors"]["details"]["python"]["status"] == "unreachable"
        # cache_manager is None
        assert result["services"]["cache_service"]["available"] is False
        # quality tools are empty
        assert result["services"]["quality_tools"]["total_tools"] == 0
        # Capabilities: no healthy executors so test_execution and quality_analysis are False
        assert result["capabilities"]["test_execution"] is False
        assert result["capabilities"]["quality_analysis"] is False
        assert result["capabilities"]["security_validation"] is True
        assert result["capabilities"]["result_caching"] is False


class TestTestingMetricsDirect:
    """Direct tests for get_testing_metrics_summary handler."""

    async def test_metrics_summary_with_analytics(self):
        """Test metrics summary with analytics data."""
        from routers.testing_modules.health_and_metrics import get_testing_metrics_summary

        mock_metrics = AsyncMock()
        mock_metrics.get_test_execution_analytics = AsyncMock(
            return_value={
                "period_days": 7,
                "test_executions": 42,
                "quality_analyses": 10,
                "resource_usage": {"cpu": 0.5},
                "languages": ["python"],
            }
        )

        with patch(
            "routers.testing_modules.health_and_metrics.get_metrics_service",
            return_value=mock_metrics,
        ):
            result = await get_testing_metrics_summary(
                days=7,
                current_user=_test_user,
                tenant_id="test",
            )

        assert result["period_days"] == 7
        assert result["test_executions"] == 42
        assert result["quality_analyses"] == 10
        assert result["resource_usage"] == {"cpu": 0.5}
        assert result["languages"] == ["python"]

    async def test_metrics_summary_fallback(self):
        """Test metrics summary with AttributeError fallback."""
        from routers.testing_modules.health_and_metrics import get_testing_metrics_summary

        mock_metrics = AsyncMock()
        mock_metrics.get_test_execution_analytics = AsyncMock(
            side_effect=AttributeError("no method")
        )

        with patch(
            "routers.testing_modules.health_and_metrics.get_metrics_service",
            return_value=mock_metrics,
        ):
            result = await get_testing_metrics_summary(
                days=7,
                current_user=_test_user,
                tenant_id="test",
            )

        # Fallback path: metrics dict doesn't have period_days, so it falls back to days param
        assert result["period_days"] == 7
        assert result["test_executions"] == 0
        assert result["quality_analyses"] == 0
        assert result["resource_usage"] == {}
        assert result["languages"] == []


class TestTestingMetrics:
    """Tests for /api/tests/metrics/summary endpoint (requires auth)."""

    async def test_metrics_summary_default(self, aclient):
        """GET /api/tests/metrics/summary with default params returns 7-day period."""
        resp = await aclient.get("/api/tests/metrics/summary")
        assert resp.status_code == 200
        data = resp.json()
        # Default days=7
        assert data["period_days"] == 7
        assert data["test_executions"] >= 0
        assert data["quality_analyses"] >= 0
        assert isinstance(data["resource_usage"], dict)
        assert isinstance(data["languages"], list)

    async def test_metrics_summary_custom_days(self, aclient):
        """GET /api/tests/metrics/summary?days=30 returns 30-day period."""
        resp = await aclient.get("/api/tests/metrics/summary?days=30")
        assert resp.status_code == 200
        data = resp.json()
        assert data["period_days"] == 30


# ===========================================================================
# test_discovery.py — /api/tests/supported-tools, /api/tests/languages,
#                     /api/tests/languages/{lang}/test-types, /api/tests/containers
# ===========================================================================


class TestDiscoveryDirect:
    """Direct tests for test_discovery.py handler functions."""

    def _make_mock_user(self, admin=False):
        return User(
            sub="test-user",
            preferred_username="test-user",
            realm_access={"roles": ["admin"] if admin else ["user"]},
            tenant_id="n8n-dev",
        )

    def _make_mock_svc(self):
        from services.code_execution_service import CodeExecutionService

        svc = Mock(spec=CodeExecutionService)
        svc._get_supported_tools = Mock(
            side_effect=lambda lang: ["tool1", "tool2"] if lang == "python" else []
        )
        svc.test_orchestrator = Mock()
        svc.test_orchestrator.get_supported_languages = AsyncMock(
            return_value=["python", "javascript"]
        )
        svc.test_orchestrator.get_supported_test_types = AsyncMock(
            side_effect=lambda lang: ["unit", "integration"] if lang == "python" else ["unit"]
        )
        return svc

    async def test_get_supported_tools_with_language(self):
        """get_supported_tools with language filter returns exact tools."""
        from routers.testing_modules.test_discovery import get_supported_tools

        svc = self._make_mock_svc()
        result = await get_supported_tools(
            language="python",
            container_name=None,
            user=self._make_mock_user(),
            execution_service=svc,
        )
        assert result["language"] == "python"
        assert result["tools"] == ["tool1", "tool2"]
        assert result["count"] == 2
        # No container_name key since we passed None
        assert "container_name" not in result

    async def test_get_supported_tools_no_language(self):
        """get_supported_tools without language filter returns all languages with tools."""
        from routers.testing_modules.test_discovery import get_supported_tools

        svc = self._make_mock_svc()
        result = await get_supported_tools(
            language=None,
            container_name=None,
            user=self._make_mock_user(),
            execution_service=svc,
        )
        # Only python has tools in our mock (javascript/typescript return [])
        assert result["supported_languages"] == ["python"]
        assert result["tools_by_language"] == {"python": ["tool1", "tool2"]}
        assert result["total_tools"] == 2
        assert "container_name" not in result

    async def test_get_supported_tools_not_found(self):
        """get_supported_tools for unknown language raises HTTPException 404."""
        from routers.testing_modules.test_discovery import get_supported_tools

        svc = self._make_mock_svc()
        with pytest.raises(HTTPException) as exc_info:
            await get_supported_tools(
                language="brainfuck",
                container_name=None,
                user=self._make_mock_user(),
                execution_service=svc,
            )
        assert exc_info.value.status_code == 404

    async def test_get_supported_tools_exception(self):
        """get_supported_tools handles unexpected exception with 500."""
        from routers.testing_modules.test_discovery import get_supported_tools

        svc = self._make_mock_svc()
        svc._get_supported_tools = Mock(side_effect=RuntimeError("boom"))

        with pytest.raises(HTTPException) as exc_info:
            await get_supported_tools(
                language="python",
                container_name=None,
                user=self._make_mock_user(),
                execution_service=svc,
            )
        assert exc_info.value.status_code == 500

    async def test_get_supported_languages_handler(self):
        """get_supported_languages returns language details with correct structure."""
        from routers.testing_modules.test_discovery import get_supported_languages

        svc = self._make_mock_svc()
        result = await get_supported_languages(
            container_name=None,
            user=self._make_mock_user(),
            execution_service=svc,
        )
        assert result["languages"] == ["python", "javascript"]
        assert result["total"] == 2
        # Verify details for each language
        assert result["details"]["python"]["supported"] is True
        assert result["details"]["python"]["test_types"] == ["unit", "integration"]
        assert result["details"]["python"]["quality_tools"] == ["tool1", "tool2"]
        assert result["details"]["javascript"]["supported"] is True
        assert result["details"]["javascript"]["test_types"] == ["unit"]
        assert result["details"]["javascript"]["quality_tools"] == []
        assert "container_name" not in result

    async def test_get_supported_languages_with_container(self):
        """get_supported_languages with container validation includes container_name."""
        from routers.testing_modules.test_discovery import get_supported_languages

        svc = self._make_mock_svc()
        with patch(
            "routers.testing_modules.test_discovery.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
        ):
            result = await get_supported_languages(
                container_name="test-container",
                user=self._make_mock_user(),
                execution_service=svc,
            )
        assert result["container_name"] == "test-container"
        assert result["languages"] == ["python", "javascript"]
        assert result["total"] == 2

    async def test_get_supported_languages_orchestrator_fallback(self):
        """get_supported_languages falls back to hardcoded languages when orchestrator raises."""
        from routers.testing_modules.test_discovery import get_supported_languages

        svc = self._make_mock_svc()
        svc.test_orchestrator.get_supported_languages = AsyncMock(
            side_effect=AttributeError("no method")
        )
        svc.test_orchestrator.get_supported_test_types = AsyncMock(
            side_effect=AttributeError("no method")
        )
        result = await get_supported_languages(
            container_name=None,
            user=self._make_mock_user(),
            execution_service=svc,
        )
        # Fallback languages are ["python", "javascript", "typescript"]
        assert result["languages"] == ["python", "javascript", "typescript"]
        assert result["total"] == 3
        # Fallback test types for python
        assert result["details"]["python"]["test_types"] == ["unit", "integration", "quality"]
        # Fallback test types for non-python
        assert result["details"]["javascript"]["test_types"] == ["unit", "quality"]
        assert result["details"]["typescript"]["test_types"] == ["unit", "quality"]

    async def test_get_language_test_types_handler(self):
        """get_language_test_types returns test types for valid language."""
        from routers.testing_modules.test_discovery import get_language_test_types

        svc = self._make_mock_svc()
        result = await get_language_test_types(
            language="python",
            container_name=None,
            user=self._make_mock_user(),
            execution_service=svc,
        )
        assert result["language"] == "python"
        assert result["test_types"] == ["unit", "integration"]
        assert result["count"] == 2
        assert "container_name" not in result

    async def test_get_language_test_types_unsupported(self):
        """get_language_test_types raises HTTPException 404 for unsupported language."""
        from routers.testing_modules.test_discovery import get_language_test_types

        svc = self._make_mock_svc()
        with pytest.raises(HTTPException) as exc_info:
            await get_language_test_types(
                language="fortran",
                container_name=None,
                user=self._make_mock_user(),
                execution_service=svc,
            )
        assert exc_info.value.status_code == 404

    async def test_get_language_test_types_fallback(self):
        """get_language_test_types uses fallback test types when orchestrator fails."""
        from routers.testing_modules.test_discovery import get_language_test_types

        svc = self._make_mock_svc()
        svc.test_orchestrator.get_supported_languages = AsyncMock(
            side_effect=AttributeError("no method")
        )
        svc.test_orchestrator.get_supported_test_types = AsyncMock(
            side_effect=AttributeError("no method")
        )
        result = await get_language_test_types(
            language="python",
            container_name=None,
            user=self._make_mock_user(),
            execution_service=svc,
        )
        # Fallback languages include python, fallback test types for python
        assert result["language"] == "python"
        assert result["test_types"] == ["unit", "integration", "quality"]
        assert result["count"] == 3

    async def test_list_test_containers_handler(self):
        """list_test_containers returns empty container list when manager unavailable."""
        from routers.testing_modules.test_discovery import list_test_containers

        # ContainerManager may not be available in test, handler returns empty
        result = await list_test_containers(
            filter_type="all",
            user=self._make_mock_user(),
        )
        assert result["containers"] == []
        assert result["count"] == 0
        assert result["filter"] == "all"

    async def test_list_test_containers_import_error(self):
        """list_test_containers gracefully handles ImportError (get_container_manager missing)."""
        from routers.testing_modules.test_discovery import list_test_containers

        # get_container_manager doesn't exist in shared.container_validation,
        # so the import naturally fails with ImportError which is caught.
        result = await list_test_containers(
            filter_type="all",
            user=self._make_mock_user(),
        )
        assert result["containers"] == []
        assert result["count"] == 0
        assert result["filter"] == "all"


class TestDiscoveryUncoveredPaths:
    """Cover remaining uncovered lines in test_discovery.py."""

    def _make_mock_user(self, admin=False):
        return User(
            sub="test-user",
            preferred_username="test-user",
            realm_access={"roles": ["admin"] if admin else ["user"]},
            tenant_id="n8n-dev",
        )

    def _make_mock_svc(self):
        from services.code_execution_service import CodeExecutionService

        svc = Mock(spec=CodeExecutionService)
        svc._get_supported_tools = Mock(
            side_effect=lambda lang: ["tool1", "tool2"] if lang == "python" else []
        )
        svc.test_orchestrator = Mock()
        svc.test_orchestrator.get_supported_languages = AsyncMock(
            return_value=["python", "javascript"]
        )
        svc.test_orchestrator.get_supported_test_types = AsyncMock(
            side_effect=lambda lang: ["unit", "integration"] if lang == "python" else ["unit"]
        )
        return svc

    async def test_get_supported_languages_empty_list_fallback(self):
        """When orchestrator returns empty list, fallback languages are used (line 224)."""
        from routers.testing_modules.test_discovery import get_supported_languages

        svc = self._make_mock_svc()
        svc.test_orchestrator.get_supported_languages = AsyncMock(return_value=[])
        result = await get_supported_languages(
            container_name=None,
            user=self._make_mock_user(),
            execution_service=svc,
        )
        assert result["languages"] == ["python", "javascript", "typescript"]
        assert result["total"] == 3

    async def test_get_supported_languages_general_exception(self):
        """Non-HTTP exception in get_supported_languages raises 500 (lines 259-263)."""
        from routers.testing_modules.test_discovery import get_supported_languages

        svc = self._make_mock_svc()
        # Make orchestrator return a non-iterable to trigger TypeError in the for loop
        svc.test_orchestrator.get_supported_languages = AsyncMock(return_value=["python"])
        svc.test_orchestrator.get_supported_test_types = AsyncMock(return_value=["unit"])

        def bad_iter():
            yield "python"
            raise TypeError("generator broke")

        svc.test_orchestrator.get_supported_languages = AsyncMock(return_value=bad_iter())
        with pytest.raises(HTTPException) as exc_info:
            await get_supported_languages(
                container_name=None,
                user=self._make_mock_user(),
                execution_service=svc,
            )
        assert exc_info.value.status_code == 500

    async def test_get_language_test_types_general_exception(self):
        """Non-HTTP exception in get_language_test_types raises 500 (lines 373-375)."""
        from routers.testing_modules.test_discovery import get_language_test_types

        svc = self._make_mock_svc()
        svc.test_orchestrator.get_supported_languages = AsyncMock(return_value=["python"])
        svc.test_orchestrator.get_supported_test_types = AsyncMock(return_value=["unit"])

        bad_user = Mock()
        bad_user.is_admin = Mock(return_value=False)
        bad_user.effective_tenant_id = property(lambda s: (_ for _ in ()).throw(TypeError("boom")))

        with pytest.raises(HTTPException) as exc_info:
            await get_language_test_types(
                language="python",
                container_name="c1",
                user=bad_user,
                execution_service=svc,
            )
        assert exc_info.value.status_code == 500

    async def test_list_test_containers_with_real_containers(self):
        """list_test_containers with working container manager filters correctly."""
        from routers.testing_modules.test_discovery import list_test_containers

        mock_container_test = Mock()
        mock_container_test.provision_type = "test"
        mock_container_test.model_dump = Mock(return_value={"name": "c1", "type": "test"})

        mock_container_prov = Mock()
        mock_container_prov.provision_type = "provisioned"
        mock_container_prov.model_dump = Mock(return_value={"name": "c2", "type": "provisioned"})

        mock_mgr = Mock()
        mock_mgr.list_containers = Mock(return_value=[mock_container_test, mock_container_prov])

        with patch(
            "shared.container_validation.get_container_manager",
            create=True,
            return_value=mock_mgr,
        ):
            # Patch the import inside list_test_containers
            import shared.container_validation as scv

            original = getattr(scv, "get_container_manager", None)
            scv.get_container_manager = Mock(return_value=mock_mgr)
            try:
                # Test filter — only "test" type containers
                result = await list_test_containers(
                    filter_type="test",
                    user=self._make_mock_user(),
                )
                assert result["count"] == 1
                assert result["filter"] == "test"
                assert result["containers"] == [{"name": "c1", "type": "test"}]

                # Provisioned filter — only "provisioned" type containers
                result = await list_test_containers(
                    filter_type="provisioned",
                    user=self._make_mock_user(),
                )
                assert result["count"] == 1
                assert result["filter"] == "provisioned"
                assert result["containers"] == [{"name": "c2", "type": "provisioned"}]

                # All filter — both containers
                result = await list_test_containers(
                    filter_type="all",
                    user=self._make_mock_user(),
                )
                assert result["count"] == 2
                assert result["filter"] == "all"
                assert len(result["containers"]) == 2
            finally:
                if original is None:
                    delattr(scv, "get_container_manager")
                else:
                    scv.get_container_manager = original

    async def test_list_test_containers_general_exception(self):
        """Non-ImportError exception in list_test_containers raises 500 (lines 476-480)."""
        import shared.container_validation as scv
        from routers.testing_modules.test_discovery import list_test_containers

        original = getattr(scv, "get_container_manager", None)
        mock_mgr = Mock()
        mock_mgr.list_containers = Mock(side_effect=ValueError("db error"))
        scv.get_container_manager = Mock(return_value=mock_mgr)
        try:
            with pytest.raises(HTTPException) as exc_info:
                await list_test_containers(
                    filter_type="all",
                    user=self._make_mock_user(),
                )
            assert exc_info.value.status_code == 500
        finally:
            if original is None:
                delattr(scv, "get_container_manager")
            else:
                scv.get_container_manager = original


class TestSupportedTools:
    """Tests for /api/tests/supported-tools endpoint."""

    async def test_get_all_tools(self, aclient):
        """GET /api/tests/supported-tools returns all tools grouped by language."""
        resp = await aclient.get("/api/tests/supported-tools")
        assert resp.status_code == 200
        data = resp.json()
        # Our stub has python, javascript, typescript tools
        assert "python" in data["supported_languages"]
        assert "javascript" in data["supported_languages"]
        assert "typescript" in data["supported_languages"]
        assert data["tools_by_language"]["python"] == [
            "ruff",
            "ruff-format",
            "ty",
            "bandit",
        ]
        assert data["tools_by_language"]["javascript"] == ["eslint", "prettier", "jshint"]
        assert data["tools_by_language"]["typescript"] == ["eslint", "prettier", "tsc"]
        assert data["total_tools"] == 4 + 3 + 3  # 10 total

    async def test_get_tools_by_language(self, aclient):
        """GET /api/tests/supported-tools?language=python returns python tools."""
        resp = await aclient.get("/api/tests/supported-tools?language=python")
        assert resp.status_code == 200
        data = resp.json()
        assert data["language"] == "python"
        assert data["tools"] == ["ruff", "ruff-format", "ty", "bandit"]
        assert data["count"] == 4

    async def test_get_tools_unknown_language(self, aclient):
        """GET /api/tests/supported-tools?language=brainfuck returns 404."""
        resp = await aclient.get("/api/tests/supported-tools?language=brainfuck")
        assert resp.status_code == 404

    async def test_get_tools_with_container(self, aclient):
        """GET /api/tests/supported-tools with container_name includes it in response."""
        with patch(
            "routers.testing_modules.test_discovery.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
        ):
            resp = await aclient.get(
                "/api/tests/supported-tools?container_name=test-container&language=python"
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["container_name"] == "test-container"
            assert data["language"] == "python"
            assert data["count"] == 4


class TestLanguages:
    """Tests for /api/tests/languages endpoint."""

    async def test_get_languages(self, aclient):
        """GET /api/tests/languages returns language list with details."""
        resp = await aclient.get("/api/tests/languages")
        assert resp.status_code == 200
        data = resp.json()
        assert data["languages"] == ["python", "javascript", "typescript"]
        assert data["total"] == 3
        # Each language has details with correct structure
        for lang in ["python", "javascript", "typescript"]:
            assert data["details"][lang]["supported"] is True
            assert isinstance(data["details"][lang]["test_types"], list)
            assert len(data["details"][lang]["test_types"]) > 0
            assert isinstance(data["details"][lang]["quality_tools"], list)

    async def test_get_languages_with_container(self, aclient):
        """GET /api/tests/languages?container_name=test includes container_name."""
        with patch(
            "routers.testing_modules.test_discovery.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
        ):
            resp = await aclient.get("/api/tests/languages?container_name=test-container")
            assert resp.status_code == 200
            data = resp.json()
            assert data["container_name"] == "test-container"
            assert data["total"] == 3

    async def test_language_details_include_tools(self, aclient):
        """Language details include quality tools for each language with correct values."""
        resp = await aclient.get("/api/tests/languages")
        assert resp.status_code == 200
        data = resp.json()
        # Python tools from stub
        assert data["details"]["python"]["quality_tools"] == [
            "ruff",
            "ruff-format",
            "ty",
            "bandit",
        ]
        # JavaScript tools from stub
        assert data["details"]["javascript"]["quality_tools"] == ["eslint", "prettier", "jshint"]
        # Typescript tools from stub
        assert data["details"]["typescript"]["quality_tools"] == ["eslint", "prettier", "tsc"]
        # Test types from stub
        assert data["details"]["python"]["test_types"] == [
            "unit",
            "integration",
            "coverage",
            "security",
        ]
        assert data["details"]["javascript"]["test_types"] == ["unit", "integration", "coverage"]
        assert data["details"]["typescript"]["test_types"] == ["unit", "integration", "coverage"]


class TestLanguageTestTypes:
    """Tests for /api/tests/languages/{lang}/test-types endpoint."""

    async def test_get_python_test_types(self, aclient):
        """GET /api/tests/languages/python/test-types returns python test types."""
        resp = await aclient.get("/api/tests/languages/python/test-types")
        assert resp.status_code == 200
        data = resp.json()
        assert data["language"] == "python"
        assert data["test_types"] == ["unit", "integration", "coverage", "security"]
        assert data["count"] == 4

    async def test_get_unsupported_language_404(self, aclient):
        """GET /api/tests/languages/fortran/test-types returns 404."""
        resp = await aclient.get("/api/tests/languages/fortran/test-types")
        assert resp.status_code == 404

    async def test_get_test_types_with_container(self, aclient):
        """GET /api/tests/languages/python/test-types?container_name=test includes it."""
        with patch(
            "routers.testing_modules.test_discovery.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
        ):
            resp = await aclient.get(
                "/api/tests/languages/python/test-types?container_name=test-container"
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["container_name"] == "test-container"
            assert data["language"] == "python"
            assert data["count"] == 4

    async def test_get_javascript_test_types(self, aclient):
        """GET /api/tests/languages/javascript/test-types returns JS test types."""
        resp = await aclient.get("/api/tests/languages/javascript/test-types")
        assert resp.status_code == 200
        data = resp.json()
        assert data["language"] == "javascript"
        assert data["test_types"] == ["unit", "integration", "coverage"]
        assert data["count"] == 3

    async def test_case_normalization(self, aclient):
        """Language name is normalized to lowercase."""
        resp = await aclient.get("/api/tests/languages/Python/test-types")
        assert resp.status_code == 200
        data = resp.json()
        assert data["language"] == "python"
        assert data["count"] == 4


class TestListContainers:
    """Tests for /api/tests/containers endpoint."""

    async def test_list_containers_default(self, aclient):
        """GET /api/tests/containers returns empty container list with filter=all."""
        resp = await aclient.get("/api/tests/containers")
        assert resp.status_code == 200
        data = resp.json()
        assert data["containers"] == []
        assert data["count"] == 0
        assert data["filter"] == "all"

    async def test_list_containers_test_filter(self, aclient):
        """GET /api/tests/containers?filter_type=test uses test filter."""
        resp = await aclient.get("/api/tests/containers?filter_type=test")
        assert resp.status_code == 200
        data = resp.json()
        assert data["filter"] == "test"
        assert data["containers"] == []
        assert data["count"] == 0

    async def test_list_containers_provisioned_filter(self, aclient):
        """GET /api/tests/containers?filter_type=provisioned uses provisioned filter."""
        resp = await aclient.get("/api/tests/containers?filter_type=provisioned")
        assert resp.status_code == 200
        data = resp.json()
        assert data["filter"] == "provisioned"
        assert data["containers"] == []
        assert data["count"] == 0


# ===========================================================================
# test_execution.py — /api/tests/validate-security
# ===========================================================================


class TestValidateSecurity:
    """Tests for /api/tests/validate-security endpoint."""

    async def test_validate_code_returns_result(self, aclient):
        """POST /api/tests/validate-security returns is_safe=False for stub that raises ValueError."""
        # The stub raises ValueError which the endpoint catches and returns is_safe=False
        resp = await aclient.post(
            "/api/tests/validate-security?code=x%20%3D%201%20%2B%202&language=python"
        )
        assert resp.status_code == 200
        data = resp.json()
        # Stub raises ValueError("Not available in test"), so the ValueError path is hit
        assert data["is_safe"] is False
        assert data["violations"] == ["Not available in test"]
        assert data["warnings"] == []
        assert data["risk_level"] == 100.0
        assert data["language"] == "python"
        assert data["code_length"] == len("x = 1 + 2")
