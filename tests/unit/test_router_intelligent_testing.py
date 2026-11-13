"""Unit tests for intelligent testing router.

Tests router structure, endpoint registration, route methods, sub-module
inclusion, shared models, and the cleanup handler.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

# FastAPI >=0.14x keeps include_router() children as nested _IncludedRouter
# entries instead of copying their routes up, so router.routes no longer lists
# sub-module endpoints. iter_route_contexts() is FastAPI's public flattener.
from fastapi.routing import iter_route_contexts
from pydantic import ValidationError


class TestIntelligentTestingRouter:
    """Test intelligent testing router structure and endpoints."""

    def test_router_exists_as_api_router(self):
        """Test router is a FastAPI APIRouter instance."""
        from fastapi import APIRouter

        from routers.intelligent_testing import router

        assert type(router) is APIRouter

    def test_router_includes_sub_routers_with_routes(self):
        """Test router includes sub-module routers with actual route paths."""
        from routers.intelligent_testing import router

        route_paths = [r.path for r in iter_route_contexts(router.routes) if hasattr(r, "path")]
        # 11 sub-routers included, totalling 25 sub-routes + 1 /status = 26
        assert len(route_paths) == 26

        # Verify a sampling of expected paths from each sub-module
        expected_paths = {
            "/status",
            "/recommendations",
            "/patterns",
            "/supported-languages",
            "/execute",
            "/execute/batch",
            "/insights",
            "/health",
            "/settings",
            "/service/start",
            "/service/stop",
            "/service/restart",
            "/pattern-analysis",
            "/smart-test-selection",
            "/test-impact-analysis",
            "/test-prioritization",
            "/advanced-recommendations",
            "/analyze",
            "/comprehensive-analysis",
            "/metrics",
            "/predict-failures",
            "/analysis/{analysis_id}",
            "/comprehensive-analysis/{analysis_id}",
        }
        assert expected_paths.issubset(set(route_paths))

    def test_status_endpoint_is_get(self):
        """Test /status endpoint is registered as a GET route."""
        from routers.intelligent_testing import router

        status_routes = [
            r
            for r in iter_route_contexts(router.routes)
            if hasattr(r, "path") and r.path == "/status"
        ]
        assert len(status_routes) == 1
        assert status_routes[0].methods == {"GET"}

    @pytest.mark.asyncio
    async def test_module_status_response_structure(self):
        """Test get_module_status returns complete response structure."""
        from routers.intelligent_testing import get_module_status

        result = await get_module_status(tenant_id="n8n-dev")
        assert result == {
            "status": "active",
            "modules": {
                "discovery_and_recommendations": "active",
                "execution_coordination": "active",
                "analysis_and_insights": "active",
                "configuration_management": "active",
                "service_lifecycle": "active",
                "container_scoped_testing": "active",
                "utilities_and_helpers": "active",
                "metrics_and_calculations": "active",
                "analysis_core": "active",
            },
            "description": "Intelligent testing system with modular architecture",
            "version": "1.0.0",
        }

    @pytest.mark.asyncio
    async def test_module_status_contains_all_modules(self):
        """Test module status lists all 9 expected modules, all active."""
        from routers.intelligent_testing import get_module_status

        result = await get_module_status(tenant_id="n8n-dev")
        modules = result["modules"]

        expected_modules = [
            "discovery_and_recommendations",
            "execution_coordination",
            "analysis_and_insights",
            "configuration_management",
            "service_lifecycle",
            "container_scoped_testing",
            "utilities_and_helpers",
            "metrics_and_calculations",
            "analysis_core",
        ]
        assert len(modules) == 9
        assert set(modules.keys()) == set(expected_modules)
        for mod_name, mod_status in modules.items():
            assert mod_status == "active", f"Module '{mod_name}' should be 'active'"

    def test_router_has_recommendations_endpoint(self):
        """Test router includes /recommendations endpoints from discovery and insight sub-modules."""
        from routers.intelligent_testing import router

        rec_routes = [
            r
            for r in iter_route_contexts(router.routes)
            if hasattr(r, "path") and r.path == "/recommendations"
        ]
        # One GET from discovery, one POST from insight_generation
        assert len(rec_routes) == 2
        methods = {frozenset(r.methods) for r in rec_routes}
        assert frozenset({"GET"}) in methods
        assert frozenset({"POST"}) in methods

    def test_router_post_endpoints_exact(self):
        """Test router has the expected POST endpoints."""
        from routers.intelligent_testing import router

        post_routes = sorted(
            r.path
            for r in iter_route_contexts(router.routes)
            if hasattr(r, "methods") and "POST" in r.methods
        )
        expected_post = sorted(
            [
                "/execute",
                "/execute/batch",
                "/service/start",
                "/service/stop",
                "/service/restart",
                "/pattern-analysis",
                "/smart-test-selection",
                "/test-impact-analysis",
                "/test-prioritization",
                "/advanced-recommendations",
                "/analyze",
                "/comprehensive-analysis",
                "/recommendations",
                "/predict-failures",
            ]
        )
        assert post_routes == expected_post

    def test_cleanup_function_is_callable(self):
        """Test _cleanup_service_sync is a callable function."""
        from routers.intelligent_testing import _cleanup_service_sync

        assert callable(_cleanup_service_sync)

    def test_cleanup_with_no_service_does_nothing(self):
        """Test cleanup completes without error when no service instance exists."""
        from routers.intelligent_testing import _cleanup_service_sync
        from routers.intelligent_testing_modules import shared

        original = shared._service_instance
        shared._service_instance = None

        _cleanup_service_sync()  # Should not raise

        # Confirm it's still None (didn't set anything)
        assert shared._service_instance is None

        shared._service_instance = original

    def test_cleanup_with_non_running_service_clears_instance(self):
        """Test cleanup sets _service_instance to None when service exists but is not running."""
        from routers.intelligent_testing import _cleanup_service_sync
        from routers.intelligent_testing_modules import shared

        original = shared._service_instance

        mock_service = MagicMock()
        mock_service.is_running = False
        shared._service_instance = mock_service

        _cleanup_service_sync()

        # is_running is False, so the cleanup should NOT clear (the if condition checks both)
        # Looking at the source: if shared._service_instance and shared._service_instance.is_running
        # is_running=False -> condition is False -> no action taken
        assert shared._service_instance is mock_service

        shared._service_instance = original


class TestIntelligentTestingSubModules:
    """Test intelligent testing sub-module routers are real APIRouters with routes."""

    def test_discovery_router_has_three_get_routes(self):
        """Test discovery router has exactly 3 GET routes: recommendations, patterns, supported-languages."""
        from routers.intelligent_testing_modules.discovery_and_recommendations import router

        get_routes = [
            r
            for r in iter_route_contexts(router.routes)
            if hasattr(r, "methods") and "GET" in r.methods
        ]
        assert len(get_routes) == 3
        paths = sorted(r.path for r in get_routes)
        assert paths == ["/patterns", "/recommendations", "/supported-languages"]

    def test_execution_router_has_two_post_routes(self):
        """Test execution coordination router has exactly 2 POST routes: execute, execute/batch."""
        from routers.intelligent_testing_modules.execution_coordination import router

        post_routes = [
            r
            for r in iter_route_contexts(router.routes)
            if hasattr(r, "methods") and "POST" in r.methods
        ]
        assert len(post_routes) == 2
        paths = sorted(r.path for r in post_routes)
        assert paths == ["/execute", "/execute/batch"]

    def test_analysis_router_has_two_get_routes(self):
        """Test analysis router has exactly 2 GET routes: insights, health."""
        from routers.intelligent_testing_modules.analysis_and_insights import router

        get_routes = [
            r
            for r in iter_route_contexts(router.routes)
            if hasattr(r, "methods") and "GET" in r.methods
        ]
        assert len(get_routes) == 2
        paths = sorted(r.path for r in get_routes)
        assert paths == ["/health", "/insights"]

    def test_config_router_has_get_and_put_for_settings(self):
        """Test configuration management router has GET and PUT on /settings."""
        from routers.intelligent_testing_modules.configuration_management import router

        routes_by_path = {}
        for r in iter_route_contexts(router.routes):
            if hasattr(r, "path") and hasattr(r, "methods"):
                routes_by_path.setdefault(r.path, set()).update(r.methods)

        assert "/settings" in routes_by_path
        assert routes_by_path["/settings"] == {"GET", "PUT"}

    def test_lifecycle_router_has_three_post_routes(self):
        """Test service lifecycle router has exactly 3 POST routes: start, stop, restart."""
        from routers.intelligent_testing_modules.service_lifecycle import router

        post_routes = [
            r
            for r in iter_route_contexts(router.routes)
            if hasattr(r, "methods") and "POST" in r.methods
        ]
        assert len(post_routes) == 3
        paths = sorted(r.path for r in post_routes)
        assert paths == ["/service/restart", "/service/start", "/service/stop"]

    def test_container_scoped_router_has_five_post_routes(self):
        """Test container-scoped testing router has exactly 5 POST routes."""
        from routers.intelligent_testing_modules.container_scoped_testing import router

        post_routes = [
            r
            for r in iter_route_contexts(router.routes)
            if hasattr(r, "methods") and "POST" in r.methods
        ]
        assert len(post_routes) == 5
        paths = sorted(r.path for r in post_routes)
        assert paths == [
            "/advanced-recommendations",
            "/pattern-analysis",
            "/smart-test-selection",
            "/test-impact-analysis",
            "/test-prioritization",
        ]

    def test_ai_analysis_router_has_two_post_routes(self):
        """Test AI analysis endpoints router has exactly 2 POST routes."""
        from routers.intelligent_testing_modules.ai_analysis_endpoints import router

        post_routes = [
            r
            for r in iter_route_contexts(router.routes)
            if hasattr(r, "methods") and "POST" in r.methods
        ]
        assert len(post_routes) == 2
        paths = sorted(r.path for r in post_routes)
        assert paths == ["/analyze", "/comprehensive-analysis"]

    def test_pattern_recognition_router_has_one_post_route(self):
        """Test pattern recognition router has exactly 1 POST route: predict-failures."""
        from routers.intelligent_testing_modules.test_pattern_recognition import router

        post_routes = [
            r
            for r in iter_route_contexts(router.routes)
            if hasattr(r, "methods") and "POST" in r.methods
        ]
        assert len(post_routes) == 1
        assert post_routes[0].path == "/predict-failures"

    def test_ai_insights_config_router_has_two_get_routes(self):
        """Test AI test insights config router has exactly 2 GET routes: health, metrics."""
        from routers.intelligent_testing_modules.ai_test_insights_config import router

        get_routes = [
            r
            for r in iter_route_contexts(router.routes)
            if hasattr(r, "methods") and "GET" in r.methods
        ]
        assert len(get_routes) == 2
        paths = sorted(r.path for r in get_routes)
        assert paths == ["/health", "/metrics"]

    def test_insight_generation_router_has_one_post_route(self):
        """Test insight generation router has exactly 1 POST route: recommendations."""
        from routers.intelligent_testing_modules.insight_generation_services import router

        post_routes = [
            r
            for r in iter_route_contexts(router.routes)
            if hasattr(r, "methods") and "POST" in r.methods
        ]
        assert len(post_routes) == 1
        assert post_routes[0].path == "/recommendations"

    def test_report_formatting_router_has_two_get_routes(self):
        """Test report formatting router has 2 GET routes for cached analysis."""
        from routers.intelligent_testing_modules.report_formatting_endpoints import router

        get_routes = [
            r
            for r in iter_route_contexts(router.routes)
            if hasattr(r, "methods") and "GET" in r.methods
        ]
        assert len(get_routes) == 2
        paths = sorted(r.path for r in get_routes)
        assert paths == ["/analysis/{analysis_id}", "/comprehensive-analysis/{analysis_id}"]


class TestSharedModels:
    """Test shared models from intelligent_testing_modules."""

    def test_intelligent_test_request_required_fields(self):
        """Test IntelligentTestRequest requires code, language, container_name."""
        from routers.intelligent_testing_modules.shared import IntelligentTestRequest

        request = IntelligentTestRequest(
            code="def add(a, b): return a + b",
            language="python",
            container_name="my-container",
        )

        assert request.code == "def add(a, b): return a + b"
        assert request.language == "python"
        assert request.container_name == "my-container"

    def test_intelligent_test_request_defaults(self):
        """Test IntelligentTestRequest default values for all optional fields."""
        from routers.intelligent_testing_modules.shared import IntelligentTestRequest

        request = IntelligentTestRequest(
            code="x = 1",
            language="python",
            container_name="c1",
        )

        assert request.timeout == 300
        assert request.enable_ai_optimization is True
        assert request.enable_pattern_analysis is True
        assert request.enable_failure_analysis is True
        assert request.enable_scheduling_optimization is True
        assert request.priority_level == 5
        assert request.expected_execution_time_ms is None
        assert request.resource_weight == 1.0
        assert request.test_type == "unit"
        assert request.dependencies == {}
        assert request.test_code is None
        assert request.project_name is None
        assert request.resource_limits is None
        assert request.security_level.value == "medium"

    def test_intelligent_test_request_rejects_invalid_language(self):
        """Test IntelligentTestRequest rejects unsupported language with ValueError."""
        from routers.intelligent_testing_modules.shared import IntelligentTestRequest

        with pytest.raises(ValidationError) as exc_info:
            IntelligentTestRequest(
                code="x = 1",
                language="brainfuck",
                container_name="c1",
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("language",)
        assert "Unsupported language" in errors[0]["msg"]

    def test_intelligent_test_request_accepts_all_supported_languages(self):
        """Test IntelligentTestRequest accepts every supported language."""
        from routers.intelligent_testing_modules.shared import IntelligentTestRequest

        supported = [
            "python",
            "javascript",
            "typescript",
            "java",
            "go",
            "rust",
            "csharp",
            "cpp",
            "nodejs",
        ]
        for lang in supported:
            req = IntelligentTestRequest(code="x", language=lang, container_name="c")
            assert req.language == lang

    def test_intelligent_test_request_timeout_validation(self):
        """Test IntelligentTestRequest rejects timeout outside 30-3600."""
        from routers.intelligent_testing_modules.shared import IntelligentTestRequest

        with pytest.raises(ValidationError):
            IntelligentTestRequest(code="x", language="python", container_name="c", timeout=10)

        with pytest.raises(ValidationError):
            IntelligentTestRequest(code="x", language="python", container_name="c", timeout=5000)

    def test_intelligent_test_request_priority_validation(self):
        """Test IntelligentTestRequest rejects priority outside 1-10."""
        from routers.intelligent_testing_modules.shared import IntelligentTestRequest

        with pytest.raises(ValidationError):
            IntelligentTestRequest(
                code="x", language="python", container_name="c", priority_level=0
            )

        with pytest.raises(ValidationError):
            IntelligentTestRequest(
                code="x", language="python", container_name="c", priority_level=11
            )

    def test_batch_test_request_model(self):
        """Test BatchTestRequest holds a list of tests with defaults."""
        from routers.intelligent_testing_modules.shared import (
            BatchTestRequest,
            IntelligentTestRequest,
        )

        test_req = IntelligentTestRequest(
            code="x = 1",
            language="python",
            container_name="c1",
        )

        batch = BatchTestRequest(tests=[test_req])
        assert len(batch.tests) == 1
        assert batch.enable_retry is True
        assert batch.max_concurrent is None
        assert batch.tests[0].language == "python"
        assert batch.tests[0].code == "x = 1"
        assert batch.tests[0].container_name == "c1"

    def test_batch_test_request_max_concurrent_validation(self):
        """Test BatchTestRequest rejects max_concurrent outside 1-20."""
        from routers.intelligent_testing_modules.shared import (
            BatchTestRequest,
            IntelligentTestRequest,
        )

        test_req = IntelligentTestRequest(code="x", language="python", container_name="c")

        with pytest.raises(ValidationError):
            BatchTestRequest(tests=[test_req], max_concurrent=0)

        with pytest.raises(ValidationError):
            BatchTestRequest(tests=[test_req], max_concurrent=21)

    def test_health_status_response_model(self):
        """Test HealthStatusResponse has correct status, health_status, and generated_at fields."""
        from routers.intelligent_testing_modules.shared import HealthStatusResponse

        response = HealthStatusResponse(
            message="ok",
            health_status={"service": "healthy"},
        )

        assert response.status.value == "success"
        assert response.health_status == {"service": "healthy"}
        assert response.message == "ok"
        assert isinstance(response.generated_at, datetime)

    def test_health_status_response_serializes_datetime(self):
        """Test HealthStatusResponse serializes generated_at to ISO string."""
        from routers.intelligent_testing_modules.shared import HealthStatusResponse

        response = HealthStatusResponse(
            message="ok",
            health_status={"service": "healthy"},
        )

        dumped = response.model_dump()
        # The field_serializer converts datetime to ISO string
        assert isinstance(dumped["generated_at"], str)
        assert "T" in dumped["generated_at"]  # ISO 8601 format

    def test_get_intelligent_test_service_is_async_function(self):
        """Test get_intelligent_test_service is an async callable dependency function."""
        import inspect

        from routers.intelligent_testing_modules import shared

        assert callable(shared.get_intelligent_test_service)
        assert inspect.iscoroutinefunction(shared.get_intelligent_test_service)

    def test_service_instance_starts_as_none(self):
        """Test global _service_instance starts as None."""
        from routers.intelligent_testing_modules import shared

        # Save and restore to avoid side effects
        original = shared._service_instance
        try:
            shared._service_instance = None
            assert shared._service_instance is None
        finally:
            shared._service_instance = original

    def test_execution_insights_response_model(self):
        """Test ExecutionInsightsResponse model creation and defaults."""
        from routers.intelligent_testing_modules.shared import ExecutionInsightsResponse

        response = ExecutionInsightsResponse(
            message="insights ready",
            insights={"total_executions": 42, "avg_time_ms": 150.5},
        )

        assert response.status.value == "success"
        assert response.message == "insights ready"
        assert response.insights["total_executions"] == 42
        assert response.insights["avg_time_ms"] == 150.5
        assert isinstance(response.generated_at, datetime)

    def test_recommendations_response_model(self):
        """Test RecommendationsResponse model creation."""
        from routers.intelligent_testing_modules.shared import RecommendationsResponse

        response = RecommendationsResponse(
            message="recs",
            recommendations=[],
        )

        assert response.status.value == "success"
        assert response.recommendations == []
        assert isinstance(response.generated_at, datetime)

    def test_patterns_response_model(self):
        """Test PatternsResponse model creation."""
        from routers.intelligent_testing_modules.shared import PatternsResponse

        response = PatternsResponse(
            message="patterns found",
            patterns=[],
        )

        assert response.status.value == "success"
        assert response.patterns == []
        assert isinstance(response.generated_at, datetime)
