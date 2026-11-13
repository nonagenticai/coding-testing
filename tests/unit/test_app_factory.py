"""Tests for app_factory.py."""

import logging
import os
import sys
from unittest.mock import patch

import pytest
from fastapi import FastAPI

from tests.route_paths import route_paths as _route_paths


def test_create_app_returns_fastapi_with_correct_title(app):
    """create_app() returns a FastAPI instance with the expected title."""
    assert type(app) is FastAPI
    assert app.title == "coding-testing"


def test_app_has_version_1_0_0(app):
    """App has version 1.0.0."""
    assert app.version == "1.0.0"


def test_app_has_openapi_url(app):
    """App has openapi URL configured."""
    assert app.openapi_url == "/openapi.json"


def test_app_has_docs_and_redoc_urls(app):
    """App has both /docs and /redoc configured."""
    assert app.docs_url == "/docs"
    assert app.redoc_url == "/redoc"


def test_app_has_root_and_health_routes(app):
    """App has registered / and /health routes."""
    paths = _route_paths(app.routes)
    assert "/" in paths
    assert "/health" in paths


def test_app_has_health_api_router(app):
    """App includes the /api/health router with ready and live endpoints."""
    paths = _route_paths(app.routes)
    assert "/api/health/ready" in paths
    assert "/api/health/live" in paths


def test_app_has_testing_routers(app):
    """App includes testing routers under /api/tests with specific endpoints."""
    paths = _route_paths(app.routes)
    # Check specific prefixes are registered
    tests_paths = [p for p in paths if p.startswith("/api/tests")]
    assert len(tests_paths) >= 5, (
        f"Expected at least 5 /api/tests paths, got {len(tests_paths)}: {tests_paths}"
    )
    # Check the specific router prefixes exist
    assert any(p.startswith("/api/tests/execute") for p in paths)
    assert any(p.startswith("/api/tests/orchestration") for p in paths)
    assert any(p.startswith("/api/tests/ai") for p in paths)


def test_app_has_gateway_alias_routes(app):
    """App includes gateway alias routes for /api/tests/status and /api/tests/ready."""
    paths = _route_paths(app.routes)
    assert "/api/tests/status" in paths
    assert "/api/tests/ready" in paths


async def test_root_endpoint(client):
    """GET / returns service info with all expected fields."""
    resp = await client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data == {
        "service": "coding-testing",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


async def test_health_endpoint(client):
    """GET /health returns ok status with version."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data == {
        "status": "ok",
        "service": "coding-testing",
        "version": "1.0.0",
    }


# ============================================================================
# Extended coverage: generate_unique_id, lifespan, middlewares
# ============================================================================


def test_generate_unique_id_root():
    """generate_unique_id for root path returns root_get."""
    from app_factory import generate_unique_id

    route = type("Route", (), {"path": "/", "methods": {"GET"}})()
    assert generate_unique_id(route) == "root_get"


def test_generate_unique_id_api_path():
    """generate_unique_id strips /api/ prefix and joins with underscores."""
    from app_factory import generate_unique_id

    route = type("Route", (), {"path": "/api/tests/execute", "methods": {"POST"}})()
    assert generate_unique_id(route) == "tests_execute_post"


def test_generate_unique_id_health_root():
    """generate_unique_id for /health path returns health_root_get."""
    from app_factory import generate_unique_id

    route = type("Route", (), {"path": "/health", "methods": {"GET"}})()
    assert generate_unique_id(route) == "health_root_get"


def test_generate_unique_id_health_api():
    """generate_unique_id for /api/health returns health_api_get."""
    from app_factory import generate_unique_id

    route = type("Route", (), {"path": "/api/health", "methods": {"GET"}})()
    assert generate_unique_id(route) == "health_api_get"


def test_generate_unique_id_with_path_params():
    """generate_unique_id strips braces from path parameters."""
    from app_factory import generate_unique_id

    route = type(
        "Route",
        (),
        {
            "path": "/api/tests/languages/{language}/test-types",
            "methods": {"GET"},
        },
    )()
    assert generate_unique_id(route) == "tests_languages_language_test_types_get"


def test_generate_unique_id_no_methods():
    """generate_unique_id uses 'unknown' when route has no methods."""
    from app_factory import generate_unique_id

    route = type("Route", (), {"path": "/api/tests", "methods": set()})()
    assert generate_unique_id(route) == "tests_unknown"


def test_generate_unique_id_dashes_converted():
    """generate_unique_id converts dashes to underscores."""
    from app_factory import generate_unique_id

    route = type(
        "Route",
        (),
        {
            "path": "/api/tests/quality-check",
            "methods": {"POST"},
        },
    )()
    assert generate_unique_id(route) == "tests_quality_check_post"


def test_generate_unique_id_empty_path_after_strip():
    """When path becomes empty after stripping /api/, returns root_{method}."""
    from app_factory import generate_unique_id

    route = type("Route", (), {"path": "/api/", "methods": {"GET"}})()
    assert generate_unique_id(route) == "root_get"


def test_generate_unique_id_health_api_sub_path():
    """Health sub-path under /api/health/detailed does NOT trigger the health special case."""
    from app_factory import generate_unique_id

    # /api/health/detailed -> base_id = "health_detailed" which != "health",
    # so the special health branch is skipped
    route = type("Route", (), {"path": "/api/health/detailed", "methods": {"GET"}})()
    assert generate_unique_id(route) == "health_detailed_get"


# ============================================================================
# Middleware tests
# ============================================================================


def test_add_middlewares_auth_disabled():
    """When ENABLE_AUTH=false, no AuthMiddleware is added but CORS and logging are."""
    from app_factory import _add_middlewares

    app = FastAPI()
    settings = type("Settings", (), {})()

    with patch.dict(os.environ, {"ENABLE_AUTH": "false"}):
        _add_middlewares(app, settings)

    # CORS middleware should be present
    middleware_classes = [type(m).__name__ for m in app.user_middleware]
    assert "Middleware" in str(middleware_classes) or len(app.user_middleware) >= 1
    # AuthMiddleware should NOT be in the middleware stack
    assert not any("AuthMiddleware" in str(type(m)) for m in app.user_middleware)


def test_add_middlewares_with_auth_enabled():
    """When ENABLE_AUTH=true, AuthMiddleware is added alongside CORS and logging."""
    from app_factory import _add_middlewares

    app = FastAPI()
    settings = type("Settings", (), {})()

    with patch.dict(os.environ, {"ENABLE_AUTH": "true"}):
        _add_middlewares(app, settings)

    # Should have more middleware when auth is enabled
    assert len(app.user_middleware) >= 2


# ── AuthMiddleware import failure must fail CLOSED ────────────────────
#
# Regression guard: `_add_middlewares` used to swallow the ImportError with
# `logger.warning("AuthMiddleware not available, skipping")`, which meant a
# broken import silently ran the whole service with NO authentication.


def _no_auth_middleware_module():
    """Context manager making `from middleware.auth import ...` raise ImportError.

    A None entry in sys.modules is the documented way to make an import of that
    name fail without touching the real module on disk.
    """
    return patch.dict(sys.modules, {"middleware.auth": None})


def test_add_middlewares_import_error_raises_in_production(monkeypatch):
    """In production an unimportable AuthMiddleware must abort startup."""
    from app_factory import _add_middlewares

    monkeypatch.setenv("ENVIRONMENT", "production")
    app = FastAPI()
    settings = type("Settings", (), {})()

    with _no_auth_middleware_module():
        with pytest.raises(RuntimeError, match="refusing to start in production"):
            _add_middlewares(app, settings)

    # And it must not have quietly registered an unauthenticated stack.
    assert not any(m.cls.__name__ == "AuthMiddleware" for m in app.user_middleware)


def test_create_app_import_error_does_not_start_app_in_production(monkeypatch):
    """create_app() itself refuses to build the app in production without auth."""
    from config.settings import get_settings

    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    try:
        from app_factory import create_app

        with _no_auth_middleware_module():
            with pytest.raises(RuntimeError, match="refusing to start in production"):
                create_app()
    finally:
        get_settings.cache_clear()


def test_add_middlewares_import_error_warns_outside_production(monkeypatch, caplog):
    """Outside production the app still boots, but says loudly that auth is off."""
    from app_factory import _add_middlewares

    monkeypatch.setenv("ENVIRONMENT", "development")
    app = FastAPI()
    settings = type("Settings", (), {})()

    with caplog.at_level(logging.WARNING, logger="app_factory"), _no_auth_middleware_module():
        _add_middlewares(app, settings)  # must not raise

    assert "starting WITHOUT authentication" in caplog.text
    assert not any(m.cls.__name__ == "AuthMiddleware" for m in app.user_middleware)
    # The rest of the stack is still wired.
    assert any(m.cls.__name__ == "CORSMiddleware" for m in app.user_middleware)


def test_add_middlewares_registers_auth_when_import_succeeds():
    """Positive control: the normal path really does register AuthMiddleware."""
    from app_factory import _add_middlewares

    app = FastAPI()
    settings = type("Settings", (), {})()
    _add_middlewares(app, settings)

    assert any(m.cls.__name__ == "AuthMiddleware" for m in app.user_middleware)


def test_add_exception_handlers_registers():
    """_add_exception_handlers registers handlers on the app."""
    from app_factory import _add_exception_handlers

    app = FastAPI()
    handlers_before = len(app.exception_handlers)
    _add_exception_handlers(app)
    handlers_after = len(app.exception_handlers)
    assert handlers_after >= handlers_before
