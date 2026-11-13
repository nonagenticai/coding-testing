"""Integration test fixtures for coding-testing."""

import pytest
from coding_client.auth.models import User

_test_user = User(
    sub="test-user",
    preferred_username="test-user",
    realm_access={"roles": ["user", "developer", "admin"]},
    tenant_id="n8n-dev",
)


@pytest.fixture(autouse=True)
def _integration_env(monkeypatch):
    """Set integration test environment variables."""
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("ENABLE_AUTH", "false")
    monkeypatch.setenv("ENABLE_DOCS", "true")
    monkeypatch.setenv("DEFAULT_TENANT_ID", "n8n-dev")
    monkeypatch.setenv("SERVICE_NAME", "coding-testing")
    monkeypatch.setenv("VERSION", "1.0.0")


@pytest.fixture
def integration_app():
    """Create FastAPI app for integration tests with auth overrides.

    ⚠️ THIS FIXTURE IS WHY coding_client.testing EXISTS.

    It used to hand-roll exactly three overrides — `get_current_user`,
    `requires_read_permission`, `requires_write_permission` — and NOT
    `require_tenant`. That list was incomplete, and nobody knew, because line 19
    of this file also set `SKIP_AUTH=true` and covered the gap. Remove the hatch
    (coding-client 1.16.0) and the omission becomes visible immediately.

    A hand-maintained list in four repos is four chances to miss the next
    dependency the SDK adds. `override_tenant_auth` neutralises all of them from
    a single tuple and picks up future ones at the next pin bump. The
    service-specific permission dependencies still go through `extra`, because
    they are this repo's, not the SDK's.
    """
    from coding_client.testing import override_tenant_auth

    from app_factory import create_app
    from shared.dependencies import requires_read_permission, requires_write_permission

    app = create_app()
    override_tenant_auth(
        app,
        user=_test_user,
        extra={
            requires_read_permission: lambda: _test_user,
            requires_write_permission: lambda: _test_user,
        },
    )
    return app


@pytest.fixture
async def integration_client(integration_app):
    """Async HTTP client for integration tests."""
    import httpx

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=integration_app),
        base_url="http://testserver",
    ) as client:
        yield client
