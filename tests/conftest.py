"""Shared test fixtures for coding-testing.

Uses real dependencies — no sys.modules stubs.
cachetools and coding_client are installed in the venv. uuid7 is stdlib in Python 3.14+.

The suite requires a real Postgres 18. There is deliberately NO SQLite fallback:
the store speaks Postgres-only SQL (jsonb_set/to_jsonb/::jsonb casts), so a
SQLite stand-in does not run the same code — it runs *different* code and
reports a different result than CI, which is how a green CI hides a real
failure. Missing/wrong DATABASE_URL fails the session loudly (see
``pytest_configure``) instead of silently swapping the engine.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest

# Split across two source lines so detect-secrets' BasicAuthDetector never sees
# a whole userinfo-and-host authority on one line. These are throwaway
# local-container credentials, not a secret.
_LOCAL_DSN = (
    "postgresql+psycopg://postgres:postgres"  # pragma: allowlist secret
    "@127.0.0.1:5439/coding_testing_test"
)

_DB_REQUIRED_MESSAGE = """\
DATABASE_URL must point at a real Postgres 18 — got: {actual}

The coding-testing suite has no SQLite fallback (house standard: postgres18 +
psycopg 3.3+ everywhere, no sqlite). CI provisions a per-job ephemeral
postgres:18 and exports DATABASE_URL. For a local run, start a throwaway one:

  docker run -d --name coding-testing-pg \\
    -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres \\
    -e POSTGRES_DB=coding_testing_test \\
    -p 127.0.0.1:5439:5432 postgres:18
  export DATABASE_URL={dsn}
  uv run alembic upgrade head
  uv run pytest
"""


def pytest_configure(config: pytest.Config) -> None:
    """Refuse to run the suite without a real Postgres DSN.

    A loud abort here is the point: the previous behaviour (fall back to
    ``sqlite+aiosqlite:///test.db`` when DATABASE_URL was unset) meant a local
    run and a CI run exercised different database engines, and two orchestration
    tests failed locally on Postgres-only SQL while CI stayed green.
    """
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url.startswith("postgresql"):
        raise pytest.UsageError(
            _DB_REQUIRED_MESSAGE.format(actual=db_url or "<unset>", dsn=_LOCAL_DSN)
        )


@pytest.fixture(autouse=True)
def _test_env(monkeypatch):
    """Set test environment variables.

    Service URLs MUST be injected here (or via gitops Deployment env in prod) —
    `config.settings.Settings` no longer has hardcoded URL defaults. We set
    test values that match the historical hardcoded defaults so existing
    tests' substring assertions ("python-test-executor", etc.) still hold.
    """
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("ENABLE_AUTH", "false")
    monkeypatch.setenv("ENABLE_DOCS", "true")
    # DATABASE_URL is left exactly as exported. ``pytest_configure`` has already
    # asserted it is a real Postgres DSN, so there is nothing to default here —
    # and nothing that could quietly substitute a different engine.
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("SKIP_TEST_EXECUTORS", "true")
    # Test executor URLs (test-only fixtures — not consulted in production code paths)
    monkeypatch.setenv("PYTHON_TEST_SERVICE_URL", "http://python-test-executor:8087")
    monkeypatch.setenv("JAVASCRIPT_TEST_SERVICE_URL", "http://javascript-test-executor:8085")
    monkeypatch.setenv("QUALITY_ANALYSIS_SERVICE_URL", "http://coding-analysis:8688")
    monkeypatch.setenv("JAVA_TEST_SERVICE_URL", "http://java-test-executor:8088")
    monkeypatch.setenv("RUST_TEST_SERVICE_URL", "http://rust-test-executor:8089")
    monkeypatch.setenv("CSHARP_TEST_SERVICE_URL", "http://csharp-test-executor:8090")
    monkeypatch.setenv("GO_TEST_SERVICE_URL", "http://go-test-executor:8091")
    monkeypatch.setenv("CPP_TEST_SERVICE_URL", "http://cpp-test-executor:8094")
    monkeypatch.setenv("CONTAINERS_SERVICE_URL", "http://coding-containers:8691")
    monkeypatch.setenv("SECURITY_SERVICE_URL", "http://coding-security:8693")
    monkeypatch.setenv("LICENSE_INTROSPECT_URL", "http://licenses:8544/license/introspect")
    monkeypatch.setenv("LLM_API_BASE_URL", "http://inference-router:8150/api")

    # coding-client 1.9.0 dropped the built-in DEFAULT_TENANT_ID literal
    # ("n8n-dev"), so it is None unless configured. Production sets it on the
    # Deployment configmap; mirror that here so the executor header paths — which
    # now fail loud rather than send an empty X-Tenant-Id — exercise their real
    # behaviour. The constant is captured at import in these two modules, so
    # patch the bindings (setenv would not reach an already-captured constant).
    # The fail-loud-when-unset path is covered directly in
    # test_executor_tenant_header.py, which patches these back to None.
    import processing.test_execution.http_test_runner as _http_runner
    import services.code_execution_service as _code_exec

    monkeypatch.setattr(_http_runner, "DEFAULT_TENANT_ID", "test-default-tenant")
    monkeypatch.setattr(_code_exec, "DEFAULT_TENANT_ID", "test-default-tenant")

    # Reset the lru_cache so each test sees the env values above.
    from config.settings import get_settings

    get_settings.cache_clear()


@pytest.fixture
def settings():
    """Fresh Settings instance."""
    from config.settings import Settings

    return Settings()


@pytest.fixture
def mock_cache_manager():
    """Mock cache manager."""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock()
    cache.close = AsyncMock()
    cache.stats = {"hits": 0, "misses": 0}
    return cache


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    session = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest.fixture
def trusted_networks(monkeypatch):
    """Set ``middleware.auth.TRUSTED_NETWORKS`` for the duration of a test.

    ``TRUSTED_CIDRS`` is parsed once at import time (deliberately — a malformed
    value must crash at startup, not per request), so tests configure the
    parsed list directly rather than the env var. Auto-reverted by monkeypatch.
    """

    def _set(raw: str):
        import middleware.auth as auth_mod

        monkeypatch.setattr(auth_mod, "TRUSTED_NETWORKS", auth_mod._parse_trusted_cidrs(raw))

    return _set


@pytest.fixture
def app():
    """Create FastAPI test application with the identity dependencies stubbed.

    Replaces the `monkeypatch.setenv("SKIP_AUTH", "true")` that used to sit in
    the autouse env fixture above; SKIP_AUTH was removed from coding-client in
    1.16.0.

    ⚠️ That blanket was LOAD-BEARING here. Deleting it and running bare gave
    `5 failed, 3977 passed, 28 errors` against a baseline of `4010 passed` on
    unmodified main, same DB — so every one of those was caused by the removal,
    not pre-existing. (coding-analysis's identical-looking line turned out to be
    inert; the two repos cannot be reasoned about from each other.)

    `override_tenant_auth` neutralises every identity dependency the SDK exposes
    from one list, so this cannot drift the way a hand-rolled override list does
    — see tests/integration/conftest.py, which had exactly that bug.
    """
    from coding_client.testing import override_tenant_auth

    from app_factory import create_app

    application = create_app()
    override_tenant_auth(application)
    return application


def _user_with_roles(*roles: str):
    from coding_client.auth.models import User
    from coding_client.testing import DEFAULT_TEST_TENANT, DEFAULT_TEST_USER

    return User(
        sub=DEFAULT_TEST_USER,
        preferred_username=DEFAULT_TEST_USER,
        tenant_id=DEFAULT_TEST_TENANT,
        realm_access={"roles": list(roles)},
    )


@pytest.fixture
def as_roles(app):
    """Re-stub the identity on ``app`` as a user holding exactly ``roles``.

    The default ``app`` user has NO realm roles, so ``requires_write_permission``
    (which is not stubbed) answers 403 for it — the real behaviour for a roleless,
    unlicensed caller. Route tests that exercise a write route call
    ``as_roles("developer")``; permission tests use it to vary the role.
    """
    from coding_client.testing import override_tenant_auth

    def _set(*roles: str):
        override_tenant_auth(app, user=_user_with_roles(*roles))

    return _set


@pytest.fixture
def as_writer(as_roles):
    """``app`` with a user that has the write role (``developer``)."""
    as_roles("developer")


@pytest.fixture
async def client(app):
    """Async HTTP test client."""
    import httpx

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as c:
        yield c


@pytest.fixture
async def pg_db():
    """Provide a clean Postgres session factory for DB-backed tests.

    ``pytest_configure`` guarantees DATABASE_URL is a real Postgres DSN, so this
    fixture never skips — a skip here would be the same silent-degradation bug
    in a different costume (a DB outage would render these tests vacuously
    green). Tables are created by ``alembic upgrade head`` (run in CI before
    pytest; locally by hand). Truncates the store tables before yielding so each
    test starts from a clean slate.
    """
    from sqlalchemy import text

    import db as db_module
    from config.settings import get_settings

    get_settings.cache_clear()
    # Reset any engine a previous test left so we bind to the current env DSN.
    await db_module.dispose_engine()
    factory = await db_module.ensure_session_factory()
    async with factory() as session:
        await session.execute(
            text(
                "TRUNCATE test_results, analysis_results, "
                "orchestration_test_suite, orchestration_plan, "
                "orchestration_execution_history, rust_test_runs, workspace_run_specs "
                "RESTART IDENTITY CASCADE"
            )
        )
        await session.commit()
    yield factory
    await db_module.dispose_engine()
