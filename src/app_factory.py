"""Application factory for creating and configuring the FastAPI testing app."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from coding_client.api.logging_setup import configure_logging
from coding_client.api.request_logging import HealthCheckFilter, RequestLoggingMiddleware
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute

logger = logging.getLogger(__name__)


def _run_alembic_upgrade(database_url: str) -> None:
    """Run ``alembic upgrade head`` synchronously against ``database_url``.

    Driven from lifespan so the first replica self-heals a fresh database;
    re-runs are no-ops because alembic_version tracks the applied set. Called
    via the alembic Python API so no extra binary is needed at runtime.
    """
    from pathlib import Path

    from alembic.config import Config

    from alembic import command
    from db import normalize_async_dsn

    project_root = Path(__file__).resolve().parent.parent
    cfg = Config(str(project_root / "alembic.ini"))
    cfg.set_main_option("script_location", str(project_root / "alembic"))
    db_url = normalize_async_dsn(database_url)
    cfg.set_main_option("sqlalchemy.url", db_url)
    # env.py prefers $DATABASE_URL; keep it aligned with what we upgrade.
    os.environ["DATABASE_URL"] = db_url
    command.upgrade(cfg, "head")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Manage application lifespan."""
    import asyncio

    startup_time = time.time()

    service_name = os.getenv("SERVICE_NAME", "coding-testing")
    version = os.getenv("VERSION", "1.0.0")
    environment = os.getenv("ENVIRONMENT", "production")

    logger.info("Starting %s v%s (%s)", service_name, version, environment)

    try:
        from config.settings import get_settings
        from db import init_db

        settings = get_settings()
        # Postgres store for test results (and orchestration state). Only wire
        # the DB when a Postgres DSN is configured — a non-postgres/empty URL
        # (some local/test shells) would otherwise hang on connect.
        db_enabled = settings.DATABASE_URL.startswith("postgresql")
        if db_enabled:
            await init_db(settings)
            # Migrate from the lifespan (not a K8s initContainer) so the first
            # replica self-heals a fresh DB and the rest no-op.
            # Bounded retry. The DSN points at pgbouncer, whose `coding_testing`
            # entry inherits `default_pool_size = 2`; during a RollingUpdate the
            # two draining replicas plus the new one contend for it, so the very
            # first connection of a starting pod can transiently fail. Without a
            # retry that single failure raises, uvicorn exits 3, and the pod
            # crashloops with a 5-minute backoff — which is how one transient
            # blip wedged the 2026-08-24 rollout for 8 hours while the Deployment
            # still read READY 2/2. Fail closed after the last attempt: a genuine
            # migration error must still stop the pod, never start it on a schema
            # it did not verify.
            attempts = int(os.getenv("ALEMBIC_UPGRADE_ATTEMPTS", "5"))
            for attempt in range(1, attempts + 1):
                try:
                    await asyncio.to_thread(_run_alembic_upgrade, settings.DATABASE_URL)
                    logger.info("alembic_upgrade_complete attempt=%d", attempt)
                    break
                except Exception:
                    if attempt >= attempts:
                        logger.error(
                            "alembic_upgrade_failed attempts=%d giving_up",
                            attempt,
                            exc_info=True,
                        )
                        raise
                    backoff = min(2 ** (attempt - 1), 8)
                    # WARNING, with the traceback, even when a later attempt wins:
                    # a retry that hides its cause is how a degrading dependency
                    # stays invisible until it stops being transient.
                    logger.warning(
                        "alembic_upgrade_retry attempt=%d/%d backoff=%ss",
                        attempt,
                        attempts,
                        backoff,
                        exc_info=True,
                    )
                    await asyncio.sleep(backoff)
        else:
            logger.warning("DATABASE_URL not a postgres DSN; DB store disabled")

        app.state.background_tasks = set()

        startup_duration = time.time() - startup_time
        logger.info("%s ready! Started in %.2fs", service_name, startup_duration)
        yield

    except Exception as e:
        logger.error("Error during %s startup: %s", service_name, e)
        raise
    finally:
        logger.info("Shutting down %s", service_name)
        try:
            from db import dispose_engine

            await dispose_engine()
        except Exception:
            logger.warning("Error disposing DB engine", exc_info=True)


def generate_unique_id(route: APIRoute) -> str:
    """Generate unique operation IDs from route paths and HTTP methods."""
    path = route.path.replace("/api/", "").strip("/")
    method = list(route.methods)[0].lower() if route.methods else "unknown"

    if not path:
        return f"root_{method}"

    path = path.replace("-", "_")
    parts = path.split("/")
    clean_parts = []
    for part in parts:
        clean_part = part.replace("{", "").replace("}", "")
        if clean_part:
            clean_parts.append(clean_part)

    base_id = "_".join(clean_parts) if clean_parts else "root"

    if base_id == "health":
        if route.path == "/health":
            return f"health_root_{method}"
        elif route.path.startswith("/api/health"):
            return f"health_api_{method}"

    return f"{base_id}_{method}"


def create_app() -> FastAPI:
    """Create and configure the FastAPI testing application."""
    configure_logging()

    creation_start = time.time()
    logger.info("Creating FastAPI testing application...")

    from config.settings import get_settings

    settings = get_settings()

    app = FastAPI(
        title=os.getenv("SERVICE_NAME", "coding-testing"),
        version=os.getenv("VERSION", "1.0.0"),
        description=(
            "Multi-language test execution and AI-powered test analysis service.\n\n"
            "## Auth\n"
            "Istio-forwarded headers (`x-user-id`, `x-tenant-id`, `x-user-roles`), "
            "optional trusted-network bypass for source IPs listed in `TRUSTED_CIDRS` "
            "(default: empty — no source IP is trusted), "
            "API key fallback (`Authorization: Bearer <API_KEY>`) in dev/test.\n\n"
            "## Endpoint Groups\n"
            "- **Testing & Quality** (`/api/tests`): execute, batch, quality-check, quality-scan, "
            "language/tool discovery, containers, health, metrics\n"
            "- **Test Execution** (`/api/tests/execute`): containerized execution, batch, "
            "active executions, cleanup, container specs\n"
            "- **Test Orchestration** (`/api/tests/orchestration`): suites, plans, "
            "execution history, orchestration metrics\n"
            "- **Intelligent Testing** (`/api/tests/ai`): AI-powered analysis, pattern detection, "
            "failure prediction, smart test selection, impact analysis, recommendations\n"
            "- **Test Results Data** (`/api/tests`): raw result retrieval, history, "
            "store externally-analyzed results\n"
            "- **Health** (`/api/health`): readiness, liveness, detailed, system metrics"
        ),
        openapi_url="/openapi.json",
        lifespan=lifespan,
        docs_url="/docs",
        swagger_ui_parameters={"persistAuthorization": True},
        redoc_url="/redoc",
        generate_unique_id_function=generate_unique_id,
    )

    _add_middlewares(app, settings)
    _add_exception_handlers(app)
    _include_routers(app)

    # --- OpenAPI security (KAG standard) ---
    from coding_client.api.openapi_config import configure_openapi

    configure_openapi(
        app,
        protected_prefixes={"/api/tests", "/api/parity"},
        public_endpoints={"/", "/health", "/docs", "/redoc", "/openapi.json"},
    )

    total_creation_time = time.time() - creation_start
    logger.info("FastAPI testing application created in %.3fs", total_creation_time)

    return app


def _add_middlewares(app: FastAPI, settings) -> None:
    """Add middlewares in KAG standard order: CORS -> RequestLogging -> Auth."""
    # Auth middleware — always added. It handles ENABLE_AUTH=false internally
    # by injecting an anonymous user with DEFAULT_TENANT_ID, ensuring
    # request.state.user is always set for downstream dependencies.
    #
    # A failed import must fail CLOSED: silently skipping the middleware would
    # run the whole service with no authentication at all. Mirrors the
    # production-fail-fast contract in config.settings
    # (BaseMicroserviceSettings._enforce_required_in_production): raise
    # RuntimeError when ENVIRONMENT == "production", warn otherwise so a
    # partially-installed dev checkout can still boot.
    try:
        from middleware.auth import AuthMiddleware
    except ImportError as exc:
        environment = os.getenv("ENVIRONMENT", "production")
        if environment == "production":
            raise RuntimeError(
                "coding-testing cannot import middleware.auth.AuthMiddleware; "
                "refusing to start in production without authentication. "
                f"Original import error: {exc}"
            ) from exc
        logger.warning(
            "AuthMiddleware not available (%s); starting WITHOUT authentication "
            "because ENVIRONMENT=%s. This would be a hard failure in production.",
            exc,
            environment,
        )
    else:
        app.add_middleware(AuthMiddleware)  # type: ignore[invalid-argument-type]
        enable_auth = os.getenv("ENABLE_AUTH", "true").lower() == "true"
        logger.info("Auth middleware enabled (enforce=%s)", enable_auth)

    # Request logging (generates request_id, adds X-Request-ID + X-Processing-Time)
    app.add_middleware(RequestLoggingMiddleware)  # type: ignore[invalid-argument-type]

    # CORS
    app.add_middleware(
        CORSMiddleware,  # type: ignore[invalid-argument-type]
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # Suppress /health noise from uvicorn access logs
    logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())


def _add_exception_handlers(app: FastAPI) -> None:
    """Register KAG-standard exception handlers."""
    from coding_client.api.exception_handlers import register_exception_handlers

    register_exception_handlers(app)


def _include_routers(app: FastAPI) -> None:
    """Include all testing service routers."""

    @app.get("/health", operation_id="health_check")
    async def root_health():
        return {
            "status": "ok",
            "service": os.getenv("SERVICE_NAME", "coding-testing"),
            "version": os.getenv("VERSION", "1.0.0"),
        }

    @app.get("/", operation_id="root")
    async def root():
        return {
            "service": os.getenv("SERVICE_NAME", "coding-testing"),
            "version": os.getenv("VERSION", "1.0.0"),
            "docs": "/docs",
            "health": "/health",
        }

    # Health API router
    from routers.health import router as health_router

    app.include_router(health_router, prefix="/api/health", tags=["Health"])

    # Behaviour parity (py2rust A5c): own module + own include line, so the sibling
    # parity slices (A5a schema, A5b openapi) register separately.
    from routers.parity_behaviour import router as parity_behaviour_router

    app.include_router(parity_behaviour_router, prefix="/api/parity", tags=["Behaviour Parity"])

    # Performance comparison (source(s) vs target under load).
    from routers.parity_perf import router as parity_perf_router

    app.include_router(parity_perf_router, prefix="/api/parity", tags=["Performance Parity"])

    # Rust test runs: the runner Job's one-time archive download (not in the OpenAPI).
    from routers.internal_test_runs import router as internal_test_runs_router

    app.include_router(internal_test_runs_router, prefix="/internal")

    # Testing routers
    from routers.intelligent_testing import router as intelligent_testing_router
    from routers.test_execution import router as test_execution_router
    from routers.test_orchestration import router as test_orchestration_router
    from routers.test_results_data import router as test_results_data_router
    from routers.testing import router as testing_router

    app.include_router(testing_router, prefix="/api/tests", tags=["Testing & Quality"])
    app.include_router(test_execution_router, prefix="/api/tests/execute", tags=["Test Execution"])
    app.include_router(
        test_orchestration_router,
        prefix="/api/tests/orchestration",
        tags=["Test Orchestration"],
    )
    app.include_router(
        intelligent_testing_router, prefix="/api/tests/ai", tags=["Intelligent Testing"]
    )
    app.include_router(test_results_data_router, prefix="/api/tests", tags=["Test Results Data"])

    from routers.workspace_test_runs import router as workspace_test_runs_router

    app.include_router(
        workspace_test_runs_router, prefix="/api/tests/workspace", tags=["Workspace Test Runs"]
    )

    # Contract parity (py2rust A5b): additive, own prefix.
    from routers.parity import router as parity_router

    app.include_router(parity_router, prefix="/api/parity", tags=["Contract Parity"])

    # ---- Gateway route aliases ----
    # The gateway proxies /api/tests/* to this service, forwarding the full path.
    # These aliases bridge the mismatch between what the gateway sends and where
    # the actual handlers are mounted.

    from auth.tenant import require_tenant

    @app.get("/api/tests/status", tags=["Gateway Aliases"], operation_id="gateway_alias_status")
    async def gateway_alias_status(
        tenant_id: str = Depends(require_tenant),
    ):
        """Alias for /api/tests/ai/status (gateway sends /api/tests/status)."""
        from routers.intelligent_testing import get_module_status

        return await get_module_status(tenant_id=tenant_id)

    @app.get("/api/tests/ready", tags=["Gateway Aliases"], operation_id="gateway_alias_ready")
    async def gateway_alias_ready():
        """Alias for /api/health/ready (gateway sends /api/tests/ready)."""
        from routers.health import service_readiness_check

        return await service_readiness_check()

    # Schema parity (py2rust A5a): its own module and include line, so sibling
    # parity slices register separately and never share a router file.
    from routers.parity_schema import router as parity_schema_router

    app.include_router(parity_schema_router, prefix="/api/parity", tags=["Parity"])
