"""SQLAlchemy async engine + session factory for coding-testing.

Postgres 18 + psycopg 3.x async driver only — never asyncpg (per global
CLAUDE.md). Pattern mirrors inference-router/src/db.py.

Node-local per-container SQLite result files were replaced by this shared
Postgres store so the service is safe to run with >1 replica: a request that
lands on replica B can read state written by replica A.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool


class Base(DeclarativeBase):
    """Shared declarative base for coding-testing ORM models."""


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
_init_lock = asyncio.Lock()


def normalize_async_dsn(url: str) -> str:
    """Normalise a Postgres DSN to the async psycopg3 dialect.

    Accepts bare ``postgresql://`` or legacy ``postgresql+asyncpg://`` and
    returns ``postgresql+psycopg://`` (the only driver this fleet uses).

    Any non-Postgres scheme is rejected rather than passed through. The store
    speaks Postgres-only SQL (jsonb_set/to_jsonb/``::jsonb``), so handing a
    sqlite (or any other) DSN to ``create_async_engine`` builds an engine that
    cannot execute it — failing here names the cause instead of surfacing as an
    ``unrecognized token: ":"`` a dozen frames away.
    """
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    if not url.startswith("postgresql+psycopg://"):
        raise ValueError(
            "DATABASE_URL must be a Postgres DSN (postgresql[+psycopg]://...); "
            f"refusing non-Postgres scheme: {url!r}"
        )
    return url


async def init_db(settings: object) -> None:
    """Create the async engine and session factory from ``settings.DATABASE_URL``."""
    global _engine, _session_factory  # noqa: PLW0603

    # Emptiness is checked BEFORE normalisation so an unconfigured deployment
    # still gets the "not configured" message rather than a scheme complaint.
    raw_url = getattr(settings, "DATABASE_URL", "")
    if not raw_url:
        raise RuntimeError("DATABASE_URL is not configured")
    database_url = normalize_async_dsn(raw_url)

    # NullPool: pgbouncer (transaction-mode, in-cluster) does the pooling.
    # App-side pooling would double-pool. Memory: feedback-pgbouncer-nullpool.
    _engine = create_async_engine(
        database_url,
        echo=False,
        poolclass=NullPool,
        pool_pre_ping=True,
    )
    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


async def ensure_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the session factory, lazily initialising the engine if needed.

    The app lifespan calls :func:`init_db` on startup, but the storage service
    is a module-level singleton that may be exercised (in tests, or a cold
    call path) before that. Lazy-init keeps it self-sufficient and avoids
    startup-ordering bugs, while the lock serialises concurrent first calls.
    """
    global _session_factory  # noqa: PLW0603
    if _session_factory is None:
        async with _init_lock:
            if _session_factory is None:
                from config.settings import get_settings

                await init_db(get_settings())
    assert _session_factory is not None
    return _session_factory


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the session factory (raises if the engine is not initialised)."""
    if _session_factory is None:
        raise RuntimeError("Database not initialised - call init_db() first")
    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession]:
    """FastAPI dependency that yields an async session."""
    factory = await ensure_session_factory()
    async with factory() as session:
        yield session


async def check_connectivity() -> bool:
    """Execute a trivial query to verify DB connectivity."""
    if _engine is None:
        return False
    try:
        async with _engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except SQLAlchemyError:
        return False
    return True


async def dispose_engine() -> None:
    """Dispose of the engine pool on shutdown."""
    global _engine, _session_factory  # noqa: PLW0603
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
