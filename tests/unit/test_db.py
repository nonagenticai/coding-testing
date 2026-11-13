"""Unit tests for the async DB layer (src/db.py)."""

import pytest
from sqlalchemy import text


def test_normalize_async_dsn_variants():
    """DSNs normalise to the psycopg3 async dialect."""
    from db import normalize_async_dsn

    assert normalize_async_dsn("postgresql://u:p@h:5432/db") == "postgresql+psycopg://u:p@h:5432/db"
    assert normalize_async_dsn("postgresql+asyncpg://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    # Already-correct dialect is unchanged.
    assert normalize_async_dsn("postgresql+psycopg://x/y") == "postgresql+psycopg://x/y"


@pytest.mark.parametrize(
    "dsn",
    [
        "sqlite+aiosqlite:///t.db",
        "sqlite:///t.db",
        "mysql://u:p@h/db",
    ],
)
def test_normalize_async_dsn_rejects_non_postgres(dsn):
    """A non-Postgres DSN is refused, never silently accepted.

    The store speaks Postgres-only SQL; accepting sqlite here is what let a
    local run and a CI run exercise different engines.
    """
    from db import normalize_async_dsn

    with pytest.raises(ValueError, match="Postgres DSN"):
        normalize_async_dsn(dsn)


async def test_init_db_rejects_non_postgres_url():
    """init_db refuses a sqlite DATABASE_URL instead of building a sqlite engine."""
    import db as db_module

    class _S:
        DATABASE_URL = "sqlite+aiosqlite:///test.db"

    with pytest.raises(ValueError, match="Postgres DSN"):
        await db_module.init_db(_S())


async def test_init_db_missing_url_raises():
    """init_db with an empty DATABASE_URL fails loud."""
    import db as db_module

    class _S:
        DATABASE_URL = ""

    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        await db_module.init_db(_S())


async def test_get_session_factory_raises_when_uninitialised():
    """get_session_factory raises until init_db has run."""
    import db as db_module

    await db_module.dispose_engine()
    with pytest.raises(RuntimeError, match="not initialised"):
        db_module.get_session_factory()


async def test_check_connectivity_false_when_uninitialised():
    """check_connectivity returns False when the engine is absent."""
    import db as db_module

    await db_module.dispose_engine()
    assert await db_module.check_connectivity() is False


async def test_dispose_engine_is_idempotent():
    """Disposing an already-disposed engine is a no-op."""
    import db as db_module

    await db_module.dispose_engine()
    await db_module.dispose_engine()  # must not raise


async def test_get_session_yields_working_session(pg_db):
    """The get_session dependency yields a live async session."""
    from db import get_session

    agen = get_session()
    session = await agen.__anext__()
    try:
        value = (await session.execute(text("SELECT 1"))).scalar_one()
        assert value == 1
    finally:
        await agen.aclose()


async def test_check_connectivity_true_when_initialised(pg_db):
    """check_connectivity returns True against a live engine."""
    import db as db_module

    # pg_db already initialised the engine.
    assert await db_module.check_connectivity() is True


async def test_get_session_factory_returns_factory(pg_db):
    """get_session_factory returns the live factory once initialised."""
    import db as db_module

    factory = db_module.get_session_factory()
    async with factory() as session:
        assert (await session.execute(text("SELECT 1"))).scalar_one() == 1
