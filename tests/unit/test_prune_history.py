"""Tests for orchestration execution-history retention (store + prune task).

DB-backed tests take the ``pg_db`` fixture (self-skips without a Postgres
DATABASE_URL). The task-module tests patch ``_run`` and need no DB.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from services.orchestration_store import get_orchestration_store


async def _seed_old_and_recent(pg_db) -> None:
    """One 'suite' history row aged 200 days + one fresh 'plan' row."""
    store = get_orchestration_store()
    await store.append_history({"type": "suite", "tenant_id": "t1", "timestamp": None})
    await store.append_history({"type": "plan", "tenant_id": "t1", "timestamp": None})
    async with pg_db() as session:
        await session.execute(
            text(
                "UPDATE orchestration_execution_history "
                "SET created_at = now() - make_interval(days => 200) "
                "WHERE data->>'type' = 'suite'"
            )
        )
        await session.commit()


class TestPruneHistoryStore:
    async def test_deletes_rows_older_than_retention(self, pg_db):
        store = get_orchestration_store()
        await _seed_old_and_recent(pg_db)
        deleted = await store.prune_history(90)
        assert deleted == 1
        remaining = await store.list_history("t1")
        assert [h["type"] for h in remaining] == ["plan"]

    async def test_keeps_everything_when_disabled(self, pg_db):
        store = get_orchestration_store()
        await _seed_old_and_recent(pg_db)
        assert await store.prune_history(0) == 0
        assert await store.prune_history(-5) == 0
        assert len(await store.list_history("t1")) == 2

    async def test_recent_rows_survive_a_generous_retention(self, pg_db):
        store = get_orchestration_store()
        await store.append_history({"type": "suite", "tenant_id": "t1", "timestamp": None})
        assert await store.prune_history(1) == 0
        assert len(await store.list_history("t1")) == 1


class TestPruneHistoryTask:
    async def test_run_prunes_via_env(self, pg_db, monkeypatch):
        import db as db_module
        from tasks.prune_history import _run

        await _seed_old_and_recent(pg_db)
        monkeypatch.setenv("ORCHESTRATION_HISTORY_RETENTION_DAYS", "90")

        deleted = await _run()
        assert deleted == 1

        # _run disposed the engine; re-acquire to assert the surviving row.
        factory = await db_module.ensure_session_factory()
        async with factory() as session:
            types = (
                (
                    await session.execute(
                        text("SELECT data->>'type' FROM orchestration_execution_history")
                    )
                )
                .scalars()
                .all()
            )
        assert types == ["plan"]

    async def test_run_requires_database_url(self, monkeypatch):
        from tasks.prune_history import _run

        monkeypatch.delenv("DATABASE_URL", raising=False)
        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            await _run()

    def test_main_success(self, monkeypatch):
        import tasks.prune_history as mod

        ran = {}

        async def fake_run():
            ran["ok"] = True
            return 3

        monkeypatch.setattr(mod, "_run", fake_run)
        mod.main()  # must not raise
        assert ran["ok"]

    def test_main_exits_nonzero_on_failure(self, monkeypatch):
        import tasks.prune_history as mod

        async def boom():
            raise RuntimeError("db unreachable")

        monkeypatch.setattr(mod, "_run", boom)
        with pytest.raises(SystemExit) as exc:
            mod.main()
        assert exc.value.code == 1
