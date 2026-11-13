"""services.run_spec_store against the real Postgres (``pg_db``)."""

from __future__ import annotations

from sqlalchemy import text

from services.run_spec_store import get_run_spec_store


async def test_spec_is_bound_to_kind_workspace_and_tenant(pg_db) -> None:
    store = get_run_spec_store()
    await store.put("a" * 32, "perf", "ws1", "t1", {"op": "perf", "n": 1})
    await store.put("b" * 32, "perf", "ws1", None, {"op": "perf", "n": 2})
    assert await store.get("a" * 32, ("perf",), "ws1", "t1") == {"op": "perf", "n": 1}
    assert await store.get("a" * 32, ("record", "perf"), "ws1", "t1") is not None
    assert await store.get("a" * 32, ("record",), "ws1", "t1") is None
    assert await store.get("a" * 32, ("perf",), "ws2", "t1") is None
    assert await store.get("a" * 32, ("perf",), "ws1", "t2") is None
    assert await store.get("a" * 32, ("perf",), "ws1", None) is None
    # A tenantless run is visible only to a tenantless poll.
    assert await store.get("b" * 32, ("perf",), "ws1", None) == {"op": "perf", "n": 2}
    assert await store.get("b" * 32, ("perf",), "ws1", "t1") is None


async def test_prune_deletes_only_old_specs(pg_db, monkeypatch) -> None:
    store = get_run_spec_store()
    await store.put("c" * 32, "perf", "ws1", None, {})
    await store.put("d" * 32, "perf", "ws1", None, {})
    async with pg_db() as session:
        await session.execute(
            text(
                "UPDATE workspace_run_specs SET created_at = now() - make_interval(days => 30) "
                "WHERE nonce = :n"
            ),
            {"n": "c" * 32},
        )
        await session.commit()
    assert await store.prune(0) == 0
    assert await store.prune(14) == 1
    assert await store.get("c" * 32, ("perf",), "ws1", None) is None
    assert await store.get("d" * 32, ("perf",), "ws1", None) == {}


async def test_prune_task_prunes_run_specs(pg_db, monkeypatch) -> None:
    import db as db_module
    from tasks.prune_history import _run

    store = get_run_spec_store()
    await store.put("e" * 32, "perf", "ws1", None, {})
    async with pg_db() as session:
        await session.execute(
            text("UPDATE workspace_run_specs SET created_at = now() - make_interval(days => 3)")
        )
        await session.commit()
    monkeypatch.setenv("RUN_SPEC_RETENTION_DAYS", "1")
    await db_module.dispose_engine()
    await _run()
    assert await store.get("e" * 32, ("perf",), "ws1", None) is None
