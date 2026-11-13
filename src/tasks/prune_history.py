"""One-shot task: prune old orchestration history, Rust test runs and run specs.

Run by a gitops CronJob against the ``coding_testing`` DB:

    python -m tasks.prune_history

``orchestration_execution_history`` is append-only (one row per suite/plan run)
with no natural bound; this trims rows older than the retention window so it
can't grow forever. ``rust_test_runs`` (services/rust_test_runs.py) is trimmed
with the same retention; ``workspace_run_specs`` (services/run_spec_store.py, one
row per workspace run) with its own.

Reads only from env — deliberately NOT via ``config.settings.Settings``, which
fails loud in production when the app's many service-URL vars are absent. A
maintenance job should not need the whole app config to delete rows, so it
initialises the DB engine from ``DATABASE_URL`` alone:

- ``DATABASE_URL``                         — required (postgresql+psycopg DSN)
- ``ORCHESTRATION_HISTORY_RETENTION_DAYS`` — optional, default 90; <=0 disables
- ``RUN_SPEC_RETENTION_DAYS``              — optional, default 14; <=0 disables

Exit 0 on success (including a no-op); non-zero on failure so the CronJob
surfaces a real error.
"""

from __future__ import annotations

import asyncio
import logging
import os
from types import SimpleNamespace

logger = logging.getLogger("tasks.prune_history")

DEFAULT_RETENTION_DAYS = 90
#: A workspace run is pollable (graded from its spec) for this long after it started.
DEFAULT_RUN_SPEC_RETENTION_DAYS = 14


async def _run() -> int:
    """Prune history older than the configured retention; return rows deleted."""
    from db import dispose_engine, init_db
    from services.orchestration_store import get_orchestration_store
    from services.run_spec_store import get_run_spec_store
    from services.rust_test_runs import prune_runs

    database_url = os.environ.get("DATABASE_URL", "")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")
    retention_days = int(
        os.environ.get("ORCHESTRATION_HISTORY_RETENTION_DAYS", DEFAULT_RETENTION_DAYS)
    )
    spec_days = int(os.environ.get("RUN_SPEC_RETENTION_DAYS", DEFAULT_RUN_SPEC_RETENTION_DAYS))

    # Initialise the shared engine directly from the DSN (a minimal settings-like
    # object), so the store's ensure_session_factory finds a live factory and
    # never falls back to get_settings() / the full production-config validation.
    await init_db(SimpleNamespace(DATABASE_URL=database_url))
    try:
        deleted = await get_orchestration_store().prune_history(retention_days)
        runs = await prune_runs(retention_days)
        specs = await get_run_spec_store().prune(spec_days)
    finally:
        await dispose_engine()
    logger.info(
        "pruned %s orchestration_execution_history row(s) older than %s day(s)",
        deleted,
        retention_days,
    )
    logger.info("pruned %s rust_test_runs row(s)", runs)
    logger.info("pruned %s workspace_run_specs row(s) older than %s day(s)", specs, spec_days)
    return deleted


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        asyncio.run(_run())
    except Exception:
        logger.exception("orchestration history prune failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
