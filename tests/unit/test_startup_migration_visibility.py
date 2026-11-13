"""Regression guards for the 2026-08-24 invisible-crashloop incident.

A `coding-testing` pod exited code 3 (uvicorn STARTUP_FAILURE) roughly one second
into its FastAPI lifespan, with a COMPLETELY EMPTY log — no traceback, no error
line, not even uvicorn's own "Application startup failed". The rollout wedged and
sat unnoticed for 8 hours because the Deployment still reported ``READY 2/2``
(only ``UP-TO-DATE`` betrayed it, and nothing alerts on that).

Two independent defects produced that outcome, and each gets a test here:

1. ``alembic/env.py`` called ``logging.config.fileConfig(path)`` with the DEFAULT
   ``disable_existing_loggers=True``. Because migrations run from the lifespan —
   i.e. AFTER the app and uvicorn have configured their loggers — that call
   destroyed every existing logger, so the failure could not report itself.
2. There was no retry. A single transient failure on the first connection raised
   straight out of the lifespan, and the 5-minute CrashLoopBackOff did the rest.
"""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- 1


def test_alembic_env_does_not_disable_existing_loggers() -> None:
    """env.py must opt OUT of fileConfig's logger-killing default.

    Static, because the damage happens at import time of a module that also
    drives a live migration; asserting the call shape is the cheap, durable
    guard. `test_fileConfig_default_kills_a_preconfigured_logger` below pins the
    behaviour this is protecting against, so the pair is not merely cosmetic.
    """
    env_py = (PROJECT_ROOT / "alembic" / "env.py").read_text()
    assert "fileConfig(config.config_file_name)" not in env_py, (
        "alembic/env.py calls fileConfig with the default "
        "disable_existing_loggers=True, which silences the app's own logger and "
        "uvicorn.error — a startup failure then produces an EMPTY log."
    )
    assert "disable_existing_loggers=False" in env_py, (
        "alembic/env.py must pass disable_existing_loggers=False to fileConfig."
    )


def test_fileconfig_default_kills_a_preconfigured_logger(tmp_path: Path) -> None:
    """Pin WHY the flag matters — this is the mechanism, not an opinion.

    Also the negative control: with the flag set to False the same logger
    survives, so the assertion cannot pass for a trivial reason.
    """
    ini = tmp_path / "alembic.ini"
    ini.write_text(
        "[loggers]\nkeys = root\n\n"
        "[handlers]\nkeys = console\n\n"
        "[formatters]\nkeys = generic\n\n"
        "[logger_root]\nlevel = WARN\nhandlers = console\n\n"
        "[handler_console]\nclass = StreamHandler\nargs = (sys.stderr,)\n"
        "level = NOTSET\nformatter = generic\n\n"
        "[formatter_generic]\nformat = %(message)s\n"
    )

    victim = logging.getLogger("coding_testing.startup_victim")
    victim.disabled = False

    logging.config.fileConfig(str(ini))  # the old, defaulted call
    assert victim.disabled is True, (
        "expected fileConfig's default to disable the pre-existing logger; if "
        "this ever stops being true the guard above is no longer needed"
    )

    victim.disabled = False
    logging.config.fileConfig(str(ini), disable_existing_loggers=False)
    assert victim.disabled is False, (
        "disable_existing_loggers=False must leave an existing logger alive"
    )


# --------------------------------------------------------------------------- 2


@pytest.mark.asyncio
async def test_startup_migration_retries_a_transient_failure(monkeypatch) -> None:
    """A blip on the first attempt must NOT kill the pod."""
    import app_factory

    calls: list[int] = []

    def flaky(_url: str) -> None:
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            raise OSError("connection refused (transient)")

    _stub_lifespan_deps(monkeypatch)
    monkeypatch.setattr(app_factory, "_run_alembic_upgrade", flaky)
    monkeypatch.setenv("ALEMBIC_UPGRADE_ATTEMPTS", "5")

    started = False
    async with app_factory.lifespan(_FakeApp()):
        started = True
    assert started, "lifespan must complete when the retry succeeds"
    assert calls == [1, 2], f"expected one retry then success, got {calls}"


@pytest.mark.asyncio
async def test_startup_migration_fails_closed_after_last_attempt(monkeypatch) -> None:
    """Negative control: a PERSISTENT failure must still stop the pod.

    Without this, 'add a retry' could silently become 'start on a schema we never
    verified', which is strictly worse than the crashloop it replaced.
    """
    import app_factory

    calls: list[int] = []

    def always_fails(_url: str) -> None:
        calls.append(len(calls) + 1)
        raise RuntimeError("migration is genuinely broken")

    _stub_lifespan_deps(monkeypatch)
    monkeypatch.setattr(app_factory, "_run_alembic_upgrade", always_fails)
    monkeypatch.setenv("ALEMBIC_UPGRADE_ATTEMPTS", "3")

    with pytest.raises(RuntimeError, match="genuinely broken"):
        async with app_factory.lifespan(_FakeApp()):
            pass
    assert calls == [1, 2, 3], f"expected exactly 3 attempts, got {calls}"


@pytest.mark.asyncio
async def test_the_probe_can_actually_fail(monkeypatch) -> None:
    """Anti-vacuity: prove the harness reaches the migration at all.

    If `_stub_lifespan_deps` ever stops wiring a postgres DSN, `db_enabled` goes
    false, the migration block is skipped entirely, and BOTH tests above would
    pass while measuring nothing.
    """
    import app_factory

    seen: list[str] = []
    _stub_lifespan_deps(monkeypatch)
    monkeypatch.setattr(app_factory, "_run_alembic_upgrade", lambda url: seen.append(url))
    monkeypatch.setenv("ALEMBIC_UPGRADE_ATTEMPTS", "1")

    async with app_factory.lifespan(_FakeApp()):
        pass
    assert seen and seen[0].startswith("postgresql"), (
        "the lifespan never reached the migration — the stub DSN is not a "
        "postgres DSN, so db_enabled was false and these tests measure nothing"
    )


class _FakeApp:
    """Minimal stand-in for FastAPI: the lifespan only touches ``app.state``."""

    class _State:
        pass

    def __init__(self) -> None:
        self.state = _FakeApp._State()


def _stub_lifespan_deps(monkeypatch) -> None:
    """Neutralise everything the lifespan touches EXCEPT the migration block.

    The lifespan imports ``get_settings``/``init_db``/``dispose_engine`` inside the
    function body, so they must be patched on their OWN modules — patching
    app_factory would silently miss them and the test would exercise the real DB.
    """
    import asyncio

    import config.settings as settings_mod
    import db as db_mod

    class _Settings:
        DATABASE_URL = "postgresql://stub-user@stub-host:5432/stub"

    async def _noop_init_db(_settings) -> None:
        return None

    async def _noop_dispose() -> None:
        return None

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(settings_mod, "get_settings", lambda: _Settings())
    monkeypatch.setattr(db_mod, "init_db", _noop_init_db)
    monkeypatch.setattr(db_mod, "dispose_engine", _noop_dispose)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
