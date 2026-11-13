"""Behaviour parity on the REAL py2rs fixture service (py2rust A5c, coding-testing#82).

The real driver runs as a SUBPROCESS (exactly as it runs in a workspace) against
the real FastAPI + SQLAlchemy/psycopg + Alembic fixture on a real postgres:18, with
a local stand-in for ``ws-pg`` (tests/data/behaviour_parity/ws_pg_shim.py). The
request/spec/config plumbing is the production code (``build_driver_config``,
``build_spec``, ``grade``); only ``ws_pg`` and the absolute workdirs are swapped for
local paths.

* vendored copy (tests/data/behaviour_parity/py2rs_fixture_svc, always runs, also in CI);
* the fixture repository itself via ``uv run`` when it is checked out locally
  (``PY2RS_FIXTURE_DIR``, default ~/Movies/conveyor-coding/py2rs-fixture-svc; read-only).

Postgres: ``PARITY_TEST_PG_ADMIN_URL`` if set; else an ephemeral
``docker run --rm --tmpfs ... postgres:18`` on a random port (removed at teardown);
else the server behind ``DATABASE_URL`` (needs CREATEDB). Nothing found -> fail.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import psycopg
import pytest

from models.parity_behaviour_models import (
    CompareRequest,
    RecordRequest,
    ReplayRequest,
)
from services.behaviour_parity import (
    DRIVER_PATH,
    build_driver_config,
    build_spec,
    grade,
)

pytestmark = pytest.mark.integration

DATA = Path(__file__).resolve().parent.parent / "data" / "behaviour_parity"
VENDORED = DATA / "py2rs_fixture_svc"
SHIM = DATA / "ws_pg_shim.py"
LIVE = Path(
    os.environ.get(
        "PY2RS_FIXTURE_DIR", str(Path.home() / "Movies/conveyor-coding/py2rs-fixture-svc")
    )
)
CASES = [json.loads(line) for line in (VENDORED / "parity/requests.jsonl").read_text().splitlines()]


def _wait(url: str, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            with psycopg.connect(url, connect_timeout=2) as conn:
                conn.execute("SELECT 1")
            return
        except psycopg.OperationalError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.5)


@pytest.fixture(scope="module")
def pg_admin_url():
    explicit = os.environ.get("PARITY_TEST_PG_ADMIN_URL")
    if explicit:
        yield explicit
        return
    docker = shutil.which("docker")
    if docker and subprocess.run([docker, "info"], capture_output=True).returncode == 0:
        name = f"ct-a5c-parity-pg-{uuid.uuid4().hex[:8]}"
        subprocess.run(
            [docker, "run", "--rm", "-d", "--name", name, "--tmpfs", "/var/lib/postgresql",
             "-e", "POSTGRES_PASSWORD=test", "-p", "127.0.0.1::5432", "postgres:18"],
            check=True,
            capture_output=True,
        )  # fmt: skip
        try:
            out = subprocess.run(
                [docker, "port", name, "5432/tcp"], check=True, capture_output=True, text=True
            ).stdout
            port = out.strip().splitlines()[0].rsplit(":", 1)[1]
            url = (
                f"postgresql://postgres:test@127.0.0.1:{port}/postgres"  # pragma: allowlist secret
            )
            _wait(url)
            yield url
        finally:
            subprocess.run([docker, "rm", "-f", "-v", name], capture_output=True)
        return
    db = os.environ.get("DATABASE_URL")
    if not db:
        pytest.fail("no postgres for the parity driver tests (no docker, no DATABASE_URL)")
    db = db.replace("postgresql+psycopg://", "postgresql://", 1)
    yield db.rsplit("/", 1)[0] + "/postgres"


def _vendored_side(workdir: Path) -> dict[str, Any]:
    return {
        "workdir": str(workdir),
        "migrate": [sys.executable, "-m", "alembic", "upgrade", "head"],
        "start": [sys.executable, "-m", "uvicorn", "py2rs_fixture_svc.main:app",
                  "--host", "127.0.0.1", "--port", "{port}"],
        "env": {"PYTHONPATH": str(workdir / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
    }  # fmt: skip


def _live_side() -> dict[str, Any]:
    return {
        "workdir": str(LIVE),
        "migrate": ["uv", "run", "--frozen", "alembic", "upgrade", "head"],
        "start": ["uv", "run", "--frozen", "uvicorn", "py2rs_fixture_svc.main:app",
                  "--host", "127.0.0.1", "--port", "{port}"],
        "env": {"PYTHONDONTWRITEBYTECODE": "1"},
    }  # fmt: skip


def _svc(side: dict[str, Any]) -> dict[str, Any]:
    """The request-model view of a side (workdir is checked to be workspace-relative)."""
    return {k: v for k, v in side.items() if k != "workdir"}


def _run(
    tmp_path: Path,
    pg_admin_url: str,
    op: str,
    request: RecordRequest | ReplayRequest | CompareRequest,
    workdirs: dict[str, str],
) -> dict[str, Any]:
    sides = []
    if op in ("record", "compare"):
        sides.append(("source", request.source))  # type: ignore[union-attr]
    if op in ("replay", "compare"):
        sides.append(("target", request.target))  # type: ignore[union-attr]
    config = build_driver_config(request, sides)
    config["ws_pg"] = [sys.executable, str(SHIM)]
    for side in config["sides"]:
        side["workdir"] = workdirs[side["role"]]
    run_dir = tmp_path / uuid.uuid4().hex
    run_dir.mkdir()
    (run_dir / "driver_config.json").write_text(json.dumps(config))
    env = {**os.environ, "WS_PG_SHIM_ADMIN_URL": pg_admin_url}
    done = subprocess.run(
        [sys.executable, "-I", str(DRIVER_PATH), str(run_dir)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    graded = grade(
        build_spec(op, request),  # type: ignore[arg-type]
        result,
        exit_code=done.returncode,
        job_output=done.stdout + done.stderr,
    )
    return {"graded": graded, "raw": result, "proc": done}


def _failed(graded) -> dict[str, list[str]]:
    return {c.name: sorted({d.kind for d in c.diffs}) for c in graded.cases if not c.passed}


def _record_then_replay(tmp_path, pg_admin_url, side: dict[str, Any]) -> None:
    rec = _run(
        tmp_path,
        pg_admin_url,
        "record",
        RecordRequest(cases=CASES, source=_svc(side)),
        {"source": side["workdir"]},
    )
    g = rec["graded"]
    assert g.computed is True, (g.error, g.driver_log_tail)
    # Recording + the source checked against the hand-written expectations.
    assert g.baseline == "expected"
    assert (g.cases_total, g.cases_passed, g.cases_failed) == (26, 26, 0), _failed(g)
    assert g.equal is True
    assert g.recording is not None and len(g.recording["cases"]) == 26
    enrich = next(c for c in g.recording["cases"] if c["name"] == "enrich_ok")
    assert enrich["upstream_calls"][0]["path"] == "/enrich"
    assert enrich["upstream_calls"][0]["body"]["text"] == "first body"

    rep = _run(
        tmp_path,
        pg_admin_url,
        "replay",
        ReplayRequest(cases=CASES, target=_svc(side), baseline="recording", recording=g.recording),
        {"target": side["workdir"]},
    )
    r = rep["graded"]
    assert r.computed is True, (r.error, r.driver_log_tail)
    assert r.baseline == "recording"
    assert (r.equal, r.cases_total, r.cases_passed) == (True, 26, 26), _failed(r)


def test_record_and_replay_vendored_fixture(tmp_path, pg_admin_url):
    _record_then_replay(tmp_path, pg_admin_url, _vendored_side(VENDORED))


@pytest.mark.skipif(
    not (LIVE / "pyproject.toml").exists() or shutil.which("uv") is None,
    reason="fixture repository not checked out here (the vendored copy covers CI)",
)
def test_record_and_replay_live_fixture_repo_via_uv(tmp_path, pg_admin_url):
    _record_then_replay(tmp_path, pg_admin_url, _live_side())


def _mutant(tmp_path: Path) -> Path:
    dst = tmp_path / "mutant"
    shutil.copytree(VENDORED, dst, ignore=shutil.ignore_patterns("__pycache__"))
    main = dst / "src/py2rs_fixture_svc/main.py"
    text = main.read_text()
    status_from, body_from = "status_code=201,", '"text": note.body}'
    assert text.count(status_from) == 1 and text.count(body_from) == 1
    text = text.replace(status_from, "status_code=200,")
    text = text.replace(body_from, '"text": note.body, "lang": "en"}')
    main.write_text(text)
    return dst


#: The cases the two mutations must flag, derived from the case file (not hand-listed):
#: every POST /notes that answers 201, and every case that makes an outbound call.
STATUS_CASES = {c["name"] for c in CASES if c["expected_status"] == 201}
UPSTREAM_CASES = {c["name"] for c in CASES if c["expected_upstream_calls"]}


def test_mutated_service_flags_exactly_the_mutated_cases(tmp_path, pg_admin_url):
    assert len(STATUS_CASES) == 5 and len(UPSTREAM_CASES) == 4
    mutant = _mutant(tmp_path)
    expected = {n: ["status"] for n in STATUS_CASES} | {
        n: ["upstream_body"] for n in UPSTREAM_CASES
    }

    # compare: source (original) and target (mutant) in ONE driver run, target vs source.
    cmp = _run(
        tmp_path,
        pg_admin_url,
        "compare",
        CompareRequest(
            cases=CASES, source=_svc(_vendored_side(VENDORED)), target=_svc(_vendored_side(mutant))
        ),
        {"source": str(VENDORED), "target": str(mutant)},
    )
    g = cmp["graded"]
    assert g.computed is True, (g.error, g.driver_log_tail)
    assert g.baseline == "source" and g.equal is False
    assert (g.cases_total, g.cases_failed) == (26, 9)
    assert _failed(g) == expected
    status_diff = next(c for c in g.cases if c.name == "create_n1").diffs[0]
    assert (status_diff.pointer, status_diff.expected, status_diff.actual) == ("/status", 201, 200)
    body_diff = next(c for c in g.cases if c.name == "enrich_ok").diffs[0]
    assert (body_diff.pointer, body_diff.detail, body_diff.actual) == (
        "/upstream_calls/0/body/lang",
        "unexpected",
        "en",
    )
    assert g.recording is not None and len(g.recording["cases"]) == 26

    # replay vs the hand-written expectations flags the same cases.
    rep = _run(
        tmp_path,
        pg_admin_url,
        "replay",
        ReplayRequest(cases=CASES, target=_svc(_vendored_side(mutant))),
        {"target": str(mutant)},
    )
    r = rep["graded"]
    assert r.computed is True and r.equal is False and r.baseline == "expected"
    assert _failed(r) == expected

    # ...and an ignore path on the outbound body leaves only the status cases.
    rep = _run(
        tmp_path,
        pg_admin_url,
        "replay",
        ReplayRequest(
            cases=CASES, target=_svc(_vendored_side(mutant)), upstream_ignore_paths=["/lang"]
        ),
        {"target": str(mutant)},
    )
    assert _failed(rep["graded"]) == {n: ["status"] for n in STATUS_CASES}


def test_broken_migration_is_not_computed(tmp_path, pg_admin_url):
    side = _vendored_side(VENDORED)
    side["migrate"] = [sys.executable, "-m", "alembic", "upgrade", "no_such_revision"]
    run = _run(
        tmp_path,
        pg_admin_url,
        "replay",
        ReplayRequest(cases=CASES, target=_svc(side)),
        {"target": str(VENDORED)},
    )
    g = run["graded"]
    assert run["proc"].returncode == 1
    assert g.computed is False and g.equal is False and "migrate exited" in (g.error or "")
