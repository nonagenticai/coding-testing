"""/api/parity/{container}/record|replay|compare|runs over a fake workspace transport."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import zlib
from unittest.mock import patch

import pytest
from coding_client.clients import ServiceError
from sqlalchemy import text

from routers.parity_behaviour import get_behaviour_parity_runner
from services import behaviour_parity as bp
from services.behaviour_parity import (
    _READ_FILE_PY,
    _WRITE_FILE_PY,
    RECORDING_FORMAT,
    BehaviourParityRunner,
    driver_source,
    encode_chunks,
    grade,
)
from services.run_spec_store import get_run_spec_store
from tests.run_specs import stored_spec

# Write routes (they execute in the workspace) need the write role; see
# tests/unit/test_workspace_route_permissions.py for the 403 side.
pytestmark = pytest.mark.usefixtures("as_writer", "pg_db")

SVC = {"workdir": "svc", "start": "uv run uvicorn app:app --port {port}", "migrate": None}
CASE = {
    "name": "c1",
    "method": "GET",
    "path": "/x",
    "upstream_responses": [],
    "expected_status": 200,
    "expected_body": {"ok": True},
    "expected_upstream_calls": [],
}


def _side_case(status=200, body=None):
    return {
        "name": "c1",
        "status": status,
        "headers": {"content-type": "application/json"},
        "body": {"ok": True} if body is None else body,
        "upstream_calls": [],
        "upstream_unconsumed": 0,
    }


def _driver_result(roles=("target",), status=200, cases=None):
    return {
        "driver_version": 1,
        "ok": True,
        "error": None,
        "cases_total": 1,
        "cases": cases or [CASE],
        "sides": {
            r: {"role": r, "ok": True, "cases": [_side_case(status)], "log_tail": "svc log"}
            for r in roles
        },
    }


class FakeWorkspace:
    """A workspace with a file system: python3 -c writes, cat reads, the driver 'runs'."""

    def __init__(self, driver_result: dict | None = None, job: dict | None = None):
        self.files: dict[str, str] = {}
        self.staging: dict[str, str] = {}
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.driver_result = driver_result
        self.job = job
        self.fail: Exception | None = None

    def _write(self, argv: list[str]) -> dict:
        _, _, _, run_dir, name, chunk, first, last = argv
        key = f"{run_dir}/{name}"
        self.staging[key] = chunk if first == "1" else self.staging[key] + chunk
        if last == "1":
            self.files[key] = zlib.decompress(base64.b64decode(self.staging.pop(key))).decode()
        return {"exit_code": 0, "stdout": "", "stderr": ""}

    async def exec(self, container, tenant_id, argv, workdir, timeout):
        self.calls.append(("exec", tuple(argv)))
        if self.fail:
            raise self.fail
        if argv[:2] == ["python3", "-c"]:
            return self._write(argv)
        if argv[:3] == ["python3", "-I", "-c"]:
            assert argv[3] == _READ_FILE_PY
            if argv[4] in self.files:
                data = base64.b64encode(zlib.compress(self.files[argv[4]].encode())).decode()
                return {"exit_code": 0, "stdout": data, "stderr": ""}
            return {"exit_code": 1, "stdout": "", "stderr": "no such file"}
        if argv[:2] == ["python3", "-I"]:
            run_dir = argv[3]
            if self.driver_result is not None:
                self.files[f"{run_dir}/result.json"] = json.dumps(self.driver_result)
            return {"exit_code": 0 if self.driver_result else 1, "stdout": "", "stderr": "e"}
        raise AssertionError(argv)

    async def exec_async(self, container, tenant_id, argv, workdir, timeout):
        self.calls.append(("exec_async", tuple(argv)))
        self.async_argv = argv
        return {"job_id": "job-1", "status": "queued"}

    async def get_exec_job(self, container, tenant_id, job_id):
        self.calls.append(("get_exec_job", (job_id,)))
        if isinstance(self.job, Exception):
            raise self.job
        if self.job and self.job.get("status") in ("completed", "failed"):
            run_dir = self.async_argv[3]
            if self.driver_result is not None:
                self.files[f"{run_dir}/result.json"] = json.dumps(self.driver_result)
        return self.job


async def _fake_validate(container_name: str, tenant_id: str | None = None) -> None:
    if container_name == "missing":
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="not found")


@pytest.fixture
def ws(app):
    holder: dict[str, FakeWorkspace] = {}

    def _use(w: FakeWorkspace) -> FakeWorkspace:
        holder["w"] = w
        app.dependency_overrides[get_behaviour_parity_runner] = lambda: BehaviourParityRunner(w)
        return w

    with patch("routers.parity_behaviour.validate_and_get_container_for_tenant", _fake_validate):
        yield _use
    app.dependency_overrides.pop(get_behaviour_parity_runner, None)


def _run_dir(w: FakeWorkspace) -> str:
    return next(k for k in w.files if k.endswith("/driver_config.json")).rsplit("/", 1)[0]


async def test_compare_async_writes_driver_config_spec_and_polls(client, ws):
    w = ws(FakeWorkspace(_driver_result(("source", "target"), status=201)))
    r = await client.post(
        "/api/parity/ws-1/compare",
        json={
            "cases_file": "svc/parity/requests.jsonl",
            "default_headers": {"X-Tenant-Id": "t"},
            "ignore_paths": ["/id"],
            "source": SVC,
            "target": {"workdir": ".", "start": ["target/debug/svc"], "upstream_env": ["A", "B"]},
            "timeout": 3600,
        },
    )
    assert r.status_code == 202, r.text
    body = r.json()
    nonce = body["run_id"].split(".")[0]
    assert body["run_id"] == f"{nonce}.job-1" and body["status"] == "queued"
    assert body["links"]["status"] == f"/api/parity/ws-1/runs/{nonce}.job-1"
    run_dir = f"/tmp/coding-testing/{nonce}"
    assert w.files[f"{run_dir}/parity_driver.py"] == driver_source()
    cfg = json.loads(w.files[f"{run_dir}/driver_config.json"])
    assert cfg["cases_file"] == "/workspace/svc/parity/requests.jsonl" and "cases" not in cfg
    assert cfg["ws_pg"] == ["ws-pg"] and cfg["default_headers"] == {"X-Tenant-Id": "t"}
    assert [s["role"] for s in cfg["sides"]] == ["source", "target"]
    assert cfg["sides"][0]["workdir"] == "/workspace/svc"
    assert cfg["sides"][0]["start"] == ["uv", "run", "uvicorn", "app:app", "--port", "{port}"]
    assert cfg["sides"][1]["workdir"] == "/workspace"
    assert cfg["sides"][1]["upstream_env"] == ["A", "B"]
    assert f"{run_dir}/spec.json" not in w.files  # kept service-side
    spec = await stored_spec(nonce)
    assert spec["op"] == "compare" and spec["baseline"] == "source"
    assert spec["ignore_paths"] == ["/id"]
    assert w.async_argv == ["python3", "-I", f"{run_dir}/parity_driver.py", run_dir]

    w.job = {"status": "running"}
    r = await client.get(f"/api/parity/ws-1/runs/{body['run_id']}")
    assert r.json()["status"] == "running" and r.json()["op"] == "compare"
    assert r.json()["result"] is None

    w.job = {"status": "completed", "exit_code": 0, "stdout": "{}", "stderr": ""}
    r = await client.get(f"/api/parity/ws-1/runs/{body['run_id']}")
    out = r.json()
    assert out["status"] == "completed" and out["op"] == "compare"
    res = out["result"]
    assert res["computed"] is True and res["equal"] is True and res["baseline"] == "source"
    assert res["recording"]["format"] == RECORDING_FORMAT


async def test_replay_sync_against_expected_reports_status_diff(client, ws):
    ws(FakeWorkspace(_driver_result(("target",), status=500)))
    r = await client.post(
        "/api/parity/ws-1/replay",
        json={"cases": [CASE], "target": SVC, "async": False, "timeout": 120},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["status"] == "completed" and len(out["run_id"]) == 32
    res = out["result"]
    assert res["computed"] is True and res["equal"] is False and res["cases_failed"] == 1
    assert res["cases"][0]["diffs"][0] == {
        "kind": "status",
        "pointer": "/status",
        "expected": 200,
        "actual": 500,
        "detail": None,
    }


async def test_replay_against_recording_run_id(client, ws, pg_db):
    w = ws(FakeWorkspace(_driver_result(("source",), status=404)))
    r = await client.post(
        "/api/parity/ws-1/record",
        json={"cases": [CASE], "source": SVC, "async": False, "timeout": 60},
    )
    other = r.json()["run_id"]  # a real record run of this workspace
    w.driver_result = _driver_result(("target",))

    def replay(rid: str) -> dict:
        return {
            "cases": [CASE],
            "target": SVC,
            "baseline": "recording",
            "recording_run_id": rid,
            "async": False,
            "timeout": 60,
        }

    r = await client.post("/api/parity/ws-1/replay", json=replay(f"{other}.job-9"))
    res = r.json()["result"]
    assert res["baseline"] == "recording" and res["computed"] is True
    assert res["cases"][0]["diffs"][0]["expected"] == 404

    # The recorded result is gone from the pod -> not computed, never a vacuous pass.
    del w.files[f"/tmp/coding-testing/{other}/result.json"]
    out = (await client.post("/api/parity/ws-1/replay", json=replay(other))).json()
    assert out["status"] == "failed" and out["result"]["computed"] is False
    assert "recording_run_id" in out["result"]["error"]

    # Another workspace's run, or one whose spec was pruned: a clear 422.
    r = await client.post("/api/parity/ws-2/replay", json=replay(other))
    assert r.status_code == 422 and "expired" in r.text
    async with pg_db() as session:
        await session.execute(
            text("UPDATE workspace_run_specs SET created_at = now() - make_interval(days => 30)")
        )
        await session.commit()
    assert await get_run_spec_store().prune(14) >= 1
    r = await client.post("/api/parity/ws-1/replay", json=replay(other))
    assert r.status_code == 422 and "expired" in r.text and "RUN_SPEC_RETENTION_DAYS" in r.text


async def test_replay_inline_recording(client, ws):
    ws(FakeWorkspace(_driver_result(("target",))))
    rec = {"format": RECORDING_FORMAT, "cases": [_side_case()]}
    r = await client.post(
        "/api/parity/ws-1/replay",
        json={
            "cases": [CASE],
            "target": SVC,
            "baseline": "recording",
            "recording": rec,
            "async": False,
            "timeout": 60,
        },
    )
    assert r.json()["result"]["equal"] is True
    r = await client.post(
        "/api/parity/ws-1/replay",
        json={
            "cases": [CASE],
            "target": SVC,
            "baseline": "recording",
            "recording": {"cases": []},
            "async": False,
            "timeout": 60,
        },
    )
    res = r.json()["result"]
    assert res["computed"] is False and "recording is not" in res["error"]


async def test_record_sync_and_driver_failure(client, ws):
    w = ws(FakeWorkspace(_driver_result(("source",))))
    r = await client.post(
        "/api/parity/ws-1/record",
        json={"cases": [CASE], "source": SVC, "async": False, "timeout": 60},
    )
    res = r.json()["result"]
    assert res["op"] == "record" and res["baseline"] == "expected" and res["equal"] is True
    assert res["recording"]["cases"][0]["status"] == 200

    w.driver_result = None  # the driver wrote nothing and exited 1
    r = await client.post(
        "/api/parity/ws-1/record",
        json={"cases": [CASE], "source": SVC, "async": False, "timeout": 60},
    )
    out = r.json()
    assert out["status"] == "failed"
    assert out["result"]["computed"] is False and out["result"]["equal"] is False
    assert "Driver result not found" in out["result"]["error"]


@pytest.mark.parametrize(
    "body, needle",
    [
        ({"cases": [CASE], "cases_file": "a.jsonl", "source": SVC}, "exactly one of"),
        ({"source": SVC}, "exactly one of"),
        ({"cases": [], "source": SVC}, "must not be empty"),
        ({"cases_file": "/etc/passwd", "source": SVC}, "workspace-relative"),
        ({"cases": [CASE], "source": {**SVC, "workdir": "../x"}}, "workspace-relative"),
        ({"cases": [CASE], "source": {**SVC, "env": {"DATABASE_URL": "x"}}}, "set by the driver"),
        ({"cases": [CASE], "source": {**SVC, "upstream_env": ["1BAD"]}}, "invalid"),
        ({"cases": [CASE], "source": {**SVC, "health_path": "health"}}, "start with"),
        ({"cases": [CASE], "source": {**SVC, "start": ["a\x00"]}}, "NUL"),
        ({"cases": [CASE], "source": SVC, "async": False, "timeout": 900}, "needs async=true"),
    ],
)
async def test_record_validation(client, ws, body, needle):
    ws(FakeWorkspace())
    r = await client.post("/api/parity/ws-1/record", json=body)
    assert r.status_code == 422, r.text
    assert needle in r.text


@pytest.mark.parametrize(
    "extra, needle",
    [
        ({"baseline": "recording"}, "exactly one of recording or recording_run_id"),
        ({"baseline": "recording", "recording_run_id": "nope"}, "not a run id"),
        ({"recording": {"a": 1}}, "need baseline='recording'"),
    ],
)
async def test_replay_validation(client, ws, extra, needle):
    ws(FakeWorkspace())
    r = await client.post("/api/parity/ws-1/replay", json={"cases": [CASE], "target": SVC, **extra})
    assert r.status_code == 422 and needle in r.text


async def test_poll_errors(client, ws):
    w = ws(FakeWorkspace())
    r = await client.get(f"/api/parity/ws-1/runs/{'a' * 32}")
    assert r.status_code == 404  # sync run ids are not pollable
    r = await client.get("/api/parity/ws-1/runs/" + "z" * 40)
    assert r.status_code == 404
    w.job = ServiceError(404, "gone")
    r = await client.get(f"/api/parity/ws-1/runs/{'a' * 32}.job-1")
    assert r.status_code == 404
    w.job = ServiceError(500, "boom")
    r = await client.get(f"/api/parity/ws-1/runs/{'a' * 32}.job-1")
    assert r.status_code == 502
    w.job = RuntimeError("down")
    r = await client.get(f"/api/parity/ws-1/runs/{'a' * 32}.job-1")
    assert r.status_code == 502
    w.job = None
    r = await client.get(f"/api/parity/ws-1/runs/{'a' * 32}.job-1")
    assert r.status_code == 502
    w.job = {"status": "weird"}
    r = await client.get(f"/api/parity/ws-1/runs/{'a' * 32}.job-1")
    assert r.status_code == 502
    w.job = {"status": "queued"}  # no spec in this workspace
    r = await client.get(f"/api/parity/ws-1/runs/{'a' * 32}.job-1")
    assert r.status_code == 404
    r = await client.get(f"/api/parity/missing/runs/{'a' * 32}.job-1")
    assert r.status_code == 404


async def test_failed_job_without_exit_code_is_not_computed(client, ws):
    w = ws(FakeWorkspace(None))
    r = await client.post("/api/parity/ws-1/replay", json={"cases": [CASE], "target": SVC})
    run_id = r.json()["run_id"]
    w.job = {"status": "failed", "exit_code": None, "error": "orphaned: replica died"}
    out = (await client.get(f"/api/parity/ws-1/runs/{run_id}")).json()
    assert out["status"] == "failed" and out["result"]["computed"] is False
    assert out["result"]["error"] == "orphaned: replica died"

    # The pod was replaced: the spec survives service-side, the missing result is
    # "not computed", never a verdict.
    w.files.clear()
    w.job = {"status": "completed", "exit_code": 0}
    out = (await client.get(f"/api/parity/ws-1/runs/{run_id}")).json()
    assert out["status"] == "failed" and out["result"]["computed"] is False
    # Another workspace never sees the run.
    r = await client.get(f"/api/parity/ws-2/runs/{run_id}")
    assert r.status_code == 404


async def test_transport_failures(client, ws):
    w = ws(FakeWorkspace())
    w.fail = ServiceError(404, "nope")
    r = await client.post("/api/parity/ws-1/record", json={"cases": [CASE], "source": SVC})
    assert r.status_code == 404
    w.fail = ServiceError(500, "nope")
    r = await client.post("/api/parity/ws-1/record", json={"cases": [CASE], "source": SVC})
    assert r.status_code == 502
    w.fail = RuntimeError("socket")
    r = await client.post("/api/parity/ws-1/record", json={"cases": [CASE], "source": SVC})
    assert r.status_code == 502
    r = await client.post("/api/parity/missing/record", json={"cases": [CASE], "source": SVC})
    assert r.status_code == 404


async def test_async_submit_failures(client, ws):
    w = ws(FakeWorkspace())

    async def boom(*a, **k):
        raise ServiceError(503, "busy")

    w.exec_async = boom  # type: ignore[method-assign]
    r = await client.post("/api/parity/ws-1/record", json={"cases": [CASE], "source": SVC})
    assert r.status_code == 503

    async def no_job(*a, **k):
        return {"status": "queued"}

    w.exec_async = no_job  # type: ignore[method-assign]
    r = await client.post("/api/parity/ws-1/record", json={"cases": [CASE], "source": SVC})
    assert r.status_code == 502 and "job_id" in r.text

    async def crash(*a, **k):
        raise RuntimeError("x")

    w.exec_async = crash  # type: ignore[method-assign]
    r = await client.post("/api/parity/ws-1/record", json={"cases": [CASE], "source": SVC})
    assert r.status_code == 502


def test_write_helper_reassembles_multi_chunk_files(tmp_path):
    """The real _WRITE_FILE_PY, run locally, rebuilds a file sent in several chunks."""
    content = json.dumps([{"name": f"c{i}", "x": "y" * 50, "n": i} for i in range(3000)])
    chunks = encode_chunks(content, chunk=1000)
    assert len(chunks) > 3
    for i, chunk in enumerate(chunks):
        subprocess.run(
            [sys.executable, "-c", _WRITE_FILE_PY, str(tmp_path / "run"), "cases.json", chunk,
             "1" if i == 0 else "0", "1" if i == len(chunks) - 1 else "0"],
            check=True,
        )  # fmt: skip
    assert (tmp_path / "run" / "cases.json").read_text() == content
    assert not (tmp_path / "run" / "cases.json.b64").exists()
    out = subprocess.run(
        [sys.executable, "-I", "-c", _READ_FILE_PY, str(tmp_path / "run" / "cases.json")],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert zlib.decompress(base64.b64decode(out)).decode() == content
    assert len(out) < len(content) / 5


def test_grade_record_without_expectations_and_bad_results():
    spec = bp.build_spec(
        "record", bp.RecordRequest(cases=[{"name": "c1", "method": "GET", "path": "/"}], source=SVC)
    )
    result = _driver_result(("source",), cases=[{"name": "c1", "method": "GET", "path": "/"}])
    g = grade(spec, result, exit_code=0)
    assert g.computed is True and g.equal is None and g.baseline == "none"
    assert g.recording is not None and g.cases_total == 1

    g = grade(spec, None, exit_code=1, job_output="Traceback")
    assert g.computed is False and g.equal is False and "Traceback" in g.driver_log_tail
    g = grade(spec, {**result, "ok": False, "error": "ws-pg missing"}, exit_code=1)
    assert g.error == "ws-pg missing"
    broken = _driver_result(("source",))
    broken["sides"]["source"]["cases"] = []
    g = grade(spec, broken, exit_code=0)
    assert g.computed is False and "no complete case list" in (g.error or "")
    g = grade(spec, result, exit_code=None, exec_error="timeout: 3600s")
    assert g.computed is False and g.error == "timeout: 3600s"


async def test_unreadable_result_documents_are_not_computed(client, ws):
    w = ws(FakeWorkspace([1, 2]))  # result.json holds a JSON list, not an object
    r = await client.post(
        "/api/parity/ws-1/replay",
        json={"cases": [CASE], "target": SVC, "async": False, "timeout": 60},
    )
    res = r.json()["result"]
    assert res["computed"] is False and "not a JSON object" in res["error"]

    async def garbage(container, tenant_id, argv, workdir, timeout):
        if argv[:3] == ["python3", "-I", "-c"]:
            return {"exit_code": 0, "stdout": "!!notbase64", "stderr": ""}
        return await FakeWorkspace.exec(w, container, tenant_id, argv, workdir, timeout)

    w.exec = garbage  # type: ignore[method-assign]
    w.job = {"status": "completed", "exit_code": 0}
    w.async_argv = ["python3", "-I", "x", "/tmp/coding-testing/" + "a" * 32]
    r = await client.get(f"/api/parity/ws-1/runs/{'a' * 32}.job-1")
    assert r.status_code == 404  # no stored spec: not a run of this workspace


# --------------------------------------------------------------- N sources -> 1 target


async def test_multi_corpus_validation_over_http(client, ws):
    ws(FakeWorkspace())
    r = await client.post(
        "/api/parity/ws-1/compare",
        json={"target": SVC, "corpora": [{"source_id": "a", "cases": [CASE]}]},
    )
    assert r.status_code == 422 and "needs a source" in r.text
    r = await client.post(
        "/api/parity/ws-1/replay",
        json={"target": SVC, "cases": [CASE], "corpora": [{"source_id": "a", "cases": [CASE]}]},
    )
    assert r.status_code == 422 and "inside each corpus" in r.text


# ------------------------------------------------------------------ cli / library sides

CLI_SVC = {"kind": "cli", "workdir": "lib", "start": ["uv", "run", "python", "{lib_harness}"]}
CLI_CASE = {
    "name": "add",
    "kind": "cli",
    "args": ["mylib:add"],
    "stdin": {"args": [1, 2]},
    "stdout_format": "json",
}


async def test_cli_compare_stages_the_harness_and_a_cli_side_config(client, ws):
    w = ws(FakeWorkspace())
    r = await client.post(
        "/api/parity/ws-1/compare",
        json={
            "cases": [CLI_CASE],
            "source": CLI_SVC,
            "target": {"kind": "cli", "start": "target/release/parity_harness"},
        },
    )
    assert r.status_code == 202, r.text
    run_dir = _run_dir(w)
    assert w.files[f"{run_dir}/lib_call_harness.py"] == bp.lib_harness_source()
    cfg = json.loads(w.files[f"{run_dir}/driver_config.json"])
    assert cfg["sides"] == [
        {
            "role": "source",
            "kind": "cli",
            "workdir": "/workspace/lib",
            "start": ["uv", "run", "python", "{lib_harness}"],
            "env": {},
        },
        {
            "role": "target",
            "kind": "cli",
            "workdir": "/workspace",
            "start": ["target/release/parity_harness"],
            "env": {},
        },
    ]
    assert cfg["cases"] == [CLI_CASE]


@pytest.mark.parametrize(
    "path, body, needle",
    [
        (
            "record",
            {"cases": [CLI_CASE], "source": {**CLI_SVC, "migrate": ["sqlx", "migrate", "run"]}},
            "drop ['migrate']",
        ),
        (
            "record",
            {"cases": [CLI_CASE], "source": {**CLI_SVC, "upstreams": {"kag": ["KAG_URL"]}}},
            "drop ['upstreams']",
        ),
        ("record", {"cases": [CLI_CASE, CASE], "source": CLI_SVC}, "cannot mix"),
        ("record", {"cases": [CASE], "source": CLI_SVC}, "runs only cli cases"),
        (
            "record",
            {"cases": [{**CLI_CASE, "timeout": 301}], "source": CLI_SVC},
            "timeout must be > 0 and <= 300 s",
        ),
        (
            "replay",
            {
                "target": CLI_SVC,
                "corpora": [{"source_id": "a", "cases": [{**CLI_CASE, "timeout": 0}]}],
            },
            "timeout must be > 0 and <= 300 s",
        ),
        ("replay", {"cases": [CLI_CASE], "target": SVC}, "need sides with kind='cli'"),
        (
            "compare",
            {"cases": [CLI_CASE], "source": CLI_SVC, "target": SVC},
            "need sides with kind='cli'",
        ),
        (
            "compare",
            {
                "target": CLI_SVC,
                "corpora": [{"source_id": "a", "cases": [CLI_CASE], "source": SVC}],
            },
            "need sides with kind='cli'",
        ),
        (
            "replay",
            {"target": SVC, "corpora": [{"source_id": "a", "cases": [CLI_CASE]}]},
            "need sides with kind='cli'",
        ),
        ("record", {"cases": [CLI_CASE], "source": {**CLI_SVC, "kind": "x"}}, "kind"),
    ],
)
async def test_cli_validation(client, ws, path, body, needle):
    ws(FakeWorkspace())
    r = await client.post(f"/api/parity/ws-1/{path}", json=body)
    assert r.status_code == 422, r.text
    assert needle in r.text
