"""Language dispatch, run/poll flow and grading of workspace test runs.

The coding-containers transport is a recording fake; the JUnit it hands back
is the real sample data in tests/data/junit (see test_junit_parser.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from coding_client.clients import ServiceError
from fastapi import HTTPException

from models.unified_test_models import WorkspaceTestRunRequest
from routers.workspace_test_runs import get_workspace_test_runner
from services.workspace_exec import ContainersTransport
from services.workspace_test_runner import (
    RUN_ROOT,
    WorkspaceTestRunner,
    build_command,
    build_result,
    detect_language,
    parse_ws_pg_output,
    redact_command,
)

# Write routes (they execute in the workspace) need the write role; see
# tests/unit/test_workspace_route_permissions.py for the 403 side.
from tests.run_specs import stored_spec

pytestmark = pytest.mark.usefixtures("as_writer", "pg_db")

DATA = Path(__file__).resolve().parent.parent / "data" / "junit"
PYTEST_MIXED = (DATA / "pytest-mixed.xml").read_text()
PYTEST_EMPTY = (DATA / "pytest-empty.xml").read_text()
PASSING = '<testsuites time="0.5"><testsuite name="s"><testcase name="ok" time="0.5"/></testsuite></testsuites>'
DB_URL = "postgresql://ws:secret@127.0.0.1:5432/ws"  # pragma: allowlist secret


def _ok(stdout: str = "", exit_code: int = 0, **extra: Any) -> dict[str, Any]:
    return {"exit_code": exit_code, "stdout": stdout, "stderr": "", "duration": 1.25, **extra}


class FakeTransport:
    """Scripted coding-containers: answers by program name, records every call."""

    def __init__(
        self,
        *,
        entries: str = "pyproject.toml\nuv.lock\nsrc\n",
        test_exit: int = 0,
        report: str | None = PASSING,
        ws_pg: dict[str, Any] | None = None,
        job: dict[str, Any] | None = None,
    ) -> None:
        self.entries = entries
        self.test_exit = test_exit
        self.report = report
        self.ws_pg = ws_pg or _ok(f"DATABASE_URL={DB_URL}\n")
        self.job = job
        self.calls: list[tuple[str, list[str], str, int]] = []
        self.files: dict[str, str] = {}
        self.dirs: list[str] = []
        self.async_calls: list[tuple[list[str], str, int]] = []

    async def exec(self, container, tenant_id, argv, workdir, timeout):
        self.calls.append((container, argv, workdir, timeout))
        prog = argv[0]
        if prog == "ls":
            return _ok(self.entries)
        if prog == "ws-pg":
            return self.ws_pg
        if prog == "python3":
            self.dirs.append(argv[3])
            return _ok()
        if prog == "cat":
            path = argv[1]
            if path.endswith("junit.xml"):
                return _ok(self.report) if self.report is not None else _ok(exit_code=1)
            if path in self.files:
                return _ok(self.files[path])
            return _ok(exit_code=1)
        return _ok("== 1 passed ==", exit_code=self.test_exit)

    async def exec_async(self, container, tenant_id, argv, workdir, timeout):
        self.async_calls.append((argv, workdir, timeout))
        return {"job_id": "job-1", "status": "queued"}

    async def get_exec_job(self, container, tenant_id, job_id):
        assert job_id == "job-1"
        return self.job or {"status": "running"}


def _test_call(t: FakeTransport) -> tuple[str, list[str], str, int]:
    return next(c for c in t.calls if c[1][0] not in ("ls", "ws-pg", "python3", "cat"))


# --------------------------------------------------------------------- dispatch


@pytest.mark.parametrize(
    ("entries", "expected"),
    [
        ({"Cargo.toml", "src"}, "rust"),
        ({"pyproject.toml"}, "python"),
        ({"uv.lock"}, "python"),
    ],
)
def test_detect_language(entries: set[str], expected: str) -> None:
    assert detect_language(entries) == expected


@pytest.mark.parametrize("entries", [{"Cargo.toml", "pyproject.toml"}, {"README.md"}, set()])
def test_detect_language_refuses_to_guess(entries: set[str]) -> None:
    with pytest.raises(HTTPException) as e:
        detect_language(entries)
    assert e.value.status_code == 422


def test_build_command_python_with_and_without_lock() -> None:
    argv, junit = build_command("/tmp/coding-testing/n", True, ["--no-cov"])
    assert argv[:4] == ["uv", "run", "--frozen", "pytest"]
    assert "--junitxml=/tmp/coding-testing/n/junit.xml" in argv
    assert argv[-1] == "--no-cov" and junit == "/tmp/coding-testing/n/junit.xml"
    argv, _ = build_command("/tmp/coding-testing/n", False, [])
    assert argv[:3] == ["python", "-m", "pytest"]


def test_ws_pg_output_and_redaction() -> None:
    assert parse_ws_pg_output(f"starting...\nDATABASE_URL={DB_URL}\n") == DB_URL
    assert parse_ws_pg_output(f"export DATABASE_URL='{DB_URL}'") == DB_URL
    assert parse_ws_pg_output("postgresql+psycopg://a@h/db") == "postgresql+psycopg://a@h/db"
    assert parse_ws_pg_output("no url here") is None
    assert redact_command(["env", f"DATABASE_URL={DB_URL}", "x"]) == [
        "env",
        "DATABASE_URL=<redacted>",
        "x",
    ]


@pytest.mark.parametrize("bad", ["/etc", "../x", "a/../../b", "-rf"])
def test_workdir_must_be_checkout_relative(bad: str) -> None:
    with pytest.raises(ValueError):
        WorkspaceTestRunRequest(workdir=bad)


def test_request_defaults_and_async_alias() -> None:
    req = WorkspaceTestRunRequest.model_validate({"async": True, "workdir": "svc/api/"})
    assert req.run_async is True
    assert req.absolute_workdir() == "/workspace/svc/api"
    assert WorkspaceTestRunRequest().absolute_workdir() == "/workspace"
    with pytest.raises(ValueError):
        WorkspaceTestRunRequest.model_validate({"bogus": 1})
    with pytest.raises(ValueError):
        WorkspaceTestRunRequest(extra_args=["a\x00b"])


# --------------------------------------------------------------------- grading


def _grade(**kw: Any):
    base: dict[str, Any] = {
        "framework": "pytest",
        "language": "python",
        "command": ["pytest"],
        "workdir": "/workspace",
        "with_postgres": False,
        "exit_code": 0,
        "stdout": "out",
        "stderr": "",
        "exec_duration": 2.0,
        "report_text": PASSING,
        "report_error": None,
    }
    base.update(kw)
    return build_result(**base)


def test_grade_pass() -> None:
    r = _grade()
    assert r.success and r.error is None and r.passed == 1 and r.total == 1
    assert r.report_found and r.duration == pytest.approx(0.5)


def test_grade_zero_tests_is_not_a_pass() -> None:
    r = _grade(report_text=PYTEST_EMPTY, exit_code=5)
    assert not r.success and r.total == 0 and "no tests ran" in (r.error or "")


def test_grade_all_skipped_is_not_a_pass() -> None:
    xml = '<testsuite><testcase name="a"><skipped/></testcase></testsuite>'
    r = _grade(report_text=xml)
    assert not r.success and r.skipped == 1 and "no tests ran" in (r.error or "")


def test_grade_failures() -> None:
    r = _grade(report_text=PYTEST_MIXED, exit_code=1)
    assert not r.success
    assert (r.passed, r.failed, r.errored, r.skipped, r.total) == (2, 1, 1, 1, 5)
    assert r.error == "1 failed, 1 errored"


def test_grade_nonzero_exit_with_all_passed_is_not_a_pass() -> None:
    r = _grade(exit_code=1)
    assert not r.success and "exit code 1" in (r.error or "")


def test_grade_missing_report_or_garbage() -> None:
    r = _grade(report_text=None, report_error="JUnit report not found at x", exec_duration=None)
    assert not r.success and not r.report_found and r.error == "JUnit report not found at x"
    assert r.duration == 0.0
    r = _grade(report_text="garbage")
    assert not r.success and "unparseable" in (r.error or "")


def test_grade_exec_error_wins_and_tail_is_capped() -> None:
    r = _grade(exec_error="timeout: 10s", stdout="o" * 5000, stderr="e")
    assert not r.success and r.error == "timeout: 10s"
    assert len(r.raw_tail) == 4000 and r.raw_tail.endswith("\ne")
    assert _grade(stdout="", stderr="only err").raw_tail == "only err"


# --------------------------------------------------------------------- run / poll


async def test_sync_python_run_detected_and_graded() -> None:
    t = FakeTransport()
    run = await WorkspaceTestRunner(t).run("ws1", "tenant-a", WorkspaceTestRunRequest())
    assert run.status == "completed" and run.result is not None
    assert (
        run.result.success and run.result.framework == "pytest" and run.result.language == "python"
    )
    container, argv, workdir, timeout = _test_call(t)
    assert container == "ws1" and workdir == "/workspace" and timeout == 300
    assert argv[:4] == ["uv", "run", "--frozen", "pytest"]
    assert t.dirs == [f"{RUN_ROOT}/{run.run_id}"]
    spec = await stored_spec(run.run_id)
    assert spec is not None
    assert spec["framework"] == "pytest" and spec["junit"] == f"{RUN_ROOT}/{run.run_id}/junit.xml"


async def test_python_run_with_postgres_in_subdir() -> None:
    t = FakeTransport(report=PYTEST_MIXED, test_exit=1)
    req = WorkspaceTestRunRequest(workdir="svc/api", with_postgres=True)
    run = await WorkspaceTestRunner(t).run("ws1", None, req)
    assert run.result is not None
    r = run.result
    assert r.framework == "pytest" and not r.success and r.with_postgres
    _, argv, workdir, _ = _test_call(t)
    assert workdir == "/workspace/svc/api"
    assert argv[:2] == ["env", f"DATABASE_URL={DB_URL}"]
    assert argv[2:6] == ["uv", "run", "--frozen", "pytest"]
    assert "DATABASE_URL=<redacted>" in r.command and DB_URL not in json.dumps(r.model_dump())
    ws_pg_call = next(c for c in t.calls if c[1][0] == "ws-pg")
    assert ws_pg_call[2] == "/workspace/svc/api"


async def test_presets_are_rust_only() -> None:
    with pytest.raises(HTTPException) as e:
        await WorkspaceTestRunner(FakeTransport()).run(
            "ws1", None, WorkspaceTestRunRequest(presets=["clippy"])
        )
    assert e.value.status_code == 422 and "rust" in e.value.detail


async def test_explicit_language_overrides_detection() -> None:
    t = FakeTransport(entries="Cargo.toml\npyproject.toml\n")
    run = await WorkspaceTestRunner(t).run("ws1", None, WorkspaceTestRunRequest(language="python"))
    assert run.result is not None and run.result.framework == "pytest"
    _, argv, _, _ = _test_call(t)
    assert argv[:3] == ["python", "-m", "pytest"]  # no uv.lock -> no `uv run`


async def test_sync_timeout_over_exec_ceiling_needs_async() -> None:
    with pytest.raises(HTTPException) as e:
        await WorkspaceTestRunner(FakeTransport()).run(
            "ws1", None, WorkspaceTestRunRequest(timeout=301)
        )
    assert e.value.status_code == 422


async def test_sync_timeout_exit_124_is_reported() -> None:
    t = FakeTransport(test_exit=124, report=None)
    run = await WorkspaceTestRunner(t).run("ws1", None, WorkspaceTestRunRequest(timeout=10))
    assert run.result is not None and run.result.error == "timed out after 10s"


async def test_truncated_report_is_not_computed() -> None:
    class Clipping(FakeTransport):
        async def exec(self, container, tenant_id, argv, workdir, timeout):
            res = await super().exec(container, tenant_id, argv, workdir, timeout)
            if argv[0] == "cat" and argv[1].endswith("junit.xml"):
                res["truncated"] = True
            return res

    run = await WorkspaceTestRunner(Clipping()).run("ws1", None, WorkspaceTestRunRequest())
    assert run.result is not None and not run.result.success
    assert "clipped" in (run.result.error or "")


async def test_unreadable_workdir_is_422() -> None:
    class NoDir(FakeTransport):
        async def exec(self, container, tenant_id, argv, workdir, timeout):
            if argv[0] == "ls":
                return {"exit_code": 2, "stdout": "", "stderr": "No such file"}
            return await super().exec(container, tenant_id, argv, workdir, timeout)

    with pytest.raises(HTTPException) as e:
        await WorkspaceTestRunner(NoDir()).run("ws1", None, WorkspaceTestRunRequest(workdir="x"))
    assert e.value.status_code == 422


async def test_ws_pg_failure_is_502() -> None:
    t = FakeTransport(ws_pg=_ok("", exit_code=1))
    with pytest.raises(HTTPException) as e:
        await WorkspaceTestRunner(t).run("ws1", None, WorkspaceTestRunRequest(with_postgres=True))
    assert e.value.status_code == 502 and "ws-pg" in e.value.detail
    assert not any(c[1][0] in ("uv", "python", "cargo", "env") for c in t.calls)


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (ServiceError(404, "nf"), 404),
        (ServiceError(400, "command not permitted", body={"detail": "x"}), 502),
        (RuntimeError("boom"), 502),
    ],
)
async def test_transport_failures_map_to_http(exc: Exception, code: int) -> None:
    class Broken(FakeTransport):
        async def exec(self, *a, **k):
            raise exc

    with pytest.raises(HTTPException) as e:
        await WorkspaceTestRunner(Broken()).run("ws1", None, WorkspaceTestRunRequest())
    assert e.value.status_code == code


async def test_exec_without_exit_code_is_502() -> None:
    class NoExit(FakeTransport):
        async def exec(self, *a, **k):
            return {"stdout": "x"}

    with pytest.raises(HTTPException) as e:
        await WorkspaceTestRunner(NoExit()).run("ws1", None, WorkspaceTestRunRequest())
    assert e.value.status_code == 502


async def test_prepare_failure_is_502() -> None:
    class NoWrite(FakeTransport):
        async def exec(self, container, tenant_id, argv, workdir, timeout):
            if argv[0] == "python3":
                return {"exit_code": 1, "stderr": "read-only"}
            return await super().exec(container, tenant_id, argv, workdir, timeout)

    with pytest.raises(HTTPException) as e:
        await WorkspaceTestRunner(NoWrite()).run("ws1", None, WorkspaceTestRunRequest())
    assert e.value.status_code == 502


async def test_async_run_then_poll_running_then_completed() -> None:
    t = FakeTransport(report=PYTEST_MIXED)
    runner = WorkspaceTestRunner(t)
    run = await runner.run(
        "ws1", "tn", WorkspaceTestRunRequest.model_validate({"async": True, "timeout": 3600})
    )
    assert run.status == "queued" and run.result is None
    nonce, job = run.run_id.split(".")
    assert job == "job-1" and len(nonce) == 32
    assert (
        run.links is not None and run.links.status == f"/api/tests/workspace/ws1/runs/{run.run_id}"
    )
    argv, workdir, timeout = t.async_calls[0]
    assert argv[:4] == ["uv", "run", "--frozen", "pytest"] and timeout == 3600
    assert not any(c[1][0] == "uv" for c in t.calls)  # never ran synchronously

    polled = await runner.poll("ws1", "tn", run.run_id)
    assert polled.status == "running" and polled.result is None

    t.job = {"status": "completed", "exit_code": 1, "stdout": "pytest out", "duration": 3.0}
    done = await runner.poll("ws1", "tn", run.run_id)
    assert done.status == "completed" and done.result is not None
    assert done.result.framework == "pytest" and done.result.failed == 1
    assert done.result.exit_code == 1 and "pytest out" in done.result.raw_tail


async def test_poll_failed_job_reports_exec_error() -> None:
    t = FakeTransport(report=None)
    runner = WorkspaceTestRunner(t)
    run = await runner.run("ws1", None, WorkspaceTestRunRequest.model_validate({"async": True}))
    t.job = {"status": "failed", "exit_code": None, "error": "orphaned: replica died"}
    done = await runner.poll("ws1", None, run.run_id)
    assert done.status == "failed" and done.result is not None
    assert done.result.error == "orphaned: replica died" and not done.result.success


async def test_poll_rejects_sync_and_malformed_ids() -> None:
    runner = WorkspaceTestRunner(FakeTransport())
    for bad in ("0" * 32, "nothex", "0" * 32 + ".bad/id"):
        with pytest.raises(HTTPException) as e:
            await runner.poll("ws1", None, bad)
        assert e.value.status_code == 404


async def test_poll_unknown_job_and_status() -> None:
    class Gone(FakeTransport):
        async def get_exec_job(self, *a):
            raise ServiceError(404, "nf")

    rid = "a" * 32 + ".job-1"
    with pytest.raises(HTTPException) as e:
        await WorkspaceTestRunner(Gone()).poll("ws1", None, rid)
    assert e.value.status_code == 404
    with pytest.raises(HTTPException) as e:
        await WorkspaceTestRunner(FakeTransport(job={"status": "weird"})).poll("ws1", None, rid)
    assert e.value.status_code == 502


async def test_poll_without_a_stored_spec_is_404() -> None:
    rid = "b" * 32 + ".job-1"
    t = FakeTransport(job={"status": "completed", "exit_code": 0})
    with pytest.raises(HTTPException) as e:
        await WorkspaceTestRunner(t).poll("ws1", None, rid)
    assert e.value.status_code == 404


async def test_poll_is_bound_to_the_workspace_and_tenant_that_started_it() -> None:
    t = FakeTransport(job={"status": "completed", "exit_code": 0})
    req = WorkspaceTestRunRequest.model_validate({"async": True})
    run = await WorkspaceTestRunner(t).run("ws1", "tn", req)
    assert (await WorkspaceTestRunner(t).poll("ws1", "tn", run.run_id)).status == "completed"
    for container, tenant in (("ws2", "tn"), ("ws1", "other"), ("ws1", None)):
        with pytest.raises(HTTPException) as e:
            await WorkspaceTestRunner(t).poll(container, tenant, run.run_id)
        assert e.value.status_code == 404


async def test_async_submit_failures() -> None:
    class Unavailable(FakeTransport):
        async def exec_async(self, *a):
            raise ServiceError(503, "no job store")

    class NoJob(FakeTransport):
        async def exec_async(self, *a):
            return {"status": "queued"}

    req = WorkspaceTestRunRequest.model_validate({"async": True})
    with pytest.raises(HTTPException) as e:
        await WorkspaceTestRunner(Unavailable()).run("ws1", None, req)
    assert e.value.status_code == 503
    with pytest.raises(HTTPException) as e:
        await WorkspaceTestRunner(NoJob()).run("ws1", None, req)
    assert e.value.status_code == 502


# --------------------------------------------------------------------- transport


class _RecordingClient:
    instances: list[_RecordingClient] = []

    def __init__(self, base_url: str, timeout: float, max_retries: int = 3) -> None:
        self.base_url, self.timeout, self.max_retries = base_url, timeout, max_retries
        self.calls: list[tuple[str, tuple, dict]] = []
        self.closed = False
        _RecordingClient.instances.append(self)

    async def exec(self, *a, **k):
        self.calls.append(("exec", a, k))
        return {"exit_code": 0}

    async def post(self, *a, **k):
        self.calls.append(("post", a, k))
        return {"job_id": "j"}

    async def get(self, *a, **k):
        self.calls.append(("get", a, k))
        return {"status": "running"}

    async def close(self) -> None:
        self.closed = True


async def test_containers_transport_calls() -> None:
    class _S:
        CONTAINERS_SERVICE_URL = "http://containers:8691"

    _RecordingClient.instances.clear()
    with (
        patch("services.workspace_exec.CodingWorkspacesClient", _RecordingClient),
        patch("services.workspace_exec.get_settings", lambda: _S()),
        patch("services.workspace_exec.service_token", lambda: None),
    ):
        tr = ContainersTransport()
        await tr.exec("ws1", "t1", ["ls"], "/workspace", 60)
        await tr.exec_async("ws1", "t1", ["cargo"], "/workspace/x", 3600)
        await tr.get_exec_job("ws1", "t1", "j")
    exec_c, async_c, get_c = _RecordingClient.instances
    assert exec_c.calls[0][2]["command"] == ["ls"] and exec_c.timeout == 70.0
    assert async_c.max_retries == 1
    assert async_c.calls[0][1][0] == "/api/workspaces/ws1/exec/async"
    assert async_c.calls[0][2]["json"] == {
        "command": ["cargo"],
        "workdir": "/workspace/x",
        "timeout": 3600,
    }
    assert get_c.calls[0][1][0] == "/api/workspaces/ws1/exec/jobs/j"
    assert all(c.closed for c in _RecordingClient.instances)


async def test_containers_transport_unconfigured() -> None:
    class _S:
        CONTAINERS_SERVICE_URL = ""

    with patch("services.workspace_exec.get_settings", lambda: _S()):
        with pytest.raises(HTTPException) as e:
            await ContainersTransport().exec("ws1", None, ["ls"], "/workspace", 5)
    assert e.value.status_code == 503


# --------------------------------------------------------------------- HTTP routes


async def _fake_validate(container_name: str, tenant_id: str | None = None):
    if container_name == "missing":
        raise HTTPException(status_code=404, detail="not found")
    return None


async def test_routes_sync_async_and_poll(app, client) -> None:
    t = FakeTransport()
    app.dependency_overrides[get_workspace_test_runner] = lambda: WorkspaceTestRunner(t)
    with patch("routers.workspace_test_runs.validate_and_get_container_for_tenant", _fake_validate):
        r = await client.post("/api/tests/workspace/ws1/runs", json={})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "completed" and body["result"]["success"] is True
        assert body["result"]["framework"] == "pytest"

        r = await client.post("/api/tests/workspace/ws1/runs")  # no body at all
        assert r.status_code == 200, r.text

        r = await client.post("/api/tests/workspace/ws1/runs", json={"async": True})
        assert r.status_code == 202, r.text
        run_id = r.json()["run_id"]

        r = await client.get(f"/api/tests/workspace/ws1/runs/{run_id}")
        assert r.status_code == 200 and r.json()["status"] == "running"

        r = await client.post("/api/tests/workspace/ws1/runs", json={"workdir": "../x"})
        assert r.status_code == 422

        r = await client.post("/api/tests/workspace/missing/runs", json={})
        assert r.status_code == 404
    app.dependency_overrides.pop(get_workspace_test_runner, None)


def test_default_runner_dependency_uses_containers_transport() -> None:
    assert isinstance(get_workspace_test_runner().transport, ContainersTransport)
