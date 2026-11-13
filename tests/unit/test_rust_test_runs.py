"""Rust workspace test runs: pack the working tree, evaluate it in a runner k8s Job.

No mocks: the in-pod programs (archive, chunked read-back, cleanup) run for real
with ``subprocess``; the archive store, the one-time download route and the poll
run against the real Postgres; the Job spec, log parsing and result mapping are
pure functions over the runner's RESULT_JSON line.

``tests/data/rust_runner/result-json.txt`` and ``result-json-passing.txt`` are REAL
lines captured verbatim from the runner image (nonagenticai/rust-test-executor#30
@ c7be4b6, ``rust-repo-run`` as uid 65532): the first over a fixture crate with
deliberate findings (1 failing test, clippy, fmt, smallvec 1.6.0 RUSTSEC-2021-0003
in audit + deny), the second over a clean crate (preset ``test`` only). The runner
contract this code targets is #30 @ e0e39e1; its RESULT_JSON shape is unchanged
since c7be4b6 (later commits add DATABASE_URL pass-through and NEXTEST_EXTRA_ARGS),
so the captures stand.

Not covered here: the API-server calls themselves (KubeApi) — they need a cluster;
the stand-ins answer with real Job / pod objects (tests/data/k8s).
"""

from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
import os
import subprocess  # nosec B404 - runs the real in-pod programs locally
import sys
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import HTTPException
from sqlalchemy import update

from config.settings import get_settings
from db_models import RustTestRunRow
from models.unified_test_models import WorkspaceTestRunRequest
from services import k8s_jobs, rust_test_runs
from services.k8s_jobs import KubeError, build_job_spec, parse_result_log
from services.rust_test_runs import (
    ARCHIVE_EXIT_TOO_LARGE,
    ARCHIVE_PY,
    CHUNK_PY,
    map_result,
    poll_rust,
    prune_runs,
    store_run,
    take_archive,
)

RUNNER_DATA = Path(__file__).resolve().parent.parent / "data" / "rust_runner"
RESULT_LINE = (RUNNER_DATA / "result-json.txt").read_text()
PASSING_LINE = (RUNNER_DATA / "result-json-passing.txt").read_text()
DOC: dict[str, Any] = parse_result_log(RESULT_LINE)
PRESETS = [p for p in ("test", "clippy", "fmt", "audit", "deny", "coverage") if p in DOC["presets"]]


def _py(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(  # nosec B603
        [sys.executable, "-I", "-c", *args], capture_output=True, check=False, timeout=60
    )


#: What cargo writes into its target dir (https://bford.info/cachedir/).
CACHEDIR_TAG = "Signature: 8a477f597d28d172789f06886806bc55\n# cargo build dir\n"


def _crate(root: Path) -> Path:
    ws = root / "workspace"
    (ws / "src").mkdir(parents=True)
    (ws / "Cargo.toml").write_text('[package]\nname = "demo"\nversion = "0.1.0"\n')
    (ws / "src" / "lib.rs").write_text("pub fn add(a: i32, b: i32) -> i32 { a + b }\n")
    (ws / "crates" / "api" / "target").mkdir(parents=True)
    (ws / "crates" / "api" / "Cargo.toml").write_text('[package]\nname = "api"\n')
    (ws / "crates" / "api" / "target" / "big").write_bytes(b"x" * 1000)
    (ws / "crates" / "api" / "target" / "CACHEDIR.TAG").write_text(CACHEDIR_TAG)
    (ws / "target" / "debug").mkdir(parents=True)
    (ws / "target" / "debug" / "demo").write_bytes(b"\x7fELF")
    (ws / "target" / "CACHEDIR.TAG").write_text(CACHEDIR_TAG)
    # a source module merely NAMED target (no CACHEDIR.TAG) is part of the crate
    (ws / "src" / "target").mkdir()
    (ws / "src" / "target" / "mod.rs").write_text("pub fn triple() {}\n")
    (ws / ".git").mkdir()
    (ws / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (ws / "assets.bin").write_bytes(os.urandom(700 * 1024))  # > one 512 KiB chunk
    (ws / "link").symlink_to("/etc/hosts")
    return ws


def _members(data: bytes) -> dict[str, tarfile.TarInfo]:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        return {m.name: m for m in tf.getmembers()}


# ----------------------------------------------------------------- in-pod programs


def test_archive_leaves_out_target_and_git_and_keeps_symlinks(tmp_path: Path) -> None:
    ws = _crate(tmp_path)
    out = tmp_path / "run" / "src.tgz"
    done = _py(ARCHIVE_PY, str(ws), str(out), str(50 * 1024 * 1024))
    assert done.returncode == 0, done.stderr
    data = out.read_bytes()
    assert int(done.stdout) == len(data)
    names = _members(data)
    assert {"Cargo.toml", "src/lib.rs", "crates/api/Cargo.toml", "assets.bin"} <= set(names)
    assert "src/target/mod.rs" in names  # named target, but not a cache dir
    assert not any(n == "target" or n.startswith(("target/", "crates/api/target", ".git"))
                   for n in names)  # fmt: skip
    assert names["link"].issym() and names["link"].linkname == "/etc/hosts"


def test_archive_leaves_out_an_untagged_cargo_target_dir(tmp_path: Path) -> None:
    """A cargo target dir WITHOUT CACHEDIR.TAG (measured live: /workspace/target, 2.6 GB,
    no tag) is left out when its parent holds a Cargo.toml, at any depth; a dir named
    target with no sibling Cargo.toml is data and is kept, and so is .repos/."""
    ws = tmp_path / "ws"
    (ws / "src" / "target").mkdir(parents=True)
    (ws / "Cargo.toml").write_text('[workspace]\nmembers = ["crates/api"]\n')
    (ws / "src" / "lib.rs").write_text("pub fn f() {}\n")
    (ws / "src" / "target" / "data.txt").write_text("kept\n")
    (ws / "target" / "debug").mkdir(parents=True)
    (ws / "target" / "debug" / "big").write_bytes(os.urandom(2 * 1024 * 1024))
    (ws / "crates" / "api" / "target" / "debug").mkdir(parents=True)
    (ws / "crates" / "api" / "Cargo.toml").write_text('[package]\nname = "api"\n')
    (ws / "crates" / "api" / "target" / "debug" / "big").write_bytes(os.urandom(1024 * 1024))
    (ws / ".repos" / "dep" / "target").mkdir(parents=True)
    (ws / ".repos" / "dep" / "target" / "keep.txt").write_text("no Cargo.toml here\n")
    cap = 1024 * 1024  # under the incompressible 3 MiB of build output
    out = tmp_path / "run" / "src.tgz"
    done = _py(ARCHIVE_PY, str(ws), str(out), str(cap))
    assert done.returncode == 0, done.stderr
    names = _members(out.read_bytes())
    assert out.stat().st_size < cap
    assert {"Cargo.toml", "src/lib.rs", "crates/api/Cargo.toml", "src/target/data.txt",
            ".repos/dep/target/keep.txt"} <= set(names)  # fmt: skip
    assert not any(n == "target" or n.startswith(("target/", "crates/api/target"))
                   for n in names)  # fmt: skip


def test_archive_over_the_cap_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "run" / "src.tgz"
    done = _py(ARCHIVE_PY, str(_crate(tmp_path)), str(out), "1000")
    assert done.returncode == ARCHIVE_EXIT_TOO_LARGE and b"over 1000" in done.stderr
    assert not out.exists()


def test_chunk_reader_round_trips(tmp_path: Path) -> None:
    blob = os.urandom(3000)
    (tmp_path / "f").write_bytes(blob)
    parts = [_py(CHUNK_PY, str(tmp_path / "f"), str(o), "1024").stdout for o in (0, 1024, 2048)]
    assert b"".join(base64.b64decode(p) for p in parts) == blob


class LocalWorkspace:
    """Runs the exec argv coding-containers would run in the pod, here, for real;
    ``/workspace`` and ``/tmp/coding-testing`` map to tmp dirs."""

    def __init__(self, ws: Path, runs: Path):
        self.map = {"/workspace": str(ws), rust_test_runs.RUN_ROOT: str(runs)}

    def _path(self, a: str) -> str:
        for prefix, real in self.map.items():
            if a == prefix or a.startswith(prefix + "/"):
                return real + a[len(prefix) :]
        return a

    async def exec(self, container, tenant_id, argv, workdir, timeout, what) -> dict[str, Any]:
        argv = [sys.executable if a == "python3" else self._path(a) for a in argv]
        done = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=timeout)  # nosec B603
        return {"exit_code": done.returncode, "stdout": done.stdout, "stderr": done.stderr}


async def test_pack_reads_the_archive_back_in_chunks_and_cleans_up(tmp_path: Path) -> None:
    ws = _crate(tmp_path)
    (tmp_path / "runs").mkdir()
    pod = LocalWorkspace(ws, tmp_path / "runs")
    data = await rust_test_runs.pack(pod.exec, "ws1", None, f"{rust_test_runs.RUN_ROOT}/n1")
    assert len(data) > rust_test_runs.CHUNK_BYTES  # really took more than one chunk
    assert "assets.bin" in _members(data)
    assert not (tmp_path / "runs" / "n1").exists()


async def test_pack_over_the_cap_is_413(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "runs").mkdir()
    pod = LocalWorkspace(_crate(tmp_path), tmp_path / "runs")
    monkeypatch.setattr(rust_test_runs, "ARCHIVE_MAX_BYTES", 1000)
    with pytest.raises(HTTPException) as e:
        await rust_test_runs.pack(pod.exec, "ws1", None, f"{rust_test_runs.RUN_ROOT}/n2")
    assert e.value.status_code == 413


# ----------------------------------------------------------------- RESULT_JSON -> result


def _clean() -> dict[str, Any]:
    doc = copy.deepcopy(DOC)
    doc["errors"] = []
    for name, body in doc["presets"].items():
        if name == "test":
            body.update(failed=0, passed=max(1, body.get("passed", 0)))
        for key in rust_test_runs._FINDING_LISTS.get(name, ()):
            body[key] = []
    return doc


def test_parse_result_log_takes_the_last_result_line() -> None:
    log = 'Compiling demo\nRESULT_JSON:{"a": 1}\n' + RESULT_LINE + "trailing output\n"
    assert parse_result_log(log) == DOC
    for bad, needle in (("compiling...\n", "no RESULT_JSON"), ("RESULT_JSON:{nope\n", "not JSON")):
        with pytest.raises(KubeError, match=needle):
            parse_result_log(bad)


def test_map_the_runner_line() -> None:
    r = map_result(DOC, PRESETS, "/workspace")
    test = DOC["presets"]["test"]
    assert (r.passed, r.failed, r.skipped, r.total) == (
        test["passed"], test["failed"], test["skipped"], test["total"]
    )  # fmt: skip
    assert r.framework == "cargo-nextest" and r.language == "rust"
    assert r.presets == DOC["presets"] and r.tool_versions == DOC["tool_versions"]
    assert r.duration == DOC["duration_s"] and len(r.cases) == len(test.get("cases", []))


def test_real_capture_with_findings_is_not_a_success() -> None:
    r = map_result(DOC, PRESETS, "/workspace")
    assert not r.success
    for needle in ("test: 1 failed", "clippy: 1 findings", "fmt: 1 files", "audit: 1 advisories",
                   "deny: 1 advisories"):  # fmt: skip
        assert needle in (r.error or ""), r.error


def test_real_passing_capture_is_a_success() -> None:
    doc = parse_result_log(PASSING_LINE)
    r = map_result(doc, ["test"], "/workspace")
    assert r.success and r.error is None and r.passed == doc["presets"]["test"]["passed"] >= 1


def test_real_captures_report_the_test_presets_exit_code() -> None:
    """exit_code is the runner's test-preset exit, read from the REAL captures: the
    crate with 1 failing of 2 tests exits 100 (cargo-nextest), the clean crate 0."""
    failing = map_result(DOC, PRESETS, "/workspace")
    assert DOC["presets"]["test"]["exit_code"] == 100
    assert failing.exit_code == 100 and not failing.success
    doc = parse_result_log(PASSING_LINE)
    passing = map_result(doc, ["test"], "/workspace")
    assert passing.exit_code == 0 and passing.success
    # the runner document has no output tail: raw_tail is not invented
    assert failing.raw_tail == "" and passing.raw_tail == ""


def test_exit_code_is_none_without_a_test_exit() -> None:
    assert rust_test_runs.failed_result(["test"], "/workspace", "Job lost").exit_code is None
    no_test = map_result(DOC, ["clippy"], "/workspace")  # test preset not requested
    assert no_test.exit_code is None
    doc = parse_result_log(PASSING_LINE)
    doc["presets"]["test"].pop("exit_code")
    assert map_result(doc, ["test"], "/workspace").exit_code is None


def test_map_clean_run_is_a_success() -> None:
    r = map_result(_clean(), PRESETS, "/workspace")
    assert r.success and r.error is None


@pytest.mark.parametrize(
    ("change", "needle"),
    [
        (lambda d: d["presets"]["test"].update(failed=1), "1 failed"),
        (lambda d: d["presets"]["test"].update(passed=0, failed=0), "no tests ran"),
        (lambda d: d["presets"]["clippy"]["findings"].append({"code": "x"}), "clippy: 1"),
        (lambda d: d["presets"]["audit"]["advisories"].append({"id": "R"}), "audit: 1"),
        (lambda d: d["presets"].pop("fmt"), "fmt: not reported"),
        (lambda d: d["errors"].append("cargo exploded"), "cargo exploded"),
    ],
)
def test_map_anything_not_clean_is_not_a_success(change, needle: str) -> None:
    doc = _clean()
    change(doc)
    r = map_result(doc, ["test", "clippy", "fmt", "audit"], "/workspace")
    assert not r.success and needle in (r.error or "")


# ----------------------------------------------------------------- the Job


def test_job_spec() -> None:
    spec = build_job_spec(
        run_id="a" * 32, namespace="coding-test-runs", image="reg/rte:1",
        archive_url="http://t/internal/test-runs/x/archive", archive_token="tok",
        presets=["test", "clippy"], workdir="crates/api", timeout_s=600,
        resources={"requests": {"cpu": "1"}, "limits": {"cpu": "4"}},
    )  # fmt: skip
    assert spec["metadata"] == {
        "name": "rust-run-" + "a" * 32,
        "namespace": "coding-test-runs",
        "labels": spec["spec"]["template"]["metadata"]["labels"],
    }
    js = spec["spec"]
    assert js["backoffLimit"] == 0 and js["activeDeadlineSeconds"] == 900  # 600 + 300 slack
    # The finished Job outlives the run's poll deadline (created + timeout + GRACE_S)
    # even when it finishes the instant it is created.
    assert js["ttlSecondsAfterFinished"] > 600 + rust_test_runs.GRACE_S
    pod = js["template"]["spec"]
    assert pod["automountServiceAccountToken"] is False and pod["restartPolicy"] == "Never"
    assert pod["securityContext"]["runAsNonRoot"] is True
    # the CI nodes: their label + their taint (gitops services/arc-runners)
    assert pod["nodeSelector"] == {"node-role/ci-runner": "true"}
    assert pod["tolerations"] == [
        {"key": "ci-runner", "operator": "Equal", "value": "true", "effect": "NoSchedule"}
    ]
    assert "affinity" not in pod and "initContainers" not in pod and "volumes" not in pod
    (c,) = pod["containers"]
    assert c["command"] == ["rust-repo-run"]
    assert {e["name"]: e["value"] for e in c["env"]} == {
        "ARCHIVE_URL": "http://t/internal/test-runs/x/archive",
        "ARCHIVE_TOKEN": "tok",
        "PRESETS": "test,clippy",
        "WORKDIR": "crates/api",
        "TIMEOUT_S": "600",
    }


def test_job_spec_passes_extra_args_as_nextest_extra_args() -> None:
    """rust extra_args reach cargo nextest via the runner's NEXTEST_EXTRA_ARGS (a JSON
    array of strings, rust-test-executor#30 @ 611b976); none -> the env is absent."""
    args = ["-E", "test(adds)", "--no-capture"]
    kw: dict[str, Any] = dict(
        run_id="a" * 32, namespace="ns", image="reg/rte:1", archive_url="http://t/a",
        archive_token="tok", presets=["test"], workdir="", timeout_s=60, resources={},
    )  # fmt: skip
    (c,) = build_job_spec(**kw, nextest_extra_args=args)["spec"]["template"]["spec"]["containers"]
    env = {e["name"]: e["value"] for e in c["env"]}
    assert json.loads(env["NEXTEST_EXTRA_ARGS"]) == args
    (c,) = build_job_spec(**kw)["spec"]["template"]["spec"]["containers"]
    assert "NEXTEST_EXTRA_ARGS" not in {e["name"] for e in c["env"]}


def test_test_preset_always_runs_and_zero_tests_is_never_success() -> None:
    """Real captures: the passing line is a success only because a test passed."""
    assert WorkspaceTestRunRequest(presets=["clippy"]).rust_presets() == ["test", "clippy"]
    assert WorkspaceTestRunRequest().rust_presets() == ["test"]
    doc = parse_result_log(PASSING_LINE)
    presets = WorkspaceTestRunRequest().rust_presets()
    assert map_result(doc, presets, "/workspace").success
    no_tests = copy.deepcopy(doc)
    no_tests["presets"]["test"].update(passed=0, total=0)
    r = map_result(no_tests, presets, "/workspace")
    assert not r.success and "no tests ran" in (r.error or "")
    no_test_preset = copy.deepcopy(doc)
    del no_test_preset["presets"]["test"]
    r = map_result(no_test_preset, presets, "/workspace")
    assert not r.success and "test: not reported" in (r.error or "")


def test_job_spec_with_postgres_sidecar() -> None:
    """with_postgres: a native postgres sidecar the runner waits for, restricted-PSA shaped.

    Proven here: the spec. NOT provable without the cluster: that the kubelet holds the
    runner until the startupProbe passes, that PSA ``restricted`` admits it, that the
    namespace quota / LimitRange admit two containers, and that the image pulls."""
    spec = build_job_spec(
        run_id="a" * 32, namespace="ns", image="reg/rte:1", archive_url="http://t/a",
        archive_token="tok", presets=["test"], workdir="", timeout_s=60,
        resources={}, postgres_image="postgres:18",
    )  # fmt: skip
    pod = spec["spec"]["template"]["spec"]
    (pg,) = pod["initContainers"]
    assert pg["restartPolicy"] == "Always"  # native sidecar: runs alongside, ends with the Job
    assert pg["image"] == "postgres:18"
    assert pg["args"] == ["-c", "listen_addresses=127.0.0.1"]
    env = {e["name"]: e["value"] for e in pg["env"]}
    assert env["POSTGRES_HOST_AUTH_METHOD"] == "trust"
    assert env["PGDATA"].startswith(pg["volumeMounts"][0]["mountPath"] + "/")
    assert pod["volumes"] == [{"name": "pgdata", "emptyDir": {"sizeLimit": "2Gi"}}]
    probe = pg["startupProbe"]["exec"]["command"]
    assert probe[:3] == ["pg_isready", "-h", "127.0.0.1"]
    sc = pg["securityContext"]
    assert sc["runAsUser"] == 999 and sc["runAsNonRoot"] is True
    assert sc["allowPrivilegeEscalation"] is False and sc["capabilities"] == {"drop": ["ALL"]}
    assert pod["securityContext"]["seccompProfile"] == {"type": "RuntimeDefault"}
    assert set(pg["resources"]["limits"]) == {"cpu", "memory", "ephemeral-storage"}
    (runner,) = pod["containers"]
    renv = {e["name"]: e["value"] for e in runner["env"]}
    assert renv["DATABASE_URL"] == "postgres://postgres@127.0.0.1:5432/postgres"


def test_runner_error_with_empty_presets_is_not_completed() -> None:
    """rust-repo-run exit 2 (bad input): errors, no presets -> failed, error surfaced."""
    r = map_result({"presets": {}, "errors": ["WORKDIR does not exist"]}, ["test"], "/workspace")
    assert rust_test_runs.run_status(r) == "failed" and "WORKDIR does not exist" in (r.error or "")
    assert rust_test_runs.run_status(map_result(DOC, PRESETS, "/workspace")) == "completed"


K8S_DATA = Path(__file__).resolve().parent.parent / "data" / "k8s"
#: Real objects read from the cluster (tests/data/k8s/README.md).
REAL_FAILED_JOB: dict[str, Any] = json.loads((K8S_DATA / "job-deadline-exceeded.json").read_text())
REAL_ERROR_POD: dict[str, Any] = json.loads((K8S_DATA / "pod-container-error.json").read_text())


def test_job_spec_from_request_fields(monkeypatch) -> None:
    """Request -> Job: extra_args, with_postgres and a sub-crate workdir all reach the Job."""
    monkeypatch.setenv("RUST_RUNNER_IMAGE", "reg/rte:1")
    monkeypatch.setenv("TESTING_INTERNAL_URL", "http://ct:8689/")
    get_settings.cache_clear()
    spec = rust_test_runs.job_spec(
        get_settings(), "b" * 32, "tok", "/workspace/crates/api", ["test"], 14400,
        True, ["-E", "test(adds)"],
    )  # fmt: skip
    assert spec["spec"]["activeDeadlineSeconds"] == 14400 + k8s_jobs.JOB_DEADLINE_SLACK_S
    pod = spec["spec"]["template"]["spec"]
    env = {e["name"]: e["value"] for e in pod["containers"][0]["env"]}
    assert env["WORKDIR"] == "crates/api"
    assert json.loads(env["NEXTEST_EXTRA_ARGS"]) == ["-E", "test(adds)"]
    assert env["DATABASE_URL"] == k8s_jobs.PG_URL
    assert env["ARCHIVE_URL"] == f"http://ct:8689/internal/test-runs/{'b' * 32}/archive"
    assert pod["initContainers"][0]["image"] == get_settings().RUNNER_POSTGRES_IMAGE
    plain = rust_test_runs.job_spec(
        get_settings(), "b" * 32, "tok", "/workspace", ["test"], 60, False, []
    )["spec"]["template"]["spec"]
    penv = {e["name"]: e["value"] for e in plain["containers"][0]["env"]}
    assert penv["WORKDIR"] == "" and "DATABASE_URL" not in penv
    assert "NEXTEST_EXTRA_ARGS" not in penv and "initContainers" not in plain


def test_real_pod_and_job_failure_reasons() -> None:
    assert k8s_jobs.pod_failure_reason(REAL_ERROR_POD) == "manager: Error (exit 1)"
    assert k8s_jobs.job_failure_reason(REAL_FAILED_JOB) == (
        "DeadlineExceeded: Job was active longer than specified deadline"
    )
    assert k8s_jobs.job_outcome(REAL_FAILED_JOB) == "failed"


class RecordedKube:
    """Stand-in API server answering with the REAL objects: the Job runs (no status
    yet) while its pod exists, then fails DeadlineExceeded after the controller has
    deleted the pod, exactly as the real Job in the fixture did."""

    def __init__(self) -> None:
        self.job_reads = 0
        self.deleted: list[str] = []

    async def create_job(self, namespace: str, spec: dict[str, Any]) -> None:
        self.name = spec["metadata"]["name"]

    async def get_job(self, namespace: str, name: str) -> dict[str, Any]:
        self.job_reads += 1
        if self.job_reads == 1:
            return {**REAL_FAILED_JOB, "status": {"active": 1}}
        return REAL_FAILED_JOB

    async def job_pods(self, namespace: str, name: str) -> list[dict[str, Any]]:
        return [REAL_ERROR_POD] if self.job_reads == 1 else []

    async def pod_log(self, namespace: str, pod: str) -> str:
        # what KubeApi raises for the API server's answer once the pod is deleted
        raise KubeError(f'GET /api/v1/namespaces/{namespace}/pods/{pod}/log answered 404: '
                        f'pods "{pod}" not found')  # fmt: skip

    async def delete_job(self, namespace: str, name: str) -> None:
        self.deleted.append(name)


async def test_run_job_reports_the_last_pod_seen(monkeypatch) -> None:
    monkeypatch.setattr(k8s_jobs, "POLL_INTERVAL_S", 0.0)
    kube = RecordedKube()
    spec = build_job_spec(
        run_id="c" * 32, namespace="ns", image="reg/rte:1", archive_url="http://t/a",
        archive_token="tok", presets=["test"], workdir="", timeout_s=60, resources={},
    )  # fmt: skip
    with pytest.raises(KubeError) as e:
        await k8s_jobs.run_job(kube, spec, 60)  # type: ignore[arg-type]
    msg = str(e.value)
    assert "DeadlineExceeded" in msg and "manager: Error (exit 1)" in msg
    assert kube.deleted == []  # the caller deletes it, once the result is stored


class ExitTwoKube(RecordedKube):
    """The REAL failed Job, its pod still present, whose runner exited 2 (runner error)
    after printing a RESULT_JSON line with errors and no presets (README @ e0e39e1)."""

    async def job_pods(self, namespace: str, name: str) -> list[dict[str, Any]]:
        return [REAL_ERROR_POD]

    async def pod_log(self, namespace: str, pod: str) -> str:
        return 'RESULT_JSON:{"presets":{},"errors":["test: invalid NEXTEST_EXTRA_ARGS"]}\n'


async def test_failed_job_with_a_result_line_is_a_failed_run(monkeypatch) -> None:
    monkeypatch.setattr(k8s_jobs, "POLL_INTERVAL_S", 0.0)
    spec = build_job_spec(
        run_id="d" * 32, namespace="ns", image="reg/rte:1", archive_url="http://t/a",
        archive_token="tok", presets=["test"], workdir="", timeout_s=60, resources={},
    )  # fmt: skip
    with pytest.raises(KubeError) as e:
        await k8s_jobs.run_job(ExitTwoKube(), spec, 60)  # type: ignore[arg-type]
    assert "DeadlineExceeded" in str(e.value)
    assert "test: invalid NEXTEST_EXTRA_ARGS" in str(e.value)  # runner errors surfaced


async def test_async_rust_run_accepts_the_model_maximum_timeout(pg_db, tmp_path) -> None:
    """main accepts async runs up to timeout 14400; so does a rust run (no 1800 cap)."""
    (tmp_path / "runs").mkdir()
    pod = LocalWorkspace(_crate(tmp_path), tmp_path / "runs")
    req = WorkspaceTestRunRequest.model_validate({"async": True, "timeout": 14400})
    run = await rust_test_runs.run_rust(pod.exec, "ws1", None, req, "/workspace")
    assert run.status == "running" and run.run_id.endswith(".rte")
    await asyncio.gather(*list(rust_test_runs._TASKS.values()))
    done = await poll_rust("ws1", None, run.run_id)
    assert done.status == "failed" and "not configured" in (done.result.error or "")


def test_settings_carry_the_gitops_env_names(monkeypatch) -> None:
    """gitops services/coding-testing sets exactly these names (configmap + deployment)."""
    env = {
        "RUST_RUNNER_IMAGE": "reg/rte@sha256:1",
        "TEST_RUNS_NAMESPACE": "ns-x",
        "TESTING_INTERNAL_URL": "http://ct:8689",
        "RUNNER_CPU_REQUEST": "2",
        "RUNNER_CPU_LIMIT": "3",
        "RUNNER_MEMORY_REQUEST": "1Gi",
        "RUNNER_MEMORY_LIMIT": "5Gi",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    s = get_settings()
    assert {k: getattr(s, k) for k in env} == env


# ----------------------------------------------------------------- store + routes (real PG)


async def test_archive_route_is_one_time_and_token_bound(pg_db, client) -> None:
    await store_run("b" * 32, "ws1", None, "t0k3n", b"\x1f\x8bdata", 60)
    url = f"/internal/test-runs/{'b' * 32}/archive"
    for headers in ({}, {"X-Run-Token": "wrong"}):
        assert (await client.get(url, headers=headers)).status_code == 404
    r = await client.get(url, headers={"X-Run-Token": "t0k3n"})
    assert r.status_code == 200 and r.content == b"\x1f\x8bdata"
    assert r.headers["content-type"] == "application/gzip"
    assert (await client.get(url, headers={"X-Run-Token": "t0k3n"})).status_code == 404


async def test_archive_route_accepts_the_header_the_runner_sends(pg_db, client) -> None:
    """rust-test-executor's rust-repo-run (src/rust_repo_run.js @ 611b976) sends the token
    as lowercase ``x-run-token``; HTTP header names are case-insensitive."""
    await store_run("9" * 32, "ws1", None, "runner-tok", b"\x1f\x8btree", 60)
    url = f"/internal/test-runs/{'9' * 32}/archive"
    assert (await client.get(url)).status_code == 404
    r = await client.get(url, headers={"x-run-token": "runner-tok"})
    assert r.status_code == 200 and r.content == b"\x1f\x8btree"


async def test_archive_route_is_not_in_the_openapi(client) -> None:
    spec = (await client.get("/openapi.json")).json()
    assert not any(p.startswith("/internal") for p in spec["paths"])


async def test_auth_middleware_lets_only_the_internal_prefix_through(monkeypatch) -> None:
    from fastapi import FastAPI

    from middleware import auth

    app = FastAPI()
    app.add_middleware(auth.AuthMiddleware)  # type: ignore[invalid-argument-type]

    @app.get("/internal/test-runs/{run_id}/archive")
    async def _archive(run_id: str) -> dict:
        return {"ok": run_id}

    @app.get("/api/tests/x")
    async def _user() -> dict:
        return {"ok": True}

    monkeypatch.setattr(auth, "ENABLE_AUTH", True)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        assert (await c.get("/internal/test-runs/abc/archive")).status_code == 200
        assert (await c.get("/api/tests/x")).status_code == 401
        assert (await c.get("/internal/other")).status_code == 401


async def test_poll_reads_the_stored_result_bound_to_workspace_and_tenant(pg_db) -> None:
    nonce = "c" * 32
    await store_run(nonce, "ws1", "tn", "tok", b"x", 60)
    run_id = f"{nonce}.rte"
    assert (await poll_rust("ws1", "tn", run_id)).status == "running"
    await rust_test_runs.finish_run(nonce, map_result(DOC, PRESETS, "/workspace"))
    assert await take_archive(nonce, "tok") is None  # dropped with the result
    done = await poll_rust("ws1", "tn", run_id)
    assert done.status == "completed" and done.result is not None
    assert done.result.passed == DOC["presets"]["test"]["passed"]
    for container, tenant in (("ws2", "tn"), ("ws1", "other"), ("ws1", None)):
        with pytest.raises(HTTPException) as e:
            await poll_rust(container, tenant, run_id)
        assert e.value.status_code == 404


async def test_stored_deadline_outlasts_the_job(pg_db) -> None:
    """A poll must not call a run lost while its Job may still legitimately run."""
    before = datetime.now(UTC)
    await store_run("8" * 32, "ws1", None, "tok", b"x", 600)
    async with pg_db() as session:
        row = await session.get(RustTestRunRow, "8" * 32)
    assert row is not None
    assert row.deadline >= before + timedelta(seconds=600 + k8s_jobs.JOB_DEADLINE_SLACK_S)


async def test_lost_run_is_failed_after_its_deadline(pg_db) -> None:
    nonce = "d" * 32
    await store_run(nonce, "ws1", None, "tok", b"x", 60)
    async with pg_db() as session:
        await session.execute(
            update(RustTestRunRow)
            .where(RustTestRunRow.run_id == nonce)
            .values(deadline=datetime.now(UTC) - timedelta(seconds=1))
        )
        await session.commit()
    lost = await poll_rust("ws1", None, f"{nonce}.rte")
    assert lost.status == "failed" and "without a result" in (lost.result.error or "")


async def test_prune_runs(pg_db) -> None:
    await store_run("e" * 32, "ws1", None, "tok", b"x", 60)
    async with pg_db() as session:
        await session.execute(
            update(RustTestRunRow).values(created_at=datetime.now(UTC) - timedelta(days=100))
        )
        await session.commit()
    await store_run("f" * 32, "ws1", None, "tok", b"x", 60)
    await store_run("0" * 32, "ws1", None, "tok", b"lost", 60)
    async with pg_db() as session:  # a lost run: past its deadline, archive never taken
        await session.execute(
            update(RustTestRunRow)
            .where(RustTestRunRow.run_id == "0" * 32)
            .values(deadline=datetime.now(UTC) - timedelta(seconds=1))
        )
        await session.commit()
    assert await prune_runs(0) == 0
    async with pg_db() as session:
        lost = await session.get(RustTestRunRow, "0" * 32)
        live = await session.get(RustTestRunRow, "f" * 32)
    assert lost is not None and lost.archive is None  # 50 MiB not kept for 90 days
    assert live is not None and live.archive == b"x"  # within its deadline: untouched
    assert await prune_runs(90) == 1


# ----------------------------------------------------------------- run end to end (no cluster)


async def test_rust_run_without_a_cluster_is_a_failed_run(pg_db, tmp_path, monkeypatch) -> None:
    """Real pack + store; KUBERNETES_SERVICE_HOST is unset here, so the Job cannot be
    created: the run is failed with that reason and the archive is dropped."""
    monkeypatch.setenv("RUST_RUNNER_IMAGE", "reg/rte:1")
    monkeypatch.setenv("TESTING_INTERNAL_URL", "http://ct:8689")
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    get_settings.cache_clear()
    (tmp_path / "runs").mkdir()
    pod = LocalWorkspace(_crate(tmp_path), tmp_path / "runs")
    req = WorkspaceTestRunRequest(presets=["test", "clippy"])
    run = await rust_test_runs.run_rust(pod.exec, "ws1", None, req, "/workspace")
    assert run.status == "failed" and run.result is not None
    assert "not running in a cluster" in (run.result.error or "")
    async with pg_db() as session:
        row = await session.get(RustTestRunRow, run.run_id)
    assert row is not None and row.archive is None and row.result is not None


async def test_unconfigured_runner_is_a_failed_run(pg_db, tmp_path) -> None:
    (tmp_path / "runs").mkdir()
    pod = LocalWorkspace(_crate(tmp_path), tmp_path / "runs")
    run = await rust_test_runs.run_rust(
        pod.exec, "ws1", None, WorkspaceTestRunRequest(), "/workspace"
    )
    assert run.status == "failed" and "not configured" in (run.result.error or "")


async def test_with_postgres_rust_run_is_accepted(pg_db, tmp_path) -> None:
    """main accepted with_postgres for rust; it still does (no cluster here: failed run
    with with_postgres recorded on the result, not a 422)."""
    (tmp_path / "runs").mkdir()
    pod = LocalWorkspace(_crate(tmp_path), tmp_path / "runs")
    req = WorkspaceTestRunRequest(with_postgres=True)
    run = await rust_test_runs.run_rust(pod.exec, "ws1", None, req, "/workspace")
    assert run.result is not None and run.result.with_postgres is True


# ----------------------------------------------------------------- poll after a restart (#88)


class ClusterJobs:
    """Stand-in API server holding Jobs by name. A missing Job is the 404 KubeApi raises;
    the pod is the REAL one, its log the REAL runner line."""

    def __init__(self, log: str = RESULT_LINE) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.log = log
        self.reads = 0
        self.deleted: list[str] = []

    async def create_job(self, namespace: str, spec: dict[str, Any]) -> None:
        self.jobs[spec["metadata"]["name"]] = {
            **spec, "status": {"conditions": [{"type": "Complete", "status": "True"}]}
        }  # fmt: skip

    async def get_job(self, namespace: str, name: str) -> dict[str, Any]:
        self.reads += 1
        if name not in self.jobs:
            raise KubeError(
                f"GET /apis/batch/v1/namespaces/{namespace}/jobs/{name} answered 404: "
                f'jobs.batch "{name}" not found',
                status=404,
            )
        return self.jobs[name]

    async def job_pods(self, namespace: str, name: str) -> list[dict[str, Any]]:
        return [REAL_ERROR_POD] if name in self.jobs else []

    async def pod_log(self, namespace: str, pod: str) -> str:
        return self.log

    async def delete_job(self, namespace: str, name: str) -> None:
        self.deleted.append(name)
        self.jobs.pop(name, None)


COMPLETE = {"conditions": [{"type": "Complete", "status": "True"}]}


async def _orphan(monkeypatch, nonce: str, status: dict[str, Any] | None) -> ClusterJobs:
    """A stored async run whose task is gone (replica restarted); its Job in ``status``
    (None: no Job)."""
    kube = ClusterJobs()
    monkeypatch.setattr(k8s_jobs, "KubeApi", lambda: kube)
    await store_run(nonce, "ws1", "tn", "tok", b"x", 14400)
    if status is not None:
        spec = rust_test_runs.job_spec(
            get_settings(), nonce, "tok", "/workspace/crates/api", PRESETS, 14400, True, None
        )
        kube.jobs[spec["metadata"]["name"]] = {**spec, "status": status}
    return kube


async def test_poll_after_a_restart_stores_the_finished_jobs_result(pg_db, monkeypatch) -> None:
    nonce = "1" * 32
    kube = await _orphan(monkeypatch, nonce, COMPLETE)
    run = await poll_rust("ws1", "tn", f"{nonce}.rte")
    assert run.status == "completed" and run.result is not None
    want = map_result(DOC, PRESETS, "/workspace/crates/api")
    assert run.result.model_dump() == want.model_copy(update={"with_postgres": True}).model_dump()
    assert kube.deleted == [k8s_jobs.job_name(nonce)]
    reads = kube.reads
    again = await poll_rust("ws1", "tn", f"{nonce}.rte")  # stored: no API call
    assert again.result == run.result and kube.reads == reads
    assert await take_archive(nonce, "tok") is None


async def test_poll_after_a_restart_fails_a_failed_job(pg_db, monkeypatch) -> None:
    nonce = "2" * 32
    await _orphan(monkeypatch, nonce, REAL_FAILED_JOB["status"])
    run = await poll_rust("ws1", "tn", f"{nonce}.rte")
    assert run.status == "failed" and run.result is not None
    assert "DeadlineExceeded" in (run.result.error or "")
    assert "manager: Error (exit 1)" in (run.result.error or "")
    async with pg_db() as session:
        assert (await session.get(RustTestRunRow, nonce)).result is not None


async def test_poll_of_a_running_job_stays_running(pg_db, monkeypatch) -> None:
    nonce = "3" * 32
    kube = await _orphan(monkeypatch, nonce, {"active": 1})
    assert (await poll_rust("ws1", "tn", f"{nonce}.rte")).status == "running"
    assert kube.deleted == []
    async with pg_db() as session:
        assert (await session.get(RustTestRunRow, nonce)).result is None


async def test_poll_fails_a_run_whose_job_is_gone_before_its_deadline(pg_db, monkeypatch) -> None:
    nonce = "4" * 32
    await _orphan(monkeypatch, nonce, None)
    # just started: the run's task may not have created the Job yet
    assert (await poll_rust("ws1", "tn", f"{nonce}.rte")).status == "running"
    async with pg_db() as session:
        await session.execute(
            update(RustTestRunRow)
            .where(RustTestRunRow.run_id == nonce)
            .values(created_at=datetime.now(UTC) - timedelta(minutes=5))
        )
        await session.commit()
    run = await poll_rust("ws1", "tn", f"{nonce}.rte")  # deadline is ~4h away
    assert run.status == "failed" and "is gone without a result" in (run.result.error or "")


async def test_poll_leaves_a_run_owned_by_this_process_to_its_task(pg_db, monkeypatch) -> None:
    nonce = "5" * 32
    kube = await _orphan(monkeypatch, nonce, COMPLETE)
    monkeypatch.setitem(rust_test_runs._TASKS, nonce, asyncio.get_running_loop().create_future())
    assert (await poll_rust("ws1", "tn", f"{nonce}.rte")).status == "running"
    assert kube.reads == 0


async def test_two_replicas_finishing_one_run_store_one_result(pg_db, monkeypatch) -> None:
    nonce = "6" * 32
    await _orphan(monkeypatch, nonce, COMPLETE)
    a, b = await asyncio.gather(*(poll_rust("ws1", "tn", f"{nonce}.rte") for _ in range(2)))
    assert a.result == b.result and a.status == "completed"
    late = rust_test_runs.failed_result([], "/workspace", "the run's own task, too late")
    assert await rust_test_runs.finish_run(nonce, late) is False
    assert (await poll_rust("ws1", "tn", f"{nonce}.rte")).result == a.result


async def test_rust_run_deletes_its_job_after_storing_the_result(
    pg_db, tmp_path, monkeypatch
) -> None:
    """The run's own path: result stored first, Job deleted after (a gone Job then
    always means a lost run)."""
    monkeypatch.setenv("RUST_RUNNER_IMAGE", "reg/rte:1")
    monkeypatch.setenv("TESTING_INTERNAL_URL", "http://ct:8689")
    get_settings.cache_clear()
    kube = ClusterJobs(log=PASSING_LINE)
    monkeypatch.setattr(k8s_jobs, "KubeApi", lambda: kube)
    (tmp_path / "runs").mkdir()
    pod = LocalWorkspace(_crate(tmp_path), tmp_path / "runs")
    run = await rust_test_runs.run_rust(
        pod.exec, "ws1", None, WorkspaceTestRunRequest(), "/workspace"
    )
    assert run.status == "completed" and run.result is not None and run.result.success
    assert kube.deleted == [k8s_jobs.job_name(run.run_id)] and kube.jobs == {}
    async with pg_db() as session:
        row = await session.get(RustTestRunRow, run.run_id)
    assert row is not None and row.result is not None
