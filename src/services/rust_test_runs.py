"""Rust workspace test runs: one throwaway k8s Job per run, never the tenant workspace.

A Rust test run:

1. packs the workspace's working tree (``/workspace``) into a tar.gz INSIDE the
   workspace, skipping exactly two kinds of entry: every directory that holds a
   ``CACHEDIR.TAG`` file (cargo's ``target`` dir writes one), every directory named
   ``target`` next to a ``Cargo.toml`` (at any depth: older or CARGO_TARGET_DIR-less
   builds leave no tag), and every ``.git``; a source dir merely named ``target`` with
   no sibling ``Cargo.toml`` is kept; symlinks are stored as links, never
   followed. It runs as
   a coding-containers exec of ``python3 -I -c`` + ``tarfile`` (``tar`` is not on the
   exec allowlist, ``python3`` is in every workspace image), refusing anything over
   :data:`ARCHIVE_MAX_BYTES`, and reads it back in base64 chunks of
   :data:`CHUNK_BYTES` (each exec stays under the exec route's 1 MiB output budget);
2. stores it in Postgres (``rust_test_runs``) behind a one-time per-run token and
   starts a Job from the Rust runner image (``services.k8s_jobs``) that downloads it
   from ``GET /internal/test-runs/{run_id}/archive`` and prints one ``RESULT_JSON:``
   line;
3. maps that document onto :class:`UnifiedTestResult` (:func:`map_result`).

Runner result (rust-test-executor ``rust-repo-run``)::

    {presets: {test: {passed, failed, skipped, total, cases[], junit_truncated},
               clippy: {findings[]}, fmt: {files[]}, audit: {advisories[]},
               deny: {advisories[], licenses[], bans[], sources[]},
               coverage: {lines, functions, regions, files[]}},
     tool_versions{}, duration_s, errors[]}

Async runs are a task of THIS service; the result is written to the run's row, so
any replica answers the poll. A poll of a run with no result that no task of the
polling replica owns (the replica running it restarted, or another replica runs it)
reads the Job itself (:func:`reconcile`): finished -> its result is stored; gone ->
the run failed. The first result stored wins (``finish_run``). A run past its
deadline without a result is reported failed.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import hmac
import logging
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import delete, select, update

from config.settings import get_settings
from db import ensure_session_factory
from db_models import RustTestRunRow
from models.unified_test_models import (
    WORKSPACE_ROOT,
    RunStatus,
    UnifiedTestCase,
    UnifiedTestResult,
    WorkspaceTestRun,
    WorkspaceTestRunLinks,
    WorkspaceTestRunRequest,
)
from services import k8s_jobs

logger = logging.getLogger(__name__)

#: Largest archive a Rust run ships to its runner Job.
ARCHIVE_MAX_BYTES = 50 * 1024 * 1024
#: Raw bytes per read-back exec: base64 of it (~683 KiB) stays under the exec route's
#: 1 MiB stdout budget.
CHUNK_BYTES = 512 * 1024
#: Job part of an async Rust run id (``<nonce>.rte``).
ASYNC_JOB = "rte"
#: How long past its timeout (plus the Job's own slack) an async run may stay unanswered.
GRACE_S = k8s_jobs.JOB_DEADLINE_SLACK_S + k8s_jobs.WAIT_SLACK_S + 60
#: A run younger than this whose Job is not found may not have created it yet.
JOB_CREATE_GRACE_S = 60
ARCHIVE_EXIT_TOO_LARGE = 4
EXEC_TIMEOUT = 120
RUN_ROOT = "/tmp/coding-testing"  # nosec B108 - emptyDir in the workspace pod, not this host

#: Packs argv[1] into the gzip tar argv[2]; prints its size; exit 4 when it is over
#: argv[3] bytes. Left out: every directory holding a CACHEDIR.TAG (cargo's target
#: dir writes one), every directory named target whose parent holds a Cargo.toml (a
#: target dir with no tag, any depth) and every .git; a dir merely NAMED target with
#: no sibling Cargo.toml is kept. Symlinks
#: are stored, never followed.
ARCHIVE_PY = (
    "import os,sys,tarfile\n"
    "src,out,cap=sys.argv[1],sys.argv[2],int(sys.argv[3])\n"
    "os.makedirs(os.path.dirname(out),exist_ok=True)\n"
    "cache=lambda t:t.isdir() and os.path.isfile(os.path.join(src,t.name,'CACHEDIR.TAG'))\n"
    "cargo=lambda t:t.isdir() and os.path.basename(t.name)=='target' and "
    "os.path.isfile(os.path.join(src,os.path.dirname(t.name),'Cargo.toml'))\n"
    "skip=lambda t:None if os.path.basename(t.name)=='.git' or cache(t) or cargo(t) else t\n"
    "with tarfile.open(out,'w:gz') as tf:\n"
    " for n in sorted(os.listdir(src)):\n"
    "  tf.add(os.path.join(src,n),arcname=n,filter=skip)\n"
    "size=os.path.getsize(out)\n"
    "if size>cap:\n"
    " os.remove(out);print(f'archive is {size} bytes, over {cap}',file=sys.stderr);sys.exit(4)\n"
    "print(size)\n"
)
#: Prints base64 of argv[3] bytes of argv[1] from offset argv[2].
CHUNK_PY = (
    "import base64,sys\n"
    "f=open(sys.argv[1],'rb');f.seek(int(sys.argv[2]))\n"
    "sys.stdout.write(base64.b64encode(f.read(int(sys.argv[3]))).decode())\n"
)
REMOVE_PY = "import shutil,sys\nshutil.rmtree(sys.argv[1],ignore_errors=True)\n"

#: preset -> the lists in its document whose entries are findings (non-empty = not clean).
_FINDING_LISTS: dict[str, tuple[str, ...]] = {
    "clippy": ("findings",),
    "fmt": ("files",),
    "audit": ("advisories",),
    "deny": ("advisories", "licenses", "bans", "sources"),
}


class _TestPreset(BaseModel):
    model_config = ConfigDict(extra="ignore")
    passed: int = Field(0, ge=0)
    failed: int = Field(0, ge=0)
    skipped: int = Field(0, ge=0)
    total: int = Field(0, ge=0)
    cases: list[UnifiedTestCase] = Field(default_factory=list)


class _RunnerResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    presets: dict[str, Any] = Field(default_factory=dict)
    tool_versions: dict[str, str] = Field(default_factory=dict)
    duration_s: float = Field(0.0, ge=0)
    errors: list[str] = Field(default_factory=list)


def map_result(doc: Any, presets: list[str], workdir: str) -> UnifiedTestResult:
    """The runner's result document as a :class:`UnifiedTestResult`.

    ``success``: no runner error, every requested preset reported, the test preset
    ran at least one case with no failed one, and clippy / fmt / audit /
    deny report nothing.

    ``exit_code`` is the test preset's own ``exit_code`` (the cargo-nextest exit);
    None when the test preset was not requested, not reported, or carries no int
    exit. ``raw_tail`` stays empty: the runner document carries no output tail.
    """
    reasons: list[str] = []
    try:
        res = _RunnerResult.model_validate(doc)
    except ValidationError as e:
        res = _RunnerResult(errors=[f"runner result unreadable: {e}"])
    reasons.extend(res.errors)
    test: _TestPreset | None = None
    exit_code: int | None = None
    for name in presets:
        body = res.presets.get(name)
        if not isinstance(body, dict):
            reasons.append(f"{name}: not reported by the runner")
        elif name == "test":
            code = body.get("exit_code")
            if isinstance(code, int) and not isinstance(code, bool):
                exit_code = code
            try:
                test = _TestPreset.model_validate(body)
            except ValidationError as e:
                reasons.append(f"test: unreadable ({e})")
                continue
            if test.failed:
                reasons.append(f"test: {test.failed} failed")
            elif test.passed == 0:
                reasons.append("test: no tests ran: nothing was verified")
        elif name == "coverage":
            if not isinstance(body.get("lines"), dict):
                reasons.append("coverage: no line coverage reported")
        else:
            for key in _FINDING_LISTS[name]:
                items = body.get(key)
                if not isinstance(items, list):
                    reasons.append(f"{name}: {key} not reported")
                elif items:
                    reasons.append(f"{name}: {len(items)} {key}")
    return UnifiedTestResult(
        framework="cargo-nextest",
        language="rust",
        success=not reasons,
        passed=test.passed if test else 0,
        failed=test.failed if test else 0,
        skipped=test.skipped if test else 0,
        total=test.total if test else 0,
        duration=res.duration_s,
        cases=test.cases if test else [],
        exit_code=exit_code,
        command=["rust-repo-run", *presets],
        workdir=workdir,
        report_found=test is not None,
        error="; ".join(reasons)[:4000] if reasons else None,
        presets=res.presets,
        tool_versions=res.tool_versions,
    )


def failed_result(presets: list[str], workdir: str, why: str) -> UnifiedTestResult:
    """A Rust run that never got a runner result (Job failed / lost, no RESULT_JSON)."""
    return UnifiedTestResult(
        framework="cargo-nextest",
        language="rust",
        success=False,
        command=["rust-repo-run", *presets],
        workdir=workdir,
        error=why[:4000],
    )


def run_status(result: UnifiedTestResult) -> RunStatus:
    """completed: the runner reported at least one preset (verdict = ``success``); failed:
    it never did (no result line, or a runner error with empty presets, e.g. exit 2)."""
    return "completed" if result.presets else "failed"


# ----------------------------------------------------------------- packing


async def pack(exec_fn, container: str, tenant_id: str | None, run_dir: str) -> bytes:
    """The working tree as tar.gz bytes; 413 over the cap, 502 when unreadable.

    ``exec_fn(container, tenant_id, argv, workdir, timeout, what)`` -> exec payload
    with an int ``exit_code`` (``WorkspaceTestRunner._exec``).
    """
    path = f"{run_dir}/src.tgz"
    try:
        out = await exec_fn(
            container, tenant_id,
            ["python3", "-I", "-c", ARCHIVE_PY, WORKSPACE_ROOT, path, str(ARCHIVE_MAX_BYTES)],
            WORKSPACE_ROOT, EXEC_TIMEOUT, "Workspace archive",
        )  # fmt: skip
        if out["exit_code"] == ARCHIVE_EXIT_TOO_LARGE:
            raise HTTPException(
                status_code=413,
                detail=f"The working tree (without target/ and .git) packs to more than "
                f"{ARCHIVE_MAX_BYTES} bytes",
            )
        size_text = (out.get("stdout") or "").strip()
        if out["exit_code"] != 0 or not size_text.isdigit():
            raise HTTPException(
                status_code=502,
                detail="Workspace archive failed: "
                + (out.get("stderr") or out.get("stdout") or "")[-500:],
            )
        size = int(size_text)
        data = bytearray()
        for offset in range(0, size, CHUNK_BYTES):
            part = await exec_fn(
                container, tenant_id,
                ["python3", "-I", "-c", CHUNK_PY, path, str(offset), str(CHUNK_BYTES)],
                WORKSPACE_ROOT, EXEC_TIMEOUT, "Workspace archive read",
            )  # fmt: skip
            if part["exit_code"] != 0 or part.get("truncated"):
                raise HTTPException(
                    status_code=502, detail="Workspace archive read failed or was clipped"
                )
            try:
                data += base64.b64decode((part.get("stdout") or "").strip(), validate=True)
            except binascii.Error as e:
                raise HTTPException(
                    status_code=502, detail=f"Workspace archive chunk is not base64: {e}"
                ) from e
        if len(data) != size:
            raise HTTPException(status_code=502, detail="Workspace archive read back incomplete")
        return bytes(data)
    finally:
        try:
            await exec_fn(
                container, tenant_id, ["python3", "-I", "-c", REMOVE_PY, run_dir],
                WORKSPACE_ROOT, EXEC_TIMEOUT, "Workspace archive cleanup",
            )  # fmt: skip
        except HTTPException as e:
            logger.warning("archive cleanup in %s failed: %s", container, e.detail)


# ----------------------------------------------------------------- store


def _sha256(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


async def store_run(
    run_id: str, container: str, tenant_id: str | None, token: str, archive: bytes, timeout: int
) -> None:
    factory = await ensure_session_factory()
    async with factory() as session:
        session.add(
            RustTestRunRow(
                run_id=run_id,
                container_name=container,
                tenant_id=tenant_id,
                token_sha256=_sha256(token),
                archive=archive,
                deadline=datetime.now(UTC) + timedelta(seconds=timeout + GRACE_S),
            )
        )
        await session.commit()


async def take_archive(run_id: str, token: str) -> bytes | None:
    """Run ``run_id``'s archive, ONCE, when ``token`` matches; else None."""
    factory = await ensure_session_factory()
    async with factory() as session, session.begin():
        row = (
            await session.execute(
                select(RustTestRunRow).where(RustTestRunRow.run_id == run_id).with_for_update()
            )
        ).scalar_one_or_none()
        if row is None or row.archive is None:
            return None
        if not hmac.compare_digest(row.token_sha256, _sha256(token)):
            return None
        data, row.archive = row.archive, None
    return data


async def finish_run(run_id: str, result: UnifiedTestResult) -> bool:
    """Store the result and drop the archive (downloaded or not, the run is over).

    Only the FIRST result is stored (the run's task and a reconciling poll on another
    replica may both finish it); returns whether this one was.
    """
    factory = await ensure_session_factory()
    async with factory() as session:
        done = await session.execute(
            update(RustTestRunRow)
            .where(RustTestRunRow.run_id == run_id, RustTestRunRow.result.is_(None))
            .values(archive=None, result=result.model_dump(mode="json"))
        )
        await session.commit()
    return done.rowcount == 1


async def prune_runs(retention_days: int) -> int:
    """Drop the archive of every run past its deadline (a lost run's archive); delete runs
    older than ``retention_days`` (<= 0: deletion disabled); return the deleted count."""
    factory = await ensure_session_factory()
    async with factory() as session:
        await session.execute(
            update(RustTestRunRow)
            .where(RustTestRunRow.deadline < datetime.now(UTC), RustTestRunRow.archive.is_not(None))
            .values(archive=None)
        )
        await session.commit()
    if retention_days <= 0:
        return 0
    async with factory() as session:
        result = await session.execute(
            delete(RustTestRunRow).where(
                RustTestRunRow.created_at < datetime.now(UTC) - timedelta(days=retention_days)
            )
        )
        await session.commit()
    return result.rowcount


# ----------------------------------------------------------------- run / poll


async def evaluate(
    run_id: str,
    token: str,
    workdir: str,
    presets: list[str],
    timeout: int,
    with_postgres: bool = False,
    extra_args: list[str] | None = None,
) -> UnifiedTestResult:
    """Run one runner Job over the stored archive and map its RESULT_JSON."""
    result = await _evaluate(run_id, token, workdir, presets, timeout, with_postgres, extra_args)
    return result.model_copy(update={"with_postgres": with_postgres})


def job_spec(
    settings: Any,
    run_id: str,
    token: str,
    workdir: str,
    presets: list[str],
    timeout: int,
    with_postgres: bool,
    extra_args: list[str] | None,
) -> dict[str, Any]:
    """The runner Job of one run, from the request's fields (pure: no I/O).

    ``workdir`` is the absolute workspace path (``/workspace/crates/api``); the runner
    gets it checkout-relative (``crates/api``).
    """
    return k8s_jobs.build_job_spec(
        run_id=run_id,
        namespace=settings.TEST_RUNS_NAMESPACE,
        image=settings.RUST_RUNNER_IMAGE,
        archive_url=(
            f"{settings.TESTING_INTERNAL_URL.rstrip('/')}/internal/test-runs/{run_id}/archive"
        ),
        archive_token=token,
        presets=presets,
        workdir=workdir.removeprefix(WORKSPACE_ROOT).lstrip("/"),
        timeout_s=timeout,
        resources={
            "requests": {
                "cpu": settings.RUNNER_CPU_REQUEST,
                "memory": settings.RUNNER_MEMORY_REQUEST,
            },
            "limits": {"cpu": settings.RUNNER_CPU_LIMIT, "memory": settings.RUNNER_MEMORY_LIMIT},
        },
        postgres_image=settings.RUNNER_POSTGRES_IMAGE if with_postgres else None,
        nextest_extra_args=extra_args,
    )


async def _evaluate(
    run_id: str,
    token: str,
    workdir: str,
    presets: list[str],
    timeout: int,
    with_postgres: bool,
    extra_args: list[str] | None,
) -> UnifiedTestResult:
    settings = get_settings()
    if not settings.RUST_RUNNER_IMAGE or not settings.TESTING_INTERNAL_URL:
        return failed_result(
            presets, workdir, "RUST_RUNNER_IMAGE / TESTING_INTERNAL_URL is not configured"
        )
    spec = job_spec(settings, run_id, token, workdir, presets, timeout, with_postgres, extra_args)
    try:
        log = await k8s_jobs.run_job(k8s_jobs.KubeApi(), spec, timeout)
        return map_result(k8s_jobs.parse_result_log(log), presets, workdir)
    except k8s_jobs.KubeError as e:
        return failed_result(presets, workdir, str(e))


async def reconcile(api: k8s_jobs.KubeApi, row: RustTestRunRow) -> UnifiedTestResult | None:
    """Finish run ``row`` from its Job when nobody else will; None while it may still run.

    Finished Job: its result, mapped exactly as the run's own task does (the request's
    presets / workdir / with_postgres are read back from the Job's env). Job not found
    (TTL-reaped, or never created) past :data:`JOB_CREATE_GRACE_S`: failed. The stored
    result is the first one stored (``finish_run``), so it is re-read and returned.
    """
    namespace, name = get_settings().TEST_RUNS_NAMESPACE, k8s_jobs.job_name(row.run_id)
    try:
        job = await api.get_job(namespace, name)
    except k8s_jobs.KubeError as e:
        young = datetime.now(UTC) - row.created_at < timedelta(seconds=JOB_CREATE_GRACE_S)
        if e.status != 404 or young:
            return None  # API server unreachable, or the Job may not be created yet
        result = failed_result(
            [], WORKSPACE_ROOT,
            f"the run's Job {name} is gone without a result (the replica running it restarted?)",
        )  # fmt: skip
    else:
        if k8s_jobs.job_outcome(job) is None:
            return None
        env = {
            e["name"]: e.get("value", "")
            for e in job["spec"]["template"]["spec"]["containers"][0].get("env") or []
        }
        presets = [p for p in env.get("PRESETS", "").split(",") if p]
        workdir = f"{WORKSPACE_ROOT}/{env.get('WORKDIR', '')}".rstrip("/")
        try:
            log = await k8s_jobs.finished_log(api, namespace, name, job)
            result = map_result(k8s_jobs.parse_result_log(log), presets, workdir)
        except k8s_jobs.KubeError as e:
            result = failed_result(presets, workdir, str(e))
        result = result.model_copy(update={"with_postgres": "DATABASE_URL" in env})
    await finish_run(row.run_id, result)
    await k8s_jobs.delete_job_quietly(api, namespace, name)
    return await stored_result(row.run_id)


async def stored_result(run_id: str) -> UnifiedTestResult | None:
    factory = await ensure_session_factory()
    async with factory() as session:
        row = await session.get(RustTestRunRow, run_id)
    return UnifiedTestResult.model_validate(row.result) if row and row.result else None


#: Running async Rust runs of THIS process by nonce (also the strong references
#: asyncio does not keep).
_TASKS: dict[str, asyncio.Task] = {}


def status_path(container: str, run_id: str) -> str:
    return f"/api/tests/workspace/{container}/runs/{run_id}"


async def run_rust(
    exec_fn, container: str, tenant_id: str | None, req: WorkspaceTestRunRequest, workdir: str
) -> WorkspaceTestRun:
    """Pack the working tree and evaluate it in a runner Job (sync or async)."""
    presets = req.rust_presets()
    nonce = uuid.uuid4().hex
    archive = await pack(exec_fn, container, tenant_id, f"{RUN_ROOT}/{nonce}")
    token = secrets.token_urlsafe(32)
    await store_run(nonce, container, tenant_id, token, archive, req.timeout)

    async def finish() -> UnifiedTestResult:
        try:
            try:
                result = await evaluate(
                    nonce, token, workdir, presets, req.timeout, req.with_postgres, req.extra_args
                )
            except Exception as e:  # noqa: BLE001 - a run never ends without a result
                logger.exception("rust run %s failed", nonce)
                result = failed_result(presets, workdir, f"run failed: {e!r}")
            if not await finish_run(nonce, result):  # a reconciling poll was first
                result = await stored_result(nonce) or result
            return result
        finally:  # only once the result is stored: a gone Job means a lost run
            await k8s_jobs.delete_job_quietly(
                k8s_jobs.KubeApi(), get_settings().TEST_RUNS_NAMESPACE, k8s_jobs.job_name(nonce)
            )

    if not req.run_async:
        result = await finish()
        return WorkspaceTestRun(
            run_id=nonce, container_name=container, status=run_status(result), result=result
        )
    task = asyncio.create_task(finish())
    _TASKS[nonce] = task
    task.add_done_callback(lambda _: _TASKS.pop(nonce, None))
    run_id = f"{nonce}.{ASYNC_JOB}"
    return WorkspaceTestRun(
        run_id=run_id,
        container_name=container,
        status="running",
        links=WorkspaceTestRunLinks(status=status_path(container, run_id)),
    )


async def poll_rust(container: str, tenant_id: str | None, run_id: str) -> WorkspaceTestRun:
    """The state of async Rust run ``run_id`` (``<nonce>.rte``) started in ``container``."""
    nonce = run_id.split(".", 1)[0]
    factory = await ensure_session_factory()
    async with factory() as session:
        row = (
            await session.execute(
                select(RustTestRunRow).where(
                    RustTestRunRow.run_id == nonce,
                    RustTestRunRow.container_name == container,
                    RustTestRunRow.tenant_id.is_not_distinct_from(tenant_id),
                )
            )
        ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found in this workspace")
    links = WorkspaceTestRunLinks(status=status_path(container, run_id))
    if row.result is not None:
        result = UnifiedTestResult.model_validate(row.result)
    elif nonce not in _TASKS and (found := await reconcile(k8s_jobs.KubeApi(), row)):
        result = found
    elif datetime.now(UTC) > row.deadline:
        result = failed_result(
            [], WORKSPACE_ROOT, "the run ended without a result (the replica running it restarted?)"
        )
    else:
        return WorkspaceTestRun(
            run_id=run_id, container_name=container, status="running", links=links
        )
    return WorkspaceTestRun(
        run_id=run_id,
        container_name=container,
        status=run_status(result),
        result=result,
        links=links,
    )
