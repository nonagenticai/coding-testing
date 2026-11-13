"""Language-dispatched test runs of a coding workspace, one result format.

* Python (``pyproject.toml`` / ``uv.lock``): runs the checkout's OWN suite next to
  the checkout (coding-containers exec): ``uv run --frozen pytest --junitxml=...``
  when a lock is committed, else ``python -m pytest`` (never ``uv run`` without a
  lock: it would write a new ``uv.lock`` into the checkout); the JUnit XML is read
  back and parsed into a ``UnifiedTestResult``. ``coverage: true`` adds pytest-cov
  (``services.coverage_report``): a requested but uncomputed coverage fails the run.
* Rust (``Cargo.toml``): NOT run in the workspace. The working tree is packed and
  evaluated by a throwaway runner Job (``services.rust_test_runs``).

Every Python run gets a private directory ``/tmp/coding-testing/<nonce>/`` in the
workspace (``/tmp`` is an emptyDir in the workspace pod) for its JUnit report.
Async runs are a coding-containers async exec job; the run id is
``<nonce>.<job_id>``. The run's spec (command, workdir, report path) is kept
service-side (:mod:`services.run_spec_store`), bound to the workspace + tenant.

No shell: the exec allowlist has no ``sh``, so the DATABASE_URL from ``ws-pg`` is
passed with ``env DATABASE_URL=... <cmd>``, and the run directory is created with
``python3 -c`` (both are on the allowlist; python3 is in every workspace image).
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from coding_client.clients import ServiceError
from fastapi import HTTPException

from models.unified_test_models import (
    RUN_ID_RE,
    CoverageReport,
    Framework,
    UnifiedTestResult,
    WorkspaceLanguage,
    WorkspaceTestRun,
    WorkspaceTestRunLinks,
    WorkspaceTestRunRequest,
)
from services import rust_test_runs
from services.coverage_report import coverage_args, parse_coverage, summary_argv
from services.junit_parser import JUnitParseError, parse_junit
from services.run_spec_store import RunSpecStore, get_run_spec_store
from services.workspace_exec import ContainersTransport, WorkspaceTransport

logger = logging.getLogger(__name__)

RUN_ROOT = "/tmp/coding-testing"  # nosec B108 - emptyDir in the workspace pod, not this host
#: coding-containers' synchronous exec ceiling (ExecRequest.timeout le=300).
SYNC_EXEC_MAX_TIMEOUT = 300
HELPER_TIMEOUT = 60
WS_PG_TIMEOUT = 120
RAW_TAIL_CHARS = 4000

_DB_URL_RE = re.compile(r"(postgres(?:ql)?(?:\+[a-z0-9]+)?://\S+)")

#: The run kind of a python workspace test run in the run-spec store.
RUN_KIND = "workspace_tests"
# Creates the run directory argv[1]. Run as `python3 -c`.
_MKDIR_PY = "import pathlib,sys\npathlib.Path(sys.argv[1]).mkdir(parents=True,exist_ok=True)\n"


def status_path(container_name: str, run_id: str) -> str:
    return f"/api/tests/workspace/{container_name}/runs/{run_id}"


def redact_command(argv: list[str]) -> list[str]:
    """Hide the DATABASE_URL value (it carries a password) in a reported argv."""
    return ["DATABASE_URL=<redacted>" if a.startswith("DATABASE_URL=") else a for a in argv]


def parse_ws_pg_output(stdout: str) -> str | None:
    """The last postgres URL ``ws-pg`` printed (bare or as ``DATABASE_URL=...``)."""
    matches = _DB_URL_RE.findall(stdout or "")
    return matches[-1].strip("'\"") if matches else None


def detect_language(entries: set[str]) -> WorkspaceLanguage:
    """Cargo.toml -> rust; pyproject.toml / uv.lock -> python; both or neither -> 422."""
    rust = "Cargo.toml" in entries
    python = bool({"pyproject.toml", "uv.lock"} & entries)
    if rust and python:
        raise HTTPException(
            status_code=422,
            detail=(
                "Both Cargo.toml and pyproject.toml/uv.lock are present: send "
                "language='python' or language='rust'"
            ),
        )
    if rust:
        return "rust"
    if python:
        return "python"
    raise HTTPException(
        status_code=422,
        detail=(
            "No Cargo.toml, pyproject.toml or uv.lock in the workdir: cannot pick a test "
            "framework (send language, or point workdir at the project)"
        ),
    )


def coverage_path(run_dir: str) -> str:
    return f"{run_dir}/coverage.json"


def build_command(
    run_dir: str, has_uv_lock: bool, extra_args: list[str], coverage: bool = False
) -> tuple[list[str], str]:
    """(pytest argv, absolute JUnit path) of a Python run."""
    junit = f"{run_dir}/junit.xml"
    runner = ["uv", "run", "--frozen", "pytest"] if has_uv_lock else ["python", "-m", "pytest"]
    cov = coverage_args(coverage_path(run_dir)) if coverage else []
    return [*runner, "-p", "no:cacheprovider", f"--junitxml={junit}", *cov, *extra_args], junit


def build_result(
    *,
    framework: Framework,
    language: WorkspaceLanguage,
    command: list[str],
    workdir: str,
    with_postgres: bool,
    exit_code: int | None,
    stdout: str,
    stderr: str,
    exec_duration: float | None,
    report_text: str | None,
    report_error: str | None,
    exec_error: str | None = None,
    coverage: CoverageReport | None = None,
) -> UnifiedTestResult:
    """Grade one run. Zero cases run, a missing/clipped report, or a non-zero exit is never a pass."""
    cases = []
    duration: float | None = None
    parse_error = None
    if report_text is not None:
        try:
            parsed = parse_junit(report_text)
            cases, duration = parsed.cases, parsed.duration
        except JUnitParseError as e:
            parse_error = str(e)
    report_found = report_text is not None and parse_error is None
    counts = {
        s: sum(1 for c in cases if c.status == s)
        for s in ("passed", "failed", "skipped", "errored")
    }
    ran = counts["passed"] + counts["failed"] + counts["errored"]

    error: str | None
    if exec_error:
        error = exec_error
    elif not report_found:
        error = parse_error or report_error or "no JUnit report was written"
    elif ran == 0:
        error = "no tests ran: nothing was verified"
    elif counts["failed"] or counts["errored"]:
        error = f"{counts['failed']} failed, {counts['errored']} errored"
    elif exit_code != 0:
        error = (
            f"exit code {exit_code} although every reported case passed "
            "(coverage floor, plugin or teardown failure?)"
        )
    else:
        error = None
    if error is None and coverage is not None and not coverage.computed:
        error = f"coverage not computed: {coverage.error}"

    tail = stdout or ""
    if stderr:
        tail = f"{tail}\n{stderr}" if tail else stderr
    return UnifiedTestResult(
        framework=framework,
        language=language,
        success=error is None,
        passed=counts["passed"],
        failed=counts["failed"],
        skipped=counts["skipped"],
        errored=counts["errored"],
        total=len(cases),
        duration=max(0.0, duration if duration is not None else float(exec_duration or 0.0)),
        cases=cases,
        raw_tail=tail[-RAW_TAIL_CHARS:],
        exit_code=exit_code,
        command=redact_command(command),
        workdir=workdir,
        with_postgres=with_postgres,
        report_found=report_found,
        error=error,
        coverage=coverage,
    )


def _exit_payload(payload: Any, what: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("exit_code"), int):
        raise HTTPException(status_code=502, detail=f"{what} returned no exit code: not computed")
    return payload


class WorkspaceTestRunner:
    """Detect, prepare, run (sync or async), read back and grade a workspace test run."""

    def __init__(
        self, transport: WorkspaceTransport | None = None, specs: RunSpecStore | None = None
    ):
        self.transport: WorkspaceTransport = transport or ContainersTransport()
        self.specs: RunSpecStore = specs or get_run_spec_store()

    async def _exec(
        self,
        container: str,
        tenant_id: str | None,
        argv: list[str],
        workdir: str,
        timeout: int,
        what: str,
    ) -> dict[str, Any]:
        try:
            payload = await self.transport.exec(container, tenant_id, argv, workdir, timeout)
        except HTTPException:
            raise
        except ServiceError as e:
            if e.status_code == 404:
                raise HTTPException(
                    status_code=404, detail=f"Workspace '{container}' not found"
                ) from e
            raise HTTPException(
                status_code=502, detail=f"{what} failed to execute: {e} {e.body!r}"[:1000]
            ) from e
        except Exception as e:  # noqa: BLE001 - any transport failure is "not run"
            raise HTTPException(status_code=502, detail=f"{what} failed to execute: {e}") from e
        return _exit_payload(payload, what)

    async def _list_entries(self, container: str, tenant_id: str | None, workdir: str) -> set[str]:
        payload = await self._exec(
            container, tenant_id, ["ls", "-A1"], workdir, HELPER_TIMEOUT, "Workdir listing"
        )
        if payload["exit_code"] != 0:
            raise HTTPException(
                status_code=422,
                detail=f"workdir {workdir} is not readable in the workspace: "
                f"{(payload.get('stderr') or '')[:300]}",
            )
        return {line.strip() for line in (payload.get("stdout") or "").splitlines() if line.strip()}

    async def _start_postgres(self, container: str, tenant_id: str | None, workdir: str) -> str:
        payload = await self._exec(container, tenant_id, ["ws-pg"], workdir, WS_PG_TIMEOUT, "ws-pg")
        url = parse_ws_pg_output(payload.get("stdout") or "")
        if payload["exit_code"] != 0 or not url:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"with_postgres: ws-pg exited {payload['exit_code']} without a DATABASE_URL: "
                    f"{((payload.get('stderr') or '') + (payload.get('stdout') or ''))[-500:]}"
                ),
            )
        return url

    async def _make_run_dir(self, container: str, tenant_id: str | None, run_dir: str) -> None:
        payload = await self._exec(
            container,
            tenant_id,
            ["python3", "-c", _MKDIR_PY, run_dir],
            "/workspace",
            HELPER_TIMEOUT,
            "Run preparation",
        )
        if payload["exit_code"] != 0:
            raise HTTPException(
                status_code=502,
                detail=f"Run preparation failed: {(payload.get('stderr') or '')[-500:]}",
            )

    async def _read_file(
        self, container: str, tenant_id: str | None, path: str, what: str
    ) -> tuple[str | None, str | None]:
        """(content, error). A missing file or a clipped read is an error, never content."""
        payload = await self._exec(
            container, tenant_id, ["cat", path], "/workspace", HELPER_TIMEOUT, what
        )
        if payload["exit_code"] != 0:
            return None, f"{what} not found at {path}"
        if payload.get("truncated"):
            return None, f"{what} exceeded the exec output budget and was clipped: not computed"
        return payload.get("stdout") or "", None

    async def _coverage(
        self, container: str, tenant_id: str | None, path: str, workdir: str
    ) -> CoverageReport:
        """Reduce the report to its summary IN the workspace, then read only that back."""
        payload = await self._exec(
            container,
            tenant_id,
            summary_argv(path, workdir),
            "/workspace",
            HELPER_TIMEOUT,
            "Coverage summary",
        )
        return parse_coverage(
            payload["exit_code"], payload.get("stdout") or "", payload.get("stderr") or ""
        )

    async def run(
        self, container: str, tenant_id: str | None, req: WorkspaceTestRunRequest
    ) -> WorkspaceTestRun:
        if not req.run_async and req.timeout > SYNC_EXEC_MAX_TIMEOUT:
            raise HTTPException(
                status_code=422,
                detail=f"timeout > {SYNC_EXEC_MAX_TIMEOUT}s needs async=true",
            )
        workdir = req.absolute_workdir()
        entries = await self._list_entries(container, tenant_id, workdir)
        language = req.language or detect_language(entries)
        if language == "rust":
            return await rust_test_runs.run_rust(self._exec, container, tenant_id, req, workdir)
        if req.presets:
            raise HTTPException(status_code=422, detail="presets apply to rust runs only")
        framework: Framework = "pytest"
        nonce = uuid.uuid4().hex
        run_dir = f"{RUN_ROOT}/{nonce}"
        argv, junit = build_command(run_dir, "uv.lock" in entries, req.extra_args, req.coverage)
        if req.with_postgres:
            url = await self._start_postgres(container, tenant_id, workdir)
            argv = ["env", f"DATABASE_URL={url}", *argv]
        spec = {
            "framework": framework,
            "language": language,
            "command": redact_command(argv),
            "workdir": workdir,
            "with_postgres": req.with_postgres,
            "junit": junit,
            "coverage_path": coverage_path(run_dir) if req.coverage else None,
        }
        await self.specs.put(nonce, RUN_KIND, container, tenant_id, spec)
        await self._make_run_dir(container, tenant_id, run_dir)

        if req.run_async:
            try:
                accepted = await self.transport.exec_async(
                    container, tenant_id, argv, workdir, req.timeout
                )
            except ServiceError as e:
                raise HTTPException(
                    status_code=503 if e.status_code == 503 else 502,
                    detail=f"Async test run could not be submitted: {e} {e.body!r}"[:1000],
                ) from e
            except Exception as e:  # noqa: BLE001
                raise HTTPException(
                    status_code=502, detail=f"Async test run could not be submitted: {e}"
                ) from e
            job_id = accepted.get("job_id") if isinstance(accepted, dict) else None
            if not isinstance(job_id, str) or not RUN_ID_RE.match(f"{nonce}.{job_id}"):
                raise HTTPException(status_code=502, detail="Async exec returned no usable job_id")
            run_id = f"{nonce}.{job_id}"
            return WorkspaceTestRun(
                run_id=run_id,
                container_name=container,
                status="queued",
                links=WorkspaceTestRunLinks(status=status_path(container, run_id)),
            )

        payload = await self._exec(
            container, tenant_id, argv, workdir, req.timeout, "Workspace test run"
        )
        exit_code = payload["exit_code"]
        report, report_error = await self._read_file(container, tenant_id, junit, "JUnit report")
        coverage = (
            await self._coverage(container, tenant_id, coverage_path(run_dir), workdir)
            if req.coverage
            else None
        )
        result = build_result(
            framework=framework,
            language=language,
            command=argv,
            workdir=workdir,
            with_postgres=req.with_postgres,
            exit_code=exit_code,
            stdout=payload.get("stdout") or "",
            stderr=payload.get("stderr") or "",
            exec_duration=payload.get("duration"),
            report_text=report,
            report_error=report_error,
            exec_error=f"timed out after {req.timeout}s" if exit_code == 124 else None,
            coverage=coverage,
        )
        return WorkspaceTestRun(
            run_id=nonce, container_name=container, status="completed", result=result
        )

    async def poll(self, container: str, tenant_id: str | None, run_id: str) -> WorkspaceTestRun:
        m = RUN_ID_RE.match(run_id)
        if not m or not m.group("job"):
            raise HTTPException(
                status_code=404, detail="Unknown run id (only async runs are pollable)"
            )
        nonce, job_id = m.group("nonce"), m.group("job")
        if job_id == rust_test_runs.ASYNC_JOB:
            return await rust_test_runs.poll_rust(container, tenant_id, run_id)
        try:
            job = await self.transport.get_exec_job(container, tenant_id, job_id)
        except ServiceError as e:
            if e.status_code == 404:
                raise HTTPException(status_code=404, detail="Run not found") from e
            raise HTTPException(status_code=502, detail=f"Run status unavailable: {e}") from e
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Run status unavailable: {e}") from e
        if not isinstance(job, dict):
            raise HTTPException(
                status_code=502, detail="Run status unavailable: empty job document"
            )
        links = WorkspaceTestRunLinks(status=status_path(container, run_id))
        status = job.get("status")
        if status in ("queued", "running"):
            return WorkspaceTestRun(
                run_id=run_id, container_name=container, status=status, links=links
            )
        if status not in ("completed", "failed"):
            raise HTTPException(status_code=502, detail=f"Unknown job status {status!r}")

        spec = await self.specs.get(nonce, (RUN_KIND,), container, tenant_id)
        if spec is None:
            raise HTTPException(status_code=404, detail="Run not found in this workspace")
        report, report_error = await self._read_file(
            container, tenant_id, spec["junit"], "JUnit report"
        )
        coverage = (
            await self._coverage(container, tenant_id, spec["coverage_path"], spec["workdir"])
            if spec.get("coverage_path")
            else None
        )
        exit_code = job.get("exit_code") if isinstance(job.get("exit_code"), int) else None
        exec_error = None
        if status == "failed" or exit_code is None:
            exec_error = str(job.get("error") or "the job ended without an exit code")
        result = build_result(
            framework=spec["framework"],
            language=spec["language"],
            command=spec["command"],
            workdir=spec["workdir"],
            with_postgres=spec["with_postgres"],
            exit_code=exit_code,
            stdout=job.get("stdout") or "",
            stderr=job.get("stderr") or "",
            exec_duration=job.get("duration"),
            report_text=report,
            report_error=report_error,
            exec_error=exec_error,
            coverage=coverage,
        )
        return WorkspaceTestRun(
            run_id=run_id,
            container_name=container,
            status="completed" if status == "completed" else "failed",
            result=result,
            links=links,
        )
