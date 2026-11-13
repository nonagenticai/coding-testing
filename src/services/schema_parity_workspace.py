"""Schema parity inside a workspace pod (py2rust slice A5a).

coding-testing cannot reach the pod-local Postgres, so every step runs in the
workspace through coding-containers exec:

1. ``ws-pg start``                       -> server up (idempotent); prints a URL
2. ``ws-pg createdb <name>`` (x2)        -> prints the URL of a fresh, empty DB
3. ``env DATABASE_URL=<url> <migrate>``  -> in each side's workdir
4. ``ws-pg psql <db> -c BEGIN READ ONLY -c <snapshot SQL> -c COMMIT`` (x2)
5. diff the two snapshots (:func:`services.schema_parity.diff_snapshots`)

``env`` is used to pass DATABASE_URL because the exec API has no env field and
``env`` is on its command allowlist. A failure at any step returns
``computed: false`` with the failing ``stage``, never a guessed verdict.

ws-pg contract assumed (coding-containers A1 lane): ``ws-pg start`` starts
Postgres 18 on 127.0.0.1 (no-op if already running) and prints a DATABASE_URL;
``ws-pg createdb <name>`` creates database ``<name>`` and prints its URL. The
URL is taken as the LAST ``postgres://`` / ``postgresql://`` token on stdout,
so log lines around it are tolerated.

Raw ``psql`` is NOT on the workspace exec allowlist (coding-containers#182): SQL
runs only through ``ws-pg psql <db> -c <sql>...``, which pins host/port/user and
the output flags (``-X -q -A -t -v ON_ERROR_STOP=1``), accepts only ``-c`` pairs
and refuses any SQL containing a backslash. The db is addressed by the NAME this
runner chose for ``ws-pg createdb`` (``^[a-z_][a-z0-9_]{0,62}$``), never by URL.
"""

from __future__ import annotations

import json
import re
import shlex
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from models.parity_schema_models import (
    MigrationSide,
    SchemaDiffOptions,
    SchemaDiffResult,
    SchemaDiffRun,
    SchemaDiffRunLinks,
    SchemaDiffWorkspaceRequest,
    SchemaDiffWorkspaceRunRequest,
    SchemaMapping,
)
from services.schema_parity import (
    SnapshotError,
    SourceSnapshot,
    build_snapshot_sql,
    diff_union,
    parse_snapshot,
)
from services.workspace_driver_run import (
    WorkspaceDriverRuns,
    job_exec_error,
    new_nonce,
    run_dir,
)

WORKSPACE_ROOT = "/workspace"
_URL_RE = re.compile(r"postgres(?:ql)?://\S+")
_TAIL = 2000

# (container_name, tenant_id, argv, timeout, workdir) -> coding-containers ExecResponse
WorkspaceExecIn = Callable[[str, str | None, list[str], int, str], Awaitable[Any]]


class _StepFailedError(Exception):
    def __init__(self, stage: str, error: str) -> None:
        super().__init__(error)
        self.stage = stage
        self.error = error


async def containers_exec_in(
    container_name: str, tenant_id: str | None, argv: list[str], timeout: int, workdir: str
) -> Any:
    """coding-containers exec with an explicit workdir (same auth as _containers_exec)."""
    from coding_client.clients import CodingWorkspacesClient

    from auth.outbound import optional_tenant_headers, service_token
    from config.settings import get_settings
    from services.intelligent_testing_service import WorkspaceReadError

    containers_url = get_settings().CONTAINERS_SERVICE_URL
    if not containers_url:
        raise WorkspaceReadError("CONTAINERS_SERVICE_URL is not configured")
    headers = optional_tenant_headers(tenant_id, service_token=service_token())
    client = CodingWorkspacesClient(base_url=containers_url, timeout=float(timeout) + 10.0)
    try:
        return await client.exec(
            container_name, headers=headers, command=argv, workdir=workdir, timeout=timeout
        )
    finally:
        await client.close()


def resolve_workdir(rel: str) -> str:
    """Workspace-relative path -> absolute path under /workspace; refuse escapes."""
    rel = (rel or ".").strip()
    if rel.startswith(("/", "-")) or ".." in rel.split("/") or "\x00" in rel:
        raise ValueError(f"workdir must be workspace-relative: {rel!r}")
    rel = rel.strip("/")
    return WORKSPACE_ROOT if rel in ("", ".") else f"{WORKSPACE_ROOT}/{rel}"


def migrate_argv(side: MigrationSide) -> list[str]:
    """Split the migration command; refuse an empty one."""
    cmd = side.migrate_command
    argv = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
    if not argv or not argv[0].strip():
        raise ValueError("migrate_command is empty")
    return argv


def parse_url(stdout: str) -> str | None:
    found = _URL_RE.findall(stdout or "")
    return found[-1] if found else None


_DBNAME_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def psql_argv(dbname: str, sql: str) -> list[str]:
    """``ws-pg psql`` argv for one READ ONLY snapshot of ``dbname``.

    Refuses here what ws-pg would refuse in the pod (bad name, backslash), so a
    violation is a local bug report rather than an opaque exec failure.
    """
    if not _DBNAME_RE.match(dbname):
        raise ValueError(f"invalid ws-pg database name: {dbname!r}")
    if "\\" in sql:
        raise ValueError("snapshot SQL must not contain a backslash (ws-pg refuses it)")
    return ["ws-pg", "psql", dbname, "-c", "BEGIN READ ONLY", "-c", sql, "-c", "COMMIT"]


def _redact(argv: list[str]) -> list[str]:
    """Hide credentials in URLs before echoing a command back."""
    return [re.sub(r"(postgres(?:ql)?://[^:/@\s]+):[^@\s]*@", r"\1:***@", a) for a in argv]


class SchemaParityWorkspaceRunner:
    """Runs the workspace schema-diff flow over an injectable exec."""

    def __init__(self, exec_in: WorkspaceExecIn | None = None) -> None:
        self._exec = exec_in or containers_exec_in

    async def _run(
        self,
        stage: str,
        container: str,
        tenant_id: str | None,
        argv: list[str],
        timeout: int,
        workdir: str,
        steps: list[dict[str, Any]],
        *,
        echo: list[str] | None = None,
    ) -> str:
        try:
            payload = await self._exec(container, tenant_id, argv, timeout, workdir)
        except Exception as exc:  # noqa: BLE001 - any transport failure = not computed
            steps.append({"stage": stage, "command": _redact(echo or argv), "error": str(exc)})
            raise _StepFailedError(stage, f"{stage}: exec failed: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("exit_code"), int):
            steps.append(
                {"stage": stage, "command": _redact(echo or argv), "error": "no exit code"}
            )
            raise _StepFailedError(stage, f"{stage}: exec returned no exit code")
        stdout = payload.get("stdout") or ""
        stderr = payload.get("stderr") or ""
        exit_code = payload["exit_code"]
        steps.append(
            {
                "stage": stage,
                "command": _redact(echo or argv),
                "workdir": workdir,
                "exit_code": exit_code,
                "truncated": bool(payload.get("truncated")),
                "stdout_tail": "" if echo is not None else _redact([stdout[-_TAIL:]])[0],
                "stderr_tail": _redact([stderr[-_TAIL:]])[0],
            }
        )
        if exit_code != 0:
            hint = " (timed out)" if exit_code == 124 else ""
            raise _StepFailedError(
                stage, f"{stage}: exit {exit_code}{hint}: {_redact([stderr[-500:].strip()])[0]}"
            )
        if payload.get("truncated"):
            raise _StepFailedError(stage, f"{stage}: output truncated; result not computed")
        return stdout

    async def run(
        self,
        container_name: str,
        request: SchemaDiffWorkspaceRequest,
        tenant_id: str | None,
    ) -> SchemaDiffResult:
        steps: list[dict[str, Any]] = []
        options: SchemaDiffOptions = request.options
        timeout = request.timeout
        run_id = uuid.uuid4().hex[:12]
        plan = side_plan(request, run_id)
        sql = build_snapshot_sql(options.schemas)
        try:
            await self._run(
                "ws_pg_start",
                container_name,
                tenant_id,
                ["ws-pg", "start"],
                timeout,
                WORKSPACE_ROOT,
                steps,
            )
            urls: dict[str, str] = {}
            for side in plan:
                stage = f"createdb_{side.stage}"
                out = await self._run(
                    stage,
                    container_name,
                    tenant_id,
                    ["ws-pg", "createdb", side.dbname],
                    timeout,
                    WORKSPACE_ROOT,
                    steps,
                )
                url = parse_url(out)
                if not url:
                    raise _StepFailedError(stage, f"{stage}: ws-pg printed no postgres URL")
                urls[side.stage] = url
            for side in plan:
                await self._run(
                    f"migrate_{side.stage}",
                    container_name,
                    tenant_id,
                    ["env", f"DATABASE_URL={urls[side.stage]}", *side.argv],
                    timeout,
                    side.workdir,
                    steps,
                )
            snaps: dict[str, dict[str, Any]] = {}
            for side in plan:
                stage = f"introspect_{side.stage}"
                argv = psql_argv(side.dbname, sql)
                out = await self._run(
                    stage,
                    container_name,
                    tenant_id,
                    argv,
                    timeout,
                    WORKSPACE_ROOT,
                    steps,
                    echo=[*argv[:-3], "<snapshot sql>", *argv[-2:]],
                )
                try:
                    snaps[side.stage] = parse_snapshot(out)
                except SnapshotError as exc:
                    raise _StepFailedError(stage, f"{stage}: {exc}") from exc
        except _StepFailedError as failed:
            return SchemaDiffResult(
                computed=False,
                equal=False,
                error=failed.error,
                stage=failed.stage,
                options=options,
                steps=steps,
            )
        result = diff_plan(plan, snaps, options)
        result.steps = steps
        return result


@dataclass(frozen=True)
class PlannedSide:
    """One database to create, migrate and introspect."""

    role: str  # source | target
    name: str  # source name ("source" for the single-source shape) or "target"
    stage: str  # stage suffix: source | target | source:<name>
    dbname: str
    workdir: str
    argv: list[str]
    mapping: SchemaMapping | None


def side_plan(request: SchemaDiffWorkspaceRequest, run_id: str) -> list[PlannedSide]:
    """Sources first (in request order), then the target. Validates workdirs + commands."""
    plan: list[PlannedSide] = []
    if request.sources is not None:
        for i, src in enumerate(request.sources):
            plan.append(
                PlannedSide(
                    role="source",
                    name=src.name,
                    stage=f"source:{src.name}",
                    dbname=f"parity_s{i}_{run_id}",
                    workdir=resolve_workdir(src.workdir),
                    argv=migrate_argv(src),
                    mapping=src.mapping,
                )
            )
    else:
        assert request.source is not None  # nosec B101 - the model requires one shape
        plan.append(
            PlannedSide(
                role="source",
                name="source",
                stage="source",
                dbname=f"parity_sou_{run_id}",
                workdir=resolve_workdir(request.source.workdir),
                argv=migrate_argv(request.source),
                mapping=request.source.mapping,
            )
        )
    plan.append(
        PlannedSide(
            role="target",
            name="target",
            stage="target",
            dbname=f"parity_tar_{run_id}",
            workdir=resolve_workdir(request.target.workdir),
            argv=migrate_argv(request.target),
            mapping=None,
        )
    )
    return plan


def diff_plan(
    plan: list[PlannedSide], snaps: dict[str, dict[str, Any]], options: SchemaDiffOptions
) -> SchemaDiffResult:
    sources = [
        SourceSnapshot(
            side.name,
            snaps[side.stage],
            side.mapping.schema_map if side.mapping else None,
            side.mapping.table_prefix if side.mapping else None,
        )
        for side in plan
        if side.role == "source"
    ]
    return diff_union(sources, snaps["target"], options)


# ---------------------------------------------------------------------------
# Async job mode: the same steps as ONE workspace job (services.schema_parity_driver)
# ---------------------------------------------------------------------------

SCHEMA_DRIVER_PATH = Path(__file__).resolve().parent / "schema_parity_driver.py"
SCHEMA_DRIVER_NAME = "schema_driver.py"
SCHEMA_RUN_KIND = "schema_diff"


def schema_status_path(container_name: str, run_id: str) -> str:
    return f"/api/parity/{container_name}/schema-diff/runs/{run_id}"


class SchemaParityRuns(WorkspaceDriverRuns):
    """Start / poll async schema-diff runs (run store = coding-containers' exec jobs)."""

    async def start(
        self, container: str, tenant_id: str | None, request: SchemaDiffWorkspaceRunRequest
    ) -> SchemaDiffRun:
        nonce = new_nonce()
        plan = side_plan(request, nonce[:12])
        config = {
            "ws_pg": ["ws-pg"],
            "sql": build_snapshot_sql(request.options.schemas),
            "timeout": request.timeout,
            "sides": [
                {"stage": p.stage, "dbname": p.dbname, "workdir": p.workdir, "argv": p.argv}
                for p in plan
            ],
        }
        spec = {
            "kind": SCHEMA_RUN_KIND,
            "options": request.options.model_dump(),
            "plan": [
                {
                    **asdict(p),
                    "mapping": p.mapping.model_dump() if p.mapping else None,
                }
                for p in plan
            ],
        }
        directory = await self.stage(
            container,
            tenant_id,
            nonce,
            {
                SCHEMA_DRIVER_NAME: SCHEMA_DRIVER_PATH.read_text(encoding="utf-8"),
                "driver_config.json": json.dumps(config),
            },
            kind=SCHEMA_RUN_KIND,
            spec=spec,
        )
        argv = ["python3", "-I", f"{directory}/{SCHEMA_DRIVER_NAME}", directory]
        run_id = await self.submit(container, tenant_id, nonce, argv, request.job_timeout)
        return SchemaDiffRun(
            run_id=run_id,
            container_name=container,
            status="queued",
            links=SchemaDiffRunLinks(status=schema_status_path(container, run_id)),
        )

    async def poll(self, container: str, tenant_id: str | None, run_id: str) -> SchemaDiffRun:
        nonce, job = await self.job(container, tenant_id, run_id)
        links = SchemaDiffRunLinks(status=schema_status_path(container, run_id))
        spec = await self.spec(container, tenant_id, nonce, (SCHEMA_RUN_KIND,))
        if spec is None:
            raise HTTPException(status_code=404, detail="Run not found in this workspace")
        if job["status"] in ("queued", "running"):
            return SchemaDiffRun(
                run_id=run_id, container_name=container, status=job["status"], links=links
            )
        result = await self._result(container, tenant_id, nonce, spec, job)
        return SchemaDiffRun(
            run_id=run_id,
            container_name=container,
            status="completed" if result.computed else "failed",
            result=result,
            links=links,
        )

    async def _result(
        self,
        container: str,
        tenant_id: str | None,
        nonce: str,
        spec: dict[str, Any],
        job: dict[str, Any],
    ) -> SchemaDiffResult:
        options = SchemaDiffOptions.model_validate(spec.get("options") or {})

        def not_computed(error: str, stage: str, steps: list | None = None) -> SchemaDiffResult:
            return SchemaDiffResult(
                computed=False,
                equal=False,
                error=error,
                stage=stage,
                options=options,
                steps=steps or [],
            )

        exec_error = job_exec_error(job)
        if exec_error:
            return not_computed(exec_error, "job")
        result, err = await self._read_json(
            container, tenant_id, f"{run_dir(nonce)}/result.json", "Driver result"
        )
        if result is None:
            tail = ((job.get("stdout") or "") + (job.get("stderr") or ""))[-1000:]
            return not_computed(f"{err}; driver exit {job.get('exit_code')}: {tail}", "driver")
        steps = result.get("steps") if isinstance(result.get("steps"), list) else []
        if not result.get("ok") or job.get("exit_code") != 0:
            return not_computed(
                str(result.get("error") or f"driver exited {job.get('exit_code')}"),
                str(result.get("stage") or "driver"),
                steps,
            )
        plan = [
            PlannedSide(
                role=str(p["role"]),
                name=str(p["name"]),
                stage=str(p["stage"]),
                dbname=str(p["dbname"]),
                workdir=str(p["workdir"]),
                argv=[str(a) for a in p["argv"]],
                mapping=SchemaMapping.model_validate(p["mapping"]) if p.get("mapping") else None,
            )
            for p in spec.get("plan") or []
        ]
        snaps: dict[str, dict[str, Any]] = {}
        for side in plan:
            stage = f"introspect_{side.stage}"
            name = (result.get("snapshots") or {}).get(side.stage)
            if not isinstance(name, str) or not re.match(r"^snapshot_\d+\.json$", name):
                return not_computed(f"{stage}: the driver recorded no snapshot", stage, steps)
            text, read_err = await self._read_text(
                container, tenant_id, f"{run_dir(nonce)}/{name}", f"{stage} snapshot"
            )
            if text is None:
                return not_computed(f"{stage}: {read_err}", stage, steps)
            try:
                snaps[side.stage] = parse_snapshot(text)
            except SnapshotError as exc:
                return not_computed(f"{stage}: {exc}", stage, steps)
        diffed = diff_plan(plan, snaps, options)
        diffed.steps = steps
        return diffed
