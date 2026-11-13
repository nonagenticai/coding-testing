"""Obtain OpenAPI documents inside a coding workspace (coding-containers exec).

Each document comes from a command run in a workdir (``uv run python
scripts/export_openapi.py``, ``cargo run -q -- --print-openapi``), from a file in
the checkout (``openapi.json``), or from a command followed by reading the file it
wrote. Commands up to 300 s use the synchronous exec route; longer ones (a cold
``cargo run``) run as a coding-containers async exec job that this service polls
until it ends. No shell: argv goes to the exec allowlist as-is.

Every failure to obtain a document is explicit (non-zero exit, clipped output,
not JSON): nothing is diffed from a document that was not fully read.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from coding_client.clients import ServiceError
from fastapi import HTTPException

from models.parity_models import (
    ObtainedSpec,
    OpenApiDiffIgnore,
    OpenApiDiffRun,
    OpenApiDiffRunLinks,
    WorkspaceOpenApiDiffRequest,
    WorkspaceOpenApiDiffResult,
    WorkspaceOpenApiDiffRunRequest,
    WorkspaceSpecSource,
)
from services.openapi_diff import (
    IgnoreRules,
    OpenApiDiffError,
    SpecInput,
    ignore_rules_from_parity_toml,
    openapi_diff,
)
from services.workspace_driver_run import (
    WorkspaceDriverRuns,
    job_exec_error,
    new_nonce,
    run_dir,
)
from services.workspace_exec import ContainersTransport, WorkspaceTransport

logger = logging.getLogger(__name__)

#: coding-containers' synchronous exec ceiling (ExecRequest.timeout le=300).
SYNC_EXEC_MAX_TIMEOUT = 300
READ_TIMEOUT = 60
POLL_INTERVAL = 2.0
TAIL_CHARS = 2000


def _tail(text: str | None) -> str:
    return (text or "")[-TAIL_CHARS:]


class WorkspaceOpenApiFetcher:
    def __init__(
        self, transport: WorkspaceTransport | None = None, poll_interval: float = POLL_INTERVAL
    ):
        self.transport: WorkspaceTransport = transport or ContainersTransport()
        self.poll_interval = poll_interval

    async def _call(self, what: str, coro: Any) -> Any:
        try:
            return await coro
        except HTTPException:
            raise
        except ServiceError as e:
            if e.status_code == 404:
                raise HTTPException(status_code=404, detail=f"{what}: workspace not found") from e
            raise HTTPException(
                status_code=502, detail=f"{what} failed to execute: {e} {e.body!r}"[:1000]
            ) from e
        except Exception as e:  # noqa: BLE001 - any transport failure is "not run"
            raise HTTPException(status_code=502, detail=f"{what} failed to execute: {e}") from e

    async def run(
        self,
        container: str,
        tenant_id: str | None,
        argv: list[str],
        workdir: str,
        timeout: int,
        what: str,
    ) -> dict[str, Any]:
        """Run argv; returns {exit_code, stdout, stderr, truncated, duration}."""
        if timeout <= SYNC_EXEC_MAX_TIMEOUT:
            payload = await self._call(
                what, self.transport.exec(container, tenant_id, argv, workdir, timeout)
            )
        else:
            payload = await self._run_async(container, tenant_id, argv, workdir, timeout, what)
        if not isinstance(payload, dict) or not isinstance(payload.get("exit_code"), int):
            raise HTTPException(
                status_code=502, detail=f"{what} returned no exit code: not computed"
            )
        return payload

    async def _run_async(
        self,
        container: str,
        tenant_id: str | None,
        argv: list[str],
        workdir: str,
        timeout: int,
        what: str,
    ) -> Any:
        accepted = await self._call(
            what, self.transport.exec_async(container, tenant_id, argv, workdir, timeout)
        )
        job_id = accepted.get("job_id") if isinstance(accepted, dict) else None
        if not isinstance(job_id, str) or not job_id:
            raise HTTPException(status_code=502, detail=f"{what}: async exec returned no job_id")
        deadline = time.monotonic() + timeout + 60
        while True:
            job = await self._call(what, self.transport.get_exec_job(container, tenant_id, job_id))
            status = job.get("status") if isinstance(job, dict) else None
            if status in ("completed", "failed"):
                if job.get("exit_code") is None:
                    raise HTTPException(
                        status_code=422,
                        detail=f"{what} did not exit normally: {job.get('error') or status}",
                    )
                return job
            if time.monotonic() > deadline:
                raise HTTPException(
                    status_code=504, detail=f"{what}: async job {job_id} still {status}"
                )
            await asyncio.sleep(self.poll_interval)

    async def obtain(
        self, container: str, tenant_id: str | None, spec: WorkspaceSpecSource, name: str, role: str
    ) -> tuple[dict[str, Any], ObtainedSpec]:
        workdir = spec.absolute_workdir()
        record = ObtainedSpec(
            name=name, role=role, workdir=workdir, command=spec.command, file=spec.file
        )
        text: str | None = None
        if spec.command:
            what = f"{role} {name!r} command"
            started = time.monotonic()
            payload = await self.run(
                container, tenant_id, spec.command, workdir, spec.timeout, what
            )
            record.exit_code = payload["exit_code"]
            record.duration = float(payload.get("duration") or (time.monotonic() - started))
            if payload["exit_code"] != 0:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": f"{what} exited {payload['exit_code']}: no document obtained",
                        "stderr_tail": _tail(payload.get("stderr")),
                        "stdout_tail": _tail(payload.get("stdout")),
                    },
                )
            if not spec.file:
                if payload.get("truncated"):
                    raise HTTPException(
                        status_code=422,
                        detail=f"{what}: stdout exceeded the exec output budget and was clipped",
                    )
                text = payload.get("stdout") or ""
        if spec.file:
            what = f"{role} {name!r} file {spec.file}"
            payload = await self.run(
                container, tenant_id, ["cat", "--", spec.file], workdir, READ_TIMEOUT, what
            )
            if payload["exit_code"] != 0:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": f"{what} could not be read (exit {payload['exit_code']})",
                        "stderr_tail": _tail(payload.get("stderr")),
                    },
                )
            if payload.get("truncated"):
                raise HTTPException(
                    status_code=422,
                    detail=f"{what} exceeded the exec output budget and was clipped",
                )
            text = payload.get("stdout") or ""
        assert text is not None  # nosec B101 - the request model requires command or file
        record.bytes = len(text.encode())
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": f"{role} {name!r}: output is not a JSON document ({e.msg})",
                    "stdout_head": text[:TAIL_CHARS],
                },
            ) from e
        if not isinstance(doc, dict):
            raise HTTPException(
                status_code=422, detail=f"{role} {name!r}: output is JSON but not an object"
            )
        return doc, record

    async def read_text(self, container: str, tenant_id: str | None, path: str, what: str) -> str:
        payload = await self.run(container, tenant_id, ["cat", "--", path], "/workspace", 60, what)
        if payload["exit_code"] != 0 or payload.get("truncated"):
            raise HTTPException(
                status_code=422,
                detail=f"{what} could not be read completely (exit {payload['exit_code']})",
            )
        return payload.get("stdout") or ""


# ---------------------------------------------------------------------------
# N sources: bounded-concurrent fetch, one diff function, async job mode
# ---------------------------------------------------------------------------


def spec_names(request: WorkspaceOpenApiDiffRequest) -> tuple[list[str], str]:
    names = [s.name or f"source[{i}]" for i, s in enumerate(request.sources)]
    target_name = request.target.name or "target"
    if target_name in names or len(set(names)) != len(names):
        raise HTTPException(status_code=422, detail="source/target names must be unique")
    return names, target_name


async def obtain_all(
    fetcher: WorkspaceOpenApiFetcher,
    container: str,
    tenant_id: str | None,
    request: WorkspaceOpenApiDiffRequest,
) -> tuple[list[tuple[str, dict[str, Any]]], tuple[str, dict[str, Any]], list[ObtainedSpec]]:
    """Every source + the target, at most ``request.concurrency`` at a time.

    Waits for all of them; the first failure IN REQUEST ORDER is raised (so the
    error a caller sees does not depend on scheduling).
    """
    names, target_name = spec_names(request)
    gate = asyncio.Semaphore(request.concurrency)
    jobs = [(n, s, "source") for n, s in zip(names, request.sources, strict=True)]
    jobs.append((target_name, request.target, "target"))

    async def one(name: str, spec: WorkspaceSpecSource, role: str) -> Any:
        async with gate:
            return await fetcher.obtain(container, tenant_id, spec, name, role)

    results = await asyncio.gather(*(one(*j) for j in jobs), return_exceptions=True)
    for r in results:
        if isinstance(r, BaseException):
            raise r
    docs = [r for r in results if not isinstance(r, BaseException)]
    sources = [(n, d[0]) for n, d in zip(names, docs[:-1], strict=True)]
    return sources, (target_name, docs[-1][0]), [d[1] for d in docs]


def merged_rules(ignore: OpenApiDiffIgnore | None, parity_toml_text: str | None) -> IgnoreRules:
    base = IgnoreRules.from_mapping(ignore.model_dump() if ignore else None)
    if parity_toml_text is None:
        return base
    extra = ignore_rules_from_parity_toml(parity_toml_text)
    return IgnoreRules(
        paths=tuple(sorted(set(base.paths) | set(extra.paths))),
        operations=tuple(sorted(set(base.operations) | set(extra.operations))),
        response_codes=tuple(sorted(set(base.response_codes) | set(extra.response_codes))),
    )


def diff_workspace_docs(
    request: WorkspaceOpenApiDiffRequest,
    container: str,
    sources: list[tuple[str, dict[str, Any]]],
    target: tuple[str, dict[str, Any]],
    obtained: list[ObtainedSpec],
    toml_text: str | None,
) -> WorkspaceOpenApiDiffResult:
    """Raises OpenApiDiffError when the documents cannot be compared."""
    maps = [tuple(s.path_prefix_map.items()) for s in request.sources]
    result = openapi_diff(
        [SpecInput(n, d, m) for (n, d), m in zip(sources, maps, strict=True)],
        SpecInput(target[0], target[1]),
        merged_rules(request.ignore, toml_text),
        strict_numeric_bounds=request.strict_numeric_bounds,
        auth=request.auth,
    )
    return WorkspaceOpenApiDiffResult.model_validate(
        {**result, "container_name": container, "obtained": [o.model_dump() for o in obtained]}
    )


OPENAPI_DRIVER_PATH = Path(__file__).resolve().parent / "openapi_fetch_driver.py"
OPENAPI_DRIVER_NAME = "openapi_driver.py"
OPENAPI_RUN_KIND = "openapi_diff"


def openapi_status_path(container_name: str, run_id: str) -> str:
    return f"/api/parity/{container_name}/openapi-diff/runs/{run_id}"


class OpenApiDiffRuns(WorkspaceDriverRuns):
    """Start / poll async openapi-diff runs (run store = coding-containers' exec jobs)."""

    async def start(
        self, container: str, tenant_id: str | None, request: WorkspaceOpenApiDiffRunRequest
    ) -> OpenApiDiffRun:
        names, target_name = spec_names(request)
        specs = [
            {
                "name": name,
                "role": role,
                "workdir": spec.absolute_workdir(),
                "command": spec.command,
                "file": spec.file,
                "timeout": spec.timeout,
            }
            for name, spec, role in [
                *((n, s, "source") for n, s in zip(names, request.sources, strict=True)),
                (target_name, request.target, "target"),
            ]
        ]
        config = {
            "concurrency": request.concurrency,
            "parity_toml": f"/workspace/{request.parity_toml}" if request.parity_toml else None,
            "specs": specs,
        }
        nonce = new_nonce()
        directory = await self.stage(
            container,
            tenant_id,
            nonce,
            {
                OPENAPI_DRIVER_NAME: OPENAPI_DRIVER_PATH.read_text(encoding="utf-8"),
                "driver_config.json": json.dumps(config),
            },
            kind=OPENAPI_RUN_KIND,
            spec={"kind": OPENAPI_RUN_KIND, "request": request.model_dump(mode="json")},
        )
        argv = ["python3", "-I", f"{directory}/{OPENAPI_DRIVER_NAME}", directory]
        run_id = await self.submit(container, tenant_id, nonce, argv, request.job_timeout)
        return OpenApiDiffRun(
            run_id=run_id,
            container_name=container,
            status="queued",
            links=OpenApiDiffRunLinks(status=openapi_status_path(container, run_id)),
        )

    async def poll(self, container: str, tenant_id: str | None, run_id: str) -> OpenApiDiffRun:
        nonce, job = await self.job(container, tenant_id, run_id)
        links = OpenApiDiffRunLinks(status=openapi_status_path(container, run_id))
        spec = await self.spec(container, tenant_id, nonce, (OPENAPI_RUN_KIND,))
        if spec is None:
            raise HTTPException(status_code=404, detail="Run not found in this workspace")
        if job["status"] in ("queued", "running"):
            return OpenApiDiffRun(
                run_id=run_id, container_name=container, status=job["status"], links=links
            )

        def failed(error: Any) -> OpenApiDiffRun:
            return OpenApiDiffRun(
                run_id=run_id, container_name=container, status="failed", error=error, links=links
            )

        exec_error = job_exec_error(job)
        if exec_error:
            return failed(exec_error)
        request = WorkspaceOpenApiDiffRunRequest.model_validate(spec["request"])
        directory = run_dir(nonce)
        result, err = await self._read_json(
            container, tenant_id, f"{directory}/result.json", "Driver result"
        )
        if result is None:
            tail = ((job.get("stdout") or "") + (job.get("stderr") or ""))[-1000:]
            return failed(f"{err}; driver exit {job.get('exit_code')}: {tail}")
        records = result.get("specs") if isinstance(result.get("specs"), list) else []
        if not result.get("ok") or job.get("exit_code") != 0:
            return failed(
                {
                    "error": result.get("error") or f"driver exited {job.get('exit_code')}",
                    "specs": [r for r in records if isinstance(r, dict) and r.get("error")],
                }
            )
        docs: list[dict[str, Any]] = []
        obtained: list[ObtainedSpec] = []
        for i, rec in enumerate(records):
            label = f"{rec.get('role')} {rec.get('name')!r}"
            text, read_err = await self._read_text(
                container, tenant_id, f"{directory}/doc_{i}.json", f"{label} document"
            )
            if text is None:
                return failed(read_err)
            try:
                doc = json.loads(text)
            except json.JSONDecodeError as e:
                return failed(
                    {
                        "error": f"{label}: output is not a JSON document ({e.msg})",
                        "stdout_head": text[:TAIL_CHARS],
                    }
                )
            if not isinstance(doc, dict):
                return failed(f"{label}: output is JSON but not an object")
            docs.append(doc)
            obtained.append(
                ObtainedSpec.model_validate(
                    {
                        k: rec.get(k)
                        for k in ("name", "role", "workdir", "command", "file", "exit_code")
                    }
                    | {"duration": rec.get("duration"), "bytes": rec.get("bytes") or 0}
                )
            )
        names, target_name = spec_names(request)
        if len(docs) != len(names) + 1:
            return failed("the driver did not record every document")
        toml_text = None
        if result.get("parity_toml"):
            toml_text, read_err = await self._read_text(
                container, tenant_id, f"{directory}/parity_toml.txt", "parity_toml"
            )
            if toml_text is None:
                return failed(read_err)
        try:
            diffed = diff_workspace_docs(
                request,
                container,
                list(zip(names, docs[:-1], strict=True)),
                (target_name, docs[-1]),
                obtained,
                toml_text,
            )
        except OpenApiDiffError as e:
            return failed(str(e))
        return OpenApiDiffRun(
            run_id=run_id, container_name=container, status="completed", result=diffed, links=links
        )
