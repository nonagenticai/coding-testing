"""Run a stdlib driver script as ONE coding-containers exec job in a workspace.

The pattern of the workspace test runs (A2) and the behaviour-parity runs (A5c),
shared so every async parity operation stores its runs the same way:

* per run a private directory ``/tmp/coding-testing/<nonce>/`` in the workspace
  (``/tmp`` is an emptyDir of the workspace pod) holding the driver, its config
  and the ``result.json`` the driver writes;
* the run's spec (what to grade and how) is kept service-side
  (:mod:`services.run_spec_store`), bound to the workspace + tenant;
* the run id is ``<nonce>.<coding-containers job id>`` (async) or ``<nonce>``
  (sync): the job lives in coding-containers' job store.

No shell: files are written and read back with ``python3 -c`` (zlib + base64;
writes are chunked so no single exec argument grows large) and a driver runs as
``python3 -I <driver> <run_dir>``; both are on every workspace image's exec
allowlist.
"""

from __future__ import annotations

import base64
import json
import re
import uuid
import zlib
from typing import Any

from coding_client.clients import ServiceError
from fastapi import HTTPException

from services.run_spec_store import RunSpecStore, get_run_spec_store
from services.workspace_exec import ContainersTransport, WorkspaceTransport

RUN_ROOT = "/tmp/coding-testing"  # nosec B108 - emptyDir in the workspace pod, not this host
#: coding-containers' synchronous exec ceiling (ExecRequest.timeout le=300).
SYNC_EXEC_MAX_TIMEOUT = 300
HELPER_TIMEOUT = 60
#: base64 characters per exec argument (kept far below kernel/URL limits).
WRITE_CHUNK = 32_000
#: ``<nonce>.<coding-containers job id>`` (async) or ``<nonce>`` (sync).
RUN_ID_RE = re.compile(r"^(?P<nonce>[0-9a-f]{32})(?:\.(?P<job>[A-Za-z0-9_-]{1,64}))?$")
JOB_STATUSES = ("queued", "running", "completed", "failed")

# argv: <dir> <name> <b64 chunk> <first 0|1> <last 0|1>. Chunks append to <name>.b64;
# the last one decodes + inflates into <name> and removes the staging file.
WRITE_FILE_PY = (
    "import base64,pathlib,sys,zlib\n"
    "d=pathlib.Path(sys.argv[1]);d.mkdir(parents=True,exist_ok=True)\n"
    "s=d/(sys.argv[2]+'.b64')\n"
    "with s.open('w' if sys.argv[4]=='1' else 'a') as f:f.write(sys.argv[3])\n"
    "if sys.argv[5]=='1':\n"
    " (d/sys.argv[2]).write_bytes(zlib.decompress(base64.b64decode(s.read_text())));s.unlink()\n"
)

# argv: <path>. Prints the file zlib-compressed + base64 (a recording is repetitive JSON:
# ~10x more fits in the exec route's 1 MiB per-stream budget than with `cat`).
READ_FILE_PY = (
    "import base64,sys,zlib\n"
    "sys.stdout.write(base64.b64encode(zlib.compress(open(sys.argv[1],'rb').read(),9)).decode())\n"
)


def encode_chunks(content: str, chunk: int = WRITE_CHUNK) -> list[str]:
    data = base64.b64encode(zlib.compress(content.encode("utf-8"), 9)).decode("ascii")
    return [data[i : i + chunk] for i in range(0, len(data), chunk)] or [""]


def new_nonce() -> str:
    return uuid.uuid4().hex


def run_dir(nonce: str) -> str:
    return f"{RUN_ROOT}/{nonce}"


class WorkspaceDriverRuns:
    """Exec / file staging / async submit / job lookup over a :class:`WorkspaceTransport`."""

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
        timeout: int,
        what: str,
    ) -> dict[str, Any]:
        try:
            payload = await self.transport.exec(container, tenant_id, argv, "/workspace", timeout)
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
        if not isinstance(payload, dict) or not isinstance(payload.get("exit_code"), int):
            raise HTTPException(status_code=502, detail=f"{what} returned no exit code")
        return payload

    async def _write_file(
        self, container: str, tenant_id: str | None, directory: str, name: str, content: str
    ) -> None:
        chunks = encode_chunks(content)
        for i, chunk in enumerate(chunks):
            payload = await self._exec(
                container,
                tenant_id,
                [
                    "python3",
                    "-c",
                    WRITE_FILE_PY,
                    directory,
                    name,
                    chunk,
                    "1" if i == 0 else "0",
                    "1" if i == len(chunks) - 1 else "0",
                ],
                HELPER_TIMEOUT,
                "Run preparation",
            )
            if payload["exit_code"] != 0:
                raise HTTPException(
                    status_code=502,
                    detail=f"Run preparation failed ({name}): "
                    f"{(payload.get('stderr') or '')[-500:]}",
                )

    async def _read_text(
        self, container: str, tenant_id: str | None, path: str, what: str
    ) -> tuple[str | None, str | None]:
        """(file text, None) or (None, why it could not be read completely)."""
        payload = await self._exec(
            container, tenant_id, ["python3", "-I", "-c", READ_FILE_PY, path], HELPER_TIMEOUT, what
        )
        if payload["exit_code"] != 0:
            return None, f"{what} not found at {path}"
        if payload.get("truncated"):
            return None, f"{what} exceeded the exec output budget and was clipped: not computed"
        try:
            raw = zlib.decompress(base64.b64decode((payload.get("stdout") or "").strip()))
            return raw.decode("utf-8"), None
        except (ValueError, zlib.error) as e:
            return None, f"{what} is not readable: {e}"

    async def _read_json(
        self, container: str, tenant_id: str | None, path: str, what: str
    ) -> tuple[Any, str | None]:
        """(parsed JSON object, None) or (None, why it could not be read)."""
        text, err = await self._read_text(container, tenant_id, path, what)
        if text is None:
            return None, err
        try:
            doc = json.loads(text)
        except ValueError as e:
            return None, f"{what} is not JSON: {e}"
        if not isinstance(doc, dict):
            return None, f"{what} is not a JSON object"
        return doc, None

    async def stage(
        self,
        container: str,
        tenant_id: str | None,
        nonce: str,
        files: dict[str, str],
        kind: str,
        spec: dict[str, Any],
    ) -> str:
        """Store the run's ``spec`` as ``kind`` and write ``files`` ({name: text}) into
        the run directory; returns the directory."""
        await self.specs.put(nonce, kind, container, tenant_id, spec)
        directory = run_dir(nonce)
        for name, content in files.items():
            await self._write_file(container, tenant_id, directory, name, content)
        return directory

    async def submit(
        self,
        container: str,
        tenant_id: str | None,
        nonce: str,
        argv: list[str],
        timeout: int,
    ) -> str:
        """Start ``argv`` as an async exec job; returns the run id ``<nonce>.<job_id>``."""
        try:
            accepted = await self.transport.exec_async(
                container, tenant_id, argv, "/workspace", timeout
            )
        except ServiceError as e:
            raise HTTPException(
                status_code=503 if e.status_code == 503 else 502,
                detail=f"Parity run could not be submitted: {e} {e.body!r}"[:1000],
            ) from e
        except Exception as e:  # noqa: BLE001
            raise HTTPException(
                status_code=502, detail=f"Parity run could not be submitted: {e}"
            ) from e
        job_id = accepted.get("job_id") if isinstance(accepted, dict) else None
        run_id = f"{nonce}.{job_id}"
        if not isinstance(job_id, str) or not RUN_ID_RE.match(run_id):
            raise HTTPException(status_code=502, detail="Async exec returned no usable job_id")
        return run_id

    async def job(
        self, container: str, tenant_id: str | None, run_id: str
    ) -> tuple[str, dict[str, Any]]:
        """(nonce, coding-containers job) of an async run id; 404 / 502 otherwise."""
        m = RUN_ID_RE.match(run_id)
        if not m or not m.group("job"):
            raise HTTPException(
                status_code=404, detail="Unknown run id (only async runs are pollable)"
            )
        try:
            job = await self.transport.get_exec_job(container, tenant_id, m.group("job"))
        except ServiceError as e:
            if e.status_code == 404:
                raise HTTPException(status_code=404, detail="Run not found") from e
            raise HTTPException(status_code=502, detail=f"Run status unavailable: {e}") from e
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Run status unavailable: {e}") from e
        if not isinstance(job, dict):
            raise HTTPException(status_code=502, detail="Run status unavailable: empty job")
        if job.get("status") not in JOB_STATUSES:
            raise HTTPException(status_code=502, detail=f"Unknown job status {job.get('status')!r}")
        return m.group("nonce"), job

    async def spec(
        self, container: str, tenant_id: str | None, nonce: str, kinds: tuple[str, ...]
    ) -> dict[str, Any] | None:
        """The run's stored spec when it was started as one of ``kinds`` in this
        workspace by this tenant; None otherwise."""
        return await self.specs.get(nonce, kinds, container, tenant_id)

    async def recording_nonce(self, container: str, tenant_id: str | None, run_id: str) -> str:
        """The nonce of ``recording_run_id``: a record/compare run of this workspace and
        tenant whose spec is still stored; 422 (unknown or expired) otherwise."""
        m = RUN_ID_RE.match(run_id)
        nonce = m.group("nonce") if m else ""
        if not m or await self.spec(container, tenant_id, nonce, ("record", "compare")) is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"recording_run_id {run_id}: no record/compare run with this id in this "
                    "workspace. It never ran here, or it expired (run specs are pruned after "
                    "RUN_SPEC_RETENTION_DAYS, default 14 days): record the source again"
                ),
            )
        return nonce


def job_exec_error(job: dict[str, Any]) -> str | None:
    """A failed job WITHOUT an exit code never ran to the end (orphaned, timed out)."""
    if job.get("status") == "failed" and not isinstance(job.get("exit_code"), int):
        return str(job.get("error") or "the job ended without an exit code")
    return None
