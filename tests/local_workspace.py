"""A workspace that IS this machine, for tests that run the real exec path.

``exec`` runs the argv as a real subprocess (``python3`` -> this interpreter) with
``/workspace`` mapped to a local directory; ``exec_async`` does the same for a
driver run, after rewriting ``/workspace`` paths in its ``driver_config.json``,
and records the finished job for ``get_exec_job``. The run directories go wherever
``services.workspace_driver_run.RUN_ROOT`` points (tests monkeypatch it to tmp).
"""

from __future__ import annotations

import json
import subprocess  # nosec B404 - test-local stand-in for the workspace exec
import sys
from pathlib import Path
from typing import Any

WORKSPACE = "/workspace"


class LocalWorkspace:
    def __init__(self, root: Path, env: dict[str, str] | None = None) -> None:
        self.root = root
        self.env = env
        self.job: dict[str, Any] = {}

    def _map(self, value: Any) -> Any:
        if isinstance(value, str) and (value == WORKSPACE or value.startswith(WORKSPACE + "/")):
            return str(self.root) + value[len(WORKSPACE) :]
        if isinstance(value, list):
            return [self._map(v) for v in value]
        if isinstance(value, dict):
            return {k: self._map(v) for k, v in value.items()}
        return value

    def _run(self, argv: list[str], workdir: str, timeout: int) -> dict[str, Any]:
        argv = [sys.executable if a == "python3" else a for a in argv]
        proc = subprocess.run(  # noqa: S603 # nosec B603
            argv,
            cwd=self._map(workdir),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=self.env,
            check=False,
        )
        return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}

    async def exec(self, container, tenant_id, argv, workdir, timeout):
        return self._run(list(argv), workdir, timeout)

    async def exec_async(self, container, tenant_id, argv, workdir, timeout):
        cfg_path = Path(argv[-1]) / "driver_config.json"
        cfg_path.write_text(json.dumps(self._map(json.loads(cfg_path.read_text()))))
        self.job = {"status": "completed", **self._run(list(argv), workdir, timeout)}
        return {"job_id": "local-1", "status": "queued"}

    async def get_exec_job(self, container, tenant_id, job_id):
        return self.job
