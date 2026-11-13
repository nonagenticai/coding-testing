"""OpenAPI fetch driver: obtain N source documents + the target in ONE workspace job.

THIS FILE RUNS INSIDE A CODING WORKSPACE, not in coding-testing (async job mode of
``POST /api/parity/{container}/openapi-diff/runs``). coding-testing writes it to
``/tmp/coding-testing/<nonce>/`` and runs it as one async exec job::

    python3 -I openapi_driver.py <run_dir>

Standard library only, Python 3.9+. Every document is obtained the way the
synchronous route obtains it (``services.workspace_openapi``): ``command`` run in
``workdir`` (its stdout is the document unless ``file`` is set), then ``file``
read relative to ``workdir`` - but CONCURRENTLY, at most ``concurrency`` at a time.

Input ``<run_dir>/driver_config.json``::

    {"concurrency": 4, "parity_toml": "/workspace/parity.toml" | null,
     "specs": [{"name": "notes", "role": "source", "workdir": "/workspace/a",
                "command": ["uv", "run", "python", "export.py"] | null,
                "file": "openapi.json" | null, "timeout": 300}, ...]}

Output: ``doc_<i>.json`` per spec (the raw text, NOT validated here) and
``result.json`` = ``{"ok", "specs": [{index, name, role, exit_code, duration,
bytes, error?, stderr_tail?, stdout_tail?}], "parity_toml": file name | null}``.
Exit 0 when every document was obtained.
"""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404 - running the export commands is the driver's job
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

TAIL = 2000


def obtain(spec: dict, index: int, run_dir: str) -> dict:
    rec: dict = {
        "index": index,
        "name": spec.get("name"),
        "role": spec.get("role"),
        "workdir": spec.get("workdir"),
        "command": spec.get("command"),
        "file": spec.get("file"),
        "exit_code": None,
        "duration": None,
        "bytes": 0,
    }
    what = f"{spec.get('role')} {spec.get('name')!r}"
    workdir = spec.get("workdir") or "/workspace"
    text = None
    if spec.get("command"):
        started = time.monotonic()
        try:
            done = subprocess.run(  # nosec B603 - argv from the run's own config
                list(spec["command"]),
                cwd=workdir,
                capture_output=True,
                timeout=int(spec.get("timeout") or 300),
                check=False,
            )
        except subprocess.TimeoutExpired:
            rec["error"] = f"{what} command timed out after {spec.get('timeout')}s"
            return rec
        except OSError as e:
            rec["error"] = f"{what} command failed to run: {e}"
            return rec
        rec["exit_code"] = done.returncode
        rec["duration"] = round(time.monotonic() - started, 3)
        stdout = done.stdout.decode("utf-8", "replace")
        if done.returncode != 0:
            rec["error"] = f"{what} command exited {done.returncode}: no document obtained"
            rec["stderr_tail"] = done.stderr.decode("utf-8", "replace")[-TAIL:]
            rec["stdout_tail"] = stdout[-TAIL:]
            return rec
        text = stdout
    if spec.get("file"):
        path = os.path.join(workdir, spec["file"])
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except OSError as e:
            rec["error"] = f"{what} file {spec['file']} could not be read: {e}"
            return rec
    text = text or ""
    rec["bytes"] = len(text.encode())
    with open(os.path.join(run_dir, f"doc_{index}.json"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return rec


def run(config: dict, run_dir: str) -> dict:
    specs = config.get("specs") or []
    if not specs:
        return {"ok": False, "error": "driver_config has no specs", "specs": []}
    workers = max(1, min(int(config.get("concurrency") or 4), 8))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(lambda pair: obtain(pair[1], pair[0], run_dir), enumerate(specs)))
    out: dict = {"ok": True, "error": None, "specs": records, "parity_toml": None}
    failed = [r["error"] for r in records if r.get("error")]
    if failed:
        out["ok"] = False
        out["error"] = "; ".join(failed)
    toml_path = config.get("parity_toml")
    if toml_path:
        try:
            with open(toml_path, encoding="utf-8") as fh:
                toml = fh.read()
        except OSError as e:
            out["ok"] = False
            out["error"] = ((out["error"] + "; ") if out["error"] else "") + (
                f"parity_toml {toml_path} could not be read: {e}"
            )
        else:
            with open(os.path.join(run_dir, "parity_toml.txt"), "w", encoding="utf-8") as fh:
                fh.write(toml)
            out["parity_toml"] = "parity_toml.txt"
    return out


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: openapi_driver.py <run_dir>", file=sys.stderr)
        return 2
    run_dir = args[0]
    try:
        with open(os.path.join(run_dir, "driver_config.json"), encoding="utf-8") as fh:
            config = json.load(fh)
        out = run(config, run_dir)
    except Exception as e:  # noqa: BLE001 - every failure must still produce result.json
        out = {
            "ok": False,
            "error": f"driver crashed: {type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-4000:],
            "specs": [],
        }
    tmp = os.path.join(run_dir, "result.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    os.replace(tmp, os.path.join(run_dir, "result.json"))
    if not out["ok"]:
        print(f"openapi driver: {out['error']}", file=sys.stderr)
        return 1
    print(json.dumps({"ok": True}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
