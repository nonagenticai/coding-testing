"""Schema-parity driver: migrate + introspect N source DBs and one target DB, in ONE job.

THIS FILE RUNS INSIDE A CODING WORKSPACE, not in coding-testing (async job mode of
``POST /api/parity/{container}/schema-diff/runs``). coding-testing writes it to
``/tmp/coding-testing/<nonce>/`` and runs it as one async exec job::

    python3 -I schema_driver.py <run_dir>

Standard library only, Python 3.9+. It performs exactly the steps the synchronous
route performs one exec at a time (``services.schema_parity_workspace``), with
the same stage names, so a slow cold migration is no longer bound by the gateway's
request budget:

1. ``ws-pg start``
2. ``ws-pg createdb <db>`` per side (URL = last postgres URL on stdout)
3. the side's migrate argv in its workdir with ``DATABASE_URL`` set
4. ``ws-pg psql <db> -c 'BEGIN READ ONLY' -c <snapshot sql> -c COMMIT`` per side

Input ``<run_dir>/driver_config.json``::

    {"ws_pg": ["ws-pg"], "sql": "...", "timeout": 600,
     "sides": [{"stage": "source:notes", "dbname": "parity_s0_x", "workdir": "/workspace/a",
                "argv": ["uv", "run", "alembic", "upgrade", "head"]}, ...]}

Output ``<run_dir>/result.json`` = ``{"ok", "stage", "error", "steps",
"snapshots": {stage: "snapshot_<i>.json"}}``; each snapshot is the raw psql output
in its own file (read back one by one, so N large schemas never share one exec
output budget). Exit 0 when every step succeeded.
"""

from __future__ import annotations

import json
import os
import re
import subprocess  # nosec B404 - running the migrations under test is the driver's job
import sys
import time
import traceback

_URL_RE = re.compile(r"postgres(?:ql)?://\S+")
_TAIL = 2000


class StepFailedError(Exception):
    def __init__(self, stage: str, error: str) -> None:
        super().__init__(error)
        self.stage = stage
        self.error = error


def redact(argv: list[str]) -> list[str]:
    return [re.sub(r"(postgres(?:ql)?://[^:/@\s]+):[^@\s]*@", r"\1:***@", a) for a in argv]


def run_step(
    stage: str,
    argv: list[str],
    cwd: str,
    timeout: int,
    steps: list,
    env: dict | None = None,
    echo: list[str] | None = None,
    hide_stdout: bool = False,
) -> str:
    started = time.monotonic()
    shown = redact(echo or argv)
    try:
        done = subprocess.run(  # nosec B603 - argv from the run's own config
            argv,
            cwd=cwd,
            env=env,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        steps.append({"stage": stage, "command": shown, "workdir": cwd, "exit_code": 124})
        raise StepFailedError(stage, f"{stage}: exit 124 (timed out)") from None
    except OSError as e:
        steps.append({"stage": stage, "command": shown, "error": str(e)})
        raise StepFailedError(stage, f"{stage}: exec failed: {e}") from e
    stdout = done.stdout.decode("utf-8", "replace")
    stderr = done.stderr.decode("utf-8", "replace")
    steps.append(
        {
            "stage": stage,
            "command": shown,
            "workdir": cwd,
            "exit_code": done.returncode,
            "truncated": False,
            "duration": round(time.monotonic() - started, 3),
            "stdout_tail": "" if hide_stdout else redact([stdout[-_TAIL:]])[0],
            "stderr_tail": redact([stderr[-_TAIL:]])[0],
        }
    )
    if done.returncode != 0:
        tail = redact([stderr[-500:].strip()])[0]
        raise StepFailedError(stage, f"{stage}: exit {done.returncode}: {tail}")
    return stdout


def run(config: dict, run_dir: str) -> dict:
    ws_pg = list(config.get("ws_pg") or ["ws-pg"])
    sql = config["sql"]
    timeout = int(config.get("timeout") or 600)
    sides = config.get("sides") or []
    root = config.get("workspace_root") or "/workspace"  # cwd of the ws-pg steps
    steps: list = []
    out: dict = {"ok": False, "stage": None, "error": None, "steps": steps, "snapshots": {}}
    try:
        if not sides:
            raise StepFailedError("config", "driver_config has no sides")
        run_step("ws_pg_start", [*ws_pg, "start"], root, timeout, steps)
        urls = {}
        for side in sides:
            stage = f"createdb_{side['stage']}"
            text = run_step(stage, [*ws_pg, "createdb", side["dbname"]], root, timeout, steps)
            found = _URL_RE.findall(text)
            if not found:
                raise StepFailedError(stage, f"{stage}: ws-pg printed no postgres URL")
            urls[side["stage"]] = found[-1]
        for side in sides:
            env = dict(os.environ)
            env["DATABASE_URL"] = urls[side["stage"]]
            argv = list(side["argv"])
            run_step(
                f"migrate_{side['stage']}",
                argv,
                side["workdir"],
                timeout,
                steps,
                env=env,
                echo=["env", f"DATABASE_URL={urls[side['stage']]}", *argv],
            )
        for i, side in enumerate(sides):
            argv = [*ws_pg, "psql", side["dbname"], "-c", "BEGIN READ ONLY", "-c", sql, "-c"]
            argv.append("COMMIT")
            text = run_step(
                f"introspect_{side['stage']}",
                argv,
                root,
                timeout,
                steps,
                echo=[*argv[:-3], "<snapshot sql>", *argv[-2:]],
                hide_stdout=True,
            )
            name = f"snapshot_{i}.json"
            with open(os.path.join(run_dir, name), "w", encoding="utf-8") as fh:
                fh.write(text)
            out["snapshots"][side["stage"]] = name
        out["ok"] = True
    except StepFailedError as e:
        out["stage"] = e.stage
        out["error"] = e.error
    return out


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: schema_driver.py <run_dir>", file=sys.stderr)
        return 2
    run_dir = args[0]
    try:
        with open(os.path.join(run_dir, "driver_config.json"), encoding="utf-8") as fh:
            config = json.load(fh)
        out = run(config, run_dir)
    except Exception as e:  # noqa: BLE001 - every failure must still produce result.json
        out = {
            "ok": False,
            "stage": "driver",
            "error": f"driver crashed: {type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-4000:],
            "steps": [],
            "snapshots": {},
        }
    tmp = os.path.join(run_dir, "result.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    os.replace(tmp, os.path.join(run_dir, "result.json"))
    if not out["ok"]:
        print(f"schema driver: {out['error']}", file=sys.stderr)
        return 1
    print(json.dumps({"ok": True}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
