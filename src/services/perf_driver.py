"""Performance driver: replay a corpus under load against source(s) and target.

THIS FILE RUNS INSIDE A CODING WORKSPACE (``python3 -I perf_driver.py <run_dir>``)
next to ``parity_driver.py`` (the behaviour driver, staged in the same run
directory), whose service startup it reuses: fresh database, migrate, upstream
stubs, start, health wait. Python standard library only, 3.9+.

Input ``<run_dir>/driver_config.json``::

    {"ws_pg": ["ws-pg"], "default_headers": {}, "request_timeout": 30,
     "levels": [1, 8], "duration_seconds": 10 | null, "requests": null | 5000,
     "warmup_rounds": 1, "case_ids": null | ["c1", ...],
     "sides": [{...behaviour side..., "key": "s0" | "t0", "source_id": "notes",
                "cases" | "cases_file", "route_map", "default_headers",
                "expected_status": {case name: status},
                "reference_key": "s0" | null}]}

Per side: the cases are filtered to ``case_ids`` and to HTTP (websocket/sse cases
are listed in ``cases_skipped``). Warmup round 1 replays every case sequentially
with its scripted upstream responses; every upstream call it makes is learned into
a (method, path) -> response table that answers the stubs during the load phase.
Expected status per case, first hit wins: ``expected_status`` (from a recording),
the case's ``expected_status``, the status the REFERENCE side (the paired source)
answered in its warmup. Under load a response with another status is an error.

Per level, ``concurrency`` threads (one keep-alive connection each) send the
cases round-robin for ``duration_seconds`` or until ``requests`` were sent. Per
route (the source's ``METHOD path``) and overall: count, errors, throughput,
latency p50/p95/p99/max (nearest rank). The service's peak RSS is the largest
``VmHWM`` in /proc over its process tree (null without /proc).
``client_cpu_ratio`` near 1.0 means the GIL-bound load generator, not the
service, was the bottleneck: ``load_generator_saturated``.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import sys
import threading
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(_HERE, "parity_driver.py")):  # staged in the workspace run dir
    # -I keeps the script dir off sys.path; add ONLY this run's private directory.
    sys.path.insert(0, _HERE)
    import parity_driver as base  # type: ignore[import-not-found]
else:  # inside coding-testing (tests, imports)
    from services import behaviour_parity_driver as base

DRIVER_VERSION = 1
SATURATION_RATIO = 0.9
PROC_ROOT = "/proc"


def percentile(sorted_values: list, q: float) -> float | None:
    """Nearest-rank percentile of an ascending list (None for no samples)."""
    if not sorted_values:
        return None
    rank = max(1, -(-len(sorted_values) * q // 100))  # ceil(n*q/100), at least 1
    return sorted_values[int(rank) - 1]


def _r(v: float | None) -> float | None:
    return None if v is None else round(v, 3)


def stats(latencies: list, mismatches: int, transport: int, wall: float) -> dict:
    lat = sorted(latencies)
    count = len(lat)
    return {
        "count": count,
        "errors": mismatches + transport,
        "status_mismatches": mismatches,
        "transport_errors": transport,
        "throughput_rps": round(count / wall, 3) if wall > 0 else None,
        "p50_ms": _r(percentile(lat, 50)),
        "p95_ms": _r(percentile(lat, 95)),
        "p99_ms": _r(percentile(lat, 99)),
        "max_ms": _r(lat[-1] if lat else None),
        "mean_ms": _r(sum(lat) / count if count else None),
    }


def route_key(case: dict) -> str:
    return f"{case['method'].upper()} {case['path'].split('?')[0]}"


def peak_rss_kb(root_pid: int) -> int | None:
    """Largest VmHWM (kB) over ``root_pid``'s process tree; None without /proc."""
    if not os.path.isdir(PROC_ROOT):
        return None
    children: dict = {}
    for name in os.listdir(PROC_ROOT):
        if not name.isdigit():
            continue
        try:
            with open(f"{PROC_ROOT}/{name}/stat", encoding="utf-8") as fh:
                ppid = int(fh.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(name))
    todo, best = [root_pid], None
    while todo:
        pid = todo.pop()
        todo.extend(children.get(pid, []))
        try:
            with open(f"{PROC_ROOT}/{pid}/status", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("VmHWM:"):
                        kb = int(line.split()[1])
                        best = kb if best is None else max(best, kb)
        except (OSError, ValueError):
            continue
    return best


def select_cases(cases: list, case_ids: list | None) -> tuple[list, list, list]:
    """(HTTP cases to load, skipped [{name, kind, reason}], unknown case ids)."""
    wanted = set(case_ids) if case_ids else None
    used, skipped = [], []
    for c in cases:
        if wanted is not None and c["name"] not in wanted:
            continue
        kind = c.get("kind") or "http"
        if kind != "http":
            skipped.append({"name": c["name"], "kind": kind, "reason": "not an HTTP case"})
            continue
        used.append(c)
    unknown = sorted(wanted - {c["name"] for c in cases}) if wanted else []
    return used, skipped, unknown


class Prepared:
    """One case ready to send many times."""

    __slots__ = ("route", "method", "path", "headers", "body", "expected")

    def __init__(self, case: dict, config: dict, side: dict, expected: int | None) -> None:
        self.route = route_key(case)
        self.method, self.path = base.map_route(
            case["method"].upper(), case["path"], side.get("route_map") or []
        )
        self.headers = base._merge_headers(
            config.get("default_headers") or {},
            side.get("default_headers") or {},
            case.get("headers") or {},
            side.get("request_headers") or {},
        )
        self.body = None
        if isinstance(case.get("raw_body"), str):
            self.body = case["raw_body"].encode()
        elif case.get("body") is not None:
            self.body = json.dumps(case["body"]).encode()
            if not any(k.lower() == "content-type" for k in self.headers):
                self.headers["Content-Type"] = "application/json"
        self.expected = expected


def run_level(
    port: int,
    prepared: list,
    concurrency: int,
    duration: float | None,
    requests: int | None,
    timeout: float,
) -> dict:
    lock = threading.Lock()
    lat: dict = {}
    mismatch: dict = {}
    transport: dict = {}
    sent = [0]
    deadline = None if duration is None else time.monotonic() + duration

    def take() -> Prepared | None:
        with lock:
            i = sent[0]
            if requests is not None and i >= requests:
                return None
            if deadline is not None and time.monotonic() >= deadline:
                return None
            sent[0] = i + 1
        return prepared[i % len(prepared)]

    def worker() -> None:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        try:
            while True:
                p = take()
                if p is None:
                    return
                t0 = time.perf_counter()
                try:
                    conn.request(p.method, p.path, body=p.body, headers=p.headers)
                    resp = conn.getresponse()
                    resp.read()
                except (OSError, http.client.HTTPException):
                    conn.close()
                    with lock:
                        transport[p.route] = transport.get(p.route, 0) + 1
                    continue
                ms = (time.perf_counter() - t0) * 1000
                with lock:
                    lat.setdefault(p.route, []).append(ms)
                    if p.expected is not None and resp.status != p.expected:
                        mismatch[p.route] = mismatch.get(p.route, 0) + 1
        finally:
            conn.close()

    cpu0, t0 = time.process_time(), time.monotonic()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - t0
    ratio = (time.process_time() - cpu0) / wall if wall > 0 else 0.0
    routes = {
        r: stats(lat.get(r, []), mismatch.get(r, 0), transport.get(r, 0), wall)
        for r in sorted(set(lat) | set(transport))
    }
    every = [ms for v in lat.values() for ms in v]
    return {
        "concurrency": concurrency,
        "requests_sent": sent[0],
        "wall_s": round(wall, 3),
        "client_cpu_ratio": round(ratio, 3),
        "load_generator_saturated": ratio >= SATURATION_RATIO,
        "overall": stats(every, sum(mismatch.values()), sum(transport.values()), wall),
        "routes": routes,
    }


def _warmup(
    side: dict, config: dict, port: int, used: list, stubs: dict, timeout: float
) -> tuple[dict, dict]:
    """(status per case in the first pass, learned upstream tables)."""
    tables: dict = {name: {} for name in stubs}
    warm: dict = {}
    for rnd in range(max(1, int(config.get("warmup_rounds") or 1))):
        for case in used:
            rec = base.replay_case(
                case,
                port,
                stubs,
                config.get("default_headers") or {},
                [],
                timeout,
                route_map=side.get("route_map") or [],
                side_headers=side.get("default_headers") or {},
                request_headers=side.get("request_headers") or {},
            )
            if rnd:
                continue
            warm[case["name"]] = rec.get("status")
            scripted = case.get("upstream_responses") or []
            if not isinstance(scripted, dict):
                scripted = {"default": scripted}
            used_n: dict = {}
            for call in rec.get("upstream_calls") or []:
                up = call.get("upstream") or "default"
                i = used_n.get(up, 0)
                used_n[up] = i + 1
                responses = scripted.get(up) or []
                if up in tables and i < len(responses):
                    tables[up].setdefault(
                        (call["method"], str(call["path"]).split("?")[0]), responses[i]
                    )
    return warm, tables


def run_perf_side(side: dict, config: dict, run_dir: str, nonce: str, references: dict) -> dict:
    key = side.get("key") or side.get("role") or "side"
    result: dict = {
        "role": side.get("role"),
        "key": key,
        "source_id": side.get("source_id"),
        "ok": False,
        "error": None,
        "levels": [],
        "peak_rss_kb": None,
    }
    used, skipped, unknown = select_cases(base.load_cases(side), config.get("case_ids"))
    result["cases_used"] = [c["name"] for c in used]
    result["cases_skipped"] = skipped
    if unknown:
        raise base.DriverError(f"{key}: case_ids not in the corpus: {unknown}")
    if not used:
        raise base.DriverError(f"{key}: no HTTP case to load")
    expected = dict(side.get("expected_status") or {})
    for c in used:
        if "expected_status" in c:
            expected.setdefault(c["name"], c["expected_status"])
    reference = references.get(side.get("reference_key") or "") or {}
    timeout = float(config.get("request_timeout") or 30)
    with base.running_service(side, config, run_dir, nonce, result) as (proc, port, stubs):
        warm, tables = _warmup(side, config, port, used, stubs, timeout)
        for c in used:
            if c["name"] not in expected and c["name"] in reference:
                expected[c["name"]] = reference[c["name"]]
        result["reference_status"] = warm
        result["warmup_status_mismatches"] = [
            {"name": c["name"], "expected": expected[c["name"]], "actual": warm.get(c["name"])}
            for c in used
            if c["name"] in expected and warm.get(c["name"]) != expected[c["name"]]
        ]
        for name, s in stubs.items():
            s.set_table(tables[name])
        prepared = [Prepared(c, config, side, expected.get(c["name"])) for c in used]
        duration, requests = config.get("duration_seconds"), config.get("requests")
        for level in config.get("levels") or [1]:
            result["levels"].append(
                run_level(
                    port,
                    prepared,
                    int(level),
                    None if duration is None else float(duration),
                    None if requests is None else int(requests),
                    timeout,
                )
            )
            code = base._exit_status(proc)  # never reap: _kill_group must signal the group first
            if code is not None:
                result["service_exited"] = code
                break
        result["peak_rss_kb"] = peak_rss_kb(proc.pid)
        result["upstream_table_misses"] = sum(s.table_misses for s in stubs.values())
        result["ok"] = True
    return result


def main(argv: list | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: perf_driver.py <run_dir>", file=sys.stderr)
        return 2
    run_dir = args[0]
    out: dict = {"driver_version": DRIVER_VERSION, "ok": False, "error": None, "sides": {}}
    started = time.monotonic()
    try:
        with open(os.path.join(run_dir, "driver_config.json"), encoding="utf-8") as fh:
            config = json.load(fh)
        sides = config.get("sides") or []
        if not sides:
            raise base.DriverError("driver_config has no sides")
        nonce = re.sub(r"[^a-z0-9]", "", os.path.basename(os.path.normpath(run_dir)).lower())
        references: dict = {}
        for side in sides:
            key = side.get("key") or side.get("role") or "side"
            try:
                res = run_perf_side(side, config, run_dir, nonce or "run", references)
            except base.DriverError as e:
                out["sides"][key] = {"key": key, "role": side.get("role"), "ok": False,
                                     "error": str(e), "levels": []}  # fmt: skip
                raise
            out["sides"][key] = res
            references[key] = res.get("reference_status") or {}
        out["ok"] = True
    except base.DriverError as e:
        out["error"] = str(e)
    except Exception as e:  # noqa: BLE001 - every failure must still produce result.json
        out["error"] = f"driver crashed: {type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()[-base.LOG_TAIL_CHARS :]
    out["duration"] = round(time.monotonic() - started, 3)
    tmp = os.path.join(run_dir, "result.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    os.replace(tmp, os.path.join(run_dir, "result.json"))
    if not out["ok"]:
        print(f"perf driver: {out['error']}", file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "sides": list(out["sides"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
