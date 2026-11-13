"""Performance comparison: N sources vs ONE target under the same load, same pod.

One run = ONE async driver job (``perf_driver.py`` + the behaviour driver whose
service startup it reuses, both stdlib) through :class:`WorkspaceDriverRuns`. Per
corpus ``i`` the driver runs source ``s<i>`` then a fresh target ``t<i>``
(route-mapped), sequentially, at identical concurrency levels.

:func:`grade_perf` (pure) turns the driver's per-side stats into target/source
ratios per corpus, level and route, plus ``regressions`` beyond the thresholds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from models.parity_perf_models import (
    PerfRequest,
    PerfResult,
    PerfRun,
    PerfRunLinks,
    PerfThresholds,
)
from services.behaviour_parity import DRIVER_NAME, driver_source, side_config
from services.workspace_driver_run import WorkspaceDriverRuns, job_exec_error, new_nonce, run_dir

PERF_DRIVER_PATH = Path(__file__).resolve().parent / "perf_driver.py"
PERF_DRIVER_NAME = "perf_driver.py"
TAIL_CHARS = 4000
RUN_KIND = "perf"


def status_path(container_name: str, run_id: str) -> str:
    return f"/api/parity/{container_name}/perf/runs/{run_id}"


def statuses_of_recording(recording: Any) -> dict[str, int]:
    """{case name: status} of a behaviour recording (or a recorded side's result)."""
    cases = recording.get("cases") if isinstance(recording, dict) else None
    if not isinstance(cases, list):
        raise HTTPException(status_code=422, detail="recording has no cases list")
    return {
        str(c["name"]): c["status"]
        for c in cases
        if isinstance(c, dict) and "name" in c and isinstance(c.get("status"), int)
    }


def build_perf_config(req: PerfRequest, expected: list[dict[str, int]]) -> dict[str, Any]:
    """Driver config; ``expected[i]``: corpus i's recorded statuses (may be empty)."""
    sides: list[dict[str, Any]] = []
    for i, corpus in enumerate(req.corpora or []):
        assert corpus.source is not None  # nosec B101 - PerfRequest validated it
        common = {
            **(
                {"cases": corpus.cases}
                if corpus.cases is not None
                else {"cases_file": f"/workspace/{corpus.cases_file}"}
            ),
            "source_id": corpus.source_id,
            "default_headers": dict(corpus.default_headers),
            "expected_status": expected[i],
        }
        sides.append({**side_config("source", corpus.source), "key": f"s{i}", **common})
        sides.append(
            {
                **side_config("target", req.target),
                "key": f"t{i}",
                **common,
                "reference_key": f"s{i}",
                "route_map": [
                    r.model_dump(by_alias=True, exclude_none=True) for r in corpus.route_map
                ],
            }
        )
    return {
        "ws_pg": ["ws-pg"],
        "default_headers": dict(req.default_headers),
        "request_timeout": req.request_timeout,
        "levels": list(req.concurrency),
        "duration_seconds": req.duration_seconds,
        "requests": req.requests,
        "warmup_rounds": req.warmup_rounds,
        "case_ids": list(req.case_ids) if req.case_ids else None,
        "sides": sides,
    }


def build_perf_spec(req: PerfRequest) -> dict[str, Any]:
    return {
        "op": RUN_KIND,
        "thresholds": req.thresholds.model_dump(),
        "corpora": [
            {"source_id": c.source_id, "source_key": f"s{i}", "target_key": f"t{i}"}
            for i, c in enumerate(req.corpora or [])
        ],
    }


def _ratio(target: Any, source: Any) -> float | None:
    if target is None or source is None or source == 0:
        return None
    return round(target / source, 4)


def _error_rate(stats: dict[str, Any]) -> float | None:
    total = int(stats.get("count") or 0) + int(stats.get("transport_errors") or 0)
    return None if total == 0 else round(int(stats.get("errors") or 0) / total, 6)


def _compare(
    sid: str,
    level: int,
    route: str,
    src: dict[str, Any],
    tgt: dict[str, Any],
    th: PerfThresholds,
    regressions: list[dict[str, Any]],
    saturated: bool = False,
) -> dict[str, Any]:
    cmp = {
        "source_id": sid,
        "concurrency": level,
        "route": route,
        "source": src,
        "target": tgt,
        "throughput_ratio": _ratio(tgt.get("throughput_rps"), src.get("throughput_rps")),
        **{
            f"{m}_ratio": _ratio(tgt.get(f"{m}_ms"), src.get(f"{m}_ms"))
            for m in ("p50", "p95", "p99", "max")
        },
        "source_error_rate": _error_rate(src),
        "target_error_rate": _error_rate(tgt),
        # The load generator (not the service) was the bottleneck on a side at this
        # level: nothing measured there is the service's; no verdict, no regression.
        "saturated": saturated,
    }
    if saturated:
        return cmp

    def flag(metric: str, s: Any, t: Any, ratio: Any, threshold: float) -> None:
        regressions.append(
            {
                "source_id": sid,
                "concurrency": level,
                "route": route,
                "metric": metric,
                "source": s,
                "target": t,
                "ratio": ratio,
                "threshold": threshold,
            }
        )

    se, te = cmp["source_error_rate"], cmp["target_error_rate"]
    if te is not None and te - (se or 0.0) > th.error_rate_delta:
        flag("error_rate", se, te, None, th.error_rate_delta)
    # Throughput is completed requests over the level's time window: it needs no
    # sample floor (a slow target completing few requests IS the regression).
    r = _ratio(tgt.get("throughput_rps"), src.get("throughput_rps"))
    if r is not None and r < th.throughput_ratio:
        flag(
            "throughput",
            src.get("throughput_rps"),
            tgt.get("throughput_rps"),
            r,
            th.throughput_ratio,
        )
    # Percentiles of too few samples are not a measurement: the comparison is
    # under-sampled and the run is not computed (never a pass).
    cmp["under_sampled"] = (
        min(int(src.get("count") or 0), int(tgt.get("count") or 0)) < th.min_route_samples
    )
    if cmp["under_sampled"]:
        return cmp
    for metric, limit in (("p50", th.p50_ratio), ("p95", th.p95_ratio), ("p99", th.p99_ratio)):
        r = _ratio(tgt.get(f"{metric}_ms"), src.get(f"{metric}_ms"))
        if r is not None and r > limit:
            flag(metric, src.get(f"{metric}_ms"), tgt.get(f"{metric}_ms"), r, limit)
    return cmp


def grade_perf(
    spec: dict[str, Any],
    result: dict[str, Any] | None,
    *,
    exit_code: int | None,
    job_output: str = "",
    exec_error: str | None = None,
) -> PerfResult:
    """Driver result + run spec -> ratios and regressions. Pure (no I/O)."""
    th = PerfThresholds.model_validate(spec.get("thresholds") or {})
    sides = (result or {}).get("sides") or {}
    tail = job_output[-TAIL_CHARS:]

    def not_computed(error: str) -> PerfResult:
        logs = "\n".join(
            f"--- {k} service log ---\n{v['log_tail']}"
            for k, v in sides.items()
            if isinstance(v, dict) and v.get("log_tail")
        )
        return PerfResult.model_validate(
            {
                "computed": False,
                "error": error,
                "sides": {k: v for k, v in sides.items() if isinstance(v, dict)},
                "thresholds": th,
                "exit_code": exit_code,
                "driver_log_tail": (tail + "\n" + logs)[-TAIL_CHARS:].strip(),
            }
        )

    if exec_error:
        return not_computed(exec_error)
    if not isinstance(result, dict):
        return not_computed(f"the driver wrote no result (exit {exit_code})")
    if not result.get("ok") or exit_code != 0:
        return not_computed(str(result.get("error") or f"driver exited {exit_code}"))

    comparisons: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    rss: list[dict[str, Any]] = []
    for corpus in spec.get("corpora") or []:
        sid = corpus["source_id"]
        s_side, t_side = sides.get(corpus["source_key"]), sides.get(corpus["target_key"])
        if not isinstance(s_side, dict) or not isinstance(t_side, dict):
            return not_computed(f"{sid}: a side is missing from the driver result")
        s_levels = {lv["concurrency"]: lv for lv in s_side.get("levels") or []}
        t_levels = {lv["concurrency"]: lv for lv in t_side.get("levels") or []}
        if not s_levels or set(s_levels) != set(t_levels):
            return not_computed(f"{sid}: source and target did not run the same levels")
        for level in sorted(s_levels):
            s_lv, t_lv = s_levels[level], t_levels[level]
            sat = bool(s_lv.get("load_generator_saturated") or t_lv.get("load_generator_saturated"))
            comparisons.append(
                _compare(sid, level, "*", s_lv["overall"], t_lv["overall"], th, regressions, sat)
            )
            s_routes, t_routes = s_lv.get("routes") or {}, t_lv.get("routes") or {}
            for route in sorted(set(s_routes) | set(t_routes)):
                comparisons.append(
                    _compare(
                        sid,
                        level,
                        route,
                        s_routes.get(route) or {},
                        t_routes.get(route) or {},
                        th,
                        regressions,
                        sat,
                    )
                )
        s_rss, t_rss = s_side.get("peak_rss_kb"), t_side.get("peak_rss_kb")
        r = _ratio(t_rss, s_rss)
        rss.append({"source_id": sid, "source_kb": s_rss, "target_kb": t_rss, "ratio": r})
        if r is not None and r > th.rss_ratio:
            regressions.append(
                {
                    "source_id": sid,
                    "concurrency": None,
                    "route": "*",
                    "metric": "rss",
                    "source": s_rss,
                    "target": t_rss,
                    "ratio": r,
                    "threshold": th.rss_ratio,
                }
            )
    idle = [
        c
        for c in comparisons
        if c["route"] == "*" and not (c["source"].get("count") and c["target"].get("count"))
    ]
    thin = [c for c in comparisons if c.get("under_sampled")]
    busy = [c for c in comparisons if c["saturated"] and c["route"] == "*"]
    error = None
    if idle:
        error = "a side completed no request at some level"
    elif busy:
        error = (
            f"the load generator was saturated (client_cpu_ratio >= 0.9) at "
            f"{', '.join(f'{c["source_id"]} concurrency {c["concurrency"]}' for c in busy)}: "
            "the driver, not the service, was the bottleneck; lower concurrency"
        )
    elif thin:
        error = (
            f"{len(thin)} comparison(s) have fewer than min_route_samples "
            f"{th.min_route_samples} requests on a side (e.g. {thin[0]['source_id']} "
            f"concurrency {thin[0]['concurrency']} {thin[0]['route']}): latency is not "
            "comparable; raise duration_seconds / requests"
        )
    computed = bool(comparisons) and error is None
    return PerfResult.model_validate(
        {
            "computed": computed,
            "passed": computed and not regressions,
            "error": error,
            "sides": sides,
            "comparisons": comparisons,
            "rss": rss,
            "regressions": regressions,
            "thresholds": th,
            "exit_code": exit_code,
            "driver_log_tail": tail,
        }
    )


class PerfRunner(WorkspaceDriverRuns):
    async def _recorded_statuses(
        self, container: str, tenant_id: str | None, run_id: str
    ) -> dict[str, int]:
        """Statuses of the SOURCE side of a finished record/compare run in this workspace."""
        nonce = await self.recording_nonce(container, tenant_id, run_id)
        doc, err = await self._read_json(
            container, tenant_id, f"{run_dir(nonce)}/result.json", "Recorded run result"
        )
        source = ((doc or {}).get("sides") or {}).get("source")
        if not isinstance(source, dict):
            raise HTTPException(
                status_code=422,
                detail=f"recording_run_id {run_id}: no recorded source side ({err})",
            )
        return statuses_of_recording(source)

    async def start(self, container: str, tenant_id: str | None, req: PerfRequest) -> PerfRun:
        expected: list[dict[str, int]] = []
        for corpus in req.corpora or []:
            if corpus.recording is not None:
                expected.append(statuses_of_recording(corpus.recording))
            elif corpus.recording_run_id:
                expected.append(
                    await self._recorded_statuses(container, tenant_id, corpus.recording_run_id)
                )
            else:
                expected.append({})
        nonce = new_nonce()
        directory = await self.stage(
            container,
            tenant_id,
            nonce,
            {
                DRIVER_NAME: driver_source(),
                PERF_DRIVER_NAME: PERF_DRIVER_PATH.read_text(encoding="utf-8"),
                "driver_config.json": json.dumps(build_perf_config(req, expected)),
            },
            kind=RUN_KIND,
            spec=build_perf_spec(req),
        )
        argv = ["python3", "-I", f"{directory}/{PERF_DRIVER_NAME}", directory]
        run_id = await self.submit(container, tenant_id, nonce, argv, req.timeout)
        return PerfRun(
            run_id=run_id,
            container_name=container,
            status="queued",
            links=PerfRunLinks(status=status_path(container, run_id)),
        )

    async def poll(self, container: str, tenant_id: str | None, run_id: str) -> PerfRun:
        nonce, job = await self.job(container, tenant_id, run_id)
        spec = await self.spec(container, tenant_id, nonce, (RUN_KIND,))
        if spec is None:
            raise HTTPException(status_code=404, detail="Perf run not found in this workspace")
        links = PerfRunLinks(status=status_path(container, run_id))
        if job["status"] in ("queued", "running"):
            return PerfRun(
                run_id=run_id, container_name=container, status=job["status"], links=links
            )
        result, result_error = await self._read_json(
            container, tenant_id, f"{run_dir(nonce)}/result.json", "Perf driver result"
        )
        exit_code = job.get("exit_code") if isinstance(job.get("exit_code"), int) else None
        output = (job.get("stdout") or "") + (job.get("stderr") or "")
        exec_error = job_exec_error(job)
        if result is None and not exec_error:
            exec_error = f"{result_error}; driver exit {exit_code}: {output[-1000:]}"
        graded = grade_perf(
            spec, result, exit_code=exit_code, job_output=output, exec_error=exec_error
        )
        return PerfRun(
            run_id=run_id,
            container_name=container,
            status="completed" if graded.computed else "failed",
            result=graded,
            links=links,
        )
