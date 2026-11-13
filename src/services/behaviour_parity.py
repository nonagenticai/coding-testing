"""Behaviour parity (py2rust A5c): record / replay / compare inside a coding workspace.

One run = ONE driver job in the workspace (``behaviour_parity_driver.py``, stdlib
only, shipped with this service). Per run a private directory
``/tmp/coding-testing/<nonce>/`` holds:

* ``parity_driver.py``    the driver itself,
* ``driver_config.json``  cases + the side(s) to run (source and/or target),
* ``result.json``         written by the driver: per side, per case status /
                          headers / body / upstream calls.

What to diff against and how (the run's spec) is kept service-side
(:mod:`services.run_spec_store`). The run id is ``<nonce>.<job_id>`` (async): the
job lives in coding-containers. The diff itself is :mod:`services.behaviour_diff` (pure library).

No shell: files are written and read back with ``python3 -c`` (zlib + base64;
writes are chunked so no single exec argument grows large) and the driver runs as ``python3 -I <driver>
<run_dir>``; both binaries are on the exec allowlist of every workspace image.
The driver itself needs ``ws-pg`` (python+rust workspace image) plus whatever the
service's own migrate/start commands use.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from models.parity_behaviour_models import (
    RUN_ID_RE,
    BehaviourCorpus,
    BehaviourOp,
    BehaviourOptions,
    BehaviourParityLinks,
    BehaviourParityResult,
    BehaviourParityRun,
    CompareRequest,
    RecordRequest,
    ReplayRequest,
    ServiceUnderTest,
)
from services.behaviour_diff import (
    DiffOptions,
    baseline_from_expected,
    diff_behaviour,
    has_expectations,
)
from services.workspace_driver_run import (
    READ_FILE_PY as _READ_FILE_PY,
)
from services.workspace_driver_run import (
    RUN_ROOT,
    SYNC_EXEC_MAX_TIMEOUT,
    WorkspaceDriverRuns,
    encode_chunks,
    job_exec_error,
    new_nonce,
)
from services.workspace_driver_run import (
    WRITE_FILE_PY as _WRITE_FILE_PY,
)

__all__ = [
    "RUN_ROOT",
    "SYNC_EXEC_MAX_TIMEOUT",
    "_READ_FILE_PY",
    "_WRITE_FILE_PY",
    "BehaviourParityRunner",
    "encode_chunks",
    "grade",
]

logger = logging.getLogger(__name__)

DRIVER_PATH = Path(__file__).resolve().parent / "behaviour_parity_driver.py"
DRIVER_NAME = "parity_driver.py"
LIB_HARNESS_PATH = Path(__file__).resolve().parent / "lib_call_harness.py"
LIB_HARNESS_NAME = "lib_call_harness.py"  # == behaviour_parity_driver.LIB_HARNESS_NAME
RECORDING_FORMAT = "coding-testing.behaviour-recording/1"
TAIL_CHARS = 4000
BEHAVIOUR_OPS = ("record", "replay", "compare")


def status_path(container_name: str, run_id: str) -> str:
    return f"/api/parity/{container_name}/runs/{run_id}"


def driver_source() -> str:
    return DRIVER_PATH.read_text(encoding="utf-8")


def lib_harness_source() -> str:
    return LIB_HARNESS_PATH.read_text(encoding="utf-8")


def side_config(role: str, svc: ServiceUnderTest) -> dict[str, Any]:
    if svc.kind == "cli":
        return {
            "role": role,
            "kind": "cli",
            "workdir": svc.absolute_workdir(),
            "start": list(svc.start),
            "env": dict(svc.env),
        }
    cfg: dict[str, Any] = {
        "role": role,
        "workdir": svc.absolute_workdir(),
        "start": list(svc.start),
        "migrate": list(svc.migrate) if svc.migrate else None,
        "env": dict(svc.env),
        "upstream_env": list(svc.upstream_env),
        "health_path": svc.health_path,
        "startup_timeout": svc.startup_timeout,
        "migrate_timeout": svc.migrate_timeout,
    }
    if svc.upstreams:
        cfg["upstreams"] = {k: list(v) for k, v in svc.upstreams.items()}
    if svc.request_headers:
        cfg["request_headers"] = dict(svc.request_headers)
    return cfg


def _corpus_cases(corpus: BehaviourCorpus) -> dict[str, Any]:
    if corpus.cases is not None:
        return {"cases": corpus.cases}
    return {"cases_file": f"/workspace/{corpus.cases_file}"}


def corpus_sides(
    corpora: list[BehaviourCorpus], target: ServiceUnderTest, with_sources: bool
) -> list[dict[str, Any]]:
    """Driver sides of a multi-corpus run: per corpus i, ``s<i>`` (compare) then ``t<i>``."""
    sides: list[dict[str, Any]] = []
    for i, corpus in enumerate(corpora):
        common = {**_corpus_cases(corpus), "default_headers": dict(corpus.default_headers)}
        if with_sources and corpus.source is not None:
            sides.append({**side_config("source", corpus.source), "key": f"s{i}", **common})
        sides.append(
            {
                **side_config("target", target),
                "key": f"t{i}",
                **common,
                "route_map": [
                    r.model_dump(by_alias=True, exclude_none=True) for r in corpus.route_map
                ],
            }
        )
    return sides


def build_driver_config(
    req: BehaviourOptions, sides: list[tuple[str, ServiceUnderTest]]
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "ws_pg": ["ws-pg"],
        "default_headers": dict(req.default_headers),
        "response_headers": [h.lower() for h in req.response_headers],
        "request_timeout": req.request_timeout,
    }
    corpora = getattr(req, "corpora", None)
    if corpora is not None:
        target = req.target  # type: ignore[attr-defined]
        cfg["sides"] = corpus_sides(corpora, target, isinstance(req, CompareRequest))
        return cfg
    cfg["sides"] = [side_config(role, svc) for role, svc in sides]
    if req.cases is not None:
        cfg["cases"] = req.cases
    else:
        cfg["cases_file"] = f"/workspace/{req.cases_file}"
    return cfg


def build_spec(op: BehaviourOp, req: BehaviourOptions) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "op": op,
        "ignore_paths": list(req.ignore_paths),
        "upstream_ignore_paths": list(req.upstream_ignore_paths),
        "upstream_headers": None if req.upstream_headers is None else list(req.upstream_headers),
        "response_headers": [h.lower() for h in req.response_headers],
        "baseline": {"record": "expected", "compare": "source"}.get(op, "expected"),
    }
    if req.frame_ignore_paths:
        spec["frame_ignore_paths"] = list(req.frame_ignore_paths)
    if req.drop_frames:
        spec["drop_frames"] = [r.model_dump() for r in req.drop_frames]
    if isinstance(req, RecordRequest) and req.source_id:
        spec["source_id"] = req.source_id
    corpora = getattr(req, "corpora", None)
    if corpora is not None:
        kinds = {c.baseline for c in corpora}
        spec["baseline"] = (
            "source" if op == "compare" else (kinds.pop() if len(kinds) == 1 else "mixed")
        )
        spec["corpora"] = [
            {
                "source_id": c.source_id,
                "source_key": f"s{i}" if op == "compare" else None,
                "target_key": f"t{i}",
                "baseline": "source" if op == "compare" else c.baseline,
                "recording": c.recording,
                "recording_run_id": c.recording_run_id,
                "ignore_upstreams": list(c.ignore_upstreams),
            }
            for i, c in enumerate(corpora)
        ]
        return spec
    if isinstance(req, ReplayRequest):
        spec["baseline"] = req.baseline
        spec["recording"] = req.recording
        spec["recording_run_id"] = req.recording_run_id
    return spec


def _recording_cases(recording: Any) -> list[dict[str, Any]]:
    if (
        not isinstance(recording, dict)
        or recording.get("format") != RECORDING_FORMAT
        or not isinstance(recording.get("cases"), list)
        or not recording["cases"]
    ):
        raise ValueError(f"recording is not a non-empty {RECORDING_FORMAT} object")
    return recording["cases"]


def make_recording(side: dict[str, Any], source_id: str | None = None) -> dict[str, Any]:
    rec: dict[str, Any] = {"format": RECORDING_FORMAT, "cases": side.get("cases") or []}
    if source_id:
        rec["source_id"] = source_id
    return rec


def _side_tails(result: dict[str, Any]) -> str:
    parts = []
    for key, side in (result.get("sides") or {}).items():
        if isinstance(side, dict) and side.get("log_tail"):
            parts.append(f"--- {key} service log ---\n{side['log_tail']}")
    return "\n".join(parts)


def _diff_options(spec: dict[str, Any], ignore_upstreams: list[str] | None = None) -> DiffOptions:
    return DiffOptions.build(
        spec.get("ignore_paths"),
        spec.get("upstream_ignore_paths"),
        spec.get("upstream_headers"),
        spec.get("response_headers"),
        ignore_upstreams,
        spec.get("frame_ignore_paths"),
        spec.get("drop_frames"),
    )


def grade(
    spec: dict[str, Any],
    result: dict[str, Any] | None,
    *,
    exit_code: int | None,
    job_output: str = "",
    exec_error: str | None = None,
    recording_from_run: dict[str, Any] | None = None,
    recordings_from_runs: dict[str, dict[str, Any] | None] | None = None,
) -> BehaviourParityResult:
    """Turn a driver result + run spec into a verdict. Pure (no I/O).

    ``recordings_from_runs`` (multi-corpus replay): ``recording_run_id`` -> the
    recording read from that run, or None when it could not be read.
    """
    op: BehaviourOp = spec["op"]
    baseline_kind = spec.get("baseline") or "expected"
    opts = _diff_options(spec)
    tail = (job_output or "")[-TAIL_CHARS:]

    def not_computed(error: str, recording: dict[str, Any] | None = None) -> BehaviourParityResult:
        return BehaviourParityResult(
            op=op,
            baseline=baseline_kind,
            computed=False,
            equal=False,
            error=error,
            exit_code=exit_code,
            recording=recording,
            driver_log_tail=(tail + "\n" + _side_tails(result or {}))[-TAIL_CHARS:].strip(),
        )

    if exec_error:
        return not_computed(exec_error)
    if not isinstance(result, dict):
        return not_computed(f"the driver wrote no result (exit {exit_code})")
    if not result.get("ok") or exit_code != 0:
        return not_computed(str(result.get("error") or f"driver exited {exit_code}"))
    sides = result.get("sides") or {}
    if spec.get("corpora") is not None:
        return _grade_corpora(
            spec, result, opts, exit_code, tail, recordings_from_runs or {}, not_computed
        )
    cases = result.get("cases") or []

    recording = None
    if op in ("record", "compare"):
        recording = make_recording(sides.get("source") or {}, spec.get("source_id"))
    actual_role = "source" if op == "record" else "target"
    actual = (sides.get(actual_role) or {}).get("cases")
    if not isinstance(actual, list) or len(actual) != len(cases) or not cases:
        return not_computed(f"{actual_role} side recorded no complete case list", recording)

    try:
        if op == "compare":
            baseline = (sides.get("source") or {}).get("cases") or []
        elif op == "replay" and baseline_kind == "recording":
            if spec.get("recording") is not None:
                baseline = _recording_cases(spec["recording"])
            else:
                if recording_from_run is None:
                    return not_computed("recording_run_id: the recorded run has no result")
                baseline = _recording_cases(recording_from_run)
        else:  # expected
            if op == "record" and not all(has_expectations(c) for c in cases):
                return BehaviourParityResult(
                    op=op,
                    baseline="none",
                    computed=True,
                    equal=None,
                    cases_total=len(actual),
                    recording=recording,
                    exit_code=exit_code,
                    driver_log_tail=tail,
                )
            baseline = baseline_from_expected(cases)
    except ValueError as e:
        return not_computed(str(e), recording)

    diff = diff_behaviour(baseline, actual, opts)
    return BehaviourParityResult.model_validate(
        {
            **diff,
            "op": op,
            "baseline": baseline_kind,
            "recording": recording,
            "exit_code": exit_code,
            "driver_log_tail": tail,
        }
    )


def _grade_corpora(
    spec: dict[str, Any],
    result: dict[str, Any],
    opts: DiffOptions,
    exit_code: int | None,
    tail: str,
    recordings_from_runs: dict[str, dict[str, Any] | None],
    not_computed: Any,
) -> BehaviourParityResult:
    """Multi-corpus replay/compare: one diff per corpus, attributed to its source_id."""
    op = spec["op"]
    sides = result.get("sides") or {}
    case_sets = result.get("case_sets") or {}
    outcomes: list[dict[str, Any]] = []
    all_cases: list[dict[str, Any]] = []
    recordings: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for corpus in spec["corpora"]:
        sid = corpus["source_id"]
        out: dict[str, Any] = {"source_id": sid, "baseline": corpus["baseline"]}
        actual = (sides.get(corpus["target_key"]) or {}).get("cases")
        cases = case_sets.get(corpus["target_key"]) or []
        try:
            if op == "compare":
                src_side = sides.get(corpus["source_key"]) or {}
                recordings[sid] = make_recording(src_side, sid)
                baseline = src_side.get("cases") or []
            elif corpus["baseline"] == "recording":
                if corpus.get("recording") is not None:
                    baseline = _recording_cases(corpus["recording"])
                else:
                    rec = recordings_from_runs.get(str(corpus.get("recording_run_id")))
                    if rec is None:
                        raise ValueError("recording_run_id: the recorded run has no result")
                    baseline = _recording_cases(rec)
            else:
                baseline = baseline_from_expected(cases)
            if not isinstance(actual, list) or not cases or len(actual) != len(cases):
                raise ValueError("target side recorded no complete case list")
        except ValueError as e:
            out.update(computed=False, equal=False, error=f"{sid}: {e}")
            errors.append(out["error"])
            outcomes.append(out)
            continue
        diff = diff_behaviour(
            baseline,
            actual,
            DiffOptions(
                ignore_paths=opts.ignore_paths,
                upstream_ignore_paths=opts.upstream_ignore_paths,
                upstream_headers=opts.upstream_headers,
                response_headers=opts.response_headers,
                ignore_upstreams=tuple(corpus.get("ignore_upstreams") or ()),
                frame_ignore_paths=opts.frame_ignore_paths,
                drop_frames=opts.drop_frames,
            ),
            source_id=sid,
        )
        out.update(
            computed=diff["computed"],
            equal=diff["equal"],
            cases_total=diff["cases_total"],
            cases_passed=diff["cases_passed"],
            cases_failed=diff["cases_failed"],
        )
        outcomes.append(out)
        all_cases.extend(diff["cases"])
    if errors:
        res = not_computed("; ".join(errors))
        return BehaviourParityResult.model_validate(
            {
                **res.model_dump(),
                "corpora": outcomes,
                "cases": all_cases,
                "recordings": recordings or None,
            }
        )
    total = len(all_cases)
    passed = sum(1 for c in all_cases if c["passed"])
    computed = total > 0 and all(o["computed"] for o in outcomes)
    return BehaviourParityResult.model_validate(
        {
            "op": op,
            "baseline": spec["baseline"],
            "computed": computed,
            "equal": computed and passed == total,
            "cases_total": total,
            "cases_passed": passed,
            "cases_failed": total - passed,
            "cases": all_cases,
            "corpora": outcomes,
            "recordings": recordings or None,
            "exit_code": exit_code,
            "driver_log_tail": tail,
        }
    )


class BehaviourParityRunner(WorkspaceDriverRuns):
    async def start(
        self,
        op: BehaviourOp,
        container: str,
        tenant_id: str | None,
        req: RecordRequest | ReplayRequest | CompareRequest,
    ) -> BehaviourParityRun:
        if not req.run_async and req.timeout > SYNC_EXEC_MAX_TIMEOUT:
            raise HTTPException(
                status_code=422, detail=f"timeout > {SYNC_EXEC_MAX_TIMEOUT}s needs async=true"
            )
        sides: list[tuple[str, ServiceUnderTest]]
        if isinstance(req, CompareRequest):
            sides = [("source", req.source), ("target", req.target)] if req.source else []
        elif isinstance(req, RecordRequest):
            sides = [("source", req.source)]
        else:
            sides = [("target", req.target)]
        rids = [getattr(req, "recording_run_id", None)]
        rids += [c.recording_run_id for c in getattr(req, "corpora", None) or []]
        for rid in rids:
            if rid:
                await self.recording_nonce(container, tenant_id, rid)
        nonce = new_nonce()
        run_dir = await self.stage(
            container,
            tenant_id,
            nonce,
            {
                DRIVER_NAME: driver_source(),
                LIB_HARNESS_NAME: lib_harness_source(),
                "driver_config.json": json.dumps(build_driver_config(req, sides)),
            },
            kind=op,
            spec=build_spec(op, req),
        )
        # -I: isolated mode, so nothing in the checkout or PYTHON* env can shadow a stdlib
        # module the driver imports (its own dir is not put on sys.path either).
        argv = ["python3", "-I", f"{run_dir}/{DRIVER_NAME}", run_dir]

        if req.run_async:
            run_id = await self.submit(container, tenant_id, nonce, argv, req.timeout)
            return BehaviourParityRun(
                run_id=run_id,
                container_name=container,
                op=op,
                status="queued",
                links=BehaviourParityLinks(status=status_path(container, run_id)),
            )

        payload = await self._exec(container, tenant_id, argv, req.timeout, "Parity driver")
        result = await self._grade(container, tenant_id, nonce, payload, None)
        return BehaviourParityRun(
            run_id=nonce,
            container_name=container,
            op=op,
            status="completed" if result.computed else "failed",
            result=result,
        )

    async def _recording_of_run(
        self, container: str, tenant_id: str | None, rid: str
    ) -> dict[str, Any] | None:
        m = RUN_ID_RE.match(str(rid))
        if not m:
            return None
        other, _ = await self._read_json(
            container,
            tenant_id,
            f"{RUN_ROOT}/{m.group('nonce')}/result.json",
            "Recorded run result",
        )
        src = ((other or {}).get("sides") or {}).get("source")
        if other and other.get("ok") and isinstance(src, dict):
            return make_recording(src)
        return None

    async def _grade(
        self,
        container: str,
        tenant_id: str | None,
        nonce: str,
        job: dict[str, Any],
        exec_error: str | None,
    ) -> BehaviourParityResult:
        spec = await self.spec(container, tenant_id, nonce, BEHAVIOUR_OPS)
        if spec is None:
            raise HTTPException(status_code=404, detail="Run not found in this workspace")
        result, result_error = await self._read_json(
            container, tenant_id, f"{RUN_ROOT}/{nonce}/result.json", "Driver result"
        )
        recording_from_run = None
        recordings_from_runs: dict[str, dict[str, Any] | None] = {}
        rid = spec.get("recording_run_id")
        if spec.get("op") == "replay" and spec.get("baseline") == "recording" and rid:
            recording_from_run = await self._recording_of_run(container, tenant_id, rid)
        for corpus in spec.get("corpora") or []:
            crid = corpus.get("recording_run_id")
            if corpus.get("baseline") == "recording" and crid:
                recordings_from_runs[crid] = await self._recording_of_run(
                    container, tenant_id, crid
                )
        exit_code = job.get("exit_code") if isinstance(job.get("exit_code"), int) else None
        output = (job.get("stdout") or "") + (job.get("stderr") or "")
        if result is None and not exec_error:
            exec_error = f"{result_error}; driver exit {exit_code}: {output[-1000:]}"
        return grade(
            spec,
            result,
            exit_code=exit_code,
            job_output=output,
            exec_error=exec_error,
            recording_from_run=recording_from_run,
            recordings_from_runs=recordings_from_runs,
        )

    async def poll(self, container: str, tenant_id: str | None, run_id: str) -> BehaviourParityRun:
        nonce, job = await self.job(container, tenant_id, run_id)
        links = BehaviourParityLinks(status=status_path(container, run_id))
        status = job["status"]
        if status in ("queued", "running"):
            spec = await self.spec(container, tenant_id, nonce, BEHAVIOUR_OPS)
            if spec is None:
                raise HTTPException(status_code=404, detail="Run not found in this workspace")
            return BehaviourParityRun(
                run_id=run_id, container_name=container, op=spec["op"], status=status, links=links
            )
        result = await self._grade(container, tenant_id, nonce, job, job_exec_error(job))
        return BehaviourParityRun(
            run_id=run_id,
            container_name=container,
            op=result.op,
            status="completed" if result.computed else "failed",
            result=result,
            links=links,
        )
