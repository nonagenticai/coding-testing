"""Performance parity: the REAL perf driver runs a stdlib source and target under load
in a local workspace (tests/local_workspace.py), each on a fresh database of the test's
real Postgres (ws-pg shim); the REAL grader compares them.

Negative controls: a target slowed by 40 ms per request must be flagged as a latency
regression, and a target answering 500 where the source answers 200 as an error-rate
regression."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from models.parity_perf_models import PerfRequest
from routers.parity_perf import get_perf_runner
from services import workspace_driver_run as wdr
from services.perf_driver import percentile, select_cases
from services.perf_parity import PerfRunner, grade_perf
from tests.local_workspace import LocalWorkspace

pytestmark = pytest.mark.usefixtures("as_writer", "pg_db")

DATA = Path(__file__).resolve().parent.parent / "data"
CASES = [
    {"name": "list", "method": "GET", "path": "/items"},
    {"name": "create", "method": "POST", "path": "/items", "body": {"n": 1},
     "upstream_responses": [{"status": 200, "body": {"enriched": True}}]},
    {"name": "stream", "kind": "sse", "method": "GET", "path": "/events"},
]  # fmt: skip
LOOSE = {"p50_ratio": 3, "p95_ratio": 3, "p99_ratio": 5, "throughput_ratio": 0.3, "rss_ratio": 3}


@pytest.fixture
def workspace(tmp_path, monkeypatch) -> LocalWorkspace:
    monkeypatch.setattr(wdr, "RUN_ROOT", str(tmp_path / "runs"))
    root = tmp_path / "ws"
    root.mkdir()
    shutil.copy(DATA / "perf" / "svc.py", root / "svc.py")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "ws-pg").write_text(
        f"#!{sys.executable}\nimport runpy, sys\n"
        f"runpy.run_path({str(DATA / 'behaviour_parity' / 'ws_pg_shim.py')!r}, run_name='__main__')\n"
    )
    (bin_dir / "ws-pg").chmod(0o755)
    (bin_dir / "python3").symlink_to(sys.executable)
    admin = os.environ["DATABASE_URL"].replace("+psycopg", "")
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "WS_PG_SHIM_ADMIN_URL": admin}
    return LocalWorkspace(root, env=env)


def _body(**target_env: str) -> dict[str, Any]:
    return {
        "corpora": [
            {
                "source_id": "notes",
                "cases": CASES,
                "route_map": {"/": "/t/"},
                # 5 ms per request keeps the (GIL-bound) load generator below
                # saturation, which would make the run not computed.
                "source": {"start": ["python3", "svc.py"], "env": {"DELAY_MS": "5"}},
            }
        ],
        "target": {
            "start": ["python3", "svc.py"],
            "env": {"PREFIX": "/t", "DELAY_MS": "5", **target_env},
        },
        "concurrency": [1, 2],
        "requests": 60,  # 30 per route per level: >= the default min_route_samples 20
        "thresholds": LOOSE,
        "timeout": 120,
    }


def _req(**target_env: str) -> PerfRequest:
    return PerfRequest.model_validate(_body(**target_env))


async def _run(ws: LocalWorkspace, req: PerfRequest) -> Any:
    runner = PerfRunner(ws)
    run = await runner.start("ws1", "t1", req)
    assert run.status == "queued"
    return await runner.poll("ws1", "t1", run.run_id)


async def test_equal_source_and_target_pass(workspace) -> None:
    out = await _run(workspace, _req())
    r = out.result
    assert out.status == "completed" and r.computed, r.error or r.driver_log_tail
    assert r.passed, r.regressions
    s0, t0 = r.sides["s0"], r.sides["t0"]
    assert [lv.concurrency for lv in t0.levels] == [1, 2]
    assert t0.cases_skipped[0]["name"] == "stream"  # SSE is not loaded
    overall = t0.levels[0].overall
    assert overall.count > 0 and overall.errors == 0 and overall.p50_ms is not None
    # The learned upstream table answered every POST under load.
    assert t0.upstream_table_misses == 0 and s0.upstream_table_misses == 0
    assert {c.route for c in r.comparisons} == {"*", "GET /items", "POST /items"}


async def test_slow_target_is_a_latency_regression(workspace) -> None:
    # Default thresholds; 100 ms per request against a fast source, 20 requests per
    # route on both sides.
    body = {**_body(DELAY_MS="100"), "concurrency": [4], "requests": 40}
    del body["thresholds"]
    out = await _run(workspace, PerfRequest.model_validate(body))
    r = out.result
    assert r.computed and not r.passed, r.error
    assert {"p50", "throughput"} <= {g.metric for g in r.regressions}


async def test_under_sampled_slow_target_never_passes(workspace) -> None:
    # Default thresholds; a 300 ms target completes only a few requests in the window:
    # its throughput is a regression and its latency is not comparable -> not computed.
    body = {**_body(DELAY_MS="300"), "duration_seconds": 0.5}
    del body["thresholds"], body["requests"]
    out = await _run(workspace, PerfRequest.model_validate(body))
    r = out.result
    assert out.status == "failed" and not r.computed and not r.passed
    assert "min_route_samples" in (r.error or "")
    assert "throughput" in {g.metric for g in r.regressions}


def test_grade_at_defaults_flags_a_starved_target() -> None:
    """5000 source requests at 500 rps vs 10 target requests at 1 rps (default thresholds)."""

    def level(count: int, rps: float, ms: float) -> dict[str, Any]:
        stats = {"count": count, "errors": 0, "transport_errors": 0, "throughput_rps": rps,
                 "p50_ms": ms, "p95_ms": ms, "p99_ms": ms, "max_ms": ms}  # fmt: skip
        return {"concurrency": 1, "wall_s": 10.0, "client_cpu_ratio": 0.1,
                "load_generator_saturated": False, "overall": stats,
                "routes": {"GET /x": stats}}  # fmt: skip

    spec = {
        "thresholds": {},
        "corpora": [{"source_id": "a", "source_key": "s0", "target_key": "t0"}],
    }
    result = {
        "ok": True,
        "sides": {
            "s0": {"key": "s0", "ok": True, "levels": [level(5000, 500.0, 2.0)]},
            "t0": {"key": "t0", "ok": True, "levels": [level(10, 1.0, 1000.0)]},
        },
    }
    r = grade_perf(spec, result, exit_code=0)
    assert not r.passed and not r.computed
    assert all(c.under_sampled for c in r.comparisons)
    assert "throughput" in {g.metric for g in r.regressions}


async def test_wrong_status_is_an_error_rate_regression(workspace) -> None:
    out = await _run(workspace, _req(FAIL_GET="1"))
    r = out.result
    assert r.computed and not r.passed
    assert any(g.metric == "error_rate" and g.route == "GET /items" for g in r.regressions)
    assert r.sides["t0"].warmup_status_mismatches[0]["name"] == "list"


async def test_poll_is_bound_to_the_workspace(workspace) -> None:
    from fastapi import HTTPException

    runner = PerfRunner(workspace)
    run = await runner.start("ws1", "t1", _req())
    with pytest.raises(HTTPException) as e:
        await runner.poll("ws2", "t1", run.run_id)
    assert e.value.status_code == 404


def test_grade_without_result_is_not_computed() -> None:
    spec = {"thresholds": {}, "corpora": []}
    assert not grade_perf(spec, None, exit_code=1).computed
    assert "boom" in (grade_perf(spec, {"ok": False, "error": "boom"}, exit_code=1).error or "")
    assert grade_perf(spec, None, exit_code=None, exec_error="lost").error == "lost"


def test_driver_helpers() -> None:
    assert percentile([], 50) is None
    assert percentile([1, 2, 3, 4], 50) == 2 and percentile([1, 2, 3, 4], 99) == 4
    used, skipped, unknown = select_cases(CASES, ["list", "nope"])
    assert [c["name"] for c in used] == ["list"] and not skipped and unknown == ["nope"]


def test_request_validation() -> None:
    with pytest.raises(ValueError, match="at most one"):
        PerfRequest.model_validate(
            {"source": {"start": ["x"]}, "cases": CASES, "target": {"start": ["x"]},
             "duration_seconds": 1, "requests": 5}
        )  # fmt: skip
    with pytest.raises(ValueError, match="needs a source"):
        PerfRequest.model_validate(
            {"target": {"start": ["x"]}, "corpora": [{"source_id": "a", "cases": CASES}]}
        )


async def test_routes(app, client, workspace) -> None:
    async def _ok(container_name: str, tenant_id: str | None = None) -> None:
        return None

    app.dependency_overrides[get_perf_runner] = lambda: PerfRunner(workspace)
    try:
        with patch("routers.parity_perf.validate_and_get_container_for_tenant", _ok):
            body = _body()
            r = await client.post("/api/parity/ws1/perf/runs", json=body)
            assert r.status_code == 202, r.text
            run_id = r.json()["run_id"]
            r = await client.get(f"/api/parity/ws1/perf/runs/{run_id}")
            assert r.status_code == 200 and r.json()["result"]["computed"] is True
    finally:
        app.dependency_overrides.pop(get_perf_runner, None)


async def test_unknown_or_expired_recording_run_is_a_clear_422(workspace) -> None:
    from fastapi import HTTPException

    body = _body()
    body["corpora"][0].update(baseline="recording", recording_run_id="e" * 32 + ".job-1")
    with pytest.raises(HTTPException) as e:
        await PerfRunner(workspace).start("ws1", "t1", PerfRequest.model_validate(body))
    assert e.value.status_code == 422 and "expired" in str(e.value.detail)


def test_grade_at_defaults_saturated_level_has_no_verdict() -> None:
    """A 10x slower target, but the load generator was saturated on the source side."""

    def level(rps: float, ms: float, saturated: bool) -> dict[str, Any]:
        stats = {"count": 500, "errors": 0, "transport_errors": 0, "throughput_rps": rps,
                 "p50_ms": ms, "p95_ms": ms, "p99_ms": ms, "max_ms": ms}  # fmt: skip
        return {"concurrency": 8, "wall_s": 10.0, "client_cpu_ratio": 0.95 if saturated else 0.1,
                "load_generator_saturated": saturated, "overall": stats,
                "routes": {"GET /x": stats}}  # fmt: skip

    spec = {
        "thresholds": {},
        "corpora": [{"source_id": "a", "source_key": "s0", "target_key": "t0"}],
    }
    sides = {
        "s0": {"key": "s0", "ok": True, "levels": [level(500.0, 2.0, True)]},
        "t0": {"key": "t0", "ok": True, "levels": [level(50.0, 20.0, False)]},
    }
    r = grade_perf(spec, {"ok": True, "sides": sides}, exit_code=0)
    assert not r.computed and not r.passed and r.regressions == []
    assert all(c.saturated for c in r.comparisons) and "saturated" in (r.error or "")
    # Negative control: the same numbers unsaturated are a regression.
    sides["s0"]["levels"] = [level(500.0, 2.0, False)]
    r = grade_perf(spec, {"ok": True, "sides": sides}, exit_code=0)
    assert r.computed and not r.passed
    assert {"p50", "throughput"} <= {g.metric for g in r.regressions}


def _sat_level(count: int, rps: float, ms: float, saturated: bool) -> dict[str, Any]:
    stats = {"count": count, "errors": 0, "transport_errors": 0, "throughput_rps": rps,
             "p50_ms": ms, "p95_ms": ms, "p99_ms": ms, "max_ms": ms}  # fmt: skip
    return {"concurrency": 8, "wall_s": 10.0, "client_cpu_ratio": 0.95 if saturated else 0.1,
            "load_generator_saturated": saturated, "overall": stats,
            "routes": {"GET /x": stats}}  # fmt: skip


_SAT_SPEC = {
    "thresholds": {},
    "corpora": [{"source_id": "a", "source_key": "s0", "target_key": "t0"}],
}


def test_grade_at_defaults_saturated_target_has_no_verdict() -> None:
    """Mirror of the source case: the load generator saturated on the TARGET side only."""
    sides = {
        "s0": {"key": "s0", "ok": True, "levels": [_sat_level(500, 500.0, 2.0, False)]},
        "t0": {"key": "t0", "ok": True, "levels": [_sat_level(500, 50.0, 20.0, True)]},
    }
    r = grade_perf(_SAT_SPEC, {"ok": True, "sides": sides}, exit_code=0)
    assert not r.computed and not r.passed and r.regressions == []
    assert r.comparisons and all(c.saturated for c in r.comparisons)
    assert "saturated" in (r.error or "") and "a concurrency 8" in (r.error or "")
    # Negative control: the same numbers with an unsaturated target are a regression.
    sides["t0"]["levels"] = [_sat_level(500, 50.0, 20.0, False)]
    r = grade_perf(_SAT_SPEC, {"ok": True, "sides": sides}, exit_code=0)
    assert r.computed and not r.passed
    assert {"p50", "throughput"} <= {g.metric for g in r.regressions}


def test_grade_at_defaults_saturated_fast_target_never_passes() -> None:
    """A saturated target whose numbers LOOK better than the source must not pass:
    nothing measured through an overloaded load generator is the service's."""
    sides = {
        "s0": {"key": "s0", "ok": True, "levels": [_sat_level(500, 500.0, 2.0, False)]},
        "t0": {"key": "t0", "ok": True, "levels": [_sat_level(500, 900.0, 1.0, True)]},
    }
    r = grade_perf(_SAT_SPEC, {"ok": True, "sides": sides}, exit_code=0)
    assert not r.computed and not r.passed and "saturated" in (r.error or "")
    # Control: unsaturated, the same faster target is a computed pass.
    sides["t0"]["levels"] = [_sat_level(500, 900.0, 1.0, False)]
    r = grade_perf(_SAT_SPEC, {"ok": True, "sides": sides}, exit_code=0)
    assert r.computed and r.passed


def test_grade_at_defaults_under_measured_target_never_passes() -> None:
    """Target under-measured (< min_route_samples) but no slower: not computed, no pass."""
    sides = {
        "s0": {"key": "s0", "ok": True, "levels": [_sat_level(5000, 500.0, 2.0, False)]},
        "t0": {"key": "t0", "ok": True, "levels": [_sat_level(10, 500.0, 2.0, False)]},
    }
    r = grade_perf(_SAT_SPEC, {"ok": True, "sides": sides}, exit_code=0)
    assert not r.computed and not r.passed
    assert all(c.under_sampled for c in r.comparisons)
    assert "min_route_samples" in (r.error or "")
