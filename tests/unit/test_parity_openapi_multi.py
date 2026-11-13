"""openapi-diff, N sources -> 1 target (gap items 3 + 16).

Source A = the real FastAPI fixture spec, source B = a small second service; the
target is the merged Rust spec (A's utoipa spec under /a, B under /b). Negative
controls: a planted change in B's operation must be caught and attributed to B,
and without the prefix map the same documents must NOT compare equal.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from models.parity_models import WorkspaceOpenApiDiffRequest, WorkspaceOpenApiDiffRunRequest
from services import openapi_fetch_driver as fetch_drv
from services import workspace_driver_run as wdr
from services.openapi_diff import SpecInput, openapi_diff
from services.workspace_openapi import (
    OpenApiDiffRuns,
    WorkspaceOpenApiFetcher,
    diff_workspace_docs,
    obtain_all,
)
from tests.local_workspace import LocalWorkspace

pytestmark = pytest.mark.usefixtures("as_writer", "pg_db")

DATA = Path(__file__).resolve().parent.parent / "data" / "openapi_diff"
FASTAPI = json.loads((DATA / "py2rs_fixture_fastapi.json").read_text())
UTOIPA = json.loads((DATA / "py2rs_fixture_utoipa.json").read_text())

B_SPEC: dict[str, Any] = {
    "openapi": "3.1.0",
    "info": {"title": "things", "version": "1"},
    "paths": {
        "/things/{id}": {
            "get": {
                "parameters": [
                    {"name": "id", "in": "path", "required": True, "schema": {"type": "integer"}}
                ],
                "responses": {
                    "200": {
                        "description": "ok",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"qty": {"type": "integer"}},
                                    "required": ["qty"],
                                }
                            }
                        },
                    }
                },
            }
        }
    },
}

BEARER = {"bearer": {"type": "http", "scheme": "bearer"}}


def merged_target(*, auth: bool = False) -> dict[str, Any]:
    tgt = copy.deepcopy(UTOIPA)
    tgt["paths"] = {f"/a{p}": item for p, item in UTOIPA["paths"].items()}
    tgt["paths"].update({f"/b{p}": copy.deepcopy(i) for p, i in B_SPEC["paths"].items()})
    if auth:  # the merged service unified auth: every op needs a bearer token
        tgt.setdefault("components", {})["securitySchemes"] = BEARER
        tgt["security"] = [{"bearer": []}]
    return tgt


def _sources(maps: bool = True) -> list[SpecInput]:
    return [
        SpecInput("a", FASTAPI, (("/", "/a/"),) if maps else ()),
        SpecInput("b", B_SPEC, (("/things", "/b/things"),) if maps else ()),
    ]


def test_prefix_maps_make_moved_routes_equal():
    r = openapi_diff(_sources(), merged_target())
    assert r["equal"] is True, r
    assert r["progress"] == {"matched": 7, "total": 7, "ratio": 1.0}


def test_without_prefix_maps_moved_routes_are_missing_and_extra():
    r = openapi_diff(_sources(maps=False), merged_target())
    assert r["equal"] is False
    assert r["counts"]["missing"] == 7 and r["counts"]["extra"] == 7


def test_planted_change_in_source_b_is_attributed_to_b():
    tgt = merged_target()
    resp = tgt["paths"]["/b/things/{id}"]["get"]["responses"]
    resp["201"] = resp.pop("200")
    r = openapi_diff(_sources(), tgt)
    assert r["equal"] is False and r["counts"]["changed"] == 1
    [changed] = r["changed_operations"]
    assert changed["sources"] == ["b"] and changed["path"] == "/b/things/{id}"
    assert changed["source_path"] == "/things/{id}"
    del tgt["paths"]["/b/things/{id}"]
    r = openapi_diff(_sources(), tgt)
    assert r["missing_operations"] == [
        {"method": "GET", "path": "/b/things/{id}", "sources": ["b"], "source_path": "/things/{id}"}
    ]


@pytest.mark.parametrize(
    "auth, equal, changed, auth_ops",
    [("compare", False, 7, 0), ("separate", False, 0, 7), ("ignore", True, 0, 7)],
)
def test_auth_normalisation_modes(auth, equal, changed, auth_ops):
    r = openapi_diff(_sources(), merged_target(auth=True), auth=auth)
    assert (r["equal"], r["counts"]["changed"], r["counts"]["auth"]) == (equal, changed, auth_ops)
    if auth != "compare":
        kinds = {d["kind"] for op in r["auth_differences"] for d in op["differences"]}
        assert kinds == {"security_changed"}


def test_auth_ignore_still_catches_a_real_change():
    """Negative control: ignoring auth must not hide a non-auth difference."""
    tgt = merged_target(auth=True)
    tgt["paths"]["/b/things/{id}"]["get"]["responses"]["200"]["content"]["application/json"][
        "schema"
    ]["properties"]["qty"]["type"] = "string"
    r = openapi_diff(_sources(), tgt, auth="ignore")
    assert r["equal"] is False and [c["sources"] for c in r["changed_operations"]] == [["b"]]


def test_authorization_header_parameter_is_auth():
    src = copy.deepcopy(B_SPEC)
    src["paths"]["/things/{id}"]["get"]["parameters"].append(
        {"name": "Authorization", "in": "header", "required": True, "schema": {"type": "string"}}
    )
    r = openapi_diff([SpecInput("b", src)], B_SPEC, auth="ignore")
    assert r["equal"] is True and r["counts"]["auth"] == 1


# --------------------------------------------------------------------------- routes


async def test_inline_route_two_sources_with_maps_and_auth(client):
    body = {
        "sources": [
            {"name": "a", "document": FASTAPI, "path_prefix_map": {"/": "/a/"}},
            {"name": "b", "document": B_SPEC, "path_prefix_map": {"/things": "/b/things"}},
        ],
        "target": {"document": merged_target(auth=True)},
        "auth": "ignore",
    }
    r = await client.post("/api/parity/openapi-diff", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["equal"] is True and r.json()["auth"] == "ignore"
    body["auth"] = "compare"
    assert (await client.post("/api/parity/openapi-diff", json=body)).json()["equal"] is False
    body["target"]["path_prefix_map"] = {"/": "/x/"}
    r = await client.post("/api/parity/openapi-diff", json=body)
    assert r.status_code == 422 and "sources only" in r.text


def test_fetch_driver_runs_concurrently_and_reports_failures(tmp_path):
    """The REAL driver: commands + a file, bounded concurrency, one failing command."""
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "openapi.json").write_text(json.dumps(B_SPEC))
    slow = (
        "import json,sys,time; s=time.time(); time.sleep(0.5); "
        "open(sys.argv[1],'w').write(json.dumps([s,time.time()])); "
        "print(json.dumps({'openapi': '3.1.0'}))"
    )
    specs = [
        {"name": "a", "role": "source", "workdir": str(tmp_path), "timeout": 30,
         "command": [sys.executable, "-c", slow, str(tmp_path / "a.t")], "file": None},
        {"name": "b", "role": "source", "workdir": str(tmp_path / "b"), "timeout": 30,
         "command": None, "file": "openapi.json"},
        {"name": "rs", "role": "target", "workdir": str(tmp_path), "timeout": 30,
         "command": [sys.executable, "-c", slow, str(tmp_path / "rs.t")], "file": None},
    ]  # fmt: skip
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "driver_config.json").write_text(json.dumps({"concurrency": 3, "specs": specs}))
    assert fetch_drv.main([str(run_dir)]) == 0
    (a0, a1), (r0, r1) = (json.loads((tmp_path / n).read_text()) for n in ("a.t", "rs.t"))
    assert a0 < r1 and r0 < a1  # the two 0.5 s commands overlapped: concurrent
    result = json.loads((run_dir / "result.json").read_text())
    assert result["ok"] is True and [s["name"] for s in result["specs"]] == ["a", "b", "rs"]
    assert json.loads((run_dir / "doc_1.json").read_text()) == B_SPEC

    specs[1]["command"] = [sys.executable, "-c", "import sys; sys.exit(3)"]
    (run_dir / "driver_config.json").write_text(json.dumps({"concurrency": 1, "specs": specs}))
    assert fetch_drv.main([str(run_dir)]) == 1
    result = json.loads((run_dir / "result.json").read_text())
    assert result["ok"] is False and "source 'b' command exited 3" in result["error"]


def test_fetch_driver_failure_paths_and_parity_toml(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (tmp_path / "parity.toml").write_text('[openapi_diff.ignore]\npaths = ["/x"]\n')
    ok_spec = {"name": "a", "role": "source", "workdir": str(tmp_path), "timeout": 30,
               "command": [sys.executable, "-c", "print('{}')"], "file": None}  # fmt: skip

    def run(config: dict) -> tuple[int, dict]:
        (run_dir / "driver_config.json").write_text(json.dumps(config))
        code = fetch_drv.main([str(run_dir)])
        return code, json.loads((run_dir / "result.json").read_text())

    code, out = run({"specs": [ok_spec], "parity_toml": str(tmp_path / "parity.toml")})
    assert code == 0 and out["parity_toml"] == "parity_toml.txt"
    assert (run_dir / "parity_toml.txt").read_text().startswith("[openapi_diff.ignore]")
    code, out = run({"specs": [ok_spec], "parity_toml": str(tmp_path / "nope.toml")})
    assert code == 1 and "parity_toml" in out["error"]
    code, out = run({"specs": [{**ok_spec, "file": "missing.json", "command": None}]})
    assert code == 1 and "could not be read" in out["error"]
    code, out = run({"specs": [{**ok_spec, "command": [str(tmp_path / "no-such-binary")]}]})
    assert code == 1 and "failed to run" in out["error"]
    sleepy = [sys.executable, "-c", "import time; time.sleep(5)"]
    code, out = run({"specs": [{**ok_spec, "command": sleepy, "timeout": 1}]})
    assert code == 1 and "timed out" in out["error"]
    code, out = run({"specs": []})
    assert code == 1 and "no specs" in out["error"]
    (run_dir / "driver_config.json").write_text("{broken")
    assert fetch_drv.main([str(run_dir)]) == 1
    assert json.loads((run_dir / "result.json").read_text())["error"].startswith("driver crashed")
    assert fetch_drv.main([]) == 2


# ------------------------------------------- workspace paths, real exec on this machine


@pytest.fixture
def local_ws(tmp_path, monkeypatch):
    monkeypatch.setattr(wdr, "RUN_ROOT", str(tmp_path / "runs"))
    root = tmp_path / "ws"
    for name, doc in (("a", FASTAPI), ("b", B_SPEC)):
        (root / ".repos" / name).mkdir(parents=True)
        (root / ".repos" / name / "openapi.json").write_text(json.dumps(doc))
    (root / "target.json").write_text(json.dumps(merged_target()))
    return LocalWorkspace(root)


WS_BODY: dict[str, Any] = {
    "sources": [
        {"name": "a", "workdir": ".repos/a", "file": "openapi.json",
         "path_prefix_map": {"/": "/a/"}},
        {"name": "b", "workdir": ".repos/b", "command": ["python3", "-c",
         "import sys; sys.stdout.write(open('openapi.json').read())"],
         "path_prefix_map": {"/things": "/b/things"}},
    ],
    "target": {"name": "rs", "command": ["python3", "-c",
               "import sys; sys.stdout.write(open('target.json').read())"]},
    "concurrency": 2,
}  # fmt: skip


async def test_workspace_sync_path_obtains_every_document_for_real(local_ws):
    req = WorkspaceOpenApiDiffRequest.model_validate(WS_BODY)
    sources, target, obtained = await obtain_all(
        WorkspaceOpenApiFetcher(local_ws), "ws-1", None, req
    )
    result = diff_workspace_docs(req, "ws-1", sources, target, obtained, None)
    assert result.equal is True
    assert [o.name for o in result.obtained] == ["a", "b", "rs"]


async def test_workspace_async_run_attributes_a_planted_b_change(local_ws):
    runs = OpenApiDiffRuns(local_ws)
    req = WorkspaceOpenApiDiffRunRequest.model_validate(WS_BODY)
    started = await runs.start("ws-1", None, req)
    assert started.status == "queued"
    done = await runs.poll("ws-1", None, started.run_id)
    assert done.status == "completed" and done.result is not None, done.error
    assert done.result.equal is True

    tgt = merged_target()
    tgt["paths"]["/b/things/{id}"]["get"]["parameters"][0]["schema"]["type"] = "string"
    (local_ws.root / "target.json").write_text(json.dumps(tgt))
    done = await runs.poll("ws-1", None, (await runs.start("ws-1", None, req)).run_id)
    assert done.result is not None and done.result.equal is False
    changed = done.result.model_dump()["changed_operations"]
    assert [(c["sources"], c["source_path"]) for c in changed] == [(["b"], "/things/{id}")]

    # Negative control: a source whose export fails -> failed, never a verdict.
    broken = copy.deepcopy(WS_BODY)
    broken["sources"][1]["command"] = ["python3", "-c", "import sys; sys.exit(3)"]
    req = WorkspaceOpenApiDiffRunRequest.model_validate(broken)
    done = await runs.poll("ws-1", None, (await runs.start("ws-1", None, req)).run_id)
    assert done.status == "failed" and done.result is None
    assert "exited 3" in json.dumps(done.error)
