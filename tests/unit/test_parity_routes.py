"""/api/parity routes: inline diff, and workspace diff over a fake exec transport."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from coding_client.clients import ServiceError

from routers.parity import get_workspace_openapi_fetcher
from services.workspace_openapi import WorkspaceOpenApiFetcher

# Write routes (they execute in the workspace) need the write role; see
# tests/unit/test_workspace_route_permissions.py for the 403 side.
pytestmark = pytest.mark.usefixtures("as_writer")

DATA = Path(__file__).resolve().parent.parent / "data" / "openapi_diff"
FASTAPI_TEXT = (DATA / "py2rs_fixture_fastapi.json").read_text()
UTOIPA_TEXT = (DATA / "py2rs_fixture_utoipa.json").read_text()
FASTAPI = json.loads(FASTAPI_TEXT)
UTOIPA = json.loads(UTOIPA_TEXT)

SRC_CMD = ["uv", "run", "python", "scripts/export_openapi.py"]
TGT_CMD = ["cargo", "run", "-q", "--", "--print-openapi"]


async def _fake_validate(container_name: str, tenant_id: str | None = None) -> None:
    if container_name == "missing":
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="not found")


class FakeTransport:
    """Records every call; answers from a {(workdir, argv-tuple): payload} map."""

    def __init__(
        self, answers: dict[tuple[str, tuple[str, ...]], Any], jobs: list[Any] | None = None
    ):
        self.answers = answers
        self.jobs = list(jobs or [])
        self.calls: list[tuple[str, Any]] = []

    def _answer(self, argv: list[str], workdir: str) -> Any:
        value = self.answers[(workdir, tuple(argv))]
        if isinstance(value, Exception):
            raise value
        return value

    async def exec(self, container, tenant_id, argv, workdir, timeout):
        self.calls.append(("exec", (container, tuple(argv), workdir, timeout)))
        return self._answer(argv, workdir)

    async def exec_async(self, container, tenant_id, argv, workdir, timeout):
        self.calls.append(("exec_async", (container, tuple(argv), workdir, timeout)))
        return {"job_id": "job-1", "status": "queued"}

    async def get_exec_job(self, container, tenant_id, job_id):
        self.calls.append(("get_exec_job", job_id))
        return self.jobs.pop(0)


def ok(stdout: str, **kw: Any) -> dict[str, Any]:
    return {"exit_code": 0, "stdout": stdout, "stderr": "", "duration": 0.5, **kw}


@pytest.fixture
def use_transport(app):
    def _use(t: FakeTransport) -> FakeTransport:
        app.dependency_overrides[get_workspace_openapi_fetcher] = lambda: WorkspaceOpenApiFetcher(
            t, poll_interval=0
        )
        return t

    yield _use
    app.dependency_overrides.pop(get_workspace_openapi_fetcher, None)


@pytest.fixture
def validated():
    with patch("routers.parity.validate_and_get_container_for_tenant", _fake_validate):
        yield


# --------------------------------------------------------------------------- inline


async def test_inline_diff_equal(client) -> None:
    r = await client.post(
        "/api/parity/openapi-diff",
        json={"sources": [{"name": "py", "document": FASTAPI}], "target": {"document": UTOIPA}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["equal"] is True and body["progress"] == {"matched": 6, "total": 6, "ratio": 1.0}


async def test_inline_diff_reports_mutation_and_parity_toml_ignore(client) -> None:
    target = json.loads(UTOIPA_TEXT)
    del target["paths"]["/health"]
    r = await client.post(
        "/api/parity/openapi-diff",
        json={"sources": [{"document": FASTAPI}], "target": {"document": target}},
    )
    body = r.json()
    assert body["equal"] is False
    assert body["missing_operations"] == [
        {"method": "GET", "path": "/health", "sources": ["source[0]"]}
    ]
    r = await client.post(
        "/api/parity/openapi-diff",
        json={
            "sources": [{"document": FASTAPI}],
            "target": {"document": target},
            "parity_toml": '[openapi_diff.ignore]\noperations = ["GET /health"]\n',
        },
    )
    body = r.json()
    assert body["equal"] is True
    assert body["ignored_operations"] == [
        {"method": "GET", "path": "/health", "document": "source[0]"}
    ]
    assert body["ignore"]["operations"] == ["GET /health"]


async def test_inline_diff_rejects_unsupported(client) -> None:
    r = await client.post(
        "/api/parity/openapi-diff",
        json={
            "sources": [{"document": {"swagger": "2.0", "paths": {}}}],
            "target": {"document": UTOIPA},
        },
    )
    assert r.status_code == 422
    assert "OpenAPI 3.0/3.1" in r.text
    r = await client.post(
        "/api/parity/openapi-diff", json={"sources": [], "target": {"document": UTOIPA}}
    )
    assert r.status_code == 422


# --------------------------------------------------------------------------- workspace


def _ws_body(**over: Any) -> dict[str, Any]:
    body = {
        "sources": [
            {
                "name": "py",
                "workdir": ".repos/conveyor-coding__py2rs-fixture-svc",
                "command": SRC_CMD,
                "file": "openapi.json",
            }
        ],
        "target": {"name": "rs", "command": " ".join(TGT_CMD)},
    }
    body.update(over)
    return body


async def test_workspace_diff_command_and_file(client, use_transport, validated) -> None:
    src_wd = "/workspace/.repos/conveyor-coding__py2rs-fixture-svc"
    t = use_transport(
        FakeTransport(
            {
                (src_wd, tuple(SRC_CMD)): ok("wrote openapi.json\n"),
                (src_wd, ("cat", "--", "openapi.json")): ok(FASTAPI_TEXT),
                ("/workspace", tuple(TGT_CMD)): ok(UTOIPA_TEXT),
            }
        )
    )
    r = await client.post("/api/parity/ws1/openapi-diff", json=_ws_body())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["equal"] is True and body["container_name"] == "ws1"
    assert [(o["name"], o["role"], o["exit_code"]) for o in body["obtained"]] == [
        ("py", "source", 0),
        ("rs", "target", 0),
    ]
    assert body["obtained"][1]["command"] == TGT_CMD  # string was split, no shell
    assert body["obtained"][1]["bytes"] == len(UTOIPA_TEXT.encode())
    assert [c[0] for c in t.calls] == ["exec", "exec", "exec"]


async def test_workspace_diff_detects_mutation(client, use_transport, validated) -> None:
    target = json.loads(UTOIPA_TEXT)
    responses = target["paths"]["/notes"]["post"]["responses"]
    responses["200"] = responses.pop("201")
    use_transport(
        FakeTransport(
            {
                ("/workspace", ("cat", "--", "openapi.json")): ok(FASTAPI_TEXT),
                ("/workspace", tuple(TGT_CMD)): ok(json.dumps(target)),
            }
        )
    )
    r = await client.post(
        "/api/parity/ws1/openapi-diff",
        json=_ws_body(sources=[{"file": "openapi.json"}]),
    )
    body = r.json()
    assert body["equal"] is False
    assert body["changed_operations"][0]["differences"] == [
        {"location": "responses.200", "kind": "response_added", "source": None, "target": "200"},
        {"location": "responses.201", "kind": "response_removed", "source": "201", "target": None},
    ]


async def test_workspace_diff_long_command_runs_async_and_polls(
    client, use_transport, validated
) -> None:
    t = use_transport(
        FakeTransport(
            {("/workspace", ("cat", "--", "openapi.json")): ok(FASTAPI_TEXT)},
            jobs=[
                {"status": "running"},
                {"status": "completed", "exit_code": 0, "stdout": UTOIPA_TEXT, "duration": 412.0},
            ],
        )
    )
    body = _ws_body(sources=[{"file": "openapi.json"}])
    body["target"]["timeout"] = 1800
    r = await client.post("/api/parity/ws1/openapi-diff", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["equal"] is True
    assert [c[0] for c in t.calls] == ["exec", "exec_async", "get_exec_job", "get_exec_job"]
    assert r.json()["obtained"][1]["duration"] == 412.0


async def test_workspace_async_job_without_exit_is_422(client, use_transport, validated) -> None:
    use_transport(
        FakeTransport(
            {("/workspace", ("cat", "--", "openapi.json")): ok(FASTAPI_TEXT)},
            jobs=[{"status": "failed", "exit_code": None, "error": "timeout: 1800s"}],
        )
    )
    body = _ws_body(sources=[{"file": "openapi.json"}])
    body["target"]["timeout"] = 1800
    r = await client.post("/api/parity/ws1/openapi-diff", json=body)
    assert r.status_code == 422 and "timeout" in r.text


@pytest.mark.parametrize(
    "payload, needle",
    [
        (
            {"exit_code": 101, "stdout": "", "stderr": "error[E0425]: cannot find value"},
            "exited 101",
        ),
        (ok(UTOIPA_TEXT[:500], truncated=True), "clipped"),
        (ok("Compiling foo v0.1.0\n" + UTOIPA_TEXT), "not a JSON document"),
        (ok("[1, 2]"), "not an object"),
    ],
)
async def test_workspace_target_not_obtained_is_422(
    client, use_transport, validated, payload, needle
) -> None:
    use_transport(
        FakeTransport(
            {
                ("/workspace", ("cat", "--", "openapi.json")): ok(FASTAPI_TEXT),
                ("/workspace", tuple(TGT_CMD)): payload,
            }
        )
    )
    r = await client.post(
        "/api/parity/ws1/openapi-diff", json=_ws_body(sources=[{"file": "openapi.json"}])
    )
    assert r.status_code == 422, r.text
    assert needle in r.text


async def test_workspace_transport_failures(client, use_transport, validated) -> None:
    use_transport(
        FakeTransport({("/workspace", ("cat", "--", "openapi.json")): ServiceError(404, "gone")})
    )
    r = await client.post(
        "/api/parity/ws1/openapi-diff", json=_ws_body(sources=[{"file": "openapi.json"}])
    )
    assert r.status_code == 404
    use_transport(
        FakeTransport(
            {("/workspace", ("cat", "--", "openapi.json")): RuntimeError("connect refused")}
        )
    )
    r = await client.post(
        "/api/parity/ws1/openapi-diff", json=_ws_body(sources=[{"file": "openapi.json"}])
    )
    assert r.status_code == 502
    use_transport(FakeTransport({("/workspace", ("cat", "--", "openapi.json")): {"stdout": "x"}}))
    r = await client.post(
        "/api/parity/ws1/openapi-diff", json=_ws_body(sources=[{"file": "openapi.json"}])
    )
    assert r.status_code == 502 and "no exit code" in r.text


async def test_workspace_parity_toml_is_read(client, use_transport, validated) -> None:
    target = json.loads(UTOIPA_TEXT)
    del target["paths"]["/health"]
    use_transport(
        FakeTransport(
            {
                ("/workspace", ("cat", "--", "openapi.json")): ok(FASTAPI_TEXT),
                ("/workspace", tuple(TGT_CMD)): ok(json.dumps(target)),
                ("/workspace", ("cat", "--", "parity.toml")): ok(
                    '[openapi_diff.ignore]\npaths = ["/health"]\n'
                ),
            }
        )
    )
    r = await client.post(
        "/api/parity/ws1/openapi-diff",
        json=_ws_body(sources=[{"file": "openapi.json"}], parity_toml="parity.toml"),
    )
    assert r.status_code == 200, r.text
    assert r.json()["equal"] is True and r.json()["counts"]["ignored"] == 1


@pytest.mark.parametrize(
    "body",
    [
        _ws_body(sources=[{"workdir": "../etc", "file": "x"}]),
        _ws_body(sources=[{"file": "/etc/passwd"}]),
        _ws_body(sources=[{"workdir": "."}]),  # neither command nor file
        _ws_body(sources=[{"file": "a.json", "bogus": 1}]),
        _ws_body(sources=[{"file": "a.json", "timeout": 99999}]),
        _ws_body(sources=[{"name": "rs", "file": "a.json"}]),  # collides with target name
        _ws_body(parity_toml="../parity.toml"),
    ],
)
async def test_workspace_request_validation(client, use_transport, validated, body) -> None:
    use_transport(FakeTransport({}))
    r = await client.post("/api/parity/ws1/openapi-diff", json=body)
    assert r.status_code == 422, r.text


async def test_workspace_missing_container(client, use_transport, validated) -> None:
    use_transport(FakeTransport({}))
    r = await client.post("/api/parity/missing/openapi-diff", json=_ws_body())
    assert r.status_code == 404


async def test_inline_strict_numeric_bounds_flag(client) -> None:
    source = json.loads(FASTAPI_TEXT)
    source["components"]["schemas"]["Health"]["properties"]["n"] = {"type": "integer"}
    target = json.loads(UTOIPA_TEXT)
    target["components"]["schemas"]["HealthBody"]["properties"]["n"] = {
        "type": "integer",
        "format": "uint32",
        "minimum": 0,
    }
    body = {"sources": [{"document": source}], "target": {"document": target}}
    r = await client.post("/api/parity/openapi-diff", json=body)
    out = r.json()
    assert out["equal"] is True and out["counts"]["cosmetic"] == 1
    assert out["cosmetic_differences"][0]["path"] == "/health"
    r = await client.post("/api/parity/openapi-diff", json={**body, "strict_numeric_bounds": True})
    out = r.json()
    assert out["equal"] is False and out["strict_numeric_bounds"] is True
    assert out["changed_operations"][0]["differences"][0]["kind"] == "unsigned_minimum_added"
