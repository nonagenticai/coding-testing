"""N sources -> 1 target behaviour parity (gap items 1 + 16).

The REAL driver (in-process, stdlib fake target, no DB) runs two corpora - one
per source - against ONE merged target, then the REAL grade() turns the result
into a verdict. Negative controls: a planted mismatch in source B's routes must
fail exactly B's case, attributed to B, and a route the target lacks must be
reported as ``route_missing``.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from models.parity_behaviour_models import CompareRequest, RecordRequest, ReplayRequest
from services import behaviour_parity as bp
from services import behaviour_parity_driver as drv
from services.behaviour_diff import DiffOptions, diff_behaviour

FAKE_URL_WS_PG = [sys.executable, "-c", "print('postgresql://u@127.0.0.1:1/x')"]

# The merged target: source A's routes under /a, source B's under /b; A calls the
# "kag" upstream, B the "billing" one and needs the TARGET's auth header.
TARGET = textwrap.dedent(
    """
    import http.client, json, os, urllib.parse
    from http.server import BaseHTTPRequestHandler, HTTPServer

    def call(env, path, body):
        up = urllib.parse.urlparse(os.environ[env])
        c = http.client.HTTPConnection(up.hostname, up.port, timeout=5)
        c.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"})
        r = c.getresponse()
        return json.loads(r.read() or b"null")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, status, body):
            data = b"" if body is None else json.dumps(body).encode()
            self.send_response(status)
            if data:
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"ok": True})
            self._send(404, None)  # axum-style: no route -> empty 404

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"null")
            if self.path == "/a/echo":
                return self._send(200, call("KAG_URL", "/enrich", body))
            if self.path == "/b/echo":
                if self.headers.get("Authorization") != "Bearer tgt":
                    return self._send(401, {"detail": "auth"})
                status = 201 if os.environ.get("MUTATE_B") else 200
                return self._send(status, call("BILL_URL", "/charge", body))
            self._send(404, None)

    HTTPServer(("127.0.0.1", int(os.environ["PORT"])), H).serve_forever()
    """
)

CASES_A = [
    {"name": "echo", "method": "POST", "path": "/echo", "body": {"a": 1},
     "upstream_responses": {"kag": [{"status": 200, "body": {"s": "A"}}]},
     "expected_status": 200, "expected_body": {"s": "A"},
     "expected_upstream_calls": [{"upstream": "kag", "method": "POST", "path": "/enrich",
                                  "body": {"a": 1}}]},
]  # fmt: skip
CASES_B = [
    {"name": "echo", "method": "POST", "path": "/echo", "body": {"b": 1},
     "headers": {"Authorization": "Bearer src"},
     "upstream_responses": {"billing": [{"status": 200, "body": {"s": "B"}}]},
     "expected_status": 200, "expected_body": {"s": "B"},
     "expected_upstream_calls": [{"upstream": "billing", "method": "POST", "path": "/charge",
                                  "body": {"b": 1}}]},
]  # fmt: skip
GONE = {
    "name": "gone",
    "method": "GET",
    "path": "/gone",
    "expected_status": 200,
    "expected_body": {"ok": 1},
}

TARGET_SVC = {
    "workdir": ".",
    "start": ["python3", "target.py"],
    "upstream_env": [],
    "upstreams": {"kag": ["KAG_URL"], "billing": ["BILL_URL"]},
    "request_headers": {"Authorization": "Bearer tgt"},
}


def _replay(cases_b: list[dict], **target_env: str) -> ReplayRequest:
    return ReplayRequest.model_validate(
        {
            "target": {**TARGET_SVC, "env": target_env},
            "corpora": [
                {"source_id": "svc-a", "cases": CASES_A, "route_map": {"/": "/a/"}},
                {"source_id": "svc-b", "cases": cases_b, "route_map": [{"from": "/", "to": "/b/"}]},
            ],
            "async": False,
            "timeout": 120,
        }
    )


def _drive(
    tmp_path: Path, req: ReplayRequest | CompareRequest, op: str
) -> bp.BehaviourParityResult:
    script = tmp_path / "target.py"
    script.write_text(TARGET)
    cfg = bp.build_driver_config(req, [])
    cfg["ws_pg"] = FAKE_URL_WS_PG
    for side in cfg["sides"]:
        side["workdir"] = str(tmp_path)
        side["start"] = [sys.executable, str(script)]
        side["startup_timeout"] = 20
    run_dir = tmp_path / ("c" * 32)
    run_dir.mkdir()
    (run_dir / "driver_config.json").write_text(json.dumps(cfg))
    code = drv.main([str(run_dir)])
    result = json.loads((run_dir / "result.json").read_text())
    assert code == 0, result.get("error")
    return bp.grade(bp.build_spec(op, req), result, exit_code=code)  # type: ignore[arg-type]


def test_two_corpora_against_one_target_pass(tmp_path):
    res = _drive(tmp_path, _replay(CASES_B), "replay")
    assert res.computed is True and res.equal is True, res.model_dump()
    assert [(c.source_id, c.cases_passed) for c in res.corpora] == [("svc-a", 1), ("svc-b", 1)]
    a, b = res.cases
    assert (a.source_id, a.method, a.path, a.target_path) == ("svc-a", "POST", "/echo", "/a/echo")
    assert (b.source_id, b.target_method, b.target_path) == ("svc-b", "POST", "/b/echo")


def test_planted_mismatch_in_source_b_is_attributed_to_b(tmp_path):
    """Negative control: B's route answers 201; A must still pass, B must fail."""
    res = _drive(tmp_path, _replay(CASES_B, MUTATE_B="1"), "replay")
    assert res.computed is True and res.equal is False
    by_source = {c.source_id: c for c in res.corpora}
    assert by_source["svc-a"].equal is True and by_source["svc-b"].equal is False
    failed = [c for c in res.cases if not c.passed]
    assert len(failed) == 1 and failed[0].source_id == "svc-b"
    assert failed[0].target_path == "/b/echo"
    assert failed[0].diffs[0].kind == "status" and failed[0].diffs[0].actual == 201


def test_missing_route_on_target_is_reported_as_route_missing(tmp_path):
    res = _drive(tmp_path, _replay([*CASES_B, GONE]), "replay")
    assert res.equal is False
    gone = next(c for c in res.cases if c.name == "gone")
    assert gone.source_id == "svc-b" and gone.target_path == "/b/gone"
    kinds = [d.kind for d in gone.diffs]
    assert kinds[0] == "route_missing" and "status" not in kinds and "body" in kinds
    assert (gone.diffs[0].detail or "").startswith("heuristic route_missing:")


def test_per_upstream_stubs_record_each_upstream_separately(tmp_path):
    """Without the target's billing stub B's call lands nowhere: compared per upstream."""
    script = tmp_path / "target.py"
    script.write_text(TARGET)
    req = _replay(CASES_B)
    cfg = bp.build_driver_config(req, [])
    side = cfg["sides"][1]
    assert side["key"] == "t1" and side["upstreams"] == {
        "kag": ["KAG_URL"],
        "billing": ["BILL_URL"],
    }
    assert side["request_headers"] == {"Authorization": "Bearer tgt"}
    assert side["route_map"] == [{"from": "/", "to": "/b/"}]
    res = _drive(tmp_path, req, "replay")
    assert res.equal is True  # each upstream got exactly its own scripted response


def test_without_target_auth_override_b_fails(tmp_path):
    """Negative control for request_headers: the source's own auth reaches the target."""
    req = ReplayRequest.model_validate(
        {
            **_replay(CASES_B).model_dump(by_alias=True, exclude_none=True),
            "target": {**TARGET_SVC, "request_headers": {}},
        }
    )
    res = _drive(tmp_path, req, "replay")
    b = next(c for c in res.cases if c.source_id == "svc-b")
    assert not b.passed and b.diffs[0].kind == "status" and b.diffs[0].actual == 401


def test_compare_records_each_source_then_the_target(tmp_path):
    src = {**TARGET_SVC}  # the merged target doubles as both sources' implementation
    req = CompareRequest.model_validate(
        {
            "target": TARGET_SVC,
            "corpora": [
                {"source_id": "svc-a", "cases": CASES_A, "route_map": {"/": "/a/"},
                 "source": {**src, "env": {}}},
            ],
            "async": False,
            "timeout": 120,
        }
    )  # fmt: skip
    cfg = bp.build_driver_config(req, [])
    assert [(s["key"], s["role"]) for s in cfg["sides"]] == [("s0", "source"), ("t0", "target")]
    assert "route_map" not in cfg["sides"][0]
    res = _drive(tmp_path, req, "compare")
    # s0 replays the unmapped /echo -> 404 on the fake, t0 the mapped /a/echo -> 200:
    # target != source, attributed to svc-a, and the source recording is returned per id.
    assert res.computed is True and res.equal is False
    assert res.recordings is not None and set(res.recordings) == {"svc-a"}
    assert res.cases[0].source_id == "svc-a"


def test_map_route_semantics():
    rules = [
        {"from": "/", "to": "/a/"},
        {"from": "/api", "to": "/v2"},
        {"from": "/api/items", "to": "/things", "method": "PUT", "to_method": "PATCH"},
    ]
    assert drv.map_route("GET", "/x?q=1", rules) == ("GET", "/a/x?q=1")
    assert drv.map_route("GET", "/", rules) == ("GET", "/a/")
    assert drv.map_route("GET", "/api/x", rules) == ("GET", "/v2/x")
    assert drv.map_route("GET", "/api", rules) == ("GET", "/v2")
    assert drv.map_route("GET", "/apix", rules) == ("GET", "/a/apix")  # segment-aligned
    assert drv.map_route("PUT", "/api/items/3", rules) == ("PATCH", "/things/3")
    assert drv.map_route("GET", "/api/items/3", rules) == ("GET", "/v2/items/3")
    assert drv.map_route("GET", "/x", []) == ("GET", "/x")


def test_map_route_method_specific_beats_generic_on_the_same_prefix():
    """Listed AFTER the generic rule, the method-specific rule must still win."""
    rules = [
        {"from": "/api/items", "to": "/v2/items"},
        {"from": "/api/items", "to": "/things", "method": "POST", "to_method": "PUT"},
    ]
    assert drv.map_route("POST", "/api/items/3", rules) == ("PUT", "/things/3")
    assert drv.map_route("GET", "/api/items/3", rules) == ("GET", "/v2/items/3")
    assert drv.map_route("POST", "/api/items/3", rules[::-1]) == ("PUT", "/things/3")
    # Two equally specific rules: the first listed wins.
    twins = [{"from": "/a", "to": "/one"}, {"from": "/a", "to": "/two"}]
    assert drv.map_route("GET", "/a/x", twins) == ("GET", "/one/x")
    # A LONGER generic prefix still beats a shorter method-specific one.
    longer = [{"from": "/api", "to": "/m", "method": "GET"}, {"from": "/api/items", "to": "/i"}]
    assert drv.map_route("GET", "/api/items/1", longer) == ("GET", "/i/1")


@pytest.mark.parametrize(
    "rule, path, mapped",
    [
        ({"from": "/api/", "to": "/v/"}, "/api", "/v"),  # exact: no slash added
        ({"from": "/api/", "to": "/v/"}, "/api/", "/v/"),
        ({"from": "/api/", "to": "/v/"}, "/api/x", "/v/x"),
        ({"from": "/api/", "to": "/v/"}, "/api?q=1", "/v?q=1"),
        ({"from": "/api", "to": "/v/"}, "/api", "/v"),
        ({"from": "/api", "to": "/v/"}, "/api/", "/v/"),
        ({"from": "/api", "to": "/v"}, "/api/", "/v/"),
        ({"from": "/api/", "to": "/"}, "/api", "/"),
        ({"from": "/", "to": "/a/"}, "/", "/a/"),
    ],
)
def test_map_route_trailing_slash_follows_the_source_path(rule, path, mapped):
    assert drv.map_route("GET", path, [rule]) == ("GET", mapped)


def test_diff_compares_calls_per_upstream_and_honours_ignore_upstreams():
    call_k = {"upstream": "kag", "method": "POST", "path": "/k", "headers": {}, "body": None}
    call_b = {"upstream": "billing", "method": "POST", "path": "/b", "headers": {}, "body": None}
    base = [{"name": "c", "status": 200, "body": None, "upstream_calls": [call_k, call_b]}]
    # Interleaving across upstreams differs: not a difference.
    got = [{"name": "c", "status": 200, "body": None, "upstream_calls": [call_b, call_k]}]
    assert diff_behaviour(base, got)["equal"] is True
    # The billing call is missing on the target: reported under its upstream.
    got = [{"name": "c", "status": 200, "body": None, "upstream_calls": [call_k]}]
    d = diff_behaviour(base, got)
    assert d["equal"] is False
    assert d["cases"][0]["diffs"][0]["pointer"] == "/upstreams/billing"
    assert diff_behaviour(base, got, DiffOptions(ignore_upstreams=("billing",)))["equal"] is True


def test_deliberate_404_app_route_still_gets_every_diff():
    """A target route that DELIBERATELY answers 404 {"detail": "Not Found"} looks like a
    missing route; the heuristic names itself and the other comparisons still run."""
    up = {"method": "GET", "path": "/notes/7", "headers": {}, "body": None}
    base = [{"name": "c", "status": 200, "body": {"id": 7},
             "headers": {"content-type": "application/json", "x-trace": "a"},
             "upstream_calls": [up]}]  # fmt: skip
    deliberate = [{"name": "c", "status": 404, "body": {"detail": "Not Found"},
                   "headers": {"content-type": "application/json", "x-trace": "b"},
                   "upstream_calls": [up, up]}]  # fmt: skip
    out = diff_behaviour(base, deliberate, DiffOptions(response_headers=("x-trace",)))
    diffs = out["cases"][0]["diffs"]
    kinds = [d["kind"] for d in diffs]
    assert kinds[0] == "route_missing" and "status" not in kinds  # named once, not twice
    assert {"body", "header", "upstream_count"} <= set(kinds)
    assert "heuristic route_missing" in diffs[0]["detail"]
    assert "deliberately" in diffs[0]["detail"]


def test_route_missing_needs_an_empty_or_default_body():
    base = [{"name": "c", "status": 200, "body": {"x": 1}}]
    app_404 = [{"name": "c", "status": 404, "body": {"detail": "note 7 not found"}}]
    assert diff_behaviour(base, app_404)["cases"][0]["diffs"][0]["kind"] == "status"
    fastapi_404 = [{"name": "c", "status": 404, "body": {"detail": "Not Found"}}]
    assert diff_behaviour(base, fastapi_404)["cases"][0]["diffs"][0]["kind"] == "route_missing"
    same = [{"name": "c", "status": 404, "body": None}]
    assert (
        diff_behaviour([{"name": "c", "status": 404, "body": None}], same)["equal"] is True
    )  # a 404 the source also gave is no finding


def test_record_stamps_source_id_on_the_recording():
    req = RecordRequest.model_validate(
        {"cases": CASES_A, "source": TARGET_SVC, "source_id": "svc-a"}
    )
    spec = bp.build_spec("record", req)
    result = {
        "ok": True,
        "cases": CASES_A,
        "sides": {"source": {"cases": [{"name": "echo", "status": 200, "body": {"s": "A"}}]}},
    }
    g = bp.grade(spec, result, exit_code=0)
    assert g.recording is not None and g.recording["source_id"] == "svc-a"


@pytest.mark.parametrize(
    "body, needle",
    [
        ({"cases": CASES_A, "corpora": [{"source_id": "a", "cases": CASES_A}]}, "inside each"),
        ({"corpora": [{"source_id": "a", "cases": CASES_A}] * 2}, "unique"),
        ({"corpora": [{"source_id": "a"}]}, "exactly one of cases_file or cases"),
        ({"corpora": [{"source_id": "a b", "cases": CASES_A}]}, "source_id"),
        ({"corpora": [{"source_id": "a", "cases": CASES_A, "route_map": {"x": "/a"}}]}, "start"),
        ({"corpora": [{"source_id": "a", "cases": CASES_A, "baseline": "recording"}]}, "exactly"),
        (
            {"corpora": [{"source_id": "a", "cases": CASES_A}], "baseline": "recording"},
            "inside each corpus",
        ),
        (
            {"corpora": [{"source_id": "a", "cases": CASES_A, "source": TARGET_SVC}]},
            "corpus.source is for compare",
        ),
    ],
)
def test_replay_corpora_validation(body, needle):
    with pytest.raises(ValueError, match=needle):
        ReplayRequest.model_validate({"target": TARGET_SVC, **body})


def test_service_upstream_validation():
    with pytest.raises(ValueError, match="reserved"):
        ReplayRequest.model_validate(
            {"cases": CASES_A, "target": {**TARGET_SVC, "upstreams": {"default": ["X"]}}}
        )
    with pytest.raises(ValueError, match="both"):
        ReplayRequest.model_validate(
            {
                "cases": CASES_A,
                "target": {**TARGET_SVC, "upstream_env": ["KAG_URL"]},
            }
        )
    with pytest.raises(ValueError, match="invalid name or value"):
        ReplayRequest.model_validate(
            {"cases": CASES_A, "target": {**TARGET_SVC, "request_headers": {"A": "x\r\ny"}}}
        )
    with pytest.raises(ValueError, match="needs a source"):
        CompareRequest.model_validate(
            {"target": TARGET_SVC, "corpora": [{"source_id": "a", "cases": CASES_A}]}
        )
    with pytest.raises(ValueError, match="send `source`"):
        CompareRequest.model_validate({"target": TARGET_SVC, "cases": CASES_A})
