"""WebSocket + SSE record/replay/compare (gap 15): the stdlib driver against a real server.

The server (tests/data/behaviour_parity/ws_sse_service.py) has its own server-side
RFC 6455 codec; each planted mutation (env knob) must surface as exactly the
expected difference, and an identical server must compare equal.
"""

from __future__ import annotations

import base64
import json
import socket
import struct
import sys
from pathlib import Path

import pytest

from services import behaviour_parity_driver as drv
from services.behaviour_diff import DiffOptions, baseline_from_expected, diff_behaviour
from services.behaviour_parity import grade

SERVICE = Path(__file__).resolve().parent.parent / "data" / "behaviour_parity" / "ws_sse_service.py"
FAKE_WS_PG = [sys.executable, "-c", "print('postgresql://u@127.0.0.1:1/x')"]

WS_ECHO = {
    "name": "ws_echo",
    "kind": "websocket",
    "path": "/ws",
    "subprotocols": ["v1.parity"],
    "steps": [
        {"recv": 1},
        {"send": {"op": "echo", "v": 1}},
        {"recv": 1},
        {"send": {"op": "echo", "v": 2}},
        {"recv": 1},
        {"send": {"op": "bye"}},
    ],
}
WS_BURST = {
    "name": "ws_burst",
    "kind": "websocket",
    "path": "/ws",
    "ordering": "unordered",
    "steps": [{"recv": 1}, {"send": {"op": "burst", "n": 3}}, {"recv": 3}, {"close": 1000}],
}
SSE = {"name": "ticks", "kind": "sse", "path": "/events", "max_events": 10, "timeout": 3}
HTTP = {"name": "items", "method": "GET", "path": "/items"}
MIXED = [HTTP, WS_ECHO, SSE, WS_BURST]
IGNORE = ["/data/ts"]


def _side(key: str, role: str, cases: list[dict], **env: str) -> dict:
    return {
        "role": role,
        "key": key,
        "workdir": "/",
        "start": [sys.executable, str(SERVICE)],
        "migrate": None,
        "env": env,
        "upstream_env": [],
        "health_path": "/health",
        "startup_timeout": 20,
        "cases": cases,
    }


def _compare(tmp_path: Path, cases: list[dict], *, opts: dict | None = None, **target_env: str):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    cfg = {
        "ws_pg": FAKE_WS_PG,
        "response_headers": ["content-type"],
        "request_timeout": 5,
        "sides": [_side("source", "source", cases), _side("target", "target", cases, **target_env)],
    }
    (run_dir / "driver_config.json").write_text(json.dumps(cfg))
    code = drv.main([str(run_dir)])
    result = json.loads((run_dir / "result.json").read_text())
    assert code == 0, result.get("error")
    # per-side case lists: compare via the plain two-side shape the grader reads
    result["cases"] = cases
    spec = {
        "op": "compare",
        "baseline": "source",
        "ignore_paths": [],
        "upstream_ignore_paths": [],
        "upstream_headers": None,
        "response_headers": ["content-type"],
        "frame_ignore_paths": IGNORE,
        **(opts or {}),
    }
    return result, grade(spec, result, exit_code=0)


def _diffs(verdict) -> list[tuple[str, str, str]]:
    return [(c.name, d.kind, d.pointer) for c in verdict.cases for d in c.diffs]


@pytest.fixture(scope="module")
def module_tmp(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("streams")


def test_mixed_corpus_identical_service_is_equal(tmp_path: Path) -> None:
    result, verdict = _compare(tmp_path, MIXED)
    assert _diffs(verdict) == []
    assert verdict.computed and verdict.equal and verdict.cases_total == 4
    src = {c["name"]: c for c in result["sides"]["source"]["cases"]}
    assert "kind" not in src["items"]  # an HTTP case records exactly as before
    ws = src["ws_echo"]
    assert ws["status"] == 101 and ws["subprotocol"] == "v1.parity"
    assert [len(g) for g in ws["frames"]] == [1, 1, 1, 0]  # 3 recv groups + empty tail
    assert ws["frames"][0][0]["data"]["type"] == "hello"
    assert ws["frames"][2][0]["data"] == {"type": "echo", "v": 2}  # the ping was not recorded
    assert ws["close"] == {"code": 4000, "reason": "bye", "by": "server"}
    sse = src["ticks"]
    assert sse["ended"] == "eof" and [e["id"] for e in sse["events"]] == ["1", "2", "3"]
    assert sse["events"][0]["event"] == "tick" and sse["events"][0]["data"]["n"] == 1
    burst = src["ws_burst"]
    assert burst["close"]["by"] == "client" and burst["close"]["code"] == 1000
    kinds = {c.name: c.kind for c in verdict.cases}
    assert kinds == {"items": None, "ws_echo": "websocket", "ticks": "sse", "ws_burst": "websocket"}


def test_planted_ws_mutation_is_exactly_one_frame_diff(tmp_path: Path) -> None:
    _, verdict = _compare(tmp_path, [WS_ECHO], MUTATE="echo")
    assert _diffs(verdict) == [("ws_echo", "frame", "/frames/2/0/data/v")]
    d = verdict.cases[0].diffs[0]
    assert (d.expected, d.actual) == (2, 3)
    assert verdict.computed and verdict.equal is False


def test_planted_sse_mutation_is_exactly_one_event_diff(tmp_path: Path) -> None:
    _, verdict = _compare(tmp_path, [SSE], MUTATE="sse")
    assert _diffs(verdict) == [("ticks", "frame", "/events/1/data/n")]


def test_without_frame_ignore_the_timestamps_differ(tmp_path: Path) -> None:
    """Negative control for the normalizer: the hello/ts and tick/ts values DO differ."""
    _, verdict = _compare(tmp_path, [WS_ECHO, SSE], opts={"frame_ignore_paths": []})
    pointers = {p for _, k, p in _diffs(verdict) if k == "frame"}
    assert "/frames/0/0/data/ts" in pointers and "/events/0/data/ts" in pointers


def test_unordered_group_tolerates_reordering_strict_does_not(tmp_path: Path) -> None:
    _, verdict = _compare(tmp_path, [WS_BURST], SHUFFLE="1")
    assert verdict.equal is True  # ordering: unordered
    strict = {**WS_BURST, "ordering": "strict"}
    _, verdict = _compare(tmp_path / "strict", [strict], SHUFFLE="1")
    assert {(k, p) for _, k, p in _diffs(verdict)} == {
        ("frame", "/frames/1/0/data/i"),
        ("frame", "/frames/1/2/data/i"),
    }


def test_drop_frames_removes_heartbeats(tmp_path: Path) -> None:
    """Only the target sends heartbeats; everything lands in the drained tail group."""
    case = {"name": "hb", "kind": "websocket", "path": "/ws", "drain": 1.0,
            "steps": [{"send": {"op": "echo", "v": 1}}]}  # fmt: skip
    _, verdict = _compare(tmp_path / "plain", [case], HB="1")
    assert ("hb", "frame_count", "/frames/0") in _diffs(verdict)  # negative control
    drop = {"drop_frames": [{"pointer": "/data/type", "equals": "hb"}]}
    _, verdict = _compare(tmp_path / "drop", [case], opts=drop, HB="1")
    assert verdict.equal is True, _diffs(verdict)


def test_rejected_handshake_records_status_and_body(tmp_path: Path) -> None:
    case = {**WS_ECHO, "name": "rejected", "headers": {"X-Reject": "1"}}
    result, verdict = _compare(tmp_path, [case])
    rec = result["sides"]["source"]["cases"][0]
    assert rec["status"] == 403 and rec["body"] == {"detail": "forbidden"}
    assert rec["frames"] == [] and rec["close"] is None
    assert verdict.equal


def test_ws_route_missing_on_target(tmp_path: Path) -> None:
    src_case = dict(WS_ECHO)
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    tgt = _side("target", "target", [src_case])
    tgt["route_map"] = [{"from": "/ws", "to": "/nope"}]
    cfg = {"ws_pg": FAKE_WS_PG, "sides": [_side("source", "source", [src_case]), tgt]}
    (run_dir / "driver_config.json").write_text(json.dumps(cfg))
    assert drv.main([str(run_dir)]) == 0
    sides = json.loads((run_dir / "result.json").read_text())["sides"]
    out = diff_behaviour(sides["source"]["cases"], sides["target"]["cases"], DiffOptions())
    kinds = [d["kind"] for d in out["cases"][0]["diffs"]]
    assert kinds[0] == "route_missing" and "frame_groups" in kinds and "status" not in kinds
    assert out["cases"][0]["target_path"] == "/nope"


# ---------------------------------------------------------------------------
# Pure pieces: codec, SSE parser, expected baselines, kind mismatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [0, 5, 125, 126, 65535, 65536, 70000])
def test_frame_round_trip_masked_and_not(size: int) -> None:
    payload = bytes(range(256)) * (size // 256) + bytes(range(size % 256))
    for mask in (True, False):
        raw = drv.ws_encode_frame(drv.OP_BINARY, payload, mask=mask)
        assert drv.ws_parse_frame(raw[:-1] if raw else raw) is None or size == 0
        op, fin, got, used = drv.ws_parse_frame(raw)
        assert (op, fin, got, used) == (drv.OP_BINARY, True, payload, len(raw))


def test_frame_size_cap() -> None:
    head = struct.pack("!BBQ", 0x82, 127, drv.WS_MAX_MESSAGE + 1)
    with pytest.raises(ValueError):
        drv.ws_parse_frame(head)


def test_client_reassembles_fragments_and_answers_ping() -> None:
    a, b = socket.socketpair()
    try:
        part1 = struct.pack("!BB", 0x01, 3) + b'{"a'  # text, FIN=0
        ping = struct.pack("!BB", 0x89, 1) + b"x"
        part2 = struct.pack("!BB", 0x80, 4) + b'":1}'  # continuation, FIN=1
        a.sendall(part1 + ping + part2)
        client = drv.WsClient(b)
        import time as _t

        msg = client.recv_message(_t.monotonic() + 2)
        assert msg == {"type": "text", "data": {"a": 1}}
        pong = drv.ws_parse_frame(a.recv(100))
        assert pong is not None and pong[0] == drv.OP_PONG and pong[2] == b"x"
        a.sendall(struct.pack("!BB", 0x82, 2) + b"\x00\xff")
        assert client.recv_message(_t.monotonic() + 2) == {
            "type": "binary",
            "b64": base64.b64encode(b"\x00\xff").decode(),
        }
        a.sendall(struct.pack("!BB", 0x81, 2) + b"hi")
        assert client.recv_message(_t.monotonic() + 2) == {"type": "text", "text": "hi"}
        with pytest.raises(drv.WsTimeoutError):
            client.recv_message(_t.monotonic() + 0.05)
        a.close()
        with pytest.raises(drv.WsClosedError):
            client.recv_message(_t.monotonic() + 1)
    finally:
        b.close()


def test_ws_accept_matches_rfc_example() -> None:
    # RFC 6455 section 1.3 published handshake example, not a credential.
    expected = "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="  # pragma: allowlist secret
    assert drv.ws_accept("dGhlIHNhbXBsZSBub25jZQ==") == expected


def test_parse_sse_lines() -> None:
    lines = [": c", "event: a", "id: 7", "retry: 10", "data: {\"x\":", "data: 1}", "",
             "data: plain", "", "event: empty", "", "data:nospace", ""]  # fmt: skip
    assert drv.parse_sse_lines(lines) == [
        {"event": "a", "id": "7", "retry": 10, "data": {"x": 1}},
        {"event": "message", "text": "plain"},
        {"event": "message", "text": "nospace"},
    ]


def test_load_cases_validates_kinds() -> None:
    with pytest.raises(drv.DriverError, match="unknown kind"):
        drv.load_cases({"cases": [{"name": "x", "kind": "grpc", "path": "/"}]})
    with pytest.raises(drv.DriverError, match="steps"):
        drv.load_cases({"cases": [{"name": "x", "kind": "websocket", "path": "/"}]})
    cases = drv.load_cases({"cases": [{"name": "x", "kind": "sse", "path": "/e"}]})
    assert cases[0]["method"] == "GET"


def test_expected_baseline_for_streams() -> None:
    ws_case = {**WS_ECHO, "expected_status": 101,
               "expected_frames": [[{"type": "text", "data": {"type": "hello"}}],
                                   [{"type": "text", "data": {"type": "echo", "v": 1}}],
                                   [{"type": "text", "data": {"type": "echo", "v": 2}}]],
               "expected_close": {"code": 4000}}  # fmt: skip
    sse_case = {**SSE, "expected_status": 200,
                "expected_events": [{"event": "tick", "id": str(n), "data": {"n": n}}
                                    for n in (1, 2, 3)]}  # fmt: skip
    base = baseline_from_expected([ws_case, sse_case])
    assert base[0]["frames"][-1] == []  # tail appended
    actual = [
        {"name": "ws_echo", "kind": "websocket", "status": 101, "body": None,
         "frames": [[{"type": "text", "data": {"type": "hello", "ts": 1}}],
                    [{"type": "text", "data": {"type": "echo", "v": 1}}],
                    [{"type": "text", "data": {"type": "echo", "v": 2}}], []],
         "close": {"code": 4000, "reason": "bye", "by": "server"}, "subprotocol": "v1.parity",
         "upstream_calls": []},
        {"name": "ticks", "kind": "sse", "status": 200, "body": None, "ended": "eof",
         "events": [{"event": "tick", "id": str(n), "data": {"n": n, "ts": 5}} for n in (1, 2, 3)],
         "upstream_calls": []},
    ]  # fmt: skip
    opts = DiffOptions.build(frame_ignore_paths=["/data/ts"])
    assert diff_behaviour(base, actual, opts)["equal"]
    actual[0]["close"]["code"] = 1000
    out = diff_behaviour(base, actual, opts)
    assert [(d["kind"], d["pointer"]) for d in out["cases"][0]["diffs"]] == [
        ("ws_close", "/close/code")
    ]


def test_kind_mismatch_is_one_diff() -> None:
    base = [{"name": "x", "status": 200, "body": None}]
    got = [{"name": "x", "kind": "sse", "status": 200, "events": [], "body": None}]
    out = diff_behaviour(base, got)
    assert [d["kind"] for d in out["cases"][0]["diffs"]] == ["kind"]


def test_unordered_missing_and_unexpected() -> None:
    base = [{"name": "w", "kind": "websocket", "ordering": "unordered", "status": 101,
             "frames": [[{"d": 1}, {"d": 2}]], "close": None}]  # fmt: skip
    got = [{"name": "w", "kind": "websocket", "ordering": "unordered", "status": 101,
            "frames": [[{"d": 2}, {"d": 3}]], "close": None}]  # fmt: skip
    diffs = diff_behaviour(base, got)["cases"][0]["diffs"]
    assert [(d["kind"], d["pointer"]) for d in diffs] == [
        ("frame_missing", "/frames/0/0"),
        ("frame_unexpected", "/frames/0/1"),
    ]
