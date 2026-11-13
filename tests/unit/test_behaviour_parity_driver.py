"""The in-workspace behaviour-parity driver, run in-process (stdlib fake service, no DB).

The real-fixture runs (real FastAPI service, real postgres:18, driver as a
subprocess) are in tests/integration/test_behaviour_parity_fixture.py.
"""

from __future__ import annotations

import http.client
import json
import socket
import sys
import textwrap
from pathlib import Path

import pytest

from services import behaviour_parity_driver as drv

FAKE_URL_WS_PG = [sys.executable, "-c", "print('postgresql://u@127.0.0.1:1/x')"]

# A tiny service: /health; POST /echo forwards its JSON body to $UPSTREAM_URL/enrich
# and answers with the upstream's status + body (502 on transport failure).
FAKE_SERVICE = textwrap.dedent(
    """
    import http.client, json, os, sys, urllib.parse
    from http.server import BaseHTTPRequestHandler, HTTPServer

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
                return self._send(200, {"status": "ok", "db": os.environ["DATABASE_URL"]})
            if self.path == "/text":
                data = b"plain"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", "5")
                self.end_headers()
                return self.wfile.write(data)
            if self.path == "/die":
                os._exit(3)
            self._send(404, {"detail": "nope"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"null")
            up = urllib.parse.urlparse(os.environ["UPSTREAM_URL"])
            try:
                c = http.client.HTTPConnection(up.hostname, up.port, timeout=5)
                c.request("POST", "/enrich?x=1", body=json.dumps(body),
                          headers={"Content-Type": "application/json",
                                   "X-Tenant-Id": self.headers.get("X-Tenant-Id", "")})
                r = c.getresponse()
                raw = r.read()
                return self._send(r.status, json.loads(raw) if raw else None)
            except (OSError, http.client.HTTPException):
                return self._send(502, {"detail": "upstream_unavailable"})

    HTTPServer(("127.0.0.1", int(os.environ["PORT"])), H).serve_forever()
    """
)


def _write(run_dir: Path, config: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "driver_config.json").write_text(json.dumps(config))


def _run(run_dir: Path) -> tuple[int, dict]:
    code = drv.main([str(run_dir)])
    return code, json.loads((run_dir / "result.json").read_text())


@pytest.fixture
def fake_service(tmp_path: Path) -> list[str]:
    script = tmp_path / "fake_service.py"
    script.write_text(FAKE_SERVICE)
    return [sys.executable, str(script)]


def _side(start: list[str], **kw) -> dict:
    return {
        "role": "source",
        "workdir": "/",
        "start": start,
        "migrate": None,
        "upstream_env": ["UPSTREAM_URL"],
        "health_path": "/health",
        "startup_timeout": 20,
        **kw,
    }


CASES = [
    {"name": "health", "method": "GET", "path": "/health", "headers": {}, "body": None,
     "upstream_responses": []},
    {"name": "echo_ok", "method": "POST", "path": "/echo", "headers": {"X-Tenant-Id": "t1"},
     "body": {"a": 1}, "upstream_responses": [{"status": 200, "body": {"summary": "s"}}]},
    {"name": "echo_connect", "method": "POST", "path": "/echo", "headers": {},
     "body": {"a": 2}, "upstream_responses": [{"error": "connect"}]},
    {"name": "echo_unscripted", "method": "POST", "path": "/echo", "headers": {},
     "body": {"a": 3}, "upstream_responses": []},
    {"name": "unconsumed", "method": "GET", "path": "/missing", "headers": {}, "body": None,
     "upstream_responses": [{"status": 200, "body": {}}]},
    {"name": "text", "method": "GET", "path": "/text", "headers": {}, "body": None,
     "upstream_responses": []},
]  # fmt: skip


def test_driver_replays_cases_and_records_upstream_calls(tmp_path, fake_service):
    run_dir = tmp_path / ("a" * 32)
    _write(
        run_dir,
        {
            "ws_pg": FAKE_URL_WS_PG,
            "cases": CASES,
            "default_headers": {"X-Tenant-Id": "default", "X-Trace": "1"},
            "response_headers": ["content-type"],
            "sides": [_side(fake_service)],
        },
    )
    code, out = _run(run_dir)
    assert code == 0, out
    assert out["ok"] is True and out["cases_total"] == 6 and out["cases"] == CASES
    side = out["sides"]["source"]
    assert side["database"] == "parity_aaaaaaaaaaaa_source"
    by = {c["name"]: c for c in side["cases"]}
    assert by["health"]["status"] == 200
    assert by["health"]["body"]["db"] == "postgresql://u@127.0.0.1:1/x"
    assert by["health"]["headers"] == {"content-type": "application/json"}

    ok = by["echo_ok"]
    assert ok["status"] == 200 and ok["body"] == {"summary": "s"}
    assert ok["upstream_unconsumed"] == 0
    [call] = ok["upstream_calls"]
    assert call["method"] == "POST" and call["path"] == "/enrich?x=1"
    assert call["body"] == {"a": 1}
    assert call["headers"]["x-tenant-id"] == "t1"  # the case header wins over default
    assert call["headers"]["content-type"] == "application/json"

    assert by["echo_connect"]["status"] == 502
    assert by["echo_connect"]["upstream_calls"][0]["body"] == {"a": 2}
    assert by["echo_connect"]["upstream_calls"][0]["headers"]["x-tenant-id"] == "default"
    assert by["echo_unscripted"]["status"] == 599
    assert by["echo_unscripted"]["upstream_calls"][0]["unscripted"] is True
    assert by["unconsumed"]["status"] == 404 and by["unconsumed"]["upstream_unconsumed"] == 1
    assert by["text"]["body"] is None and by["text"]["body_text"] == "plain"


def test_service_crash_mid_run_records_per_case_errors(tmp_path, fake_service):
    run_dir = tmp_path / ("b" * 32)
    cases = [
        {"name": "die", "method": "GET", "path": "/die"},
        {"name": "after", "method": "GET", "path": "/health"},
    ]
    _write(run_dir, {"ws_pg": FAKE_URL_WS_PG, "cases": cases, "sides": [_side(fake_service)]})
    code, out = _run(run_dir)
    assert code == 0
    side = out["sides"]["source"]
    assert all("error" in c for c in side["cases"])
    assert side["service_exited"] == 3


@pytest.mark.parametrize(
    "config_patch, side_patch, needle",
    [
        ({"ws_pg": [sys.executable, "-c", "import sys; sys.exit(3)"]}, {}, "exited 3 without a URL"),
        ({"ws_pg": ["/nonexistent/ws-pg"]}, {}, "failed to run"),
        ({}, {"migrate": [sys.executable, "-c", "print('migrating'); raise SystemExit(4)"]}, "migrate exited 4"),
        ({}, {"migrate": ["/nonexistent/alembic"]}, "migrate command failed to run"),
        ({}, {"start": [sys.executable, "-c", "raise SystemExit(5)"]}, "exited with 5 before"),
        ({}, {"start": [sys.executable, "-c", "import time; time.sleep(30)"], "startup_timeout": 1}, "not healthy"),
        ({}, {"start": ["/nonexistent/svc"]}, "start command failed to run"),
        ({}, {"start": []}, "start must be a non-empty argv list"),
        ({}, {"role": "Bad-Role"}, "invalid role"),
        ({"sides": []}, {}, "no sides"),
        ({"cases": []}, {}, "no cases"),
        ({"cases": [{"name": "a", "method": "GET", "path": "/"}] * 2}, {}, "duplicate case name"),
        ({"cases": [{"name": "a", "method": "GET", "path": "x"}]}, {}, "must start with '/'"),
        ({"cases": [{"method": "GET", "path": "/"}]}, {}, "has no name"),
        ({"cases": [{"name": "a", "path": "/"}]}, {}, "needs method and path"),
    ],
)  # fmt: skip
def test_driver_errors_are_explicit(tmp_path, fake_service, config_patch, side_patch, needle):
    run_dir = tmp_path / ("c" * 32)
    config = {"ws_pg": FAKE_URL_WS_PG, "cases": CASES[:1], "sides": [_side(fake_service)]}
    config["sides"][0].update(side_patch)
    config.update(config_patch)
    _write(run_dir, config)
    code, out = _run(run_dir)
    assert code == 1
    assert out["ok"] is False and needle in out["error"], out["error"]


def test_cases_file_loading(tmp_path, fake_service):
    cases_file = tmp_path / "requests.jsonl"
    cases_file.write_text("\n".join(json.dumps(c) for c in CASES[:2]) + "\n\n")
    run_dir = tmp_path / ("d" * 32)
    _write(
        run_dir,
        {"ws_pg": FAKE_URL_WS_PG, "cases_file": str(cases_file), "sides": [_side(fake_service)]},
    )
    code, out = _run(run_dir)
    assert code == 0 and out["cases_total"] == 2

    cases_file.write_text("{not json\n")
    code, out = _run(run_dir)
    assert code == 1 and "line 1 is not JSON" in out["error"]
    _write(run_dir, {"ws_pg": FAKE_URL_WS_PG, "cases_file": str(tmp_path / "nope"), "sides": []})
    code, out = _run(run_dir)
    assert code == 1 and "not readable" in out["error"]
    _write(run_dir, {"sides": []})
    code, out = _run(run_dir)
    assert code == 1 and "neither cases nor cases_file" in out["error"]


def test_unreadable_config_still_writes_result(tmp_path):
    run_dir = tmp_path / ("e" * 32)
    run_dir.mkdir()
    (run_dir / "driver_config.json").write_text("{broken")
    code, out = _run(run_dir)
    assert code == 1 and out["error"].startswith("driver crashed") and "traceback" in out


def test_usage():
    assert drv.main([]) == 2


def test_stub_chunked_body_custom_headers_and_text():
    stub = drv.UpstreamStub().start()
    try:
        stub.begin_case(
            [
                {"status": 201, "body": {"x": 1}, "headers": {"X-Up": "1"}},
                {"status": 200, "body_text": "raw", "headers": {"Content-Type": "text/plain"}},
                {"status": 204, "body": None},
            ]
        )
        conn = http.client.HTTPConnection("127.0.0.1", stub.port, timeout=5)
        conn.putrequest("PUT", "/chunked")
        conn.putheader("Transfer-Encoding", "chunked")
        conn.endheaders()
        conn.send(b'4\r\n{"a"\r\n3\r\n:1}\r\n0\r\n\r\n')
        r = conn.getresponse()
        assert r.status == 201 and r.getheader("X-Up") == "1" and json.loads(r.read()) == {"x": 1}
        conn.request("POST", "/form", body=b"a=1", headers={"Content-Type": "x-www-form"})
        r = conn.getresponse()
        assert r.read() == b"raw" and r.getheader("Content-Type") == "text/plain"
        conn.request("DELETE", "/x")
        r = conn.getresponse()
        assert r.status == 204 and r.read() == b""
        conn.close()
        calls, left = stub.end_case()
        assert left == 0
        assert [c["method"] for c in calls] == ["PUT", "POST", "DELETE"]
        assert calls[0]["body"] == {"a": 1}
        assert calls[1]["body"] is None and calls[1]["body_text"] == "a=1"
        assert calls[2]["body"] is None and "body_text" not in calls[2]
    finally:
        stub.stop()


def test_stub_connect_error_drops_the_connection():
    stub = drv.UpstreamStub().start()
    try:
        stub.begin_case([{"error": "connect"}])
        conn = http.client.HTTPConnection("127.0.0.1", stub.port, timeout=5)
        conn.request("POST", "/enrich", body=b"{}", headers={"Content-Type": "application/json"})
        with pytest.raises((http.client.HTTPException, OSError)):
            conn.getresponse()
        conn.close()
        calls, left = stub.end_case()
        assert left == 0 and calls[0]["path"] == "/enrich"
    finally:
        stub.stop()


def test_free_port_and_parse_body():
    port = drv.free_port()
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))
    assert drv.parse_body(b"") == (None, None)
    assert drv.parse_body(b'{"a":1}') == ({"a": 1}, None)
    assert drv.parse_body(b"x") == (None, "x")


def test_kill_group_handles_exited_and_stubborn_processes():
    import subprocess

    done = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    done.wait()
    drv._kill_group(done)  # already exited: no-op
    stubborn = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('r', flush=True); time.sleep(60)",
        ],  # fmt: skip
        start_new_session=True,
        stdout=subprocess.PIPE,
    )
    assert stubborn.stdout is not None and stubborn.stdout.readline() == b"r\n"
    drv._kill_group(stubborn, grace=0.5)
    assert stubborn.returncode is not None
