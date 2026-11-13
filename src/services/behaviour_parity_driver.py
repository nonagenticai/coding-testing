"""Behaviour-parity driver: record one service's behaviour over an ordered case list.

THIS FILE RUNS INSIDE A CODING WORKSPACE, not in coding-testing. coding-testing
ships it as package data, writes it to ``/tmp/coding-testing/<nonce>/`` in the
workspace and runs it as ONE async exec job::

    python3 -I parity_driver.py <run_dir>

It therefore uses the Python standard library ONLY (the workspace has python3 and
nothing else is guaranteed) and stays compatible with Python 3.9+.

Input ``<run_dir>/driver_config.json``::

    {"ws_pg": ["ws-pg"],                  # argv prefix; `<ws_pg> createdb <name>` prints a URL
     "cases_file": "/workspace/.../requests.jsonl",   # or "cases": [...] inline
     "default_headers": {"X-Tenant-Id": "t"},         # merged under each case's headers
     "response_headers": ["content-type"],            # response headers captured per case
     "request_timeout": 30,
     "sides": [{"role": "source", "workdir": "/workspace/src",
                "migrate": ["uv", "run", "alembic", "upgrade", "head"] | null,
                "start": ["uv", "run", "uvicorn", "app:app", "--port", "{port}"],
                "env": {}, "upstream_env": ["UPSTREAM_URL"], "health_path": "/health",
                "startup_timeout": 120, "migrate_timeout": 600}]}

For each side, in order: a FRESH database (``ws-pg createdb``), the migrate command
with ``DATABASE_URL``, an upstream STUB on a free loopback port, the service on
another free loopback port (env ``DATABASE_URL``, every ``upstream_env`` name ->
stub URL, ``PORT``, ``HOST=127.0.0.1``, ``BIND_ADDR=127.0.0.1:<port>``; ``{port}``,
``{bind_addr}``, ``{database_url}``, ``{upstream_url}`` substituted in argv), a
health wait, then every case IN ORDER. Per case the stub serves the case's
``upstream_responses`` in call order (``{"error": "connect"}`` closes the
connection without a response) and records every inbound request.

Output ``<run_dir>/result.json``; exit 0 when every side completed, 1 otherwise
(the error is in the JSON and on stderr). Logs of the migrate command and the
service go to ``<run_dir>/<key>.*.log``.

N sources -> 1 target (multi-corpus runs), per side, all optional:

* ``key``: the side's id in ``result.sides`` (default: its role); sides may share
  a role (``s0``/``s1`` are sources, ``t0``/``t1`` the target once per corpus);
* ``cases`` / ``cases_file``: this side's own case list (default: the global one);
  the loaded list is echoed as ``result.case_sets[key]``;
* ``route_map``: ``[{"from", "to", "method"?, "to_method"?}]`` rewrites each
  case's method + path before it is sent (:func:`map_route`); the result keeps
  the sent ``method``/``path`` and the recorded ``source_method``/``source_path``;
* ``default_headers`` (under the case's headers) and ``request_headers`` (over
  them, e.g. the target's auth);
* ``upstreams``: ``{name: [env names]}``: one stub PER named upstream besides the
  default one; ``upstream_responses`` may then be ``{name: [...]}`` (a list is
  the default upstream's) and every recorded call carries its ``upstream``.

Case kinds (``kind``, default ``http``; a corpus may mix them):

* ``websocket``: ``path`` + ``steps`` run over a stdlib RFC 6455 client
  (:class:`WsClient`): ``{"send": <json>}``, ``{"send_text": str}``,
  ``{"send_binary_b64": str}``, ``{"recv": n, "timeout"?: s}`` (wait for the next
  ``n`` data frames: one recorded GROUP per recv step, so it does not matter
  whether a frame arrived before or after the client's next send),
  ``{"sleep": s}`` and ``{"close": code?, "reason"?}``. After the steps the
  client drains for ``drain`` seconds (default 0.3), closes with 1000 when the
  server has not, and records the frames seen meanwhile as the final (tail)
  group. Recorded: handshake ``status`` (101, or the rejection's status + body),
  ``subprotocol``, ``frames`` (list of groups), ``close`` ({code, reason, by}).
  Ping is answered with pong and never recorded.
* ``sse``: ``method`` (default GET) + ``path`` (+ ``body``) read as an
  ``text/event-stream`` over http.client until ``max_events`` (default 100),
  end of stream, or ``timeout`` seconds without a line (default 5). Recorded:
  ``status``, headers, ``events`` ([{event, id?, data | text, retry?}]) and
  ``ended`` (``max_events`` | ``eof`` | ``idle_timeout``).

Both carry ``ordering`` (``strict`` default, or ``unordered`` = compared as a
multiset within each group) from the case into the recording.

CLI / library sides (``"kind": "cli"`` on a side; its cases are all ``kind: cli``,
a corpus never mixes cli with http/websocket/sse cases): no database, no stub, no
port. ``start`` is the argv PREFIX (``["uv", "run", "mytool"]`` or
``["target/release/mytool"]``); ``migrate``/``upstreams``/health are not used. Per
case ``{name, kind: "cli", args: [str], stdin?: str | <json>, env?: {}, timeout?: s,
stdout_format?: "text" | "json", compare_stderr?: bool}`` the driver runs
``start + args`` once (:func:`run_cli_case`) with cwd = the side's workdir, env =
os.environ | side env | case env, a non-str ``stdin`` sent as JSON, and records
``{name, kind, args, exit_code, stdout | stdout_json, stderr, duration_ms}``
(each stream read incrementally and capped at 1 MiB: past it the command is killed
and the case records ``truncated`` plus ``stdout_clipped``/``stderr_clipped``; a
case ``timeout`` is at most 300 s). In
``start`` and ``args``, ``{case_dir}`` is a fresh empty directory for the case (the
SAME path on every side, so outputs naming it compare equal) and ``{lib_harness}``
the library call harness shipped next to this file (``lib_call_harness.py``).
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import http.client
import json
import os
import re
import shutil
import signal
import socket
import struct
import subprocess  # nosec B404 - the driver's whole job is running the service under test
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DRIVER_VERSION = 4
LOG_TAIL_CHARS = 4000
_URL_RE = re.compile(r"(postgres(?:ql)?(?:\+[a-z0-9]+)?://\S+)")


CASE_KINDS = ("http", "websocket", "sse", "cli")
CLI_STREAM_CAP = 1 << 20  # bytes of stdout / stderr kept per cli case
CLI_MAX_TIMEOUT = 300  # seconds; a cli case may not ask for more
CLI_READ_CHUNK = 64 * 1024
LIB_HARNESS_NAME = "lib_call_harness.py"


class DriverError(Exception):
    """A side could not be run: nothing about its behaviour was computed."""


def _tail(path: str, limit: int = LOG_TAIL_CHARS) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - limit))
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def parse_body(raw: bytes) -> tuple[object, str | None]:
    """(JSON value or None, raw text when the body is non-empty and not JSON)."""
    if not raw:
        return None, None
    text = raw.decode("utf-8", "replace")
    try:
        return json.loads(text), None
    except ValueError:
        return None, text


def load_cases(config: dict) -> list[dict]:
    """The case list of ``config`` (a driver config or one side of it)."""
    cases = config.get("cases")
    if cases is None:
        path = config.get("cases_file")
        if not path:
            raise DriverError("driver_config has neither cases nor cases_file")
        try:
            with open(path, encoding="utf-8") as fh:
                lines = fh.read().splitlines()
        except OSError as e:
            raise DriverError(f"cases_file {path} is not readable: {e}") from e
        cases = []
        for i, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                cases.append(json.loads(line))
            except ValueError as e:
                raise DriverError(f"cases_file {path} line {i} is not JSON: {e}") from e
    if not isinstance(cases, list) or not cases:
        raise DriverError("no cases to replay")
    names = set()
    for i, c in enumerate(cases):
        if not isinstance(c, dict) or not isinstance(c.get("name"), str):
            raise DriverError(f"case #{i} has no name")
        if c["name"] in names:
            raise DriverError(f"duplicate case name {c['name']!r}")
        names.add(c["name"])
        kind = c.get("kind") or "http"
        if kind not in CASE_KINDS:
            raise DriverError(f"case {c['name']!r}: unknown kind {kind!r}")
        if kind == "cli":
            _check_cli_case(c)
            continue
        if kind != "http" and not isinstance(c.get("method"), str):
            c = cases[i] = {**c, "method": "GET"}
        if kind == "websocket" and not isinstance(c.get("steps"), list):
            raise DriverError(f"case {c['name']!r}: a websocket case needs a steps list")
        if not isinstance(c.get("method"), str) or not isinstance(c.get("path"), str):
            raise DriverError(f"case {c['name']!r} needs method and path")
        if not c["path"].startswith("/"):
            raise DriverError(f"case {c['name']!r}: path must start with '/'")
    kinds = {c.get("kind") or "http" for c in cases}
    if "cli" in kinds and len(kinds) > 1:
        raise DriverError("a corpus cannot mix cli cases with http/websocket/sse cases")
    return cases


def _check_cli_case(c: dict) -> None:
    name = c["name"]
    args = c.get("args", [])
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise DriverError(f"case {name!r}: a cli case needs args as a list of strings")
    env = c.get("env") or {}
    if not isinstance(env, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in env.items()
    ):
        raise DriverError(f"case {name!r}: env must map strings to strings")
    timeout = c.get("timeout")
    if timeout is not None and (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not 0 < timeout <= CLI_MAX_TIMEOUT
    ):
        raise DriverError(
            f"case {name!r}: timeout must be a positive number of seconds <= {CLI_MAX_TIMEOUT}"
        )
    if c.get("stdout_format", "text") not in ("text", "json"):
        raise DriverError(f"case {name!r}: stdout_format must be 'text' or 'json'")
    if not isinstance(c.get("compare_stderr", False), bool):
        raise DriverError(f"case {name!r}: compare_stderr must be a boolean")


def _prefix_matches(path: str, prefix: str) -> bool:
    if prefix.endswith("/"):
        return path.startswith(prefix) or path == prefix[:-1]
    return path == prefix or path.startswith(prefix + "/")


def map_route(method: str, path: str, rules: list) -> tuple[str, str]:
    """Rewrite (method, path) by the best matching rule (``from`` is segment-aligned).

    Best = the longest ``from``; on an equally long ``from`` a rule with ``method``
    beats one without; then the first listed. The matched prefix is replaced by
    ``to``; the mapped path ends with ``/`` exactly when the source path does (so
    ``{"from": "/api/", "to": "/v/"}`` maps ``/api`` to ``/v`` and ``/api/`` to
    ``/v/``). coding-testing imports THIS function for the OpenAPI prefix map and
    for the mapped paths it reports, so a route maps the same way everywhere.
    """
    raw_path, sep, query = path.partition("?")
    best = None
    best_rank = (-1, -1)
    for rule in rules or []:
        frm = rule.get("from") or "/"
        want = rule.get("method")
        if want and want.upper() != method.upper():
            continue
        rank = (len(frm), 1 if want else 0)
        if _prefix_matches(raw_path, frm) and rank > best_rank:
            best, best_rank = rule, rank
    if best is None:
        return method, path
    frm, to = best.get("from") or "/", best.get("to") or "/"
    rest = raw_path[len(frm) :] if raw_path.startswith(frm) else ""
    if to.endswith("/") and rest.startswith("/"):
        rest = rest[1:]
    elif not to.endswith("/") and rest and not rest.startswith("/"):
        rest = "/" + rest
    new_path = (to + rest) or "/"
    if not rest and not raw_path.endswith("/"):
        new_path = new_path.rstrip("/") or "/"
    return (best.get("to_method") or method).upper(), new_path + (sep + query if sep else "")


# ---------------------------------------------------------------------------
# Upstream stub
# ---------------------------------------------------------------------------


class UpstreamStub:
    """Serves scripted responses in call order and records every inbound request."""

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._lock = threading.Lock()
        self._responses: list = []
        self._calls: list[dict] = []
        self._inflight = 0
        #: load mode (perf driver): {(METHOD, path without query): response}; calls
        #: are answered from it and only COUNTED, never recorded.
        self._table: dict | None = None
        self.table_hits = 0
        self.table_misses = 0
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format, *args):  # noqa: A002 - stdlib signature
                return

            def _read_body(self) -> bytes:
                if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
                    chunks = []
                    while True:
                        size_line = self.rfile.readline().strip().split(b";")[0]
                        size = int(size_line or b"0", 16)
                        if size == 0:
                            while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                                pass
                            break
                        chunks.append(self.rfile.read(size))
                        self.rfile.readline()
                    return b"".join(chunks)
                length = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(length) if length > 0 else b""

            def _handle(self) -> None:
                with stub._lock:
                    stub._inflight += 1
                try:
                    raw = self._read_body()
                    body, body_text = parse_body(raw)
                    call = {
                        "upstream": stub.name,
                        "method": self.command,
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": body,
                    }
                    if body_text is not None:
                        call["body_text"] = body_text
                    with stub._lock:
                        if stub._table is not None:
                            scripted = stub._table.get((self.command, self.path.split("?")[0]))
                            if scripted is None:
                                stub.table_misses += 1
                            else:
                                stub.table_hits += 1
                        else:
                            scripted = stub._responses.pop(0) if stub._responses else None
                            if scripted is None:
                                call["unscripted"] = True
                            stub._calls.append(call)
                    if isinstance(scripted, dict) and scripted.get("error"):
                        # Transport failure: drop the connection without any response.
                        self.close_connection = True
                        try:
                            self.connection.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                        return
                    if scripted is None:
                        status, payload, headers = 599, {"detail": "parity_stub_unscripted"}, {}
                    else:
                        status = int(scripted.get("status", 200))
                        headers = scripted.get("headers") or {}
                        payload = scripted.get("body")
                    if isinstance(scripted, dict) and "body_text" in scripted:
                        data = str(scripted["body_text"]).encode()
                    elif payload is None:
                        data = b""
                    else:
                        data = json.dumps(payload).encode()
                    self.send_response(status)
                    lowered = {k.lower() for k in headers}
                    if data and "content-type" not in lowered:
                        self.send_header("Content-Type", "application/json")
                    for k, v in headers.items():
                        self.send_header(k, str(v))
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                finally:
                    with stub._lock:
                        stub._inflight -= 1

            do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _handle  # noqa: N815

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = int(self.server.server_address[1])
        self.url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> UpstreamStub:
        self._thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def set_table(self, table: dict | None) -> None:
        """Answer every call from ``table`` (load mode) or, with None, back to scripted."""
        with self._lock:
            self._table = None if table is None else dict(table)
            self._responses, self._calls = [], []

    def begin_case(self, responses: list) -> None:
        with self._lock:
            self._responses = list(responses or [])
            self._calls = []

    def end_case(self, settle: float = 2.0) -> tuple[list[dict], int]:
        """(recorded calls, number of scripted responses never consumed)."""
        deadline = time.monotonic() + settle
        while time.monotonic() < deadline:
            with self._lock:
                if self._inflight == 0:
                    break
            time.sleep(0.01)
        with self._lock:
            calls, left = self._calls, len(self._responses)
            self._calls, self._responses = [], []
        return calls, left


# ---------------------------------------------------------------------------
# Service process
# ---------------------------------------------------------------------------


def _subst(argv: list[str], values: dict[str, str]) -> list[str]:
    out = []
    for a in argv:
        for k, v in values.items():
            a = a.replace("{" + k + "}", v)
        out.append(a)
    return out


def _exit_status(proc: subprocess.Popen) -> int | None:
    """The process's exit code WITHOUT reaping it (None while it runs).

    An exited-but-unreaped leader stays a zombie, so its pid -- and therefore its
    process group id -- cannot be reused until :func:`_kill_group` reaps it.
    """
    if proc.returncode is not None:
        return proc.returncode
    try:
        info = os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        return proc.poll()
    if info is None:
        return None
    if info.si_code in (os.CLD_KILLED, os.CLD_DUMPED):
        return -info.si_status
    return info.si_status


def _wait_exit(proc: subprocess.Popen, timeout: float) -> int | None:
    """Wait up to ``timeout`` s for the process to exit, without reaping it."""
    deadline = time.monotonic() + timeout
    while True:
        code = _exit_status(proc)
        if code is not None or time.monotonic() >= deadline:
            return code
        time.sleep(0.02)


def _kill_group(proc: subprocess.Popen, grace: float = 10.0) -> None:
    """End ``proc`` and EVERYTHING in its process group (it was started with its own session).

    SIGTERM the group, give the leader up to ``grace`` seconds, then SIGKILL the whole
    group UNCONDITIONALLY: a background child left behind by a leader that already
    exited, or one that ignores SIGTERM, must not outlive the case/service. The group
    is signalled BEFORE the leader is reaped (callers only check exit via
    :func:`_exit_status`), so the pgid cannot have been reused by an unrelated process.
    """
    if proc.returncode is not None:
        return  # already reaped elsewhere: its pid may be reused, never signal it
    pgid = proc.pid  # start_new_session=True: the leader's pid is the group id
    if _exit_status(proc) is None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGTERM)
        _wait_exit(proc, grace)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGKILL)
    if _wait_exit(proc, 5.0) is None:
        with contextlib.suppress(OSError):
            proc.kill()  # not a group leader after all
    proc.wait(timeout=10)  # reap only now


def create_database(ws_pg: list[str], name: str, log_path: str) -> str:
    with open(log_path, "ab") as log:
        try:
            done = subprocess.run(  # nosec B603 - argv from the run's own config
                [*ws_pg, "createdb", name],
                stdout=subprocess.PIPE,
                stderr=log,
                timeout=180,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            raise DriverError(f"ws-pg createdb {name} failed to run: {e}") from e
    out = done.stdout.decode("utf-8", "replace")
    urls = _URL_RE.findall(out)
    if done.returncode != 0 or not urls:
        raise DriverError(
            f"ws-pg createdb {name} exited {done.returncode} without a URL: "
            f"{(out + _tail(log_path, 600))[-600:]}"
        )
    return urls[-1].strip("'\"")


def http_request(
    port: int, method: str, path: str, headers: dict, body: bytes | None, timeout: float
) -> tuple[int, dict, bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, data
    finally:
        conn.close()


def wait_healthy(proc: subprocess.Popen, port: int, path: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last = "no answer"
    while time.monotonic() < deadline:
        code = _exit_status(proc)
        if code is not None:
            raise DriverError(f"service exited with {code} before it became healthy")
        try:
            status, _, _ = http_request(port, "GET", path, {}, None, 3)
            if 200 <= status < 400:
                return
            last = f"HTTP {status}"
        except (OSError, http.client.HTTPException) as e:
            last = str(e)
        time.sleep(0.25)
    raise DriverError(f"service not healthy on {path} after {timeout:.0f}s ({last})")


# ---------------------------------------------------------------------------
# WebSocket (RFC 6455) client + SSE reader, stdlib only
# ---------------------------------------------------------------------------

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
OP_CONT, OP_TEXT, OP_BINARY, OP_CLOSE, OP_PING, OP_PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA
_WS_RESERVED_HEADERS = {
    "host",
    "upgrade",
    "connection",
    "sec-websocket-key",
    "sec-websocket-version",
    "sec-websocket-protocol",
    "content-length",
    "content-type",
}
WS_MAX_MESSAGE = 16 * 1024 * 1024


class WsTimeoutError(Exception):
    """No complete frame arrived before the deadline."""


class WsClosedError(Exception):
    """The peer closed the TCP connection."""


def ws_accept(key: str) -> str:
    """The Sec-WebSocket-Accept value a server must answer for ``key``."""
    digest = hashlib.sha1((key + WS_GUID).encode(), usedforsecurity=False).digest()
    return base64.b64encode(digest).decode()


def _xor_mask(data: bytes, key: bytes) -> bytes:
    if not data:
        return b""
    k = (key * (len(data) // 4 + 1))[: len(data)]
    return (int.from_bytes(data, "big") ^ int.from_bytes(k, "big")).to_bytes(len(data), "big")


def ws_encode_frame(opcode: int, payload: bytes, mask: bool) -> bytes:
    """One FIN frame; clients MUST mask (RFC 6455 5.3), servers must not."""
    n = len(payload)
    mbit = 0x80 if mask else 0
    if n < 126:
        head = struct.pack("!BB", 0x80 | opcode, mbit | n)
    elif n < 65536:
        head = struct.pack("!BBH", 0x80 | opcode, mbit | 126, n)
    else:
        head = struct.pack("!BBQ", 0x80 | opcode, mbit | 127, n)
    if not mask:
        return head + payload
    key = os.urandom(4)
    return head + key + _xor_mask(payload, key)


def ws_parse_frame(buf) -> tuple[int, bool, bytes, int] | None:
    """(opcode, fin, unmasked payload, bytes consumed), or None when ``buf`` is incomplete."""
    if len(buf) < 2:
        return None
    b0, b1 = buf[0], buf[1]
    n = b1 & 0x7F
    pos = 2
    if n == 126:
        if len(buf) < 4:
            return None
        n = struct.unpack("!H", bytes(buf[2:4]))[0]
        pos = 4
    elif n == 127:
        if len(buf) < 10:
            return None
        n = struct.unpack("!Q", bytes(buf[2:10]))[0]
        pos = 10
    if n > WS_MAX_MESSAGE:
        raise ValueError(f"frame of {n} bytes exceeds {WS_MAX_MESSAGE}")
    key = b""
    if b1 & 0x80:
        if len(buf) < pos + 4:
            return None
        key = bytes(buf[pos : pos + 4])
        pos += 4
    if len(buf) < pos + n:
        return None
    payload = bytes(buf[pos : pos + n])
    if key:
        payload = _xor_mask(payload, key)
    return b0 & 0x0F, bool(b0 & 0x80), payload, pos + n


def ws_frame_record(opcode: int, payload: bytes) -> dict:
    """How a data message is recorded: text (JSON-parsed when it is JSON) or base64 binary."""
    if opcode == OP_BINARY:
        return {"type": "binary", "b64": base64.b64encode(payload).decode()}
    text = payload.decode("utf-8", "replace")
    try:
        return {"type": "text", "data": json.loads(text)}
    except ValueError:
        return {"type": "text", "text": text}


class WsClient:
    """A minimal RFC 6455 client over one blocking socket with deadlines."""

    def __init__(self, sock: socket.socket, buffered: bytes = b"") -> None:
        self.sock = sock
        self.buf = bytearray(buffered)
        self.close: dict | None = None  # {code, reason, by}
        self._sent_close = False
        self._frag_op: int | None = None
        self._frag = bytearray()

    def _fill(self, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WsTimeoutError()
        self.sock.settimeout(remaining)
        try:
            chunk = self.sock.recv(65536)
        except socket.timeout as e:  # noqa: UP041 - 3.9 in the pod: not TimeoutError there
            raise WsTimeoutError() from e
        if not chunk:
            raise WsClosedError("connection closed by peer")
        self.buf += chunk

    def send(self, opcode: int, payload: bytes) -> None:
        self.sock.sendall(ws_encode_frame(opcode, payload, mask=True))

    def send_close(self, code: int = 1000, reason: str = "") -> None:
        if self._sent_close:
            return
        self._sent_close = True
        try:
            self.send(OP_CLOSE, struct.pack("!H", code) + reason.encode())
        except OSError:
            pass

    def recv_message(self, deadline: float) -> dict | None:
        """The next data message as a frame record; None once the close handshake ran."""
        while self.close is None:
            frame = ws_parse_frame(self.buf)
            if frame is None:
                self._fill(deadline)
                continue
            opcode, fin, payload, used = frame
            del self.buf[:used]
            if opcode == OP_PING:
                try:
                    self.send(OP_PONG, payload)
                except OSError:
                    pass
            elif opcode == OP_PONG:
                continue
            elif opcode == OP_CLOSE:
                code = struct.unpack("!H", payload[:2])[0] if len(payload) >= 2 else None
                self.close = {
                    "code": code,
                    "reason": payload[2:].decode("utf-8", "replace"),
                    "by": "client" if self._sent_close else "server",
                }
                self.send_close(code or 1000)
                return None
            elif opcode == OP_CONT:
                self._frag += payload
                if fin and self._frag_op is not None:
                    op, data = self._frag_op, bytes(self._frag)
                    self._frag_op, self._frag = None, bytearray()
                    return ws_frame_record(op, data)
            elif fin:
                return ws_frame_record(opcode, payload)
            else:
                self._frag_op, self._frag = opcode, bytearray(payload)
        return None


def ws_handshake(
    port: int, path: str, headers: dict, subprotocols: list, timeout: float
) -> tuple[int, dict, bytes, WsClient | None]:
    """(status, lower-cased headers, body of a rejection, client when upgraded to 101)."""
    sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    try:
        key = base64.b64encode(os.urandom(16)).decode()
        lines = [
            f"GET {path} HTTP/1.1",
            f"Host: 127.0.0.1:{port}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {key}",
            "Sec-WebSocket-Version: 13",
        ]
        if subprotocols:
            lines.append("Sec-WebSocket-Protocol: " + ", ".join(str(p) for p in subprotocols))
        for k, v in headers.items():
            if str(k).lower() not in _WS_RESERVED_HEADERS:
                lines.append(f"{k}: {v}")
        sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
        deadline = time.monotonic() + timeout
        client = WsClient(sock)
        while b"\r\n\r\n" not in client.buf:
            if len(client.buf) > 65536:
                raise DriverError("websocket handshake response headers exceed 64 KiB")
            client._fill(deadline)
        head, _, rest = bytes(client.buf).partition(b"\r\n\r\n")
        status_line, *header_lines = head.decode("latin-1").split("\r\n")
        parts = status_line.split(" ", 2)
        status = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        got: dict = {}
        for line in header_lines:
            k, _, v = line.partition(":")
            got[k.strip().lower()] = v.strip()
        client.buf = bytearray(rest)
        if status != 101:
            length = int(got.get("content-length") or 0)
            try:
                while len(client.buf) < length:
                    client._fill(deadline)
            except (WsTimeoutError, WsClosedError):
                pass
            sock.close()
            return status, got, bytes(client.buf[:length] if length else client.buf), None
        if got.get("sec-websocket-accept") != ws_accept(key):
            raise DriverError("websocket handshake: wrong Sec-WebSocket-Accept")
        return status, got, b"", client
    except BaseException:
        sock.close()
        raise


def _ws_recv_group(client: WsClient, n: int, timeout: float) -> list:
    group: list = []
    deadline = time.monotonic() + timeout
    while len(group) < n and client.close is None:
        try:
            msg = client.recv_message(deadline)
        except (WsTimeoutError, WsClosedError):
            break
        if msg is not None:
            group.append(msg)
    return group


def _ws_drain(client: WsClient, seconds: float) -> list:
    out: list = []
    deadline = time.monotonic() + max(0.0, seconds)
    while client.close is None:
        try:
            msg = client.recv_message(deadline)
        except (WsTimeoutError, WsClosedError):
            break
        if msg is not None:
            out.append(msg)
    return out


def run_ws_case(case: dict, port: int, path: str, headers: dict, out: dict, timeout: float) -> None:
    """Run a websocket case's steps; fills ``out`` (status, frames, close, ...)."""
    status, got, body, client = ws_handshake(
        port, path, headers, list(case.get("subprotocols") or []), timeout
    )
    out["status"] = status
    out["_headers"] = got
    out["subprotocol"] = got.get("sec-websocket-protocol")
    if client is None:
        parsed, text = parse_body(body)
        out["body"] = parsed
        if text is not None:
            out["body_text"] = text
        out["frames"] = []
        out["close"] = None
        return
    groups: list = []
    try:
        for step in case.get("steps") or []:
            if not isinstance(step, dict):
                raise DriverError(f"case {case['name']!r}: every step is an object")
            if client.close is not None:
                if "recv" in step:
                    groups.append([])
                continue
            if "send" in step:
                client.send(OP_TEXT, json.dumps(step["send"]).encode())
            elif "send_text" in step:
                client.send(OP_TEXT, str(step["send_text"]).encode())
            elif "send_binary_b64" in step:
                client.send(OP_BINARY, base64.b64decode(step["send_binary_b64"]))
            elif "recv" in step:
                groups.append(
                    _ws_recv_group(client, int(step["recv"]), float(step.get("timeout") or timeout))
                )
            elif "sleep" in step:
                time.sleep(min(float(step["sleep"]), 10.0))
            elif "close" in step:
                client.send_close(int(step.get("close") or 1000), str(step.get("reason") or ""))
            else:
                raise DriverError(f"case {case['name']!r}: unknown step {sorted(step)}")
        tail = _ws_drain(client, float(case.get("drain", 0.3)))
        if client.close is None:
            client.send_close(1000)
            tail.extend(_ws_drain(client, 2.0))
        groups.append(tail)
    except OSError as e:
        out["error"] = f"websocket failed: {type(e).__name__}: {e}"
    finally:
        client.sock.close()
    out["frames"] = groups
    out["close"] = client.close


def parse_sse_lines(lines) -> list:
    """SSE events (WHATWG event-stream parsing) from an iterable of decoded lines."""
    events: list = []
    data: list = []
    event, eid, retry = None, None, None
    for line in lines:
        if line == "":
            if data:
                text = "\n".join(data)
                ev: dict = {"event": event or "message"}
                if eid is not None:
                    ev["id"] = eid
                if retry is not None:
                    ev["retry"] = retry
                try:
                    ev["data"] = json.loads(text)
                except ValueError:
                    ev["text"] = text
                events.append(ev)
            data, event, eid, retry = [], None, None, None
            continue
        if line.startswith(":"):
            continue
        field, sep, value = line.partition(":")
        if sep and value.startswith(" "):
            value = value[1:]
        if field == "data":
            data.append(value)
        elif field == "event":
            event = value
        elif field == "id":
            eid = value
        elif field == "retry" and value.isdigit():
            retry = int(value)
    return events


def run_sse_case(
    case: dict, port: int, method: str, path: str, headers: dict, body, out: dict, timeout: float
) -> None:
    """Read a text/event-stream; fills ``out`` (status, events, ended, ...)."""
    max_events = int(case.get("max_events") or 100)
    idle = float(case.get("timeout") or 5)
    if not any(k.lower() == "accept" for k in headers):
        headers = {**headers, "Accept": "text/event-stream"}
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        out["status"] = resp.status
        got = {k.lower(): v for k, v in resp.getheaders()}
        out["_headers"] = got
        if "text/event-stream" not in (got.get("content-type") or ""):
            parsed, text = parse_body(resp.read())
            out["body"] = parsed
            if text is not None:
                out["body_text"] = text
            out["events"] = []
            out["ended"] = "not_a_stream"
            return
        if conn.sock is not None:
            conn.sock.settimeout(idle)
        events: list = []
        pending: list = []
        ended = "eof"
        while True:
            try:
                raw = resp.readline()
            except (socket.timeout, TimeoutError):  # noqa: UP041 - 3.9 compat
                ended = "idle_timeout"
                break
            if not raw:
                pending.append("")
                break
            pending.append(raw.decode("utf-8", "replace").rstrip("\r\n"))
            if pending[-1] == "":
                events.extend(parse_sse_lines(pending))
                pending = []
                if len(events) >= max_events:
                    ended = "max_events"
                    break
        if ended == "eof":
            events.extend(parse_sse_lines(pending))
        out["events"] = events[:max_events]
        out["ended"] = ended
    finally:
        conn.close()


def _merge_headers(*layers: dict) -> dict:
    """Later layers win; header names compare case-insensitively."""
    out: dict = {}
    for layer in layers:
        for k, v in (layer or {}).items():
            for existing in [e for e in out if e.lower() == str(k).lower()]:
                del out[existing]
            out[k] = v
    return out


def replay_case(
    case: dict,
    port: int,
    stub,
    default_headers: dict,
    response_headers: list[str],
    timeout: float,
    *,
    route_map: list | None = None,
    side_headers: dict | None = None,
    request_headers: dict | None = None,
) -> dict:
    """Send one case; ``stub`` is the default UpstreamStub or a {name: UpstreamStub} map."""
    stubs = stub if isinstance(stub, dict) else {"default": stub}
    headers = _merge_headers(
        default_headers or {}, side_headers or {}, case.get("headers") or {}, request_headers or {}
    )
    body: bytes | None = None
    if isinstance(case.get("raw_body"), str):
        body = case["raw_body"].encode()
    elif case.get("body") is not None:
        body = json.dumps(case["body"]).encode()
        if not any(k.lower() == "content-type" for k in headers):
            headers["Content-Type"] = "application/json"
    scripted = case.get("upstream_responses") or []
    if not isinstance(scripted, dict):
        scripted = {"default": scripted}
    orphan = 0  # responses scripted for an upstream this side has no stub for
    for name, responses in scripted.items():
        if name not in stubs:
            orphan += len(responses or [])
    for name, s in stubs.items():
        s.begin_case(scripted.get(name) or [])
    method, path = map_route(case["method"].upper(), case["path"], route_map or [])
    out: dict = {"name": case["name"], "method": method, "path": path}
    if (method, path) != (case["method"].upper(), case["path"]):
        out["source_method"] = case["method"].upper()
        out["source_path"] = case["path"]
    kind = case.get("kind") or "http"
    if kind != "http":
        out["kind"] = kind
        out["ordering"] = case.get("ordering") or "strict"
    started = time.monotonic()
    try:
        if kind == "websocket":
            run_ws_case(case, port, path, headers, out, timeout)
            got_headers = out.pop("_headers", {})
        elif kind == "sse":
            run_sse_case(case, port, method, path, headers, body, out, timeout)
            got_headers = out.pop("_headers", {})
        else:
            status, got_headers, data = http_request(port, method, path, headers, body, timeout)
            parsed, text = parse_body(data)
            out["status"] = status
            out["body"] = parsed
            if text is not None:
                out["body_text"] = text
        out["headers"] = {
            h.lower(): got_headers[h.lower()] for h in response_headers if h.lower() in got_headers
        }
    except (OSError, http.client.HTTPException, WsTimeoutError, WsClosedError, ValueError) as e:
        out.pop("_headers", None)
        out["error"] = f"request failed: {type(e).__name__}: {e}"
    out["duration_ms"] = round((time.monotonic() - started) * 1000, 1)
    calls: list = []
    left = orphan
    for name, s in stubs.items():
        got, n = s.end_case()
        calls.extend(got)
        left += n
    out["upstream_calls"] = calls
    out["upstream_unconsumed"] = left
    return out


def _side_basics(side: dict) -> tuple[str, str, str, list[str]]:
    """(role, key, workdir, start) of a side, validated."""
    role = side.get("role") or "side"
    key = side.get("key") or role
    for what, value in (("role", role), ("key", key)):
        if not re.match(r"^[a-z][a-z0-9_]{0,15}$", value):
            raise DriverError(f"invalid {what} {value!r}")
    workdir = side.get("workdir") or "/workspace"
    start = side.get("start")
    if not isinstance(start, list) or not start or not all(isinstance(a, str) for a in start):
        raise DriverError(f"{role}: start must be a non-empty argv list")
    return role, key, workdir, start


class _CappedReader(threading.Thread):
    """Reads one pipe incrementally, keeping at most CLI_STREAM_CAP bytes.

    On the first byte past the cap it sets ``over`` and fires ``on_over`` (which kills
    the process group), so a flooding command never fills the driver's memory.
    """

    def __init__(self, pipe, on_over) -> None:
        super().__init__(daemon=True)
        self.pipe = pipe
        self.on_over = on_over
        self.buf = bytearray()
        self.over = False

    def run(self) -> None:
        try:
            while True:
                chunk = self.pipe.read1(CLI_READ_CHUNK)
                if not chunk:
                    return
                room = CLI_STREAM_CAP - len(self.buf)
                if len(chunk) > room:
                    self.buf += chunk[: max(room, 0)]
                    if not self.over:
                        self.over = True
                        self.on_over()
                    continue  # keep draining (discarding) so the writer is not blocked
                self.buf += chunk
        except (OSError, ValueError):
            return


def _write_stdin(pipe, data: bytes | None) -> None:
    try:
        if data:
            pipe.write(data)
    except (OSError, ValueError):
        pass  # the command exited without reading all of its input
    finally:
        with contextlib.suppress(OSError, ValueError):
            pipe.close()


def run_cli_case(
    case: dict, index: int, side: dict, run_dir: str, workdir: str, start: list, timeout: float
) -> dict:
    """Run ``start + args`` once for one cli case and record exit code + streams.

    The command runs in its own session (process group). It is killed as a group when
    it exceeds the case timeout or writes past CLI_STREAM_CAP on either stream
    (``truncated: true``). Draining the pipes is bounded by the same deadline, so a
    background grandchild that escaped the group (setsid) and still holds a pipe
    cannot hang the driver: the case records an error instead.
    """
    case_dir = os.path.join(run_dir, "cli_cases", str(index))
    shutil.rmtree(case_dir, ignore_errors=True)
    os.makedirs(case_dir)
    values = {"case_dir": case_dir, "lib_harness": os.path.join(run_dir, LIB_HARNESS_NAME)}
    args = list(case.get("args") or [])
    argv = _subst(list(start), values) + _subst(args, values)
    env = dict(os.environ)
    env.update({k: str(v) for k, v in (side.get("env") or {}).items()})
    env.update(case.get("env") or {})
    stdin = case.get("stdin")
    data = None
    if stdin is not None:
        data = (stdin if isinstance(stdin, str) else json.dumps(stdin)).encode()
    limit = min(float(case.get("timeout") or timeout), float(CLI_MAX_TIMEOUT))
    out: dict = {"name": case["name"], "kind": "cli", "args": args}
    if case.get("compare_stderr"):
        out["compare_stderr"] = True
    started = time.monotonic()
    deadline = started + limit
    try:
        proc = subprocess.Popen(  # nosec B603 - argv from the run's own config
            argv,
            cwd=workdir,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as e:
        out["error"] = f"command failed to run: {type(e).__name__}: {e}"
        out["duration_ms"] = round((time.monotonic() - started) * 1000, 1)
        return out

    over = threading.Event()
    readers = [_CappedReader(proc.stdout, over.set), _CappedReader(proc.stderr, over.set)]
    writer = threading.Thread(target=_write_stdin, args=(proc.stdin, data), daemon=True)
    for t in (*readers, writer):
        t.start()
    timed_out = False
    while _exit_status(proc) is None:  # never reap before _kill_group signalled the group
        if over.is_set():
            break
        if time.monotonic() >= deadline:
            timed_out = True
            break
        _wait_exit(proc, 0.05)
    # Always, also after a normal exit: same-group background children die with the case.
    _kill_group(proc, grace=2.0)
    for t in readers:
        t.join(timeout=max(deadline - time.monotonic(), 0.5))
    held_open = any(t.is_alive() for t in readers)
    for reader in readers:
        if not reader.is_alive():  # a live reader holds the buffer lock: leave its pipe be
            with contextlib.suppress(OSError):
                reader.pipe.close()
    out["duration_ms"] = round((time.monotonic() - started) * 1000, 1)

    out["exit_code"] = proc.returncode
    if over.is_set():
        out["truncated"] = True
        out["exit_code"] = None
        out["error"] = f"output exceeded {CLI_STREAM_CAP} bytes; the command was killed"
    elif timed_out:
        out["exit_code"] = None
        out["error"] = f"timed out after {limit:g}s"
    elif held_open:
        out["error"] = (
            "stdout/stderr still open at the timeout after the command exited "
            "(a background process holds the pipe)"
        )
    for reader, field in zip(readers, ("stdout", "stderr")):
        if reader.over:
            out[f"{field}_clipped"] = True
    text = bytes(readers[0].buf).decode("utf-8", "replace")
    if case.get("stdout_format") == "json":
        try:
            out["stdout_json"] = json.loads(text)
        except ValueError as e:
            out["stdout"] = text
            out["stdout_json_error"] = f"stdout is not JSON: {e}"
    else:
        out["stdout"] = text
    out["stderr"] = bytes(readers[1].buf).decode("utf-8", "replace")
    return out


def run_cli_side(side: dict, config: dict, cases: list[dict], run_dir: str) -> dict:
    role, key, workdir, start = _side_basics(side)
    result: dict = {"role": role, "key": key, "kind": "cli", "ok": False, "error": None}
    result["cases"] = []
    timeout = float(config.get("request_timeout") or 30)
    for i, case in enumerate(cases):
        result["cases"].append(run_cli_case(case, i, side, run_dir, workdir, start, timeout))
    result["ok"] = True
    return result


@contextlib.contextmanager
def running_service(side: dict, config: dict, run_dir: str, nonce: str, result: dict):
    """Fresh DB + migrate + stubs + the started, healthy service; yields (proc, port, stubs).

    ``result`` (the side's result dict) receives ``database``, ``migrate_exit_code``
    and, on exit, ``log_tail``. The service's process group is killed on exit.
    """
    role, key, workdir, start = _side_basics(side)
    db_name = f"parity_{nonce[:12]}_{key}"
    result["database"] = db_name
    database_url = create_database(
        list(config.get("ws_pg") or ["ws-pg"]), db_name, os.path.join(run_dir, f"{key}.ws-pg.log")
    )

    env = dict(os.environ)
    env.update({k: str(v) for k, v in (side.get("env") or {}).items()})
    env["DATABASE_URL"] = database_url

    migrate = side.get("migrate")
    if migrate:
        log_path = os.path.join(run_dir, f"{key}.migrate.log")
        with open(log_path, "wb") as log:
            try:
                done = subprocess.run(  # nosec B603 - argv from the run's own config
                    _subst(list(migrate), {"database_url": database_url}),
                    cwd=workdir,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=int(side.get("migrate_timeout") or 600),
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as e:
                raise DriverError(f"{role}: migrate command failed to run: {e}") from e
        result["migrate_exit_code"] = done.returncode
        if done.returncode != 0:
            raise DriverError(f"{role}: migrate exited {done.returncode}: {_tail(log_path, 1500)}")

    stubs = {"default": UpstreamStub().start()}
    for uname in side.get("upstreams") or {}:
        stubs[uname] = UpstreamStub(uname).start()
    stub = stubs["default"]
    port = free_port()
    bind = f"127.0.0.1:{port}"
    for name in side.get("upstream_env") or []:
        env[name] = stub.url
    for uname, names in (side.get("upstreams") or {}).items():
        for name in names:
            env[name] = stubs[uname].url
    env.update({"PORT": str(port), "HOST": "127.0.0.1", "BIND_ADDR": bind})
    values = {
        "port": str(port),
        "bind_addr": bind,
        "database_url": database_url,
        "upstream_url": stub.url,
    }
    for uname, s in stubs.items():
        values[f"upstream_url:{uname}"] = s.url
    argv = _subst(start, values)
    log_path = os.path.join(run_dir, f"{key}.service.log")
    log = open(log_path, "wb")  # noqa: SIM115 - closed in finally after the process ends
    proc = None
    try:
        try:
            proc = subprocess.Popen(  # nosec B603 - argv from the run's own config
                argv,
                cwd=workdir,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as e:
            raise DriverError(f"{role}: start command failed to run: {e}") from e
        try:
            wait_healthy(
                proc,
                port,
                side.get("health_path") or "/health",
                float(side.get("startup_timeout") or 120),
            )
        except DriverError as e:
            log.flush()
            raise DriverError(f"{role}: {e}: {_tail(log_path, 1500)}") from e
        yield proc, port, stubs
    finally:
        if proc is not None:
            _kill_group(proc)
        log.close()
        for s in stubs.values():
            s.stop()
        result["log_tail"] = _tail(log_path)


def run_side(side: dict, config: dict, cases: list[dict], run_dir: str, nonce: str) -> dict:
    role = side.get("role") or "side"
    key = side.get("key") or role
    cli_side = side.get("kind") == "cli"
    cli_cases = bool(cases) and (cases[0].get("kind") or "http") == "cli"
    if cli_side != cli_cases:
        raise DriverError(
            f"{key}: a cli side runs only cli cases"
            if cli_side
            else f"{key}: cli cases need a side with kind 'cli'"
        )
    if cli_side:
        return run_cli_side(side, config, cases, run_dir)
    result: dict = {"role": role, "key": key, "ok": False, "error": None, "cases": []}
    with running_service(side, config, run_dir, nonce, result) as (proc, port, stubs):
        timeout = float(config.get("request_timeout") or 30)
        for case in cases:
            result["cases"].append(
                replay_case(
                    case,
                    port,
                    stubs,
                    config.get("default_headers") or {},
                    list(config.get("response_headers") or []),
                    timeout,
                    route_map=side.get("route_map") or [],
                    side_headers=side.get("default_headers") or {},
                    request_headers=side.get("request_headers") or {},
                )
            )
            code = _exit_status(proc)
            if code is not None:
                # Keep going: the remaining cases record connection errors, which the
                # diff reports per case; the exit is noted for the reader.
                result["service_exited"] = code
        if "service_exited" not in result and any("error" in c for c in result["cases"]):
            # A service that dies mid-request closes its socket before the kernel
            # reaps it, so poll() right after the last case can still read "alive".
            # Only when a case already failed, give the exit a moment to land.
            code = _wait_exit(proc, 1.0)
            if code is not None:
                result["service_exited"] = code
        result["ok"] = True
    return result


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: parity_driver.py <run_dir>", file=sys.stderr)
        return 2
    run_dir = args[0]
    out: dict = {"driver_version": DRIVER_VERSION, "ok": False, "error": None, "sides": {}}
    started = time.monotonic()
    try:
        with open(os.path.join(run_dir, "driver_config.json"), encoding="utf-8") as fh:
            config = json.load(fh)
        sides = config.get("sides") or []
        per_side = bool(sides) and all("cases" in s or "cases_file" in s for s in sides)
        cases: list[dict] = []
        if not per_side:
            cases = load_cases(config)
            out["cases_total"] = len(cases)
            out["cases"] = cases  # the exact input replayed (expected_* baseline is read from it)
        else:
            out["case_sets"] = {}
        if not sides:
            raise DriverError("driver_config has no sides")
        nonce = os.path.basename(os.path.normpath(run_dir)) or "run"
        nonce = re.sub(r"[^a-z0-9]", "", nonce.lower()) or "run"
        keys = [s.get("key") or s.get("role") or "side" for s in sides]
        if len(set(keys)) != len(keys):
            raise DriverError(f"side keys must be unique: {keys}")
        for side, key in zip(sides, keys):
            side_cases = cases
            if "cases" in side or "cases_file" in side:
                try:
                    side_cases = load_cases(side)
                except DriverError as e:
                    raise DriverError(f"{key}: {e}") from e
                out["case_sets"] = out.get("case_sets") or {}
                out["case_sets"][key] = side_cases
            try:
                out["sides"][key] = run_side(side, config, side_cases, run_dir, nonce)
            except DriverError as e:
                out["sides"][key] = {
                    "role": side.get("role"),
                    "key": key,
                    "ok": False,
                    "error": str(e),
                    "cases": [],
                }
                raise
        if per_side:
            out["cases_total"] = sum(len(v) for v in out["case_sets"].values())
        out["ok"] = True
    except DriverError as e:
        out["error"] = str(e)
    except Exception as e:  # noqa: BLE001 - every failure must still produce result.json
        out["error"] = f"driver crashed: {type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()[-LOG_TAIL_CHARS:]
    out["duration"] = round(time.monotonic() - started, 3)
    tmp = os.path.join(run_dir, "result.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    os.replace(tmp, os.path.join(run_dir, "result.json"))
    if not out["ok"]:
        print(f"parity driver: {out['error']}", file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "cases_total": out.get("cases_total")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
