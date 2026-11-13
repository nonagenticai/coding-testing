"""A tiny stdlib HTTP + WebSocket + SSE service for the stream-replay tests.

Its server-side RFC 6455 codec is written independently of the driver's client,
so a record/replay round trip exercises two implementations against each other.

Routes:
* ``GET /health``, ``GET /items`` (plain HTTP, for mixed corpora);
* ``GET /ws`` (websocket): sends ``{"type": "hello", "ts": <now>}`` right after the
  upgrade (before the client says anything), then per text message
  ``{"op": "echo", "v": x}`` -> a ping, then ``{"type": "echo", "v": x}``;
  ``{"op": "burst", "n": k}`` -> k frames ``{"type": "burst", "i": i}``;
  ``{"op": "bye"}`` -> close 4000 "bye". ``X-Reject: 1`` -> 403 JSON, no upgrade.
* ``GET /events`` (SSE): three ``tick`` events with ids, then end of stream.

Environment knobs (the PLANTED mutations of the tests):
``MUTATE=echo`` (echo of v=2 answers 3), ``MUTATE=sse`` (2nd event n=20),
``SHUFFLE=1`` (burst frames in reverse order), ``HB=1`` (a heartbeat frame
before every reply), ``PORT``.
"""

import base64
import hashlib
import json
import os
import struct
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
MUTATE = os.environ.get("MUTATE", "")


def frame(opcode, payload):
    n = len(payload)
    if n < 126:
        head = struct.pack("!BB", 0x80 | opcode, n)
    elif n < 65536:
        head = struct.pack("!BBH", 0x80 | opcode, 126, n)
    else:
        head = struct.pack("!BBQ", 0x80 | opcode, 127, n)
    return head + payload


def read_frame(rfile):
    b0, b1 = rfile.read(2)
    n = b1 & 0x7F
    if n == 126:
        n = struct.unpack("!H", rfile.read(2))[0]
    elif n == 127:
        n = struct.unpack("!Q", rfile.read(8))[0]
    key = rfile.read(4) if b1 & 0x80 else b""
    data = bytearray(rfile.read(n))
    for i in range(len(data)):
        data[i] ^= key[i % 4] if key else 0
    return b0 & 0x0F, bytes(data)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"status": "ok"})
        if self.path == "/items":
            return self._json(200, {"items": [1, 2]})
        if self.path == "/events":
            return self._sse()
        if self.path.startswith("/ws"):
            return self._ws()
        return self._json(404, {"detail": "Not Found"})

    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(b": comment line, ignored\n\n")
        for n in (1, 2, 3):
            v = 20 if (MUTATE == "sse" and n == 2) else n
            data = json.dumps({"n": v, "ts": time.time()})
            self.wfile.write(f"event: tick\nid: {n}\ndata: {data}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.02)
        self.close_connection = True

    def _send(self, obj):
        self.wfile.write(frame(0x1, json.dumps(obj).encode()))
        self.wfile.flush()

    def _ws(self):
        if self.headers.get("X-Reject"):
            return self._json(403, {"detail": "forbidden"})
        key = self.headers.get("Sec-WebSocket-Key", "")
        accept = base64.b64encode(
            hashlib.sha1((key + GUID).encode(), usedforsecurity=False).digest()
        ).decode()
        self.send_response(101)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        offered = self.headers.get("Sec-WebSocket-Protocol")
        if offered:
            self.send_header("Sec-WebSocket-Protocol", offered.split(",")[0].strip())
        self.end_headers()
        self.close_connection = True
        self._send({"type": "hello", "ts": time.time()})
        while True:
            try:
                op, data = read_frame(self.rfile)
            except (ValueError, OSError):
                return
            if op == 0x8:  # close: echo it and stop
                self.wfile.write(frame(0x8, data[:2]))
                return
            if op in (0x9, 0xA):
                continue
            msg = json.loads(data)
            if os.environ.get("HB"):
                self._send({"type": "hb"})
            if msg.get("op") == "echo":
                self.wfile.write(frame(0x9, b"p"))  # a ping the client must answer, not record
                v = msg["v"] + 1 if (MUTATE == "echo" and msg["v"] == 2) else msg["v"]
                self._send({"type": "echo", "v": v})
            elif msg.get("op") == "burst":
                order = range(msg["n"])
                if os.environ.get("SHUFFLE"):
                    order = reversed(list(order))
                for i in order:
                    self._send({"type": "burst", "i": i})
            elif msg.get("op") == "bye":
                self.wfile.write(frame(0x8, struct.pack("!H", 4000) + b"bye"))
                try:
                    read_frame(self.rfile)  # the client's close reply
                except (ValueError, OSError):
                    pass
                return


ThreadingHTTPServer(("127.0.0.1", int(os.environ["PORT"])), H).serve_forever()
