"""A stdlib service for the perf-parity tests: routes under $PREFIX, optional $DELAY_MS
per request, $FAIL_GET=1 answers GET items with 500. POST items calls the upstream."""

import http.client
import json
import os
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PREFIX = os.environ.get("PREFIX", "")
DELAY = float(os.environ.get("DELAY_MS", "0")) / 1000


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True

    def log_message(self, *a):
        pass

    def _send(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True})
        time.sleep(DELAY)
        if self.path == PREFIX + "/items":
            return self._send(500 if os.environ.get("FAIL_GET") else 200, [1, 2])
        self._send(404, {})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        time.sleep(DELAY)
        if self.path != PREFIX + "/items":
            return self._send(404, {})
        up = urllib.parse.urlparse(os.environ["UPSTREAM_URL"])
        c = http.client.HTTPConnection(up.hostname, up.port, timeout=5)
        c.request("POST", "/enrich", body=body, headers={"Content-Type": "application/json"})
        r = c.getresponse()
        self._send(201, json.loads(r.read() or b"null"))


ThreadingHTTPServer(("127.0.0.1", int(os.environ["PORT"])), H).serve_forever()
