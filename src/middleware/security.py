"""Security middleware for coding-testing.

Provides HTTP-layer security: method validation, request size limits,
and security response headers.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

ALLOWED_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"})
MAX_BODY_SIZE = 10 * 1024 * 1024  # 10 MB
PERMISSIVE_PREFIXES = ("/api/tests/execute", "/api/tests/ai")
PERMISSIVE_MAX_SIZE = 50 * 1024 * 1024  # 50 MB

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Referrer-Policy": "strict-origin-when-cross-origin",
}


class SecurityMiddleware(BaseHTTPMiddleware):
    """HTTP-layer security validation middleware."""

    def __init__(self, app, **kwargs):
        super().__init__(app, **kwargs)
        self._doc_access: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path

        # Skip security checks for documentation endpoints (with rate limiting)
        if path in ("/docs", "/redoc", "/openapi.json"):
            if not self._check_doc_rate_limit(request):
                return JSONResponse(
                    status_code=429, content={"error": "Rate limit exceeded for docs"}
                )
            response = await call_next(request)
            self._add_security_headers(response)
            return response

        # Validate HTTP method
        if request.method not in ALLOWED_METHODS:
            return JSONResponse(
                status_code=405,
                content={"error": "Method not allowed", "error_code": "SECURITY_ERROR"},
            )

        # Validate request size
        content_length = request.headers.get("content-length")
        if content_length:
            max_size = (
                PERMISSIVE_MAX_SIZE
                if any(path.startswith(p) for p in PERMISSIVE_PREFIXES)
                else MAX_BODY_SIZE
            )
            try:
                if int(content_length) > max_size:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": "Request entity too large",
                            "error_code": "SECURITY_ERROR",
                        },
                    )
            except ValueError:
                pass

        # Process request
        response = await call_next(request)

        # Add security headers
        self._add_security_headers(response)

        return response

    @staticmethod
    def _add_security_headers(response: Response) -> None:
        for header, value in SECURITY_HEADERS.items():
            response.headers[header] = value

    def _check_doc_rate_limit(self, request: Request) -> bool:
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = 60.0
        max_requests = 30

        self._doc_access[client_ip] = [t for t in self._doc_access[client_ip] if now - t < window]

        if len(self._doc_access[client_ip]) >= max_requests:
            return False

        self._doc_access[client_ip].append(now)
        return True
