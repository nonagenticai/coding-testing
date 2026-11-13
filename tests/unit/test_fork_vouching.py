"""The fork must honour on-behalf-of, not silently ignore it.

This service keeps its own AuthMiddleware rather than consuming the SDK's, so it
does not inherit the vouching arm. Without it the tenant came from the
introspection body ONLY — which for a caller acting on behalf of a customer is
not "safer", it is WRONG: coding-ai's calls silently resolved to svc-coding-ai's
own tenant, filing reads and writes under the wrong one with a 200.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from coding_client.auth import _default_tenant as dt_mod
from coding_client.auth import _trusted_networks as tn_mod
from coding_client.auth.tenant import clear_license_caches
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

SERVICE_SUBJECT = "019ff970-e327-74cf-aa8c-e0ce39997d44"  # svc-coding-ai
OTHER_SUBJECT = "019ff971-7cbf-7505-8bad-1f86ab493f6e"  # svc-coding-security
SERVICE_TENANT = "service-own-tenant-0001"
CUSTOMER_TENANT = "customer-tenant-0042"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, trusted_networks):
    # Empty the fork's OWN trusted-network list, using the conftest fixture that
    # exists for it. Without this the TestClient's 127.0.0.1 source is admitted
    # by the CIDR branch and every assertion below passes for the wrong reason —
    # including the forged-header one, which would report 200 and look like a
    # hole rather than a broken test.
    trusted_networks("")
    # ENABLE_AUTH is read ONCE at import time and, when false, short-circuits the
    # whole middleware into an anonymous DEFAULT_TENANT_ID user. The test env
    # sets it false, so without this every assertion here would see 200 and pass
    # or fail for reasons unrelated to vouching.
    import middleware.auth as _auth_mod

    monkeypatch.setattr(_auth_mod, "ENABLE_AUTH", True)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("TRUST_TENANT_HEADER", "false")
    monkeypatch.setenv("LICENSE_INTROSPECT_URL", "http://test-licenses/license/introspect")
    monkeypatch.delenv("CODING_ONBEHALF_SUBJECTS", raising=False)
    monkeypatch.delenv("SKIP_AUTH", raising=False)
    monkeypatch.setattr(tn_mod, "TRUSTED_NETWORKS", [])
    monkeypatch.setattr(dt_mod, "DEFAULT_TENANT_ID", None)
    clear_license_caches()
    yield
    clear_license_caches()


def _introspection(payload: dict | None):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    client = AsyncMock()
    client.post = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return patch("coding_client.auth.tenant.httpx.AsyncClient", return_value=client)


def _client() -> TestClient:
    # Imported INSIDE the helper, not at module scope. middleware.auth parses
    # TRUSTED_CIDRS ONCE at import time (deliberately, so a typo crashes the pod
    # rather than silently widening trust). A module-level import here runs at
    # COLLECTION — before conftest sets that env — so the module would cache an
    # empty network list for the whole session and every later test that relies
    # on the trusted-network branch would 401. Measured: it broke 47 of them,
    # all passing individually, which is exactly what an import-order bug looks
    # like.
    from middleware.auth import AuthMiddleware

    app = FastAPI()

    @app.get("/api/testing/whoami")
    async def whoami(request: Request):
        return {"tenant_id": getattr(request.state, "tenant_id", None)}

    app.add_middleware(AuthMiddleware)
    return TestClient(app, raise_server_exceptions=False)


def _svc_licence(subject: str = SERVICE_SUBJECT) -> dict:
    return {"active": True, "tenant_id": SERVICE_TENANT, "subject_id": subject}


class TestForkHonoursOnBehalfOf:
    def test_an_allowlisted_caller_is_scoped_to_the_customer(self, monkeypatch):
        """The defect this closes: it used to resolve to the SERVICE's tenant."""
        monkeypatch.setenv("CODING_ONBEHALF_SUBJECTS", SERVICE_SUBJECT)
        with _introspection(_svc_licence()):
            r = _client().get(
                "/api/testing/whoami",
                headers={
                    "Authorization": "Bearer svc-token",
                    "X-Tenant-Id": CUSTOMER_TENANT,
                },
            )
        assert r.status_code == 200
        assert r.json()["tenant_id"] == CUSTOMER_TENANT
        assert SERVICE_TENANT not in r.text

    def test_a_non_allowlisted_caller_is_refused_not_silently_ignored(self, monkeypatch):
        """403, not a 200 scoped to the caller's own tenant.

        Silently ignoring the assertion is what made this a data-scoping bug
        rather than a rejection.
        """
        monkeypatch.setenv("CODING_ONBEHALF_SUBJECTS", OTHER_SUBJECT)
        with _introspection(_svc_licence()):
            r = _client().get(
                "/api/testing/whoami",
                headers={
                    "Authorization": "Bearer svc-token",
                    "X-Tenant-Id": CUSTOMER_TENANT,
                },
            )
        assert r.status_code == 403
        assert r.json()["error"] == "TENANT_ASSERTION_NOT_PERMITTED"
        assert CUSTOMER_TENANT not in r.text

    def test_an_empty_allowlist_refuses(self):
        with _introspection(_svc_licence()):
            r = _client().get(
                "/api/testing/whoami",
                headers={
                    "Authorization": "Bearer svc-token",
                    "X-Tenant-Id": CUSTOMER_TENANT,
                },
            )
        assert r.status_code == 403

    def test_an_agreeing_header_is_not_an_assertion(self):
        """The gateway re-injects the tenant it resolved from this same token."""
        with _introspection(_svc_licence()):
            r = _client().get(
                "/api/testing/whoami",
                headers={
                    "Authorization": "Bearer svc-token",
                    "X-Tenant-Id": SERVICE_TENANT,
                },
            )
        assert r.status_code == 200
        assert r.json()["tenant_id"] == SERVICE_TENANT

    def test_no_assertion_is_unchanged_behaviour(self):
        with _introspection(_svc_licence()):
            r = _client().get("/api/testing/whoami", headers={"Authorization": "Bearer svc-token"})
        assert r.status_code == 200
        assert r.json()["tenant_id"] == SERVICE_TENANT

    def test_a_forged_bare_header_is_still_401(self):
        """Non-regression: the flip's own guarantee must survive this change."""
        r = _client().get(
            "/api/testing/whoami",
            headers={"X-Tenant-Id": CUSTOMER_TENANT, "X-User-Roles": "admin"},
        )
        assert r.status_code == 401
        assert CUSTOMER_TENANT not in r.text
