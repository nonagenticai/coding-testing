"""Tests for middleware package: auth, security, exception_handlers, validation."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException
from starlette.responses import JSONResponse

# ── Package import test ───────────────────────────────────────────────


def test_middleware_package_importable():
    """Middleware package exports AuthMiddleware and register_exception_handlers."""
    from middleware import (
        AuthMiddleware,
        register_exception_handlers,
    )

    assert AuthMiddleware.__name__ == "AuthMiddleware"
    assert callable(register_exception_handlers)


async def test_security_headers_on_health(client):
    """Health endpoint returns 200 with expected JSON body."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


# ── Auth middleware tests ─────────────────────────────────────────────


class _CaseInsensitiveDict(dict):
    """Dict that does case-insensitive key lookups for .get()."""

    def get(self, key, default=None):
        key_lower = key.lower()
        for k, v in self.items():
            if k.lower() == key_lower:
                return v
        return default


def _make_auth_request(
    path="/api/test",
    method="GET",
    headers=None,
    client_host="127.0.0.1",
):
    """Helper to create a mock request for AuthMiddleware."""
    request = MagicMock()
    request.url = MagicMock()
    request.url.path = path
    request.method = method
    request.headers = _CaseInsensitiveDict(headers or {})
    request.client = MagicMock()
    request.client.host = client_host
    request.state = MagicMock()
    return request


async def test_auth_middleware_public_path_skip():
    """Auth middleware skips authentication for public paths and calls next handler."""
    from middleware.auth import AuthMiddleware

    middleware = AuthMiddleware(app=MagicMock())
    request = _make_auth_request(path="/health")
    response = MagicMock()
    call_next = AsyncMock(return_value=response)

    result = await middleware.dispatch(request, call_next)
    assert result is response
    call_next.assert_awaited_once_with(request)


async def test_auth_middleware_public_path_docs():
    """Auth middleware skips auth for /docs endpoint and calls next handler."""
    from middleware.auth import AuthMiddleware

    middleware = AuthMiddleware(app=MagicMock())
    request = _make_auth_request(path="/docs")
    response = MagicMock()
    call_next = AsyncMock(return_value=response)

    result = await middleware.dispatch(request, call_next)
    assert result is response
    call_next.assert_awaited_once_with(request)


async def test_auth_middleware_enable_auth_false(monkeypatch):
    """Auth middleware creates anonymous user when ENABLE_AUTH=false."""
    # We need to patch the module-level variable
    import middleware.auth as auth_mod

    original = auth_mod.ENABLE_AUTH
    auth_mod.ENABLE_AUTH = False
    try:
        middleware = auth_mod.AuthMiddleware(app=MagicMock())
        request = _make_auth_request(path="/api/data")
        response = MagicMock()
        call_next = AsyncMock(return_value=response)

        result = await middleware.dispatch(request, call_next)
        assert result is response
        assert request.state.user.sub == "anonymous"
        assert request.state.user.preferred_username == "anonymous"
        call_next.assert_awaited_once_with(request)
    finally:
        auth_mod.ENABLE_AUTH = original


async def test_auth_middleware_x_user_id_header():
    """Auth middleware creates user from X-User-Id header with correct roles and tenant."""
    import middleware.auth as auth_mod

    original = auth_mod.ENABLE_AUTH
    auth_mod.ENABLE_AUTH = True
    try:
        middleware = auth_mod.AuthMiddleware(app=MagicMock())
        request = _make_auth_request(
            path="/api/data",
            headers={
                "x-user-id": "user-42",
                "x-user-name": "Bob",
                "x-user-email": "bob@test.com",
                "x-user-roles": "admin, editor",
                "x-tenant-id": "tenant-99",
            },
            client_host="8.8.8.8",
        )
        response = MagicMock()
        call_next = AsyncMock(return_value=response)

        result = await middleware.dispatch(request, call_next)
        assert result is response
        assert request.state.user.sub == "user-42"
        assert request.state.user.preferred_username == "Bob"
        assert request.state.user.email == "bob@test.com"
        assert request.state.user.realm_access["roles"] == ["admin", "editor"]
        assert request.state.tenant_id == "tenant-99"
        call_next.assert_awaited_once_with(request)
    finally:
        auth_mod.ENABLE_AUTH = original


async def test_auth_middleware_forged_headers_rejected_when_gate_off(monkeypatch):
    """With TRUST_TENANT_HEADER=false the identity headers carry NO identity.

    This is the security contract: a forged x-tenant-id/x-user-roles from an
    untrusted pod IP no longer mints a User — the request falls through to the
    401. (Reproduced live 2026-08-03: a sibling coding-* pod read a fabricated
    tenant off coding-analysis via exactly these headers.) The previous test is
    the false-positive control — the SAME request is still honoured with the gate
    at its default (True), so this 401 is the gate closing, not a broken path.
    """
    import middleware.auth as auth_mod

    monkeypatch.setenv("TRUST_TENANT_HEADER", "false")
    original = auth_mod.ENABLE_AUTH
    auth_mod.ENABLE_AUTH = True
    try:
        middleware = auth_mod.AuthMiddleware(app=MagicMock())
        request = _make_auth_request(
            path="/api/data",
            headers={
                "x-user-id": "attacker",
                "x-tenant-id": "victim-tenant",
                "x-user-roles": "admin,platform-admin",
            },
            client_host="10.42.7.9",  # a pod IP — the real attacker position
        )
        call_next = AsyncMock()

        result = await middleware.dispatch(request, call_next)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 401
        call_next.assert_not_awaited()
    finally:
        auth_mod.ENABLE_AUTH = original


async def test_auth_middleware_x_user_id_no_roles_trusted(trusted_networks):
    """Auth middleware defaults roles to ['user'] for a CONFIGURED trusted network."""
    import middleware.auth as auth_mod

    original = auth_mod.ENABLE_AUTH
    auth_mod.ENABLE_AUTH = True
    trusted_networks("192.168.0.0/16")
    try:
        middleware = auth_mod.AuthMiddleware(app=MagicMock())
        request = _make_auth_request(
            path="/api/data",
            headers={"x-user-id": "user-1"},
            client_host="192.168.1.10",
        )
        response = MagicMock()
        call_next = AsyncMock(return_value=response)

        result = await middleware.dispatch(request, call_next)
        assert result is response
        assert request.state.user.realm_access["roles"] == ["user"]
        assert request.state.user.sub == "user-1"
        # preferred_username defaults to user_id when x-user-name absent
        assert request.state.user.preferred_username == "user-1"
    finally:
        auth_mod.ENABLE_AUTH = original


async def test_auth_middleware_x_user_id_no_roles_untrusted_gets_no_roles(trusted_networks):
    """A forwarded user from an UNTRUSTED IP is not silently granted the 'user' role.

    Negative control for the role half of the TRUSTED_CIDRS knob: with the
    default (empty) trust list, the source IP must not upgrade an empty
    x-user-roles header into ["user"].
    """
    import middleware.auth as auth_mod

    original = auth_mod.ENABLE_AUTH
    auth_mod.ENABLE_AUTH = True
    trusted_networks("")  # default: trust nothing
    try:
        middleware = auth_mod.AuthMiddleware(app=MagicMock())
        request = _make_auth_request(
            path="/api/data",
            headers={"x-user-id": "user-1"},
            client_host="10.42.7.9",  # k3s pod CIDR — RFC1918, but NOT configured
        )
        response = MagicMock()
        call_next = AsyncMock(return_value=response)

        result = await middleware.dispatch(request, call_next)
        assert result is response
        assert request.state.user.realm_access["roles"] == []
    finally:
        auth_mod.ENABLE_AUTH = original


async def test_auth_middleware_trusted_cidr_bypass_when_configured(trusted_networks):
    """Auth middleware creates the internal user only for a CONFIGURED trusted CIDR."""
    import middleware.auth as auth_mod

    original = auth_mod.ENABLE_AUTH
    auth_mod.ENABLE_AUTH = True
    trusted_networks("10.0.0.0/8")
    try:
        middleware = auth_mod.AuthMiddleware(app=MagicMock())
        request = _make_auth_request(
            path="/api/data",
            headers={},
            client_host="10.0.0.5",
        )
        response = MagicMock()
        call_next = AsyncMock(return_value=response)

        result = await middleware.dispatch(request, call_next)
        assert result is response
        assert request.state.user.sub == "internal"
        assert request.state.user.preferred_username == "internal"
        assert request.state.user.realm_access["roles"] == ["user"]
        call_next.assert_awaited_once_with(request)
    finally:
        auth_mod.ENABLE_AUTH = original


# ── NEGATIVE CONTROLS: the pod-CIDR anonymous-read defect ─────────────
#
# Before the TRUSTED_CIDRS fix, `_is_rfc1918` hardcoded "any private or
# loopback address is trusted". 10.42.0.0/16 (the k3s pod CIDR) ⊂ 10.0.0.0/8,
# so ANY pod could read tenant data with no credential. These tests assert the
# closed default, so they fail if the hardcoded bypass ever comes back.


async def test_auth_middleware_pod_cidr_no_credential_is_rejected(trusted_networks):
    """An uncredentialed request from the k3s pod CIDR is REJECTED by default."""
    import middleware.auth as auth_mod

    original_auth = auth_mod.ENABLE_AUTH
    original_env = auth_mod.ENVIRONMENT
    original_key = auth_mod.API_KEY
    auth_mod.ENABLE_AUTH = True
    auth_mod.ENVIRONMENT = "production"
    auth_mod.API_KEY = ""
    trusted_networks("")  # absent TRUSTED_CIDRS == this
    try:
        middleware = auth_mod.AuthMiddleware(app=MagicMock())
        request = _make_auth_request(
            path="/api/tests/orchestration/suites",
            headers={},
            client_host="10.42.3.14",  # a k3s pod
        )
        call_next = AsyncMock(return_value=MagicMock())

        result = await middleware.dispatch(request, call_next)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 401
        assert json.loads(result.body)["error"]["error_code"] == "UNAUTHORIZED"
        call_next.assert_not_awaited()
    finally:
        auth_mod.ENABLE_AUTH = original_auth
        auth_mod.ENVIRONMENT = original_env
        auth_mod.API_KEY = original_key


async def test_auth_middleware_loopback_no_credential_is_rejected(trusted_networks):
    """Loopback is not implicitly trusted either — in-pod localhost needs a credential."""
    import middleware.auth as auth_mod

    original_auth = auth_mod.ENABLE_AUTH
    original_env = auth_mod.ENVIRONMENT
    original_key = auth_mod.API_KEY
    auth_mod.ENABLE_AUTH = True
    auth_mod.ENVIRONMENT = "production"
    auth_mod.API_KEY = ""
    trusted_networks("")
    try:
        middleware = auth_mod.AuthMiddleware(app=MagicMock())
        request = _make_auth_request(path="/api/tests/active", headers={}, client_host="127.0.0.1")
        call_next = AsyncMock(return_value=MagicMock())

        result = await middleware.dispatch(request, call_next)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 401
        call_next.assert_not_awaited()
    finally:
        auth_mod.ENABLE_AUTH = original_auth
        auth_mod.ENVIRONMENT = original_env
        auth_mod.API_KEY = original_key


async def test_auth_middleware_ip_outside_configured_cidr_is_rejected(trusted_networks):
    """A private IP outside the configured CIDR is rejected (not "private == trusted")."""
    import middleware.auth as auth_mod

    original_auth = auth_mod.ENABLE_AUTH
    original_env = auth_mod.ENVIRONMENT
    original_key = auth_mod.API_KEY
    auth_mod.ENABLE_AUTH = True
    auth_mod.ENVIRONMENT = "production"
    auth_mod.API_KEY = ""
    trusted_networks("192.168.5.0/24")
    try:
        middleware = auth_mod.AuthMiddleware(app=MagicMock())
        request = _make_auth_request(path="/api/data", headers={}, client_host="10.42.3.14")
        call_next = AsyncMock(return_value=MagicMock())

        result = await middleware.dispatch(request, call_next)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 401
        call_next.assert_not_awaited()
    finally:
        auth_mod.ENABLE_AUTH = original_auth
        auth_mod.ENVIRONMENT = original_env
        auth_mod.API_KEY = original_key


# ── Bare X-Tenant-Id parity with coding-client / coding-analysis ──────
#
# The forwarded-identity branch used to be gated on x-user-id alone, so a
# request carrying only X-Tenant-Id fell through to the trusted-network check
# and — once TRUSTED_CIDRS defaulted to empty — to a 401. That made this
# service strictly stricter than every sibling: coding-client's AuthMiddleware
# and coding-analysis's copy both branch on ``x_user_id or x_tenant_id``.
# Every in-mesh caller (coding-containers' clients/_tenant.py) sends exactly
# and only ``X-Tenant-Id``, so the divergence 401'd them.
#
# The pair below is deliberate: the first asserts the header is ACCEPTED, the
# second group asserts the acceptance did not widen into "accept anything".
# Without the negative half, "return the request unchanged" would pass.


async def test_auth_middleware_bare_tenant_header_is_accepted(trusted_networks):
    """A bare X-Tenant-Id authenticates, from an untrusted IP, with no trusted CIDRs.

    This is the exact shape coding-containers sends. It must pass auth on the
    header alone — never on the source IP — so it still works with the
    network-position bypass closed (TRUSTED_CIDRS empty).
    """
    import middleware.auth as auth_mod

    original_auth = auth_mod.ENABLE_AUTH
    original_env = auth_mod.ENVIRONMENT
    original_key = auth_mod.API_KEY
    auth_mod.ENABLE_AUTH = True
    auth_mod.ENVIRONMENT = "production"
    auth_mod.API_KEY = ""
    trusted_networks("")  # trust NO source IP — the deployed default
    try:
        middleware = auth_mod.AuthMiddleware(app=MagicMock())
        request = _make_auth_request(
            path="/api/tests/ai/metrics",
            headers={"x-tenant-id": "tenant-abc"},
            client_host="10.42.3.14",  # k3s pod CIDR, deliberately NOT trusted
        )
        response = MagicMock()
        call_next = AsyncMock(return_value=response)

        result = await middleware.dispatch(request, call_next)

        # Passed auth: the real handler ran, no JSONResponse short-circuit.
        assert not isinstance(result, JSONResponse), f"bare X-Tenant-Id was rejected: {result!r}"
        assert result is response
        call_next.assert_awaited_once_with(request)
        # The tenant comes from the header, not from DEFAULT_TENANT_ID.
        assert request.state.user.tenant_id == "tenant-abc"
        assert request.state.tenant_id == "tenant-abc"
        # No user id was supplied, and the untrusted IP grants no roles.
        assert request.state.user.sub == "anonymous"
        assert request.state.user.realm_access["roles"] == []
    finally:
        auth_mod.ENABLE_AUTH = original_auth
        auth_mod.ENVIRONMENT = original_env
        auth_mod.API_KEY = original_key


async def test_auth_middleware_tenant_header_does_not_reopen_anonymous_access(
    trusted_networks,
):
    """FALSE-POSITIVE CONTROL: no identity headers is still 401 from a pod IP.

    Accepting a bare X-Tenant-Id must not degrade into accepting requests with
    no tenant claim at all. This is the anonymous closure the TRUSTED_CIDRS
    change bought; it must survive.
    """
    import middleware.auth as auth_mod

    original_auth = auth_mod.ENABLE_AUTH
    original_env = auth_mod.ENVIRONMENT
    original_key = auth_mod.API_KEY
    auth_mod.ENABLE_AUTH = True
    auth_mod.ENVIRONMENT = "production"
    auth_mod.API_KEY = ""
    trusted_networks("")
    try:
        middleware = auth_mod.AuthMiddleware(app=MagicMock())
        request = _make_auth_request(
            path="/api/tests/ai/metrics",
            headers={},
            client_host="10.42.3.14",
        )
        call_next = AsyncMock(return_value=MagicMock())

        result = await middleware.dispatch(request, call_next)

        assert isinstance(result, JSONResponse)
        assert result.status_code == 401
        assert json.loads(result.body)["error"]["error_code"] == "UNAUTHORIZED"
        call_next.assert_not_awaited()
    finally:
        auth_mod.ENABLE_AUTH = original_auth
        auth_mod.ENVIRONMENT = original_env
        auth_mod.API_KEY = original_key


async def test_auth_middleware_non_identity_header_alone_is_still_401(trusted_networks):
    """FALSE-POSITIVE CONTROL: only x-user-id / x-tenant-id are identity claims.

    A request carrying some *other* forwarded header (here x-user-name) and no
    tenant or user id must still be rejected. Guards against a "fix" that
    keys the branch off header presence in general.
    """
    import middleware.auth as auth_mod

    original_auth = auth_mod.ENABLE_AUTH
    original_env = auth_mod.ENVIRONMENT
    original_key = auth_mod.API_KEY
    auth_mod.ENABLE_AUTH = True
    auth_mod.ENVIRONMENT = "production"
    auth_mod.API_KEY = ""
    trusted_networks("")
    try:
        middleware = auth_mod.AuthMiddleware(app=MagicMock())
        request = _make_auth_request(
            path="/api/tests/ai/metrics",
            headers={"x-user-name": "mallory", "x-user-roles": "admin"},
            client_host="10.42.3.14",
        )
        call_next = AsyncMock(return_value=MagicMock())

        result = await middleware.dispatch(request, call_next)

        assert isinstance(result, JSONResponse)
        assert result.status_code == 401
        call_next.assert_not_awaited()
    finally:
        auth_mod.ENABLE_AUTH = original_auth
        auth_mod.ENVIRONMENT = original_env
        auth_mod.API_KEY = original_key


async def test_auth_middleware_empty_tenant_header_is_still_401(trusted_networks):
    """FALSE-POSITIVE CONTROL: an empty X-Tenant-Id is not a tenant claim.

    ``X-Tenant-Id: ""`` must be treated as absent rather than as a tenant
    named "", which would otherwise mint a User scoped to nothing.
    """
    import middleware.auth as auth_mod

    original_auth = auth_mod.ENABLE_AUTH
    original_env = auth_mod.ENVIRONMENT
    original_key = auth_mod.API_KEY
    auth_mod.ENABLE_AUTH = True
    auth_mod.ENVIRONMENT = "production"
    auth_mod.API_KEY = ""
    trusted_networks("")
    try:
        middleware = auth_mod.AuthMiddleware(app=MagicMock())
        request = _make_auth_request(
            path="/api/tests/ai/metrics",
            headers={"x-tenant-id": ""},
            client_host="10.42.3.14",
        )
        call_next = AsyncMock(return_value=MagicMock())

        result = await middleware.dispatch(request, call_next)

        assert isinstance(result, JSONResponse)
        assert result.status_code == 401
        call_next.assert_not_awaited()
    finally:
        auth_mod.ENABLE_AUTH = original_auth
        auth_mod.ENVIRONMENT = original_env
        auth_mod.API_KEY = original_key


async def test_auth_middleware_tenant_header_is_scoped_to_the_header_value(
    trusted_networks,
):
    """A supplied tenant scopes the request to THAT tenant, never a default.

    Cross-tenant guard: the middleware must not quietly substitute
    DEFAULT_TENANT_ID (which is what the old RFC1918 bypass did — it minted an
    "internal" user on DEFAULT_TENANT_ID and ignored the header entirely).
    """
    import middleware.auth as auth_mod

    original_auth = auth_mod.ENABLE_AUTH
    auth_mod.ENABLE_AUTH = True
    trusted_networks("")
    try:
        middleware = auth_mod.AuthMiddleware(app=MagicMock())
        request = _make_auth_request(
            path="/api/tests/ai/metrics",
            headers={"x-tenant-id": "some-other-tenant"},
            client_host="10.42.3.14",
        )
        call_next = AsyncMock(return_value=MagicMock())

        await middleware.dispatch(request, call_next)

        assert request.state.tenant_id == "some-other-tenant"
        assert request.state.tenant_id != auth_mod.DEFAULT_TENANT_ID
        assert request.state.user.sub != "internal"
    finally:
        auth_mod.ENABLE_AUTH = original_auth


async def test_auth_middleware_missing_auth_401():
    """Auth middleware returns 401 JSON response for public IP without auth."""
    import middleware.auth as auth_mod

    original_auth = auth_mod.ENABLE_AUTH
    original_env = auth_mod.ENVIRONMENT
    original_key = auth_mod.API_KEY
    auth_mod.ENABLE_AUTH = True
    auth_mod.ENVIRONMENT = "production"
    auth_mod.API_KEY = ""
    try:
        middleware = auth_mod.AuthMiddleware(app=MagicMock())
        request = _make_auth_request(
            path="/api/data",
            headers={},
            client_host="8.8.8.8",
        )
        response = MagicMock()
        call_next = AsyncMock(return_value=response)

        result = await middleware.dispatch(request, call_next)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 401
        body = json.loads(result.body)
        assert body["error"]["message"] == "Authentication required"
        assert body["error"]["error_code"] == "UNAUTHORIZED"
        call_next.assert_not_awaited()
    finally:
        auth_mod.ENABLE_AUTH = original_auth
        auth_mod.ENVIRONMENT = original_env
        auth_mod.API_KEY = original_key


async def test_auth_middleware_api_key_success():
    """Auth middleware accepts valid API key in dev/test environment."""
    import middleware.auth as auth_mod

    original_auth = auth_mod.ENABLE_AUTH
    original_env = auth_mod.ENVIRONMENT
    original_key = auth_mod.API_KEY
    auth_mod.ENABLE_AUTH = True
    auth_mod.ENVIRONMENT = "test"
    auth_mod.API_KEY = "my-secret-key"
    try:
        middleware = auth_mod.AuthMiddleware(app=MagicMock())
        request = _make_auth_request(
            path="/api/data",
            headers={"authorization": "Bearer my-secret-key"},
            client_host="8.8.8.8",
        )
        response = MagicMock()
        call_next = AsyncMock(return_value=response)

        result = await middleware.dispatch(request, call_next)
        assert result is response
        assert request.state.user.sub == "api-key-user"
        assert request.state.user.preferred_username == "api-key-user"
        assert request.state.user.realm_access["roles"] == ["admin"]
        call_next.assert_awaited_once_with(request)
    finally:
        auth_mod.ENABLE_AUTH = original_auth
        auth_mod.ENVIRONMENT = original_env
        auth_mod.API_KEY = original_key


async def test_auth_middleware_api_key_wrong():
    """Auth middleware rejects wrong API key with 401 and error body."""
    import middleware.auth as auth_mod

    original_auth = auth_mod.ENABLE_AUTH
    original_env = auth_mod.ENVIRONMENT
    original_key = auth_mod.API_KEY
    auth_mod.ENABLE_AUTH = True
    auth_mod.ENVIRONMENT = "test"
    auth_mod.API_KEY = "correct-key"
    try:
        middleware = auth_mod.AuthMiddleware(app=MagicMock())
        request = _make_auth_request(
            path="/api/data",
            headers={"authorization": "Bearer wrong-key"},
            client_host="8.8.8.8",
        )
        response = MagicMock()
        call_next = AsyncMock(return_value=response)

        result = await middleware.dispatch(request, call_next)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 401
        body = json.loads(result.body)
        assert body["error"]["error_code"] == "UNAUTHORIZED"
        call_next.assert_not_awaited()
    finally:
        auth_mod.ENABLE_AUTH = original_auth
        auth_mod.ENVIRONMENT = original_env
        auth_mod.API_KEY = original_key


# ── Auth middleware: TRUSTED_CIDRS parsing + _is_trusted_network ──────


def test_no_hardcoded_rfc1918_helper_remains():
    """The hardcoded RFC1918 trust helper must not exist any more.

    Regression guard: `_is_rfc1918` trusted every private/loopback source IP,
    which on k3s (pod CIDR 10.42.0.0/16 ⊂ 10.0.0.0/8) meant any pod could read
    tenant data with no credential, and no env var could close it.
    """
    import middleware.auth as auth_mod

    assert not hasattr(auth_mod, "_is_rfc1918")


def test_trusted_cidrs_default_is_empty():
    """Absent TRUSTED_CIDRS must mean 'trust no source IP', never 'wide open'."""
    from middleware.auth import _TRUSTED_CIDRS_DEFAULT, _parse_trusted_cidrs

    assert _TRUSTED_CIDRS_DEFAULT == ""
    assert _parse_trusted_cidrs(_TRUSTED_CIDRS_DEFAULT) == []


def test_parse_trusted_cidrs_empty_string_trusts_nothing():
    """An explicitly empty TRUSTED_CIDRS trusts nothing."""
    from middleware.auth import _parse_trusted_cidrs

    assert _parse_trusted_cidrs("") == []
    assert _parse_trusted_cidrs("   ") == []
    assert _parse_trusted_cidrs(",, ,") == []


def test_parse_trusted_cidrs_parses_list():
    """A comma-separated list parses into networks, tolerating whitespace."""
    import ipaddress

    from middleware.auth import _parse_trusted_cidrs

    nets = _parse_trusted_cidrs(" 10.42.0.0/16 , 127.0.0.1/32,::1/128 ")
    assert nets == [
        ipaddress.ip_network("10.42.0.0/16"),
        ipaddress.ip_network("127.0.0.1/32"),
        ipaddress.ip_network("::1/128"),
    ]


def test_parse_trusted_cidrs_rejects_malformed_loudly():
    """A malformed CIDR raises at parse (startup) time — never a silent skip."""
    import pytest

    from middleware.auth import _parse_trusted_cidrs

    with pytest.raises(ValueError, match="TRUSTED_CIDRS contains an invalid entry"):
        _parse_trusted_cidrs("10.0.0.0/8,not-a-cidr")

    with pytest.raises(ValueError, match="TRUSTED_CIDRS contains an invalid entry"):
        _parse_trusted_cidrs("10.0.0.0/99")


def test_is_trusted_network_matches_configured_cidr(trusted_networks):
    """_is_trusted_network is True only inside the configured CIDRs."""
    from middleware.auth import _is_trusted_network

    trusted_networks("192.168.0.0/16,127.0.0.1/32")
    assert _is_trusted_network("192.168.1.1") is True
    assert _is_trusted_network("127.0.0.1") is True
    # Private, but not configured:
    assert _is_trusted_network("10.0.0.1") is False
    assert _is_trusted_network("172.16.0.1") is False


def test_is_trusted_network_public_never_trusted(trusted_networks):
    """Public addresses are not trusted under the default configuration."""
    from middleware.auth import _is_trusted_network

    trusted_networks("")
    assert _is_trusted_network("8.8.8.8") is False
    assert _is_trusted_network("1.1.1.1") is False


def test_is_trusted_network_invalid_or_missing(trusted_networks):
    """Invalid / empty / None hosts are never trusted."""
    from middleware.auth import _is_trusted_network

    trusted_networks("0.0.0.0/0")  # even with everything trusted
    assert _is_trusted_network("not-an-ip") is False
    assert _is_trusted_network("") is False
    assert _is_trusted_network(None) is False


# ── Security middleware tests ─────────────────────────────────────────


async def test_security_middleware_adds_headers():
    """SecurityMiddleware adds all security headers to response."""
    from middleware.security import SECURITY_HEADERS, SecurityMiddleware

    middleware = SecurityMiddleware(app=MagicMock())
    request = MagicMock()
    request.url = MagicMock()
    request.url.path = "/api/test"
    request.method = "GET"
    request.headers = MagicMock()
    request.headers.get = MagicMock(return_value=None)
    request.client = MagicMock()
    request.client.host = "127.0.0.1"

    response = MagicMock()
    response.headers = {}
    call_next = AsyncMock(return_value=response)

    result = await middleware.dispatch(request, call_next)
    assert result.headers["X-Content-Type-Options"] == "nosniff"
    assert result.headers["X-Frame-Options"] == "DENY"
    assert result.headers["X-XSS-Protection"] == "1; mode=block"
    assert result.headers["Strict-Transport-Security"] == "max-age=31536000; includeSubDomains"
    assert result.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert len(result.headers) == len(SECURITY_HEADERS)


async def test_security_middleware_blocked_method_405():
    """SecurityMiddleware returns 405 with error body for blocked HTTP methods."""
    from middleware.security import SecurityMiddleware

    middleware = SecurityMiddleware(app=MagicMock())
    request = MagicMock()
    request.url = MagicMock()
    request.url.path = "/api/test"
    request.method = "TRACE"
    request.headers = MagicMock()
    request.headers.get = MagicMock(return_value=None)
    request.client = MagicMock()
    request.client.host = "127.0.0.1"

    call_next = AsyncMock()
    result = await middleware.dispatch(request, call_next)
    assert isinstance(result, JSONResponse)
    assert result.status_code == 405
    body = json.loads(result.body)
    assert body["error"] == "Method not allowed"
    assert body["error_code"] == "SECURITY_ERROR"
    call_next.assert_not_awaited()


async def test_security_middleware_content_length_413():
    """SecurityMiddleware returns 413 with error body for oversized requests."""
    from middleware.security import SecurityMiddleware

    middleware = SecurityMiddleware(app=MagicMock())
    request = MagicMock()
    request.url = MagicMock()
    request.url.path = "/api/test"
    request.method = "POST"

    def header_get(name, default=None):
        if name == "content-length":
            return str(50 * 1024 * 1024)  # 50 MB, over the 10 MB limit
        return default

    request.headers = MagicMock()
    request.headers.get = header_get
    request.client = MagicMock()
    request.client.host = "127.0.0.1"

    call_next = AsyncMock()
    result = await middleware.dispatch(request, call_next)
    assert isinstance(result, JSONResponse)
    assert result.status_code == 413
    body = json.loads(result.body)
    assert body["error"] == "Request entity too large"
    assert body["error_code"] == "SECURITY_ERROR"
    call_next.assert_not_awaited()


async def test_security_middleware_permissive_path_larger_limit():
    """SecurityMiddleware allows larger body on permissive paths like /api/tests/execute/run."""
    from middleware.security import SecurityMiddleware

    middleware = SecurityMiddleware(app=MagicMock())
    request = MagicMock()
    request.url = MagicMock()
    request.url.path = "/api/tests/execute/run"
    request.method = "POST"

    # 15 MB, over default but under permissive 50 MB limit
    def header_get(name, default=None):
        if name == "content-length":
            return str(15 * 1024 * 1024)
        return default

    request.headers = MagicMock()
    request.headers.get = header_get
    request.client = MagicMock()
    request.client.host = "127.0.0.1"

    response = MagicMock()
    response.headers = {}
    call_next = AsyncMock(return_value=response)

    result = await middleware.dispatch(request, call_next)
    # Should pass through, not 413
    assert result is response
    call_next.assert_awaited_once()


async def test_security_middleware_doc_rate_limit():
    """SecurityMiddleware rate-limits documentation endpoints at 30 per 60s."""
    from middleware.security import SecurityMiddleware

    middleware = SecurityMiddleware(app=MagicMock())

    request = MagicMock()
    request.url = MagicMock()
    request.url.path = "/docs"
    request.method = "GET"
    request.headers = MagicMock()
    request.headers.get = MagicMock(return_value=None)
    request.client = MagicMock()
    request.client.host = "192.168.1.1"

    response = MagicMock()
    response.headers = {}
    call_next = AsyncMock(return_value=response)

    # First 30 should succeed
    for _ in range(30):
        result = await middleware.dispatch(request, call_next)
        assert result is response

    # 31st should be rate limited
    result = await middleware.dispatch(request, call_next)
    assert isinstance(result, JSONResponse)
    assert result.status_code == 429
    body = json.loads(result.body)
    assert body["error"] == "Rate limit exceeded for docs"


# ── Exception handlers tests (KAG standard via coding_client) ────────


def test_register_exception_handlers_delegates_to_coding_client():
    """register_exception_handlers registers handlers for Exception and RequestValidationError."""
    from fastapi import FastAPI

    from middleware.exception_handlers import register_exception_handlers

    app = FastAPI()
    register_exception_handlers(app)
    # KAG standard registers handlers for Exception and RequestValidationError
    assert Exception in app.exception_handlers
    assert callable(app.exception_handlers[Exception])


def test_register_exception_handlers_with_string_debug(monkeypatch):
    """register_exception_handlers handles debug as string (e.g. from env var)."""
    from fastapi import FastAPI

    from middleware.exception_handlers import register_exception_handlers

    monkeypatch.setenv("DEBUG", "true")
    app = FastAPI()
    register_exception_handlers(app)
    assert Exception in app.exception_handlers


# ── Validation middleware tests ───────────────────────────────────────


async def test_validation_middleware_excluded_path():
    """ValidationMiddleware skips excluded paths and delegates to next handler."""
    from middleware.validation import ValidationMiddleware

    middleware = ValidationMiddleware(app=MagicMock())
    request = MagicMock()
    request.url = MagicMock()
    request.url.path = "/health"

    response = MagicMock()
    call_next = AsyncMock(return_value=response)

    result = await middleware.dispatch(request, call_next)
    assert result is response
    call_next.assert_awaited_once_with(request)


async def test_validation_middleware_double_slash():
    """ValidationMiddleware normalizes double slashes to single slashes."""
    from middleware.validation import ValidationMiddleware

    middleware = ValidationMiddleware(app=MagicMock())
    request = MagicMock()
    request.url = MagicMock()
    request.url.path = "/api//test"
    request.scope = {"path": "/api//test"}
    raw_url = "http://testserver/api//test"
    request.url.__str__ = MagicMock(return_value=raw_url)

    response = MagicMock()
    call_next = AsyncMock(return_value=response)

    result = await middleware.dispatch(request, call_next)
    # Path should be normalized
    assert request.scope["path"] == "/api/test"
    assert result is response
    call_next.assert_awaited_once()


async def test_validation_middleware_trailing_slash():
    """ValidationMiddleware strips trailing slashes."""
    from middleware.validation import ValidationMiddleware

    middleware = ValidationMiddleware(app=MagicMock())
    request = MagicMock()
    request.url = MagicMock()
    request.url.path = "/api/test/"
    request.scope = {"path": "/api/test/"}
    raw_url = "http://testserver/api/test/"
    request.url.__str__ = MagicMock(return_value=raw_url)

    response = MagicMock()
    call_next = AsyncMock(return_value=response)

    result = await middleware.dispatch(request, call_next)
    assert request.scope["path"] == "/api/test"
    assert result is response
    call_next.assert_awaited_once()


async def test_validation_middleware_control_chars():
    """ValidationMiddleware rejects URLs with control characters with 400."""
    from middleware.validation import ValidationMiddleware

    middleware = ValidationMiddleware(app=MagicMock())
    request = MagicMock()
    request.url = MagicMock()
    request.url.path = "/api/test"
    request.scope = {"path": "/api/test"}
    # URL contains control character \x00
    raw_url = "http://testserver/api/test\x00exploit"
    request.url.__str__ = MagicMock(return_value=raw_url)

    call_next = AsyncMock()
    result = await middleware.dispatch(request, call_next)
    assert isinstance(result, JSONResponse)
    assert result.status_code == 400
    body = json.loads(result.body)
    assert body["error_code"] == "CONTROL_CHAR_DETECTED"
    assert body["error"] == "Request contains invalid control characters"
    call_next.assert_not_awaited()


async def test_validation_middleware_clean_path_no_change():
    """ValidationMiddleware does not modify clean paths."""
    from middleware.validation import ValidationMiddleware

    middleware = ValidationMiddleware(app=MagicMock())
    request = MagicMock()
    request.url = MagicMock()
    request.url.path = "/api/test"
    request.scope = {"path": "/api/test"}
    raw_url = "http://testserver/api/test"
    request.url.__str__ = MagicMock(return_value=raw_url)

    response = MagicMock()
    call_next = AsyncMock(return_value=response)

    result = await middleware.dispatch(request, call_next)
    assert result is response
    assert request.scope["path"] == "/api/test"
    call_next.assert_awaited_once()


# ── SecurityValidationException ───────────────────────────────────────


def test_security_validation_exception_defaults():
    """SecurityValidationException has correct defaults."""
    from middleware.validation import SecurityValidationException

    exc = SecurityValidationException()
    assert exc.status_code == 400
    assert exc.detail == "Security validation failed"
    assert exc.is_security_error is True
    assert exc.error_code == "SECURITY_VALIDATION_ERROR"


def test_security_validation_exception_custom():
    """SecurityValidationException accepts custom values."""
    from middleware.validation import SecurityValidationException

    exc = SecurityValidationException(
        status_code=403,
        detail="Custom security error",
        is_security_error=False,
        error_code="CUSTOM_SEC",
    )
    assert exc.status_code == 403
    assert exc.detail == "Custom security error"
    assert exc.is_security_error is False
    assert exc.error_code == "CUSTOM_SEC"


# ── ValidationMiddleware static helpers ───────────────────────────────


def test_preprocess_request_normalises():
    """_preprocess_request normalises double slashes and trailing slash."""
    from middleware.validation import ValidationMiddleware

    assert ValidationMiddleware._preprocess_request("/api//test/") == "/api/test"
    assert ValidationMiddleware._preprocess_request("///a///b///") == "/a/b"
    assert ValidationMiddleware._preprocess_request("/") == "/"


def test_create_validation_error_response():
    """_create_validation_error_response returns correct JSON with error_code."""
    from middleware.validation import SecurityValidationException, ValidationMiddleware

    exc = SecurityValidationException(status_code=400, detail="Bad chars", error_code="CTRL_CHAR")
    resp = ValidationMiddleware._create_validation_error_response(exc)
    assert resp.status_code == 400
    body = json.loads(resp.body)
    assert body["error"] == "Bad chars"
    assert body["error_code"] == "CTRL_CHAR"
    assert body["status_code"] == 400


def test_create_http_error_response():
    """_create_http_error_response returns correct JSON with status_code."""
    from middleware.validation import ValidationMiddleware

    exc = HTTPException(status_code=422, detail="Invalid payload")
    resp = ValidationMiddleware._create_http_error_response(exc)
    assert resp.status_code == 422
    body = json.loads(resp.body)
    assert body["error"] == "Invalid payload"
    assert body["status_code"] == 422


def test_create_system_error_response():
    """_create_system_error_response returns 500 with request_id in UUID format."""
    from middleware.validation import ValidationMiddleware

    exc = RuntimeError("oops")
    resp = ValidationMiddleware._create_system_error_response(exc)
    assert resp.status_code == 500
    body = json.loads(resp.body)
    assert body["error"] == "Internal server error"
    assert body["error_code"] == "INTERNAL_ERROR"
    assert body["status_code"] == 500
    # Full UUID format: 8-4-4-4-12 = 36 chars
    assert len(body["request_id"]) == 36
    # Validate UUID format (contains 4 hyphens)
    parts = body["request_id"].split("-")
    assert len(parts) == 5
    assert [len(p) for p in parts] == [8, 4, 4, 4, 12]


# ── KAG RequestLoggingMiddleware ───────────────────────────────────────


def test_request_logging_middleware_importable():
    """RequestLoggingMiddleware and HealthCheckFilter can be imported and are classes."""
    from coding_client.api.request_logging import HealthCheckFilter, RequestLoggingMiddleware

    assert RequestLoggingMiddleware.__name__ == "RequestLoggingMiddleware"
    assert HealthCheckFilter.__name__ == "HealthCheckFilter"
    assert callable(RequestLoggingMiddleware)
    assert callable(HealthCheckFilter)


# ── Licence Bearer authentication ─────────────────────────────────────
#
# The capability this service lacked entirely until coding-client 1.10.1. Before
# it, the ONLY working production credential was a bare, forgeable X-Tenant-Id:
# the API-key branch is gated on ENVIRONMENT in ("development","test") and dead
# in production. Since the header arm already honours TRUST_TENANT_HEADER,
# setting that flag to false would have closed the forged arm and left nothing —
# a 401-everything switch. These tests pin the arm that makes the flip a
# hardening instead.
#
# ⚠️ Mutation guard: delete the `_user_from_license` arm from
# middleware/auth.py, or pin coding-client below 1.10.0, and these fail BY NAME.
# They patch coding_client.auth.tenant.httpx.AsyncClient rather than the
# introspection function itself, so the SDK's real parsing and caching run — a
# stub of the function under test would prove nothing about it.


def _introspection(payload, status_code=200):
    from unittest.mock import patch

    response = MagicMock()
    response.status_code = status_code
    response.json = MagicMock(return_value=payload)
    client = AsyncMock()
    client.post = AsyncMock(return_value=response)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return patch("coding_client.auth.tenant.httpx.AsyncClient", return_value=ctx), client


def _licence_request():
    """An otherwise credential-less request carrying only a licence Bearer."""
    return _make_auth_request(
        path="/api/data",
        headers={"authorization": "Bearer a-real-licence"},
        client_host="10.42.9.9",  # untrusted; only the Bearer can grant
    )


async def test_auth_middleware_licence_bearer_authenticates():
    """A valid licence Bearer authenticates and sets the introspected tenant."""
    from unittest.mock import patch

    from coding_client.auth.tenant import clear_license_caches

    import middleware.auth as auth_mod

    clear_license_caches()
    introspect, http = _introspection(
        {"active": True, "tenant_id": "tenant-from-licence", "subject_id": "subject-1"}
    )
    original_auth, original_env = auth_mod.ENABLE_AUTH, auth_mod.ENVIRONMENT
    auth_mod.ENABLE_AUTH, auth_mod.ENVIRONMENT = True, "production"
    try:
        with (
            patch.dict(
                "os.environ", {"LICENSE_INTROSPECT_URL": "http://licences.invalid/introspect"}
            ),
            introspect,
        ):
            middleware = auth_mod.AuthMiddleware(app=MagicMock())
            request = _licence_request()
            call_next = AsyncMock(return_value="response")
            await middleware.dispatch(request, call_next)
    finally:
        auth_mod.ENABLE_AUTH, auth_mod.ENVIRONMENT = original_auth, original_env
        clear_license_caches()

    call_next.assert_awaited()  # not a 401
    assert request.state.user.tenant_id == "tenant-from-licence"
    # This service's routes read state.tenant_id; the SDK does not set it.
    assert request.state.tenant_id == "tenant-from-licence"
    # The tenant came off the wire, not from a header or a default.
    http.post.assert_awaited_once()


async def test_auth_middleware_inactive_licence_is_rejected():
    """Non-vacuity: the Bearer arm can say no, so the test above means something."""
    from unittest.mock import patch

    from coding_client.auth.tenant import clear_license_caches

    import middleware.auth as auth_mod

    clear_license_caches()
    introspect, _ = _introspection({"active": False, "tenant_id": "tenant-from-licence"})
    original_auth, original_env, original_key = (
        auth_mod.ENABLE_AUTH,
        auth_mod.ENVIRONMENT,
        auth_mod.API_KEY,
    )
    auth_mod.ENABLE_AUTH, auth_mod.ENVIRONMENT, auth_mod.API_KEY = True, "production", ""
    try:
        with (
            patch.dict(
                "os.environ", {"LICENSE_INTROSPECT_URL": "http://licences.invalid/introspect"}
            ),
            introspect,
        ):
            middleware = auth_mod.AuthMiddleware(app=MagicMock())
            request = _licence_request()
            call_next = AsyncMock()
            result = await middleware.dispatch(request, call_next)
    finally:
        auth_mod.ENABLE_AUTH, auth_mod.ENVIRONMENT, auth_mod.API_KEY = (
            original_auth,
            original_env,
            original_key,
        )
        clear_license_caches()

    assert result.status_code == 401
    call_next.assert_not_awaited()


async def test_auth_middleware_licence_bearer_works_with_header_trust_disabled():
    """The exact combination the campaign is aiming at.

    TRUST_TENANT_HEADER=false closes the forged-header arm AND the licence
    Bearer still authenticates — a hardening, not a 401-everything switch.
    """
    from unittest.mock import patch

    from coding_client.auth.tenant import clear_license_caches

    import middleware.auth as auth_mod

    clear_license_caches()
    introspect, _ = _introspection(
        {"active": True, "tenant_id": "tenant-from-licence", "subject_id": "subject-1"}
    )
    original_auth, original_env = auth_mod.ENABLE_AUTH, auth_mod.ENVIRONMENT
    auth_mod.ENABLE_AUTH, auth_mod.ENVIRONMENT = True, "production"
    try:
        with (
            patch.dict(
                "os.environ",
                {
                    "LICENSE_INTROSPECT_URL": "http://licences.invalid/introspect",
                    "TRUST_TENANT_HEADER": "false",
                },
            ),
            introspect,
        ):
            middleware = auth_mod.AuthMiddleware(app=MagicMock())
            request = _make_auth_request(
                path="/api/data",
                headers={
                    "authorization": "Bearer a-real-licence",
                    # Forged ROLES only. The forged x-tenant-id that used to sit
                    # here was removed, not because it stopped mattering but
                    # because it now provokes a different, stronger answer: a
                    # licence asserting a tenant it may not name is refused 403
                    # by the on-behalf-of arm (see test_fork_vouching.py). The
                    # comment here used to say "there is no on-behalf-of arm",
                    # which was true when written and false since coding-client
                    # 1.11.0. This test's own claim — a licence Bearer works
                    # with header trust disabled — is unchanged and still pinned.
                    "x-user-roles": "admin",
                },
                client_host="10.42.9.9",
            )
            call_next = AsyncMock(return_value="response")
            await middleware.dispatch(request, call_next)
    finally:
        auth_mod.ENABLE_AUTH, auth_mod.ENVIRONMENT = original_auth, original_env
        clear_license_caches()

    call_next.assert_awaited()
    assert request.state.user.tenant_id == "tenant-from-licence"
    assert request.state.tenant_id == "tenant-from-licence"
