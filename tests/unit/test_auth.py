"""Tests for auth models, dependencies, and tenant module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from coding_client.auth.models import User
from fastapi import HTTPException

# ── User model tests ──────────────────────────────────────────────────


def test_user_model_basic():
    """User model can be created with required fields."""
    user = User(sub="user1", preferred_username="testuser")
    assert user.sub == "user1"
    assert user.preferred_username == "testuser"


def test_user_tenant_id_defaults_to_sub():
    """Tenant ID defaults to sub when not provided."""
    user = User(sub="user1", preferred_username="testuser")
    assert user.tenant_id == "user1"


def test_user_tenant_id_explicit():
    """Tenant ID can be set explicitly."""
    user = User(sub="user1", preferred_username="testuser", tenant_id="tenant-abc")
    assert user.tenant_id == "tenant-abc"


def test_user_has_realm_role():
    """has_realm_role checks roles correctly."""
    user = User(
        sub="user1",
        preferred_username="testuser",
        realm_access={"roles": ["admin", "user"]},
    )
    assert user.has_realm_role("admin")
    assert user.has_realm_role("user")
    assert not user.has_realm_role("superadmin")


def test_user_is_admin():
    """is_admin checks for admin role."""
    admin = User(
        sub="admin1",
        preferred_username="admin",
        realm_access={"roles": ["admin"]},
    )
    assert admin.is_admin()

    regular = User(sub="user1", preferred_username="user")
    assert not regular.is_admin()


def test_user_get_all_roles():
    """get_all_roles returns realm + client roles."""
    user = User(
        sub="user1",
        preferred_username="testuser",
        realm_access={"roles": ["user"]},
        resource_access={"client1": {"roles": ["editor"]}},
    )
    roles = user.get_all_roles()
    assert "user" in roles
    assert "editor" in roles


def test_user_effective_tenant_id():
    """effective_tenant_id returns correct fallback chain."""
    user = User(sub="user1", preferred_username="testuser", tenant_id="my-tenant")
    assert user.effective_tenant_id == "my-tenant"


# ── User model — post_init / effective_tenant_id / is_admin / get_all_roles ──


def test_user_model_post_init_uses_organization_id():
    """model_post_init prefers organization_id over sub when tenant_id is None."""
    user = User(
        sub="user1",
        preferred_username="testuser",
        tenant_id=None,
        organization_id="org-123",
    )
    # post_init sets tenant_id = organization_id since tenant_id was None
    assert user.tenant_id == "org-123"


def test_user_model_post_init_falls_back_to_sub():
    """model_post_init falls back to sub when both tenant_id and organization_id are None."""
    user = User(
        sub="user1",
        preferred_username="testuser",
        tenant_id=None,
        organization_id=None,
    )
    assert user.tenant_id == "user1"


def test_user_effective_tenant_id_fallback_chain():
    """effective_tenant_id falls back through tenant_id -> organization_id -> sub."""
    # tenant_id is set explicitly
    u1 = User(sub="u1", preferred_username="u1", tenant_id="t1", organization_id="org1")
    assert u1.effective_tenant_id == "t1"

    # After post_init, tenant_id is set from org_id
    u2 = User(sub="u2", preferred_username="u2", organization_id="org2")
    assert u2.effective_tenant_id == "org2"

    # Falls all the way back to sub
    u3 = User(sub="u3", preferred_username="u3")
    assert u3.effective_tenant_id == "u3"


def test_user_is_admin_platform_admin():
    """is_admin returns True for platform-admin role."""
    user = User(
        sub="pa1",
        preferred_username="platform-admin",
        realm_access={"roles": ["platform-admin"]},
    )
    assert user.is_admin()


def test_user_has_client_role():
    """has_client_role checks client-specific roles."""
    user = User(
        sub="u1",
        preferred_username="u1",
        resource_access={"my-client": {"roles": ["viewer", "editor"]}},
    )
    assert user.has_client_role("my-client", "viewer")
    assert user.has_client_role("my-client", "editor")
    assert not user.has_client_role("my-client", "admin")
    assert not user.has_client_role("other-client", "viewer")


def test_user_get_all_roles_multiple_clients():
    """get_all_roles aggregates roles from all clients."""
    user = User(
        sub="u1",
        preferred_username="u1",
        realm_access={"roles": ["realm-role"]},
        resource_access={
            "client-a": {"roles": ["a-role1", "a-role2"]},
            "client-b": {"roles": ["b-role1"]},
        },
    )
    roles = user.get_all_roles()
    assert set(roles) == {"realm-role", "a-role1", "a-role2", "b-role1"}


def test_user_get_all_roles_empty():
    """get_all_roles returns empty list when no roles."""
    user = User(sub="u1", preferred_username="u1")
    assert user.get_all_roles() == []


def test_user_has_realm_role_no_roles_key():
    """has_realm_role returns False when realm_access has no 'roles' key."""
    user = User(sub="u1", preferred_username="u1", realm_access={})
    assert not user.has_realm_role("admin")


# ── Auth dependencies ─────────────────────────────────────────────────


def test_get_current_user_success():
    """get_current_user returns user from request.state."""
    from coding_client.auth.dependencies import get_current_user

    user = User(sub="u1", preferred_username="u1")
    request = MagicMock()
    request.state.user = user
    result = get_current_user(request)
    assert result is user
    assert result.sub == "u1"


def test_get_current_user_missing_raises_401():
    """get_current_user raises 401 when no user on request.state."""
    from coding_client.auth.dependencies import get_current_user

    request = MagicMock(spec=[])
    request.state = MagicMock(spec=[])  # no 'user' attribute

    with pytest.raises(HTTPException) as exc_info:
        get_current_user(request)
    assert exc_info.value.status_code == 401
    assert "Authentication required" in str(exc_info.value.detail)


def test_require_role_allowed():
    """require_role passes when user has the role."""
    from coding_client.auth.dependencies import require_role

    checker = require_role("admin")
    user = User(
        sub="u1",
        preferred_username="u1",
        realm_access={"roles": ["admin", "user"]},
    )
    request = MagicMock()
    request.state.user = user

    result = checker(request)
    assert result is user


def test_require_role_denied():
    """require_role raises 403 when user lacks the role."""
    from coding_client.auth.dependencies import require_role

    checker = require_role("superadmin")
    user = User(sub="u1", preferred_username="u1", realm_access={"roles": ["user"]})
    request = MagicMock()
    request.state.user = user

    with pytest.raises(HTTPException) as exc_info:
        checker(request)
    assert exc_info.value.status_code == 403
    assert "superadmin" in str(exc_info.value.detail)


def test_require_any_role_one_match():
    """require_any_role passes when user has at least one of the roles."""
    from coding_client.auth.dependencies import require_any_role

    checker = require_any_role("editor", "admin")
    user = User(sub="u1", preferred_username="u1", realm_access={"roles": ["editor"]})
    request = MagicMock()
    request.state.user = user

    result = checker(request)
    assert result is user


def test_require_any_role_no_match():
    """require_any_role raises 403 when user has none of the roles."""
    from coding_client.auth.dependencies import require_any_role

    checker = require_any_role("editor", "admin")
    user = User(sub="u1", preferred_username="u1", realm_access={"roles": ["viewer"]})
    request = MagicMock()
    request.state.user = user

    with pytest.raises(HTTPException) as exc_info:
        checker(request)
    assert exc_info.value.status_code == 403
    assert "editor" in str(exc_info.value.detail)
    assert "admin" in str(exc_info.value.detail)


# ── Tenant: _is_trusted_network ───────────────────────────────────────


def test_is_trusted_network_private_not_trusted_by_default():
    """RFC1918 addresses are NOT trusted by default (coding-client >= 1.7.0).

    10.0.0.0/8 contains the k3s pod CIDR 10.42.0.0/16, so an RFC1918 default
    means every pod is 'trusted'. The SDK default is now empty; coding-testing
    additionally pins TRUSTED_CIDRS="" in gitops. The network bypass is only live
    for a CIDR an operator explicitly names — see the configured control below.
    """
    from auth.tenant import _is_trusted_network

    assert _is_trusted_network("192.168.1.1") is False
    assert _is_trusted_network("10.0.0.1") is False
    assert _is_trusted_network("172.16.0.1") is False


def test_is_trusted_network_configured_cidr_is_trusted():
    """False-positive control: an operator-named CIDR still resolves True."""
    import ipaddress

    import coding_client.auth._trusted_networks as tn

    with patch.object(tn, "TRUSTED_NETWORKS", [ipaddress.ip_network("192.168.0.0/16")]):
        assert _tn_is_trusted("192.168.1.1") is True
        assert _tn_is_trusted("10.0.0.1") is False


def _tn_is_trusted(ip):
    from auth.tenant import _is_trusted_network

    return _is_trusted_network(ip)


def test_is_trusted_network_loopback_not_trusted_by_default():
    """Loopback is not trusted by default either (empty TRUSTED_CIDRS)."""
    from auth.tenant import _is_trusted_network

    assert _is_trusted_network("127.0.0.1") is False


def test_is_trusted_network_public():
    """Public addresses are not trusted."""
    from auth.tenant import _is_trusted_network

    assert _is_trusted_network("8.8.8.8") is False
    assert _is_trusted_network("1.1.1.1") is False


def test_is_trusted_network_invalid():
    """Invalid addresses return False."""
    from auth.tenant import _is_trusted_network

    assert _is_trusted_network("not-an-ip") is False
    assert _is_trusted_network("") is False


def test_is_trusted_network_none():
    """None returns False."""
    from auth.tenant import _is_trusted_network

    assert _is_trusted_network(None) is False


# ── Tenant: _introspect_license ───────────────────────────────────────


async def test_introspect_license_cache_hit():
    """_introspect_license returns cached data when present."""
    import hashlib

    from auth.tenant import _introspect_license, _license_cache

    token = "cached-token-xyz"
    cache_key = f"lic:{hashlib.sha256(token.encode()).hexdigest()[:16]}"
    cached_data = {"active": True, "subject_id": "tenant-1"}
    _license_cache[cache_key] = cached_data

    try:
        result = await _introspect_license(token)
        assert result == cached_data
    finally:
        _license_cache.pop(cache_key, None)


async def test_introspect_license_active_success():
    """_introspect_license returns data for active license."""
    import hashlib

    from auth.tenant import _introspect_license, _license_cache

    token = "active-token-abc"
    cache_key = f"lic:{hashlib.sha256(token.encode()).hexdigest()[:16]}"
    _license_cache.pop(cache_key, None)  # ensure no cache

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"active": True, "subject_id": "t-active"}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("coding_client.auth.tenant.httpx.AsyncClient", return_value=mock_client):
        result = await _introspect_license(token)

    assert result == {"active": True, "subject_id": "t-active"}
    # Should be cached now
    assert cache_key in _license_cache
    _license_cache.pop(cache_key, None)


async def test_introspect_license_inactive():
    """_introspect_license returns data but does NOT cache inactive licenses."""
    import hashlib

    from auth.tenant import _introspect_license, _license_cache

    token = "inactive-token-def"
    cache_key = f"lic:{hashlib.sha256(token.encode()).hexdigest()[:16]}"
    _license_cache.pop(cache_key, None)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"active": False}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("coding_client.auth.tenant.httpx.AsyncClient", return_value=mock_client):
        result = await _introspect_license(token)

    assert result == {"active": False}
    # Inactive should NOT be cached
    assert cache_key not in _license_cache


async def test_introspect_license_non_200():
    """_introspect_license returns None on non-200 response."""
    import hashlib

    from auth.tenant import _introspect_license, _license_cache

    token = "bad-token-ghi"
    cache_key = f"lic:{hashlib.sha256(token.encode()).hexdigest()[:16]}"
    _license_cache.pop(cache_key, None)

    mock_response = MagicMock()
    mock_response.status_code = 500

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("coding_client.auth.tenant.httpx.AsyncClient", return_value=mock_client):
        result = await _introspect_license(token)

    assert result is None


async def test_introspect_license_request_exception():
    """_introspect_license returns None on request exception."""
    import hashlib

    from auth.tenant import _introspect_license, _license_cache

    token = "exception-token-jkl"
    cache_key = f"lic:{hashlib.sha256(token.encode()).hexdigest()[:16]}"
    _license_cache.pop(cache_key, None)

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("coding_client.auth.tenant.httpx.AsyncClient", return_value=mock_client):
        result = await _introspect_license(token)

    assert result is None


# ── Tenant: TenantMiddleware ──────────────────────────────────────────


async def test_tenant_middleware_sets_tenant_from_header():
    """TenantMiddleware updates user.tenant_id from X-Tenant-Id header."""
    from auth.tenant import TenantMiddleware

    user = User(sub="u1", preferred_username="u1", tenant_id="old-tenant")

    request = MagicMock()
    request.headers = {"X-Tenant-Id": "new-tenant"}
    request.state.user = user

    response = MagicMock()
    call_next = AsyncMock(return_value=response)

    middleware = TenantMiddleware(app=MagicMock())
    result = await middleware.dispatch(request, call_next)

    assert user.tenant_id == "new-tenant"
    assert request.state.tenant_id == user.effective_tenant_id
    assert result is response


async def test_tenant_middleware_defaults_to_sub_when_no_header():
    """TenantMiddleware defaults tenant_id to user.sub when no header and no existing tenant."""
    from auth.tenant import TenantMiddleware

    user = User.model_construct(
        sub="u1",
        preferred_username="u1",
        tenant_id=None,
        realm_access={},
        resource_access={},
        organization_id=None,
        email=None,
        email_verified=False,
        name=None,
    )

    request = MagicMock()
    request.headers = {}
    request.state.user = user

    response = MagicMock()
    call_next = AsyncMock(return_value=response)

    middleware = TenantMiddleware(app=MagicMock())
    result = await middleware.dispatch(request, call_next)

    assert user.tenant_id == "u1"
    assert result is response


async def test_tenant_middleware_header_only_no_user():
    """TenantMiddleware sets tenant_id from header when no user is present."""
    from auth.tenant import TenantMiddleware

    request = MagicMock()
    request.headers = {"X-Tenant-Id": "header-tenant"}
    # No user attribute
    request.state = MagicMock(spec=[])

    response = MagicMock()
    call_next = AsyncMock(return_value=response)

    middleware = TenantMiddleware(app=MagicMock())
    result = await middleware.dispatch(request, call_next)

    assert request.state.tenant_id == "header-tenant"
    assert result is response


async def test_tenant_middleware_no_header_no_user():
    """TenantMiddleware proceeds without setting tenant when no header and no user."""
    from auth.tenant import TenantMiddleware

    request = MagicMock()
    request.headers = {}
    request.state = MagicMock(spec=[])

    response = MagicMock()
    call_next = AsyncMock(return_value=response)

    middleware = TenantMiddleware(app=MagicMock())
    result = await middleware.dispatch(request, call_next)

    assert result is response


# ── Tenant: require_tenant ────────────────────────────────────────────


@pytest.mark.parametrize("value", ["true", "True", "TRUE", "1", "yes"])
async def test_require_tenant_skip_auth_no_longer_bypasses_anything(monkeypatch, value):
    """SKIP_AUTH was deleted from coding-client in 1.16.0.

    Inverted rather than deleted: the old assertion — SKIP_AUTH=true yields
    "test-tenant" — WAS the hatch. Without a test in its place, nothing here
    would notice it returning through a pin downgrade.
    """
    monkeypatch.setenv("SKIP_AUTH", value)
    from auth.tenant import require_tenant

    request = MagicMock()
    request.state = MagicMock(spec=[])
    request.headers = {}
    request.client.host = "10.42.0.1"
    with pytest.raises(HTTPException) as exc:
        await require_tenant(request)
    assert exc.value.status_code == 401


async def test_require_tenant_from_state(monkeypatch):
    """require_tenant returns tenant from request.state when present."""
    monkeypatch.setenv("SKIP_AUTH", "false")
    from auth.tenant import require_tenant

    request = MagicMock()
    request.state.tenant_id = "state-tenant"
    result = await require_tenant(request)
    assert result == "state-tenant"


async def test_require_tenant_from_license_introspection(monkeypatch):
    """require_tenant falls back to license introspection."""
    monkeypatch.setenv("SKIP_AUTH", "false")
    from auth.tenant import require_tenant

    request = MagicMock()
    request.state = MagicMock(spec=[])  # no tenant_id
    request.headers = {"Authorization": "Bearer valid-license-token"}
    request.client = MagicMock()
    request.client.host = "8.8.8.8"  # not trusted

    # coding-client 1.5.7: the tenant is `tenant_id` (workspace); `subject_id` is
    # the USER and must NOT be used for tenant scoping. Provide both and assert we
    # resolve the tenant, not the user (regression guard for the introspection fix).
    license_data = {"active": True, "tenant_id": "license-tenant", "subject_id": "the-user"}
    with patch(
        "coding_client.auth.tenant._introspect_license",
        new_callable=AsyncMock,
        return_value=license_data,
    ):
        result = await require_tenant(request)

    assert result == "license-tenant"
    assert result != "the-user"


async def test_require_tenant_from_user_state(monkeypatch):
    """require_tenant falls back to user.effective_tenant_id."""
    monkeypatch.setenv("SKIP_AUTH", "false")
    from auth.tenant import require_tenant

    user = User(sub="u1", preferred_username="u1", tenant_id="user-tenant")
    request = MagicMock()
    request.state = MagicMock(spec=[])  # no tenant_id attribute
    request.headers = {}  # no auth header
    request.state.user = user
    request.client = MagicMock()
    request.client.host = "8.8.8.8"  # not trusted

    result = await require_tenant(request)
    assert result == "user-tenant"


async def test_require_tenant_trusted_network_bypass_closed_by_default(monkeypatch):
    """With TRUSTED_CIDRS empty (the 1.7.0+ default AND coding-testing's gitops
    configmap), a header-less caller from an RFC1918 IP no longer falls into the
    priority-4 bypass — it 401s like any other credential-less request.
    Previously this returned DEFAULT_TENANT_ID to any pod in the cluster.
    """
    monkeypatch.setenv("SKIP_AUTH", "false")
    from auth.tenant import require_tenant

    request = MagicMock()
    request.state = MagicMock(spec=[])  # no tenant_id, no user
    request.headers = {}
    request.client = MagicMock()
    request.client.host = "192.168.1.100"

    with pytest.raises(HTTPException) as exc_info:
        await require_tenant(request)
    assert exc_info.value.status_code == 401


async def test_require_tenant_trusted_network_bypass_when_configured(monkeypatch):
    """False-positive control: a named CIDR + a configured DEFAULT_TENANT_ID
    still grants — proving the 401 above is the empty default, not a dead path.
    """
    import ipaddress

    import coding_client.auth._default_tenant as dt
    import coding_client.auth._trusted_networks as tn

    monkeypatch.setenv("SKIP_AUTH", "false")
    from auth.tenant import require_tenant

    request = MagicMock()
    request.state = MagicMock(spec=[])
    request.headers = {}
    request.client = MagicMock()
    request.client.host = "192.168.1.100"

    with (
        patch.object(tn, "TRUSTED_NETWORKS", [ipaddress.ip_network("192.168.0.0/16")]),
        patch.object(dt, "DEFAULT_TENANT_ID", "tenant-configured"),
    ):
        result = await require_tenant(request)
        assert result == "tenant-configured"


async def test_require_tenant_raises_401_no_context(monkeypatch):
    """require_tenant raises 401 when no tenant context available."""
    monkeypatch.setenv("SKIP_AUTH", "false")
    from auth.tenant import require_tenant

    request = MagicMock()
    request.state = MagicMock(spec=[])  # no tenant_id, no user
    request.headers = {}
    request.client = MagicMock()
    request.client.host = "8.8.8.8"  # not trusted, no auth

    with pytest.raises(HTTPException) as exc_info:
        await require_tenant(request)
    assert exc_info.value.status_code == 401


# ── Tenant: get_tenant_id ────────────────────────────────────────────


def test_get_tenant_id_present():
    """get_tenant_id returns tenant_id from request.state."""
    from auth.tenant import get_tenant_id

    request = MagicMock()
    request.state.tenant_id = "t-123"
    assert get_tenant_id(request) == "t-123"


def test_get_tenant_id_absent():
    """get_tenant_id returns None when tenant_id not set."""
    from auth.tenant import get_tenant_id

    request = MagicMock()
    request.state = MagicMock(spec=[])  # no tenant_id
    assert get_tenant_id(request) is None


# ── Tenant: get_current_user_with_tenant ──────────────────────────────


def test_get_current_user_with_tenant_success(monkeypatch):
    """get_current_user_with_tenant returns user with tenant_id set."""
    monkeypatch.setenv("SKIP_AUTH", "false")
    from auth.tenant import get_current_user_with_tenant

    user = User(sub="u1", preferred_username="u1", tenant_id="t1")
    request = MagicMock()
    request.state.user = user

    result = get_current_user_with_tenant(request)
    assert result is user
    assert result.tenant_id == "t1"


def test_get_current_user_with_tenant_defaults_tenant_to_sub(monkeypatch):
    """get_current_user_with_tenant defaults tenant_id to sub."""
    monkeypatch.setenv("SKIP_AUTH", "false")
    from auth.tenant import get_current_user_with_tenant

    user = User.model_construct(
        sub="u1",
        preferred_username="u1",
        tenant_id=None,
        realm_access={},
        resource_access={},
        organization_id=None,
        email=None,
        email_verified=False,
        name=None,
    )
    request = MagicMock()
    request.state.user = user

    result = get_current_user_with_tenant(request)
    assert result.tenant_id == "u1"


def test_get_current_user_with_tenant_raises_401(monkeypatch):
    """get_current_user_with_tenant raises 401 when no user and no forwarded headers."""
    monkeypatch.setenv("SKIP_AUTH", "false")
    from auth.tenant import get_current_user_with_tenant

    request = MagicMock()
    request.state = MagicMock(spec=[])
    # No forwarded gateway headers and untrusted client → SDK falls through to 401.
    request.headers = {}
    request.client = MagicMock()
    request.client.host = "8.8.8.8"  # not trusted

    with pytest.raises(HTTPException) as exc_info:
        get_current_user_with_tenant(request)
    assert exc_info.value.status_code == 401


# ── Tenant: check_tenant_access ───────────────────────────────────────


def test_check_tenant_access_admin_always():
    """Admin users can access any tenant."""
    from auth.tenant import check_tenant_access

    admin = User(sub="a1", preferred_username="admin", realm_access={"roles": ["admin"]})
    assert check_tenant_access(admin, "any-tenant") is True


def test_check_tenant_access_same_tenant():
    """Non-admin users can access their own tenant."""
    from auth.tenant import check_tenant_access

    user = User(sub="u1", preferred_username="u1", tenant_id="t1")
    assert check_tenant_access(user, "t1") is True


def test_check_tenant_access_different_tenant():
    """Non-admin users cannot access a different tenant."""
    from auth.tenant import check_tenant_access

    user = User(sub="u1", preferred_username="u1", tenant_id="t1")
    assert check_tenant_access(user, "t2") is False


# ── Tenant: require_same_tenant ───────────────────────────────────────


def test_require_same_tenant_admin_bypass():
    """require_same_tenant allows admin to access any tenant."""
    from auth.tenant import require_same_tenant

    validator = require_same_tenant("other-tenant")
    admin = User(sub="a1", preferred_username="admin", realm_access={"roles": ["admin"]})
    # Simulate the Depends(get_current_user_with_tenant) injection
    result = validator(user=admin)
    assert result is admin


def test_require_same_tenant_same():
    """require_same_tenant allows user with same tenant."""
    from auth.tenant import require_same_tenant

    validator = require_same_tenant("t1")
    user = User(sub="u1", preferred_username="u1", tenant_id="t1")
    result = validator(user=user)
    assert result is user


def test_require_same_tenant_different():
    """require_same_tenant raises 403 for different tenant."""
    from auth.tenant import require_same_tenant

    validator = require_same_tenant("t2")
    user = User(sub="u1", preferred_username="u1", tenant_id="t1")
    with pytest.raises(HTTPException) as exc_info:
        validator(user=user)
    assert exc_info.value.status_code == 403
