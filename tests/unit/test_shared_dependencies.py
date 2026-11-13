"""Tests for shared.dependencies module."""

from __future__ import annotations

import pytest
from coding_client.auth._onbehalf import STATE_VOUCHED
from coding_client.auth.models import User
from fastapi import HTTPException
from starlette.requests import Request

from shared.dependencies import (
    PermissionException,
    create_json_http_exception,
    requires_read_permission,
    requires_write_permission,
)


def _request(*, vouched: bool = False) -> Request:
    """A request whose tenant was (or was not) resolved from a verified licence."""
    request = Request({"type": "http", "headers": []})
    if vouched:
        setattr(request.state, STATE_VOUCHED, "svc-coding-gateway")
    return request


# ── create_json_http_exception ────────────────────────────────────────


def test_create_json_http_exception_basic():
    """create_json_http_exception returns HTTPException with correct fields."""
    exc = create_json_http_exception(status_code=400, detail="Bad request")
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 400
    assert exc.detail == "Bad request"
    assert exc.headers["Content-Type"] == "application/json"


def test_create_json_http_exception_with_headers():
    """create_json_http_exception merges custom headers."""
    exc = create_json_http_exception(
        status_code=404, detail="Not found", headers={"X-Custom": "value"}
    )
    assert exc.headers["X-Custom"] == "value"
    assert exc.headers["Content-Type"] == "application/json"


def test_create_json_http_exception_none_headers():
    """create_json_http_exception works with no custom headers."""
    exc = create_json_http_exception(status_code=500, detail="Error")
    assert exc.headers["Content-Type"] == "application/json"


# ── PermissionException ──────────────────────────────────────────────


def test_permission_exception_defaults():
    """PermissionException has correct default values."""
    exc = PermissionException()
    assert exc.status_code == 403
    assert exc.error_code == "FORBIDDEN"
    assert isinstance(exc.detail, dict)
    assert exc.detail["message"] == "Insufficient permissions"
    assert exc.error_details is None


def test_permission_exception_with_message():
    """PermissionException accepts custom message."""
    exc = PermissionException(message="Custom forbidden")
    assert exc.detail["message"] == "Custom forbidden"


def test_permission_exception_with_required_permission():
    """PermissionException adds required_permission to details."""
    exc = PermissionException(required_permission="write")
    assert exc.error_details["required_permission"] == "write"
    assert exc.detail["details"]["required_permission"] == "write"


def test_permission_exception_with_details():
    """PermissionException accepts extra details."""
    exc = PermissionException(
        message="Denied",
        error_code="ACCESS_DENIED",
        details={"resource": "test-123"},
    )
    assert exc.error_code == "ACCESS_DENIED"
    assert exc.error_details["resource"] == "test-123"


def test_permission_exception_with_required_permission_and_details():
    """PermissionException merges required_permission into existing details."""
    exc = PermissionException(
        required_permission="admin",
        details={"extra": "info"},
    )
    assert exc.error_details["required_permission"] == "admin"
    assert exc.error_details["extra"] == "info"


# ── requires_read_permission ─────────────────────────────────────────


async def test_requires_read_permission_allowed():
    """requires_read_permission passes for user with viewer role."""
    user = User(sub="u1", preferred_username="u1", realm_access={"roles": ["viewer"]})
    result = await requires_read_permission(request=_request(), user=user)
    assert result is user


async def test_requires_read_permission_allowed_admin():
    """requires_read_permission passes for admin role."""
    user = User(sub="u1", preferred_username="u1", realm_access={"roles": ["admin"]})
    result = await requires_read_permission(request=_request(), user=user)
    assert result is user


async def test_requires_read_permission_allowed_user():
    """requires_read_permission passes for user role."""
    user = User(sub="u1", preferred_username="u1", realm_access={"roles": ["user"]})
    result = await requires_read_permission(request=_request(), user=user)
    assert result is user


async def test_requires_read_permission_allowed_developer():
    """requires_read_permission passes for developer role."""
    user = User(sub="u1", preferred_username="u1", realm_access={"roles": ["developer"]})
    result = await requires_read_permission(request=_request(), user=user)
    assert result is user


async def test_requires_read_permission_denied():
    """requires_read_permission raises PermissionException for unauthorized role."""
    user = User(sub="u1", preferred_username="u1", realm_access={"roles": ["guest"]})
    with pytest.raises(PermissionException) as exc_info:
        await requires_read_permission(request=_request(), user=user)
    assert exc_info.value.status_code == 403
    assert exc_info.value.error_code == "READ_PERMISSION_REQUIRED"
    assert exc_info.value.error_details["required_permission"] == "read"


async def test_requires_read_permission_no_roles():
    """requires_read_permission raises PermissionException when user has no roles."""
    user = User(sub="u1", preferred_username="u1", realm_access={"roles": []})
    with pytest.raises(PermissionException):
        await requires_read_permission(request=_request(), user=user)


# ── requires_write_permission ────────────────────────────────────────


async def test_requires_write_permission_allowed_developer():
    """requires_write_permission passes for developer role."""
    user = User(sub="u1", preferred_username="u1", realm_access={"roles": ["developer"]})
    result = await requires_write_permission(request=_request(), user=user)
    assert result is user


async def test_requires_write_permission_allowed_admin():
    """requires_write_permission passes for admin role."""
    user = User(sub="u1", preferred_username="u1", realm_access={"roles": ["admin"]})
    result = await requires_write_permission(request=_request(), user=user)
    assert result is user


async def test_requires_write_permission_denied_user():
    """requires_write_permission raises PermissionException for user role."""
    user = User(sub="u1", preferred_username="u1", realm_access={"roles": ["user"]})
    with pytest.raises(PermissionException) as exc_info:
        await requires_write_permission(request=_request(), user=user)
    assert exc_info.value.status_code == 403
    assert exc_info.value.error_code == "WRITE_PERMISSION_REQUIRED"
    assert exc_info.value.error_details["required_permission"] == "write"


async def test_requires_write_permission_denied_viewer():
    """requires_write_permission raises PermissionException for viewer role."""
    user = User(sub="u1", preferred_username="u1", realm_access={"roles": ["viewer"]})
    with pytest.raises(PermissionException):
        await requires_write_permission(request=_request(), user=user)


# ── licensed callers (no realm roles by design) ───────────────────────


async def test_verified_licence_without_roles_may_read_and_write():
    """Live 2026-10-12: every role-gated route 403'd the conveyor licence (no roles)."""
    user = User(sub="svc-n8n-local-conveyor", preferred_username="svc", realm_access={"roles": []})
    assert await requires_read_permission(request=_request(vouched=True), user=user) is user
    assert await requires_write_permission(request=_request(vouched=True), user=user) is user


async def test_unverified_caller_without_roles_is_still_refused():
    """A header-asserted tenant (no licence verification) gains nothing."""
    user = User(sub="someone", preferred_username="someone", realm_access={"roles": []})
    with pytest.raises(PermissionException):
        await requires_read_permission(request=_request(), user=user)
    with pytest.raises(PermissionException):
        await requires_write_permission(request=_request(), user=user)
