"""Dependency injection for FastAPI endpoints."""

from typing import Any

from coding_client.auth import may_vouch, tenant_license_verified
from coding_client.auth.dependencies import get_current_user
from coding_client.auth.models import User
from fastapi import Depends, HTTPException, Request, status


def create_json_http_exception(status_code: int, detail: str, headers: dict | None = None) -> Any:
    """Create JSON HTTP exception."""
    return HTTPException(
        status_code=status_code,
        detail=detail,
        headers={**(headers or {}), "Content-Type": "application/json"},
    )


class PermissionException(HTTPException):
    """Permission/authorization failure (403).

    Mirrors the monolith's ``PermissionException`` so that the global
    exception handler produces the same ``StandardErrorResponse`` shape.
    """

    def __init__(
        self,
        message: str = "Insufficient permissions",
        error_code: str = "FORBIDDEN",
        required_permission: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        if required_permission:
            if details is None:
                details = {}
            details["required_permission"] = required_permission

        detail: dict[str, Any] = {"message": message, "error_code": error_code}
        if details:
            detail["details"] = details

        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail)
        self.error_code = error_code
        self.error_details = details


async def requires_read_permission(request: Request, user: User = Depends(get_current_user)):
    """Require read permission: a realm role, or a verified tenant licence.

    A licensed caller carries no realm roles by design (see
    ``requires_service_account``), so a role-only check refused EVERY licensed
    caller: live 2026-10-12 every route behind it answered 403 to the
    conveyor's licence through the gateway (containerized execution,
    orchestration, history, metrics). The licence is the tenant's member-level
    credential; every route behind this dependency also takes
    ``require_tenant`` and scopes its data to that tenant.
    """
    if tenant_license_verified(request):
        return user
    allowed_roles = ["user", "developer", "admin", "viewer"]
    if not any(user.has_realm_role(role) for role in allowed_roles):
        raise PermissionException(
            message="Insufficient permissions for this operation",
            error_code="READ_PERMISSION_REQUIRED",
            required_permission="read",
        )
    return user


async def requires_write_permission(request: Request, user: User = Depends(get_current_user)):
    """Require write permission: a realm role, or a verified tenant licence (tenant-scoped)."""
    if tenant_license_verified(request):
        return user
    allowed_roles = ["developer", "admin"]
    if not any(user.has_realm_role(role) for role in allowed_roles):
        raise PermissionException(
            message="Insufficient permissions for this operation",
            error_code="WRITE_PERMISSION_REQUIRED",
            required_permission="write",
        )
    return user


async def requires_service_account(request: Request, user: User = Depends(get_current_user)):
    """Admit only a SERVICE principal: a verified licence whose subject is allowlisted.

    Built from the two SDK predicates the on-behalf-of mechanism already uses,
    not a new notion of "service account":

    * ``tenant_license_verified(request)`` — the principal came from licence
      introspection (``resolve_tenant_assertion`` ran), not from a forwarded
      ``x-user-id`` header claim, which any pod can type when header trust is on.
    * ``may_vouch(user.sub)`` — that licence's subject is in
      ``CODING_ONBEHALF_SUBJECTS``, the allowlist of service subjects minted by
      ``licenses/scripts/provision_service_token.py``. Fail-closed: an empty
      allowlist admits nobody.

    Deliberately NOT stacked on ``requires_write_permission``: a licensed caller
    carries no realm roles by design (coding_client.auth.middleware
    ``_user_from_license``), so a role check would refuse every service principal
    and the route could never open under TRUST_TENANT_HEADER=false.
    """
    if not (tenant_license_verified(request) and may_vouch(user.sub)):
        raise PermissionException(
            message="This endpoint accepts service-account principals only",
            error_code="SERVICE_ACCOUNT_REQUIRED",
            required_permission="service_account",
        )
    return user
