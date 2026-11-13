"""Authentication middleware for coding-testing.

Extracts user identity from Istio-forwarded headers or falls back to
API-key authentication for dev/test mode.

Source-IP trust is driven exclusively by the ``TRUSTED_CIDRS`` env var
(same name and parse semantics as ``coding_client.auth.middleware`` /
``coding_client.auth.tenant``, so one Deployment env key governs every
IP-trust layer this service runs). See ``_TRUSTED_CIDRS_DEFAULT`` for why
the default here is "trust nothing" rather than the SDK's RFC1918 list.
"""

from __future__ import annotations

import logging
import os
from ipaddress import IPv4Network, IPv6Network, ip_address, ip_network

from coding_client.auth._header_trust import trust_tenant_header
from coding_client.auth._onbehalf import forwarded_license_id, resolve_tenant_assertion
from coding_client.auth.middleware import _user_from_license
from coding_client.auth.models import User
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from auth.tenant import DEFAULT_TENANT_ID

logger = logging.getLogger(__name__)

# Paths that never require auth
PUBLIC_PATHS: frozenset[str] = frozenset(
    {
        "/health",
        "/healthz",
        "/ready",
        "/readyz",
        "/",
        "/openapi.json",
        "/docs",
        "/redoc",
        "/favicon.ico",
    }
)

#: Cluster-internal routes authenticated by their own per-run token, not a user
#: identity (routers.internal_test_runs: a runner Job has no licence).
INTERNAL_PREFIX = "/internal/test-runs/"

ENABLE_AUTH = os.getenv("ENABLE_AUTH", "true").lower() == "true"
API_KEY = os.getenv("API_KEY", "")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")


# Default: trust NO source IP. Absent ≡ empty ≡ deny.
#
# This deliberately diverges from the coding_client SDK default
# ("127.0.0.1/32,::1/128,192.168.0.0/16,172.16.0.0/12,10.0.0.0/8"). On k3s the
# pod CIDR is 10.42.0.0/16 ⊂ 10.0.0.0/8, so an RFC1918 default means every pod
# in the cluster — and anything that can reach the Service — is served tenant
# data with no credential at all. A default that is wide open in the only
# environment we actually deploy to is not a default, it is the bug. gitops has
# already settled this for the two siblings that were closed on 2026-07-29:
# coding-analysis and coding-security both pin `TRUSTED_CIDRS: ""`.
#
# Widening is therefore an explicit, reviewable diff in the Deployment env
# block, never an accident of an unset variable.
_TRUSTED_CIDRS_DEFAULT = ""
_TRUSTED_CIDRS_RAW = os.getenv("TRUSTED_CIDRS", _TRUSTED_CIDRS_DEFAULT)


def _parse_trusted_cidrs(raw: str) -> list[IPv4Network | IPv6Network]:
    """Parse a comma-separated CIDR list, rejecting malformed entries loudly.

    Called once at import time so a typo in ``TRUSTED_CIDRS`` crashes the
    process at startup (visible as a CrashLoopBackOff) instead of silently
    degrading to "trust nothing" — or worse, being re-evaluated per request.
    An empty/whitespace-only value yields an empty list: trust no source IP.
    """
    networks: list[IPv4Network | IPv6Network] = []
    for token in raw.split(","):
        cidr = token.strip()
        if not cidr:
            continue
        try:
            networks.append(ip_network(cidr, strict=False))
        except ValueError as exc:
            raise ValueError(
                f"TRUSTED_CIDRS contains an invalid entry {cidr!r}: {exc}. "
                "Expected a comma-separated list of CIDRs "
                "(e.g. '127.0.0.1/32,::1/128'). "
                "Use TRUSTED_CIDRS='' to trust no source IP."
            ) from exc
    return networks


TRUSTED_NETWORKS: list[IPv4Network | IPv6Network] = _parse_trusted_cidrs(_TRUSTED_CIDRS_RAW)


def _is_trusted_network(ip: str | None) -> bool:
    """Return True if *ip* falls inside a configured ``TRUSTED_CIDRS`` network."""
    if not ip:
        return False
    try:
        addr = ip_address(ip)
    except ValueError:
        return False
    return any(addr in net for net in TRUSTED_NETWORKS)


class AuthMiddleware(BaseHTTPMiddleware):
    """Resolve user identity and attach to ``request.state``."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Always allow public paths
        if request.url.path in PUBLIC_PATHS or request.url.path.startswith(INTERNAL_PREFIX):
            return await call_next(request)

        # Auth disabled globally
        if not ENABLE_AUTH:
            request.state.user = User(
                sub="anonymous", preferred_username="anonymous", tenant_id=DEFAULT_TENANT_ID
            )
            request.state.tenant_id = DEFAULT_TENANT_ID
            return await call_next(request)

        client_ip = request.client.host if request.client else ""

        # Check Istio/gateway-forwarded headers FIRST.
        #
        # EITHER x-user-id OR x-tenant-id is a forwarded identity claim — the
        # same condition as ``coding_client.auth.middleware.AuthMiddleware``
        # (``if x_user_id or x_tenant_id:``) and its copy in coding-analysis.
        # Gating on x-user-id alone made this service strictly stricter than
        # every sibling: in-mesh callers send only ``X-Tenant-Id`` (see
        # coding-containers ``src/clients/_tenant.py``), so they used to be let
        # through by the RFC1918 source-IP bypass rather than by their header.
        # Closing that bypass therefore 401'd them. Accepting the header here
        # restores parity WITHOUT reopening the network-position hole: an
        # uncredentialed request still falls through to the 401 below.
        #
        # Note both are *claims*, not credentials — at the mesh edge the
        # ext_authz EnvoyFilter strips and re-injects them from a validated
        # license, but pod-to-pod they are whatever the caller typed. A forged
        # x-tenant-id/x-user-roles is cross-tenant admin with no credential (a
        # sibling coding-* pod read a fabricated tenant off coding-analysis this
        # way, 2026-08-03). That question is governed centrally by
        # ``TRUST_TENANT_HEADER`` in coding-client (>=1.9.0, default true =
        # today's behaviour): with the gate off these headers carry no identity,
        # so the request falls through to the trusted-network check and then the
        # 401 below. This is the same gate coding_client.auth.middleware /
        # .tenant apply; wiring it here keeps this local ingress fork from being
        # the one unlocked door once gitops flips the knob.
        trust_headers = trust_tenant_header()
        user_id = request.headers.get("x-user-id") if trust_headers else None
        tenant_header = request.headers.get("x-tenant-id") if trust_headers else None
        if user_id or tenant_header:
            roles_header = request.headers.get("x-user-roles", "")
            roles = [r.strip() for r in roles_header.split(",") if r.strip()]
            if not roles and _is_trusted_network(client_ip):
                roles = ["user"]  # Default role for trusted network
            # An empty header is treated as absent, never as a tenant named "".
            tenant_id = tenant_header or DEFAULT_TENANT_ID
            # Matches the SDK: a tenant claim with no user id is subject
            # "anonymous" — authenticated as a tenant, not as a person.
            sub = user_id or "anonymous"
            user = User(
                sub=sub,
                preferred_username=request.headers.get("x-user-name", sub),
                email=request.headers.get("x-user-email"),
                realm_access={"roles": roles},
                tenant_id=tenant_id,
            )
            request.state.user = user
            request.state.tenant_id = tenant_id
            return await call_next(request)

        # Licence Bearer — the ONLY credential that survives
        # TRUST_TENANT_HEADER=false, and the reason this arm exists.
        #
        # Until coding-client 1.10.0 this service had no Bearer path at all: the
        # API-key branch below is gated on ENVIRONMENT in ("development","test")
        # and so is dead in production, which left the forgeable X-Tenant-Id
        # header above as the only working production credential. Measured live
        # on the running pod 2026-08-12 against /api/tests/supported-tools: a
        # forged X-Tenant-Id + X-User-Roles: admin returned 200, while a valid
        # licence Bearer returned 401. Because the header arm above already
        # honours TRUST_TENANT_HEADER, flipping that flag would have closed the
        # forged arm and left NOTHING — a 401-everything switch rather than a
        # hardening. This arm is what makes the flip safe here.
        #
        # It delegates to the SDK's ``_user_from_license`` rather than
        # reimplementing introspection: that function owns the httpx call, the
        # 2s timeout, the fail-CLOSED error handling and the positive/negative
        # caches, and it must not fork again — a second copy of an auth decision
        # is how this service ended up without one. Placed after the header arm
        # and before the CIDR bypass, matching the SDK's own arm order.
        #
        # On-behalf-of. The comment that used to sit here said "there is no
        # on-behalf-of arm in coding-client: an asserted X-Tenant-Id is never
        # combined with a Bearer". That was true when it was written and became
        # FALSE in coding-client 1.11.0, which added exactly that arm — and a
        # stale comment about an auth boundary is worse than none, because the
        # next reader trusts it.
        #
        # This fork does not inherit AuthMiddleware.dispatch, so it does not get
        # the arm for free. Without it the tenant came from the introspection
        # body ONLY, which is not merely "less permissive" — it is WRONG for a
        # service acting for a customer. coding-ai calls this service on behalf
        # of a customer, presenting its own licence and naming the customer in
        # X-Tenant-Id; resolving from the body alone silently scoped that request
        # to svc-coding-ai's own tenant instead. Reads and writes filed under the
        # wrong tenant, with a 200 and no log line. Measured live: an asserted
        # tenant returned 404 (route missing) where every sibling returned 403,
        # which is what exposed the gap.
        #
        # Same decision function as the SDK's middleware and dependencies, so the
        # three cannot drift. A refusal is 403, not 401: the caller IS
        # authenticated, it is simply not authorised to speak for that tenant.
        licence_user = await _user_from_license(request)
        if licence_user is not None:
            effective_tenant, refused = resolve_tenant_assertion(
                request,
                subject_id=licence_user.sub,
                token_tenant=licence_user.tenant_id or "",
            )
            if refused:
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "TENANT_ASSERTION_NOT_PERMITTED",
                        "detail": (
                            "This credential may not assert a tenant other than the "
                            "one its license names."
                        ),
                    },
                )
            if effective_tenant != licence_user.tenant_id:
                licence_user = licence_user.model_copy(update={"tenant_id": effective_tenant})
                request.state.vouching_subject_id = licence_user.sub
                request.state.license_id = forwarded_license_id(request)
            request.state.user = licence_user
            # This service's routes read request.state.tenant_id (the SDK's own
            # middleware sets only request.state.user and leaves this to
            # TenantMiddleware/require_tenant, which this service does not run).
            request.state.tenant_id = licence_user.tenant_id
            return await call_next(request)

        # Trusted network bypass (no forwarded headers). Empty TRUSTED_CIDRS —
        # the default — means this branch is never taken.
        if _is_trusted_network(client_ip):
            request.state.user = User(
                sub="internal",
                preferred_username="internal",
                tenant_id=DEFAULT_TENANT_ID,
                realm_access={"roles": ["user"]},
            )
            request.state.tenant_id = DEFAULT_TENANT_ID
            return await call_next(request)

        # API key fallback (dev/test)
        if API_KEY and ENVIRONMENT in ("development", "test"):
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer ") and auth_header[7:] == API_KEY:
                request.state.user = User(
                    sub="api-key-user",
                    preferred_username="api-key-user",
                    realm_access={"roles": ["admin"]},
                    tenant_id=DEFAULT_TENANT_ID,
                )
                request.state.tenant_id = DEFAULT_TENANT_ID
                return await call_next(request)

        # Reject unauthenticated requests
        logger.warning(
            "Unauthenticated request rejected: %s %s from %s (no forwarded identity "
            "headers, source IP not in TRUSTED_CIDRS)",
            request.method,
            request.url.path,
            client_ip or "<unknown>",
        )
        return JSONResponse(
            status_code=401,
            content={"error": {"message": "Authentication required", "error_code": "UNAUTHORIZED"}},
            headers={"WWW-Authenticate": "Bearer"},
        )
