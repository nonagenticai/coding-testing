"""Tenant middleware + dependencies — re-exported from coding_client SDK.

This module is a thin re-export of ``coding_client.auth.tenant`` so the
canonical implementation lives in the SDK and all coding-* services share
identical tenant-extraction behaviour. The previous local copy was a near
byte-clone of the SDK module; the only divergence (a ``_license_introspect_url``
helper reading from pydantic settings) is removed because pydantic settings
reads ``LICENSE_INTROSPECT_URL`` from env at startup — the same env var the
SDK reads at call time — so behaviour is equivalent.
"""

from coding_client.auth.tenant import (
    DEFAULT_TENANT_ID,
    LICENSE_INTROSPECT_URL,
    TENANT_ID_HEADER,
    TRUSTED_NETWORKS,
    TenantMiddleware,
    _introspect_license,
    _is_trusted_network,
    _license_cache,
    check_tenant_access,
    get_current_user_with_tenant,
    get_tenant_id,
    require_same_tenant,
    require_tenant,
)

__all__ = [
    "DEFAULT_TENANT_ID",
    "LICENSE_INTROSPECT_URL",
    "TENANT_ID_HEADER",
    "TRUSTED_NETWORKS",
    "TenantMiddleware",
    "_introspect_license",
    "_is_trusted_network",
    "_license_cache",
    "check_tenant_access",
    "get_current_user_with_tenant",
    "get_tenant_id",
    "require_same_tenant",
    "require_tenant",
]
