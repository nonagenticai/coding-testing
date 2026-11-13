"""Identity headers for outbound calls — re-exported from the coding_client SDK.

Mirrors the ``src/auth/tenant.py`` convention in this package: the canonical
implementation lives in the SDK so every coding-* service builds identical
headers, and this module exists only so in-repo imports read as local.

Why it is needed at all
-----------------------
Every sibling call this service makes today sends a bare ``X-Tenant-Id`` and
nothing else. That is not a credential — it is accepted only because the callees
still trust the header. As each callee flips to ``TRUST_TENANT_HEADER=false`` a
bare header proves nothing and the call 401s. ``service_token=`` sends this
service's OWN licence as ``Authorization: Bearer <token>`` **without touching
``X-Tenant-Id``**, so the callee reads "I am coding-testing, acting for customer
T" and honours the asserted tenant iff coding-testing's subject is in its
``CODING_ONBEHALF_SUBJECTS`` allowlist.

Keeping the two headers separate is the entire point. Folding the credential into
``X-Tenant-Id`` collapses every customer into coding-testing's own tenant,
because the callee resolves the tenant from whatever it is handed.

And the rule that predates all of this: **a missing tenant is an error, never a
default.** ``tenant_headers`` raises ``TenantMissingError`` (deliberately not a
``ValueError``/``RuntimeError`` subclass, so the broad ``except`` clauses that
wrap these call sites cannot absorb it); ``optional_tenant_headers`` omits the
header. Neither invents one. Substituting a literal makes the call succeed while
attributing it to a tenant that does not own the resource.
"""

import os

from coding_client.auth.outbound import (
    LICENSE_ID_HEADER,
    TENANT_ID_HEADER,
    TenantMissingError,
    optional_tenant_headers,
    tenant_headers,
)

__all__ = [
    "LICENSE_ID_HEADER",
    "TENANT_ID_HEADER",
    "TenantMissingError",
    "optional_tenant_headers",
    "service_token",
    "tenant_headers",
]


def service_token() -> str | None:
    """This service's own licence, read from the environment at CALL time.

    Never at import: the value is injected into the pod env, and binding it to a
    module-level constant would pin whatever happened to be set when the first
    import ran — which in tests is "nothing", and in production survives a secret
    rotation that should have taken effect immediately.

    ``or None`` normalises a set-but-empty variable to absent. The SDK already
    tests the token for truthiness, so this is belt-and-braces — but an empty
    credential is a malformed one, not "no credential", and a bare ``Bearer ``
    would still burn an introspection round-trip on the fleet's hottest auth path.
    """
    return os.getenv("CODING_SERVICE_TOKEN") or None
