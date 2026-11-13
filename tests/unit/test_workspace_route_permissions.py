"""Permissions on the py2rust routes.

Every NEW route that executes commands inside a workspace requires the same write
permission as ``/api/parity/{container_name}/schema-diff``
(``requires_write_permission``: realm role developer/admin, or a verified tenant
licence). Pure computation (inline ``/api/parity/openapi-diff``) and GET run
status stay at authenticated + tenant-owned.

The ``app`` fixture's default user is authenticated but holds NO realm role and no
licence, so it is exactly the caller the write gate must refuse.
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

import pytest
from fastapi import HTTPException

WRITE_ROUTES = [
    "/api/tests/workspace/ws1/runs",
    "/api/parity/ws1/openapi-diff",
    "/api/parity/ws1/schema-diff",
    "/api/parity/ws1/schema-diff/runs",
    "/api/parity/ws1/openapi-diff/runs",
    "/api/parity/ws1/record",
    "/api/parity/ws1/replay",
    "/api/parity/ws1/compare",
    "/api/parity/ws1/perf/runs",
]

_VALIDATORS = (
    "routers.workspace_test_runs.validate_and_get_container_for_tenant",
    "routers.parity.validate_and_get_container_for_tenant",
    "routers.parity_behaviour.validate_and_get_container_for_tenant",
    "routers.parity_schema.validate_and_get_container_for_tenant",
    "routers.parity_perf.validate_and_get_container_for_tenant",
)


async def _no_such_workspace(container_name: str, tenant_id: str | None = None) -> None:
    raise HTTPException(status_code=404, detail="not found")


@pytest.fixture(autouse=True)
def _workspace_lookup_is_404():
    """Never reach coding-containers: a request that passes auth stops at 404/422."""
    with ExitStack() as stack:
        for target in _VALIDATORS:
            stack.enter_context(patch(target, _no_such_workspace))
        yield


@pytest.mark.parametrize("path", WRITE_ROUTES)
async def test_write_routes_refuse_a_caller_without_write_permission(client, path):
    resp = await client.post(path, json={})
    assert resp.status_code == 403, resp.text


@pytest.mark.parametrize("path", WRITE_ROUTES)
async def test_write_routes_refuse_a_read_only_role(client, as_roles, path):
    as_roles("viewer", "user")
    resp = await client.post(path, json={})
    assert resp.status_code == 403, resp.text


@pytest.mark.parametrize("role", ["developer", "admin"])
@pytest.mark.parametrize("path", WRITE_ROUTES)
async def test_write_routes_admit_the_write_roles(client, as_roles, path, role):
    as_roles(role)
    resp = await client.post(path, json={})
    # Past the permission gate: body validation (422) or the workspace lookup (404).
    assert resp.status_code in (404, 422), resp.text


async def test_read_and_pure_computation_routes_need_no_write_permission(client):
    # Default user: authenticated, tenant-scoped, no roles.
    for path in ("/api/tests/workspace/ws1/runs/run-x", "/api/parity/ws1/runs/run-x"):
        resp = await client.get(path)
        assert resp.status_code != 403, (path, resp.text)
    resp = await client.post("/api/parity/openapi-diff", json={})
    assert resp.status_code != 403, resp.text
    assert resp.status_code == 422  # reached body validation, not the permission gate
