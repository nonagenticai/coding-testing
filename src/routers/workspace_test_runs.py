"""Workspace test runs: the checkout's own suite, language-dispatched, one result format.

A NEW route pair rather than new fields on ``POST /api/tests/execute``: that
route's request model (``TestRequest``) is shared with ``/api/tests/batch``
and its response model (``TestResult``) cannot carry a ``UnifiedTestResult`` or a
202 job handle without changing its schema. Two new operations keep every
existing contract byte-identical (gate G1, pinned by
``tests/unit/test_openapi_existing_routes_unchanged.py``).
"""

from __future__ import annotations

import logging

from coding_client.auth.models import User
from fastapi import APIRouter, Body, Depends, Path, Response, status

from auth.tenant import get_current_user_with_tenant
from models.error_models import (
    Error401Response,
    Error403Response,
    Error500Response,
    ValidationErrorResponse,
)
from models.unified_test_models import WorkspaceTestRun, WorkspaceTestRunRequest
from services.workspace_test_runner import WorkspaceTestRunner
from shared.container_validation import validate_and_get_container_for_tenant
from shared.dependencies import requires_write_permission

logger = logging.getLogger(__name__)
router = APIRouter()

_RESPONSES: dict[int | str, dict] = {
    401: {"description": "Authentication required", "model": Error401Response},
    403: {"description": "Insufficient permissions", "model": Error403Response},
    404: {"description": "Workspace or run not found"},
    422: {
        "description": "Validation error / no framework detected",
        "model": ValidationErrorResponse,
    },
    500: {"description": "Internal error", "model": Error500Response},
    502: {"description": "The workspace could not run the command: nothing was computed"},
}


def get_workspace_test_runner() -> WorkspaceTestRunner:
    return WorkspaceTestRunner()


def _tenant(user: User) -> str | None:
    return None if user.is_admin() else user.effective_tenant_id


@router.post(
    "/{container_name}/runs",
    response_model=WorkspaceTestRun,
    operation_id="run_workspace_tests",
    # Executes commands inside the workspace: same write permission as
    # /{container_name}/schema-diff (requires_write_permission).
    dependencies=[Depends(requires_write_permission)],
    summary="Run the workspace's own test suite (pytest, or a Rust runner Job)",
    description=(
        "Runs the checkout's tests and returns a framework-independent "
        "`UnifiedTestResult`. Language is detected from the workdir (pyproject.toml/"
        "uv.lock -> pytest via uv IN the workspace; Cargo.toml -> the working tree is "
        "packed and evaluated by a throwaway k8s Job from the Rust runner image with the "
        "requested `presets`: test, clippy, fmt, audit, deny, coverage) unless `language` "
        "is sent. python: `with_postgres` runs `ws-pg` first and exports its DATABASE_URL; "
        "`coverage: true` adds pytest-cov totals + per file in `result.coverage` (rust: "
        "the runner's `coverage` preset). "
        "`async: true` answers 202 with a `run_id`; poll "
        "`GET /api/tests/workspace/{container_name}/runs/{run_id}`. "
        "A run with zero executed tests is never `success`."
    ),
    responses={**_RESPONSES, 202: {"description": "Async run accepted", "model": WorkspaceTestRun}},
)
async def run_workspace_tests(
    response: Response,
    container_name: str = Path(..., min_length=1, max_length=253),
    request: WorkspaceTestRunRequest = Body(default_factory=WorkspaceTestRunRequest),
    user: User = Depends(get_current_user_with_tenant),
    runner: WorkspaceTestRunner = Depends(get_workspace_test_runner),
) -> WorkspaceTestRun:
    tenant_id = _tenant(user)
    await validate_and_get_container_for_tenant(container_name, tenant_id=tenant_id)
    run = await runner.run(container_name, tenant_id, request)
    if request.run_async:
        response.status_code = status.HTTP_202_ACCEPTED
    logger.info(
        "workspace test run %s on %s: status=%s success=%s",
        run.run_id,
        container_name,
        run.status,
        run.result.success if run.result else None,
    )
    return run


@router.get(
    "/{container_name}/runs/{run_id}",
    response_model=WorkspaceTestRun,
    operation_id="get_workspace_test_run",
    summary="Get an async workspace test run",
    description=(
        "Status of an async run. While the job is queued/running `result` is null; once "
        "it ended `result` holds the graded `UnifiedTestResult`."
    ),
    responses=_RESPONSES,
)
async def get_workspace_test_run(
    container_name: str = Path(..., min_length=1, max_length=253),
    run_id: str = Path(..., min_length=32, max_length=97),
    user: User = Depends(get_current_user_with_tenant),
    runner: WorkspaceTestRunner = Depends(get_workspace_test_runner),
) -> WorkspaceTestRun:
    tenant_id = _tenant(user)
    await validate_and_get_container_for_tenant(container_name, tenant_id=tenant_id)
    return await runner.poll(container_name, tenant_id, run_id)
