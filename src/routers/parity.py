"""Contract parity: does a (Rust) target expose the same HTTP contract as its source(s)?

Two NEW operations (additive; every existing route's contract is unchanged,
pinned by ``tests/unit/test_openapi_snapshot_a5b.py``):

* ``POST /api/parity/openapi-diff`` - documents sent inline.
* ``POST /api/parity/{container_name}/openapi-diff`` - documents obtained inside
  a workspace by running the given commands / reading the given files.

Both compare ONE target against the UNION of N sources and return a computed
verdict (``equal``) plus route-level progress (``progress.matched/total``).
"""

from __future__ import annotations

import logging

from coding_client.auth.models import User
from fastapi import APIRouter, Depends, HTTPException, Path

from auth.tenant import get_current_user_with_tenant
from models.error_models import (
    Error401Response,
    Error403Response,
    Error500Response,
    ValidationErrorResponse,
)
from models.parity_models import (
    OpenApiDiffIgnore,
    OpenApiDiffRequest,
    OpenApiDiffResult,
    OpenApiDiffRun,
    WorkspaceOpenApiDiffRequest,
    WorkspaceOpenApiDiffResult,
    WorkspaceOpenApiDiffRunRequest,
)
from services.openapi_diff import IgnoreRules, OpenApiDiffError, SpecInput, openapi_diff
from services.workspace_openapi import (
    OpenApiDiffRuns,
    WorkspaceOpenApiFetcher,
    diff_workspace_docs,
    merged_rules,
    obtain_all,
)
from shared.container_validation import validate_and_get_container_for_tenant
from shared.dependencies import requires_write_permission

logger = logging.getLogger(__name__)
router = APIRouter()

_RESPONSES: dict[int | str, dict] = {
    401: {"description": "Authentication required", "model": Error401Response},
    403: {"description": "Insufficient permissions", "model": Error403Response},
    422: {
        "description": "Invalid documents / a document could not be obtained",
        "model": ValidationErrorResponse,
    },
    500: {"description": "Internal error", "model": Error500Response},
}


def get_workspace_openapi_fetcher() -> WorkspaceOpenApiFetcher:
    return WorkspaceOpenApiFetcher()


def get_openapi_diff_runs() -> OpenApiDiffRuns:
    return OpenApiDiffRuns()


def _tenant(user: User) -> str | None:
    return None if user.is_admin() else user.effective_tenant_id


def _rules(ignore: OpenApiDiffIgnore | None, parity_toml_text: str | None) -> IgnoreRules:
    return merged_rules(ignore, parity_toml_text)


def _names(explicit: list[str | None], prefix: str) -> list[str]:
    return [n or f"{prefix}[{i}]" for i, n in enumerate(explicit)]


@router.post(
    "/openapi-diff",
    response_model=OpenApiDiffResult,
    operation_id="parity_openapi_diff",
    summary="Diff a target OpenAPI document against the union of source documents",
    description=(
        "Normalises every document (local $ref resolved; titles/descriptions/examples/"
        "operationId ignored; 3.0 nullable == 3.1 type [x, null] == anyOf with null; "
        "integer formats ignored; path templates compared by shape) and compares the "
        "target against the union of the sources (each source's `path_prefix_map` "
        "applied first, so a moved prefix compares equal). `equal` is computed: no "
        "missing, extra or changed operation and no source conflict (`auth` decides "
        "whether security differences count). `progress` = matched / total source "
        "operations."
    ),
    responses=_RESPONSES,
)
async def parity_openapi_diff(
    request: OpenApiDiffRequest,
    user: User = Depends(get_current_user_with_tenant),
) -> OpenApiDiffResult:
    try:
        rules = _rules(request.ignore, request.parity_toml)
        names = _names([s.name for s in request.sources], "source")
        result = openapi_diff(
            [
                SpecInput(n, s.document, tuple(s.path_prefix_map.items()))
                for n, s in zip(names, request.sources, strict=True)
            ],
            SpecInput(request.target.name or "target", request.target.document),
            rules,
            strict_numeric_bounds=request.strict_numeric_bounds,
            auth=request.auth,
        )
    except OpenApiDiffError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return OpenApiDiffResult.model_validate(result)


@router.post(
    "/{container_name}/openapi-diff",
    response_model=WorkspaceOpenApiDiffResult,
    operation_id="parity_workspace_openapi_diff",
    # Executes commands inside the workspace: same write permission as
    # /{container_name}/schema-diff (requires_write_permission).
    dependencies=[Depends(requires_write_permission)],
    summary="Obtain OpenAPI documents in a workspace and diff target vs union of sources",
    description=(
        "Each source and the target is obtained inside the workspace: `command` is run "
        "in `workdir` (its stdout is the document unless `file` is set), `file` is read "
        "relative to `workdir`. Commands above 300 s run as an async exec job that is "
        "polled to completion. Documents are obtained concurrently (`concurrency`, "
        "default 4). A document that cannot be obtained completely (non-zero "
        "exit, clipped output, not JSON) is a 422, never a partial diff. Then the same "
        "comparison as `POST /api/parity/openapi-diff`. For many sources or cold builds "
        "use the async job mode `POST /api/parity/{container_name}/openapi-diff/runs`."
    ),
    responses={
        **_RESPONSES,
        404: {"description": "Workspace not found"},
        502: {"description": "The workspace could not run the command: nothing was computed"},
        504: {"description": "An async command did not finish within its timeout"},
    },
)
async def parity_workspace_openapi_diff(
    request: WorkspaceOpenApiDiffRequest,
    container_name: str = Path(..., min_length=1, max_length=253),
    user: User = Depends(get_current_user_with_tenant),
    fetcher: WorkspaceOpenApiFetcher = Depends(get_workspace_openapi_fetcher),
) -> WorkspaceOpenApiDiffResult:
    tenant_id = _tenant(user)
    await validate_and_get_container_for_tenant(container_name, tenant_id=tenant_id)
    sources, target, obtained = await obtain_all(fetcher, container_name, tenant_id, request)
    toml_text = None
    if request.parity_toml:
        toml_text = await fetcher.read_text(
            container_name, tenant_id, request.parity_toml, f"parity_toml {request.parity_toml}"
        )
    try:
        out = diff_workspace_docs(request, container_name, sources, target, obtained, toml_text)
    except OpenApiDiffError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    logger.info(
        "openapi-diff on %s: equal=%s progress=%s/%s",
        container_name,
        out.equal,
        out.progress.matched,
        out.progress.total,
    )
    return out


@router.post(
    "/{container_name}/openapi-diff/runs",
    response_model=OpenApiDiffRun,
    status_code=202,
    operation_id="parity_workspace_openapi_diff_start",
    # Executes commands inside the workspace: same write permission as the sync route.
    dependencies=[Depends(requires_write_permission)],
    summary="Start an async workspace openapi-diff (N sources vs target) as one job",
    description=(
        "Async job mode of `POST /api/parity/{container_name}/openapi-diff` (same body "
        "plus `job_timeout`): every document is obtained inside ONE workspace exec job, "
        "`concurrency` at a time, so N cold builds are not bound by the gateway's "
        "request budget. Answers 202 with a `run_id`; poll `GET /api/parity/"
        "{container_name}/openapi-diff/runs/{run_id}`: `result` is the same "
        "`WorkspaceOpenApiDiffResult` the sync route returns; status `failed` carries "
        "`error` (what the sync route would have answered as 422) and no result."
    ),
    responses={
        **_RESPONSES,
        404: {"description": "Workspace not found"},
        502: {"description": "The run could not be prepared or submitted"},
        503: {"description": "Workspace job queue full"},
    },
)
async def parity_workspace_openapi_diff_start(
    request: WorkspaceOpenApiDiffRunRequest,
    container_name: str = Path(..., min_length=1, max_length=253),
    user: User = Depends(get_current_user_with_tenant),
    runs: OpenApiDiffRuns = Depends(get_openapi_diff_runs),
) -> OpenApiDiffRun:
    tenant_id = _tenant(user)
    await validate_and_get_container_for_tenant(container_name, tenant_id=tenant_id)
    return await runs.start(container_name, tenant_id, request)


@router.get(
    "/{container_name}/openapi-diff/runs/{run_id}",
    response_model=OpenApiDiffRun,
    operation_id="get_parity_workspace_openapi_diff_run",
    summary="Get an async workspace openapi-diff run",
    description=(
        "While the job is queued/running `result` is null. `completed`: `result` holds "
        "the verdict. `failed`: `error` says why nothing was compared."
    ),
    responses={
        **_RESPONSES,
        404: {"description": "Workspace or run not found"},
        502: {"description": "Run status unavailable / the run's files are gone"},
    },
)
async def get_parity_workspace_openapi_diff_run(
    container_name: str = Path(..., min_length=1, max_length=253),
    run_id: str = Path(..., min_length=32, max_length=97),
    user: User = Depends(get_current_user_with_tenant),
    runs: OpenApiDiffRuns = Depends(get_openapi_diff_runs),
) -> OpenApiDiffRun:
    tenant_id = _tenant(user)
    await validate_and_get_container_for_tenant(container_name, tenant_id=tenant_id)
    return await runs.poll(container_name, tenant_id, run_id)
