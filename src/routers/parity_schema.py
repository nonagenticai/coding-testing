"""Schema parity routes (py2rust slice A5a).

* ``POST /api/parity/schema-diff`` — two DSNs reachable from this service
  (local/test use ONLY: allowed when PARITY_DSN_DIFF_ENABLED is set AND
  ENVIRONMENT is one of :data:`DSN_DIFF_ENVIRONMENTS`; refused otherwise).
* ``POST /api/parity/{container_name}/schema-diff`` — inside the workspace:
  two fresh pod-local DBs via ``ws-pg``, each side's migration command applied,
  both introspected in the pod, then diffed.
* ``POST /api/parity/{container_name}/schema-diff/runs`` +
  ``GET .../schema-diff/runs/{run_id}`` — the same as ONE async workspace job
  (202 + run id; no gateway request budget).

Every route takes either ONE source (``source`` / ``source_dsn``) or N
``sources`` whose union is compared to the single target (optional per-source
``mapping``: schema rename + table prefix; conflicting sources fail).

Both return :class:`SchemaDiffResult`; ``equal`` is only ever true when the
verdict was computed.
"""

from __future__ import annotations

import logging

from coding_client.auth.models import User
from fastapi import APIRouter, Depends, HTTPException, Path

from auth.tenant import get_current_user_with_tenant, require_tenant
from config.settings import get_settings
from models.error_models import (
    Error401Response,
    Error403Response,
    Error500Response,
    ValidationErrorResponse,
)
from models.parity_schema_models import (
    SchemaDiffDsnRequest,
    SchemaDiffResult,
    SchemaDiffRun,
    SchemaDiffWorkspaceRequest,
    SchemaDiffWorkspaceRunRequest,
    SchemaMapping,
)
from services.schema_parity import SnapshotError, SourceSnapshot, diff_union, introspect_dsn
from services.schema_parity_workspace import SchemaParityRuns, SchemaParityWorkspaceRunner
from shared.container_validation import validate_and_get_container_for_tenant
from shared.dependencies import requires_write_permission

logger = logging.getLogger(__name__)
router = APIRouter()


#: The only ENVIRONMENT values (case-insensitive) in which the DSN route may run.
#: An allowlist, not a "!= production" check: a typo, "Production", "prod",
#: "staging" or an unset value must all fail CLOSED, because this route makes
#: coding-testing open connections to caller-chosen hosts.
DSN_DIFF_ENVIRONMENTS = frozenset({"development", "dev", "test", "testing", "local", "ci"})

_COMMON_RESPONSES: dict[int | str, dict] = {
    401: {"description": "Authentication required", "model": Error401Response},
    403: {
        "description": "Insufficient permissions (or DSN route disabled in this environment)",
        "model": Error403Response,
    },
    422: {"description": "Validation error", "model": ValidationErrorResponse},
    500: {"description": "Internal error", "model": Error500Response},
    502: {"description": "A database or the workspace could not be reached: nothing computed"},
}
_WORKSPACE_RESPONSES: dict[int | str, dict] = {
    **_COMMON_RESPONSES,
    404: {"description": "Workspace not found"},
}


def dsn_diff_allowed(settings: object) -> bool:
    """True only with the explicit flag AND an allowlisted ENVIRONMENT."""
    env = str(getattr(settings, "ENVIRONMENT", "") or "").strip().lower()
    return bool(getattr(settings, "PARITY_DSN_DIFF_ENABLED", False)) and (
        env in DSN_DIFF_ENVIRONMENTS
    )


def get_schema_parity_runner() -> SchemaParityWorkspaceRunner:
    return SchemaParityWorkspaceRunner()


def get_schema_parity_runs() -> SchemaParityRuns:
    return SchemaParityRuns()


@router.post(
    "/schema-diff",
    response_model=SchemaDiffResult,
    operation_id="parity_schema_diff",
    summary="Diff the schemas of two Postgres databases (local/test use)",
    responses=_COMMON_RESPONSES,
)
async def schema_diff_dsn(
    request: SchemaDiffDsnRequest,
    current_user: User = Depends(requires_write_permission),
    tenant_id: str = Depends(require_tenant),
) -> SchemaDiffResult:
    """Introspect both DSNs (read-only transaction) and diff source vs target."""
    if not dsn_diff_allowed(get_settings()):
        raise HTTPException(
            status_code=403,
            detail=(
                "DSN schema-diff is a local/test tool (needs PARITY_DSN_DIFF_ENABLED and "
                "a development/test ENVIRONMENT); use "
                "POST /api/parity/{container_name}/schema-diff"
            ),
        )
    named: list[tuple[str, str, SchemaMapping | None]] = (
        [(s.name, s.dsn, s.mapping) for s in request.sources]
        if request.sources is not None
        else [("source", request.source_dsn or "", None)]
    )
    snaps: dict[str, dict] = {}
    for label, dsn in [*((n, d) for n, d, _ in named), ("target", request.target_dsn)]:
        side = "target" if label == "target" else ("source" if request.sources is None else label)
        what = "target" if label == "target" else "source"
        if not dsn.strip().startswith(("postgres://", "postgresql://", "postgresql+")):
            raise HTTPException(status_code=422, detail=f"{what}_dsn must be a postgres URL")
        stage = (
            f"introspect_{what}" if side in ("source", "target") else f"introspect_source:{side}"
        )
        try:
            snaps[label] = await introspect_dsn(dsn, request.options.schemas)
        except SnapshotError as exc:
            return SchemaDiffResult(
                computed=False,
                equal=False,
                error=f"{stage}: {exc}",
                stage=stage,
                options=request.options,
            )
        except Exception as exc:  # noqa: BLE001 - connect/query failure = not computed
            logger.warning("schema-diff %s introspection failed: %s", label, type(exc).__name__)
            raise HTTPException(
                status_code=502,
                detail=f"{side} database could not be introspected: {type(exc).__name__}",
            ) from exc
    return diff_union(
        [
            SourceSnapshot(
                n,
                snaps[n],
                m.schema_map if m else None,
                m.table_prefix if m else None,
            )
            for n, _, m in named
        ],
        snaps["target"],
        request.options,
    )


@router.post(
    "/{container_name}/schema-diff",
    response_model=SchemaDiffResult,
    operation_id="parity_workspace_schema_diff",
    summary="Migrate two fresh workspace DBs (source vs target) and diff their schemas",
    responses=_WORKSPACE_RESPONSES,
)
async def schema_diff_workspace(
    request: SchemaDiffWorkspaceRequest,
    container_name: str = Path(..., description="Workspace container name"),
    current_user: User = Depends(requires_write_permission),
    tenant_id: str = Depends(require_tenant),
    runner: SchemaParityWorkspaceRunner = Depends(get_schema_parity_runner),
) -> SchemaDiffResult:
    """Run both migration sets in the workspace and compute the schema verdict."""
    await validate_and_get_container_for_tenant(container_name, tenant_id)
    try:
        return await runner.run(container_name, request, tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


_RUN_DESCRIPTION = (
    "Async job mode of `POST /api/parity/{container_name}/schema-diff` (same body; "
    "`timeout` per command up to 3600 s, `job_timeout` for the whole job): every "
    "createdb / migrate / introspect step runs inside ONE workspace exec job, so cold "
    "migrations of N sources are not bound by the gateway's request budget. Answers "
    "202 with a `run_id`; poll `GET /api/parity/{container_name}/schema-diff/runs/"
    "{run_id}`. The run lives in coding-containers' job store + the workspace's /tmp."
)


@router.post(
    "/{container_name}/schema-diff/runs",
    response_model=SchemaDiffRun,
    status_code=202,
    operation_id="parity_workspace_schema_diff_start",
    summary="Start an async workspace schema diff (N sources vs target) as one job",
    description=_RUN_DESCRIPTION,
    responses={**_WORKSPACE_RESPONSES, 503: {"description": "Workspace job queue full"}},
)
async def schema_diff_workspace_start(
    request: SchemaDiffWorkspaceRunRequest,
    container_name: str = Path(..., min_length=1, max_length=253),
    current_user: User = Depends(requires_write_permission),
    tenant_id: str = Depends(require_tenant),
    runs: SchemaParityRuns = Depends(get_schema_parity_runs),
) -> SchemaDiffRun:
    await validate_and_get_container_for_tenant(container_name, tenant_id)
    try:
        return await runs.start(container_name, tenant_id, request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get(
    "/{container_name}/schema-diff/runs/{run_id}",
    response_model=SchemaDiffRun,
    operation_id="get_parity_workspace_schema_diff_run",
    summary="Get an async workspace schema-diff run",
    description=(
        "While the job is queued/running `result` is null. Once it ended `result` is "
        "the same `SchemaDiffResult` the synchronous route returns; status `failed` "
        "means nothing was computed (`result.stage` names the failing step)."
    ),
    responses=_WORKSPACE_RESPONSES,
)
async def get_schema_diff_workspace_run(
    container_name: str = Path(..., min_length=1, max_length=253),
    run_id: str = Path(..., min_length=32, max_length=97),
    current_user: User = Depends(get_current_user_with_tenant),
    tenant_id: str = Depends(require_tenant),
    runs: SchemaParityRuns = Depends(get_schema_parity_runs),
) -> SchemaDiffRun:
    await validate_and_get_container_for_tenant(container_name, tenant_id)
    return await runs.poll(container_name, tenant_id, run_id)
