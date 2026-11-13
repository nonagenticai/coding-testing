"""Behaviour parity (py2rust A5c): does a (Rust) target BEHAVE like its source?

Additive routes under ``/api/parity`` (every existing route's contract is
unchanged, pinned by ``tests/unit/test_openapi_snapshot_a5c.py``):

* ``POST /api/parity/{container_name}/record``  - run the cases against the SOURCE.
* ``POST /api/parity/{container_name}/replay``  - run them against the TARGET and diff
  against the cases' ``expected_*`` fields or a recording.
* ``POST /api/parity/{container_name}/compare`` - source then target in one job,
  diff target vs source.
* ``GET  /api/parity/{container_name}/runs/{run_id}`` - poll an async run.

Each service runs on a fresh migrated database with its external upstreams
served by a stub that also records every outbound call; the diff covers status,
JSON body, selected headers and the outbound-call sequence.
"""

from __future__ import annotations

import logging

from coding_client.auth.models import User
from fastapi import APIRouter, Depends, Path, Response, status

from auth.tenant import get_current_user_with_tenant
from models.error_models import (
    Error401Response,
    Error403Response,
    Error500Response,
    ValidationErrorResponse,
)
from models.parity_behaviour_models import (
    BehaviourOp,
    BehaviourParityRun,
    CompareRequest,
    RecordRequest,
    ReplayRequest,
)
from services.behaviour_parity import BehaviourParityRunner
from shared.container_validation import validate_and_get_container_for_tenant
from shared.dependencies import requires_write_permission

logger = logging.getLogger(__name__)
router = APIRouter()

_RESPONSES: dict[int | str, dict] = {
    202: {"description": "Async run accepted", "model": BehaviourParityRun},
    401: {"description": "Authentication required", "model": Error401Response},
    403: {"description": "Insufficient permissions", "model": Error403Response},
    404: {"description": "Workspace or run not found"},
    422: {"description": "Validation error", "model": ValidationErrorResponse},
    500: {"description": "Internal error", "model": Error500Response},
    502: {"description": "The workspace could not run the driver: nothing was computed"},
}

_COMMON = (
    " Each side gets a FRESH database (`ws-pg createdb`), its `migrate` command, an "
    "upstream stub on loopback (every `upstream_env` var points at it; per case it "
    "serves the case's `upstream_responses` in order and records each outbound call) "
    "and is started with `start` on a free loopback port (`PORT`, `BIND_ADDR`, "
    "`{port}`), then the cases replay IN ORDER. Default `async: true` answers 202 "
    "with a `run_id`; poll `GET /api/parity/{container_name}/runs/{run_id}`. "
    "`computed` is false when any side failed to run; `equal` is never true then or "
    "for zero cases. N sources -> 1 target (replay/compare): send `corpora`, one per "
    "source (recording stays one source per record run), each with a `source_id`, its "
    "cases, a `route_map` (where its routes live on the target) and its baseline; each "
    "corpus is replayed against a FRESH target and every case outcome carries "
    "`source_id`, `method`/`path` and the mapped `target_method`/`target_path`. "
    "`target.request_headers` carries the target's own auth; `upstreams` gives every "
    "named upstream its own stub."
)


def get_behaviour_parity_runner() -> BehaviourParityRunner:
    return BehaviourParityRunner()


def _tenant(user: User) -> str | None:
    return None if user.is_admin() else user.effective_tenant_id


async def _start(
    op: BehaviourOp,
    request: RecordRequest | ReplayRequest | CompareRequest,
    response: Response,
    container_name: str,
    user: User,
    runner: BehaviourParityRunner,
) -> BehaviourParityRun:
    tenant_id = _tenant(user)
    await validate_and_get_container_for_tenant(container_name, tenant_id=tenant_id)
    run = await runner.start(op, container_name, tenant_id, request)
    if request.run_async:
        response.status_code = status.HTTP_202_ACCEPTED
    logger.info(
        "behaviour parity %s on %s: run=%s status=%s equal=%s",
        op,
        container_name,
        run.run_id,
        run.status,
        run.result.equal if run.result else None,
    )
    return run


@router.post(
    "/{container_name}/record",
    response_model=BehaviourParityRun,
    operation_id="parity_behaviour_record",
    # Executes commands inside the workspace: same write permission as
    # /{container_name}/schema-diff (requires_write_permission).
    dependencies=[Depends(requires_write_permission)],
    summary="Record the source service's behaviour over ordered cases",
    description=(
        "Runs the cases against `source` and returns a `recording` (per case: status, "
        "headers subset, JSON body, outbound upstream calls). When every case carries "
        "`expected_status`, the source is also diffed against the expectations." + _COMMON
    ),
    responses=_RESPONSES,
)
async def parity_behaviour_record(
    request: RecordRequest,
    response: Response,
    container_name: str = Path(..., min_length=1, max_length=253),
    user: User = Depends(get_current_user_with_tenant),
    runner: BehaviourParityRunner = Depends(get_behaviour_parity_runner),
) -> BehaviourParityRun:
    return await _start("record", request, response, container_name, user, runner)


@router.post(
    "/{container_name}/replay",
    response_model=BehaviourParityRun,
    operation_id="parity_behaviour_replay",
    # Executes commands inside the workspace: same write permission as
    # /{container_name}/schema-diff (requires_write_permission).
    dependencies=[Depends(requires_write_permission)],
    summary="Replay ordered cases against the target and diff against a baseline",
    description=(
        "Runs the cases against `target` and diffs per case against the cases' "
        "`expected_*` fields (`baseline: expected`) or a recording (`baseline: "
        "recording` with `recording` inline or `recording_run_id` of a record/compare "
        "run in the same workspace)." + _COMMON
    ),
    responses=_RESPONSES,
)
async def parity_behaviour_replay(
    request: ReplayRequest,
    response: Response,
    container_name: str = Path(..., min_length=1, max_length=253),
    user: User = Depends(get_current_user_with_tenant),
    runner: BehaviourParityRunner = Depends(get_behaviour_parity_runner),
) -> BehaviourParityRun:
    return await _start("replay", request, response, container_name, user, runner)


@router.post(
    "/{container_name}/compare",
    response_model=BehaviourParityRun,
    operation_id="parity_behaviour_compare",
    # Executes commands inside the workspace: same write permission as
    # /{container_name}/schema-diff (requires_write_permission).
    dependencies=[Depends(requires_write_permission)],
    summary="Record the source and replay the target in one job; diff target vs source",
    description=(
        "Two sequential driver passes in ONE job (source, then target, each on its own "
        "fresh database) and a per-case diff of target against source. The source "
        "recording is returned too." + _COMMON
    ),
    responses=_RESPONSES,
)
async def parity_behaviour_compare(
    request: CompareRequest,
    response: Response,
    container_name: str = Path(..., min_length=1, max_length=253),
    user: User = Depends(get_current_user_with_tenant),
    runner: BehaviourParityRunner = Depends(get_behaviour_parity_runner),
) -> BehaviourParityRun:
    return await _start("compare", request, response, container_name, user, runner)


@router.get(
    "/{container_name}/runs/{run_id}",
    response_model=BehaviourParityRun,
    operation_id="get_parity_behaviour_run",
    summary="Get an async behaviour-parity run",
    description=(
        "While the driver job is queued/running `result` is null. Once it ended the "
        "result holds the verdict; status `failed` means nothing was computed."
    ),
    responses={k: v for k, v in _RESPONSES.items() if k != 202},
)
async def get_parity_behaviour_run(
    container_name: str = Path(..., min_length=1, max_length=253),
    run_id: str = Path(..., min_length=32, max_length=97),
    user: User = Depends(get_current_user_with_tenant),
    runner: BehaviourParityRunner = Depends(get_behaviour_parity_runner),
) -> BehaviourParityRun:
    tenant_id = _tenant(user)
    await validate_and_get_container_for_tenant(container_name, tenant_id=tenant_id)
    return await runner.poll(container_name, tenant_id, run_id)
