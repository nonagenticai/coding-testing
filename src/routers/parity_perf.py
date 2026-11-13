"""Performance comparison: does the (Rust) target perform at least like its source(s)?

* ``POST /api/parity/{container_name}/perf/runs`` - start an async load run.
* ``GET  /api/parity/{container_name}/perf/runs/{run_id}`` - poll it.

Same corpora as behaviour parity; source(s) and target run sequentially in the
same workspace pod under identical concurrency levels and duration/request
budgets. Responses whose status differs from the expected one count as errors.
"""

from __future__ import annotations

import logging

from coding_client.auth.models import User
from fastapi import APIRouter, Depends, Path, status

from auth.tenant import get_current_user_with_tenant
from models.error_models import (
    Error401Response,
    Error403Response,
    Error500Response,
    ValidationErrorResponse,
)
from models.parity_perf_models import PerfRequest, PerfRun
from services.perf_parity import PerfRunner
from shared.container_validation import validate_and_get_container_for_tenant
from shared.dependencies import requires_write_permission

logger = logging.getLogger(__name__)
router = APIRouter()

_RESPONSES: dict[int | str, dict] = {
    401: {"description": "Authentication required", "model": Error401Response},
    403: {"description": "Insufficient permissions", "model": Error403Response},
    404: {"description": "Workspace or run not found"},
    422: {"description": "Validation error", "model": ValidationErrorResponse},
    500: {"description": "Internal error", "model": Error500Response},
    502: {"description": "The workspace could not run the driver: nothing was computed"},
}


def get_perf_runner() -> PerfRunner:
    return PerfRunner()


def _tenant(user: User) -> str | None:
    return None if user.is_admin() else user.effective_tenant_id


@router.post(
    "/{container_name}/perf/runs",
    response_model=PerfRun,
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="parity_perf_start",
    # Executes commands inside the workspace: same write permission as the other parity runs.
    dependencies=[Depends(requires_write_permission)],
    summary="Load-test source(s) and target on the same corpus; compare performance",
    description=(
        "Async (202 + `run_id`). Per corpus the source, then a fresh route-mapped target, "
        "each on a fresh database with stubbed upstreams: `warmup_rounds` sequential passes "
        "(the first is the correctness reference and teaches the stubs their responses), "
        "then per `concurrency` level `duration_seconds` (or `requests`) of round-robin "
        "load over the HTTP cases (`case_ids` narrows them; websocket/sse cases are "
        "skipped and listed). Reports per side, level and route: count, errors (a status "
        "other than the expected one is an error), throughput, latency p50/p95/p99/max, and "
        "the service's peak RSS (VmHWM; null when unknown). `comparisons` holds "
        "target/source ratios, `regressions` every metric beyond `thresholds`."
    ),
    responses={**_RESPONSES, 202: {"description": "Run accepted", "model": PerfRun}},
)
async def parity_perf_start(
    request: PerfRequest,
    container_name: str = Path(..., min_length=1, max_length=253),
    user: User = Depends(get_current_user_with_tenant),
    runner: PerfRunner = Depends(get_perf_runner),
) -> PerfRun:
    tenant_id = _tenant(user)
    await validate_and_get_container_for_tenant(container_name, tenant_id=tenant_id)
    run = await runner.start(container_name, tenant_id, request)
    logger.info("perf run %s on %s queued", run.run_id, container_name)
    return run


@router.get(
    "/{container_name}/perf/runs/{run_id}",
    response_model=PerfRun,
    operation_id="get_parity_perf_run",
    summary="Get an async performance-comparison run",
    description=(
        "While the job is queued/running `result` is null. Once it ended, `result.computed` "
        "false (status `failed`) means nothing was measured; otherwise `passed` is true "
        "only without regressions."
    ),
    responses=_RESPONSES,
)
async def get_parity_perf_run(
    container_name: str = Path(..., min_length=1, max_length=253),
    run_id: str = Path(..., min_length=32, max_length=97),
    user: User = Depends(get_current_user_with_tenant),
    runner: PerfRunner = Depends(get_perf_runner),
) -> PerfRun:
    tenant_id = _tenant(user)
    await validate_and_get_container_for_tenant(container_name, tenant_id=tenant_id)
    return await runner.poll(container_name, tenant_id, run_id)
