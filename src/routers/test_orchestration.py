"""
V2 Test Orchestration Endpoints

Provides advanced test orchestration capabilities including test suite management,
distributed execution, and result aggregation across multiple containers.
"""

import asyncio
import contextlib
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid7

from coding_client.auth.models import User
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from auth.tenant import require_tenant
from config.settings import get_settings
from config.test_framework_config import TestConfig
from services.orchestration_models import (
    OrchestrationTestSuite,
    TestOrchestrationPlan,
)
from services.orchestration_store import get_orchestration_store
from services.stubs import (
    get_docker_service,
)
from services.test_execution.test_orchestrator import (
    OrchestratorType,
    TestExecutionRequest,
    TestExecutionResult,
    TestLanguage,
    TestOrchestrator,
    create_test_orchestrator,
)
from shared.container_validation import validate_and_get_container_for_tenant
from shared.dependencies import (
    requires_read_permission,
)
from shared.responses import (
    bad_request,
    forbidden,
    internal_error,
    not_found,
    success_response,
)

logger = logging.getLogger(__name__)
router = APIRouter()  # Tags are set at main API level as "V2 🎭 Test Orchestration"
_orchestrator: TestOrchestrator | None = None


async def get_orchestrator() -> TestOrchestrator:
    """Get or create test orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        docker_service = get_docker_service()
        config = TestConfig(timeout=300, parallel=True, max_workers=8)
        _orchestrator = create_test_orchestrator(docker_service, config)
        logger.info("Test orchestrator initialized for orchestration endpoints")
    return _orchestrator


# Domain classes now live in ``services.orchestration_models`` (so the Postgres
# store can rehydrate them without importing this router). Re-exported here for
# backwards compatibility with ``from routers.test_orchestration import ...``.
_store = get_orchestration_store()


def summarize_suite(suite: OrchestrationTestSuite) -> dict[str, Any]:
    """The ONE server-side verdict for a suite. Every caller must use this.

    Three things a suite can be, and the whole point is that they are three and
    not two:

    * ``passed``  - at least one result, all of them successful, suite completed.
    * ``failed``  - at least one result that genuinely ran and did not succeed.
    * ``not_run`` - nothing executed at all, or everything that "executed" was a
      NOT-RUN outcome (no test command for the ``(language, test_type)`` pair,
      or an executor reporting zero tests collected). A wrong PASS and a wrong
      FAIL are both lies; this is the honest answer for "we could not tell".

    ``passed`` requires ``not_run == 0`` **explicitly** rather than relying on
    those results having ``success=False``. That is deliberate belt-and-braces:
    the model-level invariant already guarantees it for anything constructed or
    rehydrated through validation, but this endpoint must not become a green
    light for a row that gets in some other way.

    ``failed`` is left INCLUSIVE of not-run results (``total - successful``) on
    purpose. Subtracting them would make ``failed == 0`` newly true for exactly
    the population that the not-run work turned red, i.e. it would re-open this
    bug for every caller that grades on ``failed``. A reader that wants the
    distinction reads ``not_run`` or ``verdict``; no existing reader is made
    more green by this function.
    """
    total = len(suite.results)
    successful = sum(1 for r in suite.results if r.success)
    not_run = sum(1 for r in suite.results if getattr(r, "not_run", False))
    failed = total - successful
    executed = total > 0
    completed = suite.status == "completed"
    passed = executed and failed == 0 and not_run == 0 and completed

    if not executed:
        verdict = "not_run"
    elif not completed:
        verdict = "incomplete"
    elif max(failed - not_run, 0) > 0:
        verdict = "failed"
    elif not_run > 0:
        verdict = "not_run"
    else:
        verdict = "passed"

    return {
        "total": total,
        "successful": successful,
        "failed": failed,
        # Results that verified nothing. Counted in `failed` as well (see above).
        "not_run": not_run,
        # None (not 0) when nothing ran: a rate over an empty suite is
        # undefined, and 0.0 was indistinguishable from "all failed".
        "success_rate": (successful / total * 100 if executed else None),
        "executed": executed,
        "status": suite.status,
        "passed": passed,
        "verdict": verdict,
    }


@router.post(
    "/suites",
    operation_id="create_test_suite",
    summary="Create test suite",
    responses={
        400: {"description": "Invalid test suite data or malformed test execution requests"},
        404: {"description": "Container not found"},
    },
)
async def create_test_suite(
    container_name: str,
    name: str,
    description: str,
    tests: list[TestExecutionRequest],
    metadata: dict[str, Any] | None = None,
    current_user: User = Depends(requires_read_permission),
    tenant_id: str = Depends(require_tenant),
) -> dict[str, Any]:
    """
    Create a new test suite with multiple tests scoped to a container.

    Args:
        container_name: Container name for test suite execution
        name: Suite name
        description: Suite description
        tests: List of test execution requests
        metadata: Optional metadata

    Returns:
        Created test suite information

    Raises:
        400: Invalid test suite data or malformed test execution requests
        404: Container not found
    """
    try:
        # Validate container exists
        container = await validate_and_get_container_for_tenant(container_name, tenant_id)
        logger.info(
            f"Creating test suite '{name}' for container '{container_name}' (status: {container.status.value})"
        )

        # Validate input parameters
        if not name or not name.strip():
            raise bad_request("Suite name cannot be empty")

        if not tests:
            raise bad_request("Test suite must contain at least one test")

        # Validate each test request and ensure container_name matches
        for i, test in enumerate(tests):
            if not hasattr(test, "language") or not test.language:
                raise bad_request(f"Test {i + 1} is missing required language parameter")
            # Ensure all tests use the same container_name
            if hasattr(test, "container_name") and test.container_name != container_name:
                raise bad_request(
                    f"Test {i + 1} has container_name '{test.container_name}' but suite requires '{container_name}'"
                )

        suite_id = str(uuid7())
        suite = OrchestrationTestSuite(suite_id, name, description, container_name, tenant_id)
        suite.tests = tests
        if metadata:
            suite.metadata = metadata
        await _store.save_suite(suite)
        logger.info(
            f"Test suite '{name}' created by {current_user.preferred_username} for container '{container_name}' (tenant: {tenant_id}) with {len(tests)} tests"
        )
        return {
            "suite_id": suite_id,
            "name": name,
            "description": description,
            "container_name": container_name,
            "tenant_id": tenant_id,
            "test_count": len(tests),
            "languages": list(set(test.language.value for test in tests)),
            "created_at": suite.created_at.isoformat(),
            "status": suite.status,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create test suite: {str(e)}")
        raise internal_error(f"Failed to create test suite: {str(e)}")


@router.get(
    "/suites",
    operation_id="list_test_suites",
    summary="List test suites",
    responses={404: {"description": "No test suites found matching the specified filters"}},
)
async def list_test_suites(
    container_name: str | None = Query(None, description="Filter by container name"),
    status_filter: str | None = Query(None, description="Filter by status"),
    language_filter: str | None = Query(None, description="Filter by language"),
    current_user: User = Depends(requires_read_permission),
    tenant_id: str = Depends(require_tenant),
) -> dict[str, Any]:
    """
    List all test suites with optional filters, including container scoping.

    Args:
        container_name: Filter by container name (recommended for container isolation)
        status_filter: Filter by suite status
        language_filter: Filter by programming language

    Returns:
        List of test suites

    Raises:
        404: No test suites found matching the specified filters
    """
    try:
        # If container_name is provided, validate it exists
        if container_name:
            await validate_and_get_container_for_tenant(container_name, tenant_id)
            logger.debug(f"Listing test suites for container '{container_name}'")

        suites = []
        # Tenant isolation is enforced in SQL by the store.
        for suite in await _store.list_suites(tenant_id):
            # Filter by container name
            if container_name and suite.container_name != container_name:
                continue
            if status_filter and suite.status != status_filter:
                continue
            suite_languages = set(test.language.value for test in suite.tests)
            if language_filter and language_filter not in suite_languages:
                continue
            suites.append(
                {
                    "suite_id": suite.suite_id,
                    "name": suite.name,
                    "description": suite.description,
                    "container_name": suite.container_name,
                    "tenant_id": suite.tenant_id,
                    "test_count": len(suite.tests),
                    "languages": list(suite_languages),
                    "status": suite.status,
                    "created_at": suite.created_at.isoformat(),
                }
            )

        # Don't raise 404 if no suites found, just return empty list

        return success_response(data={"suites": suites, "total": len(suites)})
    except HTTPException:
        raise
    except Exception as e:
        # A failure here (e.g. the backing store is unreachable) is a genuine
        # server error, not "no suites" — surface it rather than masking a broken
        # backend as an empty 200.
        logger.error(f"Failed to list test suites: {str(e)}")
        raise internal_error("Failed to list test suites")


@router.post(
    "/suites/{suite_id}/execute",
    operation_id="execute_test_suite",
    summary="Execute test suite",
    responses={
        400: {"description": "Invalid execution parameters or suite is already running"},
        404: {"description": "Test suite or container not found"},
    },
)
async def execute_test_suite(
    suite_id: str,
    background_tasks: BackgroundTasks,
    parallel: bool = Query(True, description="Execute tests in parallel"),
    max_workers: int = Query(4, ge=1, le=16, description="Maximum parallel workers"),
    current_user: User = Depends(requires_read_permission),
    tenant_id: str = Depends(require_tenant),
) -> dict[str, Any]:
    """
    Execute all tests in a suite with container validation.

    Args:
        suite_id: Test suite ID
        parallel: Whether to run tests in parallel
        max_workers: Maximum parallel executions

    Returns:
        Execution status and initial results

    Raises:
        400: Invalid execution parameters or suite is already running
        404: Test suite or container not found
    """
    try:
        suite = await _store.get_suite(suite_id)
        if suite is None:
            raise not_found("Test suite", suite_id)

        # Validate tenant isolation
        if suite.tenant_id != tenant_id:
            raise forbidden("Access denied: test suite belongs to a different tenant")

        # Validate container still exists before execution
        container = await validate_and_get_container_for_tenant(suite.container_name, tenant_id)
        logger.info(
            f"Executing test suite '{suite.name}' for container '{suite.container_name}' (status: {container.status.value})"
        )

        # Validate execution parameters (belt-and-suspenders for direct/coroutine
        # callers — the HTTP layer's Query(ge=1, le=16) 422s before reaching here).
        if max_workers < 1 or max_workers > 16:
            raise bad_request("max_workers must be between 1 and 16")

        if not suite.tests:
            raise bad_request("Cannot execute empty test suite")

        # Build the orchestrator BEFORE claiming the suite, so a failure here can't
        # leave the row wedged in a persisted "running" state (the CAS flag now
        # survives restarts — there is no in-memory reset).
        orchestrator = await get_orchestrator()

        # Atomically claim the suite (pending/completed/… -> running). Only the
        # caller that wins the compare-and-swap schedules execution; a concurrent
        # request (this replica or another) sees it already running -> 400. This
        # replaces the old process-local check-then-set, which races across replicas.
        if not await _store.try_mark_suite_running(suite_id, tenant_id):
            raise bad_request(f"Test suite '{suite_id}' is already running")
        suite.status = "running"
        background_tasks.add_task(
            execute_suite_async,
            suite=suite,
            orchestrator=orchestrator,
            parallel=parallel,
            max_workers=max_workers,
            user_id=current_user.preferred_username,
        )
        logger.info(
            f"Test suite '{suite.name}' execution started by {current_user.preferred_username}"
        )
        return {
            "suite_id": suite_id,
            "status": "started",
            "message": f"Test suite execution started with {len(suite.tests)} tests",
            "parallel": parallel,
            "max_workers": max_workers,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute test suite: {str(e)}")
        raise internal_error(f"Failed to execute test suite: {str(e)}")


@router.get(
    "/suites/{suite_id}/results",
    operation_id="get_test_suite_results",
    summary="Get suite results",
    responses={
        200: {"description": "Suite results retrieved successfully"},
        404: {"description": "Test suite not found"},
        500: {"description": "Failed to get suite results"},
    },
)
async def get_suite_results(
    suite_id: str,
    current_user: User = Depends(requires_read_permission),
    tenant_id: str = Depends(require_tenant),
) -> dict[str, Any]:
    """
    Get execution results for a test suite with container context.

    Args:
        suite_id: Test suite ID

    Returns:
        Suite execution results including container information
    """
    try:
        suite = await _store.get_suite(suite_id)
        if suite is None:
            raise not_found("Test suite", suite_id)

        # Validate tenant isolation
        if suite.tenant_id != tenant_id:
            raise forbidden("Access denied: test suite belongs to a different tenant")

        logger.debug(
            f"Retrieving results for test suite '{suite.name}' in container '{suite.container_name}'"
        )

        summary = summarize_suite(suite)
        passed = summary["passed"]
        results_by_language: dict[str, list[Any]] = defaultdict(list)
        for result in suite.results:
            results_by_language[result.language.value].append(
                {
                    "test_id": result.test_id,
                    "success": result.success,
                    "execution_time_ms": result.execution_time_ms,
                    "test_type": result.test_type.value,
                }
            )
        return {
            "suite_id": suite_id,
            "name": suite.name,
            "container_name": suite.container_name,
            "status": suite.status,
            # Server-side verdict, computed by `summarize_suite`. `passed` is
            # the only field that may be read as "the tests passed";
            # `failed == 0` on its own must not be. `summary.verdict` carries
            # the honest third value for a suite that simply never ran.
            "passed": passed,
            "summary": summary,
            "results_by_language": dict(results_by_language),
            "execution_details": [
                {
                    "test_id": r.test_id,
                    "language": r.language.value,
                    "test_type": r.test_type.value,
                    "success": r.success,
                    "exit_code": r.exit_code,
                    "execution_time_ms": r.execution_time_ms,
                    "container_name": r.container_name,
                    # Surfaces "this test never executed" separately from
                    # "this test failed" (e.g. no test command is defined).
                    "not_run": getattr(r, "not_run", False),
                    "not_run_reason": getattr(r, "not_run_reason", None),
                }
                for r in suite.results
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get suite results: {str(e)}")
        raise internal_error(f"Failed to get suite results: {str(e)}")


@router.post(
    "/plans",
    operation_id="create_orchestration_plan",
    summary="Create orchestration plan",
    responses={
        200: {"description": "Orchestration plan created successfully"},
        400: {"description": "Invalid plan data or missing test suites"},
        404: {"description": "One or more test suites or container not found"},
        500: {"description": "Failed to create orchestration plan"},
    },
)
async def create_orchestration_plan(
    container_name: str,
    suite_ids: list[str],
    dependencies: dict[str, list[str]] | None = None,
    current_user: User = Depends(requires_read_permission),
    tenant_id: str = Depends(require_tenant),
) -> dict[str, Any]:
    """
    Create a test orchestration plan with dependencies scoped to a container.

    Args:
        container_name: Container name for orchestration plan execution
        suite_ids: List of test suite IDs (must all belong to same container)
        dependencies: Optional dependency map (suite_id -> [dependent_suite_ids])

    Returns:
        Created orchestration plan

    Raises:
        400: Invalid plan data, missing test suites, or suites from different containers
        404: Container or test suites not found
    """
    try:
        # Validate container exists
        container = await validate_and_get_container_for_tenant(container_name, tenant_id)
        logger.info(
            f"Creating orchestration plan for container '{container_name}' (status: {container.status.value})"
        )

        # Validate input parameters
        if not suite_ids:
            raise bad_request("At least one test suite ID is required")

        # Check for duplicate suite IDs
        if len(suite_ids) != len(set(suite_ids)):
            raise bad_request("Duplicate suite IDs are not allowed")

        # Validate all suites exist, belong to the same container, and same tenant
        validated_suites: dict[str, OrchestrationTestSuite] = {}
        for suite_id in suite_ids:
            suite = await _store.get_suite(suite_id)
            if suite is None:
                raise not_found("Test suite", suite_id)
            # Validate tenant isolation
            if suite.tenant_id != tenant_id:
                raise forbidden(
                    f"Access denied: test suite '{suite_id}' belongs to a different tenant"
                )
            if suite.container_name != container_name:
                raise bad_request(
                    f"Test suite '{suite_id}' belongs to container '{suite.container_name}' but plan requires '{container_name}'"
                )
            validated_suites[suite_id] = suite

        plan_id = str(uuid7())
        plan = TestOrchestrationPlan(plan_id, container_name, tenant_id)
        for suite_id in suite_ids:
            plan.suites.append(validated_suites[suite_id])
        if dependencies:
            for suite_id, deps in dependencies.items():
                plan.dependencies[suite_id] = set(deps)
        plan.execution_order = _calculate_execution_order(suite_ids, plan.dependencies)
        plan.parallel_groups = _calculate_parallel_groups(plan.execution_order, plan.dependencies)
        await _store.save_plan(plan)
        logger.info(
            f"Orchestration plan created by {current_user.preferred_username} for container '{container_name}' (tenant: {tenant_id}) with {len(suite_ids)} suites"
        )
        return {
            "plan_id": plan_id,
            "container_name": container_name,
            "tenant_id": tenant_id,
            "suite_count": len(plan.suites),
            "execution_order": plan.execution_order,
            "parallel_groups": plan.parallel_groups,
            "created_at": plan.created_at.isoformat(),
            "status": plan.status,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create orchestration plan: {str(e)}")
        raise internal_error(f"Failed to create orchestration plan: {str(e)}")


@router.post(
    "/plans/{plan_id}/execute",
    operation_id="execute_orchestration_plan",
    summary="Execute orchestration plan",
    responses={
        200: {"description": "Plan execution started successfully"},
        400: {"description": "Invalid plan state or execution parameters"},
        404: {"description": "Orchestration plan or container not found"},
        500: {"description": "Failed to execute orchestration plan"},
    },
)
async def execute_orchestration_plan(
    plan_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(requires_read_permission),
    tenant_id: str = Depends(require_tenant),
) -> dict[str, Any]:
    """
    Execute a test orchestration plan with container validation.

    Args:
        plan_id: Orchestration plan ID

    Returns:
        Execution status
    """
    try:
        plan = await _store.get_plan(plan_id)
        if plan is None:
            raise not_found("Orchestration plan", plan_id)

        # Validate tenant isolation
        if plan.tenant_id != tenant_id:
            raise forbidden("Access denied: orchestration plan belongs to a different tenant")

        # Validate container still exists before execution
        container = await validate_and_get_container_for_tenant(plan.container_name, tenant_id)
        logger.info(
            f"Executing orchestration plan for container '{plan.container_name}' (status: {container.status.value})"
        )

        if not plan.suites:
            raise bad_request("Cannot execute plan with no test suites")

        # Build the orchestrator before claiming the plan (see execute_test_suite).
        orchestrator = await get_orchestrator()

        # Atomically claim the plan (CAS, mirrors the suite guard) so concurrent /
        # cross-replica executes cannot both schedule.
        if not await _store.try_mark_plan_executing(plan_id, tenant_id):
            raise bad_request(f"Orchestration plan '{plan_id}' is already executing")
        plan.status = "executing"
        background_tasks.add_task(
            execute_plan_async,
            plan=plan,
            orchestrator=orchestrator,
            user_id=current_user.preferred_username,
        )
        logger.info(f"Orchestration plan execution started by {current_user.preferred_username}")
        return {
            "plan_id": plan_id,
            "status": "started",
            "message": f"Orchestration plan execution started with {len(plan.suites)} suites",
            "parallel_groups": len(plan.parallel_groups),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute orchestration plan: {str(e)}")
        raise internal_error(f"Failed to execute orchestration plan: {str(e)}")


@router.get(
    "/execution-history",
    operation_id="list_orchestration_history",
    summary="Get execution history",
    responses={
        200: {"description": "Execution history retrieved successfully"},
        404: {"description": "No execution history found"},
        500: {"description": "Failed to get execution history"},
    },
)
async def get_execution_history(
    container_name: str | None = Query(None, description="Filter by container name"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    since: datetime | None = Query(None, description="Filter by date"),
    current_user: User = Depends(requires_read_permission),
    tenant_id: str = Depends(require_tenant),
) -> dict[str, Any]:
    """
    Get test execution history with optional container filtering.

    Args:
        container_name: Filter by container name (recommended for container isolation)
        limit: Maximum number of results
        since: Filter executions since this date

    Returns:
        Execution history
    """
    try:
        # If container_name is provided, validate it exists
        if container_name:
            await validate_and_get_container_for_tenant(container_name, tenant_id)
            logger.debug(f"Retrieving execution history for container '{container_name}'")

        # Normalize a client-supplied naive `since` to UTC-aware; stored timestamps
        # are always aware, and comparing naive >= aware raises TypeError.
        if since is not None and since.tzinfo is None:
            since = since.replace(tzinfo=UTC)

        # Tenant isolation is enforced in SQL by the store.
        history = await _store.list_history(tenant_id)
        # Filter by container name
        if container_name:
            history = [h for h in history if h.get("container_name") == container_name]
        if since:
            history = [h for h in history if h["timestamp"] >= since]
        # `total` reflects the same filters (minus the page limit).
        total = len(history)
        history = sorted(history, key=lambda x: x["timestamp"], reverse=True)[:limit]

        # Don't raise 404 if no history found, just return empty list

        return success_response(
            data={
                "executions": history,
                "count": len(history),
                "total": total,
                "filtered_by_container": container_name,
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        # A store failure is a real 500, not "no history".
        logger.error(f"Failed to get execution history: {str(e)}")
        raise internal_error("Failed to get execution history")


@router.get(
    "/metrics/summary",
    operation_id="get_orchestration_metrics_summary",
    summary="Get orchestration metrics",
    responses={
        200: {"description": "Orchestration metrics retrieved successfully"},
        404: {"description": "No metrics found for specified period"},
        500: {"description": "Failed to get orchestration metrics"},
    },
)
async def get_orchestration_metrics(
    container_name: str | None = Query(None, description="Filter by container name"),
    days: int = Query(7, ge=1, le=90, description="Number of days"),
    current_user: User = Depends(requires_read_permission),
    tenant_id: str = Depends(require_tenant),
) -> dict[str, Any]:
    """
    Get test orchestration metrics summary with optional container filtering.

    Args:
        container_name: Filter by container name (recommended for container isolation)
        days: Number of days to include

    Returns:
        Orchestration metrics
    """
    try:
        # If container_name is provided, validate it exists
        if container_name:
            await validate_and_get_container_for_tenant(container_name, tenant_id)
            logger.debug(f"Retrieving orchestration metrics for container '{container_name}'")

        since = datetime.now(UTC) - timedelta(days=days)
        # Tenant isolation is enforced in SQL by the store.
        recent_history = [
            h for h in await _store.list_history(tenant_id) if h["timestamp"] >= since
        ]

        # Filter by container name
        if container_name:
            recent_history = [
                h for h in recent_history if h.get("container_name") == container_name
            ]

        total_executions = len(recent_history)
        suite_executions = [h for h in recent_history if h["type"] == "suite"]
        plan_executions = [h for h in recent_history if h["type"] == "plan"]
        language_counts: dict[str, int] = defaultdict(int)
        for history in recent_history:
            for lang in history.get("languages", []):
                language_counts[lang] += 1
        successful_executions = [h for h in recent_history if h.get("success", False)]
        success_rate = (
            len(successful_executions) / total_executions * 100 if total_executions > 0 else 0
        )

        # Count active suites and plans for the tenant (and optionally container).
        # Tenant scoping is enforced in SQL by the store.
        active_suites_count = len(
            [
                s
                for s in await _store.list_suites(tenant_id)
                if not container_name or s.container_name == container_name
            ]
        )
        active_plans_count = len(
            [
                p
                for p in await _store.list_plans(tenant_id)
                if not container_name or p.container_name == container_name
            ]
        )

        # Don't raise 404 if no metrics found, just return zero metrics

        return success_response(
            data={
                "container_name": container_name,
                "period_days": days,
                "total_executions": total_executions,
                "suite_executions": len(suite_executions),
                "plan_executions": len(plan_executions),
                "success_rate": round(success_rate, 2),
                "language_distribution": dict(language_counts),
                "active_suites": active_suites_count,
                "active_plans": active_plans_count,
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        # A store failure is a real 500, not "zero metrics".
        logger.error(f"Failed to get orchestration metrics: {str(e)}")
        raise internal_error("Failed to get orchestration metrics")


def _calculate_execution_order(
    suite_ids: list[str], dependencies: dict[str, set[str]]
) -> list[str]:
    """Calculate topological order for suite execution."""
    visited = set()
    order = []

    def visit(suite_id: str) -> None:
        if suite_id in visited:
            return
        visited.add(suite_id)
        for dep in dependencies.get(suite_id, []):
            if dep in suite_ids:
                visit(dep)
        order.append(suite_id)

    for suite_id in suite_ids:
        visit(suite_id)
    return order


def _calculate_parallel_groups(
    execution_order: list[str], dependencies: dict[str, set[str]]
) -> list[list[str]]:
    """Calculate which suites can run in parallel."""
    groups = []
    remaining: set[str] = set(execution_order)
    completed: set[str] = set()
    while remaining:
        current_group = []
        for suite_id in remaining:
            deps = dependencies.get(suite_id, set())
            if deps.issubset(completed):
                current_group.append(suite_id)
        if not current_group:
            current_group = list(remaining)
        groups.append(current_group)
        completed.update(current_group)
        remaining.difference_update(current_group)
    return groups


def _lease_heartbeat_interval() -> int:
    """Seconds between lease renewals for a running suite/plan.

    A fraction of the stale-reclaim threshold so a live run touches its row
    several times before it could ever be considered stale (default 3600 → 900s).
    """
    stale = get_settings().ORCHESTRATION_STALE_RUNNING_SECONDS
    return max(15, stale // 4)


async def _renew_lease(touch: Callable[[], Awaitable[bool]], interval: int) -> None:
    """Periodically renew a running suite/plan's lease so the stale-flag reaper
    never reclaims a legitimately long-running (but alive) execution.

    Runs until cancelled (in the executor's ``finally``). Best-effort: a transient
    DB error is logged and retried on the next tick, never propagated — a failed
    heartbeat must not crash the execution it is guarding.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            await touch()
        except Exception:
            logger.warning("lease heartbeat failed; will retry next tick", exc_info=True)


async def execute_suite_async(
    suite: OrchestrationTestSuite,
    orchestrator: TestOrchestrator,
    parallel: bool,
    max_workers: int,
    user_id: str,
) -> None:
    """Execute test suite asynchronously."""
    heartbeat = asyncio.create_task(
        _renew_lease(lambda: _store.heartbeat_suite(suite.suite_id), _lease_heartbeat_interval())
    )
    try:
        logger.info(f"Starting execution of suite '{suite.name}' with {len(suite.tests)} tests")
        suite.results = []
        if parallel:
            for i in range(0, len(suite.tests), max_workers):
                chunk = suite.tests[i : i + max_workers]
                tasks = [orchestrator.execute_tests(test) for test in chunk]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        error_result = TestExecutionResult(
                            test_id=str(uuid7()),
                            success=False,
                            exit_code=1,
                            stdout="",
                            stderr=str(result),
                            execution_time_ms=0.0,
                            language=TestLanguage.PYTHON,
                            test_type=OrchestratorType.UNIT,
                            container_name=suite.container_name,
                            created_at=datetime.now(UTC),
                        )
                        suite.results.append(error_result)
                    elif isinstance(result, TestExecutionResult):
                        suite.results.append(result)
        else:
            for test in suite.tests:
                try:
                    result = await orchestrator.execute_tests(test)
                    suite.results.append(result)
                except Exception as e:
                    error_result = TestExecutionResult(
                        test_id=str(uuid7()),
                        success=False,
                        exit_code=1,
                        stdout="",
                        stderr=str(e),
                        execution_time_ms=0.0,
                        language=test.language,
                        test_type=test.test_type,
                        container_name=suite.container_name,
                        created_at=datetime.now(UTC),
                    )
                    suite.results.append(error_result)
        suite.status = "completed"
        # Merge-save ONLY the execution-owned fields (results + status) into the
        # live row. The `suite` object is the snapshot captured at schedule time;
        # over a multi-minute run a concurrent writer may have changed another
        # field, so a full save_suite rewrite would silently revert it (lost
        # update). save_suite_execution_result touches only results+status.
        await _store.save_suite_execution_result(
            suite.suite_id, results=suite.results, status=suite.status
        )
        # Second reader of the same question -> same authority. This used to be
        # `all(r.success for r in suite.results)`, which is a SECOND verdict
        # implementation that disagreed with the results endpoint: it ignored
        # `not_run` entirely and `all([])` is True, so a suite with no results
        # at all recorded `success: True` in history. (The empty case is not
        # reachable through the HTTP path - `create_test_suite` and
        # `execute_test_suite` both reject an empty suite, and the executor
        # appends one result per test - so this is not a live defect being
        # closed; it is a divergent copy being deleted.)
        history_summary = summarize_suite(suite)
        await _store.append_history(
            {
                "type": "suite",
                "suite_id": suite.suite_id,
                "suite_name": suite.name,
                "container_name": suite.container_name,
                "tenant_id": suite.tenant_id,
                "timestamp": datetime.now(UTC),
                "user_id": user_id,
                "test_count": len(suite.tests),
                "successful": history_summary["successful"],
                "languages": list(set(test.language.value for test in suite.tests)),
                "success": history_summary["passed"],
                "verdict": history_summary["verdict"],
                "not_run": history_summary["not_run"],
            }
        )
        logger.info(
            f"Suite '{suite.name}' execution completed for container '{suite.container_name}' (tenant: {suite.tenant_id})"
        )
    except Exception as e:
        logger.error(f"Suite execution failed: {str(e)}")
        suite.status = "failed"
        # Best-effort: never let a persistence error in the failure handler crash
        # the background task or mask the original error. Merge-save only the
        # execution-owned fields (see the terminal save above) so the stale
        # snapshot can't revert a concurrent writer's change.
        try:
            await _store.save_suite_execution_result(
                suite.suite_id, results=suite.results, status=suite.status
            )
        except Exception:
            logger.exception("Failed to persist failed status for suite %s", suite.suite_id)
    finally:
        heartbeat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat


async def execute_plan_async(
    plan: TestOrchestrationPlan, orchestrator: TestOrchestrator, user_id: str
) -> None:
    """Execute orchestration plan asynchronously."""
    heartbeat = asyncio.create_task(
        _renew_lease(lambda: _store.heartbeat_plan(plan.plan_id), _lease_heartbeat_interval())
    )
    try:
        logger.info(f"Starting execution of plan with {len(plan.suites)} suites")
        # ``parallel_groups`` stores suite *ids* verbatim, but ``plan.suites`` only
        # holds suites whose rows still exist — ``plan_from_data`` drops any that
        # vanished (a suite may be deleted out-of-band after the plan was built).
        # Look up by id and skip a missing suite so the remaining ones still run;
        # ``next(...)`` here would raise ``StopIteration`` and fail the whole plan.
        suites_by_id = {s.suite_id: s for s in plan.suites}
        for group_idx, group in enumerate(plan.parallel_groups):
            logger.info(
                f"Executing parallel group {group_idx + 1}/{len(plan.parallel_groups)} with {len(group)} suites"
            )
            tasks = []
            for suite_id in group:
                suite = suites_by_id.get(suite_id)
                if suite is None:
                    logger.warning(
                        "Plan %s references missing suite %s (deleted mid-run); "
                        "skipping it and continuing with the rest of the plan",
                        plan.plan_id,
                        suite_id,
                    )
                    continue
                tasks.append(
                    execute_suite_async(
                        suite=suite,
                        orchestrator=orchestrator,
                        parallel=True,
                        max_workers=4,
                        user_id=user_id,
                    )
                )
            await asyncio.gather(*tasks)
        plan.status = "completed"
        # execute_suite_async persisted each member suite; persist the plan's
        # terminal status (member suites are stored as id references).
        await _store.save_plan(plan)
        all_results = []
        for suite in plan.suites:
            all_results.extend(suite.results)
        await _store.append_history(
            {
                "type": "plan",
                "plan_id": plan.plan_id,
                "container_name": plan.container_name,
                "tenant_id": plan.tenant_id,
                "timestamp": datetime.now(UTC),
                "user_id": user_id,
                "suite_count": len(plan.suites),
                "test_count": sum(len(s.tests) for s in plan.suites),
                "successful": sum(1 for r in all_results if r.success),
                "languages": list(set(r.language.value for r in all_results)),
                "success": all(r.success for r in all_results),
            }
        )
        logger.info(
            f"Orchestration plan execution completed for container '{plan.container_name}' (tenant: {plan.tenant_id})"
        )
    except Exception as e:
        logger.error(f"Plan execution failed: {str(e)}")
        plan.status = "failed"
        # Best-effort (see execute_suite_async).
        try:
            await _store.save_plan(plan)
        except Exception:
            logger.exception("Failed to persist failed status for plan %s", plan.plan_id)
    finally:
        heartbeat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat
