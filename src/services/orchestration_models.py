"""
Orchestration domain models + (de)serialization.

The two hand-written domain classes (``OrchestrationTestSuite`` /
``TestOrchestrationPlan``) live here rather than in the router so the
Postgres-backed store can rehydrate them without importing the router (which
would create an import cycle). The router re-exports both for backwards
compatibility with existing ``from routers.test_orchestration import ...`` paths.

Serialization targets the single-blob ``data: JSONB`` columns in
``db_models.OrchestrationTestSuiteRow`` / ``OrchestrationPlanRow``. Plans store
their member suites as ``suite_ids`` *references* (rehydrated from the suite
table) rather than inlining them, mirroring how ``create_orchestration_plan``
builds a plan out of already-stored suites. Sets (the dependency graph) are
stored as sorted lists because JSON has no set type.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from services.test_execution.test_orchestrator import (
    TestExecutionRequest,
    TestExecutionResult,
)


class OrchestrationTestSuite:
    """Represents a collection of related tests."""

    def __init__(
        self,
        suite_id: str,
        name: str,
        description: str,
        container_name: str,
        tenant_id: str,
    ) -> None:
        self.suite_id = suite_id
        self.name = name
        self.description = description
        self.container_name = container_name
        self.tenant_id = tenant_id
        self.tests: list[TestExecutionRequest] = []
        self.created_at = datetime.now(UTC)
        self.status = "pending"
        self.results: list[TestExecutionResult] = []
        self.metadata: dict[str, Any] = {}


class TestOrchestrationPlan:
    """Execution plan for orchestrated test runs."""

    def __init__(self, plan_id: str, container_name: str, tenant_id: str) -> None:
        self.plan_id = plan_id
        self.container_name = container_name
        self.tenant_id = tenant_id
        self.suites: list[OrchestrationTestSuite] = []
        self.execution_order: list[str] = []
        self.dependencies: dict[str, set[str]] = defaultdict(set)
        self.parallel_groups: list[list[str]] = []
        self.created_at = datetime.now(UTC)
        self.status = "draft"


def _parse_dt(value: Any) -> datetime:
    """Parse an ISO string back to an aware datetime (tolerating a live datetime)."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


# --- Suite (de)serialization -------------------------------------------------


def suite_to_data(suite: OrchestrationTestSuite) -> dict[str, Any]:
    """Serialize an ``OrchestrationTestSuite`` to a JSON-native dict."""
    return {
        "suite_id": suite.suite_id,
        "name": suite.name,
        "description": suite.description,
        "container_name": suite.container_name,
        "tenant_id": suite.tenant_id,
        "tests": [t.model_dump(mode="json") for t in suite.tests],
        "results": [r.model_dump(mode="json") for r in suite.results],
        "created_at": suite.created_at.isoformat(),
        "status": suite.status,
        "metadata": suite.metadata,
    }


def suite_from_data(data: dict[str, Any]) -> OrchestrationTestSuite:
    """Rehydrate an ``OrchestrationTestSuite`` from its stored dict."""
    suite = OrchestrationTestSuite(
        suite_id=data["suite_id"],
        name=data["name"],
        description=data["description"],
        container_name=data["container_name"],
        tenant_id=data["tenant_id"],
    )
    suite.tests = [TestExecutionRequest.model_validate(t) for t in data.get("tests", [])]
    suite.results = [TestExecutionResult.model_validate(r) for r in data.get("results", [])]
    suite.created_at = _parse_dt(data["created_at"])
    suite.status = data.get("status", "pending")
    suite.metadata = data.get("metadata", {}) or {}
    return suite


# --- Plan (de)serialization --------------------------------------------------


def plan_to_data(plan: TestOrchestrationPlan) -> dict[str, Any]:
    """Serialize a ``TestOrchestrationPlan`` (suites stored as id references)."""
    return {
        "plan_id": plan.plan_id,
        "container_name": plan.container_name,
        "tenant_id": plan.tenant_id,
        "suite_ids": [s.suite_id for s in plan.suites],
        "execution_order": plan.execution_order,
        "dependencies": {k: sorted(v) for k, v in plan.dependencies.items()},
        "parallel_groups": plan.parallel_groups,
        "created_at": plan.created_at.isoformat(),
        "status": plan.status,
    }


def plan_from_data(
    data: dict[str, Any],
    load_suite: Callable[[str], OrchestrationTestSuite | None],
) -> TestOrchestrationPlan:
    """Rehydrate a ``TestOrchestrationPlan``; member suites loaded via ``load_suite``.

    ``load_suite`` is called for each stored ``suite_id``; missing suites are
    skipped gracefully (a referenced suite may have been deleted).
    """
    plan = TestOrchestrationPlan(
        plan_id=data["plan_id"],
        container_name=data["container_name"],
        tenant_id=data["tenant_id"],
    )
    plan.execution_order = data.get("execution_order", [])
    plan.dependencies = defaultdict(
        set, {k: set(v) for k, v in data.get("dependencies", {}).items()}
    )
    plan.parallel_groups = data.get("parallel_groups", [])
    plan.created_at = _parse_dt(data["created_at"])
    plan.status = data.get("status", "draft")
    for suite_id in data.get("suite_ids", []):
        suite = load_suite(suite_id)
        if suite is not None:
            plan.suites.append(suite)
    return plan
