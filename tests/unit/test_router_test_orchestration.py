"""Unit tests for test orchestration router.

State lives in Postgres via ``services.orchestration_store`` (no module globals).
DB-backed tests take the ``pg_db`` fixture (truncates the orchestration tables and
self-skips when DATABASE_URL isn't Postgres). Suite ``tests`` and ``results`` are
real ``TestExecutionRequest`` / ``TestExecutionResult`` models — MagicMocks cannot
be JSONB-serialized by the store.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from services.orchestration_models import OrchestrationTestSuite, TestOrchestrationPlan
from services.orchestration_store import get_orchestration_store
from services.test_execution.test_orchestrator import (
    OrchestratorType,
    TestExecutionRequest,
    TestExecutionResult,
    TestLanguage,
)


def _req(
    container_name: str = "c1",
    language: TestLanguage = TestLanguage.PYTHON,
    test_type: OrchestratorType = OrchestratorType.UNIT,
) -> TestExecutionRequest:
    """Build a real, JSON-serializable test execution request."""
    return TestExecutionRequest(
        language=language,
        test_type=test_type,
        source_code="print(1)",
        container_name=container_name,
    )


def _result(
    test_id: str = "t1",
    success: bool = True,
    exit_code: int = 0,
    container_name: str = "c1",
    language: TestLanguage = TestLanguage.PYTHON,
    test_type: OrchestratorType = OrchestratorType.UNIT,
    execution_time_ms: float = 100.0,
    stdout: str = "",
    stderr: str = "",
) -> TestExecutionResult:
    """Build a real, JSON-serializable test execution result."""
    return TestExecutionResult(
        test_id=test_id,
        success=success,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        execution_time_ms=execution_time_ms,
        language=language,
        test_type=test_type,
        container_name=container_name,
        created_at=datetime.now(UTC),
    )


class TestCalculateExecutionOrder:
    """Test topological sort for execution order."""

    def test_no_dependencies(self):
        """Test order with no dependencies."""
        from routers.test_orchestration import _calculate_execution_order

        result = _calculate_execution_order(["a", "b", "c"], {})
        assert set(result) == {"a", "b", "c"}
        assert len(result) == 3

    def test_with_dependencies(self):
        """Test order respects dependencies."""
        from routers.test_orchestration import _calculate_execution_order

        deps = {"b": {"a"}, "c": {"b"}}
        result = _calculate_execution_order(["a", "b", "c"], deps)
        assert result.index("a") < result.index("b")
        assert result.index("b") < result.index("c")

    def test_parallel_items(self):
        """Test items without mutual dependencies."""
        from routers.test_orchestration import _calculate_execution_order

        deps = {"c": {"a"}}
        result = _calculate_execution_order(["a", "b", "c"], deps)
        assert result.index("a") < result.index("c")
        assert len(result) == 3

    def test_single_item(self):
        """Test with a single item."""
        from routers.test_orchestration import _calculate_execution_order

        result = _calculate_execution_order(["x"], {})
        assert result == ["x"]


class TestCalculateParallelGroups:
    """Test parallel group calculation."""

    def test_all_parallel(self):
        """Test all items can run in parallel when no deps."""
        from routers.test_orchestration import _calculate_parallel_groups

        result = _calculate_parallel_groups(["a", "b", "c"], {})
        assert len(result) == 1
        assert set(result[0]) == {"a", "b", "c"}

    def test_sequential_deps(self):
        """Test sequential dependencies create multiple groups."""
        from routers.test_orchestration import _calculate_parallel_groups

        deps = {"b": {"a"}, "c": {"b"}}
        result = _calculate_parallel_groups(["a", "b", "c"], deps)
        assert len(result) == 3
        assert result[0] == ["a"]
        assert result[1] == ["b"]
        assert result[2] == ["c"]

    def test_partial_deps(self):
        """Test mix of parallel and sequential."""
        from routers.test_orchestration import _calculate_parallel_groups

        deps = {"c": {"a"}}
        result = _calculate_parallel_groups(["a", "b", "c"], deps)
        # a and b can be parallel (no deps), c depends on a -> 2 groups
        assert len(result) == 2
        assert set(result[0]) == {"a", "b"}
        assert result[1] == ["c"]


class TestOrchestrationTestSuiteClass:
    """Test OrchestrationTestSuite dataclass."""

    def test_suite_creation(self):
        """Test creating a test suite."""
        from routers.test_orchestration import OrchestrationTestSuite

        suite = OrchestrationTestSuite(
            suite_id="s1",
            name="Test Suite",
            description="A test suite",
            container_name="container-1",
            tenant_id="n8n-dev",
        )
        assert suite.suite_id == "s1"
        assert suite.name == "Test Suite"
        assert suite.status == "pending"
        assert suite.tests == []
        assert suite.results == []

    def test_suite_has_metadata(self):
        """Test suite has metadata dict."""
        from routers.test_orchestration import OrchestrationTestSuite

        suite = OrchestrationTestSuite(
            suite_id="s1",
            name="Suite",
            description="Desc",
            container_name="c1",
            tenant_id="t1",
        )
        assert suite.metadata == {}

    def test_suite_has_created_at(self):
        """Test suite has created_at timestamp that is recent."""
        from routers.test_orchestration import OrchestrationTestSuite

        before = datetime.now(UTC)
        suite = OrchestrationTestSuite(
            suite_id="s1",
            name="Suite",
            description="Desc",
            container_name="c1",
            tenant_id="t1",
        )
        after = datetime.now(UTC)
        assert isinstance(suite.created_at, datetime)
        assert before <= suite.created_at <= after


class TestTestOrchestrationPlan:
    """Test TestOrchestrationPlan dataclass."""

    def test_plan_creation(self):
        """Test creating an orchestration plan."""
        from routers.test_orchestration import TestOrchestrationPlan

        plan = TestOrchestrationPlan(
            plan_id="p1",
            container_name="container-1",
            tenant_id="n8n-dev",
        )
        assert plan.plan_id == "p1"
        assert plan.status == "draft"
        assert plan.suites == []
        assert plan.execution_order == []
        assert plan.parallel_groups == []

    def test_plan_has_container_name(self):
        """Test plan stores container_name."""
        from routers.test_orchestration import TestOrchestrationPlan

        plan = TestOrchestrationPlan(
            plan_id="p1",
            container_name="my-container",
            tenant_id="n8n-dev",
        )
        assert plan.container_name == "my-container"


class TestCreateTestSuite:
    """Test the create_test_suite endpoint."""

    async def test_create_suite_success(self, pg_db):
        """Test successful test suite creation."""
        from routers.test_orchestration import create_test_suite

        mock_test = _req(container_name="my-container")

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        mock_container = MagicMock()
        mock_container.status.value = "running"

        with patch(
            "routers.test_orchestration.validate_and_get_container_for_tenant",
            return_value=mock_container,
        ):
            result = await create_test_suite(
                container_name="my-container",
                name="My Suite",
                description="Test suite",
                tests=[mock_test],
                metadata={"key": "val"},
                current_user=mock_user,
                tenant_id="n8n-dev",
            )

        assert result["name"] == "My Suite"
        assert result["description"] == "Test suite"
        assert result["test_count"] == 1
        assert result["container_name"] == "my-container"
        assert result["tenant_id"] == "n8n-dev"
        assert result["status"] == "pending"
        assert result["languages"] == ["python"]
        # suite_id is a UUID v7 string
        suite_id = result["suite_id"]
        assert isinstance(suite_id, str) and len(suite_id) == 36 and suite_id.count("-") == 4
        # Verify suite was persisted to the store
        stored = await get_orchestration_store().get_suite(suite_id)
        assert stored is not None
        assert stored.metadata == {"key": "val"}
        # created_at is an ISO-format timestamp string
        assert "T" in result["created_at"]

    async def test_create_suite_empty_name(self):
        """Test creating suite with empty name raises 400."""
        from routers.test_orchestration import create_test_suite

        mock_test = _req(container_name="c1")

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        mock_container = MagicMock()
        mock_container.status.value = "running"

        with patch(
            "routers.test_orchestration.validate_and_get_container_for_tenant",
            return_value=mock_container,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await create_test_suite(
                    container_name="c1",
                    name="",
                    description="desc",
                    tests=[mock_test],
                    current_user=mock_user,
                    tenant_id="n8n-dev",
                )
            assert exc_info.value.status_code == 400
            assert "empty" in str(exc_info.value.detail).lower()

    async def test_create_suite_empty_tests(self):
        """Test creating suite with no tests raises 400."""
        from routers.test_orchestration import create_test_suite

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        mock_container = MagicMock()
        mock_container.status.value = "running"

        with patch(
            "routers.test_orchestration.validate_and_get_container_for_tenant",
            return_value=mock_container,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await create_test_suite(
                    container_name="c1",
                    name="My Suite",
                    description="desc",
                    tests=[],
                    current_user=mock_user,
                    tenant_id="n8n-dev",
                )
            assert exc_info.value.status_code == 400
            assert "at least one test" in str(exc_info.value.detail).lower()


class TestListTestSuites:
    """Test the list_test_suites endpoint."""

    async def test_list_suites_no_filters(self, pg_db):
        """Test listing all suites without filters."""
        from routers.test_orchestration import list_test_suites

        suite = OrchestrationTestSuite(
            suite_id="s1",
            name="Suite 1",
            description="First suite",
            container_name="c1",
            tenant_id="n8n-dev",
        )
        suite.tests = [_req(container_name="c1")]
        await get_orchestration_store().save_suite(suite)

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        result = await list_test_suites(
            container_name=None,
            status_filter=None,
            language_filter=None,
            current_user=mock_user,
            tenant_id="n8n-dev",
        )

        assert result["status"] == "success"
        assert result["data"]["total"] == 1
        assert len(result["data"]["suites"]) == 1
        suite_data = result["data"]["suites"][0]
        assert suite_data["suite_id"] == "s1"
        assert suite_data["name"] == "Suite 1"
        assert suite_data["description"] == "First suite"
        assert suite_data["container_name"] == "c1"
        assert suite_data["tenant_id"] == "n8n-dev"
        assert suite_data["test_count"] == 1
        assert suite_data["status"] == "pending"
        assert suite_data["languages"] == ["python"]

    async def test_list_suites_with_container_filter(self, pg_db):
        """Test listing suites filtered by container name."""
        from routers.test_orchestration import list_test_suites

        store = get_orchestration_store()

        suite1 = OrchestrationTestSuite("s1", "Suite 1", "desc", "c1", "n8n-dev")
        suite1.tests = [_req(container_name="c1")]
        await store.save_suite(suite1)

        suite2 = OrchestrationTestSuite("s2", "Suite 2", "desc", "c2", "n8n-dev")
        suite2.tests = [_req(container_name="c2")]
        await store.save_suite(suite2)

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        mock_container = MagicMock()
        mock_container.status.value = "running"

        with patch(
            "routers.test_orchestration.validate_and_get_container_for_tenant",
            return_value=mock_container,
        ):
            result = await list_test_suites(
                container_name="c1",
                status_filter=None,
                language_filter=None,
                current_user=mock_user,
                tenant_id="n8n-dev",
            )

        assert result["status"] == "success"
        assert result["data"]["total"] == 1
        assert len(result["data"]["suites"]) == 1
        assert result["data"]["suites"][0]["suite_id"] == "s1"
        assert result["data"]["suites"][0]["name"] == "Suite 1"
        assert result["data"]["suites"][0]["container_name"] == "c1"

    async def test_list_suites_empty(self, pg_db):
        """Test listing suites when none exist."""
        from routers.test_orchestration import list_test_suites

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        result = await list_test_suites(
            container_name=None,
            status_filter=None,
            language_filter=None,
            current_user=mock_user,
            tenant_id="n8n-dev",
        )

        assert result["status"] == "success"
        assert result["data"]["total"] == 0
        assert result["data"]["suites"] == []

    async def test_list_suites_store_error_raises_500(self, pg_db):
        """A store failure surfaces as HTTP 500 (no longer masked as empty 200)."""
        from routers.test_orchestration import list_test_suites

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        with patch(
            "services.orchestration_store.OrchestrationStore.list_suites",
            side_effect=RuntimeError("store down"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await list_test_suites(
                    container_name=None,
                    status_filter=None,
                    language_filter=None,
                    current_user=mock_user,
                    tenant_id="n8n-dev",
                )
        assert exc_info.value.status_code == 500


class TestExecuteTestSuite:
    """Test the execute_test_suite endpoint."""

    async def test_execute_suite_success(self, pg_db):
        """Test successful suite execution."""
        from routers.test_orchestration import execute_test_suite

        store = get_orchestration_store()
        suite = OrchestrationTestSuite("s1", "Suite 1", "desc", "c1", "n8n-dev")
        suite.tests = [_req(container_name="c1")]
        await store.save_suite(suite)

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_bg = MagicMock()

        mock_container = MagicMock()
        mock_container.status.value = "running"

        mock_orchestrator = AsyncMock()

        with (
            patch(
                "routers.test_orchestration.validate_and_get_container_for_tenant",
                return_value=mock_container,
            ),
            patch(
                "routers.test_orchestration.get_orchestrator",
                return_value=mock_orchestrator,
            ),
        ):
            result = await execute_test_suite(
                suite_id="s1",
                background_tasks=mock_bg,
                parallel=True,
                max_workers=4,
                current_user=mock_user,
                tenant_id="n8n-dev",
            )

        assert result["suite_id"] == "s1"
        assert result["status"] == "started"
        assert result["parallel"] is True
        assert result["max_workers"] == 4
        assert result["message"] == "Test suite execution started with 1 tests"
        # The compare-and-swap persisted status=running to the store.
        assert (await store.get_suite("s1")).status == "running"
        # Verify background task was scheduled
        mock_bg.add_task.assert_called_once()

    async def test_execute_suite_not_found(self, pg_db):
        """Test executing nonexistent suite raises 404."""
        from routers.test_orchestration import execute_test_suite

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_bg = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await execute_test_suite(
                suite_id="nonexistent",
                background_tasks=mock_bg,
                current_user=mock_user,
                tenant_id="n8n-dev",
            )
        assert exc_info.value.status_code == 404
        assert "nonexistent" in str(exc_info.value.detail)

    async def test_execute_suite_already_running(self, pg_db):
        """Test executing already-running suite raises 400."""
        from routers.test_orchestration import execute_test_suite

        suite = OrchestrationTestSuite("s1", "Suite 1", "desc", "c1", "n8n-dev")
        suite.tests = [_req(container_name="c1")]
        suite.status = "running"
        await get_orchestration_store().save_suite(suite)

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_bg = MagicMock()

        mock_container = MagicMock()
        mock_container.status.value = "running"

        with patch(
            "routers.test_orchestration.validate_and_get_container_for_tenant",
            return_value=mock_container,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await execute_test_suite(
                    suite_id="s1",
                    background_tasks=mock_bg,
                    max_workers=4,
                    current_user=mock_user,
                    tenant_id="n8n-dev",
                )
            assert exc_info.value.status_code == 400
            assert "already running" in str(exc_info.value.detail)


class TestStaleRunningReclaim:
    """The status CAS self-heals a suite/plan flag wedged by a crash mid-run.

    ``updated_at`` holds the mark time; a row stuck in running/executing longer
    than the threshold is reclaimable so a fresh execute can proceed, while a
    genuinely-fresh run (recent ``updated_at``) is never stolen.
    """

    async def test_stale_running_suite_is_reclaimed(self, pg_db):
        from sqlalchemy import text

        store = get_orchestration_store()
        suite = OrchestrationTestSuite("s1", "Suite 1", "desc", "c1", "n8n-dev")
        suite.tests = [_req(container_name="c1")]
        suite.status = "running"
        await store.save_suite(suite)
        # Backdate updated_at so the row looks wedged (crash mid-run).
        async with pg_db() as session:
            await session.execute(
                text(
                    "UPDATE orchestration_test_suite "
                    "SET updated_at = now() - make_interval(secs => 7200) "
                    "WHERE suite_id = 's1'"
                )
            )
            await session.commit()
        assert (
            await store.try_mark_suite_running("s1", "n8n-dev", stale_running_seconds=3600) is True
        )

    async def test_fresh_running_suite_is_not_reclaimed(self, pg_db):
        store = get_orchestration_store()
        suite = OrchestrationTestSuite("s2", "Suite 2", "desc", "c1", "n8n-dev")
        suite.tests = [_req(container_name="c1")]
        suite.status = "running"
        await store.save_suite(suite)  # updated_at = now()
        assert (
            await store.try_mark_suite_running("s2", "n8n-dev", stale_running_seconds=3600) is False
        )

    async def test_stale_executing_plan_is_reclaimed(self, pg_db):
        from sqlalchemy import text

        store = get_orchestration_store()
        suite = OrchestrationTestSuite("ps1", "S", "d", "c1", "n8n-dev")
        suite.tests = [_req(container_name="c1")]
        await store.save_suite(suite)
        plan = TestOrchestrationPlan("p1", "c1", "n8n-dev")
        plan.suites = [suite]
        plan.status = "executing"
        await store.save_plan(plan)
        async with pg_db() as session:
            await session.execute(
                text(
                    "UPDATE orchestration_plan "
                    "SET updated_at = now() - make_interval(secs => 7200) "
                    "WHERE plan_id = 'p1'"
                )
            )
            await session.commit()
        assert (
            await store.try_mark_plan_executing("p1", "n8n-dev", stale_running_seconds=3600) is True
        )

    async def test_fresh_executing_plan_is_not_reclaimed(self, pg_db):
        store = get_orchestration_store()
        suite = OrchestrationTestSuite("ps2", "S", "d", "c1", "n8n-dev")
        suite.tests = [_req(container_name="c1")]
        await store.save_suite(suite)
        plan = TestOrchestrationPlan("p2", "c1", "n8n-dev")
        plan.suites = [suite]
        plan.status = "executing"
        await store.save_plan(plan)
        assert (
            await store.try_mark_plan_executing("p2", "n8n-dev", stale_running_seconds=3600)
            is False
        )

    async def test_concurrent_mark_running_has_single_winner(self, pg_db):
        """Racing CAS attempts on a pending suite → exactly one winner.

        Guards the READ COMMITTED serialization: the eligibility qual reads the
        LIVE row, so a loser's EvalPlanQual re-check sees status='running'. If it
        read the CTE snapshot instead, multiple racers would win → double-execute.
        """
        import asyncio

        store = get_orchestration_store()
        suite = OrchestrationTestSuite("cc1", "S", "d", "c1", "n8n-dev")
        suite.tests = [_req(container_name="c1")]
        await store.save_suite(suite)  # status 'pending'

        results = await asyncio.gather(
            *(store.try_mark_suite_running("cc1", "n8n-dev") for _ in range(5))
        )
        assert sum(results) == 1

    async def test_concurrent_reclaim_of_stale_suite_has_single_winner(self, pg_db):
        """Racing reclaims of one wedged suite → exactly one winner (not all)."""
        import asyncio

        from sqlalchemy import text

        store = get_orchestration_store()
        suite = OrchestrationTestSuite("cc2", "S", "d", "c1", "n8n-dev")
        suite.tests = [_req(container_name="c1")]
        suite.status = "running"
        await store.save_suite(suite)
        async with pg_db() as session:
            await session.execute(
                text(
                    "UPDATE orchestration_test_suite "
                    "SET updated_at = now() - make_interval(secs => 7200) "
                    "WHERE suite_id = 'cc2'"
                )
            )
            await session.commit()

        results = await asyncio.gather(
            *(
                store.try_mark_suite_running("cc2", "n8n-dev", stale_running_seconds=3600)
                for _ in range(5)
            )
        )
        assert sum(results) == 1


class TestLeaseHeartbeat:
    """A live long run renews its lease so the reaper never steals it."""

    async def test_heartbeat_suite_prevents_reclaim(self, pg_db):
        from sqlalchemy import text

        store = get_orchestration_store()
        suite = OrchestrationTestSuite("hb1", "S", "d", "c1", "n8n-dev")
        suite.tests = [_req(container_name="c1")]
        suite.status = "running"
        await store.save_suite(suite)
        # Make it look wedged, then renew the lease.
        async with pg_db() as session:
            await session.execute(
                text(
                    "UPDATE orchestration_test_suite "
                    "SET updated_at = now() - make_interval(secs => 7200) WHERE suite_id = 'hb1'"
                )
            )
            await session.commit()
        assert await store.heartbeat_suite("hb1") is True
        # Now fresh → the reaper must NOT reclaim it.
        assert (
            await store.try_mark_suite_running("hb1", "n8n-dev", stale_running_seconds=3600)
            is False
        )

    async def test_heartbeat_ignores_non_running_suite(self, pg_db):
        store = get_orchestration_store()
        suite = OrchestrationTestSuite("hb2", "S", "d", "c1", "n8n-dev")
        suite.tests = [_req(container_name="c1")]
        suite.status = "completed"
        await store.save_suite(suite)
        assert await store.heartbeat_suite("hb2") is False

    async def test_heartbeat_plan_prevents_reclaim(self, pg_db):
        from sqlalchemy import text

        store = get_orchestration_store()
        suite = OrchestrationTestSuite("hbps", "S", "d", "c1", "n8n-dev")
        suite.tests = [_req(container_name="c1")]
        await store.save_suite(suite)
        plan = TestOrchestrationPlan("hbp1", "c1", "n8n-dev")
        plan.suites = [suite]
        plan.status = "executing"
        await store.save_plan(plan)
        async with pg_db() as session:
            await session.execute(
                text(
                    "UPDATE orchestration_plan "
                    "SET updated_at = now() - make_interval(secs => 7200) WHERE plan_id = 'hbp1'"
                )
            )
            await session.commit()
        assert await store.heartbeat_plan("hbp1") is True
        assert (
            await store.try_mark_plan_executing("hbp1", "n8n-dev", stale_running_seconds=3600)
            is False
        )

    def test_interval_derives_from_threshold(self):
        from routers.test_orchestration import _lease_heartbeat_interval

        # default ORCHESTRATION_STALE_RUNNING_SECONDS=3600 → a quarter = 900s
        assert _lease_heartbeat_interval() == 900

    async def test_renew_lease_ticks_until_cancelled(self):
        import asyncio

        from routers.test_orchestration import _renew_lease

        calls = 0

        async def touch():
            nonlocal calls
            calls += 1
            return True

        task = asyncio.create_task(_renew_lease(touch, 0.01))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert calls >= 1

    async def test_renew_lease_survives_touch_errors(self):
        import asyncio

        from routers.test_orchestration import _renew_lease

        calls = 0

        async def touch():
            nonlocal calls
            calls += 1
            raise RuntimeError("transient db blip")

        task = asyncio.create_task(_renew_lease(touch, 0.01))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert calls >= 2  # kept ticking despite the errors


class TestExecutePlanAsyncSuiteVanish:
    """``execute_plan_async`` tolerates a suite that vanished after plan build.

    ``parallel_groups`` stores suite ids verbatim, but ``plan_from_data`` drops
    any suite whose row no longer exists. The missing id must be skipped so the
    surviving suites still run — the old ``next(...)`` lookup raised
    ``StopIteration`` and failed the entire plan.
    """

    async def test_missing_suite_is_skipped_and_plan_completes(self, pg_db):
        from routers.test_orchestration import execute_plan_async

        present = OrchestrationTestSuite("live", "Live", "d", "c1", "n8n-dev")
        present.tests = [_req(container_name="c1")]
        plan = TestOrchestrationPlan("p1", "c1", "n8n-dev")
        # "gone" was deleted out-of-band → dropped from plan.suites by
        # plan_from_data, but still referenced in parallel_groups.
        plan.suites = [present]
        plan.parallel_groups = [["live", "gone"]]

        mock_orchestrator = AsyncMock()
        mock_orchestrator.execute_tests = AsyncMock(return_value=_result(container_name="c1"))

        await execute_plan_async(plan, mock_orchestrator, user_id="u1")

        # Plan completes rather than failing on the missing suite.
        assert plan.status == "completed"
        # The surviving suite actually ran.
        assert present.status == "completed"
        assert len(present.results) == 1
        # Only the present suite's one test executed; the missing id was skipped.
        mock_orchestrator.execute_tests.assert_awaited_once()

    async def test_all_suites_missing_plan_still_completes(self, pg_db):
        from routers.test_orchestration import execute_plan_async

        plan = TestOrchestrationPlan("p2", "c1", "n8n-dev")
        plan.suites = []  # every referenced suite vanished
        plan.parallel_groups = [["gone-a", "gone-b"]]

        mock_orchestrator = AsyncMock()
        mock_orchestrator.execute_tests = AsyncMock(return_value=_result(container_name="c1"))

        await execute_plan_async(plan, mock_orchestrator, user_id="u1")

        assert plan.status == "completed"
        mock_orchestrator.execute_tests.assert_not_awaited()


class TestGetSuiteResults:
    """Test the get_suite_results endpoint."""

    async def test_get_suite_results_success(self, pg_db):
        """Test getting results for a completed suite."""
        from routers.test_orchestration import get_suite_results

        suite = OrchestrationTestSuite("s1", "Suite 1", "desc", "c1", "n8n-dev")
        suite.status = "completed"
        suite.results = [
            _result(
                test_id="t1",
                success=True,
                exit_code=0,
                container_name="c1",
                execution_time_ms=100.0,
            )
        ]
        await get_orchestration_store().save_suite(suite)

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        result = await get_suite_results(
            suite_id="s1",
            current_user=mock_user,
            tenant_id="n8n-dev",
        )

        assert result["suite_id"] == "s1"
        assert result["name"] == "Suite 1"
        assert result["container_name"] == "c1"
        assert result["status"] == "completed"
        assert result["summary"]["total"] == 1
        assert result["summary"]["successful"] == 1
        assert result["summary"]["failed"] == 0
        assert result["summary"]["success_rate"] == 100.0
        assert result["summary"]["executed"] is True
        assert result["summary"]["status"] == "completed"
        assert result["summary"]["passed"] is True
        assert result["passed"] is True
        # Verify results_by_language grouping
        assert "python" in result["results_by_language"]
        assert len(result["results_by_language"]["python"]) == 1
        assert result["results_by_language"]["python"][0]["test_id"] == "t1"
        assert result["results_by_language"]["python"][0]["success"] is True
        assert result["results_by_language"]["python"][0]["execution_time_ms"] == 100.0
        assert result["results_by_language"]["python"][0]["test_type"] == "unit"
        # Verify execution_details
        assert len(result["execution_details"]) == 1
        detail = result["execution_details"][0]
        assert detail["test_id"] == "t1"
        assert detail["language"] == "python"
        assert detail["success"] is True
        assert detail["exit_code"] == 0
        assert detail["container_name"] == "c1"

    # -----------------------------------------------------------------------
    # Defect 2 control: a suite that NEVER RAN also reports failed == 0.
    #
    # Revert guard. These fail if `passed`/`executed`/`status` are dropped from
    # the payload or if the server-side verdict stops requiring
    # `total > 0 AND failed == 0 AND status == "completed"`.
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize(
        "status",
        ["pending", "running", "failed", "completed"],
    )
    async def test_never_run_suite_is_not_a_pass(self, pg_db, status):
        """Zero results means nothing was verified -- at ANY suite status."""
        from routers.test_orchestration import get_suite_results

        suite = OrchestrationTestSuite("s-empty", "Empty", "desc", "c1", "n8n-dev")
        suite.status = status
        suite.results = []
        await get_orchestration_store().save_suite(suite)

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        result = await get_suite_results(
            suite_id="s-empty", current_user=mock_user, tenant_id="n8n-dev"
        )

        # The trap itself: failed == 0 on a suite that ran nothing.
        assert result["summary"]["failed"] == 0
        assert result["summary"]["total"] == 0
        # ...which must NOT be readable as success.
        assert result["summary"]["executed"] is False
        assert result["summary"]["passed"] is False
        assert result["passed"] is False
        # A rate over an empty suite is undefined, not 0%.
        assert result["summary"]["success_rate"] is None
        # And the status the safe predicate needs is published.
        assert result["summary"]["status"] == status

    async def test_incomplete_suite_with_passing_results_is_not_a_pass(self, pg_db):
        """total > 0 and failed == 0 is still not a pass while status != completed."""
        from routers.test_orchestration import get_suite_results

        suite = OrchestrationTestSuite("s-running", "Running", "desc", "c1", "n8n-dev")
        suite.status = "running"
        suite.results = [_result(test_id="t1", success=True, exit_code=0)]
        await get_orchestration_store().save_suite(suite)

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        result = await get_suite_results(
            suite_id="s-running", current_user=mock_user, tenant_id="n8n-dev"
        )

        assert result["summary"]["total"] == 1
        assert result["summary"]["failed"] == 0
        assert result["summary"]["executed"] is True
        assert result["summary"]["status"] == "running"
        assert result["summary"]["passed"] is False
        assert result["passed"] is False

    async def test_completed_suite_with_a_failure_is_not_a_pass(self, pg_db):
        """Negative control: the verdict must still be able to see a real failure."""
        from routers.test_orchestration import get_suite_results

        suite = OrchestrationTestSuite("s-red", "Red", "desc", "c1", "n8n-dev")
        suite.status = "completed"
        suite.results = [
            _result(test_id="t1", success=True, exit_code=0),
            _result(test_id="t2", success=False, exit_code=1),
        ]
        await get_orchestration_store().save_suite(suite)

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        result = await get_suite_results(
            suite_id="s-red", current_user=mock_user, tenant_id="n8n-dev"
        )

        assert result["summary"]["total"] == 2
        assert result["summary"]["failed"] == 1
        assert result["summary"]["executed"] is True
        assert result["summary"]["passed"] is False
        assert result["passed"] is False
        assert result["summary"]["success_rate"] == 50.0

    async def test_get_suite_results_not_found(self, pg_db):
        """Test getting results for nonexistent suite raises 404."""
        from routers.test_orchestration import get_suite_results

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        with pytest.raises(HTTPException) as exc_info:
            await get_suite_results(
                suite_id="nonexistent",
                current_user=mock_user,
                tenant_id="n8n-dev",
            )
        assert exc_info.value.status_code == 404
        assert "nonexistent" in str(exc_info.value.detail)

    async def test_get_suite_results_wrong_tenant(self, pg_db):
        """Test access denied for wrong tenant."""
        from routers.test_orchestration import get_suite_results

        suite = OrchestrationTestSuite("s1", "Suite", "desc", "c1", "tenant-a")
        await get_orchestration_store().save_suite(suite)

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        with pytest.raises(HTTPException) as exc_info:
            await get_suite_results(
                suite_id="s1",
                current_user=mock_user,
                tenant_id="tenant-b",
            )
        assert exc_info.value.status_code == 403
        assert "Access denied" in str(exc_info.value.detail)


class TestCreateOrchestrationPlan:
    """Test the create_orchestration_plan endpoint."""

    async def test_create_plan_success(self, pg_db):
        """Test successful plan creation."""
        from routers.test_orchestration import create_orchestration_plan

        suite = OrchestrationTestSuite("s1", "Suite", "desc", "c1", "n8n-dev")
        suite.tests = [_req(container_name="c1")]
        await get_orchestration_store().save_suite(suite)

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        mock_container = MagicMock()
        mock_container.status.value = "running"

        with patch(
            "routers.test_orchestration.validate_and_get_container_for_tenant",
            return_value=mock_container,
        ):
            result = await create_orchestration_plan(
                container_name="c1",
                suite_ids=["s1"],
                dependencies=None,
                current_user=mock_user,
                tenant_id="n8n-dev",
            )

        assert result["suite_count"] == 1
        assert result["container_name"] == "c1"
        assert result["tenant_id"] == "n8n-dev"
        assert result["status"] == "draft"
        assert result["execution_order"] == ["s1"]
        assert result["parallel_groups"] == [["s1"]]
        # plan_id is a UUID v7 string
        plan_id = result["plan_id"]
        assert isinstance(plan_id, str) and len(plan_id) == 36 and plan_id.count("-") == 4
        # Verify plan was persisted to the store
        assert await get_orchestration_store().get_plan(plan_id) is not None
        # created_at is an ISO-format timestamp string
        assert "T" in result["created_at"]

    async def test_create_plan_empty_suite_ids(self):
        """Test creating plan with empty suite IDs raises 400."""
        from routers.test_orchestration import create_orchestration_plan

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        mock_container = MagicMock()
        mock_container.status.value = "running"

        with patch(
            "routers.test_orchestration.validate_and_get_container_for_tenant",
            return_value=mock_container,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await create_orchestration_plan(
                    container_name="c1",
                    suite_ids=[],
                    current_user=mock_user,
                    tenant_id="n8n-dev",
                )
            assert exc_info.value.status_code == 400
            assert "at least one" in str(exc_info.value.detail).lower()


class TestExecuteOrchestrationPlan:
    """Test the execute_orchestration_plan endpoint."""

    async def test_execute_plan_success(self, pg_db):
        """Test successful plan execution."""
        from routers.test_orchestration import execute_orchestration_plan

        store = get_orchestration_store()

        suite = OrchestrationTestSuite("s1", "Suite", "desc", "c1", "n8n-dev")
        suite.tests = [_req(container_name="c1")]
        await store.save_suite(suite)

        plan = TestOrchestrationPlan("p1", "c1", "n8n-dev")
        plan.suites = [suite]
        plan.parallel_groups = [["s1"]]
        await store.save_plan(plan)

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_bg = MagicMock()

        mock_container = MagicMock()
        mock_container.status.value = "running"

        mock_orchestrator = AsyncMock()

        with (
            patch(
                "routers.test_orchestration.validate_and_get_container_for_tenant",
                return_value=mock_container,
            ),
            patch(
                "routers.test_orchestration.get_orchestrator",
                return_value=mock_orchestrator,
            ),
        ):
            result = await execute_orchestration_plan(
                plan_id="p1",
                background_tasks=mock_bg,
                current_user=mock_user,
                tenant_id="n8n-dev",
            )

        assert result["plan_id"] == "p1"
        assert result["status"] == "started"
        assert result["parallel_groups"] == 1
        assert result["message"] == "Orchestration plan execution started with 1 suites"
        # The compare-and-swap persisted status=executing to the store.
        assert (await store.get_plan("p1")).status == "executing"
        # Verify background task was scheduled
        mock_bg.add_task.assert_called_once()

    async def test_execute_plan_not_found(self, pg_db):
        """Test executing nonexistent plan raises 404."""
        from routers.test_orchestration import execute_orchestration_plan

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_bg = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await execute_orchestration_plan(
                plan_id="nonexistent",
                background_tasks=mock_bg,
                current_user=mock_user,
                tenant_id="n8n-dev",
            )
        assert exc_info.value.status_code == 404
        assert "nonexistent" in str(exc_info.value.detail)


class TestExecutionHistory:
    """Test the get_execution_history endpoint."""

    async def test_execution_history_empty(self, pg_db):
        """Test execution history returns empty when no history."""
        from routers.test_orchestration import get_execution_history

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        result = await get_execution_history(
            container_name=None,
            limit=100,
            since=None,
            current_user=mock_user,
            tenant_id="n8n-dev",
        )

        assert result["status"] == "success"
        assert result["data"]["count"] == 0
        assert result["data"]["total"] == 0
        assert result["data"]["executions"] == []
        assert result["data"]["filtered_by_container"] is None

    async def test_execution_history_with_data(self, pg_db):
        """Test execution history returns filtered data."""
        from routers.test_orchestration import get_execution_history

        await get_orchestration_store().append_history(
            {
                "type": "suite",
                "suite_id": "s1",
                "container_name": "c1",
                "tenant_id": "n8n-dev",
                "timestamp": datetime.now(UTC),
                "user_id": "testuser",
                "test_count": 5,
                "successful": 4,
                "languages": ["python"],
                "success": False,
            }
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        result = await get_execution_history(
            container_name=None,
            limit=100,
            since=None,
            current_user=mock_user,
            tenant_id="n8n-dev",
        )

        assert result["status"] == "success"
        assert result["data"]["count"] == 1
        assert result["data"]["total"] == 1
        assert result["data"]["filtered_by_container"] is None
        assert len(result["data"]["executions"]) == 1
        execution = result["data"]["executions"][0]
        assert execution["type"] == "suite"
        assert execution["suite_id"] == "s1"
        assert execution["container_name"] == "c1"
        assert execution["tenant_id"] == "n8n-dev"
        assert execution["test_count"] == 5
        assert execution["successful"] == 4
        assert execution["success"] is False
        assert execution["languages"] == ["python"]

    async def test_execution_history_store_error_raises_500(self, pg_db):
        """A store failure surfaces as HTTP 500 (no longer masked as empty 200)."""
        from routers.test_orchestration import get_execution_history

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        with patch(
            "services.orchestration_store.OrchestrationStore.list_history",
            side_effect=RuntimeError("store down"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_execution_history(
                    container_name=None,
                    limit=100,
                    since=None,
                    current_user=mock_user,
                    tenant_id="n8n-dev",
                )
        assert exc_info.value.status_code == 500


class TestOrchestrationMetrics:
    """Test the get_orchestration_metrics endpoint."""

    async def test_metrics_empty(self, pg_db):
        """Test metrics with no data returns zeros."""
        from routers.test_orchestration import get_orchestration_metrics

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        result = await get_orchestration_metrics(
            container_name=None,
            days=7,
            current_user=mock_user,
            tenant_id="n8n-dev",
        )

        assert result["status"] == "success"
        assert result["data"]["total_executions"] == 0
        assert result["data"]["suite_executions"] == 0
        assert result["data"]["plan_executions"] == 0
        assert result["data"]["success_rate"] == 0.0
        assert result["data"]["active_suites"] == 0
        assert result["data"]["active_plans"] == 0
        assert result["data"]["language_distribution"] == {}
        assert result["data"]["period_days"] == 7
        assert result["data"]["container_name"] is None

    async def test_metrics_with_data(self, pg_db):
        """Test metrics with execution data."""
        from routers.test_orchestration import get_orchestration_metrics

        store = get_orchestration_store()

        # Add history entry
        await store.append_history(
            {
                "type": "suite",
                "suite_id": "s1",
                "container_name": "c1",
                "tenant_id": "n8n-dev",
                "timestamp": datetime.now(UTC),
                "user_id": "user",
                "test_count": 2,
                "successful": 2,
                "languages": ["python", "go"],
                "success": True,
            }
        )

        # Add active suite
        suite = OrchestrationTestSuite("s1", "Suite", "desc", "c1", "n8n-dev")
        suite.tests = [_req(container_name="c1")]
        await store.save_suite(suite)

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        result = await get_orchestration_metrics(
            container_name=None,
            days=7,
            current_user=mock_user,
            tenant_id="n8n-dev",
        )

        assert result["data"]["total_executions"] == 1
        assert result["data"]["suite_executions"] == 1
        assert result["data"]["plan_executions"] == 0
        assert result["data"]["success_rate"] == 100.0
        assert result["data"]["active_suites"] == 1
        assert result["data"]["active_plans"] == 0
        assert result["data"]["language_distribution"] == {"python": 1, "go": 1}
        assert result["data"]["period_days"] == 7
        assert result["data"]["container_name"] is None

    async def test_metrics_store_error_raises_500(self, pg_db):
        """A store failure surfaces as HTTP 500 (no longer masked as zero metrics)."""
        from routers.test_orchestration import get_orchestration_metrics

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        with patch(
            "services.orchestration_store.OrchestrationStore.list_history",
            side_effect=RuntimeError("store down"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_orchestration_metrics(
                    container_name=None,
                    days=7,
                    current_user=mock_user,
                    tenant_id="n8n-dev",
                )
        assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# execute_suite_async / execute_plan_async
# ---------------------------------------------------------------------------


class TestExecuteSuiteAsync:
    """Test the background suite execution function.

    ``execute_suite_async`` now persists results + history to the store, so these
    tests take ``pg_db`` and seed real ``TestExecutionRequest`` tests.
    """

    async def test_sequential_execution(self, pg_db):
        """Test sequential suite execution."""
        from routers.test_orchestration import execute_suite_async

        suite = OrchestrationTestSuite("s1", "Suite", "desc", "c1", "n8n-dev")
        real_test = _req(container_name="c1")
        suite.tests = [real_test]

        mock_result = _result(test_id="t1", stdout="ok", execution_time_ms=100.0)
        orchestrator = AsyncMock()
        orchestrator.execute_tests = AsyncMock(return_value=mock_result)

        await execute_suite_async(
            suite=suite,
            orchestrator=orchestrator,
            parallel=False,
            max_workers=1,
            user_id="testuser",
        )
        assert suite.status == "completed"
        assert len(suite.results) == 1
        assert suite.results[0].test_id == "t1"
        assert suite.results[0].success is True
        assert suite.results[0].exit_code == 0
        assert suite.results[0].execution_time_ms == 100.0
        assert suite.results[0].language == TestLanguage.PYTHON
        # Verify orchestrator was called exactly once with our test
        orchestrator.execute_tests.assert_awaited_once_with(real_test)

    async def test_sequential_with_exception(self, pg_db):
        """Test sequential execution handles exceptions."""
        from routers.test_orchestration import execute_suite_async

        suite = OrchestrationTestSuite("s1", "Suite", "desc", "c1", "n8n-dev")
        suite.tests = [_req(container_name="c1")]

        orchestrator = AsyncMock()
        orchestrator.execute_tests = AsyncMock(side_effect=RuntimeError("fail"))

        await execute_suite_async(
            suite=suite,
            orchestrator=orchestrator,
            parallel=False,
            max_workers=1,
            user_id="testuser",
        )
        assert suite.status == "completed"
        assert len(suite.results) == 1
        assert suite.results[0].success is False
        assert suite.results[0].exit_code == 1
        assert suite.results[0].stderr == "fail"
        assert suite.results[0].stdout == ""
        assert suite.results[0].execution_time_ms == 0.0
        assert suite.results[0].container_name == "c1"

    async def test_parallel_execution(self, pg_db):
        """Test parallel suite execution."""
        from routers.test_orchestration import execute_suite_async

        suite = OrchestrationTestSuite("s1", "Suite", "desc", "c1", "n8n-dev")
        suite.tests = [_req(container_name="c1"), _req(container_name="c1")]

        mock_result = _result(test_id="t1", stdout="ok", execution_time_ms=100.0)
        orchestrator = AsyncMock()
        orchestrator.execute_tests = AsyncMock(return_value=mock_result)

        await execute_suite_async(
            suite=suite,
            orchestrator=orchestrator,
            parallel=True,
            max_workers=4,
            user_id="testuser",
        )
        assert suite.status == "completed"
        assert len(suite.results) == 2
        # Both results should be the same mock_result
        for r in suite.results:
            assert r.test_id == "t1"
            assert r.success is True
            assert r.exit_code == 0
        # Orchestrator was called twice (once per test)
        assert orchestrator.execute_tests.await_count == 2


class TestSaveSuiteExecutionResultMerge:
    """save_suite_execution_result merges results+status into the LIVE row.

    The executor holds a suite snapshot captured at schedule time; a full
    save_suite rewrite at run end would revert any field a concurrent writer
    changed mid-run (lost update). The merge-save touches only results+status.
    """

    async def test_concurrent_field_change_survives_merge_save(self, pg_db):
        from sqlalchemy import text

        store = get_orchestration_store()
        # The executor's snapshot: saved as 'running' with the original name.
        suite = OrchestrationTestSuite("m1", "Original Name", "desc", "c1", "n8n-dev")
        suite.tests = [_req(container_name="c1")]
        suite.status = "running"
        await store.save_suite(suite)

        # A concurrent writer changes a DIFFERENT field (name) on the live row.
        async with pg_db() as session:
            await session.execute(
                text(
                    "UPDATE orchestration_test_suite "
                    "SET data = jsonb_set(data, '{name}', '\"Renamed Concurrently\"') "
                    "WHERE suite_id = 'm1'"
                )
            )
            await session.commit()

        # Executor finishes and merge-saves ONLY its results + terminal status
        # from the (now stale) snapshot.
        assert (
            await store.save_suite_execution_result(
                "m1", results=[_result(test_id="rt1")], status="completed"
            )
            is True
        )

        reloaded = await store.get_suite("m1")
        assert reloaded is not None
        # The concurrent name change SURVIVED (a full rewrite would have reverted it).
        assert reloaded.name == "Renamed Concurrently"
        # The execution-owned fields were persisted.
        assert reloaded.status == "completed"
        assert len(reloaded.results) == 1
        assert reloaded.results[0].test_id == "rt1"

    async def test_merge_save_missing_suite_returns_false(self, pg_db):
        store = get_orchestration_store()
        assert (
            await store.save_suite_execution_result(
                "does-not-exist", results=[], status="completed"
            )
            is False
        )


class TestExecutePlanAsync:
    """Test the background plan execution function."""

    async def test_plan_execution(self, pg_db):
        """Test plan execution completes."""
        from routers.test_orchestration import execute_plan_async

        suite = OrchestrationTestSuite("s1", "Suite", "desc", "c1", "n8n-dev")
        suite.tests = [_req(container_name="c1")]

        plan = TestOrchestrationPlan("p1", "c1", "n8n-dev")
        plan.suites = [suite]
        plan.parallel_groups = [["s1"]]

        mock_result = _result(test_id="t1", stdout="ok", execution_time_ms=100.0)
        orchestrator = AsyncMock()
        orchestrator.execute_tests = AsyncMock(return_value=mock_result)

        await execute_plan_async(
            plan=plan,
            orchestrator=orchestrator,
            user_id="testuser",
        )
        assert plan.status == "completed"
        # Suite within the plan should also be completed
        assert suite.status == "completed"
        assert len(suite.results) == 1
        assert suite.results[0].success is True
        # Orchestrator was called once (1 test in 1 suite)
        orchestrator.execute_tests.assert_awaited_once()


# ---------------------------------------------------------------------------
# create_orchestration_plan advanced
# ---------------------------------------------------------------------------


class TestCreateOrchestrationPlanAdvanced:
    """Additional tests for create_orchestration_plan edge cases."""

    async def test_duplicate_suite_ids(self):
        """Test duplicate suite IDs raises 400."""
        from routers.test_orchestration import create_orchestration_plan

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        mock_container = MagicMock()
        mock_container.status.value = "running"

        with patch(
            "routers.test_orchestration.validate_and_get_container_for_tenant",
            return_value=mock_container,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await create_orchestration_plan(
                    container_name="c1",
                    suite_ids=["s1", "s1"],
                    current_user=mock_user,
                    tenant_id="n8n-dev",
                )
            assert exc_info.value.status_code == 400
            assert "Duplicate" in str(exc_info.value.detail)

    async def test_suite_not_found(self, pg_db):
        """Test missing suite raises 404."""
        from routers.test_orchestration import create_orchestration_plan

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        mock_container = MagicMock()
        mock_container.status.value = "running"

        with patch(
            "routers.test_orchestration.validate_and_get_container_for_tenant",
            return_value=mock_container,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await create_orchestration_plan(
                    container_name="c1",
                    suite_ids=["nonexistent"],
                    current_user=mock_user,
                    tenant_id="n8n-dev",
                )
            assert exc_info.value.status_code == 404
            assert "nonexistent" in str(exc_info.value.detail)


class TestExecuteOrchestrationPlanAdvanced:
    """Additional tests for execute_orchestration_plan."""

    async def test_wrong_tenant(self, pg_db):
        """Test wrong tenant raises 403."""
        from routers.test_orchestration import execute_orchestration_plan

        plan = TestOrchestrationPlan("p1", "c1", "tenant-a")
        await get_orchestration_store().save_plan(plan)

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_bg = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await execute_orchestration_plan(
                plan_id="p1",
                background_tasks=mock_bg,
                current_user=mock_user,
                tenant_id="tenant-b",
            )
        assert exc_info.value.status_code == 403
        assert "Access denied" in str(exc_info.value.detail)

    async def test_already_executing(self, pg_db):
        """Test already executing plan raises 400."""
        from routers.test_orchestration import execute_orchestration_plan

        store = get_orchestration_store()

        suite = OrchestrationTestSuite("s1", "Suite", "desc", "c1", "n8n-dev")
        suite.tests = [_req(container_name="c1")]
        await store.save_suite(suite)

        plan = TestOrchestrationPlan("p1", "c1", "n8n-dev")
        plan.status = "executing"
        plan.suites = [suite]
        await store.save_plan(plan)

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_bg = MagicMock()

        mock_container = MagicMock()
        mock_container.status.value = "running"

        with patch(
            "routers.test_orchestration.validate_and_get_container_for_tenant",
            return_value=mock_container,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await execute_orchestration_plan(
                    plan_id="p1",
                    background_tasks=mock_bg,
                    current_user=mock_user,
                    tenant_id="n8n-dev",
                )
            assert exc_info.value.status_code == 400
            assert "already executing" in str(exc_info.value.detail)


class TestReadPathTenantScoping:
    """list_suites/list_plans/list_history filter by tenant in SQL — a caller
    only ever sees its own rows, never another tenant's."""

    async def test_list_suites_scoped_to_tenant(self, pg_db):
        store = get_orchestration_store()
        a = OrchestrationTestSuite("ta1", "A", "d", "c1", "tenant-a")
        a.tests = [_req(container_name="c1")]
        b = OrchestrationTestSuite("tb1", "B", "d", "c1", "tenant-b")
        b.tests = [_req(container_name="c1")]
        await store.save_suite(a)
        await store.save_suite(b)

        a_ids = {s.suite_id for s in await store.list_suites("tenant-a")}
        b_ids = {s.suite_id for s in await store.list_suites("tenant-b")}
        assert a_ids == {"ta1"}
        assert b_ids == {"tb1"}
        assert not await store.list_suites("tenant-c")

    async def test_list_history_scoped_to_tenant(self, pg_db):
        store = get_orchestration_store()
        await store.append_history({"type": "suite", "tenant_id": "tenant-a", "timestamp": None})
        await store.append_history({"type": "plan", "tenant_id": "tenant-b", "timestamp": None})

        a = await store.list_history("tenant-a")
        assert len(a) == 1 and a[0]["tenant_id"] == "tenant-a"
        assert await store.list_history("tenant-b") != a
        assert await store.list_history("tenant-c") == []

    async def test_list_plans_scoped_and_batches_suites(self, pg_db):
        store = get_orchestration_store()
        s1 = OrchestrationTestSuite("ps-a1", "S", "d", "c1", "tenant-a")
        s1.tests = [_req(container_name="c1")]
        s2 = OrchestrationTestSuite("ps-a2", "S", "d", "c1", "tenant-a")
        s2.tests = [_req(container_name="c1")]
        await store.save_suite(s1)
        await store.save_suite(s2)
        plan = TestOrchestrationPlan("pa1", "c1", "tenant-a")
        plan.suites = [s1, s2]
        await store.save_plan(plan)
        other = TestOrchestrationPlan("pb1", "c1", "tenant-b")
        await store.save_plan(other)

        a_plans = await store.list_plans("tenant-a")
        assert {p.plan_id for p in a_plans} == {"pa1"}
        # member suites rehydrated (batched) — not empty
        assert {s.suite_id for s in a_plans[0].suites} == {"ps-a1", "ps-a2"}
        assert {p.plan_id for p in await store.list_plans("tenant-b")} == {"pb1"}

    async def test_get_plan_batch_rehydrates_member_suites(self, pg_db):
        store = get_orchestration_store()
        s1 = OrchestrationTestSuite("gp-s1", "S", "d", "c1", "tenant-a")
        s1.tests = [_req(container_name="c1")]
        await store.save_suite(s1)
        plan = TestOrchestrationPlan("gp1", "c1", "tenant-a")
        plan.suites = [s1]
        await store.save_plan(plan)

        loaded = await store.get_plan("gp1")
        assert loaded is not None
        assert {s.suite_id for s in loaded.suites} == {"gp-s1"}
