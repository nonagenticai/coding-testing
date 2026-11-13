"""
Orchestration Store

Postgres-backed persistence for test-suite / orchestration-plan / execution-history
state, replacing the former module-level dicts in ``routers.test_orchestration``.
Because the state lives in a shared database rather than process memory, the
service is safe to run with more than one replica: a request that lands on
replica B sees suites/plans/history written by replica A.

Mirrors ``services.test_results_storage_service`` (async, module-singleton getter,
``factory = await ensure_session_factory(); async with factory() as session: ...``).
Each domain object is stored as a single ``data: JSONB`` blob keyed by its id.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import select, text

from config.settings import get_settings
from db import ensure_session_factory
from db_models import (
    OrchestrationExecutionHistoryRow,
    OrchestrationPlanRow,
    OrchestrationTestSuiteRow,
)
from services.orchestration_models import (
    OrchestrationTestSuite,
    TestOrchestrationPlan,
    plan_from_data,
    plan_to_data,
    suite_from_data,
    suite_to_data,
)

if TYPE_CHECKING:
    from services.test_execution.test_orchestrator import TestExecutionResult

logger = logging.getLogger(__name__)


def _iso(value: Any) -> Any:
    """Render a datetime as an ISO string (JSONB has no datetime type)."""
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _parse_history_timestamps(entry: dict[str, Any]) -> dict[str, Any]:
    """Parse a history entry's ``timestamp`` back to an aware datetime.

    Endpoints compare (``>= since``) and sort on ``timestamp`` as a datetime, so
    it must be rehydrated from its stored ISO string.
    """
    ts = entry.get("timestamp")
    if isinstance(ts, str):
        parsed = datetime.fromisoformat(ts)
        entry = {**entry, "timestamp": parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)}
    return entry


class OrchestrationStore:
    """Async Postgres store for orchestration suites, plans, and history."""

    # --- Suites --------------------------------------------------------------

    async def save_suite(self, suite: OrchestrationTestSuite) -> None:
        """Upsert a test suite by ``suite_id``."""
        factory = await ensure_session_factory()
        data = suite_to_data(suite)
        async with factory() as session:
            row = await session.get(OrchestrationTestSuiteRow, suite.suite_id)
            if row is None:
                session.add(OrchestrationTestSuiteRow(suite_id=suite.suite_id, data=data))
            else:
                row.data = data
                row.updated_at = datetime.now(UTC)
            await session.commit()

    async def get_suite(self, suite_id: str) -> OrchestrationTestSuite | None:
        """Load a single test suite, or ``None`` if it does not exist."""
        factory = await ensure_session_factory()
        async with factory() as session:
            row = await session.get(OrchestrationTestSuiteRow, suite_id)
            if row is None:
                return None
            return suite_from_data(row.data)

    async def list_suites(self, tenant_id: str) -> list[OrchestrationTestSuite]:
        """Load a tenant's test suites.

        Filters by ``tenant_id`` in SQL (backed by ``ix_orchestration_test_suite_tenant``)
        so a caller only loads and deserializes its own rows, never every tenant's.
        The endpoint still applies its container/status/language filters in Python
        on this already-scoped set.
        """
        factory = await ensure_session_factory()
        async with factory() as session:
            rows = (
                (
                    await session.execute(
                        select(OrchestrationTestSuiteRow).where(
                            OrchestrationTestSuiteRow.data["tenant_id"].astext == tenant_id
                        )
                    )
                )
                .scalars()
                .all()
            )
        return [suite_from_data(row.data) for row in rows]

    async def _load_suites_by_ids(
        self, session: Any, suite_ids: list[str]
    ) -> dict[str, OrchestrationTestSuite]:
        """Batch-load suites by id within an existing session (one round-trip).

        Replaces the per-suite ``get_suite`` calls (each of which opened its own
        session) that made plan rehydration an N+1 connection storm.
        """
        if not suite_ids:
            return {}
        rows = (
            (
                await session.execute(
                    select(OrchestrationTestSuiteRow).where(
                        OrchestrationTestSuiteRow.suite_id.in_(suite_ids)
                    )
                )
            )
            .scalars()
            .all()
        )
        return {row.suite_id: suite_from_data(row.data) for row in rows}

    async def try_mark_suite_running(
        self, suite_id: str, tenant_id: str, *, stale_running_seconds: int | None = None
    ) -> bool:
        """Atomically transition a suite ``pending/…`` → ``running``.

        A compare-and-swap: only the caller whose UPDATE actually flips the status
        gets ``True`` (and should schedule execution). Concurrent callers — on this
        replica or another — see the row already ``running`` and get ``False`` (→ the
        "already running" 400). Replaces the former check-then-set guard, which was
        atomic only within a single process and races across replicas after the port.

        Self-heal: a row also wins if it has been ``running`` longer than
        ``stale_running_seconds`` (default ``ORCHESTRATION_STALE_RUNNING_SECONDS``).
        Without this, a status flag wedged by an out-of-band crash (OOM / node loss
        mid-run, before the completion write) would block re-execution forever now
        that the flag is durable. The reclaim is logged (a silent self-heal hides a
        recurring crash). A live long-running execution renews ``updated_at`` via
        :meth:`heartbeat_suite` (see ``execute_suite_async``), so only a genuinely
        dead run ever crosses the threshold — a slow-but-alive suite is never stolen.
        """
        if stale_running_seconds is None:
            stale_running_seconds = get_settings().ORCHESTRATION_STALE_RUNNING_SECONDS
        factory = await ensure_session_factory()
        async with factory() as session:
            result = await session.execute(
                text(
                    # The eligibility check (status/staleness) MUST reference the
                    # LIVE table columns (s.data / s.updated_at), not the CTE
                    # snapshot: under READ COMMITTED, a concurrent CAS that loses the
                    # row lock re-evaluates this qual against the just-committed row
                    # (EvalPlanQual). If the qual read the CTE's stale snapshot both
                    # racers would pass and double-execute; reading s.* makes the
                    # loser see status='running'/fresh updated_at and correctly fail.
                    # The CTE exists only to RETURN the prior values for the log line.
                    "WITH current AS ("
                    "  SELECT suite_id, data->>'status' AS old_status, updated_at AS old_updated_at "
                    "  FROM orchestration_test_suite "
                    "  WHERE suite_id = :suite_id AND data->>'tenant_id' = :tenant_id"
                    ") "
                    "UPDATE orchestration_test_suite AS s "
                    "SET data = jsonb_set(s.data, '{status}', '\"running\"'), "
                    "updated_at = now() "
                    "FROM current "
                    "WHERE s.suite_id = current.suite_id "
                    "AND (s.data->>'status' <> 'running' "
                    "     OR s.updated_at < now() - make_interval(secs => :stale_secs)) "
                    "RETURNING current.old_status AS old_status, "
                    "current.old_updated_at AS old_updated_at"
                ),
                {
                    "suite_id": suite_id,
                    "tenant_id": tenant_id,
                    "stale_secs": stale_running_seconds,
                },
            )
            row = result.first()
            await session.commit()
        if row is not None and row.old_status == "running":
            logger.warning(
                "Reclaimed stale 'running' suite %s (stuck since %s, > %ss threshold)",
                suite_id,
                row.old_updated_at,
                stale_running_seconds,
            )
        return row is not None

    async def heartbeat_suite(self, suite_id: str) -> bool:
        """Renew a running suite's lease by bumping ``updated_at`` to now.

        Called periodically by the executor so a legitimately long run keeps its
        row fresh and is never mistaken for a crashed one by the stale-flag
        reclaim in :meth:`try_mark_suite_running`. Only a row still ``running`` is
        touched (a completed/failed/reclaimed-away suite is left alone). Returns
        True if a row was renewed.
        """
        factory = await ensure_session_factory()
        async with factory() as session:
            result = await session.execute(
                text(
                    "UPDATE orchestration_test_suite SET updated_at = now() "
                    "WHERE suite_id = :suite_id AND data->>'status' = 'running' "
                    "RETURNING suite_id"
                ),
                {"suite_id": suite_id},
            )
            renewed = result.scalar_one_or_none() is not None
            await session.commit()
        return renewed

    async def save_suite_execution_result(
        self, suite_id: str, *, results: list[TestExecutionResult], status: str
    ) -> bool:
        """Persist ONLY the execution-owned fields (results + status) of a suite.

        Called by execute_suite_async at run end INSTEAD of save_suite. save_suite
        rewrites the whole `data` blob from the in-memory snapshot captured when the
        run was scheduled; over a multi-minute run that snapshot goes stale, so a
        full rewrite silently reverts any field a concurrent writer changed on the
        row. This merges results+status into the LIVE row via jsonb_set (mirroring
        the status CAS in try_mark_suite_running) — atomic, single statement, no
        read-modify-write, no row lock — leaving every other key at its DB value.
        Returns True if a row was updated (False if the suite was deleted mid-run).
        """
        serialized = json.dumps([r.model_dump(mode="json") for r in results])
        factory = await ensure_session_factory()
        async with factory() as session:
            result = await session.execute(
                text(
                    "UPDATE orchestration_test_suite SET "
                    "data = jsonb_set("
                    "  jsonb_set(data, '{results}', :results ::jsonb), "
                    "  '{status}', to_jsonb(:status ::text)"
                    "), "
                    "updated_at = now() "
                    "WHERE suite_id = :suite_id "
                    "RETURNING suite_id"
                ),
                {"results": serialized, "status": status, "suite_id": suite_id},
            )
            updated = result.scalar_one_or_none() is not None
            await session.commit()
        return updated

    # --- Plans ---------------------------------------------------------------

    async def save_plan(self, plan: TestOrchestrationPlan) -> None:
        """Upsert an orchestration plan by ``plan_id`` (suites stored as references)."""
        factory = await ensure_session_factory()
        data = plan_to_data(plan)
        async with factory() as session:
            row = await session.get(OrchestrationPlanRow, plan.plan_id)
            if row is None:
                session.add(OrchestrationPlanRow(plan_id=plan.plan_id, data=data))
            else:
                row.data = data
                row.updated_at = datetime.now(UTC)
            await session.commit()

    async def get_plan(self, plan_id: str) -> TestOrchestrationPlan | None:
        """Load a plan, rehydrating its member suites from the suite table."""
        factory = await ensure_session_factory()
        async with factory() as session:
            row = await session.get(OrchestrationPlanRow, plan_id)
            if row is None:
                return None
            data = row.data
            # Batch-load member suites in the same session (was an N+1 of one
            # session per suite via get_suite).
            loaded = await self._load_suites_by_ids(session, data.get("suite_ids", []))
        return plan_from_data(data, loaded.get)

    async def try_mark_plan_executing(
        self, plan_id: str, tenant_id: str, *, stale_running_seconds: int | None = None
    ) -> bool:
        """Atomically transition a plan ``draft/…`` → ``executing`` (CAS, see suite).

        Self-heals a wedged ``executing`` flag past ``stale_running_seconds`` exactly
        as :meth:`try_mark_suite_running` does for suites.
        """
        if stale_running_seconds is None:
            stale_running_seconds = get_settings().ORCHESTRATION_STALE_RUNNING_SECONDS
        factory = await ensure_session_factory()
        async with factory() as session:
            result = await session.execute(
                text(
                    # Eligibility on LIVE p.* columns, not the CTE snapshot — see the
                    # concurrency note in try_mark_suite_running.
                    "WITH current AS ("
                    "  SELECT plan_id, data->>'status' AS old_status, updated_at AS old_updated_at "
                    "  FROM orchestration_plan "
                    "  WHERE plan_id = :plan_id AND data->>'tenant_id' = :tenant_id"
                    ") "
                    "UPDATE orchestration_plan AS p "
                    "SET data = jsonb_set(p.data, '{status}', '\"executing\"'), "
                    "updated_at = now() "
                    "FROM current "
                    "WHERE p.plan_id = current.plan_id "
                    "AND (p.data->>'status' <> 'executing' "
                    "     OR p.updated_at < now() - make_interval(secs => :stale_secs)) "
                    "RETURNING current.old_status AS old_status, "
                    "current.old_updated_at AS old_updated_at"
                ),
                {
                    "plan_id": plan_id,
                    "tenant_id": tenant_id,
                    "stale_secs": stale_running_seconds,
                },
            )
            row = result.first()
            await session.commit()
        if row is not None and row.old_status == "executing":
            logger.warning(
                "Reclaimed stale 'executing' plan %s (stuck since %s, > %ss threshold)",
                plan_id,
                row.old_updated_at,
                stale_running_seconds,
            )
        return row is not None

    async def heartbeat_plan(self, plan_id: str) -> bool:
        """Renew an executing plan's lease (``updated_at``); see
        :meth:`heartbeat_suite`. Only touches a row still ``executing``."""
        factory = await ensure_session_factory()
        async with factory() as session:
            result = await session.execute(
                text(
                    "UPDATE orchestration_plan SET updated_at = now() "
                    "WHERE plan_id = :plan_id AND data->>'status' = 'executing' "
                    "RETURNING plan_id"
                ),
                {"plan_id": plan_id},
            )
            renewed = result.scalar_one_or_none() is not None
            await session.commit()
        return renewed

    async def list_plans(self, tenant_id: str) -> list[TestOrchestrationPlan]:
        """Load a tenant's plans, member suites rehydrated per plan.

        Filters by ``tenant_id`` in SQL (``ix_orchestration_plan_tenant``) and
        batch-loads each plan's member suites within the shared session (no more
        one-session-per-suite N+1).
        """
        factory = await ensure_session_factory()
        plans: list[TestOrchestrationPlan] = []
        async with factory() as session:
            rows = (
                (
                    await session.execute(
                        select(OrchestrationPlanRow).where(
                            OrchestrationPlanRow.data["tenant_id"].astext == tenant_id
                        )
                    )
                )
                .scalars()
                .all()
            )
            for row in rows:
                data = row.data
                loaded = await self._load_suites_by_ids(session, data.get("suite_ids", []))
                plans.append(plan_from_data(data, loaded.get))
        return plans

    # --- History -------------------------------------------------------------

    async def append_history(self, entry: dict[str, Any]) -> None:
        """Append one execution-history record (``timestamp`` stored as ISO)."""
        factory = await ensure_session_factory()
        stored = {**entry, "timestamp": _iso(entry.get("timestamp"))}
        async with factory() as session:
            session.add(OrchestrationExecutionHistoryRow(data=stored))
            await session.commit()

    async def list_history(self, tenant_id: str) -> list[dict[str, Any]]:
        """Load a tenant's history entries (oldest-first by insertion), timestamps rehydrated.

        Filters by ``tenant_id`` in SQL (``ix_orchestration_execution_history_tenant_created``)
        so a caller never loads every tenant's history. The endpoint still applies
        its container/``since``/``limit`` filters in Python on this scoped set.
        """
        factory = await ensure_session_factory()
        async with factory() as session:
            rows = (
                (
                    await session.execute(
                        select(OrchestrationExecutionHistoryRow)
                        .where(
                            OrchestrationExecutionHistoryRow.data["tenant_id"].astext == tenant_id
                        )
                        .order_by(OrchestrationExecutionHistoryRow.created_at)
                    )
                )
                .scalars()
                .all()
            )
        return [_parse_history_timestamps(row.data) for row in rows]

    async def prune_history(self, retention_days: int) -> int:
        """Delete execution-history rows older than ``retention_days``; return the count.

        ``orchestration_execution_history`` is append-only (one row per suite/plan
        run) with no natural bound. Run periodically (a gitops CronJob invoking
        ``tasks.prune_history``) to keep it from growing without limit. A
        non-positive ``retention_days`` is a no-op (retention disabled). Prunes
        across all tenants — it is a global maintenance op, not a per-tenant read.
        """
        if retention_days <= 0:
            return 0
        factory = await ensure_session_factory()
        async with factory() as session:
            result = await session.execute(
                text(
                    "DELETE FROM orchestration_execution_history "
                    "WHERE created_at < now() - make_interval(days => :days)"
                ),
                {"days": retention_days},
            )
            await session.commit()
        return result.rowcount


# Global singleton instance
_orchestration_store: OrchestrationStore | None = None


def get_orchestration_store() -> OrchestrationStore:
    """Get or create the global orchestration store instance."""
    global _orchestration_store
    if _orchestration_store is None:
        _orchestration_store = OrchestrationStore()
    return _orchestration_store
