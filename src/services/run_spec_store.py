"""Service-side store of workspace run specs.

A workspace run (behaviour / perf / schema / OpenAPI parity, python workspace test
runs) is graded at poll time from its SPEC: the operation, the corpora, the
``recording_run_id`` to read a baseline from, the JUnit / coverage paths. The spec
is stored HERE when the run starts, keyed by the run's nonce and bound to the
workspace (container) and tenant that started it, so a poll from another
workspace or tenant finds nothing, and any replica can answer the poll.

A spec carries no secret values: the target's auth headers live only in the
driver config the workspace pod already holds. So nothing is sealed and no key
is needed.

``tenant_id`` is compared with ``IS NOT DISTINCT FROM``: a run started without a
tenant is visible only to tenantless polls. Rows are pruned by
``tasks.prune_history`` (``RUN_SPEC_RETENTION_DAYS``).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import delete, select

from db import ensure_session_factory
from db_models import WorkspaceRunSpecRow


class RunSpecStore:
    """Async Postgres store of run specs, one row per run nonce."""

    async def put(
        self, nonce: str, kind: str, container: str, tenant_id: str | None, spec: dict[str, Any]
    ) -> None:
        factory = await ensure_session_factory()
        async with factory() as session:
            session.add(
                WorkspaceRunSpecRow(
                    nonce=nonce, kind=kind, container_name=container, tenant_id=tenant_id, spec=spec
                )
            )
            await session.commit()

    async def get(
        self, nonce: str, kinds: tuple[str, ...], container: str, tenant_id: str | None
    ) -> dict[str, Any] | None:
        """The spec of run ``nonce`` when it was started as one of ``kinds`` in
        ``container`` by ``tenant_id``; None otherwise."""
        factory = await ensure_session_factory()
        async with factory() as session:
            spec = (
                await session.execute(
                    select(WorkspaceRunSpecRow.spec).where(
                        WorkspaceRunSpecRow.nonce == nonce,
                        WorkspaceRunSpecRow.kind.in_(kinds),
                        WorkspaceRunSpecRow.container_name == container,
                        WorkspaceRunSpecRow.tenant_id.is_not_distinct_from(tenant_id),
                    )
                )
            ).scalar_one_or_none()
        return spec if isinstance(spec, dict) else None

    async def prune(self, retention_days: int) -> int:
        """Delete specs older than ``retention_days`` (<= 0: disabled); returns the count."""
        if retention_days <= 0:
            return 0
        factory = await ensure_session_factory()
        async with factory() as session:
            result = await session.execute(
                delete(WorkspaceRunSpecRow).where(
                    WorkspaceRunSpecRow.created_at
                    < datetime.now(UTC) - timedelta(days=retention_days)
                )
            )
            await session.commit()
        return result.rowcount


_store: RunSpecStore | None = None


def get_run_spec_store() -> RunSpecStore:
    global _store  # noqa: PLW0603
    if _store is None:
        _store = RunSpecStore()
    return _store
