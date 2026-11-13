"""Read a run's stored spec (services.run_spec_store) by nonce alone, for assertions."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from db import ensure_session_factory
from db_models import WorkspaceRunSpecRow


async def stored_spec(nonce: str) -> dict[str, Any] | None:
    factory = await ensure_session_factory()
    async with factory() as session:
        return (
            await session.execute(
                select(WorkspaceRunSpecRow.spec).where(WorkspaceRunSpecRow.nonce == nonce)
            )
        ).scalar_one_or_none()
