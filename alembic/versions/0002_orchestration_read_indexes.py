"""orchestration read-path indexes

Adds expression indexes on ``data ->> 'tenant_id'`` for the three orchestration
tables so the list/metrics/history endpoints — which now filter by tenant in
SQL rather than loading every tenant's rows and filtering in Python — can seek
per-tenant instead of scanning the whole table. The history index is composite
with ``created_at`` to also serve the ordered read (and future retention prune).

Revision ID: 0002_orchestration_read_indexes
Revises: 0001_initial
Create Date: 2026-07-17

"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_orchestration_read_indexes"
down_revision: str | Sequence[str] | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create per-tenant expression indexes on the orchestration tables."""
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_orchestration_test_suite_tenant "
        "ON orchestration_test_suite ((data ->> 'tenant_id'))"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_orchestration_plan_tenant "
        "ON orchestration_plan ((data ->> 'tenant_id'))"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_orchestration_execution_history_tenant_created "
        "ON orchestration_execution_history ((data ->> 'tenant_id'), created_at)"
    )


def downgrade() -> None:
    """Drop the read-path indexes."""
    op.execute("DROP INDEX IF EXISTS ix_orchestration_execution_history_tenant_created")
    op.execute("DROP INDEX IF EXISTS ix_orchestration_plan_tenant")
    op.execute("DROP INDEX IF EXISTS ix_orchestration_test_suite_tenant")
