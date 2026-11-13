"""test_results card context columns (R10-A2 computed results producer)

Adds the columns that let a test_results row be attributed to the conveyor card
whose QA produced it, plus a ``source`` discriminator so the externally computed
rows (``aa_qa``) are distinguishable from rows written by this service's own
``POST /api/tests/execute`` (``execute``, the server default — every existing
row and every existing writer keeps that value without a code change).

All card columns are NULLABLE: the execute path has no card, and ``project_id``
is filled only once the QA context carries it.

``tenant_id`` is TEXT, not UUID, deliberately: everywhere else in this service a
tenant id is a plain string (orchestration rows store ``data ->> 'tenant_id'``,
every route types it ``str``), and the configured default tenant is not
guaranteed to be a UUID. A UUID column would turn a non-UUID tenant into a 500
on write.

Revision ID: 0003_test_results_card_context
Revises: 0002_orchestration_read_indexes
Create Date: 2026-10-04

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003_test_results_card_context"
down_revision: str | Sequence[str] | None = "0002_orchestration_read_indexes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NULLABLE_TEXT_COLUMNS = ("card_id", "tenant_id", "project_id", "commit_sha", "correlation_id")


def upgrade() -> None:
    """Add card-context columns + read indexes to test_results."""
    for name in _NULLABLE_TEXT_COLUMNS:
        op.add_column("test_results", sa.Column(name, sa.Text(), nullable=True))
    op.add_column(
        "test_results",
        sa.Column("source", sa.Text(), nullable=False, server_default=sa.text("'execute'")),
    )
    op.create_index("ix_test_results_card_id", "test_results", ["card_id"])
    op.create_index(
        "ix_test_results_container_created", "test_results", ["container_name", "created_at"]
    )


def downgrade() -> None:
    """Drop the indexes and columns added by upgrade()."""
    op.drop_index("ix_test_results_container_created", table_name="test_results")
    op.drop_index("ix_test_results_card_id", table_name="test_results")
    op.drop_column("test_results", "source")
    for name in reversed(_NULLABLE_TEXT_COLUMNS):
        op.drop_column("test_results", name)
