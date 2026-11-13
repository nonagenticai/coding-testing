"""workspace_run_specs: run specs kept service-side

Every workspace run (behaviour / perf / schema / OpenAPI parity, python workspace
test runs) is graded from its spec: the operation, the corpora, the
``recording_run_id`` to read a baseline from, the JUnit / coverage paths. It is
stored here when the run starts, bound to the workspace and tenant that started
it, instead of as a ``spec.json`` in the workspace pod. A spec holds no secret
values (target auth headers stay in the pod's driver config only). Rows are
pruned by tasks.prune_history (``RUN_SPEC_RETENTION_DAYS``).

Revision ID: 0005_workspace_run_specs
Revises: 0004_rust_test_runs
Create Date: 2026-10-30

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0005_workspace_run_specs"
down_revision: str | Sequence[str] | None = "0004_rust_test_runs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create workspace_run_specs + its retention index."""
    op.create_table(
        "workspace_run_specs",
        sa.Column("nonce", sa.Text(), primary_key=True),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("container_name", sa.Text(), nullable=False),
        sa.Column("tenant_id", sa.Text(), nullable=True),
        sa.Column("spec", JSONB(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_workspace_run_specs_created", "workspace_run_specs", ["created_at"])


def downgrade() -> None:
    """Drop workspace_run_specs."""
    op.drop_index("ix_workspace_run_specs_created", table_name="workspace_run_specs")
    op.drop_table("workspace_run_specs")
