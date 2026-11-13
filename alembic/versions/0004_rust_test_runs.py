"""rust_test_runs: one row per Rust workspace test run

A Rust test run packs the workspace's working tree (tar.gz, <= 50 MiB) and runs it
in a throwaway k8s Job, which downloads it ONCE from
GET /internal/test-runs/{run_id}/archive with a per-run token (only its sha256 is
stored). ``archive`` is cleared by that download or when the run ends; ``result``
holds the mapped UnifiedTestResult so any replica answers an async poll. Rows are
pruned by tasks.prune_history.

Revision ID: 0004_rust_test_runs
Revises: 0003_test_results_card_context
Create Date: 2026-10-29

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004_rust_test_runs"
down_revision: str | Sequence[str] | None = "0003_test_results_card_context"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create rust_test_runs."""
    op.create_table(
        "rust_test_runs",
        sa.Column("run_id", sa.Text(), primary_key=True),
        sa.Column("container_name", sa.Text(), nullable=False),
        sa.Column("tenant_id", sa.Text(), nullable=True),
        sa.Column("token_sha256", sa.Text(), nullable=False),
        sa.Column("archive", sa.LargeBinary(), nullable=True),
        sa.Column("result", JSONB(), nullable=True),
        sa.Column("deadline", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_rust_test_runs_created", "rust_test_runs", ["created_at"])


def downgrade() -> None:
    """Drop rust_test_runs."""
    op.drop_index("ix_rust_test_runs_created", table_name="rust_test_runs")
    op.drop_table("rust_test_runs")
