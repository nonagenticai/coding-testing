"""initial — results store + orchestration store

Creates the shared Postgres schema that replaces the node-local per-container
SQLite result files and (forward-ready) the module-level orchestration dicts,
so the service is safe to run with >1 replica.

Revision ID: 0001_initial
Revises:
Create Date: 2026-07-16

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001_initial"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create results-store and orchestration-store tables."""
    op.create_table(
        "test_results",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("test_id", sa.Text(), nullable=False),
        sa.Column("container_name", sa.Text(), nullable=False),
        sa.Column("container_id", sa.Text(), nullable=True),
        sa.Column("test_name", sa.Text(), nullable=False),
        sa.Column("language", sa.Text(), nullable=False),
        sa.Column("test_type", sa.Text(), nullable=False),
        sa.Column("test_framework", sa.Text(), nullable=True),
        sa.Column("exit_code", sa.Integer(), nullable=False),
        sa.Column("stdout", sa.Text(), nullable=False),
        sa.Column("stderr", sa.Text(), nullable=False),
        sa.Column("execution_time_ms", sa.Double(), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("source_code", sa.Text(), nullable=True),
        sa.Column("test_code", sa.Text(), nullable=True),
        sa.Column("file_path", sa.Text(), nullable=True),
        sa.Column("dependencies", postgresql.JSONB(), nullable=True),
        sa.Column("resource_usage", postgresql.JSONB(), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("test_id", name="uq_test_results_test_id"),
    )
    op.create_index("idx_test_results_test_id", "test_results", ["test_id"])
    op.create_index("idx_test_results_container", "test_results", ["container_name"])
    op.create_index("idx_test_results_test_name", "test_results", ["test_name"])
    op.create_index(
        "idx_test_results_timestamp",
        "test_results",
        [sa.text("timestamp DESC")],
    )
    op.create_index("idx_test_results_success", "test_results", ["success"])
    op.create_index("idx_test_results_language", "test_results", ["language"])

    op.create_table(
        "analysis_results",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("analysis_id", sa.Text(), nullable=False),
        sa.Column("test_id", sa.Text(), nullable=False),
        sa.Column("container_name", sa.Text(), nullable=False),
        sa.Column("root_causes", postgresql.JSONB(), nullable=False),
        sa.Column("fix_suggestions", postgresql.JSONB(), nullable=False),
        sa.Column("detected_patterns", postgresql.JSONB(), nullable=False),
        sa.Column("analysis_metadata", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("analysis_id", name="uq_analysis_results_analysis_id"),
        sa.ForeignKeyConstraint(
            ["test_id"],
            ["test_results.test_id"],
            name="fk_analysis_results_test_id",
            ondelete="CASCADE",
        ),
    )
    op.create_index("idx_analysis_results_test_id", "analysis_results", ["test_id"])
    op.create_index("idx_analysis_results_analysis_id", "analysis_results", ["analysis_id"])

    op.create_table(
        "orchestration_test_suite",
        sa.Column("suite_id", sa.Text(), primary_key=True),
        sa.Column("data", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_table(
        "orchestration_plan",
        sa.Column("plan_id", sa.Text(), primary_key=True),
        sa.Column("data", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_table(
        "orchestration_execution_history",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("data", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )


def downgrade() -> None:
    """Drop all coding-testing tables."""
    op.drop_table("orchestration_execution_history")
    op.drop_table("orchestration_plan")
    op.drop_table("orchestration_test_suite")
    op.drop_index("idx_analysis_results_analysis_id", table_name="analysis_results")
    op.drop_index("idx_analysis_results_test_id", table_name="analysis_results")
    op.drop_table("analysis_results")
    op.drop_index("idx_test_results_language", table_name="test_results")
    op.drop_index("idx_test_results_success", table_name="test_results")
    op.drop_index("idx_test_results_timestamp", table_name="test_results")
    op.drop_index("idx_test_results_test_name", table_name="test_results")
    op.drop_index("idx_test_results_container", table_name="test_results")
    op.drop_index("idx_test_results_test_id", table_name="test_results")
    op.drop_table("test_results")
