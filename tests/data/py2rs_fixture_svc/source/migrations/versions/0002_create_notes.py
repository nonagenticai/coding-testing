"""create notes (FK to tags + non-unique tenant/tag index)

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "notes",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column(
            "tag_id",
            sa.Uuid(),
            sa.ForeignKey("tags.id", ondelete="RESTRICT", name="fk_notes_tag_id_tags"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    # Non-unique index; the planted-bug Rust variant drops this one.
    op.create_index("ix_notes_tenant_tag", "notes", ["tenant_id", "tag_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_notes_tenant_tag", table_name="notes")
    op.drop_table("notes")
