"""create tags

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tags",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("name", sa.String(64), nullable=False),
        sa.UniqueConstraint("tenant_id", "name", name="uq_tags_tenant_name"),
    )


def downgrade() -> None:
    op.drop_table("tags")
