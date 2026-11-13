"""Alembic environment: URL from DATABASE_URL (psycopg v3 driver forced).

Stand-in for the fixture's env.py, which imports its package for database_url().
"""

import os
import re

from alembic import context
from sqlalchemy import create_engine


def database_url() -> str:
    url = os.environ["DATABASE_URL"]
    return re.sub(r"^postgres(?:ql)?(?:\+[A-Za-z0-9_]+)?://", "postgresql+psycopg://", url)


def run_migrations_online() -> None:
    engine = create_engine(database_url())
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


run_migrations_online()
