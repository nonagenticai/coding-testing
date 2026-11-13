"""Apply ``<dir>/*.sql`` in order and record them like ``sqlx migrate run``.

Usage: DATABASE_URL=... python apply_sql_migrations.py <migrations_dir>
"""

import hashlib
import os
import re
import sys
from pathlib import Path

import psycopg


def main(migrations_dir: str) -> None:
    url = re.sub(r"^postgres(?:ql)?\+[A-Za-z0-9_]+://", "postgresql://", os.environ["DATABASE_URL"])
    with psycopg.connect(url) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS _sqlx_migrations ("
            "version bigint PRIMARY KEY, description text NOT NULL, "
            "installed_on timestamptz NOT NULL DEFAULT now(), success boolean NOT NULL, "
            "checksum bytea NOT NULL, execution_time bigint NOT NULL)"
        )
        for path in sorted(Path(migrations_dir).glob("*.sql")):
            version, _, desc = path.stem.partition("_")
            sql = path.read_text()
            conn.execute(sql.encode())
            conn.execute(
                "INSERT INTO _sqlx_migrations (version, description, success, checksum, "
                "execution_time) VALUES (%s, %s, true, %s, 0)",
                (int(version), desc, hashlib.sha384(sql.encode()).digest()),
            )


if __name__ == "__main__":
    main(sys.argv[1])
