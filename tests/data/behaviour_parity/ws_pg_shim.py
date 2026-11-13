"""Local stand-in for the workspace's `ws-pg` (py2rust A1) in the A5c driver tests.

Same stdout contract: `ws_pg_shim.py createdb <name>` creates a FRESH database on the
test's ephemeral postgres:18 (admin URL in WS_PG_SHIM_ADMIN_URL) and prints ONLY its URL;
an existing name fails non-zero, like ws-pg.
"""

import os
import re
import sys

import psycopg


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] != "createdb":
        print("usage: ws_pg_shim.py createdb <name>", file=sys.stderr)
        return 2
    name = sys.argv[2]
    if not re.fullmatch(r"[A-Za-z0-9_]{1,63}", name):
        print(f"ws-pg: invalid name {name!r}", file=sys.stderr)
        return 1
    admin = os.environ["WS_PG_SHIM_ADMIN_URL"]
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{name}"')
    print(admin.rsplit("/", 1)[0] + "/" + name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
