Schema-parity fixture (py2rust A5a).

- `source/migrations/versions/0001_create_tags.py`, `0002_create_notes.py` and
  `script.py.mako` are copied VERBATIM from the private fixture
  `conveyor-coding/py2rs-fixture-svc` @ `9aad739`. Only `env.py` and
  `alembic.ini` are replaced: the fixture's `env.py` imports its own package
  for `database_url()`; this one reads `DATABASE_URL` directly (same psycopg v3
  driver forcing) so the migrations run without installing that package.
- `target/migrations/*.sql` is the equivalent plain-SQL set a Rust sqlx port
  would carry (`conveyor-coding/py2rs-fixture-svc-rs` is still empty).
- `apply_sql_migrations.py` applies `target/migrations/*.sql` in order and
  records them in `_sqlx_migrations`, the way `sqlx migrate run` does.
