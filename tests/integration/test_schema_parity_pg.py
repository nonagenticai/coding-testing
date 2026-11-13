"""Schema parity against a real Postgres 18 (py2rust A5a).

Source = the real py2rs-fixture-svc Alembic migrations (copied verbatim, see
tests/data/py2rs_fixture_svc/README.md). Target = the equivalent plain-SQL set a
Rust sqlx port would carry. Each test gets fresh databases on the server named
by DATABASE_URL (the per-job ephemeral postgres:18), dropped afterwards.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import psycopg
import pytest
from fastapi.testclient import TestClient

from models.parity_schema_models import SchemaDiffOptions
from services.schema_parity import diff_snapshots, introspect_dsn

FIXTURE = Path(__file__).resolve().parent.parent / "data" / "py2rs_fixture_svc"
SOURCE_DIR = FIXTURE / "source"
TARGET_DIR = FIXTURE / "target"

pytestmark = pytest.mark.integration


def _libpq(url: str) -> str:
    return re.sub(r"^postgres(?:ql)?\+[A-Za-z0-9_]+://", "postgresql://", url)


def _with_db(url: str, dbname: str) -> str:
    return re.sub(r"/[^/?]*(\?|$)", f"/{dbname}\\1", _libpq(url), count=1)


ADMIN_DSN = _libpq(os.environ.get("DATABASE_URL", ""))


@pytest.fixture
def make_db() -> Iterator[Callable[[], str]]:
    created: list[str] = []

    def _make() -> str:
        name = f"parity_t_{uuid.uuid4().hex[:10]}"
        with psycopg.connect(ADMIN_DSN, autocommit=True) as conn:
            conn.execute(f'CREATE DATABASE "{name}"'.encode())
        created.append(name)
        return _with_db(ADMIN_DSN, name)

    yield _make
    with psycopg.connect(ADMIN_DSN, autocommit=True) as conn:
        for name in created:
            conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'.encode())


def _run(argv: list[str], dsn: str, cwd: Path) -> None:
    proc = subprocess.run(  # noqa: S603 - fixed argv, test-only
        argv,
        cwd=cwd,
        env={**os.environ, "DATABASE_URL": dsn},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def migrate_source(dsn: str) -> None:
    _run([sys.executable, "-m", "alembic", "-c", "alembic.ini", "upgrade", "head"], dsn, SOURCE_DIR)


def migrate_target(dsn: str) -> None:
    _run([sys.executable, str(FIXTURE / "apply_sql_migrations.py"), "migrations"], dsn, TARGET_DIR)


def _exec(dsn: str, sql: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.encode())


@pytest.fixture
def migrated(make_db) -> tuple[str, str]:
    src, tgt = make_db(), make_db()
    migrate_source(src)
    migrate_target(tgt)
    return src, tgt


async def _diff(src: str, tgt: str, **opts: Any):
    return diff_snapshots(
        await introspect_dsn(src), await introspect_dsn(tgt), SchemaDiffOptions(**opts)
    )


# ---------------------------------------------------------------------------
# Library against the real fixture
# ---------------------------------------------------------------------------


async def test_two_empty_databases_are_vacuous_not_equal(make_db):
    result = await _diff(make_db(), make_db())
    assert result.computed is True
    assert (result.vacuous, result.equal) == (True, False)


async def test_only_migration_bookkeeping_tables_is_vacuous(make_db):
    src, tgt = make_db(), make_db()
    _exec(src, "CREATE TABLE alembic_version (version_num varchar(32) PRIMARY KEY)")
    _exec(tgt, "CREATE TABLE _sqlx_migrations (version bigint PRIMARY KEY, description text)")
    result = await _diff(src, tgt)
    assert (result.vacuous, result.equal) == (True, False)


async def test_alembic_vs_equivalent_sql_is_equal(migrated):
    src, tgt = migrated
    result = await _diff(src, tgt)
    assert result.computed is True
    assert result.equal is True, result.model_dump()
    assert result.name_only_differences == []
    counts = result.source.object_counts  # type: ignore[union-attr]
    assert counts["table"] == 2
    assert counts["column"] == 9
    assert counts["primary_key"] == 2
    assert counts["unique"] == 1
    assert counts["foreign_key"] == 1
    assert counts["index"] == 1
    assert result.source.object_counts == result.target.object_counts  # type: ignore[union-attr]
    assert result.source.server_version.startswith("18")  # type: ignore[union-attr]


async def test_bookkeeping_tables_are_real_and_only_ignored_by_option(migrated):
    src, tgt = migrated
    result = await _diff(src, tgt, ignore_tables=[])
    assert result.equal is False
    assert [(d.kind, d.key) for d in result.missing_in_target] == [
        ("table", "public.alembic_version")
    ]
    assert [(d.kind, d.key) for d in result.extra_in_target] == [
        ("table", "public._sqlx_migrations")
    ]


async def test_dropped_index_is_missing_in_target(migrated):
    src, tgt = migrated
    _exec(tgt, "DROP INDEX ix_notes_tenant_tag")
    result = await _diff(src, tgt)
    assert result.equal is False
    assert result.extra_in_target == [] and result.changed == []
    assert len(result.missing_in_target) == 1
    miss = result.missing_in_target[0]
    assert (miss.kind, miss.key, miss.name) == (
        "index",
        "public.notes USING btree (tenant_id, tag_id)",
        "ix_notes_tenant_tag",
    )


async def test_nullability_change_is_a_column_field_change(migrated):
    src, tgt = migrated
    _exec(tgt, "ALTER TABLE notes ALTER COLUMN title DROP NOT NULL")
    result = await _diff(src, tgt)
    assert result.equal is False
    assert result.missing_in_target == [] and result.extra_in_target == []
    assert len(result.changed) == 1
    ch = result.changed[0]
    assert (ch.kind, ch.key) == ("column", "public.notes.title")
    assert [(f.field, f.source, f.target) for f in ch.fields] == [("not_null", True, False)]


async def test_varchar_length_change_is_a_type_change(migrated):
    src, tgt = migrated
    _exec(tgt, "ALTER TABLE tags ALTER COLUMN name TYPE varchar(128)")
    result = await _diff(src, tgt)
    assert result.equal is False
    assert result.missing_in_target == [] and result.extra_in_target == []
    assert len(result.changed) == 1
    ch = result.changed[0]
    assert (ch.kind, ch.key) == ("column", "public.tags.name")
    assert [(f.field, f.source, f.target) for f in ch.fields] == [
        ("type", "character varying(64)", "character varying(128)")
    ]


async def test_fk_without_on_delete_is_detected(migrated):
    src, tgt = migrated
    _exec(
        tgt,
        "ALTER TABLE notes DROP CONSTRAINT fk_notes_tag_id_tags; "
        "ALTER TABLE notes ADD CONSTRAINT fk_notes_tag_id_tags "
        "FOREIGN KEY (tag_id) REFERENCES tags (id)",
    )
    result = await _diff(src, tgt)
    assert result.equal is False
    assert result.missing_in_target == [] and result.extra_in_target == []
    assert len(result.changed) == 1
    ch = result.changed[0]
    assert (ch.kind, ch.key, ch.name) == (
        "foreign_key",
        "public.notes(tag_id) -> public.tags(id)",
        "fk_notes_tag_id_tags",
    )
    assert [(f.field, f.source, f.target) for f in ch.fields] == [
        ("on_delete", "RESTRICT", "NO ACTION")
    ]


async def test_index_rename_only_is_name_only_and_still_equal(migrated):
    src, tgt = migrated
    _exec(tgt, "ALTER INDEX ix_notes_tenant_tag RENAME TO idx_notes_tenant_tag")
    result = await _diff(src, tgt)
    assert result.equal is True
    assert len(result.name_only_differences) == 1
    nd = result.name_only_differences[0]
    assert (nd.kind, nd.source_name, nd.target_name) == (
        "index",
        "ix_notes_tenant_tag",
        "idx_notes_tenant_tag",
    )

    strict = await _diff(src, tgt, strict_names=True)
    assert strict.equal is False
    assert strict.name_only_differences == []
    assert [(f.field, f.source, f.target) for f in strict.changed[0].fields] == [
        ("name", "ix_notes_tenant_tag", "idx_notes_tenant_tag")
    ]


async def test_unique_index_is_not_a_unique_constraint(migrated):
    src, tgt = migrated
    _exec(
        tgt,
        "ALTER TABLE tags DROP CONSTRAINT uq_tags_tenant_name; "
        "CREATE UNIQUE INDEX uq_tags_tenant_name ON tags (tenant_id, name)",
    )
    result = await _diff(src, tgt)
    assert result.equal is False
    assert [(d.kind, d.key) for d in result.missing_in_target] == [
        ("unique", "public.tags(tenant_id, name)")
    ]
    assert [(d.kind, d.key) for d in result.extra_in_target] == [
        ("index", "UNIQUE public.tags USING btree (tenant_id, name)")
    ]


async def test_missing_table_is_one_finding(migrated):
    src, tgt = migrated
    _exec(tgt, "DROP TABLE notes")
    result = await _diff(src, tgt)
    assert [(d.kind, d.key) for d in result.missing_in_target] == [("table", "public.notes")]
    assert result.changed == [] and result.extra_in_target == []


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@pytest.fixture
def dsn_route_enabled(monkeypatch):
    """The DSN route needs PARITY_DSN_DIFF_ENABLED and an allowlisted ENVIRONMENT."""
    from routers import parity_schema

    monkeypatch.setattr(
        parity_schema,
        "get_settings",
        lambda: SimpleNamespace(ENVIRONMENT="test", PARITY_DSN_DIFF_ENABLED=True),
    )


def test_dsn_route_is_refused_without_the_flag(integration_app, migrated):
    src, tgt = migrated
    resp = TestClient(integration_app).post(
        "/api/parity/schema-diff", json={"source_dsn": src, "target_dsn": tgt}
    )
    assert resp.status_code == 403


def test_dsn_route_equal_then_changed(integration_app, migrated, dsn_route_enabled):
    src, tgt = migrated
    client = TestClient(integration_app)
    resp = client.post("/api/parity/schema-diff", json={"source_dsn": src, "target_dsn": tgt})
    assert resp.status_code == 200, resp.text
    assert resp.json()["equal"] is True
    assert resp.json()["computed"] is True

    _exec(tgt, "DROP INDEX ix_notes_tenant_tag")
    resp = client.post("/api/parity/schema-diff", json={"source_dsn": src, "target_dsn": tgt})
    body = resp.json()
    assert body["equal"] is False
    assert body["summary"]["missing_in_target"] == 1
    assert body["missing_in_target"][0]["kind"] == "index"


def test_dsn_route_unreachable_db_is_502(integration_app, migrated, dsn_route_enabled):
    src, _ = migrated
    client = TestClient(integration_app)
    bad = _with_db(ADMIN_DSN, f"does_not_exist_{uuid.uuid4().hex[:6]}")
    resp = client.post("/api/parity/schema-diff", json={"source_dsn": src, "target_dsn": bad})
    assert resp.status_code == 502


class _PodEmulator:
    """Stands in for coding-containers exec + the workspace's ws-pg.

    Maps /workspace to tests/data/py2rs_fixture_svc and really runs the commands
    against the test Postgres, so the runner's argv, env passing, URL parsing and
    psql snapshot path are exercised for real; only the HTTP hop is replaced.
    """

    def __init__(self, make_db: Callable[[], str]) -> None:
        self.make_db = make_db
        self.calls: list[list[str]] = []
        self.dbs: dict[str, str] = {}

    async def __call__(
        self, container: str, tenant: str | None, argv: list[str], timeout: int, workdir: str
    ) -> dict[str, Any]:
        self.calls.append(argv)
        cwd = FIXTURE / workdir.removeprefix("/workspace").lstrip("/")
        if argv[:2] == ["ws-pg", "start"]:
            return {"exit_code": 0, "stdout": f"DATABASE_URL={ADMIN_DSN}\n", "stderr": ""}
        if argv[:2] == ["ws-pg", "createdb"]:
            url = self.make_db()
            self.dbs[argv[2]] = url
            return {"exit_code": 0, "stdout": f"created {argv[2]}\n{url}\n", "stderr": ""}
        if argv[:2] == ["ws-pg", "psql"]:
            return self._ws_pg_psql(argv[2], argv[3:])
        env = dict(os.environ)
        if argv[0] == "env":
            key, _, value = argv[1].partition("=")
            env[key] = value
            argv = argv[2:]
        proc = subprocess.run(  # noqa: S603 - test emulation of the workspace exec
            argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout, check=False
        )
        return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}

    def _ws_pg_psql(self, dbname: str, args: list[str]) -> dict[str, Any]:
        """ws-pg psql's contract: a db it created, only ``-c`` pairs, no backslash."""
        pairs = list(zip(args[::2], args[1::2], strict=False))
        if (
            dbname not in self.dbs
            or len(args) % 2
            or not pairs
            or any(flag != "-c" or "\\" in sql for flag, sql in pairs)
        ):
            return {"exit_code": 1, "stdout": "", "stderr": "ws-pg: refused"}
        url = self.dbs[dbname]
        if shutil.which("psql"):
            fixed = ["psql", "-X", "--no-psqlrc", "-q", "-A", "-t", "-v", "ON_ERROR_STOP=1"]
            proc = subprocess.run(  # noqa: S603 - test emulation of ws-pg psql
                [*fixed, "-d", url, *args], capture_output=True, text=True, timeout=60, check=False
            )
            return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
        sql = args[-3]  # BEGIN READ ONLY / <snapshot> / COMMIT
        with psycopg.connect(url) as conn:
            conn.read_only = True
            row = conn.execute(sql.encode()).fetchone()
        return {"exit_code": 0, "stdout": (row[0] if row else "") + "\n", "stderr": ""}


@pytest.fixture
def workspace_client(integration_app, make_db, monkeypatch):
    from routers import parity_schema
    from services.schema_parity_workspace import SchemaParityWorkspaceRunner

    pod = _PodEmulator(make_db)
    integration_app.dependency_overrides[parity_schema.get_schema_parity_runner] = lambda: (
        SchemaParityWorkspaceRunner(exec_in=pod)
    )
    validate = AsyncMock(return_value=None)
    monkeypatch.setattr(parity_schema, "validate_and_get_container_for_tenant", validate)
    return TestClient(integration_app), pod, validate


SOURCE_CMD = f"{sys.executable} -m alembic -c alembic.ini upgrade head"
TARGET_CMD = [sys.executable, "../apply_sql_migrations.py", "migrations"]


def test_workspace_route_migrates_both_sides_and_is_equal(workspace_client):
    client, pod, validate = workspace_client
    resp = client.post(
        "/api/parity/ws-1/schema-diff",
        json={
            "source": {"workdir": "source", "migrate_command": SOURCE_CMD},
            "target": {"workdir": "target", "migrate_command": TARGET_CMD},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["computed"] is True, body
    assert body["equal"] is True, body
    validate.assert_awaited_once()
    assert validate.await_args.args[0] == "ws-1"
    stages = [s["stage"] for s in body["steps"]]
    assert stages == [
        "ws_pg_start",
        "createdb_source",
        "createdb_target",
        "migrate_source",
        "migrate_target",
        "introspect_source",
        "introspect_target",
    ]
    assert all(s["exit_code"] == 0 for s in body["steps"])
    # DATABASE_URL reached the migration through `env`, and credentials are not echoed.
    migrate = [c for c in pod.calls if c[0] == "env"]
    assert len(migrate) == 2 and all(c[1].startswith("DATABASE_URL=postgres") for c in migrate)
    assert all(":***@" in s["command"][1] for s in body["steps"] if s["stage"].startswith("migr"))


def test_workspace_route_failed_target_migration_is_not_computed(workspace_client):
    client, _, _ = workspace_client
    resp = client.post(
        "/api/parity/ws-1/schema-diff",
        json={
            "source": {"workdir": "source", "migrate_command": SOURCE_CMD},
            "target": {
                "workdir": "target",
                "migrate_command": [sys.executable, "-c", "import sys; sys.exit(3)"],
            },
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["computed"] is False
    assert body["equal"] is False
    assert body["stage"] == "migrate_target"
    assert "exit 3" in body["error"]


def test_workspace_route_refuses_workdir_escape(workspace_client):
    client, pod, _ = workspace_client
    resp = client.post(
        "/api/parity/ws-1/schema-diff",
        json={
            "source": {"workdir": "../etc", "migrate_command": SOURCE_CMD},
            "target": {"workdir": "target", "migrate_command": TARGET_CMD},
        },
    )
    assert resp.status_code == 422
    assert pod.calls == []
