"""Schema parity unit tests (py2rust A5a): pure diff logic + runner edges + gates."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from models.parity_schema_models import (
    MigrationSide,
    SchemaDiffDsnRequest,
    SchemaDiffOptions,
    SchemaDiffWorkspaceRequest,
)
from services.schema_parity import (
    SnapshotError,
    build_snapshot_sql,
    diff_snapshots,
    normalise_dsn,
    normalise_expression,
    parse_snapshot,
)
from services.schema_parity_workspace import (
    SchemaParityWorkspaceRunner,
    migrate_argv,
    parse_url,
    psql_argv,
    resolve_workdir,
)


def _snap() -> dict[str, Any]:
    return {
        "server_version": "18.6",
        "server_version_num": 180006,
        "tables": [
            {"schema": "public", "name": "t", "kind": "r", "persistence": "p"},
            {"schema": "public", "name": "alembic_version", "kind": "r", "persistence": "p"},
        ],
        "columns": [
            {"schema": "public", "table": "t", "name": "id", "type": "bigint", "not_null": True,
             "identity": "d", "generated": ""},
            {"schema": "public", "table": "t", "name": "ts", "type": "timestamp with time zone",
             "not_null": False, "default": "now()", "identity": "", "generated": ""},
            {"schema": "public", "table": "alembic_version", "name": "version_num",
             "type": "character varying(32)", "not_null": True, "identity": "", "generated": ""},
        ],
        "constraints": [
            {"schema": "public", "table": "t", "name": "t_pkey", "type": "p", "columns": ["id"]},
            {"schema": "public", "table": "t", "name": "t_c", "type": "c",
             "check_expression": "(id > 0)"},
        ],
        "indexes": [
            {"schema": "public", "table": "t", "name": "ix_a", "method": "btree",
             "unique": False, "keys": ["ts"], "options": [3], "opclasses": [None]},
            {"schema": "public", "table": "t", "name": "t_pkey", "method": "btree",
             "unique": True, "keys": ["id"], "constraint_backed": True},
        ],
        "sequences": [
            {"schema": "public", "name": "t_id_seq", "data_type": "bigint", "start": 1,
             "increment": 1, "dependency": "i", "owned_by_schema": "public",
             "owned_by_table": "t", "owned_by_column": "id"},
        ],
        "enums": [{"schema": "public", "name": "mood", "labels": ["a", "b"]}],
        "views": [],
        "extensions": [{"name": "plpgsql", "version": "1.0", "schema": "pg_catalog"}],
    }  # fmt: skip


def test_identical_snapshots_are_equal():
    result = diff_snapshots(_snap(), _snap())
    assert result.computed and result.equal
    assert result.summary == {
        "missing_in_target": 0,
        "extra_in_target": 0,
        "changed": 0,
        "name_only_differences": 0,
        "source_conflicts": 0,
        "column_order_differences": 0,
    }
    # alembic_version ignored by default; identity sequence folded into its column.
    assert result.source.object_counts["table"] == 1  # type: ignore[union-attr]
    assert result.source.object_counts["sequence"] == 0  # type: ignore[union-attr]


def _empty_snap(*, migration_tables: bool = False) -> dict[str, Any]:
    doc = {k: [] for k in ("tables", "columns", "constraints", "indexes", "sequences",
                           "enums", "views")}  # fmt: skip
    doc |= {
        "server_version": "18.6",
        "server_version_num": 180006,
        "extensions": [{"name": "plpgsql", "version": "1.0", "schema": "pg_catalog"}],
    }
    if migration_tables:
        doc["tables"] = [
            {"schema": "public", "name": n, "kind": "r", "persistence": "p"}
            for n in ("alembic_version", "_sqlx_migrations")
        ]
        doc["columns"] = [
            {"schema": "public", "table": "alembic_version", "name": "version_num",
             "type": "character varying(32)", "not_null": True, "identity": "", "generated": ""},
        ]  # fmt: skip
        doc["constraints"] = [
            {"schema": "public", "table": "alembic_version", "name": "alembic_version_pkc",
             "type": "p", "columns": ["version_num"]},
        ]  # fmt: skip
    return doc


@pytest.mark.parametrize("migration_tables", [False, True])
def test_two_empty_schemas_are_vacuous_never_equal(migration_tables):
    snap = _empty_snap(migration_tables=migration_tables)
    result = diff_snapshots(snap, copy.deepcopy(snap))
    assert result.computed is True
    assert (result.vacuous, result.equal) == (True, False)
    assert any("vacuous" in w for w in result.warnings)


def test_one_empty_side_is_a_real_difference_not_vacuous():
    result = diff_snapshots(_snap(), _empty_snap())
    assert (result.vacuous, result.equal) == (False, False)
    assert result.summary["missing_in_target"] > 0


def test_non_vacuous_equal_result_is_flagged_non_vacuous():
    result = diff_snapshots(_snap(), _snap())
    assert (result.vacuous, result.equal) == (False, True)


def test_current_timestamp_alias_and_whitespace_normalised():
    tgt = _snap()
    tgt["columns"][1]["default"] = "CURRENT_TIMESTAMP"
    tgt["constraints"][1]["check_expression"] = "(id  >\n 0)"
    assert diff_snapshots(_snap(), tgt).equal is True
    assert normalise_expression("CURRENT_TIMESTAMP(3)") == "CURRENT_TIMESTAMP(3)"
    assert normalise_expression(None) is None


def test_index_desc_option_is_part_of_identity():
    tgt = _snap()
    tgt["indexes"][0]["options"] = [0]
    result = diff_snapshots(_snap(), tgt)
    assert [d.key for d in result.missing_in_target] == ["public.t USING btree (ts DESC)"]
    assert [d.key for d in result.extra_in_target] == ["public.t USING btree (ts)"]


def test_identity_sequence_change_reported_on_column():
    tgt = _snap()
    tgt["sequences"][0]["increment"] = 5
    result = diff_snapshots(_snap(), tgt)
    assert [(c.kind, c.key) for c in result.changed] == [("column", "public.t.id")]
    assert result.changed[0].fields[0].field == "identity_sequence"


def test_enum_label_order_matters_and_output_is_ordered():
    tgt = _snap()
    tgt["enums"][0]["labels"] = ["b", "a"]
    tgt["columns"][0]["type"] = "integer"
    result = diff_snapshots(_snap(), tgt)
    # enum sorts before column (KIND_ORDER), regardless of discovery order.
    assert [c.kind for c in result.changed] == ["enum", "column"]


def test_ignore_tables_schema_qualified_and_empty():
    keep = diff_snapshots(_snap(), _snap(), SchemaDiffOptions(ignore_tables=[]))
    assert keep.source.object_counts["table"] == 2  # type: ignore[union-attr]
    q = diff_snapshots(_snap(), _snap(), SchemaDiffOptions(ignore_tables=["public.t"]))
    assert q.source.object_counts["table"] == 1  # type: ignore[union-attr]
    assert q.options.ignore_tables == ["public.t"]  # type: ignore[union-attr]


def test_duplicate_index_definitions_do_not_collapse():
    tgt = _snap()
    dup = copy.deepcopy(tgt["indexes"][0])
    dup["name"] = "ix_b"
    tgt["indexes"].append(dup)
    result = diff_snapshots(_snap(), tgt)
    assert [d.key for d in result.extra_in_target] == ["public.t USING btree (ts DESC) #2"]


def test_major_version_mismatch_warns():
    tgt = _snap()
    tgt["server_version_num"] = 170004
    assert diff_snapshots(_snap(), tgt).warnings


def test_parse_snapshot_errors():
    with pytest.raises(SnapshotError):
        parse_snapshot("")
    with pytest.raises(SnapshotError):
        parse_snapshot("not json")
    with pytest.raises(SnapshotError):
        parse_snapshot('{"tables": []}')
    assert parse_snapshot(json.dumps(_snap()).encode())["server_version"] == "18.6"


def test_schema_filter_validated_and_inlined():
    assert "nspname IN ('app', 'public')" in build_snapshot_sql(["public", "app"])
    with pytest.raises(ValueError):
        build_snapshot_sql(["x'; DROP"])
    with pytest.raises(ValueError):
        SchemaDiffOptions(schemas=["bad name"])


def test_normalise_dsn_strips_sqlalchemy_driver():
    assert normalise_dsn("postgresql+psycopg://h/db") == "postgresql://h/db"
    assert normalise_dsn("postgres://h/db") == "postgres://h/db"


# ---------------------------------------------------------------------------
# Workspace runner helpers and failure stages
# ---------------------------------------------------------------------------


def test_workdir_resolution():
    assert resolve_workdir(".") == "/workspace"
    assert resolve_workdir("svc/") == "/workspace/svc"
    for bad in ("/etc", "../x", "a/../b", "-rf"):
        with pytest.raises(ValueError):
            resolve_workdir(bad)


def test_migrate_argv_and_url_parsing():
    assert migrate_argv(MigrationSide(migrate_command="sqlx migrate run")) == [
        "sqlx",
        "migrate",
        "run",
    ]
    with pytest.raises(ValueError):
        migrate_argv(MigrationSide(migrate_command="  "))
    assert (
        parse_url("log\nDATABASE_URL=postgresql://a@127.0.0.1/x\n") == "postgresql://a@127.0.0.1/x"
    )
    assert parse_url("nothing here") is None
    argv = psql_argv("parity_sou_ab12", "SELECT 1")
    assert argv == [
        "ws-pg", "psql", "parity_sou_ab12",
        "-c", "BEGIN READ ONLY", "-c", "SELECT 1", "-c", "COMMIT",
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("dbname", "sql"),
    [
        ("postgresql://u@h/d", "SELECT 1"),
        ("Parity", "SELECT 1"),
        ("-h", "SELECT 1"),
        ("d", "\\connect postgresql://evil/x"),
        ("d", "SELECT 1; \\! id"),
    ],
)
def test_psql_argv_refuses_what_ws_pg_refuses(dbname, sql):
    with pytest.raises(ValueError):
        psql_argv(dbname, sql)


@pytest.mark.parametrize("schemas", [None, ["public"], ["public", "app_1", "x$y"]])
def test_snapshot_sql_has_no_backslash(schemas):
    # ws-pg psql refuses any -c value with a backslash (no psql meta-commands).
    assert "\\" not in build_snapshot_sql(schemas)


def _req() -> SchemaDiffWorkspaceRequest:
    return SchemaDiffWorkspaceRequest(
        source=MigrationSide(workdir="py", migrate_command="uv run alembic upgrade head"),
        target=MigrationSide(workdir="rs", migrate_command="sqlx migrate run"),
    )


async def test_runner_createdb_without_url_is_not_computed():
    async def exec_in(container, tenant, argv, timeout, workdir):
        return {"exit_code": 0, "stdout": "ok, no url", "stderr": ""}

    result = await SchemaParityWorkspaceRunner(exec_in).run("ws", _req(), "t1")
    assert (result.computed, result.equal, result.stage) == (False, False, "createdb_source")


# A URL WITH a password, so the redaction below is actually exercised. Assembled
# from parts so no literal user:password@host string sits in the source (the
# archetype's detect-secrets gate flags one as Basic Auth Credentials).
_CRED_URL = "postgresql://" + ":".join(("u", "pw")) + "@127.0.0.1/d"


async def test_runner_transport_error_and_truncation_are_not_computed():
    async def boom(container, tenant, argv, timeout, workdir):
        raise RuntimeError("connection refused")

    r = await SchemaParityWorkspaceRunner(boom).run("ws", _req(), "t1")
    assert (r.computed, r.stage) == (False, "ws_pg_start")

    async def truncated(container, tenant, argv, timeout, workdir):
        if argv[:2] == ["ws-pg", "psql"]:
            return {"exit_code": 0, "stdout": "{", "stderr": "", "truncated": True}
        return {"exit_code": 0, "stdout": _CRED_URL, "stderr": ""}

    r = await SchemaParityWorkspaceRunner(truncated).run("ws", _req(), "t1")
    assert (r.computed, r.stage) == (False, "introspect_source")
    assert "truncated" in (r.error or "")
    # credentials in echoed commands are redacted
    assert all("pw@" not in " ".join(s["command"]) for s in r.steps)


async def test_runner_bad_snapshot_and_no_exit_code():
    async def bad_json(container, tenant, argv, timeout, workdir):
        if argv[:2] == ["ws-pg", "psql"]:
            return {"exit_code": 0, "stdout": "not json", "stderr": ""}
        return {"exit_code": 0, "stdout": "postgresql://127.0.0.1/d", "stderr": ""}

    r = await SchemaParityWorkspaceRunner(bad_json).run("ws", _req(), None)
    assert (r.computed, r.stage) == (False, "introspect_source")

    async def no_code(container, tenant, argv, timeout, workdir):
        return {"stdout": ""}

    r = await SchemaParityWorkspaceRunner(no_code).run("ws", _req(), None)
    assert (r.computed, r.stage) == (False, "ws_pg_start")


async def test_runner_timeout_exit_is_labelled():
    async def slow(container, tenant, argv, timeout, workdir):
        if argv[0] == "env":
            return {"exit_code": 124, "stdout": "", "stderr": ""}
        return {"exit_code": 0, "stdout": "postgresql://127.0.0.1/d", "stderr": ""}

    r = await SchemaParityWorkspaceRunner(slow).run("ws", _req(), None)
    assert r.stage == "migrate_source" and "timed out" in (r.error or "")


async def test_runner_passes_workdirs_and_env():
    seen: list[tuple[list[str], str]] = []

    async def record(container, tenant, argv, timeout, workdir):
        seen.append((argv, workdir))
        if argv[:2] == ["ws-pg", "psql"]:
            return {"exit_code": 0, "stdout": json.dumps(_snap()), "stderr": ""}
        return {"exit_code": 0, "stdout": f"postgresql://127.0.0.1/{len(seen)}", "stderr": ""}

    r = await SchemaParityWorkspaceRunner(record).run("ws", _req(), "t1")
    assert r.computed and r.equal
    migrations = [(a, w) for a, w in seen if a[0] == "env"]
    assert migrations == [
        (["env", "DATABASE_URL=postgresql://127.0.0.1/2", "uv", "run", "alembic", "upgrade",
          "head"], "/workspace/py"),
        (["env", "DATABASE_URL=postgresql://127.0.0.1/3", "sqlx", "migrate", "run"],
         "/workspace/rs"),
    ]  # fmt: skip


# ---------------------------------------------------------------------------
# Route gates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("environment", "enabled"),
    [
        ("production", False),
        ("production", True),
        ("Production", True),
        ("prod", True),
        ("staging", True),
        ("", True),
        ("test", False),
        ("development", False),
    ],
)
async def test_dsn_route_refused_unless_flag_and_allowlisted_env(monkeypatch, environment, enabled):
    from routers import parity_schema

    monkeypatch.setattr(
        parity_schema,
        "get_settings",
        lambda: SimpleNamespace(ENVIRONMENT=environment, PARITY_DSN_DIFF_ENABLED=enabled),
    )
    req = SchemaDiffDsnRequest(source_dsn="postgresql://a/b", target_dsn="postgresql://a/c")
    with pytest.raises(HTTPException) as exc:
        await parity_schema.schema_diff_dsn(req, current_user=None, tenant_id="t")  # type: ignore[arg-type]
    assert exc.value.status_code == 403


@pytest.mark.parametrize("environment", ["development", "DEV", "test", "Testing", "local", "ci"])
def test_dsn_route_allowed_with_flag_in_allowlisted_env(environment):
    from routers import parity_schema

    settings = SimpleNamespace(ENVIRONMENT=environment, PARITY_DSN_DIFF_ENABLED=True)
    assert parity_schema.dsn_diff_allowed(settings) is True
    settings.PARITY_DSN_DIFF_ENABLED = False
    assert parity_schema.dsn_diff_allowed(settings) is False


async def test_dsn_route_rejects_non_postgres_dsn(monkeypatch):
    from routers import parity_schema

    monkeypatch.setattr(
        parity_schema,
        "get_settings",
        lambda: SimpleNamespace(ENVIRONMENT="test", PARITY_DSN_DIFF_ENABLED=True),
    )
    req = SchemaDiffDsnRequest(source_dsn="mysql://a/b", target_dsn="postgresql://a/c")
    with pytest.raises(HTTPException) as exc:
        await parity_schema.schema_diff_dsn(req, current_user=None, tenant_id="t")  # type: ignore[arg-type]
    assert exc.value.status_code == 422


# ---------------------------------------------------------------------------
# G1: every route that exists on origin/main keeps a byte-identical OpenAPI
# ---------------------------------------------------------------------------

_SNAPSHOT = Path(__file__).resolve().parent.parent / "data" / "openapi_existing_routes.json"


def test_existing_routes_openapi_unchanged(monkeypatch):
    monkeypatch.delenv("DEFAULT_TENANT_ID", raising=False)
    from app_factory import create_app

    spec = json.loads(json.dumps(create_app().openapi(), sort_keys=True))
    snap = json.loads(_SNAPSHOT.read_text())
    assert snap["paths"], "snapshot is empty: the gate would pass vacuously"
    changed_paths = [p for p, item in snap["paths"].items() if spec["paths"].get(p) != item]
    assert changed_paths == []
    for section, items in snap["components"].items():
        changed = [k for k, v in items.items() if spec["components"][section].get(k) != v]
        assert changed == [], f"components.{section} changed: {changed}"
    added = sorted(set(spec["paths"]) - set(snap["paths"]))
    # The integrated py2rust phase-1 PR (coding-testing#79-82) adds exactly these paths;
    # anything else appearing is an unreviewed contract change.
    assert added == [
        "/api/parity/openapi-diff",
        "/api/parity/schema-diff",
        "/api/parity/{container_name}/compare",
        "/api/parity/{container_name}/openapi-diff",
        # gaps-n-to-1: async job mode of the workspace openapi diff.
        "/api/parity/{container_name}/openapi-diff/runs",
        "/api/parity/{container_name}/openapi-diff/runs/{run_id}",
        # #87: performance comparison of N sources vs the target under load.
        "/api/parity/{container_name}/perf/runs",
        "/api/parity/{container_name}/perf/runs/{run_id}",
        "/api/parity/{container_name}/record",
        "/api/parity/{container_name}/replay",
        "/api/parity/{container_name}/runs/{run_id}",
        "/api/parity/{container_name}/schema-diff",
        # gaps-n-to-1 (N sources -> 1 target): async job mode of the workspace schema diff.
        "/api/parity/{container_name}/schema-diff/runs",
        "/api/parity/{container_name}/schema-diff/runs/{run_id}",
        "/api/tests/workspace/{container_name}/runs",
        "/api/tests/workspace/{container_name}/runs/{run_id}",
    ]


def test_added_parity_routes_have_deliberate_operation_ids(monkeypatch):
    """n8n-nodes-coding actions are generated from operationIds: no auto ids on new routes."""
    monkeypatch.delenv("DEFAULT_TENANT_ID", raising=False)
    from app_factory import create_app

    paths = create_app().openapi()["paths"]
    ids = {
        (method.upper(), path): op["operationId"]
        for path, item in paths.items()
        if path.startswith("/api/parity/")
        for method, op in item.items()
    }
    assert ids == {
        ("POST", "/api/parity/openapi-diff"): "parity_openapi_diff",
        ("POST", "/api/parity/{container_name}/openapi-diff"): "parity_workspace_openapi_diff",
        (
            "POST",
            "/api/parity/{container_name}/openapi-diff/runs",
        ): "parity_workspace_openapi_diff_start",
        (
            "GET",
            "/api/parity/{container_name}/openapi-diff/runs/{run_id}",
        ): "get_parity_workspace_openapi_diff_run",
        ("POST", "/api/parity/schema-diff"): "parity_schema_diff",
        ("POST", "/api/parity/{container_name}/schema-diff"): "parity_workspace_schema_diff",
        (
            "POST",
            "/api/parity/{container_name}/schema-diff/runs",
        ): "parity_workspace_schema_diff_start",
        (
            "GET",
            "/api/parity/{container_name}/schema-diff/runs/{run_id}",
        ): "get_parity_workspace_schema_diff_run",
        ("POST", "/api/parity/{container_name}/record"): "parity_behaviour_record",
        ("POST", "/api/parity/{container_name}/replay"): "parity_behaviour_replay",
        ("POST", "/api/parity/{container_name}/compare"): "parity_behaviour_compare",
        ("GET", "/api/parity/{container_name}/runs/{run_id}"): "get_parity_behaviour_run",
        ("POST", "/api/parity/{container_name}/perf/runs"): "parity_perf_start",
        ("GET", "/api/parity/{container_name}/perf/runs/{run_id}"): "get_parity_perf_run",
    }
