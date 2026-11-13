"""Schema parity, N sources -> 1 target (gap items 2 + 16): the pure union diff.

Real-Postgres coverage of the same paths (and of the new catalog sections) is in
tests/integration/test_schema_parity_multi_pg.py.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from models.parity_schema_models import (
    SchemaDiffDsnRequest,
    SchemaDiffOptions,
    SchemaDiffWorkspaceRequest,
)
from services.schema_parity import SourceSnapshot, diff_union, map_snapshot

pytestmark = pytest.mark.usefixtures("as_writer", "pg_db")


def _empty() -> dict[str, Any]:
    return {
        "server_version": "18.4",
        "server_version_num": 180004,
        "tables": [],
        "columns": [],
        "constraints": [],
        "indexes": [],
        "sequences": [],
        "enums": [],
        "views": [],
        "extensions": [{"name": "plpgsql", "version": "1.0", "schema": "pg_catalog"}],
        "domains": [],
        "functions": [],
        "triggers": [],
        "policies": [],
        "grants": [],
    }


def _table(doc: dict, table: str, cols: list[tuple[str, str]], schema: str = "public") -> dict:
    doc["tables"].append({"schema": schema, "name": table, "kind": "r", "persistence": "p"})
    for pos, (name, typ) in enumerate(cols, 1):
        doc["columns"].append(
            {"schema": schema, "table": table, "name": name, "type": typ, "not_null": False,
             "identity": "", "generated": "", "position": pos}
        )  # fmt: skip
    doc["constraints"].append(
        {"schema": schema, "table": table, "name": f"{table}_pkey", "type": "p",
         "columns": [cols[0][0]]}
    )  # fmt: skip
    return doc


def _source_a() -> dict:
    return _table(_empty(), "notes", [("id", "bigint"), ("body", "text")])


def _source_b() -> dict:
    doc = _table(_empty(), "things", [("id", "bigint"), ("qty", "integer"), ("tenant", "text")])
    doc["sequences"].append(
        {"schema": "public", "name": "things_id_seq", "data_type": "bigint", "start": 1,
         "increment": 1, "dependency": "a", "owned_by_schema": "public",
         "owned_by_table": "things", "owned_by_column": "id"}
    )  # fmt: skip
    doc["columns"][0]["default"] = "nextval('things_id_seq'::regclass)"
    doc["domains"].append(
        {"schema": "public", "name": "pos", "base_type": "integer", "not_null": False,
         "checks": ["CHECK (VALUE > 0)"]}
    )  # fmt: skip
    doc["functions"].append(
        {"schema": "public", "name": "touch", "identity_args": "", "args": "",
         "result": "trigger", "kind": "f", "language": "plpgsql", "volatility": "v",
         "strict": False, "security_definer": False, "parallel": "u", "config": None,
         "body": " BEGIN RETURN NEW; END "}
    )  # fmt: skip
    doc["triggers"].append(
        {"schema": "public", "table": "things", "name": "things_touch", "relation": "things",
         "definition": "CREATE TRIGGER things_touch BEFORE INSERT ON things FOR EACH ROW "
         "EXECUTE FUNCTION touch()", "enabled": "O"}
    )  # fmt: skip
    doc["policies"].append(
        {"schema": "public", "table": "things", "name": "tenant_iso", "command": "*",
         "permissive": True, "roles": ["PUBLIC"], "using": "(tenant = current_user)",
         "with_check": None}
    )  # fmt: skip
    doc["grants"].append(
        {"object_kind": "table", "schema": "public", "name": "things", "grantee": "reader",
         "privilege": "SELECT", "grantable": False}
    )  # fmt: skip
    doc["tables"][0]["rls"] = True
    return doc


def _target() -> dict:
    """Both sources merged: A unchanged, B's relations prefixed b_ (what the Rust repo has)."""
    tgt = _source_a()
    b = map_snapshot(_source_b(), table_prefix="b_")
    for key in tgt:
        if isinstance(tgt[key], list) and key in b:
            tgt[key] = [*tgt[key], *[x for x in b[key] if x not in tgt[key]]]
    # The deparser names the prefixed relation in the trigger text; the pkey keeps its name.
    for tg in tgt["triggers"]:
        tg["definition"] = tg["definition"].replace(" ON things ", " ON b_things ")
    return tgt


def _sources() -> list[SourceSnapshot]:
    return [SourceSnapshot("a", _source_a()), SourceSnapshot("b", _source_b(), None, "b_")]


def test_union_of_mapped_sources_equals_merged_target():
    result = diff_union(_sources(), _target())
    assert result.computed is True and result.equal is True, result.model_dump()
    assert [s.name for s in result.sources] == ["a", "b"]
    assert result.sources[1].mapping is not None and result.sources[1].mapping.table_prefix == "b_"
    counts = result.source.object_counts  # type: ignore[union-attr]
    assert counts["table"] == 2 and counts["trigger"] == 1 and counts["policy"] == 1
    assert counts["domain"] == 1 and counts["function"] == 1 and counts["grant"] == 1


def test_without_the_prefix_map_b_is_missing_and_extra():
    """Negative control for the mapping: unmapped, B's table is missing + b_things extra."""
    result = diff_union(
        [SourceSnapshot("a", _source_a()), SourceSnapshot("b", _source_b())], _target()
    )
    assert result.equal is False
    assert {d.key for d in result.missing_in_target if d.kind == "table"} == {"public.things"}
    assert {d.key for d in result.extra_in_target if d.kind == "table"} == {"public.b_things"}
    assert next(d for d in result.missing_in_target if d.kind == "table").sources == ["b"]


@pytest.mark.parametrize(
    "mutate, kind, bucket",
    [
        (lambda t: t["policies"][0].update(using="(true)"), "policy", "changed"),
        (lambda t: t["triggers"].clear(), "trigger", "missing_in_target"),
        (lambda t: t["domains"][0].update(checks=["CHECK (VALUE >= 0)"]), "domain", "changed"),
        (lambda t: t["functions"][0].update(body="BEGIN RETURN OLD; END"), "function", "changed"),
        (lambda t: t["grants"].clear(), "grant", "missing_in_target"),
        (lambda t: t["tables"][-1].update(rls=False), "table", "changed"),
    ],
)
def test_planted_mismatch_in_source_b_is_attributed_to_b(mutate, kind, bucket):
    tgt = _target()
    mutate(tgt)
    result = diff_union(_sources(), tgt)
    assert result.equal is False
    found = [d for d in getattr(result, bucket) if d.kind == kind]
    assert len(found) == 1 and found[0].sources == ["b"], result.model_dump()


def test_two_sources_defining_one_object_differently_is_a_conflict():
    a = _source_a()
    b = _table(_empty(), "notes", [("id", "bigint"), ("body", "character varying(10)")])
    result = diff_union([SourceSnapshot("a", a), SourceSnapshot("b", b)], _source_a())
    assert result.equal is False
    assert [(c.kind, c.key, c.kept) for c in result.source_conflicts] == [
        ("column", "public.notes.body", "a")
    ]
    assert result.source_conflicts[0].conflicts[0].source == "b"
    # Identical definitions merge and are attributed to both.
    same = diff_union([SourceSnapshot("a", a), SourceSnapshot("b", _source_a())], _source_a())
    assert same.equal is True and same.source_conflicts == []


def test_column_order_is_its_own_classification_and_ignorable():
    tgt = _target()
    for col in tgt["columns"]:
        if col["table"] == "b_things":
            col["position"] = {"id": 1, "qty": 3, "tenant": 2}[col["name"]]
    result = diff_union(_sources(), tgt)
    assert result.equal is False and result.changed == [] and result.missing_in_target == []
    [order] = result.column_order_differences
    assert order.table == "public.b_things" and order.sources == ["b"]
    assert (order.source_order, order.target_order) == (
        ["id", "qty", "tenant"],
        ["id", "tenant", "qty"],
    )
    relaxed = diff_union(_sources(), tgt, SchemaDiffOptions(ignore_column_order=True))
    assert relaxed.equal is True and len(relaxed.column_order_differences) == 1


def test_a_source_with_no_objects_is_never_equal():
    result = diff_union(
        [SourceSnapshot("a", _source_a()), SourceSnapshot("b", _empty())], _source_a()
    )
    assert result.vacuous is False and result.equal is False
    assert result.vacuous_sources == ["b"]


def test_ignore_kinds_and_schema_map():
    tgt = _target()
    tgt["grants"].clear()
    assert diff_union(_sources(), tgt, SchemaDiffOptions(ignore_kinds=["grant"])).equal is True
    moved = copy.deepcopy(_source_a())
    for section in ("tables", "columns", "constraints"):
        for item in moved[section]:
            item["schema"] = "notes"
    mapped = map_snapshot(_source_a(), schema_map={"public": "notes"})
    assert diff_union([SourceSnapshot("a", mapped)], moved).equal is True
    assert diff_union([SourceSnapshot("a", _source_a(), {"public": "notes"})], moved).equal is True
    with pytest.raises(ValueError, match="unknown kinds"):
        SchemaDiffOptions(ignore_kinds=["nope"])


def test_request_shapes():
    with pytest.raises(ValueError, match="exactly one of source or sources"):
        SchemaDiffWorkspaceRequest.model_validate({"target": {"migrate_command": "x"}})
    with pytest.raises(ValueError, match="unique"):
        SchemaDiffWorkspaceRequest.model_validate(
            {
                "sources": [{"name": "a", "migrate_command": "x"}] * 2,
                "target": {"migrate_command": "y"},
            }
        )
    with pytest.raises(ValueError, match="reserved"):
        SchemaDiffDsnRequest.model_validate(
            {"sources": [{"name": "target", "dsn": "postgresql://x/y"}], "target_dsn": "p://"}
        )
    with pytest.raises(ValueError, match="table_prefix"):
        SchemaDiffWorkspaceRequest.model_validate(
            {
                "sources": [
                    {"name": "a", "migrate_command": "x", "mapping": {"table_prefix": "1-"}}
                ],
                "target": {"migrate_command": "y"},
            }
        )
