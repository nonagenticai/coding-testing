"""openapi_diff: normalisation, union of sources, and precise mutation detection.

The real fixture is ``py2rs-fixture-svc``'s FastAPI document (copied verbatim from
conveyor-coding/py2rs-fixture-svc/openapi.json) and a hand-constructed utoipa-style
OpenAPI 3.1 equivalent: different component names, path parameter names, key
order, nullable spelling (type lists / oneOf), operationIds, tags, descriptions,
lower-case header names and path-item-level parameters. They must compare equal;
each single mutation must then be reported exactly where it was made.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest

from services.openapi_diff import (
    IgnoreRules,
    OpenApiDiffError,
    SpecInput,
    ignore_rules_from_parity_toml,
    openapi_diff,
    path_shape,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "openapi_diff"
FASTAPI = json.loads((DATA / "py2rs_fixture_fastapi.json").read_text())
UTOIPA = json.loads((DATA / "py2rs_fixture_utoipa.json").read_text())


def src() -> dict[str, Any]:
    return copy.deepcopy(FASTAPI)


def tgt() -> dict[str, Any]:
    return copy.deepcopy(UTOIPA)


def doc(paths: dict[str, Any], schemas: dict[str, Any] | None = None, **extra: Any) -> dict:
    d: dict[str, Any] = {"openapi": "3.1.0", "info": {"title": "t", "version": "1"}, "paths": paths}
    if schemas is not None:
        d["components"] = {"schemas": schemas}
    d.update(extra)
    return d


def get_op(schema: dict[str, Any], *, status: str = "200", params: list | None = None) -> dict:
    op: dict[str, Any] = {
        "responses": {
            status: {"description": "ok", "content": {"application/json": {"schema": schema}}}
        }
    }
    if params is not None:
        op["parameters"] = params
    return op


def only_changed(result: dict[str, Any]) -> tuple[str, str, list[dict[str, Any]]]:
    assert len(result["changed_operations"]) == 1, result["changed_operations"]
    c = result["changed_operations"][0]
    return c["method"], c["path"], c["differences"]


# --------------------------------------------------------------------------- fixture


def test_fixture_fastapi_vs_utoipa_is_equal() -> None:
    r = openapi_diff([src()], tgt())
    assert r["equal"] is True
    assert r["vacuous"] is False
    assert r["progress"] == {"matched": 6, "total": 6, "ratio": 1.0}
    assert r["counts"]["source_operations"] == 6
    assert r["counts"]["target_operations"] == 6
    assert r["missing_operations"] == r["extra_operations"] == r["changed_operations"] == []


def test_fixture_is_equal_to_itself_both_ways() -> None:
    assert openapi_diff([src()], src())["equal"] is True
    assert openapi_diff([tgt()], src())["equal"] is True


def test_mutation_drop_route() -> None:
    t = tgt()
    del t["paths"]["/notes/{id}"]["delete"]
    r = openapi_diff([src()], t)
    assert r["equal"] is False
    assert r["missing_operations"] == [
        {"method": "DELETE", "path": "/notes/{note_id}", "sources": ["source[0]"]}
    ]
    assert r["changed_operations"] == [] and r["extra_operations"] == []
    assert r["progress"] == {"matched": 5, "total": 6, "ratio": 0.8333}


def test_mutation_status_code_201_to_200() -> None:
    t = tgt()
    responses = t["paths"]["/notes"]["post"]["responses"]
    responses["200"] = responses.pop("201")
    r = openapi_diff([src()], t)
    method, path, diffs = only_changed(r)
    assert (method, path) == ("POST", "/notes")
    assert diffs == [
        {"location": "responses.200", "kind": "response_added", "target": "200"},
        {"location": "responses.201", "kind": "response_removed", "source": "201"},
    ]
    assert r["progress"]["matched"] == 5


def test_mutation_param_made_optional() -> None:
    """X-Tenant-Id is optional in the source; the query param `tag` too. Make the
    source's `tag` required, keep the target's optional: required -> optional."""
    s = src()
    s["paths"]["/notes"]["get"]["parameters"][0]["required"] = True
    r = openapi_diff([s], tgt())
    method, path, diffs = only_changed(r)
    assert (method, path) == ("GET", "/notes")
    assert diffs == [
        {
            "location": "parameters.query:tag",
            "kind": "parameter_required_changed",
            "source": True,
            "target": False,
        }
    ]


def test_mutation_body_field_made_optional() -> None:
    t = tgt()
    t["components"]["schemas"]["NewNote"]["required"] = ["body"]
    r = openapi_diff([src()], t)
    method, path, diffs = only_changed(r)
    assert (method, path) == ("POST", "/notes")
    assert diffs == [
        {
            "location": "request_body.content.application/json.schema.required",
            "kind": "required_removed",
            "source": "title",
        }
    ]


def test_mutation_request_body_made_optional() -> None:
    t = tgt()
    t["paths"]["/notes"]["post"]["requestBody"]["required"] = False
    _, _, diffs = only_changed(openapi_diff([src()], t))
    assert diffs == [
        {
            "location": "request_body",
            "kind": "request_body_required_changed",
            "source": True,
            "target": False,
        }
    ]


def test_mutation_response_field_type() -> None:
    t = tgt()
    t["components"]["schemas"]["Enriched"]["properties"]["summary"] = {"type": "integer"}
    _, path, diffs = only_changed(openapi_diff([src()], t))
    assert path == "/notes/{note_id}/enrich"
    assert diffs == [
        {
            "location": "responses.200.content.application/json.schema.properties.summary.type",
            "kind": "type_changed",
            "source": "string",
            "target": "integer",
        }
    ]


def test_mutation_shared_schema_change_hits_every_user() -> None:
    t = tgt()
    t["components"]["schemas"]["Note"]["properties"]["title"] = {"type": "integer"}
    r = openapi_diff([src()], t)
    hit = {(c["method"], c["path"]) for c in r["changed_operations"]}
    assert hit == {("GET", "/notes"), ("POST", "/notes"), ("GET", "/notes/{note_id}")}
    list_op = next(
        c for c in r["changed_operations"] if c["path"] == "/notes" and c["method"] == "GET"
    )
    assert list_op["differences"][0]["location"] == (
        "responses.200.content.application/json.schema.properties.items.items.properties.title.type"
    )


def test_mutation_nullable_dropped_is_detected() -> None:
    t = tgt()
    t["components"]["schemas"]["Note"]["properties"]["tag"] = {"type": "string"}
    r = openapi_diff([src()], t)
    get_note = next(c for c in r["changed_operations"] if c["path"] == "/notes/{note_id}")
    assert get_note["differences"] == [
        {
            "location": "responses.200.content.application/json.schema.properties.tag.nullable",
            "kind": "nullable_changed",
            "source": True,
            "target": False,
        }
    ]


def test_mutation_extra_route_and_property() -> None:
    t = tgt()
    t["paths"]["/metrics"] = {"get": get_op({"type": "string"})}
    t["components"]["schemas"]["HealthBody"]["properties"]["uptime"] = {"type": "integer"}
    r = openapi_diff([src()], t)
    assert r["extra_operations"] == [{"method": "GET", "path": "/metrics"}]
    _, path, diffs = only_changed(r)
    assert path == "/health"
    assert diffs == [
        {
            "location": "responses.200.content.application/json.schema.properties.uptime",
            "kind": "property_added",
            "target": {"type": "integer"},
        }
    ]
    assert r["equal"] is False and r["progress"]["matched"] == 5


def test_mutation_param_removed_and_added() -> None:
    t = tgt()
    params = t["paths"]["/notes"]["get"]["parameters"]
    params[0] = {"name": "label", "in": "query", "schema": {"type": "string"}}
    _, _, diffs = only_changed(openapi_diff([src()], t))
    assert [(d["location"], d["kind"]) for d in diffs] == [
        ("parameters.query:label", "parameter_added"),
        ("parameters.query:tag", "parameter_removed"),
    ]


def test_mutation_media_type_changed() -> None:
    t = tgt()
    body = t["paths"]["/notes"]["post"]["requestBody"]["content"]
    body["application/x-www-form-urlencoded"] = body.pop("application/json")
    _, _, diffs = only_changed(openapi_diff([src()], t))
    assert [(d["location"], d["kind"]) for d in diffs] == [
        ("request_body.content.application/json", "media_type_removed"),
        ("request_body.content.application/x-www-form-urlencoded", "media_type_added"),
    ]


def test_mutation_constraint_changed() -> None:
    t = tgt()
    t["components"]["schemas"]["NewNote"]["properties"]["title"]["maxLength"] = 100
    _, _, diffs = only_changed(openapi_diff([src()], t))
    assert diffs == [
        {
            "location": "request_body.content.application/json.schema.properties.title.maxLength",
            "kind": "keyword_changed",
            "source": 200,
            "target": 100,
        }
    ]


def test_deterministic_regardless_of_key_order() -> None:
    def shuffle(o: Any, rnd: random.Random) -> Any:
        if isinstance(o, dict):
            items = list(o.items())
            rnd.shuffle(items)
            return {k: shuffle(v, rnd) for k, v in items}
        if isinstance(o, list):
            return [shuffle(x, rnd) for x in o]
        return o

    t = tgt()
    del t["paths"]["/health"]
    t["components"]["schemas"]["Note"]["required"] = ["id"]
    base = json.dumps(openapi_diff([src()], t))
    for seed in range(5):
        rnd = random.Random(seed)
        s2, t2 = shuffle(src(), rnd), shuffle(t, rnd)
        t2["components"]["schemas"]["Note"]["required"] = ["id"]
        assert json.dumps(openapi_diff([s2], t2)) == base


# --------------------------------------------------------------------------- normalisation


@pytest.mark.parametrize(
    "a, b",
    [
        # 3.0 nullable vs 3.1 type list vs anyOf-null vs oneOf-null
        ({"type": "string", "nullable": True}, {"type": ["string", "null"]}),
        ({"anyOf": [{"type": "string"}, {"type": "null"}]}, {"type": ["null", "string"]}),
        ({"oneOf": [{"type": "null"}, {"type": "string"}]}, {"type": "string", "nullable": True}),
        # integer / number formats are generator cosmetics
        ({"type": "integer"}, {"type": "integer", "format": "int64"}),
        ({"type": "number", "format": "double"}, {"type": "number", "format": "float"}),
        # docs-only keys
        (
            {"type": "string", "title": "X", "description": "d", "examples": ["a"]},
            {"type": "string"},
        ),
        ({"type": "string", "x-internal": True, "example": "a"}, {"type": "string"}),
        # set-like lists
        (
            {"type": "object", "required": ["a", "b"], "properties": {"a": {}, "b": {}}},
            {"type": "object", "required": ["b", "a"], "properties": {"b": {}, "a": {}}},
        ),
        ({"enum": ["b", "a"]}, {"enum": ["a", "b"]}),
        ({"enum": ["only"]}, {"const": "only"}),
        ({"type": "string", "enum": ["a", None]}, {"type": ["string", "null"], "enum": ["a"]}),
        # 3.0 boolean exclusive bounds vs 3.1 numeric
        (
            {"type": "integer", "minimum": 0, "exclusiveMinimum": True},
            {"type": "integer", "exclusiveMinimum": 0},
        ),
        (
            {"type": "integer", "minimum": 0, "exclusiveMinimum": False},
            {"type": "integer", "minimum": 0},
        ),
        # absent meanings
        ({"type": "object", "additionalProperties": True}, {"type": "object"}),
        ({"type": "string", "default": None}, {"type": "string"}),
        # single-member allOf (utoipa wraps refs) and nullable ref
        ({"allOf": [{"$ref": "#/components/schemas/S"}]}, {"$ref": "#/components/schemas/S"}),
        (
            {"anyOf": [{"$ref": "#/components/schemas/S"}, {"type": "null"}]},
            {"oneOf": [{"type": "null"}, {"$ref": "#/components/schemas/S"}]},
        ),
    ],
)
def test_cosmetic_differences_are_equal(a: dict, b: dict) -> None:
    schemas = {"S": {"type": "object", "properties": {"x": {"type": "integer"}}}}
    s = doc({"/x": {"get": get_op(a)}}, schemas)
    t = doc({"/x": {"get": get_op(b)}}, schemas)
    r = openapi_diff([s], t)
    assert r["equal"] is True, r["changed_operations"]


@pytest.mark.parametrize(
    "a, b, kind",
    [
        ({"type": "string"}, {"type": "integer"}, "type_changed"),
        ({"type": "string", "format": "uuid"}, {"type": "string"}, "keyword_removed"),
        ({"type": "integer", "minimum": 0}, {"type": "integer"}, "keyword_removed"),
        ({"type": "object", "additionalProperties": False}, {"type": "object"}, "keyword_removed"),
        ({"type": "string"}, {"type": "string", "nullable": True}, "nullable_changed"),
        ({"enum": ["a", "b"]}, {"enum": ["a", "c"]}, "keyword_changed"),
    ],
)
def test_real_differences_are_reported(a: dict, b: dict, kind: str) -> None:
    r = openapi_diff([doc({"/x": {"get": get_op(a)}})], doc({"/x": {"get": get_op(b)}}))
    _, _, diffs = only_changed(r)
    assert [d["kind"] for d in diffs] == [kind]


def test_openapi_30_source_vs_31_target() -> None:
    s = doc(
        {"/x": {"get": get_op({"$ref": "#/components/schemas/A"})}},
        {"A": {"type": "object", "properties": {"v": {"type": "integer", "nullable": True}}}},
    )
    s["openapi"] = "3.0.3"
    t = doc(
        {"/x": {"get": get_op({"$ref": "#/components/schemas/B"})}},
        {
            "B": {
                "type": "object",
                "properties": {"v": {"type": ["integer", "null"], "format": "int32"}},
            }
        },
    )
    assert openapi_diff([s], t)["equal"] is True


def test_recursive_schemas_compare_by_structure_not_name() -> None:
    tree_a = {
        "type": "object",
        "properties": {"kids": {"type": "array", "items": {"$ref": "#/components/schemas/Node"}}},
    }
    tree_b = {
        "type": "object",
        "properties": {
            "kids": {"type": "array", "items": {"$ref": "#/components/schemas/TreeNode"}}
        },
    }
    s = doc({"/t": {"get": get_op({"$ref": "#/components/schemas/Node"})}}, {"Node": tree_a})
    t = doc(
        {"/t": {"get": get_op({"$ref": "#/components/schemas/TreeNode"})}}, {"TreeNode": tree_b}
    )
    assert openapi_diff([s], t)["equal"] is True
    tree_b["properties"]["kids"]["items"] = {"type": "string"}
    assert openapi_diff([s], t)["equal"] is False


def test_path_templates_compared_by_shape_and_param_position() -> None:
    p = lambda n: {"name": n, "in": "path", "required": True, "schema": {"type": "integer"}}  # noqa: E731
    s = doc({"/a/{x}/b/{y}": {"get": get_op({"type": "string"}, params=[p("x"), p("y")])}})
    t = doc(
        {
            "/a/{first}/b/{second}": {
                "get": get_op({"type": "string"}, params=[p("second"), p("first")])
            }
        }
    )
    assert openapi_diff([s], t)["equal"] is True
    t["paths"]["/a/{first}/b/{second}"]["get"]["parameters"][1]["schema"] = {"type": "string"}
    _, _, diffs = only_changed(openapi_diff([s], t))
    assert diffs[0]["location"] == "parameters.path#0.schema.type"
    assert path_shape("/notes/{note_id}") == path_shape("/notes/{id}") == "/notes/{}"


def test_header_names_case_insensitive_and_param_refs_resolved() -> None:
    s = doc(
        {"/x": {"get": get_op({"type": "string"}, params=[{"$ref": "#/components/parameters/T"}])}}
    )
    s["components"] = {
        "parameters": {
            "T": {
                "name": "X-Tenant-ID",
                "in": "header",
                "required": True,
                "schema": {"type": "string"},
            }
        }
    }
    t = doc(
        {
            "/x": {
                "get": get_op(
                    {"type": "string"},
                    params=[
                        {
                            "name": "x-tenant-id",
                            "in": "header",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                )
            }
        }
    )
    assert openapi_diff([s], t)["equal"] is True


def test_security_compared_by_scheme_definition() -> None:
    bearer = {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}
    s = doc({"/x": {"get": get_op({"type": "string"})}}, security=[{"BearerAuth": []}])
    s["components"] = {"securitySchemes": {"BearerAuth": bearer}}
    t = doc({"/x": {"get": {**get_op({"type": "string"}), "security": [{"bearer_auth": []}]}}})
    t["components"] = {"securitySchemes": {"bearer_auth": {"type": "http", "scheme": "Bearer"}}}
    assert openapi_diff([s], t)["equal"] is True
    t["components"]["securitySchemes"]["bearer_auth"] = {
        "type": "apiKey",
        "in": "header",
        "name": "X-Key",
    }
    _, _, diffs = only_changed(openapi_diff([s], t))
    assert [d["kind"] for d in diffs] == ["security_changed"]
    t2 = doc({"/x": {"get": get_op({"type": "string"})}})
    _, _, diffs = only_changed(openapi_diff([s], t2))
    assert diffs[0]["kind"] == "security_changed" and diffs[0]["target"] is None


# --------------------------------------------------------------------------- union


def _split_fixture() -> tuple[dict, dict]:
    a, b = src(), src()
    a["paths"] = {k: v for k, v in a["paths"].items() if k.startswith("/notes")}
    b["paths"] = {"/health": b["paths"]["/health"]}
    return a, b


def test_union_of_two_sources_vs_one_target() -> None:
    a, b = _split_fixture()
    r = openapi_diff([SpecInput("notes-svc", a), SpecInput("health-svc", b)], tgt())
    assert r["equal"] is True
    assert r["counts"]["sources"] == 2
    assert r["progress"] == {"matched": 6, "total": 6, "ratio": 1.0}
    # One source alone leaves the other's routes as extras in the target.
    r1 = openapi_diff([SpecInput("notes-svc", a)], tgt())
    assert r1["extra_operations"] == [{"method": "GET", "path": "/health"}]


def test_union_identical_operation_counted_once() -> None:
    a, b = _split_fixture()
    b["paths"]["/notes"] = copy.deepcopy(FASTAPI["paths"]["/notes"])
    r = openapi_diff([SpecInput("A", a), SpecInput("B", b)], tgt())
    assert r["equal"] is True
    assert r["counts"]["source_operations"] == 6
    t = tgt()
    del t["paths"]["/notes"]["get"]
    r = openapi_diff([SpecInput("A", a), SpecInput("B", b)], t)
    assert r["missing_operations"] == [{"method": "GET", "path": "/notes", "sources": ["A", "B"]}]


def test_union_conflict_is_reported() -> None:
    a, b = _split_fixture()
    b["paths"]["/notes/{nid}"] = {
        "delete": copy.deepcopy(FASTAPI["paths"]["/notes/{note_id}"]["delete"])
    }
    b["paths"]["/notes/{nid}"]["delete"]["parameters"][0]["name"] = "nid"
    b["paths"]["/notes/{nid}"]["delete"]["responses"]["200"] = b["paths"]["/notes/{nid}"]["delete"][
        "responses"
    ].pop("204")
    r = openapi_diff([SpecInput("A", a), SpecInput("B", b)], tgt())
    assert r["equal"] is False
    assert r["counts"]["source_conflicts"] == 1
    assert r["source_conflicts"] == [
        {
            "method": "DELETE",
            "path": "/notes/{note_id}",
            "kept": "A",
            "conflicts": [
                {
                    "source": "B",
                    "path": "/notes/{nid}",
                    "differences": [
                        {"location": "responses.200", "kind": "response_added", "target": "200"},
                        {"location": "responses.204", "kind": "response_removed", "source": "204"},
                    ],
                }
            ],
        }
    ]
    # The kept (first) definition matches the target, so it still counts as matched.
    assert r["progress"]["matched"] == 6 and r["changed_operations"] == []


# --------------------------------------------------------------------------- ignore / errors


def test_ignore_rules_paths_operations_and_status_codes() -> None:
    t = tgt()
    del t["paths"]["/health"]
    for item in t["paths"].values():
        for op in item.values():
            if isinstance(op, dict) and "responses" in op:
                op["responses"].pop("422", None)
    r = openapi_diff([src()], t, {"paths": ["/health"], "response_codes": ["422"]})
    assert r["equal"] is True
    assert r["counts"]["ignored"] == 1 and r["progress"]["total"] == 5
    r = openapi_diff([src()], t, {"operations": ["GET /health"], "response_codes": [422]})
    assert r["equal"] is True
    r = openapi_diff([src()], t, {"operations": ["* /notes/{x}*"], "paths": ["/health"]})
    assert r["progress"]["total"] == 2  # only /notes GET+POST remain, both still lack 422
    assert {c["method"] for c in r["changed_operations"]} == {"GET", "POST"}


def test_ignore_rules_from_parity_toml() -> None:
    rules = ignore_rules_from_parity_toml(
        'target_openapi_cmd = "x"\n[openapi_diff.ignore]\npaths = ["/internal/*"]\n'
        'operations = ["GET /health"]\nresponse_codes = ["422"]\n'
    )
    assert rules == IgnoreRules(("/internal/*",), ("GET /health",), ("422",))
    assert ignore_rules_from_parity_toml("source_repos = []\n") == IgnoreRules()
    with pytest.raises(OpenApiDiffError):
        ignore_rules_from_parity_toml("not = [toml")
    with pytest.raises(OpenApiDiffError):
        IgnoreRules.from_mapping({"operations": ["/no-method"]})
    with pytest.raises(OpenApiDiffError):
        IgnoreRules.from_mapping({"bogus": []})


def test_vacuous_comparison_is_never_equal() -> None:
    r = openapi_diff([doc({})], doc({}))
    assert r["vacuous"] is True and r["equal"] is False
    assert r["progress"] == {"matched": 0, "total": 0, "ratio": 0.0}


@pytest.mark.parametrize(
    "bad",
    [
        {"swagger": "2.0", "paths": {}},
        {"openapi": "4.0.0", "paths": {}},
        [],
        {"openapi": "3.1.0", "paths": []},
    ],
)
def test_unsupported_documents_raise(bad: Any) -> None:
    with pytest.raises(OpenApiDiffError):
        openapi_diff([bad], tgt())


def test_external_and_dangling_refs_raise() -> None:
    with pytest.raises(OpenApiDiffError, match="only local"):
        openapi_diff([doc({"/x": {"get": get_op({"$ref": "other.yaml#/S"})}})], tgt())
    with pytest.raises(OpenApiDiffError, match="unresolvable"):
        openapi_diff([doc({"/x": {"get": get_op({"$ref": "#/components/schemas/Nope"})}})], tgt())


def test_names_must_be_unique_and_sources_nonempty() -> None:
    with pytest.raises(OpenApiDiffError):
        openapi_diff([], tgt())
    with pytest.raises(OpenApiDiffError):
        openapi_diff([SpecInput("a", src()), SpecInput("a", src())], tgt())
    with pytest.raises(OpenApiDiffError):
        openapi_diff([SpecInput("target", src())], tgt())


def test_target_duplicate_shapes_reported() -> None:
    t = tgt()
    t["paths"]["/notes/{other}"] = {"get": copy.deepcopy(t["paths"]["/notes/{id}"]["get"])}
    r = openapi_diff([src()], t)
    assert r["target_duplicate_shapes"] == [
        {"method": "GET", "paths": ["/notes/{id}", "/notes/{other}"]}
    ]
    assert r["equal"] is False


# --------------------------------------------------------------------------- unsigned / binary


def _with_counters(d: dict[str, Any], schema_name: str, counter: dict[str, Any]) -> dict:
    d = copy.deepcopy(d)
    d["components"]["schemas"][schema_name]["properties"]["views"] = counter
    d["components"]["schemas"][schema_name]["required"].append("views")
    return d


def test_utoipa_unsigned_minimum_is_cosmetic_by_default() -> None:
    """A u32 field in the Rust port: utoipa emits format uint32 + minimum 0; the
    Python source declares a plain int. Equal by default, recorded as cosmetic."""
    s = _with_counters(src(), "NoteOut", {"type": "integer", "title": "Views"})
    t = _with_counters(tgt(), "Note", {"type": "integer", "format": "uint32", "minimum": 0})
    t["paths"]["/notes"]["get"]["parameters"][0] = {
        "name": "tag",
        "in": "query",
        "required": False,
        "schema": {"type": ["string", "null"], "minLength": 1, "maxLength": 64},
    }
    r = openapi_diff([s], t)
    assert r["equal"] is True, r["changed_operations"]
    assert r["changed_operations"] == []
    assert r["counts"]["cosmetic"] == 3  # GET /notes, POST /notes, GET /notes/{id}
    get_note = next(c for c in r["cosmetic_differences"] if c["path"] == "/notes/{note_id}")
    assert get_note["differences"] == [
        {
            "location": "responses.200.content.application/json.schema.properties.views.minimum",
            "kind": "unsigned_minimum_added",
            "target": 0,
        }
    ]
    assert r["progress"]["matched"] == 6


def test_strict_numeric_bounds_flags_unsigned_minimum() -> None:
    s = _with_counters(src(), "NoteOut", {"type": "integer"})
    t = _with_counters(tgt(), "Note", {"type": "integer", "format": "uint64", "minimum": 0})
    r = openapi_diff([s], t, strict_numeric_bounds=True)
    assert r["equal"] is False and r["strict_numeric_bounds"] is True
    assert r["cosmetic_differences"] == []
    assert r["counts"]["changed"] == 3 and r["progress"]["matched"] == 3
    kinds = {d["kind"] for c in r["changed_operations"] for d in c["differences"]}
    assert kinds == {"unsigned_minimum_added"}


@pytest.mark.parametrize(
    "a, b, kind",
    [
        # a source minimum that differs is real, not cosmetic
        ({"type": "integer", "minimum": 1}, {"type": "integer", "minimum": 0}, "keyword_changed"),
        # target REMOVES a minimum: real
        ({"type": "integer", "minimum": 0}, {"type": "integer"}, "keyword_removed"),
        # minimum 5 added: real
        ({"type": "integer"}, {"type": "integer", "minimum": 5}, "keyword_added"),
        # minimum 0 on a number (not integer): real
        ({"type": "number"}, {"type": "number", "minimum": 0}, "keyword_added"),
    ],
)
def test_only_unsigned_minimum_zero_is_cosmetic(a: dict, b: dict, kind: str) -> None:
    r = openapi_diff([doc({"/x": {"get": get_op(a)}})], doc({"/x": {"get": get_op(b)}}))
    _, _, diffs = only_changed(r)
    assert [d["kind"] for d in diffs] == [kind]
    assert r["cosmetic_differences"] == []


def test_unsigned_minimum_in_query_param_and_array_items() -> None:
    p = lambda schema: {"name": "limit", "in": "query", "schema": schema}  # noqa: E731
    s = doc(
        {
            "/x": {
                "get": get_op(
                    {"type": "array", "items": {"type": "integer"}}, params=[p({"type": "integer"})]
                )
            }
        }
    )
    t = doc(
        {
            "/x": {
                "get": get_op(
                    {
                        "type": "array",
                        "items": {"type": "integer", "format": "uint8", "minimum": 0},
                    },
                    params=[p({"type": ["integer", "null"], "format": "uint32", "minimum": 0})],
                )
            }
        }
    )
    r = openapi_diff([s], t)
    assert r["equal"] is True
    locs = [d["location"] for d in r["cosmetic_differences"][0]["differences"]]
    assert locs == [
        "parameters.query:limit.schema.minimum",
        "responses.200.content.application/json.schema.items.minimum",
    ]
    assert openapi_diff([s], t, strict_numeric_bounds=True)["equal"] is False


def _upload(schema: dict[str, Any], media: str = "multipart/form-data") -> dict[str, Any]:
    return {
        "post": {
            "requestBody": {
                "required": True,
                "content": {
                    media: {
                        "schema": {
                            "type": "object",
                            "required": ["file"],
                            "properties": {"file": schema},
                        }
                    }
                },
            },
            "responses": {"204": {"description": "ok"}},
        }
    }


@pytest.mark.parametrize(
    "a, b",
    [
        (
            {"type": "string", "format": "binary"},
            {"type": "string", "contentMediaType": "application/octet-stream"},
        ),
        (
            {"type": "string", "format": "binary"},
            {"type": "string", "contentMediaType": "image/png"},
        ),
        ({"type": "string", "format": "byte"}, {"type": "string", "contentEncoding": "base64"}),
        (
            {"type": "string", "format": "byte"},
            {"type": "string", "contentEncoding": "base64", "contentMediaType": "image/png"},
        ),
    ],
)
def test_binary_30_vs_31_file_bodies_are_equal(a: dict, b: dict) -> None:
    s = doc({"/upload": _upload(a)})
    s["openapi"] = "3.0.3"
    t = doc({"/upload": _upload(b)})
    r = openapi_diff([s], t)
    assert r["equal"] is True, r["changed_operations"]
    # octet-stream body too (whole body is the file)
    s2 = doc(
        {
            "/raw": {
                "post": {
                    "requestBody": {"content": {"application/octet-stream": {"schema": a}}},
                    "responses": {"204": {"description": "ok"}},
                }
            }
        }
    )
    t2 = doc(
        {
            "/raw": {
                "post": {
                    "requestBody": {"content": {"application/octet-stream": {"schema": b}}},
                    "responses": {"204": {"description": "ok"}},
                }
            }
        }
    )
    assert openapi_diff([s2], t2)["equal"] is True


def test_binary_vs_plain_string_and_binary_vs_base64_differ() -> None:
    s = doc({"/upload": _upload({"type": "string", "format": "binary"})})
    t = doc({"/upload": _upload({"type": "string"})})
    _, _, diffs = only_changed(openapi_diff([s], t))
    assert [d["kind"] for d in diffs] == ["keyword_removed"]
    t = doc({"/upload": _upload({"type": "string", "contentEncoding": "base64"})})
    _, _, diffs = only_changed(openapi_diff([s], t))
    assert sorted(d["kind"] for d in diffs) == ["keyword_added", "keyword_removed"]
