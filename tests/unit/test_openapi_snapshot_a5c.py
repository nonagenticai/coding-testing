"""Gate G1 for py2rust A5c: every route that existed on origin/main keeps its contract.

``tests/data/openapi_existing_routes.json`` is the OpenAPI document generated from
origin/main @ 0b757a7 (before any /api/parity route was added). New routes and new
component schemas are allowed (additive); any change to an existing path item or to an
existing component schema fails this test. (The A2 and A5b lanes pin the same baseline file,
byte-identical, so the branches merge in any order.)

Regenerate only on purpose: ``UPDATE_OPENAPI_SNAPSHOT=1 uv run pytest <this file>``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from app_factory import create_app

SNAPSHOT = Path(__file__).resolve().parent.parent / "data" / "openapi_existing_routes.json"
PARITY_POSTS = {
    "/api/parity/{container_name}/record",
    "/api/parity/{container_name}/replay",
    "/api/parity/{container_name}/compare",
}
PARITY_GETS = {"/api/parity/{container_name}/runs/{run_id}"}
PARITY_PATHS = PARITY_POSTS | PARITY_GETS


def _spec() -> dict:
    return json.loads(json.dumps(create_app().openapi(), sort_keys=True))


def test_existing_routes_openapi_unchanged() -> None:
    spec = _spec()
    if os.environ.get("UPDATE_OPENAPI_SNAPSHOT") == "1":
        SNAPSHOT.write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n")
    baseline = json.loads(SNAPSHOT.read_text())

    missing = sorted(set(baseline["paths"]) - set(spec["paths"]))
    assert not missing, f"existing routes removed: {missing}"
    changed = sorted(p for p in baseline["paths"] if baseline["paths"][p] != spec["paths"][p])
    assert not changed, f"existing route contracts changed: {changed}"

    base_schemas = baseline.get("components", {}).get("schemas", {})
    new_schemas = spec.get("components", {}).get("schemas", {})
    changed_schemas = sorted(
        name for name in base_schemas if base_schemas[name] != new_schemas.get(name)
    )
    assert not changed_schemas, f"existing component schemas changed: {changed_schemas}"
    for key in ("securitySchemes", "parameters"):
        assert baseline["components"].get(key) == spec["components"].get(key), key


def test_parity_routes_are_added_and_protected() -> None:
    spec = _spec()
    assert PARITY_PATHS <= set(spec["paths"])
    for path in PARITY_PATHS:
        method = "post" if path in PARITY_POSTS else "get"
        op = spec["paths"][path][method]
        assert op["security"] == [{"BearerAuth": []}]


def test_snapshot_is_nonvacuous() -> None:
    """The baseline must actually hold the routes it guards (not an empty file)."""
    baseline = json.loads(SNAPSHOT.read_text())
    assert len(baseline["paths"]) >= 60
    assert "/api/tests/execute" in baseline["paths"]
    assert "TestRequest" in baseline["components"]["schemas"]
    assert not PARITY_PATHS & set(baseline["paths"])
