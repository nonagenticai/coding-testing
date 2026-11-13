"""Behaviour-parity diff engine (py2rust A5c): pure library, no workspace."""

from __future__ import annotations

import copy

from services.behaviour_diff import (
    DiffOptions,
    baseline_from_expected,
    diff_behaviour,
    json_diff,
    pointer_matches,
)


def _case(name="c1", status=200, body=None, calls=None, **kw):
    return {
        "name": name,
        "status": status,
        "headers": kw.pop("headers", {"content-type": "application/json"}),
        "body": {"id": "1", "title": "t"} if body is None else body,
        "upstream_calls": calls or [],
        "upstream_unconsumed": 0,
        **kw,
    }


def _call(body=None, path="/enrich", method="POST", headers=None):
    return {
        "method": method,
        "path": path,
        "headers": headers or {"content-type": "application/json", "x-tenant-id": "t-alpha"},
        "body": {"note_id": "1", "text": "b"} if body is None else body,
    }


def _kinds(result, i=0):
    return [d["kind"] for d in result["cases"][i]["diffs"]]


def test_identical_lists_are_equal():
    base = [_case("a"), _case("b", calls=[_call()])]
    r = diff_behaviour(base, copy.deepcopy(base))
    assert r == {
        "computed": True,
        "equal": True,
        "cases_total": 2,
        "cases_passed": 2,
        "cases_failed": 0,
        "cases": [
            {"name": "a", "passed": True, "diffs": []},
            {"name": "b", "passed": True, "diffs": []},
        ],
    }


def test_zero_cases_is_never_equal():
    r = diff_behaviour([], [])
    assert r["computed"] is False and r["equal"] is False and r["cases_total"] == 0


def test_status_mismatch():
    r = diff_behaviour([_case(status=201)], [_case(status=200)])
    assert r["equal"] is False and r["cases_failed"] == 1
    assert r["cases"][0]["diffs"] == [
        {"kind": "status", "pointer": "/status", "expected": 201, "actual": 200}
    ]


def test_body_diff_has_pointers_and_key_order_is_irrelevant():
    base = _case(body={"a": 1, "items": [{"id": "x", "n": 1}], "gone": True})
    got = _case(body={"items": [{"n": 2, "id": "x"}], "a": 1, "new": None})
    r = diff_behaviour([base], [got])
    diffs = {(d["pointer"], d.get("detail")) for d in r["cases"][0]["diffs"]}
    assert diffs == {
        ("/body/gone", "missing"),
        ("/body/new", "unexpected"),
        ("/body/items/0/n", "value"),
    }


def test_list_order_and_length_matter():
    r = diff_behaviour([_case(body=[1, 2])], [_case(body=[2, 1])])
    assert [d["pointer"] for d in r["cases"][0]["diffs"]] == ["/body/0", "/body/1"]
    r = diff_behaviour([_case(body=[1, 2])], [_case(body=[1])])
    assert r["cases"][0]["diffs"][0]["detail"] == "list length"


def test_types_bool_vs_int_and_int_vs_float():
    assert json_diff(True, 1) == [("", True, 1, "type")]
    assert json_diff(1, 1.0) == []
    assert json_diff("1", 1) == [("", "1", 1, "type")]
    assert json_diff(None, {}) == [("", None, {}, "type")]


def test_ignore_paths_wildcards_and_prefixes():
    base = _case(body={"id": "a", "items": [{"ts": 1, "v": 1}, {"ts": 2, "v": 2}], "m": {"x": 1}})
    got = _case(body={"id": "b", "items": [{"ts": 9, "v": 1}, {"ts": 8, "v": 2}], "m": {"x": 2}})
    opts = DiffOptions.build(ignore_paths=["/id", "/items/*/ts", "/m"])
    assert diff_behaviour([base], [got], opts)["equal"] is True
    opts = DiffOptions.build(ignore_paths=["/**/ts", "/id"])
    r = diff_behaviour([base], [got], opts)
    assert [d["pointer"] for d in r["cases"][0]["diffs"]] == ["/body/m/x"]


def test_pointer_matches():
    assert pointer_matches("/a/b", "/a")
    assert pointer_matches("/items/3/ts", "/items/*/ts")
    assert not pointer_matches("/items/3/v", "/items/*/ts")
    assert pointer_matches("/x/y/z/ts", "/**/ts")
    assert pointer_matches("/created_at", "/created_*")
    assert pointer_matches("/anything", "/")
    assert not pointer_matches("/a", "/a/b")
    assert pointer_matches("/a~1b", "/a~1b")


def test_ignored_missing_key_is_not_reported():
    opts = DiffOptions.build(ignore_paths=["/ts"])
    r = diff_behaviour([_case(body={"ts": 1})], [_case(body={})], opts)
    assert r["equal"] is True


def test_upstream_count_order_method_path_body():
    base = _case(calls=[_call(), _call(path="/b")])
    r = diff_behaviour([base], [_case(calls=[_call()])])
    assert _kinds(r) == ["upstream_count"]
    r = diff_behaviour([base], [_case(calls=[_call(path="/b"), _call()])])
    assert _kinds(r) == ["upstream_path", "upstream_path"]
    r = diff_behaviour(
        [_case(calls=[_call()])], [_case(calls=[_call(method="PUT", body={"note_id": "2"})])]
    )
    assert _kinds(r) == ["upstream_method", "upstream_body", "upstream_body"]
    ptrs = [d["pointer"] for d in r["cases"][0]["diffs"]]
    assert ptrs == [
        "/upstream_calls/0/method",
        "/upstream_calls/0/body/note_id",
        "/upstream_calls/0/body/text",
    ]


def test_upstream_header_allowlist():
    noisy = _call(headers={"content-type": "application/json; charset=utf-8", "x-tenant-id": "t-alpha", "user-agent": "reqwest"})  # fmt: skip
    # Default allowlist (content-type by media type, x-tenant-id): user-agent ignored.
    assert diff_behaviour([_case(calls=[_call()])], [_case(calls=[noisy])])["equal"] is True
    wrong_tenant = _call(headers={"content-type": "application/json", "x-tenant-id": "t-beta"})
    r = diff_behaviour([_case(calls=[_call()])], [_case(calls=[wrong_tenant])])
    assert r["cases"][0]["diffs"][0]["pointer"] == "/upstream_calls/0/headers/x-tenant-id"
    opts = DiffOptions.build(upstream_headers=["User-Agent"])
    r = diff_behaviour([_case(calls=[_call()])], [_case(calls=[noisy])], opts)
    assert _kinds(r) == ["upstream_header"]
    opts = DiffOptions.build(upstream_headers=[])
    assert diff_behaviour([_case(calls=[_call()])], [_case(calls=[wrong_tenant])], opts)["equal"]


def test_upstream_ignore_paths_and_raw_bodies():
    opts = DiffOptions.build(upstream_ignore_paths=["/note_id"])
    r = diff_behaviour([_case(calls=[_call()])], [_case(calls=[_call(body={"note_id": "z", "text": "b"})])], opts)  # fmt: skip
    assert r["equal"] is True
    raw_a = {**_call(), "body": None, "body_text": "a=1"}
    raw_b = {**_call(), "body": None, "body_text": "a=2"}
    r = diff_behaviour([_case(calls=[raw_a])], [_case(calls=[raw_b])])
    assert r["cases"][0]["diffs"][0]["expected"] == "a=1"


def test_unscripted_and_unconsumed_upstream():
    got = _case(calls=[{**_call(), "unscripted": True}])
    r = diff_behaviour([_case(calls=[_call()])], [got])
    assert _kinds(r) == ["upstream_unscripted"]
    r = diff_behaviour([_case()], [_case(upstream_unconsumed=1)])
    assert _kinds(r) == ["upstream_unconsumed"]


def test_transport_errors_fail_the_case():
    r = diff_behaviour([_case()], [_case(error="request failed: ConnectionRefusedError")])
    assert _kinds(r) == ["error"]
    r = diff_behaviour([_case(error="boom")], [_case()])
    assert _kinds(r) == ["baseline_error"]


def test_case_mismatch_order_and_length():
    r = diff_behaviour([_case("a"), _case("b")], [_case("b"), _case("a")])
    assert r["cases_failed"] == 2 and _kinds(r) == ["case_mismatch"]
    r = diff_behaviour([_case("a")], [_case("a"), _case("extra")])
    assert r["cases_total"] == 2 and r["cases"][1]["diffs"][0]["expected"] is None


def test_response_headers_only_against_recordings():
    opts = DiffOptions.build(response_headers=["content-type", "location"])
    got = _case(headers={"content-type": "application/json; charset=utf-8", "location": "/x"})
    r = diff_behaviour([_case()], [got], opts)
    assert r["cases"][0]["diffs"] == [
        {"kind": "header", "pointer": "/headers/location", "expected": None, "actual": "/x"}
    ]
    expected = baseline_from_expected(
        [{"name": "c1", "expected_status": 200, "expected_body": {"id": "1", "title": "t"}}]
    )
    assert diff_behaviour(expected, [got], opts)["equal"] is True


def test_baseline_from_expected_uses_listed_headers_only():
    cases = [
        {
            "name": "enrich",
            "expected_status": 200,
            "expected_body": {"ok": True},
            "expected_upstream_calls": [
                {"method": "POST", "path": "/enrich", "headers": {"X-Tenant-Id": "t"}, "body": {}}
            ],
        }
    ]
    base = baseline_from_expected(cases)
    got = _case(
        name="enrich",
        body={"ok": True},
        calls=[
            {
                "method": "post",
                "path": "/enrich",
                "headers": {"x-tenant-id": "t", "content-type": "text/plain"},
                "body": {},
            }
        ],  # fmt: skip
    )
    assert diff_behaviour(base, [got])["equal"] is True
    got["upstream_calls"][0]["headers"]["x-tenant-id"] = "u"
    assert diff_behaviour(base, [got])["equal"] is False


def test_baseline_from_expected_requires_status():
    try:
        baseline_from_expected([{"name": "x"}])
    except ValueError as e:
        assert "expected_status" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_empty_body_vs_text_body():
    empty = {**_case(status=204), "body": None}
    assert diff_behaviour([empty], [dict(empty)])["equal"] is True
    r = diff_behaviour([empty], [{**empty, "body_text": "oops"}])
    assert r["cases"][0]["diffs"] == [
        {"kind": "body", "pointer": "/body", "expected": None, "actual": "oops", "detail": "type"}
    ]


def test_diffs_are_clipped():
    base = _case(body={str(i): i for i in range(80)})
    got = _case(body={str(i): -i - 1 for i in range(80)})
    r = diff_behaviour([base], [got])
    assert len(r["cases"][0]["diffs"]) == 50 and r["cases"][0]["diffs_clipped"] == 30
