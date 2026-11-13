"""Behaviour-parity diff: compare recorded case results against a baseline.

Pure library, no I/O: usable without a workspace. A *case result* is what the
in-workspace driver records per case (``behaviour_parity_driver.replay_case``)::

    {"name", "status", "headers": {...}, "body", "body_text"?, "error"?,
     "upstream_calls": [{"method", "path", "headers", "body", "body_text"?}],
     "upstream_unconsumed": int}

A *baseline* has the same shape. It is either another recording (the source
service, for ``replay``-against-recording and ``compare``) or derived from the
hand-written ``expected_*`` fields of ``parity/requests.jsonl``
(:func:`baseline_from_expected`).

Per case: status, JSON body (structural, JSON-Pointer paths, ignore patterns),
the configured response headers, and the upstream-call SEQUENCE (count, order,
method, path, allowlisted headers, body). A case is never "passed" when either
side carries a transport error, and ``equal`` is never true for zero cases.

WebSocket and SSE cases (``kind``) compare the handshake/response status, the
captured headers, and the MESSAGE sequence: a websocket case's ``frames`` per recv
group (+ the tail group) and close code, an SSE case's ``events`` and how the
stream ``ended``. Per-message normalizers: ``frame_ignore_paths`` (JSON-Pointer
patterns over the frame record, e.g. ``/data/ts`` or ``/id``) and ``drop_frames``
(``{"pointer", "equals"}``: frames matching are removed before comparing, e.g.
heartbeats). ``ordering: unordered`` compares each group as a multiset.

CLI cases (``kind: cli``, also libraries through the call harness) compare the
``exit_code`` (diff kind ``exit_code``), then stdout: structurally under ``/stdout``
honouring ``ignore_paths`` when both sides recorded ``stdout_json``, else as text
(diff kind ``stdout``); stderr (diff kind ``stderr``) only when the case sets
``compare_stderr``, because tracebacks and panics differ by nature. An expected
baseline uses ``expected_exit_code`` (required) + ``expected_stdout`` /
``expected_stdout_json`` / ``expected_stderr`` (each compared only when given).
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from typing import Any

#: Upstream request headers compared when the caller names none and the baseline is
#: a recording (an expected baseline compares exactly the names it lists per call).
DEFAULT_UPSTREAM_HEADERS: tuple[str, ...] = ("content-type", "x-tenant-id")
MAX_DIFFS_PER_CASE = 50


@dataclass(frozen=True)
class DiffOptions:
    ignore_paths: tuple[str, ...] = ()
    upstream_ignore_paths: tuple[str, ...] = ()
    upstream_headers: tuple[str, ...] | None = None
    response_headers: tuple[str, ...] = ()
    ignore_upstreams: tuple[str, ...] = ()
    frame_ignore_paths: tuple[str, ...] = ()
    drop_frames: tuple[tuple[str, str], ...] = ()  # (pointer, canonical JSON of `equals`)

    @classmethod
    def build(
        cls,
        ignore_paths: list[str] | None = None,
        upstream_ignore_paths: list[str] | None = None,
        upstream_headers: list[str] | None = None,
        response_headers: list[str] | None = None,
        ignore_upstreams: list[str] | None = None,
        frame_ignore_paths: list[str] | None = None,
        drop_frames: list[dict[str, Any]] | None = None,
    ) -> DiffOptions:
        return cls(
            frame_ignore_paths=tuple(frame_ignore_paths or ()),
            drop_frames=tuple(
                (str(d.get("pointer") or ""), _canon(d.get("equals"))) for d in (drop_frames or ())
            ),
            ignore_upstreams=tuple(ignore_upstreams or ()),
            ignore_paths=tuple(ignore_paths or ()),
            upstream_ignore_paths=tuple(upstream_ignore_paths or ()),
            upstream_headers=(
                None if upstream_headers is None else tuple(h.lower() for h in upstream_headers)
            ),
            response_headers=tuple(h.lower() for h in (response_headers or ())),
        )


@dataclass
class _Case:
    name: str
    diffs: list[dict[str, Any]] = field(default_factory=list)
    clipped: int = 0

    def add(self, kind: str, pointer: str, expected: Any, actual: Any, detail: str = "") -> None:
        if len(self.diffs) >= MAX_DIFFS_PER_CASE:
            self.clipped += 1
            return
        entry: dict[str, Any] = {
            "kind": kind,
            "pointer": pointer,
            "expected": expected,
            "actual": actual,
        }
        if detail:
            entry["detail"] = detail
        self.diffs.append(entry)


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _escape(token: str | int) -> str:
    return str(token).replace("~", "~0").replace("/", "~1")


def pointer_matches(pointer: str, pattern: str) -> bool:
    """JSON-Pointer pattern match: ``*`` = one segment (fnmatch within it), ``**`` = any depth.

    A pattern also matches everything below the node it names (ignoring ``/a``
    ignores ``/a/b``).
    """
    p_parts = pointer.split("/")[1:] if pointer else []
    q_parts = pattern.split("/")[1:] if pattern.startswith("/") else pattern.split("/")
    if pattern in ("", "/"):
        return True

    def match(i: int, j: int) -> bool:
        if j == len(q_parts):
            return True  # prefix match: the pattern names an ancestor
        if q_parts[j] == "**":
            return any(match(k, j + 1) for k in range(i, len(p_parts) + 1))
        if i == len(p_parts):
            return False
        return fnmatch.fnmatchcase(p_parts[i], q_parts[j]) and match(i + 1, j + 1)

    return match(0, 0)


def _ignored(pointer: str, patterns: tuple[str, ...]) -> bool:
    return any(pointer_matches(pointer, p) for p in patterns)


def json_diff(
    expected: Any, actual: Any, ignore: tuple[str, ...] = (), base: str = ""
) -> list[tuple[str, Any, Any, str]]:
    """Structural diff -> [(pointer, expected, actual, detail)].

    Object key order is irrelevant; list order is significant. ``base`` is the
    pointer of the compared node; ignore patterns are matched against pointers
    RELATIVE to the compared root (``/id``, ``/items/*/created_at``).
    """
    out: list[tuple[str, Any, Any, str]] = []

    def walk(e: Any, a: Any, rel: str) -> None:
        if rel and _ignored(rel, ignore):
            return
        if isinstance(e, dict) and isinstance(a, dict):
            for k in sorted(set(e) | set(a), key=str):
                child = f"{rel}/{_escape(k)}"
                if k not in a:
                    if not _ignored(child, ignore):
                        out.append((base + child, e[k], None, "missing"))
                elif k not in e:
                    if not _ignored(child, ignore):
                        out.append((base + child, None, a[k], "unexpected"))
                else:
                    walk(e[k], a[k], child)
            return
        if isinstance(e, list) and isinstance(a, list):
            if len(e) != len(a):
                out.append((base + rel, len(e), len(a), "list length"))
            for i in range(min(len(e), len(a))):
                walk(e[i], a[i], f"{rel}/{i}")
            return
        # bool is an int in Python: 1 == True must still be a difference.
        if type(e) is not type(a) and not (
            isinstance(e, (int, float))
            and isinstance(a, (int, float))
            and not isinstance(e, bool)
            and not isinstance(a, bool)
        ):
            out.append((base + rel, e, a, "type"))
        elif e != a:
            out.append((base + rel, e, a, "value"))

    walk(expected, actual, "")
    return out


def _media_type(value: Any) -> str:
    return str(value or "").split(";")[0].strip().lower()


def has_expectations(case: dict[str, Any]) -> bool:
    """Whether a case carries the expected_* field its kind's baseline needs."""
    if (case.get("kind") or "http") == "cli":
        return "expected_exit_code" in case
    return "expected_status" in case


def _cli_expected(c: dict[str, Any]) -> dict[str, Any]:
    if "expected_exit_code" not in c:
        raise ValueError(f"case {c.get('name')!r} has no expected_exit_code")
    entry: dict[str, Any] = {
        "name": c.get("name"),
        "kind": "cli",
        "exit_code": c["expected_exit_code"],
        "_expected": True,
    }
    if "expected_stdout_json" in c:
        entry["stdout_json"] = c["expected_stdout_json"]
    elif "expected_stdout" in c:
        entry["stdout"] = c["expected_stdout"]
    if "expected_stderr" in c:
        entry["stderr"] = c["expected_stderr"]
        entry["compare_stderr"] = True
    return entry


def baseline_from_expected(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Baseline case results from the ``expected_*`` fields of requests.jsonl cases."""
    out = []
    for c in cases:
        if (c.get("kind") or "http") == "cli":
            out.append(_cli_expected(c))
            continue
        if "expected_status" not in c:
            raise ValueError(f"case {c.get('name')!r} has no expected_status")
        calls = []
        for call in c.get("expected_upstream_calls") or []:
            calls.append(
                {
                    "upstream": call.get("upstream") or DEFAULT_UPSTREAM,
                    "method": call.get("method"),
                    "path": call.get("path"),
                    "headers": {k.lower(): v for k, v in (call.get("headers") or {}).items()},
                    "body": call.get("body"),
                    # The expected call names exactly the headers it constrains.
                    "_header_names": [k.lower() for k in (call.get("headers") or {})],
                }
            )
        entry: dict[str, Any] = {
            "name": c.get("name"),
            "status": c["expected_status"],
            "body": c.get("expected_body"),
            "upstream_calls": calls,
            "upstream_unconsumed": 0,
            "_expected": True,
        }
        kind = c.get("kind") or "http"
        if kind != "http":
            entry.update(kind=kind, ordering=c.get("ordering") or "strict")
            if kind == "websocket" and "expected_frames" in c:
                groups = [list(g) for g in c["expected_frames"]]
                recvs = sum(1 for st in c.get("steps") or [] if "recv" in st)
                if len(groups) == recvs:
                    groups.append([])  # no tail frames expected
                entry["frames"] = groups
            if kind == "websocket" and "expected_close" in c:
                entry["close"] = c["expected_close"]
            if kind == "sse" and "expected_events" in c:
                entry["events"] = list(c["expected_events"])
        out.append(entry)
    return out


DEFAULT_UPSTREAM = "default"
#: Bodies frameworks send for a request no route matched (axum: empty).
_NO_ROUTE_BODIES = ({"detail": "Not Found"}, {"detail": "Method Not Allowed"})


def _group_calls(calls: list[dict]) -> dict[str, list[dict]]:
    """Recorded calls per upstream, in call order (a legacy call has no ``upstream``)."""
    out: dict[str, list[dict]] = {}
    for c in calls:
        out.setdefault(str(c.get("upstream") or DEFAULT_UPSTREAM), []).append(c)
    return out


def _diff_upstreams(case: _Case, base: list[dict], got: list[dict], opts: DiffOptions) -> None:
    """Compare the call sequence PER upstream (order across upstreams is not compared).

    With only the default upstream the pointers stay ``/upstream_calls/<i>``; a named
    upstream's calls are under ``/upstreams/<name>/<i>``.
    """
    bg, gg = _group_calls(base), _group_calls(got)
    names = sorted(set(bg) | set(gg), key=lambda n: (n != DEFAULT_UPSTREAM, n))
    for name in names:
        if name in opts.ignore_upstreams:
            continue
        ptr = "/upstream_calls" if name == DEFAULT_UPSTREAM else f"/upstreams/{_escape(name)}"
        _diff_upstream(case, bg.get(name, []), gg.get(name, []), opts, ptr)


def _route_missing(baseline: dict[str, Any], actual: dict[str, Any]) -> bool:
    status = actual.get("status")
    if status not in (404, 405) or baseline.get("status") == status:
        return False
    body = actual.get("body")
    return (body is None and not str(actual.get("body_text") or "").strip()) or (
        body in _NO_ROUTE_BODIES
    )


def _diff_upstream(
    case: _Case, base: list[dict], got: list[dict], opts: DiffOptions, root: str = "/upstream_calls"
) -> None:
    if len(base) != len(got):
        case.add(
            "upstream_count",
            root,
            len(base),
            len(got),
            "; ".join(f"{c.get('method')} {c.get('path')}" for c in got)[:500],
        )
    for i in range(min(len(base), len(got))):
        b, g = base[i], got[i]
        ptr = f"{root}/{i}"
        if str(b.get("method", "")).upper() != str(g.get("method", "")).upper():
            case.add("upstream_method", f"{ptr}/method", b.get("method"), g.get("method"))
        if b.get("path") != g.get("path"):
            case.add("upstream_path", f"{ptr}/path", b.get("path"), g.get("path"))
        if "_header_names" in b:
            names: tuple[str, ...] = tuple(b["_header_names"])
        elif opts.upstream_headers is not None:
            names = opts.upstream_headers
        else:
            names = DEFAULT_UPSTREAM_HEADERS
        bh = {k.lower(): v for k, v in (b.get("headers") or {}).items()}
        gh = {k.lower(): v for k, v in (g.get("headers") or {}).items()}
        for h in names:
            bv, gv = bh.get(h), gh.get(h)
            if h == "content-type":
                bv, gv = (_media_type(bv) or None), (_media_type(gv) or None)
            if bv != gv:
                case.add("upstream_header", f"{ptr}/headers/{_escape(h)}", bv, gv)
        if g.get("unscripted"):
            case.add(
                "upstream_unscripted",
                ptr,
                None,
                f"{g.get('method')} {g.get('path')}",
                "the stub had no scripted response left for this call",
            )
        b_body = b.get("body_text") if b.get("body") is None and "body_text" in b else b.get("body")
        g_body = g.get("body_text") if g.get("body") is None and "body_text" in g else g.get("body")
        for p, e, a, d in json_diff(b_body, g_body, opts.upstream_ignore_paths, f"{ptr}/body"):
            case.add("upstream_body", p, e, a, d)


def _pointer_get(doc: Any, pointer: str) -> tuple[bool, Any]:
    """(found, value) of an exact RFC 6901 pointer."""
    if pointer in ("", "/"):
        return True, doc
    cur = doc
    for raw in pointer.split("/")[1:]:
        tok = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, dict) and tok in cur:
            cur = cur[tok]
        elif isinstance(cur, list) and tok.isdigit() and int(tok) < len(cur):
            cur = cur[int(tok)]
        else:
            return False, None
    return True, cur


def _strip(value: Any, patterns: tuple[str, ...], rel: str = "") -> Any:
    """``value`` with every node an ignore pattern names removed (for multiset compare)."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            child = f"{rel}/{_escape(k)}"
            if not _ignored(child, patterns):
                out[k] = _strip(v, patterns, child)
        return out
    if isinstance(value, list):
        return [
            _strip(v, patterns, f"{rel}/{i}")
            for i, v in enumerate(value)
            if not _ignored(f"{rel}/{i}", patterns)
        ]
    return value


def normalize_frames(frames: list[Any], opts: DiffOptions) -> list[Any]:
    """Drop the frames a ``drop_frames`` rule matches (e.g. heartbeats)."""
    out = []
    for f in frames or []:
        dropped = False
        for pointer, equals in opts.drop_frames:
            found, value = _pointer_get(f, pointer)
            if found and _canon(value) == equals:
                dropped = True
                break
        if not dropped:
            out.append(f)
    return out


def _diff_messages(
    case: _Case, root: str, base: list[Any], got: list[Any], ordering: str, opts: DiffOptions
) -> None:
    """One group of frames/events: strict = positional structural diff, else a multiset."""
    base, got = normalize_frames(base, opts), normalize_frames(got, opts)
    if ordering == "unordered":
        b_keys = [_canon(_strip(f, opts.frame_ignore_paths)) for f in base]
        g_keys = [_canon(_strip(f, opts.frame_ignore_paths)) for f in got]
        remaining = list(g_keys)
        for i, k in enumerate(b_keys):
            if k in remaining:
                remaining.remove(k)
            else:
                case.add("frame_missing", f"{root}/{i}", base[i], None, "not received (unordered)")
        extra = list(g_keys)
        for k in b_keys:
            if k in extra:
                extra.remove(k)
        for k in extra:
            j = g_keys.index(k)
            case.add("frame_unexpected", f"{root}/{j}", None, got[j], "received, not expected")
        return
    if len(base) != len(got):
        case.add("frame_count", root, len(base), len(got))
    for i in range(min(len(base), len(got))):
        for p, e, a, d in json_diff(base[i], got[i], opts.frame_ignore_paths, f"{root}/{i}"):
            case.add("frame", p, e, a, d)


def _diff_stream(baseline: dict[str, Any], actual: dict[str, Any], case: _Case, opts: DiffOptions):
    """The message part of a websocket / sse case."""
    kind = actual.get("kind")
    ordering = str(baseline.get("ordering") or actual.get("ordering") or "strict")
    if actual.get("subprotocol") != baseline.get("subprotocol") and not baseline.get("_expected"):
        case.add(
            "ws_subprotocol", "/subprotocol", baseline.get("subprotocol"), actual.get("subprotocol")
        )
    if kind == "websocket":
        if "frames" in baseline:
            bg, ag = baseline.get("frames") or [], actual.get("frames") or []
            if len(bg) != len(ag):
                case.add("frame_groups", "/frames", len(bg), len(ag), "recv groups (+ tail)")
            for g in range(min(len(bg), len(ag))):
                _diff_messages(case, f"/frames/{g}", bg[g], ag[g], ordering, opts)
        if "close" in baseline:
            bc, ac = baseline.get("close") or {}, actual.get("close") or {}
            for k in ("code", "reason", "by"):
                if k in bc or not baseline.get("_expected"):
                    if bc.get(k) != ac.get(k):
                        case.add("ws_close", f"/close/{k}", bc.get(k), ac.get(k))
    else:
        if "events" in baseline:
            _diff_messages(
                case, "/events", baseline.get("events") or [], actual.get("events") or [],
                ordering, opts,
            )  # fmt: skip
        if not baseline.get("_expected") and baseline.get("ended") != actual.get("ended"):
            case.add("sse_end", "/ended", baseline.get("ended"), actual.get("ended"))


def diff_case(baseline: dict[str, Any], actual: dict[str, Any], opts: DiffOptions) -> _Case:
    case = _Case(name=str(actual.get("name") or baseline.get("name")))
    if baseline.get("error"):
        case.add("baseline_error", "", None, None, str(baseline["error"]))
    if actual.get("error"):
        case.add("error", "", None, None, str(actual["error"]))
    if case.diffs:
        return case
    b_kind, a_kind = baseline.get("kind") or "http", actual.get("kind") or "http"
    if b_kind != a_kind:
        case.add("kind", "/kind", b_kind, a_kind, "the case was recorded as another kind")
        return case
    if a_kind == "cli":
        _diff_cli(baseline, actual, case, opts)
        return case
    if _route_missing(baseline, actual):
        # A HEURISTIC, not a proof: an app route that deliberately answers 404 with an
        # empty or {"detail": "Not Found"} body looks identical. So it only names the
        # likely cause of the status difference; every other comparison still runs.
        case.add(
            "route_missing",
            "/status",
            baseline.get("status"),
            actual.get("status"),
            f"heuristic route_missing: {actual.get('status')} with an empty or framework-"
            f"default body for {actual.get('method')} {actual.get('path')} is what a "
            "framework answers when NO route matched (route not implemented, or a wrong "
            "route_map); an app route that deliberately answers this looks the same, so "
            "the header/body/upstream differences below are still compared",
        )
    elif baseline.get("status") != actual.get("status"):
        case.add("status", "/status", baseline.get("status"), actual.get("status"))
    if not baseline.get("_expected"):
        bh = baseline.get("headers") or {}
        ah = actual.get("headers") or {}
        for h in opts.response_headers:
            bv, av = bh.get(h), ah.get(h)
            if h == "content-type":
                bv, av = (_media_type(bv) or None), (_media_type(av) or None)
            if bv != av:
                case.add("header", f"/headers/{_escape(h)}", bv, av)
    b_body = baseline.get("body")
    a_body = actual.get("body")
    if b_body is None and "body_text" in baseline:
        b_body = baseline["body_text"]
    if a_body is None and "body_text" in actual:
        a_body = actual["body_text"]
    if a_kind == "http" or b_body is not None or a_body is not None:
        for p, e, a, d in json_diff(b_body, a_body, opts.ignore_paths, "/body"):
            case.add("body", p, e, a, d)
    if a_kind != "http":
        _diff_stream(baseline, actual, case, opts)
    _diff_upstreams(
        case, baseline.get("upstream_calls") or [], actual.get("upstream_calls") or [], opts
    )
    left = int(actual.get("upstream_unconsumed") or 0)
    if left:
        case.add(
            "upstream_unconsumed",
            "/upstream_unconsumed",
            0,
            left,
            "scripted upstream responses were never requested",
        )
    return case


def _diff_cli(baseline: dict[str, Any], actual: dict[str, Any], case: _Case, opts: DiffOptions):
    """exit code, stdout (JSON-structural or text), stderr when the case asks for it."""
    expected_only = bool(baseline.get("_expected"))
    if baseline.get("exit_code") != actual.get("exit_code"):
        case.add("exit_code", "/exit_code", baseline.get("exit_code"), actual.get("exit_code"))
    if "stdout_json" in baseline and "stdout_json" in actual:
        for p, e, a, d in json_diff(
            baseline["stdout_json"], actual["stdout_json"], opts.ignore_paths, "/stdout"
        ):
            case.add("stdout", p, e, a, d)
    elif not expected_only or "stdout" in baseline or "stdout_json" in baseline:
        b_out = baseline.get("stdout_json", baseline.get("stdout"))
        a_out = actual.get("stdout_json", actual.get("stdout"))
        if b_out != a_out or ("stdout_json" in baseline) != ("stdout_json" in actual):
            detail = actual.get("stdout_json_error") or baseline.get("stdout_json_error") or ""
            case.add("stdout", "/stdout", b_out, a_out, detail)
    if baseline.get("compare_stderr") or actual.get("compare_stderr"):
        if baseline.get("stderr") != actual.get("stderr"):
            case.add("stderr", "/stderr", baseline.get("stderr"), actual.get("stderr"))


def _route(entry: dict[str, Any], a: dict[str, Any] | None, b: dict[str, Any] | None) -> None:
    """method/path as recorded for the source + as sent to the target (when known)."""
    src = a if a is not None else b
    if not src:
        return
    sent_m, sent_p = src.get("method"), src.get("path")
    rec_m = src.get("source_method") or sent_m
    rec_p = src.get("source_path") or sent_p
    if rec_m is not None:
        entry["method"] = str(rec_m).upper()
    if rec_p is not None:
        entry["path"] = rec_p
    if a is not None and sent_p is not None:
        entry["target_method"] = str(sent_m).upper() if sent_m else None
        entry["target_path"] = sent_p


def diff_behaviour(
    baseline: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    opts: DiffOptions | None = None,
    *,
    source_id: str | None = None,
) -> dict[str, Any]:
    """Compare two ordered case lists -> {computed, equal, cases_total/passed/failed, cases}.

    Cases are matched by position AND name: a reordered, missing or extra case is
    itself a difference (the replay is stateful, order is part of the contract).
    """
    opts = opts or DiffOptions()
    results = []
    for i in range(max(len(baseline), len(actual))):
        b = baseline[i] if i < len(baseline) else None
        a = actual[i] if i < len(actual) else None
        if b is None or a is None or b.get("name") != a.get("name"):
            name = str((a or b or {}).get("name"))
            c = _Case(name=name)
            c.add(
                "case_mismatch",
                f"/cases/{i}",
                None if b is None else b.get("name"),
                None if a is None else a.get("name"),
                "case lists differ in order, length or names",
            )
        else:
            c = diff_case(b, a, opts)
        entry: dict[str, Any] = {"name": c.name, "passed": not c.diffs, "diffs": c.diffs}
        kind = (a or b or {}).get("kind")
        if kind and kind != "http":
            entry["kind"] = kind
        if source_id is not None:
            entry["source_id"] = source_id
        _route(entry, a, b)
        if c.clipped:
            entry["diffs_clipped"] = c.clipped
        results.append(entry)
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    computed = total > 0
    return {
        "computed": computed,
        "equal": computed and passed == total,
        "cases_total": total,
        "cases_passed": passed,
        "cases_failed": total - passed,
        "cases": results,
    }
