"""Contract parity: diff one target OpenAPI document against the UNION of N sources.

Pure library, no I/O. The acceptance criterion of a Python -> Rust rewrite is that
the Rust service exposes the same public HTTP contract as the Python service(s) it
replaces; for a re-architecture several Python services collapse into one Rust
target, so the target is compared against the union of every source spec. The
result is computed, never judged: ``equal`` is true only when every source
operation exists in the target with an identical normalised contract, the target
exposes nothing extra, and the sources do not contradict each other.

Normalisation (applied to both sides before comparing):

* local ``$ref`` (``#/...``) resolved inline, for schemas, parameters, request
  bodies and responses; a recursive reference becomes ``{"$recursive": n}`` where
  ``n`` is how many levels up the cycle closes (component names never matter);
* documentation-only keys dropped: ``title``, ``description``, ``summary``,
  ``example(s)``, ``operationId``, ``tags``, ``externalDocs``, ``deprecated``,
  ``readOnly``/``writeOnly``, ``xml``, ``$comment``, ``x-*`` extensions;
* null-ability has one canonical form: OpenAPI 3.0 ``nullable: true``, 3.1
  ``type: [X, "null"]``, and ``anyOf``/``oneOf`` with a ``{"type": "null"}``
  branch all become ``{"nullable": true, ...X}``; a single remaining branch is
  merged into its parent, ``allOf`` with one member likewise; ``oneOf`` is read
  as ``anyOf`` (FastAPI emits anyOf where utoipa emits oneOf);
* ``format`` is dropped on ``integer``/``number`` (utoipa's int32/int64/float/
  double vs FastAPI's none); string formats (uuid, date-time, ...) are kept;
* 3.0 boolean ``exclusiveMinimum``/``exclusiveMaximum`` become the 3.1 numeric
  form; ``enum: [x]`` becomes ``const: x``; ``additionalProperties: true`` and
  ``default: null`` are dropped (both are the absent meaning);
* parameters (query/header/cookie) ignore null-ability (a query string cannot
  carry JSON null); header names compare case-insensitively; path parameters
  are keyed by position, so ``/notes/{id}`` == ``/notes/{note_id}``;
* security requirements compare by the scheme *definition* (type, scheme, in,
  name) and scopes, not by the component name.

Ordering never matters (dict keys, ``required``, ``enum``, ``type`` lists) and
every list in the result is sorted, so the output is deterministic.

N sources -> 1 target: each source may carry a ``path_prefix_map`` (``{"/":
"/notes/"}``) applied to its paths BEFORE comparison (same segment-aligned
longest-prefix rule as the behaviour driver's ``route_map``), so a source whose
routes moved under a prefix in the merged target compares equal instead of
missing + extra. ``auth`` decides how security differences count: ``compare``
(default: a ``security_changed`` difference like any other), ``separate`` (moved
to ``auth_differences``, still failing ``equal``) or ``ignore`` (moved there,
informational) - unifying auth across sources then does not flag every operation.
"""

from __future__ import annotations

import copy
import fnmatch
import json
import re
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from services.behaviour_parity_driver import map_route

HTTP_METHODS = ("get", "put", "post", "delete", "options", "head", "patch", "trace")

#: Keys that document a schema/operation but are not part of its wire contract.
DOC_KEYS = frozenset(
    {
        "title",
        "description",
        "summary",
        "example",
        "examples",
        "operationId",
        "tags",
        "externalDocs",
        "deprecated",
        "readOnly",
        "writeOnly",
        "xml",
        "$comment",
        "$schema",
        "$id",
    }
)

#: A difference kind that is informational unless ``strict_numeric_bounds``.
UNSIGNED_MINIMUM_KIND = "unsigned_minimum_added"
COSMETIC_KINDS = frozenset({UNSIGNED_MINIMUM_KIND})
AuthMode = Literal["compare", "separate", "ignore"]
AUTH_MODES = ("compare", "separate", "ignore")
#: Header parameters that carry credentials (compared as auth, not as a parameter).
AUTH_HEADER_LOCATIONS = frozenset({"parameters.header:authorization"})

_PATH_PARAM_RE = re.compile(r"\{[^}/]*\}")


class OpenApiDiffError(ValueError):
    """The input cannot be compared (not an OpenAPI 3.x document, external $ref, ...)."""


# --------------------------------------------------------------------------- ignore


@dataclass(frozen=True)
class IgnoreRules:
    """Operations excluded from the comparison (on both sides).

    * ``paths``: glob patterns over the path template (``/internal/*``); path
      parameters may be written with any name or as ``{}``.
    * ``operations``: ``"METHOD /path"`` entries (method case-insensitive, ``*``
      for any method; the path is a glob as above).
    * ``response_codes``: status codes ignored in every operation's responses
      (e.g. ``"422"`` when the target does not document validation errors).
    """

    paths: tuple[str, ...] = ()
    operations: tuple[str, ...] = ()
    response_codes: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> IgnoreRules:
        if not data:
            return cls()
        unknown = set(data) - {"paths", "operations", "response_codes"}
        if unknown:
            raise OpenApiDiffError(f"unknown ignore keys: {sorted(unknown)}")

        def _strs(key: str) -> tuple[str, ...]:
            value = data.get(key) or []
            if not isinstance(value, list) or not all(isinstance(v, str | int) for v in value):
                raise OpenApiDiffError(f"ignore.{key} must be a list of strings")
            return tuple(sorted(str(v) for v in value))

        ops = _strs("operations")
        for op in ops:
            if len(op.split()) != 2:
                raise OpenApiDiffError(f"ignore.operations entry {op!r} is not 'METHOD /path'")
        codes = tuple(sorted(c.upper() for c in _strs("response_codes")))
        return cls(paths=_strs("paths"), operations=ops, response_codes=codes)

    def matches(self, method: str, path: str) -> bool:
        shape = path_shape(path)
        for pattern in self.paths:
            if fnmatch.fnmatchcase(shape, path_shape(pattern)):
                return True
        for entry in self.operations:
            m, p = entry.split()
            if m != "*" and m.lower() != method.lower():
                continue
            if fnmatch.fnmatchcase(shape, path_shape(p)):
                return True
        return False

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "paths": list(self.paths),
            "operations": list(self.operations),
            "response_codes": list(self.response_codes),
        }


def ignore_rules_from_parity_toml(text: str) -> IgnoreRules:
    """Read ``[openapi_diff.ignore]`` (paths / operations / response_codes) from parity.toml."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise OpenApiDiffError(f"parity.toml is not valid TOML: {e}") from e
    section = data.get("openapi_diff", {})
    if not isinstance(section, dict):
        raise OpenApiDiffError("[openapi_diff] must be a table")
    return IgnoreRules.from_mapping(section.get("ignore"))


# --------------------------------------------------------------------------- helpers


def path_shape(path: str) -> str:
    """``/notes/{note_id}`` -> ``/notes/{}`` (templates compared by shape)."""
    return _PATH_PARAM_RE.sub("{}", path)


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


class _Resolver:
    def __init__(self, doc: Mapping[str, Any]):
        self.doc = doc

    def lookup(self, ref: str) -> Any:
        if not isinstance(ref, str) or not ref.startswith("#"):
            raise OpenApiDiffError(f"only local $ref is supported, got {ref!r}")
        node: Any = self.doc
        for raw in ref[1:].lstrip("/").split("/") if ref != "#" else []:
            part = raw.replace("~1", "/").replace("~0", "~")
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
                node = node[int(part)]
            else:
                raise OpenApiDiffError(f"unresolvable $ref {ref!r}")
        return node

    def deref(self, obj: Any, seen: tuple[str, ...] = ()) -> Any:
        """Follow a chain of $ref on a non-schema object (parameter, response, ...)."""
        while isinstance(obj, Mapping) and "$ref" in obj:
            ref = obj["$ref"]
            if ref in seen:
                raise OpenApiDiffError(f"circular $ref {ref!r}")
            seen = (*seen, ref)
            obj = self.lookup(ref)
        return obj


# --------------------------------------------------------------------------- schemas


def _is_null_schema(s: Any) -> bool:
    return isinstance(s, Mapping) and (
        s.get("type") == "null"
        or (isinstance(s.get("type"), list) and list(s["type"]) == ["null"])
        or ("const" in s and s["const"] is None and len(s) == 1)
        or (s.get("enum") == [None] and len(s) == 1)
    )


def _merge(parent: dict[str, Any], branch: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(parent)
    for k, v in branch.items():
        if k == "nullable":
            out["nullable"] = bool(out.get("nullable")) or bool(v)
        elif k not in out:
            out[k] = v
        elif k == "required" and isinstance(v, list):
            out[k] = sorted(set(out[k]) | set(v))
        else:
            # Conflicting keyword between wrapper and branch: keep both visible.
            out[k] = v if out[k] == v else {"allOf": [out[k], v]}
    return out


def normalize_schema(
    schema: Any,
    resolver: _Resolver,
    *,
    stack: tuple[str, ...] = (),
    in_parameter: bool = False,
) -> Any:
    """Canonical, comparable form of a JSON Schema (see module docstring)."""
    if isinstance(schema, bool):
        return {} if schema else {"not": {}}
    if not isinstance(schema, Mapping):
        return schema

    if "$ref" in schema:
        ref = schema["$ref"]
        if ref in stack:
            return {"$recursive": len(stack) - stack.index(ref)}
        target = resolver.lookup(ref)
        siblings = {k: v for k, v in schema.items() if k != "$ref"}
        resolved = normalize_schema(
            target, resolver, stack=(*stack, ref), in_parameter=in_parameter
        )
        if siblings:
            sib = normalize_schema(siblings, resolver, stack=stack, in_parameter=in_parameter)
            if sib:
                resolved = _merge(resolved, sib) if isinstance(resolved, dict) else resolved
        return resolved

    s: dict[str, Any] = {
        k: v for k, v in schema.items() if k not in DOC_KEYS and not k.startswith("x-")
    }
    nullable = bool(s.pop("nullable", False))

    # 3.1 type lists
    t = s.get("type")
    if isinstance(t, list):
        types = sorted({str(x) for x in t})
        if "null" in types:
            nullable = True
            types = [x for x in types if x != "null"]
        if len(types) == 1:
            s["type"] = types[0]
        elif types:
            s["type"] = types
        else:
            s.pop("type")
            if not any(k in s for k in ("anyOf", "oneOf", "allOf", "properties", "items")):
                return {"nullable": True, "type": "null"} if not in_parameter else {}

    # oneOf is read as anyOf; null branches lift to the nullable flag.
    if "oneOf" in s:
        branches = s.pop("oneOf")
        s["anyOf"] = list(s.get("anyOf", [])) + list(branches)
    if "anyOf" in s:
        branches = [
            normalize_schema(b, resolver, stack=stack, in_parameter=in_parameter)
            for b in s.pop("anyOf")
        ]
        kept = []
        for b in branches:
            if _is_null_schema(b) or b == {"nullable": True, "type": "null"}:
                nullable = True
            else:
                if isinstance(b, dict) and b.get("nullable"):
                    nullable = True
                    b = {k: v for k, v in b.items() if k != "nullable"}
                kept.append(b)
        uniq = {_canon(b): b for b in kept}
        kept = [uniq[k] for k in sorted(uniq)]
        if len(kept) == 1:
            s = _merge(s, kept[0])
        elif kept:
            s["anyOf"] = kept
    if "allOf" in s:
        members = [
            normalize_schema(b, resolver, stack=stack, in_parameter=in_parameter)
            for b in s.pop("allOf")
        ]
        members = [m for m in members if m != {}]
        if len(members) == 1:
            s = _merge(s, members[0])
        elif members:
            uniq = {_canon(m): m for m in members}
            s["allOf"] = [uniq[k] for k in sorted(uniq)]
    if s.get("nullable"):
        nullable = True
        s.pop("nullable")

    # Recurse into sub-schemas.
    for key in ("items", "additionalProperties", "not", "contains", "propertyNames"):
        if key in s and isinstance(s[key], Mapping | bool):
            s[key] = normalize_schema(s[key], resolver, stack=stack)
    for key in ("properties", "patternProperties", "dependentSchemas"):
        if key in s and isinstance(s[key], Mapping):
            s[key] = {
                name: normalize_schema(sub, resolver, stack=stack) for name, sub in s[key].items()
            }
    if "prefixItems" in s and isinstance(s["prefixItems"], list):
        s["prefixItems"] = [normalize_schema(x, resolver, stack=stack) for x in s["prefixItems"]]

    # Cosmetic equivalences.
    # Binary / base64 strings: 3.0 ``format: binary|byte`` vs 3.1
    # ``contentMediaType`` / ``contentEncoding: base64`` (file bodies).
    if s.get("contentEncoding") == "base64" or s.get("format") == "byte":
        s.pop("contentEncoding", None)
        s.pop("contentMediaType", None)
        s.pop("format", None)
        s["$base64"] = True
    elif s.get("format") == "binary" or "contentMediaType" in s:
        s.pop("contentMediaType", None)
        s.pop("format", None)
        s.pop("contentEncoding", None)
        s["$binary"] = True
    if s.get("additionalProperties") == {}:
        s.pop("additionalProperties")  # true == absent
    if "default" in s and s["default"] is None:
        s.pop("default")
    if s.get("type") in ("integer", "number"):
        s.pop("format", None)
    for bound, base in (("exclusiveMinimum", "minimum"), ("exclusiveMaximum", "maximum")):
        if isinstance(s.get(bound), bool):
            if s[bound] and base in s:
                s[bound] = s.pop(base)
            else:
                s.pop(bound)
    if "enum" in s and isinstance(s["enum"], list):
        values = [v for v in s["enum"] if v is not None]
        if len(values) != len(s["enum"]):
            nullable = True
        uniq = {_canon(v): v for v in values}
        values = [uniq[k] for k in sorted(uniq)]
        if len(values) == 1 and "const" not in s:
            s.pop("enum")
            s["const"] = values[0]
        else:
            s["enum"] = values
    if "required" in s and isinstance(s["required"], list):
        s["required"] = sorted(set(s["required"]))
        if not s["required"]:
            s.pop("required")

    if nullable and not in_parameter:
        s["nullable"] = True
    return s


# --------------------------------------------------------------------------- operations


@dataclass
class _Op:
    method: str
    path: str
    contract: dict[str, Any]
    source: str = ""
    sources: list[str] = field(default_factory=list)
    source_path: str | None = None  # the path as the source declares it, when mapped

    @property
    def key(self) -> tuple[str, str]:
        return (path_shape(self.path), self.method)


def _media_map(content: Any, resolver: _Resolver) -> dict[str, Any]:
    if not isinstance(content, Mapping):
        return {}
    out: dict[str, Any] = {}
    for media, obj in content.items():
        obj = resolver.deref(obj)
        schema = obj.get("schema") if isinstance(obj, Mapping) else None
        out[str(media).split(";")[0].strip().lower()] = (
            normalize_schema(schema, resolver) if schema is not None else None
        )
    return out


def _security_schemes(doc: Mapping[str, Any], resolver: _Resolver) -> dict[str, dict[str, Any]]:
    schemes = (doc.get("components") or {}).get("securitySchemes") or {}
    out: dict[str, dict[str, Any]] = {}
    for name, raw in schemes.items():
        d = resolver.deref(raw)
        if not isinstance(d, Mapping):
            continue
        canon: dict[str, Any] = {"type": str(d.get("type", "")).lower()}
        if canon["type"] == "http":
            canon["scheme"] = str(d.get("scheme", "")).lower()
        elif canon["type"] == "apikey":
            canon["in"] = d.get("in")
            canon["name"] = str(d.get("name", "")).lower()
        elif canon["type"] == "openidconnect":
            canon["openIdConnectUrl"] = d.get("openIdConnectUrl")
        elif canon["type"] == "oauth2":
            canon["flows"] = sorted((d.get("flows") or {}).keys())
        out[name] = canon
    return out


def _normalize_security(
    reqs: Any, schemes: Mapping[str, dict[str, Any]]
) -> list[list[dict[str, Any]]] | None:
    if reqs is None:
        return None
    alternatives: dict[str, list[dict[str, Any]]] = {}
    for req in reqs:
        if not isinstance(req, Mapping):
            continue
        parts = []
        for name, scopes in req.items():
            scheme = schemes.get(name, {"type": "unknown", "name": name})
            parts.append({"scheme": scheme, "scopes": sorted(scopes or [])})
        parts.sort(key=_canon)
        alternatives[_canon(parts)] = parts
    return [alternatives[k] for k in sorted(alternatives)]


def _param_key(param: Mapping[str, Any], path_names: list[str]) -> str:
    loc = str(param.get("in", ""))
    name = str(param.get("name", ""))
    if loc == "path":
        idx = path_names.index(name) if name in path_names else -1
        return f"path#{idx}" if idx >= 0 else f"path:{name}"
    if loc == "header":
        return f"header:{name.lower()}"
    return f"{loc}:{name}"


def _operation_contract(
    path: str,
    path_item: Mapping[str, Any],
    op: Mapping[str, Any],
    resolver: _Resolver,
    schemes: Mapping[str, dict[str, Any]],
    doc_security: Any,
    ignore: IgnoreRules,
) -> dict[str, Any]:
    path_names = [m.group(0)[1:-1] for m in _PATH_PARAM_RE.finditer(path)]
    params: dict[str, Any] = {}
    for raw in list(path_item.get("parameters") or []) + list(op.get("parameters") or []):
        p = resolver.deref(raw)
        if not isinstance(p, Mapping):
            continue
        key = _param_key(p, path_names)
        loc = p.get("in")
        schema = p.get("schema")
        if schema is None and isinstance(p.get("content"), Mapping) and p["content"]:
            schema = next(iter(resolver.deref(v) for v in p["content"].values())).get("schema")
        norm = normalize_schema(schema, resolver, in_parameter=True) if schema is not None else None
        params[key] = {
            "required": True if loc == "path" else bool(p.get("required", False)),
            "schema": norm,
        }

    body = None
    if op.get("requestBody") is not None:
        rb = resolver.deref(op["requestBody"])
        body = {
            "required": bool(rb.get("required", False)),
            "content": _media_map(rb.get("content"), resolver),
        }

    responses: dict[str, Any] = {}
    for code, raw in (op.get("responses") or {}).items():
        code = str(code).upper()
        if code in ignore.response_codes:
            continue
        r = resolver.deref(raw)
        if not isinstance(r, Mapping):
            continue
        headers = {}
        for hname, hraw in (r.get("headers") or {}).items():
            h = resolver.deref(hraw)
            hs = h.get("schema") if isinstance(h, Mapping) else None
            headers[str(hname).lower()] = (
                normalize_schema(hs, resolver, in_parameter=True) if hs is not None else None
            )
        entry: dict[str, Any] = {"content": _media_map(r.get("content"), resolver)}
        if headers:
            entry["headers"] = headers
        responses[code] = entry

    security = op.get("security", doc_security)
    return {
        "parameters": params,
        "request_body": body,
        "responses": responses,
        "security": _normalize_security(security, schemes),
    }


def _check_doc(doc: Any, label: str) -> None:
    if not isinstance(doc, Mapping):
        raise OpenApiDiffError(f"{label}: not a JSON object")
    version = doc.get("openapi")
    if not isinstance(version, str) or not version.startswith("3."):
        raise OpenApiDiffError(
            f"{label}: only OpenAPI 3.0/3.1 is supported (openapi={version!r}, "
            f"swagger={doc.get('swagger')!r})"
        )
    if not isinstance(doc.get("paths", {}), Mapping):
        raise OpenApiDiffError(f"{label}: 'paths' is not an object")


def extract_operations(
    doc: Mapping[str, Any],
    label: str,
    ignore: IgnoreRules,
    path_prefix_map: Iterable[tuple[str, str]] = (),
) -> tuple[list[_Op], list[dict[str, str]]]:
    """Normalised operations of one document, plus the ones the ignore rules excluded.

    ``path_prefix_map`` rewrites each path before it is keyed; an ignore rule
    matching either the declared or the mapped path excludes the operation.
    """
    rules = [{"from": f, "to": t} for f, t in path_prefix_map]
    _check_doc(doc, label)
    doc = copy.deepcopy(dict(doc))
    resolver = _Resolver(doc)
    schemes = _security_schemes(doc, resolver)
    ops: list[_Op] = []
    ignored: list[dict[str, str]] = []
    for path in sorted(doc.get("paths") or {}):
        item = resolver.deref(doc["paths"][path])
        if not isinstance(item, Mapping):
            continue
        mapped = map_route("GET", path, rules)[1] if rules else path
        for method in HTTP_METHODS:
            op = item.get(method)
            if not isinstance(op, Mapping):
                continue
            if ignore.matches(method, path) or ignore.matches(method, mapped):
                ignored.append({"method": method.upper(), "path": path, "document": label})
                continue
            contract = _operation_contract(
                path, item, op, resolver, schemes, doc.get("security"), ignore
            )
            ops.append(
                _Op(
                    method=method.upper(),
                    path=mapped,
                    contract=contract,
                    source=label,
                    source_path=path if mapped != path else None,
                )
            )
    return ops, ignored


# --------------------------------------------------------------------------- diffing


def _diff_schema(a: Any, b: Any, loc: str, out: list[dict[str, Any]]) -> None:
    if a == b:
        return
    if not isinstance(a, dict) or not isinstance(b, dict):
        out.append({"location": loc, "kind": "schema_changed", "source": a, "target": b})
        return
    for key in sorted(set(a) | set(b)):
        here = f"{loc}.{key}" if loc else key
        av, bv = a.get(key), b.get(key)
        if av == bv:
            continue
        if key == "properties" and isinstance(av, dict | None) and isinstance(bv, dict | None):
            av, bv = av or {}, bv or {}
            for prop in sorted(set(av) | set(bv)):
                ploc = f"{here}.{prop}"
                if prop not in bv:
                    out.append({"location": ploc, "kind": "property_removed", "source": av[prop]})
                elif prop not in av:
                    out.append({"location": ploc, "kind": "property_added", "target": bv[prop]})
                else:
                    _diff_schema(av[prop], bv[prop], ploc, out)
        elif key == "required":
            sa, sb = set(av or []), set(bv or [])
            for name in sorted(sa - sb):
                out.append({"location": here, "kind": "required_removed", "source": name})
            for name in sorted(sb - sa):
                out.append({"location": here, "kind": "required_added", "target": name})
        elif key == "type":
            out.append({"location": here, "kind": "type_changed", "source": av, "target": bv})
        elif key == "nullable":
            out.append(
                {
                    "location": here,
                    "kind": "nullable_changed",
                    "source": bool(av),
                    "target": bool(bv),
                }
            )
        elif key == "minimum" and key not in a and bv == 0 and b.get("type") == "integer":
            # utoipa adds minimum 0 to every unsigned integer (u8..u64).
            out.append({"location": here, "kind": UNSIGNED_MINIMUM_KIND, "target": bv})
        elif key not in a:
            out.append({"location": here, "kind": "keyword_added", "target": bv})
        elif key not in b:
            out.append({"location": here, "kind": "keyword_removed", "source": av})
        elif isinstance(av, dict) and isinstance(bv, dict) and key in ("items", "not"):
            _diff_schema(av, bv, here, out)
        else:
            out.append({"location": here, "kind": "keyword_changed", "source": av, "target": bv})


def _diff_media(a: dict[str, Any], b: dict[str, Any], loc: str, out: list[dict[str, Any]]) -> None:
    for media in sorted(set(a) | set(b)):
        here = f"{loc}.{media}"
        if media not in b:
            out.append({"location": here, "kind": "media_type_removed", "source": media})
        elif media not in a:
            out.append({"location": here, "kind": "media_type_added", "target": media})
        else:
            _diff_schema(a[media], b[media], f"{here}.schema", out)


def diff_contracts(a: dict[str, Any], b: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-operation differences, source ``a`` vs target ``b`` (sorted)."""
    out: list[dict[str, Any]] = []
    pa, pb = a["parameters"], b["parameters"]
    for key in sorted(set(pa) | set(pb)):
        loc = f"parameters.{key}"
        if key not in pb:
            out.append({"location": loc, "kind": "parameter_removed", "source": pa[key]})
        elif key not in pa:
            out.append({"location": loc, "kind": "parameter_added", "target": pb[key]})
        else:
            if pa[key]["required"] != pb[key]["required"]:
                out.append(
                    {
                        "location": loc,
                        "kind": "parameter_required_changed",
                        "source": pa[key]["required"],
                        "target": pb[key]["required"],
                    }
                )
            _diff_schema(pa[key]["schema"], pb[key]["schema"], f"{loc}.schema", out)

    ba, bb = a["request_body"], b["request_body"]
    if ba is None and bb is not None:
        out.append({"location": "request_body", "kind": "request_body_added", "target": bb})
    elif ba is not None and bb is None:
        out.append({"location": "request_body", "kind": "request_body_removed", "source": ba})
    elif ba is not None and bb is not None:
        if ba["required"] != bb["required"]:
            out.append(
                {
                    "location": "request_body",
                    "kind": "request_body_required_changed",
                    "source": ba["required"],
                    "target": bb["required"],
                }
            )
        _diff_media(ba["content"], bb["content"], "request_body.content", out)

    ra, rb = a["responses"], b["responses"]
    for code in sorted(set(ra) | set(rb)):
        loc = f"responses.{code}"
        if code not in rb:
            out.append({"location": loc, "kind": "response_removed", "source": code})
        elif code not in ra:
            out.append({"location": loc, "kind": "response_added", "target": code})
        else:
            _diff_media(ra[code]["content"], rb[code]["content"], f"{loc}.content", out)
            ha, hb = ra[code].get("headers", {}), rb[code].get("headers", {})
            for h in sorted(set(ha) | set(hb)):
                hloc = f"{loc}.headers.{h}"
                if h not in hb:
                    out.append({"location": hloc, "kind": "response_header_removed", "source": h})
                elif h not in ha:
                    out.append({"location": hloc, "kind": "response_header_added", "target": h})
                else:
                    _diff_schema(ha[h], hb[h], f"{hloc}.schema", out)

    if a["security"] != b["security"]:
        out.append(
            {
                "location": "security",
                "kind": "security_changed",
                "source": a["security"],
                "target": b["security"],
            }
        )
    out.sort(key=lambda d: (d["location"], d["kind"], _canon(d)))
    return out


# --------------------------------------------------------------------------- entry point


@dataclass(frozen=True)
class SpecInput:
    """One named OpenAPI document."""

    name: str
    document: Mapping[str, Any]
    path_prefix_map: tuple[tuple[str, str], ...] = ()


def _is_auth(diff: Mapping[str, Any]) -> bool:
    return diff["kind"] == "security_changed" or diff["location"] in AUTH_HEADER_LOCATIONS


def _named(docs: Iterable[SpecInput | Mapping[str, Any]], prefix: str) -> list[SpecInput]:
    out = []
    for i, d in enumerate(docs):
        out.append(d if isinstance(d, SpecInput) else SpecInput(name=f"{prefix}[{i}]", document=d))
    names = [s.name for s in out]
    if len(set(names)) != len(names):
        raise OpenApiDiffError(f"{prefix} names must be unique: {names}")
    return out


def openapi_diff(
    sources: Iterable[SpecInput | Mapping[str, Any]],
    target: SpecInput | Mapping[str, Any],
    ignore: IgnoreRules | Mapping[str, Any] | None = None,
    *,
    strict_numeric_bounds: bool = False,
    auth: AuthMode = "compare",
) -> dict[str, Any]:
    """Compare ``target`` against the union of ``sources``. Returns a JSON-able dict.

    ``minimum: 0`` that the target adds to an integer the source declares without a
    minimum (utoipa does this for every unsigned type) is listed under
    ``cosmetic_differences`` and does not affect ``equal`` - unless
    ``strict_numeric_bounds`` is true, then it is an ordinary difference.
    """
    if auth not in AUTH_MODES:
        raise OpenApiDiffError(f"auth must be one of {AUTH_MODES}, got {auth!r}")
    rules = ignore if isinstance(ignore, IgnoreRules) else IgnoreRules.from_mapping(ignore)
    srcs = _named(sources, "source")
    if not srcs:
        raise OpenApiDiffError("at least one source document is required")
    tgt = target if isinstance(target, SpecInput) else SpecInput(name="target", document=target)
    if tgt.name in {s.name for s in srcs}:
        raise OpenApiDiffError(f"target name {tgt.name!r} collides with a source name")

    union: dict[tuple[str, str], _Op] = {}
    conflicts: dict[tuple[str, str], dict[str, Any]] = {}
    ignored: list[dict[str, str]] = []
    for spec in srcs:
        ops, ign = extract_operations(spec.document, spec.name, rules, spec.path_prefix_map)
        ignored.extend(ign)
        for op in ops:
            existing = union.get(op.key)
            if existing is None:
                op.sources = [spec.name]
                union[op.key] = op
            elif existing.contract == op.contract:
                existing.sources.append(spec.name)
            else:
                c = conflicts.setdefault(
                    op.key,
                    {
                        "method": op.method,
                        "path": existing.path,
                        "kept": existing.source,
                        "conflicts": [],
                    },
                )
                c["conflicts"].append(
                    {
                        "source": spec.name,
                        "path": op.path,
                        "differences": diff_contracts(existing.contract, op.contract),
                    }
                )

    tops, tign = extract_operations(tgt.document, tgt.name, rules)
    ignored.extend(tign)
    target_ops: dict[tuple[str, str], _Op] = {}
    target_dupes: list[dict[str, Any]] = []
    for op in tops:
        if op.key in target_ops:
            target_dupes.append({"method": op.method, "paths": [target_ops[op.key].path, op.path]})
        else:
            target_ops[op.key] = op

    missing, changed, cosmetic, auth_diffs, matched = [], [], [], [], 0
    for key in sorted(union):
        src = union[key]
        t = target_ops.get(key)
        if t is None:
            entry: dict[str, Any] = {"method": src.method, "path": src.path, "sources": src.sources}
            if src.source_path:
                entry["source_path"] = src.source_path
            missing.append(entry)
            continue
        diffs = diff_contracts(src.contract, t.contract)
        if auth != "compare":
            authd = [d for d in diffs if _is_auth(d)]
            diffs = [d for d in diffs if not _is_auth(d)]
            if authd:
                auth_diffs.append(
                    {
                        "method": src.method,
                        "path": src.path,
                        "target_path": t.path,
                        "sources": src.sources,
                        "differences": authd,
                    }
                )
        if not strict_numeric_bounds:
            soft = [d for d in diffs if d["kind"] in COSMETIC_KINDS]
            diffs = [d for d in diffs if d["kind"] not in COSMETIC_KINDS]
            if soft:
                cosmetic.append(
                    {
                        "method": src.method,
                        "path": src.path,
                        "target_path": t.path,
                        "sources": src.sources,
                        "differences": soft,
                    }
                )
        if diffs:
            entry = {
                "method": src.method,
                "path": src.path,
                "target_path": t.path,
                "sources": src.sources,
                "differences": diffs,
            }
            if src.source_path:
                entry["source_path"] = src.source_path
            changed.append(entry)
        else:
            matched += 1
    extra = [
        {"method": target_ops[k].method, "path": target_ops[k].path}
        for k in sorted(target_ops)
        if k not in union
    ]
    source_conflicts = [conflicts[k] for k in sorted(conflicts)]
    total = len(union)
    vacuous = total == 0
    equal = (
        not vacuous
        and not missing
        and not extra
        and not changed
        and not source_conflicts
        and not target_dupes
        and not (auth == "separate" and auth_diffs)
    )
    ignored.sort(key=lambda d: (d["document"], path_shape(d["path"]), d["method"]))
    return {
        "equal": equal,
        "vacuous": vacuous,
        "counts": {
            "sources": len(srcs),
            "source_operations": total,
            "target_operations": len(target_ops),
            "matched": matched,
            "missing": len(missing),
            "extra": len(extra),
            "changed": len(changed),
            "cosmetic": len(cosmetic),
            "source_conflicts": len(source_conflicts),
            "ignored": len(ignored),
            "auth": len(auth_diffs),
        },
        "progress": {
            "matched": matched,
            "total": total,
            "ratio": round(matched / total, 4) if total else 0.0,
        },
        "missing_operations": missing,
        "extra_operations": extra,
        "changed_operations": changed,
        "cosmetic_differences": cosmetic,
        "strict_numeric_bounds": strict_numeric_bounds,
        "auth": auth,
        "auth_differences": auth_diffs,
        "source_conflicts": source_conflicts,
        "target_duplicate_shapes": target_dupes,
        "ignored_operations": ignored,
        "ignore": rules.to_dict(),
    }
