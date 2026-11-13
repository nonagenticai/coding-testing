"""Request/response models for behaviour parity (record / replay / compare)."""

from __future__ import annotations

import posixpath
import re
import shlex
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

WORKSPACE_ROOT = "/workspace"
#: ``<nonce>.<coding-containers job id>`` (async) or ``<nonce>`` (sync).
RUN_ID_RE = re.compile(r"^(?P<nonce>[0-9a-f]{32})(?:\.(?P<job>[A-Za-z0-9_-]{1,64}))?$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_RESERVED_ENV = {"DATABASE_URL", "PORT", "HOST", "BIND_ADDR"}
_UPSTREAM_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,127}$")
_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]{1,256}$")
HTTP_METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE"})
MAX_CORPORA = 50
CLI_MAX_TIMEOUT = 300  # == behaviour_parity_driver.CLI_MAX_TIMEOUT

BehaviourOp = Literal["record", "replay", "compare"]
RunStatus = Literal["queued", "running", "completed", "failed"]


def _relative(value: str, what: str) -> str:
    value = value.strip() or "."
    if value.startswith(("/", "-")) or ".." in value.split("/") or "\x00" in value:
        raise ValueError(f"{what} must be workspace-relative (no leading '/', '-' or '..')")
    return posixpath.normpath(value)


def _argv(v: Any) -> Any:
    return shlex.split(v) if isinstance(v, str) else v


def _check_headers(v: dict[str, str]) -> dict[str, str]:
    for k, val in v.items():
        if not _HEADER_NAME_RE.match(k) or any(c in str(val) for c in "\r\n\x00"):
            raise ValueError(f"header {k!r} has an invalid name or value")
    return v


class RouteRule(BaseModel):
    """Rewrite one source route to where the merged target serves it.

    ``from`` is a path prefix matched on SEGMENT boundaries (``/api`` matches
    ``/api`` and ``/api/x``, never ``/apix``; ``/api/`` also matches ``/api``; ``/``
    matches every path); the matched prefix is replaced by ``to``, and the mapped
    path ends with ``/`` exactly when the source path does (``/api/`` -> ``/v/``
    maps ``/api`` to ``/v``). The longest matching ``from`` wins; on an equally long
    ``from`` a rule with ``method`` beats a generic one; then the first listed.
    ``method`` restricts the rule to one request method; ``to_method`` also rewrites
    the method. The query string is kept as-is.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_prefix: str = Field(..., alias="from", min_length=1, max_length=512)
    to: str = Field(..., min_length=1, max_length=512)
    method: str | None = Field(None, description="Only requests with this method")
    to_method: str | None = Field(None, description="Send the request with this method")

    @field_validator("from_prefix", "to")
    @classmethod
    def _check_prefix(cls, v: str) -> str:
        if not v.startswith("/") or any(c in v for c in "?#\x00 "):
            raise ValueError("route prefixes start with '/' and carry no query/space/NUL")
        return v

    @field_validator("method", "to_method")
    @classmethod
    def _check_method(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if v.upper() not in HTTP_METHODS:
            raise ValueError(f"unknown HTTP method {v!r}")
        return v.upper()


def _route_map(v: Any) -> Any:
    """``{"/": "/a/"}`` is shorthand for ``[{"from": "/", "to": "/a/"}]``."""
    if isinstance(v, dict):
        return [{"from": k, "to": t} for k, t in v.items()]
    return v


def _check_case_kinds(cases: list[dict[str, Any]] | None) -> str | None:
    """'cli' / 'service' for an inline case list (None when not inline); mixing is a 422."""
    if not cases:
        return None
    kinds = {(c.get("kind") or "http") if isinstance(c, dict) else "http" for c in cases}
    if "cli" in kinds and len(kinds) > 1:
        raise ValueError("a corpus cannot mix cli cases with http/websocket/sse cases")
    for c in cases:
        t = c.get("timeout") if isinstance(c, dict) and c.get("kind") == "cli" else None
        if t is not None and (
            isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 < t <= CLI_MAX_TIMEOUT
        ):
            raise ValueError(
                f"cli case {c.get('name')!r}: timeout must be > 0 and <= {CLI_MAX_TIMEOUT} s"
            )
    return "cli" if kinds == {"cli"} else "service"


def _check_side_kind(cases: list[dict[str, Any]] | None, *sides: ServiceUnderTest | None) -> None:
    kind = _check_case_kinds(cases)
    for side in sides:
        if kind is not None and side is not None and side.kind != kind:
            raise ValueError(
                "cli cases need sides with kind='cli'"
                if kind == "cli"
                else "a side with kind='cli' runs only cli cases"
            )


class ServiceUnderTest(BaseModel):
    """How to migrate and start one service inside the workspace.

    ``kind='cli'`` (a CLI, or a library through the call harness): ``start`` is the
    argv PREFIX each cli case's ``args`` are appended to (``['uv','run','mytool']``,
    ``['target/release/mytool']``, or ``['uv','run','python','{lib_harness}']``); no
    database, stub or port, so ``migrate``/``upstreams``/``request_headers`` are
    rejected and ``health_path``/``startup_timeout`` are ignored. Build the binary
    beforehand (exec ``cargo build --release``): the parity run does not build.

    The driver sets ``DATABASE_URL`` (a fresh database), every ``upstream_env``
    name to the upstream stub's URL, and ``PORT`` / ``HOST=127.0.0.1`` /
    ``BIND_ADDR=127.0.0.1:<port>``. ``{port}``, ``{bind_addr}``, ``{database_url}``
    and ``{upstream_url}`` are substituted in ``start`` (and ``{database_url}`` in
    ``migrate``).
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["service", "cli"] = Field(
        "service", description="'service' (HTTP, default) or 'cli' (argv cases, see above)"
    )
    workdir: str = Field(".", max_length=1024, description="Directory under /workspace")
    start: list[str] = Field(
        ...,
        min_length=1,
        max_length=64,
        description="e.g. ['uv','run','uvicorn','app.main:app','--port','{port}'] or "
        "['target/debug/svc'] (reads BIND_ADDR)",
    )
    migrate: list[str] | None = Field(
        None,
        max_length=64,
        description="e.g. ['uv','run','alembic','upgrade','head'] or ['sqlx','migrate','run']",
    )
    env: dict[str, str] = Field(default_factory=dict, description="Extra environment")
    upstream_env: list[str] = Field(
        default_factory=lambda: ["UPSTREAM_URL"],
        max_length=32,
        description="Env var names that receive the DEFAULT upstream stub's base URL",
    )
    upstreams: dict[str, list[str]] = Field(
        default_factory=dict,
        max_length=32,
        description="Named upstreams, each with its OWN stub: {name: [env var names]}, e.g. "
        "{'kag': ['KAG_URL'], 'billing': ['BILLING_BASE_URL']}. A case scripts them with "
        "`upstream_responses: {name: [...]}` and every recorded outbound call carries its "
        "`upstream` name, so calls are compared per upstream. `{upstream_url:<name>}` is "
        "substituted in `start`.",
    )
    request_headers: dict[str, str] = Field(
        default_factory=dict,
        description="Headers sent with EVERY request to this service, overriding the "
        "case's own (e.g. the target's auth: {'Authorization': 'Bearer ...'})",
    )
    health_path: str = Field("/health", max_length=512)
    startup_timeout: int = Field(120, ge=1, le=1800, description="Seconds to become healthy")
    migrate_timeout: int = Field(600, ge=1, le=3600)

    @field_validator("start", "migrate", mode="before")
    @classmethod
    def _split_argv(cls, v: Any) -> Any:
        return _argv(v)

    @field_validator("start", "migrate")
    @classmethod
    def _check_argv(cls, v: list[str] | None) -> list[str] | None:
        if v is not None and any("\x00" in a or len(a) > 4096 for a in v):
            raise ValueError("argv entries must be <=4096 chars without NUL")
        return v

    @field_validator("workdir")
    @classmethod
    def _check_workdir(cls, v: str) -> str:
        return _relative(v, "workdir")

    @field_validator("health_path")
    @classmethod
    def _check_health(cls, v: str) -> str:
        if not v.startswith("/"):
            raise ValueError("health_path must start with '/'")
        return v

    @field_validator("env")
    @classmethod
    def _check_env(cls, v: dict[str, str]) -> dict[str, str]:
        for k in v:
            if not _ENV_NAME_RE.match(k) or k in _RESERVED_ENV:
                raise ValueError(f"env name {k!r} is invalid or set by the driver")
        return v

    @field_validator("upstream_env")
    @classmethod
    def _check_upstream_env(cls, v: list[str]) -> list[str]:
        for k in v:
            if not _ENV_NAME_RE.match(k) or k in _RESERVED_ENV:
                raise ValueError(f"upstream_env name {k!r} is invalid or reserved")
        return v

    @field_validator("upstreams")
    @classmethod
    def _check_upstreams(cls, v: dict[str, list[str]]) -> dict[str, list[str]]:
        seen: set[str] = set()
        for name, envs in v.items():
            if not _UPSTREAM_NAME_RE.match(name) or name == "default":
                raise ValueError(f"upstream name {name!r} is invalid or reserved ('default')")
            if not envs:
                raise ValueError(f"upstream {name!r} needs at least one env var name")
            for k in envs:
                if not _ENV_NAME_RE.match(k) or k in _RESERVED_ENV or k in seen:
                    raise ValueError(f"upstream env name {k!r} is invalid, reserved or repeated")
                seen.add(k)
        return v

    @field_validator("request_headers")
    @classmethod
    def _check_request_headers(cls, v: dict[str, str]) -> dict[str, str]:
        return _check_headers(v)

    @model_validator(mode="after")
    def _distinct_upstream_env(self) -> ServiceUnderTest:
        if self.kind == "cli":
            unused = [
                n
                for n, v in (
                    ("migrate", self.migrate),
                    ("upstreams", self.upstreams),
                    ("request_headers", self.request_headers),
                )
                if v
            ]
            if unused:
                raise ValueError(f"a cli side has no database/stub/HTTP: drop {unused}")
        named = {k for envs in self.upstreams.values() for k in envs}
        if named & set(self.upstream_env):
            raise ValueError(
                f"env names {sorted(named & set(self.upstream_env))} are in both "
                "upstream_env and upstreams"
            )
        return self

    def absolute_workdir(self) -> str:
        return WORKSPACE_ROOT if self.workdir == "." else f"{WORKSPACE_ROOT}/{self.workdir}"


class BehaviourCorpus(BaseModel):
    """The recorded behaviour of ONE source, replayed against the single target.

    n8n records every source in its own run (``POST .../record``); a replay or
    compare then sends one corpus per source. Each corpus is replayed against a
    FRESH target (own database, own process, own stubs), in the order given, with
    its paths rewritten by ``route_map``.
    """

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(
        ..., description="Names the source in every outcome, e.g. 'notes-svc' or a repo slug"
    )
    cases_file: str | None = Field(None, max_length=1024, description="Workspace-relative JSONL")
    cases: list[dict[str, Any]] | None = Field(None, max_length=5000, description="Inline cases")
    route_map: list[RouteRule] = Field(
        default_factory=list,
        max_length=200,
        description="Where this source's routes live on the target, e.g. [{'from': '/', "
        "'to': '/notes/'}] (or the shorthand {'/': '/notes/'}). Applied before replay.",
    )
    default_headers: dict[str, str] = Field(
        default_factory=dict,
        description="Merged under each of this corpus's case headers",
    )
    ignore_upstreams: list[str] = Field(
        default_factory=list,
        max_length=32,
        description="Upstream names whose calls are not compared (e.g. a call from source A "
        "to source B that the merged target makes in-process)",
    )
    baseline: Literal["expected", "recording"] = Field(
        "expected", description="replay only: 'expected' fields, or `recording`/`recording_run_id`"
    )
    recording: dict[str, Any] | None = Field(
        None, description="replay only: the `recording` a record run of this source returned"
    )
    recording_run_id: str | None = Field(
        None, max_length=97, description="replay only: the record run of this source"
    )
    source: ServiceUnderTest | None = Field(
        None, description="compare only: how to run THIS source (recorded in the same job)"
    )

    @field_validator("source_id")
    @classmethod
    def _check_source_id(cls, v: str) -> str:
        if not _SOURCE_ID_RE.match(v):
            raise ValueError(f"source_id {v!r} must match {_SOURCE_ID_RE.pattern}")
        return v

    @field_validator("route_map", mode="before")
    @classmethod
    def _shorthand(cls, v: Any) -> Any:
        return _route_map(v)

    @field_validator("cases_file")
    @classmethod
    def _check_cases_file(cls, v: str | None) -> str | None:
        return None if v is None else _relative(v, "cases_file")

    @field_validator("default_headers")
    @classmethod
    def _check_default_headers(cls, v: dict[str, str]) -> dict[str, str]:
        return _check_headers(v)

    @model_validator(mode="after")
    def _check_corpus(self) -> BehaviourCorpus:
        if (self.cases_file is None) == (self.cases is None):
            raise ValueError(f"corpus {self.source_id!r}: send exactly one of cases_file or cases")
        if self.cases is not None and not self.cases:
            raise ValueError(f"corpus {self.source_id!r}: cases must not be empty")
        _check_side_kind(self.cases, self.source)
        if self.baseline == "recording":
            if (self.recording is None) == (self.recording_run_id is None):
                raise ValueError(
                    f"corpus {self.source_id!r}: baseline=recording needs exactly one of "
                    "recording or recording_run_id"
                )
            if self.recording_run_id and not RUN_ID_RE.match(self.recording_run_id):
                raise ValueError(f"corpus {self.source_id!r}: recording_run_id is not a run id")
        elif self.recording is not None or self.recording_run_id is not None:
            raise ValueError(
                f"corpus {self.source_id!r}: recording/recording_run_id need baseline='recording'"
            )
        return self


def _check_corpora(corpora: list[BehaviourCorpus]) -> None:
    ids = [c.source_id for c in corpora]
    if len(set(ids)) != len(ids):
        raise ValueError(f"corpora source_id values must be unique: {ids}")
    if sum(len(c.cases or []) for c in corpora) > 20_000:
        raise ValueError("at most 20000 inline cases over all corpora")


class DropFrameRule(BaseModel):
    """Remove a websocket frame / SSE event before comparing when ``pointer`` == ``equals``."""

    model_config = ConfigDict(extra="forbid")

    pointer: str = Field(..., max_length=512, description="Exact JSON Pointer, e.g. '/data/type'")
    equals: Any = Field(..., description="e.g. 'heartbeat'")

    @field_validator("pointer")
    @classmethod
    def _check_pointer(cls, v: str) -> str:
        if v and not v.startswith("/"):
            raise ValueError("pointer must be '' or start with '/'")
        return v


class BehaviourOptions(BaseModel):
    """Cases + comparison settings shared by record, replay and compare."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    cases_file: str | None = Field(
        None,
        max_length=1024,
        description="Workspace-relative JSON Lines file (format of parity/requests.jsonl)",
    )
    cases: list[dict[str, Any]] | None = Field(
        None, max_length=5000, description="Inline cases (same objects as the JSONL lines)"
    )
    default_headers: dict[str, str] = Field(
        default_factory=dict,
        description="Merged into every case's headers (the case wins), e.g. trusted "
        "X-Tenant-* headers",
    )
    ignore_paths: list[str] = Field(
        default_factory=list,
        max_length=200,
        description="JSON-Pointer patterns ignored in response bodies ('*' = one segment, "
        "'**' = any depth), e.g. ['/id', '/items/*/created_at']",
    )
    upstream_ignore_paths: list[str] = Field(
        default_factory=list, max_length=200, description="Same, for upstream request bodies"
    )
    upstream_headers: list[str] | None = Field(
        None,
        max_length=64,
        description="Upstream request headers compared (lower-cased). Default: the names "
        "each expected call lists; against a recording ['content-type','x-tenant-id']",
    )
    response_headers: list[str] = Field(
        default_factory=lambda: ["content-type"],
        max_length=64,
        description="Response headers captured and compared against a recording "
        "(content-type by media type only)",
    )
    frame_ignore_paths: list[str] = Field(
        default_factory=list,
        max_length=200,
        description="websocket/sse cases: JSON-Pointer patterns ignored in every frame/event "
        "record ({type, data|text|b64} / {event, id, data|text}), e.g. ['/data/ts', '/id']",
    )
    drop_frames: list[DropFrameRule] = Field(
        default_factory=list,
        max_length=50,
        description="websocket/sse cases: frames/events removed before comparing when the "
        "value at `pointer` equals `equals` (e.g. heartbeats)",
    )
    request_timeout: int = Field(30, ge=1, le=600, description="Seconds per replayed request")
    timeout: int = Field(1800, ge=1, le=14400, description="Seconds for the whole driver job")
    run_async: bool = Field(
        True,
        alias="async",
        description="Default true: 202 + run_id, poll GET .../runs/{run_id}. false = wait "
        "(timeout <= 300)",
    )

    @field_validator("cases_file")
    @classmethod
    def _check_cases_file(cls, v: str | None) -> str | None:
        return None if v is None else _relative(v, "cases_file")

    @model_validator(mode="after")
    def _one_case_source(self) -> BehaviourOptions:
        if getattr(self, "corpora", None) is not None:
            if self.cases_file is not None or self.cases is not None:
                raise ValueError("with corpora, cases/cases_file go inside each corpus")
            return self
        if (self.cases_file is None) == (self.cases is None):
            raise ValueError("send exactly one of cases_file or cases")
        if self.cases is not None and not self.cases:
            raise ValueError("cases must not be empty")
        _check_case_kinds(self.cases)
        return self


class RecordRequest(BehaviourOptions):
    """Record the SOURCE service's behaviour over the cases (always ONE source per run)."""

    source: ServiceUnderTest
    source_id: str | None = Field(
        None, description="Stamped on the returned recording (N sources: one record run each)"
    )

    @field_validator("source_id")
    @classmethod
    def _check_source_id(cls, v: str | None) -> str | None:
        if v is not None and not _SOURCE_ID_RE.match(v):
            raise ValueError(f"source_id {v!r} must match {_SOURCE_ID_RE.pattern}")
        return v

    @model_validator(mode="after")
    def _source_kind(self) -> RecordRequest:
        _check_side_kind(self.cases, self.source)
        return self


class ReplayRequest(BehaviourOptions):
    """Replay the cases against the TARGET and diff against a baseline.

    N sources -> 1 target: send ``corpora`` (one per source) instead of
    ``cases``/``cases_file``; each corpus names its own baseline.
    """

    target: ServiceUnderTest
    corpora: list[BehaviourCorpus] | None = Field(
        None,
        min_length=1,
        max_length=MAX_CORPORA,
        description="One recorded corpus per source, each replayed against a fresh target",
    )
    baseline: Literal["expected", "recording"] = Field(
        "expected",
        description="'expected' = the cases' expected_* fields; 'recording' = `recording` "
        "or `recording_run_id`",
    )
    recording: dict[str, Any] | None = Field(
        None, description="A `recording` object returned by a record run"
    )
    recording_run_id: str | None = Field(
        None, max_length=97, description="A record/compare run in the SAME workspace"
    )

    @model_validator(mode="after")
    def _baseline_source(self) -> ReplayRequest:
        if self.corpora is not None:
            if self.fields_set_baseline():
                raise ValueError("with corpora, baseline/recording go inside each corpus")
            if any(c.source is not None for c in self.corpora):
                raise ValueError("replay runs no source: corpus.source is for compare")
            _check_corpora(self.corpora)
            for c in self.corpora:
                _check_side_kind(c.cases, self.target)
            return self
        _check_side_kind(self.cases, self.target)
        if self.baseline == "recording":
            if (self.recording is None) == (self.recording_run_id is None):
                raise ValueError(
                    "baseline=recording needs exactly one of recording or recording_run_id"
                )
            if self.recording_run_id and not RUN_ID_RE.match(self.recording_run_id):
                raise ValueError("recording_run_id is not a run id")
        elif self.recording is not None or self.recording_run_id is not None:
            raise ValueError("recording/recording_run_id need baseline='recording'")
        return self

    def fields_set_baseline(self) -> bool:
        return (
            self.baseline != "expected"
            or self.recording is not None
            or self.recording_run_id is not None
        )


class CompareRequest(BehaviourOptions):
    """Record the SOURCE, then replay against the TARGET and diff target vs source.

    N sources -> 1 target: send ``corpora`` (each with its own ``source``) instead
    of ``source`` + ``cases``; every source is recorded on its own fresh database,
    then the target replays each corpus on a fresh one.
    """

    source: ServiceUnderTest | None = Field(None, description="Required unless `corpora` is set")
    target: ServiceUnderTest
    corpora: list[BehaviourCorpus] | None = Field(
        None,
        min_length=1,
        max_length=MAX_CORPORA,
        description="One corpus per source, each carrying the `source` to record",
    )

    @model_validator(mode="after")
    def _sources(self) -> CompareRequest:
        if self.corpora is None:
            if self.source is None:
                raise ValueError("send `source` (or `corpora`, each with its own source)")
            _check_side_kind(self.cases, self.source, self.target)
            return self
        if self.source is not None:
            raise ValueError("with corpora, each corpus carries its own source")
        for c in self.corpora:
            if c.source is None:
                raise ValueError(f"corpus {c.source_id!r} needs a source to compare")
            if c.baseline != "expected" or c.recording is not None or c.recording_run_id:
                raise ValueError(f"corpus {c.source_id!r}: compare's baseline is its source")
            _check_side_kind(c.cases, self.target)
        _check_corpora(self.corpora)
        return self


class CaseDifference(BaseModel):
    kind: str = Field(
        ...,
        description="status | route_missing | body | header | upstream_count | "
        "upstream_method | upstream_path | upstream_header | upstream_body | "
        "upstream_unscripted | upstream_unconsumed | error | baseline_error | case_mismatch | "
        "kind | frame | frame_count | frame_groups | frame_missing | frame_unexpected | "
        "ws_close | ws_subprotocol | sse_end (the last nine: websocket/sse cases) | "
        "exit_code | stdout | stderr (cli cases). "
        "route_missing (a heuristic, replaces `status`): the target answered 404/405 with "
        "an empty or framework-default body where the baseline did not -- most likely the "
        "route is not implemented or the route_map is wrong; an app route that deliberately "
        "answers that looks the same, so the case's other differences are still listed",
    )
    pointer: str = Field(..., description="JSON Pointer into the case result")
    expected: Any = None
    actual: Any = None
    detail: str | None = None


class CaseOutcome(BaseModel):
    name: str
    passed: bool
    kind: Literal["websocket", "sse", "cli"] | None = Field(
        None, description="Set for websocket/sse/cli cases (absent = an HTTP case)"
    )
    source_id: str | None = Field(None, description="The corpus (source) the case came from")
    method: str | None = Field(None, description="The case's method as recorded for the source")
    path: str | None = Field(None, description="The case's path as recorded for the source")
    target_method: str | None = Field(None, description="Method sent to the target (mapped)")
    target_path: str | None = Field(None, description="Path sent to the target (mapped)")
    diffs: list[CaseDifference] = Field(default_factory=list)
    diffs_clipped: int = 0


class CorpusOutcome(BaseModel):
    """Per-source verdict of a multi-corpus replay/compare."""

    source_id: str
    baseline: str
    computed: bool
    equal: bool | None = None
    cases_total: int = 0
    cases_passed: int = 0
    cases_failed: int = 0
    error: str | None = None


class BehaviourParityResult(BaseModel):
    op: BehaviourOp
    baseline: Literal["expected", "recording", "source", "none", "mixed"] = Field(
        ...,
        description="What the diff compared against ('none': record without expectations; "
        "'mixed': corpora with different baselines, see `corpora[].baseline`)",
    )
    computed: bool = Field(..., description="Every side ran every case; false = no verdict")
    equal: bool | None = Field(
        None, description="True only when computed, >0 cases and every case passed"
    )
    cases_total: int = 0
    cases_passed: int = 0
    cases_failed: int = 0
    cases: list[CaseOutcome] = Field(default_factory=list)
    recording: dict[str, Any] | None = Field(
        None, description="record/compare: the source recording (feed it to replay)"
    )
    corpora: list[CorpusOutcome] = Field(
        default_factory=list, description="Multi-corpus runs: the verdict per source"
    )
    recordings: dict[str, dict[str, Any]] | None = Field(
        None, description="Multi-corpus compare: each source's recording by source_id"
    )
    error: str | None = None
    exit_code: int | None = None
    driver_log_tail: str = ""


class BehaviourParityLinks(BaseModel):
    status: str


class BehaviourParityRun(BaseModel):
    run_id: str
    container_name: str
    op: BehaviourOp
    status: RunStatus
    result: BehaviourParityResult | None = None
    links: BehaviourParityLinks | None = None
