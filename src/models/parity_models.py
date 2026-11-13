"""Request/response models for the contract-parity routes (``/api/parity``)."""

from __future__ import annotations

import posixpath
import re
import shlex
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: The checkout root inside every coding workspace.
WORKSPACE_ROOT = "/workspace"


class OpenApiDiffIgnore(BaseModel):
    """Operations / status codes excluded from the comparison (both sides)."""

    model_config = ConfigDict(extra="forbid")

    paths: list[str] = Field(
        default_factory=list,
        description="Glob patterns over path templates, e.g. '/internal/*' (param names ignored)",
    )
    operations: list[str] = Field(
        default_factory=list,
        description="'METHOD /path' entries (method '*' for any; path is a glob)",
    )
    response_codes: list[str] = Field(
        default_factory=list,
        description="Status codes ignored in every operation's responses, e.g. ['422']",
    )


_PREFIX_RE = re.compile(r"^/[^?#\s\x00]*$")
AuthMode = Literal["compare", "separate", "ignore"]
_AUTH_DESCRIPTION = (
    "How security differences count: 'compare' (default; `security_changed` like any "
    "difference), 'separate' (listed under `auth_differences`, still failing `equal`), "
    "'ignore' (listed there, informational). The `Authorization` header parameter is "
    "classified as auth too."
)
_PREFIX_MAP_DESCRIPTION = (
    "Sources only: where this source's paths live in the merged target, e.g. "
    "{'/': '/notes/'} or {'/api': '/notes/api'}. Applied BEFORE comparison (segment-"
    "aligned, longest prefix wins), so a moved prefix compares equal, not missing+extra."
)


def _check_prefix_map(v: dict[str, str]) -> dict[str, str]:
    for k, t in v.items():
        if not _PREFIX_RE.match(k) or not _PREFIX_RE.match(t):
            raise ValueError(f"path_prefix_map entries must be paths starting with '/': {k!r}")
    return v


class InlineSpec(BaseModel):
    """One OpenAPI 3.0/3.1 document sent inline."""

    name: str | None = Field(None, max_length=200, description="Label used in the result")
    document: dict[str, Any] = Field(..., description="The OpenAPI document (JSON object)")
    path_prefix_map: dict[str, str] = Field(
        default_factory=dict, max_length=50, description=_PREFIX_MAP_DESCRIPTION
    )

    @field_validator("path_prefix_map")
    @classmethod
    def _check_map(cls, v: dict[str, str]) -> dict[str, str]:
        return _check_prefix_map(v)


class OpenApiDiffRequest(BaseModel):
    """Compare one target document against the union of N source documents."""

    sources: list[InlineSpec] = Field(..., min_length=1, max_length=50)
    target: InlineSpec
    ignore: OpenApiDiffIgnore | None = None
    strict_numeric_bounds: bool = Field(
        False,
        description="Count `minimum: 0` added by the target on unsigned integers as a difference",
    )
    parity_toml: str | None = Field(
        None,
        max_length=200_000,
        description=(
            "Contents of a parity.toml; its [openapi_diff.ignore] table is merged with `ignore`"
        ),
    )
    auth: AuthMode = Field("compare", description=_AUTH_DESCRIPTION)

    @model_validator(mode="after")
    def _target_unmapped(self) -> OpenApiDiffRequest:
        if self.target.path_prefix_map:
            raise ValueError("path_prefix_map applies to sources only (the target is the map)")
        return self


def _relative(value: str, what: str) -> str:
    value = value.strip() or "."
    if value.startswith(("/", "-")) or ".." in value.split("/"):
        raise ValueError(f"{what} must be workspace-relative (no leading '/', '-' or '..')")
    return posixpath.normpath(value)


class WorkspaceSpecSource(BaseModel):
    """How to obtain one OpenAPI document inside the workspace.

    ``command`` (argv list, or a string split like a shell would without running
    one) is run in ``workdir``; when ``file`` is unset its stdout must be the JSON
    document. ``file`` is read after the command (if any), relative to ``workdir``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(None, max_length=200, description="Label used in the result")
    workdir: str = Field(".", max_length=1024, description="Directory under /workspace")
    command: list[str] | None = Field(
        None, description="e.g. ['cargo', 'run', '-q', '--', '--print-openapi']"
    )
    file: str | None = Field(None, max_length=1024, description="e.g. 'openapi.json'")
    timeout: int = Field(
        300,
        ge=1,
        le=3600,
        description="Seconds for the command; above 300 it runs as an async exec job",
    )
    path_prefix_map: dict[str, str] = Field(
        default_factory=dict, max_length=50, description=_PREFIX_MAP_DESCRIPTION
    )

    @field_validator("path_prefix_map")
    @classmethod
    def _check_map(cls, v: dict[str, str]) -> dict[str, str]:
        return _check_prefix_map(v)

    @field_validator("command", mode="before")
    @classmethod
    def _split_command(cls, v: Any) -> Any:
        if isinstance(v, str):
            return shlex.split(v)
        return v

    @field_validator("workdir")
    @classmethod
    def _check_workdir(cls, v: str) -> str:
        return _relative(v, "workdir")

    @field_validator("file")
    @classmethod
    def _check_file(cls, v: str | None) -> str | None:
        return None if v is None else _relative(v, "file")

    @model_validator(mode="after")
    def _one_way_in(self) -> WorkspaceSpecSource:
        if not self.command and not self.file:
            raise ValueError("send `command`, `file`, or both")
        return self

    def absolute_workdir(self) -> str:
        return WORKSPACE_ROOT if self.workdir == "." else f"{WORKSPACE_ROOT}/{self.workdir}"


class WorkspaceOpenApiDiffRequest(BaseModel):
    """Obtain the source and target documents inside a workspace, then diff them."""

    sources: list[WorkspaceSpecSource] = Field(..., min_length=1, max_length=20)
    target: WorkspaceSpecSource
    ignore: OpenApiDiffIgnore | None = None
    strict_numeric_bounds: bool = Field(
        False,
        description="Count `minimum: 0` added by the target on unsigned integers as a difference",
    )
    parity_toml: str | None = Field(
        None,
        max_length=1024,
        description="Workspace-relative path of a parity.toml whose [openapi_diff.ignore] applies",
    )
    auth: AuthMode = Field("compare", description=_AUTH_DESCRIPTION)
    concurrency: int = Field(
        4, ge=1, le=8, description="Documents obtained concurrently (bounded), default 4"
    )

    @field_validator("parity_toml")
    @classmethod
    def _check_toml(cls, v: str | None) -> str | None:
        return None if v is None else _relative(v, "parity_toml")

    @model_validator(mode="after")
    def _target_unmapped(self) -> WorkspaceOpenApiDiffRequest:
        if self.target.path_prefix_map:
            raise ValueError("path_prefix_map applies to sources only (the target is the map)")
        return self


class WorkspaceOpenApiDiffRunRequest(WorkspaceOpenApiDiffRequest):
    """Async job mode: every document obtained inside ONE workspace job."""

    job_timeout: int = Field(
        3600, ge=60, le=14400, description="Seconds for the whole job (all documents)"
    )


class Difference(BaseModel):
    location: str = Field(..., description="Where in the operation, e.g. responses.201")
    kind: str = Field(..., description="e.g. response_removed, type_changed, property_added")
    source: Any = None
    target: Any = None


class OperationRef(BaseModel):
    method: str
    path: str
    sources: list[str] = Field(default_factory=list)
    source_path: str | None = Field(
        None,
        description="The path as the source declares it, when path_prefix_map moved it "
        "(absent otherwise)",
        exclude_if=lambda v: v is None,
    )


class ChangedOperation(OperationRef):
    target_path: str
    differences: list[Difference]


class SourceConflictEntry(BaseModel):
    source: str
    path: str
    differences: list[Difference]


class SourceConflict(BaseModel):
    method: str
    path: str
    kept: str = Field(..., description="The source whose definition was compared to the target")
    conflicts: list[SourceConflictEntry]


class IgnoredOperation(BaseModel):
    method: str
    path: str
    document: str


class TargetDuplicateShape(BaseModel):
    method: str
    paths: list[str]


class DiffCounts(BaseModel):
    sources: int
    source_operations: int
    target_operations: int
    matched: int
    missing: int
    extra: int
    changed: int
    cosmetic: int = Field(..., description="Operations with only informational differences")
    source_conflicts: int
    ignored: int
    auth: int = Field(0, description="Operations with auth differences (auth != 'compare')")


class DiffProgress(BaseModel):
    matched: int = Field(..., description="Source operations present in the target, identical")
    total: int = Field(..., description="Operations in the union of the sources")
    ratio: float


class OpenApiDiffResult(BaseModel):
    """Computed contract-parity verdict.

    ``equal`` is true only when every source operation is in the target with an
    identical normalised contract, the target has no extra operation, and the
    sources do not conflict. A comparison with zero source operations is
    ``vacuous`` and never ``equal``.
    """

    equal: bool
    vacuous: bool
    counts: DiffCounts
    progress: DiffProgress
    missing_operations: list[OperationRef]
    extra_operations: list[OperationRef]
    changed_operations: list[ChangedOperation]
    cosmetic_differences: list[ChangedOperation] = Field(
        ...,
        description=(
            "Informational only (do not affect `equal`): e.g. `minimum: 0` the target adds "
            "to an unsigned integer. Counted as changes when `strict_numeric_bounds` is true."
        ),
    )
    strict_numeric_bounds: bool
    auth: AuthMode = "compare"
    auth_differences: list[ChangedOperation] = Field(
        default_factory=list,
        description="auth separate|ignore: security differences, classified on their own",
    )
    source_conflicts: list[SourceConflict]
    target_duplicate_shapes: list[TargetDuplicateShape]
    ignored_operations: list[IgnoredOperation]
    ignore: OpenApiDiffIgnore


class ObtainedSpec(BaseModel):
    """How one document was obtained in the workspace."""

    name: str
    role: str = Field(..., description="source | target")
    workdir: str
    command: list[str] | None = None
    file: str | None = None
    exit_code: int | None = None
    duration: float | None = None
    bytes: int = 0


class WorkspaceOpenApiDiffResult(OpenApiDiffResult):
    container_name: str
    obtained: list[ObtainedSpec]


class OpenApiDiffRunLinks(BaseModel):
    status: str


class OpenApiDiffRun(BaseModel):
    """An async workspace openapi-diff run."""

    run_id: str
    container_name: str
    status: str = Field(description="queued | running | completed | failed")
    result: WorkspaceOpenApiDiffResult | None = None
    error: Any = Field(
        None, description="failed: why no diff was computed (same detail the sync route's 422 has)"
    )
    links: OpenApiDiffRunLinks | None = None
