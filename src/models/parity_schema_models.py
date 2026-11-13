"""Request/response models for schema parity (py2rust slice A5a)."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_SOURCE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
SCHEMA_KINDS = (
    "extension",
    "enum",
    "sequence",
    "table",
    "column",
    "primary_key",
    "unique",
    "foreign_key",
    "check",
    "exclusion",
    "index",
    "view",
    "domain",
    "function",
    "trigger",
    "policy",
    "grant",
)


class SchemaDiffOptions(BaseModel):
    """How two schemas are compared."""

    strict_names: bool = Field(
        default=False,
        description=(
            "When true, a constraint/index whose definition matches but whose name differs "
            "is a failing `changed` entry. Default false: reported in `name_only_differences`."
        ),
    )
    ignore_tables: list[str] | None = Field(
        default=None,
        description=(
            "Tables (bare `name` or `schema.name`) excluded together with their columns, "
            "constraints, indexes and owned sequences. Null = the default "
            "['alembic_version', '_sqlx_migrations']; [] ignores nothing."
        ),
    )
    schemas: list[str] | None = Field(
        default=None,
        description="Restrict introspection to these schemas. Null = every non-system schema.",
    )
    ignore_column_order: bool = Field(
        default=False,
        description=(
            "Column order is compared on its own (`column_order_differences`) and fails "
            "`equal` by default; true keeps it informational."
        ),
    )
    ignore_kinds: list[str] = Field(
        default_factory=list,
        description=f"Object kinds left out of the comparison entirely, from {list(SCHEMA_KINDS)}",
    )

    @field_validator("schemas")
    @classmethod
    def _plain_identifiers(cls, value: list[str] | None) -> list[str] | None:
        for name in value or []:
            if not _IDENT_RE.match(name):
                raise ValueError(f"invalid schema name: {name!r}")
        return value

    @field_validator("ignore_kinds")
    @classmethod
    def _known_kinds(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(SCHEMA_KINDS))
        if unknown:
            raise ValueError(f"unknown kinds {unknown}; valid: {list(SCHEMA_KINDS)}")
        return value


class SchemaMapping(BaseModel):
    """Where ONE source's objects live in the merged target."""

    model_config = ConfigDict(extra="forbid")

    schema_map: dict[str, str] = Field(
        default_factory=dict,
        description="Source schema -> target schema, e.g. {'public': 'notes'}",
    )
    table_prefix: str | None = Field(
        default=None,
        max_length=48,
        description="Prepended to every relation name of this source, e.g. 'notes_'",
    )

    @field_validator("schema_map")
    @classmethod
    def _idents(cls, value: dict[str, str]) -> dict[str, str]:
        for k, v in value.items():
            if not _IDENT_RE.match(k) or not _IDENT_RE.match(v):
                raise ValueError(f"schema_map entries must be plain identifiers: {k!r}->{v!r}")
        return value

    @field_validator("table_prefix")
    @classmethod
    def _prefix(cls, value: str | None) -> str | None:
        if value is not None and not re.match(r"^[A-Za-z_][A-Za-z0-9_$]*$", value):
            raise ValueError(f"table_prefix must be an identifier prefix: {value!r}")
        return value


def check_source_names(names: list[str]) -> None:
    for n in names:
        if not _SOURCE_NAME_RE.match(n) or n == "target":
            raise ValueError(f"source name {n!r} is invalid or reserved ('target')")
    if len(set(names)) != len(names):
        raise ValueError(f"source names must be unique: {names}")


class FieldChange(BaseModel):
    """One field that differs between source and target."""

    field: str
    source: Any = None
    target: Any = None


class SchemaObjectDiff(BaseModel):
    """One schema object that is missing, extra or changed."""

    kind: str = Field(description=" | ".join(SCHEMA_KINDS))
    key: str = Field(description="Definition-based identity of the object (names excluded).")
    name: str | None = Field(default=None, description="Constraint/index name, when it has one.")
    source: dict[str, Any] | None = None
    target: dict[str, Any] | None = None
    fields: list[FieldChange] = Field(default_factory=list)
    sources: list[str] = Field(
        default_factory=list, description="The source(s) that define this object"
    )


class SourceConflictEntry(BaseModel):
    source: str
    fields: list[FieldChange]


class SchemaSourceConflict(BaseModel):
    """Two sources define one object (same kind + key) differently."""

    kind: str
    key: str
    kept: str = Field(description="The source whose definition was compared to the target")
    conflicts: list[SourceConflictEntry]


class ColumnOrderDifference(BaseModel):
    """The common columns of a table are in a different order on the target."""

    table: str
    source_order: list[str]
    target_order: list[str]
    sources: list[str] = Field(default_factory=list)


class NameOnlyDifference(BaseModel):
    """Same definition on both sides, different name (informational by default)."""

    kind: str
    key: str
    source_name: str | None
    target_name: str | None


class SchemaSideInfo(BaseModel):
    server_version: str
    object_counts: dict[str, int]


class NamedSchemaSideInfo(SchemaSideInfo):
    name: str
    mapping: SchemaMapping | None = None


class SchemaDiffResult(BaseModel):
    """Computed schema parity verdict. `equal` is true only when `computed` is true.

    A comparison where NEITHER side has a single non-ignored schema object (both
    databases empty, or holding only ignored tables such as ``alembic_version``)
    is ``vacuous`` and never ``equal`` -- it proves nothing about the migrations.
    """

    computed: bool
    equal: bool
    vacuous: bool = Field(
        default=False,
        description=(
            "True when neither side has any non-ignored schema object (extensions "
            "excluded: plpgsql is in every database). A vacuous result is never equal."
        ),
    )
    missing_in_target: list[SchemaObjectDiff] = Field(default_factory=list)
    extra_in_target: list[SchemaObjectDiff] = Field(default_factory=list)
    changed: list[SchemaObjectDiff] = Field(default_factory=list)
    name_only_differences: list[NameOnlyDifference] = Field(default_factory=list)
    source_conflicts: list[SchemaSourceConflict] = Field(
        default_factory=list, description="Sources that define one object differently (fail)"
    )
    column_order_differences: list[ColumnOrderDifference] = Field(
        default_factory=list,
        description="Fail `equal` unless options.ignore_column_order (then informational)",
    )
    vacuous_sources: list[str] = Field(
        default_factory=list,
        description="Sources that contributed no schema object at all (never equal)",
    )
    summary: dict[str, int] = Field(default_factory=dict)
    options: SchemaDiffOptions | None = None
    source: SchemaSideInfo | None = Field(
        default=None, description="The (union of the) source side(s)"
    )
    sources: list[NamedSchemaSideInfo] = Field(
        default_factory=list, description="N sources: each source on its own"
    )
    target: SchemaSideInfo | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = Field(
        default=None, description="Why the verdict was not computed (computed=false)."
    )
    stage: str | None = Field(
        default=None,
        description=(
            "Workspace mode, when not computed: the step that failed "
            "(ws_pg_start | createdb_source | createdb_target | migrate_source | "
            "migrate_target | introspect_source | introspect_target; with `sources` the "
            "source stages are suffixed ':<name>', e.g. migrate_source:notes)."
        ),
    )
    steps: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Workspace mode: each command run, with exit code and output tail.",
    )


class NamedDsn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Names the source in every finding")
    dsn: str
    mapping: SchemaMapping | None = None


class SchemaDiffDsnRequest(BaseModel):
    """Diff two databases reachable from this service (local/test use)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "source_dsn": "postgresql://user:pw@127.0.0.1:5432/src",  # pragma: allowlist secret
                "target_dsn": "postgresql://user:pw@127.0.0.1:5432/tgt",  # pragma: allowlist secret
                "options": {"strict_names": False},
            }
        }
    )

    source_dsn: str | None = Field(
        default=None,
        description="DSN of the DB the source (Python/Alembic) side migrated. "
        "Required unless `sources` is set.",
    )
    sources: list[NamedDsn] | None = Field(
        default=None,
        min_length=1,
        max_length=20,
        description="N sources -> 1 target: their union is compared to the target",
    )
    target_dsn: str = Field(description="DSN of the DB the target (Rust/sqlx) side migrated.")
    options: SchemaDiffOptions = Field(default_factory=SchemaDiffOptions)

    @model_validator(mode="after")
    def _one_source_shape(self) -> SchemaDiffDsnRequest:
        if (self.source_dsn is None) == (self.sources is None):
            raise ValueError("send exactly one of source_dsn or sources")
        if self.sources is not None:
            check_source_names([s.name for s in self.sources])
        return self


class MigrationSide(BaseModel):
    """One side's checkout and migration command, inside the workspace."""

    workdir: str = Field(
        default=".",
        description="Workspace-relative directory the migration command runs in.",
    )
    migrate_command: list[str] | str = Field(
        description=(
            "Command that applies this side's migrations, run with DATABASE_URL set, "
            "e.g. 'uv run alembic upgrade head' or 'sqlx migrate run'. A string is split "
            "shell-style (no shell is invoked)."
        )
    )
    mapping: SchemaMapping | None = Field(
        default=None, description="Where this source's objects live in the target"
    )


class NamedMigrationSide(MigrationSide):
    """One of N sources."""

    name: str = Field(description="Names the source in every finding")


class SchemaDiffWorkspaceRequest(BaseModel):
    """Migrate two fresh pod-local databases, then diff them."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "source": {
                    "workdir": "py2rs-fixture-svc",
                    "migrate_command": "uv run alembic upgrade head",
                },
                "target": {
                    "workdir": "py2rs-fixture-svc-rs",
                    "migrate_command": "sqlx migrate run",
                },
                "options": {"strict_names": False},
                "timeout": 300,
            }
        }
    )

    source: MigrationSide | None = Field(
        default=None, description="Required unless `sources` is set"
    )
    sources: list[NamedMigrationSide] | None = Field(
        default=None,
        min_length=1,
        max_length=20,
        description="N sources -> 1 target: each migrated on its own fresh DB, union compared",
    )
    target: MigrationSide
    options: SchemaDiffOptions = Field(default_factory=SchemaDiffOptions)
    timeout: int = Field(
        default=300, ge=1, le=300, description="Per-command budget in seconds (1-300)."
    )

    @model_validator(mode="after")
    def _one_source_shape(self) -> SchemaDiffWorkspaceRequest:
        if (self.source is None) == (self.sources is None):
            raise ValueError("send exactly one of source or sources")
        if self.sources is not None:
            check_source_names([s.name for s in self.sources])
        return self


class SchemaDiffWorkspaceRunRequest(SchemaDiffWorkspaceRequest):
    """Async job mode: the same request, run as ONE workspace job (no 630 s gateway limit)."""

    timeout: int = Field(
        default=600, ge=1, le=3600, description="Per-command budget in seconds (1-3600)."
    )
    job_timeout: int = Field(
        default=3600, ge=60, le=14400, description="Seconds for the whole job."
    )


class SchemaDiffRunLinks(BaseModel):
    status: str


class SchemaDiffRun(BaseModel):
    run_id: str
    container_name: str
    status: str = Field(description="queued | running | completed | failed")
    result: SchemaDiffResult | None = None
    links: SchemaDiffRunLinks | None = None
