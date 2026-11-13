"""Schema parity: introspect two Postgres databases and diff their schemas.

Slice A5a of the py2rust plan. A Python service (Alembic migrations) is
rewritten as a Rust service (sqlx ``migrations/*.sql``); both migration sets,
applied to a fresh database each, must produce the same schema — that is what
makes the one-way data copy at cutover safe. The verdict is COMPUTED here from
the catalogs, never judged.

Design:

* ONE catalog query (:func:`build_snapshot_sql`) returns the whole schema as a
  single JSON document. The same SQL runs over psycopg (two DSNs supplied) and
  over ``psql`` inside a workspace pod (the pod-local Postgres is unreachable
  from this service), so both paths introspect identically.
* Normalisation and diffing are pure functions over that JSON
  (:func:`diff_snapshots`), so they are unit-testable without a database.
* Constraints and indexes are matched by DEFINITION, not by name. A name
  difference is reported in ``name_only_differences`` (informational) unless
  ``strict_names`` is set, in which case it is a ``changed`` field and fails.
* Migration bookkeeping tables (``alembic_version``, ``_sqlx_migrations``) are
  ignored by default, together with every object that belongs to them.
* The snapshot is read inside ``BEGIN READ ONLY`` — never a session-level SET.

* N sources -> 1 target: each source snapshot is first mapped (optional
  ``schema_map`` + ``table_prefix``) and the UNION is compared to the target. Two
  sources defining one object differently is a ``source_conflict`` (fails), the
  same object defined identically by both is merged and attributed to both.
* Also compared: functions/procedures, triggers, RLS policies (and RLS
  enabled/forced on the table), domains, relation + function grants (the
  owner's own implicit privileges excluded). Column ORDER is classified on its
  own (``column_order_differences``) and can be ignored with
  ``ignore_column_order``; columns are still matched by name.

Known gaps (not introspected): composite types, comments, ownership, column-level
grants, per-column index collations. View definitions and function bodies are
compared as text: a ``table_prefix`` is NOT applied inside them.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from models.parity_schema_models import (
    ColumnOrderDifference,
    FieldChange,
    NamedSchemaSideInfo,
    NameOnlyDifference,
    SchemaDiffOptions,
    SchemaDiffResult,
    SchemaMapping,
    SchemaObjectDiff,
    SchemaSideInfo,
    SchemaSourceConflict,
    SourceConflictEntry,
)

DEFAULT_IGNORE_TABLES: tuple[str, ...] = ("alembic_version", "_sqlx_migrations")

# Deterministic output order: parents before children.
KIND_ORDER: tuple[str, ...] = (
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
_KIND_RANK = {k: i for i, k in enumerate(KIND_ORDER)}
# Kinds whose objects belong to a table (suppressed when the table itself is
# missing/extra, so one missing table is one finding, not twenty).
_TABLE_CHILD_KINDS = frozenset(
    {
        "column",
        "primary_key",
        "unique",
        "foreign_key",
        "check",
        "exclusion",
        "index",
        "trigger",
        "policy",
        "grant",
    }
)
# Kinds matched by definition, whose name is reported separately.
_NAMED_KINDS = frozenset(
    {"primary_key", "unique", "foreign_key", "check", "exclusion", "index", "trigger"}
)
_PROKIND = {"f": "function", "p": "procedure", "a": "aggregate", "w": "window"}
_VOLATILITY = {"i": "IMMUTABLE", "s": "STABLE", "v": "VOLATILE"}
_PARALLEL = {"s": "SAFE", "r": "RESTRICTED", "u": "UNSAFE"}
_POLCMD = {"r": "SELECT", "a": "INSERT", "w": "UPDATE", "d": "DELETE", "*": "ALL"}
_TGENABLED = {"O": "ORIGIN", "D": "DISABLED", "R": "REPLICA", "A": "ALWAYS"}
_TRIGGER_HEAD_RE = re.compile(r"^CREATE (?:CONSTRAINT )?TRIGGER \S+ ", re.IGNORECASE)
_TRIGGER_ON_RE = re.compile(r" ON \S+ ")

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")

_FK_ACTION = {
    "a": "NO ACTION",
    "r": "RESTRICT",
    "c": "CASCADE",
    "n": "SET NULL",
    "d": "SET DEFAULT",
}
_FK_MATCH = {"s": "SIMPLE", "f": "FULL", "p": "PARTIAL"}
_IDENTITY = {"a": "ALWAYS", "d": "BY DEFAULT"}
_GENERATED = {"s": "STORED", "v": "VIRTUAL"}
_RELKIND = {"r": "table", "p": "partitioned table", "f": "foreign table", "v": "view"}
_RELKIND.update({"m": "materialized view"})
_PERSISTENCE = {"p": "permanent", "u": "unlogged", "t": "temporary"}

# Expressions that the deparser renders differently but that mean the same
# thing. Kept deliberately tiny: every entry is a place a real difference could
# hide, so only add an alias that is semantically identical.
_EXPR_ALIASES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bCURRENT_TIMESTAMP\b(?!\s*\()", re.IGNORECASE), "now()"),
    (re.compile(r"\btransaction_timestamp\(\)", re.IGNORECASE), "now()"),
)


class SnapshotError(ValueError):
    """A snapshot document could not be read; the verdict is not computed."""


# ---------------------------------------------------------------------------
# Catalog query
# ---------------------------------------------------------------------------

_SNAPSHOT_SQL = """
WITH ns AS (
  SELECT n.oid, n.nspname
  FROM pg_namespace n
  WHERE n.nspname NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
    -- a regex, not an escaped LIKE: this SQL must contain NO backslash, because
    -- the workspace path runs it through `ws-pg psql`, which refuses one.
    AND n.nspname !~ '^pg_(toast_)?temp_'
    {schema_filter}
),
ext_members AS (
  SELECT d.classid, d.objid FROM pg_depend d WHERE d.deptype = 'e'
),
rels AS (
  SELECT c.oid, ns.nspname AS schema_name, c.relname, c.relkind::text AS relkind,
         c.relpersistence::text AS relpersistence, c.relrowsecurity, c.relforcerowsecurity,
         c.relacl, c.relowner
  FROM pg_class c
  JOIN ns ON ns.oid = c.relnamespace
  WHERE c.relkind IN ('r', 'p', 'f', 'v', 'm', 'S')
    AND NOT EXISTS (
      SELECT 1 FROM ext_members e
      WHERE e.classid = 'pg_class'::regclass AND e.objid = c.oid
    )
)
SELECT json_build_object(
  'server_version', current_setting('server_version'),
  'server_version_num', current_setting('server_version_num')::int,
  'tables', (
    SELECT coalesce(json_agg(json_build_object(
      'schema', r.schema_name, 'name', r.relname, 'kind', r.relkind,
      'persistence', r.relpersistence, 'rls', r.relrowsecurity,
      'force_rls', r.relforcerowsecurity,
      'partition_key', CASE WHEN r.relkind = 'p' THEN pg_get_partkeydef(r.oid) END
    ) ORDER BY r.schema_name, r.relname), '[]'::json)
    FROM rels r WHERE r.relkind IN ('r', 'p', 'f')
  ),
  'columns', (
    SELECT coalesce(json_agg(json_build_object(
      'schema', r.schema_name, 'table', r.relname, 'name', a.attname,
      'position', a.attnum,
      'type', format_type(a.atttypid, a.atttypmod),
      'not_null', a.attnotnull,
      'default', CASE WHEN a.attgenerated = '' THEN pg_get_expr(ad.adbin, ad.adrelid) END,
      'identity', a.attidentity::text,
      'generated', a.attgenerated::text,
      'generation_expression',
        CASE WHEN a.attgenerated <> '' THEN pg_get_expr(ad.adbin, ad.adrelid) END,
      'collation',
        CASE WHEN a.attcollation <> 0 AND a.attcollation <> t.typcollation
             THEN co.collname END
    ) ORDER BY r.schema_name, r.relname, a.attnum), '[]'::json)
    FROM rels r
    JOIN pg_attribute a ON a.attrelid = r.oid AND a.attnum > 0 AND NOT a.attisdropped
    JOIN pg_type t ON t.oid = a.atttypid
    LEFT JOIN pg_attrdef ad ON ad.adrelid = a.attrelid AND ad.adnum = a.attnum
    LEFT JOIN pg_collation co ON co.oid = a.attcollation
    WHERE r.relkind IN ('r', 'p', 'f')
  ),
  'constraints', (
    SELECT coalesce(json_agg(json_build_object(
      'schema', r.schema_name, 'table', r.relname, 'name', con.conname,
      'type', con.contype::text,
      'columns', (
        SELECT json_agg(a.attname ORDER BY k.ord)
        FROM unnest(con.conkey) WITH ORDINALITY AS k(attnum, ord)
        JOIN pg_attribute a ON a.attrelid = con.conrelid AND a.attnum = k.attnum
      ),
      'ref_schema', fns.nspname,
      'ref_table', fc.relname,
      'ref_columns', (
        SELECT json_agg(a.attname ORDER BY k.ord)
        FROM unnest(con.confkey) WITH ORDINALITY AS k(attnum, ord)
        JOIN pg_attribute a ON a.attrelid = con.confrelid AND a.attnum = k.attnum
      ),
      'delete_set_columns', (
        SELECT json_agg(a.attname ORDER BY k.ord)
        FROM unnest(con.confdelsetcols) WITH ORDINALITY AS k(attnum, ord)
        JOIN pg_attribute a ON a.attrelid = con.conrelid AND a.attnum = k.attnum
      ),
      'on_update', nullif(con.confupdtype::text, ' '),
      'on_delete', nullif(con.confdeltype::text, ' '),
      'match', nullif(con.confmatchtype::text, ' '),
      'deferrable', con.condeferrable,
      'deferred', con.condeferred,
      'validated', con.convalidated,
      'no_inherit', con.connoinherit,
      'check_expression',
        CASE WHEN con.contype = 'c' THEN pg_get_expr(con.conbin, con.conrelid) END,
      'definition', pg_get_constraintdef(con.oid, true),
      'nulls_not_distinct', coalesce(i.indnullsnotdistinct, false)
    ) ORDER BY r.schema_name, r.relname, con.conname), '[]'::json)
    FROM pg_constraint con
    JOIN rels r ON r.oid = con.conrelid
    LEFT JOIN pg_class fc ON fc.oid = con.confrelid
    LEFT JOIN pg_namespace fns ON fns.oid = fc.relnamespace
    LEFT JOIN pg_index i ON i.indexrelid = con.conindid AND con.contype IN ('p', 'u', 'x')
    WHERE con.contype IN ('p', 'u', 'f', 'c', 'x')
  ),
  'indexes', (
    SELECT coalesce(json_agg(json_build_object(
      'schema', r.schema_name, 'table', r.relname, 'name', ic.relname,
      'method', am.amname,
      'unique', i.indisunique,
      'nulls_not_distinct', i.indnullsnotdistinct,
      'keys', (
        SELECT json_agg(pg_get_indexdef(i.indexrelid, k, true) ORDER BY k)
        FROM generate_series(1, i.indnkeyatts) AS k
      ),
      'include', (
        SELECT json_agg(pg_get_indexdef(i.indexrelid, k, true) ORDER BY k)
        FROM generate_series(i.indnkeyatts + 1, i.indnatts) AS k
      ),
      'options', (
        SELECT json_agg(o.opt ORDER BY o.ord)
        FROM unnest(i.indoption::int2[]) WITH ORDINALITY AS o(opt, ord)
      ),
      'opclasses', (
        SELECT json_agg(CASE WHEN oc.opcdefault THEN NULL ELSE oc.opcname END ORDER BY k.ord)
        FROM unnest(i.indclass::oid[]) WITH ORDINALITY AS k(opc, ord)
        JOIN pg_opclass oc ON oc.oid = k.opc
      ),
      'predicate', pg_get_expr(i.indpred, i.indrelid),
      'constraint_backed', EXISTS (
        SELECT 1 FROM pg_constraint c2
        WHERE c2.conindid = i.indexrelid AND c2.contype IN ('p', 'u', 'x')
      ),
      'definition', pg_get_indexdef(i.indexrelid)
    ) ORDER BY r.schema_name, r.relname, ic.relname), '[]'::json)
    FROM pg_index i
    JOIN rels r ON r.oid = i.indrelid
    JOIN pg_class ic ON ic.oid = i.indexrelid
    JOIN pg_am am ON am.oid = ic.relam
  ),
  'enums', (
    SELECT coalesce(json_agg(json_build_object(
      'schema', ns.nspname, 'name', t.typname,
      'labels', (
        SELECT json_agg(e.enumlabel ORDER BY e.enumsortorder)
        FROM pg_enum e WHERE e.enumtypid = t.oid
      )
    ) ORDER BY ns.nspname, t.typname), '[]'::json)
    FROM pg_type t
    JOIN ns ON ns.oid = t.typnamespace
    WHERE t.typtype = 'e'
      AND NOT EXISTS (
        SELECT 1 FROM ext_members e
        WHERE e.classid = 'pg_type'::regclass AND e.objid = t.oid
      )
  ),
  'sequences', (
    SELECT coalesce(json_agg(json_build_object(
      'schema', r.schema_name, 'name', r.relname,
      'data_type', format_type(s.seqtypid, NULL),
      'start', s.seqstart, 'increment', s.seqincrement,
      'min', s.seqmin, 'max', s.seqmax, 'cache', s.seqcache, 'cycle', s.seqcycle,
      'owned_by_schema', ons.nspname, 'owned_by_table', oc.relname,
      'owned_by_column', oa.attname,
      'dependency', d.deptype::text
    ) ORDER BY r.schema_name, r.relname), '[]'::json)
    FROM rels r
    JOIN pg_sequence s ON s.seqrelid = r.oid
    LEFT JOIN pg_depend d
      ON d.classid = 'pg_class'::regclass AND d.objid = r.oid
     AND d.refclassid = 'pg_class'::regclass AND d.deptype IN ('a', 'i')
    LEFT JOIN pg_class oc ON oc.oid = d.refobjid
    LEFT JOIN pg_namespace ons ON ons.oid = oc.relnamespace
    LEFT JOIN pg_attribute oa ON oa.attrelid = d.refobjid AND oa.attnum = d.refobjsubid
    WHERE r.relkind = 'S'
  ),
  'views', (
    SELECT coalesce(json_agg(json_build_object(
      'schema', r.schema_name, 'name', r.relname, 'kind', r.relkind,
      'definition', pg_get_viewdef(r.oid, true)
    ) ORDER BY r.schema_name, r.relname), '[]'::json)
    FROM rels r WHERE r.relkind IN ('v', 'm')
  ),
  'extensions', (
    SELECT coalesce(json_agg(json_build_object(
      'name', x.extname, 'version', x.extversion, 'schema', xn.nspname
    ) ORDER BY x.extname), '[]'::json)
    FROM pg_extension x JOIN pg_namespace xn ON xn.oid = x.extnamespace
  ),
  'domains', (
    SELECT coalesce(json_agg(json_build_object(
      'schema', ns.nspname, 'name', t.typname,
      'base_type', format_type(t.typbasetype, t.typtypmod),
      'not_null', t.typnotnull,
      'default', t.typdefault,
      'collation',
        CASE WHEN t.typcollation <> 0 AND t.typcollation <> bt.typcollation
             THEN co.collname END,
      'checks', (
        SELECT json_agg(pg_get_constraintdef(c.oid, true) ORDER BY c.conname)
        FROM pg_constraint c WHERE c.contypid = t.oid AND c.contype = 'c'
      )
    ) ORDER BY ns.nspname, t.typname), '[]'::json)
    FROM pg_type t
    JOIN ns ON ns.oid = t.typnamespace
    JOIN pg_type bt ON bt.oid = t.typbasetype
    LEFT JOIN pg_collation co ON co.oid = t.typcollation
    WHERE t.typtype = 'd'
      AND NOT EXISTS (
        SELECT 1 FROM ext_members e
        WHERE e.classid = 'pg_type'::regclass AND e.objid = t.oid
      )
  ),
  'functions', (
    SELECT coalesce(json_agg(json_build_object(
      'schema', ns.nspname, 'name', p.proname,
      'identity_args', pg_get_function_identity_arguments(p.oid),
      'args', pg_get_function_arguments(p.oid),
      'result', pg_get_function_result(p.oid),
      'kind', p.prokind::text,
      'language', l.lanname,
      'volatility', p.provolatile::text,
      'strict', p.proisstrict,
      'security_definer', p.prosecdef,
      'parallel', p.proparallel::text,
      'config', p.proconfig,
      'body', p.prosrc
    ) ORDER BY ns.nspname, p.proname, pg_get_function_identity_arguments(p.oid)), '[]'::json)
    FROM pg_proc p
    JOIN ns ON ns.oid = p.pronamespace
    JOIN pg_language l ON l.oid = p.prolang
    WHERE NOT EXISTS (
      SELECT 1 FROM ext_members e
      WHERE e.classid = 'pg_proc'::regclass AND e.objid = p.oid
    )
  ),
  'triggers', (
    SELECT coalesce(json_agg(json_build_object(
      'schema', r.schema_name, 'table', r.relname, 'name', tg.tgname,
      'relation', tg.tgrelid::regclass::text,
      'definition', pg_get_triggerdef(tg.oid, true),
      'enabled', tg.tgenabled::text
    ) ORDER BY r.schema_name, r.relname, tg.tgname), '[]'::json)
    FROM pg_trigger tg
    JOIN rels r ON r.oid = tg.tgrelid
    WHERE NOT tg.tgisinternal
  ),
  'policies', (
    SELECT coalesce(json_agg(json_build_object(
      'schema', r.schema_name, 'table', r.relname, 'name', pol.polname,
      'command', pol.polcmd::text,
      'permissive', pol.polpermissive,
      'roles', (
        SELECT json_agg(CASE WHEN x.oid = 0 THEN 'PUBLIC' ELSE pg_get_userbyid(x.oid)::text END)
        FROM unnest(pol.polroles) AS x(oid)
      ),
      'using', pg_get_expr(pol.polqual, pol.polrelid),
      'with_check', pg_get_expr(pol.polwithcheck, pol.polrelid)
    ) ORDER BY r.schema_name, r.relname, pol.polname), '[]'::json)
    FROM pg_policy pol
    JOIN rels r ON r.oid = pol.polrelid
  ),
  'grants', (
    SELECT coalesce(json_agg(g ORDER BY g.object_kind, g.schema, g.name, g.grantee, g.privilege),
                    '[]'::json)
    FROM (
      SELECT CASE r.relkind WHEN 'S' THEN 'sequence' WHEN 'v' THEN 'view'
                            WHEN 'm' THEN 'view' ELSE 'table' END AS object_kind,
             r.schema_name AS schema, r.relname AS name, NULL::text AS identity_args,
             CASE WHEN a.grantee = 0 THEN 'PUBLIC'
                  ELSE pg_get_userbyid(a.grantee)::text END AS grantee,
             a.privilege_type AS privilege, a.is_grantable AS grantable
      FROM rels r CROSS JOIN LATERAL aclexplode(r.relacl) a
      WHERE a.grantee <> r.relowner
      UNION ALL
      SELECT 'function', ns.nspname, p.proname, pg_get_function_identity_arguments(p.oid),
             CASE WHEN a.grantee = 0 THEN 'PUBLIC'
                  ELSE pg_get_userbyid(a.grantee)::text END,
             a.privilege_type, a.is_grantable
      FROM pg_proc p
      JOIN ns ON ns.oid = p.pronamespace
      CROSS JOIN LATERAL aclexplode(p.proacl) a
      WHERE a.grantee <> p.proowner
        AND NOT EXISTS (
          SELECT 1 FROM ext_members e
          WHERE e.classid = 'pg_proc'::regclass AND e.objid = p.oid
        )
    ) g
  )
)::text AS snapshot
"""


def build_snapshot_sql(schemas: Iterable[str] | None = None) -> str:
    """Return the catalog query; ``schemas`` restricts it to those schemas.

    Schema names are validated as plain identifiers before being inlined (the
    psql path cannot bind parameters), so nothing caller-controlled reaches the
    SQL text unchecked.
    """
    names = sorted(set(schemas or ()))
    for name in names:
        if not _IDENT_RE.match(name):
            raise ValueError(f"invalid schema name: {name!r}")
    schema_filter = ""
    if names:
        quoted = ", ".join(f"'{n}'" for n in names)
        schema_filter = f"AND n.nspname IN ({quoted})"
    return _SNAPSHOT_SQL.format(schema_filter=schema_filter)


def parse_snapshot(raw: str | bytes | dict[str, Any]) -> dict[str, Any]:
    """Parse the snapshot JSON (psql stdout or a psycopg value)."""
    if isinstance(raw, dict):
        doc = raw
    else:
        text = raw.decode() if isinstance(raw, bytes) else raw
        text = text.strip()
        if not text:
            raise SnapshotError("empty snapshot output")
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SnapshotError(f"snapshot is not valid JSON: {exc}") from exc
    required = ("server_version", "tables", "columns", "constraints", "indexes")
    if not isinstance(doc, dict) or any(k not in doc for k in required):
        raise SnapshotError("snapshot document is missing required sections")
    return doc


async def introspect_dsn(dsn: str, schemas: Iterable[str] | None = None) -> dict[str, Any]:
    """Snapshot the database at ``dsn`` inside a READ ONLY transaction."""
    import psycopg

    conninfo = normalise_dsn(dsn)
    sql = build_snapshot_sql(schemas)
    conn = await psycopg.AsyncConnection.connect(conninfo, connect_timeout=10)
    try:
        # read_only makes psycopg open the transaction with BEGIN READ ONLY:
        # transaction-scoped, never a session-level SET.
        await conn.set_read_only(True)
        async with conn.transaction(), conn.cursor() as cur:
            await cur.execute(sql.encode())
            row = await cur.fetchone()
    finally:
        await conn.close()
    if row is None:
        raise SnapshotError("snapshot query returned no row")
    return parse_snapshot(row[0])


def normalise_dsn(dsn: str) -> str:
    """Accept SQLAlchemy-style ``postgresql+psycopg://`` URLs as plain libpq URLs."""
    return re.sub(r"^postgres(?:ql)?\+[A-Za-z0-9_]+://", "postgresql://", dsn.strip())


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalise_expression(expr: str | None) -> str | None:
    """Whitespace-collapse an expression and fold semantically-identical aliases."""
    if expr is None:
        return None
    out = " ".join(expr.split())
    for pattern, repl in _EXPR_ALIASES:
        out = pattern.sub(repl, out)
    return out


def _q(schema: str, name: str) -> str:
    return f"{schema}.{name}"


def _cols(cols: list[str] | None) -> str:
    return ", ".join(cols or [])


def _index_key_parts(idx: dict[str, Any]) -> list[str]:
    keys: list[str] = list(idx.get("keys") or [])
    options: list[int] = list(idx.get("options") or [])
    opclasses: list[str | None] = list(idx.get("opclasses") or [])
    parts: list[str] = []
    for pos, key in enumerate(keys):
        part = normalise_expression(key) or ""
        if pos < len(opclasses) and opclasses[pos]:
            part += f" {opclasses[pos]}"
        opt = options[pos] if pos < len(options) else 0
        if opt & 1:  # INDOPTION_DESC
            part += " DESC"
            if not opt & 2:
                part += " NULLS LAST"
        elif opt & 2:  # INDOPTION_NULLS_FIRST on an ASC column
            part += " NULLS FIRST"
        parts.append(part)
    return parts


class _Obj:
    __slots__ = ("fields", "key", "kind", "name", "parent", "position", "sources")

    def __init__(
        self,
        kind: str,
        key: str,
        fields: dict[str, Any],
        name: str | None = None,
        parent: str | None = None,
        position: int | None = None,
    ) -> None:
        self.kind = kind
        self.key = key
        self.fields = fields
        self.name = name
        self.parent = parent
        self.position = position
        self.sources: list[str] = []


def _ignored(schema: str, table: str | None, ignore: set[str]) -> bool:
    if table is None:
        return False
    return table in ignore or f"{schema}.{table}" in ignore


def normalise_snapshot(
    doc: dict[str, Any], ignore_tables: Iterable[str] = DEFAULT_IGNORE_TABLES
) -> dict[str, dict[str, _Obj]]:
    """Turn a raw snapshot into ``{kind: {key: object}}`` with ignores applied."""
    ignore = set(ignore_tables)
    out: dict[str, dict[str, _Obj]] = {k: {} for k in KIND_ORDER}

    def add(obj: _Obj) -> None:
        bucket = out[obj.kind]
        key = obj.key
        n = 2
        # Two objects with one definition (e.g. a duplicate index under another
        # name) must not overwrite each other; suffix deterministically.
        while key in bucket:
            key = f"{obj.key} #{n}"
            n += 1
        obj.key = key
        bucket[key] = obj

    for ext in doc.get("extensions") or []:
        add(
            _Obj(
                "extension",
                ext["name"],
                {"schema": ext.get("schema"), "version": ext.get("version")},
            )
        )

    for enum in doc.get("enums") or []:
        add(_Obj("enum", _q(enum["schema"], enum["name"]), {"labels": enum.get("labels") or []}))

    identity_seq: dict[str, dict[str, Any]] = {}
    for seq in sorted(
        doc.get("sequences") or [],
        key=lambda s: (s["schema"], s["name"], s.get("dependency") or ""),
    ):
        owner_t = seq.get("owned_by_table")
        owner_s = seq.get("owned_by_schema") or seq["schema"]
        if _ignored(owner_s, owner_t, ignore):
            continue
        params = {
            "data_type": seq.get("data_type"),
            "start": seq.get("start"),
            "increment": seq.get("increment"),
            "min": seq.get("min"),
            "max": seq.get("max"),
            "cache": seq.get("cache"),
            "cycle": seq.get("cycle"),
        }
        if seq.get("dependency") == "i" and owner_t and seq.get("owned_by_column"):
            # Identity sequences are compared as part of their column; their
            # generated names are an implementation detail.
            identity_seq[f"{owner_s}.{owner_t}.{seq['owned_by_column']}"] = params
            continue
        owned_by = (
            f"{owner_s}.{owner_t}.{seq['owned_by_column']}"
            if owner_t and seq.get("owned_by_column")
            else None
        )
        add(_Obj("sequence", _q(seq["schema"], seq["name"]), {**params, "owned_by": owned_by}))

    for tbl in doc.get("tables") or []:
        if _ignored(tbl["schema"], tbl["name"], ignore):
            continue
        add(
            _Obj(
                "table",
                _q(tbl["schema"], tbl["name"]),
                {
                    "kind": _RELKIND.get(tbl.get("kind") or "", tbl.get("kind")),
                    "persistence": _PERSISTENCE.get(
                        tbl.get("persistence") or "", tbl.get("persistence")
                    ),
                    "row_level_security": bool(tbl.get("rls")),
                    "force_row_level_security": bool(tbl.get("force_rls")),
                    "partition_key": normalise_expression(tbl.get("partition_key")),
                },
            )
        )

    for col in doc.get("columns") or []:
        if _ignored(col["schema"], col["table"], ignore):
            continue
        tkey = _q(col["schema"], col["table"])
        ckey = f"{tkey}.{col['name']}"
        add(
            _Obj(
                "column",
                ckey,
                {
                    "type": col.get("type"),
                    "not_null": bool(col.get("not_null")),
                    "default": normalise_expression(col.get("default")),
                    "identity": _IDENTITY.get(col.get("identity") or ""),
                    "identity_sequence": identity_seq.get(ckey),
                    "generated": _GENERATED.get(col.get("generated") or ""),
                    "generation_expression": normalise_expression(col.get("generation_expression")),
                    "collation": col.get("collation"),
                },
                parent=tkey,
                position=col.get("position") if isinstance(col.get("position"), int) else None,
            )
        )

    for con in doc.get("constraints") or []:
        if _ignored(con["schema"], con["table"], ignore):
            continue
        tkey = _q(con["schema"], con["table"])
        ctype = con.get("type")
        common = {"deferrable": bool(con.get("deferrable")), "deferred": bool(con.get("deferred"))}
        name = con.get("name")
        if ctype == "p":
            add(
                _Obj(
                    "primary_key",
                    tkey,
                    {"columns": con.get("columns") or [], **common},
                    name=name,
                    parent=tkey,
                )
            )
        elif ctype == "u":
            nnd = " NULLS NOT DISTINCT" if con.get("nulls_not_distinct") else ""
            add(
                _Obj(
                    "unique",
                    f"{tkey}({_cols(con.get('columns'))}){nnd}",
                    common,
                    name=name,
                    parent=tkey,
                )
            )
        elif ctype == "f":
            ref = _q(con.get("ref_schema") or "", con.get("ref_table") or "")
            add(
                _Obj(
                    "foreign_key",
                    f"{tkey}({_cols(con.get('columns'))}) -> {ref}({_cols(con.get('ref_columns'))})",
                    {
                        "on_delete": _FK_ACTION.get(
                            con.get("on_delete") or "", con.get("on_delete")
                        ),
                        "on_update": _FK_ACTION.get(
                            con.get("on_update") or "", con.get("on_update")
                        ),
                        "match": _FK_MATCH.get(con.get("match") or "", con.get("match")),
                        "delete_set_columns": con.get("delete_set_columns"),
                        "validated": bool(con.get("validated")),
                        **common,
                    },
                    name=name,
                    parent=tkey,
                )
            )
        elif ctype == "c":
            expr = normalise_expression(con.get("check_expression") or con.get("definition"))
            add(
                _Obj(
                    "check",
                    f"{tkey} CHECK {expr}",
                    {
                        "no_inherit": bool(con.get("no_inherit")),
                        "validated": bool(con.get("validated")),
                    },
                    name=name,
                    parent=tkey,
                )
            )
        elif ctype == "x":
            add(
                _Obj(
                    "exclusion",
                    f"{tkey} {normalise_expression(con.get('definition'))}",
                    common,
                    name=name,
                    parent=tkey,
                )
            )

    for idx in doc.get("indexes") or []:
        if idx.get("constraint_backed") or _ignored(idx["schema"], idx["table"], ignore):
            continue
        tkey = _q(idx["schema"], idx["table"])
        unique = "UNIQUE " if idx.get("unique") else ""
        nnd = " NULLS NOT DISTINCT" if idx.get("nulls_not_distinct") else ""
        key = f"{unique}{tkey} USING {idx.get('method')} ({', '.join(_index_key_parts(idx))})"
        if idx.get("include"):
            key += f" INCLUDE ({_cols(idx.get('include'))})"
        key += nnd
        pred = normalise_expression(idx.get("predicate"))
        if pred:
            key += f" WHERE {pred}"
        add(_Obj("index", key, {}, name=idx.get("name"), parent=tkey))

    for view in doc.get("views") or []:
        add(
            _Obj(
                "view",
                _q(view["schema"], view["name"]),
                {
                    "kind": _RELKIND.get(view.get("kind") or "", view.get("kind")),
                    "definition": normalise_expression(view.get("definition")),
                },
            )
        )

    for dom in doc.get("domains") or []:
        add(
            _Obj(
                "domain",
                _q(dom["schema"], dom["name"]),
                {
                    "base_type": dom.get("base_type"),
                    "not_null": bool(dom.get("not_null")),
                    "default": normalise_expression(dom.get("default")),
                    "collation": dom.get("collation"),
                    "checks": sorted(
                        normalise_expression(c) or "" for c in dom.get("checks") or []
                    ),
                },
            )
        )

    for fn in doc.get("functions") or []:
        add(
            _Obj(
                "function",
                f"{_q(fn['schema'], fn['name'])}({fn.get('identity_args') or ''})",
                {
                    "kind": _PROKIND.get(fn.get("kind") or "", fn.get("kind")),
                    "arguments": fn.get("args"),
                    "result": fn.get("result"),
                    "language": fn.get("language"),
                    "volatility": _VOLATILITY.get(fn.get("volatility") or "", fn.get("volatility")),
                    "strict": bool(fn.get("strict")),
                    "security_definer": bool(fn.get("security_definer")),
                    "parallel": _PARALLEL.get(fn.get("parallel") or "", fn.get("parallel")),
                    "config": sorted(fn.get("config") or []),
                    "body": normalise_expression(fn.get("body")),
                },
            )
        )

    for tg in doc.get("triggers") or []:
        if _ignored(tg["schema"], tg["table"], ignore):
            continue
        tkey = _q(tg["schema"], tg["table"])
        definition = normalise_expression(tg.get("definition")) or ""
        definition = _TRIGGER_HEAD_RE.sub("", definition, count=1)
        # The table is the key's prefix already; its deparsed (possibly mapped) name
        # must not make two identical triggers differ.
        definition = _TRIGGER_ON_RE.sub(" ON <table> ", definition, count=1)
        add(
            _Obj(
                "trigger",
                f"{tkey} {definition}",
                {"enabled": _TGENABLED.get(tg.get("enabled") or "", tg.get("enabled"))},
                name=tg.get("name"),
                parent=tkey,
            )
        )

    for pol in doc.get("policies") or []:
        if _ignored(pol["schema"], pol["table"], ignore):
            continue
        tkey = _q(pol["schema"], pol["table"])
        add(
            _Obj(
                "policy",
                f"{tkey} POLICY {pol.get('name')}",
                {
                    "command": _POLCMD.get(pol.get("command") or "", pol.get("command")),
                    "permissive": bool(pol.get("permissive")),
                    "roles": sorted(pol.get("roles") or []),
                    "using": normalise_expression(pol.get("using")),
                    "with_check": normalise_expression(pol.get("with_check")),
                },
                parent=tkey,
            )
        )

    for g in doc.get("grants") or []:
        okind = g.get("object_kind") or "table"
        if okind != "function" and _ignored(g["schema"], g["name"], ignore):
            continue
        obj = _q(g["schema"], g["name"])
        if okind == "function":
            obj += f"({g.get('identity_args') or ''})"
        add(
            _Obj(
                "grant",
                f"{okind} {obj} TO {g.get('grantee')} {g.get('privilege')}",
                {"grantable": bool(g.get("grantable"))},
                parent=obj if okind == "table" else None,
            )
        )

    return out


# ---------------------------------------------------------------------------
# Source mapping (N sources -> 1 target)
# ---------------------------------------------------------------------------


def map_snapshot(
    doc: dict[str, Any],
    schema_map: dict[str, str] | None = None,
    table_prefix: str | None = None,
    ignore_tables: Iterable[str] = DEFAULT_IGNORE_TABLES,
) -> dict[str, Any]:
    """Rename one SOURCE snapshot to where the merged target keeps its objects.

    ``schema_map`` renames schemas everywhere (objects, references, schema-qualified
    type names); ``table_prefix`` prefixes every relation name (tables, views,
    sequences and every reference to them: constraints, FK targets, indexes,
    triggers, policies, grants, sequence owners and ``nextval('<seq>'...)``
    defaults). Ignored tables (migration bookkeeping) keep their names so the
    ignore still applies. Pure: returns a new document.
    """
    smap = dict(schema_map or {})
    prefix = table_prefix or ""
    if not smap and not prefix:
        return doc
    ignore = set(ignore_tables)
    out = copy.deepcopy(doc)

    def sch(name: str | None) -> str | None:
        return smap.get(name, name) if name is not None else None

    def rel(schema: str | None, name: str | None) -> str | None:
        if name is None or not prefix or _ignored(schema or "", name, ignore):
            return name
        return prefix + name

    seq_names = {(q.get("schema"), q.get("name")) for q in out.get("sequences") or []}

    def fix_default(expr: str | None) -> str | None:
        if not expr:
            return expr
        for schema, name in seq_names:
            new = rel(schema, name)
            new_schema = sch(schema)
            expr = expr.replace(f"'{schema}.{name}'::regclass", f"'{new_schema}.{new}'::regclass")
            expr = expr.replace(f"'{name}'::regclass", f"'{new}'::regclass")
        return expr

    def fix_type(t: str | None) -> str | None:
        if not t:
            return t
        for old, new in smap.items():
            if t.startswith(f"{old}."):
                return f"{new}.{t[len(old) + 1 :]}"
        return t

    for tbl in out.get("tables") or []:
        tbl["name"] = rel(tbl["schema"], tbl["name"])
        tbl["schema"] = sch(tbl["schema"])
    for col in out.get("columns") or []:
        col["table"] = rel(col["schema"], col["table"])
        col["schema"] = sch(col["schema"])
        col["default"] = fix_default(col.get("default"))
        col["type"] = fix_type(col.get("type"))
    for con in out.get("constraints") or []:
        con["table"] = rel(con["schema"], con["table"])
        con["schema"] = sch(con["schema"])
        if con.get("ref_table"):
            con["ref_table"] = rel(con.get("ref_schema"), con["ref_table"])
            con["ref_schema"] = sch(con.get("ref_schema"))
    for section in ("indexes", "triggers", "policies"):
        for item in out.get(section) or []:
            item["table"] = rel(item["schema"], item["table"])
            item["schema"] = sch(item["schema"])
    for seq in out.get("sequences") or []:
        seq["name"] = rel(seq["schema"], seq["name"])
        seq["schema"] = sch(seq["schema"])
        if seq.get("owned_by_table"):
            seq["owned_by_table"] = rel(seq.get("owned_by_schema"), seq["owned_by_table"])
        seq["owned_by_schema"] = sch(seq.get("owned_by_schema"))
    for view in out.get("views") or []:
        view["name"] = rel(view["schema"], view["name"])
        view["schema"] = sch(view["schema"])
    for section in ("enums", "domains", "functions"):
        for item in out.get(section) or []:
            item["schema"] = sch(item["schema"])
    for dom in out.get("domains") or []:
        dom["base_type"] = fix_type(dom.get("base_type"))
    for g in out.get("grants") or []:
        if g.get("object_kind") != "function":
            g["name"] = rel(g["schema"], g["name"])
        g["schema"] = sch(g["schema"])
    return out


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


def _sort_key(kind: str, key: str) -> tuple[int, str]:
    return (_KIND_RANK[kind], key)


def _counts(norm: dict[str, dict[str, _Obj]]) -> dict[str, int]:
    return {k: len(v) for k, v in norm.items()}


def _major(doc: dict[str, Any]) -> int | None:
    num = doc.get("server_version_num")
    return int(num) // 10000 if isinstance(num, int) else None


@dataclass(frozen=True)
class SourceSnapshot:
    """One named source snapshot and where its objects live in the merged target."""

    name: str
    doc: dict[str, Any]
    schema_map: dict[str, str] | None = None
    table_prefix: str | None = None


def diff_snapshots(
    source: dict[str, Any],
    target: dict[str, Any],
    options: SchemaDiffOptions | None = None,
) -> SchemaDiffResult:
    """Diff two raw snapshots. ``source`` is the reference (the Python side)."""
    return diff_union([SourceSnapshot("source", source)], target, options)


def _union(
    sources: list[SourceSnapshot], ignore: Iterable[str], skip: set[str]
) -> tuple[dict[str, dict[str, _Obj]], list[SchemaSourceConflict], dict[str, dict[str, int]]]:
    union: dict[str, dict[str, _Obj]] = {k: {} for k in KIND_ORDER}
    conflicts: dict[tuple[str, str], SchemaSourceConflict] = {}
    counts: dict[str, dict[str, int]] = {}
    for src in sources:
        mapped = map_snapshot(src.doc, src.schema_map, src.table_prefix, ignore)
        norm = normalise_snapshot(mapped, ignore)
        for kind in skip:
            norm[kind] = {}
        counts[src.name] = _counts(norm)
        for kind, objs in norm.items():
            for key, obj in objs.items():
                kept = union[kind].get(key)
                if kept is None:
                    obj.sources = [src.name]
                    union[kind][key] = obj
                elif kept.fields == obj.fields:
                    kept.sources.append(src.name)
                else:
                    c = conflicts.setdefault(
                        (kind, key),
                        SchemaSourceConflict(
                            kind=kind, key=key, kept=kept.sources[0], conflicts=[]
                        ),
                    )
                    c.conflicts.append(
                        SourceConflictEntry(
                            source=src.name,
                            fields=[
                                FieldChange(
                                    field=f, source=kept.fields.get(f), target=obj.fields.get(f)
                                )
                                for f in sorted(set(kept.fields) | set(obj.fields))
                                if kept.fields.get(f) != obj.fields.get(f)
                            ],
                        )
                    )
    ordered = sorted(conflicts.values(), key=lambda c: _sort_key(c.kind, c.key))
    return union, ordered, counts


def _column_order(
    src: dict[str, dict[str, _Obj]], tgt: dict[str, dict[str, _Obj]]
) -> list[ColumnOrderDifference]:
    """Relative order of the columns BOTH sides have, per table both sides have."""
    by_table: dict[str, dict[str, tuple[_Obj | None, _Obj | None]]] = {}
    for key, obj in src["column"].items():
        if key in tgt["column"] and obj.parent in tgt["table"]:
            by_table.setdefault(obj.parent or "", {})[key] = (obj, tgt["column"][key])
    out = []
    for table in sorted(by_table):
        pairs = by_table[table]

        def order(side: int, pairs: dict = pairs, table: str = table) -> list[str]:
            ranked = sorted(pairs.items(), key=lambda kv: (kv[1][side].position or 0, kv[0]))
            return [k[len(table) + 1 :] for k, _ in ranked]

        s_order, t_order = order(0), order(1)
        if s_order != t_order:
            owners = src["table"].get(table)
            out.append(
                ColumnOrderDifference(
                    table=table,
                    source_order=s_order,
                    target_order=t_order,
                    sources=list(owners.sources) if owners else [],
                )
            )
    return out


def diff_union(
    sources: list[SourceSnapshot],
    target: dict[str, Any],
    options: SchemaDiffOptions | None = None,
) -> SchemaDiffResult:
    """Diff the UNION of N mapped source snapshots against one target snapshot."""
    if not sources:
        raise ValueError("at least one source snapshot is required")
    opts = options or SchemaDiffOptions()
    ignore = DEFAULT_IGNORE_TABLES if opts.ignore_tables is None else opts.ignore_tables
    skip = set(opts.ignore_kinds or ())
    src, conflicts, per_source = _union(sources, ignore, skip)
    tgt = normalise_snapshot(target, ignore)
    for kind in skip:
        tgt[kind] = {}
    multi = len(sources) > 1

    missing: list[SchemaObjectDiff] = []
    extra: list[SchemaObjectDiff] = []
    changed: list[SchemaObjectDiff] = []
    name_only: list[NameOnlyDifference] = []

    missing_tables = set(src["table"]) - set(tgt["table"])
    extra_tables = set(tgt["table"]) - set(src["table"])

    for kind in KIND_ORDER:
        s_objs, t_objs = src[kind], tgt[kind]
        for key in sorted(set(s_objs) - set(t_objs)):
            obj = s_objs[key]
            if kind in _TABLE_CHILD_KINDS and obj.parent in missing_tables:
                continue
            missing.append(
                SchemaObjectDiff(
                    kind=kind, key=key, name=obj.name, source=obj.fields, sources=obj.sources
                )
            )
        for key in sorted(set(t_objs) - set(s_objs)):
            obj = t_objs[key]
            if kind in _TABLE_CHILD_KINDS and obj.parent in extra_tables:
                continue
            extra.append(SchemaObjectDiff(kind=kind, key=key, name=obj.name, target=obj.fields))
        for key in sorted(set(s_objs) & set(t_objs)):
            s_obj, t_obj = s_objs[key], t_objs[key]
            fields = [
                FieldChange(field=f, source=s_obj.fields.get(f), target=t_obj.fields.get(f))
                for f in sorted(set(s_obj.fields) | set(t_obj.fields))
                if s_obj.fields.get(f) != t_obj.fields.get(f)
            ]
            if kind in _NAMED_KINDS and s_obj.name != t_obj.name:
                if opts.strict_names:
                    fields.append(FieldChange(field="name", source=s_obj.name, target=t_obj.name))
                else:
                    name_only.append(
                        NameOnlyDifference(
                            kind=kind, key=key, source_name=s_obj.name, target_name=t_obj.name
                        )
                    )
            if fields:
                changed.append(
                    SchemaObjectDiff(
                        kind=kind, key=key, name=s_obj.name, fields=fields, sources=s_obj.sources
                    )
                )

    for bucket in (missing, extra, changed):
        bucket.sort(key=lambda d: _sort_key(d.kind, d.key))
    name_only.sort(key=lambda d: _sort_key(d.kind, d.key))
    column_order = [] if "column" in skip else _column_order(src, tgt)

    warnings: list[str] = []
    t_major = _major(target)
    for snap in sources:
        s_major = _major(snap.doc)
        if s_major != t_major:
            label = "source" if not multi else f"source {snap.name!r}"
            warnings.append(
                f"server major versions differ ({label} {s_major}, target {t_major}): "
                "deparsed expressions may differ for reasons unrelated to the migrations"
            )

    # Extensions are database-level and plpgsql exists in EVERY database, so they
    # cannot make a comparison non-vacuous; ignored tables are already gone.
    vacuous = not any(len(v) for k, v in (*src.items(), *tgt.items()) if k != "extension")
    if vacuous:
        warnings.append(
            "vacuous comparison: neither side has any non-ignored schema object "
            "(did the migrations run against these databases?)"
        )
    vacuous_sources = (
        [n for n, c in per_source.items() if not any(v for k, v in c.items() if k != "extension")]
        if multi
        else []
    )
    for name in vacuous_sources:
        warnings.append(
            f"source {name!r} contributed no schema object (did its migrations run?): "
            "the comparison cannot be equal"
        )

    order_fails = bool(column_order) and not opts.ignore_column_order
    first = sources[0].doc
    return SchemaDiffResult(
        computed=True,
        equal=not vacuous
        and not vacuous_sources
        and not (missing or extra or changed or conflicts or order_fails),
        vacuous=vacuous,
        missing_in_target=missing,
        extra_in_target=extra,
        changed=changed,
        name_only_differences=name_only,
        source_conflicts=conflicts,
        column_order_differences=column_order,
        vacuous_sources=vacuous_sources,
        summary={
            "missing_in_target": len(missing),
            "extra_in_target": len(extra),
            "changed": len(changed),
            "name_only_differences": len(name_only),
            "source_conflicts": len(conflicts),
            "column_order_differences": len(column_order),
        },
        options=opts.model_copy(update={"ignore_tables": list(ignore)}),
        source=SchemaSideInfo(
            server_version=str(first.get("server_version")), object_counts=_counts(src)
        ),
        sources=[
            NamedSchemaSideInfo(
                name=snap.name,
                server_version=str(snap.doc.get("server_version")),
                object_counts=per_source[snap.name],
                mapping=(
                    SchemaMapping(schema_map=snap.schema_map or {}, table_prefix=snap.table_prefix)
                    if snap.schema_map or snap.table_prefix
                    else None
                ),
            )
            for snap in sources
        ]
        if multi
        else [],
        target=SchemaSideInfo(
            server_version=str(target.get("server_version")), object_counts=_counts(tgt)
        ),
        warnings=warnings,
    )
