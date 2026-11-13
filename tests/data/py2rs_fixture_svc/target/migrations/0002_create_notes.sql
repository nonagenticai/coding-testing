-- Equivalent of alembic 0002_create_notes (as a Rust sqlx port would write it).
CREATE TABLE notes (
    id uuid PRIMARY KEY,
    tenant_id varchar(64) NOT NULL,
    title varchar(200) NOT NULL,
    body text NOT NULL,
    tag_id uuid CONSTRAINT fk_notes_tag_id_tags REFERENCES tags (id) ON DELETE RESTRICT,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ix_notes_tenant_tag ON notes (tenant_id, tag_id);
