-- Equivalent of alembic 0001_create_tags (as a Rust sqlx port would write it).
CREATE TABLE tags (
    id uuid PRIMARY KEY,
    tenant_id varchar(64) NOT NULL,
    name varchar(64) NOT NULL,
    CONSTRAINT uq_tags_tenant_name UNIQUE (tenant_id, name)
);
