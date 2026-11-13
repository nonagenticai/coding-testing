ARG BASE_REGISTRY=docker.io

# --- Stage 1: Builder ---
FROM ${BASE_REGISTRY}/library/python:3.14-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md ./

# git is required because coding-client is a pinned GIT dependency (a public repository, not a
# package on an index). uv shells out to the git CLI for those; the slim base does not ship it.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN uv sync --frozen --no-dev --no-install-project

# --- Stage 2: Production ---
FROM ${BASE_REGISTRY}/library/python:3.14-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create appuser BEFORE the copies so each COPY can set ownership inline via
# --chown. A trailing `chown -R appuser:appuser /app` instead costs full-image
# rewrite time and adds a second, duplicate image layer — this repo's own
# measured cache-cold build hit 1316s with the recursive chown in place (the
# registry_build_cache flag being on does not save you: a lock or base-image
# change busts the cache and the chown then runs at full cost). /app itself is
# chowned non-recursively so the directory keeps the ownership the old
# recursive chown gave it.
RUN useradd -m -u 1000 appuser && mkdir -p /app && chown appuser:appuser /app
WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app/uv.lock /app/uv.lock
COPY --from=builder --chown=appuser:appuser /app/pyproject.toml /app/pyproject.toml
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser src/ ./src/
# alembic/ + alembic.ini are needed at runtime: the lifespan runs
# `alembic upgrade head` against the Postgres results store on boot
# (see src/app_factory.py:_run_alembic_upgrade, which resolves /app/alembic).
COPY --chown=appuser:appuser alembic/ ./alembic/
COPY --chown=appuser:appuser alembic.ini ./

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
USER appuser

EXPOSE 8689

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8689/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8689"]
