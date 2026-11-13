"""ORM models for the coding-testing Postgres store.

These populate ``Base.metadata`` so Alembic (env.py) can introspect them and
so future autogenerate stays coherent. The schema of record is the hand-written
migration ``alembic/versions/0001_initial.py`` — keep the two in sync.

Table groups:
- Results store (test_result / analysis_result): replaces the per-container
  SQLite files in the old TestResultsStorageService.
- Orchestration store (orchestration_test_suite / orchestration_plan /
  orchestration_execution_history): the live backing for the test-orchestration
  router. Its state moved out of module-level dicts into Postgres (#32) so the
  service runs >1 replica without split-brain; written via
  services/orchestration_store.py.
- Rust test runs (rust_test_runs): the packed working tree a runner Job downloads
  once, and the run's result; written via services/rust_test_runs.py (migration 0004).
- Run specs (workspace_run_specs): what each workspace run is graded against, bound
  to the workspace + tenant that started it; written via services/run_spec_store.py
  (migration 0005).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    Double,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import DateTime

from db import Base


class TestResult(Base):
    """A single test execution result, scoped by container_name."""

    __tablename__ = "test_results"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    test_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    container_name: Mapped[str] = mapped_column(Text, nullable=False)
    container_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    test_name: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(Text, nullable=False)
    test_type: Mapped[str] = mapped_column(Text, nullable=False)
    test_framework: Mapped[str | None] = mapped_column(Text, nullable=True)
    exit_code: Mapped[int] = mapped_column(Integer, nullable=False)
    stdout: Mapped[str] = mapped_column(Text, nullable=False)
    stderr: Mapped[str] = mapped_column(Text, nullable=False)
    execution_time_ms: Mapped[float] = mapped_column(Double, nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    source_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    test_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    dependencies: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    resource_usage: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # Card context (migration 0003). Nullable: the execute path has no card.
    # tenant_id is TEXT like every other tenant id in this service.
    card_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    tenant_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    project_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    commit_sha: Mapped[str | None] = mapped_column(Text, nullable=True)
    correlation_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 'execute' = POST /api/tests/execute; 'aa_qa' = conveyor-computed results.
    source: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'execute'"))

    __table_args__ = (
        Index("ix_test_results_card_id", "card_id"),
        Index("ix_test_results_container_created", "container_name", "created_at"),
    )


class AnalysisResult(Base):
    """AI analysis result linked to a test_result by test_id."""

    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    analysis_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    test_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("test_results.test_id", ondelete="CASCADE"),
        nullable=False,
    )
    container_name: Mapped[str] = mapped_column(Text, nullable=False)
    root_causes: Mapped[list[Any]] = mapped_column(JSONB, nullable=False)
    fix_suggestions: Mapped[list[Any]] = mapped_column(JSONB, nullable=False)
    detected_patterns: Mapped[list[Any]] = mapped_column(JSONB, nullable=False)
    analysis_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class OrchestrationTestSuiteRow(Base):
    """Serialized OrchestrationTestSuite, keyed by suite_id."""

    __tablename__ = "orchestration_test_suite"

    suite_id: Mapped[str] = mapped_column(Text, primary_key=True)
    data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class OrchestrationPlanRow(Base):
    """Serialized TestOrchestrationPlan, keyed by plan_id."""

    __tablename__ = "orchestration_plan"

    plan_id: Mapped[str] = mapped_column(Text, primary_key=True)
    data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class OrchestrationExecutionHistoryRow(Base):
    """Append-only orchestration execution history record."""

    __tablename__ = "orchestration_execution_history"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class RustTestRunRow(Base):
    """One Rust workspace test run (services/rust_test_runs.py; migration 0004)."""

    __tablename__ = "rust_test_runs"

    run_id: Mapped[str] = mapped_column(Text, primary_key=True)
    container_name: Mapped[str] = mapped_column(Text, nullable=False)
    tenant_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    archive: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    deadline: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("ix_rust_test_runs_created", "created_at"),)


class WorkspaceRunSpecRow(Base):
    """The spec of a workspace run (services/run_spec_store.py; migration 0005)."""

    __tablename__ = "workspace_run_specs"

    nonce: Mapped[str] = mapped_column(Text, primary_key=True)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    container_name: Mapped[str] = mapped_column(Text, nullable=False)
    tenant_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    spec: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("ix_workspace_run_specs_created", "created_at"),)
