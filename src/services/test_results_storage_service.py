"""
Test Results Storage Service

Persistent storage for test execution results with container-scoped access.
Stores test results in a shared Postgres database so the service is safe to run
with more than one replica: a request that lands on replica B can read results
written by replica A. (Previously this used per-container local SQLite files,
which broke 2-replica correctness.)

Schema lives in ``alembic/versions/0001_initial.py`` / ``src/db_models.py``.
``container_name`` remains a plain scoping column, preserving the historical
per-container isolation semantics.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from db import ensure_session_factory
from db_models import AnalysisResult, TestResult

logger = logging.getLogger(__name__)


def _iso(value: Any) -> Any:
    """Render a datetime as an ISO string (parity with the old SQLite output)."""
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _row_to_dict(row: TestResult) -> dict[str, Any]:
    """Serialize a TestResult ORM row to the historical result dict shape."""
    return {
        "id": row.id,
        "test_id": row.test_id,
        "container_name": row.container_name,
        "container_id": row.container_id,
        "test_name": row.test_name,
        "language": row.language,
        "test_type": row.test_type,
        "test_framework": row.test_framework,
        "exit_code": row.exit_code,
        "stdout": row.stdout,
        "stderr": row.stderr,
        "execution_time_ms": row.execution_time_ms,
        "success": bool(row.success),
        "source_code": row.source_code,
        "test_code": row.test_code,
        "file_path": row.file_path,
        "dependencies": row.dependencies,
        "resource_usage": row.resource_usage,
        "timestamp": _iso(row.timestamp),
        "created_at": _iso(row.created_at),
        "updated_at": _iso(row.updated_at),
        "card_id": row.card_id,
        "tenant_id": row.tenant_id,
        "project_id": row.project_id,
        "commit_sha": row.commit_sha,
        "correlation_id": row.correlation_id,
        "source": row.source,
    }


class TestIdConflictError(Exception):
    """A ``test_id`` is already stored under a different container or tenant."""

    __test__ = False  # not a pytest class, despite the name

    def __init__(self, test_id: str) -> None:
        super().__init__(f"test_id {test_id!r} already exists in another scope")
        self.test_id = test_id


class TestResultsStorageService:
    """
    Service for storing and retrieving test execution results.

    Backed by a shared Postgres database. ``container_name`` scopes reads and
    writes, preserving the container-isolation contract of the old file-based
    store while remaining correct across multiple service replicas.
    """

    async def store_test_result(
        self,
        test_id: str,
        container_name: str,
        container_id: str | None,
        test_name: str,
        language: str,
        test_type: str,
        exit_code: int,
        stdout: str,
        stderr: str,
        execution_time_ms: float,
        success: bool,
        source_code: str | None = None,
        test_code: str | None = None,
        file_path: str | None = None,
        dependencies: dict[str, str] | None = None,
        resource_usage: dict[str, float] | None = None,
        test_framework: str | None = None,
        timestamp: datetime | None = None,
        *,
        card_id: str | None = None,
        tenant_id: str | None = None,
        project_id: str | None = None,
        commit_sha: str | None = None,
        correlation_id: str | None = None,
        source: str | None = None,
    ) -> None:
        """Store a test execution result.

        The keyword-only card-context arguments are optional; existing callers
        (the execute path) omit them and get NULLs plus ``source='execute'``
        from the column's server default.
        """
        card_context: dict[str, Any] = {
            "card_id": card_id,
            "tenant_id": tenant_id,
            "project_id": project_id,
            "commit_sha": commit_sha,
            "correlation_id": correlation_id,
        }
        if source is not None:
            card_context["source"] = source
        factory = await ensure_session_factory()
        async with factory() as session:
            session.add(
                TestResult(
                    **card_context,
                    test_id=test_id,
                    container_name=container_name,
                    container_id=container_id,
                    test_name=test_name,
                    language=language,
                    test_type=test_type,
                    test_framework=test_framework,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=stderr,
                    execution_time_ms=execution_time_ms,
                    success=success,
                    source_code=source_code,
                    test_code=test_code,
                    file_path=file_path,
                    dependencies=dependencies if dependencies else None,
                    resource_usage=resource_usage if resource_usage else None,
                    timestamp=timestamp or datetime.now(UTC),
                )
            )
            await session.commit()
        logger.info(f"Stored test result: test_id={test_id}, container={container_name}")

    async def store_computed_test_result(
        self,
        *,
        test_id: str,
        container_name: str,
        tenant_id: str,
        test_name: str,
        language: str,
        test_type: str,
        exit_code: int,
        stdout: str,
        stderr: str,
        execution_time_ms: float,
        success: bool,
        test_framework: str | None = None,
        resource_usage: dict[str, Any] | None = None,
        card_id: str | None = None,
        project_id: str | None = None,
        commit_sha: str | None = None,
        correlation_id: str | None = None,
        source: str = "aa_qa",
    ) -> tuple[dict[str, Any], bool]:
        """Idempotently store an externally computed result.

        ``INSERT ... ON CONFLICT (test_id) DO NOTHING``: a repeat with the same
        ``test_id`` writes nothing and returns the row already stored, so a
        retried producer can never create a duplicate. Returns ``(row, created)``.

        Raises:
            TestIdConflictError: the ``test_id`` already exists under a different
                container or tenant. The existing row is NOT returned — handing it
                back would disclose another scope's result to this caller.
        """
        values: dict[str, Any] = {
            "test_id": test_id,
            "container_name": container_name,
            "container_id": None,
            "tenant_id": tenant_id,
            "test_name": test_name,
            "language": language,
            "test_type": test_type,
            "test_framework": test_framework,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "execution_time_ms": execution_time_ms,
            "success": success,
            "resource_usage": resource_usage if resource_usage else None,
            "timestamp": datetime.now(UTC),
            "card_id": card_id,
            "project_id": project_id,
            "commit_sha": commit_sha,
            "correlation_id": correlation_id,
            "source": source,
        }
        factory = await ensure_session_factory()
        async with factory() as session:
            inserted_id = (
                await session.execute(
                    pg_insert(TestResult)
                    .values(**values)
                    .on_conflict_do_nothing(index_elements=[TestResult.test_id])
                    .returning(TestResult.id)
                )
            ).scalar_one_or_none()
            await session.commit()
            row = (
                await session.execute(select(TestResult).where(TestResult.test_id == test_id))
            ).scalar_one()

        if row.container_name != container_name or row.tenant_id != tenant_id:
            raise TestIdConflictError(test_id)
        created = inserted_id is not None
        logger.info(
            f"Stored computed test result: test_id={test_id}, container={container_name}, "
            f"created={created}"
        )
        return _row_to_dict(row), created

    async def get_test_result(self, container_name: str, test_id: str) -> dict[str, Any] | None:
        """Retrieve a specific test result (scoped to the container)."""
        factory = await ensure_session_factory()
        async with factory() as session:
            row = (
                await session.execute(
                    select(TestResult).where(
                        TestResult.test_id == test_id,
                        TestResult.container_name == container_name,
                    )
                )
            ).scalar_one_or_none()

            if row is None:
                return None
            return _row_to_dict(row)

    async def get_test_history(
        self,
        container_name: str,
        test_name: str | None = None,
        language: str | None = None,
        success: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Retrieve historical test results with filtering + pagination."""
        factory = await ensure_session_factory()

        conditions = [TestResult.container_name == container_name]
        if test_name is not None:
            conditions.append(TestResult.test_name == test_name)
        if language is not None:
            conditions.append(TestResult.language == language)
        if success is not None:
            conditions.append(TestResult.success == success)

        async with factory() as session:
            total_count = (
                await session.execute(
                    select(func.count()).select_from(TestResult).where(*conditions)
                )
            ).scalar_one()

            rows = (
                (
                    await session.execute(
                        select(TestResult)
                        .where(*conditions)
                        .order_by(TestResult.timestamp.desc())
                        .limit(limit)
                        .offset(offset)
                    )
                )
                .scalars()
                .all()
            )

        test_results = [
            {
                "test_id": row.test_id,
                "test_name": row.test_name,
                "language": row.language,
                "test_type": row.test_type,
                "exit_code": row.exit_code,
                "success": bool(row.success),
                "execution_time_ms": row.execution_time_ms,
                "timestamp": _iso(row.timestamp),
            }
            for row in rows
        ]

        return {
            "container_name": container_name,
            "total_tests": total_count,
            "returned_tests": len(test_results),
            "test_results": test_results,
            "filters_applied": {
                "test_name": test_name,
                "language": language,
                "success": success,
            },
            "query_timestamp": datetime.now(UTC).isoformat(),
        }

    async def store_analysis_result(
        self,
        container_name: str,
        test_id: str,
        analysis_id: str,
        root_causes: list[dict[str, Any]],
        fix_suggestions: list[dict[str, Any]],
        detected_patterns: list[dict[str, Any]],
        analysis_metadata: dict[str, Any],
    ) -> None:
        """Store an AI analysis result for a test (test_id must exist in container)."""
        factory = await ensure_session_factory()
        async with factory() as session:
            exists = (
                await session.execute(
                    select(TestResult.test_id).where(
                        TestResult.test_id == test_id,
                        TestResult.container_name == container_name,
                    )
                )
            ).scalar_one_or_none()
            if exists is None:
                raise ValueError(f"Test {test_id} not found in container {container_name}")

            session.add(
                AnalysisResult(
                    analysis_id=analysis_id,
                    test_id=test_id,
                    container_name=container_name,
                    root_causes=root_causes,
                    fix_suggestions=fix_suggestions,
                    detected_patterns=detected_patterns,
                    analysis_metadata=analysis_metadata,
                )
            )
            await session.commit()
        logger.info(f"Stored analysis result: analysis_id={analysis_id}, test_id={test_id}")


# Global singleton instance
_storage_service: TestResultsStorageService | None = None


def get_test_results_storage_service() -> TestResultsStorageService:
    """Get or create global test results storage service instance."""
    global _storage_service
    if _storage_service is None:
        _storage_service = TestResultsStorageService()
    return _storage_service
