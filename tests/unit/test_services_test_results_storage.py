"""Unit tests for the Postgres-backed test results storage service.

These exercise the real store against Postgres (via the ``pg_db`` fixture,
which truncates the tables before each test). They skip automatically when no
Postgres DATABASE_URL is configured.
"""

from sqlalchemy import text


class TestTestResultsStorageService:
    """Test TestResultsStorageService against Postgres."""

    async def test_store_and_retrieve_result(self, pg_db):
        """Test storing and retrieving a test result with all fields verified."""
        from services.test_results_storage_service import TestResultsStorageService

        service = TestResultsStorageService()

        await service.store_test_result(
            test_id="test-001",
            container_name="my-container",
            container_id="docker-abc",
            test_name="test_example",
            language="python",
            test_type="unit",
            exit_code=0,
            stdout="All tests passed",
            stderr="",
            execution_time_ms=150.5,
            success=True,
        )

        result = await service.get_test_result("my-container", "test-001")
        assert result["test_id"] == "test-001"
        assert result["container_name"] == "my-container"
        assert result["container_id"] == "docker-abc"
        assert result["test_name"] == "test_example"
        assert result["language"] == "python"
        assert result["test_type"] == "unit"
        assert result["exit_code"] == 0
        assert result["stdout"] == "All tests passed"
        assert result["stderr"] == ""
        assert result["execution_time_ms"] == 150.5
        assert result["success"] is True
        assert result["source_code"] is None
        assert result["test_code"] is None
        assert result["file_path"] is None
        assert result["dependencies"] is None
        assert result["resource_usage"] is None
        assert result["test_framework"] is None
        # timestamp is rendered as an ISO string
        assert "T" in result["timestamp"]

    async def test_result_scoped_by_container(self, pg_db):
        """A result is only visible from within its own container scope."""
        from services.test_results_storage_service import TestResultsStorageService

        service = TestResultsStorageService()
        await service.store_test_result(
            test_id="scoped-1",
            container_name="container-a",
            container_id=None,
            test_name="test_scope",
            language="python",
            test_type="unit",
            exit_code=0,
            stdout="ok",
            stderr="",
            execution_time_ms=10.0,
            success=True,
        )
        # Wrong container scope → not found.
        assert await service.get_test_result("container-b", "scoped-1") is None
        # Correct scope → found.
        assert (await service.get_test_result("container-a", "scoped-1"))["test_id"] == "scoped-1"

    async def test_get_nonexistent_result(self, pg_db):
        """Test getting a result that doesn't exist returns None."""
        from services.test_results_storage_service import TestResultsStorageService

        service = TestResultsStorageService()
        result = await service.get_test_result("no-container", "no-test")
        assert result is None

    async def test_get_test_history_empty(self, pg_db):
        """Test getting history with no results returns correct empty structure."""
        from services.test_results_storage_service import TestResultsStorageService

        service = TestResultsStorageService()
        history = await service.get_test_history("no-container")
        assert history["container_name"] == "no-container"
        assert history["total_tests"] == 0
        assert history["returned_tests"] == 0
        assert history["test_results"] == []
        assert history["filters_applied"] == {
            "test_name": None,
            "language": None,
            "success": None,
        }
        assert "T" in history["query_timestamp"]

    async def test_get_test_history_with_results(self, pg_db):
        """Test getting history with stored results returns correct data."""
        from services.test_results_storage_service import TestResultsStorageService

        service = TestResultsStorageService()

        for i in range(3):
            await service.store_test_result(
                test_id=f"test-{i:03d}",
                container_name="my-container",
                container_id=None,
                test_name=f"test_{i}",
                language="python",
                test_type="unit",
                exit_code=0 if i < 2 else 1,
                stdout="ok" if i < 2 else "fail",
                stderr="" if i < 2 else "error",
                execution_time_ms=100.0 + i,
                success=i < 2,
            )

        history = await service.get_test_history("my-container")
        assert history["container_name"] == "my-container"
        assert history["total_tests"] == 3
        assert history["returned_tests"] == 3
        assert len(history["test_results"]) == 3

        returned_ids = {r["test_id"] for r in history["test_results"]}
        assert returned_ids == {"test-000", "test-001", "test-002"}

        fail_result = next(r for r in history["test_results"] if r["test_id"] == "test-002")
        assert fail_result["test_name"] == "test_2"
        assert fail_result["language"] == "python"
        assert fail_result["test_type"] == "unit"
        assert fail_result["exit_code"] == 1
        assert fail_result["success"] is False
        assert fail_result["execution_time_ms"] == 102.0

        pass_result = next(r for r in history["test_results"] if r["test_id"] == "test-000")
        assert pass_result["success"] is True
        assert pass_result["exit_code"] == 0
        assert pass_result["execution_time_ms"] == 100.0

    async def test_get_test_history_with_filter(self, pg_db):
        """Test history filtering by success status returns only matching results."""
        from services.test_results_storage_service import TestResultsStorageService

        service = TestResultsStorageService()

        await service.store_test_result(
            test_id="pass-1",
            container_name="c1",
            container_id=None,
            test_name="test_pass",
            language="python",
            test_type="unit",
            exit_code=0,
            stdout="ok",
            stderr="",
            execution_time_ms=50.0,
            success=True,
        )
        await service.store_test_result(
            test_id="fail-1",
            container_name="c1",
            container_id=None,
            test_name="test_fail",
            language="python",
            test_type="unit",
            exit_code=1,
            stdout="",
            stderr="error",
            execution_time_ms=60.0,
            success=False,
        )

        history = await service.get_test_history("c1", success=True)
        assert history["total_tests"] == 1
        assert history["returned_tests"] == 1
        assert len(history["test_results"]) == 1
        assert history["test_results"][0]["test_id"] == "pass-1"
        assert history["test_results"][0]["test_name"] == "test_pass"
        assert history["test_results"][0]["success"] is True
        assert history["filters_applied"]["success"] is True

    async def test_get_test_history_filter_by_language(self, pg_db):
        """Test history filtering by language returns only matching results."""
        from services.test_results_storage_service import TestResultsStorageService

        service = TestResultsStorageService()

        await service.store_test_result(
            test_id="py-1",
            container_name="c1",
            container_id=None,
            test_name="test_py",
            language="python",
            test_type="unit",
            exit_code=0,
            stdout="ok",
            stderr="",
            execution_time_ms=50.0,
            success=True,
        )
        await service.store_test_result(
            test_id="js-1",
            container_name="c1",
            container_id=None,
            test_name="test_js",
            language="javascript",
            test_type="unit",
            exit_code=0,
            stdout="ok",
            stderr="",
            execution_time_ms=60.0,
            success=True,
        )

        history = await service.get_test_history("c1", language="python")
        assert history["total_tests"] == 1
        assert history["returned_tests"] == 1
        assert len(history["test_results"]) == 1
        assert history["test_results"][0]["test_id"] == "py-1"
        assert history["test_results"][0]["language"] == "python"
        assert history["test_results"][0]["test_name"] == "test_py"
        assert history["filters_applied"]["language"] == "python"

    async def test_get_test_history_filter_by_test_name(self, pg_db):
        """Test history filtering by test_name returns only matching results."""
        from services.test_results_storage_service import TestResultsStorageService

        service = TestResultsStorageService()

        await service.store_test_result(
            test_id="t-a",
            container_name="c1",
            container_id=None,
            test_name="test_alpha",
            language="python",
            test_type="unit",
            exit_code=0,
            stdout="ok",
            stderr="",
            execution_time_ms=50.0,
            success=True,
        )
        await service.store_test_result(
            test_id="t-b",
            container_name="c1",
            container_id=None,
            test_name="test_beta",
            language="python",
            test_type="unit",
            exit_code=0,
            stdout="ok",
            stderr="",
            execution_time_ms=60.0,
            success=True,
        )

        history = await service.get_test_history("c1", test_name="test_alpha")
        assert history["total_tests"] == 1
        assert history["returned_tests"] == 1
        assert len(history["test_results"]) == 1
        assert history["test_results"][0]["test_id"] == "t-a"
        assert history["test_results"][0]["test_name"] == "test_alpha"
        assert history["filters_applied"]["test_name"] == "test_alpha"

    async def test_get_test_history_filter_failed(self, pg_db):
        """Test history filtering by success=False returns only failed results."""
        from services.test_results_storage_service import TestResultsStorageService

        service = TestResultsStorageService()

        await service.store_test_result(
            test_id="pass-1",
            container_name="c2",
            container_id=None,
            test_name="test_pass",
            language="python",
            test_type="unit",
            exit_code=0,
            stdout="ok",
            stderr="",
            execution_time_ms=50.0,
            success=True,
        )
        await service.store_test_result(
            test_id="fail-1",
            container_name="c2",
            container_id=None,
            test_name="test_fail",
            language="python",
            test_type="unit",
            exit_code=1,
            stdout="",
            stderr="error",
            execution_time_ms=60.0,
            success=False,
        )

        history = await service.get_test_history("c2", success=False)
        assert history["total_tests"] == 1
        assert history["returned_tests"] == 1
        assert len(history["test_results"]) == 1
        assert history["test_results"][0]["test_id"] == "fail-1"
        assert history["test_results"][0]["test_name"] == "test_fail"
        assert history["test_results"][0]["success"] is False
        assert history["test_results"][0]["exit_code"] == 1
        assert history["filters_applied"]["success"] is False

    async def test_get_test_history_pagination(self, pg_db):
        """Test history pagination with limit and offset returns correct pages."""
        from services.test_results_storage_service import TestResultsStorageService

        service = TestResultsStorageService()

        for i in range(5):
            await service.store_test_result(
                test_id=f"t-{i}",
                container_name="c3",
                container_id=None,
                test_name=f"test_{i}",
                language="python",
                test_type="unit",
                exit_code=0,
                stdout="ok",
                stderr="",
                execution_time_ms=10.0 * i,
                success=True,
            )

        history = await service.get_test_history("c3", limit=2, offset=0)
        assert history["total_tests"] == 5
        assert history["returned_tests"] == 2
        assert len(history["test_results"]) == 2
        page1_ids = {r["test_id"] for r in history["test_results"]}

        history2 = await service.get_test_history("c3", limit=2, offset=2)
        assert history2["total_tests"] == 5
        assert history2["returned_tests"] == 2
        assert len(history2["test_results"]) == 2
        page2_ids = {r["test_id"] for r in history2["test_results"]}

        history3 = await service.get_test_history("c3", limit=2, offset=4)
        assert history3["total_tests"] == 5
        assert history3["returned_tests"] == 1
        assert len(history3["test_results"]) == 1
        page3_ids = {r["test_id"] for r in history3["test_results"]}

        all_ids = page1_ids | page2_ids | page3_ids
        assert all_ids == {"t-0", "t-1", "t-2", "t-3", "t-4"}
        assert len(page1_ids & page2_ids) == 0
        assert len(page2_ids & page3_ids) == 0

    async def test_store_with_dependencies_and_resources(self, pg_db):
        """Test storing result with dependencies and resource_usage JSON fields."""
        from services.test_results_storage_service import TestResultsStorageService

        service = TestResultsStorageService()

        deps = {"pytest": "^7.0", "requests": "^2.28"}
        resources = {"cpu_percent": 45.0, "memory_mb": 256.0}

        await service.store_test_result(
            test_id="t-deps",
            container_name="c4",
            container_id="docker-xyz",
            test_name="test_with_deps",
            language="python",
            test_type="unit",
            exit_code=0,
            stdout="ok",
            stderr="",
            execution_time_ms=150.0,
            success=True,
            source_code="def foo(): pass",
            test_code="def test_foo(): assert True",
            file_path="/workspace/test_foo.py",
            dependencies=deps,
            resource_usage=resources,
            test_framework="pytest",
        )

        result = await service.get_test_result("c4", "t-deps")
        assert result["test_id"] == "t-deps"
        assert result["container_id"] == "docker-xyz"
        assert result["test_name"] == "test_with_deps"
        assert result["dependencies"] == deps
        assert result["resource_usage"] == resources
        assert result["source_code"] == "def foo(): pass"
        assert result["test_code"] == "def test_foo(): assert True"
        assert result["file_path"] == "/workspace/test_foo.py"
        assert result["test_framework"] == "pytest"
        assert result["execution_time_ms"] == 150.0
        assert result["success"] is True
        assert result["exit_code"] == 0

    async def test_store_analysis_result(self, pg_db):
        """Test storing an analysis result and verify it persists in the database."""
        from services.test_results_storage_service import TestResultsStorageService

        service = TestResultsStorageService()

        await service.store_test_result(
            test_id="t-for-analysis",
            container_name="c5",
            container_id=None,
            test_name="test_analyzed",
            language="python",
            test_type="unit",
            exit_code=1,
            stdout="",
            stderr="AssertionError",
            execution_time_ms=100.0,
            success=False,
        )

        root_causes = [{"cause": "wrong assertion"}]
        fix_suggestions = [{"fix": "use assertEqual"}]
        detected_patterns = [{"pattern": "assertion_error"}]
        analysis_meta = {"model": "gpt-4", "duration_ms": 500}

        await service.store_analysis_result(
            container_name="c5",
            test_id="t-for-analysis",
            analysis_id="a-001",
            root_causes=root_causes,
            fix_suggestions=fix_suggestions,
            detected_patterns=detected_patterns,
            analysis_metadata=analysis_meta,
        )

        # Verify the analysis was actually persisted by querying Postgres directly.
        # jsonb columns round-trip as native Python objects (no json.loads needed).
        async with pg_db() as session:
            row = (
                (
                    await session.execute(
                        text("SELECT * FROM analysis_results WHERE analysis_id = :aid"),
                        {"aid": "a-001"},
                    )
                )
                .mappings()
                .one()
            )

        assert row["analysis_id"] == "a-001"
        assert row["test_id"] == "t-for-analysis"
        assert row["container_name"] == "c5"
        assert row["root_causes"] == root_causes
        assert row["fix_suggestions"] == fix_suggestions
        assert row["detected_patterns"] == detected_patterns
        assert row["analysis_metadata"] == analysis_meta

    async def test_store_analysis_result_missing_test_id(self, pg_db):
        """Test storing analysis result for nonexistent test raises ValueError."""
        import pytest as _pytest

        from services.test_results_storage_service import TestResultsStorageService

        service = TestResultsStorageService()

        with _pytest.raises(ValueError, match="not found"):
            await service.store_analysis_result(
                container_name="c6",
                test_id="nonexistent-test",
                analysis_id="a-bad",
                root_causes=[],
                fix_suggestions=[],
                detected_patterns=[],
                analysis_metadata={},
            )

    async def test_get_test_history_has_filters_applied(self, pg_db):
        """Test history response includes filters_applied and query_timestamp metadata."""
        from services.test_results_storage_service import TestResultsStorageService

        service = TestResultsStorageService()
        history = await service.get_test_history("empty-container")
        assert history["filters_applied"] == {
            "test_name": None,
            "language": None,
            "success": None,
        }
        assert "T" in history["query_timestamp"]
        assert history["container_name"] == "empty-container"


class TestGetStorageServiceSingleton:
    """Test the singleton getter."""

    def test_returns_instance(self):
        """Test getter returns a TestResultsStorageService instance and is a singleton."""
        import services.test_results_storage_service as mod

        old = mod._storage_service
        mod._storage_service = None

        try:
            service = mod.get_test_results_storage_service()
            assert isinstance(service, mod.TestResultsStorageService)
            # Second call returns same instance (singleton)
            service2 = mod.get_test_results_storage_service()
            assert service2 is service
        finally:
            mod._storage_service = old


class TestCardContextColumns:
    """Migration 0003 columns: optional on store_test_result, idempotent computed writes."""

    async def test_existing_callers_get_execute_source_and_null_card_columns(self, pg_db):
        from services.test_results_storage_service import TestResultsStorageService

        service = TestResultsStorageService()
        await service.store_test_result(
            test_id="exec-1",
            container_name="c",
            container_id=None,
            test_name="t",
            language="python",
            test_type="unit",
            exit_code=0,
            stdout="",
            stderr="",
            execution_time_ms=1.0,
            success=True,
        )
        row = await service.get_test_result("c", "exec-1")
        assert row["source"] == "execute"
        assert row["card_id"] is None
        assert row["tenant_id"] is None

    async def test_store_test_result_accepts_card_kwargs(self, pg_db):
        from services.test_results_storage_service import TestResultsStorageService

        service = TestResultsStorageService()
        await service.store_test_result(
            test_id="card-1",
            container_name="c",
            container_id=None,
            test_name="t",
            language="python",
            test_type="unit",
            exit_code=1,
            stdout="",
            stderr="",
            execution_time_ms=1.0,
            success=False,
            card_id="k1",
            tenant_id="tenant-a",
            project_id="p1",
            commit_sha="deadbeef",
            correlation_id="corr",
            source="aa_qa",
        )
        row = await service.get_test_result("c", "card-1")
        assert (row["card_id"], row["tenant_id"], row["project_id"]) == ("k1", "tenant-a", "p1")
        assert (row["commit_sha"], row["correlation_id"], row["source"]) == (
            "deadbeef",
            "corr",
            "aa_qa",
        )

    async def test_computed_store_is_idempotent_and_scope_checked(self, pg_db):
        import pytest

        from services.test_results_storage_service import (
            TestIdConflictError,
            TestResultsStorageService,
        )

        service = TestResultsStorageService()
        kwargs = dict(
            test_id="aaqa:k:c",
            container_name="c",
            tenant_id="tenant-a",
            test_name="r3.qa.coding",
            language="python",
            test_type="unit",
            exit_code=0,
            stdout="ok",
            stderr="",
            execution_time_ms=5.0,
            success=True,
            card_id="k",
        )
        row, created = await service.store_computed_test_result(**kwargs)
        assert created is True and row["source"] == "aa_qa"
        row2, created2 = await service.store_computed_test_result(**{**kwargs, "stdout": "x"})
        assert created2 is False and row2["stdout"] == "ok" and row2["id"] == row["id"]
        with pytest.raises(TestIdConflictError):
            await service.store_computed_test_result(**{**kwargs, "tenant_id": "tenant-b"})
        async with pg_db() as session:
            count = (await session.execute(text("SELECT count(*) FROM test_results"))).scalar()
        assert count == 1
