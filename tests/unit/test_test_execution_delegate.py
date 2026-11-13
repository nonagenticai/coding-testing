"""Tests for services/test_execution_delegate.py."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from models.testing_models import TestSecurityLevel
from services.test_execution_delegate import TestExecutionDelegate


def _make_test_request(**overrides):
    """Create a mock TestRequest."""
    req = MagicMock()
    req.code = overrides.get("code", "def add(a,b): return a+b")
    req.test_code = overrides.get("test_code", "def test_add(): assert add(1,2)==3")
    req.language = overrides.get("language", "python")
    req.security_level = overrides.get("security_level", TestSecurityLevel.MEDIUM)
    req.container_name = overrides.get("container_name", "test-container")
    req.test_type = overrides.get("test_type", MagicMock(value="unit"))
    req.timeout = overrides.get("timeout", 30)
    req.dependencies = overrides.get("dependencies", {})
    req.resource_limits = overrides.get("resource_limits", None)
    req.run_quality_check = overrides.get("run_quality_check", False)
    req.quality_tools = overrides.get("quality_tools", None)
    return req


class TestTestExecutionDelegateInit:
    """Test delegate initialization."""

    def test_init(self):
        orch = MagicMock()
        cache = MagicMock()
        delegate = TestExecutionDelegate(orch, cache)
        assert delegate.test_orchestrator is orch
        assert delegate.cache_manager is cache

    def test_init_without_cache(self):
        orch = MagicMock()
        delegate = TestExecutionDelegate(orch, None)
        assert delegate.cache_manager is None


class TestGenerateCacheKey:
    """Test cache key generation."""

    def test_generates_deterministic_key(self):
        orch = MagicMock()
        delegate = TestExecutionDelegate(orch, None)
        request = _make_test_request()

        key1 = delegate.generate_cache_key(request)
        key2 = delegate.generate_cache_key(request)
        assert key1 == key2
        assert key1.startswith("test_")

    def test_different_code_different_key(self):
        orch = MagicMock()
        delegate = TestExecutionDelegate(orch, None)
        req1 = _make_test_request(code="code_a")
        req2 = _make_test_request(code="code_b")

        assert delegate.generate_cache_key(req1) != delegate.generate_cache_key(req2)

    def test_none_code_handled(self):
        orch = MagicMock()
        delegate = TestExecutionDelegate(orch, None)
        req = _make_test_request(code=None, test_code=None)

        key = delegate.generate_cache_key(req)
        assert key.startswith("test_")


class TestGetCachedResult:
    """Test cached result retrieval."""

    @pytest.mark.asyncio
    async def test_returns_none_without_cache_manager(self):
        delegate = TestExecutionDelegate(MagicMock(), None)
        result = await delegate.get_cached_result("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_cache_miss(self):
        cache = AsyncMock()
        cache.get = AsyncMock(return_value=None)
        delegate = TestExecutionDelegate(MagicMock(), cache)
        result = await delegate.get_cached_result("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_result_on_cache_hit(self):
        cached_data = {
            "test_id": "t1",
            "success": True,
            "stdout": "ok",
            "stderr": "",
            "exit_code": 0,
            "execution_time": 1.5,
            "created_at": datetime.now(UTC).isoformat(),
            "language": "python",
            "security_level": "medium",
        }
        cache = AsyncMock()
        cache.get = AsyncMock(return_value=cached_data)
        delegate = TestExecutionDelegate(MagicMock(), cache)

        result = await delegate.get_cached_result("key")
        assert result is not None
        assert result.test_id == "t1"
        assert result.success is True

    @pytest.mark.asyncio
    async def test_returns_none_on_deserialize_error(self):
        """Malformed cached data returns None."""
        cache = AsyncMock()
        cache.get = AsyncMock(return_value={"bad": "data"})
        delegate = TestExecutionDelegate(MagicMock(), cache)

        result = await delegate.get_cached_result("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_cache_error(self):
        """Cache retrieval errors return None."""
        cache = AsyncMock()
        cache.get = AsyncMock(side_effect=RuntimeError("redis down"))
        delegate = TestExecutionDelegate(MagicMock(), cache)

        result = await delegate.get_cached_result("key")
        assert result is None


class TestCacheResult:
    """Test result caching."""

    @pytest.mark.asyncio
    async def test_noop_without_cache_manager(self):
        delegate = TestExecutionDelegate(MagicMock(), None)
        mock_result = MagicMock()
        mock_result.model_dump = MagicMock(return_value={})
        await delegate.cache_result("key", mock_result)  # should not raise

    @pytest.mark.asyncio
    async def test_caches_with_ttl(self):
        cache = AsyncMock()
        delegate = TestExecutionDelegate(MagicMock(), cache)
        mock_result = MagicMock()
        mock_result.model_dump = MagicMock(return_value={"test_id": "t1"})

        await delegate.cache_result("key", mock_result)
        cache.set.assert_called_once_with("key", {"test_id": "t1"}, ttl=3600)

    @pytest.mark.asyncio
    async def test_cache_error_swallowed(self):
        """Cache storage errors are swallowed."""
        cache = AsyncMock()
        cache.set = AsyncMock(side_effect=OSError("disk full"))
        delegate = TestExecutionDelegate(MagicMock(), cache)
        mock_result = MagicMock()
        mock_result.model_dump = MagicMock(return_value={})

        await delegate.cache_result("key", mock_result)  # should not raise


class TestExecuteWithOrchestrator:
    """Test execute_with_orchestrator."""

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        exec_result = MagicMock()
        exec_result.test_id = "t1"
        exec_result.success = True
        exec_result.exit_code = 0
        exec_result.stdout = "All tests passed"
        exec_result.stderr = ""
        exec_result.execution_time_ms = 1500.0
        exec_result.created_at = datetime.now(UTC)
        exec_result.container_name = "test-container"

        orch = AsyncMock()
        orch.execute_tests = AsyncMock(return_value=exec_result)

        delegate = TestExecutionDelegate(orch, None)
        request = _make_test_request()

        result = await delegate.execute_with_orchestrator(request)

        assert result.success is True
        assert result.test_id == "t1"
        assert result.exit_code == 0
        assert result.error is None

    @pytest.mark.asyncio
    async def test_failed_execution(self):
        exec_result = MagicMock()
        exec_result.test_id = "t2"
        exec_result.success = False
        exec_result.exit_code = 1
        exec_result.stdout = ""
        exec_result.stderr = "assertion failed"
        exec_result.execution_time_ms = 500.0
        exec_result.created_at = datetime.now(UTC)
        exec_result.container_name = "test-container"

        orch = AsyncMock()
        orch.execute_tests = AsyncMock(return_value=exec_result)

        delegate = TestExecutionDelegate(orch, None)
        request = _make_test_request()

        result = await delegate.execute_with_orchestrator(request)

        assert result.success is False
        assert result.error == "assertion failed"

    @pytest.mark.asyncio
    async def test_orchestrator_exception_returns_failed_result(self):
        orch = AsyncMock()
        orch.execute_tests = AsyncMock(side_effect=RuntimeError("container crash"))

        delegate = TestExecutionDelegate(orch, None)
        request = _make_test_request()

        result = await delegate.execute_with_orchestrator(request)

        assert result.success is False
        assert result.exit_code == 1
        assert "container crash" in result.error

    @pytest.mark.asyncio
    async def test_with_quality_check(self):
        from models.testing_models import QualityResult

        exec_result = MagicMock()
        exec_result.test_id = "t3"
        exec_result.success = True
        exec_result.exit_code = 0
        exec_result.stdout = "ok"
        exec_result.stderr = ""
        exec_result.execution_time_ms = 200.0
        exec_result.created_at = datetime.now(UTC)
        exec_result.container_name = "test-container"

        orch = AsyncMock()
        orch.execute_tests = AsyncMock(return_value=exec_result)

        quality_report = QualityResult(
            analysis_id="qa-1",
            success=True,
            overall_score=90.0,
            total_issues=0,
            execution_time=0.5,
            created_at=datetime.now(UTC),
            language="python",
        )
        quality_analyzer = AsyncMock()
        quality_analyzer.analyze_quality = AsyncMock(return_value=quality_report)

        delegate = TestExecutionDelegate(orch, None)
        request = _make_test_request(run_quality_check=True)

        result = await delegate.execute_with_orchestrator(request, quality_analyzer)

        assert result.success is True
        assert result.quality_report is not None
        assert result.quality_report.overall_score == 90.0

    @pytest.mark.asyncio
    async def test_quality_check_failure_swallowed(self):
        exec_result = MagicMock()
        exec_result.test_id = "t4"
        exec_result.success = True
        exec_result.exit_code = 0
        exec_result.stdout = "ok"
        exec_result.stderr = ""
        exec_result.execution_time_ms = 100.0
        exec_result.created_at = datetime.now(UTC)
        exec_result.container_name = "test-container"

        orch = AsyncMock()
        orch.execute_tests = AsyncMock(return_value=exec_result)

        quality_analyzer = AsyncMock()
        quality_analyzer.analyze_quality = AsyncMock(side_effect=RuntimeError("quality tool down"))

        delegate = TestExecutionDelegate(orch, None)
        request = _make_test_request(run_quality_check=True)

        result = await delegate.execute_with_orchestrator(request, quality_analyzer)
        assert result.success is True
        assert result.quality_report is None

    @pytest.mark.asyncio
    async def test_no_quality_check_when_test_fails(self):
        exec_result = MagicMock()
        exec_result.test_id = "t5"
        exec_result.success = False
        exec_result.exit_code = 1
        exec_result.stdout = ""
        exec_result.stderr = "fail"
        exec_result.execution_time_ms = 100.0
        exec_result.created_at = datetime.now(UTC)
        exec_result.container_name = "test-container"

        orch = AsyncMock()
        orch.execute_tests = AsyncMock(return_value=exec_result)

        quality_analyzer = AsyncMock()

        delegate = TestExecutionDelegate(orch, None)
        request = _make_test_request(run_quality_check=True)

        await delegate.execute_with_orchestrator(request, quality_analyzer)
        # Quality check not called because test failed
        quality_analyzer.analyze_quality.assert_not_called()

    @pytest.mark.asyncio
    async def test_none_test_type_defaults_to_unit(self):
        exec_result = MagicMock()
        exec_result.test_id = "t6"
        exec_result.success = True
        exec_result.exit_code = 0
        exec_result.stdout = ""
        exec_result.stderr = ""
        exec_result.execution_time_ms = 50.0
        exec_result.created_at = datetime.now(UTC)
        exec_result.container_name = "test-container"

        orch = AsyncMock()
        orch.execute_tests = AsyncMock(return_value=exec_result)

        delegate = TestExecutionDelegate(orch, None)
        request = _make_test_request(test_type=None)

        result = await delegate.execute_with_orchestrator(request)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_timeout_error_returns_failed_result(self):
        orch = AsyncMock()
        orch.execute_tests = AsyncMock(side_effect=TimeoutError("timed out"))

        delegate = TestExecutionDelegate(orch, None)
        request = _make_test_request()

        result = await delegate.execute_with_orchestrator(request)
        assert result.success is False
        assert "timed out" in result.error
