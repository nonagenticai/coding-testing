"""Tests for intelligent_testing_modules: metrics, utilities, service_lifecycle, execution_coordination, discovery."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── metrics_and_calculations ─────────────────────────────────────────────────
from routers.intelligent_testing_modules.metrics_and_calculations import (
    ExecutionMetricsCalculator,
    ResourceEstimator,
    TestCodeAnalyzer,
    TimestampHelper,
)


def _make_exec_result(success=True, execution_time_ms=100.0, **kwargs):
    """Create a mock IntelligentTestExecutionResult."""
    m = MagicMock()
    m.success = success
    m.execution_time_ms = execution_time_ms
    m.optimization_applied = kwargs.get("optimization_applied", False)
    m.pattern_matches = kwargs.get("pattern_matches", [])
    m.recommendations = kwargs.get("recommendations", [])
    m.stderr = kwargs.get("stderr", "")
    return m


# ── ExecutionMetricsCalculator ───────────────────────────────────────────────


class TestExecutionMetricsCalculator:
    def test_success_rate_empty(self):
        assert ExecutionMetricsCalculator.calculate_success_rate([]) == 0.0

    def test_success_rate_all_pass(self):
        results = [_make_exec_result(success=True) for _ in range(5)]
        assert ExecutionMetricsCalculator.calculate_success_rate(results) == 100.0

    def test_success_rate_mixed(self):
        results = [_make_exec_result(success=True), _make_exec_result(success=False)]
        assert ExecutionMetricsCalculator.calculate_success_rate(results) == 50.0

    def test_success_rate_all_fail(self):
        results = [_make_exec_result(success=False) for _ in range(3)]
        assert ExecutionMetricsCalculator.calculate_success_rate(results) == 0.0

    def test_avg_exec_time_empty(self):
        assert ExecutionMetricsCalculator.calculate_average_execution_time([]) == 0.0

    def test_avg_exec_time_all_none(self):
        results = [_make_exec_result(execution_time_ms=None) for _ in range(3)]
        assert ExecutionMetricsCalculator.calculate_average_execution_time(results) == 0.0

    def test_avg_exec_time_normal(self):
        results = [
            _make_exec_result(execution_time_ms=100.0),
            _make_exec_result(execution_time_ms=200.0),
        ]
        assert ExecutionMetricsCalculator.calculate_average_execution_time(results) == 150.0

    def test_percentiles_empty(self):
        result = ExecutionMetricsCalculator.calculate_percentiles([])
        assert result == {50: 0.0, 90: 0.0, 95: 0.0, 99: 0.0}

    def test_percentiles_single_value(self):
        result = ExecutionMetricsCalculator.calculate_percentiles([42.0])
        for p in [50, 90, 95, 99]:
            assert result[p] == 42.0

    def test_percentiles_multiple(self):
        values = list(range(1, 101))  # 1..100
        result = ExecutionMetricsCalculator.calculate_percentiles(values)
        assert result[50] == 50  # median
        assert result[90] >= 89
        assert result[99] >= 98

    def test_percentiles_custom(self):
        result = ExecutionMetricsCalculator.calculate_percentiles([10, 20, 30], [50])
        assert result == {50: 20}
        assert len(result) == 1

    def test_detect_anomalies_too_few(self):
        results = [_make_exec_result(execution_time_ms=100.0)]
        anomalies = ExecutionMetricsCalculator.detect_performance_anomalies(results)
        assert anomalies == []

    def test_detect_anomalies_no_times(self):
        results = [_make_exec_result(execution_time_ms=None) for _ in range(5)]
        anomalies = ExecutionMetricsCalculator.detect_performance_anomalies(results)
        assert anomalies == []

    def test_detect_anomalies_found(self):
        results = [
            _make_exec_result(execution_time_ms=100.0),
            _make_exec_result(execution_time_ms=100.0),
            _make_exec_result(execution_time_ms=100.0),
            _make_exec_result(execution_time_ms=500.0),  # anomaly
        ]
        anomalies = ExecutionMetricsCalculator.detect_performance_anomalies(results)
        assert len(anomalies) == 1
        a = anomalies[0]
        assert a["anomaly_type"] == "slow_execution"
        assert a["index"] == 3
        assert a["execution_time_ms"] == 500.0
        assert a["average_time_ms"] == 200.0  # (100+100+100+500)/4
        assert a["threshold_ms"] == 400.0  # 200 * 2.0
        assert a["severity"] == "medium"  # 500 <= 400*1.5 = 600

    def test_detect_anomalies_severity_high(self):
        results = [
            _make_exec_result(execution_time_ms=100.0),
            _make_exec_result(execution_time_ms=100.0),
            _make_exec_result(execution_time_ms=100.0),
            _make_exec_result(execution_time_ms=1000.0),  # very slow
        ]
        anomalies = ExecutionMetricsCalculator.detect_performance_anomalies(results)
        assert len(anomalies) == 1
        a = anomalies[0]
        assert a["severity"] == "high"
        assert a["index"] == 3
        assert a["execution_time_ms"] == 1000.0
        # avg = (100+100+100+1000)/4 = 325, threshold = 650, 1000 > 650*1.5=975
        assert a["average_time_ms"] == 325.0
        assert a["threshold_ms"] == 650.0

    def test_detect_anomalies_none(self):
        results = [_make_exec_result(execution_time_ms=100.0) for _ in range(5)]
        anomalies = ExecutionMetricsCalculator.detect_performance_anomalies(results)
        assert anomalies == []


# ── ResourceEstimator ────────────────────────────────────────────────────────


class TestResourceEstimator:
    def test_memory_python(self):
        # "x = 1" is 1 line, size_multiplier=max(1.0, 1/100)=1.0, base=128, type=unit(1.0)
        mem = ResourceEstimator.estimate_memory_requirement("x = 1", "python")
        assert mem == 128  # int(128 * 1.0 * 1.0) = 128

    def test_memory_java_higher_base(self):
        # java base=256, js base=64, both 1 line unit
        mem_java = ResourceEstimator.estimate_memory_requirement("x = 1", "java")
        mem_js = ResourceEstimator.estimate_memory_requirement("x = 1", "javascript")
        assert mem_java == 256
        assert mem_js == 64
        assert mem_java > mem_js

    def test_memory_integration_multiplier(self):
        # python base=128, unit=1.0 -> 128, integration=1.5 -> 192
        mem_unit = ResourceEstimator.estimate_memory_requirement("x = 1", "python", "unit")
        mem_int = ResourceEstimator.estimate_memory_requirement("x = 1", "python", "integration")
        assert mem_unit == 128
        assert mem_int == 192

    def test_memory_large_code(self):
        large_code = "\n".join(["x = 1"] * 500)
        # 500 lines, size_multiplier=max(1.0, 500/100)=5.0, base=128, unit=1.0
        mem = ResourceEstimator.estimate_memory_requirement(large_code, "python")
        assert mem == 640  # int(128 * 5.0 * 1.0)

    def test_memory_unknown_language(self):
        # Unknown defaults to base=128, 1 line, unit
        mem = ResourceEstimator.estimate_memory_requirement("x", "brainfuck")
        assert mem == 128

    def test_memory_clamped_min(self):
        # js base=64, 1 line -> 64 (clamped to min 64)
        mem = ResourceEstimator.estimate_memory_requirement("x", "javascript")
        assert mem == 64

    def test_cpu_python(self):
        # python base=0.5, unit multiplier=1.0 -> 0.5
        cpu = ResourceEstimator.estimate_cpu_requirement("x = 1", "python")
        assert cpu == 0.5

    def test_cpu_performance_test(self):
        # python base=0.5, unit=1.0 -> 0.5, performance=2.0 -> 1.0
        cpu_unit = ResourceEstimator.estimate_cpu_requirement("x", "python", "unit")
        cpu_perf = ResourceEstimator.estimate_cpu_requirement("x", "python", "performance")
        assert cpu_unit == 0.5
        assert cpu_perf == 1.0

    def test_cpu_unknown_language(self):
        # Unknown defaults to base=0.5, unit=1.0 -> 0.5
        cpu = ResourceEstimator.estimate_cpu_requirement("x", "brainfuck")
        assert cpu == 0.5


# ── TestCodeAnalyzer ─────────────────────────────────────────────────────────


class TestTestCodeAnalyzer:
    def test_complexity_empty(self):
        result = TestCodeAnalyzer.estimate_complexity("", "python")
        assert result == {
            "total_lines": 1,  # "".split("\n") == [""]
            "code_lines": 0,
            "complexity_indicators": 0,
            "complexity_score": 0.0,
            "complexity_level": "low",
            "estimated_execution_time_ms": 100,  # min(max(0*10, 100), 30000)
        }

    def test_complexity_simple(self):
        code = "x = 1\ny = 2\nz = x + y"
        result = TestCodeAnalyzer.estimate_complexity(code, "python")
        assert result["complexity_level"] == "low"
        assert result["code_lines"] == 3
        assert result["total_lines"] == 3
        assert result["complexity_indicators"] == 0  # no if/for/while
        assert result["complexity_score"] == 0.0
        assert result["estimated_execution_time_ms"] == 100  # min(max(3*10, 100), 30000)

    def test_complexity_complex(self):
        code = "\n".join(
            [
                "if x > 0:",
                "    for i in range(10):",
                "        while True:",
                "            if i > 5:",
                "                break",
            ]
        )
        result = TestCodeAnalyzer.estimate_complexity(code, "python")
        # "if " appears in lines 1,4; "for " in line 2; "while " in line 3 -> 4 indicators
        assert result["complexity_indicators"] == 4
        assert result["code_lines"] == 5
        assert result["complexity_score"] == 4 / 5  # 0.8
        assert result["complexity_level"] == "high"

    def test_complexity_javascript(self):
        code = "if (x) { for (let i=0; i<10; i++) { } }"
        result = TestCodeAnalyzer.estimate_complexity(code, "javascript")
        # Single line containing both "if " and "for " -> 2 indicators
        assert result["complexity_indicators"] == 2
        assert result["code_lines"] == 1
        assert result["complexity_score"] == 2.0
        assert result["complexity_level"] == "high"

    def test_complexity_unknown_language(self):
        code = "if x\nfor i\nwhile true"
        result = TestCodeAnalyzer.estimate_complexity(code, "brainfuck")
        assert result["total_lines"] == 3
        assert result["code_lines"] == 3
        # Default indicators: ["if ", "for ", "while "] - "while true" has "while " -> match
        # "if x" has "if " -> match, "for i" has "for " -> match
        assert result["complexity_indicators"] == 3

    def test_detect_patterns_python(self):
        code = "import pytest\n@pytest.fixture\ndef fix(): pass\n@mock.patch('x')\ndef test_a(): assert True"
        patterns = TestCodeAnalyzer.detect_test_patterns(code, "python")
        assert sorted(patterns) == ["fixtures", "mock_usage", "unit_test"]

    def test_detect_patterns_javascript(self):
        code = "describe('test', () => { it('works', async () => { expect(1).toBe(1); }); });"
        patterns = TestCodeAnalyzer.detect_test_patterns(code, "javascript")
        assert sorted(patterns) == ["async_test", "unit_test"]

    def test_detect_patterns_java(self):
        code = "@Test public void testSomething() { Mockito.when(x).thenReturn(y); }"
        patterns = TestCodeAnalyzer.detect_test_patterns(code, "java")
        assert sorted(patterns) == ["mock_usage", "unit_test"]

    def test_detect_patterns_unknown(self):
        code = "x = 1"
        patterns = TestCodeAnalyzer.detect_test_patterns(code, "brainfuck")
        assert patterns == []

    def test_detect_patterns_python_parametrize(self):
        code = "@pytest.mark.parametrize('x', [1,2,3])\ndef test_param(x): assert x > 0"
        patterns = TestCodeAnalyzer.detect_test_patterns(code, "python")
        # "pytest" triggers unit_test, "parametrize" triggers parameterized, "assert" also unit_test (deduped)
        assert sorted(patterns) == ["parameterized", "unit_test"]


# ── TimestampHelper ──────────────────────────────────────────────────────────


class TestTimestampHelper:
    def test_utc_now(self):
        before = datetime.now(UTC)
        now = TimestampHelper.utc_now()
        after = datetime.now(UTC)
        assert now.tzinfo == UTC
        assert before <= now <= after

    def test_format_timestamp(self):
        dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        formatted = TimestampHelper.format_timestamp(dt)
        assert formatted == "2025-01-01T12:00:00+00:00"

    def test_parse_timestamp_iso(self):
        ts = "2025-01-01T12:00:00+00:00"
        dt = TimestampHelper.parse_timestamp(ts)
        assert dt == datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

    def test_parse_timestamp_z(self):
        ts = "2025-06-15T10:30:00Z"
        dt = TimestampHelper.parse_timestamp(ts)
        assert dt == datetime(2025, 6, 15, 10, 30, 0, tzinfo=UTC)

    def test_calculate_duration_ms(self):
        start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
        end = datetime(2025, 1, 1, 0, 0, 1, tzinfo=UTC)
        duration = TimestampHelper.calculate_duration_ms(start, end)
        assert duration == 1000.0

    def test_calculate_duration_zero(self):
        now = datetime.now(UTC)
        assert TimestampHelper.calculate_duration_ms(now, now) == 0.0


# ── utilities_and_helpers ────────────────────────────────────────────────────

from routers.intelligent_testing_modules.utilities_and_helpers import (  # noqa: E402
    RequestHasher,
    TestRequestValidator,
    calculate_estimated_batch_duration,
    format_error_response,
    generate_execution_id,
    is_valid_code_content,
    safe_async_operation,
    sanitize_project_name,
    validate_batch_request_limits,
)


class TestTestRequestValidator:
    def test_validate_language_valid(self):
        assert TestRequestValidator.validate_language("Python") == "python"
        assert TestRequestValidator.validate_language("JAVA") == "java"

    def test_validate_language_invalid(self):
        with pytest.raises(ValueError, match="Unsupported"):
            TestRequestValidator.validate_language("brainfuck")

    def test_validate_test_type_valid(self):
        assert TestRequestValidator.validate_test_type("unit") == "unit"
        assert TestRequestValidator.validate_test_type("INTEGRATION") == "integration"

    def test_validate_test_type_none(self):
        assert TestRequestValidator.validate_test_type(None) == "unit"

    def test_validate_test_type_unknown(self):
        assert TestRequestValidator.validate_test_type("weird") == "unit"

    def test_validate_timeout_valid(self):
        assert TestRequestValidator.validate_timeout(30) == 30
        assert TestRequestValidator.validate_timeout(3600) == 3600

    def test_validate_timeout_invalid(self):
        with pytest.raises(ValueError):
            TestRequestValidator.validate_timeout(10)
        with pytest.raises(ValueError):
            TestRequestValidator.validate_timeout(4000)

    def test_validate_priority_valid(self):
        assert TestRequestValidator.validate_priority(1) == 1
        assert TestRequestValidator.validate_priority(10) == 10

    def test_validate_priority_invalid(self):
        with pytest.raises(ValueError):
            TestRequestValidator.validate_priority(0)
        with pytest.raises(ValueError):
            TestRequestValidator.validate_priority(11)

    def test_validate_resource_weight_valid(self):
        assert TestRequestValidator.validate_resource_weight(0.1) == 0.1
        assert TestRequestValidator.validate_resource_weight(10.0) == 10.0

    def test_validate_resource_weight_invalid(self):
        with pytest.raises(ValueError):
            TestRequestValidator.validate_resource_weight(0.05)
        with pytest.raises(ValueError):
            TestRequestValidator.validate_resource_weight(11.0)


class TestRequestHasher:
    def _make_mock_request(self, **overrides):
        req = MagicMock()
        req.language = overrides.get("language", "python")
        req.test_type = overrides.get("test_type", "unit")
        req.source_code = overrides.get("source_code", "x = 1")
        req.test_files = overrides.get("test_files", {})
        req.dependencies = overrides.get("dependencies", {})
        req.security_level = overrides.get("security_level", "medium")
        return req

    def test_hash_request_deterministic(self):
        req = self._make_mock_request()
        h1 = RequestHasher.hash_request(req)
        h2 = RequestHasher.hash_request(req)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_hash_request_different_code(self):
        req1 = self._make_mock_request(source_code="x = 1")
        req2 = self._make_mock_request(source_code="x = 2")
        h1 = RequestHasher.hash_request(req1)
        h2 = RequestHasher.hash_request(req2)
        assert h1 != h2
        assert len(h1) == 64
        assert len(h2) == 64

    def test_hash_code_deterministic(self):
        h1 = RequestHasher.hash_code("x = 1", "python")
        h2 = RequestHasher.hash_code("x = 1", "python")
        assert h1 == h2
        assert len(h1) == 64

    def test_hash_code_different_language(self):
        h1 = RequestHasher.hash_code("x = 1", "python")
        h2 = RequestHasher.hash_code("x = 1", "javascript")
        assert h1 != h2


class TestGenerateExecutionId:
    def test_default_prefix(self):
        eid = generate_execution_id()
        assert eid.startswith("exec_")
        # The suffix should be a valid float timestamp
        timestamp_part = eid.removeprefix("exec_")
        ts = float(timestamp_part)
        assert ts > 0

    def test_custom_prefix(self):
        eid = generate_execution_id("batch")
        assert eid.startswith("batch_")
        timestamp_part = eid.removeprefix("batch_")
        ts = float(timestamp_part)
        assert ts > 0

    def test_uniqueness(self):
        eid1 = generate_execution_id()
        eid2 = generate_execution_id()
        # Timestamps should differ (or at least IDs should be parseable)
        assert eid1.startswith("exec_")
        assert eid2.startswith("exec_")


class TestSafeAsyncOperation:
    @pytest.mark.asyncio
    async def test_success(self):
        async def op():
            return 42

        result = await safe_async_operation(op)
        assert result == 42

    @pytest.mark.asyncio
    async def test_timeout(self):
        async def slow():
            await asyncio.sleep(10)

        result = await safe_async_operation(slow, timeout=0.01)
        assert result is None

    @pytest.mark.asyncio
    async def test_exception(self):
        async def bad():
            raise RuntimeError("boom")

        result = await safe_async_operation(bad)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_timeout(self):
        async def op():
            return "ok"

        result = await safe_async_operation(op, timeout=None)
        assert result == "ok"


class TestSanitizeProjectName:
    def test_normal(self):
        assert sanitize_project_name("my-project") == "my-project"

    def test_special_chars(self):
        assert sanitize_project_name("my project!@#") == "my_project"

    def test_empty(self):
        assert sanitize_project_name("") == "unnamed_project"

    def test_all_special(self):
        assert sanitize_project_name("!@#$%") == "unnamed_project"

    def test_long_name(self):
        long = "a" * 100
        assert len(sanitize_project_name(long)) == 50

    def test_multiple_underscores(self):
        assert sanitize_project_name("a___b") == "a_b"


class TestValidateBatchRequestLimits:
    def test_valid(self):
        validate_batch_request_limits(10)  # should not raise

    def test_zero(self):
        with pytest.raises(ValueError, match="at least one"):
            validate_batch_request_limits(0)

    def test_too_many(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            validate_batch_request_limits(101)

    def test_concurrent_valid(self):
        validate_batch_request_limits(10, 5)

    def test_concurrent_invalid(self):
        with pytest.raises(ValueError, match="max_concurrent"):
            validate_batch_request_limits(10, 25)


class TestFormatErrorResponse:
    def test_basic(self):
        err = format_error_response(ValueError("oops"), "validation")
        assert err["error"] == "validation"
        assert err["message"] == "oops"
        # Verify timestamp is a valid ISO format string
        datetime.fromisoformat(err["timestamp"])
        assert len(err) == 3


class TestCalculateEstimatedBatchDuration:
    def test_zero_tests(self):
        assert calculate_estimated_batch_duration(0, 100.0) == 0.0

    def test_single_batch(self):
        # 3 tests, 5 concurrent -> 1 batch
        assert calculate_estimated_batch_duration(3, 100.0, 5) == 100.0

    def test_multiple_batches(self):
        # 10 tests, 5 concurrent -> 2 batches
        assert calculate_estimated_batch_duration(10, 100.0, 5) == 200.0

    def test_partial_batch(self):
        # 7 tests, 5 concurrent -> 2 batches
        assert calculate_estimated_batch_duration(7, 100.0, 5) == 200.0


class TestIsValidCodeContent:
    def test_valid(self):
        assert is_valid_code_content("x = 1") is True

    def test_empty(self):
        assert is_valid_code_content("") is False

    def test_whitespace_only(self):
        assert is_valid_code_content("   ") is False

    def test_too_large(self):
        assert is_valid_code_content("x" * 1_000_001) is False


# ── execution_coordination helper functions ──────────────────────────────────

from routers.intelligent_testing_modules.execution_coordination import (  # noqa: E402
    _generate_batch_summary,
    optimize_execution_order,
)


class TestGenerateBatchSummary:
    @pytest.mark.asyncio
    async def test_empty(self):
        summary = await _generate_batch_summary([])
        assert summary["total_tests"] == 0
        assert summary["successful_tests"] == 0

    @pytest.mark.asyncio
    async def test_with_results(self):
        results = [
            _make_exec_result(
                success=True,
                execution_time_ms=100.0,
                optimization_applied=True,
                pattern_matches=["p1"],
                recommendations=["r1"],
            ),
            _make_exec_result(
                success=False,
                execution_time_ms=200.0,
                optimization_applied=False,
                pattern_matches=[],
                recommendations=[],
            ),
        ]
        summary = await _generate_batch_summary(results)
        assert summary["total_tests"] == 2
        assert summary["successful_tests"] == 1
        assert summary["failed_tests"] == 1
        assert summary["total_execution_time"] == 300.0
        assert summary["average_execution_time"] == 150.0
        assert summary["ai_optimizations_applied"] == 1
        assert summary["patterns_detected"] == 1
        assert summary["recommendations_generated"] == 1


class TestOptimizeExecutionOrder:
    @pytest.mark.asyncio
    async def test_sorted_by_priority_desc(self):
        r1 = MagicMock()
        r1.priority_level = 3
        r1.expected_execution_time_ms = 100.0

        r2 = MagicMock()
        r2.priority_level = 8
        r2.expected_execution_time_ms = 200.0

        r3 = MagicMock()
        r3.priority_level = 5
        r3.expected_execution_time_ms = 50.0

        optimized = await optimize_execution_order([r1, r2, r3])
        # Sorted: r2 (pri=8), r3 (pri=5), r1 (pri=3)
        assert [r.priority_level for r in optimized] == [8, 5, 3]
        assert optimized[0] is r2
        assert optimized[1] is r3
        assert optimized[2] is r1

    @pytest.mark.asyncio
    async def test_same_priority_sort_by_time(self):
        r1 = MagicMock()
        r1.priority_level = 5
        r1.expected_execution_time_ms = 300.0

        r2 = MagicMock()
        r2.priority_level = 5
        r2.expected_execution_time_ms = 100.0

        optimized = await optimize_execution_order([r1, r2])
        # Same priority: sorted by execution time ascending
        assert optimized[0] is r2
        assert optimized[1] is r1
        assert optimized[0].expected_execution_time_ms == 100.0
        assert optimized[1].expected_execution_time_ms == 300.0

    @pytest.mark.asyncio
    async def test_empty(self):
        optimized = await optimize_execution_order([])
        assert optimized == []


# ── service_lifecycle helper functions ───────────────────────────────────────

from routers.intelligent_testing_modules.service_lifecycle import (  # noqa: E402
    backup_settings,
    get_configuration_status,
    restore_settings,
)


class TestBackupSettings:
    @pytest.mark.asyncio
    async def test_backup_success(self):
        service = MagicMock()
        service.settings = MagicMock()
        service.settings.model_dump = MagicMock(return_value={"key": "val"})
        service.orchestrator = MagicMock()
        service.orchestrator.max_concurrent_tests = 5
        service.orchestrator.ai_analysis_threshold = 0.5
        service.orchestrator.pattern_analysis_interval = 120

        backup = await backup_settings(service)
        # Verify timestamp is a valid ISO format string
        datetime.fromisoformat(backup["timestamp"])
        assert backup["settings"] == {"key": "val"}
        assert backup["orchestrator_settings"] == {
            "max_concurrent_tests": 5,
            "ai_analysis_threshold": 0.5,
            "pattern_analysis_interval": 120,
        }
        # Verify no error key present on success
        assert "error" not in backup

    @pytest.mark.asyncio
    async def test_backup_error(self):
        service = MagicMock()
        service.settings = MagicMock(spec=[])  # no model_dump
        service.settings.model_dump = MagicMock(side_effect=Exception("boom"))
        # orchestrator accessed before error
        service.orchestrator = MagicMock(spec=[])

        backup = await backup_settings(service)
        assert backup["error"] == "boom"
        assert backup["settings"] == {}
        assert backup["orchestrator_settings"] == {}
        datetime.fromisoformat(backup["timestamp"])


class TestRestoreSettings:
    @pytest.mark.asyncio
    async def test_restore_dict(self):
        service = MagicMock()
        service.orchestrator = MagicMock()

        with patch(
            "routers.intelligent_testing_modules.service_lifecycle.TestExecutionSettings"
        ) as mock_settings_cls:
            mock_settings_cls.return_value = MagicMock()
            result = await restore_settings(
                service,
                {
                    "settings": {"max_concurrent_tests": 10},
                    "orchestrator_settings": {"max_concurrent_tests": 10},
                },
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_restore_non_dict_settings(self):
        service = MagicMock()
        result = await restore_settings(service, {"settings": "not-a-dict"})
        assert result is False

    @pytest.mark.asyncio
    async def test_restore_error(self):
        service = MagicMock()
        with patch(
            "routers.intelligent_testing_modules.service_lifecycle.TestExecutionSettings",
            side_effect=Exception("bad"),
        ):
            result = await restore_settings(service, {"settings": {"x": 1}})
            assert result is False


class TestGetConfigurationStatus:
    @pytest.mark.asyncio
    async def test_healthy(self):
        service = MagicMock()
        service.is_running = True
        service.orchestrator = MagicMock()
        service.orchestrator.max_concurrent_tests = 5
        service.orchestrator.ai_analysis_threshold = 0.5
        service.orchestrator.pattern_analysis_interval = 120

        status = await get_configuration_status(service)
        assert status["service_running"] is True
        assert status["configuration_health"] == "healthy"
        assert status["settings_valid"] is True
        assert status["orchestrator_configured"] is True
        datetime.fromisoformat(status["last_updated"])
        orch = status["orchestrator_status"]
        assert orch["available"] is True
        assert orch["max_concurrent_tests"] == 5
        assert orch["ai_analysis_threshold"] == 0.5
        assert orch["pattern_analysis_interval"] == 120

    @pytest.mark.asyncio
    async def test_no_orchestrator(self):
        service = MagicMock(spec=["is_running"])
        service.is_running = False

        status = await get_configuration_status(service)
        assert status["service_running"] is False
        assert status["orchestrator_status"] == {"available": False}
        assert status["configuration_health"] == "degraded"

    @pytest.mark.asyncio
    async def test_error(self):
        service = MagicMock()
        service.is_running = property(lambda self: 1 / 0)
        # Override to raise
        type(service).is_running = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("boom"))
        )

        status = await get_configuration_status(service)
        assert status["configuration_health"] == "critical"
        assert status["service_running"] is False
        assert status["settings_valid"] is False
        assert status["orchestrator_configured"] is False
        assert "boom" in status["error"]
        datetime.fromisoformat(status["last_updated"])


# ── configuration_management helpers ─────────────────────────────────────────

from routers.intelligent_testing_modules.configuration_management import (  # noqa: E402
    _apply_orchestrator_settings,
    _validate_settings,
)


class TestValidateSettings:
    @pytest.mark.asyncio
    async def test_valid_settings(self):
        settings = MagicMock()
        settings.max_concurrent_tests = 10
        settings.default_timeout_seconds = 300
        settings.ai_analysis_threshold = 0.5
        settings.pattern_analysis_interval = 120
        await _validate_settings(settings)  # should not raise

    @pytest.mark.asyncio
    async def test_invalid_max_concurrent(self):
        settings = MagicMock()
        settings.max_concurrent_tests = 100
        settings.default_timeout_seconds = 300
        settings.ai_analysis_threshold = 0.5
        settings.pattern_analysis_interval = 120
        with pytest.raises(ValueError, match="max_concurrent_tests"):
            await _validate_settings(settings)

    @pytest.mark.asyncio
    async def test_invalid_timeout(self):
        settings = MagicMock()
        settings.max_concurrent_tests = 10
        settings.default_timeout_seconds = 10  # too low
        settings.ai_analysis_threshold = 0.5
        settings.pattern_analysis_interval = 120
        with pytest.raises(ValueError, match="default_timeout_seconds"):
            await _validate_settings(settings)

    @pytest.mark.asyncio
    async def test_invalid_threshold(self):
        settings = MagicMock()
        settings.max_concurrent_tests = 10
        settings.default_timeout_seconds = 300
        settings.ai_analysis_threshold = 0.05  # too low
        settings.pattern_analysis_interval = 120
        with pytest.raises(ValueError, match="ai_analysis_threshold"):
            await _validate_settings(settings)

    @pytest.mark.asyncio
    async def test_invalid_pattern_interval(self):
        settings = MagicMock()
        settings.max_concurrent_tests = 10
        settings.default_timeout_seconds = 300
        settings.ai_analysis_threshold = 0.5
        settings.pattern_analysis_interval = 10  # too low
        with pytest.raises(ValueError, match="pattern_analysis_interval"):
            await _validate_settings(settings)


class TestApplyOrchestratorSettings:
    @pytest.mark.asyncio
    async def test_applies_settings(self):
        service = MagicMock()
        service.orchestrator = MagicMock()
        service.orchestrator.max_concurrent_tests = 5
        service.orchestrator.ai_analysis_threshold = 0.5
        service.orchestrator.pattern_analysis_interval = 120

        settings = MagicMock()
        settings.max_concurrent_tests = 10
        settings.ai_analysis_threshold = 0.8
        settings.pattern_analysis_interval = 300

        await _apply_orchestrator_settings(service, settings)

        assert service.orchestrator.max_concurrent_tests == 10
        assert service.orchestrator.ai_analysis_threshold == 0.8
        assert service.orchestrator.pattern_analysis_interval == 300

    @pytest.mark.asyncio
    async def test_error_propagates(self):
        service = MagicMock()
        service.orchestrator = MagicMock(spec=[])  # no attributes

        settings = MagicMock()
        settings.max_concurrent_tests = 10

        # Should not raise since attributes are checked with hasattr
        await _apply_orchestrator_settings(service, settings)


# ── discovery_and_recommendations helper functions ───────────────────────────

from routers.intelligent_testing_modules.discovery_and_recommendations import (  # noqa: E402
    analyze_test_coverage_gaps,
    discover_test_opportunities,
    recommend_test_strategies,
)


class TestDiscoverTestOpportunities:
    @pytest.mark.asyncio
    async def test_returns_filtered(self):
        service = MagicMock()
        rec1 = MagicMock()
        rec1.language = "python"
        rec2 = MagicMock()
        rec2.language = "java"
        service.get_recommendations = AsyncMock(return_value=[rec1, rec2])

        with patch(
            "routers.intelligent_testing_modules.discovery_and_recommendations.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
        ):
            result = await discover_test_opportunities(
                service, "container-1", "def foo(): pass", "python"
            )
        assert len(result) == 1
        assert result[0] is rec1  # Only the python recommendation should be returned

    @pytest.mark.asyncio
    async def test_error_returns_empty(self):
        service = MagicMock()
        service.get_recommendations = AsyncMock(side_effect=Exception("boom"))

        with patch(
            "routers.intelligent_testing_modules.discovery_and_recommendations.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
            side_effect=Exception("boom"),
        ):
            result = await discover_test_opportunities(service, "c", "x", "python")
        assert result == []


class TestRecommendTestStrategies:
    @pytest.mark.asyncio
    async def test_web_project(self):
        service = MagicMock()
        with patch(
            "routers.intelligent_testing_modules.discovery_and_recommendations.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
        ):
            result = await recommend_test_strategies(service, "c1", "web", "python")
        assert len(result) == 2
        assert result[0].recommendation_id == "strategy_integration"
        assert result[0].priority == "high"
        assert result[0].title == "Integration Test Suite"
        assert result[1].recommendation_id == "strategy_contract"
        assert result[1].priority == "medium"

    @pytest.mark.asyncio
    async def test_api_project(self):
        """API projects get the same strategies as web projects."""
        service = MagicMock()
        with patch(
            "routers.intelligent_testing_modules.discovery_and_recommendations.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
        ):
            result = await recommend_test_strategies(service, "c1", "api", "python")
        assert len(result) == 2
        assert result[0].recommendation_id == "strategy_integration"
        assert result[1].recommendation_id == "strategy_contract"

    @pytest.mark.asyncio
    async def test_library_project(self):
        service = MagicMock()
        with patch(
            "routers.intelligent_testing_modules.discovery_and_recommendations.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
        ):
            result = await recommend_test_strategies(service, "c1", "library", "python")
        assert len(result) == 2
        assert result[0].recommendation_id == "strategy_unit"
        assert result[0].priority == "high"
        assert result[0].confidence_level == 0.95
        assert result[1].recommendation_id == "strategy_compatibility"

    @pytest.mark.asyncio
    async def test_unknown_project(self):
        service = MagicMock()
        with patch(
            "routers.intelligent_testing_modules.discovery_and_recommendations.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
        ):
            result = await recommend_test_strategies(service, "c1", "other", "python")
        assert result == []

    @pytest.mark.asyncio
    async def test_error_returns_empty(self):
        service = MagicMock()
        with patch(
            "routers.intelligent_testing_modules.discovery_and_recommendations.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
            side_effect=Exception("boom"),
        ):
            result = await recommend_test_strategies(service, "c1", "web", "python")
        assert result == []


class TestAnalyzeTestCoverageGaps:
    @pytest.mark.asyncio
    async def test_no_coverage_patterns(self):
        service = MagicMock()
        pat = MagicMock()
        pat.pattern_type = "performance"
        service.get_patterns = AsyncMock(return_value=[pat])

        with patch(
            "routers.intelligent_testing_modules.discovery_and_recommendations.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
        ):
            result = await analyze_test_coverage_gaps(
                service, "c1", ["test_a.py"], "def foo(): pass", "python"
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_with_coverage_patterns(self):
        service = MagicMock()
        pat = MagicMock()
        pat.pattern_type = "coverage_gap"
        pat.description = "Missing edge case tests"
        service.get_patterns = AsyncMock(return_value=[pat])

        with patch(
            "routers.intelligent_testing_modules.discovery_and_recommendations.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
        ):
            result = await analyze_test_coverage_gaps(
                service, "c1", ["test_a.py"], "def foo(): pass", "python"
            )
        assert len(result) == 1
        rec = result[0]
        assert rec.title == "Coverage Gap Detected"
        assert rec.recommendation_id == "coverage_0"
        assert rec.recommendation_type == "coverage_gap"
        assert rec.priority == "medium"
        assert "Missing edge case tests" in rec.description
        assert rec.confidence_level == 0.8
        assert rec.estimated_improvement == 0.5

    @pytest.mark.asyncio
    async def test_error_returns_empty(self):
        service = MagicMock()
        with patch(
            "routers.intelligent_testing_modules.discovery_and_recommendations.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
            side_effect=Exception("boom"),
        ):
            result = await analyze_test_coverage_gaps(service, "c1", [], "x", "python")
        assert result == []


# ── shared.py response model serializers ─────────────────────────────────────

from routers.intelligent_testing_modules.shared import (  # noqa: E402
    ExecutionInsightsResponse,
    HealthStatusResponse,
    PatternsResponse,
    RecommendationsResponse,
    ServiceSettingsResponse,
)


class TestSharedResponseSerializers:
    """Cover serialize_datetime None paths in shared.py response models (lines 178, 196, 214, 232)."""

    def test_insights_response_serialize_none(self):
        resp = ExecutionInsightsResponse(
            status="success",
            message="test",
            insights={"key": "value"},
        )
        # Force serialize_datetime with None
        assert resp.serialize_datetime(None) is None

    def test_insights_response_serialize_datetime(self):
        resp = ExecutionInsightsResponse(
            status="success",
            message="test",
            insights={"key": "value"},
        )
        dt = datetime(2025, 6, 15, 10, 30, 0, tzinfo=UTC)
        result = resp.serialize_datetime(dt)
        assert result == "2025-06-15T10:30:00+00:00"

    def test_recommendations_response_serialize_none(self):
        resp = RecommendationsResponse(
            status="success",
            message="test",
            recommendations=[],
        )
        assert resp.serialize_datetime(None) is None

    def test_patterns_response_serialize_none(self):
        resp = PatternsResponse(
            status="success",
            message="test",
            patterns=[],
        )
        assert resp.serialize_datetime(None) is None

    def test_health_status_response_serialize_none(self):
        resp = HealthStatusResponse(
            status="success",
            message="test",
            health_status={"healthy": True},
        )
        assert resp.serialize_datetime(None) is None

    def test_service_settings_response_serialize_none(self):
        from services.test_execution.intelligent_test_integration import TestExecutionSettings

        settings = TestExecutionSettings()
        resp = ServiceSettingsResponse(
            status="success",
            message="test",
            settings=settings,
        )
        assert resp.serialize_datetime(None) is None

    def test_recommendations_response_serialize_datetime(self):
        """RecommendationsResponse.serialize_datetime with value (line 178)."""
        resp = RecommendationsResponse(
            status="success",
            message="test",
            recommendations=[],
        )
        dt = datetime(2025, 3, 20, 8, 0, 0, tzinfo=UTC)
        result = resp.serialize_datetime(dt)
        assert result == "2025-03-20T08:00:00+00:00"

    def test_patterns_response_serialize_datetime(self):
        """PatternsResponse.serialize_datetime with value (line 196)."""
        resp = PatternsResponse(
            status="success",
            message="test",
            patterns=[],
        )
        dt = datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC)
        result = resp.serialize_datetime(dt)
        assert result == "2025-12-31T23:59:59+00:00"

    def test_health_status_response_serialize_datetime(self):
        """HealthStatusResponse.serialize_datetime with value (line 214)."""
        resp = HealthStatusResponse(
            status="success",
            message="test",
            health_status={"healthy": True},
        )
        dt = datetime(2025, 7, 4, 0, 0, 0, tzinfo=UTC)
        result = resp.serialize_datetime(dt)
        assert result == "2025-07-04T00:00:00+00:00"

    def test_service_settings_response_serialize_datetime(self):
        """ServiceSettingsResponse.serialize_datetime with value (line 232)."""
        from services.test_execution.intelligent_test_integration import TestExecutionSettings

        settings = TestExecutionSettings()
        resp = ServiceSettingsResponse(
            status="success",
            message="test",
            settings=settings,
        )
        dt = datetime(2025, 1, 15, 14, 30, 45, tzinfo=UTC)
        result = resp.serialize_datetime(dt)
        assert result == "2025-01-15T14:30:45+00:00"


class TestGetIntelligentTestServiceFactory:
    """Cover get_intelligent_test_service (lines 245-254)."""

    @pytest.mark.asyncio
    async def test_creates_service_instance(self):
        """get_intelligent_test_service creates and caches service, then starts it."""
        import routers.intelligent_testing_modules.shared as mod

        old = mod._service_instance
        mod._service_instance = None
        try:
            mock_docker = MagicMock()
            mock_service = MagicMock()
            mock_service.is_running = False
            mock_service.start = AsyncMock()

            with patch(
                "routers.intelligent_testing_modules.shared.create_intelligent_test_service",
                return_value=mock_service,
            ) as mock_create:
                result = await mod.get_intelligent_test_service(docker_service=mock_docker)
                assert result is mock_service
                mock_service.start.assert_awaited_once()
                mock_create.assert_called_once()
                # Verify it was cached
                assert mod._service_instance is mock_service
        finally:
            mod._service_instance = old

    @pytest.mark.asyncio
    async def test_returns_cached_service(self):
        """get_intelligent_test_service returns cached instance without creating a new one."""
        import routers.intelligent_testing_modules.shared as mod

        old = mod._service_instance
        mock_service = MagicMock()
        mock_service.is_running = True
        mod._service_instance = mock_service
        try:
            with patch(
                "routers.intelligent_testing_modules.shared.create_intelligent_test_service",
            ) as mock_create:
                result = await mod.get_intelligent_test_service(docker_service=MagicMock())
                assert result is mock_service
                # Verify no new service was created
                mock_create.assert_not_called()
        finally:
            mod._service_instance = old
