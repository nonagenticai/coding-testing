"""Coverage tests for routers/testing_modules/testing_utils.py.

Covers uncovered lines: 36-51, 60-64, 69-75, 81-90, 94, 106-200, 296-297.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from models.testing_models import QualityResult, TestResult, TestSecurityLevel
from routers.testing_modules.testing_utils import (
    _log_batch_execution,
    _log_quality_analysis,
    _log_test_execution,
    aggregate_test_metrics,
    format_test_results,
    validate_test_request,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_result(**overrides) -> TestResult:
    defaults = dict(
        test_id="t-1",
        success=True,
        stdout="OK",
        stderr="",
        exit_code=0,
        execution_time=1.5,
        created_at=datetime.now(UTC),
        language="python",
        security_level=TestSecurityLevel.MEDIUM,
        container_name="c1",
    )
    defaults.update(overrides)
    return TestResult(**defaults)


# ---------------------------------------------------------------------------
# validate_test_request
# ---------------------------------------------------------------------------


class TestValidateTestRequest:
    def test_empty_request_raises(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_test_request({})
        assert exc_info.value.status_code == 400

    def test_none_request_raises(self):
        with pytest.raises(HTTPException):
            validate_test_request(None)

    def test_missing_code_raises(self):
        with pytest.raises(HTTPException):
            validate_test_request({"language": "python"})

    def test_empty_code_raises(self):
        with pytest.raises(HTTPException):
            validate_test_request({"code": "   "})

    def test_unsupported_language_raises(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_test_request({"code": "print(1)", "language": "brainfuck"})
        assert "not supported" in exc_info.value.detail

    def test_valid_request_defaults(self):
        result = validate_test_request({"code": "print(1)"})
        assert result["language"] == "python"
        assert result["timeout"] == 30
        assert result["memory_limit"] == 128
        assert result["test_type"] == "unit"

    def test_timeout_capped_at_300(self):
        result = validate_test_request({"code": "x=1", "timeout": 600})
        assert result["timeout"] == 300

    def test_memory_capped_at_512(self):
        result = validate_test_request({"code": "x=1", "memory_limit": 1024})
        assert result["memory_limit"] == 512

    def test_language_lowercased(self):
        result = validate_test_request({"code": "x=1", "language": "PYTHON"})
        assert result["language"] == "python"

    def test_code_stripped(self):
        result = validate_test_request({"code": "  print(1)  "})
        assert result["code"] == "print(1)"


# ---------------------------------------------------------------------------
# format_test_results
# ---------------------------------------------------------------------------


class TestFormatTestResults:
    def test_empty_results(self):
        result = format_test_results([])
        assert result["total_tests"] == 0
        assert result["successful"] == 0
        assert result["avg_execution_time"] == 0.0

    def test_single_success(self):
        results = [_make_test_result()]
        formatted = format_test_results(results)
        assert formatted["total_tests"] == 1
        assert formatted["successful"] == 1
        assert formatted["failed"] == 0
        assert formatted["success_rate"] == 1.0

    def test_mixed_results(self):
        results = [
            _make_test_result(success=True, execution_time=1.0),
            _make_test_result(test_id="t-2", success=False, exit_code=1, execution_time=2.0),
        ]
        formatted = format_test_results(results)
        assert formatted["total_tests"] == 2
        assert formatted["successful"] == 1
        assert formatted["failed"] == 1
        assert formatted["avg_execution_time"] == 1.5


# ---------------------------------------------------------------------------
# aggregate_test_metrics
# ---------------------------------------------------------------------------


class TestAggregateTestMetrics:
    def test_empty_results(self):
        result = aggregate_test_metrics([])
        assert result["success_rate"] == 0.0
        assert result["languages"] == {}

    def test_single_language(self):
        results = [
            _make_test_result(language="python", execution_time=1.0),
            _make_test_result(
                test_id="t-2",
                language="python",
                success=False,
                exit_code=1,
                execution_time=2.0,
                stderr="syntax error found",
            ),
        ]
        metrics = aggregate_test_metrics(results)
        assert "python" in metrics["languages"]
        assert metrics["languages"]["python"]["total"] == 2
        assert metrics["languages"]["python"]["successful"] == 1
        assert metrics["total_tests"] == 2

    def test_multiple_languages(self):
        results = [
            _make_test_result(language="python", execution_time=1.0),
            _make_test_result(test_id="t-2", language="go", execution_time=0.5),
        ]
        metrics = aggregate_test_metrics(results)
        assert "python" in metrics["languages"]
        assert "go" in metrics["languages"]

    def test_error_patterns(self):
        results = [
            _make_test_result(
                test_id="t-1", success=False, exit_code=1, stderr="SyntaxError: invalid syntax"
            ),
            _make_test_result(
                test_id="t-2", success=False, exit_code=1, stderr="ImportError: no module"
            ),
            _make_test_result(test_id="t-3", success=False, exit_code=1, stderr="Test timed out"),
            _make_test_result(
                test_id="t-4", success=False, exit_code=1, stderr="MemoryError: out of memory"
            ),
            _make_test_result(
                test_id="t-5", success=False, exit_code=1, stderr="Something went wrong"
            ),
        ]
        metrics = aggregate_test_metrics(results)
        assert "syntax_error" in metrics["error_patterns"]
        assert "import_error" in metrics["error_patterns"]
        assert "timeout" in metrics["error_patterns"]
        assert "memory_error" in metrics["error_patterns"]
        assert "unknown" in metrics["error_patterns"]

    def test_execution_times(self):
        results = [
            _make_test_result(language="python", execution_time=1.0),
            _make_test_result(test_id="t-2", language="python", execution_time=3.0),
        ]
        metrics = aggregate_test_metrics(results)
        assert metrics["execution_times"]["python"]["avg"] == 2.0
        assert metrics["execution_times"]["python"]["min"] == 1.0
        assert metrics["execution_times"]["python"]["max"] == 3.0


# ---------------------------------------------------------------------------
# Background task logging functions
# ---------------------------------------------------------------------------


class TestLogTestExecution:
    @pytest.mark.asyncio
    async def test_logs_successfully(self):
        test_result = _make_test_result()
        with patch("routers.testing_modules.testing_utils.get_metrics_service") as mock_metrics:
            mock_service = MagicMock()
            mock_service.record_test_execution = AsyncMock()
            mock_metrics.return_value = mock_service

            await _log_test_execution("user-1", test_result, {"language": "python"})
            mock_service.record_test_execution.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_error(self):
        test_result = _make_test_result()
        with patch("routers.testing_modules.testing_utils.get_metrics_service") as mock_metrics:
            mock_metrics.side_effect = AttributeError("no metrics service")
            # Should not raise
            await _log_test_execution("user-1", test_result, {})


class TestLogQualityAnalysis:
    @pytest.mark.asyncio
    async def test_logs_successfully(self):
        quality_result = QualityResult(
            analysis_id="q-1",
            success=True,
            overall_score=9.0,
            total_issues=0,
            issues=[],
            tools_results={},
            execution_time=1.0,
            created_at=datetime.now(UTC),
            language="python",
        )
        with patch("routers.testing_modules.testing_utils.get_metrics_service") as mock_metrics:
            mock_service = MagicMock()
            mock_service.record_quality_analysis = AsyncMock()
            mock_metrics.return_value = mock_service

            await _log_quality_analysis("user-1", quality_result)
            mock_service.record_quality_analysis.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_error(self):
        quality_result = MagicMock()
        quality_result.analysis_id = "q-1"
        with patch("routers.testing_modules.testing_utils.get_metrics_service") as mock_metrics:
            mock_metrics.side_effect = TypeError("bad")
            await _log_quality_analysis("user-1", quality_result)


class TestLogBatchExecution:
    @pytest.mark.asyncio
    async def test_logs_successfully(self):
        await _log_batch_execution("user-1", 10, 8, ["python", "go"])

    @pytest.mark.asyncio
    async def test_handles_error(self):
        # Pass invalid args that would cause attribute error
        with patch("routers.testing_modules.testing_utils.logger") as mock_logger:
            mock_logger.info.side_effect = AttributeError("boom")
            await _log_batch_execution("user-1", 10, 8, ["python"])
