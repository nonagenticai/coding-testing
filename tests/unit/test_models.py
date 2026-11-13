"""Model validation tests."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from models.testing_models import (
    QualityRequest,
    ResourceLimits,
    TestRequest,
    TestResult,
    TestSecurityLevel,
    TestType,
)


class TestTestRequest:
    def test_valid_request(self):
        req = TestRequest(
            code="def add(a, b): return a + b",
            test_code="def test_add(): assert add(1, 2) == 3",
            language="python",
        )
        assert req.language == "python"
        assert req.security_level == TestSecurityLevel.HIGH

    def test_language_validation(self):
        with pytest.raises(ValidationError):
            TestRequest(
                code="code",
                test_code="test",
                language="brainfuck",
            )

    def test_empty_language_rejected(self):
        with pytest.raises(ValidationError):
            TestRequest(
                code="code",
                test_code="test",
                language="",
            )

    def test_defaults_code_path_when_no_code(self):
        """When no code or code_path is provided, model defaults code_path to '.' for container scanning."""
        req = TestRequest(
            test_code="test",
            language="python",
        )
        assert req.code is None
        assert req.code_path == "."

    def test_defaults_test_path_when_no_test_code(self):
        """When no test_code or test_path is provided, model defaults test_path to '.' for container scanning."""
        req = TestRequest(
            code="code",
            language="python",
        )
        assert req.test_code is None
        assert req.test_path == "."

    def test_file_path_based_request(self):
        req = TestRequest(
            code_path="/app/code.py",
            test_path="/app/test_code.py",
            language="go",
        )
        assert req.code_file_path_resolved == "/app/code.py"
        assert req.test_file_path_resolved == "/app/test_code.py"


class TestTestResult:
    def test_valid_result(self):
        result = TestResult(
            test_id="test-123",
            success=True,
            exit_code=0,
            execution_time=1.5,
            created_at=datetime.now(UTC),
            language="python",
            security_level=TestSecurityLevel.HIGH,
        )
        assert result.success is True
        assert result.exit_code == 0


class TestQualityRequest:
    def test_valid_request(self):
        req = QualityRequest(
            container_name="test-container",
            code="print('hello')",
            language="python",
        )
        assert req.container_name == "test-container"

    def test_defaults_code_path_when_no_code_source(self):
        """When no code source is provided, model defaults code_path to '.' for container scanning."""
        req = QualityRequest(container_name="test-container")
        assert req.code is None
        assert req.code_content is None
        assert req.code_path == "."


class TestEnums:
    def test_test_type_values(self):
        assert TestType.UNIT == "unit"
        assert TestType.INTEGRATION == "integration"

    def test_security_level_values(self):
        assert TestSecurityLevel.LOW == "low"
        assert TestSecurityLevel.HIGH == "high"
        assert TestSecurityLevel.CRITICAL == "critical"


class TestResourceLimits:
    def test_valid_limits(self):
        limits = ResourceLimits(memory_mb=512, cpu_cores=2.0, timeout_seconds=60)
        assert limits.memory_mb == 512

    def test_memory_bounds(self):
        with pytest.raises(ValidationError):
            ResourceLimits(memory_mb=50)  # below 128 minimum


# ---------------------------------------------------------------------------
# LogLevel
# ---------------------------------------------------------------------------

from models.logging_models import LogLevel  # noqa: E402


class TestLogLevel:
    def test_from_lowercase(self):
        """Covers line 25: LogLevel.from_lowercase."""
        assert LogLevel.from_lowercase("info") == LogLevel.INFO
        assert LogLevel.from_lowercase("debug") == LogLevel.DEBUG
        assert LogLevel.from_lowercase("error") == LogLevel.ERROR

    def test_to_lowercase(self):
        """Covers line 29: LogLevel.to_lowercase."""
        assert LogLevel.INFO.to_lowercase() == "info"
        assert LogLevel.DEBUG.to_lowercase() == "debug"
        assert LogLevel.CRITICAL.to_lowercase() == "critical"


# ---------------------------------------------------------------------------
# StandardErrorResponse
# ---------------------------------------------------------------------------

from models.error_models import StandardErrorResponse  # noqa: E402


class TestStandardErrorResponse:
    def test_serialize_timestamp_naive(self):
        """Covers lines 58-60: naive datetime gets UTC tzinfo."""
        naive_dt = datetime(2025, 1, 1, 12, 0, 0)  # no tzinfo
        resp = StandardErrorResponse(
            message="test error",
            request_id="req-123",
            path="/api/test",
        )
        result = resp.serialize_timestamp(naive_dt)
        assert "+00:00" in result or "Z" in result

    def test_serialize_timestamp_aware(self):
        """Covers normal path: aware datetime passes through."""
        aware_dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        resp = StandardErrorResponse(
            message="test error",
            request_id="req-123",
            path="/api/test",
        )
        result = resp.serialize_timestamp(aware_dt)
        assert "2025-01-01" in result
