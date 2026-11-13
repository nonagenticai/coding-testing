"""Tests for src/services/ai_test_integration_service.py.

Verifies:
- Service initialization with/without AI analysis
- execute_test_with_analysis routes correctly based on test success/failure
- Failure analysis workflow (timeout, exceptions, success)
- Batch analysis with concurrency control
- Framework detection logic
- Result integration maps all fields correctly
- Health check aggregation
- Cleanup and factory function
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from models.ai_test_failure_models import (
    ConfidenceLevel,
    SeverityLevel,
    TestFramework,
)
from models.testing_models import (
    TestRequest,
    TestResult,
    TestSecurityLevel,
)
from services.ai_failure_analysis_service import AIFailureAnalysisService
from services.ai_test_integration_service import (
    AITestIntegrationService,
    create_ai_test_integration_service,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_result(**overrides) -> TestResult:
    defaults = {
        "test_id": "test-001",
        "success": False,
        "stdout": "FAIL",
        "stderr": "AssertionError: assert 1 == 2",
        "exit_code": 1,
        "execution_time": 2.5,
        "created_at": datetime.now(UTC),
        "language": "python",
        "security_level": TestSecurityLevel.HIGH,
    }
    defaults.update(overrides)
    return TestResult(**defaults)


def _make_test_request(**overrides) -> TestRequest:
    defaults = {
        "code": "def add(a,b): return a+b",
        "test_code": "def test_add(): assert add(1,2)==3",
        "language": "python",
    }
    defaults.update(overrides)
    return TestRequest(**defaults)


@pytest.fixture
def mock_test_execution_service():
    svc = AsyncMock()
    svc.execute_test = AsyncMock(return_value=_make_test_result())
    svc.health_check = AsyncMock(return_value={"status": "healthy"})
    return svc


@pytest.fixture
def mock_ai_analysis_service():
    svc = AsyncMock(spec=AIFailureAnalysisService)
    svc.get_health_status = MagicMock(return_value={"status": "healthy"})
    svc.cleanup = AsyncMock()
    svc.analyze_test_failure = AsyncMock()
    return svc


@pytest.fixture
def integration_service(mock_test_execution_service, mock_ai_analysis_service):
    return AITestIntegrationService(
        test_execution_service=mock_test_execution_service,
        ai_analysis_service=mock_ai_analysis_service,
        enable_ai_analysis=True,
        analysis_timeout=5.0,
        max_concurrent_analyses=2,
    )


@pytest.fixture
def disabled_service(mock_test_execution_service):
    return AITestIntegrationService(
        test_execution_service=mock_test_execution_service,
        ai_analysis_service=None,
        enable_ai_analysis=False,
    )


def _make_analysis_result():
    """Build a MagicMock that looks like an AIAnalysisResult."""
    ar = MagicMock()
    ar.analysis_id = "a1"
    ar.timestamp = datetime.now(UTC)
    ar.analysis_confidence = ConfidenceLevel.HIGH
    ar.analysis_confidence_score = 0.85
    ar.processing_time = 1.0
    ar.ai_model_version = "v1"
    ar.root_causes = []
    ar.fix_suggestions = []
    ar.detected_patterns = []
    ar.assertion_details = None
    return ar


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_stores_ai_service(self, integration_service, mock_ai_analysis_service):
        assert integration_service.ai_analysis_service is mock_ai_analysis_service
        assert integration_service.enable_ai_analysis is True

    def test_disabled_has_no_ai_service(self, disabled_service):
        assert disabled_service.ai_analysis_service is None
        assert disabled_service.enable_ai_analysis is False

    def test_auto_creates_ai_service_when_none_provided(self, mock_test_execution_service):
        svc = AITestIntegrationService(
            test_execution_service=mock_test_execution_service,
            enable_ai_analysis=True,
            analysis_timeout=30.0,
            max_concurrent_analyses=4,
        )
        # Verify AIFailureAnalysisService was auto-created with forwarded config
        ai_svc = svc.ai_analysis_service
        assert type(ai_svc) is AIFailureAnalysisService
        assert ai_svc.max_concurrent_analyses == 4
        assert ai_svc.analysis_timeout == 30.0

    def test_metrics_start_at_zero(self, integration_service):
        metrics = integration_service._metrics
        assert metrics["total_tests"] == 0
        assert metrics["failed_tests"] == 0
        assert metrics["analyzed_failures"] == 0

    @pytest.mark.parametrize("timeout,max_conc", [(10.0, 5), (120.0, 1), (0.5, 10)])
    def test_custom_settings(self, mock_test_execution_service, timeout, max_conc):
        svc = AITestIntegrationService(
            test_execution_service=mock_test_execution_service,
            enable_ai_analysis=False,
            analysis_timeout=timeout,
            max_concurrent_analyses=max_conc,
        )
        assert svc.analysis_timeout == timeout
        assert svc.max_concurrent_analyses == max_conc


# ---------------------------------------------------------------------------
# execute_test_with_analysis
# ---------------------------------------------------------------------------


class TestExecuteTestWithAnalysis:
    @pytest.mark.asyncio
    async def test_passing_test_skips_analysis(
        self, integration_service, mock_test_execution_service
    ):
        """Passing tests should not trigger AI analysis."""
        mock_test_execution_service.execute_test.return_value = _make_test_result(success=True)
        request = _make_test_request()
        result = await integration_service.execute_test_with_analysis(request)
        assert result.success is True
        assert integration_service._metrics["total_tests"] == 1
        assert integration_service._metrics["failed_tests"] == 0

    @pytest.mark.asyncio
    async def test_failed_test_with_analysis_disabled(
        self, integration_service, mock_test_execution_service
    ):
        """Explicitly disabling analysis skips it even for failures."""
        request = _make_test_request()
        result = await integration_service.execute_test_with_analysis(
            request, enable_failure_analysis=False
        )
        assert result.success is False
        assert integration_service._metrics["total_tests"] == 1
        assert integration_service._metrics["failed_tests"] == 0

    @pytest.mark.asyncio
    async def test_failed_test_analysis_fails(self, integration_service, mock_ai_analysis_service):
        """When analysis returns unsuccessful response, metrics track the failure."""
        mock_response = MagicMock()
        mock_response.success = False
        mock_response.error_message = "model unavailable"
        mock_response.analysis_result = None
        mock_ai_analysis_service.analyze_test_failure.return_value = mock_response

        result = await integration_service.execute_test_with_analysis(_make_test_request())
        assert result.success is False
        assert result.test_id == "test-001"
        assert integration_service._metrics["total_tests"] == 1
        assert integration_service._metrics["failed_tests"] == 1
        assert integration_service._metrics["analyzed_failures"] == 1
        assert integration_service._metrics["analysis_failures"] == 1
        assert integration_service._metrics["analysis_successes"] == 0

    @pytest.mark.asyncio
    async def test_failed_test_analysis_exception(
        self, integration_service, mock_ai_analysis_service
    ):
        """When _integrate_analysis_results raises, the error is captured in metadata."""
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.analysis_result = MagicMock()
        mock_ai_analysis_service.analyze_test_failure.return_value = mock_response

        integration_service._integrate_analysis_results = MagicMock(
            side_effect=RuntimeError("boom")
        )

        result = await integration_service.execute_test_with_analysis(_make_test_request())
        assert result.success is False
        assert result.test_metadata is not None
        assert "ai_analysis_error" in result.test_metadata
        assert "boom" in result.test_metadata["ai_analysis_error"]

    @pytest.mark.asyncio
    async def test_execute_test_raises_propagates(
        self, integration_service, mock_test_execution_service
    ):
        """Runtime errors from execute_test propagate and bump integration_errors."""
        mock_test_execution_service.execute_test.side_effect = RuntimeError("crash")
        with pytest.raises(RuntimeError, match="crash"):
            await integration_service.execute_test_with_analysis(_make_test_request())
        assert integration_service._metrics["integration_errors"] == 1


# ---------------------------------------------------------------------------
# analyze_existing_failure
# ---------------------------------------------------------------------------


class TestAnalyzeExistingFailure:
    @pytest.mark.asyncio
    async def test_disabled_returns_none(self, disabled_service):
        assert await disabled_service.analyze_existing_failure(_make_test_result()) is None

    @pytest.mark.asyncio
    async def test_with_source_and_test_code(self, integration_service, mock_ai_analysis_service):
        analysis = _make_analysis_result()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.analysis_result = analysis
        mock_ai_analysis_service.analyze_test_failure.return_value = mock_response

        result = await integration_service.analyze_existing_failure(
            _make_test_result(),
            source_code="def add(a,b): return a+b",
            test_code="def test_add(): assert add(1,2)==3",
        )
        # Verify the exact analysis mock was returned
        assert result is analysis
        assert result.analysis_id == "a1"
        assert result.analysis_confidence == ConfidenceLevel.HIGH
        # Verify analyze_test_failure was called with a FailureAnalysisRequest
        call_args = mock_ai_analysis_service.analyze_test_failure.call_args
        req = call_args[0][0]
        assert req.failure_data.failure_id == "test-001"

    @pytest.mark.asyncio
    async def test_exception_returns_none(self, integration_service, mock_ai_analysis_service):
        mock_ai_analysis_service.analyze_test_failure.side_effect = RuntimeError("err")
        result = await integration_service.analyze_existing_failure(_make_test_result())
        assert result is None


# ---------------------------------------------------------------------------
# batch_analyze_failures
# ---------------------------------------------------------------------------


class TestBatchAnalyzeFailures:
    @pytest.mark.asyncio
    async def test_disabled_returns_all_nones(self, disabled_service):
        results = await disabled_service.batch_analyze_failures(
            [_make_test_result(), _make_test_result()]
        )
        assert results == [None, None]

    @pytest.mark.asyncio
    async def test_no_failed_tests(self, integration_service):
        passing = _make_test_result(success=True)
        results = await integration_service.batch_analyze_failures([passing, passing])
        assert results == [None, None]

    @pytest.mark.asyncio
    async def test_batch_preserves_indices(self, integration_service, mock_ai_analysis_service):
        """Passing test at index 1 gets None, failed tests at 0 and 2 get results."""
        analysis = _make_analysis_result()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.analysis_result = analysis
        mock_ai_analysis_service.analyze_test_failure.return_value = mock_response

        results = await integration_service.batch_analyze_failures(
            [_make_test_result(), _make_test_result(success=True), _make_test_result()],
            max_concurrent=2,
        )
        assert len(results) == 3
        # Index 0 and 2 are failed tests that got analysis; index 1 is passing → None
        assert results[0] is analysis
        assert results[1] is None
        assert results[2] is analysis
        # AI service was called exactly twice (for the two failed tests)
        assert mock_ai_analysis_service.analyze_test_failure.call_count == 2

    @pytest.mark.asyncio
    async def test_batch_handles_exception(self, integration_service, mock_ai_analysis_service):
        mock_ai_analysis_service.analyze_test_failure.side_effect = RuntimeError("err")
        results = await integration_service.batch_analyze_failures([_make_test_result()])
        assert results[0] is None


# ---------------------------------------------------------------------------
# _analyze_test_failure internal
# ---------------------------------------------------------------------------


class TestAnalyzeTestFailureInternal:
    @pytest.mark.asyncio
    async def test_no_ai_service_returns_none(self, integration_service):
        integration_service.ai_analysis_service = None
        result = await integration_service._analyze_test_failure(
            _make_test_result(), _make_test_request()
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_timeout_returns_none(self, integration_service, mock_ai_analysis_service):
        async def slow_analysis(*args, **kwargs):
            await asyncio.sleep(100)

        mock_ai_analysis_service.analyze_test_failure.side_effect = slow_analysis
        integration_service.analysis_timeout = 0.01

        result = await integration_service._analyze_test_failure(
            _make_test_result(), _make_test_request()
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_success_returns_analysis_result(
        self, integration_service, mock_ai_analysis_service
    ):
        mock_response = MagicMock()
        mock_response.success = True
        analysis = _make_analysis_result()
        mock_response.analysis_result = analysis
        mock_ai_analysis_service.analyze_test_failure.return_value = mock_response

        result = await integration_service._analyze_test_failure(
            _make_test_result(), _make_test_request()
        )
        assert result is analysis

    @pytest.mark.asyncio
    async def test_failed_response_returns_none(
        self, integration_service, mock_ai_analysis_service
    ):
        mock_response = MagicMock()
        mock_response.success = False
        mock_response.error_message = "bad"
        mock_response.analysis_result = None
        mock_ai_analysis_service.analyze_test_failure.return_value = mock_response

        result = await integration_service._analyze_test_failure(
            _make_test_result(), _make_test_request()
        )
        assert result is None


# ---------------------------------------------------------------------------
# _create_analysis_request
# ---------------------------------------------------------------------------


class TestCreateAnalysisRequest:
    def test_creates_request_with_valid_fields(self, integration_service):
        test_result = _make_test_result()
        test_request = _make_test_request()
        request = integration_service._create_analysis_request(test_result, test_request)

        # request_id is a UUID v7 string (36 chars with hyphens)
        assert len(request.request_id) == 36
        assert request.request_id.count("-") == 4

        # Failure data carries through from test result
        assert request.failure_data.failure_id == "test-001"
        assert request.failure_data.exit_code == 1
        assert request.failure_data.execution_time == 2.5
        assert "FAIL" in request.failure_data.raw_output
        assert "AssertionError" in request.failure_data.raw_output

        # Default analysis options
        assert request.include_suggestions is True
        assert request.include_patterns is True
        assert request.max_root_causes == 5
        assert request.max_suggestions == 10
        assert request.minimum_confidence == 0.1

    def test_no_test_code_gives_none_file_metadata(self, integration_service):
        req = _make_test_request(test_code=None)
        request = integration_service._create_analysis_request(_make_test_result(), req)
        assert request.failure_data.context.test_file is None
        # Context should still have test name and framework
        assert request.failure_data.context.test_name == "test_test-001"
        assert request.failure_data.context.framework == TestFramework.PYTEST


# ---------------------------------------------------------------------------
# _determine_test_framework
# ---------------------------------------------------------------------------


class TestDetermineTestFramework:
    @pytest.mark.parametrize(
        "language,stdout,expected",
        [
            ("python", "pytest collected 5 items", TestFramework.PYTEST),
            ("python", "unittest: Ran 5 tests", TestFramework.UNITTEST),
            ("python", "OK", TestFramework.PYTEST),  # default for python
            ("javascript", "jest passed", TestFramework.JEST),
            ("typescript", "Tests: 3 passed", TestFramework.JEST),
            ("go", "ok", TestFramework.UNKNOWN),
            ("rust", "running 3 tests", TestFramework.UNKNOWN),
        ],
    )
    def test_framework_detection(self, integration_service, language, stdout, expected):
        result = _make_test_result(language=language, stdout=stdout)
        fw = integration_service._determine_test_framework(result)
        assert fw == expected


# ---------------------------------------------------------------------------
# _extract_execution_environment
# ---------------------------------------------------------------------------


class TestExtractExecutionEnvironment:
    def test_basic_fields(self, integration_service):
        result = _make_test_result()
        env = integration_service._extract_execution_environment(result)
        assert env["language"] == "python"
        assert env["security_level"] == "high"
        assert env["execution_time"] == 2.5
        assert env["container_name"] is None

    def test_metadata_merged(self, integration_service):
        result = _make_test_result(test_metadata={"custom_key": "value"})
        env = integration_service._extract_execution_environment(result)
        assert env["custom_key"] == "value"
        assert env["language"] == "python"


# ---------------------------------------------------------------------------
# _integrate_analysis_results
# ---------------------------------------------------------------------------


class TestIntegrateAnalysisResults:
    def test_basic_analysis_metadata(self, integration_service):
        test_result = _make_test_result(test_metadata=None)
        analysis = _make_analysis_result()
        result = integration_service._integrate_analysis_results(test_result, analysis)
        ai = result.test_metadata["ai_analysis"]
        assert ai["analysis_id"] == "a1"
        assert ai["confidence"] == ConfidenceLevel.HIGH
        assert ai["confidence_score"] == 0.85
        assert ai["processing_time"] == 1.0
        assert ai["ai_model_version"] == "v1"
        # Timestamp is an ISO string
        assert "T" in ai["timestamp"]
        # No root causes / fix suggestions / patterns since analysis has empty lists
        assert "root_causes" not in result.test_metadata
        assert "fix_suggestions" not in result.test_metadata
        assert "detected_patterns" not in result.test_metadata

    def test_root_causes_mapped(self, integration_service):
        analysis = _make_analysis_result()
        cause = MagicMock()
        cause.cause_id = "c1"
        cause.category = "logic"
        cause.description = "Wrong logic"
        cause.severity = SeverityLevel.HIGH
        cause.confidence = ConfidenceLevel.HIGH
        cause.confidence_score = 0.9
        cause.evidence = ["line 10"]
        analysis.root_causes = [cause]

        result = integration_service._integrate_analysis_results(_make_test_result(), analysis)
        assert len(result.test_metadata["root_causes"]) == 1
        rc = result.test_metadata["root_causes"][0]
        assert rc == {
            "id": "c1",
            "category": "logic",
            "description": "Wrong logic",
            "severity": SeverityLevel.HIGH,
            "confidence": ConfidenceLevel.HIGH,
            "confidence_score": 0.9,
            "evidence": ["line 10"],
        }

    def test_fix_suggestions_mapped(self, integration_service):
        analysis = _make_analysis_result()
        suggestion = MagicMock()
        suggestion.suggestion_id = "s1"
        suggestion.title = "Fix It"
        suggestion.description = "Do this"
        suggestion.fix_type = "code_change"
        suggestion.priority = 1
        suggestion.confidence = ConfidenceLevel.MEDIUM
        suggestion.confidence_score = 0.6
        suggestion.estimated_effort = "5 min"
        suggestion.code_examples = ["code"]
        suggestion.files_to_modify = ["test.py"]
        suggestion.verification_steps = ["run tests"]
        analysis.fix_suggestions = [suggestion]

        result = integration_service._integrate_analysis_results(_make_test_result(), analysis)
        assert len(result.test_metadata["fix_suggestions"]) == 1
        fix = result.test_metadata["fix_suggestions"][0]
        assert fix == {
            "id": "s1",
            "title": "Fix It",
            "description": "Do this",
            "fix_type": "code_change",
            "priority": 1,
            "confidence": ConfidenceLevel.MEDIUM,
            "confidence_score": 0.6,
            "estimated_effort": "5 min",
            "code_examples": ["code"],
            "files_to_modify": ["test.py"],
            "verification_steps": ["run tests"],
        }

    def test_detected_patterns_mapped(self, integration_service):
        analysis = _make_analysis_result()
        pattern = MagicMock()
        pattern.pattern_id = "p1"
        pattern.pattern_name = "Flaky Test"
        pattern.pattern_type = "intermittent"
        pattern.confidence = ConfidenceLevel.LOW
        pattern.confidence_score = 0.3
        pattern.frequency = 5
        pattern.typical_causes = ["race condition"]
        pattern.common_fixes = ["add retry"]
        analysis.detected_patterns = [pattern]

        result = integration_service._integrate_analysis_results(_make_test_result(), analysis)
        assert len(result.test_metadata["detected_patterns"]) == 1
        pat = result.test_metadata["detected_patterns"][0]
        assert pat == {
            "id": "p1",
            "name": "Flaky Test",
            "type": "intermittent",
            "confidence": ConfidenceLevel.LOW,
            "confidence_score": 0.3,
            "frequency": 5,
            "typical_causes": ["race condition"],
            "common_fixes": ["add retry"],
        }

    def test_assertion_details_mapped(self, integration_service):
        analysis = _make_analysis_result()
        assertion = MagicMock()
        assertion.assertion_type = "assertEqual"
        assertion.expected_value = "5"
        assertion.actual_value = "3"
        assertion.comparison_operator = "=="
        assertion.assertion_message = "Values differ"
        assertion.value_difference = "2"
        analysis.assertion_details = assertion

        result = integration_service._integrate_analysis_results(_make_test_result(), analysis)
        ad = result.test_metadata["assertion_details"]
        assert ad == {
            "assertion_type": "assertEqual",
            "expected_value": "5",
            "actual_value": "3",
            "comparison_operator": "==",
            "assertion_message": "Values differ",
            "value_difference": "2",
        }


# ---------------------------------------------------------------------------
# Health & metrics
# ---------------------------------------------------------------------------


class TestHealthAndMetrics:
    @pytest.mark.asyncio
    async def test_disabled_health(self, disabled_service):
        result = await disabled_service.get_analysis_service_health()
        assert result["status"] == "disabled"
        assert result["service"] == "ai_analysis"
        assert "not enabled" in result["message"]

    @pytest.mark.asyncio
    async def test_healthy(self, integration_service):
        result = await integration_service.get_analysis_service_health()
        assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_exception(self, integration_service, mock_ai_analysis_service):
        mock_ai_analysis_service.get_health_status.side_effect = RuntimeError("err")
        result = await integration_service.get_analysis_service_health()
        assert result["status"] == "error"
        assert result["service"] == "ai_analysis"
        assert "err" in result["message"]

    def test_metrics_structure(self, integration_service):
        metrics = integration_service.get_integration_metrics()
        assert metrics["service"] == "ai_test_integration"
        assert metrics["metrics"] == {
            "total_tests": 0,
            "failed_tests": 0,
            "analyzed_failures": 0,
            "analysis_successes": 0,
            "analysis_failures": 0,
            "integration_errors": 0,
        }
        assert metrics["analysis_enabled"] is True
        assert metrics["analysis_service_available"] is True
        assert metrics["analysis_timeout"] == 5.0
        assert metrics["max_concurrent_analyses"] == 2

    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self, integration_service):
        result = await integration_service.health_check()
        assert result["status"] == "healthy"
        assert result["service"] == "ai_test_integration"
        assert result["checks"]["test_execution"]["status"] == "healthy"
        assert result["checks"]["ai_analysis"]["status"] == "healthy"
        assert result["metrics"]["service"] == "ai_test_integration"

    @pytest.mark.asyncio
    async def test_health_check_degraded(self, integration_service, mock_test_execution_service):
        """One failing check makes status degraded."""
        mock_test_execution_service.health_check.side_effect = RuntimeError("down")
        result = await integration_service.health_check()
        assert result["status"] == "degraded"
        assert result["checks"]["test_execution"]["status"] == "error"
        assert "down" in result["checks"]["test_execution"]["error"]
        # AI analysis is still healthy
        assert result["checks"]["ai_analysis"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_ai_error(self, integration_service, mock_ai_analysis_service):
        mock_ai_analysis_service.get_health_status.side_effect = RuntimeError("err")
        result = await integration_service.health_check()
        assert result["status"] == "degraded"
        assert result["checks"]["ai_analysis"]["status"] == "error"
        # Test execution is still healthy
        assert result["checks"]["test_execution"]["status"] == "healthy"


# ---------------------------------------------------------------------------
# Cleanup & factory
# ---------------------------------------------------------------------------


class TestCleanupAndFactory:
    @pytest.mark.asyncio
    async def test_cleanup_calls_ai_service(self, integration_service, mock_ai_analysis_service):
        await integration_service.cleanup()
        mock_ai_analysis_service.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_without_service(self, disabled_service):
        await disabled_service.cleanup()  # should not raise

    @pytest.mark.asyncio
    async def test_cleanup_handles_exception(self, integration_service, mock_ai_analysis_service):
        mock_ai_analysis_service.cleanup.side_effect = RuntimeError("err")
        await integration_service.cleanup()  # should not raise

    def test_factory_creates_instance(self, mock_test_execution_service):
        svc = create_ai_test_integration_service(
            test_execution_service=mock_test_execution_service,
            enable_ai_analysis=False,
            analysis_timeout=99.0,
            max_concurrent_analyses=7,
        )
        assert type(svc) is AITestIntegrationService
        assert svc.enable_ai_analysis is False
        assert svc.ai_analysis_service is None
        assert svc.analysis_timeout == 99.0
        assert svc.max_concurrent_analyses == 7
        assert svc.test_execution_service is mock_test_execution_service
