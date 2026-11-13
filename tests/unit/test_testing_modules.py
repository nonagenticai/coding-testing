"""Unit tests for testing_modules (health_and_metrics, quality_analysis, test_discovery, test_execution, testing_utils)."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

# ============================================================================
# testing_utils.py tests
# ============================================================================


class TestValidateTestRequest:
    """Test validate_test_request utility."""

    def test_valid_request(self):
        """Test validation with valid request data."""
        from routers.testing_modules.testing_utils import validate_test_request

        data = {"code": "print('hello')", "language": "python", "timeout": 30}
        result = validate_test_request(data)

        assert result["code"] == "print('hello')"
        assert result["language"] == "python"
        assert result["timeout"] == 30

    def test_empty_request_raises(self):
        """Test validation with empty request raises."""
        from routers.testing_modules.testing_utils import validate_test_request

        with pytest.raises(HTTPException) as exc_info:
            validate_test_request({})
        assert exc_info.value.status_code == 400

    def test_none_request_raises(self):
        """Test validation with None request raises."""
        from routers.testing_modules.testing_utils import validate_test_request

        with pytest.raises(HTTPException) as exc_info:
            validate_test_request(None)
        assert exc_info.value.status_code == 400

    def test_missing_code_raises(self):
        """Test validation with missing code raises."""
        from routers.testing_modules.testing_utils import validate_test_request

        with pytest.raises(HTTPException) as exc_info:
            validate_test_request({"language": "python"})
        assert exc_info.value.status_code == 400

    def test_empty_code_raises(self):
        """Test validation with empty code raises."""
        from routers.testing_modules.testing_utils import validate_test_request

        with pytest.raises(HTTPException) as exc_info:
            validate_test_request({"code": "   "})
        assert exc_info.value.status_code == 400

    def test_unsupported_language_raises(self):
        """Test validation with unsupported language raises."""
        from routers.testing_modules.testing_utils import validate_test_request

        with pytest.raises(HTTPException) as exc_info:
            validate_test_request({"code": "x=1", "language": "cobol"})
        assert exc_info.value.status_code == 400

    def test_default_language(self):
        """Test default language is python."""
        from routers.testing_modules.testing_utils import validate_test_request

        result = validate_test_request({"code": "x=1"})
        assert result["language"] == "python"

    def test_timeout_capped(self):
        """Test timeout is capped at 300."""
        from routers.testing_modules.testing_utils import validate_test_request

        result = validate_test_request({"code": "x=1", "timeout": 600})
        assert result["timeout"] == 300

    def test_memory_limit_capped(self):
        """Test memory limit is capped at 512."""
        from routers.testing_modules.testing_utils import validate_test_request

        result = validate_test_request({"code": "x=1", "memory_limit": 1024})
        assert result["memory_limit"] == 512


class TestFormatTestResults:
    """Test format_test_results utility."""

    def test_empty_results(self):
        """Test formatting empty results."""
        from routers.testing_modules.testing_utils import format_test_results

        result = format_test_results([])

        assert result["total_tests"] == 0
        assert result["successful"] == 0
        assert result["failed"] == 0
        assert result["avg_execution_time"] == 0.0

    def test_mixed_results(self):
        """Test formatting mixed success/failure results."""
        from routers.testing_modules.testing_utils import format_test_results

        r1 = MagicMock()
        r1.test_id = "t1"
        r1.success = True
        r1.language = "python"
        r1.exit_code = 0
        r1.stdout = "ok"
        r1.stderr = ""
        r1.execution_time = 100.0
        r1.created_at = datetime.now(UTC)

        r2 = MagicMock()
        r2.test_id = "t2"
        r2.success = False
        r2.language = "python"
        r2.exit_code = 1
        r2.stdout = ""
        r2.stderr = "error"
        r2.execution_time = 200.0
        r2.created_at = datetime.now(UTC)

        result = format_test_results([r1, r2])

        assert result["total_tests"] == 2
        assert result["successful"] == 1
        assert result["failed"] == 1
        assert result["success_rate"] == 0.5
        assert result["avg_execution_time"] == 150.0


class TestAggregateTestMetrics:
    """Test aggregate_test_metrics utility."""

    def test_empty_results(self):
        """Test aggregation with empty results."""
        from routers.testing_modules.testing_utils import aggregate_test_metrics

        result = aggregate_test_metrics([])

        assert result["languages"] == {}
        assert result["success_rate"] == 0.0

    def test_with_results(self):
        """Test aggregation with results."""
        from routers.testing_modules.testing_utils import aggregate_test_metrics

        r1 = MagicMock()
        r1.language = "python"
        r1.success = True
        r1.execution_time = 100.0
        r1.stderr = ""

        r2 = MagicMock()
        r2.language = "python"
        r2.success = False
        r2.execution_time = 200.0
        r2.stderr = "syntax error in code"

        result = aggregate_test_metrics([r1, r2])

        assert "python" in result["languages"]
        assert result["languages"]["python"]["total"] == 2
        assert result["languages"]["python"]["successful"] == 1
        assert result["total_tests"] == 2
        assert result["success_rate"] == 0.5

    def test_error_pattern_detection(self):
        """Test error pattern categorization."""
        from routers.testing_modules.testing_utils import aggregate_test_metrics

        r1 = MagicMock()
        r1.language = "python"
        r1.success = False
        r1.execution_time = 10.0
        r1.stderr = "ImportError: No module named 'foo'"

        r2 = MagicMock()
        r2.language = "python"
        r2.success = False
        r2.execution_time = 10.0
        r2.stderr = "SyntaxError: invalid syntax"

        r3 = MagicMock()
        r3.language = "python"
        r3.success = False
        r3.execution_time = 10.0
        r3.stderr = "timeout exceeded"

        result = aggregate_test_metrics([r1, r2, r3])

        assert "import_error" in result["error_patterns"]
        assert "syntax_error" in result["error_patterns"]
        assert "timeout" in result["error_patterns"]


class TestLogTestExecution:
    """Test _log_test_execution background task with real models."""

    def _make_test_result(self, **overrides):
        from models.testing_models import TestResult, TestSecurityLevel

        defaults = {
            "test_id": "t1",
            "success": True,
            "stdout": "PASS",
            "stderr": "",
            "exit_code": 0,
            "execution_time": 100.0,
            "created_at": datetime.now(UTC),
            "language": "python",
            "security_level": TestSecurityLevel.MEDIUM,
        }
        defaults.update(overrides)
        return TestResult(**defaults)

    async def test_log_test_execution_sends_correct_data(self):
        """Test logging sends structured data with correct fields from real model."""
        from routers.testing_modules.testing_utils import _log_test_execution

        result = self._make_test_result(test_id="exec-42", language="go", execution_time=250.0)

        captured_entry = {}
        mock_metrics = AsyncMock()

        async def capture_record(entry):
            captured_entry.update(entry)

        mock_metrics.record_test_execution = capture_record

        with patch(
            "routers.testing_modules.testing_utils.get_metrics_service",
            return_value=mock_metrics,
        ):
            await _log_test_execution("user1", result, {"language": "go"})

        assert captured_entry["event_type"] == "test_execution"
        assert captured_entry["test_id"] == "exec-42"
        assert captured_entry["success"] is True
        assert captured_entry["language"] == "go"
        assert captured_entry["execution_time"] == 250.0
        assert captured_entry["exit_code"] == 0
        assert captured_entry["user_id"] == "user1"

    async def test_log_test_execution_handles_error(self):
        """Test logging handles errors gracefully without raising."""
        from routers.testing_modules.testing_utils import _log_test_execution

        result = self._make_test_result()

        with patch(
            "routers.testing_modules.testing_utils.get_metrics_service",
            side_effect=AttributeError("no metrics"),
        ):
            # Should not raise
            await _log_test_execution("user1", result, {})


class TestLogQualityAnalysis:
    """Test _log_quality_analysis background task with real models."""

    def _make_quality_result(self, **overrides):
        from models.testing_models import QualityResult

        defaults = {
            "analysis_id": "a1",
            "success": True,
            "overall_score": 85.0,
            "total_issues": 3,
            "issues": [],
            "tools_results": {"ruff": {"success": True}, "ruff-format": {"success": True}},
            "execution_time": 200.0,
            "created_at": datetime.now(UTC),
            "language": "python",
        }
        defaults.update(overrides)
        return QualityResult(**defaults)

    async def test_log_quality_analysis_sends_correct_data(self):
        """Test logging sends structured data with correct fields from real model."""
        from routers.testing_modules.testing_utils import _log_quality_analysis

        result = self._make_quality_result(
            analysis_id="qa-77",
            overall_score=92.5,
            total_issues=0,  # model validator syncs to len(issues)
            tools_results={"ruff": {}, "ty": {}, "ruff-format": {}},
        )

        captured_entry = {}
        mock_metrics = AsyncMock()

        async def capture_record(entry):
            captured_entry.update(entry)

        mock_metrics.record_quality_analysis = capture_record

        with patch(
            "routers.testing_modules.testing_utils.get_metrics_service",
            return_value=mock_metrics,
        ):
            await _log_quality_analysis("user1", result)

        assert captured_entry["event_type"] == "quality_analysis"
        assert captured_entry["analysis_id"] == "qa-77"
        assert captured_entry["overall_score"] == 92.5
        # total_issues auto-corrected by model_validator to len(issues)=0
        assert captured_entry["total_issues"] == 0
        assert set(captured_entry["tools_used"]) == {"ruff", "ty", "ruff-format"}
        assert captured_entry["user_id"] == "user1"

    async def test_log_quality_analysis_handles_error(self):
        """Test quality analysis logging handles errors gracefully without raising."""
        from routers.testing_modules.testing_utils import _log_quality_analysis

        result = self._make_quality_result()

        with patch(
            "routers.testing_modules.testing_utils.get_metrics_service",
            side_effect=TypeError("bad metrics"),
        ):
            # Should not raise
            await _log_quality_analysis("user1", result)


class TestLogBatchExecution:
    """Test _log_batch_execution background task."""

    async def test_log_batch_success(self):
        """Test batch execution logging."""
        from routers.testing_modules.testing_utils import _log_batch_execution

        # Should not raise
        await _log_batch_execution("user1", 10, 8, ["python", "javascript"])

    async def test_log_batch_handles_error(self):
        """Test batch logging handles attribute errors."""
        from routers.testing_modules.testing_utils import _log_batch_execution

        # Should not raise even with bad inputs
        await _log_batch_execution("user1", 0, 0, [])


# ============================================================================
# health_and_metrics.py tests
# ============================================================================


class TestHealthAndMetrics:
    """Test health and metrics endpoints."""

    async def test_health_check_unhealthy_when_service_fails(self):
        """Test health check returns unhealthy when execution service fails."""
        from routers.testing_modules.health_and_metrics import testing_health_check

        with patch(
            "routers.testing_modules.health_and_metrics.get_execution_service",
            side_effect=RuntimeError("Docker not available"),
        ):
            result = await testing_health_check(tenant_id="n8n-dev")

        # When get_execution_service raises RuntimeError, the health check
        # falls into the ValueError/RuntimeError catch block and returns
        # a minimal unhealthy response
        assert result["status"] == "unhealthy"
        assert "services" in result

    async def test_health_check_structure(self):
        """Test health check response structure."""
        from routers.testing_modules.health_and_metrics import testing_health_check

        with patch(
            "routers.testing_modules.health_and_metrics.get_execution_service",
            side_effect=RuntimeError("unavailable"),
        ):
            result = await testing_health_check(tenant_id="n8n-dev")

        assert "status" in result
        assert "timestamp" in result
        assert "services" in result

    async def test_metrics_summary_exception(self):
        """Test metrics summary raises 500 on unexpected error."""
        from routers.testing_modules.health_and_metrics import (
            get_testing_metrics_summary,
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        with patch(
            "routers.testing_modules.health_and_metrics.get_metrics_service",
            side_effect=RuntimeError("metrics down"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_testing_metrics_summary(
                    days=7,
                    current_user=mock_user,
                    tenant_id="n8n-dev",
                )
            assert exc_info.value.status_code == 500

    async def test_metrics_summary_success(self):
        """Test metrics summary returns data on success."""
        from routers.testing_modules.health_and_metrics import (
            get_testing_metrics_summary,
        )

        mock_metrics = AsyncMock()
        mock_metrics.get_test_execution_analytics = AsyncMock(
            return_value={
                "period_days": 7,
                "test_executions": 42,
                "quality_analyses": 10,
                "resource_usage": {},
                "languages": ["python"],
            }
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        with patch(
            "routers.testing_modules.health_and_metrics.get_metrics_service",
            return_value=mock_metrics,
        ):
            result = await get_testing_metrics_summary(
                days=7,
                current_user=mock_user,
                tenant_id="n8n-dev",
            )

        assert result["test_executions"] == 42
        assert result["period_days"] == 7

    async def test_metrics_summary_fallback(self):
        """Test metrics summary uses fallback when method missing."""
        from routers.testing_modules.health_and_metrics import (
            get_testing_metrics_summary,
        )

        mock_metrics = AsyncMock()
        mock_metrics.get_test_execution_analytics = AsyncMock(
            side_effect=AttributeError("no such method")
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        with patch(
            "routers.testing_modules.health_and_metrics.get_metrics_service",
            return_value=mock_metrics,
        ):
            result = await get_testing_metrics_summary(
                days=7,
                current_user=mock_user,
                tenant_id="n8n-dev",
            )

        # Should fall back to zeros
        assert result["test_executions"] == 0


# ============================================================================
# test_discovery.py tests
# ============================================================================


class TestTestDiscovery:
    """Test test_discovery endpoints."""

    async def test_get_supported_tools_for_language(self):
        """Test getting supported tools for a specific language."""
        from routers.testing_modules.test_discovery import get_supported_tools

        mock_service = MagicMock()
        mock_service._get_supported_tools = MagicMock(return_value=["ruff", "ruff-format", "ty"])

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_user.is_admin = MagicMock(return_value=False)
        mock_user.effective_tenant_id = "n8n-dev"

        result = await get_supported_tools(
            language="python",
            container_name=None,
            user=mock_user,
            execution_service=mock_service,
        )

        assert result["language"] == "python"
        assert result["count"] == 3
        assert "ruff" in result["tools"]

    async def test_get_supported_tools_all_languages(self):
        """Test getting supported tools for all languages."""
        from routers.testing_modules.test_discovery import get_supported_tools

        mock_service = MagicMock()
        mock_service._get_supported_tools = MagicMock(
            side_effect=lambda lang: {
                "python": ["ruff", "ruff-format"],
                "javascript": ["eslint"],
                "typescript": ["tsc"],
            }.get(lang, [])
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_user.is_admin = MagicMock(return_value=False)
        mock_user.effective_tenant_id = "n8n-dev"

        result = await get_supported_tools(
            language=None,
            container_name=None,
            user=mock_user,
            execution_service=mock_service,
        )

        assert "python" in result["supported_languages"]
        assert result["total_tools"] == 4

    async def test_get_supported_tools_unsupported_language(self):
        """Test 404 for unsupported language."""
        from routers.testing_modules.test_discovery import get_supported_tools

        mock_service = MagicMock()
        mock_service._get_supported_tools = MagicMock(return_value=[])

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_user.is_admin = MagicMock(return_value=False)
        mock_user.effective_tenant_id = "n8n-dev"

        with pytest.raises(HTTPException) as exc_info:
            await get_supported_tools(
                language="fortran",
                container_name=None,
                user=mock_user,
                execution_service=mock_service,
            )
        assert exc_info.value.status_code == 404

    async def test_get_supported_languages(self):
        """Test getting supported languages."""
        from routers.testing_modules.test_discovery import get_supported_languages

        mock_orchestrator = AsyncMock()
        mock_orchestrator.get_supported_languages = AsyncMock(return_value=["python", "javascript"])
        mock_orchestrator.get_supported_test_types = AsyncMock(return_value=["unit", "integration"])

        mock_service = MagicMock()
        mock_service.test_orchestrator = mock_orchestrator
        mock_service._get_supported_tools = MagicMock(return_value=["ruff"])

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_user.is_admin = MagicMock(return_value=False)
        mock_user.effective_tenant_id = "n8n-dev"

        result = await get_supported_languages(
            container_name=None,
            user=mock_user,
            execution_service=mock_service,
        )

        assert "python" in result["languages"]
        assert result["total"] == 2

    async def test_get_language_test_types_success(self):
        """Test getting test types for a supported language."""
        from routers.testing_modules.test_discovery import get_language_test_types

        mock_orchestrator = AsyncMock()
        mock_orchestrator.get_supported_languages = AsyncMock(return_value=["python", "javascript"])
        mock_orchestrator.get_supported_test_types = AsyncMock(
            return_value=["unit", "integration", "coverage"]
        )

        mock_service = MagicMock()
        mock_service.test_orchestrator = mock_orchestrator

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_user.is_admin = MagicMock(return_value=False)
        mock_user.effective_tenant_id = "n8n-dev"

        result = await get_language_test_types(
            language="python",
            container_name=None,
            user=mock_user,
            execution_service=mock_service,
        )

        assert result["language"] == "python"
        assert result["count"] == 3

    async def test_get_language_test_types_unsupported(self):
        """Test 404 for unsupported language test types."""
        from routers.testing_modules.test_discovery import get_language_test_types

        mock_orchestrator = AsyncMock()
        mock_orchestrator.get_supported_languages = AsyncMock(return_value=["python"])

        mock_service = MagicMock()
        mock_service.test_orchestrator = mock_orchestrator

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_user.is_admin = MagicMock(return_value=False)
        mock_user.effective_tenant_id = "n8n-dev"

        with pytest.raises(HTTPException) as exc_info:
            await get_language_test_types(
                language="cobol",
                container_name=None,
                user=mock_user,
                execution_service=mock_service,
            )
        assert exc_info.value.status_code == 404

    async def test_list_test_containers_empty(self):
        """Test listing containers when ContainerManager is not available."""
        from routers.testing_modules.test_discovery import list_test_containers

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_user.is_admin = MagicMock(return_value=False)
        mock_user.effective_tenant_id = "n8n-dev"

        # get_container_manager doesn't exist in shared.container_validation
        # so the real ImportError will trigger the fallback path naturally
        result = await list_test_containers(
            filter_type="all",
            user=mock_user,
        )

        assert result["containers"] == []
        assert result["count"] == 0


# ============================================================================
# test_execution.py (testing_modules) tests
# ============================================================================


class TestTestingModulesExecution:
    """Test testing_modules test_execution endpoints."""

    async def test_execute_test_success(self):
        """Test successful test execution via testing_modules."""
        from routers.testing_modules.test_execution import execute_test

        mock_result = MagicMock()
        mock_result.test_id = "t1"
        mock_result.success = True

        mock_service = AsyncMock()
        mock_service.execute_test = AsyncMock(return_value=mock_result)

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_user.is_admin = MagicMock(return_value=False)
        mock_user.effective_tenant_id = "n8n-dev"
        mock_user.id = "u1"

        mock_request = MagicMock()
        mock_request.container_name = None
        mock_request.language = "python"
        mock_request.timeout = 30
        mock_request.security_level = "medium"
        mock_request.with_hypothesis = False

        mock_bg = MagicMock()

        result = await execute_test(
            request=mock_request,
            background_tasks=mock_bg,
            user=mock_user,
            execution_service=mock_service,
        )

        assert result.success is True

    async def test_execute_test_value_error(self):
        """Test execute_test raises 400 on ValueError."""
        from routers.testing_modules.test_execution import execute_test

        mock_service = AsyncMock()
        mock_service.execute_test = AsyncMock(side_effect=ValueError("Invalid code"))

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_user.is_admin = MagicMock(return_value=False)
        mock_user.effective_tenant_id = "n8n-dev"

        mock_request = MagicMock()
        mock_request.container_name = None

        mock_bg = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await execute_test(
                request=mock_request,
                background_tasks=mock_bg,
                user=mock_user,
                execution_service=mock_service,
            )
        assert exc_info.value.status_code == 400

    async def test_validate_code_security_safe(self):
        """Test security validation for safe code."""
        from routers.testing_modules.test_execution import validate_code_security

        mock_service = AsyncMock()
        mock_service._validate_code_security = AsyncMock(
            return_value={"is_safe": True, "violations": [], "risk_score": 0.0}
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_user.is_admin = MagicMock(return_value=False)
        mock_user.effective_tenant_id = "n8n-dev"

        result = await validate_code_security(
            code="x = 1 + 2",
            language="python",
            user=mock_user,
            execution_service=mock_service,
        )

        assert result["is_safe"] is True
        assert result["risk_level"] == 0.0
        assert len(result["violations"]) == 0

    async def test_validate_code_security_unsafe(self):
        """Test security validation for unsafe code."""
        from routers.testing_modules.test_execution import validate_code_security

        mock_service = AsyncMock()
        mock_service._validate_code_security = AsyncMock(
            return_value={
                "is_safe": False,
                "violations": [
                    {"description": "exec() is dangerous", "severity": "high"},
                ],
                "risk_score": 80.0,
            }
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_user.is_admin = MagicMock(return_value=False)
        mock_user.effective_tenant_id = "n8n-dev"

        result = await validate_code_security(
            code="exec('import os; os.system(\"rm -rf /\")')",
            language="python",
            user=mock_user,
            execution_service=mock_service,
        )

        assert result["is_safe"] is False
        assert result["risk_level"] == 80.0


# ============================================================================
# quality_analysis.py tests
# ============================================================================


class TestQualityAnalysis:
    """Test quality analysis endpoints."""

    async def test_run_quality_check_success(self):
        """Test successful quality check."""
        from routers.testing_modules.quality_analysis import run_quality_check

        mock_result = MagicMock()
        mock_result.analysis_id = "a1"
        mock_result.overall_score = 90.0
        mock_result.issues = []
        mock_result.total_issues = 0

        mock_service = AsyncMock()
        mock_service.analyze_quality = AsyncMock(return_value=mock_result)

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_user.is_admin = MagicMock(return_value=False)
        mock_user.effective_tenant_id = "n8n-dev"
        mock_user.id = "u1"

        mock_request = MagicMock()
        mock_request.container_name = "my-container"
        mock_request.tools = ["ruff", "ruff-format"]
        mock_request.language = "python"
        mock_request.fix_issues = False

        mock_bg = MagicMock()
        mock_container = MagicMock()

        with patch(
            "routers.testing_modules.quality_analysis.validate_and_get_container_for_tenant",
            return_value=mock_container,
        ):
            result = await run_quality_check(
                request=mock_request,
                background_tasks=mock_bg,
                user=mock_user,
                execution_service=mock_service,
            )

        assert result.overall_score == 90.0

    async def test_run_quality_check_validation_error(self):
        """Test quality check raises 400 on ValueError."""
        from routers.testing_modules.quality_analysis import run_quality_check

        mock_service = AsyncMock()
        mock_service.analyze_quality = AsyncMock(side_effect=ValueError("Invalid tools"))

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_user.is_admin = MagicMock(return_value=False)
        mock_user.effective_tenant_id = "n8n-dev"

        mock_request = MagicMock()
        mock_request.container_name = "c1"
        mock_request.tools = ["invalid"]
        mock_request.language = "python"
        mock_request.fix_issues = False

        mock_bg = MagicMock()
        mock_container = MagicMock()

        with patch(
            "routers.testing_modules.quality_analysis.validate_and_get_container_for_tenant",
            return_value=mock_container,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await run_quality_check(
                    request=mock_request,
                    background_tasks=mock_bg,
                    user=mock_user,
                    execution_service=mock_service,
                )
            assert exc_info.value.status_code == 400

    async def test_run_quality_scan_success(self):
        """Test successful quality scan."""
        from routers.testing_modules.quality_analysis import run_quality_scan

        mock_result = MagicMock()
        mock_result.analysis_id = "a2"
        mock_result.overall_score = 85.0
        mock_result.total_issues = 5
        mock_result.issues = []

        mock_service = AsyncMock()
        mock_service.analyze_container_quality = AsyncMock(return_value=mock_result)

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_user.is_admin = MagicMock(return_value=False)
        mock_user.effective_tenant_id = "n8n-dev"
        mock_user.id = "u1"

        mock_request = MagicMock()
        mock_request.container_name = "my-container"
        mock_request.scan_path = "/workspace"
        mock_request.fix_issues = False

        mock_bg = MagicMock()
        mock_container = MagicMock()

        with patch(
            "routers.testing_modules.quality_analysis.validate_and_get_container_for_tenant",
            return_value=mock_container,
        ):
            result = await run_quality_scan(
                request=mock_request,
                background_tasks=mock_bg,
                user=mock_user,
                execution_service=mock_service,
            )

        assert result.overall_score == 85.0


# ============================================================================
# get_execution_service tests
# ============================================================================


class TestGetExecutionService:
    """Test get_execution_service dependency."""

    async def test_get_execution_service_returns_stub_on_failure(self):
        """Test that get_execution_service returns a stub when dependencies fail."""
        from routers.testing_modules.testing_utils import get_execution_service

        # The function is designed to return a stub when Docker/Redis are unavailable
        # which is the test environment situation
        service = await get_execution_service()
        from services.code_execution_service import CodeExecutionService

        assert isinstance(service, CodeExecutionService)

    async def test_stub_service_has_get_supported_tools(self):
        """Test stub service has _get_supported_tools method."""
        from routers.testing_modules.testing_utils import get_execution_service

        service = await get_execution_service()

        # The service should have _get_supported_tools (either real or stub)
        assert hasattr(service, "_get_supported_tools")

    async def test_stub_service_has_test_orchestrator(self):
        """Test stub service has test_orchestrator attribute."""
        from routers.testing_modules.testing_utils import get_execution_service

        service = await get_execution_service()

        # The service should have test_orchestrator (either real or stub)
        assert hasattr(service, "test_orchestrator")


# ============================================================================
# Extended testing_utils.py tests
# ============================================================================


class TestValidateTestRequestExtended:
    """Extended validate_test_request tests for more coverage."""

    def test_with_test_type(self):
        """Test validation preserves test_type."""
        from routers.testing_modules.testing_utils import validate_test_request

        result = validate_test_request({"code": "x=1", "test_type": "integration"})
        assert result["test_type"] == "integration"

    def test_default_timeout(self):
        """Test default timeout is 30."""
        from routers.testing_modules.testing_utils import validate_test_request

        result = validate_test_request({"code": "x=1"})
        assert result["timeout"] == 30

    def test_default_memory_limit(self):
        """Test default memory limit is 128."""
        from routers.testing_modules.testing_utils import validate_test_request

        result = validate_test_request({"code": "x=1"})
        assert result["memory_limit"] == 128

    def test_environment_vars_preserved(self):
        """Test environment vars are preserved."""
        from routers.testing_modules.testing_utils import validate_test_request

        result = validate_test_request(
            {
                "code": "x=1",
                "environment_vars": {"MY_VAR": "value"},
            }
        )
        assert result["environment_vars"] == {"MY_VAR": "value"}

    def test_language_case_insensitive(self):
        """Test language validation is case insensitive."""
        from routers.testing_modules.testing_utils import validate_test_request

        result = validate_test_request({"code": "x=1", "language": "Python"})
        assert result["language"] == "python"

    def test_all_supported_languages(self):
        """Test all supported languages pass validation."""
        from routers.testing_modules.testing_utils import validate_test_request

        for lang in ["python", "javascript", "java", "go", "rust", "cpp"]:
            result = validate_test_request({"code": "x=1", "language": lang})
            assert result["language"] == lang


class TestFormatTestResultsExtended:
    """Extended format_test_results tests using real TestResult model objects."""

    def _make_result(self, **overrides):
        from models.testing_models import TestResult, TestSecurityLevel

        defaults = {
            "test_id": "t1",
            "success": True,
            "stdout": "ok",
            "stderr": "",
            "exit_code": 0,
            "execution_time": 50.0,
            "created_at": datetime.now(UTC),
            "language": "python",
            "security_level": TestSecurityLevel.MEDIUM,
        }
        defaults.update(overrides)
        return TestResult(**defaults)

    def test_single_result(self):
        """Test formatting a single successful result preserves all fields."""
        from routers.testing_modules.testing_utils import format_test_results

        ts = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)
        r1 = self._make_result(
            test_id="t1",
            success=True,
            stdout="PASS",
            exit_code=0,
            execution_time=50.0,
            created_at=ts,
            language="python",
        )

        result = format_test_results([r1])

        assert result["total_tests"] == 1
        assert result["successful"] == 1
        assert result["failed"] == 0
        assert result["success_rate"] == 1.0
        assert result["avg_execution_time"] == 50.0
        assert len(result["results"]) == 1
        # Verify the formatted result preserves all original fields
        formatted = result["results"][0]
        assert formatted["test_id"] == "t1"
        assert formatted["success"] is True
        assert formatted["language"] == "python"
        assert formatted["exit_code"] == 0
        assert formatted["stdout"] == "PASS"
        assert formatted["stderr"] == ""
        assert formatted["execution_time"] == 50.0

    def test_mixed_results_computes_correct_stats(self):
        """Test that success/fail counts and avg time are computed from real data."""
        from routers.testing_modules.testing_utils import format_test_results

        results = [
            self._make_result(test_id="t1", success=True, execution_time=40.0),
            self._make_result(test_id="t2", success=False, exit_code=1, execution_time=60.0),
            self._make_result(test_id="t3", success=True, execution_time=80.0),
        ]

        result = format_test_results(results)

        assert result["total_tests"] == 3
        assert result["successful"] == 2
        assert result["failed"] == 1
        assert result["success_rate"] == pytest.approx(2.0 / 3.0)
        assert result["avg_execution_time"] == pytest.approx(60.0)

    def test_all_failed(self):
        """Test formatting with all failed results computes zero success rate."""
        from routers.testing_modules.testing_utils import format_test_results

        r1 = self._make_result(
            test_id="t1",
            success=False,
            exit_code=1,
            stdout="",
            stderr="AssertionError",
            execution_time=100.0,
        )

        result = format_test_results([r1])

        assert result["success_rate"] == 0.0
        assert result["failed"] == 1
        assert result["successful"] == 0
        assert result["results"][0]["stderr"] == "AssertionError"

    def test_empty_results(self):
        """Test formatting with empty list returns zero-value defaults."""
        from routers.testing_modules.testing_utils import format_test_results

        result = format_test_results([])

        assert result["total_tests"] == 0
        assert result["successful"] == 0
        assert result["failed"] == 0
        assert result["avg_execution_time"] == 0.0
        assert result["results"] == []


class TestAggregateTestMetricsExtended:
    """Extended aggregate_test_metrics tests using real TestResult models."""

    def _make_result(self, **overrides):
        from models.testing_models import TestResult, TestSecurityLevel

        defaults = {
            "test_id": "t1",
            "success": True,
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
            "execution_time": 100.0,
            "created_at": datetime.now(UTC),
            "language": "python",
            "security_level": TestSecurityLevel.MEDIUM,
        }
        defaults.update(overrides)
        return TestResult(**defaults)

    def test_multiple_languages(self):
        """Test aggregation correctly groups by language with real models."""
        from routers.testing_modules.testing_utils import aggregate_test_metrics

        r1 = self._make_result(test_id="t1", language="python", execution_time=100.0)
        r2 = self._make_result(test_id="t2", language="javascript", execution_time=200.0)

        result = aggregate_test_metrics([r1, r2])

        assert result["languages"]["python"]["total"] == 1
        assert result["languages"]["python"]["successful"] == 1
        assert result["languages"]["javascript"]["total"] == 1
        assert result["total_tests"] == 2
        assert result["execution_times"]["python"]["avg"] == 100.0
        assert result["execution_times"]["javascript"]["avg"] == 200.0

    def test_execution_time_stats(self):
        """Test execution time min/max/avg computed from real data."""
        from routers.testing_modules.testing_utils import aggregate_test_metrics

        results = [
            self._make_result(test_id=f"t{i}", execution_time=val)
            for i, val in enumerate([50.0, 100.0, 150.0])
        ]

        result = aggregate_test_metrics(results)

        assert result["execution_times"]["python"]["avg"] == pytest.approx(100.0)
        assert result["execution_times"]["python"]["min"] == 50.0
        assert result["execution_times"]["python"]["max"] == 150.0

    def test_all_error_pattern_categories(self):
        """Test that each error category is correctly detected from stderr."""
        from routers.testing_modules.testing_utils import aggregate_test_metrics

        test_cases = [
            ("SyntaxError: invalid syntax", "syntax_error"),
            ("ModuleNotFoundError: No module named 'foo'", "import_error"),
            ("execution timed out after 30s", "timeout"),
            ("memory limit exceeded", "memory_error"),
            ("something completely unexpected", "unknown"),
        ]

        for stderr_text, expected_category in test_cases:
            r = self._make_result(
                test_id=f"t-{expected_category}",
                success=False,
                exit_code=1,
                stderr=stderr_text,
            )
            result = aggregate_test_metrics([r])
            assert expected_category in result["error_patterns"], (
                f"Expected '{expected_category}' for stderr='{stderr_text}', "
                f"got {result['error_patterns']}"
            )
            assert result["error_patterns"][expected_category] == 1

    def test_success_rate_calculation(self):
        """Test per-language success rate computed from real pass/fail models."""
        from routers.testing_modules.testing_utils import aggregate_test_metrics

        results = [
            self._make_result(
                test_id=f"t{i}",
                language="go",
                success=(i < 3),
                exit_code=0 if i < 3 else 1,
                stderr="" if i < 3 else "test failed",
            )
            for i in range(4)
        ]

        result = aggregate_test_metrics(results)
        assert result["languages"]["go"]["total"] == 4
        assert result["languages"]["go"]["successful"] == 3
        assert result["languages"]["go"]["success_rate"] == pytest.approx(0.75)
        assert result["success_rate"] == pytest.approx(0.75)

    def test_empty_results(self):
        """Test aggregation of empty list returns zero defaults."""
        from routers.testing_modules.testing_utils import aggregate_test_metrics

        result = aggregate_test_metrics([])
        assert result["languages"] == {}
        assert result["execution_times"] == {}
        assert result["error_patterns"] == {}
        assert result["success_rate"] == 0.0

    def test_successful_results_produce_no_error_patterns(self):
        """Test that passing results generate no error patterns."""
        from routers.testing_modules.testing_utils import aggregate_test_metrics

        results = [self._make_result(test_id=f"t{i}") for i in range(5)]
        result = aggregate_test_metrics(results)
        assert result["error_patterns"] == {}
        assert result["success_rate"] == 1.0


# ============================================================================
# Extended test_discovery.py tests
# ============================================================================


class TestTestDiscoveryExtended:
    """Extended test_discovery tests for more coverage."""

    async def test_get_supported_tools_with_container_name(self):
        """Test getting tools with container_name parameter (validation skip path)."""
        from routers.testing_modules.test_discovery import get_supported_tools

        mock_service = MagicMock()
        mock_service._get_supported_tools = MagicMock(return_value=["ruff", "ruff-format"])

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_user.is_admin = MagicMock(return_value=True)
        mock_user.effective_tenant_id = "n8n-dev"

        with patch(
            "routers.testing_modules.test_discovery.validate_and_get_container_for_tenant",
            return_value=MagicMock(),
        ):
            result = await get_supported_tools(
                language="python",
                container_name="my-container",
                user=mock_user,
                execution_service=mock_service,
            )

        assert result["language"] == "python"
        assert result["container_name"] == "my-container"

    async def test_get_supported_tools_all_with_container(self):
        """Test getting all tools with container_name."""
        from routers.testing_modules.test_discovery import get_supported_tools

        mock_service = MagicMock()
        mock_service._get_supported_tools = MagicMock(
            side_effect=lambda lang: {"python": ["ruff"], "javascript": ["eslint"]}.get(lang, [])
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_user.is_admin = MagicMock(return_value=True)
        mock_user.effective_tenant_id = "n8n-dev"

        with patch(
            "routers.testing_modules.test_discovery.validate_and_get_container_for_tenant",
            return_value=MagicMock(),
        ):
            result = await get_supported_tools(
                language=None,
                container_name="my-container",
                user=mock_user,
                execution_service=mock_service,
            )

        assert result["container_name"] == "my-container"

    async def test_get_supported_languages_fallback(self):
        """Test get_supported_languages uses fallback when orchestrator fails."""
        from routers.testing_modules.test_discovery import get_supported_languages

        mock_service = MagicMock()
        mock_service.test_orchestrator = MagicMock()
        mock_service.test_orchestrator.get_supported_languages = AsyncMock(
            side_effect=AttributeError("no method")
        )
        mock_service.test_orchestrator.get_supported_test_types = AsyncMock(
            side_effect=AttributeError("no method")
        )
        mock_service._get_supported_tools = MagicMock(return_value=[])

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_user.is_admin = MagicMock(return_value=False)
        mock_user.effective_tenant_id = "n8n-dev"

        result = await get_supported_languages(
            container_name=None,
            user=mock_user,
            execution_service=mock_service,
        )

        # Should fall back to default languages
        assert "python" in result["languages"]
        assert result["total"] >= 3

    async def test_get_supported_languages_with_container(self):
        """Test getting languages with container_name."""
        from routers.testing_modules.test_discovery import get_supported_languages

        mock_service = MagicMock()
        mock_service.test_orchestrator = MagicMock()
        mock_service.test_orchestrator.get_supported_languages = AsyncMock(return_value=["python"])
        mock_service.test_orchestrator.get_supported_test_types = AsyncMock(return_value=["unit"])
        mock_service._get_supported_tools = MagicMock(return_value=["ruff"])

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_user.is_admin = MagicMock(return_value=True)
        mock_user.effective_tenant_id = "n8n-dev"

        with patch(
            "routers.testing_modules.test_discovery.validate_and_get_container_for_tenant",
            return_value=MagicMock(),
        ):
            result = await get_supported_languages(
                container_name="my-container",
                user=mock_user,
                execution_service=mock_service,
            )

        assert result["container_name"] == "my-container"

    async def test_get_language_test_types_with_container(self):
        """Test getting test types with container_name."""
        from routers.testing_modules.test_discovery import get_language_test_types

        mock_orchestrator = AsyncMock()
        mock_orchestrator.get_supported_languages = AsyncMock(return_value=["python"])
        mock_orchestrator.get_supported_test_types = AsyncMock(return_value=["unit", "integration"])

        mock_service = MagicMock()
        mock_service.test_orchestrator = mock_orchestrator

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_user.is_admin = MagicMock(return_value=True)
        mock_user.effective_tenant_id = "n8n-dev"

        with patch(
            "routers.testing_modules.test_discovery.validate_and_get_container_for_tenant",
            return_value=MagicMock(),
        ):
            result = await get_language_test_types(
                language="python",
                container_name="my-container",
                user=mock_user,
                execution_service=mock_service,
            )

        assert result["container_name"] == "my-container"

    async def test_get_language_test_types_fallback(self):
        """Test test types use fallback when orchestrator fails."""
        from routers.testing_modules.test_discovery import get_language_test_types

        mock_service = MagicMock()
        mock_service.test_orchestrator = MagicMock()
        mock_service.test_orchestrator.get_supported_languages = AsyncMock(
            side_effect=AttributeError("no method")
        )
        mock_service.test_orchestrator.get_supported_test_types = AsyncMock(
            side_effect=AttributeError("no method")
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_user.is_admin = MagicMock(return_value=False)
        mock_user.effective_tenant_id = "n8n-dev"

        result = await get_language_test_types(
            language="python",
            container_name=None,
            user=mock_user,
            execution_service=mock_service,
        )

        # Should fall back to default test types for python
        assert "unit" in result["test_types"]

    async def test_list_test_containers_with_filter_test(self):
        """Test listing containers with filter_type=test."""
        from routers.testing_modules.test_discovery import list_test_containers

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_user.is_admin = MagicMock(return_value=False)
        mock_user.effective_tenant_id = "n8n-dev"

        result = await list_test_containers(
            filter_type="test",
            user=mock_user,
        )

        assert result["containers"] == []
        assert result["filter"] == "test"

    async def test_list_test_containers_with_filter_provisioned(self):
        """Test listing containers with filter_type=provisioned."""
        from routers.testing_modules.test_discovery import list_test_containers

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_user.is_admin = MagicMock(return_value=False)
        mock_user.effective_tenant_id = "n8n-dev"

        result = await list_test_containers(
            filter_type="provisioned",
            user=mock_user,
        )

        assert result["containers"] == []
        assert result["filter"] == "provisioned"


# ============================================================================
# Extended test_execution.py tests
# ============================================================================


class TestTestingModulesExecutionExtended:
    """Extended test_execution tests."""

    async def test_execute_test_generic_exception(self):
        """Test execute_test raises 500 on generic Exception."""
        from routers.testing_modules.test_execution import execute_test

        mock_service = AsyncMock()
        mock_service.execute_test = AsyncMock(side_effect=RuntimeError("unexpected crash"))

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_user.is_admin = MagicMock(return_value=False)
        mock_user.effective_tenant_id = "n8n-dev"

        mock_request = MagicMock()
        mock_request.container_name = None

        mock_bg = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await execute_test(
                request=mock_request,
                background_tasks=mock_bg,
                user=mock_user,
                execution_service=mock_service,
            )
        assert exc_info.value.status_code == 500

    async def test_validate_code_security_not_implemented(self):
        """Test security validation handles NotImplementedError."""
        from routers.testing_modules.test_execution import validate_code_security

        mock_service = AsyncMock()
        mock_service._validate_code_security = AsyncMock(
            side_effect=ValueError("stub - not implemented")
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_user.is_admin = MagicMock(return_value=False)
        mock_user.effective_tenant_id = "n8n-dev"

        result = await validate_code_security(
            code="import os",
            language="python",
            user=mock_user,
            execution_service=mock_service,
        )

        # Should handle ValueError and return unsafe result
        assert result["is_safe"] is False
        assert result["risk_level"] == 100.0


# ============================================================================
# Extended quality_analysis.py tests
# ============================================================================


class TestQualityAnalysisExtended:
    """Extended quality_analysis tests."""

    async def test_run_quality_check_generic_exception(self):
        """Test quality check raises 500 on generic Exception."""
        from routers.testing_modules.quality_analysis import run_quality_check

        mock_service = AsyncMock()
        mock_service.analyze_quality = AsyncMock(side_effect=RuntimeError("tool crash"))

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_user.is_admin = MagicMock(return_value=False)
        mock_user.effective_tenant_id = "n8n-dev"

        mock_request = MagicMock()
        mock_request.container_name = "c1"
        mock_request.tools = ["ruff"]
        mock_request.language = "python"
        mock_request.fix_issues = False

        mock_bg = MagicMock()
        mock_container = MagicMock()

        with patch(
            "routers.testing_modules.quality_analysis.validate_and_get_container_for_tenant",
            return_value=mock_container,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await run_quality_check(
                    request=mock_request,
                    background_tasks=mock_bg,
                    user=mock_user,
                    execution_service=mock_service,
                )
            assert exc_info.value.status_code == 500

    async def test_run_quality_check_with_fix_issues(self):
        """Test quality check with fix_issues=True and fixable issues."""
        from routers.testing_modules.quality_analysis import run_quality_check

        mock_issue = MagicMock()
        mock_issue.suggested_fix = "add type annotation"

        mock_result = MagicMock()
        mock_result.analysis_id = "a1"
        mock_result.overall_score = 75.0
        mock_result.issues = [mock_issue]
        mock_result.total_issues = 1

        mock_service = AsyncMock()
        mock_service.analyze_quality = AsyncMock(return_value=mock_result)

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_user.is_admin = MagicMock(return_value=False)
        mock_user.effective_tenant_id = "n8n-dev"
        mock_user.id = "u1"

        mock_request = MagicMock()
        mock_request.container_name = "c1"
        mock_request.tools = ["ruff"]
        mock_request.language = "python"
        mock_request.fix_issues = True

        mock_bg = MagicMock()
        mock_container = MagicMock()

        with patch(
            "routers.testing_modules.quality_analysis.validate_and_get_container_for_tenant",
            return_value=mock_container,
        ):
            result = await run_quality_check(
                request=mock_request,
                background_tasks=mock_bg,
                user=mock_user,
                execution_service=mock_service,
            )

        assert result.overall_score == 75.0

    async def test_run_quality_scan_validation_error(self):
        """Test quality scan raises 400 on ValueError."""
        from routers.testing_modules.quality_analysis import run_quality_scan

        mock_service = AsyncMock()
        mock_service.analyze_container_quality = AsyncMock(
            side_effect=ValueError("Invalid scan path")
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_user.is_admin = MagicMock(return_value=False)
        mock_user.effective_tenant_id = "n8n-dev"

        mock_request = MagicMock()
        mock_request.container_name = "c1"
        mock_request.scan_path = "/workspace"
        mock_request.fix_issues = False

        mock_bg = MagicMock()
        mock_container = MagicMock()

        with patch(
            "routers.testing_modules.quality_analysis.validate_and_get_container_for_tenant",
            return_value=mock_container,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await run_quality_scan(
                    request=mock_request,
                    background_tasks=mock_bg,
                    user=mock_user,
                    execution_service=mock_service,
                )
            assert exc_info.value.status_code == 400

    async def test_run_quality_scan_generic_exception(self):
        """Test quality scan raises 500 on generic Exception."""
        from routers.testing_modules.quality_analysis import run_quality_scan

        mock_service = AsyncMock()
        mock_service.analyze_container_quality = AsyncMock(side_effect=RuntimeError("unexpected"))

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_user.is_admin = MagicMock(return_value=False)
        mock_user.effective_tenant_id = "n8n-dev"

        mock_request = MagicMock()
        mock_request.container_name = "c1"
        mock_request.scan_path = "/workspace"
        mock_request.fix_issues = False

        mock_bg = MagicMock()
        mock_container = MagicMock()

        with patch(
            "routers.testing_modules.quality_analysis.validate_and_get_container_for_tenant",
            return_value=mock_container,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await run_quality_scan(
                    request=mock_request,
                    background_tasks=mock_bg,
                    user=mock_user,
                    execution_service=mock_service,
                )
            assert exc_info.value.status_code == 500

    async def test_run_quality_scan_with_fix_issues(self):
        """Test quality scan with fix_issues=True."""
        from routers.testing_modules.quality_analysis import run_quality_scan

        mock_issue = MagicMock()
        mock_issue.suggested_fix = "reformat"

        mock_result = MagicMock()
        mock_result.analysis_id = "a2"
        mock_result.overall_score = 80.0
        mock_result.total_issues = 1
        mock_result.issues = [mock_issue]

        mock_service = AsyncMock()
        mock_service.analyze_container_quality = AsyncMock(return_value=mock_result)

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"
        mock_user.is_admin = MagicMock(return_value=True)
        mock_user.effective_tenant_id = "n8n-dev"
        mock_user.id = "u1"

        mock_request = MagicMock()
        mock_request.container_name = "c1"
        mock_request.scan_path = "/workspace"
        mock_request.fix_issues = True

        mock_bg = MagicMock()
        mock_container = MagicMock()

        with patch(
            "routers.testing_modules.quality_analysis.validate_and_get_container_for_tenant",
            return_value=mock_container,
        ):
            result = await run_quality_scan(
                request=mock_request,
                background_tasks=mock_bg,
                user=mock_user,
                execution_service=mock_service,
            )

        assert result.overall_score == 80.0


# ============================================================================
# Extended health_and_metrics.py tests
# ============================================================================


class TestHealthAndMetricsExtended:
    """Extended health and metrics tests."""

    async def test_health_check_with_service_available(self):
        """Test health check when execution service is available."""
        from routers.testing_modules.health_and_metrics import testing_health_check

        # This exercises the path where get_execution_service succeeds
        # but service health check returns status
        result = await testing_health_check(tenant_id="n8n-dev")

        # Should return a valid health structure
        assert "status" in result
        assert "timestamp" in result

    async def test_metrics_summary_with_all_fields(self):
        """Test metrics summary returns all expected fields."""
        from routers.testing_modules.health_and_metrics import (
            get_testing_metrics_summary,
        )

        mock_metrics = AsyncMock()
        mock_metrics.get_test_execution_analytics = AsyncMock(
            return_value={
                "period_days": 30,
                "test_executions": 100,
                "quality_analyses": 25,
                "resource_usage": {"cpu": 45.0, "memory": 256.0},
                "languages": ["python", "javascript", "go"],
            }
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "testuser"

        with patch(
            "routers.testing_modules.health_and_metrics.get_metrics_service",
            return_value=mock_metrics,
        ):
            result = await get_testing_metrics_summary(
                days=30,
                current_user=mock_user,
                tenant_id="n8n-dev",
            )

        assert result["period_days"] == 30
        assert result["test_executions"] == 100
        assert result["quality_analyses"] == 25
        assert len(result["languages"]) == 3
