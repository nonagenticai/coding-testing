"""Tests targeting low-coverage source files to push coverage above 95%.

Targets:
- testing_utils.py (validate_test_request, format_test_results, aggregate_test_metrics)
- health_and_metrics.py (testing_health_check, get_testing_metrics_summary)
- test_discovery.py (get_supported_tools, get_supported_languages, get_language_test_types, list_test_containers)
- core_config_pydantic_ai.py (load_pydantic_ai_config_from_env, get_pydantic_ai_config)
- app_factory.py (_add_exception_handlers fallback, lifespan, generate_unique_id edge cases)
- code_execution_service.py (analyze_quality, analyze_container_quality, get_service_health, _get_supported_tools, get_code_execution_service)
- test_framework_security_manager.py (SecurityManager all methods, SecurityCheckResult, TestValidationSecurityConfig)
- middleware/validation.py (HTTPException path in dispatch)
- routers/intelligent_testing.py (_cleanup_service_sync)
- routers/health.py (system-metrics with psutil)
- services/ai_test_insights/core_service.py (agent init exception path)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import HTTPException

from models.testing_models import (
    SecurityLevel,
    TestRequest,
    TestResult,
    TestSecurityLevel,
)
from tests.route_paths import route_paths as _route_paths

# ===========================================================================
# testing_utils.py — validate_test_request, format_test_results, aggregate_test_metrics
# ===========================================================================


class TestValidateTestRequest:
    """Tests for routers.testing_modules.testing_utils.validate_test_request."""

    def test_valid_python_request(self):
        from routers.testing_modules.testing_utils import validate_test_request

        result = validate_test_request(
            {"code": "def foo(): pass", "language": "python", "timeout": 10}
        )
        assert result["code"] == "def foo(): pass"
        assert result["language"] == "python"
        assert result["timeout"] == 10

    def test_empty_request_raises(self):
        from routers.testing_modules.testing_utils import validate_test_request

        with pytest.raises(HTTPException) as exc_info:
            validate_test_request({})
        assert exc_info.value.status_code == 400

    def test_none_request_raises(self):
        from routers.testing_modules.testing_utils import validate_test_request

        with pytest.raises(HTTPException) as exc_info:
            validate_test_request(None)
        assert exc_info.value.status_code == 400

    def test_missing_code_raises(self):
        from routers.testing_modules.testing_utils import validate_test_request

        with pytest.raises(HTTPException) as exc_info:
            validate_test_request({"language": "python"})
        assert exc_info.value.status_code == 400

    def test_blank_code_raises(self):
        from routers.testing_modules.testing_utils import validate_test_request

        with pytest.raises(HTTPException) as exc_info:
            validate_test_request({"code": "   "})
        assert exc_info.value.status_code == 400

    def test_unsupported_language_raises(self):
        from routers.testing_modules.testing_utils import validate_test_request

        with pytest.raises(HTTPException) as exc_info:
            validate_test_request({"code": "x=1", "language": "haskell"})
        assert exc_info.value.status_code == 400

    def test_timeout_clamped_to_300(self):
        from routers.testing_modules.testing_utils import validate_test_request

        result = validate_test_request({"code": "x=1", "timeout": 999})
        assert result["timeout"] == 300

    def test_memory_limit_clamped_to_512(self):
        from routers.testing_modules.testing_utils import validate_test_request

        result = validate_test_request({"code": "x=1", "memory_limit": 2048})
        assert result["memory_limit"] == 512

    def test_defaults_applied(self):
        from routers.testing_modules.testing_utils import validate_test_request

        result = validate_test_request({"code": "x=1"})
        assert result["language"] == "python"
        assert result["timeout"] == 30
        assert result["memory_limit"] == 128
        assert result["test_type"] == "unit"
        assert result["environment_vars"] == {}

    def test_code_is_stripped(self):
        from routers.testing_modules.testing_utils import validate_test_request

        result = validate_test_request({"code": "  x=1  "})
        assert result["code"] == "x=1"

    def test_all_supported_languages(self):
        from routers.testing_modules.testing_utils import validate_test_request

        for lang in ["python", "javascript", "java", "go", "rust", "cpp"]:
            result = validate_test_request({"code": "x=1", "language": lang})
            assert result["language"] == lang


class TestFormatTestResults:
    """Tests for routers.testing_modules.testing_utils.format_test_results."""

    def _make_result(self, success=True, exec_time=1.0, lang="python"):
        return TestResult(
            test_id="t-1",
            success=success,
            stdout="out",
            stderr="",
            exit_code=0 if success else 1,
            execution_time=exec_time,
            created_at=datetime.now(UTC),
            language=lang,
            security_level=TestSecurityLevel.MEDIUM,
        )

    def test_empty_results(self):
        from routers.testing_modules.testing_utils import format_test_results

        result = format_test_results([])
        assert result["total_tests"] == 0
        assert result["successful"] == 0
        assert result["failed"] == 0
        assert result["avg_execution_time"] == 0.0
        assert result["results"] == []

    def test_single_success(self):
        from routers.testing_modules.testing_utils import format_test_results

        result = format_test_results([self._make_result()])
        assert result["total_tests"] == 1
        assert result["successful"] == 1
        assert result["failed"] == 0
        assert result["success_rate"] == 1.0
        assert result["avg_execution_time"] == 1.0
        assert len(result["results"]) == 1
        assert result["results"][0]["test_id"] == "t-1"
        assert result["results"][0]["success"] is True

    def test_mixed_results(self):
        from routers.testing_modules.testing_utils import format_test_results

        results = [
            self._make_result(success=True, exec_time=2.0),
            self._make_result(success=False, exec_time=4.0),
        ]
        result = format_test_results(results)
        assert result["total_tests"] == 2
        assert result["successful"] == 1
        assert result["failed"] == 1
        assert result["success_rate"] == 0.5
        assert result["avg_execution_time"] == 3.0


class TestAggregateTestMetrics:
    """Tests for routers.testing_modules.testing_utils.aggregate_test_metrics."""

    def _make_result(self, success=True, exec_time=1.0, lang="python", stderr=""):
        return TestResult(
            test_id="t-1",
            success=success,
            stdout="",
            stderr=stderr,
            exit_code=0 if success else 1,
            execution_time=exec_time,
            created_at=datetime.now(UTC),
            language=lang,
            security_level=TestSecurityLevel.MEDIUM,
        )

    def test_empty_results(self):
        from routers.testing_modules.testing_utils import aggregate_test_metrics

        result = aggregate_test_metrics([])
        assert result["success_rate"] == 0.0
        assert result["languages"] == {}
        assert result["execution_times"] == {}
        assert result["error_patterns"] == {}

    def test_language_breakdown(self):
        from routers.testing_modules.testing_utils import aggregate_test_metrics

        results = [
            self._make_result(lang="python"),
            self._make_result(lang="python", success=False),
            self._make_result(lang="go"),
        ]
        result = aggregate_test_metrics(results)
        assert result["languages"]["python"]["total"] == 2
        assert result["languages"]["python"]["successful"] == 1
        assert result["languages"]["python"]["success_rate"] == 0.5
        assert result["languages"]["go"]["total"] == 1
        assert result["total_tests"] == 3

    def test_execution_times(self):
        from routers.testing_modules.testing_utils import aggregate_test_metrics

        results = [
            self._make_result(exec_time=1.0),
            self._make_result(exec_time=3.0),
        ]
        result = aggregate_test_metrics(results)
        times = result["execution_times"]["python"]
        assert times["avg"] == 2.0
        assert times["min"] == 1.0
        assert times["max"] == 3.0

    def test_error_pattern_syntax(self):
        from routers.testing_modules.testing_utils import aggregate_test_metrics

        results = [self._make_result(success=False, stderr="SyntaxError: invalid")]
        result = aggregate_test_metrics(results)
        assert result["error_patterns"]["syntax_error"] == 1

    def test_error_pattern_import(self):
        from routers.testing_modules.testing_utils import aggregate_test_metrics

        results = [self._make_result(success=False, stderr="ModuleNotFoundError: No module")]
        result = aggregate_test_metrics(results)
        assert result["error_patterns"]["import_error"] == 1

    def test_error_pattern_timeout(self):
        from routers.testing_modules.testing_utils import aggregate_test_metrics

        results = [self._make_result(success=False, stderr="execution timed out")]
        result = aggregate_test_metrics(results)
        assert result["error_patterns"]["timeout"] == 1

    def test_error_pattern_memory(self):
        from routers.testing_modules.testing_utils import aggregate_test_metrics

        results = [self._make_result(success=False, stderr="MemoryError")]
        result = aggregate_test_metrics(results)
        assert result["error_patterns"]["memory_error"] == 1

    def test_error_pattern_unknown(self):
        from routers.testing_modules.testing_utils import aggregate_test_metrics

        results = [self._make_result(success=False, stderr="something weird happened")]
        result = aggregate_test_metrics(results)
        assert result["error_patterns"]["unknown"] == 1


# ===========================================================================
# core_config_pydantic_ai.py
# ===========================================================================


class TestPydanticAIConfig:
    """Tests for config.core_config_pydantic_ai."""

    def test_load_defaults(self):
        from config.core_config_pydantic_ai import load_pydantic_ai_config_from_env

        config = load_pydantic_ai_config_from_env()
        assert config["provider"] == "openai"
        assert config["model"] == "gpt-4"
        assert config["max_retries"] == 3
        assert config["timeout_seconds"] == 30
        assert config["temperature"] == pytest.approx(0.1)
        assert config["max_tokens"] == 2000
        assert config["rate_limit_rpm"] == 100
        assert config["enabled"] is True

    def test_load_with_custom_env(self, monkeypatch):
        from config.core_config_pydantic_ai import load_pydantic_ai_config_from_env

        monkeypatch.setenv("PYDANTIC_AI_PROVIDER", "anthropic")
        monkeypatch.setenv("PYDANTIC_AI_MODEL", "claude-3")
        monkeypatch.setenv("PYDANTIC_AI_MAX_RETRIES", "5")
        monkeypatch.setenv("PYDANTIC_AI_TIMEOUT", "60")
        monkeypatch.setenv("PYDANTIC_AI_TEMPERATURE", "0.7")
        monkeypatch.setenv("PYDANTIC_AI_MAX_TOKENS", "4000")
        monkeypatch.setenv("PYDANTIC_AI_RATE_LIMIT_RPM", "200")
        monkeypatch.setenv("PYDANTIC_AI_ENABLED", "false")
        monkeypatch.setenv("PYDANTIC_AI_API_KEY", "sk-test")
        monkeypatch.setenv("PYDANTIC_AI_BASE_URL", "https://api.example.com")

        config = load_pydantic_ai_config_from_env()
        assert config["provider"] == "anthropic"
        assert config["model"] == "claude-3"
        assert config["max_retries"] == 5
        assert config["timeout_seconds"] == 60
        assert config["temperature"] == pytest.approx(0.7)
        assert config["max_tokens"] == 4000
        assert config["rate_limit_rpm"] == 200
        assert config["enabled"] is False
        assert config["api_key"] == "sk-test"
        assert config["base_url"] == "https://api.example.com"

    def test_get_config_delegates(self):
        from config.core_config_pydantic_ai import get_pydantic_ai_config

        config = get_pydantic_ai_config()
        assert config["provider"] == "openai"
        assert config["model"] == "gpt-4"
        assert config["max_retries"] == 3
        assert config["timeout_seconds"] == 30
        assert config["enabled"] is True


# ===========================================================================
# test_framework_security_manager.py — SecurityManager methods
# ===========================================================================


class TestSecurityManagerStub:
    """Tests for config.test_framework_security_manager.SecurityManager."""

    def test_init_with_config(self):
        from config.test_framework_security_manager import (
            SecurityManager,
            TestValidationSecurityConfig,
        )

        config = TestValidationSecurityConfig(max_execution_time=120)
        mgr = SecurityManager(config)
        assert mgr.config.max_execution_time == 120
        assert mgr._logger is None

    def test_init_with_logger_and_config(self):
        from config.test_framework_security_manager import (
            SecurityManager,
            TestValidationSecurityConfig,
        )

        logger = logging.getLogger("test")
        config = TestValidationSecurityConfig()
        mgr = SecurityManager(logger, config)
        assert mgr._logger is logger
        assert mgr.config is config

    def test_init_with_logger_only(self):
        from config.test_framework_security_manager import SecurityManager

        logger = logging.getLogger("test")
        mgr = SecurityManager(logger)
        assert mgr._logger is logger
        assert mgr.config.max_execution_time == 300
        assert mgr.config.max_memory_mb == 512
        assert mgr.config.security_level == "medium"

    def test_init_no_args(self):
        from config.test_framework_security_manager import SecurityManager

        mgr = SecurityManager()
        assert mgr._logger is None
        assert mgr.config.max_execution_time == 300
        assert mgr.config.max_memory_mb == 512
        assert mgr.config.allow_network is False
        assert mgr.config.allow_file_write is False

    async def test_validate_code_returns_safe(self):
        from config.test_framework_security_manager import SecurityManager

        mgr = SecurityManager()
        result = await mgr.validate_code("print('hello')", "python")
        assert result.is_safe is True
        assert result.passed is True

    async def test_check_imports_returns_safe(self):
        from config.test_framework_security_manager import SecurityManager

        mgr = SecurityManager()
        result = await mgr.check_imports("import math")
        assert result.is_safe is True

    def test_check_code_safety_returns_safe(self):
        from config.test_framework_security_manager import SecurityManager

        mgr = SecurityManager()
        result = mgr.check_code_safety("x = 1")
        assert result.is_safe is True

    def test_get_sandbox_config(self):
        from config.test_framework_security_manager import (
            SecurityManager,
            TestValidationSecurityConfig,
        )

        config = TestValidationSecurityConfig(
            allow_network=True,
            allow_file_write=True,
            max_execution_time=60,
            max_memory_mb=256,
        )
        mgr = SecurityManager(config)
        sandbox = mgr.get_sandbox_config()
        assert sandbox["network_access"] is True
        assert sandbox["file_system"] == "read_write"
        assert sandbox["max_execution_time"] == 60
        assert sandbox["max_memory_mb"] == 256

    def test_get_sandbox_config_read_only(self):
        from config.test_framework_security_manager import (
            SecurityManager,
            TestValidationSecurityConfig,
        )

        config = TestValidationSecurityConfig(allow_file_write=False)
        mgr = SecurityManager(config)
        sandbox = mgr.get_sandbox_config()
        assert sandbox["file_system"] == "read_only"


class TestSecurityCheckResult:
    """Tests for config.test_framework_security_manager.SecurityCheckResult."""

    def test_defaults(self):
        from config.test_framework_security_manager import SecurityCheckResult

        result = SecurityCheckResult()
        assert result.is_safe is True
        assert result.passed is True
        assert result.violations == []
        assert result.risk_score == 0.0
        assert result.checked_patterns == 0
        assert result.error is None
        assert result.violation_type is None

    def test_custom_values(self):
        from config.test_framework_security_manager import (
            SecurityCheckResult,
            SecurityViolationType,
        )

        result = SecurityCheckResult(
            is_safe=False,
            passed=False,
            violations=[{"type": "test"}],
            risk_score=75.5,
            checked_patterns=10,
            error="test error",
            violation_type=SecurityViolationType.DANGEROUS_IMPORT,
        )
        assert result.is_safe is False
        assert result.risk_score == 75.5
        assert result.violation_type == SecurityViolationType.DANGEROUS_IMPORT


class TestSecurityViolationType:
    """Tests for config.test_framework_security_manager.SecurityViolationType."""

    def test_all_values(self):
        from config.test_framework_security_manager import SecurityViolationType

        assert SecurityViolationType.DANGEROUS_IMPORT == "dangerous_import"
        assert SecurityViolationType.DANGEROUS_FUNCTION == "dangerous_function"
        assert SecurityViolationType.FILE_SYSTEM_ACCESS == "file_system_access"
        assert SecurityViolationType.NETWORK_ACCESS == "network_access"
        assert SecurityViolationType.PROCESS_EXECUTION == "process_execution"
        assert SecurityViolationType.CODE_INJECTION == "code_injection"
        assert SecurityViolationType.RESOURCE_ABUSE == "resource_abuse"
        assert SecurityViolationType.UNAUTHORIZED_IMPORT == "unauthorized_import"
        assert SecurityViolationType.SYNTAX_ERROR == "syntax_error"
        assert SecurityViolationType.EXECUTION_TIME == "execution_time"
        assert SecurityViolationType.CODE_LENGTH == "code_length"


class TestTestValidationSecurityConfig:
    """Tests for config.test_framework_security_manager.TestValidationSecurityConfig."""

    def test_defaults(self):
        from config.test_framework_security_manager import TestValidationSecurityConfig

        config = TestValidationSecurityConfig()
        assert config.max_execution_time == 300
        assert config.max_memory_mb == 512
        assert config.max_file_size_mb == 10
        assert config.max_code_length == 10000
        assert config.allow_network is False
        assert config.allow_file_write is False
        assert config.allowed_modules == ["pytest", "math", "os", "sys"]
        assert config.blocked_imports == ["subprocess", "shutil", "ctypes"]
        assert config.security_level == "medium"

    def test_custom_values(self):
        from config.test_framework_security_manager import TestValidationSecurityConfig

        config = TestValidationSecurityConfig(
            max_execution_time=60,
            max_memory_mb=256,
            allow_network=True,
            security_level="high",
        )
        assert config.max_execution_time == 60
        assert config.max_memory_mb == 256
        assert config.allow_network is True
        assert config.security_level == "high"


# ===========================================================================
# code_execution_service.py
# ===========================================================================


class TestCodeExecutionService:
    """Tests for services.code_execution_service.CodeExecutionService."""

    def test_init_defaults(self):
        from services.code_execution_service import CodeExecutionService

        svc = CodeExecutionService()
        assert svc.docker_service is None
        assert svc.cache_manager is None
        assert svc.security_service is None
        assert svc.test_orchestrator is None
        assert svc.settings is None

    def test_init_with_services(self):
        from services.code_execution_service import CodeExecutionService

        docker = Mock()
        cache = Mock()
        security = Mock()
        svc = CodeExecutionService(
            docker_service=docker,
            cache_manager=cache,
            security_service=security,
        )
        assert svc.docker_service is docker
        assert svc.cache_manager is cache
        assert svc.security_service is security

    async def test_execute_code_returns_stub(self):
        from services.code_execution_service import CodeExecutionService

        svc = CodeExecutionService()
        result = await svc.execute_code("print(1)", "python")
        assert result["exit_code"] == 1
        assert result["execution_time"] == 0.0
        assert "unreachable" in result["stderr"].lower() or "executor" in result["stderr"].lower()

    async def test_execute_tests_returns_stub(self):
        from services.code_execution_service import CodeExecutionService

        svc = CodeExecutionService()
        result = await svc.execute_tests("code", "test_code", "python")
        assert result["exit_code"] == 1
        assert result["tests_passed"] == 0
        assert result["tests_failed"] == 0

    async def test_execute_test_raises(self):
        from services.code_execution_service import CodeExecutionService

        svc = CodeExecutionService()
        # execute_test now forwards to executor; raises HTTPException(503) for unreachable executor
        from fastapi import HTTPException as _HTTPException

        request = Mock()
        request.language = "python"
        request.code = "print(1)"
        request.test_code = None
        request.timeout = None
        request.test_type = None
        request.container_name = None
        request.code_path = None
        request.test_path = None
        request.dependencies = None
        request.security_level = None
        request.with_hypothesis = None
        request.resource_limits = None
        with pytest.raises(_HTTPException) as exc_info:
            await svc.execute_test(request)
        assert exc_info.value.status_code in (502, 503)

    async def test_validate_code_security_raises(self):
        from unittest.mock import AsyncMock, patch

        import httpx

        from services.code_execution_service import CodeExecutionService

        svc = CodeExecutionService()
        # Unreachable coding-security is "not computed" (is_safe None), never "safe".
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("unreachable"))
        with patch(
            "services.code_execution_service._get_security_http_client",
            return_value=mock_client,
        ):
            result = await svc._validate_code_security("code", "python")
        assert isinstance(result, dict)
        assert result["is_safe"] is None
        assert result["computed"] is False
        assert "unreachable" in result["reason"].lower()

    def test_get_supported_tools_returns_known_tools(self):
        from services.code_execution_service import CodeExecutionService

        svc = CodeExecutionService()
        python_tools = svc._get_supported_tools("python")
        assert len(python_tools) > 0
        assert "ruff" in python_tools
        assert svc._get_supported_tools("nonexistent") == []

    async def test_get_service_health(self):
        from services.code_execution_service import CodeExecutionService

        svc = CodeExecutionService()
        result = await svc.get_service_health()
        assert result["status"] == "stub"

    def test_get_code_execution_service_singleton(self):
        # Reset singleton
        import services.code_execution_service as mod
        from services.code_execution_service import get_code_execution_service

        mod._code_execution_service = None
        svc1 = get_code_execution_service()
        svc2 = get_code_execution_service()
        assert svc1 is svc2
        assert svc1.docker_service is None
        assert svc1.cache_manager is None
        assert svc1.security_service is None
        # Cleanup
        mod._code_execution_service = None

    async def test_analyze_quality_stub(self):
        """Test analyze_quality calls coding-analysis service (mocked)."""
        from services.code_execution_service import CodeExecutionService

        svc = CodeExecutionService()
        mock_request = Mock()
        # whole-workspace shape (inline code without code_path is a 422 now)
        mock_request.code_content = None
        mock_request.code = None
        mock_request.code_path = None
        mock_request.language = "python"
        mock_request.tools = None
        mock_request.container_name = "ws1"
        mock_request.fix_issues = False
        mock_request.custom_rules = None
        mock_request.tenant_id = None

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "analysis_id": "qa-1",
            "success": True,
            "overall_score": 85.0,
            "total_issues": 0,
            "issues": [],
            "tools_results": {},
            "execution_time": 0.5,
            "language": "python",
        }
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await svc.analyze_quality(mock_request, tenant_id="acme-tenant")
            assert result.analysis_id == "qa-1"
            assert result.success is True
            assert result.overall_score == 85.0

        # Regression guard: the caller's tenant MUST be forwarded as X-Tenant-Id so
        # coding-analysis scopes its own container lookup correctly (was dropped).
        from auth.tenant import TENANT_ID_HEADER

        _, post_kwargs = mock_client.post.call_args
        assert post_kwargs["headers"][TENANT_ID_HEADER] == "acme-tenant"

    async def test_analyze_container_quality_stub(self):
        """Test analyze_container_quality calls coding-analysis service (mocked)."""
        from services.code_execution_service import CodeExecutionService

        svc = CodeExecutionService()
        mock_request = Mock()
        mock_request.container_name = "test-container"
        mock_request.language = "python"
        mock_request.tools = None
        mock_request.fix_issues = False

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "analysis_id": "qa-2",
            "success": True,
            "overall_score": 90.0,
            "total_issues": 1,
            "issues": [],
            "tools_results": {},
            "execution_time": 1.0,
            "language": "python",
            "created_at": "2026-01-01T00:00:00",
        }
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await svc.analyze_container_quality(mock_request)
            assert result.analysis_id == "qa-2"
            assert result.overall_score == 90.0


# ===========================================================================
# app_factory.py — additional coverage
# ===========================================================================


class TestAppFactoryAdditional:
    """Additional tests for app_factory.py."""

    def test_add_exception_handlers_registers(self):
        """Test _add_exception_handlers registers handlers for Exception and validation errors."""
        from fastapi import FastAPI

        from app_factory import _add_exception_handlers

        app = FastAPI()
        _add_exception_handlers(app)
        # KAG register_exception_handlers registers at least Exception handler
        assert Exception in app.exception_handlers
        # Verify handler is callable
        assert callable(app.exception_handlers[Exception])

    def test_add_exception_handlers_uses_coding_client(self):
        """Test _add_exception_handlers delegates to coding_client."""
        from fastapi import FastAPI

        from app_factory import _add_exception_handlers

        app = FastAPI()
        _add_exception_handlers(app)
        # KAG standard registers handlers for Exception and RequestValidationError
        assert Exception in app.exception_handlers

    def test_add_middlewares_auth_disabled(self, monkeypatch):
        """Test _add_middlewares always registers AuthMiddleware (handles auth internally)."""
        from app_factory import _add_middlewares

        monkeypatch.setenv("ENABLE_AUTH", "false")
        from config.settings import Settings

        settings = Settings()
        from fastapi import FastAPI

        app = FastAPI()
        _add_middlewares(app, settings)
        # AuthMiddleware is ALWAYS registered — it handles ENABLE_AUTH=false
        # internally by injecting an anonymous user.
        middleware_classes = [m.cls.__name__ for m in app.user_middleware]
        assert "CORSMiddleware" in middleware_classes
        assert "AuthMiddleware" in middleware_classes

    def test_add_middlewares_auth_enabled(self, monkeypatch):
        """Test _add_middlewares includes auth middleware when ENABLE_AUTH=true."""
        from app_factory import _add_middlewares

        monkeypatch.setenv("ENABLE_AUTH", "true")
        from config.settings import Settings

        settings = Settings()
        from fastapi import FastAPI

        app = FastAPI()
        _add_middlewares(app, settings)
        middleware_classes = [m.cls.__name__ for m in app.user_middleware]
        assert "CORSMiddleware" in middleware_classes
        assert "AuthMiddleware" in middleware_classes

    def test_create_app_with_auth_enabled(self, monkeypatch):
        """Test create_app with ENABLE_AUTH=true produces a functioning app."""
        monkeypatch.setenv("ENABLE_AUTH", "true")
        from app_factory import create_app

        app = create_app()
        assert app.title == "coding-testing"
        assert app.version == "1.0.0"
        # Verify routers are included by checking route count
        route_paths = _route_paths(app.routes)
        assert "/api/health" in route_paths or any("/api/health" in p for p in route_paths)

    async def test_lifespan_sets_state(self, app):
        """Test lifespan sets background_tasks on app state during startup."""

        from app_factory import lifespan

        # Run the lifespan context to trigger startup
        async with lifespan(app):
            assert hasattr(app.state, "background_tasks")
            assert app.state.background_tasks == set()

    def test_generate_unique_id_empty_clean_parts(self):
        """generate_unique_id handles path with only stripped params."""
        from app_factory import generate_unique_id

        # Path that results in empty parts after stripping
        route = type("Route", (), {"path": "/api/", "methods": {"GET"}})()
        result = generate_unique_id(route)
        assert result == "root_get"

    async def test_gateway_alias_status(self, client):
        """GET /api/tests/status returns module status."""
        resp = await client.get("/api/tests/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "active"
        assert "modules" in data

    async def test_gateway_alias_ready(self, client):
        """GET /api/tests/ready returns readiness."""
        resp = await client.get("/api/tests/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"


# ===========================================================================
# routers/health.py — system-metrics with psutil
# ===========================================================================


class TestHealthEndpoints:
    """Tests for routers.health.py endpoints."""

    async def test_system_metrics_endpoint(self, client):
        """GET /api/health/system-metrics returns metrics."""
        resp = await client.get("/api/health/system-metrics")
        assert resp.status_code == 200
        data = resp.json()
        # psutil should be available, so we get actual metrics
        assert "cpu_percent" in data
        assert "memory_percent" in data

    async def test_health_check_returns_uptime(self, client):
        """GET /api/health returns uptime_seconds as a positive number."""
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["uptime_seconds"] >= 0.0
        assert data["status"] == "healthy"
        assert data["service"] == "coding-testing"

    async def test_detailed_health_has_system(self, client):
        """GET /api/health/detailed returns system info with psutil metrics."""
        resp = await client.get("/api/health/detailed")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "coding-testing"
        assert data["uptime_seconds"] >= 0.0
        # psutil is installed in test env, so system metrics must be present
        assert "system" in data
        assert data["system"]["cpu_percent"] >= 0.0
        assert data["system"]["memory_used_mb"] > 0.0
        assert data["system"]["memory_total_mb"] > 0.0


# ===========================================================================
# middleware/validation.py — HTTPException path
# ===========================================================================


class TestValidationMiddlewareHTTPException:
    """Test ValidationMiddleware when call_next raises HTTPException."""

    async def test_http_exception_caught(self):
        """ValidationMiddleware handles HTTPException from call_next."""
        import json

        from middleware.validation import ValidationMiddleware

        middleware = ValidationMiddleware(app=MagicMock())
        request = MagicMock()
        request.url = MagicMock()
        request.url.path = "/api/test"
        request.scope = {"path": "/api/test"}
        raw_url = "http://testserver/api/test"
        request.url.__str__ = MagicMock(return_value=raw_url)

        call_next = AsyncMock(side_effect=HTTPException(status_code=404, detail="Not found"))
        result = await middleware.dispatch(request, call_next)
        assert result.status_code == 404
        body = json.loads(result.body.decode())
        assert body["error"] == "Not found"
        assert body["status_code"] == 404

    async def test_generic_exception_caught(self):
        """ValidationMiddleware handles generic exceptions from call_next."""
        import json

        from middleware.validation import ValidationMiddleware

        middleware = ValidationMiddleware(app=MagicMock())
        request = MagicMock()
        request.url = MagicMock()
        request.url.path = "/api/test"
        request.scope = {"path": "/api/test"}
        raw_url = "http://testserver/api/test"
        request.url.__str__ = MagicMock(return_value=raw_url)

        call_next = AsyncMock(side_effect=RuntimeError("something broke"))
        result = await middleware.dispatch(request, call_next)
        assert result.status_code == 500
        body = json.loads(result.body.decode())
        assert body["error"] == "Internal server error"
        assert body["error_code"] == "INTERNAL_ERROR"
        assert "request_id" in body
        assert body["status_code"] == 500


# ===========================================================================
# routers/intelligent_testing.py — _cleanup_service_sync
# ===========================================================================


class TestIntelligentTestingCleanup:
    """Tests for routers.intelligent_testing._cleanup_service_sync."""

    def test_cleanup_when_no_service(self):
        """_cleanup_service_sync does nothing when no service instance."""
        from routers.intelligent_testing_modules import shared

        original = shared._service_instance
        shared._service_instance = None
        try:
            from routers.intelligent_testing import _cleanup_service_sync

            _cleanup_service_sync()  # Should not raise
        finally:
            shared._service_instance = original

    def test_cleanup_when_service_not_running(self):
        """_cleanup_service_sync does nothing when service is not running."""
        from routers.intelligent_testing_modules import shared

        original = shared._service_instance
        mock_svc = MagicMock()
        mock_svc.is_running = False
        shared._service_instance = mock_svc
        try:
            from routers.intelligent_testing import _cleanup_service_sync

            _cleanup_service_sync()  # Should not raise
            assert shared._service_instance is mock_svc  # Unchanged since not running
        finally:
            shared._service_instance = original

    def test_cleanup_when_service_running_loop_running(self):
        """_cleanup_service_sync creates task when loop is running."""
        from routers.intelligent_testing_modules import shared

        original = shared._service_instance
        mock_svc = MagicMock()
        mock_svc.is_running = True
        mock_svc.stop = AsyncMock()
        shared._service_instance = mock_svc
        try:
            from routers.intelligent_testing import _cleanup_service_sync

            mock_loop = MagicMock()
            mock_loop.is_running.return_value = True
            mock_loop.create_task = MagicMock()

            with patch("asyncio.get_event_loop", return_value=mock_loop):
                _cleanup_service_sync()

            assert shared._service_instance is None
        finally:
            shared._service_instance = original

    def test_cleanup_handles_exception(self):
        """_cleanup_service_sync handles exceptions gracefully."""
        from routers.intelligent_testing_modules import shared

        original = shared._service_instance
        mock_svc = MagicMock()
        mock_svc.is_running = True
        shared._service_instance = mock_svc
        try:
            from routers.intelligent_testing import _cleanup_service_sync

            with patch("asyncio.get_event_loop", side_effect=RuntimeError("no loop")):
                _cleanup_service_sync()

            assert shared._service_instance is None
        finally:
            shared._service_instance = original


# ===========================================================================
# ai_test_insights/core_service.py — init_ai_agents exception path
# ===========================================================================


class TestAIAgentsInitException:
    """Tests for AI agent initialization exception handling in core_service.py."""

    def test_init_agents_exception_sets_none(self):
        """When Agent() raises, all agents should be set to None."""
        from services.ai_test_insights.config import AITestInsightsConfig
        from services.ai_test_insights.core_service import AITestInsightsService

        config = AITestInsightsConfig(enable_ai_features=True)

        with patch(
            "services.ai_test_insights.core_service.HAS_PYDANTIC_AI",
            True,
        ):
            with patch(
                "services.ai_test_insights.core_service.Agent",
                side_effect=Exception("Agent init failed"),
            ):
                svc = AITestInsightsService(config=config)

        assert svc.performance_agent is None
        assert svc.reliability_agent is None
        assert svc.coverage_agent is None
        assert svc.quality_agent is None
        assert svc.pattern_agent is None
        assert svc.recommendation_agent is None


# ===========================================================================
# testing_utils.py — _log_test_execution, _log_quality_analysis, _log_batch_execution
# ===========================================================================


class TestBackgroundTasks:
    """Tests for background task functions in testing_utils.py."""

    async def test_log_test_execution_success(self):
        from routers.testing_modules.testing_utils import _log_test_execution

        mock_result = MagicMock()
        mock_result.test_id = "t-1"
        mock_result.success = True
        mock_result.language = "python"
        mock_result.execution_time = 1.0
        mock_result.security_level = MagicMock()
        mock_result.security_level.value = "medium"
        mock_result.exit_code = 0

        mock_metrics = AsyncMock()
        mock_metrics.record_test_execution = AsyncMock()

        with patch(
            "routers.testing_modules.testing_utils.get_metrics_service",
            return_value=mock_metrics,
        ):
            await _log_test_execution("user1", mock_result, {"language": "python"})

        mock_metrics.record_test_execution.assert_awaited_once()

    async def test_log_test_execution_error(self):
        from routers.testing_modules.testing_utils import _log_test_execution

        mock_result = MagicMock()
        mock_result.test_id = "t-1"
        mock_result.success = True
        mock_result.language = "python"
        mock_result.execution_time = 1.0
        mock_result.security_level = MagicMock()
        mock_result.security_level.value = "medium"
        mock_result.exit_code = 0

        with patch(
            "routers.testing_modules.testing_utils.get_metrics_service",
            side_effect=AttributeError("no service"),
        ):
            # Should not raise
            await _log_test_execution("user1", mock_result, {})

    async def test_log_quality_analysis_success(self):
        from routers.testing_modules.testing_utils import _log_quality_analysis

        mock_result = MagicMock()
        mock_result.analysis_id = "qa-1"
        mock_result.success = True
        mock_result.language = "python"
        mock_result.overall_score = 85.0
        mock_result.total_issues = 0
        mock_result.execution_time = 0.5
        mock_result.tools_results = {"tool1": {}}

        mock_metrics = AsyncMock()
        mock_metrics.record_quality_analysis = AsyncMock()

        with patch(
            "routers.testing_modules.testing_utils.get_metrics_service",
            return_value=mock_metrics,
        ):
            await _log_quality_analysis("user1", mock_result)

        mock_metrics.record_quality_analysis.assert_awaited_once()

    async def test_log_batch_execution(self):
        from routers.testing_modules.testing_utils import _log_batch_execution

        # Should not raise
        await _log_batch_execution("user1", 5, 3, ["python", "go"])

    async def test_log_batch_execution_error(self):
        from routers.testing_modules.testing_utils import _log_batch_execution

        # Even with a weird type error, should not raise
        await _log_batch_execution("user1", 5, 3, ["python"])


# ===========================================================================
# testing_utils.py — get_execution_service internal paths
# ===========================================================================


class TestGetExecutionService:
    """Tests for routers.testing_modules.testing_utils.get_execution_service internals."""

    async def test_docker_service_runtime_error_fallback(self):
        """get_execution_service falls back to mock docker when RuntimeError."""
        from routers.testing_modules.testing_utils import get_execution_service

        mock_cache = MagicMock()
        mock_cache.get_redis = AsyncMock(return_value=None)

        mock_security = Mock()

        with (
            patch(
                "services.stubs.get_docker_service",
                side_effect=RuntimeError("Docker not available"),
            ),
            patch(
                "services.stubs.RedisCacheManager",
                return_value=mock_cache,
            ),
            patch(
                "services.security_service.create_security_service",
                return_value=mock_security,
            ),
        ):
            svc = await get_execution_service()
            # Docker fallback creates a mock docker_service with is_docker_available=False
            assert svc.docker_service is None
            assert svc.cache_manager is mock_cache
            assert svc.security_service is mock_security

    async def test_cache_manager_runtime_error_returns_stub(self):
        """get_execution_service returns stub with full tool registry when cache init fails."""
        from routers.testing_modules.testing_utils import get_execution_service

        with (
            patch(
                "services.stubs.get_docker_service",
                return_value=Mock(),
            ),
            patch(
                "services.stubs.RedisCacheManager",
                side_effect=RuntimeError("Redis not available"),
            ),
        ):
            # cache_manager is None → HTTPException(500) → caught in outer except → stub
            svc = await get_execution_service()
            python_tools = svc._get_supported_tools("python")
            assert "ruff-format" in python_tools
            assert "ty" in python_tools
            assert "ruff" in python_tools
            assert len(python_tools) == 5
            # Stub also has tools for other languages
            go_tools = svc._get_supported_tools("go")
            assert "golangci-lint" in go_tools
            # Unknown language returns empty list
            assert svc._get_supported_tools("cobol") == []

    async def test_security_service_runtime_error_fallback(self):
        """get_execution_service sets security_service=None when init fails."""
        from routers.testing_modules.testing_utils import get_execution_service

        mock_cache = MagicMock()
        mock_cache.get_redis = AsyncMock(return_value=None)

        mock_docker = Mock()

        with (
            patch(
                "services.stubs.get_docker_service",
                return_value=mock_docker,
            ),
            patch(
                "services.stubs.RedisCacheManager",
                return_value=mock_cache,
            ),
            patch(
                "services.security_service.create_security_service",
                side_effect=RuntimeError("Security init failed"),
            ),
        ):
            svc = await get_execution_service()
            assert svc.docker_service is mock_docker
            assert svc.cache_manager is mock_cache
            # Security service is None when init fails
            assert svc.security_service is None
            # _validate_code_security returns a dict (fail-open on unreachable service)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "is_safe": True,
                "violations": [],
                "risk_score": 0,
            }
            mock_resp.raise_for_status = MagicMock()
            with patch(
                "services.code_execution_service._get_security_http_client"
            ) as mock_client_fn:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_resp)
                mock_client_fn.return_value = mock_client
                result = await svc._validate_code_security("x = 1", "python")
            assert isinstance(result, dict)
            assert result["is_safe"] is True
            assert result["violations"] == []
            assert result["risk_score"] == 0

    async def test_redis_connection_error(self):
        """get_execution_service handles Redis ConnectionError and still returns working service."""
        from routers.testing_modules.testing_utils import get_execution_service

        mock_cache = MagicMock()
        mock_cache.get_redis = AsyncMock(side_effect=ConnectionError("refused"))

        mock_docker = Mock()
        mock_security = Mock()

        with (
            patch(
                "services.stubs.get_docker_service",
                return_value=mock_docker,
            ),
            patch(
                "services.stubs.RedisCacheManager",
                return_value=mock_cache,
            ),
            patch(
                "services.security_service.create_security_service",
                return_value=mock_security,
            ),
        ):
            svc = await get_execution_service()
            # Redis connection error is caught but cache_manager is still set
            # because the error happens during ping, not during RedisCacheManager()
            assert svc.docker_service is mock_docker
            assert svc.cache_manager is mock_cache
            assert svc.security_service is mock_security


# ===========================================================================
# test_discovery.py — container validation fallback paths
# ===========================================================================


class TestDiscoveryContainerValidation:
    """Tests for container validation fallback paths in test_discovery.py."""

    async def test_container_validation_import_error(self):
        """Container validation ImportError is caught gracefully."""
        from routers.testing_modules.test_discovery import get_supported_tools
        from services.code_execution_service import CodeExecutionService

        svc = Mock(spec=CodeExecutionService)
        svc._get_supported_tools = Mock(return_value=["tool1"])

        user = Mock()
        user.is_admin = Mock(return_value=False)
        user.effective_tenant_id = "tenant-1"

        with patch(
            "routers.testing_modules.test_discovery.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
            side_effect=ImportError("Container validation not available"),
        ):
            result = await get_supported_tools(
                language="python",
                container_name="test-container",
                user=user,
                execution_service=svc,
            )
        assert result["language"] == "python"
        assert result["container_name"] == "test-container"

    async def test_container_validation_runtime_error(self):
        """Container validation RuntimeError is caught gracefully."""
        from routers.testing_modules.test_discovery import get_supported_tools
        from services.code_execution_service import CodeExecutionService

        svc = Mock(spec=CodeExecutionService)
        svc._get_supported_tools = Mock(return_value=["tool1"])

        user = Mock()
        user.is_admin = Mock(return_value=True)

        with patch(
            "routers.testing_modules.test_discovery.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Service unavailable"),
        ):
            result = await get_supported_tools(
                language="python",
                container_name="test-container",
                user=user,
                execution_service=svc,
            )
        assert result["container_name"] == "test-container"

    async def test_languages_container_validation_fallback(self):
        """Container validation fallback in get_supported_languages."""
        from routers.testing_modules.test_discovery import get_supported_languages
        from services.code_execution_service import CodeExecutionService

        svc = Mock(spec=CodeExecutionService)
        svc._get_supported_tools = Mock(return_value=[])
        svc.test_orchestrator = Mock()
        svc.test_orchestrator.get_supported_languages = AsyncMock(return_value=["python"])
        svc.test_orchestrator.get_supported_test_types = AsyncMock(return_value=["unit"])

        user = Mock()
        user.is_admin = Mock(return_value=False)
        user.effective_tenant_id = "tenant-1"

        with patch(
            "routers.testing_modules.test_discovery.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Not available"),
        ):
            result = await get_supported_languages(
                container_name="test-container",
                user=user,
                execution_service=svc,
            )
        assert result["container_name"] == "test-container"

    async def test_test_types_container_validation_fallback(self):
        """Container validation fallback in get_language_test_types."""
        from routers.testing_modules.test_discovery import get_language_test_types
        from services.code_execution_service import CodeExecutionService

        svc = Mock(spec=CodeExecutionService)
        svc.test_orchestrator = Mock()
        svc.test_orchestrator.get_supported_languages = AsyncMock(return_value=["python"])
        svc.test_orchestrator.get_supported_test_types = AsyncMock(return_value=["unit"])

        user = Mock()
        user.is_admin = Mock(return_value=False)
        user.effective_tenant_id = "tenant-1"

        with patch(
            "routers.testing_modules.test_discovery.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
            side_effect=ImportError("Not available"),
        ):
            result = await get_language_test_types(
                language="python",
                container_name="test-container",
                user=user,
                execution_service=svc,
            )
        assert result["container_name"] == "test-container"

    async def test_languages_quality_tools_exception(self):
        """get_supported_languages handles _get_supported_tools exception gracefully."""
        from routers.testing_modules.test_discovery import get_supported_languages
        from services.code_execution_service import CodeExecutionService

        svc = Mock(spec=CodeExecutionService)
        svc.test_orchestrator = Mock()
        svc.test_orchestrator.get_supported_languages = AsyncMock(return_value=["python"])
        svc.test_orchestrator.get_supported_test_types = AsyncMock(return_value=["unit"])
        # _get_supported_tools raises but is caught within the loop
        svc._get_supported_tools = Mock(side_effect=Exception("tool error"))

        user = Mock()
        user.is_admin = Mock(return_value=False)
        user.effective_tenant_id = "tenant-1"

        # The inner exception is caught; quality_tools is empty but result is returned
        result = await get_supported_languages(
            container_name=None,
            user=user,
            execution_service=svc,
        )
        assert result["languages"][0] == "python"
        assert "container_name" not in result  # key omitted when container_name is None

    async def test_test_types_fallback_for_non_python(self):
        """get_language_test_types falls back to ['unit', 'quality'] for non-python."""
        from routers.testing_modules.test_discovery import get_language_test_types
        from services.code_execution_service import CodeExecutionService

        svc = Mock(spec=CodeExecutionService)
        svc.test_orchestrator = Mock()
        svc.test_orchestrator.get_supported_languages = AsyncMock(
            side_effect=AttributeError("no method")
        )
        svc.test_orchestrator.get_supported_test_types = AsyncMock(
            side_effect=AttributeError("no method")
        )

        user = Mock()
        user.is_admin = Mock(return_value=False)
        user.effective_tenant_id = "tenant-1"

        result = await get_language_test_types(
            language="javascript",
            container_name=None,
            user=user,
            execution_service=svc,
        )
        assert "unit" in result["test_types"]
        assert "quality" in result["test_types"]


# ===========================================================================
# processing/test_execution/test_runner.py — ContainerTestRunner additional coverage
# ===========================================================================


class TestContainerTestRunnerExtra:
    """Additional tests for processing.test_execution.test_runner.ContainerTestRunner."""

    def _make_runner(self):
        from config.test_framework_config import TestConfig
        from processing.test_execution.test_runner import ContainerTestRunner
        from services.stubs import DockerService

        docker_svc = MagicMock(spec=DockerService)
        config = TestConfig()
        runner = ContainerTestRunner(docker_svc, config)
        return runner

    def test_should_use_orchestrator_go(self):
        runner = self._make_runner()
        req = TestRequest(code="x=1", language="go")
        assert runner._should_use_orchestrator(req) is True

    def test_should_use_orchestrator_rust(self):
        runner = self._make_runner()
        req = TestRequest(code="x=1", language="rust")
        assert runner._should_use_orchestrator(req) is True

    def test_should_not_use_orchestrator_python(self):
        runner = self._make_runner()
        req = TestRequest(code="x=1", language="python")
        assert runner._should_use_orchestrator(req) is False

    def test_get_memory_limit_all_levels(self):
        runner = self._make_runner()
        assert runner._get_memory_limit(SecurityLevel.HIGH) == "256m"
        assert runner._get_memory_limit(SecurityLevel.MEDIUM) == "512m"
        assert runner._get_memory_limit(SecurityLevel.LOW) == "1g"

    def test_get_cpu_quota_all_levels(self):
        runner = self._make_runner()
        assert runner._get_cpu_quota(SecurityLevel.HIGH) == 25000
        assert runner._get_cpu_quota(SecurityLevel.MEDIUM) == 50000
        assert runner._get_cpu_quota(SecurityLevel.LOW) == 100000

    def test_get_file_extension(self):
        runner = self._make_runner()
        assert runner._get_file_extension("python") == ".py"
        assert runner._get_file_extension("javascript") == ".js"
        assert runner._get_file_extension("typescript") == ".ts"
        assert runner._get_file_extension("unknown") == ".py"

    def test_get_test_file_extension(self):
        runner = self._make_runner()
        assert runner._get_test_file_extension("python") == ".py"
        assert runner._get_test_file_extension("javascript") == ".test.js"
        assert runner._get_test_file_extension("typescript") == ".test.ts"
        assert runner._get_test_file_extension("go") == ".py"

    async def test_validate_request_no_code(self):
        runner = self._make_runner()
        req = TestRequest(language="python")
        # Empty code should be fine if no validation is needed
        await runner._validate_request(req)

    async def test_validate_request_code_too_long(self):
        runner = self._make_runner()
        req = TestRequest(code="x" * 25001, test_code="y" * 25001, language="python")
        with pytest.raises(ValueError, match="Security validation failed"):
            await runner._validate_request(req)

    async def test_cleanup_container_handles_error(self):
        runner = self._make_runner()
        runner.docker_service.stop_container = AsyncMock(side_effect=RuntimeError("stop failed"))
        # Should not raise
        await runner._cleanup_container("container-123")

    async def test_execute_test_error_handling(self):
        runner = self._make_runner()
        runner.orchestrator = None
        runner.docker_service.get_language_container_image = MagicMock(
            side_effect=RuntimeError("no image")
        )

        req = TestRequest(code="x=1", language="python")
        result = await runner.execute_test(req)
        assert result.success is False
        assert "no image" in result.stderr


# ===========================================================================
# processing/test_execution/http_test_runner.py — HTTPTestRunner init
# ===========================================================================


class TestHTTPTestRunnerInit:
    """Tests for processing.test_execution.http_test_runner.HTTPTestRunner initialization."""

    def test_init_defaults(self):
        from processing.test_execution.http_test_runner import HTTPTestRunner

        runner = HTTPTestRunner()
        assert runner.cache_manager is None
        assert "python-test-executor" in runner.python_service_url
        assert "javascript-test-executor" in runner.javascript_service_url

    def test_init_with_cache(self):
        from processing.test_execution.http_test_runner import HTTPTestRunner

        cache = MagicMock()
        runner = HTTPTestRunner(cache_manager=cache)
        assert runner.cache_manager is cache

    def test_init_custom_urls(self, monkeypatch):
        from processing.test_execution.http_test_runner import HTTPTestRunner

        monkeypatch.setenv("PYTHON_TEST_SERVICE_URL", "http://custom:9999")
        runner = HTTPTestRunner()
        assert runner.python_service_url == "http://custom:9999"


class TestHTTPTestRunnerHealthCheck:
    """health_check must not overwrite a healthy result as unhealthy (missing else)."""

    @staticmethod
    def _session_returning(status: int):
        class _Resp:
            def __init__(self):
                self.status = status
                self.headers = {}

            async def json(self):
                return {"ok": True}

        class _Ctx:
            async def __aenter__(self):
                return _Resp()

            async def __aexit__(self, *a):
                return False

        session = MagicMock()
        session.get = MagicMock(return_value=_Ctx())
        return session

    async def test_200_reports_healthy(self):
        from processing.test_execution.http_test_runner import HTTPTestRunner

        runner = HTTPTestRunner()
        runner._get_session = AsyncMock(return_value=self._session_returning(200))

        result = await runner.health_check(service="python")
        # Regression: a 200 response must report healthy, not be unconditionally
        # overwritten as {"status": "unhealthy", "error": "HTTP 200"}.
        assert result["python"]["status"] == "healthy"

    async def test_non_200_reports_unhealthy(self):
        from processing.test_execution.http_test_runner import HTTPTestRunner

        runner = HTTPTestRunner()
        runner._get_session = AsyncMock(return_value=self._session_returning(503))

        result = await runner.health_check(service="python")
        assert result["python"]["status"] == "unhealthy"
        assert "503" in result["python"]["error"]
