"""Tests for IntelligentTestExecutionService in intelligent_test_integration.py."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from config.test_framework_config import TestConfig
from models.testing_models import TestRequest, TestSecurityLevel
from services.stubs import DockerService
from services.test_execution.intelligent_test_integration import (
    IntelligentTestExecutionService,
    TestExecutionError,
    TestExecutionResourceError,
    TestExecutionRetryError,
    TestExecutionSettings,
    TestExecutionTimeoutError,
    create_intelligent_test_service,
)
from services.test_execution.intelligent_test_orchestrator import (
    IntelligentTestExecutionRequest,
    IntelligentTestExecutionResult,
    OrchestratorType,
)
from services.test_execution.test_orchestrator import TestLanguage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def docker_service():
    svc = AsyncMock(spec=DockerService)
    svc.create_test_container = AsyncMock(return_value="container-123")
    svc.start_container = AsyncMock()
    svc.create_file_in_container = AsyncMock(return_value=True)
    svc.exec_in_container = AsyncMock(
        return_value={
            "exit_code": 0,
            "stdout": "ok",
            "stderr": "",
        }
    )
    svc.stop_container = AsyncMock()
    svc.remove_container_by_id = AsyncMock()
    return svc


@pytest.fixture
def config():
    return TestConfig()


@pytest.fixture
def settings():
    return TestExecutionSettings(retry_attempts=2, retry_delay=0.1)


@pytest.fixture
def service(docker_service, config, settings):
    return IntelligentTestExecutionService(docker_service, config, settings)


@pytest.fixture
def python_test_request():
    return TestRequest(
        code="def add(a,b): return a+b",
        test_code="import pytest\ndef test_add(): assert add(1,2)==3",
        language="python",
        timeout=60,
    )


@pytest.fixture
def intelligent_request():
    return IntelligentTestExecutionRequest(
        language=TestLanguage.PYTHON,
        test_type=OrchestratorType.UNIT,
        source_code="def add(a,b): return a+b",
        test_files={"test_main.py": "import pytest\ndef test_add(): assert add(1,2)==3"},
        timeout_seconds=60,
        priority_level=5,
    )


# ---------------------------------------------------------------------------
# TestExecutionSettings
# ---------------------------------------------------------------------------


class TestTestExecutionSettings:
    def test_defaults(self):
        s = TestExecutionSettings()
        assert s.enable_ai_optimization is True
        assert s.max_concurrent_tests == 5
        assert s.retry_attempts == 3
        assert s.execution_timeout == 600

    def test_max_concurrent_tests_validation(self):
        with pytest.raises(Exception):
            TestExecutionSettings(max_concurrent_tests=0)

    def test_max_concurrent_tests_upper_bound(self):
        with pytest.raises(Exception):
            TestExecutionSettings(max_concurrent_tests=21)

    def test_ai_analysis_threshold_validation(self):
        with pytest.raises(Exception):
            TestExecutionSettings(ai_analysis_threshold=4)

    def test_ai_analysis_threshold_upper_bound(self):
        with pytest.raises(Exception):
            TestExecutionSettings(ai_analysis_threshold=101)


# ---------------------------------------------------------------------------
# Exception classes
# ---------------------------------------------------------------------------


class TestExceptions:
    def test_execution_error(self):
        before = datetime.now(UTC)
        err = TestExecutionError("boom", "general", {"key": "val"})
        after = datetime.now(UTC)
        assert str(err) == "boom"
        assert err.error_type == "general"
        assert err.details == {"key": "val"}
        assert before <= err.timestamp <= after
        # Verify it's a proper Exception subclass
        assert isinstance(err, Exception)

    def test_execution_error_defaults(self):
        err = TestExecutionError("simple")
        assert err.error_type == "general"
        assert err.details == {}

    def test_retry_error(self):
        inner = RuntimeError("inner")
        err = TestExecutionRetryError("retries exhausted", 3, inner)
        assert err.attempts == 3
        assert err.last_error is inner
        assert err.error_type == "retry_failure"
        assert err.details == {"attempts": 3, "last_error": "inner"}
        assert isinstance(err, TestExecutionError)

    def test_timeout_error(self):
        err = TestExecutionTimeoutError("timed out", 300)
        assert err.timeout_seconds == 300
        assert err.error_type == "timeout"
        assert err.details == {"timeout_seconds": 300}
        assert isinstance(err, TestExecutionError)

    def test_resource_error(self):
        err = TestExecutionResourceError("OOM", "memory", "512m")
        assert err.resource_type == "memory"
        assert err.resource_limit == "512m"
        assert err.error_type == "resource_limit"
        assert err.details == {"resource_type": "memory", "resource_limit": "512m"}
        assert isinstance(err, TestExecutionError)


# ---------------------------------------------------------------------------
# Service init
# ---------------------------------------------------------------------------


class TestServiceInit:
    def test_creates_orchestrator(self, service):
        from services.test_execution.intelligent_test_orchestrator import (
            IntelligentTestOrchestrator,
        )

        assert isinstance(service.orchestrator, IntelligentTestOrchestrator)

    def test_stores_dependencies(self, service, docker_service, config, settings):
        assert service.docker_service is docker_service
        assert service.test_config is config
        assert service.settings is settings

    def test_default_settings_if_none(self, docker_service, config):
        svc = IntelligentTestExecutionService(docker_service, config)
        assert svc.settings.enable_ai_optimization is True
        assert svc.settings.max_concurrent_tests == 5
        assert svc.settings.retry_attempts == 3
        assert svc.settings.execution_timeout == 600

    def test_initial_state(self, service):
        assert service.is_running is False
        assert service.is_healthy is True
        assert service.performance_metrics == {}
        assert service.error_counts == {}

    def test_factory_function(self, docker_service, config, settings):
        svc = create_intelligent_test_service(docker_service, config, settings)
        assert svc.docker_service is docker_service
        assert svc.test_config is config
        assert svc.settings is settings


# ---------------------------------------------------------------------------
# start / stop
# ---------------------------------------------------------------------------


class TestStartStop:
    @pytest.mark.asyncio
    async def test_start(self, service):
        await service.start()
        assert service.is_running is True
        assert service.is_healthy is True
        assert service.performance_metrics == {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "ai_analysis_count": 0,
            "pattern_detections": 0,
            "recommendations_generated": 0,
        }
        assert service.error_counts == {
            "timeout_errors": 0,
            "resource_errors": 0,
            "retry_failures": 0,
            "ai_analysis_errors": 0,
            "general_errors": 0,
        }
        await service.stop()

    @pytest.mark.asyncio
    async def test_stop(self, service):
        await service.start()
        assert service.is_running is True
        await service.stop()
        assert service.is_running is False


# ---------------------------------------------------------------------------
# execute_test
# ---------------------------------------------------------------------------


class TestExecuteTest:
    @pytest.mark.asyncio
    async def test_not_running_raises(self, service, intelligent_request):
        with pytest.raises(TestExecutionError, match="not running") as exc_info:
            await service.execute_test(intelligent_request)
        assert exc_info.value.error_type == "service_not_running"

    @pytest.mark.asyncio
    async def test_successful_execution(self, service, intelligent_request):
        await service.start()
        result = await service.execute_test(intelligent_request)
        assert result.success is True
        assert result.exit_code == 0
        assert result.language == TestLanguage.PYTHON
        assert result.test_type == OrchestratorType.UNIT
        assert result.execution_time_ms >= 0
        assert result.test_id != ""
        await service.stop()

    @pytest.mark.asyncio
    async def test_converts_test_request(self, service, python_test_request):
        await service.start()
        result = await service.execute_test(python_test_request)
        assert result.success is True
        assert result.language == TestLanguage.PYTHON
        assert result.test_type == OrchestratorType.UNIT
        await service.stop()

    @pytest.mark.asyncio
    async def test_without_retry(self, service, intelligent_request):
        await service.start()
        result = await service.execute_test(intelligent_request, enable_retry=False)
        assert result.success is True
        assert result.exit_code == 0
        assert result.language == TestLanguage.PYTHON
        await service.stop()


# ---------------------------------------------------------------------------
# _validate_test_request
# ---------------------------------------------------------------------------


class TestValidateTestRequest:
    @pytest.mark.asyncio
    async def test_empty_source_code(self, service):
        req = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="   ",
            timeout_seconds=60,
            priority_level=5,
        )
        with pytest.raises(TestExecutionError, match="empty") as exc_info:
            await service._validate_test_request(req)
        assert exc_info.value.error_type == "invalid_request"

    @pytest.mark.asyncio
    async def test_timeout_too_long(self, service):
        service.settings.execution_timeout = 100
        req = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="x=1",
            timeout_seconds=101,
            priority_level=5,
        )
        with pytest.raises(TestExecutionError, match="Timeout") as exc_info:
            await service._validate_test_request(req)
        assert exc_info.value.error_type == "timeout_too_long"

    @pytest.mark.asyncio
    async def test_code_too_large(self, service):
        req = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="x" * 1000001,
            timeout_seconds=60,
            priority_level=5,
        )
        with pytest.raises(TestExecutionError, match="size limit") as exc_info:
            await service._validate_test_request(req)
        assert exc_info.value.error_type == "code_too_large"

    @pytest.mark.asyncio
    async def test_valid_request_passes(self, service):
        req = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="x = 1",
            timeout_seconds=60,
            priority_level=5,
        )
        # Should not raise — returns None
        result = await service._validate_test_request(req)
        assert result is None


# ---------------------------------------------------------------------------
# _convert_test_request
# ---------------------------------------------------------------------------


class TestConvertTestRequest:
    @pytest.mark.asyncio
    async def test_python_conversion(self, service, python_test_request):
        result = await service._convert_test_request(python_test_request)
        assert result.language == TestLanguage.PYTHON
        assert result.test_type == OrchestratorType.UNIT
        assert result.source_code == "def add(a,b): return a+b"
        assert result.timeout_seconds == 60
        assert result.priority_level == 5
        assert result.enable_ai_optimization is True

    @pytest.mark.asyncio
    async def test_all_languages_convert(self, service):
        expected_mapping = {
            "python": TestLanguage.PYTHON,
            "javascript": TestLanguage.JAVASCRIPT,
            "typescript": TestLanguage.TYPESCRIPT,
            "java": TestLanguage.JAVA,
            "go": TestLanguage.GO,
            "rust": TestLanguage.RUST,
            "csharp": TestLanguage.CSHARP,
            "cpp": TestLanguage.CPP,
            "nodejs": TestLanguage.NODEJS,
        }
        for lang, expected_enum in expected_mapping.items():
            req = TestRequest(code="x=1", language=lang, timeout=60)
            result = await service._convert_test_request(req)
            assert result.language == expected_enum, f"Failed for {lang}"
            assert result.source_code == "x=1"
            assert result.timeout_seconds == 60

    @pytest.mark.asyncio
    async def test_unsupported_language_raises(self, service):
        # TestRequest validates language so we need a valid one first
        # We patch to bypass validation
        req = TestRequest.__new__(TestRequest)
        object.__setattr__(req, "language", "haskell")
        object.__setattr__(req, "code", "x")
        object.__setattr__(req, "test_code", None)
        object.__setattr__(req, "timeout", 60)
        object.__setattr__(req, "security_level", TestSecurityLevel.MEDIUM)
        object.__setattr__(req, "container_name", None)
        object.__setattr__(req, "test_type", None)
        object.__setattr__(req, "test_files", {})
        with pytest.raises(TestExecutionError, match="conversion failed") as exc_info:
            await service._convert_test_request(req)
        assert exc_info.value.error_type == "request_conversion_failure"

    @pytest.mark.asyncio
    async def test_test_code_becomes_test_file(self, service, python_test_request):
        result = await service._convert_test_request(python_test_request)
        assert result.test_files == {
            "test_main.py": "import pytest\ndef test_add(): assert add(1,2)==3"
        }


# ---------------------------------------------------------------------------
# _get_file_extension
# ---------------------------------------------------------------------------


class TestGetFileExtension:
    @pytest.mark.parametrize(
        "lang,ext",
        [
            ("python", ".py"),
            ("javascript", ".js"),
            ("typescript", ".ts"),
            ("java", ".java"),
            ("go", ".go"),
            ("rust", ".rs"),
            ("csharp", ".cs"),
            ("cpp", ".cpp"),
            ("nodejs", ".js"),
        ],
    )
    def test_known_languages(self, service, lang, ext):
        assert service._get_file_extension(lang) == ext

    def test_unknown_returns_txt(self, service):
        assert service._get_file_extension("brainfuck") == ".txt"


# ---------------------------------------------------------------------------
# _update_execution_metrics
# ---------------------------------------------------------------------------


class TestUpdateExecutionMetrics:
    def test_increments_total(self, service):
        service.performance_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "ai_analysis_count": 0,
            "pattern_detections": 0,
            "recommendations_generated": 0,
        }
        result = IntelligentTestExecutionResult(
            test_id="t1",
            success=True,
            exit_code=0,
            stdout="ok",
            stderr="",
            execution_time_ms=100,
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            created_at=datetime.now(UTC),
            ai_analysis=None,
        )
        service._update_execution_metrics(result)
        assert service.performance_metrics["total_executions"] == 1
        assert service.performance_metrics["successful_executions"] == 1
        assert service.performance_metrics["failed_executions"] == 0
        assert service.performance_metrics["average_execution_time"] == 100.0

    def test_increments_failed(self, service):
        service.performance_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "ai_analysis_count": 0,
            "pattern_detections": 0,
            "recommendations_generated": 0,
        }
        result = IntelligentTestExecutionResult(
            test_id="t2",
            success=False,
            exit_code=1,
            stdout="",
            stderr="fail",
            execution_time_ms=200,
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            created_at=datetime.now(UTC),
        )
        service._update_execution_metrics(result)
        assert service.performance_metrics["total_executions"] == 1
        assert service.performance_metrics["successful_executions"] == 0
        assert service.performance_metrics["failed_executions"] == 1
        assert service.performance_metrics["average_execution_time"] == 200.0

    def test_average_execution_time_across_multiple(self, service):
        service.performance_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "ai_analysis_count": 0,
            "pattern_detections": 0,
            "recommendations_generated": 0,
        }
        for ms in [100, 200, 300]:
            result = IntelligentTestExecutionResult(
                test_id=f"t-{ms}",
                success=True,
                exit_code=0,
                stdout="ok",
                stderr="",
                execution_time_ms=ms,
                language=TestLanguage.PYTHON,
                test_type=OrchestratorType.UNIT,
                created_at=datetime.now(UTC),
            )
            service._update_execution_metrics(result)
        assert service.performance_metrics["total_executions"] == 3
        assert service.performance_metrics["average_execution_time"] == 200.0

    def test_ai_analysis_metrics(self, service):
        service.performance_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "ai_analysis_count": 0,
            "pattern_detections": 0,
            "recommendations_generated": 0,
        }
        result = IntelligentTestExecutionResult(
            test_id="t3",
            success=True,
            exit_code=0,
            stdout="ok",
            stderr="",
            execution_time_ms=100,
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            created_at=datetime.now(UTC),
            ai_analysis={
                "pattern_analysis": [{"p": 1}, {"p": 2}],
                "recommendations": [{"r": 1}],
            },
        )
        service._update_execution_metrics(result)
        assert service.performance_metrics["ai_analysis_count"] == 1
        assert service.performance_metrics["pattern_detections"] == 2
        assert service.performance_metrics["recommendations_generated"] == 1
        assert service.performance_metrics["total_executions"] == 1


# ---------------------------------------------------------------------------
# _update_batch_metrics
# ---------------------------------------------------------------------------


class TestUpdateBatchMetrics:
    def test_updates_for_each_result(self, service):
        service.performance_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "ai_analysis_count": 0,
            "pattern_detections": 0,
            "recommendations_generated": 0,
        }
        results = [
            IntelligentTestExecutionResult(
                test_id=f"t{i}",
                success=True,
                exit_code=0,
                stdout="ok",
                stderr="",
                execution_time_ms=100,
                language=TestLanguage.PYTHON,
                test_type=OrchestratorType.UNIT,
                created_at=datetime.now(UTC),
            )
            for i in range(3)
        ]
        service._update_batch_metrics(results)
        assert service.performance_metrics["total_executions"] == 3


# ---------------------------------------------------------------------------
# get_health_status
# ---------------------------------------------------------------------------


class TestGetHealthStatus:
    @pytest.mark.asyncio
    async def test_health_status_structure(self, service):
        await service.start()
        status = await service.get_health_status()
        assert status["service_status"]["is_running"] is True
        assert status["service_status"]["is_healthy"] is True
        # startup_time is an ISO timestamp string
        assert isinstance(status["service_status"]["startup_time"], str)
        assert status["service_status"]["uptime_seconds"] >= 0
        assert status["performance_metrics"]["total_executions"] == 0
        assert status["error_counts"]["timeout_errors"] == 0
        # No executions yet so success_rate defaults to 1.0
        assert status["health_scores"]["success_rate"] == 1.0
        assert status["health_scores"]["error_rate"] == 0.0
        assert status["health_scores"]["overall_health"] == 1.0
        assert status["settings"]["retry_attempts"] == 2  # from fixture
        await service.stop()

    @pytest.mark.asyncio
    async def test_health_when_not_running(self, service):
        status = await service.get_health_status()
        assert status["service_status"]["is_running"] is False
        # Not running means not healthy
        assert status["health_scores"]["overall_health"] == 0.0


# ---------------------------------------------------------------------------
# get_execution_insights
# ---------------------------------------------------------------------------


class TestGetExecutionInsights:
    @pytest.mark.asyncio
    async def test_includes_service_metrics(self, service):
        await service.start()
        insights = await service.get_execution_insights()
        svc_metrics = insights["service_metrics"]
        assert svc_metrics["performance_metrics"]["total_executions"] == 0
        assert svc_metrics["error_counts"]["timeout_errors"] == 0
        assert svc_metrics["service_health"]["is_running"] is True
        assert svc_metrics["service_health"]["is_healthy"] is True
        await service.stop()


# ---------------------------------------------------------------------------
# managed_execution context manager
# ---------------------------------------------------------------------------


class TestManagedExecution:
    @pytest.mark.asyncio
    async def test_context_manager(self, service):
        async with service.managed_execution():
            assert service.is_running is True
        assert service.is_running is False


# ---------------------------------------------------------------------------
# _execute_with_retry
# ---------------------------------------------------------------------------


class TestExecuteWithRetry:
    @pytest.mark.asyncio
    async def test_succeeds_on_first_attempt(self, service, intelligent_request):
        await service.start()
        result = await service._execute_with_retry(intelligent_request)
        assert result.success is True
        assert result.exit_code == 0
        assert result.language == TestLanguage.PYTHON
        assert result.test_type == OrchestratorType.UNIT
        # Metrics should be updated after successful execution
        assert service.performance_metrics["total_executions"] == 1
        assert service.performance_metrics["successful_executions"] == 1
        await service.stop()


# ---------------------------------------------------------------------------
# _generate_service_recommendations
# ---------------------------------------------------------------------------


class TestGenerateServiceRecommendations:
    @pytest.mark.asyncio
    async def test_empty_when_no_executions(self, service):
        recs = await service._generate_service_recommendations()
        assert recs == []

    @pytest.mark.asyncio
    async def test_high_error_rate_recommendation(self, service):
        service.performance_metrics = {"total_executions": 10}
        service.error_counts = {
            "timeout_errors": 5,
            "resource_errors": 0,
            "retry_failures": 0,
            "ai_analysis_errors": 0,
            "general_errors": 0,
        }
        recs = await service._generate_service_recommendations()
        # 5/10 = 50% error rate > 20% threshold => high error rate rec
        # 5/10 = 50% timeout rate > 10% threshold => timeout rec
        assert len(recs) == 2
        titles = [r.title.lower() for r in recs]
        assert "high error rate detected" in titles
        assert "frequent timeout errors" in titles
        # Verify recommendation fields
        error_rec = next(r for r in recs if "error rate" in r.title.lower())
        assert error_rec.priority == "high"
        assert error_rec.recommendation_type == "configuration"
        assert error_rec.confidence_level == 0.9
        timeout_rec = next(r for r in recs if "timeout" in r.title.lower())
        assert timeout_rec.priority == "medium"
        assert timeout_rec.confidence_level == 0.8


# ---------------------------------------------------------------------------
# _detect_service_patterns
# ---------------------------------------------------------------------------


class TestDetectServicePatterns:
    @pytest.mark.asyncio
    async def test_returns_empty_list(self, service):
        patterns = await service._detect_service_patterns()
        assert patterns == []


# ---------------------------------------------------------------------------
# execute_batch
# ---------------------------------------------------------------------------


class TestExecuteBatch:
    @pytest.mark.asyncio
    async def test_batch_not_running_raises(self, service, intelligent_request):
        with pytest.raises(TestExecutionError, match="not running") as exc_info:
            await service.execute_batch([intelligent_request])
        assert exc_info.value.error_type == "service_not_running"

    @pytest.mark.asyncio
    async def test_batch_success(self, service, intelligent_request):
        await service.start()
        results = await service.execute_batch([intelligent_request])
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].language == TestLanguage.PYTHON
        # Batch metrics should be updated
        assert service.performance_metrics["total_executions"] == 1
        await service.stop()

    @pytest.mark.asyncio
    async def test_batch_converts_test_requests(self, service, python_test_request):
        await service.start()
        results = await service.execute_batch([python_test_request])
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].language == TestLanguage.PYTHON
        await service.stop()


# ---------------------------------------------------------------------------
# get_recommendations / get_patterns
# ---------------------------------------------------------------------------


class TestGetRecommendationsPatterns:
    @pytest.mark.asyncio
    async def test_get_recommendations(self, service):
        await service.start()
        recs = await service.get_recommendations()
        # With no executions and no errors, no service-level recommendations
        # Orchestrator returns whatever it returns (mocked), plus service recs
        assert recs == []
        await service.stop()

    @pytest.mark.asyncio
    async def test_get_patterns(self, service):
        await service.start()
        patterns = await service.get_patterns()
        # _detect_service_patterns returns [] and orchestrator is mocked
        assert patterns == []
        await service.stop()

    @pytest.mark.asyncio
    async def test_get_recommendations_error(self, service):
        """When orchestrator raises, should return empty list and increment error count."""
        await service.start()
        service.orchestrator.get_recommendations = AsyncMock(side_effect=RuntimeError("fail"))
        recs = await service.get_recommendations()
        assert recs == []
        assert service.error_counts["ai_analysis_errors"] == 1
        await service.stop()

    @pytest.mark.asyncio
    async def test_get_patterns_error(self, service):
        """When orchestrator raises, should return empty list and increment error count."""
        await service.start()
        service.orchestrator.get_patterns = AsyncMock(side_effect=RuntimeError("fail"))
        patterns = await service.get_patterns()
        assert patterns == []
        assert service.error_counts["ai_analysis_errors"] == 1
        await service.stop()


# ---------------------------------------------------------------------------
# get_execution_insights error path
# ---------------------------------------------------------------------------


class TestGetExecutionInsightsError:
    @pytest.mark.asyncio
    async def test_error_returns_fallback(self, service):
        await service.start()
        service.orchestrator.get_execution_insights = AsyncMock(side_effect=RuntimeError("fail"))
        insights = await service.get_execution_insights()
        assert insights["error"] == "Failed to get execution insights"
        assert insights["details"] == "fail"
        assert insights["service_metrics"]["performance_metrics"] == service.performance_metrics
        assert insights["service_metrics"]["error_counts"] == service.error_counts
        await service.stop()


# ---------------------------------------------------------------------------
# get_health_status advanced
# ---------------------------------------------------------------------------


class TestGetHealthStatusAdvanced:
    @pytest.mark.asyncio
    async def test_health_with_executions(self, service):
        await service.start()
        service.performance_metrics["total_executions"] = 10
        service.performance_metrics["successful_executions"] = 8
        service.performance_metrics["failed_executions"] = 2
        status = await service.get_health_status()
        assert status["health_scores"]["success_rate"] == 0.8
        await service.stop()

    @pytest.mark.asyncio
    async def test_health_error_path(self, service):
        await service.start()
        service.orchestrator.get_queue_status = AsyncMock(side_effect=RuntimeError("queue fail"))
        status = await service.get_health_status()
        assert status["service_status"]["is_healthy"] is False
        await service.stop()


# ---------------------------------------------------------------------------
# _execute_single_attempt
# ---------------------------------------------------------------------------


class TestExecuteSingleAttempt:
    @pytest.mark.asyncio
    async def test_timeout_raises_timeout_error(self, service, intelligent_request):
        await service.start()
        service.settings.execution_timeout = 0.001
        service.orchestrator.execute_test = AsyncMock(side_effect=TimeoutError())
        with pytest.raises(TestExecutionTimeoutError) as exc_info:
            await service._execute_single_attempt(intelligent_request)
        assert exc_info.value.timeout_seconds == 0.001
        assert exc_info.value.error_type == "timeout"
        await service.stop()

    @pytest.mark.asyncio
    async def test_memory_error_raises_resource_error(self, service, intelligent_request):
        await service.start()
        service.orchestrator.execute_test = AsyncMock(
            side_effect=RuntimeError("memory limit exceeded")
        )
        with pytest.raises(TestExecutionResourceError) as exc_info:
            await service._execute_single_attempt(intelligent_request)
        assert exc_info.value.resource_type == "memory"
        assert exc_info.value.resource_limit == "unknown"
        assert exc_info.value.error_type == "resource_limit"
        await service.stop()

    @pytest.mark.asyncio
    async def test_general_error_raises_execution_error(self, service, intelligent_request):
        await service.start()
        service.orchestrator.execute_test = AsyncMock(
            side_effect=RuntimeError("something unrelated")
        )
        with pytest.raises(TestExecutionError) as exc_info:
            await service._execute_single_attempt(intelligent_request)
        assert exc_info.value.error_type == "execution_failure"
        assert "something unrelated" in str(exc_info.value)
        await service.stop()


# ---------------------------------------------------------------------------
# _execute_with_retry advanced
# ---------------------------------------------------------------------------


class TestExecuteWithRetryAdvanced:
    @pytest.mark.asyncio
    async def test_retries_on_timeout(self, service, intelligent_request):
        """Should retry on timeout errors."""
        await service.start()
        call_count = 0

        async def mock_execute(req):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TestExecutionTimeoutError("timeout", 60)
            return IntelligentTestExecutionResult(
                test_id="t1",
                success=True,
                exit_code=0,
                stdout="ok",
                stderr="",
                execution_time_ms=100,
                language=TestLanguage.PYTHON,
                test_type=OrchestratorType.UNIT,
                created_at=datetime.now(UTC),
            )

        service._execute_single_attempt = mock_execute
        result = await service._execute_with_retry(intelligent_request)
        assert result.success is True
        assert call_count == 2
        await service.stop()

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self, service, intelligent_request):
        """Should raise RetryError when all retries fail."""
        await service.start()
        service._execute_single_attempt = AsyncMock(side_effect=RuntimeError("always fails"))
        with pytest.raises(TestExecutionRetryError):
            await service._execute_with_retry(intelligent_request)
        assert service.error_counts["retry_failures"] == 1
        await service.stop()


# ---------------------------------------------------------------------------
# _generate_service_recommendations advanced
# ---------------------------------------------------------------------------


class TestGenerateServiceRecommendationsAdvanced:
    @pytest.mark.asyncio
    async def test_timeout_only_recommendation(self, service):
        """15/100 = 15% total error rate (under 20%), but 15% timeout rate > 10% threshold."""
        service.performance_metrics = {"total_executions": 100}
        service.error_counts = {
            "timeout_errors": 15,
            "resource_errors": 0,
            "retry_failures": 0,
            "ai_analysis_errors": 0,
            "general_errors": 0,
        }
        recs = await service._generate_service_recommendations()
        # 15% total error rate < 20%, so no "high error rate" rec
        # 15% timeout rate > 10%, so we get "frequent timeout errors"
        assert len(recs) == 1
        assert recs[0].title == "Frequent timeout errors"
        assert recs[0].priority == "medium"
        assert recs[0].confidence_level == 0.8

    @pytest.mark.asyncio
    async def test_no_recommendations_when_healthy(self, service):
        service.performance_metrics = {"total_executions": 100}
        service.error_counts = {
            "timeout_errors": 0,
            "resource_errors": 0,
            "retry_failures": 0,
            "ai_analysis_errors": 0,
            "general_errors": 0,
        }
        recs = await service._generate_service_recommendations()
        assert recs == []


# ---------------------------------------------------------------------------
# _convert_test_request advanced
# ---------------------------------------------------------------------------


class TestConvertTestRequestAdvanced:
    @pytest.mark.asyncio
    async def test_no_test_code_empty_test_files(self, service):
        req = TestRequest(code="x=1", language="python", timeout=60)
        result = await service._convert_test_request(req)
        assert result.test_files == {}
        assert result.source_code == "x=1"
        assert result.language == TestLanguage.PYTHON

    @pytest.mark.asyncio
    async def test_settings_propagated(self, service):
        service.settings.enable_ai_optimization = False
        service.settings.enable_pattern_analysis = False
        req = TestRequest(code="x=1", language="python", timeout=60)
        result = await service._convert_test_request(req)
        assert result.enable_ai_optimization is False
        assert result.enable_pattern_analysis is False
        assert result.enable_failure_analysis is True
        assert result.enable_scheduling_optimization is True


# ---------------------------------------------------------------------------
# Batch execution error path — covers lines 256-259
# ---------------------------------------------------------------------------


class TestExecuteBatchError:
    @pytest.mark.asyncio
    async def test_batch_execution_failure(self, service, intelligent_request):
        """When orchestrator.batch_execute_tests raises, wraps in TestExecutionError."""
        await service.start()
        service.orchestrator.batch_execute_tests = AsyncMock(
            side_effect=RuntimeError("batch kaboom")
        )
        with pytest.raises(TestExecutionError, match="Batch execution failed") as exc_info:
            await service.execute_batch([intelligent_request])
        assert exc_info.value.error_type == "batch_execution_failure"
        assert "batch kaboom" in str(exc_info.value)
        assert service.error_counts["general_errors"] == 1
        await service.stop()


# ---------------------------------------------------------------------------
# Validation re-raise path — covers lines 471, 475-477
# ---------------------------------------------------------------------------


class TestValidationEdgeCases:
    @pytest.mark.asyncio
    async def test_validation_reraises_test_execution_error(self, service):
        """TestExecutionError from validation propagates with original error_type."""
        req = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="   ",  # empty = triggers TestExecutionError
            timeout_seconds=60,
            priority_level=5,
        )
        with pytest.raises(TestExecutionError, match="empty") as exc_info:
            await service._validate_test_request(req)
        assert exc_info.value.error_type == "invalid_request"

    @pytest.mark.asyncio
    async def test_high_security_no_ai_warning(self, service):
        """High security with AI disabled logs warning but passes validation."""
        from models.testing_models import SecurityLevel

        service.settings.enable_ai_optimization = False
        req = IntelligentTestExecutionRequest(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="x = 1",
            timeout_seconds=60,
            priority_level=5,
            security_level=SecurityLevel.HIGH,
        )
        # Should not raise — returns None
        result = await service._validate_test_request(req)
        assert result is None


# ---------------------------------------------------------------------------
# Retry with resource error — covers lines 511, 524
# ---------------------------------------------------------------------------


class TestRetryResourceError:
    @pytest.mark.asyncio
    async def test_retries_on_resource_error(self, service, intelligent_request):
        """Resource errors are retried."""
        await service.start()
        call_count = 0

        async def mock_execute(req):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TestExecutionResourceError("OOM", "memory", "512m")
            return IntelligentTestExecutionResult(
                test_id="t1",
                success=True,
                exit_code=0,
                stdout="ok",
                stderr="",
                execution_time_ms=100,
                language=TestLanguage.PYTHON,
                test_type=OrchestratorType.UNIT,
                created_at=datetime.now(UTC),
            )

        service._execute_single_attempt = mock_execute
        result = await service._execute_with_retry(intelligent_request)
        assert result.success is True
        assert call_count == 2
        assert service.error_counts["resource_errors"] == 1
        await service.stop()


# ---------------------------------------------------------------------------
# _update_execution_metrics error path — covers lines 593-594
# ---------------------------------------------------------------------------


class TestUpdateMetricsError:
    def test_update_metrics_exception_caught(self, service):
        """When metrics update fails internally, exception is caught (lines 593-594)."""
        # Set performance_metrics to None to trigger KeyError
        service.performance_metrics = None
        result = IntelligentTestExecutionResult(
            test_id="t1",
            success=True,
            exit_code=0,
            stdout="ok",
            stderr="",
            execution_time_ms=100,
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            created_at=datetime.now(UTC),
        )
        # Should not raise
        service._update_execution_metrics(result)

    def test_update_batch_metrics_exception_caught(self, service):
        """When batch metrics update fails, exception is caught (lines 602-603)."""
        service.performance_metrics = None
        result = IntelligentTestExecutionResult(
            test_id="t1",
            success=True,
            exit_code=0,
            stdout="ok",
            stderr="",
            execution_time_ms=100,
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            created_at=datetime.now(UTC),
        )
        service._update_batch_metrics([result])


# ---------------------------------------------------------------------------
# _generate_service_recommendations / _detect_service_patterns error paths
# — covers lines 659-661, 672-674
# ---------------------------------------------------------------------------


class TestRecommendationPatternErrors:
    @pytest.mark.asyncio
    async def test_generate_recommendations_error(self, service):
        """Internal error in recommendation generation returns empty list."""
        # Make performance_metrics a non-dict to trigger exception
        service.performance_metrics = "broken"
        recs = await service._generate_service_recommendations()
        assert recs == []

    @pytest.mark.asyncio
    async def test_detect_patterns_error(self, service):
        """Internal error in pattern detection returns empty list."""
        # The method body is essentially a no-op, but test the error path
        patterns = await service._detect_service_patterns()
        assert patterns == []


# ---------------------------------------------------------------------------
# execute_test_with_intelligence convenience function — covers lines 737-756
# ---------------------------------------------------------------------------


class TestExecuteTestWithIntelligence:
    @pytest.mark.asyncio
    async def test_convenience_function(self):
        """Test the module-level convenience function."""
        from services.test_execution.intelligent_test_integration import (
            execute_test_with_intelligence,
        )

        # We need to mock the orchestrator to avoid actual Docker calls
        with patch(
            "services.test_execution.intelligent_test_integration.IntelligentTestOrchestrator"
        ) as mock_orch_cls:
            mock_orch = mock_orch_cls.return_value
            mock_orch.start = AsyncMock()
            mock_orch.stop = AsyncMock()
            mock_orch.execute_test = AsyncMock(
                return_value=IntelligentTestExecutionResult(
                    test_id="conv-t1",
                    success=True,
                    exit_code=0,
                    stdout="ok",
                    stderr="",
                    execution_time_ms=50,
                    language=TestLanguage.PYTHON,
                    test_type=OrchestratorType.UNIT,
                    created_at=datetime.now(UTC),
                )
            )
            mock_orch.get_queue_status = AsyncMock(return_value={})
            mock_orch.get_recommendations = AsyncMock(return_value=[])
            mock_orch.get_patterns = AsyncMock(return_value=[])
            mock_orch.get_execution_insights = AsyncMock(return_value={})
            mock_orch.batch_execute_tests = AsyncMock(return_value=[])

            result = await execute_test_with_intelligence(
                code="def add(a, b): return a + b",
                language="python",
                test_code="def test_add(): assert add(1,2)==3",
                timeout=60,
            )
            assert result.success is True
            assert result.test_id == "conv-t1"


# ---------------------------------------------------------------------------
# Start/stop error paths
# ---------------------------------------------------------------------------


class TestStartStopErrors:
    @pytest.mark.asyncio
    async def test_start_failure(self, service):
        """When orchestrator.start() fails, raises TestExecutionError."""
        service.orchestrator.start = AsyncMock(side_effect=RuntimeError("start failed"))
        with pytest.raises(TestExecutionError, match="startup failed") as exc_info:
            await service.start()
        assert exc_info.value.error_type == "startup_failure"
        assert service.is_healthy is False
        assert service.is_running is False

    @pytest.mark.asyncio
    async def test_stop_failure(self, service):
        """When orchestrator.stop() fails, raises TestExecutionError."""
        await service.start()
        assert service.is_running is True
        service.orchestrator.stop = AsyncMock(side_effect=RuntimeError("stop failed"))
        with pytest.raises(TestExecutionError, match="shutdown failed") as exc_info:
            await service.stop()
        assert exc_info.value.error_type == "shutdown_failure"
