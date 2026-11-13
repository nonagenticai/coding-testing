"""Tests for low-coverage modules: container_scoped_testing, execution_coordination,
analysis_and_insights, report_formatting, ai_analysis, test_pattern_recognition,
service_lifecycle, ai_failure_analysis_service."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Helpers ──────────────────────────────────────────────────────────────────


def _mock_user(*, is_admin: bool = False, tenant_id: str = "t1") -> MagicMock:
    user = MagicMock()
    user.preferred_username = "tester"
    user.is_admin.return_value = is_admin
    user.effective_tenant_id = tenant_id
    return user


def _mock_exec_result(
    success: bool = True,
    execution_time_ms: float = 100.0,
    **kwargs: Any,
) -> MagicMock:
    m = MagicMock()
    m.success = success
    m.execution_time_ms = execution_time_ms
    m.optimization_applied = kwargs.get("optimization_applied", False)
    m.pattern_matches = kwargs.get("pattern_matches", [])
    m.recommendations = kwargs.get("recommendations", [])
    m.stderr = kwargs.get("stderr", "")
    m.error_type = kwargs.get("error_type", "unknown")
    m.error_message = kwargs.get("error_message", "No message available")
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# analysis_and_insights  (28 % → target ~90 %)
# ═══════════════════════════════════════════════════════════════════════════════


class TestAnalyzeTestPerformance:
    """Tests for the analyze_test_performance helper using real result models."""

    def _make_result(self, **overrides):
        """Build a real IntelligentTestExecutionResult."""
        from datetime import UTC, datetime

        from services.test_execution.intelligent_test_orchestrator import (
            IntelligentTestExecutionResult,
        )
        from services.test_execution.test_orchestrator import OrchestratorType, TestLanguage

        defaults = dict(
            test_id="perf-1",
            success=True,
            exit_code=0,
            stdout="ok",
            stderr="",
            execution_time_ms=100.0,
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            created_at=datetime.now(UTC),
        )
        defaults.update(overrides)
        return IntelligentTestExecutionResult(**defaults)

    async def test_empty_results(self):
        from routers.intelligent_testing_modules.analysis_and_insights import (
            analyze_test_performance,
        )

        result = await analyze_test_performance([])
        assert result["total_tests"] == 0

    async def test_with_real_results_computes_stats(self):
        """Test that real models flow through the full analysis pipeline."""
        from routers.intelligent_testing_modules.analysis_and_insights import (
            analyze_test_performance,
        )

        results = [
            self._make_result(test_id="p1", success=True, execution_time_ms=100.0),
            self._make_result(test_id="p2", success=False, exit_code=1, execution_time_ms=300.0),
            self._make_result(test_id="p3", success=True, execution_time_ms=200.0),
        ]
        # No patching -- let the real bottleneck/recommendation logic run
        result = await analyze_test_performance(results)

        summary = result["performance_summary"]
        assert summary["total_tests"] == 3
        assert summary["success_rate"] == pytest.approx(2.0 / 3.0 * 100)
        assert summary["average_execution_time"] == pytest.approx(200.0)
        assert summary["min_execution_time"] == 100.0
        assert summary["max_execution_time"] == 300.0
        assert summary["total_execution_time"] == pytest.approx(600.0)
        assert "analysis_timestamp" in result
        # Bottlenecks/recommendations are lists (real logic ran)
        assert isinstance(result["bottlenecks"], list)
        assert isinstance(result["recommendations"], list)

    async def test_low_success_rate_generates_reliability_recommendation(self):
        """Test that <95% success rate triggers a reliability recommendation."""
        from routers.intelligent_testing_modules.analysis_and_insights import (
            analyze_test_performance,
        )

        # 1 pass, 9 fails = 10% success rate
        results = [
            self._make_result(
                test_id=f"t{i}",
                success=(i == 0),
                exit_code=0 if i == 0 else 1,
                execution_time_ms=50.0,
            )
            for i in range(10)
        ]
        result = await analyze_test_performance(results)
        rec_types = [r["type"] for r in result["recommendations"]]
        assert "reliability" in rec_types


class TestAnalyzeFailurePatterns:
    """Tests for the analyze_failure_patterns helper using real result models."""

    def _make_result(self, **overrides):
        from datetime import UTC, datetime

        from services.test_execution.intelligent_test_orchestrator import (
            IntelligentTestExecutionResult,
        )
        from services.test_execution.test_orchestrator import OrchestratorType, TestLanguage

        defaults = dict(
            test_id="fail-1",
            success=False,
            exit_code=1,
            stdout="",
            stderr="assertion failed",
            execution_time_ms=50.0,
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            created_at=datetime.now(UTC),
        )
        defaults.update(overrides)
        return IntelligentTestExecutionResult(**defaults)

    async def test_no_failures(self):
        from routers.intelligent_testing_modules.analysis_and_insights import (
            analyze_failure_patterns,
        )

        results = [self._make_result(success=True, exit_code=0)]
        result = await analyze_failure_patterns(results)
        assert result["total_failures"] == 0

    async def test_with_failures_real_analysis(self):
        """Test full failure analysis pipeline without mocking internals."""
        from routers.intelligent_testing_modules.analysis_and_insights import (
            analyze_failure_patterns,
        )

        results = [
            self._make_result(
                test_id="f1",
                success=False,
                error_type="timeout",
                error_message="timed out after 30s",
            ),
            self._make_result(
                test_id="f2",
                success=False,
                error_type="timeout",
                error_message="timed out after 60s",
            ),
            self._make_result(test_id="f3", success=True, exit_code=0),
        ]
        # No patching -- let the real identification/recommendation logic run
        result = await analyze_failure_patterns(results)

        assert result["total_failures"] == 2
        assert result["failure_rate"] == pytest.approx(2.0 / 3.0 * 100)
        # timeout appeared twice, should be a common error
        assert len(result["common_errors"]) >= 1
        timeout_error = next(
            (e for e in result["common_errors"] if e["error_type"] == "timeout"), None
        )
        assert timeout_error is not None
        assert timeout_error["occurrences"] == 2
        # Real pattern/recommendation logic ran
        assert isinstance(result["failure_patterns"], list)
        assert isinstance(result["failure_recommendations"], list)

    async def test_language_specific_failure_pattern_detected(self):
        """Test that >30% failures in one language generates a pattern."""
        from routers.intelligent_testing_modules.analysis_and_insights import (
            analyze_failure_patterns,
        )
        from services.test_execution.test_orchestrator import TestLanguage

        # All failures are python -- should trigger language_specific pattern
        results = [
            self._make_result(
                test_id=f"f{i}", success=False, language=TestLanguage.PYTHON, error_type=f"err{i}"
            )
            for i in range(5)
        ]
        result = await analyze_failure_patterns(results)
        pattern_types = [p["type"] for p in result["failure_patterns"]]
        assert "language_specific" in pattern_types


class TestGenerateTestTrends:
    """Tests for the generate_test_trends helper."""

    async def test_success(self):
        from routers.intelligent_testing_modules.analysis_and_insights import (
            generate_test_trends,
        )

        mock_service = AsyncMock()
        mock_service.get_execution_insights = AsyncMock(return_value={"total_executions": 10})
        with patch(
            "routers.intelligent_testing_modules.analysis_and_insights._generate_trend_recommendations",
            new_callable=AsyncMock,
            return_value=["rec1"],
        ):
            result = await generate_test_trends(mock_service, 48)
        assert result["time_period_hours"] == 48
        assert result["execution_trends"]["total_executions"] == 10

    async def test_exception(self):
        from routers.intelligent_testing_modules.analysis_and_insights import (
            generate_test_trends,
        )

        mock_service = AsyncMock()
        mock_service.get_execution_insights = AsyncMock(side_effect=RuntimeError("x"))
        result = await generate_test_trends(mock_service)
        assert "error" in result


# ═══════════════════════════════════════════════════════════════════════════════
# execution_coordination  (28 % → target ~90 %)
# ═══════════════════════════════════════════════════════════════════════════════


class TestGenerateBatchSummary:
    def _make_result(self, **overrides):
        """Build a real IntelligentTestExecutionResult for batch summary tests."""
        from datetime import UTC, datetime

        from services.test_execution.intelligent_test_orchestrator import (
            IntelligentTestExecutionResult,
        )
        from services.test_execution.test_orchestrator import OrchestratorType, TestLanguage

        defaults = dict(
            test_id="batch-1",
            success=True,
            exit_code=0,
            stdout="ok",
            stderr="",
            execution_time_ms=100.0,
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            created_at=datetime.now(UTC),
        )
        defaults.update(overrides)
        return IntelligentTestExecutionResult(**defaults)

    async def test_empty(self):
        from routers.intelligent_testing_modules.execution_coordination import (
            _generate_batch_summary,
        )

        summary = await _generate_batch_summary([])
        assert summary["total_tests"] == 0

    async def test_with_real_results(self):
        """Test batch summary with real model objects computes correct stats."""
        from routers.intelligent_testing_modules.execution_coordination import (
            _generate_batch_summary,
        )

        results = [
            self._make_result(test_id="b1", success=True, execution_time_ms=100.0),
            self._make_result(test_id="b2", success=False, exit_code=1, execution_time_ms=200.0),
        ]
        summary = await _generate_batch_summary(results)
        assert summary["total_tests"] == 2
        assert summary["successful_tests"] == 1
        assert summary["failed_tests"] == 1
        assert summary["total_execution_time"] == pytest.approx(300.0)
        assert summary["average_execution_time"] == pytest.approx(150.0)


class TestOptimizeExecutionOrder:
    def _make_request(self, **overrides):
        from services.test_execution.intelligent_test_orchestrator import (
            IntelligentTestExecutionRequest,
        )
        from services.test_execution.test_orchestrator import OrchestratorType, TestLanguage

        defaults = dict(
            language=TestLanguage.PYTHON,
            test_type=OrchestratorType.UNIT,
            source_code="x=1",
            timeout_seconds=60,
            priority_level=5,
        )
        defaults.update(overrides)
        return IntelligentTestExecutionRequest(**defaults)

    async def test_sort_order_by_priority(self):
        """Test that higher priority requests come first."""
        from routers.intelligent_testing_modules.execution_coordination import (
            optimize_execution_order,
        )

        req_high = self._make_request(priority_level=10, expected_execution_time_ms=200)
        req_low = self._make_request(priority_level=1, expected_execution_time_ms=100)

        result = await optimize_execution_order([req_low, req_high])
        assert result[0].priority_level == 10
        assert result[1].priority_level == 1

    async def test_single_request(self):
        """Test optimization with single request returns it."""
        from routers.intelligent_testing_modules.execution_coordination import (
            optimize_execution_order,
        )

        req = self._make_request(priority_level=5)
        result = await optimize_execution_order([req])
        assert len(result) == 1
        assert result[0].priority_level == 5


class TestManageConcurrentExecution:
    async def test_success(self):
        from routers.intelligent_testing_modules.execution_coordination import (
            manage_concurrent_execution,
        )
        from services.test_execution.intelligent_test_orchestrator import (
            IntelligentTestExecutionResult,
        )

        mock_service = AsyncMock()
        mock_service.settings = MagicMock()
        mock_service.settings.max_concurrent_tests = 2
        exec_result = MagicMock(spec=IntelligentTestExecutionResult)
        mock_service.execute_test = AsyncMock(return_value=exec_result)
        req = MagicMock()
        results = await manage_concurrent_execution(mock_service, [req, req], max_concurrent=2)
        assert len(results) == 2

    async def test_with_exception(self):
        from routers.intelligent_testing_modules.execution_coordination import (
            manage_concurrent_execution,
        )

        mock_service = AsyncMock()
        mock_service.settings = MagicMock()
        mock_service.settings.max_concurrent_tests = 2

        async def side_effect(req):
            raise RuntimeError("fail")

        mock_service.execute_test = side_effect
        req = MagicMock()
        results = await manage_concurrent_execution(mock_service, [req], max_concurrent=1)
        assert results == []

    async def test_default_max_concurrent(self):
        from routers.intelligent_testing_modules.execution_coordination import (
            manage_concurrent_execution,
        )
        from services.test_execution.intelligent_test_orchestrator import (
            IntelligentTestExecutionResult,
        )

        mock_service = AsyncMock()
        mock_service.settings = MagicMock()
        mock_service.settings.max_concurrent_tests = 3
        exec_result = MagicMock(spec=IntelligentTestExecutionResult)
        mock_service.execute_test = AsyncMock(return_value=exec_result)
        req = MagicMock()
        results = await manage_concurrent_execution(mock_service, [req])
        assert len(results) == 1


class TestHandleExecutionRetry:
    async def test_success_first_try(self):
        from routers.intelligent_testing_modules.execution_coordination import (
            handle_execution_retry,
        )

        mock_service = AsyncMock()
        exec_result = _mock_exec_result(success=True)
        mock_service.execute_test = AsyncMock(return_value=exec_result)
        req = MagicMock()
        result = await handle_execution_retry(mock_service, req, max_retries=1)
        assert result.success

    async def test_all_retries_fail(self):
        from routers.intelligent_testing_modules.execution_coordination import (
            handle_execution_retry,
        )
        from services.test_execution.intelligent_test_integration import (
            TestExecutionRetryError,
        )

        mock_service = AsyncMock()
        mock_service.execute_test = AsyncMock(side_effect=RuntimeError("fail"))
        req = MagicMock()
        with pytest.raises(TestExecutionRetryError):
            await handle_execution_retry(mock_service, req, max_retries=0)

    async def test_retry_on_failed_result(self):
        from routers.intelligent_testing_modules.execution_coordination import (
            handle_execution_retry,
        )
        from services.test_execution.intelligent_test_integration import (
            TestExecutionRetryError,
        )

        mock_service = AsyncMock()
        exec_result = _mock_exec_result(success=False, stderr="test failed")
        mock_service.execute_test = AsyncMock(return_value=exec_result)
        req = MagicMock()
        with pytest.raises(TestExecutionRetryError):
            await handle_execution_retry(mock_service, req, max_retries=0)


# ═══════════════════════════════════════════════════════════════════════════════
# service_lifecycle helpers  (65 % → target ~90 %)
# ═══════════════════════════════════════════════════════════════════════════════


class TestBackupSettings:
    async def test_with_model_dump(self):
        from routers.intelligent_testing_modules.service_lifecycle import backup_settings

        mock_service = MagicMock()
        mock_service.settings = MagicMock()
        mock_service.settings.model_dump.return_value = {"timeout": 30}
        mock_service.orchestrator = MagicMock()
        mock_service.orchestrator.max_concurrent_tests = 5
        mock_service.orchestrator.ai_analysis_threshold = 0.5
        mock_service.orchestrator.pattern_analysis_interval = 60

        result = await backup_settings(mock_service)
        assert "timestamp" in result
        assert result["settings"] == {"timeout": 30}
        assert result["orchestrator_settings"]["max_concurrent_tests"] == 5

    async def test_without_model_dump(self):
        from routers.intelligent_testing_modules.service_lifecycle import backup_settings

        mock_service = MagicMock()
        mock_service.settings = "plain_string"
        mock_service.orchestrator = MagicMock(spec=[])  # no attrs

        result = await backup_settings(mock_service)
        assert result["settings"] == "plain_string"

    async def test_exception(self):
        from routers.intelligent_testing_modules.service_lifecycle import backup_settings

        mock_service = MagicMock()
        mock_service.settings = MagicMock()
        mock_service.settings.model_dump.side_effect = RuntimeError("boom")

        result = await backup_settings(mock_service)
        assert "error" in result


class TestRestoreSettings:
    async def test_restore_dict(self):
        from routers.intelligent_testing_modules.service_lifecycle import restore_settings

        mock_service = MagicMock()
        mock_service.orchestrator = MagicMock()

        with patch(
            "routers.intelligent_testing_modules.service_lifecycle.TestExecutionSettings"
        ) as mock_settings_cls:
            mock_settings_cls.return_value = MagicMock()
            result = await restore_settings(
                mock_service,
                {
                    "settings": {"timeout": 30},
                    "orchestrator_settings": {"max_concurrent_tests": 5},
                },
            )
        assert result is True

    async def test_restore_non_dict_settings(self):
        from routers.intelligent_testing_modules.service_lifecycle import restore_settings

        mock_service = MagicMock()
        result = await restore_settings(mock_service, {"settings": "not_a_dict"})
        assert result is False

    async def test_restore_exception(self):
        from routers.intelligent_testing_modules.service_lifecycle import restore_settings

        mock_service = MagicMock()
        with patch(
            "routers.intelligent_testing_modules.service_lifecycle.TestExecutionSettings",
            side_effect=RuntimeError("x"),
        ):
            result = await restore_settings(mock_service, {"settings": {"a": 1}})
        assert result is False


class TestGetConfigurationStatus:
    async def test_with_orchestrator(self):
        from routers.intelligent_testing_modules.service_lifecycle import (
            get_configuration_status,
        )

        mock_service = MagicMock()
        mock_service.is_running = True
        mock_service.orchestrator = MagicMock()
        mock_service.orchestrator.max_concurrent_tests = 5
        mock_service.orchestrator.ai_analysis_threshold = 0.8
        mock_service.orchestrator.pattern_analysis_interval = 120

        result = await get_configuration_status(mock_service)
        assert result["service_running"] is True
        assert result["orchestrator_status"]["available"] is True
        assert result["configuration_health"] == "healthy"

    async def test_without_orchestrator(self):
        from routers.intelligent_testing_modules.service_lifecycle import (
            get_configuration_status,
        )

        mock_service = MagicMock(spec=["is_running"])
        mock_service.is_running = False

        result = await get_configuration_status(mock_service)
        assert result["configuration_health"] == "degraded"
        assert result["orchestrator_status"]["available"] is False

    async def test_exception(self):
        from routers.intelligent_testing_modules.service_lifecycle import (
            get_configuration_status,
        )

        mock_service = MagicMock()
        type(mock_service).is_running = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("x"))
        )

        result = await get_configuration_status(mock_service)
        assert result["configuration_health"] == "critical"


# ═══════════════════════════════════════════════════════════════════════════════
# ai_failure_analysis_service  (63 % → target ~95 %)
# ═══════════════════════════════════════════════════════════════════════════════


class TestAIFailureAnalysisService:
    def test_init_defaults(self):
        from services.ai_failure_analysis_service import AIFailureAnalysisService

        svc = AIFailureAnalysisService()
        assert svc.config == {}
        assert svc.max_concurrent_analyses == 3
        assert svc.analysis_timeout == 60.0

    def test_init_custom(self):
        from services.ai_failure_analysis_service import AIFailureAnalysisService

        svc = AIFailureAnalysisService(
            config={"key": "val"},
            max_concurrent_analyses=10,
            analysis_timeout=120.0,
        )
        assert svc.config == {"key": "val"}
        assert svc.max_concurrent_analyses == 10

    async def test_analyze_failure(self):
        from services.ai_failure_analysis_service import AIFailureAnalysisService

        svc = AIFailureAnalysisService()
        req = MagicMock()
        req.request_id = "req1"
        resp = await svc.analyze_failure(req)
        assert resp.request_id == "req1"
        assert resp.success is False

    async def test_analyze_failures_batch(self):
        from services.ai_failure_analysis_service import AIFailureAnalysisService

        svc = AIFailureAnalysisService()
        req = MagicMock()
        req.request_id = "req1"
        results = await svc.analyze_failures_batch([req, req])
        assert len(results) == 2

    async def test_analyze_test_failure_alias(self):
        from services.ai_failure_analysis_service import AIFailureAnalysisService

        svc = AIFailureAnalysisService()
        req = MagicMock()
        req.request_id = "req2"
        resp = await svc.analyze_test_failure(req)
        assert resp.request_id == "req2"

    def test_get_health_status(self):
        from services.ai_failure_analysis_service import AIFailureAnalysisService

        svc = AIFailureAnalysisService()
        status = svc.get_health_status()
        assert status["status"] == "stub"

    async def test_cleanup(self):
        from services.ai_failure_analysis_service import AIFailureAnalysisService

        svc = AIFailureAnalysisService()
        svc._cache["key"] = "val"
        await svc.cleanup()
        assert svc._cache == {}

    def test_clear_cache(self):
        from services.ai_failure_analysis_service import AIFailureAnalysisService

        svc = AIFailureAnalysisService()
        svc._cache["key"] = "val"
        svc.clear_cache()
        assert svc._cache == {}

    def test_get_singleton(self):
        # Reset singleton
        import services.ai_failure_analysis_service as mod
        from services.ai_failure_analysis_service import (
            AIFailureAnalysisService,
            get_ai_failure_analysis_service,
        )

        mod._ai_failure_analysis_service = None

        svc = get_ai_failure_analysis_service()
        assert isinstance(svc, AIFailureAnalysisService)
        svc2 = get_ai_failure_analysis_service()
        assert svc is svc2

        # Cleanup
        mod._ai_failure_analysis_service = None


# ═══════════════════════════════════════════════════════════════════════════════
# Shared response models serialization (79 % → target 100 %)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSharedResponseSerializers:
    def test_execution_insights_response_serialize_none(self):
        from routers.intelligent_testing_modules.shared import ExecutionInsightsResponse

        resp = ExecutionInsightsResponse(
            status="success",
            message="ok",
            insights={"a": 1},
        )
        # Serialize generated_at
        s = resp.serialize_datetime(resp.generated_at)
        assert isinstance(s, str)
        # Serialize None
        assert resp.serialize_datetime(None) is None

    def test_recommendations_response_serialize_none(self):
        from routers.intelligent_testing_modules.shared import RecommendationsResponse

        resp = RecommendationsResponse(
            status="success",
            message="ok",
            recommendations=[],
        )
        assert resp.serialize_datetime(None) is None

    def test_patterns_response_serialize_none(self):
        from routers.intelligent_testing_modules.shared import PatternsResponse

        resp = PatternsResponse(
            status="success",
            message="ok",
            patterns=[],
        )
        assert resp.serialize_datetime(None) is None

    def test_health_status_response_serialize_none(self):
        from routers.intelligent_testing_modules.shared import HealthStatusResponse

        resp = HealthStatusResponse(
            status="success",
            message="ok",
            health_status={"ok": True},
        )
        assert resp.serialize_datetime(None) is None

    def test_service_settings_response_serialize_none(self):
        from routers.intelligent_testing_modules.shared import ServiceSettingsResponse
        from services.test_execution.intelligent_test_integration import TestExecutionSettings

        resp = ServiceSettingsResponse(
            status="success",
            message="ok",
            settings=TestExecutionSettings(),
        )
        assert resp.serialize_datetime(None) is None
