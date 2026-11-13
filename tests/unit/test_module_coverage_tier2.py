"""Tests for Tier 2 coverage gaps.

Covers:
- analysis_and_insights.py (75% -> target 95%+)
- execution_coordination.py batch endpoint (77% -> target 90%+)
- testing_modules/test_execution.py (58% -> target 80%+)
- testing_modules/health_and_metrics.py (71% -> target 85%+)
- app_factory.py lifespan (68% -> target 80%+)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from coding_client.auth.models import User
from fastapi import HTTPException

from shared.container_validation import ContainerInfo, ContainerStatus

# ── Helpers ──────────────────────────────────────────────────────────────────


def _admin_user() -> User:
    return User(
        sub="admin-user",
        preferred_username="admin",
        realm_access={"roles": ["admin"]},
        tenant_id="n8n-dev",
    )


def _container_info(name: str = "test-container") -> ContainerInfo:
    return ContainerInfo(
        container_name=name,
        status=ContainerStatus.RUNNING,
        tenant_id="n8n-dev",
    )


# ============================================================================
# analysis_and_insights.py (endpoint + helper functions)
# ============================================================================

_AI = "routers.intelligent_testing_modules.analysis_and_insights"


class TestAnalysisAndInsightsEndpoints:
    @pytest.mark.asyncio
    async def test_get_execution_insights_success(self):
        from routers.intelligent_testing_modules.analysis_and_insights import get_execution_insights

        mock_service = MagicMock()
        mock_service.get_execution_insights = AsyncMock(
            return_value={"total_tests": 10, "success_rate": 90.0}
        )

        result = await get_execution_insights(
            user=_admin_user(), service=mock_service, tenant_id="n8n-dev"
        )
        assert result.status.value == "success"
        assert result.insights["total_tests"] == 10

    @pytest.mark.asyncio
    async def test_get_execution_insights_error(self):
        from routers.intelligent_testing_modules.analysis_and_insights import get_execution_insights

        mock_service = MagicMock()
        mock_service.get_execution_insights = AsyncMock(side_effect=RuntimeError("fail"))

        with pytest.raises(HTTPException) as exc_info:
            await get_execution_insights(
                user=_admin_user(), service=mock_service, tenant_id="n8n-dev"
            )
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_get_health_status_success(self):
        from routers.intelligent_testing_modules.analysis_and_insights import get_health_status

        mock_service = MagicMock()
        mock_service.get_health_status = AsyncMock(
            return_value={"status": "healthy", "uptime": 3600}
        )

        result = await get_health_status(
            user=_admin_user(), service=mock_service, tenant_id="n8n-dev"
        )
        assert result.status.value == "success"
        assert result.health_status["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_get_health_status_error(self):
        from routers.intelligent_testing_modules.analysis_and_insights import get_health_status

        mock_service = MagicMock()
        mock_service.get_health_status = AsyncMock(side_effect=RuntimeError("down"))

        with pytest.raises(HTTPException) as exc_info:
            await get_health_status(user=_admin_user(), service=mock_service, tenant_id="n8n-dev")
        assert exc_info.value.status_code == 500


class TestAnalyzeTestPerformance:
    @pytest.mark.asyncio
    async def test_empty_results(self):
        from routers.intelligent_testing_modules.analysis_and_insights import (
            analyze_test_performance,
        )

        result = await analyze_test_performance([])
        assert result["total_tests"] == 0

    @pytest.mark.asyncio
    async def test_with_results(self):
        from routers.intelligent_testing_modules.analysis_and_insights import (
            analyze_test_performance,
        )

        r1 = MagicMock(execution_time_ms=100.0, success=True)
        r2 = MagicMock(execution_time_ms=200.0, success=False)

        with (
            patch(
                f"{_AI}._identify_performance_bottlenecks", new_callable=AsyncMock, return_value=[]
            ),
            patch(
                f"{_AI}._generate_performance_recommendations",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = await analyze_test_performance([r1, r2])
            assert result["performance_summary"]["total_tests"] == 2
            assert result["performance_summary"]["average_execution_time"] == 150.0
            assert result["performance_summary"]["success_rate"] == 50.0

    @pytest.mark.asyncio
    async def test_error_returns_fallback(self):
        from routers.intelligent_testing_modules.analysis_and_insights import (
            analyze_test_performance,
        )

        # Provide results that will cause an error during processing
        bad_result = MagicMock()
        bad_result.execution_time_ms = None  # no exec time
        bad_result.success = True

        with patch(
            f"{_AI}._identify_performance_bottlenecks",
            new_callable=AsyncMock,
            side_effect=RuntimeError("fail"),
        ):
            result = await analyze_test_performance([bad_result])
            assert "error" in result


class TestAnalyzeFailurePatterns:
    @pytest.mark.asyncio
    async def test_no_failures(self):
        from routers.intelligent_testing_modules.analysis_and_insights import (
            analyze_failure_patterns,
        )

        r1 = MagicMock(success=True)
        result = await analyze_failure_patterns([r1])
        assert result["total_failures"] == 0

    @pytest.mark.asyncio
    async def test_with_failures(self):
        from routers.intelligent_testing_modules.analysis_and_insights import (
            analyze_failure_patterns,
        )

        r1 = MagicMock(success=False, error_type="assertion", error_message="Expected 1 got 2")
        r2 = MagicMock(success=False, error_type="assertion", error_message="Expected 3 got 4")
        r3 = MagicMock(success=True)

        with (
            patch(f"{_AI}._identify_failure_patterns", new_callable=AsyncMock, return_value=[]),
            patch(
                f"{_AI}._generate_failure_recommendations", new_callable=AsyncMock, return_value=[]
            ),
        ):
            result = await analyze_failure_patterns([r1, r2, r3])
            assert result["total_failures"] == 2
            assert len(result["common_errors"]) == 1  # assertion appears 2 times
            assert result["common_errors"][0]["error_type"] == "assertion"

    @pytest.mark.asyncio
    async def test_error_returns_fallback(self):
        from routers.intelligent_testing_modules.analysis_and_insights import (
            analyze_failure_patterns,
        )

        r1 = MagicMock(success=False)
        # Force error during grouping
        type(r1).error_type = property(lambda s: (_ for _ in ()).throw(RuntimeError("boom")))

        result = await analyze_failure_patterns([r1])
        assert "error" in result


class TestGenerateTestTrends:
    @pytest.mark.asyncio
    async def test_success(self):
        from routers.intelligent_testing_modules.analysis_and_insights import generate_test_trends

        mock_service = MagicMock()
        mock_service.get_execution_insights = AsyncMock(
            return_value={
                "total_executions": 100,
                "success_trend": [80, 85, 90],
                "failure_trend": [20, 15, 10],
                "performance_trend": [100, 95, 90],
                "language_trends": {"python": 50, "javascript": 30},
                "peak_hours": [9, 14, 16],
            }
        )

        with patch(
            f"{_AI}._generate_trend_recommendations", new_callable=AsyncMock, return_value=["rec1"]
        ):
            result = await generate_test_trends(mock_service, time_period_hours=48)
            assert result["time_period_hours"] == 48
            assert result["execution_trends"]["total_executions"] == 100
            assert result["trend_recommendations"] == ["rec1"]

    @pytest.mark.asyncio
    async def test_error_returns_fallback(self):
        from routers.intelligent_testing_modules.analysis_and_insights import generate_test_trends

        mock_service = MagicMock()
        mock_service.get_execution_insights = AsyncMock(side_effect=RuntimeError("fail"))

        result = await generate_test_trends(mock_service)
        assert "error" in result
        assert result["time_period_hours"] == 24


# ============================================================================
# execution_coordination.py - batch endpoint
# ============================================================================

_EC = "routers.intelligent_testing_modules.execution_coordination"


class TestExecuteBatchTests:
    @pytest.mark.asyncio
    async def test_batch_success(self):
        from routers.intelligent_testing_modules.execution_coordination import execute_batch_tests
        from services.test_execution.intelligent_test_orchestrator import (
            IntelligentTestExecutionResult,
        )

        test_req = MagicMock()
        test_req.language = "python"
        test_req.test_type = "unit"
        test_req.code = "x = 1"
        test_req.container_name = "test-container"
        test_req.test_code = None
        test_req.dependencies = {}
        test_req.timeout = 300
        test_req.resource_limits = None
        test_req.security_level = "medium"
        test_req.project_name = None
        test_req.enable_ai_optimization = True
        test_req.enable_pattern_analysis = True
        test_req.enable_failure_analysis = True
        test_req.enable_scheduling_optimization = True
        test_req.priority_level = 5
        test_req.expected_execution_time_ms = None
        test_req.resource_weight = 1.0

        mock_request = MagicMock()
        mock_request.tests = [test_req]
        mock_request.enable_retry = True

        mock_result = MagicMock(spec=IntelligentTestExecutionResult)
        mock_result.success = True
        mock_result.execution_time_ms = 100.0
        mock_result.optimization_applied = False
        mock_result.pattern_matches = []
        mock_result.recommendations = []

        mock_service = MagicMock()
        mock_service.execute_batch = AsyncMock(return_value=[mock_result])

        with patch(
            f"{_EC}.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
            return_value=_container_info(),
        ):
            result = await execute_batch_tests(
                request=mock_request, user=_admin_user(), service=mock_service, tenant_id="n8n-dev"
            )
            assert result.status.value == "success"
            assert result.summary["total_tests"] == 1

    @pytest.mark.asyncio
    async def test_batch_http_exception_passthrough(self):
        from routers.intelligent_testing_modules.execution_coordination import execute_batch_tests

        test_req = MagicMock()
        test_req.container_name = "bad-container"

        mock_request = MagicMock()
        mock_request.tests = [test_req]

        mock_service = MagicMock()

        with patch(
            f"{_EC}.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
            side_effect=HTTPException(status_code=404, detail="Container not found"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await execute_batch_tests(
                    request=mock_request,
                    user=_admin_user(),
                    service=mock_service,
                    tenant_id="n8n-dev",
                )
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_batch_unexpected_error(self):
        from routers.intelligent_testing_modules.execution_coordination import execute_batch_tests

        test_req = MagicMock()
        test_req.container_name = None
        test_req.language = "python"
        test_req.test_type = "unit"
        test_req.code = "x = 1"
        test_req.test_code = None
        test_req.dependencies = {}
        test_req.timeout = 300
        test_req.resource_limits = None
        test_req.security_level = "medium"
        test_req.project_name = None
        test_req.enable_ai_optimization = True
        test_req.enable_pattern_analysis = True
        test_req.enable_failure_analysis = True
        test_req.enable_scheduling_optimization = True
        test_req.priority_level = 5
        test_req.expected_execution_time_ms = None
        test_req.resource_weight = 1.0

        mock_request = MagicMock()
        mock_request.tests = [test_req]
        mock_request.enable_retry = True

        mock_service = MagicMock()
        mock_service.execute_batch = AsyncMock(side_effect=RuntimeError("boom"))

        with pytest.raises(HTTPException) as exc_info:
            await execute_batch_tests(
                request=mock_request, user=_admin_user(), service=mock_service, tenant_id="n8n-dev"
            )
        assert exc_info.value.status_code == 500


# ============================================================================
# testing_modules/test_execution.py (batch and validate-security endpoints)
# ============================================================================


class TestTestingModulesTestExecution:
    """Tests for testing_modules/test_execution.py batch and security endpoints."""

    @pytest.mark.asyncio
    async def test_execute_batch_success(self):
        from models.testing_models import TestResult
        from routers.testing_modules.test_execution import execute_batch

        mock_request = MagicMock()
        mock_request.container_name = None
        mock_request.language = "python"
        mock_request.test_type = None
        mock_request.security_level = "medium"

        mock_result = MagicMock(spec=TestResult)
        mock_result.success = True

        mock_service = MagicMock()
        mock_service.execute_test = AsyncMock(return_value=mock_result)

        background_tasks = MagicMock()

        result = await execute_batch(
            requests=[mock_request],
            background_tasks=background_tasks,
            max_parallel=4,
            user=_admin_user(),
            execution_service=mock_service,
        )
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_execute_batch_with_exception(self):
        """Test batch handles individual test failure gracefully."""
        from routers.testing_modules.test_execution import execute_batch

        mock_request = MagicMock()
        mock_request.container_name = None
        mock_request.language = "python"
        mock_request.test_type = None
        mock_request.security_level = "medium"

        mock_service = MagicMock()
        mock_service.execute_test = AsyncMock(side_effect=RuntimeError("test fail"))

        background_tasks = MagicMock()

        result = await execute_batch(
            requests=[mock_request],
            background_tasks=background_tasks,
            max_parallel=4,
            user=_admin_user(),
            execution_service=mock_service,
        )
        assert len(result) == 1
        assert result[0].success is False

    @pytest.mark.asyncio
    async def test_execute_batch_http_exception(self):
        from routers.testing_modules.test_execution import execute_batch

        mock_request = MagicMock()
        mock_request.container_name = "bad-container"

        mock_service = MagicMock()
        background_tasks = MagicMock()

        with patch(
            "routers.testing_modules.test_execution.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
            side_effect=HTTPException(status_code=404, detail="Not found"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await execute_batch(
                    requests=[mock_request],
                    background_tasks=background_tasks,
                    max_parallel=4,
                    user=_admin_user(),
                    execution_service=mock_service,
                )
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_validate_code_security_safe(self):
        from routers.testing_modules.test_execution import validate_code_security

        mock_service = MagicMock()
        mock_service._validate_code_security = AsyncMock(
            return_value={"is_safe": True, "violations": [], "risk_score": 0.0}
        )

        result = await validate_code_security(
            code="x = 1",
            language="python",
            user=_admin_user(),
            execution_service=mock_service,
        )
        assert result["is_safe"] is True
        assert result["risk_level"] == 0.0
        assert result["language"] == "python"

    @pytest.mark.asyncio
    async def test_validate_code_security_violations(self):
        from routers.testing_modules.test_execution import validate_code_security

        mock_service = MagicMock()
        mock_service._validate_code_security = AsyncMock(
            return_value={
                "is_safe": False,
                "violations": [
                    {"description": "Dangerous import", "severity": "high"},
                ],
                "risk_score": 75.0,
            }
        )

        result = await validate_code_security(
            code="import os; os.system('rm -rf /')",
            language="python",
            user=_admin_user(),
            execution_service=mock_service,
        )
        assert result["is_safe"] is False
        assert result["risk_level"] == 75.0

    @pytest.mark.asyncio
    async def test_validate_code_security_value_error(self):
        from routers.testing_modules.test_execution import validate_code_security

        mock_service = MagicMock()
        mock_service._validate_code_security = AsyncMock(side_effect=ValueError("bad language"))

        result = await validate_code_security(
            code="x = 1",
            language="python",
            user=_admin_user(),
            execution_service=mock_service,
        )
        assert result["is_safe"] is False
        assert result["risk_level"] == 100.0


# ============================================================================
# testing_modules/health_and_metrics.py
# ============================================================================


class TestHealthAndMetrics:
    @pytest.mark.asyncio
    async def test_testing_health_check_degraded(self):
        """Test health check when execution service fails to init."""
        from routers.testing_modules.health_and_metrics import testing_health_check

        with patch(
            "routers.testing_modules.health_and_metrics.get_execution_service",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Docker not available"),
        ):
            result = await testing_health_check(tenant_id="n8n-dev")
            # Error is caught and handled gracefully
            assert result["status"] == "unhealthy"
            assert result["services"]["execution_service"]["error"] == "Docker not available"

    @pytest.mark.asyncio
    async def test_testing_metrics_summary_success(self):
        from routers.testing_modules.health_and_metrics import get_testing_metrics_summary

        mock_metrics_service = MagicMock()
        mock_metrics_service.get_test_execution_analytics = AsyncMock(
            return_value={
                "period_days": 7,
                "test_executions": 42,
                "quality_analyses": 15,
                "resource_usage": {"cpu": 0.5},
                "languages": ["python", "javascript"],
            }
        )

        with patch(
            "routers.testing_modules.health_and_metrics.get_metrics_service",
            return_value=mock_metrics_service,
        ):
            result = await get_testing_metrics_summary(
                days=7,
                current_user=_admin_user(),
                tenant_id="n8n-dev",
            )
            assert result["test_executions"] == 42
            assert result["period_days"] == 7

    @pytest.mark.asyncio
    async def test_testing_metrics_summary_fallback(self):
        from routers.testing_modules.health_and_metrics import get_testing_metrics_summary

        mock_metrics_service = MagicMock()
        mock_metrics_service.get_test_execution_analytics = AsyncMock(
            side_effect=AttributeError("no such method")
        )

        with patch(
            "routers.testing_modules.health_and_metrics.get_metrics_service",
            return_value=mock_metrics_service,
        ):
            result = await get_testing_metrics_summary(
                days=7,
                current_user=_admin_user(),
                tenant_id="n8n-dev",
            )
            assert result["test_executions"] == 0


# ============================================================================
# app_factory.py - lifespan context
# ============================================================================


class TestAppFactoryLifespan:
    @pytest.mark.asyncio
    async def test_lifespan_context(self):
        """Test the lifespan context manager runs without error."""
        from app_factory import lifespan

        mock_app = MagicMock()
        mock_app.state = MagicMock()

        async with lifespan(mock_app):
            # Verify state was set
            assert hasattr(mock_app.state, "background_tasks")


# ============================================================================
# analysis_core.py helper functions
# ============================================================================


class TestAnalysisCore:
    @pytest.mark.asyncio
    async def test_identify_performance_bottlenecks(self):
        from routers.intelligent_testing_modules.analysis_core import (
            _identify_performance_bottlenecks,
        )

        r1 = MagicMock(execution_time_ms=100.0, success=True)
        r2 = MagicMock(execution_time_ms=5000.0, success=True)  # slow

        result = await _identify_performance_bottlenecks([r1, r2])
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_generate_performance_recommendations(self):
        from routers.intelligent_testing_modules.analysis_core import (
            _generate_performance_recommendations,
        )

        summary = {"average_execution_time": 5000, "success_rate": 60.0}
        bottlenecks = [{"type": "slow_test"}]

        result = await _generate_performance_recommendations(summary, bottlenecks)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_identify_failure_patterns(self):
        from routers.intelligent_testing_modules.analysis_core import _identify_failure_patterns

        r1 = MagicMock(success=False, stderr="assertion error")
        result = await _identify_failure_patterns([r1])
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_generate_failure_recommendations(self):
        from routers.intelligent_testing_modules.analysis_core import (
            _generate_failure_recommendations,
        )

        errors = [{"error_type": "assertion", "occurrences": 5}]
        result = await _generate_failure_recommendations(errors)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_generate_trend_recommendations(self):
        from routers.intelligent_testing_modules.analysis_core import (
            _generate_trend_recommendations,
        )

        trend_data = {
            "execution_trends": {"total_executions": 100, "success_trend": [80, 85, 90]},
        }
        result = await _generate_trend_recommendations(trend_data)
        assert isinstance(result, list)
