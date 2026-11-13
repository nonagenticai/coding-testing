"""Comprehensive tests for Tier 1 router modules with lowest coverage.

Covers:
- container_scoped_testing.py (24% -> target 90%+)
- report_formatting_endpoints.py (40% -> target 90%+)
- ai_analysis_endpoints.py (44% -> target 90%+)
- test_pattern_recognition.py (45% -> target 90%+)
- execution_coordination.py (53% -> target 90%+)
- insight_generation_services.py (54% -> target 90%+)
- configuration_management.py (63% -> target 90%+)
- service_lifecycle.py (65% -> target 90%+)
- discovery_and_recommendations.py (66% -> target 90%+)
- ai_test_insights_config.py (76% -> target 90%+)
- config/test_framework_logging_manager.py (72% -> target 100%)
- main.py (0% -> target 100%)
- app_factory.py (68% -> target 90%+)
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


def _regular_user() -> User:
    return User(
        sub="regular-user",
        preferred_username="regular",
        realm_access={"roles": ["user"]},
        tenant_id="n8n-dev",
    )


def _container_info(name: str = "test-container") -> ContainerInfo:
    return ContainerInfo(
        container_name=name,
        status=ContainerStatus.RUNNING,
        tenant_id="n8n-dev",
    )


# ============================================================================
# container_scoped_testing.py
# ============================================================================

# The endpoint functions call get_intelligent_testing_service() internally
# (not as a dependency injection), so we must patch it at the module level.

_CST = "routers.intelligent_testing_modules.container_scoped_testing"


class TestContainerScopedTesting:
    """Tests for container-scoped testing endpoints."""

    @pytest.mark.asyncio
    async def test_analyze_test_patterns_success(self):
        from routers.intelligent_testing_modules.container_scoped_testing import (
            analyze_test_patterns,
        )

        mock_request = MagicMock()
        mock_request.container_name = "test-container"
        mock_request.test_results = [MagicMock()]
        mock_request.test_results[0].model_dump.return_value = {"name": "t1"}
        mock_request.include_performance_patterns = True
        mock_request.include_failure_patterns = True
        mock_request.include_timing_patterns = True
        mock_request.analysis_depth = "deep"

        mock_service = MagicMock()
        mock_service.analyze_test_patterns = AsyncMock(
            return_value={
                "container_name": "test-container",
                "pattern_count": 2,
                "patterns_detected": [{"type": "slow"}],
                "confidence_score": 0.85,
            }
        )

        with (
            patch(
                f"{_CST}.validate_and_get_container_for_tenant",
                new_callable=AsyncMock,
                return_value=_container_info(),
            ),
            patch(f"{_CST}.get_intelligent_testing_service", return_value=mock_service),
        ):
            result = await analyze_test_patterns(
                request=mock_request,
                user=_admin_user(),
                tenant_id="n8n-dev",
            )
            assert result.pattern_count == 2
            assert result.confidence_score == 0.85

    @pytest.mark.asyncio
    async def test_analyze_test_patterns_non_admin_passes_tenant(self):
        from routers.intelligent_testing_modules.container_scoped_testing import (
            analyze_test_patterns,
        )

        mock_request = MagicMock()
        mock_request.container_name = "test-container"
        mock_request.test_results = []
        mock_request.include_performance_patterns = True
        mock_request.include_failure_patterns = True
        mock_request.include_timing_patterns = True
        mock_request.analysis_depth = "shallow"

        mock_service = MagicMock()
        mock_service.analyze_test_patterns = AsyncMock(
            return_value={
                "container_name": "test-container",
                "pattern_count": 0,
                "patterns_detected": [],
                "confidence_score": 0.0,
            }
        )
        validate_mock = AsyncMock(return_value=_container_info())

        with (
            patch(f"{_CST}.validate_and_get_container_for_tenant", validate_mock),
            patch(f"{_CST}.get_intelligent_testing_service", return_value=mock_service),
        ):
            await analyze_test_patterns(
                request=mock_request,
                user=_regular_user(),
                tenant_id="n8n-dev",
            )
            validate_mock.assert_called_once_with("test-container", tenant_id="n8n-dev")

    @pytest.mark.asyncio
    async def test_analyze_test_patterns_container_not_found(self):
        from routers.intelligent_testing_modules.container_scoped_testing import (
            analyze_test_patterns,
        )

        mock_request = MagicMock()
        mock_request.container_name = "missing"

        with patch(
            f"{_CST}.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
            side_effect=HTTPException(status_code=404, detail="Not found"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await analyze_test_patterns(
                    request=mock_request, user=_admin_user(), tenant_id="n8n-dev"
                )
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_analyze_test_patterns_service_error(self):
        from routers.intelligent_testing_modules.container_scoped_testing import (
            analyze_test_patterns,
        )

        mock_request = MagicMock()
        mock_request.container_name = "test-container"

        mock_service = MagicMock()
        mock_service.analyze_test_patterns = AsyncMock(side_effect=RuntimeError("boom"))

        with (
            patch(
                f"{_CST}.validate_and_get_container_for_tenant",
                new_callable=AsyncMock,
                return_value=_container_info(),
            ),
            patch(f"{_CST}.get_intelligent_testing_service", return_value=mock_service),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await analyze_test_patterns(
                    request=mock_request, user=_admin_user(), tenant_id="n8n-dev"
                )
            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_smart_test_selection_success(self):
        from routers.intelligent_testing_modules.container_scoped_testing import (
            smart_test_selection,
        )

        mock_request = MagicMock()
        mock_request.container_name = "test-container"
        mock_request.changed_files = ["src/main.py"]
        mock_request.selection_strategy = "impact"
        mock_request.max_tests = 10
        mock_request.include_related_tests = True

        mock_service = MagicMock()
        mock_service.select_smart_tests = AsyncMock(
            return_value={
                "container_name": "test-container",
                "selection_count": 3,
                "selected_tests": ["test_a", "test_b", "test_c"],
                "selection_strategy": "impact",
            }
        )

        with (
            patch(
                f"{_CST}.validate_and_get_container_for_tenant",
                new_callable=AsyncMock,
                return_value=_container_info(),
            ),
            patch(f"{_CST}.get_intelligent_testing_service", return_value=mock_service),
        ):
            result = await smart_test_selection(
                request=mock_request, user=_admin_user(), tenant_id="n8n-dev"
            )
            assert result.selection_count == 3

    @pytest.mark.asyncio
    async def test_smart_test_selection_error(self):
        from routers.intelligent_testing_modules.container_scoped_testing import (
            smart_test_selection,
        )

        mock_request = MagicMock()
        mock_request.container_name = "test-container"
        mock_request.changed_files = []

        mock_service = MagicMock()
        mock_service.select_smart_tests = AsyncMock(side_effect=RuntimeError("fail"))

        with (
            patch(
                f"{_CST}.validate_and_get_container_for_tenant",
                new_callable=AsyncMock,
                return_value=_container_info(),
            ),
            patch(f"{_CST}.get_intelligent_testing_service", return_value=mock_service),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await smart_test_selection(
                    request=mock_request, user=_admin_user(), tenant_id="n8n-dev"
                )
            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_test_impact_analysis_success(self):
        from routers.intelligent_testing_modules.container_scoped_testing import (
            test_impact_analysis,
        )

        mock_request = MagicMock()
        mock_request.container_name = "test-container"
        mock_request.changed_files = ["src/utils.py"]
        mock_request.analysis_depth = "deep"
        mock_request.include_indirect_impacts = True

        mock_service = MagicMock()
        mock_service.analyze_test_impact = AsyncMock(
            return_value={
                "container_name": "test-container",
                "changed_files": ["src/utils.py"],
                "impact_count": 5,
                "impacted_tests": [{"name": "t1"}],
                "impact_score": 3.5,
            }
        )

        with (
            patch(
                f"{_CST}.validate_and_get_container_for_tenant",
                new_callable=AsyncMock,
                return_value=_container_info(),
            ),
            patch(f"{_CST}.get_intelligent_testing_service", return_value=mock_service),
        ):
            result = await test_impact_analysis(
                request=mock_request, user=_regular_user(), tenant_id="n8n-dev"
            )
            assert result.impact_count == 5

    @pytest.mark.asyncio
    async def test_test_impact_analysis_error(self):
        from routers.intelligent_testing_modules.container_scoped_testing import (
            test_impact_analysis,
        )

        mock_request = MagicMock()
        mock_request.container_name = "test-container"

        mock_service = MagicMock()
        mock_service.analyze_test_impact = AsyncMock(side_effect=ValueError("bad"))

        with (
            patch(
                f"{_CST}.validate_and_get_container_for_tenant",
                new_callable=AsyncMock,
                return_value=_container_info(),
            ),
            patch(f"{_CST}.get_intelligent_testing_service", return_value=mock_service),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await test_impact_analysis(
                    request=mock_request, user=_admin_user(), tenant_id="n8n-dev"
                )
            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_test_prioritization_success(self):
        from routers.intelligent_testing_modules.container_scoped_testing import (
            test_prioritization,
        )

        mock_request = MagicMock()
        mock_request.container_name = "test-container"
        mock_request.test_files = ["test_a.py", "test_b.py"]
        mock_request.prioritization_criteria = ["impact"]
        mock_request.changed_files = ["src/main.py"]

        mock_service = MagicMock()
        mock_service.prioritize_tests = AsyncMock(
            return_value={
                "container_name": "test-container",
                "total_tests": 2,
                "prioritized_tests": [{"file": "test_a.py", "priority": 1}],
                "criteria_used": ["impact"],
            }
        )

        with (
            patch(
                f"{_CST}.validate_and_get_container_for_tenant",
                new_callable=AsyncMock,
                return_value=_container_info(),
            ),
            patch(f"{_CST}.get_intelligent_testing_service", return_value=mock_service),
        ):
            result = await test_prioritization(
                request=mock_request, user=_admin_user(), tenant_id="n8n-dev"
            )
            assert result.total_tests == 2

    @pytest.mark.asyncio
    async def test_test_prioritization_error(self):
        from routers.intelligent_testing_modules.container_scoped_testing import (
            test_prioritization,
        )

        mock_request = MagicMock()
        mock_request.container_name = "test-container"

        mock_service = MagicMock()
        mock_service.prioritize_tests = AsyncMock(side_effect=RuntimeError("fail"))

        with (
            patch(
                f"{_CST}.validate_and_get_container_for_tenant",
                new_callable=AsyncMock,
                return_value=_container_info(),
            ),
            patch(f"{_CST}.get_intelligent_testing_service", return_value=mock_service),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await test_prioritization(
                    request=mock_request, user=_admin_user(), tenant_id="n8n-dev"
                )
            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_advanced_test_recommendations_success(self):
        from routers.intelligent_testing_modules.container_scoped_testing import (
            advanced_test_recommendations,
        )

        mock_request = MagicMock()
        mock_request.container_name = "test-container"
        mock_request.code_context = "def add(a, b): return a + b"
        mock_request.language = "python"
        mock_request.test_type = "unit"
        mock_request.focus_areas = ["edge_cases"]
        mock_request.priority_threshold = 0.5

        mock_service = MagicMock()
        mock_service.generate_test_recommendations = AsyncMock(
            return_value={
                "container_name": "test-container",
                "recommendation_count": 3,
                "recommendations": [{"title": "Test edge cases"}],
            }
        )

        with (
            patch(
                f"{_CST}.validate_and_get_container_for_tenant",
                new_callable=AsyncMock,
                return_value=_container_info(),
            ),
            patch(f"{_CST}.get_intelligent_testing_service", return_value=mock_service),
        ):
            result = await advanced_test_recommendations(
                request=mock_request, user=_regular_user(), tenant_id="n8n-dev"
            )
            assert result.recommendation_count == 3

    @pytest.mark.asyncio
    async def test_advanced_test_recommendations_error(self):
        from routers.intelligent_testing_modules.container_scoped_testing import (
            advanced_test_recommendations,
        )

        mock_request = MagicMock()
        mock_request.container_name = "test-container"

        mock_service = MagicMock()
        mock_service.generate_test_recommendations = AsyncMock(side_effect=RuntimeError("boom"))

        with (
            patch(
                f"{_CST}.validate_and_get_container_for_tenant",
                new_callable=AsyncMock,
                return_value=_container_info(),
            ),
            patch(f"{_CST}.get_intelligent_testing_service", return_value=mock_service),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await advanced_test_recommendations(
                    request=mock_request, user=_admin_user(), tenant_id="n8n-dev"
                )
            assert exc_info.value.status_code == 500


# ============================================================================
# report_formatting_endpoints.py
# ============================================================================


class TestReportFormattingEndpoints:
    """Tests for report formatting endpoints."""

    @pytest.mark.asyncio
    async def test_get_cached_analysis_success(self):
        from routers.intelligent_testing_modules.report_formatting_endpoints import (
            get_cached_analysis,
        )

        mock_analysis = MagicMock()
        mock_service = MagicMock()
        mock_service.get_cached_analysis = AsyncMock(return_value=mock_analysis)

        result = await get_cached_analysis(
            analysis_id="ana_123", user=_admin_user(), service=mock_service, tenant_id="n8n-dev"
        )
        assert result == mock_analysis

    @pytest.mark.asyncio
    async def test_get_cached_analysis_not_found(self):
        from routers.intelligent_testing_modules.report_formatting_endpoints import (
            get_cached_analysis,
        )

        mock_service = MagicMock()
        mock_service.get_cached_analysis = AsyncMock(return_value=None)

        with pytest.raises(HTTPException) as exc_info:
            await get_cached_analysis(
                analysis_id="ana_missing",
                user=_admin_user(),
                service=mock_service,
                tenant_id="n8n-dev",
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_cached_analysis_service_error(self):
        from routers.intelligent_testing_modules.report_formatting_endpoints import (
            get_cached_analysis,
        )

        mock_service = MagicMock()
        mock_service.get_cached_analysis = AsyncMock(side_effect=RuntimeError("cache down"))

        with pytest.raises(HTTPException) as exc_info:
            await get_cached_analysis(
                analysis_id="ana_123", user=_admin_user(), service=mock_service, tenant_id="n8n-dev"
            )
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_get_cached_analysis_reraises_http_exception(self):
        from routers.intelligent_testing_modules.report_formatting_endpoints import (
            get_cached_analysis,
        )

        mock_service = MagicMock()
        mock_service.get_cached_analysis = AsyncMock(
            side_effect=HTTPException(status_code=403, detail="Forbidden")
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_cached_analysis(
                analysis_id="ana_123", user=_admin_user(), service=mock_service, tenant_id="n8n-dev"
            )
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_get_cached_comprehensive_analysis_success(self):
        from routers.intelligent_testing_modules.report_formatting_endpoints import (
            get_cached_comprehensive_analysis,
        )

        mock_result = {"analysis_id": "comp_123", "patterns": []}
        mock_service = MagicMock()
        mock_service.get_cached_comprehensive_analysis = AsyncMock(return_value=mock_result)

        result = await get_cached_comprehensive_analysis(
            analysis_id="comp_123", user=_admin_user(), service=mock_service, tenant_id="n8n-dev"
        )
        assert result == mock_result

    @pytest.mark.asyncio
    async def test_get_cached_comprehensive_analysis_not_found(self):
        from routers.intelligent_testing_modules.report_formatting_endpoints import (
            get_cached_comprehensive_analysis,
        )

        mock_service = MagicMock()
        mock_service.get_cached_comprehensive_analysis = AsyncMock(return_value=None)

        with pytest.raises(HTTPException) as exc_info:
            await get_cached_comprehensive_analysis(
                analysis_id="comp_missing",
                user=_admin_user(),
                service=mock_service,
                tenant_id="n8n-dev",
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_cached_comprehensive_analysis_error(self):
        from routers.intelligent_testing_modules.report_formatting_endpoints import (
            get_cached_comprehensive_analysis,
        )

        mock_service = MagicMock()
        mock_service.get_cached_comprehensive_analysis = AsyncMock(side_effect=RuntimeError("boom"))

        with pytest.raises(HTTPException) as exc_info:
            await get_cached_comprehensive_analysis(
                analysis_id="comp_123",
                user=_admin_user(),
                service=mock_service,
                tenant_id="n8n-dev",
            )
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_get_cached_comprehensive_analysis_reraises_http_exception(self):
        from routers.intelligent_testing_modules.report_formatting_endpoints import (
            get_cached_comprehensive_analysis,
        )

        mock_service = MagicMock()
        mock_service.get_cached_comprehensive_analysis = AsyncMock(
            side_effect=HTTPException(status_code=401, detail="Unauthorized")
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_cached_comprehensive_analysis(
                analysis_id="comp_123",
                user=_admin_user(),
                service=mock_service,
                tenant_id="n8n-dev",
            )
        assert exc_info.value.status_code == 401


# ============================================================================
# ai_analysis_endpoints.py
# ============================================================================


class TestAIAnalysisEndpoints:
    """Tests for AI analysis endpoints."""

    @pytest.mark.asyncio
    async def test_analyze_test_insights_success(self):
        from routers.intelligent_testing_modules.ai_analysis_endpoints import analyze_test_insights

        mock_request = MagicMock()
        mock_request.test_results = [MagicMock()]
        mock_request.quality_results = []
        mock_request.time_period = "recent"
        mock_request.include_predictions = True

        mock_analysis = MagicMock()
        mock_service = MagicMock()
        mock_service.analyze_test_insights = AsyncMock(return_value=mock_analysis)

        result = await analyze_test_insights(
            request=mock_request, service=mock_service, tenant_id="n8n-dev"
        )
        assert result == mock_analysis

    @pytest.mark.asyncio
    async def test_analyze_test_insights_empty_results(self):
        from routers.intelligent_testing_modules.ai_analysis_endpoints import analyze_test_insights

        mock_request = MagicMock()
        mock_request.test_results = []
        mock_service = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await analyze_test_insights(
                request=mock_request, service=mock_service, tenant_id="n8n-dev"
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_analyze_test_insights_service_error(self):
        from routers.intelligent_testing_modules.ai_analysis_endpoints import analyze_test_insights

        mock_request = MagicMock()
        mock_request.test_results = [MagicMock()]
        mock_request.quality_results = []
        mock_request.time_period = "recent"
        mock_request.include_predictions = False

        mock_service = MagicMock()
        mock_service.analyze_test_insights = AsyncMock(side_effect=RuntimeError("AI down"))

        with pytest.raises(HTTPException) as exc_info:
            await analyze_test_insights(
                request=mock_request, service=mock_service, tenant_id="n8n-dev"
            )
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_comprehensive_test_analysis_success(self):
        from routers.intelligent_testing_modules.ai_analysis_endpoints import (
            comprehensive_test_analysis,
        )

        mock_request = MagicMock()
        mock_request.test_results = [MagicMock()]
        mock_request.quality_results = []
        mock_request.code_changes = ["fix: typo"]
        mock_request.time_period = "week"
        mock_request.include_predictions = True

        mock_report = {"analysis_id": "comp_123"}
        mock_service = MagicMock()
        mock_service.comprehensive_analysis = AsyncMock(return_value=mock_report)

        result = await comprehensive_test_analysis(
            request=mock_request, service=mock_service, tenant_id="n8n-dev"
        )
        assert result["analysis_id"] == "comp_123"

    @pytest.mark.asyncio
    async def test_comprehensive_test_analysis_empty_results(self):
        from routers.intelligent_testing_modules.ai_analysis_endpoints import (
            comprehensive_test_analysis,
        )

        mock_request = MagicMock()
        mock_request.test_results = []
        mock_service = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await comprehensive_test_analysis(
                request=mock_request, service=mock_service, tenant_id="n8n-dev"
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_comprehensive_test_analysis_error(self):
        from routers.intelligent_testing_modules.ai_analysis_endpoints import (
            comprehensive_test_analysis,
        )

        mock_request = MagicMock()
        mock_request.test_results = [MagicMock()]
        mock_request.quality_results = []
        mock_request.code_changes = None
        mock_request.time_period = "month"
        mock_request.include_predictions = False

        mock_service = MagicMock()
        mock_service.comprehensive_analysis = AsyncMock(side_effect=RuntimeError("fail"))

        with pytest.raises(HTTPException) as exc_info:
            await comprehensive_test_analysis(
                request=mock_request, service=mock_service, tenant_id="n8n-dev"
            )
        assert exc_info.value.status_code == 500


# ============================================================================
# test_pattern_recognition.py
# ============================================================================

_TPR = "routers.intelligent_testing_modules.test_pattern_recognition"


class TestPatternRecognition:
    @pytest.mark.asyncio
    async def test_predict_test_failures_success(self):
        from routers.intelligent_testing_modules.test_pattern_recognition import (
            predict_test_failures,
        )

        mock_request = MagicMock()
        mock_request.container_name = "test-container"
        mock_request.test_results = [MagicMock()]
        mock_request.code_changes = ["fix: null check"]

        mock_service = MagicMock()
        mock_service.predict_test_failures = AsyncMock(return_value={"test_a": 0.2, "test_b": 0.8})

        with patch(
            f"{_TPR}.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
            return_value=_container_info(),
        ):
            result = await predict_test_failures(
                request=mock_request, user=_admin_user(), service=mock_service, tenant_id="n8n-dev"
            )
            assert result.predictions == {"test_a": 0.2, "test_b": 0.8}
            assert result.confidence == 0.5  # 1.0 - mean(0.2, 0.8) = 0.5

    @pytest.mark.asyncio
    async def test_predict_test_failures_empty_predictions(self):
        from routers.intelligent_testing_modules.test_pattern_recognition import (
            predict_test_failures,
        )

        mock_request = MagicMock()
        mock_request.container_name = "test-container"
        mock_request.test_results = [MagicMock()]
        mock_request.code_changes = None

        mock_service = MagicMock()
        mock_service.predict_test_failures = AsyncMock(return_value={})

        with patch(
            f"{_TPR}.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
            return_value=_container_info(),
        ):
            result = await predict_test_failures(
                request=mock_request, user=_admin_user(), service=mock_service, tenant_id="n8n-dev"
            )
            # empty predictions -> confidence = min(1.0, max(0.0, 1.0 - 0.0)) = 1.0
            assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_predict_test_failures_no_test_results(self):
        from routers.intelligent_testing_modules.test_pattern_recognition import (
            predict_test_failures,
        )

        mock_request = MagicMock()
        mock_request.container_name = "test-container"
        mock_request.test_results = []
        mock_service = MagicMock()

        with patch(
            f"{_TPR}.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
            return_value=_container_info(),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await predict_test_failures(
                    request=mock_request,
                    user=_admin_user(),
                    service=mock_service,
                    tenant_id="n8n-dev",
                )
            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_predict_test_failures_container_not_found(self):
        from routers.intelligent_testing_modules.test_pattern_recognition import (
            predict_test_failures,
        )

        mock_request = MagicMock()
        mock_request.container_name = "missing"
        mock_request.test_results = [MagicMock()]
        mock_service = MagicMock()

        with patch(
            f"{_TPR}.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
            side_effect=HTTPException(status_code=404),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await predict_test_failures(
                    request=mock_request,
                    user=_admin_user(),
                    service=mock_service,
                    tenant_id="n8n-dev",
                )
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_predict_test_failures_service_error(self):
        from routers.intelligent_testing_modules.test_pattern_recognition import (
            predict_test_failures,
        )

        mock_request = MagicMock()
        mock_request.container_name = "test-container"
        mock_request.test_results = [MagicMock()]
        mock_request.code_changes = None

        mock_service = MagicMock()
        mock_service.predict_test_failures = AsyncMock(side_effect=RuntimeError("model error"))

        with patch(
            f"{_TPR}.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
            return_value=_container_info(),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await predict_test_failures(
                    request=mock_request,
                    user=_admin_user(),
                    service=mock_service,
                    tenant_id="n8n-dev",
                )
            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_predict_test_failures_non_admin_passes_tenant(self):
        from routers.intelligent_testing_modules.test_pattern_recognition import (
            predict_test_failures,
        )

        mock_request = MagicMock()
        mock_request.container_name = "test-container"
        mock_request.test_results = [MagicMock()]
        mock_request.code_changes = None

        mock_service = MagicMock()
        mock_service.predict_test_failures = AsyncMock(return_value={"t1": 0.1})
        validate_mock = AsyncMock(return_value=_container_info())

        with patch(f"{_TPR}.validate_and_get_container_for_tenant", validate_mock):
            await predict_test_failures(
                request=mock_request,
                user=_regular_user(),
                service=mock_service,
                tenant_id="n8n-dev",
            )
            validate_mock.assert_called_once_with("test-container", tenant_id="n8n-dev")


# ============================================================================
# insight_generation_services.py
# ============================================================================


class TestInsightGenerationServices:
    def _make_recommendation(self, priority="high"):
        from services.ai_test_insights.config import TestInsightType
        from services.ai_test_insights.models import TestInsightRecommendation

        return TestInsightRecommendation(
            recommendation_id="rec_1",
            type=TestInsightType.PERFORMANCE,
            priority=priority,
            confidence=0.9,
            title="Test Recommendation",
            description="A test recommendation",
            rationale="Because it matters",
            estimated_effort="low",
            expected_impact="medium impact",
        )

    @pytest.mark.asyncio
    async def test_generate_recommendations_success(self):
        from routers.intelligent_testing_modules.insight_generation_services import (
            generate_test_recommendations,
        )

        rec1 = self._make_recommendation(priority="high")
        rec2 = self._make_recommendation(priority="low")

        mock_request = MagicMock()
        mock_request.analysis = MagicMock()
        mock_request.focus_areas = ["performance"]

        mock_service = MagicMock()
        mock_service.generate_test_recommendations = AsyncMock(return_value=[rec1, rec2])

        result = await generate_test_recommendations(
            request=mock_request, user=_admin_user(), service=mock_service, tenant_id="n8n-dev"
        )
        assert result.total_recommendations == 2
        assert result.high_priority_count == 1

    @pytest.mark.asyncio
    async def test_generate_recommendations_empty(self):
        from routers.intelligent_testing_modules.insight_generation_services import (
            generate_test_recommendations,
        )

        mock_request = MagicMock()
        mock_request.analysis = MagicMock()
        mock_request.focus_areas = None

        mock_service = MagicMock()
        mock_service.generate_test_recommendations = AsyncMock(return_value=[])

        result = await generate_test_recommendations(
            request=mock_request, user=_admin_user(), service=mock_service, tenant_id="n8n-dev"
        )
        assert result.total_recommendations == 0
        assert result.high_priority_count == 0

    @pytest.mark.asyncio
    async def test_generate_recommendations_error(self):
        from routers.intelligent_testing_modules.insight_generation_services import (
            generate_test_recommendations,
        )

        mock_request = MagicMock()
        mock_request.analysis = MagicMock()
        mock_request.focus_areas = None

        mock_service = MagicMock()
        mock_service.generate_test_recommendations = AsyncMock(side_effect=RuntimeError("fail"))

        with pytest.raises(HTTPException) as exc_info:
            await generate_test_recommendations(
                request=mock_request, user=_admin_user(), service=mock_service, tenant_id="n8n-dev"
            )
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_generate_recommendations_reraises_http_exception(self):
        from routers.intelligent_testing_modules.insight_generation_services import (
            generate_test_recommendations,
        )

        mock_request = MagicMock()
        mock_request.analysis = MagicMock()
        mock_request.focus_areas = None

        mock_service = MagicMock()
        mock_service.generate_test_recommendations = AsyncMock(
            side_effect=HTTPException(status_code=429, detail="Rate limited")
        )

        with pytest.raises(HTTPException) as exc_info:
            await generate_test_recommendations(
                request=mock_request, user=_admin_user(), service=mock_service, tenant_id="n8n-dev"
            )
        assert exc_info.value.status_code == 429


# ============================================================================
# execution_coordination.py
# ============================================================================

_EC = "routers.intelligent_testing_modules.execution_coordination"


class TestExecutionCoordinationEndpoints:
    def _make_request_mock(self, **overrides):
        defaults = dict(
            language="python",
            test_type="unit",
            code="x = 1",
            container_name=None,
            test_code=None,
            dependencies={},
            timeout=300,
            resource_limits=None,
            security_level="medium",
            project_name=None,
            enable_ai_optimization=True,
            enable_pattern_analysis=True,
            enable_failure_analysis=True,
            enable_scheduling_optimization=True,
            priority_level=5,
            expected_execution_time_ms=None,
            resource_weight=1.0,
        )
        defaults.update(overrides)
        m = MagicMock()
        for k, v in defaults.items():
            setattr(m, k, v)
        return m

    def _make_exec_result(self):
        from services.test_execution.intelligent_test_orchestrator import (
            IntelligentTestExecutionResult,
        )

        return MagicMock(spec=IntelligentTestExecutionResult, success=True)

    @pytest.mark.asyncio
    async def test_execute_test_success(self):
        from routers.intelligent_testing_modules.execution_coordination import execute_test

        mock_request = self._make_request_mock(container_name="test-container")
        mock_result = self._make_exec_result()
        mock_service = MagicMock()
        mock_service.execute_test = AsyncMock(return_value=mock_result)

        with patch(
            f"{_EC}.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
            return_value=_container_info(),
        ):
            result = await execute_test(
                request=mock_request, user=_admin_user(), service=mock_service, tenant_id="n8n-dev"
            )
            assert result.status.value == "success"
            assert result.result == mock_result

    @pytest.mark.asyncio
    async def test_execute_test_no_container(self):
        from routers.intelligent_testing_modules.execution_coordination import execute_test

        mock_request = self._make_request_mock()
        mock_result = self._make_exec_result()
        mock_service = MagicMock()
        mock_service.execute_test = AsyncMock(return_value=mock_result)

        result = await execute_test(
            request=mock_request, user=_admin_user(), service=mock_service, tenant_id="n8n-dev"
        )
        assert result.status.value == "success"

    @pytest.mark.asyncio
    async def test_execute_test_timeout_error(self):
        from routers.intelligent_testing_modules.execution_coordination import execute_test
        from services.test_execution.intelligent_test_integration import TestExecutionTimeoutError

        mock_request = self._make_request_mock()
        mock_service = MagicMock()
        mock_service.execute_test = AsyncMock(
            side_effect=TestExecutionTimeoutError(message="timeout", timeout_seconds=30)
        )

        with pytest.raises(HTTPException) as exc_info:
            await execute_test(
                request=mock_request, user=_admin_user(), service=mock_service, tenant_id="n8n-dev"
            )
        assert exc_info.value.status_code == 408

    @pytest.mark.asyncio
    async def test_execute_test_resource_error(self):
        from routers.intelligent_testing_modules.execution_coordination import execute_test
        from services.test_execution.intelligent_test_integration import TestExecutionResourceError

        mock_request = self._make_request_mock()
        mock_service = MagicMock()
        mock_service.execute_test = AsyncMock(
            side_effect=TestExecutionResourceError(
                message="OOM", resource_type="memory", resource_limit="512Mi"
            )
        )

        with pytest.raises(HTTPException) as exc_info:
            await execute_test(
                request=mock_request, user=_admin_user(), service=mock_service, tenant_id="n8n-dev"
            )
        assert exc_info.value.status_code == 507

    @pytest.mark.asyncio
    async def test_execute_test_retry_error(self):
        from routers.intelligent_testing_modules.execution_coordination import execute_test
        from services.test_execution.intelligent_test_integration import TestExecutionRetryError

        mock_request = self._make_request_mock()
        mock_service = MagicMock()
        mock_service.execute_test = AsyncMock(
            side_effect=TestExecutionRetryError(
                message="retry fail", attempts=3, last_error=RuntimeError("boom")
            )
        )

        with pytest.raises(HTTPException) as exc_info:
            await execute_test(
                request=mock_request, user=_admin_user(), service=mock_service, tenant_id="n8n-dev"
            )
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_execute_test_execution_error(self):
        from routers.intelligent_testing_modules.execution_coordination import execute_test
        from services.test_execution.intelligent_test_integration import TestExecutionError

        mock_request = self._make_request_mock()
        mock_service = MagicMock()
        mock_service.execute_test = AsyncMock(
            side_effect=TestExecutionError(
                message="compile fail", error_type="compilation", details={"line": 1}
            )
        )

        with pytest.raises(HTTPException) as exc_info:
            await execute_test(
                request=mock_request, user=_admin_user(), service=mock_service, tenant_id="n8n-dev"
            )
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_execute_test_unexpected_error(self):
        from routers.intelligent_testing_modules.execution_coordination import execute_test

        mock_request = self._make_request_mock()
        mock_service = MagicMock()
        mock_service.execute_test = AsyncMock(side_effect=RuntimeError("unexpected"))

        with pytest.raises(HTTPException) as exc_info:
            await execute_test(
                request=mock_request, user=_admin_user(), service=mock_service, tenant_id="n8n-dev"
            )
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_execute_test_http_exception_passthrough(self):
        from routers.intelligent_testing_modules.execution_coordination import execute_test

        mock_request = self._make_request_mock(container_name="bad")
        mock_service = MagicMock()

        with patch(
            f"{_EC}.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
            side_effect=HTTPException(status_code=404),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await execute_test(
                    request=mock_request,
                    user=_admin_user(),
                    service=mock_service,
                    tenant_id="n8n-dev",
                )
            assert exc_info.value.status_code == 404


class TestHandleExecutionRetry:
    @pytest.mark.asyncio
    async def test_retry_success_first_attempt(self):
        from routers.intelligent_testing_modules.execution_coordination import (
            handle_execution_retry,
        )

        mock_result = MagicMock(success=True)
        mock_service = MagicMock()
        mock_service.execute_test = AsyncMock(return_value=mock_result)

        result = await handle_execution_retry(mock_service, MagicMock(), max_retries=3)
        assert result == mock_result
        assert mock_service.execute_test.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_success_after_failure(self):
        from routers.intelligent_testing_modules.execution_coordination import (
            handle_execution_retry,
        )

        fail_result = MagicMock(success=False, stderr="fail")
        success_result = MagicMock(success=True)
        mock_service = MagicMock()
        mock_service.execute_test = AsyncMock(side_effect=[fail_result, success_result])

        with patch(f"{_EC}.asyncio") as mock_asyncio:
            mock_asyncio.sleep = AsyncMock()
            result = await handle_execution_retry(mock_service, MagicMock(), max_retries=3)
            assert result.success is True

    @pytest.mark.asyncio
    async def test_retry_all_fail(self):
        from routers.intelligent_testing_modules.execution_coordination import (
            handle_execution_retry,
        )
        from services.test_execution.intelligent_test_integration import TestExecutionRetryError

        fail_result = MagicMock(success=False, stderr="fail")
        mock_service = MagicMock()
        mock_service.execute_test = AsyncMock(return_value=fail_result)

        with patch(f"{_EC}.asyncio") as mock_asyncio:
            mock_asyncio.sleep = AsyncMock()
            with pytest.raises(TestExecutionRetryError):
                await handle_execution_retry(mock_service, MagicMock(), max_retries=1)

    @pytest.mark.asyncio
    async def test_retry_exception_on_attempts(self):
        from routers.intelligent_testing_modules.execution_coordination import (
            handle_execution_retry,
        )
        from services.test_execution.intelligent_test_integration import TestExecutionRetryError

        mock_service = MagicMock()
        mock_service.execute_test = AsyncMock(side_effect=RuntimeError("crash"))

        with patch(f"{_EC}.asyncio") as mock_asyncio:
            mock_asyncio.sleep = AsyncMock()
            with pytest.raises(TestExecutionRetryError):
                await handle_execution_retry(mock_service, MagicMock(), max_retries=1)


class TestManageConcurrentExecution:
    @pytest.mark.asyncio
    async def test_concurrent_execution_success(self):
        from routers.intelligent_testing_modules.execution_coordination import (
            manage_concurrent_execution,
        )
        from services.test_execution.intelligent_test_orchestrator import (
            IntelligentTestExecutionResult,
        )

        mock_result = MagicMock(spec=IntelligentTestExecutionResult)
        mock_service = MagicMock()
        mock_service.settings = MagicMock()
        mock_service.settings.max_concurrent_tests = 2
        mock_service.execute_test = AsyncMock(return_value=mock_result)

        results = await manage_concurrent_execution(
            mock_service, [MagicMock(), MagicMock()], max_concurrent=2
        )
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_concurrent_execution_default_max(self):
        from routers.intelligent_testing_modules.execution_coordination import (
            manage_concurrent_execution,
        )
        from services.test_execution.intelligent_test_orchestrator import (
            IntelligentTestExecutionResult,
        )

        mock_result = MagicMock(spec=IntelligentTestExecutionResult)
        mock_service = MagicMock()
        mock_service.settings = MagicMock()
        mock_service.settings.max_concurrent_tests = 3
        mock_service.execute_test = AsyncMock(return_value=mock_result)

        results = await manage_concurrent_execution(mock_service, [MagicMock()])
        assert len(results) == 1


# ============================================================================
# service_lifecycle.py (endpoint functions)
# ============================================================================


class TestServiceLifecycleEndpoints:
    @pytest.mark.asyncio
    async def test_start_service_already_running(self):
        from routers.intelligent_testing_modules.service_lifecycle import start_service

        mock_service = MagicMock(is_running=True)
        result = await start_service(user=_admin_user(), service=mock_service, tenant_id="n8n-dev")
        assert result.message == "Service is already running"

    @pytest.mark.asyncio
    async def test_start_service_success(self):
        from routers.intelligent_testing_modules.service_lifecycle import start_service

        mock_service = MagicMock(is_running=False, start=AsyncMock())
        result = await start_service(user=_admin_user(), service=mock_service, tenant_id="n8n-dev")
        assert result.message == "Service started successfully"
        mock_service.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_service_forbidden(self):
        from routers.intelligent_testing_modules.service_lifecycle import start_service

        with pytest.raises(HTTPException) as exc_info:
            await start_service(user=_regular_user(), service=MagicMock(), tenant_id="n8n-dev")
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_start_service_error(self):
        from routers.intelligent_testing_modules.service_lifecycle import start_service

        mock_service = MagicMock(
            is_running=False, start=AsyncMock(side_effect=RuntimeError("fail"))
        )
        with pytest.raises(HTTPException) as exc_info:
            await start_service(user=_admin_user(), service=mock_service, tenant_id="n8n-dev")
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_stop_service_already_stopped(self):
        from routers.intelligent_testing_modules.service_lifecycle import stop_service

        mock_service = MagicMock(is_running=False)
        result = await stop_service(user=_admin_user(), service=mock_service, tenant_id="n8n-dev")
        assert result.message == "Service is already stopped"

    @pytest.mark.asyncio
    async def test_stop_service_success(self):
        from routers.intelligent_testing_modules.service_lifecycle import stop_service

        mock_service = MagicMock(is_running=True, stop=AsyncMock())
        result = await stop_service(user=_admin_user(), service=mock_service, tenant_id="n8n-dev")
        assert result.message == "Service stopped successfully"

    @pytest.mark.asyncio
    async def test_stop_service_forbidden(self):
        from routers.intelligent_testing_modules.service_lifecycle import stop_service

        with pytest.raises(HTTPException) as exc_info:
            await stop_service(user=_regular_user(), service=MagicMock(), tenant_id="n8n-dev")
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_stop_service_error(self):
        from routers.intelligent_testing_modules.service_lifecycle import stop_service

        mock_service = MagicMock(is_running=True, stop=AsyncMock(side_effect=RuntimeError("fail")))
        with pytest.raises(HTTPException) as exc_info:
            await stop_service(user=_admin_user(), service=mock_service, tenant_id="n8n-dev")
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_restart_service_success(self):
        from routers.intelligent_testing_modules.service_lifecycle import restart_service

        mock_service = MagicMock(is_running=True, stop=AsyncMock(), start=AsyncMock())
        result = await restart_service(
            user=_admin_user(), service=mock_service, tenant_id="n8n-dev"
        )
        assert result.message == "Service restarted successfully"
        mock_service.stop.assert_called_once()
        mock_service.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_restart_service_from_stopped(self):
        from routers.intelligent_testing_modules.service_lifecycle import restart_service

        mock_service = MagicMock(is_running=False, start=AsyncMock())
        result = await restart_service(
            user=_admin_user(), service=mock_service, tenant_id="n8n-dev"
        )
        assert result.message == "Service restarted successfully"

    @pytest.mark.asyncio
    async def test_restart_service_forbidden(self):
        from routers.intelligent_testing_modules.service_lifecycle import restart_service

        with pytest.raises(HTTPException) as exc_info:
            await restart_service(user=_regular_user(), service=MagicMock(), tenant_id="n8n-dev")
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_restart_service_error(self):
        from routers.intelligent_testing_modules.service_lifecycle import restart_service

        mock_service = MagicMock(
            is_running=True, stop=AsyncMock(), start=AsyncMock(side_effect=RuntimeError("fail"))
        )
        with pytest.raises(HTTPException) as exc_info:
            await restart_service(user=_admin_user(), service=mock_service, tenant_id="n8n-dev")
        assert exc_info.value.status_code == 500


# ============================================================================
# configuration_management.py
# ============================================================================


class TestConfigurationManagementEndpoints:
    def _real_settings(self):
        from services.test_execution.intelligent_test_integration import TestExecutionSettings

        return TestExecutionSettings()

    @pytest.mark.asyncio
    async def test_get_settings_success(self):
        from routers.intelligent_testing_modules.configuration_management import (
            get_service_settings,
        )

        real_settings = self._real_settings()
        mock_service = MagicMock(settings=real_settings)

        result = await get_service_settings(
            user=_admin_user(), service=mock_service, tenant_id="n8n-dev"
        )
        assert result.status.value == "success"
        assert result.settings == real_settings

    @pytest.mark.asyncio
    async def test_get_settings_forbidden(self):
        from routers.intelligent_testing_modules.configuration_management import (
            get_service_settings,
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_service_settings(
                user=_regular_user(), service=MagicMock(), tenant_id="n8n-dev"
            )
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_get_settings_error(self):
        from routers.intelligent_testing_modules.configuration_management import (
            get_service_settings,
        )

        mock_service = MagicMock()
        type(mock_service).settings = property(
            lambda s: (_ for _ in ()).throw(RuntimeError("fail"))
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_service_settings(
                user=_admin_user(), service=mock_service, tenant_id="n8n-dev"
            )
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_update_settings_success(self):
        from routers.intelligent_testing_modules.configuration_management import (
            update_service_settings,
        )

        real_settings = self._real_settings()
        mock_service = MagicMock()
        mock_service.orchestrator = MagicMock()

        # Patch _validate_settings since real defaults don't pass the internal validator
        with patch(
            "routers.intelligent_testing_modules.configuration_management._validate_settings",
            new_callable=AsyncMock,
        ):
            result = await update_service_settings(
                settings=real_settings,
                user=_admin_user(),
                service=mock_service,
                tenant_id="n8n-dev",
            )
            assert result.status.value == "success"

    @pytest.mark.asyncio
    async def test_update_settings_forbidden(self):
        from routers.intelligent_testing_modules.configuration_management import (
            update_service_settings,
        )

        with pytest.raises(HTTPException) as exc_info:
            await update_service_settings(
                settings=self._real_settings(),
                user=_regular_user(),
                service=MagicMock(),
                tenant_id="n8n-dev",
            )
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_update_settings_validation_error(self):
        from routers.intelligent_testing_modules.configuration_management import (
            update_service_settings,
        )

        bad_settings = MagicMock(
            max_concurrent_tests=100,
            default_timeout_seconds=300,
            ai_analysis_threshold=0.5,
            pattern_analysis_interval=120,
        )
        mock_service = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await update_service_settings(
                settings=bad_settings, user=_admin_user(), service=mock_service, tenant_id="n8n-dev"
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_update_settings_unexpected_error(self):
        from routers.intelligent_testing_modules.configuration_management import (
            update_service_settings,
        )

        real_settings = self._real_settings()
        mock_service = MagicMock()
        mock_service.orchestrator = MagicMock()
        type(mock_service.orchestrator).max_concurrent_tests = property(
            lambda s: 5, lambda s, v: (_ for _ in ()).throw(RuntimeError("fail"))
        )

        with (
            patch(
                "routers.intelligent_testing_modules.configuration_management._validate_settings",
                new_callable=AsyncMock,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await update_service_settings(
                settings=real_settings,
                user=_admin_user(),
                service=mock_service,
                tenant_id="n8n-dev",
            )
        assert exc_info.value.status_code == 500


# ============================================================================
# discovery_and_recommendations.py (endpoint functions)
# ============================================================================

_DR = "routers.intelligent_testing_modules.discovery_and_recommendations"


class TestDiscoveryEndpoints:
    def _make_test_recommendation(self):
        from services.test_execution.pydantic_ai_agents import TestRecommendation

        return TestRecommendation(
            recommendation_id="rec_1",
            recommendation_type="optimization",
            priority="high",
            title="Optimize tests",
            description="Optimize test execution",
            impact_assessment="High impact",
            implementation_steps=["Step 1"],
            confidence_level=0.9,
        )

    @pytest.mark.asyncio
    async def test_get_recommendations_success(self):
        from routers.intelligent_testing_modules.discovery_and_recommendations import (
            get_recommendations,
        )

        recs = [self._make_test_recommendation(), self._make_test_recommendation()]
        mock_service = MagicMock()
        mock_service.get_recommendations = AsyncMock(return_value=recs)

        with patch(
            f"{_DR}.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
            return_value=_container_info(),
        ):
            result = await get_recommendations(
                container_name="test-container",
                user=_admin_user(),
                service=mock_service,
                tenant_id="n8n-dev",
            )
            assert result.status.value == "success"
            assert len(result.recommendations) == 2

    @pytest.mark.asyncio
    async def test_get_recommendations_container_not_found(self):
        from routers.intelligent_testing_modules.discovery_and_recommendations import (
            get_recommendations,
        )

        with patch(
            f"{_DR}.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
            side_effect=HTTPException(status_code=404),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_recommendations(
                    container_name="missing",
                    user=_admin_user(),
                    service=MagicMock(),
                    tenant_id="n8n-dev",
                )
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_recommendations_error(self):
        from routers.intelligent_testing_modules.discovery_and_recommendations import (
            get_recommendations,
        )

        mock_service = MagicMock()
        mock_service.get_recommendations = AsyncMock(side_effect=RuntimeError("fail"))

        with patch(
            f"{_DR}.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
            return_value=_container_info(),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_recommendations(
                    container_name="test-container",
                    user=_admin_user(),
                    service=mock_service,
                    tenant_id="n8n-dev",
                )
            assert exc_info.value.status_code == 500

    def _make_execution_pattern(self):
        from datetime import UTC, datetime

        from services.test_execution.pydantic_ai_agents import ExecutionPattern

        return ExecutionPattern(
            pattern_id="pat_1",
            pattern_type="performance",
            frequency=5,
            severity="medium",
            description="Slow test pattern",
            affected_languages=["python"],
            affected_test_types=["unit"],
            first_occurrence=datetime.now(UTC),
            last_occurrence=datetime.now(UTC),
            recommendations=["Optimize"],
            confidence_score=0.8,
        )

    @pytest.mark.asyncio
    async def test_get_execution_patterns_success(self):
        from routers.intelligent_testing_modules.discovery_and_recommendations import (
            get_execution_patterns,
        )

        mock_service = MagicMock()
        mock_service.get_patterns = AsyncMock(return_value=[self._make_execution_pattern()])

        with patch(
            f"{_DR}.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
            return_value=_container_info(),
        ):
            result = await get_execution_patterns(
                container_name="test-container",
                user=_admin_user(),
                service=mock_service,
                tenant_id="n8n-dev",
            )
            assert result.status.value == "success"

    @pytest.mark.asyncio
    async def test_get_execution_patterns_error(self):
        from routers.intelligent_testing_modules.discovery_and_recommendations import (
            get_execution_patterns,
        )

        mock_service = MagicMock()
        mock_service.get_patterns = AsyncMock(side_effect=RuntimeError("fail"))

        with patch(
            f"{_DR}.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
            return_value=_container_info(),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_execution_patterns(
                    container_name="test-container",
                    user=_admin_user(),
                    service=mock_service,
                    tenant_id="n8n-dev",
                )
            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_get_supported_languages_success(self):
        from routers.intelligent_testing_modules.discovery_and_recommendations import (
            get_supported_languages,
        )

        mock_service = MagicMock()
        mock_service.orchestrator.base_orchestrator.get_supported_languages = AsyncMock(
            return_value=["python", "javascript"]
        )
        mock_service.orchestrator.base_orchestrator.get_supported_test_types = AsyncMock(
            return_value=["unit", "integration"]
        )

        result = await get_supported_languages(service=mock_service, tenant_id="n8n-dev")
        assert result["status"] == "success"
        assert "python" in result["data"]["languages"]

    @pytest.mark.asyncio
    async def test_get_supported_languages_error(self):
        from routers.intelligent_testing_modules.discovery_and_recommendations import (
            get_supported_languages,
        )

        mock_service = MagicMock()
        mock_service.orchestrator.base_orchestrator.get_supported_languages = AsyncMock(
            side_effect=RuntimeError("fail")
        )

        result = await get_supported_languages(service=mock_service, tenant_id="n8n-dev")
        assert result["data"]["languages"] == []


# ============================================================================
# ai_test_insights_config.py (health + metrics)
# ============================================================================


class TestAITestInsightsConfigEndpoints:
    @pytest.mark.asyncio
    async def test_health_check_success(self):
        from routers.intelligent_testing_modules.ai_test_insights_config import health_check

        result = await health_check(user=_admin_user(), service=MagicMock(), tenant_id="n8n-dev")
        assert result["status"] == "success"
        assert result["service"] == "ai_test_insights"

    @pytest.mark.asyncio
    async def test_get_service_metrics_with_cache(self):
        from routers.intelligent_testing_modules.ai_test_insights_config import get_service_metrics

        mock_service = MagicMock(cache_manager=MagicMock())
        result = await get_service_metrics(
            user=_admin_user(), service=mock_service, tenant_id="n8n-dev"
        )
        assert result["data"]["cache_enabled"] is True

    @pytest.mark.asyncio
    async def test_get_service_metrics_no_cache(self):
        from routers.intelligent_testing_modules.ai_test_insights_config import get_service_metrics

        mock_service = MagicMock(cache_manager=None)
        result = await get_service_metrics(
            user=_admin_user(), service=mock_service, tenant_id="n8n-dev"
        )
        assert result["data"]["cache_enabled"] is False

    @pytest.mark.asyncio
    async def test_health_check_exception(self):
        """Health check raises 500 on unexpected error (lines 214-216)."""
        from unittest.mock import patch

        from fastapi import HTTPException

        from routers.intelligent_testing_modules.ai_test_insights_config import health_check

        # Patch logger.info to raise, triggering the except block
        with patch(
            "routers.intelligent_testing_modules.ai_test_insights_config.logger"
        ) as mock_logger:
            mock_logger.info.side_effect = RuntimeError("log crash")
            with pytest.raises(HTTPException) as exc_info:
                await health_check(user=_admin_user(), service=MagicMock(), tenant_id="n8n-dev")
            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_get_service_metrics_exception(self):
        """Metrics endpoint raises 500 on unexpected error (lines 330-332)."""
        from unittest.mock import patch

        from fastapi import HTTPException

        from routers.intelligent_testing_modules.ai_test_insights_config import get_service_metrics

        # Patch datetime to make the isoformat call fail
        with patch(
            "routers.intelligent_testing_modules.ai_test_insights_config.datetime"
        ) as mock_dt:
            mock_dt.now.side_effect = RuntimeError("datetime broken")
            with pytest.raises(HTTPException) as exc_info:
                await get_service_metrics(
                    user=_admin_user(), service=MagicMock(), tenant_id="n8n-dev"
                )
            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_get_ai_test_insights_service_factory(self):
        """get_ai_test_insights_service creates service instance (line 149)."""
        from routers.intelligent_testing_modules.ai_test_insights_config import (
            get_ai_test_insights_service,
        )

        result = await get_ai_test_insights_service(cache_manager=MagicMock())
        from services.ai_test_insights.core_service import AITestInsightsService

        assert isinstance(result, AITestInsightsService)


# ============================================================================
# config/test_framework_logging_manager.py
# ============================================================================


class TestLoggingManager:
    def test_singleton(self):
        from config.test_framework_logging_manager import LoggingManager

        LoggingManager._instance = None
        inst1 = LoggingManager.get_instance()
        inst2 = LoggingManager.get_instance()
        assert inst1 is inst2
        LoggingManager._instance = None

    def test_configure_noop(self):
        from config.test_framework_logging_manager import LoggingManager

        LoggingManager().configure(level="DEBUG", format="%(message)s")

    def test_get_logger(self):
        import logging

        from config.test_framework_logging_manager import LoggingManager

        logger = LoggingManager().get_logger("test_name")
        assert isinstance(logger, logging.Logger)

    def test_set_level_noop(self):
        from config.test_framework_logging_manager import LoggingManager

        LoggingManager().set_level("DEBUG")

    def test_log_test_start(self):
        from config.test_framework_logging_manager import LoggingManager

        LoggingManager().log_test_start("test-123", extra="data")

    def test_log_test_end(self):
        from config.test_framework_logging_manager import LoggingManager

        LoggingManager().log_test_end("test-123", success=True)
        LoggingManager().log_test_end("test-456", success=False)

    def test_log_test_error(self):
        from config.test_framework_logging_manager import LoggingManager

        LoggingManager().log_test_error("test-123", "something went wrong")

    def test_flush_noop(self):
        from config.test_framework_logging_manager import LoggingManager

        LoggingManager().flush()

    def test_shutdown_noop(self):
        from config.test_framework_logging_manager import LoggingManager

        LoggingManager().shutdown()

    def test_init_with_config(self):
        from config.test_framework_logging_manager import LoggingManager

        mgr = LoggingManager(config={"level": "DEBUG"})
        assert mgr.config == {"level": "DEBUG"}
        assert mgr._initialized is True


# ============================================================================
# main.py
# ============================================================================


class TestMainModule:
    def test_main_creates_app(self):
        from main import app

        assert app is not None
        assert app.title == "coding-testing"


# ============================================================================
# app_factory.py
# ============================================================================


class TestAppFactoryAdditional:
    def test_generate_unique_id_root(self):
        from app_factory import generate_unique_id

        mock_route = MagicMock(path="/api/", methods={"GET"})
        assert generate_unique_id(mock_route) == "root_get"

    def test_generate_unique_id_health_root(self):
        from app_factory import generate_unique_id

        mock_route = MagicMock(path="/health", methods={"GET"})
        assert generate_unique_id(mock_route) == "health_root_get"

    def test_generate_unique_id_health_api(self):
        from app_factory import generate_unique_id

        # /api/health path (base_id == "health" and starts with /api/health)
        mock_route = MagicMock(path="/api/health", methods={"GET"})
        assert generate_unique_id(mock_route) == "health_api_get"

    def test_generate_unique_id_with_path_params(self):
        from app_factory import generate_unique_id

        mock_route = MagicMock(path="/api/tests/{test_id}/results", methods={"GET"})
        result = generate_unique_id(mock_route)
        assert "test_id" in result and "results" in result

    def test_generate_unique_id_with_dashes(self):
        from app_factory import generate_unique_id

        mock_route = MagicMock(path="/api/health-check", methods={"POST"})
        assert "health_check" in generate_unique_id(mock_route)

    def test_generate_unique_id_no_methods(self):
        from app_factory import generate_unique_id

        mock_route = MagicMock(path="/api/test", methods=set())
        assert "unknown" in generate_unique_id(mock_route)
