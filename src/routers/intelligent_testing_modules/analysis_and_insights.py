"""
Test analysis and insights module for intelligent testing.
Handles test result analysis, performance insights, and failure analysis.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from coding_client.auth.models import User
from fastapi import APIRouter, Depends

from auth.tenant import get_current_user_with_tenant, require_tenant
from models.common_responses_models import (
    InternalServerErrorResponse,
    LegacyResponseStatus,
    NotFoundResponse,
)
from services.test_execution.intelligent_test_integration import (
    IntelligentTestExecutionService,
)
from services.test_execution.intelligent_test_orchestrator import (
    IntelligentTestExecutionResult,
)
from shared.responses import (
    internal_error,
)

from .analysis_core import (
    _generate_failure_recommendations,
    _generate_performance_recommendations,
    _generate_trend_recommendations,
    _identify_failure_patterns,
    _identify_performance_bottlenecks,
)
from .shared import (
    ExecutionInsightsResponse,
    HealthStatusResponse,
    get_intelligent_test_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/insights",
    response_model=ExecutionInsightsResponse,
    operation_id="testing_get_execution_insights",
    responses={
        200: {"description": "Execution insights generated successfully"},
        404: {
            "description": "Resource not found - No execution data available",
            "model": NotFoundResponse,
        },
        500: {
            "description": "Internal Server Error - Failed to generate insights",
            "model": InternalServerErrorResponse,
        },
    },
)
async def get_execution_insights(
    user: User = Depends(get_current_user_with_tenant),
    service: IntelligentTestExecutionService = Depends(get_intelligent_test_service),
    tenant_id: str = Depends(require_tenant),
) -> ExecutionInsightsResponse:
    """Get comprehensive execution insights from AI analysis.

    Requires authentication. Insights are service-level aggregates.
    """
    try:
        logger.info(f"Generating execution insights for user {user.preferred_username}")
        insights = await service.get_execution_insights()

        return ExecutionInsightsResponse(
            status=LegacyResponseStatus.SUCCESS,
            message="Execution insights generated successfully",
            insights=insights,
        )

    except Exception as e:
        logger.error(f"Failed to get execution insights: {e}")
        raise internal_error("Failed to generate insights", details={"message": str(e)})


@router.get(
    "/health",
    response_model=HealthStatusResponse,
    operation_id="get_ai_health_status",
    responses={
        200: {"description": "Health status retrieved successfully"},
        404: {
            "description": "Resource not found - Service not available",
            "model": NotFoundResponse,
        },
        500: {
            "description": "Internal Server Error - Failed to retrieve health status",
            "model": InternalServerErrorResponse,
        },
    },
)
async def get_health_status(
    user: User = Depends(get_current_user_with_tenant),
    service: IntelligentTestExecutionService = Depends(get_intelligent_test_service),
    tenant_id: str = Depends(require_tenant),
) -> HealthStatusResponse:
    """Get comprehensive health status of the intelligent test service.

    Requires authentication. Returns service-level health information.
    """
    try:
        logger.info(f"Checking service health status for user {user.preferred_username}")
        health_status = await service.get_health_status()

        return HealthStatusResponse(
            status=LegacyResponseStatus.SUCCESS,
            message="Health status retrieved successfully",
            health_status=health_status,
        )

    except Exception as e:
        logger.error(f"Failed to get health status: {e}")
        raise internal_error("Failed to retrieve health status", details={"message": str(e)})


async def analyze_test_performance(
    results: list[IntelligentTestExecutionResult],
) -> dict[str, Any]:
    """Analyze performance metrics from test execution results."""
    try:
        logger.info(f"Analyzing performance for {len(results)} test results")

        if not results:
            return {
                "total_tests": 0,
                "performance_summary": {},
                "bottlenecks": [],
                "recommendations": [],
            }

        # Calculate performance metrics
        execution_times = [r.execution_time_ms for r in results if r.execution_time_ms]
        success_rate = sum(1 for r in results if r.success) / len(results) * 100

        performance_summary = {
            "total_execution_time": sum(execution_times),
            "average_execution_time": (
                sum(execution_times) / len(execution_times) if execution_times else 0
            ),
            "min_execution_time": min(execution_times) if execution_times else 0,
            "max_execution_time": max(execution_times) if execution_times else 0,
            "success_rate": success_rate,
            "total_tests": len(results),
        }

        # Identify bottlenecks and generate recommendations
        bottlenecks = await _identify_performance_bottlenecks(results)
        recommendations = await _generate_performance_recommendations(
            performance_summary, bottlenecks
        )

        return {
            "performance_summary": performance_summary,
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "analysis_timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to analyze test performance: {e}")
        return {
            "error": str(e),
            "performance_summary": {},
            "bottlenecks": [],
            "recommendations": [],
        }


async def analyze_failure_patterns(
    results: list[IntelligentTestExecutionResult],
) -> dict[str, Any]:
    """Analyze failure patterns in test execution results."""
    try:
        logger.info(f"Analyzing failure patterns for {len(results)} test results")

        failed_results = [r for r in results if not r.success]

        if not failed_results:
            return {
                "total_failures": 0,
                "failure_patterns": [],
                "common_errors": [],
                "failure_recommendations": [],
            }

        # Group failures by error type
        error_groups: dict[str, list[IntelligentTestExecutionResult]] = {}
        for result in failed_results:
            error_type = getattr(result, "error_type", "unknown")
            if error_type not in error_groups:
                error_groups[error_type] = []
            error_groups[error_type].append(result)

        # Identify common error patterns
        common_errors = []
        for error_type, error_results in error_groups.items():
            if len(error_results) > 1:  # Only consider patterns with multiple occurrences
                common_errors.append(
                    {
                        "error_type": error_type,
                        "occurrences": len(error_results),
                        "percentage": len(error_results) / len(failed_results) * 100,
                        "sample_message": getattr(
                            error_results[0], "error_message", "No message available"
                        ),
                    }
                )

        # Sort by frequency
        common_errors.sort(key=lambda x: x["occurrences"], reverse=True)

        # Generate patterns and recommendations
        failure_patterns = await _identify_failure_patterns(failed_results)
        failure_recommendations = await _generate_failure_recommendations(common_errors)

        return {
            "total_failures": len(failed_results),
            "failure_rate": len(failed_results) / len(results) * 100,
            "common_errors": common_errors,
            "failure_patterns": failure_patterns,
            "failure_recommendations": failure_recommendations,
            "analysis_timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to analyze failure patterns: {e}")
        return {
            "error": str(e),
            "total_failures": 0,
            "failure_patterns": [],
            "common_errors": [],
            "failure_recommendations": [],
        }


async def generate_test_trends(
    service: IntelligentTestExecutionService, time_period_hours: int = 24
) -> dict[str, Any]:
    """Generate test execution trends over a specified time period."""
    try:
        logger.info(f"Generating test trends for the last {time_period_hours} hours")

        # Get execution insights from service
        insights = await service.get_execution_insights()

        # Extract trend data from insights
        trend_data = {
            "time_period_hours": time_period_hours,
            "execution_trends": {
                "total_executions": insights.get("total_executions", 0),
                "success_trend": insights.get("success_trend", []),
                "failure_trend": insights.get("failure_trend", []),
                "performance_trend": insights.get("performance_trend", []),
            },
            "language_trends": insights.get("language_trends", {}),
            "peak_hours": insights.get("peak_hours", []),
            "trend_recommendations": [],
        }

        # Generate trend-based recommendations
        trend_data["trend_recommendations"] = await _generate_trend_recommendations(trend_data)

        return trend_data

    except Exception as e:
        logger.error(f"Failed to generate test trends: {e}")
        return {
            "error": str(e),
            "time_period_hours": time_period_hours,
            "execution_trends": {},
            "language_trends": {},
            "peak_hours": [],
            "trend_recommendations": [],
        }
