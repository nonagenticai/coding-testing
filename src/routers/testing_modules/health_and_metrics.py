# src/routers/testing_modules/health_and_metrics.py
"""
Health and Metrics Module

Endpoints for:
- Health checks and status monitoring
- Metrics and analytics reporting
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

import httpx
from coding_client.auth.models import User
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)

from auth.tenant import require_tenant
from models.error_models import (
    Error400Response,
    Error401Response,
    Error403Response,
    Error404Response,
    Error500Response,
)
from services.stubs import get_metrics_service
from shared.dependencies import (
    create_json_http_exception,
    requires_read_permission,
)

from .testing_utils import get_execution_service

logger = logging.getLogger(__name__)

# Shared httpx client for executor health checks (avoids per-request allocation)
_executor_client: httpx.AsyncClient | None = None


def _get_executor_client() -> httpx.AsyncClient:
    global _executor_client
    if _executor_client is None:
        _executor_client = httpx.AsyncClient(timeout=3)
    return _executor_client


# Create router
router = APIRouter()


@router.get(
    "/health",
    operation_id="testing_health_check",
    summary="❤️ Testing Service Health",
    description="""
    **Check health status of testing and quality analysis services.**

    Returns detailed status information about:
    - Code execution service availability
    - Docker container runtime status
    - Quality analysis tools availability
    - Cache service connectivity
    - Overall testing capabilities

    This endpoint is resilient and will return partial status even if some services are unavailable.
    """,
    responses={
        200: {
            "description": "Health status retrieved successfully (may show degraded state)",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "status": {
                                "type": "string",
                                "enum": ["healthy", "degraded", "unhealthy"],
                            },
                            "timestamp": {"type": "string", "format": "date-time"},
                            "services": {
                                "type": "object",
                                "properties": {
                                    "execution_service": {"type": "object"},
                                    "docker_service": {"type": "object"},
                                    "cache_service": {"type": "object"},
                                    "quality_tools": {"type": "object"},
                                },
                            },
                            "capabilities": {
                                "type": "object",
                                "properties": {
                                    "test_execution": {"type": "boolean"},
                                    "quality_analysis": {"type": "boolean"},
                                    "security_validation": {"type": "boolean"},
                                    "result_caching": {"type": "boolean"},
                                },
                            },
                            "error": {"type": "string"},
                        },
                    }
                }
            },
        },
        404: {
            "description": "Health endpoint not found",
            "model": Error404Response,
        },
        500: {
            "description": "Health check completely failed - critical service error",
            "model": Error500Response,
        },
    },
)
async def testing_health_check(
    tenant_id: str = Depends(require_tenant),
):
    """
    Health check for testing services.

    Verifies that Docker containers and quality tools are available.
    This endpoint is designed to be resilient and report status even when services are failing.
    """
    try:
        # Try to get execution service, but don't fail if it's not available
        svc_instance = None
        service_init_error = None

        try:
            svc_instance = await get_execution_service()
        except (RuntimeError, HTTPException) as e:
            service_init_error = str(e)
            logger.warning(f"Execution service initialization failed in health check: {e}")

        # Check executor connectivity (test execution uses HTTP executor pods, not Docker)
        executor_urls = {
            "python": os.getenv("PYTHON_TEST_SERVICE_URL"),
            "javascript": os.getenv("JAVASCRIPT_TEST_SERVICE_URL"),
            "java": os.getenv("JAVA_TEST_SERVICE_URL"),
            "go": os.getenv("GO_TEST_SERVICE_URL"),
            "rust": os.getenv("RUST_TEST_SERVICE_URL"),
            "csharp": os.getenv("CSHARP_TEST_SERVICE_URL"),
            "cpp": os.getenv("CPP_TEST_SERVICE_URL"),
        }

        executor_status: dict[str, dict] = {}
        healthy_executors = 0
        total_executors = 0
        client = _get_executor_client()
        for lang, url in executor_urls.items():
            if not url:
                continue
            total_executors += 1
            try:
                r = await client.get(f"{url}/health")
                is_healthy = r.status_code == 200
                executor_status[lang] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "url": url,
                }
                if is_healthy:
                    healthy_executors += 1
            except Exception as e:
                executor_status[lang] = {"status": "unreachable", "url": url, "error": str(e)[:100]}

        executors_healthy = healthy_executors > 0

        # Check cache service health
        cache_healthy = (
            svc_instance is not None
            and hasattr(svc_instance, "cache_manager")
            and svc_instance.cache_manager is not None
        )
        cache_error = None if cache_healthy else "Cache manager not available"

        # Check quality tools availability
        python_tools: list[str] = []
        js_tools: list[str] = []
        tools_error = None

        if svc_instance:
            try:
                python_tools = svc_instance._get_supported_tools("python")
                js_tools = svc_instance._get_supported_tools("javascript")
            except (AttributeError, ValueError) as e:
                tools_error = str(e)

        overall_status = (
            "healthy" if (svc_instance and executors_healthy and cache_healthy) else "degraded"
        )
        if service_init_error:
            overall_status = "unhealthy"

        response = {
            "status": overall_status,
            "timestamp": datetime.now(UTC).isoformat() + "Z",
            "services": {
                "execution_service": {
                    "status": "healthy" if svc_instance else "unhealthy",
                    "available": svc_instance is not None,
                    "error": service_init_error,
                },
                "executors": {
                    "status": "healthy" if executors_healthy else "unhealthy",
                    "healthy_count": healthy_executors,
                    "total_count": total_executors,
                    "details": executor_status,
                },
                "cache_service": {
                    "status": "healthy" if cache_healthy else "unhealthy",
                    "available": cache_healthy,
                    "error": cache_error,
                },
                "quality_tools": {
                    "python_tools_count": len(python_tools),
                    "javascript_tools_count": len(js_tools),
                    "total_tools": len(python_tools) + len(js_tools),
                    "error": tools_error,
                },
            },
            "capabilities": {
                "test_execution": executors_healthy,
                "quality_analysis": executors_healthy
                and (len(python_tools) > 0 or len(js_tools) > 0),
                "security_validation": True,
                "result_caching": cache_healthy,
            },
        }

        return response

    except (ValueError, RuntimeError) as e:
        logger.error(f"❌ Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(UTC).isoformat() + "Z",
            "error": str(e),
            "services": {},
        }


@router.get(
    "/metrics/summary",
    operation_id="get_testing_metrics_summary",
    summary="📊 Testing Metrics Summary",
    description="""
    **Get aggregated testing and quality analysis metrics.**

    Returns comprehensive statistics for the specified time period:

    **Test Execution Metrics**:
    - Total test runs
    - Success/failure rates
    - Average execution time
    - Language distribution

    **Quality Analysis Metrics**:
    - Total quality scans
    - Average quality scores
    - Most common issues
    - Tool usage statistics

    **Resource Usage**:
    - CPU and memory consumption
    - Container utilization
    - Cache hit rates
    """,
    responses={
        200: {
            "description": "Metrics summary retrieved successfully",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "period_days": {"type": "integer"},
                            "test_executions": {"type": "integer"},
                            "quality_analyses": {"type": "integer"},
                            "resource_usage": {"type": "object"},
                            "languages": {"type": "array", "items": {"type": "string"}},
                        },
                    }
                }
            },
        },
        400: {
            "description": "Bad request - invalid days parameter",
            "model": Error400Response,
        },
        401: {
            "description": "Authentication required - missing or invalid authentication token",
            "model": Error401Response,
        },
        403: {
            "description": "Insufficient permissions - user lacks required permissions",
            "model": Error403Response,
        },
        404: {
            "description": "Metrics not found - no metrics available for specified period",
            "model": Error404Response,
        },
        500: {
            "description": "Failed to retrieve metrics - internal server error",
            "model": Error500Response,
        },
    },
)
async def get_testing_metrics_summary(
    days: int = Query(7, ge=1, le=90, description="Number of days to include in summary"),
    current_user: User = Depends(requires_read_permission),
    tenant_id: str = Depends(require_tenant),
):
    """
    Get testing metrics summary for the specified time period.

    Returns aggregated statistics about test executions and quality analyses.
    """
    try:
        logger.info(
            f"📊 Metrics summary requested by user {current_user.preferred_username} for {days} days"
        )

        # Create metrics service and get real data
        metrics_service = get_metrics_service()
        # Use generic metrics method since get_testing_metrics_summary may not exist
        try:
            metrics = await metrics_service.get_test_execution_analytics(days=days)
        except AttributeError:
            # Fallback to basic metrics
            metrics = {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "success_rate": 0.0,
                "average_execution_time": 0.0,
                "quality_score_average": 0.0,
            }

        result = {
            "period_days": metrics.get("period_days", days),
            "test_executions": metrics.get("test_executions", 0),
            "quality_analyses": metrics.get("quality_analyses", 0),
            "resource_usage": metrics.get("resource_usage", {}),
            "languages": metrics.get("languages", []),
        }

        # Always return result, even if empty - this is normal for new systems
        logger.info(
            f"📊 Metrics summary completed: {result['test_executions']} tests, {result['quality_analyses']} analyses"
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get metrics summary: {str(e)}")
        raise create_json_http_exception(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics summary: {str(e)}",
        )


__all__ = ["router"]
