"""Health check endpoints for coding-testing."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from auth.tenant import require_tenant
from config.settings import get_settings

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(
    responses={
        500: {"description": "Internal server error"},
        503: {"description": "Service unavailable"},
    },
)


class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Health status")
    service: str = Field(default="coding-testing", description="Service name")
    version: str = Field(default="1.0.0", description="Service version")
    uptime_seconds: float | None = Field(None, description="Uptime in seconds")
    checks: dict[str, Any] = Field(default_factory=dict, description="Component checks")


class SystemMetricsResponse(BaseModel):
    cpu_percent: float | None = None
    memory_percent: float | None = None
    memory_used_mb: float | None = None
    memory_total_mb: float | None = None
    disk_percent: float | None = None
    open_file_descriptors: int | None = None


_start_time = time.time()


@router.get(
    "",
    response_model=HealthCheckResponse,
    summary="Health Check",
    operation_id="health_api_check",
)
async def health_check(
    tenant_id: str = Depends(require_tenant),
) -> HealthCheckResponse:
    """Return overall health status."""
    uptime = time.time() - _start_time
    return HealthCheckResponse(
        status="healthy",
        service=os.getenv("SERVICE_NAME", "coding-testing"),
        version=os.getenv("VERSION", "1.0.0"),
        uptime_seconds=round(uptime, 2),
    )


@router.get(
    "/ready",
    response_model=HealthCheckResponse,
    summary="Readiness Check",
    operation_id="health_ready",
)
async def service_readiness_check() -> HealthCheckResponse:
    """Kubernetes readiness probe."""
    return HealthCheckResponse(
        status="ready",
        service=os.getenv("SERVICE_NAME", "coding-testing"),
        version=os.getenv("VERSION", "1.0.0"),
    )


@router.get(
    "/live",
    response_model=HealthCheckResponse,
    summary="Liveness Check",
    operation_id="health_live",
)
async def service_liveness_check() -> HealthCheckResponse:
    """Kubernetes liveness probe."""
    return HealthCheckResponse(
        status="alive",
        service=os.getenv("SERVICE_NAME", "coding-testing"),
        version=os.getenv("VERSION", "1.0.0"),
    )


@router.get(
    "/detailed",
    summary="Detailed Health Check",
    operation_id="health_detailed",
)
async def detailed_health_check(
    tenant_id: str = Depends(require_tenant),
) -> dict[str, Any]:
    """Return detailed health information including system metrics."""
    uptime = time.time() - _start_time
    result: dict[str, Any] = {
        "status": "healthy",
        "service": os.getenv("SERVICE_NAME", "coding-testing"),
        "version": os.getenv("VERSION", "1.0.0"),
        "environment": os.getenv("ENVIRONMENT", "production"),
        "uptime_seconds": round(uptime, 2),
    }

    if HAS_PSUTIL:
        result["system"] = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_mb": round(psutil.virtual_memory().used / (1024 * 1024), 2),
            "memory_total_mb": round(psutil.virtual_memory().total / (1024 * 1024), 2),
        }

    return result


@router.get(
    "/system-metrics",
    response_model=SystemMetricsResponse,
    summary="System Metrics",
    operation_id="get_health_system_metrics",
)
async def get_system_metrics(
    tenant_id: str = Depends(require_tenant),
) -> SystemMetricsResponse:
    """Return system resource metrics."""
    if not HAS_PSUTIL:
        return SystemMetricsResponse()

    vm = psutil.virtual_memory()
    return SystemMetricsResponse(
        cpu_percent=psutil.cpu_percent(interval=0.1),
        memory_percent=vm.percent,
        memory_used_mb=round(vm.used / (1024 * 1024), 2),
        memory_total_mb=round(vm.total / (1024 * 1024), 2),
        disk_percent=psutil.disk_usage("/").percent,
        open_file_descriptors=len(psutil.Process().open_files())
        if hasattr(psutil.Process(), "open_files")
        else None,
    )
