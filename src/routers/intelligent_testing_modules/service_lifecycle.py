"""
Service lifecycle management for intelligent testing.
Handles service start, stop, restart operations.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from coding_client.auth.models import User
from fastapi import APIRouter, Depends

from auth.tenant import get_current_user_with_tenant, require_tenant
from models.common_responses_models import (
    BadRequestResponse,
    InternalServerErrorResponse,
    LegacyBaseResponse,
    LegacyResponseStatus,
)
from services.test_execution.intelligent_test_integration import (
    IntelligentTestExecutionService,
    TestExecutionSettings,
)
from shared.responses import (
    forbidden,
    internal_error,
)

from .shared import get_intelligent_test_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/service/start",
    response_model=LegacyBaseResponse,
    operation_id="start_ai_service",
    responses={
        200: {"description": "Service started successfully or already running"},
        400: {
            "description": "Bad Request - Invalid service start request",
            "model": BadRequestResponse,
        },
        500: {
            "description": "Internal Server Error - Failed to start service",
            "model": InternalServerErrorResponse,
        },
    },
)
async def start_service(
    user: User = Depends(get_current_user_with_tenant),
    service: IntelligentTestExecutionService = Depends(get_intelligent_test_service),
    tenant_id: str = Depends(require_tenant),
) -> LegacyBaseResponse:
    """Start the intelligent test service. Requires admin privileges."""
    # Require admin for service lifecycle operations
    if not user.is_admin():
        raise forbidden("Service lifecycle operations require admin privileges")
    try:
        logger.info(f"Service start requested by admin user {user.preferred_username}")
        if service.is_running:
            return LegacyBaseResponse(
                status=LegacyResponseStatus.SUCCESS,
                message="Service is already running",
            )

        logger.info("Starting intelligent test service")
        await service.start()

        return LegacyBaseResponse(
            status=LegacyResponseStatus.SUCCESS, message="Service started successfully"
        )

    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        raise internal_error(f"Failed to start service: {str(e)}")


@router.post(
    "/service/stop",
    response_model=LegacyBaseResponse,
    operation_id="stop_ai_service",
    responses={
        200: {"description": "Service stopped successfully or already stopped"},
        400: {
            "description": "Bad Request - Invalid service stop request",
            "model": BadRequestResponse,
        },
        500: {
            "description": "Internal Server Error - Failed to stop service",
            "model": InternalServerErrorResponse,
        },
    },
)
async def stop_service(
    user: User = Depends(get_current_user_with_tenant),
    service: IntelligentTestExecutionService = Depends(get_intelligent_test_service),
    tenant_id: str = Depends(require_tenant),
) -> LegacyBaseResponse:
    """Stop the intelligent test service. Requires admin privileges."""
    # Require admin for service lifecycle operations
    if not user.is_admin():
        raise forbidden("Service lifecycle operations require admin privileges")
    try:
        logger.info(f"Service stop requested by admin user {user.preferred_username}")
        if not service.is_running:
            return LegacyBaseResponse(
                status=LegacyResponseStatus.SUCCESS,
                message="Service is already stopped",
            )

        logger.info("Stopping intelligent test service")
        await service.stop()

        return LegacyBaseResponse(
            status=LegacyResponseStatus.SUCCESS, message="Service stopped successfully"
        )

    except Exception as e:
        logger.error(f"Failed to stop service: {e}")
        raise internal_error(f"Failed to stop service: {str(e)}")


@router.post(
    "/service/restart",
    response_model=LegacyBaseResponse,
    operation_id="restart_ai_service",
    responses={
        200: {"description": "Service restarted successfully"},
        400: {
            "description": "Bad Request - Invalid service restart request",
            "model": BadRequestResponse,
        },
        500: {
            "description": "Internal Server Error - Failed to restart service",
            "model": InternalServerErrorResponse,
        },
    },
)
async def restart_service(
    user: User = Depends(get_current_user_with_tenant),
    service: IntelligentTestExecutionService = Depends(get_intelligent_test_service),
    tenant_id: str = Depends(require_tenant),
) -> LegacyBaseResponse:
    """Restart the intelligent test service. Requires admin privileges."""
    # Require admin for service lifecycle operations
    if not user.is_admin():
        raise forbidden("Service lifecycle operations require admin privileges")
    try:
        logger.info(f"Service restart requested by admin user {user.preferred_username}")

        # Stop if running
        if service.is_running:
            await service.stop()

        # Start service
        await service.start()

        return LegacyBaseResponse(
            status=LegacyResponseStatus.SUCCESS,
            message="Service restarted successfully",
        )

    except Exception as e:
        logger.error(f"Failed to restart service: {e}")
        raise internal_error(f"Failed to restart service: {str(e)}")


async def backup_settings(service: IntelligentTestExecutionService) -> dict[str, Any]:
    """Create a backup of current service settings."""
    try:
        logger.info("Creating settings backup")

        # Build orchestrator settings first
        orchestrator_settings: dict[str, Any] = {}
        if hasattr(service.orchestrator, "max_concurrent_tests"):
            orchestrator_settings["max_concurrent_tests"] = (
                service.orchestrator.max_concurrent_tests
            )
        if hasattr(service.orchestrator, "ai_analysis_threshold"):
            orchestrator_settings["ai_analysis_threshold"] = (
                service.orchestrator.ai_analysis_threshold
            )
        if hasattr(service.orchestrator, "pattern_analysis_interval"):
            orchestrator_settings["pattern_analysis_interval"] = (
                service.orchestrator.pattern_analysis_interval
            )

        backup: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "settings": (
                service.settings.model_dump()
                if hasattr(service.settings, "model_dump")
                else str(service.settings)
            ),
            "orchestrator_settings": orchestrator_settings,
        }

        logger.info("Settings backup created successfully")
        return backup

    except Exception as e:
        logger.error(f"Failed to create settings backup: {e}")
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "error": str(e),
            "settings": {},
            "orchestrator_settings": {},
        }


async def restore_settings(
    service: IntelligentTestExecutionService, backup: dict[str, Any]
) -> bool:
    """Restore service settings from a backup."""
    try:
        logger.info("Restoring settings from backup")

        # Restore main settings
        if "settings" in backup:
            settings_data = backup["settings"]
            if isinstance(settings_data, dict):
                # Create new settings object from backup data
                service.settings = TestExecutionSettings(**settings_data)
            else:
                logger.warning("Settings backup format not supported for restoration")
                return False

        # Restore orchestrator settings
        orchestrator_settings = backup.get("orchestrator_settings", {})
        for setting_name, setting_value in orchestrator_settings.items():
            if hasattr(service.orchestrator, setting_name):
                setattr(service.orchestrator, setting_name, setting_value)

        logger.info("Settings restored successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to restore settings: {e}")
        return False


async def get_configuration_status(
    service: IntelligentTestExecutionService,
) -> dict[str, Any]:
    """Get comprehensive configuration status."""
    try:
        logger.info("Getting configuration status")

        status: dict[str, Any] = {
            "service_running": service.is_running,
            "settings_valid": True,
            "orchestrator_configured": hasattr(service, "orchestrator"),
            "last_updated": datetime.now(UTC).isoformat(),
            "configuration_health": "healthy",
        }

        # Check orchestrator status
        if hasattr(service, "orchestrator"):
            status["orchestrator_status"] = {
                "available": True,
                "max_concurrent_tests": getattr(
                    service.orchestrator, "max_concurrent_tests", "unknown"
                ),
                "ai_analysis_threshold": getattr(
                    service.orchestrator, "ai_analysis_threshold", "unknown"
                ),
                "pattern_analysis_interval": getattr(
                    service.orchestrator, "pattern_analysis_interval", "unknown"
                ),
            }
        else:
            status["orchestrator_status"] = {"available": False}
            status["configuration_health"] = "degraded"

        return status

    except Exception as e:
        logger.error(f"Failed to get configuration status: {e}")
        return {
            "service_running": False,
            "settings_valid": False,
            "orchestrator_configured": False,
            "last_updated": datetime.now(UTC).isoformat(),
            "configuration_health": "critical",
            "error": str(e),
        }
