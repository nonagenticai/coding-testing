"""
Configuration management module for intelligent testing.
Handles service settings and configuration updates.
"""

from __future__ import annotations

import logging

from coding_client.auth.models import User
from fastapi import APIRouter, Depends, status

from auth.tenant import get_current_user_with_tenant, require_tenant
from models.common_responses_models import (
    BadRequestResponse,
    InternalServerErrorResponse,
    LegacyResponseStatus,
    NotFoundResponse,
    SimpleValidationErrorResponse,
)
from services.test_execution.intelligent_test_integration import (
    IntelligentTestExecutionService,
    TestExecutionSettings,
)
from shared.responses import (
    bad_request,
    error_response,
    internal_error,
)

from .shared import (
    ServiceSettingsResponse,
    get_intelligent_test_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/settings",
    response_model=ServiceSettingsResponse,
    operation_id="get_ai_service_settings",
    responses={
        200: {"description": "Service settings retrieved successfully"},
        404: {
            "description": "Resource not found - Service not available",
            "model": NotFoundResponse,
        },
        500: {
            "description": "Internal Server Error - Failed to retrieve settings",
            "model": InternalServerErrorResponse,
        },
    },
)
async def get_service_settings(
    user: User = Depends(get_current_user_with_tenant),
    service: IntelligentTestExecutionService = Depends(get_intelligent_test_service),
    tenant_id: str = Depends(require_tenant),
) -> ServiceSettingsResponse:
    """Get current service settings and configuration. Requires admin privileges."""
    # Require admin for configuration management
    if not user.is_admin():
        raise error_response(
            status_code=status.HTTP_403_FORBIDDEN,
            message="Configuration management requires admin privileges",
            error_code="FORBIDDEN",
        )
    try:
        logger.info(f"Retrieving service settings for admin user {user.preferred_username}")

        return ServiceSettingsResponse(
            status=LegacyResponseStatus.SUCCESS,
            message="Service settings retrieved successfully",
            settings=service.settings,
        )

    except Exception as e:
        logger.error(f"Failed to get service settings: {e}")
        raise internal_error(f"Failed to retrieve settings: {e}")


@router.put(
    "/settings",
    response_model=ServiceSettingsResponse,
    operation_id="update_ai_service_settings",
    responses={
        200: {"description": "Service settings updated successfully"},
        400: {
            "description": "Bad Request - Invalid settings configuration",
            "model": BadRequestResponse,
        },
        404: {"description": "Settings resource not found", "model": NotFoundResponse},
        422: {
            "description": "Validation Error - Invalid settings data",
            "model": SimpleValidationErrorResponse,
        },
        500: {
            "description": "Internal Server Error - Failed to update settings",
            "model": InternalServerErrorResponse,
        },
    },
)
async def update_service_settings(
    settings: TestExecutionSettings,
    user: User = Depends(get_current_user_with_tenant),
    service: IntelligentTestExecutionService = Depends(get_intelligent_test_service),
    tenant_id: str = Depends(require_tenant),
) -> ServiceSettingsResponse:
    """Update service settings and configuration. Requires admin privileges."""
    # Require admin for configuration management
    if not user.is_admin():
        raise error_response(
            status_code=status.HTTP_403_FORBIDDEN,
            message="Configuration management requires admin privileges",
            error_code="FORBIDDEN",
        )
    try:
        logger.info(f"Updating service settings by admin user {user.preferred_username}")

        # Validate settings before applying
        await _validate_settings(settings)

        # Update settings
        service.settings = settings

        # Update orchestrator settings if needed
        await _apply_orchestrator_settings(service, settings)

        logger.info("Service settings updated successfully")

        return ServiceSettingsResponse(
            status=LegacyResponseStatus.SUCCESS,
            message="Service settings updated successfully",
            settings=settings,
        )

    except ValueError as e:
        logger.error(f"Invalid settings configuration: {e}")
        raise bad_request(f"Invalid settings configuration: {e}")

    except Exception as e:
        logger.error(f"Failed to update service settings: {e}")
        raise internal_error(f"Failed to update settings: {e}")


async def _validate_settings(settings: TestExecutionSettings) -> None:
    """Validate service settings before applying them."""
    try:
        # Validate concurrent execution limits
        if hasattr(settings, "max_concurrent_tests"):
            max_concurrent = settings.max_concurrent_tests
            if not 1 <= max_concurrent <= 50:
                raise ValueError(
                    f"max_concurrent_tests must be between 1 and 50, got {max_concurrent}"
                )

        # Validate timeout settings
        if hasattr(settings, "default_timeout_seconds"):
            timeout: int = settings.default_timeout_seconds  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
            if not 30 <= timeout <= 3600:
                raise ValueError(
                    f"default_timeout_seconds must be between 30 and 3600, got {timeout}"
                )

        # Validate AI analysis settings
        if hasattr(settings, "ai_analysis_threshold"):
            threshold = settings.ai_analysis_threshold
            if not 0.1 <= threshold <= 1.0:
                raise ValueError(
                    f"ai_analysis_threshold must be between 0.1 and 1.0, got {threshold}"
                )

        # Validate pattern analysis interval
        if hasattr(settings, "pattern_analysis_interval"):
            interval = settings.pattern_analysis_interval
            if not 60 <= interval <= 86400:  # 1 minute to 24 hours
                raise ValueError(
                    f"pattern_analysis_interval must be between 60 and 86400 seconds, got {interval}"
                )

        logger.info("Settings validation passed")

    except Exception as e:
        logger.error(f"Settings validation failed: {e}")
        raise


async def _apply_orchestrator_settings(
    service: IntelligentTestExecutionService, settings: TestExecutionSettings
) -> None:
    """Apply settings to the orchestrator if needed."""
    try:
        orchestrator_mapping = {
            "max_concurrent_tests": "max_concurrent_tests",
            "ai_analysis_threshold": "ai_analysis_threshold",
            "pattern_analysis_interval": "pattern_analysis_interval",
        }

        for setting_attr, orchestrator_attr in orchestrator_mapping.items():
            if hasattr(service.orchestrator, orchestrator_attr) and hasattr(settings, setting_attr):
                setattr(
                    service.orchestrator,
                    orchestrator_attr,
                    getattr(settings, setting_attr),
                )

        logger.info("Orchestrator settings applied successfully")

    except Exception as e:
        logger.error(f"Failed to apply orchestrator settings: {e}")
        raise
