"""
API endpoints for intelligent test orchestration with PydanticAI integration.
Provides advanced test execution, analysis, and optimization capabilities.

This module has been refactored into focused sub-modules for better maintainability:
- discovery_and_recommendations: Test discovery and AI recommendations
- execution_coordination: Test execution and batch processing
- analysis_and_insights: Result analysis and performance insights
- configuration_management: Service settings and lifecycle
- utilities_and_helpers: Shared utilities and validation

The original monolithic file has been split to maintain sub-500 line modules
while preserving all intelligent testing functionality.
"""

from __future__ import annotations

import asyncio
import atexit
import logging

from fastapi import APIRouter, Depends

from auth.tenant import require_tenant

from .intelligent_testing_modules.ai_analysis_endpoints import (
    router as ai_analysis_router,
)
from .intelligent_testing_modules.ai_test_insights_config import (
    router as ai_insights_config_router,
)
from .intelligent_testing_modules.analysis_and_insights import router as analysis_router
from .intelligent_testing_modules.configuration_management import (
    router as config_router,
)
from .intelligent_testing_modules.container_scoped_testing import (
    router as container_scoped_router,
)
from .intelligent_testing_modules.discovery_and_recommendations import (
    router as discovery_router,
)
from .intelligent_testing_modules.execution_coordination import (
    router as execution_router,
)
from .intelligent_testing_modules.insight_generation_services import (
    router as insight_generation_router,
)
from .intelligent_testing_modules.report_formatting_endpoints import (
    router as report_formatting_router,
)
from .intelligent_testing_modules.service_lifecycle import router as lifecycle_router
from .intelligent_testing_modules.test_pattern_recognition import (
    router as pattern_recognition_router,
)

logger = logging.getLogger(__name__)

# Main router for intelligent testing
router = APIRouter()

# Include sub-module routers (tags are set at main API level as "V2 🤖 Intelligent Testing")
router.include_router(discovery_router)
router.include_router(execution_router)
router.include_router(analysis_router)
router.include_router(config_router)
router.include_router(lifecycle_router)
router.include_router(container_scoped_router)

# Include AI test insights sub-module routers
router.include_router(ai_analysis_router)
router.include_router(ai_insights_config_router)
router.include_router(insight_generation_router)
router.include_router(report_formatting_router)
router.include_router(pattern_recognition_router)


# Cleanup handler using atexit (replaces deprecated on_event("shutdown"))
def _cleanup_service_sync():
    """Synchronous cleanup wrapper for atexit."""
    from .intelligent_testing_modules import shared

    if shared._service_instance and shared._service_instance.is_running:
        logger.info("Shutting down intelligent test service")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(shared._service_instance.stop())
            else:
                asyncio.run(shared._service_instance.stop())
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        shared._service_instance = None


atexit.register(_cleanup_service_sync)


# Health check endpoint for the main router
@router.get("/status", operation_id="get_ai_module_status")
async def get_module_status(
    tenant_id: str = Depends(require_tenant),
):
    """Get the status of intelligent testing modules."""
    return {
        "status": "active",
        "modules": {
            "discovery_and_recommendations": "active",
            "execution_coordination": "active",
            "analysis_and_insights": "active",
            "configuration_management": "active",
            "service_lifecycle": "active",
            "container_scoped_testing": "active",
            "utilities_and_helpers": "active",
            "metrics_and_calculations": "active",
            "analysis_core": "active",
        },
        "description": "Intelligent testing system with modular architecture",
        "version": "1.0.0",
    }
