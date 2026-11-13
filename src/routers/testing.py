# src/routers/testing.py
"""
V2 Testing and Quality Analysis Endpoint

Modular testing endpoints split into focused components:
- test_execution: Core test execution and batch processing
- test_discovery: Language, tool, and container discovery
- test_results: Quality analysis, health checks, and metrics
- testing_utils: Shared utilities and background tasks

This refactored structure improves maintainability and reduces file size.
Each module is under 500 lines and focused on specific functionality.
"""

import logging

from fastapi import APIRouter

from routers.testing_modules.health_and_metrics import (
    router as health_and_metrics_router,
)
from routers.testing_modules.quality_analysis import (
    router as quality_analysis_router,
)
from routers.testing_modules.test_discovery import (
    router as test_discovery_router,
)

# Import modular routers
from routers.testing_modules.test_execution import (
    router as test_execution_router,
)

logger = logging.getLogger(__name__)

# Create main router following existing pattern
router = APIRouter()

# Include all modular routers with appropriate prefixes
# Note: Tags are set at the main API level as "V2 🧪 Testing & Quality"
router.include_router(test_execution_router)
router.include_router(test_discovery_router)
router.include_router(quality_analysis_router)
router.include_router(health_and_metrics_router)

logger.info("✅ Testing endpoints initialized with modular structure")

# Export router following existing pattern
__all__ = ["router"]
