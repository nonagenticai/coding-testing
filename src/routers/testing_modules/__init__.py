"""Testing modules package.

This package contains modular components for test management and execution.
"""

from .health_and_metrics import router as health_metrics_router
from .quality_analysis import router as quality_analysis_router
from .test_discovery import router as test_discovery_router
from .test_execution import router as test_execution_router

# Utility functions
from .testing_utils import (
    aggregate_test_metrics,
    format_test_results,
    validate_test_request,
)

__all__ = [
    # Routers
    "health_metrics_router",
    "quality_analysis_router",
    "test_discovery_router",
    "test_execution_router",
    # Utilities
    "validate_test_request",
    "format_test_results",
    "aggregate_test_metrics",
]
