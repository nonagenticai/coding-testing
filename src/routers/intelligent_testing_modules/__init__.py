"""Intelligent testing modules package."""

from .ai_analysis_endpoints import router as ai_analysis_router
from .configuration_management import router as config_router
from .insight_generation_services import router as insight_generation_router
from .report_formatting_endpoints import router as report_formatting_router
from .test_pattern_recognition import router as pattern_recognition_router

__all__ = [
    "ai_analysis_router",
    "pattern_recognition_router",
    "insight_generation_router",
    "report_formatting_router",
    "config_router",
]
