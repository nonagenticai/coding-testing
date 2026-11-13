"""AI-Powered Test Insights Service using PydanticAI.

This service provides intelligent analysis of test patterns, performance insights,
and quality metrics to help developers understand and improve their testing practices.
It uses PydanticAI for type-safe AI model interactions and provides actionable
recommendations for test optimization.

Features:
- Test pattern analysis and detection
- Performance bottleneck identification
- Quality metric analysis with recommendations
- Intelligent test failure prediction
- Test coverage optimization suggestions
- Test reliability scoring and tracking

The service integrates with the existing testing infrastructure to provide
comprehensive insights into test execution patterns and quality metrics.

Note: This service has been refactored into modular components under ai_test_insights/
for better maintainability. This file now provides backwards compatibility.
"""

import logging

# Import from modular components for backwards compatibility
from services.ai_test_insights import (
    DEFAULT_CONFIG,
    AITestInsightsConfig,
    AITestInsightsService,
    TestCoverageInsights,
    TestInsightAnalysis,
    TestInsightRecommendation,
    TestInsightType,
    TestPatternInsight,
    TestPatternType,
    TestPerformanceMetrics,
    TestQualityInsights,
    TestReliabilityMetrics,
    create_ai_test_insights_service,
)

# Cache manager is passed as dependency


logger = logging.getLogger(__name__)


# Backwards compatibility exports - these classes are now imported from modular components
# Users can continue to import these from this module without changing their code

# Export the imported classes for backwards compatibility
__all__ = [
    "AITestInsightsService",
    "create_ai_test_insights_service",
    "TestInsightAnalysis",
    "TestPerformanceMetrics",
    "TestReliabilityMetrics",
    "TestCoverageInsights",
    "TestQualityInsights",
    "TestPatternInsight",
    "TestInsightRecommendation",
    "TestPatternType",
    "TestInsightType",
    "AITestInsightsConfig",
    "DEFAULT_CONFIG",
]
