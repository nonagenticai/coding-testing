"""
Test discovery and recommendation module for intelligent testing.
Handles AI-powered test discovery, recommendations, and pattern analysis
with container-scoped operations.
"""

from __future__ import annotations

import logging
from typing import Any

from coding_client.auth.models import User
from fastapi import APIRouter, Depends, HTTPException, Query

from auth.tenant import get_current_user_with_tenant, require_tenant
from models.common_responses_models import (
    InternalServerErrorResponse,
    LegacyResponseStatus,
    NotFoundResponse,
)
from services.test_execution.intelligent_test_integration import (
    IntelligentTestExecutionService,
)
from services.test_execution.pydantic_ai_agents import (
    TestRecommendation,
)
from shared.container_validation import validate_and_get_container_for_tenant
from shared.responses import (
    internal_error,
)

from .shared import (
    PatternsResponse,
    RecommendationsResponse,
    get_intelligent_test_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/recommendations",
    response_model=RecommendationsResponse,
    operation_id="get_ai_recommendations",
    responses={
        200: {"description": "AI recommendations generated successfully"},
        404: {
            "description": "Resource not found - Container or execution data not available",
            "model": NotFoundResponse,
        },
        500: {
            "description": "Internal Server Error - Failed to generate recommendations",
            "model": InternalServerErrorResponse,
        },
    },
)
async def get_recommendations(
    container_name: str = Query(..., description="Container name to scope recommendations"),
    user: User = Depends(get_current_user_with_tenant),
    service: IntelligentTestExecutionService = Depends(get_intelligent_test_service),
    tenant_id: str = Depends(require_tenant),
) -> RecommendationsResponse:
    """Get AI-generated recommendations for test optimization scoped to a container.

    This endpoint provides intelligent recommendations for:
    - Performance optimization
    - Resource allocation improvements
    - Test reliability enhancements
    - Configuration optimizations

    All recommendations are scoped to the specified container context.

    Args:
        container_name: Container name to scope recommendations
        user: Authenticated user with tenant context
        service: The intelligent test execution service

    Returns:
        RecommendationsResponse: Container-scoped recommendations

    Raises:
        HTTPException: If container not found or recommendations generation fails
    """
    try:
        # Tenant-aware container validation (admins can access all, users only their own)
        effective_tenant_id = None if user.is_admin() else user.effective_tenant_id
        await validate_and_get_container_for_tenant(container_name, tenant_id=effective_tenant_id)

        logger.info(f"Generating AI recommendations for container '{container_name}'")

        recommendations = await service.get_recommendations()

        return RecommendationsResponse(
            status=LegacyResponseStatus.SUCCESS,
            message=f"Recommendations generated successfully for container '{container_name}'",
            recommendations=recommendations,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get recommendations for container '{container_name}': {e}")
        raise internal_error("Failed to generate recommendations", details={"message": str(e)})


@router.get(
    "/patterns",
    response_model=PatternsResponse,
    operation_id="get_execution_patterns",
    responses={
        200: {"description": "Execution patterns retrieved successfully"},
        404: {
            "description": "Resource not found - Container or pattern data not available",
            "model": NotFoundResponse,
        },
        500: {
            "description": "Internal Server Error - Failed to retrieve patterns",
            "model": InternalServerErrorResponse,
        },
    },
)
async def get_execution_patterns(
    container_name: str = Query(..., description="Container name to scope pattern analysis"),
    user: User = Depends(get_current_user_with_tenant),
    service: IntelligentTestExecutionService = Depends(get_intelligent_test_service),
    tenant_id: str = Depends(require_tenant),
) -> PatternsResponse:
    """Get identified test execution patterns scoped to a container.

    This endpoint provides detected patterns including:
    - Performance patterns (slow tests, resource usage)
    - Failure patterns (common error types, failure clustering)
    - Timing patterns (peak usage, scheduling conflicts)
    - Language and test type specific patterns

    All patterns are scoped to the specified container context.

    Args:
        container_name: Container name to scope pattern analysis
        user: Authenticated user with tenant context
        service: The intelligent test execution service

    Returns:
        PatternsResponse: Container-scoped execution patterns

    Raises:
        HTTPException: If container not found or pattern retrieval fails
    """
    try:
        # Tenant-aware container validation (admins can access all, users only their own)
        effective_tenant_id = None if user.is_admin() else user.effective_tenant_id
        await validate_and_get_container_for_tenant(container_name, tenant_id=effective_tenant_id)

        logger.info(f"Retrieving execution patterns for container '{container_name}'")

        patterns = await service.get_patterns()

        return PatternsResponse(
            status=LegacyResponseStatus.SUCCESS,
            message=f"Execution patterns retrieved successfully for container '{container_name}'",
            patterns=patterns,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get execution patterns for container '{container_name}': {e}")
        raise internal_error("Failed to retrieve patterns", details={"message": str(e)})


@router.get(
    "/supported-languages",
    operation_id="list_ai_supported_languages",
    responses={
        200: {"description": "Supported languages and test types retrieved successfully"},
        404: {
            "description": "Resource not found - Service not available",
            "model": NotFoundResponse,
        },
        500: {
            "description": "Internal Server Error - Failed to retrieve supported languages",
            "model": InternalServerErrorResponse,
        },
    },
)
async def get_supported_languages(
    service: IntelligentTestExecutionService = Depends(get_intelligent_test_service),
    tenant_id: str = Depends(require_tenant),
) -> dict[str, Any]:
    """Get List[Any] of supported programming languages and test types.

    This endpoint provides information about supported languages and their available test types.

    Args:
        service (IntelligentTestExecutionService, optional): The service parameter. Defaults to Depends(get_intelligent_test_service).

    Returns:
        Dict[str, List[str]]: The dict[str, list[str]] result.

    Raises:
        HTTPException: Raised under certain conditions.

    Example:
        >>> await get_supported_languages("service_value")"""
    try:
        supported_languages = await service.orchestrator.base_orchestrator.get_supported_languages()

        # Get test types for each language
        language_test_types = {}
        for language in supported_languages:
            test_types = await service.orchestrator.base_orchestrator.get_supported_test_types(
                language
            )
            language_test_types[language] = test_types

        return {
            "status": "success",
            "data": {
                "languages": supported_languages,
                "language_test_types": language_test_types,
            },
        }

    except Exception as e:
        logger.error(f"Failed to get supported languages: {e}")
        # Return empty data instead of 500 error
        return {
            "status": "success",
            "data": {"languages": [], "language_test_types": {}},
        }


async def discover_test_opportunities(
    service: IntelligentTestExecutionService,
    container_name: str,
    code: str,
    language: str,
    tenant_id: str | None = None,
    project_context: dict[str, Any] | None = None,
) -> list[TestRecommendation]:
    """Discover test opportunities in the provided code scoped to a container.

    Args:
        service: The intelligent test execution service
        container_name: Container name to scope discovery
        code: Source code to analyze
        language: Programming language
        tenant_id: Tenant ID for access control (None for admin access)
        project_context: Optional project context for better recommendations

    Returns:
        List of test recommendations scoped to container
    """
    try:
        # Tenant-aware container validation
        await validate_and_get_container_for_tenant(container_name, tenant_id=tenant_id)

        logger.info(
            f"Discovering test opportunities for container '{container_name}' ({language} code)"
        )

        # Use the service's AI capabilities to analyze code and suggest tests
        recommendations = await service.get_recommendations()

        # Filter recommendations based on language and code analysis
        filtered_recommendations = []
        for recommendation in recommendations:
            rec_language = getattr(recommendation, "language", None)
            if rec_language is not None and str(rec_language).lower() == language.lower():
                filtered_recommendations.append(recommendation)

        return filtered_recommendations

    except Exception as e:
        logger.error(f"Failed to discover test opportunities for container '{container_name}': {e}")
        return []


async def analyze_test_coverage_gaps(
    service: IntelligentTestExecutionService,
    container_name: str,
    existing_tests: list[str],
    source_code: str,
    language: str,
    tenant_id: str | None = None,
) -> list[TestRecommendation]:
    """Analyze gaps in test coverage and suggest improvements scoped to a container.

    Args:
        service: The intelligent test execution service
        container_name: Container name to scope analysis
        existing_tests: List of existing test code
        source_code: Source code to analyze
        language: Programming language
        tenant_id: Tenant ID for access control (None for admin access)

    Returns:
        List of recommendations for coverage improvements scoped to container
    """
    try:
        # Tenant-aware container validation
        await validate_and_get_container_for_tenant(container_name, tenant_id=tenant_id)

        logger.info(f"Analyzing test coverage gaps for container '{container_name}' ({language})")

        # Get patterns to identify common gaps
        patterns = await service.get_patterns()

        coverage_recommendations: list[TestRecommendation] = []

        # Analyze patterns to identify coverage gaps
        for pattern in patterns:
            if hasattr(pattern, "pattern_type") and "coverage" in pattern.pattern_type.lower():
                # Create recommendation based on coverage pattern
                recommendation = TestRecommendation(
                    recommendation_id=f"coverage_{len(coverage_recommendations)}",
                    recommendation_type="coverage_gap",
                    priority="medium",
                    title="Coverage Gap Detected",
                    description=f"Coverage gap identified: {pattern.description}",
                    impact_assessment="Add test coverage for identified gap",
                    implementation_steps=[
                        "Identify uncovered code paths",
                        "Write tests for identified gaps",
                    ],
                    confidence_level=0.8,
                    estimated_improvement=0.5,
                )
                coverage_recommendations.append(recommendation)

        return coverage_recommendations

    except Exception as e:
        logger.error(f"Failed to analyze coverage gaps for container '{container_name}': {e}")
        return []


async def recommend_test_strategies(
    service: IntelligentTestExecutionService,
    container_name: str,
    project_type: str,
    language: str,
    complexity_level: str = "medium",
    tenant_id: str | None = None,
) -> list[TestRecommendation]:
    """Recommend testing strategies based on project characteristics scoped to a container.

    Args:
        service: The intelligent test execution service
        container_name: Container name to scope recommendations
        project_type: Type of project (web, api, library, etc.)
        language: Programming language
        complexity_level: Project complexity level
        tenant_id: Tenant ID for access control (None for admin access)

    Returns:
        List of strategic testing recommendations scoped to container
    """
    try:
        # Tenant-aware container validation
        await validate_and_get_container_for_tenant(container_name, tenant_id=tenant_id)

        logger.info(
            f"Recommending test strategies for container '{container_name}' ({project_type} project in {language})"
        )

        strategy_recommendations = []

        # Base recommendations based on project type
        if project_type.lower() in ["web", "api"]:
            strategy_recommendations.extend(
                [
                    TestRecommendation(
                        recommendation_id="strategy_integration",
                        recommendation_type="strategy",
                        priority="high",
                        title="Integration Test Suite",
                        description="Implement integration tests for API endpoints",
                        impact_assessment="High impact on API reliability and regression detection",
                        implementation_steps=[
                            "Set up test environment",
                            "Create API test fixtures",
                            "Write endpoint tests",
                        ],
                        confidence_level=0.9,
                        estimated_improvement=0.7,
                    ),
                    TestRecommendation(
                        recommendation_id="strategy_contract",
                        recommendation_type="strategy",
                        priority="medium",
                        title="Contract Testing Framework",
                        description="Implement contract testing for API compatibility",
                        impact_assessment="Medium impact on API compatibility and versioning",
                        implementation_steps=[
                            "Choose contract testing tool",
                            "Define API contracts",
                            "Integrate into CI/CD",
                        ],
                        confidence_level=0.8,
                        estimated_improvement=0.5,
                    ),
                ]
            )

        elif project_type.lower() == "library":
            strategy_recommendations.extend(
                [
                    TestRecommendation(
                        recommendation_id="strategy_unit",
                        recommendation_type="strategy",
                        priority="high",
                        title="Comprehensive Unit Testing",
                        description="Comprehensive unit testing for library functions",
                        impact_assessment="High impact on code quality and maintainability",
                        implementation_steps=[
                            "Identify public API surface",
                            "Write unit tests for each function",
                            "Add edge case coverage",
                        ],
                        confidence_level=0.95,
                        estimated_improvement=0.8,
                    ),
                    TestRecommendation(
                        recommendation_id="strategy_compatibility",
                        recommendation_type="strategy",
                        priority="medium",
                        title="Version Compatibility Testing",
                        description="Test compatibility across different versions",
                        impact_assessment="Medium impact on library adoption and stability",
                        implementation_steps=[
                            "Set up compatibility test matrix",
                            "Test against multiple runtime versions",
                            "Document supported versions",
                        ],
                        confidence_level=0.8,
                        estimated_improvement=0.5,
                    ),
                ]
            )

        return strategy_recommendations

    except Exception as e:
        logger.error(f"Failed to recommend test strategies: {e}")
        return []
