"""
Container-Scoped Intelligent Testing Endpoints.

This module provides container-scoped endpoints for:
- Smart test selection
- Test impact analysis
- Test prioritization
- Advanced pattern recognition
"""

from __future__ import annotations

import logging

from coding_client.auth.models import User
from fastapi import APIRouter, Body, Depends, HTTPException

from auth.tenant import get_current_user_with_tenant, require_tenant
from models.common_responses_models import (
    InternalServerErrorResponse,
    NotFoundResponse,
)
from models.intelligent_testing_models import (
    ContainerScopedSmartTestSelectionRequest,
    ContainerScopedTestPatternRequest,
    ContainerScopedTestRecommendationRequest,
    SmartTestSelectionResponse,
    TestImpactAnalysisRequest,
    TestImpactAnalysisResponse,
    TestPatternAnalysisResponse,
    TestPrioritizationRequest,
    TestPrioritizationResponse,
    TestRecommendationResponse,
)
from services.intelligent_testing_service import (
    get_intelligent_testing_service,
)
from shared.container_validation import validate_and_get_container_for_tenant
from shared.responses import (
    internal_error,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/pattern-analysis",
    operation_id="testing_analyze_test_patterns",
    response_model=TestPatternAnalysisResponse,
    summary="Analyze test patterns within a container",
    description="Perform AI-powered pattern analysis on test results scoped to a specific container.",
    responses={
        200: {"description": "Pattern analysis completed successfully"},
        404: {
            "description": "Container not found",
            "model": NotFoundResponse,
        },
        500: {
            "description": "Internal Server Error - Pattern analysis failed",
            "model": InternalServerErrorResponse,
        },
    },
)
async def analyze_test_patterns(
    request: ContainerScopedTestPatternRequest = Body(...),
    user: User = Depends(get_current_user_with_tenant),
    tenant_id: str = Depends(require_tenant),
) -> TestPatternAnalysisResponse:
    """
    Analyze test patterns within a container context.

    This endpoint performs AI-powered pattern detection including:
    - Performance patterns (slow tests, resource usage)
    - Failure patterns (common errors, flaky tests)
    - Timing patterns (execution time trends)

    All analysis is scoped to the specified container.

    Args:
        request: Pattern analysis request with container context
        user: Authenticated user with tenant context

    Returns:
        TestPatternAnalysisResponse: Detected patterns and analysis results

    Raises:
        HTTPException: If container not found or analysis fails
    """
    try:
        # Tenant-aware container validation (admins can access all, users only their own)
        effective_tenant_id = None if user.is_admin() else user.effective_tenant_id
        container = await validate_and_get_container_for_tenant(
            request.container_name, tenant_id=effective_tenant_id
        )

        logger.info(
            f"Starting pattern analysis for container '{request.container_name}' "
            f"with {len(request.test_results)} test results"
        )

        # Get service instance
        service = get_intelligent_testing_service()

        # Prepare analysis options
        analysis_options = {
            "include_performance_patterns": request.include_performance_patterns,
            "include_failure_patterns": request.include_failure_patterns,
            "include_timing_patterns": request.include_timing_patterns,
            "analysis_depth": request.analysis_depth,
        }

        # Perform pattern analysis
        analysis_result = await service.analyze_test_patterns(
            container=container,
            test_results=[r.model_dump() for r in request.test_results],
            analysis_options=analysis_options,
        )

        logger.info(
            f"Pattern analysis completed for container '{request.container_name}': "
            f"{analysis_result['pattern_count']} patterns detected"
        )

        return TestPatternAnalysisResponse(**analysis_result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pattern analysis failed for container '{request.container_name}': {e}")
        raise internal_error(f"Pattern analysis failed: {e}")


@router.post(
    "/smart-test-selection",
    operation_id="testing_smart_test_selection",
    response_model=SmartTestSelectionResponse,
    summary="Intelligently select tests based on code changes",
    description="Use AI to select the most relevant tests based on code changes within a container.",
    responses={
        200: {"description": "Test selection completed successfully"},
        404: {
            "description": "Container not found",
            "model": NotFoundResponse,
        },
        500: {
            "description": "Internal Server Error - Test selection failed",
            "model": InternalServerErrorResponse,
        },
    },
)
async def smart_test_selection(
    request: ContainerScopedSmartTestSelectionRequest = Body(...),
    user: User = Depends(get_current_user_with_tenant),
    tenant_id: str = Depends(require_tenant),
) -> SmartTestSelectionResponse:
    """
    Intelligently select tests based on code changes.

    This endpoint performs smart test selection using:
    - Impact analysis (which tests are affected by changes)
    - Coverage-based selection (maximize coverage with minimal tests)
    - Priority-based selection (run critical tests first)
    - Hybrid strategies combining multiple approaches

    All selection is scoped to the specified container.

    Args:
        request: Test selection request with container and changed files
        user: Authenticated user with tenant context

    Returns:
        SmartTestSelectionResponse: Selected tests and analysis

    Raises:
        HTTPException: If container not found or selection fails
    """
    try:
        # Tenant-aware container validation (admins can access all, users only their own)
        effective_tenant_id = None if user.is_admin() else user.effective_tenant_id
        container = await validate_and_get_container_for_tenant(
            request.container_name, tenant_id=effective_tenant_id
        )

        logger.info(
            f"Starting smart test selection for container '{request.container_name}' "
            f"with {len(request.changed_files)} changed files"
        )

        # Get service instance
        service = get_intelligent_testing_service()

        # Prepare selection options
        selection_options = {
            "selection_strategy": request.selection_strategy,
            "max_tests": request.max_tests,
            "include_related_tests": request.include_related_tests,
        }

        # Perform smart test selection
        selection_result = await service.select_smart_tests(
            container=container,
            changed_files=request.changed_files,
            selection_options=selection_options,
            tenant_id=tenant_id,
        )

        logger.info(
            f"Smart test selection completed for container '{request.container_name}': "
            f"{selection_result['selection_count']} tests selected"
        )

        return SmartTestSelectionResponse(**selection_result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Smart test selection failed for container '{request.container_name}': {e}")
        raise internal_error(f"Smart test selection failed: {e}")


@router.post(
    "/test-impact-analysis",
    operation_id="analyze_test_impact",
    response_model=TestImpactAnalysisResponse,
    summary="Analyze test impact from code changes",
    description="Analyze which tests are impacted by code changes within a container.",
    responses={
        200: {"description": "Impact analysis completed successfully"},
        404: {
            "description": "Container not found",
            "model": NotFoundResponse,
        },
        500: {
            "description": "Internal Server Error - Impact analysis failed",
            "model": InternalServerErrorResponse,
        },
    },
)
async def test_impact_analysis(
    request: TestImpactAnalysisRequest = Body(...),
    user: User = Depends(get_current_user_with_tenant),
    tenant_id: str = Depends(require_tenant),
) -> TestImpactAnalysisResponse:
    """
    Analyze the impact of code changes on tests.

    This endpoint performs impact analysis to identify:
    - Directly impacted tests (tests for changed code)
    - Indirectly impacted tests (tests for dependent code)
    - Impact severity and priority

    All analysis is scoped to the specified container.

    Args:
        request: Impact analysis request with container and changed files
        user: Authenticated user with tenant context

    Returns:
        TestImpactAnalysisResponse: Impact analysis results

    Raises:
        HTTPException: If container not found or analysis fails
    """
    try:
        # Tenant-aware container validation (admins can access all, users only their own)
        effective_tenant_id = None if user.is_admin() else user.effective_tenant_id
        container = await validate_and_get_container_for_tenant(
            request.container_name, tenant_id=effective_tenant_id
        )

        logger.info(
            f"Starting test impact analysis for container '{request.container_name}' "
            f"with {len(request.changed_files)} changed files"
        )

        # Get service instance
        service = get_intelligent_testing_service()

        # Prepare analysis options
        analysis_options = {
            "analysis_depth": request.analysis_depth,
            "include_indirect_impacts": request.include_indirect_impacts,
        }

        # Perform impact analysis
        impact_result = await service.analyze_test_impact(
            container=container,
            changed_files=request.changed_files,
            analysis_options=analysis_options,
            tenant_id=tenant_id,
        )

        logger.info(
            f"Test impact analysis completed for container '{request.container_name}': "
            f"{impact_result['impact_count']} tests impacted"
        )

        return TestImpactAnalysisResponse(**impact_result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Test impact analysis failed for container '{request.container_name}': {e}")
        raise internal_error(f"Test impact analysis failed: {e}")


@router.post(
    "/test-prioritization",
    operation_id="prioritize_tests",
    response_model=TestPrioritizationResponse,
    summary="Prioritize tests based on multiple criteria",
    description="Use AI to prioritize tests based on impact, history, and execution time.",
    responses={
        200: {"description": "Test prioritization completed successfully"},
        404: {
            "description": "Container not found",
            "model": NotFoundResponse,
        },
        500: {
            "description": "Internal Server Error - Prioritization failed",
            "model": InternalServerErrorResponse,
        },
    },
)
async def test_prioritization(
    request: TestPrioritizationRequest = Body(...),
    user: User = Depends(get_current_user_with_tenant),
    tenant_id: str = Depends(require_tenant),
) -> TestPrioritizationResponse:
    """
    Prioritize tests based on multiple criteria.

    This endpoint prioritizes tests using:
    - Impact (how critical is the test)
    - Failure history (tests that fail more often)
    - Execution time (faster tests get higher priority)
    - Code changes (tests related to recent changes)

    All prioritization is scoped to the specified container.

    Args:
        request: Prioritization request with container and test files
        user: Authenticated user with tenant context

    Returns:
        TestPrioritizationResponse: Prioritized tests with scores

    Raises:
        HTTPException: If container not found or prioritization fails
    """
    try:
        # Tenant-aware container validation (admins can access all, users only their own)
        effective_tenant_id = None if user.is_admin() else user.effective_tenant_id
        container = await validate_and_get_container_for_tenant(
            request.container_name, tenant_id=effective_tenant_id
        )

        logger.info(
            f"Starting test prioritization for container '{request.container_name}' "
            f"with {len(request.test_files)} test files"
        )

        # Get service instance
        service = get_intelligent_testing_service()

        # Prepare prioritization options
        prioritization_options = {
            "prioritization_criteria": request.prioritization_criteria,
            "changed_files": request.changed_files or [],
        }

        # Perform test prioritization
        prioritization_result = await service.prioritize_tests(
            container=container,
            test_files=request.test_files,
            prioritization_options=prioritization_options,
        )

        logger.info(
            f"Test prioritization completed for container '{request.container_name}': "
            f"{prioritization_result['total_tests']} tests prioritized"
        )

        return TestPrioritizationResponse(**prioritization_result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Test prioritization failed for container '{request.container_name}': {e}")
        raise internal_error(f"Test prioritization failed: {e}")


@router.post(
    "/advanced-recommendations",
    operation_id="get_advanced_recommendations",
    response_model=TestRecommendationResponse,
    summary="Get advanced test recommendations",
    description="Generate AI-powered test recommendations scoped to a container.",
    responses={
        200: {"description": "Recommendations generated successfully"},
        404: {
            "description": "Container not found",
            "model": NotFoundResponse,
        },
        500: {
            "description": "Internal Server Error - Recommendation generation failed",
            "model": InternalServerErrorResponse,
        },
    },
)
async def advanced_test_recommendations(
    request: ContainerScopedTestRecommendationRequest = Body(...),
    user: User = Depends(get_current_user_with_tenant),
    tenant_id: str = Depends(require_tenant),
) -> TestRecommendationResponse:
    """
    Generate advanced test recommendations for a container.

    This endpoint provides intelligent recommendations for:
    - Missing test coverage
    - Test quality improvements
    - Performance optimizations
    - Best practices implementation

    All recommendations are scoped to the specified container.

    Args:
        request: Recommendation request with container context
        user: Authenticated user with tenant context

    Returns:
        TestRecommendationResponse: AI-generated recommendations

    Raises:
        HTTPException: If container not found or generation fails
    """
    try:
        # Tenant-aware container validation (admins can access all, users only their own)
        effective_tenant_id = None if user.is_admin() else user.effective_tenant_id
        container = await validate_and_get_container_for_tenant(
            request.container_name, tenant_id=effective_tenant_id
        )

        logger.info(
            f"Generating advanced test recommendations for container '{request.container_name}'"
        )

        # Get service instance
        service = get_intelligent_testing_service()

        # Prepare recommendation context
        context = {
            "code_context": request.code_context,
            "language": request.language,
            "test_type": request.test_type,
            "focus_areas": request.focus_areas or [],
            "priority_threshold": request.priority_threshold,
        }

        # Generate recommendations
        recommendation_result = await service.generate_test_recommendations(
            container=container,
            context=context,
        )

        logger.info(
            f"Advanced recommendations generated for container '{request.container_name}': "
            f"{recommendation_result['recommendation_count']} recommendations"
        )

        return TestRecommendationResponse(**recommendation_result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Advanced recommendations failed for container '{request.container_name}': {e}"
        )
        raise internal_error(f"Recommendation generation failed: {e}")
