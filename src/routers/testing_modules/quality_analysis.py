# src/routers/testing_modules/quality_analysis.py
"""
Quality Analysis Module

Endpoints for:
- Code quality analysis
- Container quality scanning
"""

import logging

from coding_client.auth.models import User
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    status,
)

from auth.tenant import get_current_user_with_tenant
from models.error_models import (
    Error400Response,
    Error401Response,
    Error403Response,
    Error404Response,
    Error500Response,
    ValidationErrorResponse,
)
from models.testing_models import (
    QualityRequest,
    QualityResult,
    QualityScanRequest,
)
from services.code_execution_service import CodeExecutionService
from shared.container_validation import validate_and_get_container_for_tenant
from shared.dependencies import (
    create_json_http_exception,
)

from .testing_utils import _log_quality_analysis, get_execution_service

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


@router.post(
    "/quality-check",
    response_model=QualityResult,
    operation_id="run_quality_check",
    summary="🔍 Code Quality Analysis",
    description="""
    **Run comprehensive code quality analysis with multiple tools on container-scoped code.**

    Perform static analysis, style checking, and security scanning on your code:

    **Python Tools**:
    - `ruff` - Linting, import sorting, unused imports/variables
    - `ruff-format` - Code formatting
    - `ty` - Type checking
    - `bandit` - Security vulnerability scanning
    - `radon` - Cyclomatic complexity

    **JavaScript/TypeScript Tools**:
    - `eslint` - Linting and code quality
    - `prettier` - Code formatting

    **Features**:
    - Container-scoped analysis for isolated repos
    - Parallel execution of multiple tools
    - Consolidated quality score (0-100)
    - Categorized issues by severity
    - Automated resolve suggestions
    - Performance metrics per tool

    **Note**: Requires valid `container_name` to ensure proper code isolation.
    """,
    responses={
        200: {
            "description": "Quality analysis completed successfully",
            "model": QualityResult,
        },
        400: {
            "description": "Invalid analysis request - malformed request or unsupported language/tools",
            "model": Error400Response,
        },
        401: {
            "description": "Authentication required - missing or invalid authentication token",
            "model": Error401Response,
        },
        403: {
            "description": "Insufficient permissions - user lacks required permissions for quality analysis",
            "model": Error403Response,
        },
        422: {
            "description": "Validation error - request data failed validation",
            "model": ValidationErrorResponse,
        },
        500: {
            "description": "Quality analysis failed - internal server error during analysis",
            "model": Error500Response,
        },
    },
)
async def run_quality_check(
    request: QualityRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user_with_tenant),
    execution_service: CodeExecutionService = Depends(get_execution_service),
):
    """
    Run code quality analysis with specified tools.

    Available tools by language:
    - Python: ruff, ruff-format, ty, bandit, radon
    - JavaScript/TypeScript: eslint, prettier

    Features:
    - Multi-tool parallel analysis
    - Automated resolve suggestions
    - Quality scoring
    - Security scanning
    """
    try:
        logger.info(f"🔍 Quality analysis requested by user {user.preferred_username}")
        logger.info(f"Tools: {request.tools}, Language: {request.language}")

        # Tenant-aware container validation (admins can access all, users only their own)
        tenant_id = None if user.is_admin() else user.effective_tenant_id
        await validate_and_get_container_for_tenant(request.container_name, tenant_id=tenant_id)
        logger.debug(f"Container '{request.container_name}' validated for quality check")

        # Run quality analysis (forward the caller's tenant to coding-analysis so its
        # own container-scoping resolves under the right tenant).
        result = await execution_service.analyze_quality(
            request, tenant_id=user.effective_tenant_id
        )

        # Apply resolves if requested
        if request.fix_issues and result.issues:
            # Check if any issues have suggested resolves
            fixable_issues = [issue for issue in result.issues if issue.suggested_fix]
            if fixable_issues:
                logger.info(f"🔧 Attempting to apply {len(fixable_issues)} suggested resolves")
                # This would be implemented in the service in the future
            else:
                logger.info("🔧 No fixable issues found (no suggested resolves available)")

        # Log analysis for trends (background task)
        background_tasks.add_task(
            _log_quality_analysis,
            user_id=str(user.id if hasattr(user, "id") else user.preferred_username),
            analysis_result=result,
        )

        logger.info(
            f"✅ Quality analysis completed: {result.analysis_id}, score: {result.overall_score}"
        )
        return result

    except HTTPException:
        # Let HTTPException (including 404 from BOLA check) propagate
        raise
    except ValueError as e:
        logger.warning(f"❌ Quality analysis validation error: {str(e)}")
        raise create_json_http_exception(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}",
        )
    except Exception as e:
        logger.error(f"❌ Quality analysis failed: {str(e)}")
        raise create_json_http_exception(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quality analysis failed: {str(e)}",
        )


@router.post(
    "/quality-scan",
    response_model=QualityResult,
    operation_id="run_quality_scan",
    summary="🐳 Container Quality Scan",
    description="""
    **Run comprehensive quality analysis on an entire container's codebase.**

    Scan all supported files within a container using multiple quality tools:

    **Scanning Process**:
    1. Identifies all supported files in the specified path
    2. Runs configured quality tools in parallel
    3. Aggregates results across all files
    4. Generates consolidated quality report

    **Use Cases**:
    - Pre-deployment quality gates
    - Continuous integration checks
    - Codebase health monitoring
    - Technical debt assessment

    **Performance**:
    - Parallel tool execution
    - Intelligent file filtering
    - Results caching for efficiency
    - Progress tracking for large codebases
    """,
    responses={
        200: {
            "description": "Container scan completed successfully",
            "model": QualityResult,
        },
        400: {
            "description": "Invalid scan request - malformed request or container not found",
            "model": Error400Response,
        },
        401: {
            "description": "Authentication required - missing or invalid authentication token",
            "model": Error401Response,
        },
        403: {
            "description": "Insufficient permissions - user lacks required permissions for container scanning",
            "model": Error403Response,
        },
        404: {
            "description": "Container not found - specified container does not exist",
            "model": Error404Response,
        },
        422: {
            "description": "Validation error - request data failed validation",
            "model": ValidationErrorResponse,
        },
        500: {
            "description": "Container scan failed - internal server error during scanning",
            "model": Error500Response,
        },
    },
)
async def run_quality_scan(
    request: QualityScanRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user_with_tenant),
    execution_service: CodeExecutionService = Depends(get_execution_service),
):
    """
    Run code quality scan on container.

    Available tools by language:
    - Python: ruff, ruff-format, ty, bandit, radon
    - JavaScript/TypeScript: eslint, prettier

    Features:
    - Multi-tool parallel analysis
    - Automated resolve suggestions
    - Quality scoring
    - Security scanning
    """
    try:
        logger.info(f"🔍 Container quality scan requested by user {user.preferred_username}")
        logger.info(f"Container: {request.container_name}, Scan path: {request.scan_path}")

        # Tenant-aware container validation (admins can access all, users only their own)
        tenant_id = None if user.is_admin() else user.effective_tenant_id
        await validate_and_get_container_for_tenant(request.container_name, tenant_id=tenant_id)
        logger.debug(f"Container '{request.container_name}' validated for quality scan")

        # Execute container-wide quality scan (forward the caller's tenant to
        # coding-analysis so its own container-scoping resolves under the right tenant).
        result = await execution_service.analyze_container_quality(
            request, tenant_id=user.effective_tenant_id
        )

        # Log scan completion
        logger.info(
            f"✅ Container quality scan completed: {result.analysis_id}, "
            f"score: {result.overall_score}, total issues: {result.total_issues}"
        )

        # Apply resolves if requested
        if request.fix_issues and result.issues:
            fixable_issues = [issue for issue in result.issues if issue.suggested_fix]
            if fixable_issues:
                logger.info(f"🔧 Attempting to apply {len(fixable_issues)} suggested resolves")
                # Future: Apply resolves to container files
            else:
                logger.info("🔧 No fixable issues found (no suggested resolves available)")

        # Log analysis for trends (background task)
        background_tasks.add_task(
            _log_quality_analysis,
            user_id=str(user.id if hasattr(user, "id") else user.preferred_username),
            analysis_result=result,
        )

        return result

    except HTTPException:
        # Let HTTPException (including 404 from BOLA check) propagate
        raise
    except ValueError as e:
        logger.warning(f"❌ Container quality scan validation error: {str(e)}")
        raise create_json_http_exception(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}",
        )
    except Exception as e:
        logger.error(f"❌ Container quality scan failed: {str(e)}")
        raise create_json_http_exception(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Container quality scan failed: {str(e)}",
        )


__all__ = ["router"]
