"""
Test execution coordination module for intelligent testing.
Handles test execution orchestration, batch processing, and resource management.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from coding_client.auth.models import User
from fastapi import APIRouter, Depends, HTTPException, status

from auth.tenant import get_current_user_with_tenant, require_tenant
from models.common_responses_models import (
    BadRequestResponse,
    InternalServerErrorResponse,
    LegacyResponseStatus,
    ServiceUnavailableResponse,
    SimpleValidationErrorResponse,
    TimeoutResponse,
)
from models.testing_models import TestRequest
from services.test_execution.intelligent_test_integration import (
    IntelligentTestExecutionService,
    TestExecutionError,
    TestExecutionResourceError,
    TestExecutionRetryError,
    TestExecutionTimeoutError,
)
from services.test_execution.intelligent_test_orchestrator import (
    IntelligentTestExecutionRequest,
    IntelligentTestExecutionResult,
    TestType,
)
from services.test_execution.test_orchestrator import TestLanguage
from shared.container_validation import validate_and_get_container_for_tenant
from shared.responses import (
    error_response,
    internal_error,
)

from .shared import (
    BatchTestExecutionResponse,
    BatchTestRequest,
    IntelligentTestRequest,
    TestExecutionResponse,
    get_intelligent_test_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/execute",
    response_model=TestExecutionResponse,
    operation_id="execute_intelligent_test",
    responses={
        200: {"description": "Test executed successfully"},
        400: {
            "description": "Bad Request - Invalid test request or unsupported language",
            "model": BadRequestResponse,
        },
        408: {
            "description": "Request Timeout - Test execution exceeded timeout limit",
            "model": TimeoutResponse,
        },
        422: {
            "description": "Validation Error - Invalid request body",
            "model": SimpleValidationErrorResponse,
        },
        500: {
            "description": "Internal Server Error - Test execution failed",
            "model": InternalServerErrorResponse,
        },
        503: {
            "description": "Service Unavailable - Test execution failed after retries",
            "model": ServiceUnavailableResponse,
        },
        507: {"description": "Insufficient Storage - Resource limit exceeded"},
    },
)
async def execute_test(
    request: IntelligentTestRequest,
    user: User = Depends(get_current_user_with_tenant),
    service: IntelligentTestExecutionService = Depends(get_intelligent_test_service),
    tenant_id: str = Depends(require_tenant),
) -> TestExecutionResponse:
    """Execute a test with intelligent AI-powered analysis and optimization.

    This endpoint provides advanced test execution capabilities including:
    - AI-powered pattern analysis
    - Intelligent scheduling optimization
    - Failure analysis and recommendations
    - Performance monitoring and insights

    Args:
        request (IntelligentTestRequest): The request parameter.
        user: Authenticated user with tenant context.
        service (IntelligentTestExecutionService, optional): The service parameter. Defaults to Depends(get_intelligent_test_service).

    Returns:
        TestExecutionResponse: The testexecutionresponse result.

    Raises:
        HTTPException: Raised under certain conditions.

    Example:
        >>> await execute_test("request_value", "service_value")"""
    try:
        execution_id = f"exec_{datetime.now(UTC).timestamp()}"

        logger.info(
            f"Starting intelligent test execution {execution_id} for user {user.preferred_username}"
        )

        # Tenant-aware container validation (admins can access all, users only their own)
        if request.container_name:
            effective_tenant_id = None if user.is_admin() else user.effective_tenant_id
            await validate_and_get_container_for_tenant(
                request.container_name, tenant_id=effective_tenant_id
            )

        # Convert request to internal format
        intelligent_request = IntelligentTestExecutionRequest(
            language=getattr(TestLanguage, request.language.upper()),
            test_type=(
                getattr(TestType, request.test_type.upper()) if request.test_type else TestType.UNIT
            ),
            source_code=request.code,
            container_name=request.container_name,
            test_files={"test_main.py": request.test_code} if request.test_code else {},
            dependencies=request.dependencies,
            timeout_seconds=request.timeout,
            resource_limits=request.resource_limits,
            security_level=request.security_level,
            project_name=request.project_name or execution_id,
            enable_ai_optimization=request.enable_ai_optimization,
            enable_pattern_analysis=request.enable_pattern_analysis,
            enable_failure_analysis=request.enable_failure_analysis,
            enable_scheduling_optimization=request.enable_scheduling_optimization,
            priority_level=request.priority_level,
            expected_execution_time_ms=request.expected_execution_time_ms,
            resource_weight=request.resource_weight,
        )

        # Execute test
        result = await service.execute_test(intelligent_request)

        logger.info(f"Test execution {execution_id} completed successfully")

        return TestExecutionResponse(
            status=LegacyResponseStatus.SUCCESS,
            message="Test executed successfully",
            result=result,
            execution_id=execution_id,
        )

    except TestExecutionTimeoutError as e:
        logger.error(f"Test execution timeout: {e}")
        raise error_response(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            message=f"Test execution timeout: {e}",
            error_code="TIMEOUT",
            details={"timeout_seconds": e.timeout_seconds},
        )

    except TestExecutionResourceError as e:
        logger.error(f"Resource error: {e}")
        raise error_response(
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
            message=f"Resource limit exceeded: {e}",
            error_code="RESOURCE_LIMIT_EXCEEDED",
            details={
                "resource_type": e.resource_type,
                "resource_limit": e.resource_limit,
            },
        )

    except TestExecutionRetryError as e:
        logger.error(f"Retry failure: {e}")
        raise error_response(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            message=f"Test execution failed after retries: {e}",
            error_code="SERVICE_UNAVAILABLE",
            details={"attempts": e.attempts, "last_error": str(e.last_error)},
        )

    except TestExecutionError as e:
        logger.error(f"Test execution error: {e}")
        raise internal_error(
            f"Test execution failed: {e}",
            details={"error_type": e.error_type, "execution_details": e.details},
        )

    except HTTPException:
        # Let HTTPException (including 404 from BOLA check) propagate
        raise
    except Exception as e:
        logger.error(f"Unexpected error during test execution: {e}")
        raise internal_error(f"Unexpected error during test execution: {e}")


@router.post(
    "/execute/batch",
    response_model=BatchTestExecutionResponse,
    operation_id="execute_intelligent_batch_tests",
    responses={
        200: {"description": "Batch test execution completed successfully"},
        400: {
            "description": "Bad Request - Invalid batch test request",
            "model": BadRequestResponse,
        },
        422: {
            "description": "Validation Error - Invalid request body",
            "model": SimpleValidationErrorResponse,
        },
        500: {
            "description": "Internal Server Error - Batch test execution failed",
            "model": InternalServerErrorResponse,
        },
    },
)
async def execute_batch_tests(
    request: BatchTestRequest,
    user: User = Depends(get_current_user_with_tenant),
    service: IntelligentTestExecutionService = Depends(get_intelligent_test_service),
    tenant_id: str = Depends(require_tenant),
) -> BatchTestExecutionResponse:
    """Execute multiple tests in batch with intelligent optimization.

    This endpoint provides batch test execution with:
    - AI-powered execution order optimization
    - Parallel execution with resource management
    - Comprehensive batch analysis and reporting

    Args:
        request (BatchTestRequest): The request parameter.
        user: Authenticated user with tenant context.
        service (IntelligentTestExecutionService, optional): The service parameter. Defaults to Depends(get_intelligent_test_service).

    Returns:
        BatchTestExecutionResponse: The batchtestexecutionresponse result.

    Raises:
        HTTPException: Raised under certain conditions.

    Example:
        >>> await execute_batch_tests("request_value", "service_value")"""
    try:
        execution_id = f"batch_{datetime.now(UTC).timestamp()}"

        logger.info(
            f"Starting batch test execution {execution_id} with {len(request.tests)} tests for user {user.preferred_username}"
        )

        # Tenant-aware container validation (admins can access all, users only their own)
        effective_tenant_id = None if user.is_admin() else user.effective_tenant_id
        container_names = {test.container_name for test in request.tests if test.container_name}
        for container_name in container_names:
            await validate_and_get_container_for_tenant(
                container_name, tenant_id=effective_tenant_id
            )

        # Convert requests to internal format
        intelligent_requests: list[IntelligentTestExecutionRequest] = []
        for i, test_req in enumerate(request.tests):
            intelligent_request = IntelligentTestExecutionRequest(
                language=getattr(TestLanguage, test_req.language.upper()),
                test_type=(
                    getattr(TestType, test_req.test_type.upper())
                    if test_req.test_type
                    else TestType.UNIT
                ),
                source_code=test_req.code,
                container_name=test_req.container_name,
                test_files=({"test_main.py": test_req.test_code} if test_req.test_code else {}),
                dependencies=test_req.dependencies,
                timeout_seconds=test_req.timeout,
                resource_limits=test_req.resource_limits,
                security_level=test_req.security_level,
                project_name=test_req.project_name or f"{execution_id}_test_{i}",
                enable_ai_optimization=test_req.enable_ai_optimization,
                enable_pattern_analysis=test_req.enable_pattern_analysis,
                enable_failure_analysis=test_req.enable_failure_analysis,
                enable_scheduling_optimization=test_req.enable_scheduling_optimization,
                priority_level=test_req.priority_level,
                expected_execution_time_ms=test_req.expected_execution_time_ms,
                resource_weight=test_req.resource_weight,
            )
            intelligent_requests.append(intelligent_request)

        # Execute batch (cast list to expected union type)
        from typing import cast

        results = await service.execute_batch(
            cast(
                list[TestRequest | IntelligentTestExecutionRequest],
                intelligent_requests,
            ),
            request.enable_retry,
        )

        # Generate summary
        summary = await _generate_batch_summary(results)

        logger.info(f"Batch test execution {execution_id} completed: {summary}")

        return BatchTestExecutionResponse(
            status=LegacyResponseStatus.SUCCESS,
            message="Batch test execution completed",
            results=results,
            execution_id=execution_id,
            summary=summary,
        )

    except HTTPException:
        # Let HTTPException (including 404 from BOLA check) propagate
        raise
    except Exception as e:
        logger.error(f"Batch test execution failed: {e}")
        raise internal_error(f"Batch test execution failed: {e}")


async def _generate_batch_summary(
    results: list[IntelligentTestExecutionResult],
) -> dict[str, Any]:
    """Generate summary statistics for batch execution results.

    Args:
        results: List of test execution results

    Returns:
        Dictionary containing summary statistics
    """
    if not results:
        return {
            "total_tests": 0,
            "successful_tests": 0,
            "failed_tests": 0,
            "total_execution_time": 0,
            "average_execution_time": 0,
            "ai_optimizations_applied": 0,
            "patterns_detected": 0,
            "recommendations_generated": 0,
        }

    return {
        "total_tests": len(results),
        "successful_tests": sum(1 for r in results if r.success),
        "failed_tests": sum(1 for r in results if not r.success),
        "total_execution_time": sum(r.execution_time_ms for r in results),
        "average_execution_time": sum(r.execution_time_ms for r in results) / len(results),
        "ai_optimizations_applied": sum(1 for r in results if r.optimization_applied),
        "patterns_detected": sum(len(r.pattern_matches) for r in results),
        "recommendations_generated": sum(len(r.recommendations) for r in results),
    }


async def optimize_execution_order(
    requests: list[IntelligentTestExecutionRequest],
) -> list[IntelligentTestExecutionRequest]:
    """Optimize the execution order of test requests for better performance.

    Args:
        requests: List of test execution requests

    Returns:
        Optimized list of test execution requests
    """
    try:
        logger.info(f"Optimizing execution order for {len(requests)} tests")

        # Sort by priority (higher first), then by expected execution time (shorter first)
        optimized_requests = sorted(
            requests,
            key=lambda r: (-r.priority_level, r.expected_execution_time_ms or 0),
        )

        return optimized_requests

    except Exception as e:
        logger.error(f"Failed to optimize execution order: {e}")
        return requests


async def manage_concurrent_execution(
    service: IntelligentTestExecutionService,
    requests: list[IntelligentTestExecutionRequest],
    max_concurrent: int | None = None,
) -> list[IntelligentTestExecutionResult]:
    """Manage concurrent execution of test requests with resource limits.

    Args:
        service: The intelligent test execution service
        requests: List of test execution requests
        max_concurrent: Maximum number of concurrent executions

    Returns:
        List of test execution results
    """
    try:
        if not max_concurrent:
            max_concurrent = getattr(service.settings, "max_concurrent_tests", 5)

        logger.info(
            f"Managing concurrent execution of {len(requests)} tests with limit {max_concurrent}"
        )

        # Create semaphore to limit concurrent executions
        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_with_semaphore(
            request: IntelligentTestExecutionRequest,
        ) -> IntelligentTestExecutionResult:
            async with semaphore:
                return await service.execute_test(request)

        # Execute all requests concurrently with semaphore
        tasks = [execute_with_semaphore(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and log them
        valid_results: list[IntelligentTestExecutionResult] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.error(f"Test execution {i} failed with exception: {result}")
            elif isinstance(result, IntelligentTestExecutionResult):
                valid_results.append(result)

        return valid_results

    except Exception as e:
        logger.error(f"Failed to manage concurrent execution: {e}")
        raise


async def handle_execution_retry(
    service: IntelligentTestExecutionService,
    request: IntelligentTestExecutionRequest,
    max_retries: int = 3,
) -> IntelligentTestExecutionResult:
    """Handle retry logic for failed test executions.

    Args:
        service: The intelligent test execution service
        request: Test execution request
        max_retries: Maximum number of retry attempts

    Returns:
        Test execution result

    Raises:
        TestExecutionRetryError: If all retry attempts fail
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logger.info(f"Retrying test execution, attempt {attempt}/{max_retries}")
                # Add exponential backoff
                await asyncio.sleep(2**attempt)

            result = await service.execute_test(request)

            if result.success:
                return result
            else:
                last_error = Exception(f"Test execution failed: {result.stderr}")

        except Exception as e:
            last_error = e
            logger.warning(f"Test execution attempt {attempt} failed: {e}")

    # All attempts failed
    raise TestExecutionRetryError(
        message=f"Test execution failed after {max_retries} retries",
        attempts=max_retries + 1,
        last_error=last_error or Exception("Unknown error"),
    )
