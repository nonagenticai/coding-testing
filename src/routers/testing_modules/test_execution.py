# src/routers/testing_modules/test_execution.py
"""
Test Execution Module

Core test execution endpoints including:
- Single test execution
- Batch test execution
- Security validation
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import cast
from uuid import uuid7

from coding_client.auth.models import User
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    HTTPException,
    Query,
    status,
)
from pydantic import BaseModel, ConfigDict, Field

from auth.tenant import get_current_user_with_tenant
from models.error_models import (
    Error400Response,
    Error401Response,
    Error403Response,
    Error500Response,
    ValidationErrorResponse,
)
from models.testing_models import (
    TestRequest,
    TestResult,
    TestType,
    create_test_result,
)
from services.code_execution_service import CodeExecutionService
from shared.container_validation import validate_and_get_container_for_tenant
from shared.dependencies import (
    create_json_http_exception,
)

from .testing_utils import (
    _log_batch_execution,
    _log_test_execution,
    get_execution_service,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


@router.post(
    "/execute",
    response_model=TestResult,
    operation_id="testing_execute_test",
    summary="🧪 Execute Code Tests",
    description="""
    **Execute tests with provided code in any of 9 supported programming languages.**

    This endpoint supports comprehensive test execution with:
    - **Languages**: Python, JavaScript, TypeScript, Go, Rust, Java, C#, C++, Node.js
    - **Container Support**: Dynamic creation or use existing provisioned containers
    - **Test Types**: Unit, integration, benchmark, coverage, security, and more
    - **Test Frameworks**: pytest, jest, go test, cargo test, junit, nunit, gtest
    - **Security Levels**: STRICT, STANDARD, RELAXED sandboxing
    - **Resource Limits**: CPU, memory, and time constraints
    - **Quality Integration**: Optional quality analysis after test execution
    - **Caching**: Results cached for identical test runs

    Example usage:
    ```python
    # Python test execution
    {
        "code": "def add(a, b): return a + b",
        "test_code": "def test_add(): assert add(2, 3) == 5",
        "language": "python",
        "test_type": "unit",
        "container_name": "codegraph-myproject",  # optional
        "run_quality_check": true,  # optional
        "timeout": 30
    }
    ```
    """,
    responses={
        200: {
            "description": "Test executed successfully",
            "model": TestResult,
        },
        400: {
            "description": "Invalid test request - malformed request syntax or invalid parameters",
            "model": Error400Response,
        },
        401: {
            "description": "Authentication required - missing or invalid authentication token",
            "model": Error401Response,
        },
        403: {
            "description": "Insufficient permissions - user lacks required permissions for test execution",
            "model": Error403Response,
        },
        422: {
            "description": "Validation error - request data failed validation",
            "model": ValidationErrorResponse,
        },
        500: {
            "description": "Test execution failed - internal server error during test execution",
            "model": Error500Response,
        },
    },
)
async def execute_test(
    request: TestRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user_with_tenant),
    execution_service: CodeExecutionService = Depends(get_execution_service),
):
    """
    Execute a test with the provided code and test code.

    Supports:
    - All 9 programming languages (Python, JavaScript, TypeScript, Go, Rust, Java, C#, C++, Node.js)
    - Dynamic container creation for isolated execution
    - Use existing provisioned containers with container_name
    - Different test types per language (unit, integration, benchmark, etc.)
    - Security validation and sandboxing
    - Resource monitoring and limits
    - Quality analysis integration
    - Result caching for improved performance
    """
    try:
        logger.info(f"🧪 Test execution requested by user {user.preferred_username}")

        # Tenant-aware container validation (admins can access all, users only their own)
        tenant_id = None if user.is_admin() else user.effective_tenant_id
        if request.container_name:
            await validate_and_get_container_for_tenant(request.container_name, tenant_id=tenant_id)
            logger.debug(f"Container '{request.container_name}' validated for test execution")

        # Execute test using code execution service
        result = await execution_service.execute_test(request, tenant_id=tenant_id)

        # Log execution for analytics (background task)
        background_tasks.add_task(
            _log_test_execution,
            user_id=str(user.id if hasattr(user, "id") else user.preferred_username),
            test_result=result,
            request_summary={
                "language": request.language,
                "timeout": request.timeout,
                "security_level": request.security_level,
                "with_hypothesis": request.with_hypothesis,
            },
        )

        logger.info(f"✅ Test execution completed: {result.test_id}, success: {result.success}")
        return result

    except HTTPException:
        # Let HTTPException (including 404 from BOLA check) propagate
        raise
    except ValueError as e:
        logger.warning(f"❌ Test execution validation error: {str(e)}")
        raise create_json_http_exception(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}",
        )
    except Exception as e:
        logger.error(f"❌ Test execution failed: {str(e)}")
        raise create_json_http_exception(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Test execution failed: {str(e)}",
        )


@router.post(
    "/batch",
    response_model=list[TestResult],
    operation_id="testing_execute_batch_tests",
    summary="🚀 Batch Test Execution",
    description="""
    **Execute multiple tests in parallel across different languages and containers.**

    Efficiently run multiple test cases with:
    - Parallel execution with configurable limits
    - Mixed language support in single batch
    - Progress tracking and partial results
    - Automatic resource management

    Example batch request:
    ```json
    [
        {
            "code": "def add(a, b): return a + b",
            "test_code": "def test_add(): assert add(2, 3) == 5",
            "language": "python"
        },
        {
            "code": "function add(a, b) { return a + b; }",
            "test_code": "test('add', () => { expect(add(2, 3)).toBe(5); });",
            "language": "javascript"
        }
    ]
    ```
    """,
    responses={
        200: {
            "description": "Batch execution completed successfully",
            "model": list[TestResult],
        },
        400: {
            "description": "Invalid batch request - malformed request or invalid parameters",
            "model": Error400Response,
        },
        401: {
            "description": "Authentication required - missing or invalid authentication token",
            "model": Error401Response,
        },
        403: {
            "description": "Insufficient permissions - user lacks required permissions for batch execution",
            "model": Error403Response,
        },
        422: {
            "description": "Validation error - request data failed validation",
            "model": ValidationErrorResponse,
        },
        500: {
            "description": "Batch execution failed - internal server error during batch processing",
            "model": Error500Response,
        },
    },
)
async def execute_batch(
    requests: list[TestRequest],
    background_tasks: BackgroundTasks,
    max_parallel: int = Query(4, ge=1, le=10, description="Maximum parallel executions"),
    user: User = Depends(get_current_user_with_tenant),
    execution_service: CodeExecutionService = Depends(get_execution_service),
):
    """
    Execute multiple tests in batch.

    Supports mixed languages and parallel execution with resource limits.
    """
    try:
        logger.info(f"🚀 Batch execution requested by user {user.preferred_username}")
        logger.info(f"Batch size: {len(requests)}, Max parallel: {max_parallel}")

        # Tenant-aware container validation (admins can access all, users only their own)
        tenant_id = None if user.is_admin() else user.effective_tenant_id
        container_names = {req.container_name for req in requests if req.container_name}
        for container_name in container_names:
            await validate_and_get_container_for_tenant(container_name, tenant_id=tenant_id)
            logger.debug(f"Container '{container_name}' validated for batch execution")

        # Execute batch using asyncio
        results = []

        # Process in chunks to respect max_parallel
        for i in range(0, len(requests), max_parallel):
            chunk = requests[i : i + max_parallel]
            chunk_tasks = [
                execution_service.execute_test(req, tenant_id=tenant_id) for req in chunk
            ]
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)

            # Handle exceptions
            for j, result in enumerate(chunk_results):
                if isinstance(result, Exception):
                    # Create failed result
                    test_type_value = None
                    if chunk[j].test_type is not None:
                        test_type_value = cast(TestType, chunk[j].test_type).value

                    results.append(
                        create_test_result(
                            test_id=str(uuid7()),
                            success=False,
                            exit_code=1,
                            stdout="",
                            stderr=str(result),
                            execution_time=0.0,
                            created_at=datetime.now(UTC),
                            language=chunk[j].language,
                            security_level=chunk[j].security_level,
                            error=str(result),
                            test_type=test_type_value,
                            container_name=chunk[j].container_name or "unknown",
                        )
                    )
                else:
                    # Type assertion: if not Exception, must be TestResult
                    assert isinstance(result, TestResult)
                    results.append(result)

        # Log batch completion
        successful = sum(1 for r in results if r.success)
        logger.info(f"✅ Batch execution completed: {successful}/{len(results)} successful")

        # Background metrics tracking
        background_tasks.add_task(
            _log_batch_execution,
            user_id=user.preferred_username,
            batch_size=len(requests),
            successful=successful,
            languages=[req.language for req in requests],
        )

        return results

    except HTTPException:
        # Let HTTPException (including 404 from BOLA check) propagate
        raise
    except Exception as e:
        logger.error(f"❌ Batch execution failed: {str(e)}")
        raise create_json_http_exception(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch execution failed: {str(e)}",
        )


class SecurityValidateBody(BaseModel):
    """JSON body for POST /api/tests/validate-security."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., min_length=1, max_length=50000, description="Code to validate")
    language: str = Field(
        "python", pattern="^(python|javascript|typescript)$", description="Programming language"
    )


@router.post(
    "/validate-security",
    operation_id="testing_validate_code_security",
    summary="🔒 Validate Code Security",
    description="""
    **Perform security validation on code before execution.**

    Analyzes code for potentially dangerous patterns and security risks:

    **Detection Capabilities**:
    - File system operations (read/write/delete)
    - Network access and HTTP requests
    - Process execution and shell commands
    - Code evaluation (eval, exec)
    - Import of dangerous modules
    - SQL injection patterns
    - Path traversal attempts

    **Risk Scoring**:
    - 0-25: Low risk - safe for execution
    - 26-50: Medium risk - review recommended
    - 51-75: High risk - manual approval needed
    - 76-100: Critical risk - execution blocked
    """,
    responses={
        200: {
            "description": "Security validation completed successfully",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "is_safe": {"type": ["boolean", "null"]},
                            "computed": {"type": "boolean"},
                            "reason": {"type": ["string", "null"]},
                            "violations": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "warnings": {"type": "array", "items": {"type": "string"}},
                            "risk_level": {"type": "number"},
                            "language": {"type": "string"},
                            "code_length": {"type": "integer"},
                        },
                    }
                }
            },
        },
        400: {
            "description": "Bad request - invalid code or language parameter",
            "model": Error400Response,
        },
        401: {
            "description": "Authentication required - missing or invalid authentication token",
            "model": Error401Response,
        },
        403: {
            "description": "Insufficient permissions - user lacks required permissions",
            "model": Error403Response,
        },
        422: {
            "description": "Validation error - invalid parameters",
            "model": ValidationErrorResponse,
        },
        500: {
            "description": "Security validation failed - internal server error",
            "model": Error500Response,
        },
    },
)
async def validate_code_security(
    code: str | None = Query(
        None,
        min_length=1,
        max_length=50000,
        description="Code to validate (prefer the JSON body: code in a query string is "
        "rejected by the request filter whenever it contains SQL or shell text)",
    ),
    language: str | None = Query(
        None,
        pattern="^(python|javascript|typescript)$",
        description="Programming language",
    ),
    body: SecurityValidateBody | None = Body(None),
    user: User = Depends(get_current_user_with_tenant),
    execution_service: CodeExecutionService = Depends(get_execution_service),
):
    """
    Validate code for security issues before execution.

    Performs static analysis to detect potentially dangerous patterns
    like file system access, network operations, or code injection.
    """
    try:
        logger.info(f"🔒 Security validation requested by user {user.preferred_username}")

        if isinstance(body, SecurityValidateBody):
            code, language = body.code, body.language
        if not code:
            raise HTTPException(status_code=400, detail="code is required (JSON body or query)")
        language = language or "python"

        # Use the service's security validation
        try:
            security_result = await execution_service._validate_code_security(code, language)

            # is_safe is True/False only when coding-security gave a verdict; None
            # (computed: false, with a reason) is "not checked", never "safe".
            violations = security_result.get("violations", [])
            result = {
                "is_safe": security_result.get("is_safe"),
                "computed": security_result.get("computed", False),
                "reason": security_result.get("reason"),
                "violations": [
                    (
                        f"{v.get('type', 'violation')} (line {v.get('line_number')}): "
                        f"{v.get('description', '')}"
                    )
                    if isinstance(v, dict)
                    else str(v)
                    for v in violations
                ],
                "warnings": [
                    v.get("description", str(v)) if isinstance(v, dict) else str(v)
                    for v in violations
                    if isinstance(v, dict) and v.get("severity", "") in ["low", "medium"]
                ],
                "risk_level": security_result.get("risk_score"),
                "language": language,
                "code_length": len(code),
            }
        except ValueError as e:
            result = {
                "is_safe": False,
                "violations": [str(e)],
                "warnings": [],
                "risk_level": 100.0,  # High risk score for validation errors
                "language": language,
                "code_length": len(code),
            }

        violations = result.get("violations", [])
        violations_count = len(violations) if isinstance(violations, (list, tuple)) else 0
        logger.info(
            f"🔒 Security validation completed: safe={result['is_safe']}, violations={violations_count}"
        )
        return result

    except Exception as e:
        logger.error(f"❌ Security validation failed: {str(e)}")
        raise create_json_http_exception(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Security validation failed: {str(e)}",
        )


__all__ = ["router"]
