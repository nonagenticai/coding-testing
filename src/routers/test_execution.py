"""
V2 Multi-Language Test Execution Endpoints

Provides API endpoints for executing tests in multiple programming languages
using isolated Docker containers with comprehensive security and monitoring.
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid7

from coding_client.auth.models import User
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from auth.tenant import require_tenant
from config.test_framework_config import TestConfig, TestFrameworkSecurityConfig
from models.error_models import (
    Error400Response,
    Error401Response,
    Error403Response,
    Error404Response,
    Error500Response,
    Error503Response,
    ValidationErrorResponse,
)
from services.stubs import get_docker_service, get_metrics_service
from services.test_execution.test_execution_container_registry import (
    TestExecutionContainerRegistry,
)
from services.test_execution.test_orchestrator import (
    TestExecutionRequest,
    TestExecutionResult,
    TestOrchestrator,
    create_test_orchestrator,
)
from shared.container_validation import validate_and_get_container_for_tenant
from shared.dependencies import (
    requires_read_permission,
)
from shared.responses import (
    internal_error,
    not_found,
    success_response,
)

logger = logging.getLogger(__name__)
router = APIRouter()  # Tags are set at main API level as "V2 🚀 Test Execution"
_test_orchestrator: TestOrchestrator | None = None
_container_registry: TestExecutionContainerRegistry | None = None


async def get_test_orchestrator() -> TestOrchestrator:
    """Get or create test orchestrator instance."""
    global _test_orchestrator
    if _test_orchestrator is None:
        docker_service = get_docker_service()
        TestFrameworkSecurityConfig(
            max_execution_time=30,
            max_memory_mb=1000,
            max_file_size_mb=10,
            max_code_length=10000,
        )
        config = TestConfig(timeout=30, parallel=True, max_workers=4)
        _test_orchestrator = create_test_orchestrator(docker_service, config)
        logger.info("Test orchestrator initialized")
    return _test_orchestrator


async def get_container_registry() -> TestExecutionContainerRegistry:
    """Get or create container registry instance."""
    global _container_registry
    if _container_registry is None:
        _container_registry = TestExecutionContainerRegistry()
        logger.info("Container registry initialized")
    return _container_registry


@router.get(
    "/languages",
    operation_id="list_execution_languages",
    summary="Get supported languages",
    responses={
        200: {
            "description": "Successfully retrieved supported languages",
            "content": {
                "application/json": {
                    "example": {
                        "supported_languages": [
                            "python",
                            "javascript",
                            "typescript",
                            "go",
                            "rust",
                            "java",
                            "csharp",
                            "cpp",
                        ],
                        "available_containers": [
                            {
                                "language": "python",
                                "image": "python:3.11-slim",
                                "version": "3.11",
                                "available": True,
                            }
                        ],
                        "total_languages": 8,
                    }
                }
            },
        },
        401: {"description": "Authentication required", "model": Error401Response},
        403: {"description": "Insufficient permissions", "model": Error403Response},
        404: {
            "description": "No supported languages available",
            "model": Error404Response,
        },
        500: {"description": "Internal server error", "model": Error500Response},
    },
)
async def get_supported_languages(
    current_user: User = Depends(requires_read_permission),
    tenant_id: str = Depends(require_tenant),
) -> dict[str, Any]:
    """
    Get List[Any] of supported programming languages for test execution.

    Returns:
        Dictionary containing supported and available languages
    """
    try:
        orchestrator = await get_test_orchestrator()
        registry = await get_container_registry()
        supported_languages = await orchestrator.get_supported_languages()

        if not supported_languages:
            raise not_found("Supported languages", "any")

        available_containers = []
        for lang in supported_languages:
            spec = registry.get_container_spec(lang)
            if spec:
                available_containers.append(
                    {
                        "language": lang,
                        "image": spec.image,
                        "version": spec.version,
                        "available": True,
                    }
                )
        return {
            "supported_languages": supported_languages,
            "available_containers": available_containers,
            "total_languages": len(supported_languages),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get supported languages: {str(e)}")
        raise internal_error(f"Failed to get supported languages: {str(e)}")


@router.get(
    "/languages/{language}/test-types",
    operation_id="get_execution_language_test_types",
    summary="Get test types for language",
    responses={
        200: {
            "description": "Successfully retrieved test types",
            "content": {
                "application/json": {
                    "example": {
                        "language": "python",
                        "test_types": ["unit", "integration", "coverage", "security"],
                        "count": 4,
                    }
                }
            },
        },
        401: {"description": "Authentication required", "model": Error401Response},
        403: {"description": "Insufficient permissions", "model": Error403Response},
        404: {"description": "Language not supported", "model": Error404Response},
        500: {"description": "Internal server error", "model": Error500Response},
    },
)
async def get_test_types_for_language(
    language: str,
    current_user: User = Depends(requires_read_permission),
    tenant_id: str = Depends(require_tenant),
) -> dict[str, Any]:
    """
    Get supported test types for a specific language.

    Args:
        language: Programming language name

    Returns:
        Dictionary containing test types for the language
    """
    try:
        orchestrator = await get_test_orchestrator()
        supported = await orchestrator.get_supported_languages()
        if language not in supported:
            raise not_found("Language", language)
        test_types = await orchestrator.get_supported_test_types(language)
        return success_response(
            data={
                "language": language,
                "test_types": test_types,
                "count": len(test_types),
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get test types for {language}: {str(e)}")
        # Return default test types instead of error
        return success_response(
            data={
                "language": language,
                "test_types": ["unit", "integration", "coverage"],
                "count": 3,
            }
        )


@router.post(
    "/batch",
    operation_id="execute_containerized_batch_tests",
    response_model=list[TestExecutionResult],
    summary="Batch test execution",
    responses={
        200: {
            "description": "Batch tests executed successfully",
            "model": list[TestExecutionResult],
        },
        400: {"description": "Invalid request parameters", "model": Error400Response},
        401: {"description": "Authentication required", "model": Error401Response},
        403: {"description": "Insufficient permissions", "model": Error403Response},
        422: {
            "description": "Request validation failed",
            "model": ValidationErrorResponse,
        },
        500: {"description": "Batch execution failed", "model": Error500Response},
        503: {
            "description": "Service temporarily unavailable",
            "model": Error503Response,
        },
    },
)
async def execute_batch_tests(
    requests: list[TestExecutionRequest],
    background_tasks: BackgroundTasks,
    max_parallel: int = Query(4, ge=1, le=10, description="Maximum parallel executions"),
    current_user: User = Depends(requires_read_permission),
    tenant_id: str = Depends(require_tenant),
) -> list[TestExecutionResult]:
    """
    Execute multiple tests in parallel.

    Efficiently processes multiple test requests with configurable parallelism.

    Args:
        requests: List of test execution requests
        max_parallel: Maximum number of parallel executions

    Returns:
        List of test execution results
    """
    try:
        # Validate all containers exist before processing
        for req in requests:
            if req.container_name:
                await validate_and_get_container_for_tenant(req.container_name, tenant_id)

        logger.info(
            f"Batch execution requested by {current_user.preferred_username}, size: {len(requests)}"
        )
        orchestrator = await get_test_orchestrator()
        results = []
        for i in range(0, len(requests), max_parallel):
            chunk = requests[i : i + max_parallel]
            chunk_tasks = [orchestrator.execute_tests(req, tenant_id=tenant_id) for req in chunk]
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
            for j, result in enumerate(chunk_results):
                if isinstance(result, Exception):
                    error_result = TestExecutionResult(
                        test_id=str(uuid7()),
                        success=False,
                        exit_code=1,
                        stdout="",
                        stderr=str(result),
                        execution_time_ms=0.0,
                        language=chunk[j].language,
                        test_type=chunk[j].test_type,
                        created_at=datetime.now(UTC),
                        container_name=chunk[j].container_name or "unknown",
                    )
                    results.append(error_result)
                elif isinstance(result, TestExecutionResult):
                    results.append(result)
        successful = sum(1 for r in results if r.success)
        background_tasks.add_task(
            log_batch_execution_metrics,
            user_id=current_user.preferred_username,
            total=len(results),
            successful=successful,
        )
        logger.info(f"Batch execution completed: {successful}/{len(results)} successful")
        return results
    except Exception as e:
        logger.error(f"Batch execution failed: {str(e)}")
        raise internal_error(f"Batch execution failed: {str(e)}")


@router.get(
    "/active",
    operation_id="list_active_executions",
    summary="Get active test executions",
    responses={
        200: {
            "description": "Successfully retrieved active executions",
            "content": {
                "application/json": {
                    "example": {
                        "active_count": 2,
                        "active_tests": [
                            {
                                "test_id": "test-123",
                                "container_id": "container-abc",
                                "status": "running",
                            }
                        ],
                    }
                }
            },
        },
        401: {"description": "Authentication required", "model": Error401Response},
        403: {"description": "Insufficient permissions", "model": Error403Response},
        404: {
            "description": "No active test executions found",
            "model": Error404Response,
        },
        500: {
            "description": "Failed to get active executions",
            "model": Error500Response,
        },
    },
)
async def get_active_executions(
    current_user: User = Depends(requires_read_permission),
    tenant_id: str = Depends(require_tenant),
) -> dict[str, Any]:
    """
    Get List[Any] of currently active test executions.

    Returns:
        Dictionary containing active test information
    """
    try:
        orchestrator = await get_test_orchestrator()
        active_tests = []
        for test_id, container_id in orchestrator.active_containers.items():
            # Tenant isolation: only surface executions owned by the caller's tenant.
            if orchestrator.active_container_tenants.get(test_id) != tenant_id:
                continue
            active_tests.append(
                {"test_id": test_id, "container_id": container_id, "status": "running"}
            )

        return success_response(
            data={
                "active_executions": active_tests,
                "active_count": len(active_tests),
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get active executions: {str(e)}")
        # Return empty result instead of error
        return success_response(data={"active_executions": [], "active_count": 0})


@router.delete(
    "/cleanup/{test_id}",
    operation_id="cleanup_test_execution",
    summary="Cleanup test execution",
    responses={
        200: {
            "description": "Test execution cleaned up successfully",
            "content": {
                "application/json": {
                    "example": {
                        "test_id": "test-123",
                        "status": "cleaned",
                        "message": "Test execution resources cleaned up successfully",
                    }
                }
            },
        },
        401: {"description": "Authentication required", "model": Error401Response},
        403: {"description": "Insufficient permissions", "model": Error403Response},
        404: {"description": "Test execution not found", "model": Error404Response},
        500: {
            "description": "Failed to cleanup test execution",
            "model": Error500Response,
        },
    },
)
async def cleanup_test_execution(
    test_id: str,
    current_user: User = Depends(requires_read_permission),
    tenant_id: str = Depends(require_tenant),
) -> dict[str, Any]:
    """
    Cleanup resources for a specific test execution.

    Args:
        test_id: Test execution ID to cleanup

    Returns:
        Cleanup status
    """
    try:
        orchestrator = await get_test_orchestrator()
        # Tenant isolation: a test_id belonging to another tenant must be
        # indistinguishable from one that does not exist (no cross-tenant
        # existence leak, no cross-tenant destructive cleanup).
        if (
            test_id not in orchestrator.active_containers
            or orchestrator.active_container_tenants.get(test_id) != tenant_id
        ):
            raise not_found("Test execution", test_id)
        container_id = orchestrator.active_containers[test_id]
        await orchestrator._cleanup_container(test_id, container_id)
        logger.info(f"Test execution {test_id} cleaned up by {current_user.preferred_username}")
        return {
            "test_id": test_id,
            "status": "cleaned",
            "message": "Test execution resources cleaned up successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cleanup test execution: {str(e)}")
        raise internal_error(f"Failed to cleanup test execution: {str(e)}")


@router.get(
    "/container-specs",
    operation_id="list_container_specs",
    summary="Get container specifications",
    responses={
        200: {
            "description": "Successfully retrieved container specifications",
            "content": {
                "application/json": {
                    "examples": {
                        "single_language": {
                            "summary": "Single language specification",
                            "value": {
                                "language": "python",
                                "specification": {
                                    "image": "python:3.11-slim",
                                    "version": "3.11",
                                    "memory_limit": "512m",
                                    "cpu_quota": 100000,
                                    "environment": {"PYTHONUNBUFFERED": "1"},
                                    "mounts": [],
                                    "security_opts": ["no-new-privileges:true"],
                                },
                            },
                        },
                        "all_languages": {
                            "summary": "All language specifications",
                            "value": {
                                "specifications": {
                                    "python": {
                                        "image": "python:3.11-slim",
                                        "version": "3.11",
                                        "available": True,
                                    },
                                    "javascript": {
                                        "image": "node:20-slim",
                                        "version": "20",
                                        "available": True,
                                    },
                                },
                                "count": 2,
                            },
                        },
                    }
                }
            },
        },
        401: {"description": "Authentication required", "model": Error401Response},
        403: {"description": "Insufficient permissions", "model": Error403Response},
        404: {
            "description": "No container specification for language",
            "model": Error404Response,
        },
        500: {
            "description": "Failed to get container specifications",
            "model": Error500Response,
        },
    },
)
async def get_container_specifications(
    language: str | None = Query(None, description="Filter by language"),
    current_user: User = Depends(requires_read_permission),
    tenant_id: str = Depends(require_tenant),
) -> dict[str, Any]:
    """
    Get container specifications for test execution.

    Args:
        language: Optional language filter

    Returns:
        Container specifications
    """
    try:
        registry = await get_container_registry()
        if language:
            spec = registry.get_container_spec(language)
            if not spec:
                raise not_found("Container specification", language)
            return {
                "language": language,
                "specification": {
                    "image": spec.image,
                    "version": spec.version,
                    "memory_limit": spec.memory_limit,
                    "cpu_quota": spec.cpu_quota,
                    "environment": spec.environment,
                    "mounts": spec.mounts,
                    "security_opts": spec.security_opts,
                },
            }
        else:
            all_specs = {}
            for lang in [
                "python",
                "go",
                "rust",
                "java",
                "csharp",
                "cpp",
                "javascript",
                "typescript",
            ]:
                spec = registry.get_container_spec(lang)
                if spec:
                    all_specs[lang] = {
                        "image": spec.image,
                        "version": spec.version,
                        "available": True,
                    }
            return {"specifications": all_specs, "count": len(all_specs)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get container specifications: {str(e)}")
        raise internal_error(f"Failed to get container specifications: {str(e)}")


@router.get(
    "/health",
    operation_id="test_execution_health_check",
    summary="Test execution service health",
    responses={
        200: {
            "description": "Service health status",
            "content": {
                "application/json": {
                    "examples": {
                        "healthy": {
                            "summary": "All components healthy",
                            "value": {
                                "service": "test_execution",
                                "status": "healthy",
                                "timestamp": "2025-01-23T10:30:00.000Z",
                                "components": {
                                    "orchestrator": {
                                        "status": "healthy",
                                        "active_tests": 0,
                                    },
                                    "registry": {
                                        "status": "healthy",
                                        "supported_languages": 8,
                                    },
                                    "docker": {"status": "healthy"},
                                },
                            },
                        },
                        "degraded": {
                            "summary": "Service degraded",
                            "value": {
                                "service": "test_execution",
                                "status": "degraded",
                                "timestamp": "2025-01-23T10:30:00.000Z",
                                "components": {
                                    "orchestrator": {
                                        "status": "healthy",
                                        "active_tests": 0,
                                    },
                                    "registry": {
                                        "status": "unhealthy",
                                        "error": "Registry initialization failed",
                                    },
                                    "docker": {"status": "healthy"},
                                },
                            },
                        },
                        "unhealthy": {
                            "summary": "Service unhealthy",
                            "value": {
                                "service": "test_execution",
                                "status": "unhealthy",
                                "timestamp": "2025-01-23T10:30:00.000Z",
                                "error": "Critical service failure",
                            },
                        },
                    }
                }
            },
        },
        404: {"description": "Health endpoint not found", "model": Error404Response},
        503: {"description": "Service unavailable", "model": Error503Response},
    },
)
async def test_execution_health(
    tenant_id: str = Depends(require_tenant),
) -> dict[str, Any]:
    """
    Check health of test execution service.

    Returns:
        Health status of the service
    """
    try:
        components_status = {}
        overall_status = "healthy"

        try:
            orchestrator = await get_test_orchestrator()
            active_count = len(orchestrator.active_containers)
            components_status["orchestrator"] = {
                "status": "healthy",
                "active_tests": active_count,
            }
        except Exception as e:
            components_status["orchestrator"] = {"status": "unhealthy", "error": str(e)}
            overall_status = "degraded"

        try:
            registry = await get_container_registry()
            supported_langs = registry.get_supported_languages()
            components_status["registry"] = {
                "status": "healthy",
                "supported_languages": len(supported_langs),
            }
        except Exception as e:
            components_status["registry"] = {"status": "unhealthy", "error": str(e)}
            overall_status = "degraded"

        try:
            docker_service = get_docker_service()
            components_status["docker"] = {"status": "healthy" if docker_service else "unavailable"}
        except Exception as e:
            components_status["docker"] = {"status": "unhealthy", "error": str(e)}
            overall_status = "degraded"

        return success_response(
            data={
                "service": "test_execution",
                "status": overall_status,
                "components": components_status,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return success_response(
            data={
                "service": "test_execution",
                "status": "unhealthy",
                "components": {},
                "timestamp": datetime.now(UTC).isoformat(),
                "error": str(e),
            }
        )


async def log_test_execution_metrics(
    user_id: str, result: TestExecutionResult, request: TestExecutionRequest
) -> None:
    """Log test execution metrics for analytics."""
    try:
        metrics_service = get_metrics_service()
        metrics_data = {
            "event_type": "test_execution",
            "timestamp": datetime.now(UTC).isoformat(),
            "user_id": user_id,
            "test_id": result.test_id,
            "language": result.language.value,
            "test_type": result.test_type.value,
            "success": result.success,
            "execution_time_ms": result.execution_time_ms,
            "exit_code": result.exit_code,
            "security_level": request.security_level.value,
            "timeout_seconds": request.timeout_seconds,
        }
        await metrics_service.record_test_execution(metrics_data)
        logger.info(f"Test execution metrics logged for {result.test_id}")
    except Exception as e:
        logger.error(f"Failed to log test execution metrics: {str(e)}")


async def log_batch_execution_metrics(user_id: str, total: int, successful: int) -> None:
    """Log batch execution metrics."""
    try:
        metrics_service = get_metrics_service()
        metrics_data = {
            "event_type": "batch_test_execution",
            "timestamp": datetime.now(UTC).isoformat(),
            "user_id": user_id,
            "total_tests": total,
            "successful_tests": successful,
            "failed_tests": total - successful,
            "success_rate": successful / total if total > 0 else 0.0,
        }
        await metrics_service.record_test_execution(metrics_data)
        logger.info(f"Batch execution metrics logged: {successful}/{total} successful")
    except Exception as e:
        logger.error(f"Failed to log batch execution metrics: {str(e)}")
