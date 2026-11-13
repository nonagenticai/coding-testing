# src/routers/testing_modules/test_discovery.py
"""
Test Discovery and Configuration Module

Endpoints for:
- Language discovery and support
- Test type discovery
- Tool configuration
- Container listing
"""

import logging
import os
from typing import Any

import httpx
from coding_client.auth.models import User
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)

from auth.tenant import get_current_user_with_tenant
from models.error_models import (
    Error400Response,
    Error404Response,
    Error500Response,
)
from services.code_execution_service import CodeExecutionService
from shared.container_validation import validate_and_get_container_for_tenant
from shared.dependencies import (
    create_json_http_exception,
)

from .testing_utils import get_execution_service

logger = logging.getLogger(__name__)

# Shared httpx client for executor health checks (avoids per-request allocation)
_executor_client: httpx.AsyncClient | None = None


def _get_executor_client() -> httpx.AsyncClient:
    global _executor_client
    if _executor_client is None:
        _executor_client = httpx.AsyncClient(timeout=3)
    return _executor_client


# Create router
router = APIRouter()


@router.get(
    "/supported-tools",
    operation_id="list_supported_tools",
    summary="🛠️ List Supported Quality Tools",
    description="""
    **Get a comprehensive list of supported code quality analysis tools.**

    Returns all available tools organized by programming language, including:
    - Tool names and descriptions
    - Supported file types
    - Tool categories (formatter, linter, security, etc.)

    Use the optional `language` query parameter to filter results for a specific language.
    Use the optional `container_name` query parameter to verify container exists before returning tools.
    """,
    responses={
        200: {
            "description": "List of supported tools retrieved successfully",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "language": {"type": "string"},
                            "tools": {"type": "array", "items": {"type": "string"}},
                            "count": {"type": "integer"},
                            "supported_languages": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "tools_by_language": {"type": "object"},
                            "total_tools": {"type": "integer"},
                            "container_name": {"type": "string"},
                        },
                    }
                }
            },
        },
        400: {
            "description": "Bad request - invalid language parameter",
            "model": Error400Response,
        },
        404: {
            "description": "Supported tools not found - no tools available for specified language or container not found",
            "model": Error404Response,
        },
        500: {
            "description": "Failed to retrieve tool list - internal server error",
            "model": Error500Response,
        },
    },
)
async def get_supported_tools(
    language: str | None = Query(None, description="Filter by programming language"),
    container_name: str | None = Query(None, description="Optional container name for scoping"),
    user: User = Depends(get_current_user_with_tenant),
    execution_service: CodeExecutionService = Depends(get_execution_service),
):
    """
    Get list of supported quality analysis tools.

    Returns tools available for each programming language.
    Optionally scoped to a specific container.
    """
    try:
        # Tenant-aware container validation (admins can access all, users only their own)
        if container_name:
            try:
                tenant_id = None if user.is_admin() else user.effective_tenant_id
                await validate_and_get_container_for_tenant(container_name, tenant_id=tenant_id)
                logger.debug(f"Container '{container_name}' validated for tool discovery")
            except (ImportError, RuntimeError) as e:
                logger.warning(f"Container validation not available: {e}, skipping validation")

        if language:
            tools = execution_service._get_supported_tools(language)
            if not tools:
                raise create_json_http_exception(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No tools available for language '{language}'",
                )
            result = {"language": language, "tools": tools, "count": len(tools)}
            if container_name:
                result["container_name"] = container_name
            return result
        else:
            all_tools = {}
            for lang in ["python", "javascript", "typescript"]:
                tools = execution_service._get_supported_tools(lang)
                if tools:
                    all_tools[lang] = tools

            result: dict[str, Any] = {
                "supported_languages": list(all_tools.keys()),
                "tools_by_language": all_tools,
                "total_tools": sum(len(tools) for tools in all_tools.values()),
            }
            if container_name:
                result["container_name"] = container_name
            return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get supported tools: {str(e)}")
        raise create_json_http_exception(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get supported tools: {str(e)}",
        )


@router.get(
    "/languages",
    operation_id="list_supported_languages",
    summary="🌐 List Supported Languages",
    description="""
    **Get a list of all supported programming languages for test execution.**

    Returns detailed information about:
    - Language names and versions
    - Available test types per language
    - Container status and availability
    - Language-specific features

    This endpoint helps determine which languages are available for testing.
    Use the optional `container_name` query parameter to verify container exists before returning languages.
    """,
    responses={
        200: {
            "description": "Languages retrieved successfully",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "languages": {"type": "array", "items": {"type": "string"}},
                            "details": {"type": "object"},
                            "total": {"type": "integer"},
                            "container_name": {"type": "string"},
                        },
                    }
                }
            },
        },
        404: {
            "description": "Languages not found - no supported languages available or container not found",
            "model": Error404Response,
        },
        500: {
            "description": "Failed to retrieve languages - internal server error",
            "model": Error500Response,
        },
    },
)
async def get_supported_languages(
    container_name: str | None = Query(None, description="Optional container name for scoping"),
    user: User = Depends(get_current_user_with_tenant),
    execution_service: CodeExecutionService = Depends(get_execution_service),
):
    """
    Get list of supported languages with details.
    Optionally scoped to a specific container.
    """
    try:
        # Tenant-aware container validation (admins can access all, users only their own)
        if container_name:
            try:
                tenant_id = None if user.is_admin() else user.effective_tenant_id
                await validate_and_get_container_for_tenant(container_name, tenant_id=tenant_id)
                logger.debug(f"Container '{container_name}' validated for language discovery")
            except (ImportError, RuntimeError) as e:
                logger.warning(f"Container validation not available: {e}, skipping validation")

        # Get languages from orchestrator with fallback to executor health checks
        try:
            languages = await execution_service.test_orchestrator.get_supported_languages()
        except (AttributeError, Exception):  # fmt: skip
            languages = []

        if not languages:
            # Discover languages by checking executor health endpoints directly
            executor_map = {
                "python": os.getenv("PYTHON_TEST_SERVICE_URL"),
                "javascript": os.getenv("JAVASCRIPT_TEST_SERVICE_URL"),
                "typescript": os.getenv("JAVASCRIPT_TEST_SERVICE_URL"),  # shares JS executor
                "java": os.getenv("JAVA_TEST_SERVICE_URL"),
                "go": os.getenv("GO_TEST_SERVICE_URL"),
                "rust": os.getenv("RUST_TEST_SERVICE_URL"),
                "csharp": os.getenv("CSHARP_TEST_SERVICE_URL"),
                "cpp": os.getenv("CPP_TEST_SERVICE_URL"),
            }
            discovered: list[str] = []
            client = _get_executor_client()
            for lang, url in executor_map.items():
                if not url:
                    continue
                try:
                    r = await client.get(f"{url}/health")
                    if r.status_code == 200:
                        discovered.append(lang)
                except Exception:
                    pass
            languages = discovered if discovered else ["python", "javascript", "typescript"]

        # Get detailed info for each language
        language_details = {}
        for lang in languages:
            try:
                test_types = await execution_service.test_orchestrator.get_supported_test_types(
                    lang
                )
            except (AttributeError, Exception):  # fmt: skip
                # Fallback test types
                test_types = (
                    ["unit", "integration", "quality"] if lang == "python" else ["unit", "quality"]
                )

            try:
                quality_tools = execution_service._get_supported_tools(lang)
            except Exception:
                quality_tools = []

            language_details[lang] = {
                "supported": True,
                "test_types": test_types,
                "quality_tools": quality_tools,
            }

        result = {
            "languages": languages,
            "details": language_details,
            "total": len(languages),
        }
        if container_name:
            result["container_name"] = container_name
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get supported languages: {str(e)}")
        raise create_json_http_exception(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get supported languages: {str(e)}",
        )


@router.get(
    "/languages/{language}/test-types",
    operation_id="get_language_test_types",
    summary="📋 Get Language Test Types",
    description="""
    **Get available test types for a specific programming language.**

    Different languages support different test types:
    - **Python**: unit, integration, coverage, security
    - **Go**: unit, benchmark, coverage, race, vet
    - **Rust**: unit, doc, benchmark, coverage, clippy, fmt
    - **Java**: unit, integration, coverage, security
    - And more...

    Use this to determine which test types you can run for your code.
    Use the optional `container_name` query parameter to verify container exists before returning test types.
    """,
    responses={
        200: {
            "description": "Test types retrieved successfully",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "language": {"type": "string"},
                            "test_types": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "count": {"type": "integer"},
                            "container_name": {"type": "string"},
                        },
                    }
                }
            },
        },
        404: {
            "description": "Language not supported - specified language is not available or container not found",
            "model": Error404Response,
        },
        500: {
            "description": "Failed to retrieve test types - internal server error",
            "model": Error500Response,
        },
    },
)
async def get_language_test_types(
    language: str,
    container_name: str | None = Query(None, description="Optional container name for scoping"),
    user: User = Depends(get_current_user_with_tenant),
    execution_service: CodeExecutionService = Depends(get_execution_service),
):
    """
    Get supported test types for a language.
    Optionally scoped to a specific container.
    """
    try:
        # Tenant-aware container validation (admins can access all, users only their own)
        if container_name:
            try:
                tenant_id = None if user.is_admin() else user.effective_tenant_id
                await validate_and_get_container_for_tenant(container_name, tenant_id=tenant_id)
                logger.debug(f"Container '{container_name}' validated for test type discovery")
            except (ImportError, RuntimeError) as e:
                logger.warning(f"Container validation not available: {e}, skipping validation")

        # Normalize language name
        normalized = language.lower().strip()

        # Check if language is supported (with fallback)
        try:
            supported = await execution_service.test_orchestrator.get_supported_languages()
        except (AttributeError, Exception):  # fmt: skip
            supported = ["python", "javascript", "typescript"]

        if normalized not in supported:
            raise create_json_http_exception(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Language '{language}' is not supported",
            )

        # Get test types (with fallback)
        try:
            test_types = await execution_service.test_orchestrator.get_supported_test_types(
                normalized
            )
        except (AttributeError, Exception):  # fmt: skip
            test_types = (
                ["unit", "integration", "quality"]
                if normalized == "python"
                else ["unit", "quality"]
            )

        result = {
            "language": normalized,
            "test_types": test_types,
            "count": len(test_types),
        }
        if container_name:
            result["container_name"] = container_name
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get test types for {language}: {str(e)}")
        raise create_json_http_exception(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get test types: {str(e)}",
        )


@router.get(
    "/containers",
    operation_id="list_test_containers",
    summary="🐳 List Test Containers",
    description="""
    **Get information about available test containers.**

    Lists all containers that can be used for test execution:
    - Provisioned containers from GitHub repositories
    - Language-specific test executors
    - Container status and availability

    Use the `filter_type` parameter to filter results:
    - `all` - All containers
    - `test` - Test executor containers
    - `provisioned` - GitHub provisioned containers
    """,
    responses={
        200: {
            "description": "Containers retrieved successfully",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "containers": {
                                "type": "array",
                                "items": {"type": "object"},
                            },
                            "count": {"type": "integer"},
                            "filter": {"type": "string"},
                        },
                    }
                }
            },
        },
        400: {
            "description": "Bad request - invalid filter_type parameter",
            "model": Error400Response,
        },
        404: {
            "description": "Containers not found - no containers available for specified filter",
            "model": Error404Response,
        },
        500: {
            "description": "Failed to retrieve containers - internal server error",
            "model": Error500Response,
        },
    },
)
async def list_test_containers(
    filter_type: str = Query(
        "all",
        pattern="^(all|test|provisioned)$",
        description="Container type filter",
    ),
    user: User = Depends(get_current_user_with_tenant),
):
    """
    List available test containers filtered by tenant.
    """
    try:
        # Use ContainerManager which has tenant-aware filtering
        from shared.container_validation import get_container_manager

        container_manager = get_container_manager()

        # Tenant-aware container listing (admins see all, users only their own)
        tenant_id = None if user.is_admin() else user.effective_tenant_id
        all_containers = container_manager.list_containers(tenant_id=tenant_id)

        # Filter based on type
        if filter_type == "test":
            containers = [c for c in all_containers if getattr(c, "provision_type", None) == "test"]
        elif filter_type == "provisioned":
            containers = [
                c for c in all_containers if getattr(c, "provision_type", None) == "provisioned"
            ]
        else:
            containers = all_containers

        # Always return result, even if empty - this is normal for new systems
        return {
            "containers": [c.model_dump() for c in containers] if containers else [],
            "count": len(containers),
            "filter": filter_type,
        }

    except (ImportError, RuntimeError) as e:
        # Container manager not available in microservice context
        logger.warning(f"Container manager not available: {e}, returning empty list")
        return {
            "containers": [],
            "count": 0,
            "filter": filter_type,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to list containers: {str(e)}")
        raise create_json_http_exception(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list containers: {str(e)}",
        )


__all__ = ["router"]
