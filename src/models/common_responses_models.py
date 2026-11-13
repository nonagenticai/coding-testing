"""Common response models for consistent API endpoint responses.

This module provides standardized response models that ensure consistency across all
API endpoints. These models support proper OpenAPI documentation generation and provide
a unified error handling experience.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from models.error_models import (
    Error400Response,
    Error401Response,
    Error403Response,
    Error404Response,
    Error408Response,
    Error500Response,
    Error503Response,
    StandardErrorResponse,
    ValidationErrorResponse,
)


class LegacyResponseStatus(StrEnum):
    """Legacy response status enum."""

    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"


class LegacyBaseResponse(BaseModel):
    """Legacy base response model."""

    status: LegacyResponseStatus = Field(
        default=LegacyResponseStatus.SUCCESS, description="Response status"
    )
    message: str | None = Field(default=None, description="Response message")
    data: dict[str, Any] | None = Field(default=None, description="Response data")


class ValidationErrorDetail(BaseModel):
    """Individual validation error detail."""

    loc: list[str] = Field(..., description="Location of the validation error")
    msg: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")


class LegacyStandardErrorResponse(BaseModel):
    """Legacy standard error response format."""

    detail: str = Field(..., description="Error message")
    status_code: int | None = Field(None, description="HTTP status code")
    type: str | None = Field(None, description="Error type")


class SimpleValidationErrorResponse(BaseModel):
    """Response model for validation errors (422)."""

    detail: list[ValidationErrorDetail] = Field(..., description="List of validation errors")


class NotFoundResponse(BaseModel):
    """Response model for resource not found errors (404)."""

    detail: str = Field(..., description="Resource not found message")


class UnauthorizedResponse(BaseModel):
    """Response model for authentication errors (401)."""

    detail: str = Field(..., description="Authentication error message")


class ForbiddenResponse(BaseModel):
    """Response model for authorization errors (403)."""

    detail: str = Field(..., description="Authorization error message")


class InternalServerErrorResponse(BaseModel):
    """Response model for server errors (500)."""

    detail: str = Field(..., description="Internal server error message")


class BadRequestResponse(BaseModel):
    """Response model for bad request errors (400)."""

    detail: str = Field(..., description="Bad request error message")
    status_code: int | None = Field(400, description="HTTP status code")
    type: str | None = Field("input_validation_error", description="Error type")


class ResourceMetrics(BaseModel):
    """System resource utilization metrics for monitoring and alerting."""

    cpu_usage: float = Field(default=0.0, description="CPU usage percentage")
    memory_usage: float = Field(default=0.0, description="Memory usage in MB")
    disk_usage: float = Field(default=0.0, description="Disk usage percentage")
    network_connections: int = Field(default=0, description="Number of active network connections")


class ServiceUnavailableResponse(BaseModel):
    """Response model for service unavailable errors (503)."""

    detail: str = Field(..., description="Service unavailable error message")


class GoneResponse(BaseModel):
    """Response model for gone/deprecated resource errors (410)."""

    detail: str = Field(..., description="Resource is gone or deprecated message")


class NotImplementedResponse(BaseModel):
    """Response model for not implemented errors (501)."""

    detail: str = Field(..., description="Feature not implemented error message")


class TimeoutResponse(BaseModel):
    """Response model for timeout errors (408)."""

    detail: str = Field(..., description="Request timeout error message")


def _error_example(message: str, path: str, error_code: str) -> dict[str, Any]:
    return {
        "status": "error",
        "message": message,
        "timestamp": "2025-06-02T11:45:30Z",
        "request_id": "req_example123",
        "path": path,
        "error_code": error_code,
        "details": {},
    }


# Common response schemas for OpenAPI documentation
COMMON_RESPONSES = {
    400: {
        "description": "Bad Request - Invalid input or malformed request",
        "model": Error400Response,
        "content": {
            "application/json": {
                "examples": {
                    "validation_error": {
                        "summary": "Input validation error",
                        "value": _error_example(
                            "Invalid request body: Unsafe string contains illegal characters",
                            "/api/tests/execute",
                            "BAD_REQUEST",
                        ),
                    },
                    "path_error": {
                        "summary": "Path validation error",
                        "value": _error_example(
                            "Invalid request path contains illegal characters",
                            "/api/tests/results/bad-path",
                            "BAD_REQUEST",
                        ),
                    },
                }
            }
        },
    },
    401: {
        "description": "Unauthorized - Authentication required or invalid credentials",
        "model": Error401Response,
        "content": {
            "application/json": {
                "example": _error_example(
                    "Invalid authentication credentials",
                    "/api/tests/execute",
                    "UNAUTHORIZED",
                )
            }
        },
    },
    403: {
        "description": "Forbidden - Insufficient permissions",
        "model": Error403Response,
        "content": {
            "application/json": {
                "example": _error_example(
                    "Insufficient permissions to access this resource",
                    "/api/tests/execute",
                    "FORBIDDEN",
                )
            }
        },
    },
    404: {
        "description": "Not Found - Resource does not exist",
        "model": Error404Response,
        "content": {
            "application/json": {
                "example": _error_example(
                    "Resource not found", "/api/tests/results/missing", "NOT_FOUND"
                )
            }
        },
    },
    408: {
        "description": "Request Timeout - Request took too long to process",
        "model": Error408Response,
        "content": {
            "application/json": {
                "example": _error_example(
                    "Request timeout - processing took too long",
                    "/api/tests/execute",
                    "REQUEST_TIMEOUT",
                )
            }
        },
    },
    410: {
        "description": "Gone - Resource is deprecated or no longer available",
        "model": StandardErrorResponse,
        "content": {
            "application/json": {
                "example": _error_example(
                    "This endpoint has been deprecated. Use containerized alternatives.",
                    "/api/tests/legacy-endpoint",
                    "GONE",
                )
            }
        },
    },
    422: {
        "description": "Unprocessable Entity - Validation errors",
        "model": ValidationErrorResponse,
        "content": {
            "application/json": {
                "example": {
                    "status": "error",
                    "message": "Request validation failed",
                    "timestamp": "2025-06-02T11:45:30Z",
                    "request_id": "req_validation123",
                    "path": "/api/tests/execute",
                    "error_code": "VALIDATION_ERROR",
                    "validation_errors": [
                        {
                            "field": "body.field_name",
                            "message": "field required",
                            "type": "value_error.missing",
                            "input": None,
                        }
                    ],
                }
            }
        },
    },
    500: {
        "description": "Internal Server Error - Unexpected server error",
        "model": Error500Response,
        "content": {
            "application/json": {
                "example": _error_example(
                    "Internal server error occurred",
                    "/api/tests/execute",
                    "INTERNAL_ERROR",
                )
            }
        },
    },
    501: {
        "description": "Not Implemented - Feature not yet implemented",
        "model": StandardErrorResponse,
        "content": {
            "application/json": {
                "example": _error_example(
                    "This feature is not yet implemented. Planned for future release.",
                    "/api/tests/future-feature",
                    "NOT_IMPLEMENTED",
                )
            }
        },
    },
    503: {
        "description": "Service Unavailable - External service is unavailable",
        "model": Error503Response,
        "content": {
            "application/json": {
                "example": _error_example(
                    "External service is currently unavailable",
                    "/api/tests/execute",
                    "SERVICE_UNAVAILABLE",
                )
            }
        },
    },
}


def get_common_responses(*status_codes: int) -> dict[int, dict[str, Any]]:
    """Get common response schemas for specified status codes."""
    return {code: COMMON_RESPONSES[code] for code in status_codes if code in COMMON_RESPONSES}


def get_standard_responses() -> dict[int, dict[str, Any]]:
    """Get standard responses that most endpoints should include."""
    return get_common_responses(400, 404, 422, 500)


def get_auth_responses() -> dict[int, dict[str, Any]]:
    """Get authentication-related responses."""
    return get_common_responses(401, 403)


def get_all_responses() -> dict[int, dict[str, Any]]:
    """Get all common response schemas."""
    return COMMON_RESPONSES.copy()
