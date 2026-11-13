"""Error Response Models for API.

This module contains standardized error response models that ensure consistent
error handling across all API endpoints.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_serializer

# Inline definitions replacing monolith imports
SAFE_STRING_PATTERN = r"^[\s\S]*$"


class ResponseStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"


class StandardErrorResponse(BaseModel):
    """Standard error response format for all API errors."""

    status: ResponseStatus = Field(
        default=ResponseStatus.ERROR,
        description="Response status (always 'error' for error responses)",
    )
    message: str = Field(
        ..., pattern=SAFE_STRING_PATTERN, description="Human-readable error message"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="ISO 8601 timestamp of when the error occurred",
    )
    request_id: str = Field(
        ...,
        pattern=SAFE_STRING_PATTERN,
        description="Unique request identifier for tracking",
    )
    path: str = Field(
        ...,
        pattern=SAFE_STRING_PATTERN,
        description="API endpoint path where the error occurred",
    )
    error_code: str | None = Field(
        None,
        pattern=SAFE_STRING_PATTERN,
        description="Machine-readable error code for programmatic handling",
    )
    details: dict[str, Any] | None = Field(None, description="Additional error context and details")

    @field_serializer("timestamp")
    def serialize_timestamp(self, timestamp: datetime) -> str:
        """Serialize timestamp to ISO 8601 format with timezone information."""
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        return timestamp.isoformat()

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "error",
                "message": "Resource not found",
                "timestamp": "2025-01-12T10:30:00.000Z",
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "path": "/api/workspaces/my-container",
                "error_code": "RESOURCE_NOT_FOUND",
                "details": {
                    "resource_type": "container",
                    "resource_id": "my-container",
                },
            }
        }
    )


class APIValidationErrorDetail(BaseModel):
    """Detailed information about a specific field validation error."""

    field: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Field that failed validation")
    message: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Validation error message")
    type: str = Field(..., pattern=SAFE_STRING_PATTERN, description="Type of validation error")
    input: Any = Field(None, description="The invalid input value")


class ValidationErrorResponse(StandardErrorResponse):
    """Specialized error response for request validation failures."""

    status: ResponseStatus = Field(default=ResponseStatus.ERROR, description="Response status")
    message: str = Field(
        default="Request validation failed",
        pattern=SAFE_STRING_PATTERN,
        description="Human-readable error message",
    )
    error_code: str | None = Field(
        default="VALIDATION_ERROR",
        pattern=SAFE_STRING_PATTERN,
        description="Machine-readable error code",
    )
    validation_errors: list[APIValidationErrorDetail] = Field(
        ..., description="List of validation errors"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "error",
                "message": "Request validation failed",
                "timestamp": "2025-01-12T10:30:00.000Z",
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "path": "/api/analysis/analyze",
                "error_code": "VALIDATION_ERROR",
                "validation_errors": [
                    {
                        "field": "file_path",
                        "message": "Field required",
                        "type": "missing",
                        "input": None,
                    }
                ],
            }
        }
    )


# Predefined error responses for common HTTP status codes
class Error400Response(StandardErrorResponse):
    """Bad Request (400) error response model."""

    message: str = Field(
        default="Bad request",
        pattern=SAFE_STRING_PATTERN,
        description="Human-readable error message",
    )
    error_code: str | None = Field(
        default="BAD_REQUEST",
        pattern=SAFE_STRING_PATTERN,
        description="Machine-readable error code",
    )


class Error401Response(StandardErrorResponse):
    """Unauthorized (401) error response model."""

    message: str = Field(
        default="Authentication required",
        pattern=SAFE_STRING_PATTERN,
        description="Human-readable error message",
    )
    error_code: str | None = Field(
        default="UNAUTHORIZED",
        pattern=SAFE_STRING_PATTERN,
        description="Machine-readable error code",
    )


class Error403Response(StandardErrorResponse):
    """Forbidden (403) error response model."""

    message: str = Field(
        default="Insufficient permissions",
        pattern=SAFE_STRING_PATTERN,
        description="Human-readable error message",
    )
    error_code: str | None = Field(
        default="FORBIDDEN",
        pattern=SAFE_STRING_PATTERN,
        description="Machine-readable error code",
    )


class Error404Response(StandardErrorResponse):
    """Not Found (404) error response model."""

    message: str = Field(
        default="Resource not found",
        pattern=SAFE_STRING_PATTERN,
        description="Human-readable error message",
    )
    error_code: str | None = Field(
        default="NOT_FOUND",
        pattern=SAFE_STRING_PATTERN,
        description="Machine-readable error code",
    )


class Error405Response(StandardErrorResponse):
    """Method Not Allowed error response."""

    message: str = Field(
        default="Method not allowed",
        pattern=SAFE_STRING_PATTERN,
        description="Human-readable error message",
    )
    error_code: str | None = Field(
        default="METHOD_NOT_ALLOWED",
        pattern=SAFE_STRING_PATTERN,
        description="Machine-readable error code",
    )


class Error406Response(StandardErrorResponse):
    """Not Acceptable (406) error response model."""

    message: str = Field(
        default="Not acceptable",
        pattern=SAFE_STRING_PATTERN,
        description="Human-readable error message",
    )
    error_code: str | None = Field(
        default="NOT_ACCEPTABLE",
        pattern=SAFE_STRING_PATTERN,
        description="Machine-readable error code",
    )


class Error408Response(StandardErrorResponse):
    """Request Timeout (408) error response model."""

    message: str = Field(
        default="Request timeout",
        pattern=SAFE_STRING_PATTERN,
        description="Human-readable error message",
    )
    error_code: str | None = Field(
        default="REQUEST_TIMEOUT",
        pattern=SAFE_STRING_PATTERN,
        description="Machine-readable error code",
    )


class Error409Response(StandardErrorResponse):
    """Conflict error response."""

    message: str = Field(
        default="Resource conflict",
        pattern=SAFE_STRING_PATTERN,
        description="Human-readable error message",
    )
    error_code: str | None = Field(
        default="CONFLICT",
        pattern=SAFE_STRING_PATTERN,
        description="Machine-readable error code",
    )


class Error422Response(ValidationErrorResponse):
    """Unprocessable Entity (422) error response for validation failures."""


class Error429Response(StandardErrorResponse):
    """Too Many Requests (429) error response model."""

    message: str = Field(
        default="Too many requests",
        pattern=SAFE_STRING_PATTERN,
        description="Human-readable error message",
    )
    error_code: str | None = Field(
        default="RATE_LIMIT_EXCEEDED",
        pattern=SAFE_STRING_PATTERN,
        description="Machine-readable error code",
    )


class Error500Response(StandardErrorResponse):
    """Internal Server Error (500) response model."""

    message: str = Field(
        default="Internal server error",
        pattern=SAFE_STRING_PATTERN,
        description="Human-readable error message",
    )
    error_code: str | None = Field(
        default="INTERNAL_ERROR",
        pattern=SAFE_STRING_PATTERN,
        description="Machine-readable error code",
    )


class Error503Response(StandardErrorResponse):
    """Service Unavailable (503) error response model."""

    message: str = Field(
        default="Service temporarily unavailable",
        pattern=SAFE_STRING_PATTERN,
        description="Human-readable error message",
    )
    error_code: str | None = Field(
        default="SERVICE_UNAVAILABLE",
        pattern=SAFE_STRING_PATTERN,
        description="Machine-readable error code",
    )


# Aliases for backwards compatibility
APIError = StandardErrorResponse
ValidationError = ValidationErrorResponse
NotFoundError = Error404Response
InternalServerError = Error500Response
