"""
Custom API Exceptions

These exceptions automatically convert to standardized error responses
via the global exception handlers.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status


class BaseAPIException(HTTPException):
    """
    Base class for all API exceptions.

    Provides consistent structure for error details.
    """

    def __init__(
        self,
        status_code: int,
        message: str,
        error_code: str,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ):
        # Create structured detail dict
        detail: dict[str, Any] = {"message": message, "error_code": error_code}
        if details:
            detail["details"] = details

        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code
        self.details = details
        # For compatibility with global exception handler
        self.error_details = details


class AgentConfigurationError(BaseAPIException):
    """
    Exception raised when AI agent configuration is invalid.
    """

    def __init__(
        self,
        message: str = "Agent configuration error",
        error_code: str = "AGENT_CONFIG_ERROR",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            message=message,
            error_code=error_code,
            details=details,
        )


class AITimeoutError(BaseAPIException):
    """
    Exception raised when AI service request times out.
    """

    def __init__(
        self,
        message: str = "AI service timeout",
        error_code: str = "AI_TIMEOUT",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            message=message,
            error_code=error_code,
            details=details,
        )


class AIValidationError(BaseAPIException):
    """
    Exception raised when AI service input validation fails.
    """

    def __init__(
        self,
        message: str = "AI input validation error",
        error_code: str = "AI_VALIDATION_ERROR",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            message=message,
            error_code=error_code,
            details=details,
        )


class NotFoundException(BaseAPIException):
    """
    Exception for resource not found errors (404).
    """

    def __init__(
        self,
        message: str = "Resource not found",
        error_code: str = "NOT_FOUND",
        resource_type: str | None = None,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        # Add resource info to details
        if resource_type or resource_id:
            if details is None:
                details = {}
            if resource_type:
                details["resource_type"] = resource_type
            if resource_id:
                details["resource_id"] = resource_id

        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            message=message,
            error_code=error_code,
            details=details,
        )


class ValidationException(BaseAPIException):
    """
    Exception for validation errors (400).

    Use this for business logic validation, not request validation.
    """

    def __init__(
        self,
        message: str = "Validation failed",
        error_code: str = "VALIDATION_FAILED",
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        # Add field info to details
        if field:
            if details is None:
                details = {}
            details["field"] = field

        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            message=message,
            error_code=error_code,
            details=details,
        )


class AuthenticationException(BaseAPIException):
    """
    Exception for authentication failures (401).
    """

    def __init__(
        self,
        message: str = "Authentication required",
        error_code: str = "UNAUTHORIZED",
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ):
        # Add WWW-Authenticate header if not present
        if headers is None:
            headers = {}
        if "WWW-Authenticate" not in headers:
            headers["WWW-Authenticate"] = "Bearer"

        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            message=message,
            error_code=error_code,
            details=details,
            headers=headers,
        )


class PermissionException(BaseAPIException):
    """
    Exception for permission/authorization failures (403).
    """

    def __init__(
        self,
        message: str = "Insufficient permissions",
        error_code: str = "FORBIDDEN",
        required_permission: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        # Add permission info to details
        if required_permission:
            if details is None:
                details = {}
            details["required_permission"] = required_permission

        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            message=message,
            error_code=error_code,
            details=details,
        )


class ConflictException(BaseAPIException):
    """
    Exception for resource conflicts (409).
    """

    def __init__(
        self,
        message: str = "Resource conflict",
        error_code: str = "CONFLICT",
        conflicting_resource: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        # Add conflict info to details
        if conflicting_resource:
            if details is None:
                details = {}
            details["conflicting_resource"] = conflicting_resource

        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            message=message,
            error_code=error_code,
            details=details,
        )


class ServiceUnavailableException(BaseAPIException):
    """
    Exception for service unavailable errors (503).
    """

    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        error_code: str = "SERVICE_UNAVAILABLE",
        retry_after: int | None = None,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ):
        # Add Retry-After header if specified
        if retry_after:
            if headers is None:
                headers = {}
            headers["Retry-After"] = str(retry_after)
            if details is None:
                details = {}
            details["retry_after_seconds"] = retry_after

        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            message=message,
            error_code=error_code,
            details=details,
            headers=headers,
        )


class RateLimitException(BaseAPIException):
    """
    Exception for rate limit errors (429).
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        error_code: str = "RATE_LIMIT_EXCEEDED",
        retry_after: int | None = None,
        limit: int | None = None,
        remaining: int | None = None,
        reset: int | None = None,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ):
        if details is None:
            details = {}

        # Add rate limit info to details
        if limit is not None:
            details["limit"] = limit
        if remaining is not None:
            details["remaining"] = remaining
        if reset is not None:
            details["reset"] = reset

        # Add rate limit headers
        if headers is None:
            headers = {}
        if retry_after:
            headers["Retry-After"] = str(retry_after)
        if limit is not None:
            headers["X-RateLimit-Limit"] = str(limit)
        if remaining is not None:
            headers["X-RateLimit-Remaining"] = str(remaining)
        if reset is not None:
            headers["X-RateLimit-Reset"] = str(reset)

        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            message=message,
            error_code=error_code,
            details=details,
            headers=headers,
        )


class ProcessingException(BaseAPIException):
    """
    Exception for processing errors (422).

    Use this when the request is valid but cannot be processed.
    """

    def __init__(
        self,
        message: str = "Unable to process request",
        error_code: str = "UNPROCESSABLE_ENTITY",
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        # Add reason to details
        if reason:
            if details is None:
                details = {}
            details["reason"] = reason

        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            message=message,
            error_code=error_code,
            details=details,
        )


class InternalServerException(BaseAPIException):
    """
    Exception for internal server errors (500).

    Use sparingly - most errors should have more specific types.
    """

    def __init__(
        self,
        message: str = "Internal server error",
        error_code: str = "INTERNAL_ERROR",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=message,
            error_code=error_code,
            details=details,
        )


class ContainerNotFoundException(BaseAPIException):
    """
    Exception for when a container is not found.
    """

    def __init__(
        self,
        container_name: str,
        message: str | None = None,
        error_code: str = "CONTAINER_NOT_FOUND",
        details: dict[str, Any] | None = None,
    ):
        if message is None:
            message = f"Container '{container_name}' not found"

        if details is None:
            details = {}

        # Add container info to details
        details.update(
            {
                "resource_type": "container",
                "resource_id": container_name,
            }
        )

        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            message=message,
            error_code=error_code,
            details=details,
        )
