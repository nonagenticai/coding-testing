"""Custom exceptions for PydanticAI operations."""

from __future__ import annotations

from typing import Any


class PydanticAIException(Exception):  # noqa: N818
    """Base exception for PydanticAI-related errors."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        """Initialize PydanticAI exception.

        Args:
            message: Error message
            error_code: Optional error code for categorization
            details: Optional additional details
            cause: Optional underlying cause exception
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause

    def __str__(self) -> str:
        """String representation of the exception."""
        parts = [self.message]

        if self.error_code:
            parts.append(f"Error code: {self.error_code}")

        if self.details:
            parts.append(f"Details: {self.details}")

        if self.cause:
            parts.append(f"Caused by: {self.cause}")

        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None,
        }


class AgentConfigurationError(PydanticAIException):
    """Exception raised when agent configuration is invalid."""

    def __init__(
        self,
        message: str,
        agent_id: str | None = None,
        config_field: str | None = None,
        **kwargs: Any,
    ):
        """Initialize agent configuration error.

        Args:
            message: Error message
            agent_id: Agent ID that has configuration issues
            config_field: Specific configuration field that's invalid
            **kwargs: Additional arguments for base exception
        """
        super().__init__(message, **kwargs)
        self.agent_id = agent_id
        self.config_field = config_field

        if agent_id:
            self.details["agent_id"] = agent_id
        if config_field:
            self.details["config_field"] = config_field


class ModelProviderError(PydanticAIException):
    """Exception raised when model provider encounters an error."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        **kwargs: Any,
    ):
        """Initialize model provider error.

        Args:
            message: Error message
            provider: AI provider name
            model: Model name
            status_code: HTTP status code if applicable
            **kwargs: Additional arguments for base exception
        """
        super().__init__(message, **kwargs)
        self.provider = provider
        self.model = model
        self.status_code = status_code

        if provider:
            self.details["provider"] = provider
        if model:
            self.details["model"] = model
        if status_code:
            self.details["status_code"] = status_code


class APIKeyError(ModelProviderError):
    """Exception raised when API key is missing or invalid."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        **kwargs: Any,
    ):
        """Initialize API key error.

        Args:
            message: Error message
            provider: AI provider name
            **kwargs: Additional arguments for base exception
        """
        super().__init__(message, provider=provider, error_code="API_KEY_ERROR", **kwargs)


class RateLimitError(ModelProviderError):
    """Exception raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        retry_after: int | None = None,
        **kwargs: Any,
    ):
        """Initialize rate limit error.

        Args:
            message: Error message
            provider: AI provider name
            retry_after: Seconds to wait before retrying
            **kwargs: Additional arguments for base exception
        """
        super().__init__(message, provider=provider, error_code="RATE_LIMIT", **kwargs)
        self.retry_after = retry_after

        if retry_after:
            self.details["retry_after"] = retry_after


class ModelNotFoundError(ModelProviderError):
    """Exception raised when requested model is not found."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ):
        """Initialize model not found error.

        Args:
            message: Error message
            provider: AI provider name
            model: Model name that was not found
            **kwargs: Additional arguments for base exception
        """
        super().__init__(
            message,
            provider=provider,
            model=model,
            error_code="MODEL_NOT_FOUND",
            **kwargs,
        )


class TokenLimitError(ModelProviderError):
    """Exception raised when token limit is exceeded."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        model: str | None = None,
        token_count: int | None = None,
        token_limit: int | None = None,
        **kwargs: Any,
    ):
        """Initialize token limit error.

        Args:
            message: Error message
            provider: AI provider name
            model: Model name
            token_count: Number of tokens in request
            token_limit: Maximum token limit
            **kwargs: Additional arguments for base exception
        """
        super().__init__(
            message, provider=provider, model=model, error_code="TOKEN_LIMIT", **kwargs
        )
        self.token_count = token_count
        self.token_limit = token_limit

        if token_count:
            self.details["token_count"] = token_count
        if token_limit:
            self.details["token_limit"] = token_limit


class AnalysisError(PydanticAIException):
    """Exception raised during test failure analysis."""

    def __init__(
        self,
        message: str,
        analysis_type: str | None = None,
        test_id: str | None = None,
        **kwargs: Any,
    ):
        """Initialize analysis error.

        Args:
            message: Error message
            analysis_type: Type of analysis that failed
            test_id: Test ID being analyzed
            **kwargs: Additional arguments for base exception
        """
        super().__init__(message, **kwargs)
        self.analysis_type = analysis_type
        self.test_id = test_id

        if analysis_type:
            self.details["analysis_type"] = analysis_type
        if test_id:
            self.details["test_id"] = test_id


class AIValidationError(PydanticAIException):
    """Exception raised when AI input validation fails."""

    def __init__(
        self,
        message: str,
        field_name: str | None = None,
        field_value: Any | None = None,
        **kwargs: Any,
    ):
        """Initialize validation error.

        Args:
            message: Error message
            field_name: Name of field that failed validation
            field_value: Value that failed validation
            **kwargs: Additional arguments for base exception
        """
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)
        self.field_name = field_name
        self.field_value = field_value

        if field_name:
            self.details["field_name"] = field_name
        if field_value is not None:
            self.details["field_value"] = str(field_value)


class AITimeoutError(PydanticAIException):
    """Exception raised when operation times out."""

    def __init__(
        self,
        message: str,
        timeout_seconds: float | None = None,
        operation: str | None = None,
        **kwargs: Any,
    ):
        """Initialize timeout error.

        Args:
            message: Error message
            timeout_seconds: Timeout duration in seconds
            operation: Operation that timed out
            **kwargs: Additional arguments for base exception
        """
        super().__init__(message, error_code="TIMEOUT", **kwargs)
        self.timeout_seconds = timeout_seconds
        self.operation = operation

        if timeout_seconds:
            self.details["timeout_seconds"] = timeout_seconds
        if operation:
            self.details["operation"] = operation


class ResourceExhaustedError(PydanticAIException):
    """Exception raised when resources are exhausted."""

    def __init__(
        self,
        message: str,
        resource_type: str | None = None,
        **kwargs: Any,
    ):
        """Initialize resource exhausted error.

        Args:
            message: Error message
            resource_type: Type of resource that's exhausted
            **kwargs: Additional arguments for base exception
        """
        super().__init__(message, error_code="RESOURCE_EXHAUSTED", **kwargs)
        self.resource_type = resource_type

        if resource_type:
            self.details["resource_type"] = resource_type


class RetryableError(PydanticAIException):
    """Exception that can be retried."""

    def __init__(
        self,
        message: str,
        retry_count: int = 0,
        max_retries: int = 3,
        **kwargs: Any,
    ):
        """Initialize retryable error.

        Args:
            message: Error message
            retry_count: Current retry count
            max_retries: Maximum number of retries
            **kwargs: Additional arguments for base exception
        """
        super().__init__(message, error_code="RETRYABLE", **kwargs)
        self.retry_count = retry_count
        self.max_retries = max_retries

        self.details["retry_count"] = retry_count
        self.details["max_retries"] = max_retries

    def can_retry(self) -> bool:
        """Check if error can be retried."""
        return self.retry_count < self.max_retries

    def increment_retry(self) -> None:
        """Increment retry count."""
        self.retry_count += 1
        self.details["retry_count"] = self.retry_count


class NonRetryableError(PydanticAIException):
    """Exception that should not be retried."""

    def __init__(
        self,
        message: str,
        **kwargs: Any,
    ):
        """Initialize non-retryable error.

        Args:
            message: Error message
            **kwargs: Additional arguments for base exception
        """
        super().__init__(message, error_code="NON_RETRYABLE", **kwargs)


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable.

    Args:
        error: Exception to check

    Returns:
        True if error is retryable, False otherwise
    """
    # PydanticAI specific retryable errors
    if isinstance(error, RetryableError):
        return error.can_retry()

    # Rate limit errors are retryable
    if isinstance(error, RateLimitError):
        return True

    # Non-retryable errors
    if isinstance(error, (NonRetryableError, APIKeyError, ModelNotFoundError, AIValidationError)):
        return False

    # Network/connection errors are usually retryable
    if isinstance(error, (ConnectionError, AITimeoutError)):
        return True

    # Server errors (5xx) are retryable, client errors (4xx) are not
    if isinstance(error, ModelProviderError) and error.status_code:
        return error.status_code >= 500

    # Default to non-retryable
    return False


def get_retry_delay(error: Exception, retry_count: int) -> float:
    """Get retry delay for an error.

    Args:
        error: Exception that occurred
        retry_count: Current retry count

    Returns:
        Delay in seconds before retrying
    """
    # Rate limit errors may specify retry delay
    if isinstance(error, RateLimitError) and error.retry_after:
        return float(error.retry_after)

    # Exponential backoff: 2^retry_count seconds, max 60 seconds
    delay = min(2**retry_count, 60)
    return delay


def categorize_error(error: Exception) -> str:
    """Categorize an error for logging and metrics.

    Args:
        error: Exception to categorize

    Returns:
        Error category string
    """
    if isinstance(error, APIKeyError):
        return "authentication"
    elif isinstance(error, RateLimitError):
        return "rate_limit"
    elif isinstance(error, ModelNotFoundError):
        return "model_not_found"
    elif isinstance(error, TokenLimitError):
        return "token_limit"
    elif isinstance(error, AITimeoutError):
        return "timeout"
    elif isinstance(error, AIValidationError):
        return "validation"
    elif isinstance(error, AgentConfigurationError):
        return "configuration"
    elif isinstance(error, ModelProviderError):
        return "provider"
    elif isinstance(error, AnalysisError):
        return "analysis"
    elif isinstance(error, ResourceExhaustedError):
        return "resource_exhausted"
    else:
        return "unknown"


def create_error_response(
    error: Exception,
    request_id: str,
    analysis_time_ms: float = 0.0,
) -> dict[str, Any]:
    """Create standardized error response.

    Args:
        error: Exception that occurred
        request_id: Request ID for tracking
        analysis_time_ms: Time spent on analysis

    Returns:
        Error response dictionary
    """
    error_category = categorize_error(error)

    response: dict[str, Any] = {
        "request_id": request_id,
        "success": False,
        "error_message": str(error),
        "error_category": error_category,
        "total_analysis_time_ms": analysis_time_ms,
        "retryable": is_retryable_error(error),
        "metadata": {
            "error_type": type(error).__name__,
        },
    }

    # Add additional details for PydanticAI exceptions
    if isinstance(error, PydanticAIException):
        response["error_code"] = error.error_code
        response["error_details"] = error.details
        response["metadata"].update(error.to_dict())

    return response
