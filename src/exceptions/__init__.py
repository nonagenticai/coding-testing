"""Application exceptions module.

This module provides centralized exception definitions for the Code Graph Helper application.
"""

from .api_exceptions import (
    AgentConfigurationError,
    AITimeoutError,
    AIValidationError,
    AuthenticationException,
    BaseAPIException,
    ConflictException,
    ContainerNotFoundException,
    InternalServerException,
    NotFoundException,
    PermissionException,
    ProcessingException,
    RateLimitException,
    ServiceUnavailableException,
    ValidationException,
)
from .pydantic_ai_exceptions import (
    AnalysisError,
    APIKeyError,
    ModelNotFoundError,
    ModelProviderError,
    PydanticAIException,
    RateLimitError,
    TokenLimitError,
    create_error_response,
)

# Backward compatibility aliases
APIException = BaseAPIException
AuthorizationException = PermissionException
ServerException = InternalServerException
BadRequestException = ValidationException
TimeoutException = ProcessingException

__all__ = [
    # API Exceptions
    "AgentConfigurationError",
    "AITimeoutError",
    "AIValidationError",
    "BaseAPIException",
    "NotFoundException",
    "ValidationException",
    "AuthenticationException",
    "PermissionException",
    "ConflictException",
    "ServiceUnavailableException",
    "RateLimitException",
    "ProcessingException",
    "InternalServerException",
    "ContainerNotFoundException",
    # PydanticAI Exceptions
    "AnalysisError",
    "APIKeyError",
    "ModelNotFoundError",
    "ModelProviderError",
    "PydanticAIException",
    "RateLimitError",
    "TokenLimitError",
    "create_error_response",
    # Backward compatibility aliases
    "APIException",
    "AuthorizationException",
    "ServerException",
    "BadRequestException",
    "TimeoutException",
]
