"""Tests for exceptions module (api_exceptions and pydantic_ai_exceptions)."""

from __future__ import annotations

from exceptions import (
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
from exceptions.pydantic_ai_exceptions import (
    AITimeoutError,
    AIValidationError,
    AnalysisError,
    APIKeyError,
    ModelNotFoundError,
    ModelProviderError,
    NonRetryableError,
    PydanticAIException,
    RateLimitError,
    ResourceExhaustedError,
    RetryableError,
    TokenLimitError,
    categorize_error,
    create_error_response,
    get_retry_delay,
    is_retryable_error,
)

# ── BaseAPIException ─────────────────────────────────────────────────


def test_base_api_exception():
    """BaseAPIException stores structured detail."""
    exc = BaseAPIException(
        status_code=400,
        message="Bad input",
        error_code="BAD_INPUT",
        details={"field": "name"},
    )
    assert exc.status_code == 400
    assert exc.error_code == "BAD_INPUT"
    assert exc.details == {"field": "name"}
    assert exc.error_details == {"field": "name"}
    assert isinstance(exc.detail, dict)
    assert exc.detail["message"] == "Bad input"
    assert exc.detail["error_code"] == "BAD_INPUT"
    assert exc.detail["details"] == {"field": "name"}


def test_base_api_exception_no_details():
    """BaseAPIException works without details."""
    exc = BaseAPIException(status_code=500, message="Error", error_code="ERR")
    assert exc.details is None
    assert "details" not in exc.detail


def test_base_api_exception_with_headers():
    """BaseAPIException passes headers through."""
    exc = BaseAPIException(
        status_code=401,
        message="Unauthenticated",
        error_code="UNAUTH",
        headers={"WWW-Authenticate": "Bearer"},
    )
    assert exc.headers == {"WWW-Authenticate": "Bearer"}


# ── NotFoundException ─────────────────────────────────────────────────


def test_not_found_exception_defaults():
    """NotFoundException has correct defaults."""
    exc = NotFoundException()
    assert exc.status_code == 404
    assert exc.error_code == "NOT_FOUND"
    assert exc.detail["message"] == "Resource not found"


def test_not_found_exception_with_resource():
    """NotFoundException includes resource info in details."""
    exc = NotFoundException(
        message="Graph not found",
        resource_type="graph",
        resource_id="g-123",
    )
    assert exc.details["resource_type"] == "graph"
    assert exc.details["resource_id"] == "g-123"


def test_not_found_exception_resource_type_only():
    """NotFoundException works with only resource_type."""
    exc = NotFoundException(resource_type="container")
    assert exc.details["resource_type"] == "container"
    assert "resource_id" not in exc.details


# ── ValidationException (API) ────────────────────────────────────────


def test_validation_exception_defaults():
    """ValidationException has correct defaults."""
    exc = ValidationException()
    assert exc.status_code == 400
    assert exc.error_code == "VALIDATION_FAILED"


def test_validation_exception_with_field():
    """ValidationException includes field in details."""
    exc = ValidationException(message="Field invalid", field="email")
    assert exc.details["field"] == "email"


def test_validation_exception_with_field_and_details():
    """ValidationException merges field into provided details."""
    exc = ValidationException(
        field="name",
        details={"min_length": 3},
    )
    assert exc.details["field"] == "name"
    assert exc.details["min_length"] == 3


# ── AuthenticationException ──────────────────────────────────────────


def test_authentication_exception_defaults():
    """AuthenticationException has correct defaults."""
    exc = AuthenticationException()
    assert exc.status_code == 401
    assert exc.error_code == "UNAUTHORIZED"
    assert exc.headers["WWW-Authenticate"] == "Bearer"


def test_authentication_exception_custom_header():
    """AuthenticationException does not overwrite existing WWW-Authenticate."""
    exc = AuthenticationException(headers={"WWW-Authenticate": "Custom"})
    assert exc.headers["WWW-Authenticate"] == "Custom"


# ── PermissionException ──────────────────────────────────────────────


def test_permission_exception_defaults():
    """PermissionException has correct defaults."""
    exc = PermissionException()
    assert exc.status_code == 403
    assert exc.error_code == "FORBIDDEN"


def test_permission_exception_with_required_permission():
    """PermissionException includes required_permission in details."""
    exc = PermissionException(required_permission="write")
    assert exc.details["required_permission"] == "write"


# ── ConflictException ────────────────────────────────────────────────


def test_conflict_exception_defaults():
    """ConflictException has correct defaults."""
    exc = ConflictException()
    assert exc.status_code == 409
    assert exc.error_code == "CONFLICT"


def test_conflict_exception_with_resource():
    """ConflictException includes conflicting_resource in details."""
    exc = ConflictException(conflicting_resource="container-abc")
    assert exc.details["conflicting_resource"] == "container-abc"


# ── ServiceUnavailableException ──────────────────────────────────────


def test_service_unavailable_defaults():
    """ServiceUnavailableException has correct defaults."""
    exc = ServiceUnavailableException()
    assert exc.status_code == 503
    assert exc.error_code == "SERVICE_UNAVAILABLE"


def test_service_unavailable_with_retry_after():
    """ServiceUnavailableException sets Retry-After header and details."""
    exc = ServiceUnavailableException(retry_after=30)
    assert exc.headers["Retry-After"] == "30"
    assert exc.details["retry_after_seconds"] == 30


# ── RateLimitException ───────────────────────────────────────────────


def test_rate_limit_exception_defaults():
    """RateLimitException has correct defaults."""
    exc = RateLimitException()
    assert exc.status_code == 429
    assert exc.error_code == "RATE_LIMIT_EXCEEDED"


def test_rate_limit_exception_with_info():
    """RateLimitException includes rate limit info."""
    exc = RateLimitException(
        retry_after=60,
        limit=100,
        remaining=0,
        reset=1234567890,
    )
    assert exc.details["limit"] == 100
    assert exc.details["remaining"] == 0
    assert exc.details["reset"] == 1234567890
    assert exc.headers["Retry-After"] == "60"
    assert exc.headers["X-RateLimit-Limit"] == "100"
    assert exc.headers["X-RateLimit-Remaining"] == "0"
    assert exc.headers["X-RateLimit-Reset"] == "1234567890"


# ── ProcessingException ──────────────────────────────────────────────


def test_processing_exception_defaults():
    """ProcessingException has correct defaults."""
    exc = ProcessingException()
    assert exc.status_code == 422
    assert exc.error_code == "UNPROCESSABLE_ENTITY"


def test_processing_exception_with_reason():
    """ProcessingException includes reason in details."""
    exc = ProcessingException(reason="Missing dependency")
    assert exc.details["reason"] == "Missing dependency"


# ── InternalServerException ──────────────────────────────────────────


def test_internal_server_exception_defaults():
    """InternalServerException has correct defaults."""
    exc = InternalServerException()
    assert exc.status_code == 500
    assert exc.error_code == "INTERNAL_ERROR"


# ── ContainerNotFoundException ───────────────────────────────────────


def test_container_not_found_default_message():
    """ContainerNotFoundException generates message from container_name."""
    exc = ContainerNotFoundException("python-runner")
    assert exc.status_code == 400
    assert "python-runner" in exc.detail["message"]
    assert exc.details["resource_type"] == "container"
    assert exc.details["resource_id"] == "python-runner"


def test_container_not_found_custom_message():
    """ContainerNotFoundException accepts custom message."""
    exc = ContainerNotFoundException("node-runner", message="Custom not found")
    assert exc.detail["message"] == "Custom not found"


# ── AgentConfigurationError (API) ────────────────────────────────────


def test_agent_config_error_api():
    """AgentConfigurationError (API) has correct defaults."""
    from exceptions.api_exceptions import AgentConfigurationError as APIAgentError

    exc = APIAgentError()
    assert exc.status_code == 400
    assert exc.error_code == "AGENT_CONFIG_ERROR"


# ── AITimeoutError (API) ─────────────────────────────────────────────


def test_ai_timeout_error_api():
    """AITimeoutError (API) has correct defaults."""
    from exceptions.api_exceptions import AITimeoutError as APIAITimeout

    exc = APIAITimeout()
    assert exc.status_code == 504
    assert exc.error_code == "AI_TIMEOUT"


# ── AIValidationError (API) ──────────────────────────────────────────


def test_ai_validation_error_api():
    """AIValidationError (API) has correct defaults."""
    from exceptions.api_exceptions import AIValidationError as APIAIValidation

    exc = APIAIValidation()
    assert exc.status_code == 422
    assert exc.error_code == "AI_VALIDATION_ERROR"


# ── Backward compatibility aliases ───────────────────────────────────


def test_backward_compatibility_aliases():
    """Backward compatibility aliases are set correctly."""
    from exceptions import (
        APIException,
        AuthorizationException,
        BadRequestException,
        ServerException,
        TimeoutException,
    )

    assert APIException is BaseAPIException
    assert AuthorizationException is PermissionException
    assert ServerException is InternalServerException
    assert BadRequestException is ValidationException
    assert TimeoutException is ProcessingException


# ── PydanticAIException ──────────────────────────────────────────────


def test_pydantic_ai_exception_basic():
    """PydanticAIException stores message, error_code, details."""
    exc = PydanticAIException("Test error", error_code="TEST", details={"key": "val"})
    assert exc.message == "Test error"
    assert exc.error_code == "TEST"
    assert exc.details == {"key": "val"}
    assert exc.cause is None


def test_pydantic_ai_exception_with_cause():
    """PydanticAIException stores cause exception."""
    cause = ValueError("underlying")
    exc = PydanticAIException("Wrapped", cause=cause)
    assert exc.cause is cause


def test_pydantic_ai_exception_str():
    """PydanticAIException __str__ includes all parts."""
    exc = PydanticAIException(
        "Error msg",
        error_code="EC",
        details={"d": 1},
        cause=RuntimeError("boom"),
    )
    s = str(exc)
    assert "Error msg" in s
    assert "Error code: EC" in s
    assert "Details:" in s
    assert "Caused by:" in s


def test_pydantic_ai_exception_str_minimal():
    """PydanticAIException __str__ with only message."""
    exc = PydanticAIException("Just message")
    assert str(exc) == "Just message"


def test_pydantic_ai_exception_to_dict():
    """PydanticAIException to_dict returns correct structure."""
    exc = PydanticAIException("Error", error_code="E1", details={"a": 1})
    d = exc.to_dict()
    assert d["type"] == "PydanticAIException"
    assert d["message"] == "Error"
    assert d["error_code"] == "E1"
    assert d["details"] == {"a": 1}
    assert d["cause"] is None


def test_pydantic_ai_exception_to_dict_with_cause():
    """PydanticAIException to_dict includes cause string."""
    cause = ValueError("bad value")
    exc = PydanticAIException("Wrapped", cause=cause)
    d = exc.to_dict()
    assert d["cause"] == "bad value"


# ── AgentConfigurationError (PydanticAI) ─────────────────────────────


def test_agent_configuration_error():
    """AgentConfigurationError stores agent_id and config_field."""
    from exceptions.pydantic_ai_exceptions import AgentConfigurationError as PAIAgentError

    exc = PAIAgentError("Config bad", agent_id="agent-1", config_field="model")
    assert exc.agent_id == "agent-1"
    assert exc.config_field == "model"
    assert exc.details["agent_id"] == "agent-1"
    assert exc.details["config_field"] == "model"


# ── ModelProviderError ───────────────────────────────────────────────


def test_model_provider_error():
    """ModelProviderError stores provider, model, status_code."""
    exc = ModelProviderError("Provider error", provider="openai", model="gpt-4", status_code=500)
    assert exc.provider == "openai"
    assert exc.model == "gpt-4"
    assert exc.status_code == 500
    assert exc.details["provider"] == "openai"
    assert exc.details["model"] == "gpt-4"


# ── APIKeyError ──────────────────────────────────────────────────────


def test_api_key_error():
    """APIKeyError sets error_code to API_KEY_ERROR."""
    exc = APIKeyError("Missing key", provider="anthropic")
    assert exc.error_code == "API_KEY_ERROR"
    assert exc.details["provider"] == "anthropic"


# ── RateLimitError ───────────────────────────────────────────────────


def test_rate_limit_error():
    """RateLimitError stores retry_after."""
    exc = RateLimitError("Too many requests", provider="openai", retry_after=30)
    assert exc.error_code == "RATE_LIMIT"
    assert exc.retry_after == 30
    assert exc.details["retry_after"] == 30


# ── ModelNotFoundError ───────────────────────────────────────────────


def test_model_not_found_error():
    """ModelNotFoundError sets error_code to MODEL_NOT_FOUND."""
    exc = ModelNotFoundError("No such model", provider="openai", model="gpt-5")
    assert exc.error_code == "MODEL_NOT_FOUND"
    assert exc.details["model"] == "gpt-5"


# ── TokenLimitError ──────────────────────────────────────────────────


def test_token_limit_error():
    """TokenLimitError stores token_count and token_limit."""
    exc = TokenLimitError(
        "Token limit",
        provider="openai",
        model="gpt-4",
        token_count=5000,
        token_limit=4096,
    )
    assert exc.error_code == "TOKEN_LIMIT"
    assert exc.token_count == 5000
    assert exc.token_limit == 4096
    assert exc.details["token_count"] == 5000
    assert exc.details["token_limit"] == 4096


# ── AnalysisError ────────────────────────────────────────────────────


def test_analysis_error():
    """AnalysisError stores analysis_type and test_id."""
    exc = AnalysisError("Analysis failed", analysis_type="coverage", test_id="t-123")
    assert exc.analysis_type == "coverage"
    assert exc.test_id == "t-123"
    assert exc.details["analysis_type"] == "coverage"
    assert exc.details["test_id"] == "t-123"


# ── AIValidationError (PydanticAI) ───────────────────────────────────


def test_ai_validation_error_pydantic():
    """AIValidationError stores field_name and field_value."""
    exc = AIValidationError("Bad input", field_name="code", field_value="xyz")
    assert exc.error_code == "VALIDATION_ERROR"
    assert exc.field_name == "code"
    assert exc.field_value == "xyz"
    assert exc.details["field_name"] == "code"
    assert exc.details["field_value"] == "xyz"


# ── AITimeoutError (PydanticAI) ──────────────────────────────────────


def test_ai_timeout_error_pydantic():
    """AITimeoutError stores timeout_seconds and operation."""
    exc = AITimeoutError("Timed out", timeout_seconds=30.0, operation="analysis")
    assert exc.error_code == "TIMEOUT"
    assert exc.timeout_seconds == 30.0
    assert exc.operation == "analysis"
    assert exc.details["timeout_seconds"] == 30.0
    assert exc.details["operation"] == "analysis"


# ── ResourceExhaustedError ───────────────────────────────────────────


def test_resource_exhausted_error():
    """ResourceExhaustedError stores resource_type."""
    exc = ResourceExhaustedError("No GPUs left", resource_type="gpu")
    assert exc.error_code == "RESOURCE_EXHAUSTED"
    assert exc.resource_type == "gpu"
    assert exc.details["resource_type"] == "gpu"


# ── RetryableError ───────────────────────────────────────────────────


def test_retryable_error_can_retry():
    """RetryableError.can_retry returns True when retries remain."""
    exc = RetryableError("Retry me", retry_count=1, max_retries=3)
    assert exc.can_retry() is True
    assert exc.retry_count == 1
    assert exc.max_retries == 3


def test_retryable_error_exhausted():
    """RetryableError.can_retry returns False when retries exhausted."""
    exc = RetryableError("No more", retry_count=3, max_retries=3)
    assert exc.can_retry() is False


def test_retryable_error_increment():
    """RetryableError.increment_retry increases count."""
    exc = RetryableError("Retry", retry_count=0, max_retries=3)
    exc.increment_retry()
    assert exc.retry_count == 1
    assert exc.details["retry_count"] == 1


# ── NonRetryableError ────────────────────────────────────────────────


def test_non_retryable_error():
    """NonRetryableError has NON_RETRYABLE error_code."""
    exc = NonRetryableError("Never retry")
    assert exc.error_code == "NON_RETRYABLE"


# ── is_retryable_error ──────────────────────────────────────────────


def test_is_retryable_retryable_error():
    """is_retryable_error returns True for RetryableError with retries left."""
    assert is_retryable_error(RetryableError("r", retry_count=0, max_retries=3)) is True


def test_is_retryable_retryable_error_exhausted():
    """is_retryable_error returns False for exhausted RetryableError."""
    assert is_retryable_error(RetryableError("r", retry_count=3, max_retries=3)) is False


def test_is_retryable_rate_limit():
    """is_retryable_error returns True for RateLimitError."""
    assert is_retryable_error(RateLimitError("rate limited")) is True


def test_is_retryable_non_retryable():
    """is_retryable_error returns False for NonRetryableError."""
    assert is_retryable_error(NonRetryableError("no")) is False


def test_is_retryable_api_key_error():
    """is_retryable_error returns False for APIKeyError."""
    assert is_retryable_error(APIKeyError("bad key")) is False


def test_is_retryable_model_not_found():
    """is_retryable_error returns False for ModelNotFoundError."""
    assert is_retryable_error(ModelNotFoundError("no model")) is False


def test_is_retryable_ai_validation_error():
    """is_retryable_error returns False for AIValidationError."""
    assert is_retryable_error(AIValidationError("bad input")) is False


def test_is_retryable_connection_error():
    """is_retryable_error returns True for ConnectionError."""
    assert is_retryable_error(ConnectionError("refused")) is True


def test_is_retryable_ai_timeout():
    """is_retryable_error returns True for AITimeoutError."""
    assert is_retryable_error(AITimeoutError("timeout")) is True


def test_is_retryable_provider_5xx():
    """is_retryable_error returns True for ModelProviderError with status_code >= 500."""
    exc = ModelProviderError("server error", status_code=502)
    assert is_retryable_error(exc) is True


def test_is_retryable_provider_4xx():
    """is_retryable_error returns False for ModelProviderError with status_code < 500."""
    exc = ModelProviderError("client error", status_code=400)
    assert is_retryable_error(exc) is False


def test_is_retryable_generic_exception():
    """is_retryable_error returns False for generic Exception."""
    assert is_retryable_error(Exception("random")) is False


# ── get_retry_delay ──────────────────────────────────────────────────


def test_get_retry_delay_rate_limit_with_retry_after():
    """get_retry_delay uses retry_after from RateLimitError."""
    exc = RateLimitError("limited", retry_after=45)
    assert get_retry_delay(exc, retry_count=0) == 45.0


def test_get_retry_delay_exponential_backoff():
    """get_retry_delay uses exponential backoff for other errors."""
    exc = ModelProviderError("error")
    assert get_retry_delay(exc, retry_count=0) == 1  # 2^0
    assert get_retry_delay(exc, retry_count=1) == 2  # 2^1
    assert get_retry_delay(exc, retry_count=2) == 4  # 2^2
    assert get_retry_delay(exc, retry_count=3) == 8  # 2^3


def test_get_retry_delay_max_60():
    """get_retry_delay caps at 60 seconds."""
    exc = ModelProviderError("error")
    assert get_retry_delay(exc, retry_count=10) == 60  # min(1024, 60)


# ── categorize_error ─────────────────────────────────────────────────


def test_categorize_api_key():
    assert categorize_error(APIKeyError("bad key")) == "authentication"


def test_categorize_rate_limit():
    assert categorize_error(RateLimitError("limited")) == "rate_limit"


def test_categorize_model_not_found():
    assert categorize_error(ModelNotFoundError("no model")) == "model_not_found"


def test_categorize_token_limit():
    assert categorize_error(TokenLimitError("too many tokens")) == "token_limit"


def test_categorize_timeout():
    assert categorize_error(AITimeoutError("timeout")) == "timeout"


def test_categorize_validation():
    assert categorize_error(AIValidationError("bad input")) == "validation"


def test_categorize_configuration():
    from exceptions.pydantic_ai_exceptions import AgentConfigurationError as PAIConfig

    assert categorize_error(PAIConfig("config bad")) == "configuration"


def test_categorize_provider():
    assert categorize_error(ModelProviderError("provider issue")) == "provider"


def test_categorize_analysis():
    assert categorize_error(AnalysisError("analysis failed")) == "analysis"


def test_categorize_resource_exhausted():
    assert categorize_error(ResourceExhaustedError("no resources")) == "resource_exhausted"


def test_categorize_unknown():
    assert categorize_error(RuntimeError("random")) == "unknown"


# ── create_error_response ────────────────────────────────────────────


def test_create_error_response_generic():
    """create_error_response returns correct structure for generic exception."""
    resp = create_error_response(RuntimeError("oops"), request_id="r-1", analysis_time_ms=42.0)
    assert resp["request_id"] == "r-1"
    assert resp["success"] is False
    assert resp["error_message"] == "oops"
    assert resp["error_category"] == "unknown"
    assert resp["total_analysis_time_ms"] == 42.0
    assert resp["retryable"] is False
    assert resp["metadata"]["error_type"] == "RuntimeError"


def test_create_error_response_pydantic_ai():
    """create_error_response includes PydanticAI-specific fields."""
    exc = APIKeyError("Invalid key", provider="openai")
    resp = create_error_response(exc, request_id="r-2")
    assert resp["error_category"] == "authentication"
    assert resp["error_code"] == "API_KEY_ERROR"
    assert resp["error_details"]["provider"] == "openai"
    assert resp["metadata"]["type"] == "APIKeyError"
    assert resp["retryable"] is False


def test_create_error_response_retryable():
    """create_error_response marks retryable errors."""
    exc = RateLimitError("Limited", retry_after=30)
    resp = create_error_response(exc, request_id="r-3")
    assert resp["retryable"] is True
