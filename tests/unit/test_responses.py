"""Tests for shared.responses helpers."""

from fastapi import HTTPException

from shared.responses import (
    bad_request,
    conflict,
    forbidden,
    internal_error,
    not_found,
    success_response,
    unauthorized,
)


def test_success_response():
    """success_response returns correct structure."""
    resp = success_response(data={"items": [1, 2]}, message="OK")
    assert resp["status"] == "success"
    assert resp["message"] == "OK"
    assert resp["data"] == {"items": [1, 2]}


def test_success_response_default_message():
    """success_response has default message."""
    resp = success_response(data=None)
    assert resp["message"] == "Success"


def test_not_found():
    """not_found returns 404 HTTPException."""
    exc = not_found("Container", "test-abc")
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 404
    assert isinstance(exc.detail, dict) and "test-abc" in exc.detail["message"]


def test_bad_request():
    """bad_request returns 400 HTTPException."""
    exc = bad_request("Invalid input")
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 400


def test_internal_error():
    """internal_error returns 500 HTTPException."""
    exc = internal_error("Database failed")
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 500


def test_unauthorized():
    """unauthorized returns 401 HTTPException."""
    exc = unauthorized()
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 401


def test_forbidden():
    """forbidden returns 403 HTTPException."""
    exc = forbidden()
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 403


def test_conflict():
    """conflict returns 409 HTTPException."""
    exc = conflict("Container", "Already exists")
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 409


# ============================================================================
# Extended coverage: remaining response helpers
# ============================================================================

from shared.responses import (  # noqa: E402
    error_response,
    rate_limited,
    service_unavailable,
    timeout_error,
    validation_error,
)


def test_error_response_structure():
    """error_response returns HTTPException with structured detail."""
    exc = error_response(
        status_code=418,
        message="I'm a teapot",
        error_code="TEAPOT",
        details={"brew": "earl grey"},
    )
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 418
    assert exc.detail["status"] == "error"
    assert exc.detail["message"] == "I'm a teapot"
    assert exc.detail["error_code"] == "TEAPOT"
    assert exc.detail["details"]["brew"] == "earl grey"


def test_error_response_default_details():
    """error_response defaults details to empty dict."""
    exc = error_response(status_code=500, message="fail")
    assert exc.detail["details"] == {}


def test_validation_error():
    """validation_error returns 422 HTTPException."""
    exc = validation_error("Field must be positive", "amount")
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 422
    assert exc.detail["error_code"] == "VALIDATION_ERROR"
    assert exc.detail["details"]["field"] == "amount"


def test_validation_error_with_details():
    """validation_error merges extra details."""
    exc = validation_error("Bad", "name", details={"min_length": 3})
    assert exc.detail["details"]["field"] == "name"
    assert exc.detail["details"]["min_length"] == 3


def test_service_unavailable():
    """service_unavailable returns 503 HTTPException."""
    exc = service_unavailable()
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 503
    assert exc.detail["error_code"] == "SERVICE_UNAVAILABLE"


def test_service_unavailable_custom_message():
    """service_unavailable with custom message."""
    exc = service_unavailable("Maintenance mode")
    assert exc.detail["message"] == "Maintenance mode"


def test_timeout_error():
    """timeout_error returns 504 HTTPException."""
    exc = timeout_error()
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 504
    assert exc.detail["error_code"] == "TIMEOUT"


def test_timeout_error_custom_message():
    """timeout_error with custom message."""
    exc = timeout_error("Database query timed out")
    assert exc.detail["message"] == "Database query timed out"


def test_rate_limited():
    """rate_limited returns 429 HTTPException."""
    exc = rate_limited()
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 429
    assert exc.detail["error_code"] == "RATE_LIMITED"


def test_rate_limited_custom_message():
    """rate_limited with custom message."""
    exc = rate_limited("Too many API calls")
    assert exc.detail["message"] == "Too many API calls"


def test_not_found_error_code_format():
    """not_found error_code format replaces spaces with underscores."""
    exc = not_found("Test Result", "abc-123")
    assert exc.detail["error_code"] == "TEST_RESULT_NOT_FOUND"


def test_conflict_error_code_format():
    """conflict error_code format replaces spaces with underscores."""
    exc = conflict("Test Run", "duplicate")
    assert exc.detail["error_code"] == "TEST_RUN_CONFLICT"


def test_unauthorized_custom_message():
    """unauthorized with custom message."""
    exc = unauthorized("Token expired")
    assert exc.detail["message"] == "Token expired"


def test_forbidden_custom_message():
    """forbidden with custom message."""
    exc = forbidden("Admin role required")
    assert exc.detail["message"] == "Admin role required"


def test_bad_request_with_details():
    """bad_request with details dict."""
    exc = bad_request("Invalid format", details={"expected": "JSON"})
    assert exc.detail["details"]["expected"] == "JSON"
