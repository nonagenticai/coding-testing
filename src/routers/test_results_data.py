"""
API endpoints for accessing raw test execution data without AI analysis.

These endpoints provide programmatic access to test execution results,
enabling external orchestration layers (n8n, TypeScript flows) to retrieve
test data for AI analysis via external services like worker_management.

All endpoints are container-scoped to ensure data isolation between users.

**Key Features:**
- Container-scoped access control
- Raw test execution data (no embedded AI analysis)
- Historical test result queries
- Storage endpoint for externally-analyzed results
- Comprehensive error handling and validation

**Security:**
- All endpoints validate container ownership via `validate_and_get_container`
- No cross-container data leakage
- User can only access their own container data

**Usage in Orchestration:**
1. Execute test via `/api/tests/execute`
2. Retrieve raw data via `/api/tests/results/{container_name}/{test_id}` (this file)
3. Send to worker_management for AI analysis
4. Store analyzed results via `/api/tests/results/{container_name}/analyzed` (this file)
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import Any

from coding_client.auth.models import User
from fastapi import APIRouter, Depends, Header, HTTPException, Path, Query, Response, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from auth.tenant import require_tenant
from models.error_models import (
    Error400Response,
    Error401Response,
    Error403Response,
    Error404Response,
    Error500Response,
    ValidationErrorResponse,
)
from services.test_results_storage_service import (
    TestIdConflictError,
    get_test_results_storage_service,
)
from shared.container_validation import validate_and_get_container_for_tenant
from shared.dependencies import (
    requires_read_permission,
    requires_service_account,
    requires_write_permission,
)
from shared.responses import (
    internal_error,
    not_found,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# Response Models
# ============================================================================


class RawTestResultResponse(BaseModel):
    """
    Raw test execution result without AI analysis.

    This model provides all execution data needed for external AI analysis.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "test_id": "01936b2a-8f8e-7000-9a5d-0242ac130003",
                "container_name": "my-repo-main",
                "container_id": "abc123def456",
                "test_name": "test_user_authentication",
                "language": "python",
                "test_type": "unit",
                "exit_code": 1,
                "stdout": "===== test session starts =====\nFAILED tests/test_auth.py::test_user_authentication",
                "stderr": "AssertionError: None != <User: john>",
                "execution_time_ms": 1234.56,
                "source_code": "def authenticate_user(username, password):\n    # Missing return statement\n    user = User.objects.get(username=username)",
                "test_code": "def test_user_authentication():\n    user = authenticate_user('john', 'pass123')\n    assert user is not None",
                "file_path": "tests/test_auth.py",
                "dependencies": {"pytest": "7.4.0", "django": "4.2.0"},
                "resource_usage": {
                    "memory_mb": 128.5,
                    "cpu_percent": 15.2,
                    "peak_memory_mb": 145.3,
                },
                "timestamp": "2025-01-13T10:30:45.123Z",
                "test_framework": "pytest",
                "success": False,
            }
        }
    )

    # Core identification
    test_id: str = Field(..., description="Unique test execution ID (UUID v7)")
    container_name: str = Field(..., description="Container name (scoping/ownership)")
    container_id: str = Field(..., description="Docker container ID")

    # Test metadata
    test_name: str = Field(..., description="Name of the test that was executed")
    language: str = Field(
        ...,
        description="Programming language (python, javascript, typescript, go, rust)",
    )
    test_type: str = Field(..., description="Test type (unit, integration, benchmark, etc.)")
    test_framework: str | None = Field(
        None, description="Testing framework (pytest, jest, cargo test, etc.)"
    )

    # Execution results
    exit_code: int = Field(..., description="Process exit code (0 = success, non-zero = failure)")
    stdout: str = Field(..., description="Standard output from test execution")
    stderr: str = Field(..., description="Standard error from test execution")
    execution_time_ms: float = Field(..., description="Total execution time in milliseconds")
    success: bool = Field(..., description="Whether the test execution succeeded")

    # Code context (optional but recommended for AI analysis)
    source_code: str | None = Field(
        None, description="Source code being tested", max_length=100_000
    )
    test_code: str | None = Field(
        None, description="Test code that was executed", max_length=100_000
    )
    file_path: str | None = Field(
        None, description="Path to the test file (relative to container root)"
    )

    # Environment context
    dependencies: dict[str, str] | None = Field(
        None, description="Project dependencies with versions"
    )
    resource_usage: dict[str, float] | None = Field(
        None,
        description="Resource usage metrics (memory_mb, cpu_percent, peak_memory_mb, etc.)",
    )

    # Temporal
    timestamp: datetime = Field(..., description="When the test was executed (ISO 8601)")


class TestHistoryResponse(BaseModel):
    """Historical test results for pattern analysis."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "container_name": "my-repo-main",
                "total_tests": 45,
                "returned_tests": 10,
                "test_results": [
                    {
                        "test_id": "01936b2a-...",
                        "test_name": "test_user_authentication",
                        "exit_code": 1,
                        "success": False,
                        "timestamp": "2025-01-13T10:30:45.123Z",
                    }
                ],
                "filters_applied": {
                    "test_name": "test_user_authentication",
                    "language": "python",
                },
                "query_timestamp": "2025-01-13T10:35:00.000Z",
            }
        }
    )

    container_name: str = Field(..., description="Container name for scoping")
    total_tests: int = Field(..., description="Total number of test executions matching filters")
    returned_tests: int = Field(..., description="Number of test results returned in this response")
    test_results: list[dict[str, Any]] = Field(..., description="List of historical test results")
    filters_applied: dict[str, Any] = Field(
        ..., description="Filters that were applied to the query"
    )
    query_timestamp: datetime = Field(..., description="When this query was executed")


class AnalyzedTestResultRequest(BaseModel):
    """
    Request model for storing externally-analyzed test results.

    Use this endpoint to store AI analysis results generated by
    external services (worker_management, custom LLM orchestration, etc.).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "test_id": "01936b2a-8f8e-7000-9a5d-0242ac130003",
                "analysis_id": "01936b2b-1234-7000-9a5d-0242ac130004",
                "root_causes": [
                    {
                        "category": "semantic",
                        "description": "Missing return statement in authenticate_user function",
                        "severity": "high",
                        "confidence": "high",
                        "confidence_score": 0.95,
                    }
                ],
                "fix_suggestions": [
                    {
                        "title": "Add return statement",
                        "description": "Add 'return user' at the end of authenticate_user function",
                        "fix_type": "code_change",
                        "confidence": "high",
                    }
                ],
                "detected_patterns": [],
                "analysis_metadata": {
                    "model_provider": "deepseek",
                    "model_name": "deepseek-chat",
                    "analysis_duration_ms": 4230.5,
                    "total_agents_used": 5,
                },
            }
        }
    )

    test_id: str = Field(..., description="Test ID this analysis corresponds to (UUID v7)")
    analysis_id: str = Field(..., description="Unique analysis ID (UUID v7)")

    # Analysis results (flexible structure to support various AI outputs)
    root_causes: list[dict[str, Any]] = Field(
        default_factory=list, description="Identified root causes"
    )
    fix_suggestions: list[dict[str, Any]] = Field(
        default_factory=list, description="Fix suggestions"
    )
    detected_patterns: list[dict[str, Any]] = Field(
        default_factory=list, description="Detected failure patterns"
    )

    # Analysis metadata
    analysis_metadata: dict[str, Any] = Field(
        ..., description="Metadata about the analysis (model, provider, timing, etc.)"
    )


class AnalyzedTestResultResponse(BaseModel):
    """Response after storing analyzed test results."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "test_id": "01936b2a-8f8e-7000-9a5d-0242ac130003",
                "analysis_id": "01936b2b-1234-7000-9a5d-0242ac130004",
                "container_name": "my-repo-main",
                "stored_at": "2025-01-13T10:35:00.000Z",
                "message": "Analysis results stored successfully",
            }
        }
    )

    success: bool = Field(..., description="Whether storage succeeded")
    test_id: str = Field(..., description="Test ID")
    analysis_id: str = Field(..., description="Analysis ID")
    container_name: str = Field(..., description="Container name")
    stored_at: datetime = Field(..., description="When the analysis was stored")
    message: str = Field(..., description="Status message")


# ============================================================================
# API Endpoints
# ============================================================================


@router.get(
    "/results/{container_name}/{test_id}",
    operation_id="get_raw_test_result",
    response_model=RawTestResultResponse,
    summary="Get raw test execution data",
    description="""
    Retrieve raw test execution data for a specific test without AI analysis.

    This endpoint provides all execution data needed for external AI analysis:
    - Test output (stdout, stderr)
    - Exit code and execution time
    - Source code and test code (if available)
    - Dependencies and resource usage
    - Container context

    **Container-Scoped**: Only returns data if the requesting user owns the container.

    **Use Case**: Retrieve test data → Send to worker_management for AI analysis
    """,
    responses={
        200: {
            "description": "Successfully retrieved raw test data",
            "model": RawTestResultResponse,
        },
        401: {"description": "Authentication required", "model": Error401Response},
        403: {
            "description": "Insufficient permissions or container access denied",
            "model": Error403Response,
        },
        404: {"description": "Container or test not found", "model": Error404Response},
        500: {"description": "Internal server error", "model": Error500Response},
    },
)
async def get_raw_test_result(
    container_name: str = Path(..., description="Container name for scoping"),
    test_id: str = Path(..., description="Test execution ID (UUID)"),
    current_user: User = Depends(requires_read_permission),
    tenant_id: str = Depends(require_tenant),
) -> RawTestResultResponse:
    """
    Get raw test execution data without AI analysis.

    This endpoint is designed for external orchestration layers that need
    raw test data to perform AI analysis via external services.

    **Security**: Validates that the user owns the specified container.

    Args:
        container_name: Container name (ownership validation)
        test_id: Unique test execution ID
        current_user: Authenticated user (from dependency)

    Returns:
        Raw test execution data with all context needed for AI analysis

    Raises:
        HTTPException: 403 if container access denied, 404 if test not found
    """
    try:
        logger.info(f"Fetching raw test result: test_id={test_id}, container={container_name}")

        # Validate container ownership with tenant isolation
        await validate_and_get_container_for_tenant(container_name, tenant_id)

        # Retrieve test result from storage
        storage_service = get_test_results_storage_service()
        result_data = await storage_service.get_test_result(container_name, test_id)

        if not result_data:
            logger.warning(f"Test result not found: test_id={test_id}, container={container_name}")
            raise not_found("Test result", f"test_id={test_id} in container={container_name}")

        # Transform to response model
        return RawTestResultResponse(
            test_id=result_data["test_id"],
            container_name=result_data["container_name"],
            container_id=result_data.get("container_id", ""),
            test_name=result_data["test_name"],
            language=result_data["language"],
            test_type=result_data["test_type"],
            test_framework=result_data.get("test_framework"),
            exit_code=result_data["exit_code"],
            stdout=result_data["stdout"],
            stderr=result_data["stderr"],
            execution_time_ms=result_data["execution_time_ms"],
            success=result_data["success"],
            source_code=result_data.get("source_code"),
            test_code=result_data.get("test_code"),
            file_path=result_data.get("file_path"),
            dependencies=result_data.get("dependencies"),
            resource_usage=result_data.get("resource_usage"),
            timestamp=datetime.fromisoformat(result_data["timestamp"]),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching raw test result: {str(e)}")
        raise internal_error(f"Failed to fetch test result: {str(e)}")


@router.get(
    "/history/{container_name}",
    operation_id="get_test_history",
    response_model=TestHistoryResponse,
    summary="Get historical test results",
    description="""
    Retrieve historical test execution results for pattern analysis.

    Supports filtering by:
    - Test name
    - Language
    - Success/failure status
    - Time range
    - Test type

    **Container-Scoped**: Only returns tests from the specified container.

    **Use Case**: Retrieve historical patterns → Send to worker_management for pattern detection
    """,
    responses={
        200: {
            "description": "Successfully retrieved test history",
            "model": TestHistoryResponse,
        },
        401: {"description": "Authentication required", "model": Error401Response},
        403: {
            "description": "Insufficient permissions or container access denied",
            "model": Error403Response,
        },
        404: {"description": "Container not found", "model": Error404Response},
        500: {"description": "Internal server error", "model": Error500Response},
    },
)
async def get_test_history(
    container_name: str = Path(..., description="Container name for scoping"),
    test_name: str | None = Query(None, description="Filter by test name (exact match)"),
    language: str | None = Query(None, description="Filter by programming language"),
    success: bool | None = Query(None, description="Filter by success status (true/false)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip (for pagination)"),
    current_user: User = Depends(requires_read_permission),
    tenant_id: str = Depends(require_tenant),
) -> TestHistoryResponse:
    """
    Get historical test execution results for pattern analysis.

    This endpoint enables external AI services to analyze historical test
    patterns and learn from past failures.

    **Security**: Validates that the user owns the specified container.

    Args:
        container_name: Container name (ownership validation)
        test_name: Optional filter by test name
        language: Optional filter by programming language
        success: Optional filter by success status
        limit: Maximum number of results (default 100, max 1000)
        offset: Pagination offset (default 0)
        current_user: Authenticated user (from dependency)

    Returns:
        Historical test results matching the specified filters

    Raises:
        HTTPException: 403 if container access denied, 404 if container not found
    """
    try:
        logger.info(f"Fetching test history for container={container_name}")

        # Validate container ownership with tenant isolation
        await validate_and_get_container_for_tenant(container_name, tenant_id)

        # Retrieve test history from storage
        storage_service = get_test_results_storage_service()
        history_data = await storage_service.get_test_history(
            container_name=container_name,
            test_name=test_name,
            language=language,
            success=success,
            limit=limit,
            offset=offset,
        )

        return TestHistoryResponse(**history_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching test history: {str(e)}")
        raise internal_error(f"Failed to fetch test history: {str(e)}")


@router.post(
    "/results/{container_name}/analyzed",
    operation_id="store_analyzed_test_result",
    response_model=AnalyzedTestResultResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Store externally-analyzed test results",
    description="""
    Store AI analysis results generated by external services.

    Use this endpoint to persist analysis results from:
    - worker_management AI workers
    - n8n workflows with LLM nodes
    - Custom TypeScript/Python orchestration flows
    - External AI analysis services

    **Container-Scoped**: Analysis is associated with the container's test results.

    **Use Case**: External AI analysis → Store results → Display in code_graph_helper UI
    """,
    responses={
        201: {
            "description": "Analysis results stored successfully",
            "model": AnalyzedTestResultResponse,
        },
        400: {"description": "Invalid request data", "model": Error400Response},
        401: {"description": "Authentication required", "model": Error401Response},
        403: {
            "description": "Insufficient permissions or container access denied",
            "model": Error403Response,
        },
        404: {"description": "Container or test not found", "model": Error404Response},
        422: {"description": "Validation error", "model": ValidationErrorResponse},
        500: {"description": "Internal server error", "model": Error500Response},
    },
)
async def store_analyzed_result(
    request: AnalyzedTestResultRequest,
    container_name: str = Path(..., description="Container name for scoping"),
    current_user: User = Depends(requires_write_permission),
    tenant_id: str = Depends(require_tenant),
) -> AnalyzedTestResultResponse:
    """
    Store externally-analyzed test results.

    This endpoint enables external orchestration layers to persist AI analysis
    results back into code_graph_helper for display, reporting, and historical tracking.

    **Security**: Validates that the user owns the specified container.

    **Workflow Example:**
    1. Execute test via `/api/tests/execute`
    2. Retrieve raw data via `/api/tests/results/{container_name}/{test_id}`
    3. Send to worker_management for AI analysis
    4. Store results via this endpoint
    5. Results now available in code_graph_helper UI and APIs

    Args:
        container_name: Container name (ownership validation)
        request: Analysis results with root causes, fix suggestions, patterns
        current_user: Authenticated user (from dependency)

    Returns:
        Confirmation of successful storage with metadata

    Raises:
        HTTPException: 403 if container access denied, 404 if test not found, 422 if invalid data
    """
    try:
        logger.info(
            f"Storing analyzed result: test_id={request.test_id}, container={container_name}"
        )

        # Validate container ownership with tenant isolation
        await validate_and_get_container_for_tenant(container_name, tenant_id)

        # Store analysis result
        storage_service = get_test_results_storage_service()
        try:
            await storage_service.store_analysis_result(
                container_name=container_name,
                test_id=request.test_id,
                analysis_id=request.analysis_id,
                root_causes=request.root_causes,
                fix_suggestions=request.fix_suggestions,
                detected_patterns=request.detected_patterns,
                analysis_metadata=request.analysis_metadata,
            )
        except ValueError as e:
            # Test not found in this container
            logger.warning(f"Test not found for analysis storage: {str(e)}")
            raise not_found("Test", f"{request.test_id} in container {container_name}")

        return AnalyzedTestResultResponse(
            success=True,
            test_id=request.test_id,
            analysis_id=request.analysis_id,
            container_name=container_name,
            stored_at=datetime.now(UTC),
            message="Analysis results stored successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error storing analyzed result: {str(e)}")
        raise internal_error(f"Failed to store analysis result: {str(e)}")


# ============================================================================
# Computed (externally executed) results
# ============================================================================


# Text fields derived from a junit summary are capped so one runaway failure list cannot bloat a row.
_DERIVED_TEXT_CAP = 4000
_DERIVED_FIELDS = ("success", "execution_time_ms", "stdout", "stderr", "resource_usage")


def _cap(text: str) -> str:
    if len(text) <= _DERIVED_TEXT_CAP:
        return text
    over = len(text) - _DERIVED_TEXT_CAP
    return text[:_DERIVED_TEXT_CAP] + f"…[truncated {over} chars]"


class JunitSummary(BaseModel):
    """A parsed junit report as the conveyor's compute.junit.parse emits it (unknown keys ignored)."""

    model_config = ConfigDict(extra="ignore")

    tests: int = Field(..., ge=0)
    failures: int = Field(..., ge=0)
    errors: int = Field(..., ge=0)
    skipped: int = Field(0, ge=0)
    passed: int | None = Field(None, ge=0)
    time: float | None = Field(None, ge=0)
    failing: list[str] = Field(default_factory=list)
    errored: list[str] = Field(default_factory=list)
    reason: str | None = None

    @model_validator(mode="after")
    def _counts_are_consistent(self) -> JunitSummary:
        if self.passed is not None and self.passed > self.tests:
            raise ValueError(f"junit_summary.passed ({self.passed}) exceeds tests ({self.tests})")
        if self.failures + self.errors + self.skipped > self.tests:
            raise ValueError(
                "junit_summary failures + errors + skipped "
                f"({self.failures + self.errors + self.skipped}) exceeds tests ({self.tests})"
            )
        return self


def derive_computed_test_id(
    request: ComputedTestResultRequest, *, tenant_id: str, container_name: str
) -> str:
    """The idempotency key for a summary-only body: ``{source}:{card_id}:{sha256(...)[:32]}``.

    Derived in the route, not the model, because tenant and container are not body
    fields. Both are hashed so another tenant cannot pre-empt this tenant's row
    (409 write-DoS) and a retry in the same scope collapses to one row.
    """
    summary = request.junit_summary
    canonical = json.dumps(
        {
            "tenant_id": tenant_id,
            "container_name": container_name,
            "commit_sha": request.commit_sha,
            "exit_code": request.exit_code,
            "junit_summary": summary.model_dump() if summary is not None else None,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:32]
    return f"{request.source}:{request.card_id}:{digest}"


class ComputedTestResultRequest(BaseModel):
    """A test outcome computed outside this service (e.g. conveyor QA).

    ``tenant_id`` and ``project_id`` are intentionally absent: the tenant comes from
    the authenticated caller and the project from the ``X-Project-Id`` header (sent
    by the n8n node and the gateway on every call), never from the body.

    Two input forms:

    * explicit: ``success`` / ``execution_time_ms`` / ``stdout`` / ``stderr`` /
      ``resource_usage`` sent by the caller;
    * ``junit_summary``: the raw parsed report; the service DERIVES those five fields
      (success = tests > 0 and failures + errors == 0 and passed > 0 and exit_code == 0 —
      a run that executed nothing, or whose runner exited non-zero, is never a pass). Mixing the two is a 422 — the
      generated n8n node omits false / 0 / "" values, so an explicit value next to a
      summary is ambiguous and must fail loudly.

    ``test_id`` may be omitted only with a ``junit_summary`` and a ``card_id``: it is then
    derived by the route (:func:`derive_computed_test_id`) as
    ``{source}:{card_id}:{sha256(tenant_id, container_name, commit_sha, exit_code, junit_summary)[:32]}``,
    so an identical re-run in the same scope collapses to one row and a different outcome gets its own.
    """

    model_config = ConfigDict(extra="forbid")

    test_id: str | None = Field(
        None,
        min_length=1,
        description="Idempotency key, e.g. aaqa:{card}:{corr}; derived when omitted (needs junit_summary + card_id)",
    )
    test_name: str = Field("computed", description="Name of the test run")
    language: str = Field("python", description="Programming language")
    test_type: str = Field("unit", description="Test type")
    test_framework: str | None = Field(None, description="Testing framework (pytest, ...)")
    exit_code: int = Field(..., description="Process exit code")
    # Optional, default False: the generated n8n node only sends booleans when
    # true (n8n_module_scripts#130), so a failed run omits the field. Absence of
    # a pass claim is not a pass.
    success: bool = Field(False, description="Whether the run passed (omitted = false)")
    execution_time_ms: float = Field(0.0, ge=0, description="Wall time in milliseconds")
    stdout: str = Field("", description="Captured / summarised stdout")
    stderr: str = Field("", description="Captured / summarised stderr")
    resource_usage: dict[str, Any] | None = Field(None, description="Counts / resource data")
    junit_summary: JunitSummary | None = Field(
        None,
        description="Raw parsed junit report; when present the service derives success, timing, stdout, stderr, resource_usage",
    )
    card_id: str | None = Field(None, description="Conveyor card id")
    commit_sha: str | None = Field(None, description="Commit the tests ran against")
    correlation_id: str | None = Field(None, description="Run correlation id")
    source: str = Field("aa_qa", min_length=1, description="Producer discriminator")

    @model_validator(mode="after")
    def _derive_from_summary(self) -> ComputedTestResultRequest:
        j = self.junit_summary
        if j is None:
            if self.test_id is None:
                raise ValueError("test_id is required unless junit_summary and card_id are sent")
            return self
        mixed = sorted(f for f in _DERIVED_FIELDS if f in self.model_fields_set)
        if mixed:
            raise ValueError(
                f"junit_summary cannot be combined with explicit {mixed}: they are derived from it"
            )
        passed = (
            j.passed
            if j.passed is not None
            else max(j.tests - j.failures - j.errors - j.skipped, 0)
        )
        # Not run is never a pass (cf. models.testing_models.enforce_not_run_is_never_a_pass):
        # an all-skipped suite has passed == 0, and a non-zero runner exit (coverage gate,
        # internal error after a partial report, a stale junit) overrides a clean report.
        self.success = (
            j.tests > 0 and j.failures + j.errors == 0 and passed > 0 and self.exit_code == 0
        )
        self.execution_time_ms = round((j.time or 0.0) * 1000, 3)
        out = j.reason or ""
        if j.failing:
            out = (out + "\n" if out else "") + "failing:\n" + "\n".join(j.failing)
        self.stdout = _cap(out)
        self.stderr = _cap("\n".join(j.errored))
        self.resource_usage = {
            "tests": j.tests,
            "failures": j.failures,
            "errors": j.errors,
            "skipped": j.skipped,
            "passed": passed,
        }
        if self.test_id is None and not self.card_id:
            raise ValueError(
                "test_id may be omitted only when card_id is sent (it keys the derived id)"
            )
        return self


class ComputedTestResultResponse(BaseModel):
    """Stored computed result plus whether this call created it."""

    created: bool
    result: dict[str, Any]


@router.post(
    "/results/{container_name}/computed",
    operation_id="store_computed_test_result",
    response_model=ComputedTestResultResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Store an externally computed test result (service accounts only)",
    description="""
    Persist a test outcome that was executed and graded outside this service
    (the conveyor's QA step), attributed to its card.

    **Service accounts only.** The project is the `X-Project-Id` header (required,
    422 without it). **Idempotent on test_id**: a repeat returns the
    stored row with `created=false` and HTTP 200, never a duplicate.
    """,
    responses={
        200: {"description": "Already stored (idempotent repeat)"},
        201: {"description": "Stored"},
        401: {"description": "Authentication required", "model": Error401Response},
        403: {
            "description": "Not a service account, or container belongs to another tenant",
            "model": Error403Response,
        },
        404: {"description": "Container not found", "model": Error404Response},
        409: {"description": "test_id already used in another container/tenant"},
        422: {"description": "Validation error", "model": ValidationErrorResponse},
        500: {"description": "Internal server error", "model": Error500Response},
    },
)
async def store_computed_result(
    request: ComputedTestResultRequest,
    response: Response,
    container_name: str = Path(..., description="Container name for scoping"),
    # Not in the OpenAPI: like X-Tenant-Id it is a call-wide header the n8n node and
    # the gateway send on every request, not a per-operation input.
    project_id: str | None = Header(None, alias="X-Project-Id", include_in_schema=False),
    current_user: User = Depends(requires_service_account),
    tenant_id: str = Depends(require_tenant),
) -> ComputedTestResultResponse:
    """Store an externally computed test result, idempotently."""
    if not (project_id or "").strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="X-Project-Id header is required: the project this test result belongs to",
        )
    container = await validate_and_get_container_for_tenant(container_name, tenant_id)
    if container.tenant_id and container.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Container '{container_name}' does not belong to this tenant",
        )

    # Sent, or derived here: the model validator guarantees junit_summary + card_id when omitted.
    test_id = request.test_id or derive_computed_test_id(
        request, tenant_id=tenant_id, container_name=container_name
    )
    storage_service = get_test_results_storage_service()
    try:
        row, created = await storage_service.store_computed_test_result(
            test_id=test_id,
            container_name=container_name,
            tenant_id=tenant_id,
            test_name=request.test_name,
            language=request.language,
            test_type=request.test_type,
            test_framework=request.test_framework,
            exit_code=request.exit_code,
            stdout=request.stdout,
            stderr=request.stderr,
            execution_time_ms=request.execution_time_ms,
            success=request.success,
            resource_usage=request.resource_usage,
            card_id=request.card_id,
            project_id=project_id.strip(),
            commit_sha=request.commit_sha,
            correlation_id=request.correlation_id,
            source=request.source,
        )
    except TestIdConflictError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"test_id '{test_id}' already exists in another scope",
        )
    except Exception as e:
        logger.error(f"Error storing computed result: {str(e)}")
        raise internal_error(f"Failed to store computed test result: {str(e)}")

    if not created:
        response.status_code = status.HTTP_200_OK
    logger.info(
        f"Computed result test_id={test_id} container={container_name} "
        f"created={created} by {current_user.preferred_username}"
    )
    return ComputedTestResultResponse(created=created, result=row)


# ============================================================================
# Health Check
# ============================================================================


@router.get(
    "/health",
    operation_id="test_results_data_health_check",
    summary="Health check for test results data endpoints",
    description="Check if test results data API is available and responding.",
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "service": "test_results_data_api",
                        "endpoints": [
                            "GET /results/{container_name}/{test_id}",
                            "GET /history/{container_name}",
                            "POST /results/{container_name}/analyzed",
                            "POST /results/{container_name}/computed",
                        ],
                        "timestamp": "2025-01-13T10:30:45.123Z",
                    }
                }
            },
        }
    },
)
async def health_check() -> dict[str, Any]:
    """
    Health check for test results data API.

    Returns:
        Health status and available endpoints
    """
    return {
        "status": "healthy",
        "service": "test_results_data_api",
        "endpoints": [
            "GET /results/{container_name}/{test_id}",
            "GET /history/{container_name}",
            "POST /results/{container_name}/analyzed",
            "POST /results/{container_name}/computed",
        ],
        "timestamp": datetime.now(UTC).isoformat(),
    }
