"""
Comprehensive Pydantic v2 models for AI-powered test failure analysis.

This module provides type-safe, well-documented models for analyzing test failures
using PydanticAI. All models are designed with Pydantic v2 compliance and support
comprehensive error handling and validation.

The models support:
- Test failure input data and context
- AI-powered analysis results
- Root cause identification
- Intelligent fix suggestions
- Pattern detection and classification
- Confidence scoring and reasoning
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TestFramework(StrEnum):
    """Supported test frameworks for failure analysis."""

    PYTEST = "pytest"
    UNITTEST = "unittest"
    JEST = "jest"
    MOCHA = "mocha"
    JUNIT = "junit"
    XUNIT = "xunit"
    RSPEC = "rspec"
    UNKNOWN = "unknown"


class FailureType(StrEnum):
    """Types of test failures that can be analyzed."""

    ASSERTION_ERROR = "assertion_error"
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT_ERROR = "timeout_error"
    MEMORY_ERROR = "memory_error"
    NETWORK_ERROR = "network_error"
    DEPENDENCY_ERROR = "dependency_error"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN_ERROR = "unknown_error"


class SeverityLevel(StrEnum):
    """Severity levels for test failures and analysis results."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ConfidenceLevel(StrEnum):
    """Confidence levels for AI analysis results."""

    VERY_HIGH = "very_high"  # 90-100%
    HIGH = "high"  # 75-89%
    MEDIUM = "medium"  # 50-74%
    LOW = "low"  # 25-49%
    VERY_LOW = "very_low"  # 0-24%


class TestFileMetadata(BaseModel):
    """Metadata about a test file."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
        extra="forbid",
    )

    file_path: str = Field(..., description="Path to the test file", min_length=1, max_length=1000)

    file_size: int | None = Field(default=None, description="Size of the test file in bytes", ge=0)

    line_count: int | None = Field(
        default=None, description="Number of lines in the test file", ge=0
    )

    last_modified: datetime | None = Field(default=None, description="Last modification timestamp")

    framework: TestFramework = Field(
        default=TestFramework.UNKNOWN, description="Test framework used in the file"
    )

    test_count: int | None = Field(default=None, description="Number of tests in the file", ge=0)


class TestFailureContext(BaseModel):
    """Context information about a test failure."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
        extra="forbid",
    )

    test_name: str = Field(
        ..., description="Name of the failing test", min_length=1, max_length=200
    )

    test_class: str | None = Field(
        default=None, description="Test class name if applicable", max_length=200
    )

    test_method: str | None = Field(
        default=None, description="Test method name if applicable", max_length=200
    )

    test_file: TestFileMetadata | None = Field(
        default=None, description="Metadata about the test file"
    )

    source_file: str | None = Field(
        default=None,
        description="Path to the source file being tested",
        max_length=1000,
    )

    test_line_number: int | None = Field(
        default=None, description="Line number where the test failed", ge=1
    )

    framework: TestFramework = Field(
        default=TestFramework.UNKNOWN, description="Test framework used"
    )

    execution_environment: dict[str, Any] | None = Field(
        default=None, description="Environment details (OS, Python version, etc.)"
    )


class TestFailureData(BaseModel):
    """Raw test failure data and output."""

    model_config = ConfigDict(
        str_strip_whitespace=False,  # Preserve formatting in output
        validate_default=True,
        use_enum_values=True,
        extra="forbid",
    )

    failure_id: str = Field(
        ...,
        description="Unique identifier for this test failure",
        min_length=1,
        max_length=100,
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the test failure occurred",
    )

    raw_output: str = Field(
        ...,
        description="Raw test output including failure details",
        max_length=100_000,  # 100KB limit
    )

    exit_code: int | None = Field(default=None, description="Exit code from test execution")

    execution_time: float | None = Field(
        default=None, description="Test execution time in seconds", ge=0
    )

    memory_usage: int | None = Field(default=None, description="Memory usage in bytes", ge=0)

    context: TestFailureContext = Field(
        ..., description="Context information about the test failure"
    )

    @field_validator("failure_id")
    @classmethod
    def validate_failure_id(cls, v: str) -> str:
        """Validate that failure_id contains only safe characters."""
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "failure_id must contain only alphanumeric characters, hyphens, and underscores"
            )
        return v


class ErrorDetails(BaseModel):
    """Detailed information about a specific error."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
        extra="forbid",
    )

    error_type: FailureType = Field(..., description="Type of error that occurred")

    error_message: str = Field(..., description="Error message text", max_length=5000)

    traceback: str | None = Field(
        default=None, description="Full traceback if available", max_length=50_000
    )

    file_path: str | None = Field(
        default=None, description="File where the error occurred", max_length=1000
    )

    line_number: int | None = Field(
        default=None, description="Line number where the error occurred", ge=1
    )

    column_number: int | None = Field(
        default=None, description="Column number where the error occurred", ge=1
    )

    stack_trace: list[str] | None = Field(default=None, description="Parsed stack trace entries")


class AssertionDetails(BaseModel):
    """Details about assertion failures."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
        extra="forbid",
    )

    assertion_type: Literal[
        "assertEqual",
        "assertNotEqual",
        "assertTrue",
        "assertFalse",
        "assertIs",
        "assertIsNot",
        "assertIsNone",
        "assertIsNotNone",
        "assertIn",
        "assertNotIn",
        "assertRaises",
        "assertRaisesRegex",
        "assertAlmostEqual",
        "assertNotAlmostEqual",
        "assertGreater",
        "assertGreaterEqual",
        "assertLess",
        "assertLessEqual",
        "assertRegex",
        "assertNotRegex",
        "assertCountEqual",
        "assertMultiLineEqual",
        "assertSequenceEqual",
        "assertListEqual",
        "assertTupleEqual",
        "assertSetEqual",
        "assertDictEqual",
        "custom",
        "unknown",
    ] = Field(default="unknown", description="Type of assertion that failed")

    expected_value: str | None = Field(
        default=None, description="Expected value in the assertion", max_length=10_000
    )

    actual_value: str | None = Field(
        default=None,
        description="Actual value that caused the failure",
        max_length=10_000,
    )

    comparison_operator: str | None = Field(
        default=None,
        description="Comparison operator used (==, !=, <, >, etc.)",
        max_length=10,
    )

    assertion_message: str | None = Field(
        default=None,
        description="Custom assertion message if provided",
        max_length=1000,
    )

    value_difference: str | None = Field(
        default=None,
        description="Calculated difference between expected and actual values",
        max_length=10_000,
    )


class RootCause(BaseModel):
    """Identified root cause of a test failure."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
        extra="forbid",
    )

    cause_id: str = Field(
        ...,
        description="Unique identifier for this root cause",
        min_length=1,
        max_length=100,
    )

    category: Literal[
        "logic_error",
        "type_mismatch",
        "null_reference",
        "index_out_of_bounds",
        "configuration_issue",
        "dependency_missing",
        "environment_issue",
        "data_validation",
        "race_condition",
        "resource_exhaustion",
        "network_connectivity",
        "permission_denied",
        "timeout",
        "incompatible_versions",
        "missing_initialization",
        "state_corruption",
        "external_service_failure",
        "file_system_error",
        "unknown",
    ] = Field(..., description="Category of the root cause")

    description: str = Field(
        ...,
        description="Human-readable description of the root cause",
        min_length=10,
        max_length=1000,
    )

    severity: SeverityLevel = Field(..., description="Severity level of this root cause")

    confidence: ConfidenceLevel = Field(
        ..., description="Confidence level in this root cause identification"
    )

    confidence_score: float = Field(
        ..., description="Numerical confidence score (0.0 to 1.0)", ge=0.0, le=1.0
    )

    evidence: list[str] = Field(
        default_factory=list,
        description="Evidence supporting this root cause identification",
    )

    related_patterns: list[str] | None = Field(
        default=None, description="Related failure patterns or signatures"
    )

    @field_validator("cause_id")
    @classmethod
    def validate_cause_id(cls, v: str) -> str:
        """Validate that cause_id contains only safe characters."""
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "cause_id must contain only alphanumeric characters, hyphens, and underscores"
            )
        return v

    @model_validator(mode="after")
    def validate_confidence_consistency(self) -> RootCause:
        """Ensure confidence level matches confidence score."""
        score = self.confidence_score
        level = self.confidence

        if level == ConfidenceLevel.VERY_HIGH and not (0.9 <= score <= 1.0):
            raise ValueError("VERY_HIGH confidence must have score 0.9-1.0")
        elif level == ConfidenceLevel.HIGH and not (0.75 <= score < 0.9):
            raise ValueError("HIGH confidence must have score 0.75-0.89")
        elif level == ConfidenceLevel.MEDIUM and not (0.5 <= score < 0.75):
            raise ValueError("MEDIUM confidence must have score 0.5-0.74")
        elif level == ConfidenceLevel.LOW and not (0.25 <= score < 0.5):
            raise ValueError("LOW confidence must have score 0.25-0.49")
        elif level == ConfidenceLevel.VERY_LOW and not (0.0 <= score < 0.25):
            raise ValueError("VERY_LOW confidence must have score 0.0-0.24")

        return self


class FixSuggestion(BaseModel):
    """AI-generated suggestion for resolving a test failure."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
        extra="forbid",
    )

    suggestion_id: str = Field(
        ...,
        description="Unique identifier for this suggestion",
        min_length=1,
        max_length=100,
    )

    title: str = Field(
        ...,
        description="Brief title describing the suggestion",
        min_length=5,
        max_length=200,
    )

    description: str = Field(
        ...,
        description="Detailed description of the suggested action",
        min_length=20,
        max_length=2000,
    )

    fix_type: Literal[
        "code_change",
        "configuration_change",
        "dependency_update",
        "environment_setup",
        "test_modification",
        "data_correction",
        "infrastructure_change",
        "documentation_update",
        "process_change",
        "investigation_needed",
        "unknown",
    ] = Field(..., description="Type of action required to implement the suggestion")

    priority: Literal["critical", "high", "medium", "low"] = Field(
        ..., description="Priority level for implementing this suggestion"
    )

    confidence: ConfidenceLevel = Field(..., description="Confidence level in this suggestion")

    confidence_score: float = Field(
        ..., description="Numerical confidence score (0.0 to 1.0)", ge=0.0, le=1.0
    )

    estimated_effort: Literal["trivial", "low", "medium", "high", "very_high"] = Field(
        ..., description="Estimated effort required to implement"
    )

    code_examples: list[str] | None = Field(
        default=None, description="Code examples demonstrating the suggestion"
    )

    files_to_modify: list[str] | None = Field(
        default=None, description="List of files that need to be modified"
    )

    prerequisites: list[str] | None = Field(
        default=None, description="Prerequisites that must be met before implementing"
    )

    potential_side_effects: list[str] | None = Field(
        default=None,
        description="Potential side effects of implementing this suggestion",
    )

    verification_steps: list[str] | None = Field(
        default=None,
        description="Steps to verify the suggestion was implemented correctly",
    )

    @field_validator("suggestion_id")
    @classmethod
    def validate_suggestion_id(cls, v: str) -> str:
        """Validate that suggestion_id contains only safe characters."""
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "suggestion_id must contain only alphanumeric characters, hyphens, and underscores"
            )
        return v


class TestFailurePattern(BaseModel):
    """Pattern detected in test failures for classification."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
        extra="forbid",
    )

    pattern_id: str = Field(
        ...,
        description="Unique identifier for this pattern",
        min_length=1,
        max_length=100,
    )

    pattern_name: str = Field(
        ...,
        description="Human-readable name for the pattern",
        min_length=5,
        max_length=200,
    )

    pattern_signature: str = Field(
        ...,
        description="Signature or fingerprint of the pattern",
        min_length=10,
        max_length=500,
    )

    pattern_type: Literal[
        "error_message",
        "stack_trace",
        "assertion_pattern",
        "execution_flow",
        "timing_pattern",
        "resource_usage",
        "configuration_pattern",
        "dependency_pattern",
        "unknown",
    ] = Field(..., description="Type of pattern detected")

    frequency: int = Field(
        default=1, description="How many times this pattern has been observed", ge=1
    )

    confidence: ConfidenceLevel = Field(..., description="Confidence in pattern detection")

    confidence_score: float = Field(
        ..., description="Numerical confidence score (0.0 to 1.0)", ge=0.0, le=1.0
    )

    associated_failures: list[str] | None = Field(
        default=None, description="List of failure IDs that match this pattern"
    )

    typical_causes: list[str] | None = Field(
        default=None, description="Common causes associated with this pattern"
    )

    common_fixes: list[str] | None = Field(
        default=None, description="Common fixes for this pattern"
    )


class AIAnalysisResult(BaseModel):
    """Complete AI analysis result for a test failure."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
        extra="forbid",
    )

    analysis_id: str = Field(
        ...,
        description="Unique identifier for this analysis",
        min_length=1,
        max_length=100,
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the analysis was performed",
    )

    failure_data: TestFailureData = Field(..., description="Original test failure data")

    error_details: ErrorDetails = Field(..., description="Detailed error information")

    assertion_details: AssertionDetails | None = Field(
        default=None, description="Assertion-specific details if applicable"
    )

    root_causes: list[RootCause] = Field(
        default_factory=list, description="Identified root causes ranked by confidence"
    )

    fix_suggestions: list[FixSuggestion] = Field(
        default_factory=list,
        description="AI-generated suggestions for resolving the failure",
    )

    detected_patterns: list[TestFailurePattern] = Field(
        default_factory=list, description="Patterns detected in the failure"
    )

    analysis_confidence: ConfidenceLevel = Field(
        ..., description="Overall confidence in the analysis"
    )

    analysis_confidence_score: float = Field(
        ..., description="Numerical confidence score (0.0 to 1.0)", ge=0.0, le=1.0
    )

    processing_time: float | None = Field(
        default=None, description="Time taken to perform the analysis in seconds", ge=0
    )

    ai_model_version: str | None = Field(
        default=None,
        description="Version of the AI model used for analysis",
        max_length=100,
    )

    additional_metadata: dict[str, Any] | None = Field(
        default=None, description="Additional metadata from the analysis"
    )

    @field_validator("analysis_id")
    @classmethod
    def validate_analysis_id(cls, v: str) -> str:
        """Validate that analysis_id contains only safe characters."""
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "analysis_id must contain only alphanumeric characters, hyphens, and underscores"
            )
        return v

    @field_validator("root_causes")
    @classmethod
    def validate_root_causes_sorted(cls, v: list[RootCause]) -> list[RootCause]:
        """Ensure root causes are sorted by confidence score (highest first)."""
        if len(v) > 1:
            sorted_causes = sorted(v, key=lambda x: x.confidence_score, reverse=True)
            return sorted_causes
        return v

    @field_validator("fix_suggestions")
    @classmethod
    def validate_fix_suggestions_sorted(cls, v: list[FixSuggestion]) -> list[FixSuggestion]:
        """Ensure fix suggestions are sorted by confidence score (highest first)."""
        if len(v) > 1:
            sorted_suggestions = sorted(v, key=lambda x: x.confidence_score, reverse=True)
            return sorted_suggestions
        return v


class FailureAnalysisRequest(BaseModel):
    """Request model for AI-powered test failure analysis."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
        extra="forbid",
    )

    request_id: str = Field(
        ...,
        description="Unique identifier for this analysis request",
        min_length=1,
        max_length=100,
    )

    failure_data: TestFailureData = Field(..., description="Test failure data to analyze")

    analysis_options: dict[str, Any] | None = Field(
        default=None, description="Options to customize the analysis"
    )

    include_patterns: bool = Field(
        default=True, description="Whether to include pattern detection in the analysis"
    )

    include_suggestions: bool = Field(
        default=True, description="Whether to include fix suggestions in the analysis"
    )

    max_root_causes: int = Field(
        default=5, description="Maximum number of root causes to identify", ge=1, le=20
    )

    max_suggestions: int = Field(
        default=10,
        description="Maximum number of fix suggestions to generate",
        ge=1,
        le=50,
    )

    minimum_confidence: float = Field(
        default=0.1,
        description="Minimum confidence threshold for results",
        ge=0.0,
        le=1.0,
    )

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, v: str) -> str:
        """Validate that request_id contains only safe characters."""
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "request_id must contain only alphanumeric characters, hyphens, and underscores"
            )
        return v


class FailureAnalysisResponse(BaseModel):
    """Response model for AI-powered test failure analysis."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
        extra="forbid",
    )

    request_id: str = Field(
        ...,
        description="Request ID that generated this response",
        min_length=1,
        max_length=100,
    )

    success: bool = Field(..., description="Whether the analysis was successful")

    analysis_result: AIAnalysisResult | None = Field(
        default=None, description="Complete analysis result if successful"
    )

    error_message: str | None = Field(
        default=None, description="Error message if analysis failed", max_length=1000
    )

    warnings: list[str] | None = Field(
        default=None, description="Warning messages from the analysis"
    )

    processing_time: float = Field(..., description="Total processing time in seconds", ge=0)

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the response was generated",
    )
