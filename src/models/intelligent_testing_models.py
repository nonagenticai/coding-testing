"""
Intelligent Testing API Models.

This module provides Pydantic models for intelligent testing endpoints
with container-scoped operations.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from models.testing_models import TestResult

# Inline definition replacing monolith import (from app.types)
SAFE_STRING_PATTERN = r"^[\s\S]*$"


class ContainerScopedTestPatternRequest(BaseModel):
    """Request model for container-scoped test pattern analysis."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    container_name: str = Field(
        ...,
        pattern=SAFE_STRING_PATTERN,
        description="Container name to scope pattern analysis",
    )
    test_results: list[TestResult] = Field(
        default_factory=list,
        description="Historical test results for pattern detection",
    )
    analysis_depth: str = Field(
        default="standard",
        pattern=SAFE_STRING_PATTERN,
        description="Analysis depth (quick, standard, deep)",
    )
    include_performance_patterns: bool = Field(
        default=True, description="Include performance pattern analysis"
    )
    include_failure_patterns: bool = Field(
        default=True, description="Include failure pattern analysis"
    )
    include_timing_patterns: bool = Field(
        default=True, description="Include timing pattern analysis"
    )


class ContainerScopedTestRecommendationRequest(BaseModel):
    """Request model for container-scoped test recommendations."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    container_name: str = Field(
        ...,
        pattern=SAFE_STRING_PATTERN,
        description="Container name to scope recommendations",
    )
    code_context: str | None = Field(
        None,
        pattern=SAFE_STRING_PATTERN,
        description="Code context for better recommendations",
    )
    language: str | None = Field(
        None,
        pattern=SAFE_STRING_PATTERN,
        description="Programming language for language-specific recommendations",
    )
    test_type: str | None = Field(
        None,
        pattern=SAFE_STRING_PATTERN,
        description="Test type (unit, integration, e2e, etc.)",
    )
    focus_areas: list[str] | None = Field(
        default=None, description="Specific areas to focus recommendations on"
    )
    priority_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Priority threshold (0.0-1.0)"
    )


class ContainerScopedSmartTestSelectionRequest(BaseModel):
    """Request model for container-scoped smart test selection."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    container_name: str = Field(
        ...,
        pattern=SAFE_STRING_PATTERN,
        description="Container name to scope test selection",
    )
    changed_files: list[str] = Field(..., description="List of files that have changed")
    test_files: list[str] | None = Field(
        default=None, description="Available test files (auto-detected if not provided)"
    )
    selection_strategy: str = Field(
        default="impact_based",
        pattern=SAFE_STRING_PATTERN,
        description="Selection strategy (impact_based, coverage_based, priority_based, hybrid)",
    )
    max_tests: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of tests to select (None = all relevant tests)",
    )
    include_related_tests: bool = Field(
        default=True, description="Include tests related to changed code"
    )


class TestPatternAnalysisResponse(BaseModel):
    """Response model for test pattern analysis."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    container_name: str = Field(
        ..., pattern=SAFE_STRING_PATTERN, description="Container name for the analysis"
    )
    patterns_detected: list[dict[str, Any]] = Field(..., description="List of detected patterns")
    pattern_count: int = Field(..., description="Total number of patterns detected")
    performance_patterns: list[dict[str, Any]] = Field(
        default_factory=list, description="Performance-related patterns"
    )
    failure_patterns: list[dict[str, Any]] = Field(
        default_factory=list, description="Failure-related patterns"
    )
    timing_patterns: list[dict[str, Any]] = Field(
        default_factory=list, description="Timing-related patterns"
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall confidence in pattern detection"
    )
    computed: bool = Field(
        default=True,
        description="False when the result is a heuristic placeholder or could not be derived from real data",
    )
    reason: str | None = Field(
        default=None, description="Why the result is not (fully) computed, when applicable"
    )
    analyzed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of analysis",
    )


class TestRecommendationResponse(BaseModel):
    """Response model for test recommendations."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    container_name: str = Field(
        ...,
        pattern=SAFE_STRING_PATTERN,
        description="Container name for recommendations",
    )
    recommendations: list[dict[str, Any]] = Field(..., description="List of test recommendations")
    recommendation_count: int = Field(..., description="Total number of recommendations")
    high_priority_count: int = Field(
        default=0, description="Number of high-priority recommendations"
    )
    medium_priority_count: int = Field(
        default=0, description="Number of medium-priority recommendations"
    )
    low_priority_count: int = Field(default=0, description="Number of low-priority recommendations")
    computed: bool = Field(
        default=True,
        description="False when the result is a heuristic placeholder or could not be derived from real data",
    )
    reason: str | None = Field(
        default=None, description="Why the result is not (fully) computed, when applicable"
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of recommendation generation",
    )


class SmartTestSelectionResponse(BaseModel):
    """Response model for smart test selection."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    container_name: str = Field(
        ...,
        pattern=SAFE_STRING_PATTERN,
        description="Container name for test selection",
    )
    selected_tests: list[str] = Field(..., description="List of selected test files")
    selection_count: int = Field(..., description="Number of selected tests")
    selection_strategy: str = Field(
        ..., pattern=SAFE_STRING_PATTERN, description="Strategy used for selection"
    )
    impact_analysis: dict[str, Any] = Field(
        default_factory=dict, description="Impact analysis details"
    )
    estimated_execution_time: float | None = Field(
        None, description="Estimated total execution time in seconds"
    )
    coverage_estimate: float | None = Field(
        None, ge=0.0, le=100.0, description="Estimated code coverage percentage"
    )
    computed: bool = Field(
        default=True,
        description="False when the result is a heuristic placeholder or could not be derived from real data",
    )
    reason: str | None = Field(
        default=None, description="Why the result is not (fully) computed, when applicable"
    )
    selected_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of test selection",
    )


class TestImpactAnalysisRequest(BaseModel):
    """Request model for test impact analysis."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    container_name: str = Field(
        ...,
        pattern=SAFE_STRING_PATTERN,
        description="Container name to scope impact analysis",
    )
    changed_files: list[str] = Field(..., description="List of files that have changed")
    analysis_depth: str = Field(
        default="standard",
        pattern=SAFE_STRING_PATTERN,
        description="Analysis depth (quick, standard, deep)",
    )
    include_indirect_impacts: bool = Field(
        default=True, description="Include indirect impacts (dependencies, etc.)"
    )


class TestImpactAnalysisResponse(BaseModel):
    """Response model for test impact analysis."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    container_name: str = Field(
        ..., pattern=SAFE_STRING_PATTERN, description="Container name for analysis"
    )
    changed_files: list[str] = Field(..., description="Files that were analyzed")
    impacted_tests: list[dict[str, Any]] = Field(
        ..., description="List of impacted tests with details"
    )
    impact_count: int = Field(..., description="Total number of impacted tests")
    direct_impacts: int = Field(default=0, description="Number of directly impacted tests")
    indirect_impacts: int = Field(default=0, description="Number of indirectly impacted tests")
    impact_score: float = Field(..., ge=0.0, le=10.0, description="Overall impact score (0-10)")
    status: Literal["computed", "not_computed"] = Field(
        default="computed",
        description=(
            "'not_computed' when the workspace could not be read completely "
            "(exec error, timeout, truncated output); lists are then empty"
        ),
    )
    reason: str | None = Field(default=None, description="Why the analysis was not computed")
    unknown_changed_files: list[str] = Field(
        default_factory=list,
        description="Changed Python paths that do not exist in the workspace",
    )
    unsupported_changed_files: list[str] = Field(
        default_factory=list,
        description="Changed non-Python files (import analysis is Python-only)",
    )
    analyzed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of impact analysis",
    )


class TestPrioritizationRequest(BaseModel):
    """Request model for test prioritization."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    container_name: str = Field(
        ...,
        pattern=SAFE_STRING_PATTERN,
        description="Container name to scope prioritization",
    )
    test_files: list[str] = Field(..., description="List of test files to prioritize")
    prioritization_criteria: list[str] = Field(
        default_factory=lambda: ["impact", "failure_history", "execution_time"],
        description="Criteria for prioritization",
    )
    changed_files: list[str] | None = Field(
        default=None, description="Recently changed files for context"
    )


class TestPrioritizationResponse(BaseModel):
    """Response model for test prioritization."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    container_name: str = Field(
        ...,
        pattern=SAFE_STRING_PATTERN,
        description="Container name for prioritization",
    )
    prioritized_tests: list[dict[str, Any]] = Field(
        ..., description="List of tests with priority scores"
    )
    total_tests: int = Field(..., description="Total number of tests prioritized")
    criteria_used: list[str] = Field(..., description="Criteria used for prioritization")
    computed: bool = Field(
        default=True,
        description="False when the result is a heuristic placeholder or could not be derived from real data",
    )
    reason: str | None = Field(
        default=None, description="Why the result is not (fully) computed, when applicable"
    )
    prioritized_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of prioritization",
    )
