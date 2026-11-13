"""
Shared models and dependencies for intelligent testing modules.
This module prevents circular imports by centralizing common dependencies.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import Depends
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from config.test_framework_config import TestConfig
from models.common_responses_models import LegacyBaseResponse
from models.testing_models import TestSecurityLevel
from services.stubs import DockerService
from services.test_execution.intelligent_test_integration import (
    IntelligentTestExecutionService,
    TestExecutionSettings,
    create_intelligent_test_service,
)
from services.test_execution.intelligent_test_orchestrator import (
    IntelligentTestExecutionResult,
)
from services.test_execution.pydantic_ai_agents import (
    ExecutionPattern,
    TestRecommendation,
)
from shared.core_dependencies import get_docker_service

from .utilities_and_helpers import TestRequestValidator

logger = logging.getLogger(__name__)


# Request/Response Models (shared across modules)
class IntelligentTestRequest(BaseModel):
    """Request model for intelligent test execution.

    Attributes:
        code (str): Class attribute code.
        language (str): Class attribute language.
        test_code (Optional[str]): Class attribute test_code.
        timeout (int): Class attribute timeout.
        security_level (TestSecurityLevel): Class attribute security_level.
        project_name (Optional[str]): Class attribute project_name.
        enable_ai_optimization (bool): Class attribute enable_ai_optimization.
        enable_pattern_analysis (bool): Class attribute enable_pattern_analysis.
        enable_failure_analysis (bool): Class attribute enable_failure_analysis.
        enable_scheduling_optimization (bool): Class attribute enable_scheduling_optimization.
        priority_level (int): Class attribute priority_level.
        expected_execution_time_ms (Optional[float]): Class attribute expected_execution_time_ms.
        resource_weight (float): Class attribute resource_weight.
        test_type (Optional[str]): Class attribute test_type.
        dependencies (Dict[str, str]): Class attribute dependencies.
        resource_limits (Optional[Dict[str, Any]]): Class attribute resource_limits.

    Example:
        >>> obj = IntelligentTestRequest()"""

    code: str = Field(..., description="Source code to test")
    language: str = Field(
        ...,
        description="Programming language",
        json_schema_extra={
            "enum": [
                "python",
                "javascript",
                "typescript",
                "java",
                "go",
                "rust",
                "csharp",
                "cpp",
                "nodejs",
            ]
        },
    )
    container_name: str = Field(
        ..., description="Container name for container-scoped test execution"
    )
    test_code: str | None = Field(None, description="Optional test code")
    timeout: int = Field(300, ge=30, le=3600, description="Execution timeout in seconds")
    security_level: TestSecurityLevel = Field(
        TestSecurityLevel.MEDIUM, description="Security level"
    )
    project_name: str | None = Field(None, description="Optional project name")

    # AI-specific settings
    enable_ai_optimization: bool = Field(True, description="Enable AI optimization")
    enable_pattern_analysis: bool = Field(True, description="Enable pattern analysis")
    enable_failure_analysis: bool = Field(True, description="Enable failure analysis")
    enable_scheduling_optimization: bool = Field(True, description="Enable scheduling optimization")
    priority_level: int = Field(5, ge=1, le=10, description="Priority level (1=lowest, 10=highest)")
    expected_execution_time_ms: float | None = Field(
        None, ge=0, description="Expected execution time in milliseconds"
    )
    resource_weight: float = Field(1.0, ge=0.1, le=10.0, description="Resource weight factor")

    # Test configuration
    test_type: str | None = Field(
        "unit", description="Type of test (unit, integration, benchmark, etc.)"
    )
    dependencies: dict[str, str] = Field(default_factory=dict, description="Test dependencies")
    resource_limits: dict[str, Any] | None = Field(None, description="Resource limits")

    @field_validator("language")
    @classmethod
    def validate_language(cls, v):
        """Validate supported languages using the modular validator."""
        return TestRequestValidator.validate_language(v)


class BatchTestRequest(BaseModel):
    """Request model for batch test execution."""

    tests: list[IntelligentTestRequest] = Field(..., description="List of test requests")
    enable_retry: bool = Field(True, description="Enable retry logic")
    max_concurrent: int | None = Field(None, ge=1, le=20, description="Maximum concurrent tests")


class TestExecutionResponse(LegacyBaseResponse):
    """Response model for test execution."""

    result: IntelligentTestExecutionResult | None = Field(None, description="Test execution result")
    execution_id: str = Field(..., description="Unique execution identifier")

    model_config = ConfigDict()


class BatchTestExecutionResponse(LegacyBaseResponse):
    """Response model for batch test execution."""

    results: list[IntelligentTestExecutionResult] = Field(..., description="Test execution results")
    execution_id: str = Field(..., description="Unique batch execution identifier")
    summary: dict[str, Any] = Field(..., description="Execution summary")

    model_config = ConfigDict()


class ExecutionInsightsResponse(LegacyBaseResponse):
    """Response model for execution insights."""

    insights: dict[str, Any] = Field(..., description="Comprehensive execution insights")
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of insights generation",
    )

    model_config = ConfigDict()

    @field_serializer("generated_at")
    def serialize_datetime(self, v: datetime | None) -> str | None:
        if v is None:
            return None
        return v.isoformat()


class RecommendationsResponse(LegacyBaseResponse):
    """Response model for recommendations."""

    recommendations: list[TestRecommendation] = Field(
        ..., description="AI-generated recommendations"
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of recommendations generation",
    )

    model_config = ConfigDict()

    @field_serializer("generated_at")
    def serialize_datetime(self, v: datetime | None) -> str | None:
        if v is None:
            return None
        return v.isoformat()


class PatternsResponse(LegacyBaseResponse):
    """Response model for patterns."""

    patterns: list[ExecutionPattern] = Field(..., description="Identified execution patterns")
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of patterns generation",
    )

    model_config = ConfigDict()

    @field_serializer("generated_at")
    def serialize_datetime(self, v: datetime | None) -> str | None:
        if v is None:
            return None
        return v.isoformat()


class HealthStatusResponse(LegacyBaseResponse):
    """Response model for health status."""

    health_status: dict[str, Any] = Field(..., description="Comprehensive health status")
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of health check",
    )

    model_config = ConfigDict()

    @field_serializer("generated_at")
    def serialize_datetime(self, v: datetime | None) -> str | None:
        if v is None:
            return None
        return v.isoformat()


class ServiceSettingsResponse(LegacyBaseResponse):
    """Response model for service settings."""

    settings: TestExecutionSettings = Field(..., description="Current service settings")
    applied_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of settings application",
    )

    model_config = ConfigDict()

    @field_serializer("applied_at")
    def serialize_datetime(self, v: datetime | None) -> str | None:
        if v is None:
            return None
        return v.isoformat()


# Global service instance
_service_instance: IntelligentTestExecutionService | None = None


async def get_intelligent_test_service(
    docker_service: DockerService = Depends(get_docker_service),
) -> IntelligentTestExecutionService:
    """Dependency to get or create intelligent test service."""
    global _service_instance

    if _service_instance is None:
        test_config = TestConfig()
        settings = TestExecutionSettings()
        _service_instance = create_intelligent_test_service(docker_service, test_config, settings)

        # Start service if not already running
        if not _service_instance.is_running:
            await _service_instance.start()

    return _service_instance
