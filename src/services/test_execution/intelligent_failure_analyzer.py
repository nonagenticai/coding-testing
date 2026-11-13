"""Intelligent test failure analysis service.

Routes through inference-router via :class:`TestFailureAgent`. The previous
implementation built two provider-pinned agents (primary OpenAI / fallback
Anthropic) and toggled them client-side; that's now done server-side by
inference-router based on the routing rule registered for
``test_failure_analysis``.

What this layer adds on top of :class:`TestFailureAgent`:

* Translation of :class:`TestResult` into a prompt the LLM can act on.
* Optional caching of past responses keyed on test characteristics.
* Capture of inference-router metadata
  (:func:`coding_client.llm.last_inference_id`,
  :func:`coding_client.llm.last_failover_chain_id`) into the response
  payload so audit logs / DB rows can carry provenance.
* Health check + summary for the readiness endpoint.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid7

from coding_client.llm import (
    LLMEmptyResponseError,
    TestFailureAnalysisResponse,
    last_failover_chain_id,
    last_inference_id,
    llm_request_context,
)
from pydantic import BaseModel, ConfigDict, Field

from models.pydantic_ai_models import (
    AIAnalysisRequest,
    AIAnalysisResponse,
    PydanticAIAnalysisType,
    PydanticAITestFailureContext,
)
from models.testing_models import TestResult
from services.test_execution.failure_agent import TestFailureAgent

logger = logging.getLogger(__name__)


class FailureAnalysisConfig(BaseModel):
    """Configuration for the intelligent failure analyzer.

    Provider/model selection is delegated to inference-router. The fields
    here only control client-side behaviour: enablement, caching, optional
    secondary analyses (which are local heuristics, not separate LLM
    calls), and the per-request timeout that the LLM client honours via
    :class:`coding_client.llm.LLMSettings`.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        frozen=False,
    )

    enabled: bool = Field(default=True, description="Whether analysis is enabled")
    timeout_seconds: int = Field(default=30, description="Per-request timeout (seconds)")
    max_tokens: int = Field(default=2000, description="Soft target — passed via prompt only")
    enable_code_fixes: bool = Field(default=True, description="Include code-fix block in prompt")
    enable_performance_analysis: bool = Field(
        default=True, description="Include performance heuristics"
    )
    enable_security_analysis: bool = Field(default=True, description="Include security heuristics")
    cache_results: bool = Field(default=True, description="Cache analysis results")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")


class IntelligentFailureAnalyzer:
    """Service for intelligent test failure analysis using inference-router."""

    def __init__(
        self,
        config: FailureAnalysisConfig | None = None,
        cache_manager: Any | None = None,
        agent: TestFailureAgent | None = None,
    ):
        """Initialize the intelligent failure analyzer.

        Args:
            config: Optional analyzer configuration.
            cache_manager: Optional async cache manager (must expose ``get``
                and ``set`` with TTL).
            agent: Injectable :class:`TestFailureAgent` (handy for tests).
                Defaults to a freshly-constructed instance.
        """
        self.config = config or FailureAnalysisConfig()
        self.cache_manager = cache_manager
        self.agent: TestFailureAgent = agent or TestFailureAgent()
        self.active_analyses: dict[str, asyncio.Task[Any]] = {}

        logger.info("Initialized IntelligentFailureAnalyzer (inference-router routing)")

    async def analyze_test_failure(
        self,
        test_result: TestResult,
        source_code: str | None = None,
        test_code: str | None = None,
        dependencies: dict[str, str] | None = None,
        environment: dict[str, str] | None = None,
        resource_usage: dict[str, Any] | None = None,
        *,
        tenant_id: str | None = None,
        job_id: str | None = None,
        license_id: str | None = None,
    ) -> AIAnalysisResponse:
        """Analyze a test failure using the inference-router LLM plane.

        ``tenant_id`` / ``job_id`` / ``license_id`` (when provided by the
        caller) flow into inference-router as ``X-Tenant-Id`` / ``X-Job-Id``
        / ``X-License-Id`` headers. coding-testing has no first-class
        request-scope today, so callers must thread these through manually
        — when omitted the request goes out without them and inference
        provenance falls back to whatever inference-router infers.
        """
        request_id = str(uuid7())
        start_time = datetime.now(UTC)

        if not self.config.enabled:
            logger.info("Intelligent failure analysis is disabled")
            return AIAnalysisResponse(
                request_id=request_id,
                success=False,
                error_message="Intelligent failure analysis is disabled",
                total_analysis_time_ms=0.0,
            )

        # Cache lookup
        if self.config.cache_results and self.cache_manager:
            cached = await self._get_cached_analysis(test_result)
            if cached is not None:
                logger.info(f"Returning cached analysis for test {test_result.test_id}")
                return cached

        context = PydanticAITestFailureContext(
            test_id=test_result.test_id,
            language=test_result.language,
            test_type=test_result.test_type or "unit",
            exit_code=test_result.exit_code,
            stdout=test_result.stdout,
            stderr=test_result.stderr,
            execution_time_ms=test_result.execution_time * 1000,
            container_name=test_result.container_name,
            source_code=source_code,
            test_code=test_code,
            dependencies=dependencies or {},
            environment=environment or {},
            resource_usage=resource_usage,
        )
        analysis_request = AIAnalysisRequest(
            context=context,
            analysis_types=[PydanticAIAnalysisType.FAILURE_ANALYSIS],
            max_tokens=self.config.max_tokens,
            timeout_seconds=self.config.timeout_seconds,
            include_code_fixes=self.config.enable_code_fixes,
            include_performance_analysis=self.config.enable_performance_analysis,
            include_security_analysis=self.config.enable_security_analysis,
        )

        try:
            with llm_request_context(
                tenant_id=tenant_id,
                job_id=job_id,
                license_id=license_id,
                schema_id=self.agent.schema_id,
            ):
                analysis: TestFailureAnalysisResponse = await self.agent.execute(
                    prompt=self._build_prompt(analysis_request),
                )

            inference_id = last_inference_id()
            failover_chain_id = last_failover_chain_id()
            total_time_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000

            metadata: dict[str, Any] = {
                "schema_id": self.agent.schema_id,
                "category": analysis.category,
                "confidence": analysis.confidence,
            }
            if inference_id:
                metadata["inference_id"] = inference_id
            if failover_chain_id:
                metadata["failover_chain_id"] = failover_chain_id

            response = AIAnalysisResponse(
                request_id=request_id,
                success=True,
                total_analysis_time_ms=total_time_ms,
                metadata=metadata,
                error_message=None,
            )

            # Stash the structured analysis text alongside provenance so
            # downstream consumers (audit log, DB row) can persist it.
            response.metadata["root_cause"] = analysis.root_cause
            response.metadata["suggested_fix"] = analysis.suggested_fix

            logger.info(
                "Test failure analysis complete",
                extra={
                    "test_id": test_result.test_id,
                    "request_id": request_id,
                    "inference_id": inference_id,
                    "failover_chain_id": failover_chain_id,
                    "category": analysis.category,
                },
            )

            if self.config.cache_results and self.cache_manager:
                await self._cache_analysis(test_result, response)

            return response

        except LLMEmptyResponseError as exc:
            total_time_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
            logger.warning(f"LLM returned no usable output for {test_result.test_id}: {exc}")
            return AIAnalysisResponse(
                request_id=request_id,
                success=False,
                error_message=f"LLM plane returned no usable output: {exc}",
                total_analysis_time_ms=total_time_ms,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            total_time_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
            logger.error(f"Failed to analyze test failure for {test_result.test_id}: {exc}")
            return AIAnalysisResponse(
                request_id=request_id,
                success=False,
                error_message=f"Analysis failed: {exc}",
                total_analysis_time_ms=total_time_ms,
            )

    def _build_prompt(self, request: AIAnalysisRequest) -> str:
        """Build the user prompt sent to the LLM."""
        ctx = request.context
        deps = ctx.dependencies or {}
        resource_usage = ctx.resource_usage or {}

        deps_block = (
            ", ".join(f"{name}={version}" for name, version in deps.items())
            if deps
            else "(none reported)"
        )
        resource_block = (
            ", ".join(f"{key}={value}" for key, value in resource_usage.items())
            if resource_usage
            else "(none reported)"
        )

        return (
            f"Test ID: {ctx.test_id}\n"
            f"Language: {ctx.language}\n"
            f"Test type: {ctx.test_type}\n"
            f"Exit code: {ctx.exit_code}\n"
            f"Execution time: {ctx.execution_time_ms:.0f} ms\n"
            f"Container: {ctx.container_name or 'N/A'}\n"
            f"Dependencies: {deps_block}\n"
            f"Resource usage: {resource_block}\n\n"
            f"STDOUT (truncated):\n{(ctx.stdout or '')[:2000]}\n\n"
            f"STDERR (truncated):\n{(ctx.stderr or '')[:2000]}\n\n"
            f"Source code (truncated):\n{(ctx.source_code or '(none)')[:1000]}\n\n"
            f"Test code (truncated):\n{(ctx.test_code or '(none)')[:1000]}\n\n"
            "Diagnose the failure. Return: root_cause, suggested_fix, confidence (0..1), "
            "and category (one of: assertion, timeout, import, syntax, environment, flaky, other)."
        )

    async def _get_cached_analysis(self, test_result: TestResult) -> AIAnalysisResponse | None:
        """Get cached analysis result for a test."""
        if not self.cache_manager:
            return None
        try:
            cache_key = self._create_cache_key(test_result)
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data:
                return AIAnalysisResponse(**cached_data)
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning(f"Failed to get cached analysis: {exc}")
        return None

    async def _cache_analysis(self, test_result: TestResult, response: AIAnalysisResponse) -> None:
        """Cache analysis result."""
        if not self.cache_manager:
            return
        try:
            cache_key = self._create_cache_key(test_result)
            await self.cache_manager.set(
                cache_key,
                response.model_dump(),
                ttl=self.config.cache_ttl_seconds,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning(f"Failed to cache analysis: {exc}")

    def _create_cache_key(self, test_result: TestResult) -> str:
        """Create cache key for test result."""
        key_parts = [
            test_result.language,
            test_result.test_type or "unit",
            str(test_result.exit_code),
            str(hash(test_result.stderr) if test_result.stderr else "no_stderr"),
            str(hash(test_result.stdout) if test_result.stdout else "no_stdout"),
        ]
        return "intelligent_analysis:" + ":".join(key_parts)

    async def batch_analyze_failures(
        self,
        test_results: list[TestResult],
        max_concurrent: int = 5,
    ) -> list[AIAnalysisResponse]:
        """Analyze multiple test failures concurrently."""
        if not self.config.enabled:
            return []
        failed_tests = [r for r in test_results if not r.success]
        if not failed_tests:
            return []

        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_single(test_result: TestResult) -> AIAnalysisResponse:
            async with semaphore:
                return await self.analyze_test_failure(test_result)

        tasks = [analyze_single(r) for r in failed_tests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        responses: list[AIAnalysisResponse] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.error(f"Analysis failed for test {failed_tests[i].test_id}: {result}")
                responses.append(
                    AIAnalysisResponse(
                        request_id=str(uuid7()),
                        success=False,
                        error_message=str(result),
                        total_analysis_time_ms=0.0,
                    )
                )
            else:
                responses.append(result)
        return responses

    async def get_analysis_summary(self) -> dict[str, Any]:
        """Get summary of analysis service status."""
        return {
            "service": "intelligent_failure_analyzer",
            "enabled": self.config.enabled,
            "schema_id": self.agent.schema_id,
            "active_analyses": len(self.active_analyses),
            "cache_enabled": self.config.cache_results,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on the analysis service.

        We don't ping the LLM here — that would burn an inference call per
        readiness probe. We only check enablement and (when configured)
        the cache layer.
        """
        status = "healthy" if self.config.enabled else "disabled"
        checks: dict[str, str] = {"enabled": str(self.config.enabled).lower()}

        if self.cache_manager:
            try:
                await self.cache_manager.get("health_check")
                checks["cache"] = "healthy"
            except (RuntimeError, ValueError, OSError) as exc:
                checks["cache"] = f"unhealthy: {exc}"
                if status == "healthy":
                    status = "degraded"
        else:
            checks["cache"] = "disabled"

        return {
            "service": "intelligent_failure_analyzer",
            "status": status,
            "enabled": self.config.enabled,
            "schema_id": self.agent.schema_id,
            "checks": checks,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def update_config(self, new_config: FailureAnalysisConfig) -> None:
        """Update analyzer configuration."""
        self.config = new_config


def create_intelligent_failure_analyzer(
    config: FailureAnalysisConfig | None = None,
    cache_manager: Any | None = None,
) -> IntelligentFailureAnalyzer:
    """Factory function to create an intelligent failure analyzer."""
    return IntelligentFailureAnalyzer(config=config, cache_manager=cache_manager)
