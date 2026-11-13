"""Tests for the post-migration IntelligentFailureAnalyzer.

The analyzer now routes through inference-router via
:class:`coding_client.llm.BaseLLMAgent`; this module exercises the
delegation, caching, batch behaviour, and provenance capture.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from coding_client.llm import TestFailureAnalysisResponse

from models.testing_models import TestResult, TestSecurityLevel
from services.test_execution.intelligent_failure_analyzer import (
    FailureAnalysisConfig,
    IntelligentFailureAnalyzer,
    create_intelligent_failure_analyzer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_test_result(**overrides: Any) -> TestResult:
    defaults: dict[str, Any] = {
        "test_id": "test-001",
        "success": False,
        "stdout": "FAILED test_add",
        "stderr": "AssertionError: expected 3 but got 4",
        "exit_code": 1,
        "execution_time": 1.0,
        "created_at": datetime.now(UTC),
        "language": "python",
        "security_level": TestSecurityLevel.MEDIUM,
        "container_name": "container-1",
    }
    defaults.update(overrides)
    return TestResult(**defaults)


def _stub_response(**overrides: Any) -> TestFailureAnalysisResponse:
    fields: dict[str, Any] = {
        "root_cause": "Off-by-one in addition",
        "suggested_fix": "Check the operator and operand order",
        "confidence": 0.87,
        "category": "assertion",
    }
    fields.update(overrides)
    return TestFailureAnalysisResponse(**fields)


@pytest.fixture
def stub_agent() -> MagicMock:
    agent = MagicMock()
    agent.schema_id = "test_failure_analysis"
    agent.execute = AsyncMock(return_value=_stub_response())
    return agent


# ---------------------------------------------------------------------------
# FailureAnalysisConfig
# ---------------------------------------------------------------------------


class TestFailureAnalysisConfig:
    def test_defaults(self) -> None:
        cfg = FailureAnalysisConfig()
        assert cfg.enabled is True
        assert cfg.timeout_seconds == 30
        assert cfg.max_tokens == 2000
        assert cfg.enable_code_fixes is True
        assert cfg.enable_performance_analysis is True
        assert cfg.enable_security_analysis is True
        assert cfg.cache_results is True
        assert cfg.cache_ttl_seconds == 3600

    def test_custom(self) -> None:
        cfg = FailureAnalysisConfig(enabled=False, cache_results=False, max_tokens=512)
        assert cfg.enabled is False
        assert cfg.cache_results is False
        assert cfg.max_tokens == 512

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(Exception):
            FailureAnalysisConfig(primary_provider="openai")  # legacy field, must error


# ---------------------------------------------------------------------------
# IntelligentFailureAnalyzer
# ---------------------------------------------------------------------------


class TestAnalyzerHappyPath:
    @pytest.mark.asyncio
    async def test_analyze_returns_structured_metadata(self, stub_agent: MagicMock) -> None:
        analyzer = IntelligentFailureAnalyzer(agent=stub_agent)
        result = await analyzer.analyze_test_failure(
            _make_test_result(),
            tenant_id="t1",
            job_id="j1",
            license_id="lic1",
        )

        assert result.success is True
        assert result.metadata["category"] == "assertion"
        assert pytest.approx(result.metadata["confidence"]) == 0.87
        assert result.metadata["root_cause"] == "Off-by-one in addition"
        assert result.metadata["suggested_fix"]
        assert result.total_analysis_time_ms >= 0
        stub_agent.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_inference_metadata_captured(self, stub_agent: MagicMock) -> None:
        analyzer = IntelligentFailureAnalyzer(agent=stub_agent)
        with (
            patch(
                "services.test_execution.intelligent_failure_analyzer.last_inference_id",
                return_value="inf-123",
            ),
            patch(
                "services.test_execution.intelligent_failure_analyzer.last_failover_chain_id",
                return_value="chain-abc",
            ),
        ):
            result = await analyzer.analyze_test_failure(_make_test_result())
        assert result.metadata["inference_id"] == "inf-123"
        assert result.metadata["failover_chain_id"] == "chain-abc"

    @pytest.mark.asyncio
    async def test_disabled_short_circuits(self, stub_agent: MagicMock) -> None:
        analyzer = IntelligentFailureAnalyzer(
            config=FailureAnalysisConfig(enabled=False),
            agent=stub_agent,
        )
        result = await analyzer.analyze_test_failure(_make_test_result())
        assert result.success is False
        assert "disabled" in (result.error_message or "").lower()
        stub_agent.execute.assert_not_called()


class TestAnalyzerLLMErrors:
    @pytest.mark.asyncio
    async def test_llm_empty_response_returns_error_response(self, stub_agent: MagicMock) -> None:
        from coding_client.llm import LLMEmptyResponseError

        stub_agent.execute = AsyncMock(
            side_effect=LLMEmptyResponseError("no output", agent_name="TestFailureAgent")
        )
        analyzer = IntelligentFailureAnalyzer(agent=stub_agent)
        result = await analyzer.analyze_test_failure(_make_test_result())
        assert result.success is False
        assert result.error_message is not None
        assert "no usable" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_runtime_error_caught(self, stub_agent: MagicMock) -> None:
        stub_agent.execute = AsyncMock(side_effect=RuntimeError("kaboom"))
        analyzer = IntelligentFailureAnalyzer(agent=stub_agent)
        result = await analyzer.analyze_test_failure(_make_test_result())
        assert result.success is False
        assert "kaboom" in (result.error_message or "")


class TestAnalyzerCaching:
    @pytest.mark.asyncio
    async def test_cache_hit_skips_llm(self, stub_agent: MagicMock) -> None:
        cache = AsyncMock()
        cache.get = AsyncMock(
            return_value={
                "request_id": "cached-1",
                "success": True,
                "total_analysis_time_ms": 0.0,
                "metadata": {"category": "assertion", "confidence": 0.9},
            }
        )
        analyzer = IntelligentFailureAnalyzer(cache_manager=cache, agent=stub_agent)
        result = await analyzer.analyze_test_failure(_make_test_result())
        assert result.request_id == "cached-1"
        stub_agent.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_then_set(self, stub_agent: MagicMock) -> None:
        cache = AsyncMock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock()
        analyzer = IntelligentFailureAnalyzer(cache_manager=cache, agent=stub_agent)
        await analyzer.analyze_test_failure(_make_test_result())
        cache.set.assert_awaited_once()


class TestBatchAnalyze:
    @pytest.mark.asyncio
    async def test_batch_returns_one_per_failed_test(self, stub_agent: MagicMock) -> None:
        analyzer = IntelligentFailureAnalyzer(agent=stub_agent)
        results = await analyzer.batch_analyze_failures(
            [
                _make_test_result(test_id="a", success=False),
                _make_test_result(test_id="b", success=True),
                _make_test_result(test_id="c", success=False),
            ]
        )
        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_batch_empty_returns_empty(self, stub_agent: MagicMock) -> None:
        analyzer = IntelligentFailureAnalyzer(agent=stub_agent)
        assert await analyzer.batch_analyze_failures([]) == []

    @pytest.mark.asyncio
    async def test_batch_disabled_returns_empty(self, stub_agent: MagicMock) -> None:
        analyzer = IntelligentFailureAnalyzer(
            config=FailureAnalysisConfig(enabled=False),
            agent=stub_agent,
        )
        assert await analyzer.batch_analyze_failures([_make_test_result()]) == []


class TestSummaryAndHealth:
    @pytest.mark.asyncio
    async def test_summary(self, stub_agent: MagicMock) -> None:
        analyzer = IntelligentFailureAnalyzer(agent=stub_agent)
        summary = await analyzer.get_analysis_summary()
        assert summary["service"] == "intelligent_failure_analyzer"
        assert summary["schema_id"] == "test_failure_analysis"

    @pytest.mark.asyncio
    async def test_health_check_healthy_when_enabled(self, stub_agent: MagicMock) -> None:
        analyzer = IntelligentFailureAnalyzer(agent=stub_agent)
        health = await analyzer.health_check()
        assert health["status"] == "healthy"
        assert health["enabled"] is True

    @pytest.mark.asyncio
    async def test_health_check_disabled_when_disabled(self, stub_agent: MagicMock) -> None:
        analyzer = IntelligentFailureAnalyzer(
            config=FailureAnalysisConfig(enabled=False),
            agent=stub_agent,
        )
        health = await analyzer.health_check()
        assert health["status"] == "disabled"

    @pytest.mark.asyncio
    async def test_health_cache_failure_marks_degraded(self, stub_agent: MagicMock) -> None:
        cache = AsyncMock()
        cache.get = AsyncMock(side_effect=RuntimeError("cache down"))
        analyzer = IntelligentFailureAnalyzer(cache_manager=cache, agent=stub_agent)
        health = await analyzer.health_check()
        assert health["status"] == "degraded"
        assert "unhealthy" in health["checks"]["cache"]


class TestFactoryAndUpdate:
    def test_factory(self) -> None:
        analyzer = create_intelligent_failure_analyzer()
        assert isinstance(analyzer, IntelligentFailureAnalyzer)

    def test_update_config(self, stub_agent: MagicMock) -> None:
        analyzer = IntelligentFailureAnalyzer(agent=stub_agent)
        new_cfg = FailureAnalysisConfig(enabled=False, cache_results=False)
        analyzer.update_config(new_cfg)
        assert analyzer.config.enabled is False
