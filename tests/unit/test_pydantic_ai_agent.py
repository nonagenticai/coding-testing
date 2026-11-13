"""Tests for the post-migration TestFailureAgent.

Asserts the thin :class:`TestFailureAgent` correctly delegates to
:meth:`coding_client.llm.BaseLLMAgent._call_llm`, returns a typed
:class:`TestFailureAnalysisResponse`, and round-trips its response model
unchanged.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from coding_client.llm import TestFailureAnalysisResponse

from services.test_execution.failure_agent import TestFailureAgent


def _stub_response() -> TestFailureAnalysisResponse:
    return TestFailureAnalysisResponse(
        root_cause="Off-by-one in addition",
        suggested_fix="Use a + b, not a + b - 1",
        confidence=0.91,
        category="assertion",
    )


class TestTestFailureAgent:
    def test_class_attributes(self) -> None:
        assert TestFailureAgent.schema_id == "test_failure_analysis"
        assert TestFailureAgent.response_model is TestFailureAnalysisResponse
        assert TestFailureAgent.system_prompt  # non-empty

    def test_instantiates(self) -> None:
        agent = TestFailureAgent()
        assert agent.schema_id == "test_failure_analysis"

    @pytest.mark.asyncio
    async def test_execute_returns_response_model(self) -> None:
        agent = TestFailureAgent()
        with patch.object(
            TestFailureAgent,
            "_call_llm",
            new=AsyncMock(return_value=_stub_response()),
        ) as mock_call:
            result = await agent.execute(prompt="Test failed because foo")

        assert isinstance(result, TestFailureAnalysisResponse)
        assert result.root_cause == "Off-by-one in addition"
        assert result.category == "assertion"
        mock_call.assert_awaited_once()
        kwargs = mock_call.await_args.kwargs
        assert kwargs["prompt"] == "Test failed because foo"
        assert kwargs["response_model"] is TestFailureAnalysisResponse

    @pytest.mark.asyncio
    async def test_execute_passes_system_prompt_override(self) -> None:
        agent = TestFailureAgent()
        with patch.object(
            TestFailureAgent,
            "_call_llm",
            new=AsyncMock(return_value=_stub_response()),
        ) as mock_call:
            await agent.execute(prompt="x", system_prompt="custom-system")
        assert mock_call.await_args.kwargs["system_prompt"] == "custom-system"


class TestResponseRoundtrip:
    def test_valid_response_serializes(self) -> None:
        resp = _stub_response()
        dumped = resp.model_dump()
        assert dumped["category"] == "assertion"
        round_tripped = TestFailureAnalysisResponse.model_validate(dumped)
        assert round_tripped == resp

    def test_invalid_category_rejected(self) -> None:
        with pytest.raises(Exception):
            TestFailureAnalysisResponse(
                root_cause="x",
                suggested_fix="y",
                confidence=0.5,
                category="not-a-real-category",  # type: ignore[arg-type]
            )

    def test_confidence_bounds(self) -> None:
        with pytest.raises(Exception):
            TestFailureAnalysisResponse(
                root_cause="x",
                suggested_fix="y",
                confidence=1.5,
                category="other",
            )
        with pytest.raises(Exception):
            TestFailureAnalysisResponse(
                root_cause="x",
                suggested_fix="y",
                confidence=-0.1,
                category="other",
            )

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(Exception):
            TestFailureAnalysisResponse(
                root_cause="x",
                suggested_fix="y",
                confidence=0.5,
                category="other",
                extra_field="boom",  # type: ignore[call-arg]
            )
