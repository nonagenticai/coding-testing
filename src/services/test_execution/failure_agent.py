"""Thin LLM agent for test-failure analysis.

Routes through inference-router via the shared
:class:`coding_client.llm.BaseLLMAgent`. Previously this service shipped a
~1000-line ``FailureAnalysisAgent`` that hard-coded provider routing, three
provider SDK paths, and ad-hoc heuristic analyses. Server-side fallback is
now handled by inference-router; pydantic-ai handles JSON-schema injection
+ validation; this subclass exists only to declare the system prompt and
expose a typed entry point.
"""

from __future__ import annotations

from typing import Any

from coding_client.llm import (
    AGENT_ROUTING_REGISTRY,
    BaseLLMAgent,
    TestFailureAnalysisResponse,
)

_ROUTING = AGENT_ROUTING_REGISTRY["test_failure_analysis"]


class TestFailureAgent(BaseLLMAgent):
    """LLM agent for diagnosing a single failed test execution.

    The schema_id ``test_failure_analysis`` is registered with model-registry
    via ``coding-client``'s ``register_routing_rules.py`` bootstrap; the
    routing rule there governs the provider fallback chain.
    """

    schema_id = _ROUTING.schema_id
    response_model = TestFailureAnalysisResponse
    system_prompt = _ROUTING.system_prompt

    async def execute(self, **kwargs: Any) -> TestFailureAnalysisResponse:
        """Diagnose a failed test and return a structured analysis.

        Keyword args:
            prompt: User prompt describing the test failure (logs, code,
                stderr, etc.). Callers are expected to format this from
                their failure record. Required.
            system_prompt: Optional override of the default system prompt.

        Returns:
            A validated :class:`TestFailureAnalysisResponse`.
        """
        prompt = kwargs.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("TestFailureAgent.execute requires a non-empty 'prompt' kwarg")
        system_prompt = kwargs.get("system_prompt")
        result = await self._call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            response_model=TestFailureAnalysisResponse,
        )
        # _call_llm returns the parsed response model when one is in play.
        assert isinstance(result, TestFailureAnalysisResponse)
        return result
