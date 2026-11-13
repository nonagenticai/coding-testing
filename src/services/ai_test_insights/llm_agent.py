"""Generic free-form LLM agent for AI Test Insights JSON-schema-on-prompt calls.

The insights service still has legacy call sites that send a hand-rolled
JSON schema in the system prompt and parse the model's free-text reply.
For those we want a thin agent that returns a string, not a typed pydantic
model — :class:`coding_client.llm.BaseLLMAgent` already supports that mode
when ``response_model`` is ``None``.

When new insight call sites are added, prefer declaring a real pydantic
response model in ``coding_client.llm.responses`` instead of going through
this free-form path.
"""

from __future__ import annotations

from typing import Any

from coding_client.llm import BaseLLMAgent


class AITestInsightsAgent(BaseLLMAgent):
    """Free-form LLM helper for AI Test Insights legacy call sites."""

    schema_id = "ai_test_insights_freeform"
    response_model = None
    system_prompt = (
        "You analyze test-suite metadata and return structured insights. "
        "Always honour the JSON schema described in the user-supplied system prompt."
    )

    async def execute(self, **kwargs: Any) -> str:
        """Issue a free-form LLM call and return the raw text output.

        Keyword args:
            prompt: User prompt to send to the model. Required.
            system_prompt: Optional override of the default system prompt.
        """
        prompt = kwargs.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("AITestInsightsAgent.execute requires a non-empty 'prompt' kwarg")
        system_prompt = kwargs.get("system_prompt")
        result = await self._call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            response_model=None,
        )
        return str(result)
