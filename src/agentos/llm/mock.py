from __future__ import annotations

import asyncio

from agentos.llm.base import LLMRequest, LLMResponse
from agentos.metrics.model import ModelCallMetrics


class MockLLMBackend:
    def __init__(
        self,
        responses: dict[str, str] | None = None,
        latency: float = 0.01,
    ) -> None:
        self._responses = responses or {"default": "mock response"}
        self._latency = latency
        self.call_count = 0
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        await asyncio.sleep(self._latency)
        self.call_count += 1
        self.requests.append(request)

        content = self._responses.get(
            request.agent_id, self._responses.get("default", "")
        )

        prompt_tokens = max(1, sum(len(m.get("content", "")) for m in request.messages) // 4)
        completion_tokens = len(content.split()) * 2

        metrics = ModelCallMetrics(
            backend="mock",
            model="mock",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_hit_tokens=0,
            cache_miss_tokens=prompt_tokens,
            latency_secs=self._latency,
        )

        return LLMResponse(
            content=content,
            tokens_used=len(content.split()) * 2,
            model="mock",
            latency_secs=self._latency,
            backend="mock",
            metrics=metrics,
        )
