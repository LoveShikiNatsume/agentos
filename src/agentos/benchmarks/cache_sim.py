"""Cache-simulating mock LLM backend for benchmarking.

Simulates prefix caching behavior: when multiple requests share a common
system prompt prefix, the overlapping tokens count as cache hits.
Models DeepSeek prompt cache and vLLM prefix caching.
"""
from __future__ import annotations

import asyncio

from agentos.llm.base import LLMRequest, LLMResponse
from agentos.metrics.model import ModelCallMetrics


class CacheSimulatingMockBackend:
    def __init__(
        self,
        latency_base: float = 0.05,
        cache_speedup: float = 0.6,
        model_name: str = "cache-sim-mock",
        backend_name: str = "cache-sim-mock",
    ) -> None:
        self._seen_system: list[str] = []
        self._latency_base = latency_base
        self._cache_speedup = cache_speedup
        self._model_name = model_name
        self._backend_name = backend_name
        self.call_count = 0
        self.requests: list[LLMRequest] = []

    def reset(self) -> None:
        self._seen_system.clear()
        self.call_count = 0
        self.requests.clear()

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        self.requests.append(request)

        system_content = ""
        for msg in request.messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
                break

        prompt_tokens = max(
            1,
            sum(len(m.get("content", "")) for m in request.messages) // 4,
        )

        best_hit_chars = 0
        for prev in self._seen_system:
            overlap = 0
            for a, b in zip(system_content, prev):
                if a == b:
                    overlap += 1
                else:
                    break
            best_hit_chars = max(best_hit_chars, overlap)

        self._seen_system.append(system_content)

        cache_hit_tokens = min(best_hit_chars // 4, prompt_tokens)
        cache_miss_tokens = prompt_tokens - cache_hit_tokens

        cache_ratio = cache_hit_tokens / prompt_tokens if prompt_tokens > 0 else 0
        latency = self._latency_base * (1.0 - cache_ratio * self._cache_speedup)
        await asyncio.sleep(latency)

        content = f"mock response for {request.agent_id}"
        completion_tokens = len(content.split()) * 2

        metrics = ModelCallMetrics(
            backend=self._backend_name,
            model=self._model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_hit_tokens=cache_hit_tokens,
            cache_miss_tokens=cache_miss_tokens,
            latency_secs=latency,
        )

        return LLMResponse(
            content=content,
            tokens_used=completion_tokens,
            model=self._model_name,
            latency_secs=latency,
            backend=self._backend_name,
            metrics=metrics,
        )
