from __future__ import annotations

import time

import litellm

from agentos.llm.base import LLMRequest, LLMResponse
from agentos.metrics.model import ModelCallMetrics


class LiteLLMBackend:
    def __init__(self, default_model: str = "gpt-4o-mini") -> None:
        self._default_model = default_model

    async def generate(self, request: LLMRequest) -> LLMResponse:
        model = request.model if request.model != "default" else self._default_model
        start = time.monotonic()

        response = await litellm.acompletion(
            model=model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        elapsed = time.monotonic() - start
        content = response.choices[0].message.content or ""
        usage = response.usage

        prompt_tokens = usage.prompt_tokens or 0
        completion_tokens = usage.completion_tokens or 0

        cache_hit = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
        cache_miss = getattr(usage, "prompt_cache_miss_tokens", 0) or 0
        if cache_hit == 0 and cache_miss == 0 and prompt_tokens > 0:
            cache_miss = prompt_tokens

        metrics = ModelCallMetrics(
            backend="litellm",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_hit_tokens=cache_hit,
            cache_miss_tokens=cache_miss,
            latency_secs=elapsed,
        )

        return LLMResponse(
            content=content,
            tokens_used=prompt_tokens + completion_tokens,
            model=model,
            latency_secs=elapsed,
            backend="litellm",
            metrics=metrics,
            metadata={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )

