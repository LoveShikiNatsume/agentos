from __future__ import annotations

import os
import time
from typing import Any

import httpx

from agentos.llm.base import LLMRequest, LLMResponse
from agentos.metrics.model import ModelCallMetrics


class DeepSeekBackend:
    """DeepSeek API backend with prompt cache metrics extraction.

    Uses httpx directly to parse prompt_cache_hit_tokens / prompt_cache_miss_tokens
    from the DeepSeek usage response.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout: float = 60.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    async def generate(self, request: LLMRequest) -> LLMResponse:
        model = request.model if request.model != "default" else self._model
        start = time.monotonic()

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": request.messages,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                },
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()

        elapsed = time.monotonic() - start
        content = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {})

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cache_hit = usage.get("prompt_cache_hit_tokens", 0)
        cache_miss = usage.get("prompt_cache_miss_tokens", 0)

        if cache_hit == 0 and cache_miss == 0 and prompt_tokens > 0:
            cache_miss = prompt_tokens

        metrics = ModelCallMetrics(
            backend="deepseek",
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
            backend="deepseek",
            metrics=metrics,
            metadata={"usage": usage},
        )
