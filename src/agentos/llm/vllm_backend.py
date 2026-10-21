from __future__ import annotations

import os
import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx

from agentos.llm.base import LLMRequest, LLMResponse
from agentos.metrics.model import ModelCallMetrics


@dataclass
class _VLLMCounters:
    prefix_cache_queries: float = 0.0
    prefix_cache_hits: float = 0.0
    prefill_tokens: float = 0.0
    decode_tokens: float = 0.0
    prefill_secs: float = 0.0
    decode_secs: float = 0.0


_PREFIX_QUERY_METRICS = {"vllm:prefix_cache_queries_total"}
_PREFIX_HIT_METRICS = {"vllm:prefix_cache_hits_total"}
_PREFILL_TOKEN_METRICS = {
    "vllm:prefill_tokens_total",
    "vllm:prompt_tokens_total",
}
_DECODE_TOKEN_METRICS = {
    "vllm:decode_tokens_total",
    "vllm:generation_tokens_total",
}
_PREFILL_TIME_METRICS = {
    "vllm:prefill_time_seconds_sum",
    "vllm:time_to_first_token_seconds_sum",
}
_DECODE_TIME_METRICS = {
    "vllm:decode_time_seconds_sum",
    "vllm:time_per_output_token_seconds_sum",
}


def _metric_name(line: str) -> str | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    return line.split()[0].split("{", 1)[0]


def _metric_value(line: str) -> float:
    try:
        return float(line.split()[-1])
    except (IndexError, ValueError):
        return 0.0


def _positive_delta(after: float, before: float) -> float:
    return max(0.0, after - before)


class VLLMBackend:
    """vLLM backend with request-scoped usage by default.

    Global Prometheus deltas are only used in explicit exclusive-server mode.
    That mode serializes this client's calls and assumes no other server users.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str | None = None,
        timeout: float = 120.0,
        *,
        metrics_mode: str = "request",
        max_output_tokens: int | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if metrics_mode not in {"request", "exclusive"}:
            raise ValueError("metrics_mode must be request or exclusive")
        if max_output_tokens is not None and max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        self._metrics_mode = metrics_mode
        self._max_output_tokens = max_output_tokens
        self._chat_template_kwargs = dict(chat_template_kwargs or {})
        self._exclusive_lock = asyncio.Lock()
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    async def _resolve_model(self, client: httpx.AsyncClient) -> str:
        if self._model:
            return self._model
        resp = await client.get(f"{self._base_url}/v1/models")
        resp.raise_for_status()
        data = resp.json()
        self._model = data["data"][0]["id"]
        return self._model

    async def _read_counters(self, client: httpx.AsyncClient) -> _VLLMCounters:
        try:
            resp = await client.get(f"{self._base_url}/metrics")
            counters = _VLLMCounters()
            for line in resp.text.splitlines():
                name = _metric_name(line)
                if name is None:
                    continue
                value = _metric_value(line)
                if name in _PREFIX_QUERY_METRICS:
                    counters.prefix_cache_queries += value
                elif name in _PREFIX_HIT_METRICS:
                    counters.prefix_cache_hits += value
                elif name in _PREFILL_TOKEN_METRICS:
                    counters.prefill_tokens += value
                elif name in _DECODE_TOKEN_METRICS:
                    counters.decode_tokens += value
                elif name in _PREFILL_TIME_METRICS:
                    counters.prefill_secs += value
                elif name in _DECODE_TIME_METRICS:
                    counters.decode_secs += value
            return counters
        except Exception:
            return _VLLMCounters()

    async def _read_cache_counters(self, client: httpx.AsyncClient) -> tuple[float, float]:
        counters = await self._read_counters(client)
        return counters.prefix_cache_queries, counters.prefix_cache_hits

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if self._metrics_mode == "exclusive":
            async with self._exclusive_lock:
                return await self._generate(request)
        return await self._generate(request)

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        start = time.monotonic()
        exclusive = self._metrics_mode == "exclusive"
        async with httpx.AsyncClient(timeout=self._timeout, trust_env=False) as client:
            model = request.model if request.model != "default" else await self._resolve_model(client)
            before = await self._read_counters(client) if exclusive else _VLLMCounters()
            payload: dict[str, Any] = {
                "model": model, "messages": request.messages,
                "temperature": request.temperature,
                "max_tokens": min(request.max_tokens, self._max_output_tokens)
                if self._max_output_tokens is not None else request.max_tokens,
            }
            template = request.metadata.get("chat_template_kwargs", self._chat_template_kwargs)
            if template:
                payload["chat_template_kwargs"] = template
            if "response_format" in request.metadata:
                payload["response_format"] = request.metadata["response_format"]
            resp = await client.post(
                f"{self._base_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"}, json=payload,
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            after = await self._read_counters(client) if exclusive else _VLLMCounters()

        elapsed = time.monotonic() - start
        usage = data.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        details = usage.get("prompt_tokens_details") or {}
        cache_queries = 0
        cache_available = False
        if details.get("cached_tokens") is not None:
            cache_hit = int(details["cached_tokens"])
            if not 0 <= cache_hit <= prompt_tokens:
                raise ValueError("invalid request-scoped cache counters")
            cache_miss = prompt_tokens - cache_hit
            cache_queries = prompt_tokens
            cache_available = True
            cache_source = "response_usage"
        elif exclusive:
            cache_hit = int(_positive_delta(after.prefix_cache_hits, before.prefix_cache_hits))
            cache_queries = int(_positive_delta(after.prefix_cache_queries, before.prefix_cache_queries))
            cache_miss = max(0, cache_queries - cache_hit)
            cache_available = cache_queries > 0
            cache_source = "exclusive_server_delta" if cache_available else "prompt_tokens_fallback"
            if not cache_available:
                cache_miss = prompt_tokens
        else:
            # Absence is unknown, not a measured cache miss. Keep numeric fields
            # compatible while the availability flag prevents heat feedback.
            cache_hit = cache_miss = 0
            cache_source = "unavailable"

        prefill_tokens = int(_positive_delta(after.prefill_tokens, before.prefill_tokens))
        decode_tokens = int(_positive_delta(after.decode_tokens, before.decode_tokens))
        prefill_secs = _positive_delta(after.prefill_secs, before.prefill_secs)
        decode_secs = _positive_delta(after.decode_secs, before.decode_secs)
        timing_available = exclusive and any((prefill_tokens, decode_tokens, prefill_secs, decode_secs))
        if not timing_available:
            prefill_tokens = cache_miss if cache_available else prompt_tokens
            decode_tokens = completion_tokens
        metadata = {
            "prefill_tokens": prefill_tokens, "decode_tokens": decode_tokens,
            "prefill_secs": prefill_secs, "decode_secs": decode_secs,
            "prefill_decode_available": bool(timing_available),
            "prefill_decode_fallback": not timing_available,
            "prefill_saved_tokens": cache_hit, "prefill_miss_tokens": cache_miss,
            "cache_metrics_available": cache_available, "cache_metric_source": cache_source,
            "metrics_scope": "exclusive_server_interval" if exclusive else "request",
            "token_source": "vllm_usage", "finish_reason": data["choices"][0].get("finish_reason"),
        }
        metrics = ModelCallMetrics(
            backend="vllm", model=model, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, cache_hit_tokens=cache_hit,
            cache_miss_tokens=cache_miss, latency_secs=elapsed, metadata=metadata,
        )
        return LLMResponse(
            content=data["choices"][0]["message"]["content"] or "",
            tokens_used=prompt_tokens + completion_tokens, model=model,
            latency_secs=elapsed, backend="vllm", metrics=metrics,
            metadata={"usage": usage, "cache_queries": cache_queries, **metadata},
        )
