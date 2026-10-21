"""Request-scoped OpenAI-compatible calls; retries and raw usage are never hidden."""
from __future__ import annotations

import asyncio
import os
import time
import uuid

import httpx

from agentos.llm.base import LLMResponse
from agentos.metrics.model import ModelCallMetrics


def _reported_count(value):
    # Reject booleans, strings, fractions and negative values; do not turn them
    # into plausible-looking measured counters.
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _usage_metrics(usage, *, model, elapsed, finish_reason):
    raw = usage if isinstance(usage, dict) else {}
    prompt = _reported_count(raw.get("prompt_tokens"))
    completion = _reported_count(raw.get("completion_tokens"))
    total = _reported_count(raw.get("total_tokens"))
    total_source = "response_usage" if total is not None else "unavailable"
    if total is None and prompt is not None and completion is not None:
        total, total_source = prompt + completion, "sum_of_reported_prompt_and_completion"
    details = raw.get("prompt_tokens_details")
    details = details if isinstance(details, dict) else {}
    hit_raw = raw.get("prompt_cache_hit_tokens", details.get("cached_tokens"))
    miss_raw = raw.get("prompt_cache_miss_tokens")
    hit, miss = _reported_count(hit_raw), _reported_count(miss_raw)
    source = "response_usage"
    invalid = ((hit_raw is not None and hit is None) or (miss_raw is not None and miss is None))
    if "prompt_cache_hit_tokens" in raw and "cached_tokens" in details:
        invalid |= _reported_count(details["cached_tokens"]) != hit
    if prompt is not None:
        invalid |= (hit is not None and hit > prompt) or (miss is not None and miss > prompt)
        if hit is not None and miss is not None:
            invalid |= hit + miss != prompt
        elif hit is not None:
            miss, source = prompt - hit, "reported_hit_and_prompt_minus_hit"
        elif miss is not None:
            hit, source = prompt - miss, "reported_miss_and_prompt_minus_miss"
    if invalid:
        hit = miss = None
        source = "invalid_response_usage"
    available = hit is not None and miss is not None
    if not available and not invalid:
        source = "partial_response_usage" if hit is not None or miss is not None else "unavailable"
    observed = {"prompt_tokens": prompt, "completion_tokens": completion,
                "total_tokens": total, "cache_hit_tokens": hit, "cache_miss_tokens": miss}
    metadata = {
        "metrics_scope": "request", "token_source": "response_usage",
        "token_metrics_available": prompt is not None and completion is not None,
        "prompt_tokens_available": prompt is not None,
        "completion_tokens_available": completion is not None,
        "total_tokens_available": total is not None, "total_token_source": total_source,
        "cache_hit_tokens_available": hit is not None, "cache_miss_tokens_available": miss is not None,
        "cache_metrics_available": available, "cache_metric_source": source,
        "observed_tokens": observed, "cache_counter_validation_failed": bool(invalid),
        "prefill_decode_available": False, "prefill_decode_fallback": True,
        "queue_wait_available": False, "latency_source": "client_wall_clock",
        "finish_reason": finish_reason,
    }
    # The existing metrics dataclass and gateway arithmetic require integers.
    # Unknown numeric slots remain compatibility placeholders, explicitly
    # distinguished from measured zero by metadata and nullable observed_tokens.
    metrics = ModelCallMetrics(backend="openai_compatible", model=model,
        prompt_tokens=prompt if prompt is not None else 0,
        completion_tokens=completion if completion is not None else 0,
        cache_hit_tokens=hit if hit is not None else 0,
        cache_miss_tokens=miss if miss is not None else 0,
        latency_secs=elapsed, metadata=metadata)
    return metrics, total if total is not None else 0


class OpenAIBackend:
    def __init__(self, config):
        self.config = config

    def request_payload(self, request):
        config = self.config
        payload = {"model": config["model"], "messages": request.messages}
        if config.get("max_tokens") is not None:
            payload["max_tokens"] = config["max_tokens"]
        if config.get("thinking") is not None:
            payload["thinking"] = {"type": config["thinking"]}
        if config.get("thinking") != "enabled":
            payload["temperature"] = config["temperature"]
        if config.get("reasoning_effort") is not None:
            payload["reasoning_effort"] = config["reasoning_effort"]
        if request.tools:
            payload["tools"] = request.tools
        elif config.get("response_format") is not None:
            payload["response_format"] = {"type": config["response_format"]}
        return payload

    async def generate(self, request):
        config = self.config
        key = os.environ.get(config["api_key_env"], "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        url = config["base_url"].rstrip("/")
        if not url.endswith("/v1"):
            url += "/v1"
        payload = self.request_payload(request)
        start = time.monotonic()
        async with httpx.AsyncClient(timeout=config["timeout"]) as client:
            response = await client.post(url + "/chat/completions", headers=headers, json=payload)
            # Do not include response body (may echo auth) in transport exceptions.
            if response.status_code >= 400:
                raise RuntimeError(f"model HTTP status {response.status_code}")
            payload = response.json()
        choice = payload["choices"][0]
        assistant_message = choice["message"]
        content = assistant_message.get("content")
        if content is not None and not isinstance(content, str):
            raise ValueError("model returned an unsupported non-text content type")
        usage = payload.get("usage")
        model = payload.get("model", config["model"])
        elapsed = time.monotonic() - start
        metrics, tokens = _usage_metrics(usage, model=model, elapsed=elapsed,
                                        finish_reason=choice.get("finish_reason"))
        return LLMResponse(content or "", tokens, model, elapsed,
                           backend="openai_compatible", metrics=metrics,
                           metadata={"usage": usage, "finish_reason": choice.get("finish_reason"),
                                     "assistant_message": assistant_message})



class RecordedBackend:
    def __init__(self, backend, records, config):
        self.backend, self.records, self.config = backend, records, config
        self.calls = []

    async def generate(self, request):
        call_id = uuid.uuid4().hex
        request.model = self.config["model"]
        request.max_tokens = self.config["max_tokens"]
        request.temperature = self.config["temperature"]
        for attempt in range(1, self.config["retries"] + 2):
            start = time.monotonic()
            row = dict(call_id=call_id, attempt=attempt, task_id=request.task_id,
                       agent_id=request.agent_id, messages=request.messages,
                       context_refs=request.context_refs, model=request.model,
                       tools=request.tools, prefix_id=request.prefix_id,
                       context_selection=request.metadata.get("context_selection"))
            if hasattr(self.backend, "request_payload"):
                row["request_parameters"] = {k: v for k, v in self.backend.request_payload(request).items() if k != "messages"}
            self.records.emit("execution.model_started", request.task_id, call_id=call_id, attempt=attempt)
            cancelled = False
            try:
                response = await self.backend.generate(request)
                row.update(status="completed", response=response.content, assistant_message=response.metadata.get("assistant_message"),
                           metrics=response.metrics.to_dict() if response.metrics is not None else None,
                           usage=response.metadata.get("usage"), response_model=response.model, finish_reason=response.metadata.get("finish_reason"))
            except asyncio.CancelledError:
                cancelled = True
                raise_error = True
                row.update(status="failed", error="CancelledError: model call cancelled", usage=None)
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}", usage=None)
                raise_error = attempt > self.config["retries"]
            else:
                raise_error = False
            row["duration_seconds"] = time.monotonic() - start
            self.calls.append(self.records.clean(row))
            with (self.records.output / "calls.jsonl").open("a") as stream:
                import json
                stream.write(json.dumps(self.records.clean(row), ensure_ascii=False) + "\n")
            self.records.emit("execution.model_" + row["status"], request.task_id,
                              **{k: v for k, v in row.items() if k not in {"messages", "task_id", "agent_id", "response", "assistant_message", "tools", "request_parameters"}})
            if row["status"] == "completed":
                return response
            if cancelled:
                raise asyncio.CancelledError
            if raise_error:
                raise RuntimeError(row["error"])
            self.records.emit("execution.model_retry", request.task_id, call_id=call_id, next_attempt=attempt + 1)
            await asyncio.sleep(min(0.1 * attempt, 1))
