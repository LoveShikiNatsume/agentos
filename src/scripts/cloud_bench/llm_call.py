"""Unified LLM call layer for cloud benchmark.

Provides native_call (direct DeepSeek) and runtime_call (ModelGateway with cache tracking).
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from agentos.events.log import EventLog
from agentos.events.types import RuntimeEvent, EventCategory, EventSeverity
from agentos.llm.base import LLMRequest
from agentos.llm.deepseek_backend import DeepSeekBackend
from agentos.llm.gateway import ModelGateway
from agentos.context.cache_tracker import CacheHeatTracker


def _make_request(prompt: str, agent_id: str = "cloud_bench", max_tokens: int = 512) -> LLMRequest:
    return LLMRequest(
        agent_id=agent_id,
        messages=[
            {"role": "system", "content": "You are a senior software engineer. Be concise and precise."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )


async def native_call(
    prompt: str,
    max_tokens: int = 512,
) -> tuple[str, dict[str, int]]:
    """Direct DeepSeek call. Returns (content, usage_dict)."""
    backend = DeepSeekBackend()
    req = _make_request(prompt, max_tokens)
    resp = await backend.generate(req)

    usage = {
        "prompt_tokens": resp.metrics.prompt_tokens if resp.metrics else 0,
        "completion_tokens": resp.metrics.completion_tokens if resp.metrics else 0,
        "cache_hit_tokens": resp.metrics.cache_hit_tokens if resp.metrics else 0,
        "cache_miss_tokens": resp.metrics.cache_miss_tokens if resp.metrics else 0,
        "latency_secs": resp.latency_secs,
    }
    return resp.content, usage


async def runtime_call(
    prompt: str,
    prefix_id: str = "wishlist-shared-spec",
    max_tokens: int = 512,
    event_log: EventLog | None = None,
) -> tuple[str, dict[str, int], int]:
    """ModelGateway call with cache tracking. Returns (content, usage_dict, event_count)."""
    tracker = CacheHeatTracker(ttl=300, hot_threshold=0.3)
    gateway = ModelGateway(
        backends={"deepseek": DeepSeekBackend()},
        default_backend="deepseek",
        cache_tracker=tracker,
    )

    req = _make_request(prompt, max_tokens=max_tokens)
    req = LLMRequest(
        agent_id=req.agent_id,
        messages=req.messages,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        model=req.model,
        backend="deepseek",
        prefix_id=prefix_id,
    )

    event_count = 0
    if event_log:
        # Emit llm.request_sent
        evt = RuntimeEvent(
            event_type="llm.request_sent",
            category=EventCategory.LLM,
            severity=EventSeverity.INFO,
            data={"model": "deepseek-v4-flash", "token_count": sum(len(m["content"]) for m in req.messages)},
        )
        event_log.append(evt)
        event_count += 1

    resp = await gateway.generate(req)

    usage = {
        "prompt_tokens": resp.metrics.prompt_tokens if resp.metrics else 0,
        "completion_tokens": resp.metrics.completion_tokens if resp.metrics else 0,
        "cache_hit_tokens": resp.metrics.cache_hit_tokens if resp.metrics else 0,
        "cache_miss_tokens": resp.metrics.cache_miss_tokens if resp.metrics else 0,
        "latency_secs": resp.latency_secs,
    }

    if event_log:
        # Emit llm.response_received
        evt = RuntimeEvent(
            event_type="llm.response_received",
            category=EventCategory.LLM,
            severity=EventSeverity.INFO,
            data={
                "model": "deepseek-v4-flash",
                "tokens_used": usage["prompt_tokens"] + usage["completion_tokens"],
                "latency_secs": usage["latency_secs"],
                "cache_hit_tokens": usage["cache_hit_tokens"],
                "cache_miss_tokens": usage["cache_miss_tokens"],
            },
        )
        event_log.append(evt)
        event_count += 1

    return resp.content, usage, event_count


def emit_event(event_log: EventLog, event_type: str, data: dict[str, Any], event_count: int = 0) -> int:
    """Append a synthetic RuntimeEvent. Returns updated event_count."""
    category = EventCategory.SYSTEM
    severity = EventSeverity.INFO
    if event_type.startswith("fault."):
        category = EventCategory.FAULT
        severity = EventSeverity.WARNING
    elif event_type.startswith("cloud."):
        category = EventCategory.TASK

    evt = RuntimeEvent(
        event_type=event_type,
        category=category,
        severity=severity,
        data=data,
    )
    event_log.append(evt)
    return event_count + 1
