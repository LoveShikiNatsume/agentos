from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from agentos.metrics.model import ModelCallMetrics


@dataclass
class LLMRequest:
    agent_id: str
    messages: list[dict[str, Any]]
    model: str = "default"
    backend: str = "default"
    temperature: float = 0.7
    max_tokens: int | None = 4096
    task_id: str | None = None
    trace_id: str | None = None
    context_refs: list[str] = field(default_factory=list)
    prefix_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    tools: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LLMResponse:
    content: str
    tokens_used: int
    model: str
    latency_secs: float
    backend: str = "default"
    metrics: ModelCallMetrics | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


ModelRequest = LLMRequest
ModelResponse = LLMResponse


class LLMBackend(Protocol):
    async def generate(self, request: LLMRequest) -> LLMResponse: ...
