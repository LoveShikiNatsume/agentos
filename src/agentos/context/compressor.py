from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Compressor(Protocol):
    async def compress(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
    ) -> list[dict[str, Any]]: ...


@dataclass
class SlidingWindowCompressor:
    """Keep system prompt + the last *window_size* user/assistant turns."""

    window_size: int = 10

    async def compress(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        system: list[dict[str, Any]] = []
        conversation: list[dict[str, Any]] = []

        for msg in messages:
            if msg.get("role") == "system":
                system.append(msg)
            else:
                conversation.append(msg)

        if len(conversation) <= self.window_size:
            return system + conversation

        kept = conversation[-self.window_size :]
        return system + kept


@dataclass
class TokenBudgetCompressor:
    """Keep system prompt + as many recent turns as fit within *budget*."""

    budget: int = 4096
    token_estimator: Any = None

    async def compress(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        budget = max_tokens if max_tokens is not None else self.budget

        system: list[dict[str, Any]] = []
        conversation: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system":
                system.append(msg)
            else:
                conversation.append(msg)

        system_tokens = sum(self._estimate(m) for m in system)
        remaining = budget - system_tokens
        if remaining <= 0:
            return system

        kept: list[dict[str, Any]] = []
        used = 0
        for msg in reversed(conversation):
            cost = self._estimate(msg)
            if used + cost > remaining:
                break
            kept.append(msg)
            used += cost
        kept.reverse()
        return system + kept

    def _estimate(self, msg: dict[str, Any]) -> int:
        content = msg.get("content", "")
        if self.token_estimator is not None:
            return self.token_estimator(content)
        return max(1, len(content) // 4)
