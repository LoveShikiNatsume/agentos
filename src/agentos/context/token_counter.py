from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class CharEstimateCounter:
    """Rough estimate: 1 token ~ 4 characters."""

    def count(self, text: str) -> int:
        return max(1, len(text) // 4)


class TiktokenCounter:
    """Precise counter using tiktoken. Falls back to CharEstimateCounter on import error."""

    def __init__(self, model: str = "cl100k_base") -> None:
        try:
            import tiktoken

            self._enc = tiktoken.get_encoding(model)
        except (ImportError, Exception):
            self._enc = None
        self._fallback = CharEstimateCounter()

    def count(self, text: str) -> int:
        if self._enc is not None:
            return len(self._enc.encode(text))
        return self._fallback.count(text)


_default_counter: TokenCounter = CharEstimateCounter()


def get_default_counter() -> TokenCounter:
    return _default_counter


def set_default_counter(counter: TokenCounter) -> None:
    global _default_counter
    _default_counter = counter
