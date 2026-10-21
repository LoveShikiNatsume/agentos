from __future__ import annotations

import hashlib

from agentos.context.token_counter import TokenCounter, get_default_counter


class SharedContextPool:
    def __init__(self, token_counter: TokenCounter | None = None) -> None:
        self._store: dict[str, str] = {}
        self._hits: dict[str, int] = {}
        self._ref_counts: dict[str, int] = {}
        self._token_estimates: dict[str, int] = {}
        self._counter = token_counter or get_default_counter()

    def store(self, key: str, content: str) -> str:
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        if content_hash not in self._store:
            self._store[content_hash] = content
            self._hits[content_hash] = 0
            self._ref_counts[content_hash] = 0
            self._token_estimates[content_hash] = self._counter.count(content)
        self._ref_counts[content_hash] += 1
        return content_hash

    def get(self, ref: str) -> str | None:
        content = self._store.get(ref)
        if content is not None:
            self._hits[ref] = self._hits.get(ref, 0) + 1
        return content

    def peek(self, ref: str) -> str | None:
        """Read content without counting a hit (for snapshot/inspection)."""
        return self._store.get(ref)

    def acquire(self, content_hash: str) -> None:
        if content_hash in self._ref_counts:
            self._ref_counts[content_hash] += 1

    def release(self, content_hash: str) -> None:
        if content_hash in self._ref_counts:
            self._ref_counts[content_hash] = max(0, self._ref_counts[content_hash] - 1)

    def token_estimate(self, content_hash: str) -> int:
        return self._token_estimates.get(content_hash, 0)

    def ref_count(self, content_hash: str) -> int:
        return self._ref_counts.get(content_hash, 0)

    def hits(self, content_hash: str) -> int:
        """Read-only reuse count. Unlike get(), does NOT increment hits."""
        return self._hits.get(content_hash, 0)

    def stats(self) -> dict[str, int]:
        token_savings = sum(
            self._hits[h] * self._token_estimates.get(h, 0)
            for h in self._hits
        )
        return {
            "unique_segments": len(self._store),
            "total_hits": sum(self._hits.values()),
            "total_bytes": sum(len(value) for value in self._store.values()),
            "total_estimated_tokens": sum(self._token_estimates.values()),
            "token_savings": token_savings,
        }
