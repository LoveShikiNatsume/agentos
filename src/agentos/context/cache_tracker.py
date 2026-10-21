from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from agentos.metrics.model import ModelCallMetrics


@dataclass
class PrefixHeatEntry:
    prefix_id: str
    last_call_time: float = 0.0
    last_hit_time: float = 0.0
    total_hits: int = 0
    total_calls: int = 0
    total_hit_tokens: int = 0
    total_miss_tokens: int = 0
    ewma_hit_ratio: float = 0.0
    ewma_prefill_secs_per_token: float = 0.0
    estimated_tokens: int = 0


@dataclass(frozen=True)
class CacheReusePrediction:
    prefix_id: str
    heat: float
    reuse_probability: float
    expected_hit_tokens: int
    expected_prefill_saved_secs: float
    confidence: float
    samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefix_id": self.prefix_id,
            "heat": self.heat,
            "reuse_probability": self.reuse_probability,
            "expected_hit_tokens": self.expected_hit_tokens,
            "expected_prefill_saved_secs": self.expected_prefill_saved_secs,
            "confidence": self.confidence,
            "samples": self.samples,
        }


class CacheHeatTracker:
    """Tracks which prompt prefixes are likely still hot in GPU KV cache.

    Uses exponential time decay to estimate whether a prefix is still cached.
    Exposes hot prefix info so the scheduler can prioritize cache-affine agents.
    """

    def __init__(
        self,
        ttl: float = 300.0,
        capacity: int = 64,
        hot_threshold: float = 0.3,
        ewma_alpha: float = 0.35,
    ) -> None:
        self._ttl = ttl
        self._capacity = capacity
        self._hot_threshold = hot_threshold
        self._ewma_alpha = ewma_alpha
        self._entries: dict[str, PrefixHeatEntry] = {}
        self._access_order: list[str] = []

    def record_call(
        self,
        prefix_id: str,
        metrics: ModelCallMetrics,
        timestamp: float | None = None,
    ) -> None:
        ts = timestamp if timestamp is not None else time.monotonic()

        if prefix_id not in self._entries:
            self._entries[prefix_id] = PrefixHeatEntry(prefix_id=prefix_id)

        entry = self._entries[prefix_id]
        entry.last_call_time = ts
        entry.total_calls += 1
        entry.estimated_tokens = max(entry.estimated_tokens, metrics.prompt_tokens)
        entry.total_miss_tokens += metrics.cache_miss_tokens

        token_total = metrics.cache_hit_tokens + metrics.cache_miss_tokens
        observed_hit_ratio = (
            metrics.cache_hit_tokens / token_total
            if token_total > 0
            else 0.0
        )
        entry.ewma_hit_ratio = self._ewma(
            previous=entry.ewma_hit_ratio,
            observed=observed_hit_ratio,
            sample_count=entry.total_calls,
        )

        prefill_tokens = int(
            metrics.metadata.get("prefill_miss_tokens", metrics.cache_miss_tokens)
        )
        prefill_secs = float(metrics.metadata.get("prefill_secs", 0.0))
        if prefill_tokens > 0 and prefill_secs > 0:
            entry.ewma_prefill_secs_per_token = self._ewma(
                previous=entry.ewma_prefill_secs_per_token,
                observed=prefill_secs / prefill_tokens,
                sample_count=entry.total_calls,
            )

        if metrics.cache_hit_tokens > 0:
            entry.last_hit_time = ts
            entry.total_hits += 1
            entry.total_hit_tokens += metrics.cache_hit_tokens

        if prefix_id in self._access_order:
            self._access_order.remove(prefix_id)
        self._access_order.append(prefix_id)

        self._evict_if_needed(ts)

    def get_prefix_heat(self, prefix_id: str, now: float | None = None) -> float:
        entry = self._entries.get(prefix_id)
        if entry is None or entry.total_calls == 0:
            return 0.0

        ts = now if now is not None else time.monotonic()
        ref_time = max(entry.last_call_time, entry.last_hit_time)
        elapsed = ts - ref_time
        return math.exp(-elapsed / self._ttl)

    def get_hot_prefixes(self, now: float | None = None) -> list[str]:
        ts = now if now is not None else time.monotonic()
        heated = [
            (pid, self.get_prefix_heat(pid, ts))
            for pid in self._entries
            if self.get_prefix_heat(pid, ts) >= self._hot_threshold
        ]
        heated.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in heated]

    def predict_reuse(
        self,
        prefix_id: str,
        now: float | None = None,
    ) -> CacheReusePrediction:
        entry = self._entries.get(prefix_id)
        if entry is None or entry.total_calls == 0:
            return CacheReusePrediction(
                prefix_id=prefix_id,
                heat=0.0,
                reuse_probability=0.0,
                expected_hit_tokens=0,
                expected_prefill_saved_secs=0.0,
                confidence=0.0,
                samples=0,
            )

        heat = self.get_prefix_heat(prefix_id, now)
        empirical_hit_ratio = entry.total_hits / entry.total_calls
        token_hit_ratio = (
            entry.total_hit_tokens
            / (entry.total_hit_tokens + entry.total_miss_tokens)
            if entry.total_hit_tokens + entry.total_miss_tokens > 0
            else 0.0
        )
        reuse_probability = max(
            0.0,
            min(
                1.0,
                heat
                * (
                    0.5 * entry.ewma_hit_ratio
                    + 0.3 * empirical_hit_ratio
                    + 0.2 * token_hit_ratio
                ),
            ),
        )
        expected_hit_tokens = int(round(entry.estimated_tokens * reuse_probability))
        expected_prefill_saved_secs = (
            expected_hit_tokens * entry.ewma_prefill_secs_per_token
        )
        confidence = min(1.0, entry.total_calls / 8.0)
        return CacheReusePrediction(
            prefix_id=prefix_id,
            heat=heat,
            reuse_probability=reuse_probability,
            expected_hit_tokens=expected_hit_tokens,
            expected_prefill_saved_secs=expected_prefill_saved_secs,
            confidence=confidence,
            samples=entry.total_calls,
        )

    def get_reuse_predictions(
        self,
        now: float | None = None,
    ) -> list[CacheReusePrediction]:
        predictions = [
            self.predict_reuse(prefix_id, now)
            for prefix_id in self._entries
        ]
        predictions.sort(
            key=lambda pred: (
                pred.expected_hit_tokens,
                pred.reuse_probability,
                pred.confidence,
            ),
            reverse=True,
        )
        return predictions

    def is_hot(self, prefix_id: str, now: float | None = None) -> bool:
        return self.get_prefix_heat(prefix_id, now) >= self._hot_threshold

    def get_entry(self, prefix_id: str) -> PrefixHeatEntry | None:
        return self._entries.get(prefix_id)

    def stats(self, now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.monotonic()
        hot_count = sum(1 for pid in self._entries if self.is_hot(pid, now))
        return {
            "tracked_prefixes": len(self._entries),
            "hot_prefixes": hot_count,
            "cold_prefixes": len(self._entries) - hot_count,
            "capacity": self._capacity,
            "ttl": self._ttl,
            "total_calls": sum(e.total_calls for e in self._entries.values()),
            "total_hits": sum(e.total_hits for e in self._entries.values()),
            "total_hit_tokens": sum(e.total_hit_tokens for e in self._entries.values()),
            "total_miss_tokens": sum(e.total_miss_tokens for e in self._entries.values()),
            "predictions": [
                prediction.to_dict()
                for prediction in self.get_reuse_predictions(now)
            ],
        }

    def _evict_if_needed(self, now: float) -> None:
        while len(self._entries) > self._capacity and self._access_order:
            oldest_id = self._access_order[0]
            if not self.is_hot(oldest_id, now):
                self._access_order.pop(0)
                del self._entries[oldest_id]
            else:
                break

    def _ewma(self, *, previous: float, observed: float, sample_count: int) -> float:
        if sample_count <= 1:
            return observed
        return self._ewma_alpha * observed + (1.0 - self._ewma_alpha) * previous
