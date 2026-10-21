from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from agentos.benchmarks.features import WorkloadFeature


@dataclass
class ReplayResult:
    policy: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    prefill_saved_tokens: int = 0
    duplicate_prompt_tokens: int = 0
    active_wss_tokens: int = 0
    compressed_tokens: int = 0
    latency_secs: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cache_hit_ratio(self) -> float:
        total = self.cache_hit_tokens + self.cache_miss_tokens
        return self.cache_hit_tokens / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["total_tokens"] = self.total_tokens
        data["cache_hit_ratio"] = self.cache_hit_ratio
        return data


class BenchmarkReplay:
    """Counterfactual replay over B-line workload features and context stats."""

    def __init__(
        self,
        features: Iterable[WorkloadFeature | dict[str, Any]],
        *,
        context_stats: dict[str, Any] | None = None,
        events: Iterable[dict[str, Any]] | None = None,
    ) -> None:
        self._features = [self._feature_dict(feature) for feature in features]
        self._context_stats = dict(context_stats or {})
        self._events = list(events or [])

    @staticmethod
    def _feature_dict(feature: WorkloadFeature | dict[str, Any]) -> dict[str, Any]:
        if isinstance(feature, WorkloadFeature):
            return feature.to_dict()
        return dict(feature)

    def replay(self, policy: str) -> ReplayResult:
        observed = self._observed(policy)
        duplicate_tokens = int(self._context_stats.get("duplicate_tokens", 0))
        compressed_tokens = int(self._context_stats.get("compressed_tokens", 0))
        active_wss_tokens = int(self._context_stats.get("active_wss_tokens", 0))

        if policy == "observed" or policy == "full_runtime":
            observed.duplicate_prompt_tokens = duplicate_tokens
            observed.compressed_tokens = compressed_tokens
            observed.active_wss_tokens = active_wss_tokens
            return observed

        if policy == "no_context_ref":
            observed.prompt_tokens += duplicate_tokens
            observed.duplicate_prompt_tokens = 0
            observed.metadata["counterfactual_added_prompt_tokens"] = duplicate_tokens
            return observed

        if policy == "no_prefix_cache":
            observed.cache_miss_tokens += observed.cache_hit_tokens
            observed.cache_hit_tokens = 0
            observed.prefill_saved_tokens = 0
            observed.metadata["counterfactual_removed_cache_hits"] = True
            return observed

        if policy == "no_compaction":
            observed.active_wss_tokens = active_wss_tokens + compressed_tokens
            observed.compressed_tokens = 0
            observed.metadata["counterfactual_added_wss_tokens"] = compressed_tokens
            return observed

        raise ValueError(f"unknown replay policy: {policy}")

    def compare(
        self,
        baseline_policy: str,
        runtime_policy: str = "full_runtime",
    ) -> dict[str, Any]:
        baseline = self.replay(baseline_policy)
        runtime = self.replay(runtime_policy)
        return {
            "baseline": baseline.to_dict(),
            "runtime": runtime.to_dict(),
            "delta": {
                "prompt_tokens": baseline.prompt_tokens - runtime.prompt_tokens,
                "cache_hit_tokens": runtime.cache_hit_tokens - baseline.cache_hit_tokens,
                "cache_miss_tokens": baseline.cache_miss_tokens - runtime.cache_miss_tokens,
                "prefill_saved_tokens": (
                    runtime.prefill_saved_tokens - baseline.prefill_saved_tokens
                ),
                "active_wss_tokens": baseline.active_wss_tokens - runtime.active_wss_tokens,
                "latency_secs": baseline.latency_secs - runtime.latency_secs,
            },
        }

    def _observed(self, policy: str) -> ReplayResult:
        result = ReplayResult(policy=policy)
        for feature in self._features:
            result.prompt_tokens += int(feature.get("prompt_tokens", 0))
            result.completion_tokens += int(feature.get("completion_tokens", 0))
            result.cache_hit_tokens += int(feature.get("cache_hit_tokens", 0))
            result.cache_miss_tokens += int(feature.get("cache_miss_tokens", 0))
            result.prefill_saved_tokens += int(feature.get("prefill_saved_tokens", 0))
            result.latency_secs += float(feature.get("latency_secs", 0.0))
        return result
