from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from agentos.benchmarks.features import WorkloadFeature
from agentos.metrics.model import ModelCallMetrics


@dataclass(frozen=True)
class CostModel:
    """Token/latency cost model shared by benchmark and dashboard code.

    Prices are expressed per one million tokens to match public API pricing
    tables. Local runs can leave token prices at zero and use latency/prefill
    proxy values instead.
    """

    prompt_cache_miss_per_million: float = 0.0
    prompt_cache_hit_per_million: float = 0.0
    completion_per_million: float = 0.0
    latency_per_second: float = 0.0
    prefill_saved_per_million: float = 0.0


@dataclass
class CostEstimate:
    prompt_cost: float = 0.0
    completion_cost: float = 0.0
    latency_cost: float = 0.0
    total_cost: float = 0.0
    prefill_saved_value: float = 0.0
    effective_cost: float = 0.0
    currency: str = "USD"
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metadata"] = self.metadata or {}
        return data


def _per_million(tokens: int, price: float) -> float:
    return (tokens / 1_000_000.0) * price


def estimate_feature_cost(
    feature: WorkloadFeature,
    model: CostModel,
    *,
    currency: str = "USD",
) -> CostEstimate:
    prompt_cost = (
        _per_million(feature.cache_miss_tokens, model.prompt_cache_miss_per_million)
        + _per_million(feature.cache_hit_tokens, model.prompt_cache_hit_per_million)
    )
    completion_cost = _per_million(
        feature.completion_tokens,
        model.completion_per_million,
    )
    latency_cost = feature.latency_secs * model.latency_per_second
    prefill_saved_value = _per_million(
        feature.prefill_saved_tokens,
        model.prefill_saved_per_million,
    )
    total = prompt_cost + completion_cost + latency_cost
    return CostEstimate(
        prompt_cost=prompt_cost,
        completion_cost=completion_cost,
        latency_cost=latency_cost,
        total_cost=total,
        prefill_saved_value=prefill_saved_value,
        effective_cost=max(0.0, total - prefill_saved_value),
        currency=currency,
        metadata={
            "source": feature.source,
            "task_id": feature.task_id,
            "workload_class": feature.workload_class,
            "cache_hit_ratio": feature.cache_hit_ratio,
            "prefill_saved_tokens": feature.prefill_saved_tokens,
        },
    )


def estimate_metrics_cost(
    metrics: ModelCallMetrics,
    model: CostModel,
    *,
    currency: str = "USD",
) -> CostEstimate:
    feature = WorkloadFeature(
        source="metrics",
        workload_class="ai_inference",
        prompt_tokens=metrics.prompt_tokens,
        completion_tokens=metrics.completion_tokens,
        cache_hit_tokens=metrics.cache_hit_tokens,
        cache_miss_tokens=metrics.cache_miss_tokens,
        cache_hit_ratio=metrics.cache_hit_ratio,
        prefill_saved_tokens=int(
            metrics.metadata.get("prefill_saved_tokens", metrics.cache_hit_tokens)
        ),
        latency_secs=metrics.latency_secs,
    )
    return estimate_feature_cost(feature, model, currency=currency)


def summarize_costs(estimates: list[CostEstimate]) -> dict[str, float]:
    return {
        "prompt_cost": sum(item.prompt_cost for item in estimates),
        "completion_cost": sum(item.completion_cost for item in estimates),
        "latency_cost": sum(item.latency_cost for item in estimates),
        "total_cost": sum(item.total_cost for item in estimates),
        "prefill_saved_value": sum(item.prefill_saved_value for item in estimates),
        "effective_cost": sum(item.effective_cost for item in estimates),
    }
