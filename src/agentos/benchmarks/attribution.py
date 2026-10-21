from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from agentos.benchmarks.cost import CostEstimate, CostModel, estimate_feature_cost
from agentos.benchmarks.features import WorkloadFeature
from agentos.benchmarks.replay import BenchmarkReplay


@dataclass
class RuntimeAttributionReport:
    prompt_tokens_saved_by_context_ref: int = 0
    prefill_tokens_saved_by_prefix_cache: int = 0
    active_wss_reduced_by_compression: int = 0
    cache_miss_tokens_avoided: int = 0
    latency_secs: float = 0.0
    estimated_cost: CostEstimate | None = None
    replay: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["estimated_cost"] = (
            self.estimated_cost.to_dict()
            if self.estimated_cost is not None
            else None
        )
        return data


def build_runtime_attribution(
    features: Iterable[WorkloadFeature | dict[str, Any]],
    *,
    context_stats: dict[str, Any] | None = None,
    cost_model: CostModel | None = None,
) -> RuntimeAttributionReport:
    feature_list = [
        feature if isinstance(feature, WorkloadFeature) else _feature_from_dict(feature)
        for feature in features
    ]
    stats = dict(context_stats or {})
    replay = BenchmarkReplay(feature_list, context_stats=stats)
    no_context = replay.compare("no_context_ref")
    no_cache = replay.compare("no_prefix_cache")
    no_compaction = replay.compare("no_compaction")

    cost_estimates = []
    if cost_model is not None:
        cost_estimates = [
            estimate_feature_cost(feature, cost_model)
            for feature in feature_list
        ]

    return RuntimeAttributionReport(
        prompt_tokens_saved_by_context_ref=int(
            no_context["delta"]["prompt_tokens"]
        ),
        prefill_tokens_saved_by_prefix_cache=sum(
            int(feature.prefill_saved_tokens)
            for feature in feature_list
        ),
        active_wss_reduced_by_compression=int(
            no_compaction["delta"]["active_wss_tokens"]
        ),
        cache_miss_tokens_avoided=int(
            no_cache["delta"]["cache_miss_tokens"]
        ),
        latency_secs=sum(float(feature.latency_secs) for feature in feature_list),
        estimated_cost=_sum_costs(cost_estimates) if cost_estimates else None,
        replay={
            "no_context_ref": no_context,
            "no_prefix_cache": no_cache,
            "no_compaction": no_compaction,
        },
        metadata={
            "feature_count": len(feature_list),
            "context_reuse_ratio": float(stats.get("context_reuse_ratio", 0.0)),
            "active_wss_tokens": int(stats.get("active_wss_tokens", 0)),
        },
    )


def _feature_from_dict(data: dict[str, Any]) -> WorkloadFeature:
    allowed = set(WorkloadFeature.__dataclass_fields__)
    return WorkloadFeature(**{key: value for key, value in data.items() if key in allowed})


def _sum_costs(estimates: list[CostEstimate]) -> CostEstimate:
    return CostEstimate(
        prompt_cost=sum(item.prompt_cost for item in estimates),
        completion_cost=sum(item.completion_cost for item in estimates),
        latency_cost=sum(item.latency_cost for item in estimates),
        total_cost=sum(item.total_cost for item in estimates),
        prefill_saved_value=sum(item.prefill_saved_value for item in estimates),
        effective_cost=sum(item.effective_cost for item in estimates),
        metadata={"estimate_count": len(estimates)},
    )
