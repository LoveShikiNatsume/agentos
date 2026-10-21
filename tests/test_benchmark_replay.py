import pytest

from agentos.benchmarks.attribution import build_runtime_attribution
from agentos.benchmarks.cost import CostModel
from agentos.benchmarks.features import WorkloadFeature
from agentos.benchmarks.replay import BenchmarkReplay


def _features() -> list[WorkloadFeature]:
    return [
        WorkloadFeature(
            source="llm",
            task_id="plan",
            workload_class="ai_inference",
            prompt_tokens=1000,
            completion_tokens=100,
            cache_hit_tokens=600,
            cache_miss_tokens=400,
            prefill_saved_tokens=600,
            latency_secs=1.2,
        ),
        WorkloadFeature(
            source="llm",
            task_id="code",
            workload_class="ai_inference",
            prompt_tokens=1400,
            completion_tokens=200,
            cache_hit_tokens=900,
            cache_miss_tokens=500,
            prefill_saved_tokens=900,
            latency_secs=2.0,
        ),
    ]


def test_benchmark_replay_estimates_no_context_ref_counterfactual():
    replay = BenchmarkReplay(
        _features(),
        context_stats={
            "duplicate_tokens": 700,
            "compressed_tokens": 300,
            "active_wss_tokens": 1200,
        },
    )

    comparison = replay.compare("no_context_ref")

    assert comparison["baseline"]["prompt_tokens"] == 3100
    assert comparison["runtime"]["prompt_tokens"] == 2400
    assert comparison["delta"]["prompt_tokens"] == 700


def test_benchmark_replay_estimates_no_prefix_cache_counterfactual():
    replay = BenchmarkReplay(_features())

    comparison = replay.compare("no_prefix_cache")

    assert comparison["baseline"]["cache_hit_tokens"] == 0
    assert comparison["baseline"]["cache_miss_tokens"] == 2400
    assert comparison["runtime"]["cache_hit_tokens"] == 1500
    assert comparison["delta"]["cache_miss_tokens"] == 1500


def test_runtime_attribution_summarizes_context_cache_and_cost():
    report = build_runtime_attribution(
        _features(),
        context_stats={
            "duplicate_tokens": 700,
            "compressed_tokens": 300,
            "active_wss_tokens": 1200,
            "context_reuse_ratio": 0.36,
        },
        cost_model=CostModel(
            prompt_cache_miss_per_million=1.0,
            prompt_cache_hit_per_million=0.1,
            completion_per_million=2.0,
            latency_per_second=0.05,
            prefill_saved_per_million=1.0,
        ),
    )

    assert report.prompt_tokens_saved_by_context_ref == 700
    assert report.prefill_tokens_saved_by_prefix_cache == 1500
    assert report.active_wss_reduced_by_compression == 300
    assert report.cache_miss_tokens_avoided == 1500
    assert report.estimated_cost is not None
    assert report.estimated_cost.total_cost == pytest.approx(0.16165)
