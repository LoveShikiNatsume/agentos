import pytest

from agentos.benchmarks.cost import (
    CostModel,
    estimate_feature_cost,
    estimate_metrics_cost,
    summarize_costs,
)
from agentos.benchmarks.features import WorkloadFeature, build_workload_feature
from agentos.context.eviction import SegmentEvictor
from agentos.context.manager import ContextManager
from agentos.context.prefix import ContextPriority
from agentos.core.resources import ResourceQuota
from agentos.core.task import Task
from agentos.llm.base import LLMRequest
from agentos.metrics.model import ModelCallMetrics


def test_llm_call_feature_exposes_ai_inference_metrics():
    request = LLMRequest(
        agent_id="agent-review",
        task_id="review",
        prefix_id="prefix-hot",
        messages=[{"role": "user", "content": "review this"}],
        metadata={"expected_parallelism": 1},
    )
    metrics = ModelCallMetrics(
        backend="vllm",
        model="qwen",
        prompt_tokens=1000,
        completion_tokens=80,
        cache_hit_tokens=700,
        cache_miss_tokens=300,
        latency_secs=0.42,
        metadata={
            "prefill_tokens": 300,
            "decode_tokens": 80,
            "prefill_secs": 0.12,
            "decode_secs": 0.3,
            "prefill_decode_available": True,
        },
    )

    feature = WorkloadFeature.from_llm_call(
        request,
        metrics,
        backend="vllm",
        model="qwen",
        prefix_heat=0.9,
    )

    assert feature.workload_class == "ai_inference"
    assert feature.workload_hint == "ai_inference"
    assert feature.prefix_id == "prefix-hot"
    assert feature.prefix_heat == pytest.approx(0.9)
    assert feature.prompt_tokens == 1000
    assert feature.completion_tokens == 80
    assert feature.cache_hit_ratio == pytest.approx(0.7)
    assert feature.prefill_tokens == 300
    assert feature.decode_tokens == 80
    assert feature.prefill_secs == pytest.approx(0.12)
    assert feature.decode_secs == pytest.approx(0.3)
    assert feature.prefill_decode_available is True
    assert feature.prefill_saved_tokens == 700
    assert feature.prefill_miss_tokens == 300


def test_build_workload_feature_merges_task_context_and_resource_metadata():
    cm = ContextManager()
    cm.add_shared_ref("repo", "a" * 400, priority=ContextPriority.REPO_SUMMARY)
    cm.add_private_ref("agent-a", "b" * 200)
    cm.build_messages("agent-a", [{"role": "user", "content": "hello"}])
    cm.build_messages("agent-b", [{"role": "user", "content": "hello"}])

    task = Task(
        task_id="compile",
        name="compile",
        agent_role="builder",
        resource_requirements=ResourceQuota(
            cpu_max_percent=80.0,
            memory_max_mb=512,
            io_read_bps=1024,
        ),
        workload_hint="compile",
        io_profile="sequential_read",
        expected_parallelism=4,
    )

    feature = build_workload_feature(
        task=task,
        context_stats=cm.stats(),
        prefix_heat=0.25,
    )

    assert feature.task_id == "compile"
    assert feature.agent_role == "builder"
    assert feature.workload_hint == "compile"
    assert feature.io_profile == "sequential_read"
    assert feature.expected_parallelism == 4
    assert feature.prefix_heat == pytest.approx(0.25)
    assert feature.resource_requirements["memory_max_mb"] == 512
    assert task.metadata["workload_hint"] == "compile"
    assert task.metadata["io_profile"] == "sequential_read"
    assert task.metadata["expected_parallelism"] == 4
    assert feature.shared_refs == 1
    assert feature.private_refs == 1
    assert feature.total_refs == 2
    assert feature.active_wss_tokens >= 150
    assert feature.duplicate_tokens > 0
    assert feature.context_reuse_ratio > 0


def test_context_stats_report_compression_and_active_wss():
    cm = ContextManager(evictor=SegmentEvictor(), segment_budget=120)
    cm.add_shared_ref(
        "sys",
        "system " * 20,
        priority=ContextPriority.SYSTEM_INSTRUCTION,
    )
    cm.add_shared_ref("cold-a", "a" * 400)
    cm.add_shared_ref("cold-b", "b" * 400)

    cm.build_messages("agent-a", [{"role": "user", "content": "hello"}])
    stats = cm.stats()

    assert stats["shared_refs"] == 3
    assert stats["private_refs"] == 0
    assert stats["compressed_tokens"] > 0
    assert stats["compression"]["tokens_before"] > stats["compression"]["tokens_after"]
    assert stats["active_wss_tokens"] == stats["compression"]["tokens_after"]


def test_cost_model_estimates_cloud_and_local_proxy_costs():
    feature = WorkloadFeature(
        source="llm",
        task_id="infer",
        workload_class="ai_inference",
        completion_tokens=500,
        cache_hit_tokens=8000,
        cache_miss_tokens=2000,
        cache_hit_ratio=0.8,
        prefill_saved_tokens=8000,
        latency_secs=2.0,
    )
    model = CostModel(
        prompt_cache_miss_per_million=1.0,
        prompt_cache_hit_per_million=0.1,
        completion_per_million=2.0,
        latency_per_second=0.05,
        prefill_saved_per_million=1.0,
    )

    estimate = estimate_feature_cost(feature, model)

    assert estimate.prompt_cost == pytest.approx(0.0028)
    assert estimate.completion_cost == pytest.approx(0.001)
    assert estimate.latency_cost == pytest.approx(0.1)
    assert estimate.total_cost == pytest.approx(0.1038)
    assert estimate.prefill_saved_value == pytest.approx(0.008)
    assert estimate.effective_cost == pytest.approx(0.0958)


def test_cost_model_can_estimate_from_metrics_and_summarize():
    metrics = ModelCallMetrics(
        backend="deepseek",
        model="deepseek-chat",
        prompt_tokens=1000,
        completion_tokens=100,
        cache_hit_tokens=600,
        cache_miss_tokens=400,
        latency_secs=1.0,
    )
    model = CostModel(
        prompt_cache_miss_per_million=1.0,
        prompt_cache_hit_per_million=0.5,
        completion_per_million=2.0,
    )

    estimate = estimate_metrics_cost(metrics, model)
    summary = summarize_costs([estimate, estimate])

    assert estimate.prompt_cost == pytest.approx(0.0007)
    assert estimate.completion_cost == pytest.approx(0.0002)
    assert summary["total_cost"] == pytest.approx(estimate.total_cost * 2)
