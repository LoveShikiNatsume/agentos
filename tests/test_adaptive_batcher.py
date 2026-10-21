import asyncio

import pytest

from agentos.context.cache_tracker import CacheHeatTracker
from agentos.llm.base import LLMBackend, LLMRequest, LLMResponse
from agentos.llm.batcher import AdaptiveBatcher, BatchPolicy
from agentos.llm.gateway import ModelGateway
from agentos.metrics.model import ModelCallMetrics


class _SlowEchoBackend(LLMBackend):
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request.agent_id)
        await asyncio.sleep(0.001)
        metrics = ModelCallMetrics(
            backend="echo",
            model="echo",
            prompt_tokens=100,
            completion_tokens=10,
            cache_hit_tokens=50,
            cache_miss_tokens=50,
        )
        return LLMResponse(
            content=request.agent_id,
            tokens_used=110,
            model="echo",
            latency_secs=0.001,
            backend="echo",
            metrics=metrics,
        )


def _req(agent_id: str, prefix_id: str, **metadata):
    return LLMRequest(
        agent_id=agent_id,
        prefix_id=prefix_id,
        messages=[{"role": "user", "content": agent_id}],
        metadata=metadata,
    )


def test_adaptive_batcher_groups_by_prefix_and_respects_batch_size():
    batcher = AdaptiveBatcher(
        lambda req: req.prefix_id or "__no_prefix__",
        policy=BatchPolicy(max_batch_size=2),
    )
    plan = batcher.plan([
        _req("a1", "p1"),
        _req("a2", "p1"),
        _req("a3", "p1"),
        _req("b1", "p2"),
    ])

    assert [group.request_indices for group in plan.groups] == [[0, 1], [2], [3]]
    assert plan.groups[0].reason == "prefix_affinity"


def test_adaptive_batcher_prioritizes_predicted_cache_reuse():
    tracker = CacheHeatTracker(ttl=60.0)
    tracker.record_call(
        "hot",
        ModelCallMetrics(
            backend="vllm",
            model="qwen",
            prompt_tokens=2000,
            cache_hit_tokens=1800,
            cache_miss_tokens=200,
        ),
        timestamp=0.0,
    )
    batcher = AdaptiveBatcher(
        lambda req: req.prefix_id or "__no_prefix__",
        prediction_resolver=lambda prefix_id: tracker.predict_reuse(prefix_id, now=0.0),
    )

    plan = batcher.plan([_req("cold", "cold"), _req("hot", "hot")])

    assert plan.groups[0].prefix_id == "hot"
    assert plan.groups[0].predicted_hit_tokens > 0
    assert plan.groups[0].reason == "predicted_cache_reuse"


def test_adaptive_batcher_flushes_deadline_sensitive_group():
    batcher = AdaptiveBatcher(
        lambda req: req.prefix_id or "__no_prefix__",
        policy=BatchPolicy(max_wait_ms=30),
    )

    plan = batcher.plan([_req("interactive", "p1", deadline_ms=5)])

    assert plan.groups[0].reason == "deadline_flush"
    assert plan.groups[0].max_wait_ms == 5


@pytest.mark.asyncio
async def test_gateway_generate_adaptive_batch_preserves_order_and_records_plan():
    backend = _SlowEchoBackend()
    gateway = ModelGateway(backends={"default": backend})
    requests = [
        _req("a1", "p1"),
        _req("b1", "p2"),
        _req("a2", "p1"),
    ]

    responses = await gateway.generate_adaptive_batch(requests)

    assert [resp.content for resp in responses] == ["a1", "b1", "a2"]
    assert gateway.stats()["adaptive_batch_plan_count"] == 1
    assert gateway.stats()["last_adaptive_batch_plan"]["groups"][0]["prefix_id"] == "p1"
