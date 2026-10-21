from agentos.llm.base import LLMRequest, LLMResponse, ModelRequest, ModelResponse
from agentos.metrics.model import ModelCallMetrics


def test_llm_request_accepts_runtime_metadata():
    req = LLMRequest(
        agent_id="agent-1",
        messages=[{"role": "user", "content": "hello"}],
        backend="deepseek",
        model="deepseek-v4-pro",
        task_id="task-1",
        trace_id="trace-1",
        context_refs=["ctx://task-bg"],
        prefix_id="prefix-abc",
    )
    assert req.backend == "deepseek"
    assert req.task_id == "task-1"
    assert req.context_refs == ["ctx://task-bg"]
    assert req.prefix_id == "prefix-abc"


def test_model_request_alias_keeps_existing_protocol_name():
    assert ModelRequest is LLMRequest
    assert ModelResponse is LLMResponse


def test_model_call_metrics_cache_ratio():
    metrics = ModelCallMetrics(
        backend="deepseek",
        model="deepseek-v4-pro",
        prompt_tokens=100,
        completion_tokens=20,
        cache_hit_tokens=80,
        cache_miss_tokens=20,
        latency_secs=1.25,
    )
    assert metrics.total_tokens == 120
    assert metrics.cache_hit_ratio == 0.8
    assert metrics.to_dict()["cache_miss_tokens"] == 20


def test_llm_response_carries_metrics():
    metrics = ModelCallMetrics(
        backend="mock",
        model="mock",
        prompt_tokens=10,
        completion_tokens=4,
        latency_secs=0.01,
    )
    resp = LLMResponse(
        content="ok",
        tokens_used=14,
        model="mock",
        latency_secs=0.01,
        metrics=metrics,
    )
    assert resp.metrics is metrics
    assert resp.tokens_used == 14
