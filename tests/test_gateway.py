import asyncio
import time

import pytest

from agentos.core.task import Task, TaskDAG
from agentos.llm.base import LLMBackend, LLMRequest, LLMResponse
from agentos.llm.gateway import ModelGateway
from agentos.llm.mock import MockLLMBackend
from agentos.metrics.model import ModelCallMetrics
from agentos.runtime.engine import AgentHandler, Runtime


class _BareBackend:
    """A backend that returns no metrics (simulates legacy backends)."""

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            content="bare response",
            tokens_used=10,
            model="bare-model",
            latency_secs=0.001,
            metadata={"prompt_tokens": 5, "completion_tokens": 5},
        )


async def test_routes_to_named_backend():
    alpha = MockLLMBackend(responses={"default": "alpha response"})
    beta = MockLLMBackend(responses={"default": "beta response"})
    gw = ModelGateway(backends={"alpha": alpha, "beta": beta}, default_backend="alpha")

    req = LLMRequest(agent_id="a1", messages=[{"role": "user", "content": "hi"}], backend="beta")
    resp = await gw.generate(req)
    assert resp.content == "beta response"
    assert beta.call_count == 1
    assert alpha.call_count == 0


async def test_uses_default_backend():
    mock = MockLLMBackend(responses={"default": "default response"})
    gw = ModelGateway(backends={"main": mock}, default_backend="main")

    req = LLMRequest(agent_id="a1", messages=[{"role": "user", "content": "hi"}])
    resp = await gw.generate(req)
    assert resp.content == "default response"
    assert mock.call_count == 1


async def test_fallback_to_default_on_unknown_backend():
    mock = MockLLMBackend(responses={"default": "fallback"})
    gw = ModelGateway(backends={"main": mock}, default_backend="main")

    req = LLMRequest(agent_id="a1", messages=[{"role": "user", "content": "hi"}], backend="nonexistent")
    resp = await gw.generate(req)
    assert resp.content == "fallback"


async def test_raises_on_unknown_backend_no_default():
    gw = ModelGateway(backends={"alpha": MockLLMBackend()}, default_backend="missing")

    req = LLMRequest(agent_id="a1", messages=[{"role": "user", "content": "hi"}], backend="nope")
    with pytest.raises(ValueError, match="No backend registered"):
        await gw.generate(req)


async def test_populates_metrics_when_backend_returns_none():
    gw = ModelGateway(backends={"bare": _BareBackend()}, default_backend="bare")

    req = LLMRequest(agent_id="a1", messages=[{"role": "user", "content": "hi"}])
    resp = await gw.generate(req)
    assert resp.metrics is not None
    assert isinstance(resp.metrics, ModelCallMetrics)
    assert resp.metrics.backend == "bare"
    assert resp.metrics.prompt_tokens == 5
    assert resp.metrics.completion_tokens == 5


async def test_preserves_existing_metrics():
    mock = MockLLMBackend(responses={"default": "hello"})
    gw = ModelGateway(backends={"mock": mock}, default_backend="mock")

    req = LLMRequest(agent_id="a1", messages=[{"role": "user", "content": "test content here"}])
    resp = await gw.generate(req)
    assert resp.metrics is not None
    assert resp.metrics.cache_hit_tokens == 0
    assert resp.metrics.cache_miss_tokens == resp.metrics.prompt_tokens
    assert resp.metrics.backend == "mock"


async def test_stats_tracks_calls():
    mock = MockLLMBackend(responses={"default": "ok"})
    gw = ModelGateway(backends={"main": mock}, default_backend="main")

    req = LLMRequest(agent_id="a1", messages=[{"role": "user", "content": "hi"}])
    await gw.generate(req)
    await gw.generate(req)

    stats = gw.stats()
    assert stats["total_calls"] == 2
    assert stats["calls_by_backend"]["main"] == 2
    assert "main" in stats["registered_backends"]


async def test_register_backend():
    gw = ModelGateway(default_backend="main")
    assert gw.backend_names == []

    mock = MockLLMBackend(responses={"default": "registered"})
    gw.register("main", mock)
    assert "main" in gw.backend_names
    assert gw.get_backend("main") is mock

    req = LLMRequest(agent_id="a1", messages=[{"role": "user", "content": "hi"}])
    resp = await gw.generate(req)
    assert resp.content == "registered"


class _EchoHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context):
        resp = await ctx.llm_call([{"role": "user", "content": "hello"}])
        return resp.content


async def test_works_as_runtime_llm_backend():
    mock = MockLLMBackend(responses={"default": "gateway works"})
    gw = ModelGateway(backends={"mock": mock}, default_backend="mock")

    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="echo", agent_role="echo"))

    rt = Runtime(llm_backend=gw, handlers={"echo": _EchoHandler()})
    results = await rt.run(dag)
    assert results["t1"] == "gateway works"
    assert gw.stats()["total_calls"] == 1


class _FakeDeepSeekBackend:
    """Simulates a DeepSeekBackend returning cache metrics."""

    async def generate(self, request: LLMRequest) -> LLMResponse:
        metrics = ModelCallMetrics(
            backend="deepseek",
            model="deepseek-chat",
            prompt_tokens=100,
            completion_tokens=20,
            cache_hit_tokens=80,
            cache_miss_tokens=20,
            latency_secs=0.5,
        )
        return LLMResponse(
            content="deepseek reply",
            tokens_used=120,
            model="deepseek-chat",
            latency_secs=0.5,
            backend="deepseek",
            metrics=metrics,
            metadata={"usage": {"prompt_cache_hit_tokens": 80}},
        )


async def test_gateway_routes_to_deepseek_backend():
    ds = _FakeDeepSeekBackend()
    gw = ModelGateway(backends={"deepseek": ds, "mock": MockLLMBackend()}, default_backend="mock")

    req = LLMRequest(agent_id="a1", messages=[{"role": "user", "content": "hi"}], backend="deepseek")
    resp = await gw.generate(req)
    assert resp.content == "deepseek reply"
    assert resp.metrics.backend == "deepseek"
    assert resp.metrics.cache_hit_tokens == 80


async def test_gateway_preserves_deepseek_cache_metrics():
    ds = _FakeDeepSeekBackend()
    gw = ModelGateway(backends={"deepseek": ds}, default_backend="deepseek")

    req = LLMRequest(agent_id="a1", messages=[{"role": "user", "content": "hi"}])
    resp = await gw.generate(req)
    assert resp.metrics.cache_hit_tokens == 80
    assert resp.metrics.cache_miss_tokens == 20
    assert resp.metrics.cache_hit_ratio == pytest.approx(0.8)


async def test_gateway_records_llm_workload_feature():
    ds = _FakeDeepSeekBackend()
    gw = ModelGateway(backends={"deepseek": ds}, default_backend="deepseek")

    req = LLMRequest(
        agent_id="agent-a",
        task_id="task-a",
        prefix_id="prefix-a",
        messages=[{"role": "user", "content": "hi"}],
    )
    await gw.generate(req)

    features = gw.workload_features()
    assert len(features) == 1
    feature = features[0]
    assert feature.source == "llm"
    assert feature.workload_class == "ai_inference"
    assert feature.task_id == "task-a"
    assert feature.agent_id == "agent-a"
    assert feature.prefix_id == "prefix-a"
    assert feature.cache_hit_ratio == pytest.approx(0.8)
    assert feature.prefill_saved_tokens == 80
    assert gw.stats()["workload_feature_count"] == 1


class TestGroupByPrefix:
    def test_groups_by_explicit_prefix_id(self):
        gw = ModelGateway(backends={"mock": MockLLMBackend()}, default_backend="mock")
        reqs = [
            LLMRequest(agent_id="a1", messages=[{"role": "user", "content": "hi"}], prefix_id="group-a"),
            LLMRequest(agent_id="a2", messages=[{"role": "user", "content": "hello"}], prefix_id="group-a"),
            LLMRequest(agent_id="a3", messages=[{"role": "user", "content": "hey"}], prefix_id="group-b"),
        ]
        groups = gw.group_by_prefix(reqs)
        assert len(groups) == 2
        assert len(groups["group-a"]) == 2
        assert len(groups["group-b"]) == 1

    def test_groups_by_system_prompt_hash(self):
        gw = ModelGateway(backends={"mock": MockLLMBackend()}, default_backend="mock")
        sys_prompt = "you are a helpful assistant"
        reqs = [
            LLMRequest(agent_id="a1", messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": "q1"}]),
            LLMRequest(agent_id="a2", messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": "q2"}]),
            LLMRequest(agent_id="a3", messages=[{"role": "system", "content": "different sys"}, {"role": "user", "content": "q3"}]),
        ]
        groups = gw.group_by_prefix(reqs)
        assert len(groups) == 2
        group_sizes = sorted(len(v) for v in groups.values())
        assert group_sizes == [1, 2]

    def test_no_prefix_group(self):
        gw = ModelGateway(backends={"mock": MockLLMBackend()}, default_backend="mock")
        reqs = [
            LLMRequest(agent_id="a1", messages=[{"role": "user", "content": "q1"}]),
            LLMRequest(agent_id="a2", messages=[{"role": "user", "content": "q2"}]),
        ]
        groups = gw.group_by_prefix(reqs)
        assert len(groups) == 1
        assert "__no_prefix__" in groups

    def test_empty_list(self):
        gw = ModelGateway(backends={"mock": MockLLMBackend()}, default_backend="mock")
        groups = gw.group_by_prefix([])
        assert groups == {}


class TestGenerateBatch:
    async def test_batch_returns_all_responses(self):
        mock = MockLLMBackend(responses={"default": "batch reply"})
        gw = ModelGateway(backends={"mock": mock}, default_backend="mock")
        reqs = [
            LLMRequest(agent_id="a1", messages=[{"role": "user", "content": "q1"}]),
            LLMRequest(agent_id="a2", messages=[{"role": "user", "content": "q2"}]),
            LLMRequest(agent_id="a3", messages=[{"role": "user", "content": "q3"}]),
        ]
        responses = await gw.generate_batch(reqs)
        assert len(responses) == 3
        assert all(r.content == "batch reply" for r in responses)
        assert mock.call_count == 3

    async def test_batch_empty_list(self):
        gw = ModelGateway(backends={"mock": MockLLMBackend()}, default_backend="mock")
        responses = await gw.generate_batch([])
        assert responses == []

    async def test_batch_single_request(self):
        mock = MockLLMBackend(responses={"default": "single"})
        gw = ModelGateway(backends={"mock": mock}, default_backend="mock")
        reqs = [LLMRequest(agent_id="a1", messages=[{"role": "user", "content": "q"}])]
        responses = await gw.generate_batch(reqs)
        assert len(responses) == 1
        assert responses[0].content == "single"

    async def test_batch_preserves_order_across_prefix_groups(self):
        mock = MockLLMBackend(responses={"default": "ok"})
        gw = ModelGateway(backends={"mock": mock}, default_backend="mock")
        reqs = [
            LLMRequest(agent_id="a1", messages=[{"role": "system", "content": "sys-A"}, {"role": "user", "content": "q1"}]),
            LLMRequest(agent_id="a2", messages=[{"role": "system", "content": "sys-B"}, {"role": "user", "content": "q2"}]),
            LLMRequest(agent_id="a3", messages=[{"role": "system", "content": "sys-A"}, {"role": "user", "content": "q3"}]),
        ]
        responses = await gw.generate_batch(reqs)
        assert len(responses) == 3
        received_agents = [mock.requests[i].agent_id for i in range(3)]
        assert "a1" in received_agents
        assert "a2" in received_agents
        assert "a3" in received_agents

    async def test_batch_same_prefix_runs_sequentially(self):
        """Requests with the same prefix should run one after another, not concurrently."""
        call_times: list[tuple[str, float, float]] = []

        class _TimingBackend:
            async def generate(self, request: LLMRequest) -> LLMResponse:
                start = time.monotonic()
                await asyncio.sleep(0.05)
                end = time.monotonic()
                call_times.append((request.agent_id, start, end))
                return LLMResponse(
                    content="ok", tokens_used=10, model="test", latency_secs=0.05,
                    metrics=ModelCallMetrics(backend="test", model="test", prompt_tokens=10),
                )

        gw = ModelGateway(backends={"t": _TimingBackend()}, default_backend="t")
        sys_prompt = "shared system prompt for all agents"
        reqs = [
            LLMRequest(agent_id="a1", messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": "q1"}]),
            LLMRequest(agent_id="a2", messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": "q2"}]),
        ]
        await gw.generate_batch(reqs)

        assert len(call_times) == 2
        _, start1, end1 = call_times[0]
        _, start2, _ = call_times[1]
        assert start2 >= end1 - 0.01

    async def test_batch_different_prefixes_run_in_parallel(self):
        """Requests with different prefixes should run concurrently."""
        call_times: list[tuple[str, float, float]] = []

        class _TimingBackend:
            async def generate(self, request: LLMRequest) -> LLMResponse:
                start = time.monotonic()
                await asyncio.sleep(0.05)
                end = time.monotonic()
                call_times.append((request.agent_id, start, end))
                return LLMResponse(
                    content="ok", tokens_used=10, model="test", latency_secs=0.05,
                    metrics=ModelCallMetrics(backend="test", model="test", prompt_tokens=10),
                )

        gw = ModelGateway(backends={"t": _TimingBackend()}, default_backend="t")
        reqs = [
            LLMRequest(agent_id="a1", messages=[{"role": "system", "content": "sys-A"}, {"role": "user", "content": "q1"}]),
            LLMRequest(agent_id="a2", messages=[{"role": "system", "content": "sys-B"}, {"role": "user", "content": "q2"}]),
        ]
        await gw.generate_batch(reqs)

        assert len(call_times) == 2
        _, start1, _ = call_times[0]
        _, start2, _ = call_times[1]
        assert abs(start1 - start2) < 0.03
