from pathlib import Path

from agentos.benchmarks.cache_sim import CacheSimulatingMockBackend
from agentos.llm.base import LLMRequest


async def test_cache_sim_no_cache_on_different_prefixes():
    backend = CacheSimulatingMockBackend(latency_base=0.01)
    req1 = LLMRequest(
        agent_id="a1",
        messages=[
            {"role": "system", "content": "You are a planner. Background: project X."},
            {"role": "user", "content": "Plan task"},
        ],
    )
    req2 = LLMRequest(
        agent_id="a2",
        messages=[
            {"role": "system", "content": "Context: project X. You are a coder."},
            {"role": "user", "content": "Write code"},
        ],
    )

    resp1 = await backend.generate(req1)
    resp2 = await backend.generate(req2)

    assert resp1.metrics.cache_hit_tokens == 0
    assert resp2.metrics.cache_hit_tokens < resp2.metrics.prompt_tokens


async def test_cache_sim_cache_hit_on_shared_prefix():
    backend = CacheSimulatingMockBackend(latency_base=0.01)
    shared = "A" * 400
    req1 = LLMRequest(
        agent_id="a1",
        messages=[
            {"role": "system", "content": f"{shared}\n\nYou are a planner."},
            {"role": "user", "content": "Plan"},
        ],
    )
    req2 = LLMRequest(
        agent_id="a2",
        messages=[
            {"role": "system", "content": f"{shared}\n\nYou are a coder."},
            {"role": "user", "content": "Code"},
        ],
    )

    await backend.generate(req1)
    resp2 = await backend.generate(req2)

    assert resp2.metrics.cache_hit_tokens > 0
    assert resp2.metrics.cache_hit_ratio > 0.5


async def test_cache_sim_latency_reduced_on_cache_hit():
    backend = CacheSimulatingMockBackend(latency_base=0.05, cache_speedup=0.6)
    shared = "B" * 800
    req1 = LLMRequest(
        agent_id="a1",
        messages=[{"role": "system", "content": shared}, {"role": "user", "content": "q1"}],
    )
    req2 = LLMRequest(
        agent_id="a2",
        messages=[{"role": "system", "content": shared}, {"role": "user", "content": "q2"}],
    )

    resp1 = await backend.generate(req1)
    resp2 = await backend.generate(req2)

    assert resp2.metrics.latency_secs < resp1.metrics.latency_secs


async def test_cache_sim_reset_clears_state():
    backend = CacheSimulatingMockBackend(latency_base=0.01)
    req = LLMRequest(agent_id="a", messages=[{"role": "user", "content": "hi"}])
    await backend.generate(req)
    assert backend.call_count == 1

    backend.reset()
    assert backend.call_count == 0
    assert len(backend.requests) == 0


async def test_benchmark_comparison_runtime_beats_baseline(tmp_path: Path):
    from scripts.run_benchmark_comparison import run_baseline, run_runtime

    backend = CacheSimulatingMockBackend(latency_base=0.01)
    bl = await run_baseline(backend)
    rt = await run_runtime(backend, log_path=tmp_path / "events.jsonl")

    assert rt.tasks_completed == 6
    assert bl.tasks_completed < 6
    assert rt.fault_recovered is True
    assert bl.fault_recovered is False
    assert rt.cache_hit_ratio > bl.cache_hit_ratio
    assert rt.event_count > 0
