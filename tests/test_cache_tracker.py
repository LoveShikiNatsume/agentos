import asyncio
import math

import pytest

from agentos.context.cache_tracker import CacheHeatTracker, PrefixHeatEntry
from agentos.llm.base import LLMRequest
from agentos.llm.gateway import ModelGateway
from agentos.llm.mock import MockLLMBackend
from agentos.metrics.model import ModelCallMetrics


def _metrics(prompt=100, hit=0, miss=100):
    return ModelCallMetrics(
        backend="test", model="test",
        prompt_tokens=prompt, cache_hit_tokens=hit, cache_miss_tokens=miss,
    )


class TestCacheHeatTracker:
    def test_record_and_query(self):
        tracker = CacheHeatTracker(ttl=60.0)
        tracker.record_call("pfx_a", _metrics(hit=64, miss=36), timestamp=100.0)

        assert tracker.get_prefix_heat("pfx_a", now=100.0) > 0.99
        assert "pfx_a" in tracker.get_hot_prefixes(now=100.0)

    def test_unknown_prefix_returns_zero(self):
        tracker = CacheHeatTracker()
        assert tracker.get_prefix_heat("unknown") == 0.0
        assert not tracker.is_hot("unknown")

    def test_heat_decay(self):
        tracker = CacheHeatTracker(ttl=60.0)
        tracker.record_call("pfx_a", _metrics(), timestamp=0.0)

        heat_0 = tracker.get_prefix_heat("pfx_a", now=0.0)
        heat_30 = tracker.get_prefix_heat("pfx_a", now=30.0)
        heat_60 = tracker.get_prefix_heat("pfx_a", now=60.0)
        heat_120 = tracker.get_prefix_heat("pfx_a", now=120.0)

        assert heat_0 > 0.99
        assert 0.5 < heat_30 < 0.7
        assert abs(heat_60 - math.exp(-1)) < 0.01
        assert heat_120 < 0.15

    def test_cold_prefix_not_hot(self):
        tracker = CacheHeatTracker(ttl=60.0, hot_threshold=0.3)
        tracker.record_call("pfx_a", _metrics(), timestamp=0.0)

        assert tracker.is_hot("pfx_a", now=0.0)
        assert not tracker.is_hot("pfx_a", now=200.0)
        assert "pfx_a" not in tracker.get_hot_prefixes(now=200.0)

    def test_capacity_eviction(self):
        tracker = CacheHeatTracker(ttl=60.0, capacity=3)

        tracker.record_call("a", _metrics(), timestamp=0.0)
        tracker.record_call("b", _metrics(), timestamp=1.0)
        tracker.record_call("c", _metrics(), timestamp=2.0)
        assert len(tracker.get_hot_prefixes(now=2.0)) == 3

        tracker.record_call("d", _metrics(), timestamp=100.0)
        assert "d" in tracker.get_hot_prefixes(now=100.0)
        assert tracker.get_entry("a") is None

    def test_multiple_prefixes_sorted_by_heat(self):
        tracker = CacheHeatTracker(ttl=60.0)
        tracker.record_call("old", _metrics(), timestamp=0.0)
        tracker.record_call("mid", _metrics(), timestamp=10.0)
        tracker.record_call("new", _metrics(), timestamp=20.0)

        hot = tracker.get_hot_prefixes(now=20.0)
        assert hot[0] == "new"
        assert hot[-1] == "old"

    def test_is_hot(self):
        tracker = CacheHeatTracker(ttl=60.0, hot_threshold=0.5)
        tracker.record_call("pfx", _metrics(), timestamp=0.0)

        assert tracker.is_hot("pfx", now=0.0)
        assert not tracker.is_hot("pfx", now=60.0)

    def test_stats(self):
        tracker = CacheHeatTracker(ttl=60.0, capacity=64)
        tracker.record_call("a", _metrics(hit=50, miss=50), timestamp=0.0)
        tracker.record_call("b", _metrics(hit=0, miss=100), timestamp=1.0)
        tracker.record_call("a", _metrics(hit=80, miss=20), timestamp=2.0)

        s = tracker.stats()
        assert s["tracked_prefixes"] == 2
        assert s["total_calls"] == 3
        assert s["total_hits"] == 2
        assert s["capacity"] == 64

    def test_record_updates_estimated_tokens(self):
        tracker = CacheHeatTracker(ttl=60.0)
        tracker.record_call("pfx", _metrics(prompt=100), timestamp=0.0)
        assert tracker.get_entry("pfx").estimated_tokens == 100

        tracker.record_call("pfx", _metrics(prompt=200), timestamp=1.0)
        assert tracker.get_entry("pfx").estimated_tokens == 200

    def test_hit_tokens_accumulate(self):
        tracker = CacheHeatTracker(ttl=60.0)
        tracker.record_call("pfx", _metrics(hit=64, miss=36), timestamp=0.0)
        tracker.record_call("pfx", _metrics(hit=128, miss=0), timestamp=1.0)

        entry = tracker.get_entry("pfx")
        assert entry.total_hit_tokens == 64 + 128
        assert entry.total_hits == 2
        assert entry.total_calls == 2

    def test_no_hit_still_tracks(self):
        tracker = CacheHeatTracker(ttl=60.0)
        tracker.record_call("pfx", _metrics(hit=0, miss=100), timestamp=0.0)

        entry = tracker.get_entry("pfx")
        assert entry.total_hits == 0
        assert entry.total_calls == 1
        assert tracker.is_hot("pfx", now=0.0)


class TestGatewayIntegration:
    @pytest.mark.asyncio
    async def test_gateway_with_tracker(self):
        tracker = CacheHeatTracker(ttl=60.0)
        backend = MockLLMBackend()
        gw = ModelGateway(
            backends={"default": backend},
            cache_tracker=tracker,
        )

        req = LLMRequest(
            agent_id="agent1",
            messages=[
                {"role": "system", "content": "You are a planner."},
                {"role": "user", "content": "hello"},
            ],
        )
        await gw.generate(req)

        assert tracker.stats()["total_calls"] == 1
        assert tracker.stats()["tracked_prefixes"] == 1

    @pytest.mark.asyncio
    async def test_gateway_without_tracker(self):
        backend = MockLLMBackend()
        gw = ModelGateway(backends={"default": backend})

        req = LLMRequest(
            agent_id="agent1",
            messages=[{"role": "user", "content": "hello"}],
        )
        resp = await gw.generate(req)
        assert resp.content is not None

    @pytest.mark.asyncio
    async def test_same_prefix_tracked_together(self):
        tracker = CacheHeatTracker(ttl=60.0)
        gw = ModelGateway(
            backends={"default": MockLLMBackend()},
            cache_tracker=tracker,
        )

        shared_system = "Project background: building an OS for agents."
        for agent_id in ["planner", "coder", "tester"]:
            req = LLMRequest(
                agent_id=agent_id,
                messages=[
                    {"role": "system", "content": shared_system},
                    {"role": "user", "content": f"I am {agent_id}"},
                ],
            )
            await gw.generate(req)

        assert tracker.stats()["tracked_prefixes"] == 1
        assert tracker.stats()["total_calls"] == 3
