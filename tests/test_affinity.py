"""Tests for CacheAffinityAdvisor and its integration with DAGScheduler."""

from __future__ import annotations

import time

import pytest

from agentos.context.cache_tracker import CacheHeatTracker
from agentos.context.manager import ContextManager
from agentos.core.task import Task, TaskDAG, TaskStatus
from agentos.metrics.model import ModelCallMetrics
from agentos.scheduler.affinity import CacheAffinityAdvisor
from agentos.scheduler.dag import DAGScheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(task_id: str, context_refs: list[str] | None = None, **kw) -> Task:
    return Task(
        task_id=task_id,
        name=task_id,
        agent_role="worker",
        context_refs=context_refs or [],
        **kw,
    )


def _make_metrics(prompt_tokens: int = 100, cache_hit: int = 0) -> ModelCallMetrics:
    return ModelCallMetrics(
        backend="mock",
        model="test",
        prompt_tokens=prompt_tokens,
        cache_hit_tokens=cache_hit,
        cache_miss_tokens=prompt_tokens - cache_hit,
    )


# ---------------------------------------------------------------------------
# ContextManager.resolve_prefix_id
# ---------------------------------------------------------------------------

class TestResolvePrefixId:
    def test_returns_same_id_for_same_shared_refs(self):
        ctx = ContextManager()
        ctx.add_shared_ref("bg", "project background")
        ctx.add_shared_ref("repo", "repo summary")
        id1 = ctx.resolve_prefix_id(["bg", "repo"])
        id2 = ctx.resolve_prefix_id(["repo", "bg"])
        assert id1 == id2

    def test_different_content_different_id(self):
        ctx = ContextManager()
        ctx.add_shared_ref("bg", "project A background")
        id_a = ctx.resolve_prefix_id(["bg"])
        ctx2 = ContextManager()
        ctx2.add_shared_ref("bg", "project B background")
        id_b = ctx2.resolve_prefix_id(["bg"])
        assert id_a != id_b

    def test_no_refs_returns_sentinel(self):
        ctx = ContextManager()
        assert ctx.resolve_prefix_id([]) == "__no_prefix__"

    def test_none_uses_all_shared(self):
        ctx = ContextManager()
        ctx.add_shared_ref("bg", "project background")
        ctx.add_shared_ref("repo", "repo summary")
        id_all = ctx.resolve_prefix_id(None)
        id_explicit = ctx.resolve_prefix_id(["bg", "repo"])
        assert id_all == id_explicit

    def test_ignores_unknown_ref_names(self):
        ctx = ContextManager()
        ctx.add_shared_ref("bg", "project background")
        id1 = ctx.resolve_prefix_id(["bg", "nonexistent"])
        id2 = ctx.resolve_prefix_id(["bg"])
        assert id1 == id2

    def test_ignores_private_refs(self):
        ctx = ContextManager()
        ctx.add_shared_ref("bg", "shared content")
        ctx.add_private_ref("agent-1", "private content")
        id1 = ctx.resolve_prefix_id(None)
        id2 = ctx.resolve_prefix_id(["bg"])
        assert id1 == id2


# ---------------------------------------------------------------------------
# CacheAffinityAdvisor
# ---------------------------------------------------------------------------

class TestCacheAffinityAdvisor:
    def _setup(self) -> tuple[ContextManager, CacheHeatTracker, CacheAffinityAdvisor]:
        ctx = ContextManager()
        tracker = CacheHeatTracker(ttl=300.0)
        advisor = CacheAffinityAdvisor(ctx, tracker)
        return ctx, tracker, advisor

    def test_hot_prefix_sorted_first(self):
        ctx, tracker, advisor = self._setup()
        ctx.add_shared_ref("bg-a", "background A", priority=10)
        ctx.add_shared_ref("bg-b", "background B", priority=10)

        task_a = _make_task("t-a", context_refs=["bg-a"])
        task_b = _make_task("t-b", context_refs=["bg-b"])

        pid_a = advisor.resolve_task_prefix(task_a)
        tracker.record_call(pid_a, _make_metrics(cache_hit=80))

        result = advisor.sort_by_affinity([task_b, task_a])
        assert result[0].task_id == "t-a"

    def test_same_prefix_grouped(self):
        ctx, tracker, advisor = self._setup()
        ctx.add_shared_ref("bg", "shared background")

        tasks = [
            _make_task("t1", context_refs=["bg"]),
            _make_task("t2", context_refs=["bg"]),
            _make_task("t3", context_refs=["bg"]),
        ]

        result = advisor.sort_by_affinity(tasks)
        prefixes = [advisor.resolve_task_prefix(t) for t in result]
        assert prefixes[0] == prefixes[1] == prefixes[2]

    def test_empty_input(self):
        _, _, advisor = self._setup()
        assert advisor.sort_by_affinity([]) == []

    def test_no_heat_data_falls_back_to_critical_path(self):
        ctx, _, advisor = self._setup()
        ctx.add_shared_ref("bg", "background")

        task_low = _make_task("low", context_refs=["bg"])
        task_low.critical_path_score = 1.0
        task_high = _make_task("high", context_refs=["bg"])
        task_high.critical_path_score = 5.0

        result = advisor.sort_by_affinity([task_low, task_high])
        assert result[0].task_id == "high"

    def test_tasks_without_refs_get_same_no_prefix(self):
        ctx, _, advisor = self._setup()
        t1 = _make_task("t1")
        t2 = _make_task("t2")
        p1 = advisor.resolve_task_prefix(t1)
        p2 = advisor.resolve_task_prefix(t2)
        assert p1 == p2 == "__no_prefix__"


# ---------------------------------------------------------------------------
# DAGScheduler integration
# ---------------------------------------------------------------------------

class TestDAGSchedulerWithAffinity:
    async def test_affinity_reorders_batch(self):
        ctx = ContextManager()
        ctx.add_shared_ref("hot-ctx", "hot context content")
        ctx.add_shared_ref("cold-ctx", "cold context content")
        tracker = CacheHeatTracker(ttl=300.0)
        advisor = CacheAffinityAdvisor(ctx, tracker)

        hot_pid = ctx.resolve_prefix_id(["hot-ctx"])
        tracker.record_call(hot_pid, _make_metrics(cache_hit=90))

        dag = TaskDAG()
        dag.add_task(_make_task("cold-task", context_refs=["cold-ctx"]))
        dag.add_task(_make_task("hot-task", context_refs=["hot-ctx"]))

        scheduler = DAGScheduler(dag, affinity_advisor=advisor)
        batch = await scheduler.get_next_batch()
        assert batch[0].task_id == "hot-task"

    async def test_without_advisor_uses_default_sort(self):
        dag = TaskDAG()
        dag.add_task(_make_task("a", priority=1))
        dag.add_task(_make_task("b", priority=5))

        scheduler = DAGScheduler(dag)
        batch = await scheduler.get_next_batch()
        assert batch[0].task_id == "b"
