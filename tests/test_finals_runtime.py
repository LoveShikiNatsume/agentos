from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from agentos.context.eviction import SegmentEvictor
from agentos.context.manager import ContextManager
from agentos.context.types import ContextAccessDenied
from agentos.core.task import Task, TaskDAG, TaskStatus
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import AgentHandler, Runtime, WorkflowExecutionError


class Ask(AgentHandler):
    async def run(self, ctx):
        await ctx.llm_call([{"role": "user", "content": ctx.task.name}])
        return ctx.task.task_id


class AskBatch(AgentHandler):
    async def run(self, ctx):
        replies = await ctx.llm_batch_call([
            [{"role": "user", "content": "first"}],
            [{"role": "user", "content": "second"}],
        ])
        return len(replies)


def test_selected_refs_are_canonical_and_keep_only_own_private_context():
    cm = ContextManager()
    cm.add_shared("system", "system rules")
    cm.add_shared("repo", "selected repository")
    cm.add_shared("other", "unrelated material")
    cm.add_private("a", "private a")
    cm.add_private("b", "private b")

    prepared = cm.prepare_context(
        "a", [{"role": "user", "content": "question"}],
        context_ref_names=["ctx://repo", "system", "repo"],
    )

    assert prepared.messages[0]["content"] == (
        "system rules\n\nselected repository\n\nprivate a"
    )
    assert prepared.messages[1] == {"role": "user", "content": "question"}
    assert prepared.shared_refs == ("system", "repo")
    assert prepared.prefix_id == cm.resolve_prefix_id(["system", "ctx://repo"])
    assert "private b" not in str(prepared.messages)


def test_none_retains_all_shared_but_explicit_empty_clears_prefix():
    cm = ContextManager()
    cm.add_shared("shared", "shared data")
    assert "shared data" in str(cm.build_messages("a", []))
    prepared = cm.prepare_context("a", [], context_ref_names=[])
    assert prepared.messages == []
    assert prepared.shared_refs == ()
    assert prepared.prefix_id == "__no_prefix__"
    assert cm.prefix_diagnostics()["last_prefix_id"] == "__no_prefix__"


def test_selection_precedes_budget_eviction():
    cm = ContextManager(
        evictor=SegmentEvictor(similarity_threshold=1.0),
        segment_budget=20,
    )
    cm.add_shared_ref("unrelated-pinned", "unrelated " * 100, priority=0)
    cm.add_shared_ref("needed", "small useful input", priority=40)

    prepared = cm.prepare_context("a", [], context_ref_names=["needed"])

    assert prepared.shared_refs == ("needed",)
    assert prepared.messages[0]["content"] == "small useful input"


def test_request_prefix_describes_surviving_content_after_eviction():
    cm = ContextManager(
        evictor=SegmentEvictor(similarity_threshold=1.0, pin_threshold=-1),
        segment_budget=10,
    )
    cm.add_shared_ref("keep", "small context", priority=1)
    cm.add_shared_ref("drop", "large unrelated context " * 100, priority=50)

    prepared = cm.prepare_context("a", [], context_ref_names=["keep", "drop"])

    assert prepared.shared_refs == ("keep",)
    assert prepared.prefix_id == cm.resolve_prefix_id(["keep"])
    assert prepared.prefix_id != cm.resolve_prefix_id(["keep", "drop"])


def test_unknown_and_denied_explicit_refs_fail_before_assembly():
    cm = ContextManager()
    cm.add_shared("allowed", "ok")
    denied = cm.add_shared_ref("denied", "not readable")
    cm._refs["denied"] = replace(denied, permissions=frozenset())
    private = cm.add_private_ref("owner", "owner secret")

    with pytest.raises(KeyError, match="unknown ContextRef"):
        cm.prepare_context("a", [], context_ref_names=["missing"])
    with pytest.raises(ContextAccessDenied):
        cm.prepare_context("a", [], context_ref_names=["denied"])
    with pytest.raises(ContextAccessDenied):
        cm.prepare_context("a", [], context_ref_names=[private.uri])
    with pytest.raises(TypeError, match="collection"):
        cm.prepare_context("a", [], context_ref_names="allowed")


async def test_runtime_sends_task_selected_payload_and_matching_prefix():
    cm = ContextManager()
    cm.add_shared("first", "first task material")
    cm.add_shared("second", "second task material")
    backend = MockLLMBackend()
    dag = TaskDAG()
    dag.add_task(Task("a", "a", "ask", context_refs=["ctx://first"]))
    dag.add_task(Task("b", "b", "ask", context_refs=["second"]))
    runtime = Runtime(
        backend, {"ask": Ask()}, context_manager=cm, enable_kernel_probe=False,
    )

    assert await runtime.run(dag) == {"a": "a", "b": "b"}

    requests = {request.task_id: request for request in backend.requests}
    assert requests["a"].messages[0]["content"] == "first task material"
    assert requests["b"].messages[0]["content"] == "second task material"
    assert requests["a"].prefix_id == cm.resolve_prefix_id(["first"])
    assert requests["b"].prefix_id == cm.resolve_prefix_id(["second"])
    events = [e for e in runtime.events if e.event_type == "context.assembled"]
    assert {e.task_id: e.data["shared_refs"] for e in events} == {
        "a": ["first"], "b": ["second"],
    }


async def test_runtime_batch_calls_obey_same_selection_contract():
    cm = ContextManager()
    cm.add_shared("keep", "required")
    cm.add_shared("omit", "irrelevant")
    backend = MockLLMBackend()
    dag = TaskDAG()
    dag.add_task(Task("a", "a", "ask", context_refs=["keep"]))
    runtime = Runtime(
        backend, {"ask": AskBatch()}, context_manager=cm, enable_kernel_probe=False,
    )

    assert await runtime.run(dag) == {"a": 2}
    assert len(backend.requests) == 2
    assert all(r.messages[0]["content"] == "required" for r in backend.requests)
    assert all(r.prefix_id == cm.resolve_prefix_id(["keep"]) for r in backend.requests)


async def test_runtime_uses_post_eviction_prefix_for_llm_grouping():
    cm = ContextManager(
        evictor=SegmentEvictor(similarity_threshold=1.0, pin_threshold=-1),
        segment_budget=10,
    )
    cm.add_shared_ref("keep", "small context", priority=1)
    cm.add_shared_ref("drop", "large data " * 100, priority=50)
    backend = MockLLMBackend()
    dag = TaskDAG()
    dag.add_task(Task("a", "a", "ask", context_refs=["keep", "drop"]))
    runtime = Runtime(
        backend, {"ask": Ask()}, context_manager=cm, enable_kernel_probe=False,
    )

    await runtime.run(dag)
    assert backend.requests[0].prefix_id == cm.resolve_prefix_id(["keep"])
    assert "large data" not in str(backend.requests[0].messages)


class Work(AgentHandler):
    async def run(self, ctx):
        if ctx.task.task_id == "bad":
            raise ValueError("branch failed")
        return ctx.task.task_id


def branched_dag():
    dag = TaskDAG()
    dag.add_task(Task("bad", "bad", "work", priority=10))
    dag.add_task(Task("blocked", "blocked", "work", dependencies=["bad"]))
    dag.add_task(Task("blocked-later", "blocked-later", "work", dependencies=["blocked"]))
    dag.add_task(Task("good", "good", "work"))
    dag.add_task(Task("good-later", "good-later", "work", dependencies=["good"]))
    return dag


async def test_isolation_finishes_independent_branches_and_reports_partial_results():
    dag = branched_dag()
    runtime = Runtime(
        MockLLMBackend(), {"work": Work()}, max_batch_size=1,
        failure_policy="isolate", enable_kernel_probe=False,
    )

    with pytest.raises(WorkflowExecutionError) as caught:
        await asyncio.wait_for(runtime.run(dag), timeout=3)

    assert caught.value.results == {"good": "good", "good-later": "good-later"}
    assert caught.value.failures == {"bad": "branch failed"}
    assert set(caught.value.skipped_tasks) == {"blocked", "blocked-later"}
    assert dag.tasks["bad"].status == TaskStatus.FAILED
    assert dag.tasks["good-later"].status == TaskStatus.COMPLETED
    assert runtime.active_resource_reservations == {}
    assert runtime._active_dag is None


async def test_default_failure_policy_preserves_fail_fast_exception():
    dag = branched_dag()
    runtime = Runtime(
        MockLLMBackend(), {"work": Work()}, max_batch_size=1,
        enable_kernel_probe=False,
    )

    with pytest.raises(ValueError, match="branch failed"):
        await runtime.run(dag)

    assert dag.tasks["good"].status == TaskStatus.PENDING


async def test_isolation_closes_blocked_chains_even_with_limited_propagation_depth():
    dag = branched_dag()
    runtime = Runtime(
        MockLLMBackend(), {"work": Work()}, max_batch_size=1,
        failure_policy="isolate", propagation_depth=1, enable_kernel_probe=False,
    )

    with pytest.raises(WorkflowExecutionError) as caught:
        await asyncio.wait_for(runtime.run(dag), timeout=3)

    assert set(caught.value.skipped_tasks) == {"blocked", "blocked-later"}
    assert dag.is_complete()
    assert "good-later" in caught.value.results


async def test_handler_lookup_failure_isolated_before_agent_creation():
    dag = TaskDAG()
    dag.add_task(Task("missing", "missing", "not-registered", priority=10))
    dag.add_task(Task("blocked", "blocked", "work", dependencies=["missing"]))
    dag.add_task(Task("good", "good", "work"))
    runtime = Runtime(
        MockLLMBackend(), {"work": Work()}, max_batch_size=1,
        failure_policy="isolate", enable_kernel_probe=False,
    )

    with pytest.raises(WorkflowExecutionError) as caught:
        await asyncio.wait_for(runtime.run(dag), timeout=3)

    assert caught.value.results == {"good": "good"}
    assert caught.value.skipped_tasks == ["blocked"]
    assert "missing" in caught.value.failures


async def test_invalid_task_context_does_not_block_other_tasks_in_isolation_mode():
    cm = ContextManager()
    cm.add_shared("valid", "material")
    backend = MockLLMBackend()
    dag = TaskDAG()
    dag.add_task(Task("bad-ref", "bad-ref", "ask", context_refs=["missing"], priority=10))
    dag.add_task(Task("good", "good", "ask", context_refs=["valid"]))
    runtime = Runtime(
        backend, {"ask": Ask()}, context_manager=cm, failure_policy="isolate",
        max_batch_size=1, enable_kernel_probe=False,
    )

    with pytest.raises(WorkflowExecutionError) as caught:
        await runtime.run(dag)

    assert caught.value.results == {"good": "good"}
    assert list(caught.value.failures) == ["bad-ref"]
    assert [r.task_id for r in backend.requests] == ["good"]


async def test_external_cancellation_is_not_reported_as_workflow_failure():
    started = asyncio.Event()

    class Slow(AgentHandler):
        async def run(self, ctx):
            started.set()
            await asyncio.Event().wait()

    dag = TaskDAG()
    dag.add_task(Task("slow", "slow", "slow"))
    runtime = Runtime(
        MockLLMBackend(), {"slow": Slow()}, failure_policy="isolate",
        enable_kernel_probe=False,
    )
    running = asyncio.create_task(runtime.run(dag))
    await asyncio.wait_for(started.wait(), timeout=3)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert runtime.active_resource_reservations == {}
    assert runtime._active_dag is None


def test_invalid_failure_policy_is_rejected():
    with pytest.raises(ValueError, match="failure_policy"):
        Runtime(MockLLMBackend(), {}, failure_policy="success-anyway")


def test_in_place_snapshot_restore_replaces_only_owned_private_refs():
    cm = ContextManager()
    cm.add_shared("shared", "old shared")
    original = cm.add_private_ref("a", "saved work")
    snapshot = cm.export_agent_context("a")
    scratch = cm.add_private_ref("a", "failed scratch")
    other = cm.add_private_ref("b", "another task")
    cm.add_shared("shared", "concurrent update")

    cm.import_agent_context(snapshot, restore_shared=False)

    assert cm.read("a", original.name) == "saved work"
    assert cm.get_ref(scratch.name) is None
    assert cm.read("b", other.name) == "another task"
    assert cm.read("a", "shared") == "concurrent update"
    assert cm._pool.ref_count(original.content_hash) == 1
    assert cm._pool.ref_count(scratch.content_hash) == 0
    appended = cm.add_private_ref("a", "new successful work")
    assert appended.name != original.name
    assert cm.read("a", original.name) == "saved work"


def test_snapshot_cannot_overwrite_another_agents_private_state():
    cm = ContextManager()
    own = cm.add_private_ref("a", "own")
    other = cm.add_private_ref("b", "other")
    snapshot = cm.export_agent_context("a")
    snapshot["private"][0]["name"] = other.name

    with pytest.raises(ContextAccessDenied):
        cm.import_agent_context(snapshot, restore_shared=False)
    assert cm.read("a", own.name) == "own"
    assert cm.read("b", other.name) == "other"


async def test_context_retry_keeps_concurrent_shared_and_other_private_updates():
    from agentos.fault.recovery import ContextRetryPolicy

    cm = ContextManager()
    cm.add_shared("shared", "initial")
    scratch_written = asyncio.Event()
    update_written = asyncio.Event()

    class Retrying(AgentHandler):
        calls = 0

        async def run(self, ctx):
            self.calls += 1
            if self.calls == 1:
                cm.add_private(ctx.acb.agent_id, "invalid partial output")
                scratch_written.set()
                await update_written.wait()
                raise RuntimeError("transient")
            payload = cm.build_messages(ctx.acb.agent_id, [])
            assert "invalid partial output" not in str(payload)
            assert "updated by healthy task" in str(payload)
            assert "healthy private output" not in str(payload)
            return "retried cleanly"

    class Updating(AgentHandler):
        async def run(self, ctx):
            await scratch_written.wait()
            cm.add_shared("shared", "updated by healthy task")
            cm.add_private(ctx.acb.agent_id, "healthy private output")
            update_written.set()
            return ctx.acb.agent_id

    dag = TaskDAG()
    dag.add_task(Task("retry", "retry", "retry"))
    dag.add_task(Task("update", "update", "update"))
    rt = Runtime(
        MockLLMBackend(), {"retry": Retrying(), "update": Updating()},
        context_manager=cm, recovery_policy=ContextRetryPolicy(max_retries=1),
        max_batch_size=2, enable_kernel_probe=False,
    )
    results = await asyncio.wait_for(rt.run(dag), timeout=2)
    assert results["retry"] == "retried cleanly"
    assert "healthy private output" in str(cm.build_messages(results["update"], []))
    restored = [e for e in rt.events if e.event_type == "fault.context_restored"]
    assert restored[0].data["shared_context_rewound"] is False
