import pytest

from agentos.context.manager import ContextManager
from agentos.core.task import Task, TaskDAG, TaskStatus
from agentos.core.types import AgentState
from agentos.fault.recovery import (
    ContextRetryPolicy,
    FallbackPolicy,
    RecoveryAction,
    RetryPolicy,
    SkipPolicy,
)
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import AgentHandler, Runtime


class FlakyHandler(AgentHandler):
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, ctx: AgentHandler.Context) -> str:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary")
        return "recovered"


class AlwaysFailHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        raise RuntimeError("always")


class FallbackHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        return "fallback"


async def test_retry_policy_recovers_successfully():
    handler = FlakyHandler()
    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="work", agent_role="flaky"))
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"flaky": handler},
        recovery_policy=RetryPolicy(max_retries=1),
    )

    results = await rt.run(dag)

    assert results["t1"] == "recovered"
    assert handler.calls == 2
    assert rt.agents["agent-t1"].state == AgentState.FINISHED
    recovery_events = [
        event for event in rt.events if event.event_type == "fault.recovery_completed"
    ]
    assert recovery_events
    assert recovery_events[0].data["recovery_time_secs"] >= 0.0


async def test_retry_policy_raises_after_retries_exhausted():
    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="work", agent_role="fail"))
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"fail": AlwaysFailHandler()},
        recovery_policy=RetryPolicy(max_retries=1),
    )

    with pytest.raises(RuntimeError, match="always"):
        await rt.run(dag)

    assert any(event.event_type == "fault.recovery_failed" for event in rt.events)


async def test_fallback_policy_runs_fallback_handler():
    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="work", agent_role="fail"))
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"fail": AlwaysFailHandler(), "fallback": FallbackHandler()},
        recovery_policy=FallbackPolicy(fallback_role="fallback"),
    )

    results = await rt.run(dag)

    assert results["t1"] == "fallback"
    assert any(
        event.event_type == "fault.recovery_completed"
        and event.data["action"] == "fallback"
        for event in rt.events
    )


async def test_skip_policy_completes_task_with_none():
    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="work", agent_role="fail"))
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"fail": AlwaysFailHandler()},
        recovery_policy=SkipPolicy(),
    )

    results = await rt.run(dag)

    assert results["t1"] is None
    assert dag.tasks["t1"].status.value == "completed"
    assert any(
        event.event_type == "fault.recovery_completed"
        and event.data["action"] == "skip"
        for event in rt.events
    )


class PartialProgressHandler(AgentHandler):
    """Simulates an agent that writes partial results then fails on first call."""

    def __init__(self) -> None:
        self.calls = 0

    async def run(self, ctx: AgentHandler.Context) -> str:
        self.calls += 1
        ctx.context_manager.add_private(ctx.acb.agent_id, f"partial-result-call-{self.calls}")
        if self.calls == 1:
            raise RuntimeError("fail after partial progress")
        refs = ctx.context_manager.get_refs_for_agent(ctx.acb.agent_id)
        private_count = sum(1 for r in refs if r.scope.value == "private")
        return f"recovered with {private_count} private refs"


def test_context_retry_policy_returns_context_retry_on_first_failure():
    policy = ContextRetryPolicy(max_retries=2)
    decision = policy.decide(RuntimeError("fail"), attempt=1)
    assert decision.action == RecoveryAction.CONTEXT_RETRY
    assert "context_retry" in decision.reason


def test_context_retry_policy_returns_fail_after_max_retries():
    policy = ContextRetryPolicy(max_retries=1)
    decision = policy.decide(RuntimeError("fail"), attempt=2)
    assert decision.action == RecoveryAction.FAIL
    assert "exhausted" in decision.reason


async def test_context_retry_removes_private_results_from_failed_attempt():
    ctx_mgr = ContextManager()
    ctx_mgr.add_shared("task-bg", "project background info")
    handler = PartialProgressHandler()
    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="work", agent_role="partial"))
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"partial": handler},
        context_manager=ctx_mgr,
        recovery_policy=ContextRetryPolicy(max_retries=1),
    )

    results = await rt.run(dag)

    assert results["t1"] == "recovered with 1 private refs"
    assert handler.calls == 2
    snapshot_events = [e for e in rt.events if e.event_type == "fault.context_snapshot_saved"]
    assert len(snapshot_events) == 1
    restore_events = [e for e in rt.events if e.event_type == "fault.context_restored"]
    assert len(restore_events) == 1
    assert restore_events[0].data["context_refs_restored"] == 0
    assert restore_events[0].data["shared_refs_available"] == 1


async def test_context_retry_reports_restored_state_without_claiming_token_savings():
    ctx_mgr = ContextManager()
    ctx_mgr.add_shared("bg", "A" * 400)  # ~100 tokens
    handler = PartialProgressHandler()
    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="work", agent_role="partial"))
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"partial": handler},
        context_manager=ctx_mgr,
        recovery_policy=ContextRetryPolicy(max_retries=1),
    )

    await rt.run(dag)

    restore_events = [e for e in rt.events if e.event_type == "fault.context_restored"]
    assert len(restore_events) == 1
    assert restore_events[0].data["estimated_private_tokens_restored"] == 0
    assert restore_events[0].data["shared_context_rewound"] is False
    assert "estimated_tokens_saved" not in restore_events[0].data

    completed_events = [
        e for e in rt.events
        if e.event_type == "fault.recovery_completed" and e.data.get("action") == "context_retry"
    ]
    assert len(completed_events) == 1


async def test_context_retry_exhausted_triggers_propagation():
    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="code", agent_role="fail"))
    dag.add_task(Task(task_id="t2", name="test", agent_role="ok", dependencies=["t1"]))

    class OkHandler(AgentHandler):
        async def run(self, ctx: AgentHandler.Context) -> str:
            return "ok"

    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"fail": AlwaysFailHandler(), "ok": OkHandler()},
        recovery_policy=ContextRetryPolicy(max_retries=1),
    )

    with pytest.raises(RuntimeError, match="always"):
        await rt.run(dag)

    assert dag.tasks["t2"].status == TaskStatus.SKIPPED
    assert any(e.event_type == "fault.context_restored" for e in rt.events)
    assert any(e.event_type == "fault.propagated" for e in rt.events)


async def test_recovery_demo_scenario_produces_trace(tmp_path):
    from agentos.scenarios.coding.recovery_demo import run_recovery_demo

    results = await run_recovery_demo(log_path=tmp_path / "events.jsonl")

    assert results["recover"] == "recovered"
    assert (tmp_path / "events.jsonl").exists()
