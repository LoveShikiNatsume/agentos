"""Verify ContextRef-based partial recovery.

Demonstrates:
1. Agent fails after writing partial results to context
2. Runtime snapshots and restores context
3. Retry agent sees prior partial results
4. Token savings are recorded in events

Usage:
    python3 scripts/verify_partial_recovery.py
"""
from __future__ import annotations

import asyncio
import sys

from agentos.context.manager import ContextManager
from agentos.core.task import Task, TaskDAG, TaskStatus
from agentos.fault.recovery import ContextRetryPolicy
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import AgentHandler, Runtime


class SimulatedCoderHandler(AgentHandler):
    """Simulates a coder that writes partial code, then fails, then succeeds on retry."""

    def __init__(self) -> None:
        self.calls = 0

    async def run(self, ctx: AgentHandler.Context) -> str:
        self.calls += 1
        ctx.context_manager.add_private(
            ctx.acb.agent_id,
            f"[Call {self.calls}] def function_{self.calls}(): pass  # generated code",
        )
        if self.calls == 1:
            raise RuntimeError("LLM timeout after writing partial code")
        return f"completed on attempt {self.calls}"


class TesterHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        return "tests passed"


async def main() -> int:
    print("=" * 60)
    print("  ContextRef-Based Partial Recovery Verification")
    print("=" * 60)
    print()

    ctx_mgr = ContextManager()
    ctx_mgr.add_shared("task-bg", "Project: AgentOS-Runtime. Goal: implement auth module.")
    ctx_mgr.add_shared("repo-summary", "Python project with FastAPI backend, 50 files.")

    dag = TaskDAG()
    dag.add_task(Task(task_id="plan", name="plan", agent_role="planner"))
    dag.add_task(Task(task_id="code", name="code", agent_role="coder", dependencies=["plan"]))
    dag.add_task(Task(task_id="test", name="test", agent_role="tester", dependencies=["code"]))

    class PlannerHandler(AgentHandler):
        async def run(self, ctx: AgentHandler.Context) -> str:
            ctx.context_manager.add_shared("plan", "Step 1: auth module. Step 2: tests.")
            return "plan complete"

    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={
            "planner": PlannerHandler(),
            "coder": SimulatedCoderHandler(),
            "tester": TesterHandler(),
        },
        context_manager=ctx_mgr,
        recovery_policy=ContextRetryPolicy(max_retries=1),
    )

    results = await rt.run(dag)

    print("[Task Results]")
    for task_id, result in results.items():
        status = dag.tasks[task_id].status.value
        print(f"  {task_id}: status={status}, result={result}")
    print()

    print("[Recovery Events]")
    for event in rt.events:
        if event.event_type.startswith("fault."):
            agent = event.agent_id or "-"
            print(f"  {event.event_type:<32} agent={agent:<16} data={event.data}")
    print()

    ok = True
    checks = [
        ("plan completed", dag.tasks["plan"].status == TaskStatus.COMPLETED),
        ("code completed (after retry)", dag.tasks["code"].status == TaskStatus.COMPLETED),
        ("test completed", dag.tasks["test"].status == TaskStatus.COMPLETED),
        ("context snapshot saved", any(e.event_type == "fault.context_snapshot_saved" for e in rt.events)),
        ("context restored", any(e.event_type == "fault.context_restored" for e in rt.events)),
        ("recovery completed with context_retry", any(
            e.event_type == "fault.recovery_completed" and e.data.get("action") == "context_retry"
            for e in rt.events
        )),
    ]

    print("[Checks]")
    for label, passed in checks:
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {label}")
        if not passed:
            ok = False
    print()

    restore_events = [e for e in rt.events if e.event_type == "fault.context_restored"]
    if restore_events:
        tokens_saved = restore_events[0].data.get("estimated_tokens_saved", 0)
        refs_restored = restore_events[0].data.get("context_refs_restored", 0)
        print("[Token Savings]")
        print(f"  Context refs restored:    {refs_restored}")
        print(f"  Estimated tokens saved:   {tokens_saved}")
        print(f"  (vs full retry which discards all partial context)")
    print()

    status = "ALL PASSED" if ok else "SOME FAILED"
    print(f"Result: {status}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
