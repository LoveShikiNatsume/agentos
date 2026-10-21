"""End-to-end fault injection, recovery, and dashboard demo.

Exercises in one run:
- DAG scheduling (plan -> code -> test -> fix -> deploy + independent review)
- Fault injection (kill on code task)
- ContextRef partial recovery (context_retry)
- Fault propagation (would skip downstream if recovery fails)
- Independent branch continues unaffected
- EventLog JSONL output
- agentctl trace + dashboard generation

Usage:
    .venv/bin/python scripts/demo_e2e_fault.py
    .venv/bin/python scripts/demo_e2e_fault.py --log runs/demo/events.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from agentos.context.manager import ContextManager
from agentos.core.task import Task, TaskDAG, TaskStatus
from agentos.events.log import EventLog
from agentos.fault.injector import FaultInjectionPlan
from agentos.fault.recovery import ContextRetryPolicy
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import AgentHandler, Runtime


class PlannerHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        ctx.context_manager.add_shared(
            "plan", "1. Implement auth module\n2. Write unit tests\n3. Integration test"
        )
        resp = await ctx.llm_call([{"role": "user", "content": "Plan the auth module"}])
        return resp.content


class CoderHandler(AgentHandler):
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, ctx: AgentHandler.Context) -> str:
        self.calls += 1
        ctx.context_manager.add_private(
            ctx.acb.agent_id,
            f"[attempt {self.calls}] class AuthService: ... # partial code",
        )
        resp = await ctx.llm_call([{"role": "user", "content": "Write auth code"}])
        return resp.content


class TesterHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        resp = await ctx.llm_call([{"role": "user", "content": "Write auth tests"}])
        return resp.content


class FixerHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        resp = await ctx.llm_call([{"role": "user", "content": "Fix auth issues"}])
        return resp.content


class DeployHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        resp = await ctx.llm_call([{"role": "user", "content": "Deploy auth module"}])
        return resp.content


class ReviewHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        resp = await ctx.llm_call([{"role": "user", "content": "Review code quality"}])
        return resp.content


def build_demo_dag() -> TaskDAG:
    dag = TaskDAG()
    dag.add_task(Task(task_id="plan", name="plan", agent_role="planner"))
    dag.add_task(Task(task_id="code", name="code", agent_role="coder", dependencies=["plan"]))
    dag.add_task(Task(task_id="test", name="test", agent_role="tester", dependencies=["code"]))
    dag.add_task(Task(task_id="fix", name="fix", agent_role="fixer", dependencies=["test"]))
    dag.add_task(Task(task_id="deploy", name="deploy", agent_role="deployer", dependencies=["fix"]))
    dag.add_task(Task(task_id="review", name="review", agent_role="reviewer", dependencies=["plan"]))
    return dag


async def run_demo(
    log_path: Path, llm_backend: Any | None = None
) -> tuple[dict[str, Any], Runtime, TaskDAG]:
    ctx_mgr = ContextManager()
    ctx_mgr.add_shared("project-bg", "AgentOS-Runtime: multi-agent execution runtime for OS competition.")
    ctx_mgr.add_shared("repo-summary", "Python 3.11 project, 50+ modules, pytest test suite.")

    dag = build_demo_dag()

    rt = Runtime(
        llm_backend=llm_backend or MockLLMBackend(),
        handlers={
            "planner": PlannerHandler(),
            "coder": CoderHandler(),
            "tester": TesterHandler(),
            "fixer": FixerHandler(),
            "deployer": DeployHandler(),
            "reviewer": ReviewHandler(),
        },
        context_manager=ctx_mgr,
        event_log=EventLog(log_path),
        fault_injection=FaultInjectionPlan({"code": "kill"}),
        recovery_policy=ContextRetryPolicy(max_retries=1),
    )

    results = await rt.run(dag)
    return results, rt, dag


def print_report(results: dict[str, Any], rt: Runtime, dag: TaskDAG) -> bool:
    print("=" * 64)
    print("  AgentOS-Runtime End-to-End Fault Demo")
    print("=" * 64)
    print()

    print("[DAG Task Results]")
    for task_id in ("plan", "code", "test", "fix", "deploy", "review"):
        task = dag.tasks[task_id]
        result = results.get(task_id, "-")
        result_str = str(result)[:60] if result else "-"
        print(f"  {task_id:<8} status={task.status.value:<10} result={result_str}")
    print()

    print("[Event Summary]")
    event_types: dict[str, int] = {}
    for e in rt.events:
        event_types[e.event_type] = event_types.get(e.event_type, 0) + 1
    for etype, count in sorted(event_types.items()):
        print(f"  {etype:<36} x{count}")
    print()

    print("[Fault & Recovery Chain]")
    for e in rt.events:
        if e.event_type.startswith("fault."):
            agent = e.agent_id or "-"
            print(f"  {e.event_type:<36} agent={agent:<16} {e.data}")
    print()

    checks = [
        ("all tasks completed", all(
            dag.tasks[t].status == TaskStatus.COMPLETED
            for t in ("plan", "code", "test", "fix", "deploy", "review")
        )),
        ("fault injected on code", any(
            e.event_type == "fault.injected" and e.task_id == "code" for e in rt.events
        )),
        ("context snapshot saved", any(
            e.event_type == "fault.context_snapshot_saved" for e in rt.events
        )),
        ("context restored after failure", any(
            e.event_type == "fault.context_restored" for e in rt.events
        )),
        ("recovery completed (context_retry)", any(
            e.event_type == "fault.recovery_completed"
            and e.data.get("action") == "context_retry"
            for e in rt.events
        )),
        ("review branch unaffected", dag.tasks["review"].status == TaskStatus.COMPLETED),
        ("DAG fully complete", dag.is_complete()),
    ]

    print("[Checks]")
    ok = True
    for label, passed in checks:
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {label}")
        if not passed:
            ok = False
    print()

    restore = [e for e in rt.events if e.event_type == "fault.context_restored"]
    if restore:
        print("[Partial Recovery Metrics]")
        print(f"  Context refs restored:  {restore[0].data.get('context_refs_restored', 0)}")
        print(f"  Tokens saved estimate:  {restore[0].data.get('estimated_tokens_saved', 0)}")
        print()

    return ok


def generate_dashboard(log_path: Path, output_dir: Path) -> Path | None:
    dashboard_path = output_dir / "dashboard.html"
    try:
        from agentos.dashboard.builder import build_dashboard
        build_dashboard(
            log_paths=[log_path],
            output_path=dashboard_path,
        )
        return dashboard_path
    except Exception as exc:
        print(f"  Dashboard generation failed: {exc}")
        return None


def run_agentctl_trace(log_path: Path) -> None:
    print("[agentctl trace (fault events)]")
    try:
        from agentos.cli import main as cli_main
        cli_main(["trace", "--log", str(log_path), "--category", "fault"])
    except SystemExit:
        pass
    print()


async def async_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="demo-e2e-fault")
    parser.add_argument("--log", default="runs/demo/events.jsonl")
    parser.add_argument(
        "--backend",
        default="mock",
        help="LLM backend: mock (default) or deepseek (real DeepSeek official API).",
    )
    parser.add_argument(
        "--model",
        default="deepseek-chat",
        help="Model name when --backend deepseek (default: deepseek-chat).",
    )
    args = parser.parse_args(argv)

    log_path = Path(args.log)
    if log_path.exists():
        log_path.unlink()

    llm_backend = None
    if args.backend != "mock":
        from agentos.llm.factory import create_backend

        kwargs: dict[str, Any] = {}
        if args.backend == "deepseek":
            kwargs["model"] = args.model
        llm_backend = create_backend(args.backend, **kwargs)
        print(f"[Backend] Using real LLM backend: {args.backend} ({args.model})\n")

    results, rt, dag = await run_demo(log_path, llm_backend=llm_backend)
    ok = print_report(results, rt, dag)

    run_agentctl_trace(log_path)

    output_dir = log_path.parent
    dashboard_path = generate_dashboard(log_path, output_dir)
    if dashboard_path:
        print(f"[Dashboard] Written to {dashboard_path}")
        html = dashboard_path.read_text(encoding="utf-8")
        has_agents = "agent-code" in html
        has_faults = "fault" in html.lower()
        print(f"  Contains agent data: {has_agents}")
        print(f"  Contains fault data: {has_faults}")
    print()

    status = "ALL PASSED" if ok else "SOME FAILED"
    print(f"Result: {status}")
    return 0 if ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
