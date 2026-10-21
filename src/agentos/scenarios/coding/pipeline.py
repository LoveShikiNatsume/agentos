from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any, Sequence

from agentos.core.task import Task, TaskDAG
from agentos.core.types import Message
from agentos.events.log import EventLog
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import AgentHandler, Runtime
from agentos.scenarios.coding.prompts import (
    CODER_PROMPT,
    FIXER_PROMPT,
    PLANNER_PROMPT,
    PROJECT_CONTEXT,
    TESTER_PROMPT,
)


class PlannerHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        resp = await ctx.llm_call([{"role": "user", "content": PLANNER_PROMPT}])
        await ctx.bus.send(
            Message(
                from_agent=ctx.acb.agent_id,
                to_agent="agent-code",
                content={"plan": resp.content},
                message_type="plan",
                task_id=ctx.task.task_id,
                trace_id=ctx.acb.trace_id,
            )
        )
        return resp.content


class CoderHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        msg = await ctx.bus.receive(ctx.acb.agent_id)
        plan = msg.content["plan"] if msg else "implement fibonacci"
        resp = await ctx.llm_call(
            [{"role": "user", "content": f"{CODER_PROMPT}\nPlan: {plan}"}]
        )
        await ctx.bus.send(
            Message(
                from_agent=ctx.acb.agent_id,
                to_agent="agent-test",
                content={"code": resp.content},
                message_type="code",
                task_id=ctx.task.task_id,
                trace_id=ctx.acb.trace_id,
            )
        )
        return resp.content


class TesterHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        msg = await ctx.bus.receive(ctx.acb.agent_id)
        code = msg.content["code"] if msg else "fibonacci code"
        resp = await ctx.llm_call(
            [{"role": "user", "content": f"{TESTER_PROMPT}\nCode: {code}"}]
        )
        await ctx.bus.send(
            Message(
                from_agent=ctx.acb.agent_id,
                to_agent="agent-fix",
                content={"tests": resp.content, "code": code},
                message_type="tests",
                task_id=ctx.task.task_id,
                trace_id=ctx.acb.trace_id,
            )
        )
        return resp.content


class FixerHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        msg = await ctx.bus.receive(ctx.acb.agent_id)
        payload = msg.content if msg else {"code": "", "tests": ""}
        resp = await ctx.llm_call(
            [
                {
                    "role": "user",
                    "content": (
                        f"{FIXER_PROMPT}\nCode: {payload.get('code', '')}\n"
                        f"Tests: {payload.get('tests', '')}"
                    ),
                }
            ]
        )
        return resp.content


def build_coding_dag() -> TaskDAG:
    dag = TaskDAG()
    dag.add_task(Task(task_id="plan", name="plan", agent_role="planner"))
    dag.add_task(
        Task(task_id="code", name="code", agent_role="coder", dependencies=["plan"])
    )
    dag.add_task(
        Task(task_id="test", name="test", agent_role="tester", dependencies=["code"])
    )
    dag.add_task(
        Task(task_id="fix", name="fix", agent_role="fixer", dependencies=["test"])
    )
    return dag


def build_mock_backend() -> MockLLMBackend:
    return MockLLMBackend(
        responses={
            "agent-plan": "1. Define fib(n). 2. Add tests.",
            "agent-code": "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
            "agent-test": "assert fib(0) == 0\nassert fib(5) == 5",
            "agent-fix": "Final fix: implementation and tests are consistent.",
        }
    )


async def run_coding_pipeline(
    *,
    backend: str = "mock",
    log_path: str | Path = "runs/latest/events.jsonl",
) -> dict[str, Any]:
    if backend != "mock":
        raise ValueError("Only backend='mock' is supported by the built-in scenario")

    runtime = Runtime(
        llm_backend=build_mock_backend(),
        handlers={
            "planner": PlannerHandler(),
            "coder": CoderHandler(),
            "tester": TesterHandler(),
            "fixer": FixerHandler(),
        },
        event_log=EventLog(log_path),
    )
    runtime.context.add_shared_ref("project", PROJECT_CONTEXT)
    return await runtime.run(build_coding_dag())


async def _async_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coding-pipeline")
    parser.add_argument("--backend", default="mock", choices=["mock"])
    parser.add_argument("--log", default="runs/latest/events.jsonl")
    args = parser.parse_args(argv)

    results = await run_coding_pipeline(backend=args.backend, log_path=args.log)
    for task_id, result in results.items():
        print(f"--- {task_id} ---")
        print(result)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(_async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
