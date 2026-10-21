from __future__ import annotations

import asyncio

from agentos.core.task import Task, TaskDAG
from agentos.core.types import Message
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import AgentHandler, Runtime


class PlannerHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        resp = await ctx.llm_call(
            messages=[
                {
                    "role": "user",
                    "content": "Plan a fibonacci function implementation.",
                }
            ]
        )
        await ctx.bus.send(
            Message(
                from_agent=ctx.acb.agent_id,
                to_agent="agent-code",
                content={"plan": resp.content},
            )
        )
        return resp.content


class CoderHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        msg = await ctx.bus.receive(ctx.acb.agent_id)
        plan = msg.content["plan"] if msg else "implement fibonacci"
        resp = await ctx.llm_call(
            messages=[{"role": "user", "content": f"Write code based on: {plan}"}]
        )
        await ctx.bus.send(
            Message(
                from_agent=ctx.acb.agent_id,
                to_agent="agent-test",
                content={"code": resp.content},
            )
        )
        return resp.content


class TesterHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        msg = await ctx.bus.receive(ctx.acb.agent_id)
        code = msg.content["code"] if msg else "the fibonacci function"
        resp = await ctx.llm_call(
            messages=[{"role": "user", "content": f"Write tests for: {code}"}]
        )
        return resp.content


async def main() -> None:
    dag = TaskDAG()
    dag.add_task(Task(task_id="plan", name="plan", agent_role="planner"))
    dag.add_task(
        Task(task_id="code", name="code", agent_role="coder", dependencies=["plan"])
    )
    dag.add_task(
        Task(task_id="test", name="test", agent_role="tester", dependencies=["code"])
    )

    llm = MockLLMBackend(
        responses={
            "agent-plan": (
                "1. Define fib(n)\n"
                "2. Base cases: fib(0)=0, fib(1)=1\n"
                "3. Recursive: fib(n)=fib(n-1)+fib(n-2)"
            ),
            "agent-code": (
                "def fib(n):\n"
                "    if n <= 1:\n"
                "        return n\n"
                "    return fib(n - 1) + fib(n - 2)"
            ),
            "agent-test": "assert fib(0) == 0\nassert fib(5) == 5\nassert fib(10) == 55",
        }
    )

    runtime = Runtime(
        llm_backend=llm,
        handlers={
            "planner": PlannerHandler(),
            "coder": CoderHandler(),
            "tester": TesterHandler(),
        },
    )
    runtime.context.add_shared(
        "project", "Implement a fibonacci function with full test coverage."
    )

    print("=== AgentOS Runtime: Coding Pipeline Demo ===\n")
    results = await runtime.run(dag)

    for task_id, result in results.items():
        print(f"--- {task_id} ---")
        print(result)
        print()

    print("=== Stats ===")
    print(f"Total LLM calls: {llm.call_count}")
    print(f"Total events: {len(runtime.events)}")
    print(f"Context stats: {runtime.context.stats()}")


if __name__ == "__main__":
    asyncio.run(main())

