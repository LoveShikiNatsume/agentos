"""CLI verification for fault propagation control (Task 8)."""

import asyncio

from agentos.core.task import Task, TaskDAG, TaskStatus
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import AgentHandler, Runtime


class EchoHandler(AgentHandler):
    async def run(self, ctx):
        response = await ctx.llm_call(
            messages=[{"role": "user", "content": f"Task: {ctx.task.name}"}]
        )
        return response.content


class FailHandler(AgentHandler):
    async def run(self, ctx):
        raise RuntimeError("injected failure")


async def main():
    dag = TaskDAG()
    dag.add_task(Task(task_id="plan", name="plan", agent_role="echo"))
    dag.add_task(
        Task(task_id="code", name="code", agent_role="fail", dependencies=["plan"])
    )
    dag.add_task(
        Task(task_id="test", name="test", agent_role="echo", dependencies=["code"])
    )
    dag.add_task(
        Task(task_id="fix", name="fix", agent_role="echo", dependencies=["test"])
    )
    dag.add_task(Task(task_id="deploy", name="deploy", agent_role="echo"))

    rt = Runtime(
        llm_backend=MockLLMBackend(responses={"default": "ok"}),
        handlers={"echo": EchoHandler(), "fail": FailHandler()},
        recovery_policy=None,
        event_log="runs/propagation_demo/events.jsonl",
    )

    try:
        await rt.run(dag)
    except RuntimeError:
        pass

    print("=== Final Task Statuses ===")
    for tid, task in dag.tasks.items():
        print(f"  {tid}: {task.status.value}")

    print()
    print("=== DAG Complete? ===")
    print(f"  {dag.is_complete()}")

    print()
    print("=== Event Chain ===")
    for e in rt.events:
        line = f"  [{e.event_type}]"
        if e.task_id:
            line += f" task={e.task_id}"
        if e.agent_id:
            line += f" agent={e.agent_id}"
        if e.event_type == "fault.propagated":
            line += f" source={e.data['source_task_id']} depth={e.data['depth']}"
        print(line)


asyncio.run(main())
