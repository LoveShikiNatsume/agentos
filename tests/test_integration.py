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
                    "content": "Plan the implementation of a fibonacci function.",
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
        plan = msg.content["plan"] if msg else "no plan"
        resp = await ctx.llm_call(
            messages=[{"role": "user", "content": f"Implement based on plan: {plan}"}]
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
        code = msg.content["code"] if msg else "no code"
        resp = await ctx.llm_call(
            messages=[{"role": "user", "content": f"Write tests for: {code}"}]
        )
        return resp.content


async def test_coding_pipeline_end_to_end():
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
            "agent-plan": "Step 1: Define fib(n). Step 2: Use recursion.",
            "agent-code": "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
            "agent-test": "assert fib(5) == 5; assert fib(0) == 0",
        }
    )

    rt = Runtime(
        llm_backend=llm,
        handlers={
            "planner": PlannerHandler(),
            "coder": CoderHandler(),
            "tester": TesterHandler(),
        },
    )

    rt.context.add_shared(
        "project", "Project: Fibonacci implementation for AgentOS demo"
    )

    results = await rt.run(dag)

    assert len(results) == 3
    assert "fib" in results["code"]
    assert llm.call_count == 3

    state_events = [
        event for event in rt.events if event.event_type == "agent.state_changed"
    ]
    assert len(state_events) > 0

    completed_events = [
        event for event in rt.events if event.event_type == "task.completed"
    ]
    assert len(completed_events) == 3

    for req in llm.requests:
        system_msgs = [message for message in req.messages if message["role"] == "system"]
        assert len(system_msgs) == 1
        assert "Fibonacci" in system_msgs[0]["content"]


async def test_pipeline_with_parallel_coders():
    dag = TaskDAG()
    dag.add_task(Task(task_id="plan", name="plan", agent_role="planner"))
    dag.add_task(
        Task(
            task_id="code-a",
            name="code-module-a",
            agent_role="coder",
            dependencies=["plan"],
        )
    )
    dag.add_task(
        Task(
            task_id="code-b",
            name="code-module-b",
            agent_role="coder",
            dependencies=["plan"],
        )
    )
    dag.add_task(
        Task(
            task_id="test",
            name="test-all",
            agent_role="tester",
            dependencies=["code-a", "code-b"],
        )
    )

    llm = MockLLMBackend(responses={"default": "ok"})
    rt = Runtime(
        llm_backend=llm,
        handlers={
            "planner": PlannerHandler(),
            "coder": CoderHandler(),
            "tester": TesterHandler(),
        },
    )

    results = await rt.run(dag)
    assert len(results) == 4
    assert llm.call_count == 4

