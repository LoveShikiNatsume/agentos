import asyncio

import pytest

from agentos.core.resources import ResourceQuota
from agentos.core.task import Task, TaskDAG
from agentos.fault.recovery import SkipPolicy
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import AgentHandler, Runtime
from agentos.tools.manager import (
    ToolConflictError,
    ToolManager,
    ToolPermissionError,
    ToolSpec,
    ToolTimeoutError,
)


async def test_tool_manager_calls_sync_and_async_tools():
    manager = ToolManager()
    manager.register(ToolSpec("echo", lambda text: f"sync:{text}"))

    async def upper(text: str) -> str:
        return text.upper()

    manager.register(ToolSpec("upper", upper))

    sync_result = await manager.call(
        "echo",
        kwargs={"text": "hi"},
        allowed_tools=["echo"],
    )
    async_result = await manager.call(
        "upper",
        kwargs={"text": "hi"},
        allowed_tools=["upper"],
    )

    assert sync_result.ok is True
    assert sync_result.output == "sync:hi"
    assert async_result.output == "HI"


async def test_tool_manager_rejects_unregistered_tool():
    manager = ToolManager()

    with pytest.raises(ToolPermissionError, match="not registered"):
        await manager.call("missing", allowed_tools=["missing"])


async def test_tool_manager_denies_empty_allowed_tools():
    manager = ToolManager()
    manager.register(ToolSpec("echo", lambda: "ok"))

    with pytest.raises(ToolPermissionError, match="not allowed"):
        await manager.call("echo", allowed_tools=[])


async def test_tool_manager_denies_tool_outside_whitelist():
    manager = ToolManager()
    manager.register(ToolSpec("echo", lambda: "ok"))

    with pytest.raises(ToolPermissionError, match="not allowed"):
        await manager.call("echo", allowed_tools=["other"])


async def test_tool_manager_timeout_raises_structured_error():
    manager = ToolManager()

    async def slow() -> str:
        await asyncio.sleep(0.05)
        return "late"

    manager.register(ToolSpec("slow", slow, timeout_seconds=0.001))

    with pytest.raises(ToolTimeoutError) as exc_info:
        await manager.call("slow", allowed_tools=["slow"])

    assert exc_info.value.result is not None
    assert exc_info.value.result.timed_out is True


async def test_tool_manager_exclusive_tool_conflict():
    manager = ToolManager()
    started = asyncio.Event()
    release = asyncio.Event()

    async def hold() -> str:
        started.set()
        await release.wait()
        return "done"

    manager.register(ToolSpec("hold", hold, exclusive=True))
    first = asyncio.create_task(manager.call("hold", allowed_tools=["hold"]))
    await started.wait()

    with pytest.raises(ToolConflictError):
        await manager.call("hold", allowed_tools=["hold"])

    release.set()
    result = await first
    assert result.output == "done"


class _ToolEchoHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context):
        result = await ctx.tool_call("echo", text="hi")
        return result.output


class _ToolSlowHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context):
        await ctx.tool_call("slow")
        return "never"


async def test_runtime_tool_call_emits_events_and_updates_stats():
    manager = ToolManager()
    manager.register(ToolSpec("echo", lambda text: f"tool:{text}"))
    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="t1",
            name="tool task",
            agent_role="tool",
            resource_requirements=ResourceQuota(allowed_tools=["echo"]),
        )
    )
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"tool": _ToolEchoHandler()},
        tool_manager=manager,
    )

    results = await rt.run(dag)

    acb = rt.agents["agent-t1"]
    event_types = [event.event_type for event in rt.events]
    assert results["t1"] == "tool:hi"
    assert "tool.call_started" in event_types
    assert "tool.call_finished" in event_types
    assert acb.stats.tool_calls == 1
    assert acb.stats.total_tool_wait_secs >= 0


async def test_runtime_denies_unallowed_tool_and_fails_task():
    manager = ToolManager()
    manager.register(ToolSpec("echo", lambda text: text))
    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="tool task", agent_role="tool"))
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"tool": _ToolEchoHandler()},
        tool_manager=manager,
    )

    with pytest.raises(ToolPermissionError):
        await rt.run(dag)

    assert any(event.event_type == "tool.permission_denied" for event in rt.events)
    assert dag.tasks["t1"].status.value == "failed"


async def test_runtime_tool_timeout_can_enter_recovery_path():
    manager = ToolManager()

    async def slow() -> str:
        await asyncio.sleep(0.05)
        return "late"

    manager.register(ToolSpec("slow", slow, timeout_seconds=0.001))
    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="t1",
            name="slow tool task",
            agent_role="slow",
            resource_requirements=ResourceQuota(allowed_tools=["slow"]),
        )
    )
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"slow": _ToolSlowHandler()},
        recovery_policy=SkipPolicy(),
        tool_manager=manager,
    )

    results = await rt.run(dag)

    assert results["t1"] is None
    assert any(event.event_type == "tool.timeout" for event in rt.events)
    assert any(event.event_type == "fault.recovery_completed" for event in rt.events)
