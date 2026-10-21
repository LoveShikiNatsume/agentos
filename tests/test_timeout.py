import asyncio

import pytest

from agentos.core.resources import ResourceQuota
from agentos.core.task import Task, TaskDAG
from agentos.core.types import AgentState
from agentos.fault.timeout import AgentTimeoutError, run_with_timeout
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import AgentHandler, Runtime


class SlowHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        await asyncio.sleep(0.05)
        return "slow done"


async def test_run_with_timeout_returns_result_before_deadline():
    async def work() -> str:
        await asyncio.sleep(0)
        return "ok"

    assert await run_with_timeout(work(), timeout_seconds=1.0) == "ok"


async def test_run_with_timeout_raises_agent_timeout_error():
    async def work() -> str:
        await asyncio.sleep(0.05)
        return "late"

    with pytest.raises(AgentTimeoutError, match="timed out"):
        await run_with_timeout(
            work(),
            timeout_seconds=0.001,
            agent_id="agent-1",
            task_id="task-1",
        )


async def test_runtime_timeout_marks_task_failed_and_emits_fault_event():
    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="slow",
            name="slow",
            agent_role="slow",
            resource_requirements=ResourceQuota(timeout_seconds=0.001),
        )
    )
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"slow": SlowHandler()},
    )

    with pytest.raises(AgentTimeoutError):
        await rt.run(dag)

    assert dag.tasks["slow"].status.value == "failed"
    assert rt.agents["agent-slow"].state == AgentState.FAILED

    fault_events = [event for event in rt.events if event.event_type == "fault.timeout"]
    assert len(fault_events) == 1
    assert fault_events[0].agent_id == "agent-slow"
    assert fault_events[0].task_id == "slow"
    assert fault_events[0].data["timeout_seconds"] == 0.001
