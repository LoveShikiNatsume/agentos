import pytest

from agentos.events.types import EventCategory
from agentos.core.task import Task, TaskDAG
from agentos.fault.injector import FaultInjectionPlan, FaultType, inject_fault
from agentos.fault.recovery import RetryPolicy
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import AgentHandler, Runtime


def test_inject_timeout_fault_event():
    event = inject_fault(FaultType.TIMEOUT, task_id="task-1", agent_id="agent-1")

    assert event.event_type == "fault.injected"
    assert event.category == EventCategory.FAULT
    assert event.agent_id == "agent-1"
    assert event.task_id == "task-1"
    assert event.data["fault_type"] == "timeout"


def test_inject_kill_fault_event():
    event = inject_fault("kill", task_id="task-1")

    assert event.event_type == "fault.injected"
    assert event.data["fault_type"] == "kill"


def test_inject_memory_pressure_fault_event():
    event = inject_fault(
        FaultType.MEMORY_PRESSURE,
        task_id="task-1",
        agent_id="agent-1",
        metadata={"memory_mb": 4},
    )

    assert event.event_type == "fault.injected"
    assert event.data["fault_type"] == "memory_pressure"
    assert event.data["memory_mb"] == 4


def test_fault_injection_plan_matches_task_once():
    plan = FaultInjectionPlan({"task-1": "kill"})

    first = plan.pop_for_task("task-1")
    second = plan.pop_for_task("task-1")

    assert first is not None
    assert first.data["fault_type"] == "kill"
    assert second is None


def test_fault_injection_plan_keeps_memory_pressure_metadata():
    plan = FaultInjectionPlan(
        {"task-1": {"fault_type": "memory_pressure", "memory_mb": 4}}
    )

    event = plan.pop_for_task("task-1")

    assert event is not None
    assert event.data["fault_type"] == "memory_pressure"
    assert event.data["memory_mb"] == 4
    assert plan.pop_for_task("task-1") is None


class _OkHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        return "ok"


async def test_runtime_triggers_injected_kill_fault_and_recovers():
    dag = TaskDAG()
    dag.add_task(Task(task_id="task-1", name="work", agent_role="ok"))
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"ok": _OkHandler()},
        recovery_policy=RetryPolicy(max_retries=1),
        fault_injection=FaultInjectionPlan({"task-1": "kill"}),
    )

    results = await rt.run(dag)

    assert results["task-1"] == "ok"
    event_types = [event.event_type for event in rt.events]
    assert "fault.injected" in event_types
    assert "fault.triggered" in event_types
    assert "fault.recovery_completed" in event_types


async def test_runtime_triggers_injected_timeout_fault_and_recovers():
    dag = TaskDAG()
    dag.add_task(Task(task_id="task-1", name="work", agent_role="ok"))
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"ok": _OkHandler()},
        recovery_policy=RetryPolicy(max_retries=1),
        fault_injection=FaultInjectionPlan({"task-1": "timeout"}),
    )

    results = await rt.run(dag)

    assert results["task-1"] == "ok"
    assert any(event.event_type == "fault.timeout" for event in rt.events)
    assert any(event.event_type == "fault.recovery_completed" for event in rt.events)


async def test_runtime_triggers_injected_memory_pressure_fault_and_recovers():
    dag = TaskDAG()
    dag.add_task(Task(task_id="task-1", name="work", agent_role="ok"))
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"ok": _OkHandler()},
        recovery_policy=RetryPolicy(max_retries=1),
        fault_injection=FaultInjectionPlan(
            {"task-1": {"fault_type": "memory_pressure", "memory_mb": 4}}
        ),
    )

    results = await rt.run(dag)

    assert results["task-1"] == "ok"
    assert any(event.event_type == "fault.memory_pressure" for event in rt.events)
    assert any(event.event_type == "fault.recovery_completed" for event in rt.events)


async def test_runtime_memory_pressure_without_recovery_fails():
    dag = TaskDAG()
    dag.add_task(Task(task_id="task-1", name="work", agent_role="ok"))
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"ok": _OkHandler()},
        fault_injection=FaultInjectionPlan({"task-1": "memory_pressure"}),
    )

    with pytest.raises(MemoryError):
        await rt.run(dag)

    assert any(event.event_type == "fault.memory_pressure" for event in rt.events)
    assert dag.tasks["task-1"].status.value == "failed"
