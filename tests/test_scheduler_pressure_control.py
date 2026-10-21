from __future__ import annotations

from agentos.core.resources import ResourceQuota
from agentos.core.task import Task, TaskDAG
from agentos.scheduler.dag import DAGScheduler


async def test_memory_pressure_defers_high_memory_task_when_low_memory_task_is_ready():
    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="high-memory",
            name="high-memory",
            agent_role="worker",
            resource_requirements=ResourceQuota(memory_max_mb=512),
        )
    )
    dag.add_task(
        Task(
            task_id="low-memory",
            name="low-memory",
            agent_role="worker",
            resource_requirements=ResourceQuota(memory_max_mb=64),
        )
    )
    scheduler = DAGScheduler(dag, max_batch_size=2)
    scheduler.set_pressure(
        pressure_level="critical",
        pressure_sample={
            "pressure_source": "system+cgroup",
            "memory_some_avg10": 35.0,
            "cpu_some_avg10": 2.0,
            "io_some_avg10": 1.0,
        },
    )

    batch = await scheduler.get_next_batch()

    assert [task.task_id for task in batch] == ["low-memory"]
    decision = scheduler.last_decision or {}
    assert decision["pressure_control"]["memory_pressure_deferred_task_ids"] == [
        "high-memory"
    ]
    assert "high-memory" in decision["deferred_task_ids"]


async def test_cpu_pressure_reduces_batch_width_for_ready_tasks():
    dag = TaskDAG()
    for task_id in ("a", "b", "c"):
        dag.add_task(Task(task_id=task_id, name=task_id, agent_role="worker"))

    scheduler = DAGScheduler(dag, max_batch_size=3)
    scheduler.set_pressure(
        pressure_level="critical",
        pressure_sample={
            "pressure_source": "system+cgroup",
            "cpu_some_avg10": 32.0,
            "memory_some_avg10": 3.0,
            "io_some_avg10": 1.0,
        },
    )

    batch = await scheduler.get_next_batch()

    assert len(batch) == 1
    decision = scheduler.last_decision or {}
    assert decision["pressure_control"]["cpu_pressure_batch_limit"] == 1
    assert len(decision["selected_task_ids"]) == 1
    assert len(decision["deferred_task_ids"]) == 2
