"""Service-time feedback from Runtime, using concurrent task stubs only."""

import asyncio
from types import SimpleNamespace

import pytest

from agentos.core.task import Task, TaskDAG
from agentos.isolation.kernel_adapter import PressureSample
from agentos.llm.mock import MockLLMBackend
from agentos.runtime import engine
from agentos.runtime.engine import Runtime
from agentos.scheduler.fairness import FairnessTracker


@pytest.mark.parametrize("scale", ["absolute", "service_time"])
@pytest.mark.parametrize("duration", [0.0, 2.0])
async def test_runtime_observes_batch_wall_time_and_emits_selection_inputs(monkeypatch, scale, duration):
    clock = [10.0]
    tracker = FairnessTracker(fairness_scale=scale, now=lambda: clock[0])
    runtime = Runtime(
        llm_backend=MockLLMBackend(), handlers={}, max_batch_size=4,
        fairness_scale=scale, scheduler_fairness_tracker=tracker,
        enable_kernel_probe=False,
    )
    dag = TaskDAG()
    for i in range(8):
        dag.add_task(Task(task_id=f"t{i}", name="stub", agent_role="stub"))

    async def run_task(task):
        start = clock[0]
        await asyncio.sleep(0)
        # Four concurrent tasks take two seconds together, not eight seconds.
        clock[0] = max(clock[0], start + duration)
        return task.task_id

    monkeypatch.setattr(runtime, "_run_task", run_task)
    monkeypatch.setattr(runtime, "_sample_system_pressure", lambda: PressureSample())
    monkeypatch.setattr(engine, "time", SimpleNamespace(
        monotonic=lambda: clock[0], time=engine.time.time,
    ))
    results = await runtime.run(dag)
    assert len(results) == 8
    assert clock[0] == 10.0 + 2 * duration
    decisions = [e.data for e in runtime.events if e.event_type == "scheduler.decision"]
    assert len(decisions) == 2
    first_inputs = decisions[0]["policy_score"]["fairness"]
    second_inputs = decisions[1]["policy_score"]["fairness"]
    assert first_inputs["avg_service_time_ms"] is None
    if duration:
        assert second_inputs["avg_service_time_ms"] == pytest.approx(2000.0)
    else:
        assert second_inputs["avg_service_time_ms"] is None
    assert second_inputs["fairness_scale"] == scale
    expected_scale = scale if duration else "absolute"
    assert second_inputs["effective_scale"] == expected_scale
    selected = next(e for e in runtime.events if e.event_type == "scheduler.policy_selected")
    assert selected.data["fairness"] == first_inputs


def test_runtime_rejects_invalid_or_mismatched_fairness_scale():
    with pytest.raises(ValueError, match="fairness_scale"):
        Runtime(llm_backend=MockLLMBackend(), handlers={}, fairness_scale="unknown")
    with pytest.raises(ValueError, match="match"):
        Runtime(
            llm_backend=MockLLMBackend(), handlers={}, fairness_scale="service_time",
            scheduler_fairness_tracker=FairnessTracker(),
        )
