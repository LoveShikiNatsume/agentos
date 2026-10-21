"""Deterministic fairness-scale and bounded cache-affinity bypass tests.

These tests use a manual clock and a cache-heat stub; no inference is run.
"""

from __future__ import annotations

import pytest

from agentos.core.task import Task, TaskDAG, TaskStatus
from agentos.scheduler.dag import DAGScheduler
from agentos.scheduler.fairness import FairnessTracker, StarvationRisk


class ManualClock:
    def __init__(self) -> None:
        self.seconds = 100.0

    def __call__(self) -> float:
        return self.seconds

    def advance(self, seconds: float) -> None:
        self.seconds += seconds


class PrefixHeatStub:
    """Expose only the affinity advisor interface consumed by DAGScheduler."""

    def __init__(self, heats: dict[str, float] | None = None) -> None:
        self.heats = heats or {}

    def get_task_prefix_heat(self, task: Task) -> float:
        return self.heats.get(task.context_refs[0], 0.9)

    def sort_by_affinity(self, tasks: list[Task]) -> list[Task]:
        return sorted(
            tasks,
            key=lambda task: (
                -self.get_task_prefix_heat(task),
                task.context_refs[0],
                task.created_at,
                task.task_id,
            ),
        )


def make_task(task_id: str, prefix: str, *, created_at: float = 0.0) -> Task:
    return Task(
        task_id=task_id,
        name=task_id,
        agent_role="worker",
        context_refs=[prefix],
        workload_hint="ai_inference",
        sla_class="batch",
        created_at=created_at,
    )


def make_dag(tasks: list[Task]) -> TaskDAG:
    dag = TaskDAG()
    for task in tasks:
        dag.add_task(task)
    return dag


@pytest.mark.asyncio
@pytest.mark.parametrize("scale", ["absolute", "service_time"])
async def test_six_llm_batches_do_not_treat_normal_queueing_as_starvation(scale):
    """24 interleaved tasks require six ordinary 2 s batches, not starvation."""
    clock = ManualClock()
    tracker = FairnessTracker(fairness_scale=scale, now=clock)
    tasks = [
        make_task(f"doc-{group}-task-{index}", f"doc-{group}", created_at=index * 6 + group)
        for index in range(4)
        for group in range(6)
    ]
    dag = make_dag(tasks)
    scheduler = DAGScheduler(
        dag,
        max_batch_size=4,
        fairness_scale=scale,
        fairness_tracker=tracker,
        affinity_advisor=PrefixHeatStub(),
    )
    scheduler.record_batch_service_time(2000.0)
    policies = []
    selected_ids = []
    prefix_groups = []
    for round_index in range(6):
        batch = await scheduler.get_next_batch()
        assert len(batch) == 4
        decision = scheduler.last_decision
        assert decision is not None
        policies.append(decision["policy"])
        selected_ids.extend(task.task_id for task in batch)
        prefix_groups.append({task.context_refs[0] for task in batch})
        if scale == "service_time":
            assert decision["policy_score"]["fairness"]["max_relative_wait"] == pytest.approx(
                round_index / 6
            )
            assert decision["policy_score"]["fairness"]["starvation_risk"] == "low"
            for task in batch:
                assert tracker.task_snapshot(task.task_id)["expected_wait_batches"] == 6
        for task in batch:
            task.status = TaskStatus.COMPLETED
        clock.advance(2.0)
        scheduler.record_batch_service_time(2000.0)

    assert len(selected_ids) == len(set(selected_ids)) == 24
    assert dag.is_complete()
    if scale == "absolute":
        assert policies[0] == "cache_affinity"
        assert policies[2:] == ["fairness_balanced"] * 4
    else:
        assert policies == ["cache_affinity"] * 6
        assert all(len(prefixes) == 1 for prefixes in prefix_groups)


@pytest.mark.asyncio
async def test_medium_risk_keeps_affinity_and_ages_cold_tasks_in_ready_fifo_order():
    clock = ManualClock()
    tracker = FairnessTracker(fairness_scale="service_time", now=clock)
    tracker.record_service_time(2000.0)
    tracker.observe_ready(["oldest", "historical-a", "historical-b"], batch_size=1)
    clock.advance(0.5)
    tracker.observe_ready(["oldest", "younger", "historical-a"], batch_size=1)
    clock.advance(6.0)
    # Reverse insertion/creation order and favor younger's prefix. Ready age wins.
    tasks = [
        make_task("hot", "hot", created_at=0.0),
        make_task("younger", "warm", created_at=1.0),
        make_task("oldest", "cold", created_at=2.0),
    ]
    scheduler = DAGScheduler(
        make_dag(tasks),
        max_batch_size=1,
        fairness_scale="service_time",
        fairness_tracker=tracker,
        affinity_advisor=PrefixHeatStub({"cold": 0.0, "warm": 0.8, "hot": 1.0}),
    )
    batch = await scheduler.get_next_batch()
    decision = scheduler.last_decision
    assert decision is not None
    assert decision["policy"] == "cache_affinity"
    assert decision["policy_score"]["fairness"]["starvation_risk"] == "medium"
    assert decision["ordered_candidate_task_ids"] == ["oldest", "younger", "hot"]
    assert [task.task_id for task in batch] == ["oldest"]
    assert tracker.task_snapshot("oldest")["aging_due"] is True
    assert tracker.task_snapshot("younger")["aging_due"] is True
    assert tracker.task_snapshot("hot")["aging_due"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("force_policy", [None, "cache_affinity"])
@pytest.mark.parametrize("risk_source", ["wait", "deferrals"])
async def test_overdue_cold_task_is_selected_even_when_cache_policy_is_forced(
    force_policy, risk_source
):
    clock = ManualClock()
    tracker = FairnessTracker(fairness_scale="service_time", now=clock)
    tracker.record_service_time(2000.0)
    tracker.observe_ready(["old"], batch_size=1)
    if risk_source == "wait":
        clock.advance(5.0)
    else:
        for _ in range(5):
            tracker.record_decision(deferred_ids=["old"])
    scheduler = DAGScheduler(
        make_dag([make_task("hot", "hot"), make_task("old", "cold")]),
        max_batch_size=1,
        fairness_scale="service_time",
        fairness_tracker=tracker,
        affinity_advisor=PrefixHeatStub({"hot": 1.0, "cold": 0.0}),
        force_policy=force_policy,
    )
    batch = await scheduler.get_next_batch()
    decision = scheduler.last_decision
    assert decision is not None
    assert decision["policy_score"]["fairness"]["starvation_risk"] == "high"
    assert decision["policy"] == (
        "cache_affinity" if force_policy else "fairness_balanced"
    )
    assert [task.task_id for task in batch] == ["old"]
    snapshot = tracker.task_snapshot("old")
    assert snapshot["relative_wait"] == pytest.approx(2.5 if risk_source == "wait" else 0.0)
    if risk_source == "deferrals":
        assert snapshot["relative_deferred"] == 5.0
    assert snapshot["aging_due"] is True


@pytest.mark.asyncio
async def test_absolute_forced_affinity_retains_original_order_without_aging():
    clock = ManualClock()
    tracker = FairnessTracker(now=clock)
    tracker.record_service_time(2000.0)
    tracker.observe_ready(["old"], batch_size=1)
    clock.advance(5.0)
    scheduler = DAGScheduler(
        make_dag([make_task("old", "cold"), make_task("hot", "hot")]),
        max_batch_size=1,
        fairness_tracker=tracker,
        affinity_advisor=PrefixHeatStub({"hot": 1.0, "cold": 0.0}),
        force_policy="cache_affinity",
    )
    assert [task.task_id for task in await scheduler.get_next_batch()] == ["hot"]
    assert tracker.task_snapshot("old")["aging_due"] is False


@pytest.mark.parametrize(
    ("elapsed", "deferrals", "relative_wait", "relative_deferred", "risk", "aging_due"),
    [
        (11.0, 5, 11 / 12, 5 / 6, StarvationRisk.LOW, False),
        (12.0, 0, 1.0, 0.0, StarvationRisk.MEDIUM, True),
        (0.0, 6, 0.0, 1.0, StarvationRisk.MEDIUM, True),
        (24.0, 0, 2.0, 0.0, StarvationRisk.HIGH, True),
        (0.0, 12, 0.0, 2.0, StarvationRisk.HIGH, True),
    ],
)
def test_relative_wait_and_deferral_thresholds(
    elapsed, deferrals, relative_wait, relative_deferred, risk, aging_due
):
    clock = ManualClock()
    tracker = FairnessTracker(fairness_scale="service_time", now=clock)
    tracker.record_service_time(2000.0)
    task_ids = [f"task-{index}" for index in range(24)]
    tracker.observe_ready(task_ids, batch_size=4)
    for _ in range(deferrals):
        tracker.record_decision(deferred_ids=["task-0"])
    clock.advance(elapsed)
    tracker.observe_ready(task_ids, batch_size=4)

    snapshot = tracker.task_snapshot("task-0")
    assert snapshot["expected_wait_batches"] == 6
    assert snapshot["relative_wait"] == pytest.approx(relative_wait)
    assert snapshot["relative_deferred"] == pytest.approx(relative_deferred)
    assert snapshot["starvation_risk"] == risk.value
    assert snapshot["aging_due"] is aging_due
    assert tracker.summarize().starvation_risk == risk


def test_queue_shrink_and_new_arrivals_cannot_rewrite_existing_wait_budget():
    clock = ManualClock()
    tracker = FairnessTracker(fairness_scale="service_time", now=clock)
    tracker.record_service_time(2000.0)
    initial_ids = [f"initial-{index}" for index in range(24)]
    tracker.observe_ready(initial_ids, batch_size=4)
    clock.advance(4.0)
    tracker.observe_ready(["initial-0"], batch_size=4)
    assert tracker.task_snapshot("initial-0")["expected_wait_batches"] == 6
    assert tracker.task_snapshot("initial-0")["starvation_risk"] == "low"

    clock.advance(9.0)
    arrivals = [f"new-{index}" for index in range(40)]
    tracker.observe_ready(["initial-0", *arrivals], batch_size=4)
    old = tracker.task_snapshot("initial-0")
    assert old["expected_wait_batches"] == 6
    assert old["relative_wait"] == pytest.approx(13 / 12)
    assert old["aging_due"] is True
    assert tracker.task_snapshot("new-0")["expected_wait_batches"] == 11
    assert tracker.task_snapshot("new-0")["relative_wait"] == 0.0

    clock.advance(12.0)
    tracker.observe_ready(["initial-0", *arrivals], batch_size=4)
    assert tracker.task_snapshot("initial-0")["starvation_risk"] == "high"


def test_service_time_summary_excludes_no_longer_ready_historical_task():
    clock = ManualClock()
    tracker = FairnessTracker(fairness_scale="service_time", now=clock)
    tracker.record_service_time(2000.0)
    tracker.observe_ready(["old"], batch_size=1)
    clock.advance(5.0)
    tracker.observe_ready(["old"], batch_size=1)
    assert tracker.summarize().starvation_risk == StarvationRisk.HIGH
    tracker.record_decision(selected_ids=["old"])

    tracker.observe_ready(["new"], batch_size=1)
    summary = tracker.summarize()
    assert summary.starvation_risk == StarvationRisk.LOW
    assert summary.max_wait_ms == 0.0
    assert set(summary.to_dict()["task_metrics"]) == {"new"}
    assert tracker.task_snapshot("old")["starvation_risk"] == "high"

    tracker.observe_ready([], batch_size=1)
    summary = tracker.summarize()
    assert summary.starvation_risk == StarvationRisk.LOW
    assert summary.to_dict()["task_metrics"] == {}


def test_cold_start_falls_back_to_absolute_then_observations_enable_relative_scale():
    clock = ManualClock()
    tracker = FairnessTracker(fairness_scale="service_time", now=clock)
    ids = [f"task-{index}" for index in range(24)]
    tracker.observe_ready(ids, batch_size=4)
    clock.advance(4.0)
    tracker.observe_ready(ids, batch_size=4)
    assert tracker.summarize().starvation_risk == StarvationRisk.HIGH
    snapshot = tracker.task_snapshot("task-0")
    assert snapshot["effective_scale"] == "absolute"
    assert snapshot["relative_wait"] is None
    assert snapshot["aging_due"] is False

    tracker.record_service_time(2000.0)
    tracker.observe_ready(ids, batch_size=4)
    snapshot = tracker.task_snapshot("task-0")
    assert snapshot["effective_scale"] == "service_time"
    assert snapshot["relative_wait"] == pytest.approx(1 / 3)
    assert snapshot["high_wait_ms"] == pytest.approx(24000.0)
    assert tracker.summarize().starvation_risk == StarvationRisk.LOW

    tracker.record_service_time(6000.0)
    tracker.observe_ready(ids, batch_size=4)
    snapshot = tracker.task_snapshot("task-0")
    assert snapshot["avg_service_time_ms"] == pytest.approx(3000.0)
    assert snapshot["medium_wait_ms"] == pytest.approx(18000.0)
    assert snapshot["high_wait_ms"] == pytest.approx(36000.0)


@pytest.mark.asyncio
async def test_policy_event_preserves_predecision_metrics_and_effective_thresholds():
    clock = ManualClock()
    tracker = FairnessTracker(fairness_scale="service_time", now=clock)
    tracker.record_service_time(2000.0)
    tracker.observe_ready(["hot", "cold"], batch_size=1)
    tracker.record_decision(deferred_ids=["cold"])
    clock.advance(2.0)
    scheduler = DAGScheduler(
        make_dag([make_task("cold", "cold"), make_task("hot", "hot")]),
        max_batch_size=1,
        fairness_scale="service_time",
        fairness_tracker=tracker,
        affinity_advisor=PrefixHeatStub({"hot": 1.0, "cold": 0.0}),
    )
    await scheduler.get_next_batch()
    decision = scheduler.last_decision
    assert decision is not None
    inputs = decision["policy_score"]["fairness"]
    assert inputs["starvation_risk"] == "low"
    assert inputs["avg_service_time_ms"] == 2000.0
    assert inputs["max_relative_wait"] == pytest.approx(0.5)
    assert inputs["max_relative_deferred"] == pytest.approx(0.5)
    assert inputs["medium_relative_wait"] == 1.0
    assert inputs["high_relative_wait"] == 2.0
    assert inputs["task_metrics"]["cold"]["medium_wait_ms"] == 4000.0
    assert inputs["task_metrics"]["cold"]["high_wait_ms"] == 8000.0
    assert inputs["task_metrics"]["cold"]["deferred_count"] == 1
    # This decision's new deferral must not rewrite the inputs used to select.
    assert decision["fairness"]["starvation_risk"] == "medium"
    assert decision["fairness"]["max_relative_deferred"] == 1.0
    assert inputs["task_metrics"]["cold"]["deferred_count"] == 1


@pytest.mark.parametrize("invalid_scale", ["relative", "", None])
def test_invalid_fairness_scale_is_rejected(invalid_scale):
    with pytest.raises(ValueError, match="fairness_scale"):
        FairnessTracker(fairness_scale=invalid_scale)
    with pytest.raises(ValueError, match="fairness_scale"):
        DAGScheduler(TaskDAG(), fairness_scale=invalid_scale)


@pytest.mark.parametrize(
    ("scheduler_scale", "tracker_scale"),
    [("absolute", "service_time"), ("service_time", "absolute")],
)
def test_injected_tracker_scale_must_match_scheduler(scheduler_scale, tracker_scale):
    with pytest.raises(ValueError, match="fairness_scale"):
        DAGScheduler(
            TaskDAG(),
            fairness_scale=scheduler_scale,
            fairness_tracker=FairnessTracker(fairness_scale=tracker_scale),
        )


@pytest.mark.parametrize("duration", [0.0, -1.0, float("nan"), float("inf"), True, "invalid", None])
def test_invalid_service_time_cannot_poison_observed_mean(duration):
    tracker = FairnessTracker(fairness_scale="service_time")
    tracker.record_service_time(2000.0)
    with pytest.raises(ValueError):
        tracker.record_service_time(duration)
    tracker.observe_ready(["task"], batch_size=1)
    assert tracker.task_snapshot("task")["avg_service_time_ms"] == 2000.0


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.1, float("nan"), float("inf"), True, "invalid", None])
def test_invalid_service_time_alpha_is_rejected(alpha):
    with pytest.raises(ValueError):
        FairnessTracker(fairness_scale="service_time", service_time_alpha=alpha)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_invalid_batch_width_is_rejected(batch_size):
    tracker = FairnessTracker(fairness_scale="service_time")
    with pytest.raises(ValueError):
        tracker.observe_ready(["task"], batch_size=batch_size)


class PreferCacheTunerStub:
    def recommend_policy_bias(self):
        return {
            "enabled": True,
            "policy_bias": {"cache_affinity": 5},
            "reasons": ["cache reuse"],
        }

    def observe_decision(self, decision):
        pass


@pytest.mark.asyncio
async def test_adaptive_cache_bias_cannot_bypass_service_time_high_limit():
    clock = ManualClock()
    tracker = FairnessTracker(fairness_scale="service_time", now=clock)
    tracker.record_service_time(2000.0)
    tracker.observe_ready(["old"], batch_size=1)
    clock.advance(5.0)
    scheduler = DAGScheduler(
        make_dag([make_task("old", "cold"), make_task("hot", "hot")]),
        max_batch_size=1,
        fairness_scale="service_time",
        fairness_tracker=tracker,
        affinity_advisor=PrefixHeatStub({"hot": 1.0, "cold": 0.0}),
        adaptive_tuner=PreferCacheTunerStub(),
    )
    batch = await scheduler.get_next_batch()
    assert scheduler.last_decision["policy"] == "fairness_balanced"
    assert [task.task_id for task in batch] == ["old"]


@pytest.mark.asyncio
async def test_critical_sla_still_has_priority_over_service_time_high_limit():
    clock = ManualClock()
    tracker = FairnessTracker(fairness_scale="service_time", now=clock)
    tracker.record_service_time(2000.0)
    tracker.observe_ready(["old"], batch_size=1)
    clock.advance(5.0)
    critical = make_task("critical", "hot")
    critical.sla_class = "critical"
    scheduler = DAGScheduler(
        make_dag([make_task("old", "cold"), critical]),
        max_batch_size=1,
        fairness_scale="service_time",
        fairness_tracker=tracker,
        affinity_advisor=PrefixHeatStub({"hot": 1.0, "cold": 0.0}),
    )
    await scheduler.get_next_batch()
    assert scheduler.last_decision["policy_score"]["fairness"]["starvation_risk"] == "high"
    assert scheduler.last_decision["policy"] == "risk_aware"

