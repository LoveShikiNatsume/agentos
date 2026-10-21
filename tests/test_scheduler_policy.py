import pytest

from agentos.core.task import Task, TaskDAG
from agentos.scheduler.dag import DAGScheduler
from agentos.scheduler.policy import (
    PolicySelector,
    SchedulingPolicy,
    StarvationRisk,
)
from agentos.scheduler.workload import WorkloadClass, WorkloadSample


def select(
    workload_class: WorkloadClass,
    *,
    prefix_heat: float | None = None,
    expected_parallelism: int | None = None,
    priority: int = 0,
    starvation_risk: StarvationRisk = StarvationRisk.LOW,
):
    sample = WorkloadSample(
        task_id="task",
        agent_role="worker",
        workload_hint=workload_class.value,
        prefix_heat=prefix_heat,
        expected_parallelism=expected_parallelism,
        priority=priority,
    )
    return PolicySelector().select(
        [sample],
        {sample.task_id: workload_class},
        starvation_risk=starvation_risk,
    )


def test_interactive_workload_selects_latency_first():
    decision = select(WorkloadClass.INTERACTIVE)

    assert decision.policy == SchedulingPolicy.LATENCY_FIRST
    assert decision.workload_class == WorkloadClass.INTERACTIVE
    assert "interactive" in decision.reason


def test_batch_workload_selects_throughput_first():
    decision = select(WorkloadClass.BATCH, expected_parallelism=8)

    assert decision.policy == SchedulingPolicy.THROUGHPUT_FIRST
    assert decision.score_breakdown["expected_parallelism"] == 8


def test_hot_prefix_selects_cache_affinity():
    decision = select(WorkloadClass.AI_INFERENCE, prefix_heat=0.91)

    assert decision.policy == SchedulingPolicy.CACHE_AFFINITY
    assert decision.workload_class == WorkloadClass.AI_INFERENCE
    assert decision.score_breakdown["max_prefix_heat"] == 0.91


def test_high_starvation_risk_selects_fairness_balanced():
    decision = select(
        WorkloadClass.INTERACTIVE,
        prefix_heat=0.95,
        starvation_risk=StarvationRisk.HIGH,
    )

    assert decision.policy == SchedulingPolicy.FAIRNESS_BALANCED
    assert decision.reason == "high starvation risk"


def test_unknown_workload_falls_back_to_latency_first():
    decision = select(WorkloadClass.UNKNOWN)

    assert decision.policy == SchedulingPolicy.LATENCY_FIRST
    assert decision.reason == "fallback critical path ordering"


def test_medium_starvation_risk_is_recorded_but_does_not_force_fairness_policy():
    """Medium starvation risk should be recorded in score_breakdown but not force fairness_balanced."""
    decision = select(WorkloadClass.INTERACTIVE, starvation_risk=StarvationRisk.MEDIUM)

    # Medium risk should NOT force fairness_balanced
    assert decision.policy != SchedulingPolicy.FAIRNESS_BALANCED
    # But it should be recorded in score_breakdown
    assert decision.score_breakdown["starvation_risk"] == "medium"


def test_policy_decision_serializes_to_event_data():
    decision = select(WorkloadClass.BATCH, expected_parallelism=6)

    d = decision.to_dict()
    assert d["policy"] == "throughput_first"
    assert d["reason"] == "batch workload or high expected parallelism"
    assert d["workload_class"] == "batch"
    assert d["score_breakdown"]["starvation_risk"] == "low"
    assert d["score_breakdown"]["expected_parallelism"] == 6
    assert d["score_breakdown"]["candidate_count"] == 1
    # P5: New fields in score_breakdown
    assert "has_critical_sla" in d["score_breakdown"]
    assert "max_urgency_score" in d["score_breakdown"]
    assert "max_task_risk_score" in d["score_breakdown"]
    assert "pressure_level" in d["score_breakdown"]


# --- force_policy tests (A1.1) ---


class TestForcePolicy:
    async def test_force_policy_overrides_policy_selector(self):
        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

        scheduler = DAGScheduler(dag, force_policy="throughput_first")
        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        assert decision["policy"] == "throughput_first"
        assert decision["policy_reason"] == "forced by ablation experiment"
        assert decision["score_breakdown"]["forced_policy"] is True

    async def test_force_policy_none_uses_policy_selector(self):
        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

        scheduler = DAGScheduler(dag)
        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        assert decision["policy_reason"] != "forced by ablation experiment"
        assert "forced_policy" not in decision.get("score_breakdown", {})

    async def test_force_policy_with_enum_value(self):
        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

        scheduler = DAGScheduler(dag, force_policy=SchedulingPolicy.CACHE_AFFINITY)
        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        assert decision["policy"] == "cache_affinity"
        assert decision["policy_reason"] == "forced by ablation experiment"

    async def test_force_policy_preserves_workload_classification(self):
        dag = TaskDAG()
        dag.add_task(Task(
            task_id="t1", name="work", agent_role="worker",
            workload_hint="ai_inference",
        ))

        scheduler = DAGScheduler(dag, force_policy="latency_first")
        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        # Workload classification should still run
        assert decision["workload_class"] == "ai_inference"
        assert decision["policy"] == "latency_first"

    async def test_force_policy_preserves_safety_and_fairness(self):
        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))
        dag.add_task(Task(task_id="t2", name="work2", agent_role="worker"))

        scheduler = DAGScheduler(dag, force_policy="throughput_first")
        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        # Safety and fairness should still be computed
        assert "fairness" in decision
        assert "safety" in decision
        assert "runtime_health" in decision
        assert "backpressure" in decision
