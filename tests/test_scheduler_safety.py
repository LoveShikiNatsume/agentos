"""Tests for Backpressure and Safety Envelope."""

from __future__ import annotations

import pytest

from agentos.scheduler.safety import (
    AdmissionDecision,
    BackpressureSignal,
    SafetyEnvelopeConfig,
    TaskSafetyDecision,
    backpressure_signal_from_pressure,
    evaluate_task_safety,
    summarize_safety,
)


class TestBackpressureSignalFromPressure:
    """Test backpressure_signal_from_pressure function."""

    def test_normal_pressure(self):
        assert backpressure_signal_from_pressure("normal") == BackpressureSignal.NONE

    def test_warning_pressure(self):
        assert backpressure_signal_from_pressure("warning") == BackpressureSignal.SOFT

    def test_critical_pressure(self):
        assert backpressure_signal_from_pressure("critical") == BackpressureSignal.HARD

    def test_overloaded_pressure(self):
        assert backpressure_signal_from_pressure("overloaded") == BackpressureSignal.REJECTING

    def test_none_pressure(self):
        assert backpressure_signal_from_pressure(None) == BackpressureSignal.NONE


class TestEvaluateTaskSafety:
    """Test evaluate_task_safety function."""

    def test_allow_under_normal_pressure(self):
        """Normal pressure should allow tasks."""
        decision = evaluate_task_safety(
            sla_class="batch",
            workload_class="compile",
            pressure_level="normal",
            task_risk_score=0.1,
            previous_failures=0,
            memory_max_mb=128,
            cpu_max_percent=20,
        )
        assert decision.admission_decision == "allow"
        assert decision.backpressure_signal == "none"

    def test_memory_violation_rejects_task(self):
        """Memory limit exceeded should reject task."""
        decision = evaluate_task_safety(
            sla_class="batch",
            workload_class="compile",
            pressure_level="normal",
            task_risk_score=0.1,
            previous_failures=0,
            memory_max_mb=8192,
            cpu_max_percent=50,
        )
        assert decision.admission_decision == "reject"
        assert "memory limit exceeded" in decision.safety_violations

    def test_cpu_violation_rejects_task(self):
        """CPU limit exceeded should reject task."""
        decision = evaluate_task_safety(
            sla_class="batch",
            workload_class="compile",
            pressure_level="normal",
            task_risk_score=0.1,
            previous_failures=0,
            memory_max_mb=128,
            cpu_max_percent=99,
        )
        assert decision.admission_decision == "reject"
        assert "cpu limit exceeded" in decision.safety_violations

    def test_failure_limit_rejects_task(self):
        """Failure limit exceeded should reject task."""
        decision = evaluate_task_safety(
            sla_class="batch",
            workload_class="compile",
            pressure_level="normal",
            task_risk_score=0.1,
            previous_failures=5,
            memory_max_mb=128,
            cpu_max_percent=20,
        )
        assert decision.admission_decision == "reject"
        assert "failure limit exceeded" in decision.safety_violations

    def test_best_effort_rejected_when_overloaded(self):
        """Best_effort should be rejected when overloaded."""
        decision = evaluate_task_safety(
            sla_class="best_effort",
            workload_class="compression",
            pressure_level="overloaded",
            task_risk_score=0.2,
            previous_failures=0,
            memory_max_mb=128,
            cpu_max_percent=20,
        )
        assert decision.admission_decision == "reject"
        assert decision.backpressure_signal == "rejecting"

    def test_batch_deferred_when_critical(self):
        """Batch should be deferred when critical."""
        decision = evaluate_task_safety(
            sla_class="batch",
            workload_class="compile",
            pressure_level="critical",
            task_risk_score=0.2,
            previous_failures=0,
            memory_max_mb=128,
            cpu_max_percent=20,
        )
        assert decision.admission_decision == "defer"
        assert decision.backpressure_signal == "hard"

    def test_high_risk_deferred_when_critical(self):
        """High risk task should be deferred when critical."""
        decision = evaluate_task_safety(
            sla_class="interactive",
            workload_class="ai_inference",
            pressure_level="critical",
            task_risk_score=0.9,
            previous_failures=0,
            memory_max_mb=128,
            cpu_max_percent=20,
        )
        assert decision.admission_decision == "defer"
        assert decision.backpressure_signal == "hard"

    def test_unknown_workload_degraded_when_hard_pressure(self):
        """Unknown workload should be degraded under hard pressure."""
        decision = evaluate_task_safety(
            sla_class="interactive",
            workload_class="unknown",
            pressure_level="critical",
            task_risk_score=0.2,
            previous_failures=0,
            memory_max_mb=128,
            cpu_max_percent=20,
            policy="throughput_first",
        )
        assert decision.admission_decision == "degrade"
        assert decision.backpressure_signal == "hard"

    def test_critical_allowed_under_normal_pressure(self):
        """Critical SLA should be allowed under normal pressure."""
        decision = evaluate_task_safety(
            sla_class="critical",
            workload_class="network",
            pressure_level="normal",
            task_risk_score=0.5,
            previous_failures=0,
            memory_max_mb=128,
            cpu_max_percent=20,
        )
        assert decision.admission_decision == "allow"


class TestTaskSafetyDecision:
    """Test TaskSafetyDecision data class."""

    def test_to_dict(self):
        decision = TaskSafetyDecision(
            admission_decision="reject",
            backpressure_signal="hard",
            degrade_reason="safety envelope violation",
            safety_violations=["memory limit exceeded"],
        )
        d = decision.to_dict()
        assert d["admission_decision"] == "reject"
        assert d["backpressure_signal"] == "hard"
        assert d["degrade_reason"] == "safety envelope violation"
        assert d["safety_violations"] == ["memory limit exceeded"]


class TestSummarizeSafety:
    """Test summarize_safety function."""

    def test_summarize_safety_counts(self):
        decisions = {
            "t1": {"admission_decision": "allow"},
            "t2": {"admission_decision": "defer"},
            "t3": {"admission_decision": "reject", "safety_violations": ["memory limit exceeded"]},
        }
        summary = summarize_safety(decisions)
        assert summary["admission_counts"]["allow"] == 1
        assert summary["admission_counts"]["defer"] == 1
        assert summary["admission_counts"]["reject"] == 1
        assert "t3" in summary["rejected_task_ids"]
        assert "t2" in summary["safety_deferred_task_ids"]
        assert "memory limit exceeded" in summary["safety_violations"]


# Tests for enforced admission control in DAGScheduler
import pytest

from agentos.core.resources import ResourceQuota
from agentos.core.task import Task, TaskDAG, TaskStatus
from agentos.scheduler.dag import DAGScheduler
from agentos.scheduler.safety import SafetyEnvelopeConfig


class TestEnforcedAdmissionControl:
    """Test that rejected tasks are not executed."""

    @pytest.mark.asyncio
    async def test_rejected_task_not_selected(self):
        """Test that rejected task is not in selected_task_ids."""
        dag = TaskDAG()
        dag.add_task(Task(
            task_id="normal",
            name="normal",
            agent_role="worker",
            resource_requirements=ResourceQuota(memory_max_mb=128),
        ))
        dag.add_task(Task(
            task_id="huge",
            name="huge",
            agent_role="worker",
            resource_requirements=ResourceQuota(memory_max_mb=8192),
        ))

        scheduler = DAGScheduler(dag)
        batch = await scheduler.get_next_batch()
        decision = scheduler.last_decision

        # huge should be rejected due to memory limit
        assert "huge" not in [t.task_id for t in batch]
        assert "huge" not in decision["selected_task_ids"]
        assert "huge" in decision["safety"]["rejected_task_ids"]

    @pytest.mark.asyncio
    async def test_rejected_task_not_executed_by_runtime(self):
        """Test that rejected task is not executed by runtime."""
        from agentos.llm.mock import MockLLMBackend
        from agentos.runtime.engine import AgentHandler, Runtime

        class TaskIdHandler(AgentHandler):
            async def run(self, ctx: AgentHandler.Context) -> str:
                return ctx.task.task_id

        dag = TaskDAG()
        dag.add_task(Task(
            task_id="normal",
            name="normal",
            agent_role="worker",
            resource_requirements=ResourceQuota(memory_max_mb=128),
        ))
        dag.add_task(Task(
            task_id="huge",
            name="huge",
            agent_role="worker",
            resource_requirements=ResourceQuota(memory_max_mb=8192),
        ))

        rt = Runtime(
            llm_backend=MockLLMBackend(),
            handlers={"worker": TaskIdHandler()},
        )

        await rt.run(dag)

        # huge should not be executed
        assert dag.tasks["huge"].status != TaskStatus.COMPLETED
        # normal should be executed
        assert dag.tasks["normal"].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_safety_rejection_reported_in_decision(self):
        """Test that safety rejection is reported in decision."""
        dag = TaskDAG()
        dag.add_task(Task(
            task_id="huge",
            name="huge",
            agent_role="worker",
            resource_requirements=ResourceQuota(memory_max_mb=8192),
        ))

        scheduler = DAGScheduler(dag)
        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        assert decision is not None
        assert "huge" in decision["safety"]["rejected_task_ids"]
        assert decision["safety"]["admission_counts"]["reject"] == 1
        assert "memory limit exceeded" in decision["safety"]["safety_violations"]

    @pytest.mark.asyncio
    async def test_deferred_task_reported_in_decision(self):
        """Test that deferred task is reported in decision."""
        # Use critical pressure level
        from agentos.scheduler.fairness import FairnessTracker

        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="t1", agent_role="worker"))
        dag.add_task(Task(task_id="t2", name="t2", agent_role="worker"))

        tracker = FairnessTracker(
            medium_wait_ms=10,
            high_wait_ms=20,
            medium_deferred_count=1,
            high_deferred_count=2,
            now=lambda: 10.0,
        )

        scheduler = DAGScheduler(dag, max_batch_size=1, fairness_tracker=tracker)

        # First batch: t1 selected, t2 deferred
        await scheduler.get_next_batch()
        dag.tasks["t1"].status = TaskStatus.COMPLETED

        # Advance time to trigger high starvation risk
        tracker._now = lambda: 10.025
        await scheduler.get_next_batch()

        decision = scheduler.last_decision
        # Should have backpressure due to high starvation risk
        assert decision["backpressure"]["signal"] in {"soft", "hard", "rejecting"}

    @pytest.mark.asyncio
    async def test_rejected_task_reason_is_rejected_by_safety_envelope(self):
        """Test that rejected task reason is 'rejected_by_safety_envelope'."""
        dag = TaskDAG()
        dag.add_task(Task(
            task_id="normal",
            name="normal",
            agent_role="worker",
            resource_requirements=ResourceQuota(memory_max_mb=128),
        ))
        dag.add_task(Task(
            task_id="huge",
            name="huge",
            agent_role="worker",
            resource_requirements=ResourceQuota(memory_max_mb=8192),
        ))

        scheduler = DAGScheduler(dag)
        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        # huge should be rejected with proper reason
        assert decision["tasks"]["huge"]["reason"] == "rejected_by_safety_envelope"

    @pytest.mark.asyncio
    async def test_rejected_task_not_in_deferred_task_ids(self):
        """Test that rejected task is not in deferred_task_ids."""
        dag = TaskDAG()
        dag.add_task(Task(
            task_id="normal",
            name="normal",
            agent_role="worker",
            resource_requirements=ResourceQuota(memory_max_mb=128),
        ))
        dag.add_task(Task(
            task_id="huge",
            name="huge",
            agent_role="worker",
            resource_requirements=ResourceQuota(memory_max_mb=8192),
        ))

        scheduler = DAGScheduler(dag)
        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        # huge should not be in deferred_task_ids
        assert "huge" not in decision["deferred_task_ids"]
        # huge should be in safety.rejected_task_ids
        assert "huge" in decision["safety"]["rejected_task_ids"]

    @pytest.mark.asyncio
    async def test_rejected_task_in_safety_rejected_task_ids(self):
        """Test that rejected task is in safety.rejected_task_ids."""
        dag = TaskDAG()
        dag.add_task(Task(
            task_id="normal",
            name="normal",
            agent_role="worker",
            resource_requirements=ResourceQuota(memory_max_mb=128),
        ))
        dag.add_task(Task(
            task_id="huge",
            name="huge",
            agent_role="worker",
            resource_requirements=ResourceQuota(memory_max_mb=8192),
        ))

        scheduler = DAGScheduler(dag)
        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        # huge should be in safety.rejected_task_ids
        assert "huge" in decision["safety"]["rejected_task_ids"]
        assert decision["safety"]["admission_counts"]["reject"] == 1
        assert "memory limit exceeded" in decision["safety"]["safety_violations"]
