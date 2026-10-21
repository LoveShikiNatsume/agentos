"""Tests for adaptive policy tuning (A6).

Covers:
1. Empty window produces no strong bias
2. High max_wait_ms produces fairness_balanced bias
3. High max_urgency_score produces risk_aware bias
4. Cache locality signal produces cache_affinity bias
5. Backpressure/safety pressure produces safety/risk bias
6. DAGScheduler decision includes adaptive_tuning field
7. Runtime emits scheduler.adaptive_tuning event
8. Schema validates adaptive_tuning
9. Audit counts adaptive_tuning events
10. Evidence report contains Adaptive Policy Tuning Evidence
"""

from __future__ import annotations

import pytest

from agentos.core.task import Task, TaskDAG, TaskStatus
from agentos.scheduler.adaptive import AdaptivePolicyTuner
from agentos.scheduler.dag import DAGScheduler
from agentos.scheduler.policy import PolicySelector, SchedulingPolicy
from agentos.scheduler.schema import validate_scheduler_decision_v2
from agentos.scheduler.audit import audit_scheduler_events
from scripts.generate_scheduler_evidence import generate_report


# ---------------------------------------------------------------------------
# 1. Empty window produces no strong bias
# ---------------------------------------------------------------------------


class TestAdaptiveTunerEmptyWindow:
    """Test that an empty observation window produces no strong bias."""

    def test_empty_window_returns_empty_bias(self):
        tuner = AdaptivePolicyTuner(window_size=5)
        result = tuner.recommend_policy_bias()

        assert result["enabled"] is True
        assert result["window_size"] == 5
        assert result["observed_decisions"] == 0
        assert result["policy_bias"] == {}
        assert result["reasons"] == []

    def test_snapshot_shows_empty_observations(self):
        tuner = AdaptivePolicyTuner(window_size=3)
        snap = tuner.snapshot()

        assert snap["window_size"] == 3
        assert snap["observed_decisions"] == 0
        assert snap["observations"] == []


# ---------------------------------------------------------------------------
# 2. High max_wait_ms produces fairness_balanced bias
# ---------------------------------------------------------------------------


class TestFairnessAdaptive:
    """Test that increasing wait times produce fairness_balanced bias."""

    def test_consecutive_high_wait_produces_fairness_bias(self):
        tuner = AdaptivePolicyTuner(window_size=5)

        # Feed observations with increasing max_wait_ms
        for wait in [100.0, 200.0, 300.0, 500.0, 800.0]:
            tuner.observe_decision({
                "policy": "latency_first",
                "workload_class": "interactive",
                "fairness": {
                    "avg_wait_ms": wait * 0.5,
                    "max_wait_ms": wait,
                    "fairness_score": 0.8,
                    "starvation_risk": "low",
                },
                "fairness_warnings": [],
                "deferred_task_ids": [],
                "blocked_task_ids": [],
                "runtime_health": {
                    "runtime_health_score": 0.9,
                    "pressure_level": "normal",
                },
                "sla": {"max_urgency_score": 0.3},
                "risk": {"high_risk_task_ids": []},
                "backpressure": {"signal": "none"},
                "safety": {"admission_counts": {"reject": 0}},
            })

        result = tuner.recommend_policy_bias()
        assert result["policy_bias"].get("fairness_balanced", 0) >= 1
        assert any("wait/fairness" in r for r in result["reasons"])

    def test_starvation_warning_produces_fairness_bias(self):
        tuner = AdaptivePolicyTuner(window_size=3)

        for _ in range(3):
            tuner.observe_decision({
                "policy": "latency_first",
                "workload_class": "batch",
                "fairness": {
                    "avg_wait_ms": 50.0,
                    "max_wait_ms": 100.0,
                    "fairness_score": 0.9,
                    "starvation_risk": "low",
                },
                "fairness_warnings": [
                    {"task_id": "t1", "starvation_risk": "high"},
                ],
                "deferred_task_ids": [],
                "blocked_task_ids": [],
                "runtime_health": {
                    "runtime_health_score": 0.8,
                    "pressure_level": "normal",
                },
                "sla": {"max_urgency_score": 0.2},
                "risk": {"high_risk_task_ids": []},
                "backpressure": {"signal": "none"},
                "safety": {"admission_counts": {"reject": 0}},
            })

        result = tuner.recommend_policy_bias()
        assert result["policy_bias"].get("fairness_balanced", 0) >= 1


# ---------------------------------------------------------------------------
# 3. High max_urgency_score produces risk_aware bias
# ---------------------------------------------------------------------------


class TestRiskAdaptive:
    """Test that high urgency produces risk_aware bias."""

    def test_high_urgency_produces_risk_aware_bias(self):
        tuner = AdaptivePolicyTuner(window_size=3)

        for _ in range(3):
            tuner.observe_decision({
                "policy": "latency_first",
                "workload_class": "interactive",
                "fairness": {
                    "avg_wait_ms": 10.0,
                    "max_wait_ms": 50.0,
                    "fairness_score": 0.95,
                    "starvation_risk": "low",
                },
                "fairness_warnings": [],
                "deferred_task_ids": [],
                "blocked_task_ids": [],
                "runtime_health": {
                    "runtime_health_score": 0.9,
                    "pressure_level": "normal",
                },
                "sla": {"max_urgency_score": 0.9},
                "risk": {"high_risk_task_ids": ["t1"]},
                "backpressure": {"signal": "none"},
                "safety": {"admission_counts": {"reject": 0}},
            })

        result = tuner.recommend_policy_bias()
        assert result["policy_bias"].get("risk_aware", 0) >= 1
        assert any("deadline/SLA" in r for r in result["reasons"])


# ---------------------------------------------------------------------------
# 4. Cache locality signal produces cache_affinity bias
# ---------------------------------------------------------------------------


class TestCacheAffinityAdaptive:
    """Test that cache locality signals produce cache_affinity bias."""

    def test_high_cache_hit_produces_cache_affinity_bias(self):
        tuner = AdaptivePolicyTuner(window_size=3)

        for _ in range(3):
            tuner.observe_decision({
                "policy": "latency_first",
                "workload_class": "ai_inference",
                "fairness": {
                    "avg_wait_ms": 10.0,
                    "max_wait_ms": 50.0,
                    "fairness_score": 0.95,
                    "starvation_risk": "low",
                },
                "fairness_warnings": [],
                "cache_hit_ratio": 0.8,
                "deferred_task_ids": [],
                "blocked_task_ids": [],
                "runtime_health": {
                    "runtime_health_score": 0.95,
                    "pressure_level": "normal",
                },
                "sla": {"max_urgency_score": 0.3},
                "risk": {"high_risk_task_ids": []},
                "backpressure": {"signal": "none"},
                "safety": {"admission_counts": {"reject": 0}},
            })

        result = tuner.recommend_policy_bias()
        assert result["policy_bias"].get("cache_affinity", 0) >= 1
        assert any("cache" in r.lower() for r in result["reasons"])


# ---------------------------------------------------------------------------
# 5. Backpressure/safety pressure produces safety/risk bias
# ---------------------------------------------------------------------------


class TestBackpressureSafetyAdaptive:
    """Test that backpressure/safety signals produce appropriate bias."""

    def test_hard_backpressure_produces_safety_bias(self):
        tuner = AdaptivePolicyTuner(window_size=3)

        for _ in range(3):
            tuner.observe_decision({
                "policy": "throughput_first",
                "workload_class": "batch",
                "fairness": {
                    "avg_wait_ms": 10.0,
                    "max_wait_ms": 50.0,
                    "fairness_score": 0.9,
                    "starvation_risk": "low",
                },
                "fairness_warnings": [],
                "deferred_task_ids": [],
                "blocked_task_ids": [],
                "runtime_health": {
                    "runtime_health_score": 0.5,
                    "pressure_level": "critical",
                },
                "sla": {"max_urgency_score": 0.3},
                "risk": {"high_risk_task_ids": []},
                "backpressure": {"signal": "hard"},
                "safety": {"admission_counts": {"reject": 1}},
            })

        result = tuner.recommend_policy_bias()
        assert result["policy_bias"].get("throughput_first", 0) <= -1
        assert result["policy_bias"].get("risk_aware", 0) >= 1
        assert any("backpressure" in r.lower() or "safety" in r.lower() for r in result["reasons"])


# ---------------------------------------------------------------------------
# 6. DAGScheduler decision includes adaptive_tuning field
# ---------------------------------------------------------------------------


class TestDAGSchedulerAdaptiveTuning:
    """Test that DAGScheduler includes adaptive_tuning in decisions."""

    async def test_decision_has_adaptive_tuning_when_tuner_provided(self):
        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

        tuner = AdaptivePolicyTuner(window_size=5)
        scheduler = DAGScheduler(dag, adaptive_tuner=tuner)
        await scheduler.get_next_batch()

        decision = scheduler.last_decision
        assert "adaptive_tuning" in decision
        at = decision["adaptive_tuning"]
        assert at["enabled"] is True
        assert at["window_size"] == 5
        # observed_decisions reflects count at bias computation time (before this decision)
        assert at["observed_decisions"] == 0
        assert isinstance(at["policy_bias"], dict)
        assert isinstance(at["reasons"], list)
        # After get_next_batch, the tuner should have observed the decision
        assert len(tuner._observations) == 1

    async def test_decision_no_adaptive_tuning_without_tuner(self):
        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

        scheduler = DAGScheduler(dag)
        await scheduler.get_next_batch()

        decision = scheduler.last_decision
        assert "adaptive_tuning" not in decision

    async def test_adaptive_tuner_observations_grow(self):
        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))
        dag.add_task(Task(task_id="t2", name="work2", agent_role="worker"))

        tuner = AdaptivePolicyTuner(window_size=5)
        scheduler = DAGScheduler(dag, max_batch_size=1, adaptive_tuner=tuner)

        await scheduler.get_next_batch()
        dag.tasks["t1"].status = TaskStatus.COMPLETED
        await scheduler.get_next_batch()

        snap = tuner.snapshot()
        assert snap["observed_decisions"] == 2


# ---------------------------------------------------------------------------
# 7. Runtime emits scheduler.adaptive_tuning event
# ---------------------------------------------------------------------------


class TestRuntimeAdaptiveTuningEvent:
    """Test that Runtime emits scheduler.adaptive_tuning event."""

    async def test_runtime_emits_adaptive_tuning_event(self):
        from agentos.llm.mock import MockLLMBackend
        from agentos.runtime.engine import AgentHandler, Runtime

        class TaskIdHandler(AgentHandler):
            async def run(self, ctx: AgentHandler.Context) -> str:
                return ctx.task.task_id

        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

        tuner = AdaptivePolicyTuner(window_size=5)
        rt = Runtime(
            llm_backend=MockLLMBackend(),
            handlers={"worker": TaskIdHandler()},
            adaptive_tuner=tuner,
        )

        await rt.run(dag)

        events = [
            e for e in rt.events
            if e.event_type == "scheduler.adaptive_tuning"
        ]
        assert len(events) > 0
        event = events[0]
        assert "window_size" in event.data
        assert "observed_decisions" in event.data
        assert "policy_bias" in event.data
        assert "reasons" in event.data

    async def test_no_adaptive_tuning_event_without_tuner(self):
        from agentos.llm.mock import MockLLMBackend
        from agentos.runtime.engine import AgentHandler, Runtime

        class TaskIdHandler(AgentHandler):
            async def run(self, ctx: AgentHandler.Context) -> str:
                return ctx.task.task_id

        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

        rt = Runtime(
            llm_backend=MockLLMBackend(),
            handlers={"worker": TaskIdHandler()},
        )

        await rt.run(dag)

        events = [
            e for e in rt.events
            if e.event_type == "scheduler.adaptive_tuning"
        ]
        assert len(events) == 0


# ---------------------------------------------------------------------------
# 8. Schema validates adaptive_tuning
# ---------------------------------------------------------------------------


class TestSchemaAdaptiveTuning:
    """Test that schema validation handles adaptive_tuning field."""

    def test_valid_adaptive_tuning_passes(self):
        decision = _make_valid_decision()
        decision["adaptive_tuning"] = {
            "enabled": True,
            "window_size": 5,
            "observed_decisions": 3,
            "policy_bias": {"fairness_balanced": 1},
            "reasons": ["recent wait/fairness pressure increased"],
        }
        errors = validate_scheduler_decision_v2(decision)
        # Should not have adaptive_tuning specific errors
        at_errors = [e for e in errors if "adaptive_tuning" in e]
        assert at_errors == []

    def test_invalid_adaptive_tuning_enabled_type(self):
        decision = _make_valid_decision()
        decision["adaptive_tuning"] = {
            "enabled": "yes",  # should be bool
            "window_size": 5,
            "observed_decisions": 0,
            "policy_bias": {},
            "reasons": [],
        }
        errors = validate_scheduler_decision_v2(decision)
        assert any("adaptive_tuning.enabled must be a bool" in e for e in errors)

    def test_invalid_adaptive_tuning_policy_bias_type(self):
        decision = _make_valid_decision()
        decision["adaptive_tuning"] = {
            "enabled": True,
            "window_size": 5,
            "observed_decisions": 0,
            "policy_bias": "invalid",  # should be dict
            "reasons": [],
        }
        errors = validate_scheduler_decision_v2(decision)
        assert any("adaptive_tuning.policy_bias must be a dict" in e for e in errors)

    def test_invalid_adaptive_tuning_reasons_type(self):
        decision = _make_valid_decision()
        decision["adaptive_tuning"] = {
            "enabled": True,
            "window_size": 5,
            "observed_decisions": 0,
            "policy_bias": {},
            "reasons": "invalid",  # should be list
        }
        errors = validate_scheduler_decision_v2(decision)
        assert any("adaptive_tuning.reasons must be a list" in e for e in errors)

    def test_negative_observed_decisions_fails(self):
        decision = _make_valid_decision()
        decision["adaptive_tuning"] = {
            "enabled": True,
            "window_size": 5,
            "observed_decisions": -1,
            "policy_bias": {},
            "reasons": [],
        }
        errors = validate_scheduler_decision_v2(decision)
        assert any("observed_decisions must be >= 0" in e for e in errors)


# ---------------------------------------------------------------------------
# 9. Audit counts adaptive_tuning events
# ---------------------------------------------------------------------------


class TestAuditAdaptiveTuning:
    """Test that audit tracks adaptive_tuning events."""

    def test_audit_counts_adaptive_tuning_event(self):
        events = [
            _make_valid_decision_event(),
            _make_adaptive_tuning_event(),
            _make_runtime_health_event(),
            _make_sla_summary_event(),
            _make_backpressure_event(),
        ]
        result = audit_scheduler_events(events)
        assert result.adaptive_tuning_count == 1

    def test_audit_counts_adaptive_policy_adjustment(self):
        # Decision with adaptive_tuning that has non-zero bias
        decision_event = _make_valid_decision_event()
        decision_event["data"]["adaptive_tuning"] = {
            "enabled": True,
            "window_size": 5,
            "observed_decisions": 3,
            "policy_bias": {"fairness_balanced": 2},
            "reasons": ["recent wait/fairness pressure increased"],
        }
        events = [
            decision_event,
            _make_adaptive_tuning_event(),
            _make_runtime_health_event(),
            _make_sla_summary_event(),
            _make_backpressure_event(),
        ]
        result = audit_scheduler_events(events)
        assert result.adaptive_policy_adjustment_count >= 1

    def test_audit_requires_adaptive_tuning_event_when_in_decision(self):
        decision_event = _make_valid_decision_event()
        decision_event["data"]["adaptive_tuning"] = {
            "enabled": True,
            "window_size": 5,
            "observed_decisions": 3,
            "policy_bias": {},
            "reasons": [],
        }
        events = [
            decision_event,
            _make_runtime_health_event(),
            _make_sla_summary_event(),
            _make_backpressure_event(),
        ]
        result = audit_scheduler_events(events)
        assert result.ok is False
        assert any("adaptive_tuning present in decision but no scheduler.adaptive_tuning event" in e for e in result.errors)


# ---------------------------------------------------------------------------
# 10. Evidence report contains Adaptive Policy Tuning Evidence
# ---------------------------------------------------------------------------


class TestEvidenceAdaptiveTuning:
    """Test that evidence report contains Adaptive Policy Tuning Evidence."""

    def test_report_contains_adaptive_tuning_section(self):
        events = [
            _make_valid_decision_event(),
            _make_adaptive_tuning_event(),
            _make_runtime_health_event(),
            _make_sla_summary_event(),
            _make_backpressure_event(),
        ]
        report = generate_report(events)
        assert "## Adaptive Policy Tuning Evidence" in report
        assert "sliding-window" in report.lower() or "rule-based" in report.lower()

    def test_report_shows_no_adaptive_tuning_when_absent(self):
        events = [
            _make_valid_decision_event(),
            _make_runtime_health_event(),
            _make_sla_summary_event(),
            _make_backpressure_event(),
        ]
        report = generate_report(events)
        assert "## Adaptive Policy Tuning Evidence" in report
        assert "No adaptive tuning evidence found" in report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_decision() -> dict:
    """Create a minimal valid scheduler.decision.v2 dict."""
    return {
        "schema_version": "scheduler.decision.v2",
        "strategy": "critical_path",
        "policy": "latency_first",
        "workload_class": "interactive",
        "policy_reason": "interactive workload or high priority",
        "policy_score": {"candidate_count": 1},
        "score_breakdown": {"candidate_count": 1},
        "fairness": {
            "max_wait_ms": 0.0,
            "avg_wait_ms": 0.0,
            "max_slowdown": 1.0,
            "fairness_score": 1.0,
            "starvation_risk": "low",
            "warnings": [],
        },
        "fairness_warnings": [],
        "ready_task_ids": ["t1"],
        "candidate_task_ids": ["t1"],
        "ordered_candidate_task_ids": ["t1"],
        "selected_task_ids": ["t1"],
        "blocked_task_ids": [],
        "deferred_task_ids": [],
        "selected": ["t1"],
        "blocked": [],
        "deferred": [],
        "available_resources": None,
        "max_batch_size": None,
        "runtime_mode": "user_space_runtime",
        "kernel_integration": "mock",
        "fallback_reason": "forced mock mode",
        "runtime_health": {
            "runtime_health_score": 1.0,
            "queue_pressure": 0.0,
            "resource_pressure": 0.0,
            "fairness_pressure": 0.0,
            "dependency_pressure": 0.0,
            "pressure_level": "normal",
            "health_warnings": [],
        },
        "sla": {
            "critical_count": 0,
            "interactive_count": 1,
            "batch_count": 0,
            "best_effort_count": 0,
            "deadline_task_count": 0,
            "max_urgency_score": 0.7,
        },
        "risk": {
            "max_task_risk_score": 0.1,
            "avg_task_risk_score": 0.1,
            "high_risk_task_ids": [],
            "risk_warnings": [],
        },
        "backpressure": {
            "signal": "none",
            "pressure_level": "normal",
            "defer_count": 0,
            "degrade_count": 0,
            "reject_count": 0,
            "reasons": [],
        },
        "safety": {
            "admission_counts": {"allow": 1, "defer": 0, "degrade": 0, "reject": 0},
            "rejected_task_ids": [],
            "safety_deferred_task_ids": [],
            "degraded_task_ids": [],
            "safety_violations": [],
        },
        "tasks": {
            "t1": {
                "task_id": "t1",
                "agent_role": "worker",
                "reason": "selected",
                "workload_class": "interactive",
                "policy": "latency_first",
                "policy_reason": "interactive workload or high priority",
                "critical_path_score": 0.0,
                "priority": 0,
                "dependencies": [],
                "context_refs": [],
                "resource_requirements": {},
                "resource_estimate": {},
                "wait_ms": 0.0,
                "slowdown": 1.0,
                "selected_count": 1,
                "deferred_count": 0,
                "blocked_count": 0,
                "starvation_risk": "low",
                "sla_class": "interactive",
                "deadline_ms": None,
                "urgency_score": 0.7,
                "task_risk_score": 0.1,
                "risk": {
                    "sla_class": "interactive",
                    "deadline_ms": None,
                    "urgency_score": 0.7,
                    "task_risk_score": 0.1,
                    "resource_risk": 0.0,
                    "timeout_risk": 0.0,
                    "failure_risk": 0.0,
                    "starvation_risk_score": 0.1,
                    "dependency_risk": 0.0,
                    "risk_reasons": [],
                },
                "admission_decision": "allow",
                "backpressure_signal": "none",
                "degrade_reason": None,
                "safety_violations": [],
                "safety": {
                    "admission_decision": "allow",
                    "backpressure_signal": "none",
                    "degrade_reason": None,
                    "safety_violations": [],
                },
            }
        },
    }


def _make_valid_decision_event() -> dict:
    return {"event_type": "scheduler.decision", "data": _make_valid_decision()}


def _make_adaptive_tuning_event() -> dict:
    return {
        "event_type": "scheduler.adaptive_tuning",
        "data": {
            "window_size": 5,
            "observed_decisions": 3,
            "policy_bias": {"fairness_balanced": 1},
            "reasons": ["recent wait/fairness pressure increased"],
            "selected_policy_before_bias": "latency_first",
            "selected_policy_after_bias": "fairness_balanced",
        },
    }


def _make_runtime_health_event() -> dict:
    return {
        "event_type": "scheduler.runtime_health",
        "data": {
            "runtime_health_score": 1.0,
            "queue_pressure": 0.0,
            "resource_pressure": 0.0,
            "fairness_pressure": 0.0,
            "dependency_pressure": 0.0,
            "pressure_level": "normal",
            "health_warnings": [],
        },
    }


def _make_sla_summary_event() -> dict:
    return {
        "event_type": "scheduler.sla_summary",
        "data": {
            "critical_count": 0,
            "interactive_count": 1,
            "batch_count": 0,
            "best_effort_count": 0,
            "deadline_task_count": 0,
            "max_urgency_score": 0.7,
        },
    }


def _make_backpressure_event() -> dict:
    return {
        "event_type": "scheduler.backpressure",
        "data": {
            "signal": "none",
            "pressure_level": "normal",
            "defer_count": 0,
            "degrade_count": 0,
            "reject_count": 0,
            "reasons": [],
        },
    }
