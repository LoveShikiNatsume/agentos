"""Tests for scheduler.decision.v2 schema validation."""

from __future__ import annotations

import pytest

from agentos.scheduler.schema import (
    validate_scheduler_decision_v2,
    validate_scheduler_event,
)


# A minimal valid scheduler.decision.v2 for testing
VALID_DECISION = {
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
            # P5: SLA and Risk fields
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
            # P6: Safety fields
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


class TestValidateSchedulerDecisionV2:
    """Test validate_scheduler_decision_v2 function."""

    def test_valid_scheduler_decision_v2_has_no_errors(self):
        """Test that a valid decision has no errors."""
        errors = validate_scheduler_decision_v2(VALID_DECISION)
        assert errors == []

    def test_missing_required_field_reports_error(self):
        """Test that missing required fields are reported."""
        decision = {**VALID_DECISION}
        del decision["policy"]
        errors = validate_scheduler_decision_v2(decision)
        assert any("missing required field: policy" in e for e in errors)

    def test_wrong_schema_version_reports_error(self):
        """Test that wrong schema version is reported."""
        decision = {**VALID_DECISION, "schema_version": "scheduler.decision.v1"}
        errors = validate_scheduler_decision_v2(decision)
        assert any("schema_version must be 'scheduler.decision.v2'" in e for e in errors)

    def test_missing_fairness_field_reports_error(self):
        """Test that missing fairness fields are reported."""
        decision = {**VALID_DECISION, "fairness": {}}
        errors = validate_scheduler_decision_v2(decision)
        assert any("missing required fairness field: fairness_score" in e for e in errors)
        assert any("missing required fairness field: starvation_risk" in e for e in errors)

    def test_kernel_probe_shape_is_validated(self):
        """Test that kernel_probe shape is validated when present."""
        # Valid kernel_probe
        decision = {
            **VALID_DECISION,
            "kernel_probe": {
                "sched_ext": "unavailable",
                "bpf_map": "mock",
                "fallback_mode": "mock_bpf_map",
            },
        }
        errors = validate_scheduler_decision_v2(decision)
        assert errors == []

        # Invalid kernel_probe (missing field)
        decision = {
            **VALID_DECISION,
            "kernel_probe": {
                "sched_ext": "unavailable",
                "bpf_map": "mock",
            },
        }
        errors = validate_scheduler_decision_v2(decision)
        assert any("missing required kernel_probe field: fallback_mode" in e for e in errors)

    def test_bpf_map_shape_is_validated(self):
        """Test that bpf_map shape is validated when present."""
        # Valid bpf_map
        decision = {
            **VALID_DECISION,
            "bpf_map": {
                "mode": "mock",
                "task_count": 1,
                "tasks": {},
            },
        }
        errors = validate_scheduler_decision_v2(decision)
        assert errors == []

        # Invalid bpf_map (missing field)
        decision = {
            **VALID_DECISION,
            "bpf_map": {
                "mode": "mock",
                "task_count": 1,
            },
        }
        errors = validate_scheduler_decision_v2(decision)
        assert any("missing required bpf_map field: tasks" in e for e in errors)

    def test_none_kernel_probe_and_bpf_map_are_allowed(self):
        """Test that None kernel_probe and bpf_map are allowed."""
        decision = {**VALID_DECISION}
        # kernel_probe and bpf_map not present (None) is OK
        errors = validate_scheduler_decision_v2(decision)
        assert errors == []


class TestValidateSchedulerEvent:
    """Test validate_scheduler_event function."""

    def test_valid_event_has_no_errors(self):
        """Test that a valid event has no errors."""
        event = {
            "event_type": "scheduler.decision",
            "data": VALID_DECISION,
        }
        errors = validate_scheduler_event(event)
        assert errors == []

    def test_wrong_event_type_reports_error(self):
        """Test that wrong event type is reported."""
        event = {
            "event_type": "scheduler.policy_selected",
            "data": VALID_DECISION,
        }
        errors = validate_scheduler_event(event)
        assert any("event_type must be 'scheduler.decision'" in e for e in errors)

    def test_non_dict_data_reports_error(self):
        """Test that non-dict data is reported."""
        event = {
            "event_type": "scheduler.decision",
            "data": "not a dict",
        }
        errors = validate_scheduler_event(event)
        assert any("event data must be a dict" in e for e in errors)


class TestValidateOSIntegrationFields:
    """Test validation of OS integration fields."""

    def test_valid_os_integration_fields(self):
        """Test that valid OS integration fields pass validation."""
        decision = {**VALID_DECISION}
        errors = validate_scheduler_decision_v2(decision)
        assert errors == []

    def test_missing_runtime_mode_reports_error(self):
        """Test that missing runtime_mode is reported."""
        decision = {**VALID_DECISION}
        del decision["runtime_mode"]
        errors = validate_scheduler_decision_v2(decision)
        assert any("missing required field: runtime_mode" in e for e in errors)

    def test_wrong_runtime_mode_reports_error(self):
        """Test that wrong runtime_mode is reported."""
        decision = {**VALID_DECISION, "runtime_mode": "kernel_space"}
        errors = validate_scheduler_decision_v2(decision)
        assert any("runtime_mode must be 'user_space_runtime'" in e for e in errors)

    def test_missing_kernel_integration_reports_error(self):
        """Test that missing kernel_integration is reported."""
        decision = {**VALID_DECISION}
        del decision["kernel_integration"]
        errors = validate_scheduler_decision_v2(decision)
        assert any("missing required field: kernel_integration" in e for e in errors)

    def test_invalid_kernel_integration_reports_error(self):
        """Test that invalid kernel_integration is reported."""
        decision = {**VALID_DECISION, "kernel_integration": "invalid"}
        errors = validate_scheduler_decision_v2(decision)
        assert any("kernel_integration must be one of" in e for e in errors)

    def test_valid_kernel_integration_values(self):
        """Test that valid kernel_integration values pass validation."""
        for value in ["mock", "probe", "unavailable"]:
            decision = {**VALID_DECISION, "kernel_integration": value}
            errors = validate_scheduler_decision_v2(decision)
            assert errors == []

    def test_sched_ext_requires_policy_generation_and_task_kernel_fields(self):
        decision = {
            **VALID_DECISION,
            "kernel_integration": "sched_ext",
            "policy_generation": 3,
            "tasks": {
                "t1": {
                    **VALID_DECISION["tasks"]["t1"],
                    "queue_class": "latency",
                    "kernel_weight": 250,
                    "slice_us": 2_000,
                    "policy_generation": 3,
                }
            },
        }

        assert validate_scheduler_decision_v2(decision) == []

        decision["tasks"]["t1"]["policy_generation"] = 2
        errors = validate_scheduler_decision_v2(decision)
        assert any("policy_generation must match" in error for error in errors)

    def test_missing_fallback_reason_reports_error(self):
        """Test that missing fallback_reason is reported."""
        decision = {**VALID_DECISION}
        del decision["fallback_reason"]
        errors = validate_scheduler_decision_v2(decision)
        assert any("missing required field: fallback_reason" in e for e in errors)

    def test_none_fallback_reason_is_allowed(self):
        """Test that None fallback_reason is allowed."""
        decision = {**VALID_DECISION, "fallback_reason": None}
        errors = validate_scheduler_decision_v2(decision)
        assert errors == []


class TestValidateRuntimeHealth:
    """Test validation of runtime_health fields."""

    def test_missing_runtime_health_reports_error(self):
        """Test that missing runtime_health is reported."""
        decision = dict(VALID_DECISION)
        del decision["runtime_health"]
        errors = validate_scheduler_decision_v2(decision)
        assert any("missing required field: runtime_health" in e for e in errors)

    def test_valid_runtime_health_passes(self):
        """Test that valid runtime_health passes validation."""
        decision = {**VALID_DECISION}
        errors = validate_scheduler_decision_v2(decision)
        assert errors == []

    def test_invalid_runtime_health_score_reports_error(self):
        """Test that invalid runtime_health_score is reported."""
        decision = {
            **VALID_DECISION,
            "runtime_health": {
                **VALID_DECISION["runtime_health"],
                "runtime_health_score": 1.5,
            },
        }
        errors = validate_scheduler_decision_v2(decision)
        assert any("runtime_health_score" in e for e in errors)

    def test_invalid_pressure_level_reports_error(self):
        """Test that invalid pressure_level is reported."""
        decision = {
            **VALID_DECISION,
            "runtime_health": {
                **VALID_DECISION["runtime_health"],
                "pressure_level": "panic",
            },
        }
        errors = validate_scheduler_decision_v2(decision)
        assert any("pressure_level" in e for e in errors)

    def test_invalid_health_warnings_type_reports_error(self):
        """Test that invalid health_warnings type is reported."""
        decision = {
            **VALID_DECISION,
            "runtime_health": {
                **VALID_DECISION["runtime_health"],
                "health_warnings": "not a list",
            },
        }
        errors = validate_scheduler_decision_v2(decision)
        assert any("health_warnings must be a list" in e for e in errors)

    def test_missing_runtime_health_field_reports_error(self):
        """Test that missing runtime_health field is reported."""
        decision = {
            **VALID_DECISION,
            "runtime_health": {
                "runtime_health_score": 1.0,
                "queue_pressure": 0.0,
                # Missing other fields
            },
        }
        errors = validate_scheduler_decision_v2(decision)
        assert any("missing required runtime_health field" in e for e in errors)

    def test_none_runtime_health_reports_error(self):
        """Test that None runtime_health is reported."""
        decision = {**VALID_DECISION, "runtime_health": None}
        errors = validate_scheduler_decision_v2(decision)
        assert any("runtime_health must be a dict" in e for e in errors)

    def test_non_dict_runtime_health_reports_error(self):
        """Test that non-dict runtime_health is reported."""
        decision = {**VALID_DECISION, "runtime_health": "invalid"}
        errors = validate_scheduler_decision_v2(decision)
        assert any("runtime_health must be a dict" in e for e in errors)
