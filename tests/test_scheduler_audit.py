"""Tests for scheduler event audit."""

from __future__ import annotations

import json

import pytest

from agentos.scheduler.audit import SchedulerAuditResult, audit_scheduler_events


# A minimal valid scheduler.decision event
VALID_DECISION_EVENT = {
    "event_type": "scheduler.decision",
    "data": {
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
    },
}

VALID_POLICY_SELECTED_EVENT = {
    "event_type": "scheduler.policy_selected",
    "data": {
        "policy": "latency_first",
        "workload_class": "interactive",
        "reason": "interactive workload or high priority",
        "candidate_count": 1,
    },
}

VALID_KERNEL_PROBE_EVENT = {
    "event_type": "scheduler.kernel_probe",
    "data": {
        "sched_ext": "unavailable",
        "bpf_map": "mock",
        "fallback_mode": "mock_bpf_map",
        "reason": "forced mock mode",
    },
}

VALID_BPF_MAP_UPDATED_EVENT = {
    "event_type": "scheduler.bpf_map_updated",
    "data": {
        "mode": "mock",
        "task_count": 1,
        "updated_task_ids": ["t1"],
    },
}

VALID_RUNTIME_HEALTH_EVENT = {
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

VALID_SLA_SUMMARY_EVENT = {
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

VALID_RISK_WARNING_EVENT = {
    "event_type": "scheduler.risk_warning",
    "data": {
        "warning": "critical task present",
        "max_task_risk_score": 0.8,
        "high_risk_task_ids": ["t1"],
        "policy": "risk_aware",
    },
}

VALID_BACKPRESSURE_EVENT = {
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

VALID_SAFETY_VIOLATION_EVENT = {
    "event_type": "scheduler.safety_violation",
    "data": {
        "violation": "memory limit exceeded",
        "rejected_task_ids": ["t1"],
        "degraded_task_ids": [],
        "safety_deferred_task_ids": [],
        "backpressure_signal": "none",
        "pressure_level": "normal",
    },
}


class TestAuditSchedulerEvents:
    """Test audit_scheduler_events function."""

    def test_audit_accepts_valid_scheduler_event_chain(self):
        """Test that a valid event chain passes audit."""
        events = [
            VALID_KERNEL_PROBE_EVENT,
            VALID_DECISION_EVENT,
            VALID_POLICY_SELECTED_EVENT,
            VALID_BPF_MAP_UPDATED_EVENT,
            VALID_RUNTIME_HEALTH_EVENT,
            VALID_SLA_SUMMARY_EVENT,
            VALID_BACKPRESSURE_EVENT,
        ]
        result = audit_scheduler_events(events)
        assert result.ok is True
        assert result.decision_count == 1
        assert result.policy_selected_count == 1
        assert result.kernel_probe_count == 1
        assert result.bpf_map_updated_count == 1
        assert result.runtime_health_count == 1
        assert result.sla_summary_count == 1
        assert result.backpressure_count == 1
        assert "latency_first" in result.policies_seen
        assert "interactive" in result.workload_classes_seen

    def test_audit_reports_missing_decision(self):
        """Test that missing decision events are reported."""
        events = [
            VALID_POLICY_SELECTED_EVENT,
        ]
        result = audit_scheduler_events(events)
        assert result.ok is False
        assert any("no scheduler.decision events found" in e for e in result.errors)

    def test_audit_reports_invalid_decision_schema(self):
        """Test that invalid decision schema is reported."""
        invalid_event = {
            "event_type": "scheduler.decision",
            "data": {"schema_version": "scheduler.decision.v1"},
        }
        events = [invalid_event]
        result = audit_scheduler_events(events)
        assert result.ok is False
        assert any("schema_version must be 'scheduler.decision.v2'" in e for e in result.errors)

    def test_audit_tracks_policies_and_workload_classes(self):
        """Test that policies and workload classes are tracked."""
        # Create a decision with different policy
        batch_decision = {**VALID_DECISION_EVENT["data"], "policy": "throughput_first", "workload_class": "batch"}
        batch_event = {"event_type": "scheduler.decision", "data": batch_decision}
        events = [
            VALID_DECISION_EVENT,
            batch_event,
        ]
        result = audit_scheduler_events(events)
        assert "latency_first" in result.policies_seen
        assert "throughput_first" in result.policies_seen
        assert "interactive" in result.workload_classes_seen
        assert "batch" in result.workload_classes_seen

    def test_audit_accepts_no_fairness_warning_when_no_starvation(self):
        """Test that missing fairness_warning is OK when no starvation."""
        events = [
            VALID_DECISION_EVENT,
            VALID_RUNTIME_HEALTH_EVENT,
            VALID_SLA_SUMMARY_EVENT,
            VALID_BACKPRESSURE_EVENT,
        ]
        result = audit_scheduler_events(events)
        assert result.ok is True
        assert result.fairness_warning_count == 0

    def test_audit_ignores_non_scheduler_events(self):
        """Test that non-scheduler events are ignored."""
        non_scheduler_event = {
            "event_type": "task.completed",
            "data": {"task_id": "t1"},
        }
        events = [
            non_scheduler_event,
            VALID_DECISION_EVENT,
            VALID_RUNTIME_HEALTH_EVENT,
            VALID_SLA_SUMMARY_EVENT,
            VALID_BACKPRESSURE_EVENT,
        ]
        result = audit_scheduler_events(events)
        assert result.ok is True
        assert result.decision_count == 1

    def test_audit_checks_kernel_probe_event_when_in_decision(self):
        """Test that kernel_probe event is required when present in decision."""
        decision_with_probe = {
            **VALID_DECISION_EVENT["data"],
            "kernel_probe": {
                "sched_ext": "unavailable",
                "bpf_map": "mock",
                "fallback_mode": "mock_bpf_map",
            },
        }
        decision_event = {"event_type": "scheduler.decision", "data": decision_with_probe}
        events = [decision_event]
        result = audit_scheduler_events(events)
        assert result.ok is False
        assert any("kernel_probe present in decision but no scheduler.kernel_probe event" in e for e in result.errors)

    def test_audit_checks_bpf_map_event_when_in_decision(self):
        """Test that bpf_map_updated event is required when bpf_map in decision."""
        decision_with_bpf = {
            **VALID_DECISION_EVENT["data"],
            "bpf_map": {
                "mode": "mock",
                "task_count": 1,
                "tasks": {},
            },
        }
        decision_event = {"event_type": "scheduler.decision", "data": decision_with_bpf}
        events = [decision_event]
        result = audit_scheduler_events(events)
        assert result.ok is False
        assert any("bpf_map present in decision but no scheduler.bpf_map_updated event" in e for e in result.errors)


class TestSchedulerAuditResult:
    """Test SchedulerAuditResult data class."""

    def test_to_dict(self):
        result = SchedulerAuditResult(
            decision_count=2,
            policy_selected_count=1,
            policies_seen=["latency_first"],
            workload_classes_seen=["interactive"],
        )
        d = result.to_dict()
        assert d["ok"] is True
        assert d["decision_count"] == 2
        assert d["policies_seen"] == ["latency_first"]


# Tests for audit_scheduler_events.py script
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = (Path(__file__).parent.parent / "src")


class TestAuditSchedulerEventsScript:
    """Test audit_scheduler_events.py script."""

    def test_audit_script_reads_jsonl(self, tmp_path):
        """Test script reads JSONL and outputs audit result."""
        # Write a test JSONL file
        jsonl_path = tmp_path / "events.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps(VALID_DECISION_EVENT) + "\n")
            f.write(json.dumps(VALID_POLICY_SELECTED_EVENT) + "\n")
            f.write(json.dumps(VALID_RUNTIME_HEALTH_EVENT) + "\n")
            f.write(json.dumps(VALID_SLA_SUMMARY_EVENT) + "\n")
            f.write(json.dumps(VALID_BACKPRESSURE_EVENT) + "\n")

        result = subprocess.run(
            [
                sys.executable,
                "scripts/audit_scheduler_events.py",
                str(jsonl_path),
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert payload["ok"] is True
        assert payload["decision_count"] == 1
        assert "latency_first" in payload["policies_seen"]

    def test_audit_script_exits_1_for_missing_file(self):
        """Test script exits 1 for missing input file."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/audit_scheduler_events.py",
                "/tmp/nonexistent_events.jsonl",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1

    def test_audit_script_writes_output_file(self, tmp_path):
        """Test script writes output to file."""
        # Write a test JSONL file
        jsonl_path = tmp_path / "events.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps(VALID_DECISION_EVENT) + "\n")
            f.write(json.dumps(VALID_RUNTIME_HEALTH_EVENT) + "\n")
            f.write(json.dumps(VALID_SLA_SUMMARY_EVENT) + "\n")
            f.write(json.dumps(VALID_BACKPRESSURE_EVENT) + "\n")

        out_path = tmp_path / "audit.json"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/audit_scheduler_events.py",
                str(jsonl_path),
                "--out",
                str(out_path),
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.returncode == 0
        assert out_path.exists()
        payload = json.loads(out_path.read_text())
        assert payload["ok"] is True

    def test_audit_requires_runtime_health_event_when_decision_has_health(self):
        """Test that runtime_health event is required when present in decision."""
        events = [
            VALID_DECISION_EVENT,
            VALID_SLA_SUMMARY_EVENT,
            VALID_BACKPRESSURE_EVENT,
        ]
        result = audit_scheduler_events(events)
        assert result.ok is False
        assert any("scheduler.runtime_health" in e for e in result.errors)

    def test_audit_passes_with_runtime_health_event(self):
        """Test that audit passes with runtime_health event."""
        events = [
            VALID_DECISION_EVENT,
            VALID_RUNTIME_HEALTH_EVENT,
            VALID_SLA_SUMMARY_EVENT,
            VALID_BACKPRESSURE_EVENT,
        ]
        result = audit_scheduler_events(events)
        assert result.ok is True
        assert result.runtime_health_count == 1
