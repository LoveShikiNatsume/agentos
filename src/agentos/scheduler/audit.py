"""Scheduler event audit module.

This module reads Runtime events (from EventLog or in-memory) and validates
that scheduler events form a complete evidence chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentos.scheduler.schema import validate_scheduler_decision_v2


@dataclass
class SchedulerAuditResult:
    """Result of auditing scheduler events."""

    decision_count: int = 0
    policy_selected_count: int = 0
    fairness_warning_count: int = 0
    kernel_probe_count: int = 0
    bpf_map_updated_count: int = 0
    runtime_health_count: int = 0
    sla_summary_count: int = 0
    risk_warning_count: int = 0
    backpressure_count: int = 0
    safety_violation_count: int = 0
    adaptive_tuning_count: int = 0
    adaptive_policy_adjustment_count: int = 0
    policies_seen: list[str] = field(default_factory=list)
    workload_classes_seen: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "decision_count": self.decision_count,
            "policy_selected_count": self.policy_selected_count,
            "fairness_warning_count": self.fairness_warning_count,
            "kernel_probe_count": self.kernel_probe_count,
            "bpf_map_updated_count": self.bpf_map_updated_count,
            "runtime_health_count": self.runtime_health_count,
            "sla_summary_count": self.sla_summary_count,
            "risk_warning_count": self.risk_warning_count,
            "backpressure_count": self.backpressure_count,
            "safety_violation_count": self.safety_violation_count,
            "adaptive_tuning_count": self.adaptive_tuning_count,
            "adaptive_policy_adjustment_count": self.adaptive_policy_adjustment_count,
            "policies_seen": self.policies_seen,
            "workload_classes_seen": self.workload_classes_seen,
            "errors": self.errors,
            "error_count": self.error_count,
            "warnings": self.warnings,
            "warning_count": self.warning_count,
        }


def audit_scheduler_events(events: list[dict[str, Any]]) -> SchedulerAuditResult:
    """Audit a list of scheduler events for completeness.

    Parameters
    ----------
    events:
        List of event dicts.  Each should have ``event_type`` and ``data`` keys.
        Non-scheduler events are ignored.

    Returns
    -------
    SchedulerAuditResult
        The audit result with counts and errors.
    """
    result = SchedulerAuditResult()

    # Track unique policies and workload classes
    policies_set: set[str] = set()
    workload_classes_set: set[str] = set()

    # Track what we've seen
    has_decision = False
    has_kernel_probe_in_decision = False
    has_bpf_map_in_decision = False
    has_runtime_health_in_decision = False
    has_sla_in_decision = False
    has_risk_warnings_in_decision = False
    has_backpressure_in_decision = False
    has_safety_violations_in_decision = False
    has_adaptive_tuning_in_decision = False

    for event in events:
        event_type = event.get("event_type", "")

        # Skip non-scheduler events
        if not event_type.startswith("scheduler."):
            continue

        data = event.get("data", {})

        if event_type == "scheduler.decision":
            has_decision = True
            result.decision_count += 1

            # Validate decision schema
            schema_errors = validate_scheduler_decision_v2(data)
            for error in schema_errors:
                result.errors.append(f"decision {result.decision_count}: {error}")

            # Track policy and workload class
            policy = data.get("policy")
            if policy:
                policies_set.add(policy)
            workload_class = data.get("workload_class")
            if workload_class:
                workload_classes_set.add(workload_class)

            # Check for kernel_probe in decision
            if "kernel_probe" in data:
                has_kernel_probe_in_decision = True

            # Check for bpf_map in decision
            if "bpf_map" in data:
                has_bpf_map_in_decision = True

            # Check for runtime_health in decision
            if "runtime_health" in data:
                has_runtime_health_in_decision = True

            # Check for sla in decision
            if "sla" in data:
                has_sla_in_decision = True

            # Check for risk warnings in decision
            risk_summary = data.get("risk", {})
            if risk_summary.get("risk_warnings"):
                has_risk_warnings_in_decision = True

            # Check for backpressure in decision
            if "backpressure" in data:
                has_backpressure_in_decision = True

            # Check for safety violations in decision
            safety_summary = data.get("safety", {})
            if safety_summary.get("safety_violations"):
                has_safety_violations_in_decision = True

            # Check for adaptive_tuning in decision
            adaptive_tuning = data.get("adaptive_tuning")
            if adaptive_tuning is not None and isinstance(adaptive_tuning, dict) and adaptive_tuning.get("enabled"):
                has_adaptive_tuning_in_decision = True
                # Count adjustments: policy_bias has non-zero entries
                policy_bias = adaptive_tuning.get("policy_bias", {})
                if any(v != 0 for v in policy_bias.values()):
                    result.adaptive_policy_adjustment_count += 1

        elif event_type == "scheduler.policy_selected":
            result.policy_selected_count += 1
            policy = data.get("policy")
            if policy:
                policies_set.add(policy)

        elif event_type == "scheduler.fairness_warning":
            result.fairness_warning_count += 1

        elif event_type == "scheduler.kernel_probe":
            result.kernel_probe_count += 1

        elif event_type == "scheduler.bpf_map_updated":
            result.bpf_map_updated_count += 1

        elif event_type == "scheduler.runtime_health":
            result.runtime_health_count += 1

        elif event_type == "scheduler.sla_summary":
            result.sla_summary_count += 1

        elif event_type == "scheduler.risk_warning":
            result.risk_warning_count += 1

        elif event_type == "scheduler.backpressure":
            result.backpressure_count += 1

        elif event_type == "scheduler.safety_violation":
            result.safety_violation_count += 1

        elif event_type == "scheduler.adaptive_tuning":
            result.adaptive_tuning_count += 1

    # Validation rules
    if not has_decision:
        result.errors.append("no scheduler.decision events found")

    if has_kernel_probe_in_decision and result.kernel_probe_count == 0:
        result.errors.append("kernel_probe present in decision but no scheduler.kernel_probe event")

    if has_bpf_map_in_decision and result.bpf_map_updated_count == 0:
        result.errors.append("bpf_map present in decision but no scheduler.bpf_map_updated event")

    if has_runtime_health_in_decision and result.runtime_health_count == 0:
        result.errors.append("runtime_health present in decision but no scheduler.runtime_health event")

    if has_sla_in_decision and result.sla_summary_count == 0:
        result.errors.append("sla present in decision but no scheduler.sla_summary event")

    if has_risk_warnings_in_decision and result.risk_warning_count == 0:
        result.errors.append("risk_warnings present in decision but no scheduler.risk_warning event")

    if has_backpressure_in_decision and result.backpressure_count == 0:
        result.errors.append("backpressure present in decision but no scheduler.backpressure event")

    if has_safety_violations_in_decision and result.safety_violation_count == 0:
        result.errors.append("safety_violations present in decision but no scheduler.safety_violation event")

    if has_adaptive_tuning_in_decision and result.adaptive_tuning_count == 0:
        result.errors.append("adaptive_tuning present in decision but no scheduler.adaptive_tuning event")

    # Set the seen lists
    result.policies_seen = sorted(policies_set)
    result.workload_classes_seen = sorted(workload_classes_set)

    # Generate warnings (non-fatal observations)
    if has_decision and len(policies_set) <= 1:
        result.warnings.append(f"only {len(policies_set)} policy observed across {result.decision_count} decisions")
    if has_decision and result.fairness_warning_count == 0:
        result.warnings.append("no fairness warnings emitted (may indicate low contention)")
    if has_decision and result.risk_warning_count == 0:
        result.warnings.append("no risk warnings emitted")

    return result
