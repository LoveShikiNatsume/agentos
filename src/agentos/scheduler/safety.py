"""Backpressure and Safety Envelope for Agent Runtime.

This module provides admission control, backpressure signaling, and safety
envelope enforcement for the scheduler. It is inspired by:
- Power grid dispatch (peak shaving)
- SRE overload protection (rate limiting, degradation, circuit breaking)
- Autonomous driving safety envelope

P6 is a user-space Runtime safety layer, NOT a Linux kernel sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AdmissionDecision(str, Enum):
    """Admission decision for a task."""

    ALLOW = "allow"
    DEFER = "defer"
    DEGRADE = "degrade"
    REJECT = "reject"


class BackpressureSignal(str, Enum):
    """Backpressure signal level."""

    NONE = "none"
    SOFT = "soft"
    HARD = "hard"
    REJECTING = "rejecting"


# Valid admission decisions
VALID_ADMISSION_DECISIONS = {d.value for d in AdmissionDecision}

# Valid backpressure signals
VALID_BACKPRESSURE_SIGNALS = {s.value for s in BackpressureSignal}


@dataclass
class SafetyEnvelopeConfig:
    """Configuration for safety envelope rules."""

    max_task_memory_mb: int | None = 4096
    max_task_cpu_percent: int | None = 95
    max_task_failure_count: int = 3
    max_task_risk_score: float = 0.85
    reject_best_effort_when_overloaded: bool = True
    defer_batch_when_critical: bool = True


@dataclass
class TaskSafetyDecision:
    """Safety decision for a task."""

    admission_decision: str
    backpressure_signal: str
    degrade_reason: str | None = None
    safety_violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "admission_decision": self.admission_decision,
            "backpressure_signal": self.backpressure_signal,
            "degrade_reason": self.degrade_reason,
            "safety_violations": list(self.safety_violations),
        }


def backpressure_signal_from_pressure(pressure_level: str | None) -> BackpressureSignal:
    """Map pressure level to backpressure signal.

    Parameters
    ----------
    pressure_level:
        Pressure level string (normal, warning, critical, overloaded).

    Returns
    -------
    BackpressureSignal
        Corresponding backpressure signal.
    """
    if pressure_level == "warning":
        return BackpressureSignal.SOFT
    if pressure_level == "critical":
        return BackpressureSignal.HARD
    if pressure_level == "overloaded":
        return BackpressureSignal.REJECTING
    return BackpressureSignal.NONE


def evaluate_task_safety(
    *,
    sla_class: str,
    workload_class: str,
    pressure_level: str,
    task_risk_score: float,
    previous_failures: int,
    memory_max_mb: int | None,
    cpu_max_percent: int | None,
    policy: str | None = None,
    config: SafetyEnvelopeConfig | None = None,
) -> TaskSafetyDecision:
    """Evaluate task safety and admission decision.

    Parameters
    ----------
    sla_class:
        SLA class string (critical, interactive, batch, best_effort).
    workload_class:
        Workload class string.
    pressure_level:
        Current pressure level (normal, warning, critical, overloaded).
    task_risk_score:
        Task risk score (0.0-1.0).
    previous_failures:
        Number of previous failures.
    memory_max_mb:
        Task memory requirement in MB.
    cpu_max_percent:
        Task CPU requirement in percent.
    policy:
        Current scheduling policy.
    config:
        Safety envelope configuration.

    Returns
    -------
    TaskSafetyDecision
        The safety decision for the task.
    """
    config = config or SafetyEnvelopeConfig()
    signal = backpressure_signal_from_pressure(pressure_level)
    violations: list[str] = []

    # Check resource limits
    if (
        config.max_task_memory_mb is not None
        and memory_max_mb is not None
        and memory_max_mb > config.max_task_memory_mb
    ):
        violations.append("memory limit exceeded")

    if (
        config.max_task_cpu_percent is not None
        and cpu_max_percent is not None
        and cpu_max_percent > config.max_task_cpu_percent
    ):
        violations.append("cpu limit exceeded")

    if previous_failures >= config.max_task_failure_count:
        violations.append("failure limit exceeded")

    # Reject if safety violations
    if violations:
        return TaskSafetyDecision(
            admission_decision=AdmissionDecision.REJECT.value,
            backpressure_signal=signal.value,
            degrade_reason="safety envelope violation",
            safety_violations=violations,
        )

    # Reject best_effort when overloaded
    if (
        pressure_level == "overloaded"
        and sla_class == "best_effort"
        and config.reject_best_effort_when_overloaded
    ):
        return TaskSafetyDecision(
            admission_decision=AdmissionDecision.REJECT.value,
            backpressure_signal=signal.value,
            degrade_reason="best_effort rejected under overloaded pressure",
        )

    # Defer batch when critical/overloaded
    if (
        pressure_level in {"critical", "overloaded"}
        and sla_class == "batch"
        and config.defer_batch_when_critical
    ):
        return TaskSafetyDecision(
            admission_decision=AdmissionDecision.DEFER.value,
            backpressure_signal=signal.value,
            degrade_reason="batch task deferred under critical pressure",
        )

    # Defer high risk tasks under pressure
    if (
        pressure_level in {"critical", "overloaded"}
        and task_risk_score >= config.max_task_risk_score
    ):
        return TaskSafetyDecision(
            admission_decision=AdmissionDecision.DEFER.value,
            backpressure_signal=signal.value,
            degrade_reason="high risk task deferred under pressure",
        )

    # Degrade unknown workload under pressure
    if (
        workload_class == "unknown"
        and policy in {"throughput_first", "risk_aware"}
        and signal in {BackpressureSignal.HARD, BackpressureSignal.REJECTING}
    ):
        return TaskSafetyDecision(
            admission_decision=AdmissionDecision.DEGRADE.value,
            backpressure_signal=signal.value,
            degrade_reason="unknown workload degraded under pressure",
        )

    # Allow by default
    return TaskSafetyDecision(
        admission_decision=AdmissionDecision.ALLOW.value,
        backpressure_signal=signal.value,
    )


def summarize_safety(decisions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarize safety decisions for all tasks.

    Parameters
    ----------
    decisions:
        Dict of task_id -> task decision dict.

    Returns
    -------
    dict
        Safety summary with counts and lists.
    """
    counts = {"allow": 0, "defer": 0, "degrade": 0, "reject": 0}
    violations: list[str] = []
    rejected_task_ids: list[str] = []
    deferred_task_ids: list[str] = []
    degraded_task_ids: list[str] = []

    for task_id, decision in decisions.items():
        admission = decision.get("admission_decision", "allow")
        counts[admission] = counts.get(admission, 0) + 1

        if admission == "reject":
            rejected_task_ids.append(task_id)
        elif admission == "defer":
            deferred_task_ids.append(task_id)
        elif admission == "degrade":
            degraded_task_ids.append(task_id)

        for violation in decision.get("safety_violations", []):
            if violation not in violations:
                violations.append(violation)

    return {
        "admission_counts": counts,
        "rejected_task_ids": rejected_task_ids,
        "safety_deferred_task_ids": deferred_task_ids,
        "degraded_task_ids": degraded_task_ids,
        "safety_violations": violations,
    }
