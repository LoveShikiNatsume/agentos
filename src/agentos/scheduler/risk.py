"""Risk Assessment and SLA-aware scheduling for Agent Runtime.

This module provides risk scoring and SLA-aware scheduling capabilities,
inspired by hospital triage, financial risk control, and SRE SLA management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SLAClass(str, Enum):
    """SLA class for tasks."""

    CRITICAL = "critical"
    INTERACTIVE = "interactive"
    BATCH = "batch"
    BEST_EFFORT = "best_effort"


# SLA priority mapping (higher = more important)
SLA_PRIORITY = {
    SLAClass.CRITICAL: 4,
    SLAClass.INTERACTIVE: 3,
    SLAClass.BATCH: 2,
    SLAClass.BEST_EFFORT: 1,
}

# Valid SLA classes
VALID_SLA_CLASSES = {cls.value for cls in SLAClass}


@dataclass
class TaskRiskSummary:
    """Risk summary for a task."""

    sla_class: str
    deadline_ms: int | None
    urgency_score: float
    task_risk_score: float
    sla_risk: float
    resource_risk: float
    timeout_risk: float
    failure_risk: float
    starvation_risk_score: float
    dependency_risk: float
    risk_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sla_class": self.sla_class,
            "deadline_ms": self.deadline_ms,
            "urgency_score": self.urgency_score,
            "task_risk_score": self.task_risk_score,
            "sla_risk": self.sla_risk,
            "resource_risk": self.resource_risk,
            "timeout_risk": self.timeout_risk,
            "failure_risk": self.failure_risk,
            "starvation_risk_score": self.starvation_risk_score,
            "dependency_risk": self.dependency_risk,
            "risk_reasons": list(self.risk_reasons),
        }


def clamp01(value: float) -> float:
    """Clamp value to [0.0, 1.0] range."""
    return max(0.0, min(1.0, value))


def normalize_sla_class(value: str | None) -> SLAClass:
    """Normalize SLA class string to enum.

    Parameters
    ----------
    value:
        SLA class string or None.

    Returns
    -------
    SLAClass
        Normalized SLA class (defaults to BATCH).
    """
    if value is None:
        return SLAClass.BATCH
    try:
        return SLAClass(value)
    except ValueError:
        return SLAClass.BATCH


def compute_urgency_score(
    *,
    sla_class: str | None,
    wait_ms: float,
    deadline_ms: int | None,
) -> float:
    """Compute urgency score for a task.

    Parameters
    ----------
    sla_class:
        SLA class string.
    wait_ms:
        Time task has been waiting in milliseconds.
    deadline_ms:
        Task deadline in milliseconds (None = no deadline).

    Returns
    -------
    float
        Urgency score in [0.0, 1.0].
    """
    sla = normalize_sla_class(sla_class)

    base = {
        SLAClass.CRITICAL: 0.9,
        SLAClass.INTERACTIVE: 0.7,
        SLAClass.BATCH: 0.4,
        SLAClass.BEST_EFFORT: 0.2,
    }[sla]

    wait_boost = min(0.3, max(0.0, wait_ms) / 10_000.0)

    deadline_boost = 0.0
    if deadline_ms is not None:
        if deadline_ms <= 500:
            deadline_boost = 0.3
        elif deadline_ms <= 2_000:
            deadline_boost = 0.2
        elif deadline_ms <= 10_000:
            deadline_boost = 0.1

    return round(clamp01(base + wait_boost + deadline_boost), 4)


def starvation_risk_to_score(value: str | None) -> float:
    """Convert starvation risk string to score.

    Parameters
    ----------
    value:
        Starvation risk string ("low", "medium", "high").

    Returns
    -------
    float
        Starvation risk score.
    """
    if value == "high":
        return 0.9
    if value == "medium":
        return 0.5
    return 0.1


def compute_dependency_risk(
    *,
    dependency_count: int,
    unmet_dependency_count: int,
) -> float:
    """Compute dependency risk.

    Parameters
    ----------
    dependency_count:
        Total number of dependencies.
    unmet_dependency_count:
        Number of unmet dependencies.

    Returns
    -------
    float
        Dependency risk in [0.0, 1.0].
    """
    if dependency_count <= 0:
        return 0.0
    return round(clamp01(unmet_dependency_count / dependency_count), 4)


def compute_resource_risk(
    *,
    blocked_resource: str | None,
    required_value: float | None,
    available_value: float | None,
) -> float:
    """Compute resource risk.

    Parameters
    ----------
    blocked_resource:
        Resource that is blocking the task (None = not blocked).
    required_value:
        Required resource value.
    available_value:
        Available resource value.

    Returns
    -------
    float
        Resource risk in [0.0, 1.0].
    """
    if blocked_resource is not None:
        return 0.9
    if required_value is None or available_value in (None, 0):
        return 0.0
    return round(clamp01(required_value / available_value), 4)


def compute_timeout_risk(
    *,
    deadline_ms: int | None,
    wait_ms: float,
) -> float:
    """Compute timeout risk.

    Parameters
    ----------
    deadline_ms:
        Task deadline in milliseconds (None = no deadline).
    wait_ms:
        Time task has been waiting in milliseconds.

    Returns
    -------
    float
        Timeout risk in [0.0, 1.0].
    """
    if deadline_ms is None:
        return 0.0
    if deadline_ms <= 0:
        return 1.0
    return round(clamp01(wait_ms / deadline_ms), 4)


def compute_failure_risk(
    *,
    previous_failures: int = 0,
) -> float:
    """Compute failure risk based on previous failures.

    Parameters
    ----------
    previous_failures:
        Number of previous failures.

    Returns
    -------
    float
        Failure risk in [0.0, 1.0].
    """
    if previous_failures <= 0:
        return 0.0
    return round(clamp01(previous_failures / 3.0), 4)


def compute_sla_risk(sla_class: str | None) -> float:
    """Compute SLA risk based on SLA class.

    Parameters
    ----------
    sla_class:
        SLA class string.

    Returns
    -------
    float
        SLA risk in [0.0, 1.0].
    """
    normalized = normalize_sla_class(sla_class)
    sla_risk_map = {
        SLAClass.CRITICAL: 0.8,
        SLAClass.INTERACTIVE: 0.5,
        SLAClass.BATCH: 0.2,
        SLAClass.BEST_EFFORT: 0.1,
    }
    return sla_risk_map.get(normalized, 0.2)


def compute_task_risk(
    *,
    sla_class: str | None,
    deadline_ms: int | None,
    wait_ms: float,
    starvation_risk: str | None,
    dependency_count: int,
    unmet_dependency_count: int,
    blocked_resource: str | None = None,
    required_value: float | None = None,
    available_value: float | None = None,
    previous_failures: int = 0,
) -> TaskRiskSummary:
    """Compute task risk summary.

    Parameters
    ----------
    sla_class:
        SLA class string.
    deadline_ms:
        Task deadline in milliseconds.
    wait_ms:
        Time task has been waiting in milliseconds.
    starvation_risk:
        Starvation risk string.
    dependency_count:
        Total number of dependencies.
    unmet_dependency_count:
        Number of unmet dependencies.
    blocked_resource:
        Resource that is blocking the task.
    required_value:
        Required resource value.
    available_value:
        Available resource value.
    previous_failures:
        Number of previous failures.

    Returns
    -------
    TaskRiskSummary
        Computed task risk summary.
    """
    normalized_sla = normalize_sla_class(sla_class)

    urgency_score = compute_urgency_score(
        sla_class=normalized_sla.value,
        wait_ms=wait_ms,
        deadline_ms=deadline_ms,
    )
    sla_risk = compute_sla_risk(normalized_sla.value)
    resource_risk = compute_resource_risk(
        blocked_resource=blocked_resource,
        required_value=required_value,
        available_value=available_value,
    )
    timeout_risk = compute_timeout_risk(
        deadline_ms=deadline_ms,
        wait_ms=wait_ms,
    )
    failure_risk = compute_failure_risk(previous_failures=previous_failures)
    starvation_score = starvation_risk_to_score(starvation_risk)
    dependency_risk = compute_dependency_risk(
        dependency_count=dependency_count,
        unmet_dependency_count=unmet_dependency_count,
    )

    # P5.1: Adjusted task_risk_score formula with sla_risk and urgency_score
    task_risk_score = round(clamp01(
        sla_risk * 0.20
        + urgency_score * 0.20
        + resource_risk * 0.20
        + timeout_risk * 0.15
        + failure_risk * 0.10
        + starvation_score * 0.10
        + dependency_risk * 0.05
    ), 4)

    reasons: list[str] = []
    if normalized_sla == SLAClass.CRITICAL:
        reasons.append("critical SLA")
    if urgency_score >= 0.8:
        reasons.append("high urgency")
    if deadline_ms is not None and timeout_risk >= 0.7:
        reasons.append("deadline risk")
    if deadline_ms is not None and urgency_score >= 0.7 and timeout_risk < 0.7:
        reasons.append("deadline approaching")
    if resource_risk >= 0.7:
        reasons.append("resource risk")
    if failure_risk >= 0.7:
        reasons.append("failure risk")
    if starvation_score >= 0.8:
        reasons.append("starvation risk")
    if dependency_risk >= 0.7:
        reasons.append("dependency risk")

    return TaskRiskSummary(
        sla_class=normalized_sla.value,
        deadline_ms=deadline_ms,
        urgency_score=urgency_score,
        task_risk_score=task_risk_score,
        sla_risk=sla_risk,
        resource_risk=resource_risk,
        timeout_risk=timeout_risk,
        failure_risk=failure_risk,
        starvation_risk_score=starvation_score,
        dependency_risk=dependency_risk,
        risk_reasons=reasons,
    )
