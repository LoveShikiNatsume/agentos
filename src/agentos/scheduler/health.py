"""Runtime Health Score computation.

This module provides health assessment for the Agent Runtime scheduler,
inspired by air traffic control / power grid dispatch / SRE health monitoring.

It does NOT claim to assess Linux kernel health - only user-space Runtime health.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

VALID_PRESSURE_LEVELS = {"normal", "warning", "critical", "overloaded"}


@dataclass
class RuntimeHealthSummary:
    """Runtime health assessment summary."""

    runtime_health_score: float
    queue_pressure: float
    resource_pressure: float
    fairness_pressure: float
    dependency_pressure: float
    pressure_level: str
    health_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_health_score": self.runtime_health_score,
            "queue_pressure": self.queue_pressure,
            "resource_pressure": self.resource_pressure,
            "fairness_pressure": self.fairness_pressure,
            "dependency_pressure": self.dependency_pressure,
            "pressure_level": self.pressure_level,
            "health_warnings": list(self.health_warnings),
        }


def clamp01(value: float) -> float:
    """Clamp value to [0.0, 1.0] range."""
    return max(0.0, min(1.0, value))


def pressure_level_from_score(score: float, max_pressure: float = 0.0) -> str:
    """Determine pressure level from health score and max pressure.

    Parameters
    ----------
    score:
        Runtime health score (0.0-1.0).
    max_pressure:
        Maximum single pressure dimension (0.0-1.0).

    Returns
    -------
    str
        Pressure level: "normal", "warning", "critical", or "overloaded".
    """
    if score >= 0.80:
        level = "normal"
    elif score >= 0.60:
        level = "warning"
    elif score >= 0.35:
        level = "critical"
    else:
        level = "overloaded"

    # Escalate if any single pressure is high
    if max_pressure >= 0.90 and level in {"normal", "warning"}:
        return "critical"
    if max_pressure >= 0.70 and level == "normal":
        return "warning"

    return level


def compute_queue_pressure(
    ready_count: int,
    deferred_count: int,
    blocked_count: int,
    max_batch_size: int | None = None,
) -> float:
    """Compute queue pressure from task counts.

    Parameters
    ----------
    ready_count:
        Number of ready tasks.
    deferred_count:
        Number of deferred tasks.
    blocked_count:
        Number of blocked tasks.
    max_batch_size:
        Maximum batch size (if configured).

    Returns
    -------
    float
        Queue pressure in [0.0, 1.0].
    """
    queue_base = deferred_count + blocked_count
    queue_total = max(ready_count + deferred_count + blocked_count, 1)

    queue_pressure = clamp01(queue_base / queue_total)

    # If ready tasks exceed max_batch_size, increase pressure
    if max_batch_size and ready_count > max_batch_size:
        queue_pressure = max(
            queue_pressure,
            clamp01((ready_count - max_batch_size) / max(ready_count, 1)),
        )

    return queue_pressure


def compute_fairness_pressure(fairness_score: float, starvation_risk: str) -> float:
    """Compute fairness pressure from fairness score and starvation risk.

    Parameters
    ----------
    fairness_score:
        Fairness score (0.0-1.0).
    starvation_risk:
        Starvation risk level: "low", "medium", or "high".

    Returns
    -------
    float
        Fairness pressure in [0.0, 1.0].
    """
    pressure = clamp01(1.0 - fairness_score)

    # Ensure minimum pressure based on starvation risk
    risk_minimums = {
        "low": 0.0,
        "medium": 0.5,
        "high": 0.8,
    }
    min_pressure = risk_minimums.get(starvation_risk, 0.0)
    return max(pressure, min_pressure)


def compute_resource_pressure(resource_blocked_count: int) -> float:
    """Compute resource pressure from blocked task count.

    Parameters
    ----------
    resource_blocked_count:
        Number of tasks blocked due to resource constraints.

    Returns
    -------
    float
        Resource pressure in [0.0, 1.0].
    """
    if resource_blocked_count <= 0:
        return 0.0
    return clamp01(0.7 + 0.1 * (resource_blocked_count - 1))


def compute_dependency_pressure(
    total_incomplete_count: int,
    dependency_blocked_count: int,
) -> float:
    """Compute dependency pressure from incomplete and dependency-blocked counts.

    Parameters
    ----------
    total_incomplete_count:
        Total number of incomplete tasks.
    dependency_blocked_count:
        Number of tasks blocked due to unmet dependencies.

    Returns
    -------
    float
        Dependency pressure in [0.0, 1.0].
    """
    if total_incomplete_count <= 0:
        return 0.0
    return clamp01(dependency_blocked_count / total_incomplete_count)


def compute_runtime_health(
    *,
    ready_count: int,
    candidate_count: int,
    selected_count: int,
    deferred_count: int,
    blocked_count: int,
    total_incomplete_count: int,
    dependency_blocked_count: int,
    fairness_score: float,
    starvation_risk: str,
    resource_blocked_count: int = 0,
    max_batch_size: int | None = None,
) -> RuntimeHealthSummary:
    """Compute runtime health summary.

    Parameters
    ----------
    ready_count:
        Number of ready tasks.
    candidate_count:
        Number of candidate tasks.
    selected_count:
        Number of selected tasks.
    deferred_count:
        Number of deferred tasks.
    blocked_count:
        Number of blocked tasks.
    total_incomplete_count:
        Total number of incomplete tasks.
    dependency_blocked_count:
        Number of tasks blocked due to unmet dependencies.
    fairness_score:
        Fairness score (0.0-1.0).
    starvation_risk:
        Starvation risk level: "low", "medium", or "high".
    resource_blocked_count:
        Number of tasks blocked due to resource constraints.
    max_batch_size:
        Maximum batch size (if configured).

    Returns
    -------
    RuntimeHealthSummary
        The computed health summary.
    """
    # Compute individual pressures
    queue_pressure = compute_queue_pressure(
        ready_count, deferred_count, blocked_count, max_batch_size
    )
    fairness_pressure = compute_fairness_pressure(fairness_score, starvation_risk)
    resource_pressure = compute_resource_pressure(resource_blocked_count)
    dependency_pressure = compute_dependency_pressure(
        total_incomplete_count, dependency_blocked_count
    )

    # Compute overall health score
    runtime_health_score = 1.0 - (
        queue_pressure * 0.25
        + resource_pressure * 0.30
        + fairness_pressure * 0.25
        + dependency_pressure * 0.20
    )
    runtime_health_score = clamp01(runtime_health_score)

    # Determine pressure level
    max_pressure = max(
        queue_pressure, resource_pressure, fairness_pressure, dependency_pressure
    )
    pressure_level = pressure_level_from_score(runtime_health_score, max_pressure)

    # Generate warnings
    warnings: list[str] = []
    if queue_pressure >= 0.7:
        warnings.append("queue pressure is high")
    elif queue_pressure >= 0.5:
        warnings.append("queue pressure is moderate")

    if resource_pressure >= 0.7:
        warnings.append("resource pressure is high")

    if fairness_pressure >= 0.8:
        warnings.append("fairness pressure is critical")
    elif fairness_pressure >= 0.5:
        warnings.append("fairness pressure is high")

    if dependency_pressure >= 0.7:
        warnings.append("dependency pressure is high")
    elif dependency_pressure >= 0.5:
        warnings.append("dependency pressure is moderate")

    return RuntimeHealthSummary(
        runtime_health_score=round(runtime_health_score, 4),
        queue_pressure=round(queue_pressure, 4),
        resource_pressure=round(resource_pressure, 4),
        fairness_pressure=round(fairness_pressure, 4),
        dependency_pressure=round(dependency_pressure, 4),
        pressure_level=pressure_level,
        health_warnings=warnings,
    )
