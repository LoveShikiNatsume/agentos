"""Fairness tracking for scheduler decisions.

This module provides runtime fairness observation metrics, not a Linux CFS
implementation. It tracks per-task wait time, selection/deferral/block counts,
and computes a window-level fairness score for the policy layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Any, Callable


class StarvationRisk(str, Enum):
    """Starvation risk level derived from fairness metrics."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class TaskFairnessState:
    """Per-task fairness state tracked across scheduling rounds."""

    task_id: str
    ready_since: float | None = None
    wait_ms: float = 0.0
    selected_count: int = 0
    deferred_count: int = 0
    blocked_count: int = 0
    last_selected_at: float | None = None
    estimated_service_ms: float = 1.0
    slowdown: float = 1.0
    starvation_risk: StarvationRisk = StarvationRisk.LOW
    expected_wait_batches: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "ready_since": self.ready_since,
            "wait_ms": self.wait_ms,
            "selected_count": self.selected_count,
            "deferred_count": self.deferred_count,
            "blocked_count": self.blocked_count,
            "last_selected_at": self.last_selected_at,
            "estimated_service_ms": self.estimated_service_ms,
            "slowdown": self.slowdown,
            "starvation_risk": self.starvation_risk.value,
            "expected_wait_batches": self.expected_wait_batches,
        }


@dataclass
class FairnessSummary:
    """Window-level fairness summary, including the decision's scale and limits."""

    max_wait_ms: float = 0.0
    avg_wait_ms: float = 0.0
    max_slowdown: float = 1.0
    fairness_score: float = 1.0
    starvation_risk: StarvationRisk = StarvationRisk.LOW
    warnings: list[dict[str, object]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_wait_ms": self.max_wait_ms,
            "avg_wait_ms": self.avg_wait_ms,
            "max_slowdown": self.max_slowdown,
            "fairness_score": self.fairness_score,
            "starvation_risk": self.starvation_risk.value,
            "warnings": list(self.warnings),
            **self.metrics,
        }


class FairnessTracker:
    """Track fairness using absolute limits or observed batch service time.

    ``absolute`` preserves the original wait/deferral thresholds. In
    ``service_time`` mode, a task's first ready observation freezes its queue
    budget ``R = max(1, ceil(ready_count / batch_size))``. All tasks entering
    together receive a full queue-drain budget, because cache-affinity order
    need not match their initial positions. Freezing the budget prevents later
    arrivals from extending it indefinitely, and prevents queue drainage from
    abruptly shortening the remaining tasks' budgets.

    For EWMA batch service time ``S``, relative wait is ``wait_ms / (R * S)``
    and relative deferral is ``deferred_count / R``. Either ratio reaching 1
    requests aging; either reaching 2 is HIGH starvation risk. Before the first
    service sample, the original absolute thresholds apply. Service samples
    update ``S = alpha * duration_ms + (1 - alpha) * S``; the first sample is
    used directly.

    ``now`` returns monotonic seconds and can be injected for deterministic
    tests. ``batch_size=None`` treats the entire observed ready queue as one
    batch. Service-time summaries include only the latest observed ready set;
    absolute summaries retain their historical behavior.
    """

    def __init__(
        self,
        *,
        medium_wait_ms: float = 1000.0,
        high_wait_ms: float = 3000.0,
        medium_deferred_count: int = 2,
        high_deferred_count: int = 4,
        fairness_scale: str = "absolute",
        service_time_alpha: float = 0.25,
        now: Callable[[], float] | None = None,
    ) -> None:
        if fairness_scale not in {"absolute", "service_time"}:
            raise ValueError("fairness_scale must be 'absolute' or 'service_time'")
        if (
            isinstance(service_time_alpha, bool)
            or not isinstance(service_time_alpha, Real)
            or not math.isfinite(service_time_alpha)
            or not 0.0 < service_time_alpha <= 1.0
        ):
            raise ValueError("service_time_alpha must be finite and in (0, 1]")
        self._medium_wait_ms = medium_wait_ms
        self._high_wait_ms = high_wait_ms
        self._medium_deferred_count = medium_deferred_count
        self._high_deferred_count = high_deferred_count
        self._fairness_scale = fairness_scale
        self._service_time_alpha = service_time_alpha
        self._avg_service_time_ms: float | None = None
        self._now = now or _monotonic
        self._states: dict[str, TaskFairnessState] = {}
        self._active_ids: list[str] = []

    @property
    def fairness_scale(self) -> str:
        """Configured fairness scale (the effective scale can be absolute at startup)."""
        return self._fairness_scale

    def observe_ready(
        self, task_ids: list[str], *, batch_size: int | None = None
    ) -> None:
        """Observe ready tasks and freeze each new task's queue-drain budget."""
        if batch_size is not None and (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer or None")
        self._active_ids = list(dict.fromkeys(task_ids))
        effective_batch_size = batch_size or max(1, len(task_ids))
        expected_wait_batches = max(1, math.ceil(len(task_ids) / effective_batch_size))
        now = self._now()
        for task_id in task_ids:
            state = self._get_or_create(task_id)
            if state.ready_since is None:
                state.ready_since = now
                state.expected_wait_batches = expected_wait_batches
            else:
                state.wait_ms = (now - state.ready_since) * 1000.0
            state.slowdown = max(1.0, state.wait_ms / max(state.estimated_service_ms, 0.001))
            state.starvation_risk = self._task_risk(state)

    def record_service_time(self, duration_ms: float) -> None:
        """Observe a finite, positive completed batch duration in milliseconds."""
        if (
            isinstance(duration_ms, bool)
            or not isinstance(duration_ms, Real)
            or not math.isfinite(duration_ms)
            or duration_ms <= 0
        ):
            raise ValueError("duration_ms must be finite and positive")
        if self._avg_service_time_ms is None:
            self._avg_service_time_ms = float(duration_ms)
        else:
            alpha = self._service_time_alpha
            self._avg_service_time_ms = (
                alpha * duration_ms + (1.0 - alpha) * self._avg_service_time_ms
            )
        if self._uses_service_time():
            for state in self._states.values():
                state.starvation_risk = self._task_risk(state)

    def record_decision(
        self,
        *,
        selected_ids: list[str] | None = None,
        deferred_ids: list[str] | None = None,
        blocked_ids: list[str] | None = None,
    ) -> None:
        """Update counters after a scheduling decision."""
        now = self._now()
        for task_id in selected_ids or []:
            state = self._get_or_create(task_id)
            state.selected_count += 1
            state.last_selected_at = now
            state.starvation_risk = self._task_risk(state)
        for task_id in deferred_ids or []:
            state = self._get_or_create(task_id)
            state.deferred_count += 1
            state.starvation_risk = self._task_risk(state)
        for task_id in blocked_ids or []:
            state = self._get_or_create(task_id)
            state.blocked_count += 1
            state.starvation_risk = self._task_risk(state)

    def summarize(self) -> FairnessSummary:
        """Compute fairness using historical absolute or active relative states."""
        states = self._summary_states()
        metrics = self.decision_metrics()
        if not states:
            return FairnessSummary(metrics=metrics)

        waits = [s.wait_ms for s in states if s.wait_ms > 0]
        slowdowns = [s.slowdown for s in states]
        deferred_counts = [s.deferred_count for s in states]

        max_wait = max(waits) if waits else 0.0
        avg_wait = sum(waits) / len(waits) if waits else 0.0
        max_slowdown = max(slowdowns) if slowdowns else 1.0
        max_deferred = max(deferred_counts) if deferred_counts else 0

        # Absolute mode keeps the original scoring and risk algorithm.
        if self._uses_service_time():
            max_relative_wait = max(self._relative_values(s)[0] for s in states)
            max_relative_deferred = max(self._relative_values(s)[1] for s in states)
            wait_penalty = min(1.0, max_relative_wait / 2.0)
            deferred_penalty = min(1.0, max_relative_deferred / 2.0)
            starvation_risk = self._relative_risk(max_relative_wait, max_relative_deferred)
        else:
            wait_penalty = min(1.0, max_wait / max(self._high_wait_ms, 1.0))
            deferred_penalty = min(1.0, max_deferred / max(self._high_deferred_count, 1))
            starvation_risk = self._window_risk(max_wait, max_deferred)
        fairness_score = max(0.0, 1.0 - 0.5 * wait_penalty - 0.5 * deferred_penalty)

        return FairnessSummary(
            max_wait_ms=max_wait,
            avg_wait_ms=avg_wait,
            max_slowdown=max_slowdown,
            fairness_score=round(fairness_score, 4),
            starvation_risk=starvation_risk,
            warnings=self._collect_warnings(states),
            metrics=metrics,
        )

    def task_snapshot(self, task_id: str) -> dict[str, Any]:
        """Return task counters and the exact thresholds used for this task.

        ``aging_due`` is enabled only when service-time measurements exist,
        and starts at MEDIUM risk so affinity ordering can intervene before
        the HIGH limit forces a fairness policy.
        """
        state = self._states.get(task_id) or TaskFairnessState(task_id=task_id)
        return {**state.to_dict(), **self._task_metrics(state)}

    def decision_metrics(self) -> dict[str, Any]:
        """Return serializable scale, relative values, and per-task thresholds.

        The same fields are included in ``summarize().to_dict()`` so a policy
        event can retain the measurements observed before its decision.
        """
        states = self._summary_states()
        uses_service_time = self._uses_service_time()
        task_metrics = {state.task_id: self._task_metrics(state) for state in states}
        return {
            "fairness_scale": self._fairness_scale,
            "effective_scale": "service_time" if uses_service_time else "absolute",
            "avg_service_time_ms": self._avg_service_time_ms,
            "max_relative_wait": (
                max((self._relative_values(s)[0] for s in states), default=0.0)
                if uses_service_time else None
            ),
            "max_relative_deferred": (
                max((self._relative_values(s)[1] for s in states), default=0.0)
                if uses_service_time else None
            ),
            "medium_relative_wait": 1.0 if uses_service_time else None,
            "high_relative_wait": 2.0 if uses_service_time else None,
            "medium_relative_deferred": 1.0 if uses_service_time else None,
            "high_relative_deferred": 2.0 if uses_service_time else None,
            "task_metrics": task_metrics,
        }

    def _get_or_create(self, task_id: str) -> TaskFairnessState:
        if task_id not in self._states:
            self._states[task_id] = TaskFairnessState(task_id=task_id)
        return self._states[task_id]

    def _uses_service_time(self) -> bool:
        return self._fairness_scale == "service_time" and self._avg_service_time_ms is not None

    def _summary_states(self) -> list[TaskFairnessState]:
        if self._fairness_scale == "service_time":
            return [self._states[task_id] for task_id in self._active_ids]
        return list(self._states.values())

    def _relative_values(self, state: TaskFairnessState) -> tuple[float, float]:
        assert self._avg_service_time_ms is not None
        budget = state.expected_wait_batches
        return (
            state.wait_ms / (budget * self._avg_service_time_ms),
            state.deferred_count / budget,
        )

    def _task_metrics(self, state: TaskFairnessState) -> dict[str, Any]:
        uses_service_time = self._uses_service_time()
        relative_wait, relative_deferred = (
            self._relative_values(state) if uses_service_time else (None, None)
        )
        wait_budget = state.expected_wait_batches * (self._avg_service_time_ms or 0.0)
        return {
            "fairness_scale": self._fairness_scale,
            "effective_scale": "service_time" if uses_service_time else "absolute",
            "avg_service_time_ms": self._avg_service_time_ms,
            "expected_wait_batches": state.expected_wait_batches,
            "wait_ms": state.wait_ms,
            "deferred_count": state.deferred_count,
            "relative_wait": relative_wait,
            "relative_deferred": relative_deferred,
            "medium_wait_ms": wait_budget if uses_service_time else self._medium_wait_ms,
            "high_wait_ms": 2.0 * wait_budget if uses_service_time else self._high_wait_ms,
            "medium_deferred_count": (
                state.expected_wait_batches if uses_service_time else self._medium_deferred_count
            ),
            "high_deferred_count": (
                2 * state.expected_wait_batches if uses_service_time else self._high_deferred_count
            ),
            "aging_due": uses_service_time and self._task_risk(state) != StarvationRisk.LOW,
            "starvation_risk": self._task_risk(state).value,
        }

    def _task_risk(self, state: TaskFairnessState) -> StarvationRisk:
        """Determine starvation risk for a single task."""
        if self._uses_service_time():
            return self._relative_risk(*self._relative_values(state))
        return self._window_risk(state.wait_ms, state.deferred_count)

    @staticmethod
    def _relative_risk(relative_wait: float, relative_deferred: float) -> StarvationRisk:
        if relative_wait >= 2.0 or relative_deferred >= 2.0:
            return StarvationRisk.HIGH
        if relative_wait >= 1.0 or relative_deferred >= 1.0:
            return StarvationRisk.MEDIUM
        return StarvationRisk.LOW

    def _window_risk(self, max_wait: float, max_deferred: int) -> StarvationRisk:
        """Determine absolute window-level starvation risk."""
        if max_wait >= self._high_wait_ms or max_deferred >= self._high_deferred_count:
            return StarvationRisk.HIGH
        if max_wait >= self._medium_wait_ms or max_deferred >= self._medium_deferred_count:
            return StarvationRisk.MEDIUM
        return StarvationRisk.LOW

    def _collect_warnings(self, states: list[TaskFairnessState]) -> list[dict[str, object]]:
        """Collect warnings for the states included in this summary."""
        warnings: list[dict[str, object]] = []
        for state in states:
            risk = self._task_risk(state)
            if risk in {StarvationRisk.MEDIUM, StarvationRisk.HIGH}:
                warning: dict[str, object] = {
                    "task_id": state.task_id,
                    "wait_ms": state.wait_ms,
                    "deferred_count": state.deferred_count,
                    "blocked_count": state.blocked_count,
                    "starvation_risk": risk.value,
                    "suggested_policy": "fairness_balanced" if risk == StarvationRisk.HIGH else None,
                }
                if self._fairness_scale == "service_time":
                    warning.update(self._task_metrics(state))
                warnings.append(warning)
        return warnings


def _monotonic() -> float:
    """Wrapper around time.monotonic for easy patching."""
    import time
    return time.monotonic()
