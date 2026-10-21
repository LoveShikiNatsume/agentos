"""Mock BPF Map adapter for scheduler integration.

This module provides a mock BPF Map interface that stores scheduler-side
workload/policy/fairness fields.  On unsupported kernels or missing
permissions, the runtime falls back to this mock adapter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class BPFTaskRecord:
    """A single task record stored in the mock BPF map.

    Schema version ``bpf.task_record.v2`` adds OS-facing fields for
    cgroup quotas, pressure, and backpressure signals while remaining
    fully backward-compatible with v1 consumers.
    """

    task_id: str
    workload_class: str
    policy: str
    fairness_score: float
    starvation_risk: str
    selected: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    # v2 OS-facing fields (nullable — None means "not provided")
    agent_id: str | None = None
    pid: int | None = None
    cpu_weight: int | None = None
    cpu_max_percent: float | None = None
    memory_max_mb: int | None = None
    pids_max: int | None = None
    pressure_level: str = "normal"
    backpressure_signal: str = "none"
    updated_at: float = 0.0
    schema_version: str = "bpf.task_record.v2"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "workload_class": self.workload_class,
            "policy": self.policy,
            "fairness_score": self.fairness_score,
            "starvation_risk": self.starvation_risk,
            "selected": self.selected,
            "metadata": dict(self.metadata),
            "agent_id": self.agent_id,
            "pid": self.pid,
            "cpu_weight": self.cpu_weight,
            "cpu_max_percent": self.cpu_max_percent,
            "memory_max_mb": self.memory_max_mb,
            "pids_max": self.pids_max,
            "pressure_level": self.pressure_level,
            "backpressure_signal": self.backpressure_signal,
            "updated_at": self.updated_at,
            "schema_version": self.schema_version,
        }


class BPFMapAdapter(Protocol):
    """Common boundary implemented by mock and real pinned-map adapters."""

    mode: str
    schema_version: str
    pin_path: Path | None

    def update_task(self, record: BPFTaskRecord) -> None:
        raise NotImplementedError

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def snapshot(self) -> dict[str, dict[str, Any]]:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    @property
    def task_count(self) -> int:
        raise NotImplementedError


class MockBPFMapAdapter:
    """Mock BPF map that stores task records in memory."""

    mode = "mock"
    schema_version = "bpf.task_record.v2"
    pin_path: Path | None = None

    def __init__(self) -> None:
        self._records: dict[str, BPFTaskRecord] = {}

    def update_task(self, record: BPFTaskRecord) -> None:
        """Insert or update a task record."""
        self._records[record.task_id] = record

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        """Get a single task record as a dict, or None."""
        record = self._records.get(task_id)
        if record is None:
            return None
        return record.to_dict()

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a snapshot of all task records."""
        return {task_id: record.to_dict() for task_id, record in self._records.items()}

    def clear(self) -> None:
        """Remove all records."""
        self._records.clear()

    @property
    def task_count(self) -> int:
        """Number of records in the map."""
        return len(self._records)


def task_record_from_decision(
    *,
    task_id: str,
    task_decision: dict[str, Any],
    scheduler_decision: dict[str, Any],
) -> BPFTaskRecord:
    """Build a BPFTaskRecord from scheduler decision data.

    Parameters
    ----------
    task_id:
        The task identifier.
    task_decision:
        The per-task decision dict from ``scheduler.decision.tasks[task_id]``.
    scheduler_decision:
        The top-level scheduler decision dict.

    Returns
    -------
    BPFTaskRecord
        A record suitable for writing to the mock BPF map.
    """
    # selected: compat with both selected_task_ids and old selected
    selected_ids = scheduler_decision.get("selected_task_ids", scheduler_decision.get("selected", []))
    fairness = scheduler_decision.get("fairness", {})

    # resource requirements
    resource_req = task_decision.get("resource_requirements", {})

    # pressure: top-level decision["pressure"] has highest priority,
    # then task-level effective_pressure_level, then legacy task pressure_level
    pressure_summary = scheduler_decision.get("pressure") or {}
    pressure_level = pressure_summary.get(
        "effective_pressure_level",
        task_decision.get(
            "effective_pressure_level",
            task_decision.get("pressure_level", "normal"),
        ),
    )

    # backpressure: task-level first, then top-level decision["backpressure"]["signal"],
    # then legacy scheduler_decision["backpressure_signal"]
    backpressure_summary = scheduler_decision.get("backpressure") or {}
    backpressure = task_decision.get(
        "backpressure_signal",
        backpressure_summary.get(
            "signal",
            scheduler_decision.get("backpressure_signal", "none"),
        ),
    )

    # quota fields from resource_requirements
    cpu_weight = resource_req.get("cpu_shares")
    cpu_max = resource_req.get("cpu_max_percent")
    mem_max = resource_req.get("memory_max_mb")
    pids_max = resource_req.get("pids_max")

    return BPFTaskRecord(
        task_id=task_id,
        workload_class=task_decision.get("workload_class", "unknown"),
        policy=scheduler_decision.get("policy", "latency_first"),
        fairness_score=fairness.get("fairness_score", 1.0),
        starvation_risk=task_decision.get("starvation_risk", "low"),
        selected=task_id in selected_ids,
        metadata={
            "reason": task_decision.get("reason", "unknown"),
            "wait_ms": task_decision.get("wait_ms", 0.0),
            "deferred_count": task_decision.get("deferred_count", 0),
            "blocked_count": task_decision.get("blocked_count", 0),
        },
        agent_id=task_decision.get("agent_id"),
        pid=task_decision.get("pid"),
        cpu_weight=cpu_weight,
        cpu_max_percent=cpu_max,
        memory_max_mb=mem_max,
        pids_max=pids_max,
        pressure_level=pressure_level,
        backpressure_signal=backpressure,
        updated_at=time.time(),
    )
