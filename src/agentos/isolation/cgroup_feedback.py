from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from agentos.isolation.kernel_adapter import PressureSample
from agentos.isolation.pressure import pressure_level_from_psi


_RESOURCES = ("cpu", "memory", "io")


@dataclass(frozen=True)
class CgroupFeedbackSnapshot:
    feedback_id: str
    cpu_some_avg10: float
    memory_some_avg10: float
    io_some_avg10: float
    pressure_level: str
    local_resources: tuple[str, ...]
    sample_count: int
    task_ids: tuple[str, ...]
    agent_ids: tuple[str, ...]
    phases: tuple[str, ...]
    sample_indexes: tuple[int, ...]
    first_timestamp: float
    last_timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "cpu_some_avg10": self.cpu_some_avg10,
            "memory_some_avg10": self.memory_some_avg10,
            "io_some_avg10": self.io_some_avg10,
            "pressure_level": self.pressure_level,
            "pressure_source": "cgroup",
            "local_resources": list(self.local_resources),
            "sample_count": self.sample_count,
            "task_ids": list(self.task_ids),
            "agent_ids": list(self.agent_ids),
            "phases": list(self.phases),
            "sample_indexes": list(self.sample_indexes),
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "consumed": True,
            "reusable": False,
        }


class CgroupFeedbackAccumulator:
    """Accumulate genuine cgroup-local PSI peaks for one scheduler window."""

    def __init__(self) -> None:
        self._sequence = 0
        self.clear()

    @property
    def pending(self) -> bool:
        return self._sample_count > 0

    def observe(
        self,
        sample: PressureSample,
        *,
        agent_id: str,
        task_id: str,
        phase: str,
        sample_index: int,
        timestamp: float | None = None,
    ) -> bool:
        accepted: dict[str, float] = {}
        values = {
            "cpu": sample.cpu_some_avg10,
            "memory": sample.memory_some_avg10,
            "io": sample.io_some_avg10,
        }
        for resource in _RESOURCES:
            if sample.pressure_sources.get(resource) != "cgroup":
                continue
            try:
                value = float(values[resource])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value) or value < 0:
                continue
            accepted[resource] = value

        if not accepted:
            return False

        observed_at = time.time() if timestamp is None else float(timestamp)
        if self._first_timestamp is None:
            self._first_timestamp = observed_at
        self._last_timestamp = observed_at
        self._sample_count += 1
        self._append_unique(self._task_ids, task_id)
        self._append_unique(self._agent_ids, agent_id)
        self._append_unique(self._phases, phase)
        self._append_unique(self._sample_indexes, sample_index)

        for resource, value in accepted.items():
            self._peaks[resource] = max(self._peaks[resource], value)
            self._local_resources.add(resource)
        return True

    def consume(self) -> CgroupFeedbackSnapshot | None:
        if not self.pending:
            return None

        self._sequence += 1
        snapshot = CgroupFeedbackSnapshot(
            feedback_id=f"cgroup-feedback-{self._sequence}",
            cpu_some_avg10=self._peaks["cpu"],
            memory_some_avg10=self._peaks["memory"],
            io_some_avg10=self._peaks["io"],
            pressure_level=pressure_level_from_psi(
                self._peaks["cpu"],
                self._peaks["memory"],
                self._peaks["io"],
            ),
            local_resources=tuple(
                resource
                for resource in _RESOURCES
                if resource in self._local_resources
            ),
            sample_count=self._sample_count,
            task_ids=tuple(self._task_ids),
            agent_ids=tuple(self._agent_ids),
            phases=tuple(self._phases),
            sample_indexes=tuple(self._sample_indexes),
            first_timestamp=self._first_timestamp or 0.0,
            last_timestamp=self._last_timestamp or 0.0,
        )
        self.clear()
        return snapshot

    def clear(self) -> None:
        self._peaks = {resource: 0.0 for resource in _RESOURCES}
        self._local_resources: set[str] = set()
        self._sample_count = 0
        self._task_ids: list[str] = []
        self._agent_ids: list[str] = []
        self._phases: list[str] = []
        self._sample_indexes: list[int] = []
        self._first_timestamp: float | None = None
        self._last_timestamp: float | None = None

    @staticmethod
    def _append_unique(values: list[Any], value: Any) -> None:
        if value not in values:
            values.append(value)
