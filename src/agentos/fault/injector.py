from __future__ import annotations

from enum import Enum
from collections.abc import Mapping
from typing import Any

from agentos.events.types import EventCategory, EventSeverity, RuntimeEvent


class FaultType(Enum):
    TIMEOUT = "timeout"
    KILL = "kill"
    MEMORY_PRESSURE = "memory_pressure"


_SUPPORTED_FAULT_TYPES = {fault_type.value for fault_type in FaultType}


def inject_fault(
    fault_type: FaultType | str,
    *,
    task_id: str,
    agent_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> RuntimeEvent:
    normalized = fault_type.value if isinstance(fault_type, FaultType) else fault_type
    if normalized not in _SUPPORTED_FAULT_TYPES:
        raise ValueError(f"unsupported fault type: {normalized}")
    return RuntimeEvent(
        event_type="fault.injected",
        agent_id=agent_id,
        task_id=task_id,
        category=EventCategory.FAULT,
        severity=EventSeverity.WARNING,
        data={**(metadata or {}), "fault_type": normalized},
    )


class FaultInjectionPlan:
    """One-shot fault injection plan keyed by task_id."""

    def __init__(
        self,
        faults: dict[str, FaultType | str | Mapping[str, Any]] | None = None,
    ) -> None:
        self._faults = dict(faults or {})

    def pop_for_task(
        self,
        task_id: str,
        *,
        agent_id: str | None = None,
    ) -> RuntimeEvent | None:
        fault_type = self._faults.pop(task_id, None)
        if fault_type is None:
            return None
        metadata: dict[str, Any] | None = None
        if isinstance(fault_type, Mapping):
            metadata = dict(fault_type)
            raw_type = metadata.pop("fault_type", metadata.pop("type", None))
            if raw_type is None:
                raise ValueError("fault plan mapping requires 'fault_type' or 'type'")
            fault_type = raw_type
        return inject_fault(
            fault_type,
            task_id=task_id,
            agent_id=agent_id,
            metadata=metadata,
        )
