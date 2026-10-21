from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventCategory(Enum):
    SYSTEM = "system"
    AGENT = "agent"
    TASK = "task"
    SCHEDULER = "scheduler"
    CONTEXT = "context"
    LLM = "llm"
    TOOL = "tool"
    FAULT = "fault"
    RESOURCE = "resource"


class EventSeverity(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class RuntimeEvent:
    event_type: str
    agent_id: str | None = None
    task_id: str | None = None
    trace_id: str | None = None
    category: EventCategory = EventCategory.SYSTEM
    severity: EventSeverity = EventSeverity.INFO
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    monotonic_ns: int = field(default_factory=time.monotonic_ns)

    def _json_safe_data(self) -> dict[str, Any]:
        try:
            return json.loads(json.dumps(self.data))
        except TypeError as exc:
            raise TypeError("RuntimeEvent data must be JSON serializable") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "trace_id": self.trace_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "data": self._json_safe_data(),
            "timestamp": self.timestamp,
            "monotonic_ns": self.monotonic_ns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeEvent":
        return cls(
            event_type=data["event_type"],
            agent_id=data.get("agent_id"),
            task_id=data.get("task_id"),
            trace_id=data.get("trace_id"),
            category=EventCategory(data.get("category", EventCategory.SYSTEM.value)),
            severity=EventSeverity(data.get("severity", EventSeverity.INFO.value)),
            data=dict(data.get("data", {})),
            timestamp=float(data.get("timestamp", time.time())),
            monotonic_ns=int(data.get("monotonic_ns", time.monotonic_ns())),
        )


def agent_state_changed(
    agent_id: str,
    old_state: str,
    new_state: str,
    task_id: str | None = None,
    trace_id: str | None = None,
) -> RuntimeEvent:
    return RuntimeEvent(
        event_type="agent.state_changed",
        agent_id=agent_id,
        task_id=task_id,
        trace_id=trace_id,
        category=EventCategory.AGENT,
        data={"old_state": old_state, "new_state": new_state},
    )


def task_completed(
    task_id: str,
    agent_id: str,
    result: Any,
    trace_id: str | None = None,
) -> RuntimeEvent:
    return RuntimeEvent(
        event_type="task.completed",
        agent_id=agent_id,
        task_id=task_id,
        trace_id=trace_id,
        category=EventCategory.TASK,
        data={"result_preview": str(result)[:200]},
    )


def llm_request_sent(
    agent_id: str,
    model: str,
    token_count: int,
    task_id: str | None = None,
    trace_id: str | None = None,
) -> RuntimeEvent:
    return RuntimeEvent(
        event_type="llm.request_sent",
        agent_id=agent_id,
        task_id=task_id,
        trace_id=trace_id,
        category=EventCategory.LLM,
        data={"model": model, "token_count": token_count},
    )


def llm_response_received(
    agent_id: str,
    model: str,
    tokens_used: int,
    latency: float,
    task_id: str | None = None,
    trace_id: str | None = None,
) -> RuntimeEvent:
    return RuntimeEvent(
        event_type="llm.response_received",
        agent_id=agent_id,
        task_id=task_id,
        trace_id=trace_id,
        category=EventCategory.LLM,
        data={"model": model, "tokens_used": tokens_used, "latency_secs": latency},
    )
