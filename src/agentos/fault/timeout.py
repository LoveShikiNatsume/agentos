from __future__ import annotations

import asyncio
from typing import Awaitable, TypeVar

from agentos.events.types import (
    EventCategory,
    EventSeverity,
    RuntimeEvent,
)

T = TypeVar("T")


class AgentTimeoutError(TimeoutError):
    def __init__(
        self,
        timeout_seconds: float,
        agent_id: str | None = None,
        task_id: str | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.agent_id = agent_id
        self.task_id = task_id
        target = f"agent={agent_id or '-'} task={task_id or '-'}"
        super().__init__(f"{target} timed out after {timeout_seconds}s")


async def run_with_timeout(
    awaitable: Awaitable[T],
    timeout_seconds: float | None,
    *,
    agent_id: str | None = None,
    task_id: str | None = None,
) -> T:
    if timeout_seconds is None:
        return await awaitable
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise AgentTimeoutError(timeout_seconds, agent_id=agent_id, task_id=task_id) from exc


def fault_timeout(
    *,
    agent_id: str | None,
    task_id: str | None,
    trace_id: str | None,
    timeout_seconds: float,
) -> RuntimeEvent:
    return RuntimeEvent(
        event_type="fault.timeout",
        agent_id=agent_id,
        task_id=task_id,
        trace_id=trace_id,
        category=EventCategory.FAULT,
        severity=EventSeverity.ERROR,
        data={"timeout_seconds": timeout_seconds},
    )
