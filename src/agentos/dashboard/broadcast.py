from __future__ import annotations

import asyncio
from typing import Any

from agentos.events.types import RuntimeEvent


class EventBroadcaster:
    """Fan-out RuntimeEvents to SSE subscriber queues."""

    def __init__(self, maxsize: int = 10000) -> None:
        self._subscribers: list[asyncio.Queue[RuntimeEvent]] = []
        self._history: list[RuntimeEvent] = []
        self._maxsize = maxsize

    def subscribe(self) -> asyncio.Queue[RuntimeEvent]:
        queue: asyncio.Queue[RuntimeEvent] = asyncio.Queue(maxsize=self._maxsize)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[RuntimeEvent]) -> None:
        try:
            self._subscribers.remove(queue)
        except ValueError:
            pass

    def publish(self, event: RuntimeEvent) -> None:
        self._history.append(event)
        for queue in self._subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def snapshot(self) -> list[RuntimeEvent]:
        return list(self._history)

    def subscriber_count(self) -> int:
        return len(self._subscribers)
