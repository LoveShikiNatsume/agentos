from __future__ import annotations

import heapq
import itertools
from collections import defaultdict
from typing import Protocol

from agentos.core.types import Message


class MessageTransport(Protocol):
    async def send(self, message: Message) -> None:
        ...

    async def receive(self, agent_id: str) -> Message | None:
        ...

    def queue_depth(self, agent_id: str | None = None) -> int:
        ...


class InMemoryMessageTransport:
    def __init__(self) -> None:
        self._queues: dict[str, list[tuple[int, float, int, Message]]] = defaultdict(
            list
        )
        self._sequence = itertools.count()

    async def send(self, message: Message) -> None:
        priority = int(getattr(message, "priority", 0))
        item = (-priority, message.timestamp, next(self._sequence), message)
        heapq.heappush(self._queues[message.to_agent], item)

    async def receive(self, agent_id: str) -> Message | None:
        queue = self._queues[agent_id]
        if not queue:
            return None
        return heapq.heappop(queue)[3]

    def queue_depth(self, agent_id: str | None = None) -> int:
        if agent_id is not None:
            return len(self._queues.get(agent_id, []))
        return sum(len(queue) for queue in self._queues.values())
