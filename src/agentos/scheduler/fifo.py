from __future__ import annotations

import asyncio

from agentos.core.task import Task


class FIFOScheduler:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[Task] = asyncio.Queue()

    async def submit(self, task: Task) -> None:
        await self._queue.put(task)

    async def next(self) -> Task | None:
        if self._queue.empty():
            return None
        return self._queue.get_nowait()

