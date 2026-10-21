from __future__ import annotations

from typing import Protocol

from agentos.core.task import Task


class Scheduler(Protocol):
    async def submit(self, task: Task) -> None: ...
    async def next(self) -> Task | None: ...

