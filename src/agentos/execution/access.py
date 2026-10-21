"""In-process workspace coordination; explicitly not an OS security boundary."""
import asyncio
from contextlib import asynccontextmanager


class WorkspaceAccess:
    def __init__(self):
        self.condition = asyncio.Condition()
        self.readers = 0
        self.writer = False
        self.waiting_writers = 0

    @asynccontextmanager
    async def hold(self, read_only):
        async with self.condition:
            if read_only:
                await self.condition.wait_for(lambda: not self.writer and not self.waiting_writers)
                self.readers += 1
            else:
                self.waiting_writers += 1
                try:
                    await self.condition.wait_for(lambda: not self.writer and self.readers == 0)
                    self.writer = True
                finally:
                    self.waiting_writers -= 1
                    self.condition.notify_all()
        try:
            yield
        finally:
            async with self.condition:
                if read_only:
                    self.readers -= 1
                else:
                    self.writer = False
                self.condition.notify_all()
