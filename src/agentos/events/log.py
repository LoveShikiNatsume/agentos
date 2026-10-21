from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from agentos.events.types import EventCategory, RuntimeEvent


class EventLog:
    """Append-only JSONL event log for replaying RuntimeEvent traces."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, event: RuntimeEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), ensure_ascii=False))
            f.write("\n")

    def read(
        self,
        *,
        agent_id: str | None = None,
        task_id: str | None = None,
        event_type: str | None = None,
        category: EventCategory | str | None = None,
        skip_bad_lines: bool = True,
    ) -> Iterator[RuntimeEvent]:
        if not self.path.exists():
            return

        category_value = category.value if isinstance(category, EventCategory) else category
        with self.path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    event = RuntimeEvent.from_dict(json.loads(raw))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    if skip_bad_lines:
                        continue
                    raise ValueError(
                        f"Invalid event log line {line_number} in {self.path}"
                    ) from exc

                if agent_id is not None and event.agent_id != agent_id:
                    continue
                if task_id is not None and event.task_id != task_id:
                    continue
                if event_type is not None and event.event_type != event_type:
                    continue
                if category_value is not None and event.category.value != category_value:
                    continue
                yield event
