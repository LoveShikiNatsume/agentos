from __future__ import annotations

from agentos.context.cache_tracker import CacheHeatTracker
from agentos.context.manager import ContextManager
from agentos.core.task import Task


class CacheAffinityAdvisor:
    """Advises the scheduler on task ordering based on KV cache heat.

    Given a list of ready tasks, resolves each task's prefix_id via
    ContextManager and queries CacheHeatTracker for heat scores.
    Returns tasks sorted so that cache-hot prefixes run first, and
    tasks sharing the same prefix are grouped together.
    """

    def __init__(
        self,
        context_manager: ContextManager,
        cache_tracker: CacheHeatTracker,
    ) -> None:
        self._ctx = context_manager
        self._tracker = cache_tracker

    def resolve_task_prefix(self, task: Task) -> str:
        refs = task.context_refs if task.context_refs else None
        return self._ctx.resolve_prefix_id(refs)

    def get_task_prefix_heat(self, task: Task) -> float:
        return self._tracker.get_prefix_heat(self.resolve_task_prefix(task))

    def sort_by_affinity(self, tasks: list[Task]) -> list[Task]:
        """Sort tasks so cache-hot prefixes come first, same-prefix grouped."""
        if not tasks:
            return []

        prefix_map: dict[str, str] = {}
        for task in tasks:
            prefix_map[task.task_id] = self.resolve_task_prefix(task)

        def sort_key(task: Task) -> tuple[float, str, float, int, float, str]:
            pid = prefix_map[task.task_id]
            heat = self._tracker.get_prefix_heat(pid)
            return (
                -heat,                      # hottest first
                pid,                         # group same prefix together
                -task.critical_path_score,   # within group: critical path first
                -task.priority,
                task.created_at,
                task.task_id,
            )

        return sorted(tasks, key=sort_key)
