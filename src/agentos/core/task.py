from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agentos.core.resources import ResourceQuota


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class Task:
    task_id: str
    name: str
    agent_role: str
    dependencies: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent_id: str | None = None
    priority: int = 0
    resource_requirements: ResourceQuota = field(default_factory=ResourceQuota)
    # Empty keeps all shared context for compatibility; nonempty selects named refs.
    context_refs: list[str] = field(default_factory=list)
    workload_hint: str | None = None
    io_profile: str | None = None
    expected_parallelism: int | None = None
    critical_path_score: float = 0.0
    result: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    # P5: SLA and Risk fields
    sla_class: str | None = None
    deadline_ms: int | None = None
    previous_failures: int = 0

    def __post_init__(self) -> None:
        self.workload_hint = self._sync_metadata_field(
            "workload_hint",
            self.workload_hint,
        )
        self.io_profile = self._sync_metadata_field("io_profile", self.io_profile)
        self.expected_parallelism = self._sync_metadata_field(
            "expected_parallelism",
            self.expected_parallelism,
        )
        # Sync SLA fields from metadata
        self.sla_class = self._sync_metadata_field("sla_class", self.sla_class)
        self.deadline_ms = self._sync_metadata_field("deadline_ms", self.deadline_ms)
        self.previous_failures = self._sync_metadata_field(
            "previous_failures", self.previous_failures
        ) or 0

    def _sync_metadata_field(self, key: str, value: Any) -> Any:
        if value is None:
            return self.metadata.get(key)
        self.metadata.setdefault(key, value)
        return value


class TaskDAG:
    def __init__(self) -> None:
        self.tasks: dict[str, Task] = {}

    def add_task(self, task: Task) -> None:
        if task.task_id in self.tasks:
            raise ValueError(f"Task {task.task_id} already exists")
        for dep_id in task.dependencies:
            if dep_id not in self.tasks:
                raise ValueError(
                    f"Dependency {dep_id} not found. Add tasks in dependency order."
                )
        self.tasks[task.task_id] = task
        try:
            self._validate_no_cycles()
        except ValueError:
            # Rollback: remove the task we just added
            del self.tasks[task.task_id]
            raise

    def add_dynamic_task(self, task: Task) -> None:
        self.add_task(task)

    def add_dynamic_tasks(self, tasks: list[Task]) -> None:
        self._validate_dynamic_tasks(tasks)

        for task in tasks:
            self.tasks[task.task_id] = task
        try:
            self._validate_no_cycles()
        except ValueError:
            # Rollback: remove all tasks we just added
            for task in tasks:
                self.tasks.pop(task.task_id, None)
            raise

    def cancel_task(self, task_id: str, *, reason: str | None = None) -> Task:
        task = self.tasks[task_id]
        if task.status != TaskStatus.PENDING:
            raise ValueError(
                f"Task {task_id} cannot be cancelled from status {task.status.value}"
            )
        task.status = TaskStatus.CANCELLED
        task.error = reason
        task.updated_at = time.time()
        return task

    def cancel_downstream(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        max_depth: int | None = None,
    ) -> list[Task]:
        cancelled: list[Task] = []
        for downstream_id, _depth in self.get_downstream_tasks(
            task_id,
            max_depth=max_depth,
        ):
            task = self.tasks[downstream_id]
            if task.status != TaskStatus.PENDING:
                continue
            cancelled.append(self.cancel_task(downstream_id, reason=reason))
        return cancelled

    def replace_downstream(
        self,
        task_id: str,
        replacement_tasks: list[Task],
        *,
        reason: str | None = None,
        max_depth: int | None = None,
    ) -> tuple[list[Task], list[Task]]:
        self._validate_dynamic_tasks(replacement_tasks)
        cancelled = self.cancel_downstream(
            task_id,
            reason=reason,
            max_depth=max_depth,
        )
        for task in replacement_tasks:
            self.tasks[task.task_id] = task
        try:
            self._validate_no_cycles()
        except ValueError:
            # Rollback: remove replacement tasks
            for task in replacement_tasks:
                self.tasks.pop(task.task_id, None)
            raise
        self.compute_critical_path_scores()
        return cancelled, list(replacement_tasks)

    def _validate_dynamic_tasks(self, tasks: list[Task]) -> None:
        known_ids = set(self.tasks)
        batch_ids: set[str] = set()

        for task in tasks:
            if task.task_id in batch_ids:
                raise ValueError(f"Duplicate task id in batch: {task.task_id}")
            if task.task_id in known_ids:
                raise ValueError(f"Task {task.task_id} already exists")
            for dep_id in task.dependencies:
                if dep_id not in known_ids and dep_id not in batch_ids:
                    raise ValueError(
                        f"Dependency {dep_id} not found. Add tasks in dependency order."
                    )
            batch_ids.add(task.task_id)

    def _validate_no_cycles(self) -> None:
        """Validate that the DAG has no cycles using DFS three-color marking.

        Raises ValueError with cycle details if a cycle is detected.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {tid: WHITE for tid in self.tasks}
        parent: dict[str, str | None] = {tid: None for tid in self.tasks}

        def dfs(tid: str) -> None:
            color[tid] = GRAY
            for dep_id in self.tasks[tid].dependencies:
                if dep_id not in self.tasks:
                    continue
                if color[dep_id] == GRAY:
                    # Found a cycle — reconstruct the cycle path
                    cycle = [dep_id, tid]
                    current = tid
                    while parent[current] is not None and parent[current] != dep_id:
                        current = parent[current]
                        cycle.append(current)
                    cycle.reverse()
                    raise ValueError(
                        f"Cycle detected in TaskDAG: {' -> '.join(cycle)} -> {dep_id}"
                    )
                if color[dep_id] == WHITE:
                    parent[dep_id] = tid
                    dfs(dep_id)
            color[tid] = BLACK

        for tid in self.tasks:
            if color[tid] == WHITE:
                dfs(tid)

    def compute_critical_path_scores(self) -> None:
        children: dict[str, list[str]] = {task_id: [] for task_id in self.tasks}
        for task in self.tasks.values():
            for dep_id in task.dependencies:
                children[dep_id].append(task.task_id)

        scores: dict[str, float] = {}

        def score(task_id: str) -> float:
            if task_id in scores:
                return scores[task_id]
            child_scores = [score(child_id) for child_id in children[task_id]]
            scores[task_id] = 1.0 + (max(child_scores) if child_scores else 0.0)
            return scores[task_id]

        for task_id, task in self.tasks.items():
            task.critical_path_score = score(task_id)

    def get_downstream_tasks(
        self,
        task_id: str,
        max_depth: int | None = None,
    ) -> list[tuple[str, int]]:
        """Return downstream tasks as (task_id, depth) tuples.

        Uses BFS to traverse the dependency graph.

        Args:
            task_id: The source task to find downstream of
            max_depth: Maximum propagation depth (None = unlimited)

        Returns:
            List of (task_id, depth) tuples, sorted by depth
        """
        # Build children map (reverse of dependencies)
        children: dict[str, list[str]] = {tid: [] for tid in self.tasks}
        for task in self.tasks.values():
            for dep_id in task.dependencies:
                children[dep_id].append(task.task_id)

        # BFS traversal
        result: list[tuple[str, int]] = []
        queue: list[tuple[str, int]] = [(task_id, 0)]
        visited: set[str] = {task_id}

        while queue:
            current_id, depth = queue.pop(0)

            if max_depth is not None and depth >= max_depth:
                continue

            for child_id in children[current_id]:
                if child_id not in visited:
                    visited.add(child_id)
                    result.append((child_id, depth + 1))
                    queue.append((child_id, depth + 1))

        return result

    def get_ready_tasks(self) -> list[Task]:
        ready: list[Task] = []
        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            deps_met = all(
                self.tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
            )
            if deps_met:
                ready.append(task)
        return ready

    def complete_task(self, task_id: str, result: Any) -> None:
        task = self.tasks[task_id]
        task.status = TaskStatus.COMPLETED
        task.result = result
        task.updated_at = time.time()

    def fail_task(self, task_id: str, error: str) -> None:
        task = self.tasks[task_id]
        task.status = TaskStatus.FAILED
        task.error = error
        task.updated_at = time.time()

    def is_complete(self) -> bool:
        return all(
            task.status in (
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.SKIPPED,
                TaskStatus.CANCELLED,
            )
            for task in self.tasks.values()
        )
