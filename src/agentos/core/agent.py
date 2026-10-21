from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from agentos.core.resources import ResourceQuota, ResourceUsage
from agentos.core.types import AgentState, AgentStats


VALID_TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.CREATED: {AgentState.READY},
    AgentState.READY: {AgentState.RUNNING, AgentState.BLOCKED, AgentState.SUSPENDED},
    AgentState.RUNNING: {
        AgentState.BLOCKED_LLM,
        AgentState.BLOCKED_TOOL,
        AgentState.WAITING_LLM,
        AgentState.WAITING_TOOL,
        AgentState.FINISHED,
        AgentState.COMPLETED,
        AgentState.FAILED,
        AgentState.BLOCKED,
        AgentState.SUSPENDED,
    },
    AgentState.BLOCKED_LLM: {
        AgentState.RUNNING,
        AgentState.FAILED,
        AgentState.RECOVERING,
    },
    AgentState.BLOCKED_TOOL: {
        AgentState.RUNNING,
        AgentState.FAILED,
        AgentState.RECOVERING,
    },
    AgentState.WAITING_LLM: {
        AgentState.RUNNING,
        AgentState.FAILED,
        AgentState.RECOVERING,
    },
    AgentState.WAITING_TOOL: {
        AgentState.RUNNING,
        AgentState.FAILED,
        AgentState.RECOVERING,
    },
    AgentState.BLOCKED: {
        AgentState.READY,
        AgentState.FAILED,
        AgentState.RECOVERING,
    },
    AgentState.SUSPENDED: {AgentState.READY, AgentState.FAILED},
    AgentState.FAILED: {AgentState.RECOVERING, AgentState.READY},
    AgentState.RECOVERING: {AgentState.READY, AgentState.FAILED},
    AgentState.FINISHED: set(),
    AgentState.COMPLETED: set(),
}


@dataclass
class AgentControlBlock:
    agent_id: str
    name: str
    role: str
    state: AgentState = AgentState.CREATED
    task_id: str | None = None
    pid: int | None = None
    cgroup: str | None = None
    model_backend: str = "default"
    trace_id: str | None = None
    context_refs: list[str] = field(default_factory=list)
    context_ids: list[str] = field(default_factory=list)
    resource_quota: ResourceQuota = field(default_factory=ResourceQuota)
    resource_usage: ResourceUsage = field(default_factory=ResourceUsage)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    stats: AgentStats = field(default_factory=AgentStats)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.trace_id is None:
            self.trace_id = f"trace-{self.agent_id}"


def agent_transition(acb: AgentControlBlock, new_state: AgentState) -> None:
    allowed = VALID_TRANSITIONS.get(acb.state, set())
    if new_state not in allowed:
        raise ValueError(
            f"Invalid transition: {acb.state.value} -> {new_state.value} "
            f"for agent {acb.agent_id}"
        )
    acb.state = new_state
    acb.updated_at = time.time()
