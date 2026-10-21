from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentState(Enum):
    CREATED = "created"
    READY = "ready"
    RUNNING = "running"
    BLOCKED_LLM = "blocked_llm"
    BLOCKED_TOOL = "blocked_tool"
    SUSPENDED = "suspended"
    FINISHED = "finished"
    RECOVERING = "recovering"
    WAITING_LLM = "waiting_llm"
    WAITING_TOOL = "waiting_tool"
    BLOCKED = "blocked"
    FAILED = "failed"
    COMPLETED = "completed"


RUNTIME_AGENT_STATES: tuple[AgentState, ...] = (
    AgentState.CREATED,
    AgentState.READY,
    AgentState.RUNNING,
    AgentState.BLOCKED_LLM,
    AgentState.BLOCKED_TOOL,
    AgentState.SUSPENDED,
    AgentState.FINISHED,
    AgentState.FAILED,
    AgentState.RECOVERING,
)

COMPATIBILITY_AGENT_STATES: tuple[AgentState, ...] = (
    AgentState.WAITING_LLM,
    AgentState.WAITING_TOOL,
    AgentState.BLOCKED,
    AgentState.COMPLETED,
)


@dataclass
class AgentStats:
    llm_calls: int = 0
    tokens_sent: int = 0
    tokens_received: int = 0
    token_used: int = 0
    tool_calls: int = 0
    total_llm_wait_secs: float = 0.0
    total_tool_wait_secs: float = 0.0
    failures: int = 0
    recoveries: int = 0


@dataclass
class Message:
    from_agent: str
    to_agent: str
    content: dict[str, Any]
    message_type: str = "generic"
    priority: int = 0
    task_id: str | None = None
    trace_id: str | None = None
    context_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: f"msg-{time.time_ns()}")
