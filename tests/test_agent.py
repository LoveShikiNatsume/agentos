import pytest

from agentos.core.agent import AgentControlBlock, agent_transition
from agentos.core.resources import ResourceQuota, ResourceUsage
from agentos.core.types import (
    COMPATIBILITY_AGENT_STATES,
    RUNTIME_AGENT_STATES,
    AgentState,
    Message,
)


def test_agent_state_values():
    assert AgentState.CREATED.value == "created"
    assert AgentState.WAITING_LLM.value == "waiting_llm"
    assert AgentState.COMPLETED.value == "completed"


def test_acb_creation():
    acb = AgentControlBlock(
        agent_id="agent-1",
        name="coder",
        role="code_generation",
    )
    assert acb.state == AgentState.CREATED
    assert acb.agent_id == "agent-1"
    assert acb.stats.llm_calls == 0


def test_valid_state_transitions():
    acb = AgentControlBlock(agent_id="a1", name="t", role="t")
    agent_transition(acb, AgentState.READY)
    assert acb.state == AgentState.READY
    agent_transition(acb, AgentState.RUNNING)
    assert acb.state == AgentState.RUNNING
    agent_transition(acb, AgentState.WAITING_LLM)
    assert acb.state == AgentState.WAITING_LLM
    agent_transition(acb, AgentState.RUNNING)
    agent_transition(acb, AgentState.COMPLETED)
    assert acb.state == AgentState.COMPLETED


def test_runtime_narrative_state_transitions():
    acb = AgentControlBlock(agent_id="a1", name="t", role="t")
    agent_transition(acb, AgentState.READY)
    agent_transition(acb, AgentState.RUNNING)
    agent_transition(acb, AgentState.BLOCKED_LLM)
    agent_transition(acb, AgentState.RUNNING)
    agent_transition(acb, AgentState.FINISHED)
    assert acb.state == AgentState.FINISHED


def test_failure_recovery_state_transitions():
    acb = AgentControlBlock(agent_id="a1", name="t", role="t")
    agent_transition(acb, AgentState.READY)
    agent_transition(acb, AgentState.RUNNING)
    agent_transition(acb, AgentState.FAILED)
    agent_transition(acb, AgentState.RECOVERING)
    agent_transition(acb, AgentState.READY)
    assert acb.state == AgentState.READY


def test_invalid_state_transition():
    acb = AgentControlBlock(agent_id="a1", name="t", role="t")
    with pytest.raises(ValueError, match="Invalid transition"):
        agent_transition(acb, AgentState.COMPLETED)


def test_message_creation():
    msg = Message(
        from_agent="agent-1",
        to_agent="agent-2",
        content={"type": "result", "data": "hello"},
    )
    assert msg.from_agent == "agent-1"
    assert msg.content["type"] == "result"


def test_message_carries_runtime_metadata():
    msg = Message(
        from_agent="agent-1",
        to_agent="agent-2",
        content={"data": "hello"},
        message_type="result",
        priority=10,
        task_id="task-1",
        trace_id="trace-1",
        context_refs=["ctx://repo-summary"],
    )
    assert msg.message_type == "result"
    assert msg.priority == 10
    assert msg.task_id == "task-1"
    assert msg.trace_id == "trace-1"
    assert msg.context_refs == ["ctx://repo-summary"]


def test_agent_state_includes_runtime_states():
    assert RUNTIME_AGENT_STATES == (
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


def test_legacy_states_are_marked_as_compatibility_states():
    assert COMPATIBILITY_AGENT_STATES == (
        AgentState.WAITING_LLM,
        AgentState.WAITING_TOOL,
        AgentState.BLOCKED,
        AgentState.COMPLETED,
    )


def test_tool_blocking_state_transition():
    acb = AgentControlBlock(agent_id="a1", name="tool", role="tool_user")
    agent_transition(acb, AgentState.READY)
    agent_transition(acb, AgentState.RUNNING)
    agent_transition(acb, AgentState.BLOCKED_TOOL)
    agent_transition(acb, AgentState.RUNNING)
    agent_transition(acb, AgentState.FINISHED)
    assert acb.state == AgentState.FINISHED


def test_terminal_states_reject_outgoing_transitions():
    finished = AgentControlBlock(agent_id="a1", name="done", role="worker")
    agent_transition(finished, AgentState.READY)
    agent_transition(finished, AgentState.RUNNING)
    agent_transition(finished, AgentState.FINISHED)

    with pytest.raises(ValueError, match="Invalid transition"):
        agent_transition(finished, AgentState.READY)

    completed = AgentControlBlock(agent_id="a2", name="done", role="worker")
    agent_transition(completed, AgentState.READY)
    agent_transition(completed, AgentState.RUNNING)
    agent_transition(completed, AgentState.COMPLETED)

    with pytest.raises(ValueError, match="Invalid transition"):
        agent_transition(completed, AgentState.READY)


def test_acb_carries_runtime_metadata():
    quota = ResourceQuota(memory_max_mb=256, timeout_seconds=30.0)
    usage = ResourceUsage(cpu_percent=5.0, memory_rss_mb=64.0)
    acb = AgentControlBlock(
        agent_id="agent-1",
        name="coder",
        role="code_generation",
        task_id="task-1",
        pid=1234,
        cgroup="/sys/fs/cgroup/agentos/agent-1",
        model_backend="deepseek",
        resource_quota=quota,
        resource_usage=usage,
        context_refs=["ctx://task-bg"],
        trace_id="trace-1",
        metadata={"scenario": "coding"},
    )
    assert acb.agent_id == "agent-1"
    assert acb.name == "coder"
    assert acb.role == "code_generation"
    assert acb.state == AgentState.CREATED
    assert acb.task_id == "task-1"
    assert acb.pid == 1234
    assert acb.cgroup == "/sys/fs/cgroup/agentos/agent-1"
    assert acb.model_backend == "deepseek"
    assert acb.resource_quota.memory_max_mb == 256
    assert acb.resource_usage.memory_rss_mb == 64.0
    assert acb.context_refs == ["ctx://task-bg"]
    assert acb.trace_id == "trace-1"
    assert acb.stats.llm_calls == 0
    assert acb.metadata["scenario"] == "coding"


def test_acb_initializes_trace_id_by_default():
    acb = AgentControlBlock(agent_id="agent-1", name="coder", role="code_generation")
    assert acb.trace_id == "trace-agent-1"
