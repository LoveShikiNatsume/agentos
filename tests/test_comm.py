import json

import pytest

from agentos.comm.bus import MessageBus
from agentos.core.types import Message


def _message(
    *,
    from_agent: str = "a1",
    to_agent: str = "a2",
    content: dict | None = None,
    timestamp: float = 1.0,
    message_id: str = "msg",
    message_type: str = "generic",
    priority: int = 0,
    task_id: str | None = None,
    trace_id: str | None = None,
    context_refs: list[str] | None = None,
    metadata: dict | None = None,
) -> Message:
    message = Message(
        from_agent=from_agent,
        to_agent=to_agent,
        content=content or {},
        timestamp=timestamp,
        message_id=message_id,
    )
    message.message_type = message_type
    message.priority = priority
    message.task_id = task_id
    message.trace_id = trace_id
    message.context_refs = context_refs or []
    message.metadata = metadata or {}
    return message


def _load_transport_classes():
    try:
        from agentos.comm.transport import InMemoryMessageTransport, MessageTransport
    except ModuleNotFoundError as exc:
        pytest.fail(f"transport module missing: {exc}")

    return InMemoryMessageTransport, MessageTransport


class RecordingTransport:
    def __init__(self) -> None:
        self.sent: list[Message] = []
        self.received: dict[str, list[Message]] = {}

    async def send(self, message: Message) -> None:
        self.sent.append(message)
        self.received.setdefault(message.to_agent, []).append(message)

    async def receive(self, agent_id: str) -> Message | None:
        messages = self.received.setdefault(agent_id, [])
        if not messages:
            return None
        return messages.pop(0)

    def queue_depth(self, agent_id: str | None = None) -> int:
        if agent_id is not None:
            return len(self.received.get(agent_id, []))
        return sum(len(messages) for messages in self.received.values())


class FalsyRecordingTransport(RecordingTransport):
    def __bool__(self) -> bool:
        return False


class RaisingSendTransport(RecordingTransport):
    async def send(self, message: Message) -> None:
        raise RuntimeError("transport rejected message")


@pytest.fixture
def bus():
    return MessageBus()


def test_default_bus_uses_in_memory_transport(bus: MessageBus):
    InMemoryMessageTransport, _ = _load_transport_classes()

    assert isinstance(bus.transport, InMemoryMessageTransport)


def test_comm_package_exports_transport_types():
    try:
        from agentos.comm import InMemoryMessageTransport, MessageTransport
    except ImportError as exc:
        pytest.fail(f"comm package transport exports missing: {exc}")

    assert InMemoryMessageTransport is not None
    assert MessageTransport is not None


def test_message_supports_metadata_fields_on_direct_construction():
    message = Message(
        from_agent="a",
        to_agent="b",
        content={},
        message_type="context_ref",
        priority=7,
        task_id="task",
        trace_id="trace",
        context_refs=["ctx://x"],
        metadata={"k": "v"},
    )

    assert message.message_type == "context_ref"
    assert message.priority == 7
    assert message.task_id == "task"
    assert message.trace_id == "trace"
    assert message.context_refs == ["ctx://x"]
    assert message.metadata == {"k": "v"}


def test_agent_state_baseline_tuples_are_available():
    from agentos.core.types import (
        COMPATIBILITY_AGENT_STATES,
        RUNTIME_AGENT_STATES,
        AgentState,
    )

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
    assert COMPATIBILITY_AGENT_STATES == (
        AgentState.WAITING_LLM,
        AgentState.WAITING_TOOL,
        AgentState.BLOCKED,
        AgentState.COMPLETED,
    )


async def test_explicit_transport_injection_keeps_bus_observability():
    transport = RecordingTransport()
    bus = MessageBus(transport=transport)
    message = _message(message_id="injected", content={"data": "hello"})

    await bus.send(message)
    received = await bus.receive("a2")

    assert bus.transport is transport
    assert received is message
    assert bus.get_history("a2") == [message]
    assert bus.stats()["message_count"] == 1


def test_explicit_falsy_transport_is_preserved():
    transport = FalsyRecordingTransport()

    bus = MessageBus(transport=transport)

    assert bus.transport is transport


async def test_failed_transport_send_does_not_record_history_or_stats():
    bus = MessageBus(transport=RaisingSendTransport())
    message = _message(to_agent="a2", message_id="rejected")

    with pytest.raises(RuntimeError, match="transport rejected message"):
        await bus.send(message)

    assert bus.get_history("a2") == []
    assert bus.stats()["message_count"] == 0


async def test_send_and_receive(bus: MessageBus):
    msg = _message(content={"data": "hello"})

    await bus.send(msg)
    received = await bus.receive("a2")

    assert received is not None
    assert received.content["data"] == "hello"


async def test_receive_empty(bus: MessageBus):
    result = await bus.receive("a1")

    assert result is None


async def test_messages_are_routed_correctly(bus: MessageBus):
    await bus.send(_message(to_agent="a2", content={"x": 1}, message_id="a2"))
    await bus.send(_message(to_agent="a3", content={"x": 2}, message_id="a3"))

    msg_a3 = await bus.receive("a3")
    assert msg_a3 is not None
    assert msg_a3.content["x"] == 2

    msg_a2 = await bus.receive("a2")
    assert msg_a2 is not None
    assert msg_a2.content["x"] == 1

    assert await bus.receive("a2") is None


async def test_queue_depth_reports_agent_and_total_pending_messages(bus: MessageBus):
    await bus.send(_message(to_agent="a2", message_id="a2-1"))
    await bus.send(_message(to_agent="a2", message_id="a2-2"))
    await bus.send(_message(to_agent="a3", message_id="a3-1"))

    assert bus.queue_depth("a2") == 2
    assert bus.queue_depth("a3") == 1
    assert bus.queue_depth() == 3

    await bus.receive("a2")

    assert bus.queue_depth("a2") == 1
    assert bus.queue_depth() == 2


async def test_priority_ordering_uses_priority_timestamp_and_insertion_sequence(
    bus: MessageBus,
):
    low_priority = _message(
        content={"label": "low"}, priority=1, timestamp=1.0, message_id="low"
    )
    high_newer = _message(
        content={"label": "high-newer"},
        priority=5,
        timestamp=2.0,
        message_id="high-newer",
    )
    high_older = _message(
        content={"label": "high-older"},
        priority=5,
        timestamp=1.0,
        message_id="high-older",
    )
    high_same_first = _message(
        content={"label": "high-same-first"},
        priority=5,
        timestamp=1.0,
        message_id="high-same-first",
    )

    for message in [low_priority, high_newer, high_older, high_same_first]:
        await bus.send(message)

    received = [await bus.receive("a2") for _ in range(4)]

    assert [message.content["label"] for message in received if message] == [
        "high-older",
        "high-same-first",
        "high-newer",
        "low",
    ]


async def test_history_remains_after_receive(bus: MessageBus):
    message = _message(content={"n": 1})

    await bus.send(message)
    assert await bus.receive("a2") is message

    assert bus.get_history("a2") == [message]


async def test_history_preserves_context_refs(bus: MessageBus):
    message = _message(
        content={"ref": "ctx://repo-summary"},
        context_refs=["ctx://repo-summary", "ctx://task-background"],
    )

    await bus.send(message)
    received = await bus.receive("a2")
    history = bus.get_history("a2")

    assert received is not None
    assert received.context_refs == ["ctx://repo-summary", "ctx://task-background"]
    assert history[0].context_refs == ["ctx://repo-summary", "ctx://task-background"]


async def test_history_filters_by_task_and_type(bus: MessageBus):
    design = _message(
        content={"n": 1},
        message_type="design",
        task_id="task-1",
        message_id="design",
    )
    code = _message(
        content={"n": 2},
        message_type="code",
        task_id="task-1",
        message_id="code",
    )
    other_task = _message(
        content={"n": 3},
        message_type="design",
        task_id="task-2",
        message_id="other-task",
    )

    for message in [design, code, other_task]:
        await bus.send(message)

    assert bus.get_history("a2", task_id="task-1") == [design, code]
    assert bus.get_history("a2", message_type="design") == [design, other_task]
    assert bus.get_history("a2", task_id="task-1", message_type="design") == [
        design
    ]
    assert bus.get_history(None, task_id="task-1") == [design, code]


async def test_stats_summarize_filtered_history(bus: MessageBus):
    design = _message(
        to_agent="a2",
        content={"kind": "design", "body": "db"},
        message_type="design",
        task_id="task-1",
        context_refs=["ctx://repo"],
        message_id="design",
    )
    code = _message(
        to_agent="a2",
        content={"kind": "code", "body": "api"},
        message_type="code",
        task_id="task-1",
        context_refs=["ctx://repo", "ctx://api"],
        message_id="code",
    )
    other_agent = _message(
        to_agent="a3",
        content={"kind": "design", "body": "ui"},
        message_type="design",
        task_id="task-2",
        message_id="other-agent",
    )

    for message in [design, code, other_agent]:
        await bus.send(message)

    stats = bus.stats(task_id="task-1")

    assert stats == {
        "message_count": 2,
        "context_ref_count": 3,
        "payload_bytes": _payload_bytes(design) + _payload_bytes(code),
        "by_agent": {"a2": 2},
        "by_type": {"code": 1, "design": 1},
    }
    assert bus.stats(agent_id="a3") == {
        "message_count": 1,
        "context_ref_count": 0,
        "payload_bytes": _payload_bytes(other_agent),
        "by_agent": {"a3": 1},
        "by_type": {"design": 1},
    }
    assert bus.stats(agent_id="missing") == {
        "message_count": 0,
        "context_ref_count": 0,
        "payload_bytes": 0,
        "by_agent": {},
        "by_type": {},
    }


async def test_payload_savings_compares_inline_and_context_ref_messages(
    bus: MessageBus,
):
    inline = _message(
        content={"body": "large inline payload"},
        message_type="inline",
        task_id="task-1",
        message_id="inline",
    )
    context_ref = _message(
        content={"ref": "ctx://large-payload"},
        message_type="context_ref",
        task_id="task-1",
        context_refs=["ctx://large-payload", "ctx://repo"],
        message_id="context-ref",
    )
    other_task = _message(
        content={"body": "ignore me"},
        message_type="inline",
        task_id="task-2",
        message_id="other-task",
    )

    for message in [inline, context_ref, other_task]:
        await bus.send(message)

    inline_bytes = _payload_bytes(inline)
    ref_bytes = _payload_bytes(context_ref)

    assert bus.payload_savings(task_id="task-1") == {
        "inline_message_type": "inline",
        "ref_message_type": "context_ref",
        "inline_message_count": 1,
        "ref_message_count": 1,
        "inline_payload_bytes": inline_bytes,
        "ref_payload_bytes": ref_bytes,
        "bytes_saved": inline_bytes - ref_bytes,
        "payload_reduction_ratio": (inline_bytes - ref_bytes) / inline_bytes,
        "context_ref_count": 2,
    }


def _payload_bytes(message: Message) -> int:
    return len(
        json.dumps(message.content, sort_keys=True, ensure_ascii=False).encode(
            "utf-8"
        )
    )
