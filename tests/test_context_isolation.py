import pytest

from agentos.context.manager import ContextManager
from agentos.context.types import ContextAccessDenied
from agentos.events.types import EventCategory, EventSeverity, RuntimeEvent


def test_read_shared_context_returns_content():
    manager = ContextManager()
    manager.add_shared_ref("task-bg", "project background")
    assert manager.read("agent-1", "task-bg") == "project background"


def test_owner_can_read_own_private_context():
    manager = ContextManager()
    ref = manager.add_private_ref("agent-1", "agent-1 secret notes")
    assert manager.read("agent-1", ref.name) == "agent-1 secret notes"


def test_unauthorized_agent_read_private_denied():
    manager = ContextManager()
    ref = manager.add_private_ref("agent-1", "agent-1 secret notes")
    with pytest.raises(ContextAccessDenied):
        manager.read("agent-2", ref.name)


def test_denied_read_emits_audit_event():
    events: list[RuntimeEvent] = []
    manager = ContextManager(event_hook=events.append)
    ref = manager.add_private_ref("agent-1", "agent-1 secret notes")

    with pytest.raises(ContextAccessDenied):
        manager.read("agent-2", ref.name)

    assert len(events) == 1
    event = events[0]
    assert event.event_type == "context.access_denied"
    assert event.category == EventCategory.CONTEXT
    assert event.severity == EventSeverity.WARNING
    assert event.agent_id == "agent-2"
    assert event.data["ref"] == ref.uri
    assert event.data["owner_agent_id"] == "agent-1"
    # 审计事件必须能进 JSONL EventLog
    event.to_dict()


def test_authorized_read_emits_no_audit_event():
    events: list[RuntimeEvent] = []
    manager = ContextManager(event_hook=events.append)
    manager.add_shared_ref("task-bg", "project background")
    manager.read("agent-1", "task-bg")
    assert events == []


def test_read_missing_ref_raises_keyerror():
    manager = ContextManager()
    with pytest.raises(KeyError):
        manager.read("agent-1", "no-such-ref")


def test_denied_read_does_not_leak_content():
    events: list[RuntimeEvent] = []
    manager = ContextManager(event_hook=events.append)
    ref = manager.add_private_ref("agent-1", "agent-1 secret notes")
    with pytest.raises(ContextAccessDenied) as exc_info:
        manager.read("agent-2", ref.name)
    assert "secret" not in str(exc_info.value)
    assert "secret" not in str(events[0].to_dict())
