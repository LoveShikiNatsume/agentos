import json

from agentos.events.types import (
    EventCategory,
    EventSeverity,
    RuntimeEvent,
    agent_state_changed,
    llm_request_sent,
    llm_response_received,
    task_completed,
)


def test_agent_state_changed_event():
    event = agent_state_changed(
        "agent-1",
        "ready",
        "running",
        task_id="task-1",
        trace_id="trace-1",
    )
    assert event.event_type == "agent.state_changed"
    assert event.agent_id == "agent-1"
    assert event.task_id == "task-1"
    assert event.trace_id == "trace-1"
    assert event.data == {"old_state": "ready", "new_state": "running"}


def test_task_completed_event_truncates_result_preview():
    event = task_completed("task-1", "agent-1", "x" * 250, trace_id="trace-1")
    assert event.event_type == "task.completed"
    assert event.agent_id == "agent-1"
    assert event.task_id == "task-1"
    assert event.trace_id == "trace-1"
    assert len(event.data["result_preview"]) == 200


def test_llm_events_include_resource_data():
    request_event = llm_request_sent(
        "agent-1",
        "default",
        123,
        task_id="task-1",
        trace_id="trace-1",
    )
    response_event = llm_response_received(
        "agent-1",
        "mock",
        12,
        0.01,
        task_id="task-1",
        trace_id="trace-1",
    )

    assert request_event.task_id == "task-1"
    assert request_event.trace_id == "trace-1"
    assert request_event.data["token_count"] == 123
    assert response_event.task_id == "task-1"
    assert response_event.trace_id == "trace-1"
    assert response_event.data["tokens_used"] == 12
    assert response_event.data["latency_secs"] == 0.01


def test_runtime_event_serializes_round_trip():
    event = RuntimeEvent(
        event_type="agent.state_changed",
        category=EventCategory.AGENT,
        severity=EventSeverity.INFO,
        agent_id="agent-1",
        task_id="task-1",
        trace_id="trace-1",
        data={"state": "running"},
        timestamp=123.0,
    )
    restored = RuntimeEvent.from_dict(event.to_dict())
    assert restored.event_type == "agent.state_changed"
    assert restored.category == EventCategory.AGENT
    assert restored.severity == EventSeverity.INFO
    assert restored.agent_id == "agent-1"
    assert restored.data == {"state": "running"}


def test_runtime_event_defaults_to_system_category():
    event = RuntimeEvent(event_type="runtime.started")
    assert event.category == EventCategory.SYSTEM
    assert event.severity == EventSeverity.INFO


def test_runtime_event_to_dict_does_not_alias_data():
    event = RuntimeEvent(event_type="test.event", data={"nested": {"value": 1}})
    serialized = event.to_dict()
    serialized["data"]["nested"]["value"] = 2
    assert event.data == {"nested": {"value": 1}}


def test_runtime_event_factories_are_json_serializable():
    events = [
        agent_state_changed("agent-1", "ready", "running"),
        task_completed("task-1", "agent-1", {"ok": True}),
        llm_request_sent("agent-1", "mock", 123),
        llm_response_received("agent-1", "mock", 12, 0.01),
    ]
    for event in events:
        json.dumps(event.to_dict())


def test_runtime_event_rejects_non_json_data():
    event = RuntimeEvent(event_type="bad", data={"bad": object()})
    try:
        event.to_dict()
    except TypeError as exc:
        assert "RuntimeEvent data must be JSON serializable" in str(exc)
    else:
        raise AssertionError("expected TypeError")
