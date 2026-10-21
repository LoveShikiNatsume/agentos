import json

from agentos.events.log import EventLog
from agentos.events.types import EventCategory, RuntimeEvent, agent_state_changed


def test_event_log_appends_and_reads_events(tmp_path):
    log = EventLog(tmp_path / "events.jsonl")
    event = agent_state_changed(
        "agent-1",
        "ready",
        "running",
        task_id="task-1",
        trace_id="trace-1",
    )

    log.append(event)

    events = list(log.read())
    assert len(events) == 1
    assert events[0].event_type == "agent.state_changed"
    assert events[0].agent_id == "agent-1"
    assert events[0].task_id == "task-1"
    assert events[0].trace_id == "trace-1"


def test_event_log_creates_parent_directories(tmp_path):
    path = tmp_path / "runs" / "latest" / "events.jsonl"
    log = EventLog(path)

    log.append(RuntimeEvent(event_type="runtime.started"))

    assert path.exists()


def test_event_log_filters_by_agent_task_type_and_category(tmp_path):
    log = EventLog(tmp_path / "events.jsonl")
    log.append(
        RuntimeEvent(
            event_type="agent.state_changed",
            agent_id="agent-1",
            task_id="task-1",
            category=EventCategory.AGENT,
        )
    )
    log.append(
        RuntimeEvent(
            event_type="task.completed",
            agent_id="agent-1",
            task_id="task-2",
            category=EventCategory.TASK,
        )
    )
    log.append(
        RuntimeEvent(
            event_type="agent.state_changed",
            agent_id="agent-2",
            task_id="task-1",
            category=EventCategory.AGENT,
        )
    )

    events = list(
        log.read(
            agent_id="agent-1",
            task_id="task-1",
            event_type="agent.state_changed",
            category=EventCategory.AGENT,
        )
    )

    assert len(events) == 1
    assert events[0].agent_id == "agent-1"
    assert events[0].task_id == "task-1"


def test_event_log_empty_file_reads_no_events(tmp_path):
    log = EventLog(tmp_path / "events.jsonl")

    assert list(log.read()) == []


def test_event_log_skips_bad_lines_by_default(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(RuntimeEvent(event_type="ok").to_dict()),
                "{bad json",
                "",
            ]
        ),
        encoding="utf-8",
    )
    log = EventLog(path)

    events = list(log.read())

    assert [event.event_type for event in events] == ["ok"]


def test_event_log_can_raise_on_bad_lines(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text("{bad json\n", encoding="utf-8")
    log = EventLog(path)

    try:
        list(log.read(skip_bad_lines=False))
    except ValueError as exc:
        assert "Invalid event log line" in str(exc)
    else:
        raise AssertionError("expected ValueError")
