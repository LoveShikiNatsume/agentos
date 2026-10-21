from pathlib import Path

from agentos.events.log import EventLog


async def test_e2e_fault_demo_produces_expected_events_and_dashboard(tmp_path):
    from scripts.demo_e2e_fault import run_demo, generate_dashboard

    log_path = tmp_path / "events.jsonl"
    results, rt, dag = await run_demo(log_path)

    # All 6 tasks completed
    assert len(results) == 6
    for task_id in ("plan", "code", "test", "fix", "deploy", "review"):
        assert dag.tasks[task_id].status.value == "completed"

    # Fault chain present
    event_types = {e.event_type for e in rt.events}
    assert "fault.injected" in event_types
    assert "fault.context_snapshot_saved" in event_types
    assert "fault.context_restored" in event_types
    assert "fault.recovery_completed" in event_types

    # EventLog file written
    assert log_path.exists()
    log_events = list(EventLog(log_path).read())
    assert len(log_events) > 20

    # Dashboard renders
    dashboard_path = generate_dashboard(log_path, tmp_path)
    assert dashboard_path is not None
    html = dashboard_path.read_text(encoding="utf-8")
    assert "AgentOS Runtime Dashboard" in html
    assert "agent-code" in html
