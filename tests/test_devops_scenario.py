from pathlib import Path

from agentos.core.task import TaskStatus
from agentos.events.log import EventLog


async def test_devops_incident_completes_all_tasks_with_recovery(tmp_path: Path):
    from agentos.scenarios.devops.incident import run_devops_incident

    log_path = tmp_path / "events.jsonl"
    results, rt, dag = await run_devops_incident(log_path=log_path, inject_fault=True)

    assert len(results) == 6
    for task_id in ("monitor", "diagnose", "db_check", "fix", "verify", "postmortem"):
        assert dag.tasks[task_id].status == TaskStatus.COMPLETED

    event_types = {e.event_type for e in rt.events}
    assert "task.dynamic_submitted" in event_types
    assert "fault.injected" in event_types
    assert "fault.context_snapshot_saved" in event_types
    assert "fault.context_restored" in event_types
    assert "fault.recovery_completed" in event_types

    assert log_path.exists()
    log_events = list(EventLog(log_path).read())
    assert len(log_events) > 15


async def test_devops_incident_runs_without_fault(tmp_path: Path):
    from agentos.scenarios.devops.incident import run_devops_incident

    log_path = tmp_path / "events.jsonl"
    results, rt, dag = await run_devops_incident(log_path=log_path, inject_fault=False)

    assert len(results) == 6
    for task_id in ("monitor", "diagnose", "db_check", "fix", "verify", "postmortem"):
        assert dag.tasks[task_id].status == TaskStatus.COMPLETED

    injected = [e for e in rt.events if e.event_type == "fault.injected"]
    assert len(injected) == 0
    recovered = [e for e in rt.events if e.event_type == "fault.recovery_completed"]
    assert len(recovered) == 0
