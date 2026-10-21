import json
from pathlib import Path

from agentos.dashboard import build_dashboard, render_dashboard
from agentos.events.log import EventLog
from agentos.events.types import EventCategory, RuntimeEvent


def _write_sample_log(path: Path) -> None:
    log = EventLog(path)
    log.append(
        RuntimeEvent(
            event_type="agent.created",
            agent_id="agent-1",
            task_id="task-1",
            category=EventCategory.AGENT,
            data={"role": "echo"},
        )
    )
    log.append(
        RuntimeEvent(
            event_type="agent.state_changed",
            agent_id="agent-1",
            task_id="task-1",
            category=EventCategory.AGENT,
            data={"old_state": "ready", "new_state": "running"},
        )
    )
    log.append(
        RuntimeEvent(
            event_type="worker.started",
            agent_id="agent-1",
            task_id="task-1",
            category=EventCategory.AGENT,
            data={"pid": 1234, "worker_type": "echo"},
        )
    )
    log.append(
        RuntimeEvent(
            event_type="resource.cgroup_created",
            agent_id="agent-1",
            task_id="task-1",
            category=EventCategory.RESOURCE,
            data={"path": "/sys/fs/cgroup/agentos/agent-1"},
        )
    )
    log.append(
        RuntimeEvent(
            event_type="resource.cgroup_pid_added",
            agent_id="agent-1",
            task_id="task-1",
            category=EventCategory.RESOURCE,
            data={"path": "/sys/fs/cgroup/agentos/agent-1", "pid": 1234},
        )
    )
    log.append(
        RuntimeEvent(
            event_type="resource.usage_sampled",
            agent_id="agent-1",
            task_id="task-1",
            category=EventCategory.RESOURCE,
            data={
                "path": "/sys/fs/cgroup/agentos/agent-1",
                "usage": {"memory_peak_mb": 9.5, "pids_current": 0},
            },
        )
    )
    log.append(
        RuntimeEvent(
            event_type="fault.recovery_started",
            agent_id="agent-1",
            task_id="task-1",
            category=EventCategory.FAULT,
            data={
                "action": "retry",
                "attempt": 1,
                "error": "<script>alert(1)</script>",
            },
        )
    )
    log.append(
        RuntimeEvent(
            event_type="fault.recovery_completed",
            agent_id="agent-1",
            task_id="task-1",
            category=EventCategory.FAULT,
            data={"action": "retry", "recovery_time_secs": 0.1},
        )
    )
    log.append(
        RuntimeEvent(
            event_type="resource.cgroup_removed",
            agent_id="agent-1",
            task_id="task-1",
            category=EventCategory.RESOURCE,
            data={"path": "/sys/fs/cgroup/agentos/agent-1"},
        )
    )


def _benchmark_summary() -> dict:
    return {
        "run_count": 3,
        "modes": {
            "baseline": {
                "total_latency_secs": {"mean": 10.0},
                "total_prompt_tokens": {"mean": 100.0},
                "total_completion_tokens": {"mean": 200.0},
                "cache_hit_ratio": {"mean": 0.25},
                "fault_count": {"mean": 0.0},
                "event_count": {"mean": 8.0},
            },
            "runtime": {
                "total_latency_secs": {"mean": 8.0},
                "total_prompt_tokens": {"mean": 90.0},
                "total_completion_tokens": {"mean": 180.0},
                "cache_hit_ratio": {"mean": 0.5},
                "fault_count": {"mean": 0.0},
                "event_count": {"mean": 10.0},
            },
        },
        "comparison": {
            "latency_delta_secs": -2.0,
            "latency_delta_pct": -0.2,
            "prompt_token_savings": 10.0,
            "cache_hit_ratio_delta": 0.25,
        },
    }


def test_build_dashboard_renders_agents_resources_faults_and_benchmark(tmp_path):
    log_path = tmp_path / "events.jsonl"
    summary_path = tmp_path / "benchmark_summary.json"
    output = tmp_path / "dashboard.html"
    _write_sample_log(log_path)
    summary_path.write_text(
        json.dumps(_benchmark_summary(), ensure_ascii=False), encoding="utf-8"
    )

    written = build_dashboard(
        log_paths=[log_path],
        benchmark_summary_path=summary_path,
        output_path=output,
    )

    html = written.read_text(encoding="utf-8")
    assert written == output
    assert "AgentOS Runtime Dashboard" in html
    assert "agent-1" in html
    assert "1234" in html
    assert "/sys/fs/cgroup/agentos/agent-1" in html
    assert "Recovery Mean" not in html
    assert "Latency Delta" in html
    assert "runtime" in html
    assert "50.00%" in html
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_render_dashboard_without_benchmark_summary_still_returns_html(tmp_path):
    log_path = tmp_path / "events.jsonl"
    _write_sample_log(log_path)
    events = list(EventLog(log_path).read())

    html = render_dashboard(events)

    assert "AgentOS Runtime Dashboard" in html
    assert "No benchmark summary loaded." in html
    assert "agent-1" in html
