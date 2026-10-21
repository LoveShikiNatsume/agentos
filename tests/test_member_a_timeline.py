from __future__ import annotations

import json

from scripts.member_a_timeline import generate_timeline


def _event(event_type: str, timestamp: float, **data):
    return {
        "event_type": event_type,
        "timestamp": timestamp,
        "agent_id": data.pop("agent_id", "agent-t1"),
        "task_id": data.pop("task_id", "t1"),
        "data": data,
    }


def test_generate_timeline_extracts_member_a_evidence(tmp_path):
    events_path = tmp_path / "events.jsonl"
    events = [
        _event("worker.started", 100.0, pid=1234, worker_type="sleep"),
        _event("resource.cgroup_created", 100.004, path="/sys/fs/cgroup/agentos/agent-t1"),
        _event(
            "resource.cgroup_pid_added",
            100.006,
            pid=1234,
            pid_membership_verified=True,
        ),
        _event(
            "resource.usage_sampled",
            100.010,
            phase="initial",
            usage={"memory_peak_mb": 32.5, "cpu_usage_usec": 1000},
        ),
        _event(
            "resource.pressure_sampled",
            100.020,
            scope="agent_cgroup",
            phase="running",
            pressure={"pressure_source": "cgroup", "pressure_level": "critical"},
        ),
        _event(
            "resource.cgroup_feedback_applied",
            100.050,
            pressure_source="system+cgroup",
            effective_pressure_level="critical",
        ),
        _event(
            "scheduler.decision",
            100.060,
            policy="fairness_balanced",
            pressure={"pressure_source": "system+cgroup"},
            backpressure={"signal": "defer"},
        ),
        _event(
            "kernel.decision_published",
            100.070,
            bpf_mode="mock",
            record_count=2,
            fallback_active=False,
        ),
        _event("resource.cgroup_removed", 100.090, path="/sys/fs/cgroup/agentos/agent-t1"),
    ]
    events_path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )

    result = generate_timeline(
        events_path,
        json_out=tmp_path / "timeline.json",
        markdown_out=tmp_path / "timeline.md",
    )

    assert result["event_count"] == 9
    assert result["timeline_count"] == 9
    assert result["evidence"]["worker_started"] == 1
    assert result["evidence"]["cgroup_created"] == 1
    assert result["evidence"]["usage_sampled"] == 1
    assert result["evidence"]["cgroup_feedback_applied"] == 1
    assert result["evidence"]["scheduler_decision_with_cgroup_feedback"] == 1
    assert result["timeline"][0]["offset_ms"] == 0.0
    assert result["timeline"][4]["label"] == "resource.pressure_sampled"
    assert result["timeline"][6]["summary"] == (
        "policy=fairness_balanced pressure=system+cgroup backpressure=defer"
    )

    markdown = (tmp_path / "timeline.md").read_text(encoding="utf-8")
    assert "# Member A Runtime Timeline" in markdown
    assert "resource.cgroup_feedback_applied" in markdown
    assert "kernel.decision_published" in markdown
