from __future__ import annotations

import json

from scripts.demo_member_a_feedback_loop import run_demo


def test_member_a_feedback_loop_demo_emits_p0_evidence(tmp_path):
    result = run_demo(tmp_path / "member-a-demo")

    assert result["accepted"] is True
    assert result["results"]["pressure"] == "pressure-ok"
    assert result["results"]["followup"] == "followup-ok"
    assert result["local_sample_phases"] == ["final", "initial", "running"]
    assert result["feedback_applied"] >= 1
    assert result["affected_decisions"] >= 1
    assert result["kernel_decision_published"] >= 2
    assert result["cgroup_removed"] >= 2

    event_types = [event["event_type"] for event in result["events"]]
    assert "resource.cgroup_created" in event_types
    assert "resource.cgroup_pid_added" in event_types
    assert "resource.cgroup_quota_applied" in event_types
    assert "resource.usage_sampled" in event_types
    assert "resource.pressure_sampled" in event_types
    assert "resource.cgroup_feedback_applied" in event_types
    assert "scheduler.decision" in event_types
    assert "kernel.decision_published" in event_types

    output_path = tmp_path / "member-a-demo" / "summary.json"
    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8"))["accepted"] is True
