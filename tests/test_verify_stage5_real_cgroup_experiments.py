from __future__ import annotations

import json

from agentos.benchmarks.real_cgroup_experiment import (
    aggregate_trials,
    build_trial_plan,
    write_artifacts,
)
from scripts.verify_stage5_real_cgroup_experiments import verify_directory


def _complete_trial(entry) -> dict:
    if entry.scenario == "cpu":
        cpu = 100_000.0 if entry.mode == "unrestricted" else 50_000.0
    else:
        cpu = 1_000.0
    oom = (
        1.0
        if entry.scenario == "memory" and entry.mode == "controlled"
        else 0.0
    )
    spawned = (
        7.0
        if entry.scenario == "pids" and entry.mode == "controlled"
        else (16.0 if entry.scenario == "pids" else 0.0)
    )
    failures = (
        9.0
        if entry.scenario == "pids" and entry.mode == "controlled"
        else 0.0
    )
    return {
        **entry.to_dict(),
        "actual_position": entry.planned_position,
        "trial_status": "recorded",
        "error": None,
        "adapter_mode": "cgroup_v2",
        "pid_membership_verified": True,
        "limit_readback_verified": True,
        "pressure_source": "cgroup",
        "pressure_sources": {
            "cpu": "cgroup",
            "memory": "cgroup",
            "io": "cgroup",
        },
        "cleanup_failed": 0,
        "exit_code": (
            -9
            if entry.scenario == "memory" and entry.mode == "controlled"
            else 0
        ),
        "worker_result": {
            "completed": not (
                entry.scenario == "memory" and entry.mode == "controlled"
            )
        },
        "metrics": {
            "cpu_usage_usec": cpu,
            "memory_peak_mb": (
                48.0
                if entry.scenario == "memory" and entry.mode == "controlled"
                else 96.0
            ),
            "oom_kill_count": oom,
            "children_spawned": spawned,
            "spawn_failures": failures,
            "wall_time_secs": 1.0,
        },
    }


def _write_complete_evidence(tmp_path) -> None:
    plan = build_trial_plan(repetitions=10, seed=20260703)
    trials = [_complete_trial(entry) for entry in plan]
    summary = aggregate_trials(
        trials,
        repetitions=10,
        seed=20260703,
    )
    write_artifacts(
        output_dir=tmp_path,
        plan=plan,
        trials=trials,
        summary=summary,
    )


def test_verifier_accepts_complete_recomputable_real_cgroup_evidence(
    tmp_path,
) -> None:
    _write_complete_evidence(tmp_path)

    result = verify_directory(tmp_path)

    assert result["passed"] is True
    assert result["errors"] == []
    assert result["counts"]["measured_trials"] == 60
    assert result["enforcement"]["cpu_ratio"] == 0.5


def test_verifier_rejects_missing_membership_or_cleanup_failure(tmp_path) -> None:
    _write_complete_evidence(tmp_path)
    raw_path = tmp_path / "raw_trials.jsonl"
    rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[6]["pid_membership_verified"] = False
    rows[7]["cleanup_failed"] = 1
    raw_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = verify_directory(tmp_path)

    assert result["passed"] is False
    assert any("membership" in error for error in result["errors"])
    assert any("cleanup" in error for error in result["errors"])


def test_verifier_rejects_weak_cpu_or_missing_memory_enforcement(
    tmp_path,
) -> None:
    _write_complete_evidence(tmp_path)
    raw_path = tmp_path / "raw_trials.jsonl"
    rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        if row["scenario"] == "cpu" and row["mode"] == "controlled":
            row["metrics"]["cpu_usage_usec"] = 90_000.0
        if row["scenario"] == "memory" and row["mode"] == "controlled":
            row["metrics"]["oom_kill_count"] = 0.0
    raw_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = verify_directory(tmp_path)

    assert result["passed"] is False
    assert any("CPU" in error for error in result["errors"])
    assert any("memory" in error for error in result["errors"])
