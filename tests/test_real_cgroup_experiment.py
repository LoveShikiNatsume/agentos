from __future__ import annotations

from collections import defaultdict
import json

from agentos.benchmarks.real_cgroup_experiment import (
    MODES,
    SCENARIOS,
    SCHEMA_VERSION,
    aggregate_trials,
    build_trial_plan,
    limit_readback_matches,
    quota_for,
)
from scripts import run_stage5_real_cgroup_experiments


def test_real_cgroup_plan_is_complete_paired_and_randomized() -> None:
    plan = build_trial_plan(repetitions=10, seed=20260703)

    assert len(plan) == 66
    assert sum(entry.warmup for entry in plan) == 6
    assert len({entry.trial_id for entry in plan}) == 66

    measured = [entry for entry in plan if not entry.warmup]
    blocks = defaultdict(list)
    for entry in measured:
        blocks[(entry.scenario, entry.repetition)].append(entry)

    assert len(blocks) == len(SCENARIOS) * 10
    orders = defaultdict(set)
    for (scenario, _repetition), entries in blocks.items():
        ordered = sorted(entries, key=lambda item: item.planned_position)
        assert tuple(entry.mode for entry in ordered) in (
            MODES,
            tuple(reversed(MODES)),
        )
        orders[scenario].add(tuple(entry.mode for entry in ordered))
    assert all(len(values) == 2 for values in orders.values())


def test_real_cgroup_plan_is_deterministic_and_seeded() -> None:
    first = build_trial_plan(repetitions=10, seed=17)
    second = build_trial_plan(repetitions=10, seed=17)
    third = build_trial_plan(repetitions=10, seed=18)

    assert first == second
    assert first != third


def _trial(entry, *, value: float) -> dict:
    metrics = {
        "cpu_usage_usec": value,
        "memory_peak_mb": value,
        "oom_kill_count": value,
        "children_spawned": value,
        "spawn_failures": value,
        "wall_time_secs": value,
    }
    return {
        **entry.to_dict(),
        "actual_position": entry.planned_position,
        "trial_status": "recorded",
        "pid_membership_verified": True,
        "limit_readback_verified": True,
        "pressure_source": "cgroup",
        "cleanup_failed": 0,
        "metrics": metrics,
    }


def test_aggregate_preserves_cells_statistics_and_claim_boundary() -> None:
    plan = build_trial_plan(repetitions=10, seed=11)
    trials = [
        _trial(entry, value=float(entry.repetition or 1))
        for entry in plan
    ]

    summary = aggregate_trials(
        trials,
        repetitions=10,
        seed=11,
    )

    assert summary["schema_version"] == SCHEMA_VERSION
    assert summary["metadata"]["measured_trial_count"] == 60
    assert summary["metadata"]["warmup_count"] == 6
    cell = summary["cells"]["cpu"]["controlled"]
    assert cell["recorded_trials"] == 10
    assert cell["metrics"]["cpu_usage_usec"]["n"] == 10
    assert cell["metrics"]["cpu_usage_usec"]["mean"] == 5.5
    assert cell["metrics"]["cpu_usage_usec"]["stddev"] > 0
    assert summary["claim_boundary"] == {
        "real_cgroup_v2": True,
        "bpf_program_attached": False,
        "sched_ext_active": False,
        "kernel_scheduling_takeover": False,
    }


def test_controlled_quotas_match_declared_safety_limits() -> None:
    assert quota_for("cpu", "controlled").cpu_max_percent == 20.0
    assert quota_for("memory", "controlled").memory_max_mb == 48
    assert quota_for("pids", "controlled").pids_max == 8
    assert quota_for("cpu", "unrestricted").has_limits() is False


def test_limit_readback_requires_the_scenario_specific_controller() -> None:
    assert limit_readback_matches(
        "cpu",
        "controlled",
        {"cpu_max_percent": 20.0},
    )
    assert limit_readback_matches(
        "memory",
        "controlled",
        {"memory_max_mb": 48, "memory_swap_max": 0},
    )
    assert limit_readback_matches(
        "pids",
        "controlled",
        {"pids_max": 8},
    )
    assert limit_readback_matches("cpu", "unrestricted", {})
    assert not limit_readback_matches(
        "memory",
        "controlled",
        {"memory_max_mb": 47},
    )


def test_real_cgroup_runner_rejects_formal_repetitions_below_ten(
    tmp_path,
    capsys,
) -> None:
    exit_code = run_stage5_real_cgroup_experiments.main(
        [
            "--repetitions",
            "9",
            "--out",
            str(tmp_path),
        ]
    )

    assert exit_code == 2
    assert "at least 10" in capsys.readouterr().err


def test_real_cgroup_runner_emits_skip_without_delegation(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        run_stage5_real_cgroup_experiments,
        "detect_delegation",
        lambda: ("host_root", False),
    )

    exit_code = run_stage5_real_cgroup_experiments.main(
        [
            "--out",
            str(tmp_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["passed"] is False
    assert payload["delegation_mode"] == "host_root"
    assert payload["skip_reason"]


def test_real_cgroup_overwrite_preserves_output_mountpoint(tmp_path) -> None:
    output = tmp_path / "evidence"
    nested = output / "nested"
    nested.mkdir(parents=True)
    (output / "old.json").write_text("old", encoding="utf-8")
    (nested / "old.txt").write_text("old", encoding="utf-8")

    run_stage5_real_cgroup_experiments._clear_directory(output)

    assert output.is_dir()
    assert list(output.iterdir()) == []
