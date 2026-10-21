from __future__ import annotations

from pathlib import Path

import pytest

from agentos.benchmarks.sched_ext_stage4 import (
    STAGE4_MODES,
    acceptance_passed,
    jain_fairness,
    paired_improvement_95,
    smoke_acceptance_passed,
    summarize_trials,
)
from scripts.verify_sched_ext_policy_effects import policy_effects_accepted
from scripts import run_sched_ext_stage4_ablation as stage4_ablation
from scripts import sched_ext_stage4_common as stage4_common


def _mode(runs: int) -> dict:
    return {
        "runs": runs,
        "completed_runs": runs,
        "workers_attempted": runs * 5,
        "workers_completed": runs * 5,
        "batch_progress_runs": runs,
        "affinity_violations": 0,
    }


def _ablation_summary(runs: int) -> dict:
    return {
        "modes": {mode: _mode(runs) for mode in STAGE4_MODES},
        "paired_latency_improvement": {
            "agentos_static_3dsq": {"ci95_low": 0.01},
            "agentos_full_dag_psi_fairness": {"ci95_low": 0.02},
        },
        "batch_wait_max_ns": 100_000_000,
        "batch_process_max_gap_us": 100_000,
        "worker_startup_max_us": 100_000,
        "weight_direction_verified": True,
        "slice_direction_verified": True,
        "queue_stats_verified": True,
        "checksum_consistent": True,
        "pressure_warning_observed": True,
        "warning_generation_verified": True,
        "policy_leaks": 0,
        "task_state_leaks": 0,
        "kernel_log_errors": [],
        "systemd_failed_units": 0,
        "final_sched_ext_state": "disabled",
        "failure_reason": None,
    }


def test_jain_fairness_and_trial_summary() -> None:
    assert jain_fairness([10, 10, 10]) == pytest.approx(1.0)
    assert jain_fairness([10, 0]) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        jain_fairness([-1, 1])

    trial = {
        "latency_p95_us": 5.0,
        "throughput_iterations": 100,
        "batch_max_gap_us": 50.0,
        "worker_startup_max_us": 25.0,
        "jain_fairness": 0.9,
        "completed": True,
        "affinity_violations": 0,
        "workers": [
            {"role": "batch", "iterations": 10, "completed": True, "return_code": 0},
            {"role": "normal", "iterations": 10, "completed": True, "return_code": 0},
        ],
    }
    result = summarize_trials([trial, trial])
    assert result["runs"] == 2
    assert result["completed_runs"] == 2


def test_paired_improvement_uses_baseline_minus_candidate() -> None:
    result = paired_improvement_95([100, 100, 100], [80, 75, 85])

    assert result["mean"] == pytest.approx(0.2)
    assert result["ci95_low"] > 0


def test_formal_and_smoke_have_distinct_acceptance_thresholds() -> None:
    formal = _ablation_summary(20)
    assert acceptance_passed(formal)

    smoke = _ablation_summary(3)
    smoke["paired_latency_improvement"] = {}
    smoke["pressure_warning_observed"] = False
    smoke["weight_direction_verified"] = False
    assert smoke_acceptance_passed(smoke)
    assert not acceptance_passed(smoke, expected_runs=3)


@pytest.mark.parametrize("acceptance", [acceptance_passed, smoke_acceptance_passed])
def test_acceptance_rejects_batch_wait_and_worker_startup_stalls(acceptance) -> None:
    summary = _ablation_summary(3)
    assert acceptance(summary, expected_runs=3)

    summary["batch_wait_max_ns"] = 250_000_001
    assert not acceptance(summary, expected_runs=3)

    summary = _ablation_summary(3)
    summary["worker_startup_max_us"] = 500_001
    assert not acceptance(summary, expected_runs=3)


@pytest.mark.parametrize(
    ("mode", "observed", "expected"),
    [
        ("agentos_static_3dsq", "warning", "normal"),
        ("agentos_normal", "critical", "normal"),
        ("agentos_full_dag_psi_fairness", "warning", "warning"),
    ],
)
def test_only_full_mode_consumes_observed_pressure(
    mode: str, observed: str, expected: str
) -> None:
    assert stage4_common.policy_pressure_level(mode, observed) == expected


def test_controller_is_pinned_to_reserved_cpu_and_original_mask_is_returned(
    monkeypatch,
) -> None:
    calls: list[tuple[int, set[int]]] = []

    monkeypatch.setattr(stage4_common.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3})
    monkeypatch.setattr(
        stage4_common.os,
        "sched_setaffinity",
        lambda pid, cpus: calls.append((pid, set(cpus))),
    )

    assert stage4_common.pin_controller_cpu() == {0, 1, 2, 3}
    assert calls == [(0, {0})]


def test_controller_pin_rejects_missing_reserved_cpu(monkeypatch) -> None:
    monkeypatch.setattr(stage4_common.os, "sched_getaffinity", lambda pid: {1, 2, 3})

    with pytest.raises(RuntimeError, match="reserved controller CPU 0"):
        stage4_common.pin_controller_cpu()


def test_proc_state_parser_handles_spaces_and_parentheses(monkeypatch) -> None:
    class _StatPath:
        def read_text(self, **_kwargs):
            return "123 (worker name (stage4)) S 1 2 3"

    monkeypatch.setattr(stage4_common, "Path", lambda _value: _StatPath())

    assert stage4_common._proc_state(123) == "S"


def test_mixed_workload_uses_reserved_worker_cpu_set() -> None:
    assert stage4_common.CONTROLLER_CPU == 0
    assert stage4_common.WORKER_CPU_LIST == "1,2"


def test_heartbeat_renews_initial_policy_before_pressure_changes(monkeypatch) -> None:
    class _StopAfterOneIteration:
        def __init__(self) -> None:
            self.calls = 0

        def wait(self, _timeout: float) -> bool:
            self.calls += 1
            return self.calls > 1

    class _Control:
        def __init__(self) -> None:
            self.generations: list[int] = []
            self.policies = []

        def update_global_config(self, *, generation: int, flags: int) -> None:
            del flags
            self.generations.append(generation)

        def put_policy(self, key, policy):
            self.policies.append((key, policy))
            return policy

        @staticmethod
        def policy_to_dict(policy):
            return {"generation": policy.generation}

    monkeypatch.setattr(stage4_common, "cpu_psi_avg10", lambda: 0.0)
    control = _Control()
    errors: list[str] = []

    stage4_common._heartbeat(
        _StopAfterOneIteration(),
        control,
        7,
        0,
        [],
        "agentos_normal",
        "normal",
        {},
        [
            {
                "task_id": "normal-1",
                "role": "normal",
                "weight": 100,
                "slice_us": 5_000,
                "key": "policy-key",
                "overrides": None,
            }
        ],
        [],
        errors,
    )

    assert errors == []
    assert control.generations == [7]
    assert len(control.policies) == 1
    assert control.policies[0][1].generation == 7


def test_ablation_pressure_cpu_is_disjoint_and_topology_is_available(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        stage4_ablation.os,
        "sched_getaffinity",
        lambda pid: {0, 1, 2, 3},
    )

    assert stage4_ablation.PRESSURE_CPU == 3
    assert stage4_ablation._worker_cpus() == {1, 2}
    assert stage4_ablation._validate_cpu_topology() == [0, 1, 2, 3]


def test_ablation_topology_rejects_missing_pressure_cpu(monkeypatch) -> None:
    monkeypatch.setattr(
        stage4_ablation.os,
        "sched_getaffinity",
        lambda pid: {0, 1, 2},
    )

    with pytest.raises(RuntimeError, match="missing \\[3\\]"):
        stage4_ablation._validate_cpu_topology()


def test_policy_effects_acceptance_requires_all_fallback_classes() -> None:
    summary = {
        "preflight_accepted": True,
        "queue_stats_verified": True,
        "weight_direction_verified": True,
        "slice_direction_verified": True,
        "fallback_cases_completed": True,
        "fallback_stats_delta": {
            "fallback_lookup": 1,
            "fallback_version": 1,
            "fallback_range": 3,
            "fallback_generation": 1,
            "fallback_heartbeat": 1,
            "fallback_policy_stale": 1,
        },
        "policy_leaks": 0,
        "task_state_leaks": 0,
        "batch_wait_max_ns": 1,
        "kernel_log_errors": [],
        "systemd_failed_units": 0,
        "final_sched_ext_state": "disabled",
        "failure_reason": None,
    }

    assert policy_effects_accepted(summary)
    summary["fallback_stats_delta"]["fallback_range"] = 2
    assert not policy_effects_accepted(summary)


def test_policy_effects_fallback_workers_wait_at_barrier_and_cover_stale_lease() -> None:
    source = (
        (Path(__file__).parents[1] / "src") / "scripts/verify_sched_ext_policy_effects.py"
    ).read_text(encoding="utf-8")

    assert "wait_worker_barrier(process)" in source
    assert '("policy_stale", _policy_bytes(updated_at_ns=1), False)' in source
    assert 'fallbacks.get("fallback_policy_stale", 0) > 0' in source


def test_simple_partial_compat_loader_is_started_without_upstream_p_flag(
    monkeypatch, tmp_path
) -> None:
    calls = []

    class _Process:
        returncode = None
        stdout = None
        stderr = None

        def poll(self):
            return None

    process = _Process()

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return process

    waited = []
    monkeypatch.setattr(stage4_common.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        stage4_common,
        "_wait_sysfs",
        lambda ops, **kwargs: waited.append((ops, kwargs)),
    )

    assert stage4_common._start_simple(tmp_path / "scx_simple_partial") is process
    assert calls[0][0] == [str(tmp_path / "scx_simple_partial")]
    assert waited == [("simple", {"partial": True, "process": process})]


def test_native_scheduler_kicks_target_cpu_and_tracks_batch_dispatch_per_cpu() -> None:
    source = (
        (Path(__file__).parents[1] / "src") / "native/sched_ext/agentos_scx.bpf.c"
    ).read_text(encoding="utf-8")

    assert "batch_dispatch_state SEC(\".maps\")" in source
    assert "scx_bpf_task_cpu(p)" in source
    assert "scx_bpf_kick_cpu(cpu, SCX_KICK_IDLE)" in source
    assert "static u64 last_batch_dispatch_ns;" not in source


def test_native_latency_preemption_targets_worker_cpu_with_per_target_throttle() -> None:
    source = (
        (Path(__file__).parents[1] / "src") / "native/sched_ext/agentos_scx.bpf.c"
    ).read_text(encoding="utf-8")

    assert "static void request_latency_preempt(struct task_struct *p, u64 now)" in source
    assert "cpu = scx_bpf_task_cpu(p);" in source
    assert "scx_bpf_kick_cpu(cpu, SCX_KICK_PREEMPT);" in source
    assert "__uint(type, BPF_MAP_TYPE_ARRAY);" in source
    assert "__uint(max_entries, AGENTOS_SCX_MAX_CPUS);" in source
    assert "request_latency_preempt(p, now);" in source
    assert "AGENTOS_SCX_TASK_FLAG_CONTINUOUSLY_RUNNABLE" in source
    assert "if (!was_runnable)" in source


def test_native_latency_tasks_do_not_bypass_latency_dsq_on_idle_cpu() -> None:
    source = (
        (Path(__file__).parents[1] / "src") / "native/sched_ext/agentos_scx.bpf.c"
    ).read_text(encoding="utf-8")
    select_cpu = source.split(
        "s32 BPF_STRUCT_OPS(agentos_scx_select_cpu", 1
    )[1].split("void BPF_STRUCT_OPS(agentos_scx_enqueue", 1)[0]

    latency_guard = (
        "if (policy.queue_class == AGENTOS_SCX_QUEUE_LATENCY)\n"
        "\t\t\treturn cpu;"
    )
    assert latency_guard in select_cpu
    assert select_cpu.index(latency_guard) < select_cpu.index(
        "scx_bpf_dispatch(p, SCX_DSQ_LOCAL"
    )
