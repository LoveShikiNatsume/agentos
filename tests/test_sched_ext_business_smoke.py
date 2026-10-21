from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from agentos.benchmarks.sched_ext_business_smoke import (
    SMOKE_MODES,
    assess_trial_evidence,
    build_smoke_plan,
    has_complete_mode_round_pairs,
    smoke_round_seed,
)
from agentos.scenarios.sched_ext_business.pipeline import BUSINESS_TASK_IDS


def _agentos_trial(mode: str = "agentos-static-3dsq") -> dict[str, object]:
    queues = {
        task_id: (
            "latency"
            if task_id in {"patch_validation", "security_gate"}
            else "normal"
            if task_id in {"unit_test_select", "dependency_analysis"}
            else "batch"
        )
        for task_id in BUSINESS_TASK_IDS
    }
    if mode == "agentos-normal":
        queues = {task_id: "normal" for task_id in BUSINESS_TASK_IDS}
    return {
        "mode": mode,
        "completed_tasks": 8,
        "task_loss": 0,
        "task_results": {
            task_id: {
                "runtime_observation": {
                    "policy": 7,
                    "ext_enabled": 1,
                    "affinity": [1, 2],
                }
            }
            for task_id in BUSINESS_TASK_IDS
        },
        "policy_queues": queues,
        "policy_associations": 8,
        "stats_delta": {
            "policy_hit": 8,
            "dispatch_normal": 2,
            "dispatch_latency": 2 if mode != "agentos-normal" else 0,
            "dispatch_batch": 4 if mode != "agentos-normal" else 0,
        },
        "cleanup": {
            "workers": 0,
            "cgroups": 0,
            "policies": 0,
            "task_states": 0,
            "bound_pids": 0,
            "loader": 0,
            "sched_ext_state": "disabled",
        },
        "loader_exit_code": 0,
    }


def test_smoke_plan_freezes_modes_rounds_and_rotation() -> None:
    functional = build_smoke_plan("functional")
    statistical = build_smoke_plan("statistical")

    assert len(functional) == 4
    assert [item["mode"] for item in functional] == list(SMOKE_MODES)
    assert len(statistical) == 12
    assert [item["mode"] for item in statistical[:4]] == list(SMOKE_MODES)
    assert statistical[4]["mode"] == SMOKE_MODES[1]
    assert {item["round"] for item in statistical} == {0, 1, 2}
    with pytest.raises(ValueError, match="unsupported smoke phase"):
        build_smoke_plan("formal")


def test_complete_pair_gate_handles_boolean_set_comparison_without_all() -> None:
    functional = [
        {"mode": mode, "round": 0}
        for mode in SMOKE_MODES
    ]
    assert has_complete_mode_round_pairs(functional, "functional") is True
    assert has_complete_mode_round_pairs(functional[:-1], "functional") is False


def test_all_modes_in_a_paired_round_share_one_seed() -> None:
    plan = build_smoke_plan("statistical")
    seeds_by_round = {
        round_id: {
            smoke_round_seed(20260721, int(item["round"]))
            for item in plan
            if item["round"] == round_id
        }
        for round_id in range(3)
    }
    assert seeds_by_round == {
        0: {20260721},
        1: {20260821},
        2: {20260921},
    }
    with pytest.raises(ValueError, match="non-negative"):
        smoke_round_seed(1, -1)


def test_assess_trial_accepts_real_agentos_queue_and_cleanup_evidence() -> None:
    checks = assess_trial_evidence(_agentos_trial())
    assert all(checks.values())


def test_assess_trial_rejects_wrong_dsq_and_residuals() -> None:
    trial = _agentos_trial()
    trial["policy_queues"]["patch_validation"] = "normal"  # type: ignore[index]
    trial["cleanup"]["policies"] = 1  # type: ignore[index]

    checks = assess_trial_evidence(trial)

    assert checks["queue_mapping"] is False
    assert checks["no_residuals"] is False


def test_assess_cfs_requires_workers_to_remain_outside_sched_ext() -> None:
    trial = _agentos_trial("cfs")
    trial["policy_associations"] = 0
    trial["loader_exit_code"] = None
    trial["stats_delta"] = {}
    trial["cleanup"]["sched_ext_state"] = "disabled"  # type: ignore[index]
    for result in trial["task_results"].values():  # type: ignore[union-attr]
        result["runtime_observation"]["policy"] = 0
        result["runtime_observation"]["ext_enabled"] = 0

    assert all(assess_trial_evidence(trial).values())

    first = next(iter(trial["task_results"].values()))  # type: ignore[union-attr]
    first["runtime_observation"]["ext_enabled"] = 1
    assert assess_trial_evidence(trial)["worker_class"] is False


def test_controlled_pressure_worker_hashes_on_one_cpu_and_stops() -> None:
    cpu = min(os.sched_getaffinity(0))
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "agentos.benchmarks.sched_ext_business_pressure",
            "--cpu",
            str(cpu),
            "--seed",
            "7",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )
    assert process.stderr is not None
    assert process.stderr.readline() == b"READY\n"
    assert process.stdin is not None
    process.stdin.write(b"1")
    process.stdin.flush()
    process.send_signal(signal.SIGTERM)
    stdout, _stderr = process.communicate(timeout=5)
    result = json.loads(stdout)

    assert process.returncode == 0
    assert result["completed"] is True
    assert result["affinity"] == [cpu]
    assert result["iterations"] > 0


def test_business_runtime_requires_worker_startup_barrier_before_sched_ext_bind() -> None:
    source = (
        (Path(__file__).parents[1] / "src") / "scripts/run_sched_ext_business_smoke.py"
    ).read_text(encoding="utf-8")

    assert "SubprocessWorkerManager(" in source
    assert "startup_barrier=True" in source
