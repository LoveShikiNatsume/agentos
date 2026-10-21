from __future__ import annotations

import copy
import json
import shutil
from collections import defaultdict
from pathlib import Path

import pytest

import scripts.verify_sched_ext_business_formal as verifier

from agentos.benchmarks.sched_ext_business import seal_directory, write_json, write_jsonl
from agentos.benchmarks.sched_ext_business_formal import (
    FORMAL_MODES,
    build_formal_plan,
    evaluate_primary_gate,
    formal_round_seed,
)
from agentos.benchmarks.sched_ext_business_smoke import assess_trial_evidence
from agentos.scenarios.sched_ext_business.metrics import paired_improvements, summarise_trials
from scripts.run_sched_ext_business_demo import run_functional_demo
from scripts.run_sched_ext_business_formal import (
    FORMAL_SCHEMA,
    FROZEN_FIXTURE_SHA256,
    FROZEN_HOST,
    FROZEN_KERNEL,
    FROZEN_SCHEDULER_SHA256,
    FROZEN_WORK_UNITS,
)
from scripts.verify_sched_ext_business_formal import verify_formal_bundle


# Synthetic immutable test input, not an archived kernel/performance experiment.
# Only the test's trusted input identity is injected; all 19 verifier checks run.
SYNTHETIC_FIXTURE_SHA256 = "c88321526f41c50673e2ca6eaa1b9554996f22ef8635b7dde03d59ba6466b56f"


def _formal_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(verifier, "FROZEN_FIXTURE_SHA256", SYNTHETIC_FIXTURE_SHA256)
    functional = tmp_path / "functional"
    run_functional_demo(
        source_root=Path(__file__).parent / "测试及验证材料" / "fixtures" / "sched_ext_verifier",
        output_dir=functional,
        fixture_paths=("README.md", "sample.py", "test_sample.py"),
        work_units=1,
    )
    base_trial = json.loads((functional / "trials.jsonl").read_text(encoding="utf-8"))
    base_events = [
        json.loads(line)
        for line in (functional / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads((functional / "fixture_manifest.json").read_text(encoding="utf-8"))
    assert manifest["aggregate_sha256"] == SYNTHETIC_FIXTURE_SHA256

    bundle = tmp_path / "formal"
    bundle.mkdir()
    shutil.copytree(functional / "fixture", bundle / "fixture")
    write_json(bundle / "fixture_manifest.json", manifest)
    write_json(
        bundle / "environment.json",
        {
            "schema": FORMAL_SCHEMA,
            "phase": "formal",
            "formal": True,
            "host": FROZEN_HOST,
            "kernel_release": FROZEN_KERNEL,
            "controller_cpu": 0,
            "worker_cpus": [1, 2],
            "pressure_cpu": 3,
            "modes": list(FORMAL_MODES),
            "warmup_rounds_per_mode": 2,
            "formal_rounds_per_mode": 20,
            "warmup_in_formal_statistics": False,
        },
    )
    write_json(
        bundle / "build_manifest.json",
        {
            "schema": FORMAL_SCHEMA,
            "scheduler_sha256": FROZEN_SCHEDULER_SHA256,
            "fixture_sha256": SYNTHETIC_FIXTURE_SHA256,
            "fixture_version": 2,
            "work_units": FROZEN_WORK_UNITS,
            "seed": 20260721,
            "primary_baseline": "cfs",
            "primary_candidate": "agentos-static-3dsq",
            "primary_metric": "critical_ready_ms",
        },
    )
    write_json(bundle / "preflight.json", {"accepted": True})

    all_trials: dict[str, list[dict]] = defaultdict(list)
    all_events: dict[str, list[dict]] = defaultdict(list)
    all_loader_events: dict[str, list[dict]] = defaultdict(list)
    for sample_kind in ("warmup", "formal"):
        for item in build_formal_plan(sample_kind):
            mode = str(item["mode"])
            trial = copy.deepcopy(base_trial)
            trial.update(
                schema=FORMAL_SCHEMA,
                formal=True,
                sample_kind=sample_kind,
                mode=mode,
                round=int(item["round"]),
                order=int(item["order"]),
                seed=formal_round_seed(20260721, int(item["round"]), sample_kind=sample_kind),
                work_units={task_id: FROZEN_WORK_UNITS for task_id in base_trial["work_units"]},
                policy_associations=0 if mode == "cfs" else 8,
                policy_queues=(
                    {}
                    if mode == "cfs"
                    else {
                        task_id: (
                            "normal"
                            if mode == "agentos-normal"
                            else "latency"
                            if task_id in {"patch_validation", "security_gate"}
                            else "normal"
                            if task_id in {"unit_test_select", "dependency_analysis"}
                            else "batch"
                        )
                        for task_id in base_trial["task_results"]
                    }
                ),
                loader_exit_code=None if mode == "cfs" else 0,
                cleanup={
                    "workers": 0,
                    "cgroups": 0,
                    "policies": 0,
                    "task_states": 0,
                    "bound_pids": 0,
                    "loader": 0,
                    "sched_ext_state": "disabled",
                },
            )
            for result in trial["task_results"].values():
                result["runtime_observation"].update(
                    policy=0 if mode == "cfs" else 7,
                    ext_enabled=0 if mode == "cfs" else 1,
                    affinity=[1, 2],
                )
            trial["stats_delta"] = (
                {}
                if mode == "cfs"
                else {
                    "policy_hit": 8,
                    "dispatch_normal": 8,
                    "dispatch_latency": 0 if mode == "agentos-normal" else 2,
                    "dispatch_batch": 0 if mode == "agentos-normal" else 4,
                }
            )
            trial["evidence_checks"] = assess_trial_evidence(trial)
            all_trials[sample_kind].append(trial)
            events = [
                {
                    **copy.deepcopy(event),
                    "benchmark_mode": mode,
                    "benchmark_round": int(item["round"]),
                    "sample_kind": sample_kind,
                }
                for event in base_events
            ]
            if mode != "cfs":
                queue_codes = {"latency": 1, "normal": 2, "batch": 3}
                events.extend(
                    {
                        "event_type": "kernel.sched_ext_policy_published",
                        "task_id": task_id,
                        "data": {
                            "readback_verified": True,
                            "policy": {"queue_class": queue_codes[queue]},
                        },
                        "timestamp": 1.0,
                        "monotonic_ns": 1,
                        "benchmark_mode": mode,
                        "benchmark_round": int(item["round"]),
                        "sample_kind": sample_kind,
                    }
                    for task_id, queue in trial["policy_queues"].items()
                )
                delta = trial["stats_delta"]
                all_loader_events[sample_kind].extend(
                    [
                        {
                            "event": "ready",
                            "benchmark_mode": mode,
                            "benchmark_round": int(item["round"]),
                            "sample_kind": sample_kind,
                        },
                        {
                            "event": "stats",
                            "stats": {name: 0 for name in delta},
                            "benchmark_mode": mode,
                            "benchmark_round": int(item["round"]),
                            "sample_kind": sample_kind,
                        },
                        {
                            "event": "final_stats",
                            "stats": delta,
                            "benchmark_mode": mode,
                            "benchmark_round": int(item["round"]),
                            "sample_kind": sample_kind,
                        },
                    ]
                )
            all_events[sample_kind].extend(events)

    formal_trials = all_trials["formal"]
    benchmark_summary = {
        "modes": summarise_trials(formal_trials),
        "paired_improvements": {
            mode: paired_improvements(formal_trials, baseline="cfs", candidate=mode)
            for mode in FORMAL_MODES
            if mode != "cfs"
        },
    }
    primary_gate = evaluate_primary_gate(formal_trials)
    write_jsonl(bundle / "warmup_trials.jsonl", all_trials["warmup"])
    write_jsonl(bundle / "warmup_events.jsonl", all_events["warmup"])
    write_jsonl(bundle / "warmup_loader_events.jsonl", all_loader_events["warmup"])
    write_jsonl(bundle / "trials.jsonl", formal_trials)
    write_jsonl(bundle / "events.jsonl", all_events["formal"])
    write_jsonl(bundle / "loader_events.jsonl", all_loader_events["formal"])
    write_json(bundle / "benchmark_raw.json", {"schema": FORMAL_SCHEMA, "sample_kind": "formal", "trials": formal_trials})
    write_json(bundle / "benchmark_summary.json", benchmark_summary)
    (bundle / "benchmark_summary.csv").write_text("mode,metric\n", encoding="utf-8")
    write_json(bundle / "primary_gate.json", primary_gate)
    (bundle / "kernel_window.txt").write_text("clean\n", encoding="utf-8")
    write_json(
        bundle / "summary.json",
        {
            "schema": FORMAL_SCHEMA,
            "accepted": True,
            "phase": "formal",
            "formal": True,
            "performance_gate_passed": primary_gate["passed"],
            "expected_warmup_trials": 8,
            "completed_warmup_trials": 8,
            "expected_formal_trials": 80,
            "completed_formal_trials": 80,
            "fixture_sha256": SYNTHETIC_FIXTURE_SHA256,
            "output_checksum": base_trial["output_checksum"],
            "task_loss": 0,
            "kernel_log_errors": [],
            "systemd_failed_units": 0,
            "final_sched_ext_state": "disabled",
            "final_switch_all": "0",
            "failure_reason": None,
        },
    )
    seal_directory(bundle)
    return bundle


def test_formal_strict_verifier_accepts_and_is_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _formal_bundle(tmp_path, monkeypatch)
    before = (bundle / "SHA256SUMS").read_bytes()

    result = verify_formal_bundle(bundle)

    assert result["passed"] is True
    assert result["checks"] == "19/19"
    assert (bundle / "SHA256SUMS").read_bytes() == before


def test_formal_strict_verifier_rejects_warmup_leakage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _formal_bundle(tmp_path, monkeypatch)
    raw = json.loads((bundle / "benchmark_raw.json").read_text(encoding="utf-8"))
    raw["trials"].append(json.loads((bundle / "warmup_trials.jsonl").read_text().splitlines()[0]))
    write_json(bundle / "benchmark_raw.json", raw)
    seal_directory(bundle)

    result = verify_formal_bundle(bundle)

    assert result["passed"] is False
    assert "warmup_excluded_from_statistics" in result["failed_checks"]


def test_formal_verifier_rejects_resealed_fixture_tampering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _formal_bundle(tmp_path, monkeypatch)
    (bundle / "fixture" / "sample.py").write_text("# changed after freezing\n", encoding="utf-8")
    seal_directory(bundle)

    result = verify_formal_bundle(bundle)

    assert result["passed"] is False
    assert "fixture_verified" in result["failed_checks"]


def test_synthetic_bundle_cannot_pass_production_frozen_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _formal_bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(verifier, "FROZEN_FIXTURE_SHA256", FROZEN_FIXTURE_SHA256)

    result = verify_formal_bundle(bundle)

    assert result["passed"] is False
    assert "frozen_build_identity" in result["failed_checks"]
    assert "fixture_verified" in result["failed_checks"]
