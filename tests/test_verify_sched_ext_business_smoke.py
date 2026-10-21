from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

from agentos.benchmarks.sched_ext_business import (
    seal_directory,
    write_json,
    write_jsonl,
)
from agentos.benchmarks.sched_ext_business_smoke import (
    SMOKE_MODES,
    assess_trial_evidence,
    expected_policy_queues,
)
from agentos.scenarios.sched_ext_business.pipeline import BUSINESS_TASK_IDS
from scripts.run_sched_ext_business_demo import run_functional_demo
from scripts.run_sched_ext_business_smoke import SMOKE_SCHEMA
from scripts.verify_sched_ext_business_smoke import verify_smoke_bundle


def _source(root: Path) -> list[str]:
    root.mkdir()
    (root / "code.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    (root / "README.md").write_text("# Fixture\n", encoding="utf-8")
    return ["README.md", "code.py"]


def _smoke_bundle(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    functional = tmp_path / "functional"
    run_functional_demo(
        source_root=source,
        output_dir=functional,
        fixture_paths=_source(source),
        rounds=1,
        seed=1,
        work_units=1,
    )
    base_trial = json.loads((functional / "trials.jsonl").read_text(encoding="utf-8"))
    base_events = [
        json.loads(line)
        for line in (functional / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads((functional / "fixture_manifest.json").read_text(encoding="utf-8"))

    bundle = tmp_path / "smoke"
    bundle.mkdir()
    shutil.copytree(functional / "fixture", bundle / "fixture")
    write_json(bundle / "fixture_manifest.json", manifest)
    write_json(
        bundle / "environment.json",
        {
            "schema": SMOKE_SCHEMA,
            "phase": "functional",
            "formal": False,
            "kernel_release": "6.12.95-agentos-scx",
            "controller_cpu": 0,
            "worker_cpus": [1, 2],
            "pressure_cpu": 3,
            "modes": list(SMOKE_MODES),
        },
    )
    write_json(bundle / "preflight.json", {"accepted": True})
    write_json(bundle / "build_manifest.json", {"schema": SMOKE_SCHEMA})
    trials = []
    events = []
    loader_events = []
    queue_codes = {"latency": 1, "normal": 2, "batch": 3}
    for order, mode in enumerate(SMOKE_MODES):
        trial = copy.deepcopy(base_trial)
        trial.update(
            schema=SMOKE_SCHEMA,
            mode=mode,
            round=0,
            order=order,
            policy_associations=0 if mode == "cfs" else 8,
            policy_queues=expected_policy_queues(mode),
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
        trial_events = [
            {**event, "benchmark_mode": mode, "benchmark_round": 0}
            for event in copy.deepcopy(base_events)
        ]
        for task_id, result in trial["task_results"].items():
            observation = result["runtime_observation"]
            observation.update(
                policy=0 if mode == "cfs" else 7,
                ext_enabled=0 if mode == "cfs" else 1,
                affinity=[1, 2],
            )
            if mode != "cfs":
                queue = expected_policy_queues(mode)[task_id]
                trial_events.append(
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
                        "benchmark_round": 0,
                    }
                )
        if mode == "cfs":
            trial["stats_delta"] = {}
        else:
            delta = {
                "policy_hit": 8,
                "dispatch_normal": 8,
                "dispatch_latency": 0 if mode == "agentos-normal" else 2,
                "dispatch_batch": 0 if mode == "agentos-normal" else 4,
            }
            trial["stats_delta"] = delta
            loader_events.extend(
                [
                    {
                        "event": "ready",
                        "benchmark_mode": mode,
                        "benchmark_round": 0,
                    },
                    {
                        "event": "stats",
                        "stats": {name: 0 for name in delta},
                        "benchmark_mode": mode,
                        "benchmark_round": 0,
                    },
                    {
                        "event": "final_stats",
                        "stats": delta,
                        "benchmark_mode": mode,
                        "benchmark_round": 0,
                    },
                ]
            )
        trial["evidence_checks"] = assess_trial_evidence(trial)
        trials.append(trial)
        events.extend(trial_events)

    write_jsonl(bundle / "trials.jsonl", trials)
    write_jsonl(bundle / "events.jsonl", events)
    write_jsonl(bundle / "loader_events.jsonl", loader_events)
    write_json(bundle / "benchmark_raw.json", {"schema": SMOKE_SCHEMA, "trials": trials})
    write_json(bundle / "benchmark_summary.json", {})
    (bundle / "benchmark_summary.csv").write_text("mode,metric\n", encoding="utf-8")
    (bundle / "kernel_window.txt").write_text("clean\n", encoding="utf-8")
    write_json(
        bundle / "summary.json",
        {
            "schema": SMOKE_SCHEMA,
            "accepted": True,
            "phase": "functional",
            "formal": False,
            "expected_trials": 4,
            "completed_trials": 4,
            "fixture_sha256": manifest["aggregate_sha256"],
            "output_checksum": base_trial["output_checksum"],
            "task_loss": 0,
            "kernel_log_errors": [],
            "systemd_failed_units": 0,
            "final_sched_ext_state": "disabled",
            "failure_reason": None,
        },
    )
    seal_directory(bundle)
    return bundle


def test_smoke_strict_verifier_accepts_complete_read_only_bundle(tmp_path: Path) -> None:
    bundle = _smoke_bundle(tmp_path)
    before = (bundle / "SHA256SUMS").read_bytes()

    result = verify_smoke_bundle(bundle)

    assert result["passed"] is True
    assert result["checks"] == "15/15"
    assert (bundle / "SHA256SUMS").read_bytes() == before


def test_smoke_strict_verifier_rejects_queue_tamper_and_residual(tmp_path: Path) -> None:
    bundle = _smoke_bundle(tmp_path)
    trials = [json.loads(line) for line in (bundle / "trials.jsonl").read_text().splitlines()]
    trials[2]["policy_queues"]["patch_validation"] = "normal"
    trials[2]["cleanup"]["policies"] = 1
    write_jsonl(bundle / "trials.jsonl", trials)
    seal_directory(bundle)

    result = verify_smoke_bundle(bundle)

    assert result["passed"] is False
    assert "runtime_evidence" in result["failed_checks"]
