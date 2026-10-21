#!/usr/bin/env python3
"""Run the five-mode real sched_ext Stage 4 ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentos.benchmarks.sched_ext_stage4 import (
    STAGE4_MODES,
    acceptance_passed,
    paired_improvement_95,
    smoke_acceptance_passed,
    summarize_trials,
)
from scripts.sched_ext_stage4_common import (
    CONTROLLER_CPU,
    WORKER_CPU_LIST,
    run_mixed_trial,
)
from scripts.verify_real_sched_ext import _snapshot, kernel_log_errors


PRESSURE_CPU = 3


def _worker_cpus() -> set[int]:
    return {int(cpu) for cpu in WORKER_CPU_LIST.split(",")}


def _validate_cpu_topology() -> list[int]:
    available = set(os.sched_getaffinity(0))
    worker_cpus = _worker_cpus()
    if PRESSURE_CPU in worker_cpus:
        raise RuntimeError(
            f"pressure CPU {PRESSURE_CPU} overlaps worker CPUs "
            f"{sorted(worker_cpus)}"
        )
    required = worker_cpus | {CONTROLLER_CPU, PRESSURE_CPU}
    missing = required - available
    if missing:
        raise RuntimeError(
            f"Stage 4 CPU topology requires CPUs {sorted(required)}; "
            f"missing {sorted(missing)} from affinity {sorted(available)}"
        )
    return sorted(available)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _seal(output_dir: Path) -> None:
    lines = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.relative_to(output_dir)}\n")
    (output_dir / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


def run_ablation(
    *,
    scheduler_binary: Path,
    simple_binary: Path,
    worker_binary: Path,
    output_dir: Path,
    policy_effects_summary: Path,
    iterations: int = 20,
    warmups: int = 2,
    duration_ms: int = 12_000,
    random_seed: int = 20260715,
) -> dict[str, Any]:
    if os.geteuid() != 0:
        raise PermissionError("Stage 4 ablation must run as root")
    if iterations not in {3, 20}:
        raise ValueError("iterations must be 3 for smoke or 20 for formal acceptance")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "trials"
    raw_dir.mkdir(exist_ok=True)
    for stale_trial in raw_dir.glob("*.json"):
        stale_trial.unlink()
    summary: dict[str, Any] = {
        "schema_version": 1,
        "accepted": False,
        "kernel_release": platform.release(),
        "iterations": iterations,
        "warmups": warmups,
        "duration_ms": duration_ms,
        "random_seed": random_seed,
        "cpu_topology": {
            "controller_cpu": CONTROLLER_CPU,
            "worker_cpus": sorted(_worker_cpus()),
            "pressure_cpu": PRESSURE_CPU,
        },
        "modes": {},
        "failure_reason": None,
    }
    trials: dict[str, list[dict[str, Any]]] = {mode: [] for mode in STAGE4_MODES}
    journal_since = time.time()
    background: subprocess.Popen[bytes] | None = None
    try:
        summary["cpu_topology"]["available_cpus"] = _validate_cpu_topology()
        (output_dir / "uname.txt").write_text(
            platform.platform() + "\n", encoding="utf-8"
        )
        (output_dir / "cmdline.txt").write_text(
            Path("/proc/cmdline").read_text(encoding="utf-8"), encoding="utf-8"
        )
        _write_json(output_dir / "sysfs-before.json", _snapshot())
        _write_json(
            output_dir / "artifact-sha256.json",
            [
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in (
                    scheduler_binary,
                    simple_binary,
                    worker_binary,
                    PROJECT_ROOT / "native/sched_ext/agentos_scx.bpf.c",
                    PROJECT_ROOT / "native/sched_ext/agentos_scx.c",
                    PROJECT_ROOT / "native/sched_ext/include/agentos_scx.h",
                    PROJECT_ROOT / "scripts/sched_ext_stage4_common.py",
                    Path(__file__).resolve(),
                )
            ],
        )
        background = subprocess.Popen(
            ["taskset", "-c", str(PRESSURE_CPU), "yes"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        for warmup in range(warmups):
            for mode in STAGE4_MODES:
                run_mixed_trial(
                    mode=mode,
                    scheduler_binary=scheduler_binary,
                    simple_binary=simple_binary,
                    worker_binary=worker_binary,
                    duration_ms=min(duration_ms, 3_000),
                    seed=10_000 + warmup,
                )

        rng = random.Random(random_seed)
        for run_index in range(iterations):
            block = list(STAGE4_MODES)
            rng.shuffle(block)
            for mode in block:
                trial = run_mixed_trial(
                    mode=mode,
                    scheduler_binary=scheduler_binary,
                    simple_binary=simple_binary,
                    worker_binary=worker_binary,
                    duration_ms=duration_ms,
                    seed=run_index + 1,
                )
                trial["run_index"] = run_index
                trials[mode].append(trial)
                serializable = dict(trial)
                loader_events = serializable.pop("loader_events", [])
                _write_json(raw_dir / f"{run_index:02d}-{mode}.json", serializable)
                _write_json(
                    raw_dir / f"{run_index:02d}-{mode}-loader-events.json",
                    loader_events,
                )

        summary["modes"] = {
            mode: summarize_trials(mode_trials)
            for mode, mode_trials in trials.items()
        }
        normal_p95 = [trial["latency_p95_us"] for trial in trials["agentos_normal"]]
        summary["paired_latency_improvement"] = {
            mode: paired_improvement_95(
                normal_p95,
                [trial["latency_p95_us"] for trial in trials[mode]],
            )
            for mode in (
                "agentos_static_3dsq",
                "agentos_full_dag_psi_fairness",
            )
        }
        expected_checksums = {
            trial["run_index"]: trial["validation_checksums"]
            for trial in trials["cfs"]
        }
        summary["checksum_consistent"] = all(
            trial["validation_checksums"] == expected_checksums[trial["run_index"]]
            for mode in STAGE4_MODES
            for trial in trials[mode]
        )
        summary["pressure_warning_observed"] = any(
            any(
                sample["pressure_level"] in {"warning", "critical"}
                for sample in trial.get("psi_trace", [])
            )
            for trial in trials["agentos_full_dag_psi_fairness"]
        )
        summary["warning_generation_verified"] = any(
            readback.get("pressure_level") == "warning"
            and str(readback.get("task_id", "")).startswith("batch-")
            and readback.get("readback", {}).get("weight") == 25
            for trial in trials["agentos_full_dag_psi_fairness"]
            for readback in trial.get("policy_readbacks", [])
        )
        agent_trials = [
            trial
            for mode in STAGE4_MODES
            if mode.startswith("agentos_")
            for trial in trials[mode]
        ]
        summary["policy_leaks"] = sum(trial["policy_leaks"] for trial in agent_trials)
        summary["task_state_leaks"] = sum(trial["task_state_leaks"] for trial in agent_trials)
        summary["batch_wait_max_ns"] = max(
            int(trial["stats"].get("wait_max_batch_ns", 0))
            for trial in agent_trials
        )
        summary["batch_process_max_gap_us"] = max(
            float(trial["batch_max_gap_us"])
            for mode in STAGE4_MODES
            for trial in trials[mode]
        )
        summary["worker_startup_max_us"] = max(
            float(trial["worker_startup_max_us"])
            for mode in STAGE4_MODES
            for trial in trials[mode]
        )
        effects = json.loads(policy_effects_summary.read_text(encoding="utf-8"))
        for key in (
            "weight_direction_verified",
            "slice_direction_verified",
            "queue_stats_verified",
        ):
            summary[key] = effects.get(key)
        services = subprocess.run(
            ["systemctl", "--failed", "--no-legend", "--plain"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        summary["systemd_failed_units"] = len(
            [line for line in services.stdout.splitlines() if line.strip()]
        )
        (output_dir / "systemd-failed.txt").write_text(
            services.stdout + services.stderr, encoding="utf-8"
        )
        journal = subprocess.run(
            [
                "journalctl", "-k", "-b", "--since", f"@{journal_since:.3f}",
                "--no-pager", "-o", "short-monotonic",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        (output_dir / "kernel-window.log").write_text(
            journal.stdout + journal.stderr, encoding="utf-8"
        )
        if journal.returncode:
            raise RuntimeError(f"journalctl failed with code {journal.returncode}")
        errors = kernel_log_errors(journal.stdout.splitlines())
        errors.extend(
            line for line in journal.stdout.splitlines()
            if any(term in line.lower() for term in (
                "vtime overflow", "runnable task loss", "lost runnable task"
            ))
        )
        summary["kernel_log_errors"] = errors
        summary["final_sched_ext_state"] = _snapshot().get("state")
        summary["accepted"] = (
            acceptance_passed(summary, expected_runs=iterations)
            if iterations == 20
            else smoke_acceptance_passed(summary, expected_runs=iterations)
        )
    except Exception as exc:
        summary["failure_reason"] = f"{type(exc).__name__}: {exc}"
        summary["final_sched_ext_state"] = _snapshot().get("state")
    finally:
        if background is not None and background.poll() is None:
            background.terminate()
            try:
                background.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                background.kill()
                background.communicate(timeout=2)
        _write_json(output_dir / "sysfs-after.json", _snapshot())
        _write_json(output_dir / "summary.json", summary)
        _seal(output_dir)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler-binary", type=Path, required=True)
    parser.add_argument("--simple-binary", type=Path, required=True)
    parser.add_argument("--worker-binary", type=Path, required=True)
    parser.add_argument("--policy-effects-summary", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs/member_a_sched_ext_validation/stage4_ablation",
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--duration-ms", type=int, default=12_000)
    args = parser.parse_args()
    summary = run_ablation(
        scheduler_binary=args.scheduler_binary.resolve(),
        simple_binary=args.simple_binary.resolve(),
        worker_binary=args.worker_binary.resolve(),
        policy_effects_summary=args.policy_effects_summary.resolve(),
        output_dir=args.output_dir.resolve(),
        iterations=args.iterations,
        warmups=args.warmups,
        duration_ms=args.duration_ms,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
