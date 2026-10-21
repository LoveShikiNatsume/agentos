#!/usr/bin/env python3
"""Run the authorized four-mode P2 business smoke on the fixed openEuler host."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentos.benchmarks.sched_ext_business import (
    BUSINESS_BUNDLE_SCHEMA,
    DEFAULT_FIXTURE_PATHS,
    FROZEN_FIXTURE_MANIFEST,
    FROZEN_FIXTURE_VERSION,
    seal_directory,
    write_json,
    write_jsonl,
    write_summary_csv,
)
from agentos.benchmarks.sched_ext_business_smoke import (
    SMOKE_MODES,
    assess_trial_evidence,
    build_smoke_plan,
    has_complete_mode_round_pairs,
    smoke_round_seed,
)
from agentos.isolation.kernel_adapter import UserSpaceKernelAdapter
from agentos.isolation.sched_ext_adapter import SchedExtKernelAdapter
from agentos.isolation.sched_ext_control import SchedExtControlClient
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import Runtime
from agentos.scenarios.sched_ext_business.fixture import FixtureManifest, freeze_fixture
from agentos.scenarios.sched_ext_business.metrics import (
    compute_trial_metrics,
    paired_improvements,
    summarise_trials,
)
from agentos.scenarios.sched_ext_business.pipeline import (
    BUSINESS_TASK_IDS,
    build_business_dag,
)
from agentos.worker.process import SubprocessWorkerManager
from scripts.preflight_sched_ext import probe_sched_ext
from scripts.sched_ext_stage4_common import wait_worker_barrier
from scripts.verify_real_sched_ext import (
    _snapshot,
    _start_loader,
    _stop_loader,
    kernel_log_errors,
)


SMOKE_SCHEMA = "sched_ext_business_smoke.v1"
CONTROLLER_CPU = 0
WORKER_CPUS = {1, 2}
PRESSURE_CPU = 3
QUEUE_CODES = {1: "latency", 2: "normal", 3: "batch"}


def _git_value(*args: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _map_count(path: Path) -> int:
    completed = subprocess.run(
        ["bpftool", "-j", "map", "dump", "pinned", str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return len(json.loads(completed.stdout))


def _stats_delta(events: list[dict[str, Any]]) -> dict[str, int]:
    initial = next(
        (event.get("stats", {}) for event in events if event.get("event") == "stats"),
        {},
    )
    final = next(
        (
            event.get("stats", {})
            for event in reversed(events)
            if event.get("event") == "final_stats"
        ),
        {},
    )
    return {
        name: int(final.get(name, 0)) - int(initial.get(name, 0))
        for name in sorted(set(initial) | set(final))
    }


def _start_pressure(seed: int) -> subprocess.Popen[bytes]:
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "agentos.benchmarks.sched_ext_business_pressure",
            "--cpu",
            str(PRESSURE_CPU),
            "--seed",
            str(seed),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )
    try:
        wait_worker_barrier(process)
        assert process.stdin is not None
        process.stdin.write(b"1")
        process.stdin.flush()
        return process
    except Exception:
        process.kill()
        process.communicate()
        raise


def _stop_pressure(process: subprocess.Popen[bytes]) -> dict[str, Any]:
    process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=10)
    lines = [line for line in stdout.decode("utf-8").splitlines() if line.strip()]
    if process.returncode != 0 or len(lines) != 1:
        raise RuntimeError(
            f"pressure worker failed: rc={process.returncode}, stderr={stderr!r}"
        )
    result = json.loads(lines[0])
    result["return_code"] = process.returncode
    result["stderr"] = stderr.decode("utf-8", errors="replace")
    return result


def _policy_evidence(events: list[dict[str, Any]]) -> tuple[int, dict[str, str]]:
    queues: dict[str, str] = {}
    associations = 0
    for event in events:
        if event.get("event_type") != "kernel.sched_ext_policy_published":
            continue
        task_id = event.get("task_id")
        if task_id not in BUSINESS_TASK_IDS:
            continue
        data = event.get("data", {})
        policy = data.get("policy", {}) if isinstance(data, dict) else {}
        if data.get("readback_verified") is True:
            associations += 1
        queues[str(task_id)] = QUEUE_CODES.get(int(policy.get("queue_class", 0)), "unknown")
    return associations, queues


def _alive_worker_count(events: list[dict[str, Any]]) -> int:
    pids = {
        int(event.get("data", {}).get("pid"))
        for event in events
        if event.get("event_type") == "worker.started"
        and isinstance(event.get("data", {}).get("pid"), int)
    }
    return sum(Path(f"/proc/{pid}").exists() for pid in pids)


async def _run_runtime(
    *,
    mode: str,
    fixture_root: Path,
    manifest: FixtureManifest,
    work_units: int,
    adapter: SchedExtKernelAdapter | UserSpaceKernelAdapter,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    dag = build_business_dag(
        fixture_root,
        manifest,
        work_units=work_units,
        kernel_queue_mode="normal" if mode == "agentos-normal" else "sla",
    )
    runtime = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={},
        execution_mode="subprocess",
        max_batch_size=len(BUSINESS_TASK_IDS),
        worker_manager=SubprocessWorkerManager(
            allowed_cpus=WORKER_CPUS,
            startup_barrier=True,
        ),
        enable_kernel_probe=False,
        kernel_adapter=adapter,
    )
    results = await runtime.run(dag)
    return (
        results,
        [event.to_dict() for event in runtime.events],
        adapter.snapshot(),
    )


def _trial(
    *,
    mode: str,
    round_id: int,
    order: int,
    fixture_root: Path,
    manifest: FixtureManifest,
    scheduler_binary: Path,
    work_units: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    original_affinity = set(os.sched_getaffinity(0))
    loader: subprocess.Popen[str] | None = None
    pressure: subprocess.Popen[bytes] | None = None
    loader_events: list[dict[str, Any]] = []
    loader_exit_code: int | None = None
    pressure_result: dict[str, Any] = {}
    control: SchedExtControlClient | None = None
    adapter: SchedExtKernelAdapter | UserSpaceKernelAdapter
    runtime_events: list[dict[str, Any]] = []
    adapter_snapshot: dict[str, Any] = {}
    try:
        os.sched_setaffinity(0, {CONTROLLER_CPU})
        if mode == "cfs":
            adapter = UserSpaceKernelAdapter()
        else:
            loader, loader_events = _start_loader(scheduler_binary)
            enabled = _snapshot()
            if enabled.get("ops") != "agentos_scx" or enabled.get("switch_all") != "0":
                raise RuntimeError(f"unexpected sched_ext state after loader: {enabled}")
            control = SchedExtControlClient()
            if mode == "agentos-static-3dsq":
                missing = Path("/run/agentos-p2-static-no-psi")
                psi_paths = {name: missing / name for name in ("cpu", "memory", "io")}
                base = UserSpaceKernelAdapter(psi_paths=psi_paths)
            else:
                base = UserSpaceKernelAdapter()
            adapter = SchedExtKernelAdapter(
                control=control,
                base=base,
                mode="strict",
                config_flags=1 if mode == "agentos-full" else 0,
            )

        pressure = _start_pressure(seed)
        results, runtime_events, adapter_snapshot = asyncio.run(
            _run_runtime(
                mode=mode,
                fixture_root=fixture_root,
                manifest=manifest,
                work_units=work_units,
                adapter=adapter,
            )
        )
        pressure_result = _stop_pressure(pressure)
        pressure = None

        policy_count = task_state_count = 0
        if control is not None:
            policy_count = _map_count(control.task_policy_path)
            time.sleep(0.1)
            task_state_count = _map_count(control.task_state_path)
        if loader is not None:
            loader_exit_code, _stdout, loader_stderr = _stop_loader(
                loader, loader_events, signal.SIGTERM
            )
            loader = None
            if loader_exit_code != 0:
                raise RuntimeError(f"loader exited with {loader_exit_code}: {loader_stderr}")

        task_results = {task_id: results[task_id] for task_id in BUSINESS_TASK_IDS}
        metrics = compute_trial_metrics(
            runtime_events,
            task_results,
            deadlines_ms={"patch_validation": 2_000, "security_gate": 2_000},
            work_units={task_id: work_units for task_id in BUSINESS_TASK_IDS},
        )
        associations, policy_queues = _policy_evidence(runtime_events)
        cleanup = {
            "workers": _alive_worker_count(runtime_events),
            "cgroups": len(adapter_snapshot.get("agent_cgroups", {})),
            "policies": policy_count,
            "task_states": task_state_count,
            "bound_pids": len(adapter_snapshot.get("bound_pids", [])),
            "loader": 0,
            "sched_ext_state": _snapshot().get("state"),
        }
        trial = {
            "schema": SMOKE_SCHEMA,
            "mode": mode,
            "round": round_id,
            "order": order,
            "seed": seed,
            "fixture_sha256": manifest.aggregate_sha256,
            "fixture_version": FROZEN_FIXTURE_VERSION,
            "work_units": {task_id: work_units for task_id in BUSINESS_TASK_IDS},
            "task_results": task_results,
            **metrics,
            "policy_associations": associations,
            "policy_queues": policy_queues,
            "stats_delta": _stats_delta(loader_events) if loader_events else {},
            "loader_exit_code": loader_exit_code,
            "pressure": pressure_result,
            "cleanup": cleanup,
        }
        trial["evidence_checks"] = assess_trial_evidence(trial)
        return trial, runtime_events, loader_events
    finally:
        if pressure is not None and pressure.poll() is None:
            pressure.send_signal(signal.SIGTERM)
            pressure.communicate(timeout=10)
        if loader is not None and loader.poll() is None:
            _stop_loader(loader, loader_events, signal.SIGTERM)
        os.sched_setaffinity(0, original_affinity)


def run_smoke(
    *,
    phase: str,
    scheduler_binary: Path,
    output_dir: Path,
    work_units: int,
    seed: int,
) -> dict[str, Any]:
    if os.geteuid() != 0:
        raise PermissionError("P2 openEuler smoke must run as root")
    plan = build_smoke_plan(phase)
    required_affinity = {CONTROLLER_CPU, *WORKER_CPUS, PRESSURE_CPU}
    if not required_affinity.issubset(os.sched_getaffinity(0)):
        raise RuntimeError("P2 smoke requires available CPUs 0,1,2,3")
    if platform.release() != "6.12.95-agentos-scx":
        raise RuntimeError(f"unexpected kernel: {platform.release()}")
    if not 1 <= work_units <= 10_000:
        raise ValueError("work_units must be between 1 and 10000")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if not scheduler_binary.is_file():
        raise FileNotFoundError(str(scheduler_binary))

    output_dir.mkdir(parents=True)
    journal_since = int(time.time())
    preflight = probe_sched_ext()
    write_json(output_dir / "preflight.json", preflight)
    if preflight.get("accepted") is not True:
        raise RuntimeError(f"sched_ext preflight rejected host: {preflight.get('failures')}")

    fixture_root = output_dir / "fixture"
    manifest = freeze_fixture(PROJECT_ROOT, fixture_root, DEFAULT_FIXTURE_PATHS)
    expected = FixtureManifest.from_dict(
        json.loads(FROZEN_FIXTURE_MANIFEST.read_text(encoding="utf-8"))
    )
    if manifest != expected:
        raise ValueError("source does not match frozen fixture v1")
    write_json(output_dir / "fixture_manifest.json", manifest.to_dict())
    write_json(
        output_dir / "environment.json",
        {
            "schema": SMOKE_SCHEMA,
            "phase": phase,
            "host": platform.node(),
            "kernel_release": platform.release(),
            "python": platform.python_version(),
            "controller_cpu": CONTROLLER_CPU,
            "worker_cpus": sorted(WORKER_CPUS),
            "pressure_cpu": PRESSURE_CPU,
            "modes": list(SMOKE_MODES),
            "functional_only": False,
            "formal": False,
        },
    )
    write_json(
        output_dir / "build_manifest.json",
        {
            "schema": SMOKE_SCHEMA,
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_dirty": bool(_git_value("status", "--porcelain")),
            "scheduler_binary": str(scheduler_binary),
            "scheduler_sha256": hashlib.sha256(scheduler_binary.read_bytes()).hexdigest(),
            "fixture_sha256": manifest.aggregate_sha256,
            "work_units": work_units,
            "seed": seed,
        },
    )

    trials: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    loader_events: list[dict[str, Any]] = []
    failure_reason: str | None = None
    try:
        for item in plan:
            trial_seed = smoke_round_seed(seed, int(item["round"]))
            trial, trial_events, trial_loader_events = _trial(
                mode=str(item["mode"]),
                round_id=int(item["round"]),
                order=int(item["order"]),
                fixture_root=fixture_root,
                manifest=manifest,
                scheduler_binary=scheduler_binary,
                work_units=work_units,
                seed=trial_seed,
            )
            trials.append(trial)
            events.extend(
                {
                    **event,
                    "benchmark_mode": trial["mode"],
                    "benchmark_round": trial["round"],
                }
                for event in trial_events
            )
            loader_events.extend(
                {
                    **event,
                    "benchmark_mode": trial["mode"],
                    "benchmark_round": trial["round"],
                }
                for event in trial_loader_events
            )
    except Exception as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"

    journal = subprocess.run(
        ["journalctl", "-k", "-b", f"--since=@{journal_since}", "--no-pager"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    (output_dir / "kernel_window.txt").write_text(
        journal.stdout + journal.stderr, encoding="utf-8"
    )
    kernel_errors = kernel_log_errors(journal.stdout.splitlines())
    systemd = subprocess.run(
        ["systemctl", "--failed", "--no-legend", "--plain"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    failed_units = [line for line in systemd.stdout.splitlines() if line.strip()]

    benchmark_summary = summarise_trials(trials) if trials else {}
    paired = {}
    if trials and has_complete_mode_round_pairs(trials, phase):
        paired = {
            mode: paired_improvements(trials, baseline="cfs", candidate=mode)
            for mode in SMOKE_MODES
            if mode != "cfs"
        }
    checks = [assess_trial_evidence(trial) for trial in trials]
    output_checksums = {trial.get("output_checksum") for trial in trials}
    accepted = bool(
        failure_reason is None
        and len(trials) == len(plan)
        and checks
        and all(all(item.values()) for item in checks)
        and len(output_checksums) == 1
        and kernel_errors == []
        and failed_units == []
        and journal.returncode == 0
        and _snapshot().get("state") == "disabled"
    )
    summary = {
        "schema": SMOKE_SCHEMA,
        "accepted": accepted,
        "phase": phase,
        "formal": False,
        "modes": list(SMOKE_MODES),
        "expected_trials": len(plan),
        "completed_trials": len(trials),
        "fixture_sha256": manifest.aggregate_sha256,
        "output_checksum": next(iter(output_checksums)) if len(output_checksums) == 1 else None,
        "task_loss": sum(int(trial.get("task_loss", 0)) for trial in trials),
        "kernel_log_errors": kernel_errors,
        "systemd_failed_units": len(failed_units),
        "final_sched_ext_state": _snapshot().get("state"),
        "failure_reason": failure_reason,
    }
    write_jsonl(output_dir / "trials.jsonl", trials)
    write_jsonl(output_dir / "events.jsonl", events)
    write_jsonl(output_dir / "loader_events.jsonl", loader_events)
    write_json(output_dir / "benchmark_raw.json", {"schema": SMOKE_SCHEMA, "trials": trials})
    write_json(
        output_dir / "benchmark_summary.json",
        {"modes": benchmark_summary, "paired_improvements": paired},
    )
    write_summary_csv(output_dir / "benchmark_summary.csv", benchmark_summary)
    write_json(output_dir / "summary.json", summary)
    seal_directory(output_dir)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("functional", "statistical"), required=True)
    parser.add_argument("--scheduler-binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-units", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args(argv)
    summary = run_smoke(
        phase=args.phase,
        scheduler_binary=args.scheduler_binary.resolve(),
        output_dir=args.output_dir.resolve(),
        work_units=args.work_units,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
