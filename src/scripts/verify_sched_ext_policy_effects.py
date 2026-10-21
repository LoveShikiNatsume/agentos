#!/usr/bin/env python3
"""Verify real Stage 4 queue, weight, slice, and fallback behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentos.isolation.bpf_schema import SchedPolicyKeyV1
from agentos.isolation.sched_ext_control import SchedExtControlClient
from scripts.preflight_sched_ext import probe_sched_ext
from scripts.sched_ext_stage4_common import run_mixed_trial, wait_worker_barrier
from scripts.verify_real_sched_ext import (
    _snapshot,
    _start_loader,
    _stop_loader,
    kernel_log_errors,
)

_POLICY = struct.Struct("<IIIIIIQQ")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _seal(output_dir: Path) -> None:
    lines = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.relative_to(output_dir)}\n")
    (output_dir / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


def _write_environment(
    output_dir: Path, artifacts: list[Path]
) -> None:
    (output_dir / "uname.txt").write_text(platform.platform() + "\n", encoding="utf-8")
    (output_dir / "cmdline.txt").write_text(
        Path("/proc/cmdline").read_text(encoding="utf-8"), encoding="utf-8"
    )
    _write_json(output_dir / "sysfs-before.json", _snapshot())
    hashes = []
    for path in artifacts:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes.append({"path": str(path), "sha256": digest})
    _write_json(output_dir / "artifact-sha256.json", hashes)


def _raw_policy_update(path: Path, key: SchedPolicyKeyV1, value: bytes) -> None:
    command = [
        "bpftool", "map", "update", "pinned", str(path),
        "key", "hex", *key.to_hex_tokens(),
        "value", "hex", *[f"{byte:02x}" for byte in value], "any",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)


def _policy_bytes(
    *,
    version: int = 1,
    queue: int = 2,
    weight: int = 100,
    slice_us: int = 5_000,
    pressure: int = 1,
    flags: int = 1,
    generation: int = 4242,
    updated_at_ns: int | None = None,
) -> bytes:
    return _POLICY.pack(
        version, queue, weight, slice_us, pressure, flags,
        generation,
        time.monotonic_ns() if updated_at_ns is None else updated_at_ns,
    )


def _final_stats(events: list[dict[str, Any]]) -> dict[str, int]:
    for event in reversed(events):
        if event.get("event") == "final_stats":
            return {key: int(value) for key, value in event.get("stats", {}).items()}
    raise RuntimeError("loader did not emit final_stats")


def _run_fallback_cases(
    scheduler_binary: Path, worker_binary: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    loader, events = _start_loader(scheduler_binary)
    control = SchedExtControlClient()
    generation = 4242
    control.validate_ready()
    control.update_global_config(generation=generation)
    initial = {
        key: int(value)
        for key, value in events[-1].get("stats", {}).items()
    }
    definitions: list[tuple[str, bytes | None, bool]] = [
        ("lookup", None, False),
        ("version", _policy_bytes(version=2), False),
        ("queue", _policy_bytes(queue=9), False),
        ("weight", _policy_bytes(weight=0), False),
        ("slice", _policy_bytes(slice_us=99), False),
        ("generation", _policy_bytes(generation=generation - 1), False),
        ("policy_stale", _policy_bytes(updated_at_ns=1), False),
        ("heartbeat", _policy_bytes(), True),
    ]
    results: list[dict[str, Any]] = []
    try:
        for index, (name, raw_policy, stale_heartbeat) in enumerate(definitions):
            process = subprocess.Popen(
                [
                    str(worker_binary), "--role", "normal",
                    "--duration-ms", "300", "--cpu", "1",
                    "--seed", str(90_000 + index),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
            )
            wait_worker_barrier(process)
            key = SchedPolicyKeyV1(tgid=process.pid, pid=process.pid)
            if raw_policy is not None:
                _raw_policy_update(control.task_policy_path, key, raw_policy)
            if stale_heartbeat:
                control.update_global_config(generation=generation, heartbeat_ns=1)
            os.sched_setscheduler(process.pid, 7, os.sched_param(0))
            configured_policy = os.sched_getscheduler(process.pid)
            assert process.stdin is not None
            process.stdin.write(b"1")
            process.stdin.close()
            process.stdin = None
            stdout, stderr = process.communicate(timeout=10)
            payload = json.loads(stdout.decode("utf-8"))
            payload.update(
                case=name,
                configured_policy=configured_policy,
                return_code=process.returncode,
                stderr=stderr.decode("utf-8", errors="replace"),
            )
            results.append(payload)
            if raw_policy is not None:
                control.delete_policy(key)
            if stale_heartbeat:
                control.update_global_config(generation=generation)
        time.sleep(0.2)
        policy_leaks = len(
            json.loads(
                subprocess.run(
                    ["bpftool", "-j", "map", "dump", "pinned", str(control.task_policy_path)],
                    check=True, capture_output=True, text=True, timeout=10,
                ).stdout
            )
        )
        state_leaks = len(
            json.loads(
                subprocess.run(
                    ["bpftool", "-j", "map", "dump", "pinned", str(control.task_state_path)],
                    check=True, capture_output=True, text=True, timeout=10,
                ).stdout
            )
        )
        loader_rc, stdout, stderr = _stop_loader(loader, events, signal.SIGTERM)
        loader = None
        if loader_rc:
            raise RuntimeError(f"fallback loader exited with {loader_rc}")
        final = _final_stats(events)
        delta = {key: final.get(key, 0) - initial.get(key, 0) for key in final}
        return {
            "cases": results,
            "all_completed": all(
                item.get("completed") is True and item.get("return_code") == 0
                and item.get("configured_policy") == 7
                for item in results
            ),
            "stats_before": initial,
            "stats_after": final,
            "stats_delta": delta,
            "policy_leaks": policy_leaks,
            "task_state_leaks": state_leaks,
            "loader_exit_code": loader_rc,
        }, events, stderr
    finally:
        if loader is not None and loader.poll() is None:
            _stop_loader(loader, events, signal.SIGTERM)


def policy_effects_accepted(summary: dict[str, Any]) -> bool:
    fallbacks = summary.get("fallback_stats_delta", {})
    return bool(
        summary.get("preflight_accepted") is True
        and summary.get("queue_stats_verified") is True
        and summary.get("weight_direction_verified") is True
        and summary.get("slice_direction_verified") is True
        and summary.get("fallback_cases_completed") is True
        and fallbacks.get("fallback_lookup", 0) > 0
        and fallbacks.get("fallback_version", 0) > 0
        and fallbacks.get("fallback_range", 0) >= 3
        and fallbacks.get("fallback_generation", 0) > 0
        and fallbacks.get("fallback_heartbeat", 0) > 0
        and fallbacks.get("fallback_policy_stale", 0) > 0
        and summary.get("policy_leaks") == 0
        and summary.get("task_state_leaks") == 0
        and summary.get("batch_wait_max_ns", 1 << 63) <= 250_000_000
        and summary.get("kernel_log_errors") == []
        and summary.get("systemd_failed_units") == 0
        and summary.get("final_sched_ext_state") == "disabled"
        and summary.get("failure_reason") is None
    )


def verify_policy_effects(
    *,
    scheduler_binary: Path,
    simple_binary: Path,
    worker_binary: Path,
    output_dir: Path,
    duration_ms: int = 2_000,
) -> dict[str, Any]:
    if os.geteuid() != 0:
        raise PermissionError("Stage 4 policy verification must run as root")
    output_dir.mkdir(parents=True, exist_ok=True)
    journal_since = time.time()
    summary: dict[str, Any] = {
        "schema_version": 1,
        "accepted": False,
        "kernel_release": platform.release(),
        "failure_reason": None,
    }
    try:
        _write_environment(
            output_dir,
            [
                scheduler_binary,
                simple_binary,
                worker_binary,
                PROJECT_ROOT / "native/sched_ext/agentos_scx.bpf.c",
                PROJECT_ROOT / "native/sched_ext/agentos_scx.c",
                PROJECT_ROOT / "native/sched_ext/include/agentos_scx.h",
                Path(__file__).resolve(),
            ],
        )
        preflight = probe_sched_ext()
        _write_json(output_dir / "preflight.json", preflight)
        summary["preflight_accepted"] = preflight.get("accepted") is True
        if not summary["preflight_accepted"]:
            raise RuntimeError("sched_ext preflight was not accepted")

        static = run_mixed_trial(
            mode="agentos_static_3dsq", scheduler_binary=scheduler_binary,
            simple_binary=simple_binary, worker_binary=worker_binary,
            duration_ms=duration_ms, seed=4101,
        )
        weight = run_mixed_trial(
            mode="agentos_static_3dsq", scheduler_binary=scheduler_binary,
            simple_binary=simple_binary, worker_binary=worker_binary,
            duration_ms=duration_ms, seed=4102,
            cpu_overrides={"normal-0": 1, "normal-1": 1},
            policy_overrides={
                "normal-0": {"weight": 400},
                "normal-1": {"weight": 100},
            },
        )
        short_slice = run_mixed_trial(
            mode="agentos_normal", scheduler_binary=scheduler_binary,
            simple_binary=simple_binary, worker_binary=worker_binary,
            duration_ms=duration_ms, seed=4103,
            policy_overrides={task: {"slice_us": 2_000} for task in (
                "latency-0", "normal-0", "normal-1", "batch-0", "batch-1"
            )},
        )
        long_slice = run_mixed_trial(
            mode="agentos_normal", scheduler_binary=scheduler_binary,
            simple_binary=simple_binary, worker_binary=worker_binary,
            duration_ms=duration_ms, seed=4103,
            policy_overrides={task: {"slice_us": 10_000} for task in (
                "latency-0", "normal-0", "normal-1", "batch-0", "batch-1"
            )},
        )
        for name, trial in (
            ("static-3dsq", static), ("weight-400-100", weight),
            ("slice-2ms", short_slice), ("slice-10ms", long_slice),
        ):
            events = trial.pop("loader_events", [])
            _write_json(output_dir / f"{name}.json", trial)
            _write_json(output_dir / f"{name}-loader-events.json", events)

        queue_stats = static["stats"]
        summary["queue_stats_verified"] = all(
            queue_stats.get(f"enqueue_{queue}", 0) > 0
            and queue_stats.get(f"dispatch_{queue}", 0) > 0
            and queue_stats.get(f"runtime_{queue}_ns", 0) > 0
            and queue_stats.get(f"wait_total_{queue}_ns", 0) > 0
            for queue in ("latency", "normal", "batch")
        ) and queue_stats.get("policy_hit", 0) > 0
        weight_workers = {item["task_id"]: item for item in weight["workers"]}
        summary["weight_iterations"] = {
            task: int(weight_workers[task]["iterations"])
            for task in ("normal-0", "normal-1")
        }
        summary["weight_direction_verified"] = (
            summary["weight_iterations"]["normal-0"]
            > summary["weight_iterations"]["normal-1"]
        )
        summary["slice_dispatches"] = {
            "2ms": int(short_slice["stats"].get("dispatch_normal", 0)),
            "10ms": int(long_slice["stats"].get("dispatch_normal", 0)),
        }
        summary["slice_direction_verified"] = (
            summary["slice_dispatches"]["2ms"]
            > summary["slice_dispatches"]["10ms"]
        )
        summary["batch_wait_max_ns"] = int(queue_stats.get("wait_max_batch_ns", 0))

        fallback, fallback_events, fallback_stderr = _run_fallback_cases(
            scheduler_binary, worker_binary
        )
        _write_json(output_dir / "fallback-cases.json", fallback)
        _write_json(output_dir / "fallback-loader-events.json", fallback_events)
        (output_dir / "fallback-loader-stderr.log").write_text(
            fallback_stderr, encoding="utf-8"
        )
        summary["fallback_cases_completed"] = fallback["all_completed"]
        summary["fallback_stats_delta"] = fallback["stats_delta"]
        summary["policy_leaks"] = fallback["policy_leaks"]
        summary["task_state_leaks"] = fallback["task_state_leaks"]

        services = subprocess.run(
            ["systemctl", "--failed", "--no-legend", "--plain"],
            check=False, capture_output=True, text=True, timeout=15,
        )
        (output_dir / "systemd-failed.txt").write_text(
            services.stdout + services.stderr, encoding="utf-8"
        )
        summary["systemd_failed_units"] = len(
            [line for line in services.stdout.splitlines() if line.strip()]
        )
        journal = subprocess.run(
            [
                "journalctl", "-k", "-b", "--since", f"@{journal_since:.3f}",
                "--no-pager", "-o", "short-monotonic",
            ],
            check=False, capture_output=True, text=True, timeout=30,
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
        summary["accepted"] = policy_effects_accepted(summary)
    except Exception as exc:
        summary["failure_reason"] = f"{type(exc).__name__}: {exc}"
        summary["final_sched_ext_state"] = _snapshot().get("state")
    finally:
        _write_json(output_dir / "sysfs-after.json", _snapshot())
        _write_json(output_dir / "summary.json", summary)
        _seal(output_dir)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler-binary", type=Path, required=True)
    parser.add_argument("--simple-binary", type=Path, required=True)
    parser.add_argument("--worker-binary", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "runs/member_a_sched_ext_validation/stage4_policy_effects",
    )
    parser.add_argument("--duration-ms", type=int, default=2_000)
    args = parser.parse_args()
    summary = verify_policy_effects(
        scheduler_binary=args.scheduler_binary.resolve(),
        simple_binary=args.simple_binary.resolve(),
        worker_binary=args.worker_binary.resolve(),
        output_dir=args.output_dir.resolve(),
        duration_ms=args.duration_ms,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
