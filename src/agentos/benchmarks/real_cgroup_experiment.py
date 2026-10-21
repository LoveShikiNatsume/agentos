from __future__ import annotations

import hashlib
import csv
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentos.benchmarks.experiment_stats import describe
from agentos.core.resources import ResourceQuota
from agentos.isolation.bpf import MockBPFMapAdapter
from agentos.isolation.cgroup import CgroupV2Manager
from agentos.isolation.kernel_adapter import UserSpaceKernelAdapter
from agentos.isolation.sched_ext import KernelProbeResult


SCHEMA_VERSION = "agentos.stage5.real-cgroup.v1"
SCENARIOS = ("cpu", "memory", "pids")
MODES = ("unrestricted", "controlled")
METRICS = (
    "cpu_usage_usec",
    "memory_peak_mb",
    "oom_kill_count",
    "children_spawned",
    "spawn_failures",
    "wall_time_secs",
)


@dataclass(frozen=True)
class RealCgroupTrialPlanEntry:
    trial_id: str
    scenario: str
    mode: str
    repetition: int
    sub_seed: int
    planned_position: int
    warmup: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sub_seed(seed: int, scenario: str, repetition: int) -> int:
    material = f"{seed}:{scenario}:{repetition}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def build_trial_plan(
    *,
    repetitions: int,
    seed: int,
) -> list[RealCgroupTrialPlanEntry]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")

    plan: list[RealCgroupTrialPlanEntry] = []
    for scenario in SCENARIOS:
        warmup_seed = _sub_seed(seed, scenario, 0)
        for position, mode in enumerate(MODES):
            plan.append(
                RealCgroupTrialPlanEntry(
                    trial_id=f"warmup-{scenario}-{mode}",
                    scenario=scenario,
                    mode=mode,
                    repetition=0,
                    sub_seed=warmup_seed,
                    planned_position=position,
                    warmup=True,
                )
            )
        for repetition in range(1, repetitions + 1):
            sub_seed = _sub_seed(seed, scenario, repetition)
            modes = list(MODES)
            random.Random(sub_seed).shuffle(modes)
            for position, mode in enumerate(modes):
                plan.append(
                    RealCgroupTrialPlanEntry(
                        trial_id=(
                            f"{scenario}-r{repetition:02d}-"
                            f"p{position}-{mode}"
                        ),
                        scenario=scenario,
                        mode=mode,
                        repetition=repetition,
                        sub_seed=sub_seed,
                        planned_position=position,
                    )
                )
    return plan


def quota_for(scenario: str, mode: str) -> ResourceQuota:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario!r}")
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode!r}")
    if mode == "unrestricted":
        return ResourceQuota()
    if scenario == "cpu":
        return ResourceQuota(cpu_max_percent=20.0)
    if scenario == "memory":
        return ResourceQuota(memory_max_mb=48)
    return ResourceQuota(pids_max=8)


def limit_readback_matches(
    scenario: str,
    mode: str,
    limits: dict[str, object],
) -> bool:
    if mode == "unrestricted":
        return True
    expected = {
        "cpu": ("cpu_max_percent", 20.0),
        "memory": ("memory_max_mb", 48),
        "pids": ("pids_max", 8),
    }[scenario]
    actual = limits.get(expected[0])
    if scenario == "cpu":
        return actual is not None and abs(float(actual) - 20.0) <= 1.0
    if scenario == "memory":
        return actual == expected[1] and limits.get("memory_swap_max") == 0
    return actual == expected[1]


def _worker_command(scenario: str) -> list[str]:
    prelude = "import json,subprocess,sys,time\nprint('ready',flush=True)\nsys.stdin.readline()\n"
    if scenario == "cpu":
        body = (
            "started=time.monotonic()\n"
            "value=0\n"
            "while time.monotonic()-started<0.6:\n"
            " value=(value*1664525+1013904223)&0xffffffff\n"
            "print(json.dumps({'completed':True,'value':value}),flush=True)\n"
        )
    elif scenario == "memory":
        body = (
            "blocks=[]\n"
            "for _ in range(96):\n"
            " block=bytearray(1024*1024)\n"
            " for offset in range(0,len(block),4096):\n"
            "  block[offset]=1\n"
            " blocks.append(block)\n"
            "print(json.dumps({'completed':True,'allocated_mb':len(blocks)}),flush=True)\n"
            "time.sleep(0.15)\n"
        )
    elif scenario == "pids":
        body = (
            "children=[]\n"
            "failures=0\n"
            "for _ in range(16):\n"
            " try:\n"
            "  children.append(subprocess.Popen([sys.executable,'-c','import time;time.sleep(0.2)']))\n"
            " except OSError:\n"
            "  failures+=1\n"
            "print(json.dumps({'completed':True,'children_spawned':len(children),'spawn_failures':failures}),flush=True)\n"
            "for child in children:\n"
            " child.wait()\n"
        )
    else:
        raise ValueError(f"unknown scenario: {scenario!r}")
    return [sys.executable, "-u", "-c", prelude + body]


def _read_memory_events(path: Path) -> dict[str, int]:
    try:
        lines = (path / "memory.events").read_text(
            encoding="utf-8"
        ).splitlines()
    except OSError:
        return {}
    result: dict[str, int] = {}
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            result[parts[0]] = int(parts[1])
        except ValueError:
            continue
    return result


def _parse_worker_result(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _verification_adapter(
    manager: CgroupV2Manager,
) -> UserSpaceKernelAdapter:
    return UserSpaceKernelAdapter(
        cgroup_manager=manager,
        bpf_map_adapter=MockBPFMapAdapter(),
        probe_result=KernelProbeResult(
            sched_ext="unavailable",
            bpf_map="mock",
            fallback_mode="stage5_real_cgroup",
            reason="real cgroup experiment does not claim sched_ext",
        ),
    )


def run_trial(
    entry: RealCgroupTrialPlanEntry,
    *,
    root: Path,
) -> dict[str, Any]:
    manager = CgroupV2Manager(root=root)
    adapter = _verification_adapter(manager)
    quota = quota_for(entry.scenario, entry.mode)
    agent_id = f"s5-{entry.trial_id}"
    process: subprocess.Popen[str] | None = None
    cgroup_path: Path | None = None
    cleanup_failed = 0
    error: str | None = None
    result_data: dict[str, Any] = {}
    pid_verified = False
    readback_verified = False
    pressure_source = "unavailable"
    local_pressure_sources: dict[str, str] = {}
    limits: dict[str, object] = {}
    memory_before: dict[str, int] = {}
    memory_after: dict[str, int] = {}
    usage_before = 0
    usage_after = 0
    memory_peak_mb = 0.0
    exit_code: int | None = None
    wall_started = time.monotonic()
    trial_status = "recorded"

    try:
        process = subprocess.Popen(
            _worker_command(entry.scenario),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("worker pipes unavailable")
        if process.stdout.readline().strip() != "ready":
            raise RuntimeError("worker did not enter attach barrier")

        applied = adapter.apply_agent_quota(
            agent_id=agent_id,
            task_id=entry.trial_id,
            pid=process.pid,
            quota=quota,
        )
        if not applied.available or applied.cgroup_path is None:
            raise RuntimeError(f"cgroup apply failed: {applied.reason}")
        cgroup_path = applied.cgroup_path
        if entry.scenario == "memory" and entry.mode == "controlled":
            swap_limit = cgroup_path / "memory.swap.max"
            swap_limit.write_text("0", encoding="utf-8")
        pid_verified = any(
            event.event_type == "resource.cgroup_pid_added"
            and event.data.get("pid_membership_verified") is True
            and event.data.get("membership_check_required") is True
            for event in applied.events
        )
        limits = manager.read_limits(cgroup_path)
        if entry.scenario == "memory" and entry.mode == "controlled":
            limits["memory_swap_max"] = int(
                (cgroup_path / "memory.swap.max")
                .read_text(encoding="utf-8")
                .strip()
            )
        readback_verified = limit_readback_matches(
            entry.scenario,
            entry.mode,
            limits,
        )
        initial = adapter.sample_pressure(cgroup_path=cgroup_path)
        usage_before = (
            initial.cgroup_usage.cpu_usage_usec
            if initial.cgroup_usage is not None
            else 0
        )
        memory_before = _read_memory_events(cgroup_path)

        process.stdin.write("go\n")
        process.stdin.flush()
        remaining_stdout, stderr = process.communicate(timeout=8)
        exit_code = process.returncode
        result_data = _parse_worker_result(remaining_stdout)

        final = adapter.sample_pressure(cgroup_path=cgroup_path)
        pressure_source = final.pressure_source
        local_pressure_sources = dict(final.pressure_sources)
        if final.cgroup_usage is not None:
            usage_after = final.cgroup_usage.cpu_usage_usec
            memory_peak_mb = final.cgroup_usage.memory_peak_mb
        memory_after = _read_memory_events(cgroup_path)
        if stderr.strip() and exit_code == 0:
            result_data["stderr"] = stderr.strip()
    except Exception as exc:
        trial_status = "harness_error"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=2)
            exit_code = process.returncode
        if cgroup_path is not None:
            cleanup_events = adapter.cleanup_agent(
                agent_id=agent_id,
                cgroup_path=cgroup_path,
            )
            cleanup_failed = sum(
                event.event_type == "resource.cgroup_cleanup_failed"
                for event in cleanup_events
            )

    metrics = {
        "cpu_usage_usec": float(max(0, usage_after - usage_before)),
        "memory_peak_mb": float(memory_peak_mb),
        "oom_kill_count": float(
            max(
                0,
                memory_after.get("oom_kill", 0)
                - memory_before.get("oom_kill", 0),
            )
        ),
        "children_spawned": float(result_data.get("children_spawned", 0)),
        "spawn_failures": float(result_data.get("spawn_failures", 0)),
        "wall_time_secs": max(0.0, time.monotonic() - wall_started),
    }
    return {
        **entry.to_dict(),
        "actual_position": entry.planned_position,
        "trial_status": trial_status,
        "error": error,
        "adapter_mode": "cgroup_v2",
        "cgroup_path": str(cgroup_path) if cgroup_path is not None else None,
        "pid_membership_verified": pid_verified,
        "limit_readback_verified": readback_verified,
        "limits": limits,
        "pressure_source": pressure_source,
        "pressure_sources": local_pressure_sources,
        "cleanup_failed": cleanup_failed,
        "exit_code": exit_code,
        "worker_result": result_data,
        "metrics": metrics,
        "metric_sources": {
            "cpu_usage_usec": "cgroup v2 cpu.stat",
            "memory_peak_mb": "cgroup v2 memory.peak",
            "oom_kill_count": "cgroup v2 memory.events delta",
            "children_spawned": "worker subprocess creation result",
            "spawn_failures": "worker OSError count",
            "wall_time_secs": "time.monotonic",
        },
    }


def aggregate_trials(
    trials: list[dict[str, Any]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    measured = [trial for trial in trials if not trial["warmup"]]
    warmups = [trial for trial in trials if trial["warmup"]]
    cells: dict[str, Any] = {}
    for scenario in SCENARIOS:
        cells[scenario] = {}
        for mode in MODES:
            candidates = [
                trial
                for trial in measured
                if trial["scenario"] == scenario and trial["mode"] == mode
            ]
            rows = [
                trial
                for trial in candidates
                if trial.get("trial_status") == "recorded"
            ]
            metrics = {
                metric: describe(
                    [float(row["metrics"][metric]) for row in rows]
                )
                for metric in METRICS
                if rows and all(metric in row.get("metrics", {}) for row in rows)
            }
            cells[scenario][mode] = {
                "recorded_trials": len(rows),
                "harness_failures": sum(
                    row.get("trial_status") != "recorded"
                    for row in candidates
                ),
                "pid_membership_verified": sum(
                    row.get("pid_membership_verified") is True
                    for row in rows
                ),
                "limit_readback_verified": sum(
                    row.get("limit_readback_verified") is True
                    for row in rows
                ),
                "cleanup_failures": sum(
                    int(row.get("cleanup_failed", 0)) for row in rows
                ),
                "metrics": metrics,
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "repetitions": repetitions,
            "scenarios": list(SCENARIOS),
            "modes": list(MODES),
            "measured_trial_count": len(measured),
            "warmup_count": len(warmups),
        },
        "cells": cells,
        "claim_boundary": {
            "real_cgroup_v2": True,
            "bpf_program_attached": False,
            "sched_ext_active": False,
            "kernel_scheduling_takeover": False,
        },
    }


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage 5 Real cgroup v2 Comparison",
        "",
        f"- Repetitions per cell: {summary['metadata']['repetitions']}",
        f"- Seed: {summary['metadata']['seed']}",
        "- Statistics: sample standard deviation and Student-t 95% CI.",
        "",
        "## Results",
        "",
        "| Scenario | Mode | Primary mean | 95% CI |",
        "|---|---|---:|---:|",
    ]
    primary = {
        "cpu": "cpu_usage_usec",
        "memory": "oom_kill_count",
        "pids": "children_spawned",
    }
    for scenario in SCENARIOS:
        for mode in MODES:
            stat = summary["cells"][scenario][mode]["metrics"][primary[scenario]]
            lines.append(
                f"| {scenario} | {mode} | {stat['mean']:.6f} | "
                f"[{stat['ci95_low']:.6f}, {stat['ci95_high']:.6f}] |"
            )
    lines.extend(
        [
            "",
            "## Capability boundary",
            "",
            "- This is real cgroup v2 enforcement evidence.",
            "- No BPF program attachment is claimed.",
            "- No sched_ext activation or kernel scheduling takeover is claimed.",
            "",
        ]
    )
    return "\n".join(lines)


def write_artifacts(
    *,
    output_dir: Path,
    plan: list[RealCgroupTrialPlanEntry],
    trials: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        output_dir / "trial_plan.json",
        json.dumps(
            [entry.to_dict() for entry in plan],
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    _atomic_text(
        output_dir / "raw_trials.jsonl",
        "".join(json.dumps(trial, sort_keys=True) + "\n" for trial in trials),
    )
    _atomic_text(
        output_dir / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    csv_path = output_dir / "summary.csv"
    temporary = csv_path.with_name(csv_path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "scenario",
            "mode",
            "metric",
            "n",
            "mean",
            "stddev",
            "p50",
            "p95",
            "ci95_low",
            "ci95_high",
            "minimum",
            "maximum",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scenario in SCENARIOS:
            for mode in MODES:
                for metric, stat in summary["cells"][scenario][mode][
                    "metrics"
                ].items():
                    writer.writerow(
                        {
                            "scenario": scenario,
                            "mode": mode,
                            "metric": metric,
                            **stat,
                        }
                    )
    temporary.replace(csv_path)
    _atomic_text(output_dir / "report.md", _report(summary))
    _atomic_text(
        output_dir / "environment.txt",
        "\n".join(
            (
                f"python={platform.python_version()}",
                f"platform={platform.platform()}",
                f"pid={os.getpid()}",
                "filesystem=cgroup2fs",
                "",
            )
        ),
    )
