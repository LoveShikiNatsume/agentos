#!/usr/bin/env python3
"""Run the independent Stage 5 sched_ext fault and recovery matrix.

This verifier is intentionally separate from the older cache/cgroup Stage 5
experiments.  It mutates a real sched_ext instance and must run as root on the
pinned openEuler kernel.  A fresh loader (and therefore fresh maps) is used for
every trial.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import signal
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentos.benchmarks.sched_ext_reliability import (
    AUTO_FALLBACK_CASES,
    AUTO_FAULT_CASES,
    STRICT_FAULT_CASES,
    acceptance_passed,
    summarize_trials,
)
from agentos.core.resources import ResourceQuota
from agentos.core.task import Task, TaskDAG
from agentos.fault.injector import FaultInjectionPlan
from agentos.fault.recovery import SkipPolicy
from agentos.isolation.cgroup import CgroupV2Manager
from agentos.isolation.sched_ext_control import SchedExtControlClient
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import Runtime
from agentos.worker.process import SubprocessWorkerManager
from scripts.preflight_sched_ext import probe_sched_ext
from scripts.verify_real_sched_ext import (
    _snapshot,
    _start_loader,
    kernel_log_errors,
    sched_ext_enabled,
)

PIN_DIR = Path("/sys/fs/bpf/agentos/sched_ext")
POLICY_MAP = PIN_DIR / "task_policy_map"
TASK_STATE_MAP = PIN_DIR / "task_state_map"
CGROUP_ROOT = Path("/sys/fs/cgroup/agentos")
EXPECTED_KERNEL = "6.12.95-agentos-scx"
GLOBAL_CONFIG_STRUCT = struct.Struct("<IIQQII")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True) + "\n")


def _seal(directory: Path) -> None:
    lines = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            relative = path.relative_to(directory)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {relative}\n")
    (directory / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, check=check, capture_output=True, text=True, timeout=20
    )


def _map_entries(path: Path) -> int:
    if not path.exists():
        return 0
    result = _run(["bpftool", "-j", "map", "dump", "pinned", str(path)])
    value = json.loads(result.stdout)
    return len(value) if isinstance(value, list) else 0


def _unpin_sched_ext_maps(pin_dir: Path = PIN_DIR) -> None:
    state = _snapshot().get("state")
    if state != "disabled":
        raise RuntimeError(
            f"refusing to unpin sched_ext maps while state is {state!r}"
        )
    if not pin_dir.is_dir():
        return
    for path in pin_dir.iterdir():
        if path.is_file() or path.is_symlink():
            path.unlink()


def _unpin_policy_map(path: Path = POLICY_MAP) -> None:
    """Remove one bpffs pin while retaining the map through its live references."""
    path.unlink()


def _residual_cgroups(root: Path = CGROUP_ROOT) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def _cleanup_trial_cgroups(root: Path = CGROUP_ROOT) -> None:
    if not root.is_dir():
        return
    for path in root.iterdir():
        if not path.is_dir() or not path.name.startswith("agent-stage5-"):
            continue
        try:
            path.rmdir()
        except OSError:
            pass


def _systemd_failed_units() -> int:
    result = _run(
        ["systemctl", "--failed", "--no-legend", "--plain"], check=False
    )
    return len([line for line in result.stdout.splitlines() if line.strip()])


def _artifact_hashes(paths: Iterable[Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in paths:
        if path.is_file():
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _event_dicts(runtime: Runtime) -> list[dict[str, Any]]:
    return [event.to_dict() for event in runtime.events]


async def _wait_event(
    runtime: Runtime, event_type: str, *, after: int = 0, timeout: float = 5.0
) -> tuple[dict[str, Any], int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        events = _event_dicts(runtime)
        for index in range(after, len(events)):
            if events[index].get("event_type") == event_type:
                return events[index], index + 1
        await asyncio.sleep(0.02)
    raise TimeoutError(f"timed out waiting for {event_type}")


def _task_for_case(case: str) -> Task:
    worker_type = "sleep"
    payload: dict[str, Any] = {"seconds": 6.0, "result": "stage5-ok"}
    quota = ResourceQuota(timeout_seconds=12.0)
    if case == "worker_fork":
        worker_type, payload = "fork", {}
    elif case == "worker_failure":
        worker_type, payload = "fail", {"error": "stage5 expected failure"}
    elif case == "worker_timeout":
        payload = {"seconds": 2.0}
        quota = ResourceQuota(timeout_seconds=0.5)
    elif case in {"worker_sigkill", "single_cpu_cpuset", "rt_coexistence"}:
        payload = {"seconds": 1.0, "result": "stage5-ok"}
    return Task(
        task_id="stage5-primary",
        name=f"stage5-{case}",
        agent_role=worker_type,
        resource_requirements=quota,
        metadata={
            "execution": "subprocess",
            "worker_type": worker_type,
            "worker_payload": payload,
        },
    )


def _make_runtime(mode: str, case: str) -> tuple[Runtime, TaskDAG]:
    dag = TaskDAG()
    dag.add_task(_task_for_case(case))
    if mode == "auto" and case in AUTO_FALLBACK_CASES:
        dag.add_task(
            Task(
                task_id="stage5-after-recovery",
                name="stage5-after-recovery",
                agent_role="echo",
                dependencies=["stage5-primary"],
                metadata={
                    "execution": "subprocess",
                    "worker_type": "echo",
                    "worker_payload": {"result": "stage5-rebound"},
                },
            )
        )
    fault = FaultInjectionPlan({"stage5-primary": "kill"}) if case == "worker_sigkill" else None
    runtime = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={},
        execution_mode="subprocess",
        recovery_policy=None if mode == "strict" else SkipPolicy(),
        fault_injection=fault,
        worker_manager=SubprocessWorkerManager(
            allowed_cpus={1} if case == "single_cpu_cpuset" else {1, 2}
        ),
        cgroup_manager=CgroupV2Manager(root=CGROUP_ROOT),
        sched_ext_mode=mode,
        sched_ext_control=SchedExtControlClient(),
        worker_sample_interval_seconds=0.1,
    )
    return runtime, dag


def _rewrite_schema(control: SchedExtControlClient, schema: int) -> None:
    config = control.read_global_config()
    raw = GLOBAL_CONFIG_STRUCT.pack(
        schema,
        config["flags"],
        config["generation"],
        config["heartbeat_ns"],
        config["loader_pid"],
        config["heartbeat_timeout_seconds"],
    )
    _run(
        [
            "bpftool", "map", "update", "pinned", str(control.global_config_path),
            "key", "hex", "00", "00", "00", "00", "value", "hex",
            *[f"{byte:02x}" for byte in raw], "any",
        ]
    )


@dataclass
class _CapturedLoader:
    process: subprocess.Popen[str]
    stdout_lines: list[str] = field(default_factory=list)
    stderr_lines: list[str] = field(default_factory=list)
    threads: list[threading.Thread] = field(default_factory=list)

    @property
    def pid(self) -> int:
        return self.process.pid

    def poll(self) -> int | None:
        return self.process.poll()

    def wait(self, timeout: float | None = None) -> int:
        return self.process.wait(timeout=timeout)

    def send_signal(self, signo: signal.Signals) -> None:
        self.process.send_signal(signo)


def _drain_stream(stream: Any, sink: list[str]) -> None:
    try:
        for line in stream:
            sink.append(line)
    finally:
        stream.close()


def _start_captured_loader(binary: Path) -> tuple[_CapturedLoader, list[dict[str, Any]]]:
    process, events = _start_loader(binary)
    captured = _CapturedLoader(process=process)
    for stream, sink, name in (
        (process.stdout, captured.stdout_lines, "stdout"),
        (process.stderr, captured.stderr_lines, "stderr"),
    ):
        if stream is None:
            continue
        thread = threading.Thread(
            target=_drain_stream,
            args=(stream, sink),
            name=f"stage5-loader-{process.pid}-{name}",
            daemon=True,
        )
        thread.start()
        captured.threads.append(thread)
    return captured, events


def _finish_loader(process: _CapturedLoader, signo: signal.Signals) -> dict[str, Any]:
    if process.poll() is None:
        process.send_signal(signo)
        process.send_signal(signal.SIGCONT)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.send_signal(signal.SIGKILL)
        process.wait(timeout=3)
    for thread in process.threads:
        thread.join(timeout=2)
    return {
        "pid": process.pid,
        "return_code": process.poll(),
        "stdout": "".join(process.stdout_lines),
        "stderr": "".join(process.stderr_lines),
    }


async def _run_trial(
    *, mode: str, case: str, scheduler_binary: Path, trial_dir: Path, allow_sysrq: bool
) -> dict[str, Any]:
    trial_dir.mkdir(parents=True, exist_ok=False)
    started_ns = time.monotonic_ns()
    loader, initial_loader_events = _start_captured_loader(scheduler_binary)
    loaders: list[_CapturedLoader] = [loader]
    loader_logs: list[dict[str, Any]] = []
    pressure: subprocess.Popen[str] | None = None
    rt_process: subprocess.Popen[str] | None = None
    runtime, dag = _make_runtime(mode, case)
    run_task: asyncio.Task[Any] | None = None
    injection_ns = 0
    detection_ms = 0.0
    recovery_ms = 0.0
    runtime_error: str | None = None
    results: dict[str, Any] = {}
    outcome = "unknown"
    map_id: int | None = None
    rt_ext_enabled: int | None = None
    recovered_old_worker_ext_enabled: int | None = None
    control = SchedExtControlClient()
    try:
        run_task = asyncio.create_task(runtime.run(dag))
        _started, cursor = await _wait_event(runtime, "worker.started", timeout=4.0)
        worker_pid = int(_started["data"]["pid"])
        if case == "rt_coexistence":
            rt_process = subprocess.Popen(
                [sys.executable, "-c", "import time\nfor _ in range(20):\n time.sleep(.02)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.sched_setaffinity(rt_process.pid, {1})
            os.sched_setscheduler(rt_process.pid, os.SCHED_FIFO, os.sched_param(1))
            rt_ext_enabled = sched_ext_enabled(rt_process.pid)

        fault_case = case in AUTO_FALLBACK_CASES or mode == "strict"
        if fault_case:
            injection_ns = time.monotonic_ns()
            if case in {"loader_sigkill", "loader_sigterm", "cfs_competition"}:
                if case == "cfs_competition":
                    pressure = subprocess.Popen(
                        ["yes"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                    os.sched_setaffinity(pressure.pid, {1})
                loader.send_signal(
                    signal.SIGKILL
                    if case in {"loader_sigkill", "cfs_competition"}
                    else signal.SIGTERM
                )
            elif case == "sysrq_unload":
                if not allow_sysrq:
                    raise RuntimeError("sysrq_unload requires --allow-sysrq")
                Path("/proc/sysrq-trigger").write_text("S", encoding="ascii")
            elif case == "heartbeat_timeout":
                loader.send_signal(signal.SIGSTOP)
            elif case == "schema_mismatch":
                _rewrite_schema(control, 0xFFFFFFFF)
            elif case == "policy_map_unpin":
                health = control.read_health()
                map_id = int(health["map_ids"]["task_policy_map"])
                _unpin_policy_map(POLICY_MAP)
            elif case == "policy_map_freeze":
                _run(["bpftool", "map", "freeze", "pinned", str(POLICY_MAP)])

            _fallback, cursor = await _wait_event(
                runtime, "kernel.sched_ext_fallback", after=cursor, timeout=5.0
            )
            detection_ms = (time.monotonic_ns() - injection_ns) / 1_000_000

            if mode == "auto":
                recovery_started_ns = time.monotonic_ns()
                if case in {"loader_sigkill", "loader_sigterm", "sysrq_unload", "cfs_competition"}:
                    if loader.poll() is None:
                        loader.wait(timeout=5)
                    if case in {"loader_sigkill", "cfs_competition"}:
                        _unpin_sched_ext_maps()
                    loader, events = _start_captured_loader(scheduler_binary)
                    loaders.append(loader)
                    initial_loader_events.extend(events)
                elif case == "heartbeat_timeout":
                    loader.send_signal(signal.SIGCONT)
                elif case == "schema_mismatch":
                    config = control.read_global_config()
                    control.update_global_config(generation=config["generation"])
                elif case == "policy_map_unpin":
                    if map_id is None:
                        raise RuntimeError("policy map id unavailable")
                    _run(["bpftool", "map", "pin", "id", str(map_id), str(POLICY_MAP)])
                elif case == "policy_map_freeze":
                    loader_logs.append(_finish_loader(loader, signal.SIGTERM))
                    loader, events = _start_captured_loader(scheduler_binary)
                    loaders.append(loader)
                    initial_loader_events.extend(events)
                _recovered, cursor = await _wait_event(
                    runtime, "kernel.sched_ext_recovered", after=cursor, timeout=4.0
                )
                recovery_ms = (time.monotonic_ns() - recovery_started_ns) / 1_000_000
                recovered_old_worker_ext_enabled = sched_ext_enabled(worker_pid)

        if case == "runtime_cancel":
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                outcome = "cancelled"
        else:
            try:
                results = await asyncio.wait_for(run_task, timeout=15.0)
            except Exception as exc:
                runtime_error = f"{type(exc).__name__}: {exc}"

        event_types = [event["event_type"] for event in _event_dicts(runtime)]
        attached_tasks = [
            event.get("task_id")
            for event in _event_dicts(runtime)
            if event.get("event_type") == "kernel.sched_ext_worker_attached"
        ]
        recovery_binding_ok = bool(
            mode != "auto"
            or not fault_case
            or (
                attached_tasks.count("stage5-primary") == 1
                and attached_tasks.count("stage5-after-recovery") == 1
            )
        )
        recovery_generation_ok = True
        if mode == "auto" and fault_case:
            recovered_policies = [
                event.get("data", {}).get("policy", {})
                for event in _event_dicts(runtime)
                if event.get("event_type") == "kernel.sched_ext_policy_published"
                and event.get("task_id") == "stage5-after-recovery"
            ]
            recovery_generation_ok = bool(
                len(recovered_policies) == 1
                and recovered_policies[0].get("generation")
                == control.read_global_config().get("generation")
            )
        if mode == "strict":
            outcome = (
                "strict_failed"
                if runtime_error or "task.failed" in event_types
                else "silent_success"
            )
        elif case in AUTO_FALLBACK_CASES:
            outcome = (
                "recovered"
                if "kernel.sched_ext_recovered" in event_types
                else "fallback_only"
            )
        elif case in {"worker_failure", "worker_sigkill", "worker_timeout"}:
            expected_event = {
                "worker_failure": "worker.failed",
                "worker_sigkill": "worker.failed",
                "worker_timeout": "worker.timeout",
            }[case]
            outcome = (
                "expected_worker_failure"
                if expected_event in event_types
                else "missing_worker_failure"
            )
        elif case == "runtime_cancel":
            outcome = "cancelled" if outcome == "cancelled" else "cancel_failed"
        else:
            outcome = "completed" if results.get("stage5-primary") is not None else "task_missing"

        fallback_events = event_types.count("kernel.sched_ext_fallback")
        recovered_events = event_types.count("kernel.sched_ext_recovered")
        expected = (
            outcome == "strict_failed"
            if mode == "strict"
            else outcome in {"recovered", "expected_worker_failure", "cancelled", "completed"}
        )
        task_loss = int(
            mode == "auto"
            and case
            not in {
                "worker_failure",
                "worker_sigkill",
                "worker_timeout",
                "runtime_cancel",
            }
            and (
                results.get("stage5-primary") is None
                or (
                    fault_case
                    and results.get("stage5-after-recovery") != "stage5-rebound"
                )
            )
        )
        passed = bool(
            expected
            and task_loss == 0
            and fallback_events == int(fault_case)
            and recovered_events == int(mode == "auto" and fault_case)
            and recovery_binding_ok
            and recovery_generation_ok
            and (
                not (mode == "auto" and fault_case)
                or recovered_old_worker_ext_enabled == 0
            )
            and (case != "rt_coexistence" or rt_ext_enabled == 0)
            and detection_ms
            <= (
                4_000
                if case in {"heartbeat_timeout", "policy_map_freeze"}
                else 2_000
            )
            and (not (mode == "auto" and fault_case) or recovery_ms <= 2_000)
        )
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
        passed = False
        task_loss = 1 if mode == "auto" else 0
        fallback_events = 0
        recovered_events = 0
        outcome = "verifier_error"
    finally:
        if run_task is not None and not run_task.done():
            run_task.cancel()
            try:
                await run_task
            except BaseException:
                pass
        for process in (pressure, rt_process):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=3)
        for process in loaders:
            if process.poll() is None:
                loader_logs.append(_finish_loader(process, signal.SIGTERM))
            elif not any(item["pid"] == process.pid for item in loader_logs):
                loader_logs.append(_finish_loader(process, signal.SIGTERM))
        _unpin_sched_ext_maps()
        _cleanup_trial_cgroups()

    events = _event_dicts(runtime)
    _write_jsonl(trial_dir / "events.jsonl", events)
    _write_json(trial_dir / "results.json", results)
    _write_json(trial_dir / "loader-events.json", initial_loader_events)
    _write_json(trial_dir / "loader-processes.json", loader_logs)
    policy_entries_after = _map_entries(POLICY_MAP)
    task_state_entries_after = _map_entries(TASK_STATE_MAP)
    final_state = _snapshot().get("state")
    passed = bool(
        passed
        and policy_entries_after == 0
        and task_state_entries_after == 0
        and final_state == "disabled"
    )
    record = {
        "schema_version": 1,
        "mode": mode,
        "case": case,
        "passed": passed,
        "outcome": outcome,
        "task_loss": task_loss,
        "fallback_events": fallback_events,
        "recovered_events": recovered_events,
        "detection_ms": detection_ms,
        "recovery_ms": recovery_ms,
        "runtime_error": runtime_error,
        "rt_ext_enabled": rt_ext_enabled,
        "recovered_old_worker_ext_enabled": recovered_old_worker_ext_enabled,
        "recovery_binding_ok": (
            recovery_binding_ok if "recovery_binding_ok" in locals() else False
        ),
        "recovery_generation_ok": (
            recovery_generation_ok
            if "recovery_generation_ok" in locals()
            else False
        ),
        "duration_ms": (time.monotonic_ns() - started_ns) / 1_000_000,
        "policy_entries_after": policy_entries_after,
        "task_state_entries_after": task_state_entries_after,
        "final_sched_ext_state": final_state,
    }
    _write_json(trial_dir / "trial.json", record)
    _seal(trial_dir)
    return record


async def _run_matrix(args: argparse.Namespace, output_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    selected = set(args.cases or ())
    matrix = (("auto", AUTO_FAULT_CASES), ("strict", STRICT_FAULT_CASES))
    for mode, cases in matrix:
        for case in cases:
            if selected and case not in selected and f"{mode}:{case}" not in selected:
                continue
            for iteration in range(1, args.iterations + 1):
                trial_dir = output_dir / "trials" / mode / case / f"run-{iteration:02d}"
                record = await _run_trial(
                    mode=mode,
                    case=case,
                    scheduler_binary=args.scheduler_binary,
                    trial_dir=trial_dir,
                    allow_sysrq=args.allow_sysrq,
                )
                record["iteration"] = iteration
                records.append(record)
                _write_jsonl(output_dir / "trials.jsonl", records)
    return records


async def _run_soak(
    *, scheduler_binary: Path, output_dir: Path, duration_seconds: int
) -> dict[str, Any]:
    if duration_seconds <= 0:
        return {"completed": False, "duration_seconds": 0, "task_loss": 0, "waves": 0}
    loader, loader_events = _start_captured_loader(scheduler_binary)
    started = time.monotonic()
    waves = 0
    task_loss = 0
    events: list[dict[str, Any]] = []
    failure: str | None = None
    try:
        while time.monotonic() - started < duration_seconds:
            runtime, dag = _make_runtime("auto", "single_cpu_cpuset")
            try:
                run_task = asyncio.create_task(runtime.run(dag))
                started_event, _cursor = await _wait_event(
                    runtime, "worker.started", timeout=4.0
                )
                os.sched_setaffinity(int(started_event["data"]["pid"]), {1, 2})
                results = await run_task
                if results.get("stage5-primary") != "stage5-ok":
                    task_loss += 1
            except Exception as exc:
                task_loss += 1
                failure = f"{type(exc).__name__}: {exc}"
            events.extend(_event_dicts(runtime))
            waves += 1
            await asyncio.sleep(0.1)
    finally:
        loader_log = _finish_loader(loader, signal.SIGTERM)
        _write_json(output_dir / "soak-loader-events.json", loader_events)
        _write_json(output_dir / "soak-loader-process.json", loader_log)
        _write_jsonl(output_dir / "soak-events.jsonl", events)
    elapsed = time.monotonic() - started
    result = {
        "completed": elapsed >= duration_seconds and failure is None,
        "duration_seconds": elapsed,
        "requested_duration_seconds": duration_seconds,
        "task_loss": task_loss,
        "waves": waves,
        "failure_reason": failure,
    }
    _write_json(output_dir / "soak-summary.json", result)
    return result


def _residual_processes(binary: Path) -> list[int]:
    residual: list[int] = []
    expected_exe = binary.resolve()
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            executable = (path / "exe").resolve()
            command = (path / "cmdline").read_bytes().replace(b"\0", b" ")
        except OSError:
            continue
        if executable == expected_exe or b"agentos.worker.entrypoint" in command:
            residual.append(int(path.name))
    return sorted(residual)


def _residual_bpf_links() -> list[dict[str, Any]]:
    result = _run(["bpftool", "-j", "link", "show"], check=False)
    if result.returncode:
        return [{"inspection_error": result.stderr.strip() or result.stdout.strip()}]
    try:
        links = json.loads(result.stdout)
    except json.JSONDecodeError:
        return [{"inspection_error": "invalid bpftool link JSON"}]
    if not isinstance(links, list):
        return [{"inspection_error": "bpftool link output is not a list"}]
    return [
        link for link in links
        if "agentos" in json.dumps(link, sort_keys=True).lower()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheduler-binary", type=Path,
        default=PROJECT_ROOT / "native/sched_ext/build/bin/agentos_scx",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "runs/member_a_sched_ext_validation/stage5_reliability",
    )
    parser.add_argument("--iterations", type=int, choices=(1, 20), default=1)
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--allow-sysrq", action="store_true")
    parser.add_argument("--require-soak", action="store_true")
    parser.add_argument("--soak-seconds", type=int, default=0)
    args = parser.parse_args(argv)
    args.scheduler_binary = args.scheduler_binary.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    started_at = int(time.time())
    failure_reason: str | None = None
    records: list[dict[str, Any]] = []
    soak = {"completed": False, "duration_seconds": 0, "task_loss": 0, "waves": 0}
    _write_json(output_dir / "sysfs-before.json", _snapshot())
    environment = {
        "kernel_release": platform.release(),
        "expected_kernel_release": EXPECTED_KERNEL,
        "controller_cpu": 0,
        "worker_cpus": [1, 2],
        "pressure_cpu": 3,
        "python": sys.version,
        "preflight": probe_sched_ext(),
    }
    _write_json(output_dir / "environment.json", environment)
    _write_json(
        output_dir / "artifact-sha256.json",
        _artifact_hashes(
            [
                args.scheduler_binary,
                Path(__file__),
                PROJECT_ROOT / "native/sched_ext/agentos_scx.bpf.c",
            ]
        ),
    )
    try:
        if os.geteuid() != 0:
            raise PermissionError("Stage 5 reliability verification must run as root")
        if platform.release() != EXPECTED_KERNEL:
            raise RuntimeError(f"expected kernel {EXPECTED_KERNEL}, got {platform.release()}")
        if not environment["preflight"].get("accepted"):
            failures = environment["preflight"].get("failures")
            raise RuntimeError(f"sched_ext preflight failed: {failures}")
        if len(os.sched_getaffinity(0) & {0, 1, 2, 3}) != 4:
            raise RuntimeError("Stage 5 requires CPUs 0-3")
        if args.require_soak and args.soak_seconds < 3600:
            raise ValueError("formal soak requires --soak-seconds >= 3600")
        os.sched_setaffinity(0, {0})
        records = asyncio.run(_run_matrix(args, output_dir))
        soak = asyncio.run(
            _run_soak(
                scheduler_binary=args.scheduler_binary,
                output_dir=output_dir,
                duration_seconds=args.soak_seconds,
            )
        )
    except Exception as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"

    journal = _run(
        ["journalctl", "-k", "-b", f"--since=@{started_at}", "--no-pager"],
        check=False,
    )
    (output_dir / "kernel-window.log").write_text(
        journal.stdout + journal.stderr, encoding="utf-8"
    )
    summarized = summarize_trials(records)
    summarized.update(
        {
            "schema_version": 1,
            "iterations": args.iterations,
            "policy_leaks": _map_entries(POLICY_MAP),
            "task_state_leaks": _map_entries(TASK_STATE_MAP),
            "residual_pinned_maps": (
                sorted(path.name for path in PIN_DIR.iterdir())
                if PIN_DIR.is_dir() else []
            ),
            "residual_bpf_links": _residual_bpf_links(),
            "residual_processes": _residual_processes(args.scheduler_binary),
            "residual_cgroups": _residual_cgroups(),
            "kernel_log_errors": kernel_log_errors(journal.stdout.splitlines()),
            "systemd_failed_units": _systemd_failed_units(),
            "final_sched_ext_state": _snapshot().get("state"),
            "soak": soak,
            "failure_reason": failure_reason,
        }
    )
    complete_matrix = not args.cases
    summarized["accepted"] = bool(
        complete_matrix
        and acceptance_passed(
            summarized,
            expected_iterations=args.iterations,
            require_soak=args.require_soak,
        )
    )
    _write_json(output_dir / "sysfs-after.json", _snapshot())
    _write_json(output_dir / "summary.json", summarized)
    _seal(output_dir)
    print(json.dumps(summarized, indent=2, sort_keys=True))
    return 0 if summarized["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
