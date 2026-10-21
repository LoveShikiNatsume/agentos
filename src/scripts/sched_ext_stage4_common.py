"""Real-kernel helpers shared by the sched_ext Stage 4 verifiers."""

from __future__ import annotations

import asyncio
import json
import os
import select
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from agentos.benchmarks.experiment_stats import percentile
from agentos.benchmarks.sched_ext_stage4 import jain_fairness
from agentos.core.resources import ResourceQuota
from agentos.core.task import Task, TaskDAG
from agentos.isolation.bpf_schema import SchedPolicyKeyV1, SchedPolicyValueV1
from agentos.isolation.sched_ext_control import SchedExtControlClient
from agentos.scheduler.dag import DAGScheduler
from scripts.verify_real_sched_ext import _read_json_line, _snapshot, _start_loader, _stop_loader


SCHED_EXT = 7
CONTROLLER_CPU = 0
WORKER_CPU_LIST = "1,2"
WORKERS = (
    ("latency-0", "latency", 1, 300, 2_000),
    ("normal-0", "normal", 1, 100, 5_000),
    ("normal-1", "normal", 2, 100, 5_000),
    ("batch-0", "batch", 2, 50, 10_000),
    ("batch-1", "batch", 3, 50, 10_000),
)


def proc_cpu_snapshot(
    path: Path = Path("/proc/stat"), cpus: tuple[int, ...] = (0, 1, 2, 3)
) -> dict[str, int]:
    """Return aggregate CPU ticks and context switches for Stage 6 observation."""
    total = idle = context_switches = 0
    wanted = {f"cpu{cpu}" for cpu in cpus}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] in wanted:
            ticks = [int(value) for value in fields[1:]]
            total += sum(ticks)
            idle += sum(ticks[index] for index in (3, 4) if index < len(ticks))
        elif fields[0] == "ctxt" and len(fields) == 2:
            context_switches = int(fields[1])
    if total <= 0:
        raise RuntimeError("CPU 0-3 counters are unavailable in /proc/stat")
    return {"total": total, "idle": idle, "context_switches": context_switches}


def proc_cpu_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, float | int]:
    total = after["total"] - before["total"]
    idle = after["idle"] - before["idle"]
    if total <= 0 or idle < 0:
        raise RuntimeError("invalid /proc/stat counter delta")
    return {
        "cpu_utilization": max(0.0, min(1.0, (total - idle) / total)),
        "system_context_switches": after["context_switches"] - before["context_switches"],
    }


def pin_controller_cpu(cpu: int = CONTROLLER_CPU) -> set[int]:
    original = set(os.sched_getaffinity(0))
    if cpu not in original:
        raise RuntimeError(
            f"reserved controller CPU {cpu} is outside affinity {sorted(original)}"
        )
    os.sched_setaffinity(0, {cpu})
    return original


def _proc_state(pid: int) -> str:
    stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    suffix = stat.rsplit(")", 1)
    if len(suffix) != 2:
        raise RuntimeError(f"cannot parse /proc/{pid}/stat")
    fields = suffix[1].split()
    if not fields:
        raise RuntimeError(f"missing process state for PID {pid}")
    return fields[0]


def wait_worker_barrier(
    process: subprocess.Popen[bytes], timeout: float = 5.0
) -> None:
    if process.stderr is None:
        raise RuntimeError("worker stderr pipe is unavailable")
    ready, _, _ = select.select([process.stderr], [], [], timeout)
    if not ready:
        raise TimeoutError(f"worker {process.pid} did not publish READY")
    line = process.stderr.readline()
    if line != b"READY\n":
        raise RuntimeError(
            f"worker {process.pid} readiness failed: {line.decode(errors='replace')!r}"
        )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"worker {process.pid} exited before reaching the barrier"
            )
        if _proc_state(process.pid) == "S":
            return
        time.sleep(0.001)
    raise TimeoutError(f"worker {process.pid} did not block on the startup barrier")


def cpu_psi_avg10(path: Path = Path("/proc/pressure/cpu")) -> float:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("some "):
            for field in line.split()[1:]:
                if field.startswith("avg10="):
                    return float(field.split("=", 1)[1])
    raise RuntimeError("CPU PSI some.avg10 is unavailable")


def pressure_level(avg10: float) -> str:
    if avg10 >= 80:
        return "critical"
    if avg10 >= 10:
        return "warning"
    return "normal"


def policy_pressure_level(mode: str, observed_level: str) -> str:
    """Only the full policy mode is allowed to consume live PSI."""
    return observed_level if mode == "agentos_full_dag_psi_fairness" else "normal"


async def dag_policies(level: str) -> dict[str, dict[str, Any]]:
    dag = TaskDAG()
    for task_id, role, _cpu, weight, _slice in WORKERS:
        dag.add_task(
            Task(
                task_id=task_id,
                name=task_id,
                agent_role=role,
                sla_class=(
                    "interactive" if role == "latency" else
                    "batch" if role == "batch" else "normal"
                ),
                resource_requirements=ResourceQuota(cpu_shares=weight),
            )
        )
    scheduler = DAGScheduler(dag, max_batch_size=len(WORKERS), pressure_level=level)
    await scheduler.get_next_batch()
    if scheduler.last_decision is None:
        raise RuntimeError("DAG scheduler did not produce a decision")
    return scheduler.last_decision["tasks"]


def _wait_sysfs(
    ops: str,
    timeout: float = 10.0,
    *,
    partial: bool = False,
    process: subprocess.Popen[str] | None = None,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = _snapshot()
        if process is not None and process.poll() is not None:
            _stdout, stderr = process.communicate(timeout=1)
            raise RuntimeError(
                f"scheduler {ops!r} exited before attach: "
                f"rc={process.returncode}, stderr={stderr.strip()!r}"
            )
        if (
            snapshot.get("state") == "enabled"
            and snapshot.get("ops") == ops
            and (not partial or snapshot.get("switch_all") == "0")
        ):
            return
        time.sleep(0.05)
    raise TimeoutError(f"scheduler {ops!r} did not become active")


def _start_simple(binary: Path) -> subprocess.Popen[str]:
    process = subprocess.Popen(
        [str(binary)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_sysfs("simple", partial=True, process=process)
        return process
    except Exception:
        process.terminate()
        process.communicate(timeout=3)
        raise


def _stop_simple(process: subprocess.Popen[str]) -> tuple[int, str, str]:
    process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=10)
    return process.returncode, stdout, stderr


def _map_count(path: Path) -> int:
    result = subprocess.run(
        ["bpftool", "-j", "map", "dump", "pinned", str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return len(json.loads(result.stdout))


def _heartbeat(
    stop: threading.Event,
    control: SchedExtControlClient,
    generation: int,
    flags: int,
    psi_trace: list[dict[str, Any]],
    mode: str,
    initial_level: str,
    initial_decisions: dict[str, dict[str, Any]],
    bindings: list[dict[str, Any]],
    policy_readbacks: list[dict[str, Any]],
    errors: list[str],
) -> None:
    current_level = initial_level
    current_generation = generation
    decisions = dict(initial_decisions)
    try:
        while not stop.wait(0.5):
            avg10 = cpu_psi_avg10()
            observed_level = pressure_level(avg10)
            level_changed = (
                mode == "agentos_full_dag_psi_fairness"
                and observed_level != current_level
            )
            if level_changed:
                current_generation += 1
                decisions = asyncio.run(dag_policies(observed_level))
                current_level = observed_level
            control.update_global_config(
                generation=current_generation, flags=flags
            )
            for binding in bindings:
                policy = _policy_for(
                    mode,
                    binding["task_id"],
                    binding["role"],
                    binding["weight"],
                    binding["slice_us"],
                    current_generation,
                    current_level,
                    decisions,
                    binding.get("overrides"),
                )
                readback = control.put_policy(binding["key"], policy)
                if level_changed:
                    policy_readbacks.append(
                        {
                            "phase": "pressure_update",
                            "task_id": binding["task_id"],
                            "generation": current_generation,
                            "pressure_level": observed_level,
                            "decision": decisions.get(binding["task_id"]),
                            "readback": control.policy_to_dict(readback),
                        }
                    )
            psi_trace.append(
                {
                    "monotonic_ns": time.monotonic_ns(),
                    "cpu_some_avg10": avg10,
                    "pressure_level": observed_level,
                    "policy_pressure_level": current_level,
                    "generation": current_generation,
                }
            )
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")


def _policy_for(
    mode: str,
    task_id: str,
    role: str,
    weight: int,
    slice_us: int,
    generation: int,
    level: str,
    decisions: dict[str, dict[str, Any]],
    overrides: dict[str, Any] | None = None,
) -> SchedPolicyValueV1:
    if mode == "agentos_normal":
        role, weight, slice_us = "normal", 100, 5_000
    elif mode == "agentos_full_dag_psi_fairness":
        decision = decisions.get(task_id)
        if decision is not None:
            role = str(decision["queue_class"])
            weight = int(decision["kernel_weight"])
            slice_us = int(decision["slice_us"])
    if overrides:
        role = str(overrides.get("queue_class", role))
        weight = int(overrides.get("weight", weight))
        slice_us = int(overrides.get("slice_us", slice_us))
    return SchedPolicyValueV1.from_fields(
        queue_class=role,
        weight=weight,
        slice_us=slice_us,
        pressure_level=level if mode == "agentos_full_dag_psi_fairness" else "normal",
        critical=role == "latency",
        generation=generation,
        updated_at_ns=time.monotonic_ns(),
    )


def run_mixed_trial(
    *,
    mode: str,
    scheduler_binary: Path,
    simple_binary: Path,
    worker_binary: Path,
    duration_ms: int,
    seed: int,
    policy_overrides: dict[str, dict[str, Any]] | None = None,
    cpu_overrides: dict[str, int] | None = None,
) -> dict[str, Any]:
    if os.geteuid() != 0:
        raise PermissionError("Stage 4 trials must run as root")
    original_affinity: set[int] | None = None
    loader = None
    simple = None
    loader_events: list[dict[str, Any]] = []
    control: SchedExtControlClient | None = None
    heartbeat_stop = threading.Event()
    heartbeat_thread = None
    processes: list[tuple[str, subprocess.Popen[str], SchedPolicyKeyV1 | None]] = []
    policy_readbacks: list[dict[str, Any]] = []
    policy_bindings: list[dict[str, Any]] = []
    heartbeat_errors: list[str] = []
    generation = seed + 1
    initial_psi = cpu_psi_avg10()
    observed_level = pressure_level(initial_psi)
    level = policy_pressure_level(mode, observed_level)
    psi_trace: list[dict[str, Any]] = [{
        "monotonic_ns": time.monotonic_ns(),
        "cpu_some_avg10": initial_psi,
        "pressure_level": observed_level,
        "policy_pressure_level": level,
        "generation": generation,
    }]
    decisions = asyncio.run(dag_policies(level))
    flags = 1 if mode == "agentos_full_dag_psi_fairness" else 0
    loader_stdout = loader_stderr = ""

    try:
        original_affinity = pin_controller_cpu()
        if mode == "scx_simple":
            simple = _start_simple(simple_binary)
        elif mode.startswith("agentos_"):
            loader, loader_events = _start_loader(scheduler_binary)
            control = SchedExtControlClient()
            control.validate_ready()
            control.update_global_config(generation=generation, flags=flags)

        for index, (task_id, role, cpu, weight, slice_us) in enumerate(WORKERS):
            if cpu_overrides and task_id in cpu_overrides:
                cpu = cpu_overrides[task_id]
            command = [
                str(worker_binary), "--role", role,
                "--duration-ms", str(duration_ms),
                "--seed", str(seed * 100 + index + 1),
            ]
            if cpu_overrides and task_id in cpu_overrides:
                command.extend(["--cpu", str(cpu)])
            else:
                command.extend(["--cpus", WORKER_CPU_LIST])
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
            )
            wait_worker_barrier(process)
            key = None
            if mode != "cfs":
                if control is not None:
                    key = SchedPolicyKeyV1(tgid=process.pid, pid=process.pid)
                    policy = _policy_for(
                        mode, task_id, role, weight, slice_us,
                        generation, level, decisions,
                        (policy_overrides or {}).get(task_id),
                    )
                    readback = control.put_policy(key, policy)
                    policy_readbacks.append(
                        {
                            "phase": "initial",
                            "task_id": task_id,
                            "generation": generation,
                            "pressure_level": level,
                            "requested": {
                                "queue_class": role,
                                "weight": weight,
                                "slice_us": slice_us,
                            },
                            "decision": decisions.get(task_id),
                            "readback": control.policy_to_dict(readback),
                        }
                    )
                    policy_bindings.append(
                        {
                            "task_id": task_id,
                            "role": role,
                            "weight": weight,
                            "slice_us": slice_us,
                            "key": key,
                            "overrides": (policy_overrides or {}).get(task_id),
                        }
                    )
                os.sched_setscheduler(process.pid, SCHED_EXT, os.sched_param(0))
            processes.append((task_id, process, key))

        if control is not None:
            heartbeat_thread = threading.Thread(
                target=_heartbeat,
                args=(
                    heartbeat_stop, control, generation, flags, psi_trace,
                    mode, level, decisions, policy_bindings, policy_readbacks,
                    heartbeat_errors,
                ),
                daemon=True,
            )
            heartbeat_thread.start()

        observation_before = proc_cpu_snapshot()
        release_times: dict[str, int] = {}
        for task_id, process, _key in processes:
            assert process.stdin is not None
            release_times[task_id] = time.monotonic_ns()
            process.stdin.write(b"1")
            process.stdin.close()
            process.stdin = None

        worker_results: list[dict[str, Any]] = []
        for task_id, process, key in processes:
            stdout, stderr = process.communicate(timeout=duration_ms / 1000 + 10)
            payload = json.loads(stdout.decode("utf-8"))
            payload.update(
                task_id=task_id,
                return_code=process.returncode,
                stderr=stderr.decode("utf-8", errors="replace"),
            )
            payload["startup_wait_us"] = max(
                0,
                int(payload["started_at_ns"]) - release_times[task_id],
            ) / 1_000.0
            worker_results.append(payload)

        observation = proc_cpu_delta(observation_before, proc_cpu_snapshot())

        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=2)
            heartbeat_thread = None
        if control is not None:
            for _task_id, _process, key in processes:
                if key is not None:
                    control.delete_policy(key)

        if heartbeat_errors:
            raise RuntimeError(
                "heartbeat policy loop failed: " + "; ".join(heartbeat_errors)
            )

        policy_leaks = task_state_leaks = 0
        if control is not None:
            policy_leaks = _map_count(control.task_policy_path)
            time.sleep(0.1)
            task_state_leaks = _map_count(control.task_state_path)

        if loader is not None:
            loader_rc, loader_stdout, loader_stderr = _stop_loader(
                loader, loader_events, signal.SIGTERM
            )
            loader = None
            if loader_rc:
                raise RuntimeError(f"agentos loader exited with {loader_rc}")
        elif simple is not None:
            simple_rc, loader_stdout, loader_stderr = _stop_simple(simple)
            simple = None
            if simple_rc:
                raise RuntimeError(f"scx_simple exited with {simple_rc}")

        latency = next(item for item in worker_results if item["role"] == "latency")
        continuous = [item for item in worker_results if item["role"] != "latency"]
        batch = [item for item in worker_results if item["role"] == "batch"]
        final_stats = next(
            (
                event.get("stats", {})
                for event in reversed(loader_events)
                if event.get("event") == "final_stats"
            ),
            {},
        )
        return {
            "mode": mode,
            "controller_cpu": CONTROLLER_CPU,
            "seed": seed,
            "completed": all(
                item.get("completed") is True and item.get("return_code") == 0
                for item in worker_results
            ),
            "workers": worker_results,
            "latency_p95_us": percentile(latency["latencies_us"], 0.95),
            "throughput_iterations": sum(int(item["iterations"]) for item in continuous),
            "batch_max_gap_us": max(float(item["max_gap_us"]) for item in batch),
            "worker_startup_max_us": max(
                float(item["startup_wait_us"]) for item in worker_results
            ),
            **observation,
            "jain_fairness": jain_fairness(int(item["iterations"]) for item in continuous),
            "affinity_violations": sum(int(item["affinity_violations"]) for item in worker_results),
            "validation_checksums": [item["validation_checksum"] for item in worker_results],
            "pressure_level": level,
            "observed_pressure_level": observed_level,
            "pressure_avg10": cpu_psi_avg10(),
            "psi_trace": psi_trace,
            "policy_readbacks": policy_readbacks,
            "stats": final_stats,
            "policy_leaks": policy_leaks,
            "task_state_leaks": task_state_leaks,
            "loader_events": loader_events,
            "loader_stdout": loader_stdout,
            "loader_stderr": loader_stderr,
        }
    finally:
        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=2)
        for _task_id, process, _key in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
        if loader is not None and loader.poll() is None:
            _stop_loader(loader, loader_events, signal.SIGTERM)
        if simple is not None and simple.poll() is None:
            _stop_simple(simple)
        if original_affinity is not None:
            os.sched_setaffinity(0, original_affinity)
