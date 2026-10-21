from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentos.benchmarks.cache_sim import CacheSimulatingMockBackend
from agentos.benchmarks.experiment_stats import describe, percentile
from agentos.context.manager import ContextManager
from agentos.events.types import RuntimeEvent
from agentos.fault.injector import FaultInjectionPlan
from agentos.fault.recovery import ContextRetryPolicy
from agentos.isolation.kernel_adapter import (
    KernelApplyResult,
    PressureSample,
)
from agentos.isolation.sched_ext import KernelProbeResult
from agentos.llm.base import LLMRequest
from agentos.runtime.engine import Runtime
from agentos.scheduler.adaptive import AdaptivePolicyTuner
from scripts.workloads import (
    ai_inference_pressure,
    compile_pressure,
    compression_pressure,
    network_pressure,
)
from scripts.workloads._base import _extract_token_metrics


SCHEMA_VERSION = "agentos.stage5.experiment.v1"
WORKLOAD_NAMES = ("ai_inference", "compile", "compression", "network")
LANES = ("steady_state", "fault_recovery")
REQUIRED_METRICS = (
    "makespan_secs",
    "throughput_tasks_per_sec",
    "task_latency_p50_secs",
    "task_latency_p95_secs",
    "cpu_time_secs",
    "process_rss_peak_mb",
    "jain_fairness_index",
    "starvation_count",
    "fault_recovery_rate",
    "recovery_time_secs",
    "runtime_overhead_secs",
    "cache_hit_ratio",
    "cache_hit_tokens",
    "prefill_tokens_saved",
    "tasks_completed",
    "tasks_total",
    "task_failures",
    "pressure_affected_decisions",
)


@dataclass(frozen=True)
class GroupProfile:
    name: str
    dag_scheduler: bool
    fixed_policy: bool
    workload_aware: bool
    pressure_observation: bool
    feedback_control: bool
    context_refs: bool
    adaptive_tuning: bool
    fault_recovery: bool
    kernel_adapter: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


GROUP_PROFILES = {
    "bare_fifo": GroupProfile(
        "bare_fifo", False, False, False, False, False, False, False, False, False
    ),
    "dag_fixed": GroupProfile(
        "dag_fixed", True, True, False, False, False, False, False, False, False
    ),
    "workload_aware": GroupProfile(
        "workload_aware", True, False, True, True, False, False, False, False, False
    ),
    "feedback_control": GroupProfile(
        "feedback_control", True, False, True, True, True, False, False, False, True
    ),
    "full_runtime": GroupProfile(
        "full_runtime", True, False, True, True, True, True, True, True, True
    ),
}

_WORKLOAD_MODULES = {
    "ai_inference": ai_inference_pressure,
    "compile": compile_pressure,
    "compression": compression_pressure,
    "network": network_pressure,
}


@dataclass(frozen=True)
class TrialPlanEntry:
    trial_id: str
    workload: str
    lane: str
    group: str
    repetition: int
    sub_seed: int
    planned_position: int
    warmup: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sub_seed(seed: int, workload: str, lane: str, repetition: int) -> int:
    raw = f"{seed}:{workload}:{lane}:{repetition}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def build_trial_plan(
    workloads: tuple[str, ...] | list[str],
    *,
    repetitions: int,
    seed: int,
) -> list[TrialPlanEntry]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    unknown = set(workloads) - set(WORKLOAD_NAMES)
    if unknown:
        raise ValueError(f"unknown workloads: {sorted(unknown)}")

    plan: list[TrialPlanEntry] = []
    for workload in workloads:
        for lane in LANES:
            warmup_seed = _sub_seed(seed, workload, lane, 0)
            for position, group in enumerate(GROUP_PROFILES):
                plan.append(
                    TrialPlanEntry(
                        trial_id=f"warmup-{workload}-{lane}-{group}",
                        workload=workload,
                        lane=lane,
                        group=group,
                        repetition=0,
                        sub_seed=warmup_seed,
                        planned_position=position,
                        warmup=True,
                    )
                )

            for repetition in range(1, repetitions + 1):
                sub_seed = _sub_seed(seed, workload, lane, repetition)
                groups = list(GROUP_PROFILES)
                random.Random(sub_seed).shuffle(groups)
                for position, group in enumerate(groups):
                    plan.append(
                        TrialPlanEntry(
                            trial_id=(
                                f"{workload}-{lane}-r{repetition:02d}-"
                                f"p{position}-{group}"
                            ),
                            workload=workload,
                            lane=lane,
                            group=group,
                            repetition=repetition,
                            sub_seed=sub_seed,
                            planned_position=position,
                        )
                    )
    return plan


class _ReplayKernelAdapter:
    def __init__(
        self,
        *,
        control: bool,
        trace: list[str],
        publish_enabled: bool,
    ) -> None:
        self.control = control
        self.trace = trace
        self.publish_enabled = publish_enabled
        self.sample_count = 0
        self.publish_count = 0
        self._probe = KernelProbeResult(
            sched_ext="unavailable",
            bpf_map="mock",
            fallback_mode="stage5_replay",
            reason="cache-sim stage5 replay; no kernel takeover",
        )

    def probe(self) -> KernelProbeResult:
        return self._probe

    def sample_pressure(self, *, cgroup_path=None) -> PressureSample:
        index = min(self.sample_count, len(self.trace) - 1)
        observed = self.trace[index]
        self.sample_count += 1
        level = observed if self.control else "normal"
        value = {
            "normal": 0.0,
            "warning": 12.0,
            "critical": 25.0,
            "overloaded": 55.0,
        }[level]
        return PressureSample(
            cpu_some_avg10=value,
            pressure_level=level,
            pressure_source="replay",
            pressure_sources={
                "cpu": "replay",
                "memory": "replay",
                "io": "replay",
            },
            reason="deterministic stage5 PSI replay",
        )

    def publish_decision(self, decision: dict[str, Any]) -> list[RuntimeEvent]:
        if self.publish_enabled:
            self.publish_count += 1
        return []

    def apply_agent_quota(self, **kwargs) -> KernelApplyResult:
        return KernelApplyResult(
            available=False,
            mode="stage5_cache_sim",
            reason="coroutine experiment has no real cgroup",
        )

    def cleanup_agent(self, **kwargs) -> list[RuntimeEvent]:
        return []

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": "stage5_replay",
            "real_cgroup": False,
            "bpf_mode": "mock",
        }


def _module_context(module: Any) -> dict[str, str]:
    context: dict[str, str] = {}
    for name in dir(module):
        if not name.isupper():
            continue
        if "CONTEXT" not in name and "PREFIX" not in name:
            continue
        value = getattr(module, name)
        if isinstance(value, str):
            context[name.lower().replace("_", "-")] = value
    return context


def _rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text(
            encoding="utf-8"
        ).splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def _jain(values: list[float]) -> float:
    positives = [max(value, 1e-12) for value in values]
    if not positives:
        return 1.0
    denominator = len(positives) * sum(value * value for value in positives)
    return (sum(positives) ** 2 / denominator) if denominator else 1.0


async def _monitor(awaitable) -> tuple[Any, float]:
    task = asyncio.create_task(awaitable)
    peak = _rss_mb()
    while not task.done():
        peak = max(peak, _rss_mb())
        await asyncio.sleep(0.001)
    return await task, max(peak, _rss_mb())


async def _run_bare(
    module: Any,
    backend: CacheSimulatingMockBackend,
    *,
    inject_fault: bool,
) -> tuple[bool, list[float], list[RuntimeEvent]]:
    dag = module._build_dag()
    context = "\n\n".join(_module_context(module).values())
    completed: set[str] = set()
    durations: list[float] = []
    workload_success = True

    while len(completed) < len(dag.tasks):
        ready = [
            task
            for task in dag.tasks.values()
            if task.task_id not in completed
            and all(dep in completed for dep in task.dependencies)
        ]
        if not ready:
            workload_success = False
            break
        task = ready[0]
        if inject_fault and not durations:
            workload_success = False
            completed.add(task.task_id)
            continue
        started = time.monotonic()
        await backend.generate(
            LLMRequest(
                agent_id=f"bare-{task.agent_role}",
                task_id=task.task_id,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"{context}\n\nRole: {task.agent_role}; "
                            f"task: {task.name}"
                        ),
                    },
                    {"role": "user", "content": f"Execute {task.name}."},
                ],
            )
        )
        durations.append(time.monotonic() - started)
        completed.add(task.task_id)
    return workload_success, durations, []


def _task_durations(events: list[RuntimeEvent]) -> list[float]:
    starts: dict[str, float] = {}
    durations: list[float] = []
    for event in events:
        task_id = event.task_id or ""
        if event.event_type == "task.started":
            starts[task_id] = event.timestamp
        elif event.event_type in {"task.completed", "task.failed"}:
            if task_id in starts:
                durations.append(max(0.0, event.timestamp - starts[task_id]))
    return durations


async def _run_runtime(
    module: Any,
    backend: CacheSimulatingMockBackend,
    profile: GroupProfile,
    *,
    inject_fault: bool,
    trace: list[str],
) -> tuple[bool, list[float], list[RuntimeEvent], _ReplayKernelAdapter]:
    baseline_handlers, runtime_handlers = module.build_handlers()
    handlers = runtime_handlers if profile.context_refs else baseline_handlers
    context_manager = ContextManager()
    if profile.context_refs:
        for key, value in _module_context(module).items():
            context_manager.add_shared(key, value)
    adapter = _ReplayKernelAdapter(
        control=profile.feedback_control,
        trace=trace,
        publish_enabled=profile.kernel_adapter,
    )
    fault_target = next(iter(module._build_dag().tasks))
    runtime = Runtime(
        llm_backend=backend,
        handlers=handlers,
        context_manager=context_manager,
        fault_injection=(
            FaultInjectionPlan({fault_target: "kill"})
            if inject_fault
            else None
        ),
        recovery_policy=(
            ContextRetryPolicy(max_retries=1)
            if profile.fault_recovery
            else None
        ),
        force_scheduler_policy=(
            "latency_first" if profile.fixed_policy else None
        ),
        adaptive_tuner=(
            AdaptivePolicyTuner(window_size=5)
            if profile.adaptive_tuning
            else None
        ),
        kernel_adapter=adapter,
        enable_kernel_probe=False,
    )
    workload_success = True
    try:
        await runtime.run(module._build_dag())
    except Exception:
        workload_success = False
    if any(event.event_type == "task.failed" for event in runtime.events):
        workload_success = False
    return (
        workload_success,
        _task_durations(runtime.events),
        runtime.events,
        adapter,
    )


async def run_trial(
    entry: TrialPlanEntry,
    *,
    cache_latency_ms: float = 1.0,
) -> dict[str, Any]:
    profile = GROUP_PROFILES[entry.group]
    module = _WORKLOAD_MODULES[entry.workload]
    backend = CacheSimulatingMockBackend(
        latency_base=cache_latency_ms / 1000,
    )
    trace = ["normal", "warning", "normal", "warning", "normal"]
    inject_fault = entry.lane == "fault_recovery"
    cpu_started = time.process_time()
    wall_started = time.monotonic()
    adapter: _ReplayKernelAdapter | None = None
    error: str | None = None
    try:
        if entry.group == "bare_fifo":
            outcome, peak_rss = await _monitor(
                _run_bare(module, backend, inject_fault=inject_fault)
            )
            workload_success, task_durations, events = outcome
        else:
            outcome, peak_rss = await _monitor(
                _run_runtime(
                    module,
                    backend,
                    profile,
                    inject_fault=inject_fault,
                    trace=trace,
                )
            )
            workload_success, task_durations, events, adapter = outcome
        trial_status = "recorded"
    except Exception as exc:
        workload_success = False
        task_durations = []
        events = []
        peak_rss = _rss_mb()
        trial_status = "harness_error"
        error = f"{type(exc).__name__}: {exc}"

    makespan = max(time.monotonic() - wall_started, 1e-12)
    token_metrics = _extract_token_metrics(backend)
    completed_count = sum(
        event.event_type == "task.completed" for event in events
    )
    if entry.group == "bare_fifo":
        completed_count = len(task_durations)
    failure_count = sum(
        event.event_type == "task.failed" for event in events
    )
    scheduler_decisions = sum(
        event.event_type == "scheduler.decision" for event in events
    )
    starvation_count = sum(
        len(event.data.get("fairness_warnings", []))
        for event in events
        if event.event_type == "scheduler.decision"
    )
    recovery_count = sum(
        event.event_type == "fault.recovery_completed" for event in events
    )
    llm_wait = sum(
        event.data.get("latency_secs", 0.0)
        for event in events
        if event.event_type == "llm.response_received"
    )
    policies = sorted(
        {
            str(event.data.get("policy"))
            for event in events
            if event.event_type == "scheduler.decision"
            and event.data.get("policy")
        }
    )
    pressure_affected_decisions = sum(
        event.event_type == "scheduler.decision"
        and (
            event.data.get("pressure", {}).get("os_pressure_level")
            not in {None, "normal"}
        )
        for event in events
    )
    task_count = len(module._build_dag().tasks)
    metrics = {
        "makespan_secs": makespan,
        "throughput_tasks_per_sec": completed_count / makespan,
        "task_latency_p50_secs": (
            percentile(task_durations, 0.50)
            if task_durations
            else 0.0
        ),
        "task_latency_p95_secs": (
            percentile(task_durations, 0.95)
            if task_durations
            else 0.0
        ),
        "cpu_time_secs": max(0.0, time.process_time() - cpu_started),
        "process_rss_peak_mb": peak_rss,
        "jain_fairness_index": _jain(task_durations),
        "starvation_count": float(starvation_count),
        "fault_recovery_rate": (
            1.0 if inject_fault and recovery_count else 0.0
        ),
        "recovery_time_secs": sum(
            float(event.data.get("recovery_time_secs", 0.0))
            for event in events
            if event.event_type == "fault.recovery_completed"
        ),
        "runtime_overhead_secs": max(0.0, makespan - llm_wait),
        "cache_hit_ratio": float(token_metrics["cache_hit_ratio"]),
        "cache_hit_tokens": float(token_metrics["cache_hit_tokens"]),
        "prefill_tokens_saved": float(token_metrics["cache_hit_tokens"]),
        "tasks_completed": float(completed_count),
        "tasks_total": float(task_count),
        "task_failures": float(failure_count),
        "pressure_affected_decisions": float(
            pressure_affected_decisions
        ),
    }
    return {
        **entry.to_dict(),
        "actual_position": entry.planned_position,
        "backend": "cache-sim",
        "trial_status": trial_status,
        "workload_success": workload_success,
        "error": error,
        "capabilities": profile.to_dict(),
        "metrics": metrics,
        "metric_sources": {
            "makespan_secs": "time.monotonic",
            "cpu_time_secs": "time.process_time",
            "process_rss_peak_mb": "/proc/self/status VmRSS sampled",
            "cache_hit_ratio": "CacheSimulatingMockBackend",
            "pressure": "deterministic replay, not real cgroup",
        },
        "evidence": {
            "scheduler_decisions": scheduler_decisions,
            "selected_policies": policies,
            "pressure_trace_observed": profile.pressure_observation,
            "pressure_control_applied": profile.feedback_control,
            "pressure_source": (
                "replay" if profile.pressure_observation else "none"
            ),
            "pressure_samples": (
                adapter.sample_count if adapter is not None else 0
            ),
            "kernel_publications": (
                adapter.publish_count if adapter is not None else 0
            ),
            "real_cgroup_control": False,
            "bpf_mode": "mock",
        },
        "events": [event.to_dict() for event in events],
    }


def aggregate_trials(
    trials: list[dict[str, Any]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    measured = [trial for trial in trials if not trial["warmup"]]
    warmups = [trial for trial in trials if trial["warmup"]]
    workloads = sorted({trial["workload"] for trial in trials})
    cells: dict[str, Any] = {}
    for workload in workloads:
        cells[workload] = {}
        for lane in LANES:
            cells[workload][lane] = {}
            for group in GROUP_PROFILES:
                rows = [
                    trial
                    for trial in measured
                    if trial["workload"] == workload
                    and trial["lane"] == lane
                    and trial["group"] == group
                    and trial["trial_status"] == "recorded"
                ]
                metrics = {
                    metric: describe(
                        [float(row["metrics"][metric]) for row in rows]
                    )
                    for metric in REQUIRED_METRICS
                    if rows
                }
                cells[workload][lane][group] = {
                    "recorded_trials": len(rows),
                    "harness_failures": sum(
                        trial["trial_status"] != "recorded"
                        for trial in measured
                        if trial["workload"] == workload
                        and trial["lane"] == lane
                        and trial["group"] == group
                    ),
                    "workload_success_rate": (
                        sum(bool(row["workload_success"]) for row in rows)
                        / len(rows)
                        if rows
                        else 0.0
                    ),
                    "metrics": metrics,
                }
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "backend": "cache-sim",
            "seed": seed,
            "repetitions": repetitions,
            "workloads": workloads,
            "groups": list(GROUP_PROFILES),
            "lanes": list(LANES),
            "warmup_count": len(warmups),
            "measured_trial_count": len(measured),
            "harness_failure_count": sum(
                trial["trial_status"] != "recorded" for trial in measured
            ),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pid": os.getpid(),
        },
        "capability_manifests": {
            name: profile.to_dict()
            for name, profile in GROUP_PROFILES.items()
        },
        "cells": cells,
        "claim_boundary": {
            "feedback_source": "deterministic PSI replay",
            "real_cgroup_control": False,
            "real_pinned_bpf_map": False,
            "bpf_mode": "mock",
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
    meta = summary["metadata"]
    lines = [
        "# AgentOS Stage 5 National-Level Experiment Report",
        "",
        "## Methodology",
        "",
        f"- Backend: `{meta['backend']}`",
        f"- Repetitions per cell: {meta['repetitions']}",
        f"- Master seed: {meta['seed']}",
        "- Statistics: sample standard deviation and two-sided Student-t 95% CI.",
        "- Warmups are excluded from statistics.",
        "",
        "## Steady-State Makespan",
        "",
        "| Workload | Group | Mean (s) | Stddev | 95% CI |",
        "|---|---|---:|---:|---:|",
    ]
    for workload, lanes in summary["cells"].items():
        for group, cell in lanes["steady_state"].items():
            stat = cell["metrics"].get("makespan_secs")
            if not stat:
                continue
            lines.append(
                f"| {workload} | {group} | {stat['mean']:.6f} "
                f"| {stat['stddev']:.6f} "
                f"| [{stat['ci95_low']:.6f}, {stat['ci95_high']:.6f}] |"
            )
    lines.extend(
        [
            "",
            "## Limitations and Capability Boundary",
            "",
            "- Feedback values are deterministic PSI replay, not real cgroup samples.",
            "- cache-sim results do not prove real cgroup enforcement.",
            "- Real cgroup v2 and pinned BPF map evidence is provided by the privileged stage 4 verifier.",
            "- No BPF program attachment, sched_ext activation, or kernel scheduling takeover is claimed.",
            "",
        ]
    )
    return "\n".join(lines)


def write_artifacts(
    *,
    output_dir: Path,
    plan: list[TrialPlanEntry],
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
    raw_lines: list[str] = []
    events_dir = output_dir / "events"
    events_dir.mkdir(exist_ok=True)
    for trial in trials:
        raw = dict(trial)
        events = raw.pop("events", [])
        raw_lines.append(json.dumps(raw, sort_keys=True))
        _atomic_text(
            events_dir / f"{trial['trial_id']}.jsonl",
            "".join(
                json.dumps(event, sort_keys=True) + "\n"
                for event in events
            ),
        )
    _atomic_text(
        output_dir / "raw_trials.jsonl",
        "\n".join(raw_lines) + "\n",
    )
    _atomic_text(
        output_dir / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )

    csv_path = output_dir / "summary.csv"
    csv_tmp = csv_path.with_name(csv_path.name + ".tmp")
    with csv_tmp.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "workload",
            "lane",
            "group",
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
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for workload, lanes in summary["cells"].items():
            for lane, groups in lanes.items():
                for group, cell in groups.items():
                    for metric, stat in cell["metrics"].items():
                        writer.writerow(
                            {
                                "workload": workload,
                                "lane": lane,
                                "group": group,
                                "metric": metric,
                                **stat,
                            }
                        )
    csv_tmp.replace(csv_path)
    _atomic_text(output_dir / "report.md", _report(summary))
