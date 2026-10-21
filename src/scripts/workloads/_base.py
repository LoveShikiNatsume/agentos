"""Shared helpers for workload pressure scripts."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from agentos.benchmarks.cache_sim import CacheSimulatingMockBackend
from agentos.context.manager import ContextManager
from agentos.core.task import Task, TaskDAG, TaskStatus
from agentos.events.log import EventLog
from agentos.events.types import RuntimeEvent
from agentos.fault.injector import FaultInjectionPlan
from agentos.fault.recovery import ContextRetryPolicy
from agentos.runtime.engine import AgentHandler, Runtime
from agentos.scheduler.adaptive import AdaptivePolicyTuner


@dataclass
class InteractionOverhead:
    runtime_events: int = 0
    event_latency_ms_avg: float = 0.0
    message_count: int = 0
    payload_bytes: int = 0
    context_ref_count: int = 0
    contextref_bytes_saved: int = 0
    worker_startup_ms_avg: float = 0.0
    subprocess_roundtrip_ms_avg: float = 0.0
    stdin_payload_bytes: int = 0
    stdout_payload_bytes: int = 0
    total_llm_wait_secs: float = 0.0
    total_tool_wait_secs: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NormalizedMetrics:
    prompt_tokens_per_completed_task: float = 0.0
    total_tokens_per_completed_task: float = 0.0
    latency_per_completed_task_secs: float = 0.0
    cache_hit_ratio_improvement: float = 0.0
    tasks_completed_delta: int = 0
    fault_recovery_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkloadResult:
    workload_type: str
    mode: str  # "baseline" or "runtime"
    total_latency_secs: float = 0.0
    makespan_secs: float = 0.0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    cache_hit_ratio: float = 0.0
    tasks_completed: int = 0
    tasks_total: int = 0
    fault_injected: bool = False
    fault_recovered: bool = False
    context_refs_restored: int = 0
    tokens_saved_recovery: int = 0
    event_count: int = 0
    throughput_tasks_per_sec: float = 0.0
    avg_latency_per_task_secs: float = 0.0
    interaction_overhead: InteractionOverhead = field(default_factory=InteractionOverhead)
    normalized: NormalizedMetrics = field(default_factory=NormalizedMetrics)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _extract_token_metrics(backend: Any) -> dict[str, int]:
    if not hasattr(backend, "requests"):
        return _extract_token_metrics_from_runtime(None)
    total_prompt = 0
    total_hit = 0
    for i, req in enumerate(backend.requests):
        system = ""
        for msg in req.messages:
            if msg.get("role") == "system":
                system = msg.get("content", "")
        ptok = max(
            1,
            sum(len(m.get("content", "")) for m in req.messages) // 4,
        )
        total_prompt += ptok
        best = 0
        for j in range(i):
            prev_sys = ""
            for msg in backend.requests[j].messages:
                if msg.get("role") == "system":
                    prev_sys = msg.get("content", "")
            overlap = 0
            for a, b in zip(system, prev_sys):
                if a == b:
                    overlap += 1
                else:
                    break
            best = max(best, overlap // 4)
        total_hit += min(best, ptok)

    total_miss = total_prompt - total_hit
    hit_plus_miss = total_hit + total_miss
    ratio = total_hit / hit_plus_miss if hit_plus_miss > 0 else 0.0
    return {
        "prompt_tokens": total_prompt,
        "cache_hit_tokens": total_hit,
        "cache_miss_tokens": total_miss,
        "cache_hit_ratio": ratio,
    }


def _extract_token_metrics_from_runtime(
    rt: Runtime | None,
) -> dict[str, int]:
    """Fallback: extract token metrics from Runtime AgentStats."""
    if rt is None:
        return {"prompt_tokens": 0, "cache_hit_tokens": 0, "cache_miss_tokens": 0, "cache_hit_ratio": 0.0}
    total_sent = 0
    total_received = 0
    for acb in rt.agents.values():
        total_sent += acb.stats.tokens_sent
        total_received += acb.stats.tokens_received
    return {
        "prompt_tokens": total_sent,
        "cache_hit_tokens": 0,
        "cache_miss_tokens": total_sent,
        "cache_hit_ratio": 0.0,
    }


def _extract_fault_info(events: list[RuntimeEvent]) -> dict[str, Any]:
    fault_recovered = any(
        e.event_type == "fault.recovery_completed"
        and e.data.get("action") == "context_retry"
        for e in events
    )
    restore_events = [
        e for e in events if e.event_type == "fault.context_restored"
    ]
    refs_restored = (
        restore_events[0].data.get("context_refs_restored", 0)
        if restore_events
        else 0
    )
    tokens_saved = (
        restore_events[0].data.get("estimated_tokens_saved", 0)
        if restore_events
        else 0
    )
    return {
        "fault_injected": any(
            e.event_type == "fault.injected" for e in events
        ),
        "fault_recovered": fault_recovered,
        "context_refs_restored": refs_restored,
        "tokens_saved_recovery": tokens_saved,
    }


def _extract_interaction_overhead(
    rt: Runtime,
    events: list[RuntimeEvent],
    shared_context: dict[str, str],
) -> InteractionOverhead:
    overhead = InteractionOverhead()
    overhead.runtime_events = len(events)

    timestamps = [e.timestamp for e in events if e.timestamp > 0]
    if len(timestamps) >= 2:
        span = max(timestamps) - min(timestamps)
        overhead.event_latency_ms_avg = (span / len(timestamps)) * 1000

    bus_stats = rt._bus.stats()
    overhead.message_count = bus_stats.get("message_count", 0)
    overhead.payload_bytes = bus_stats.get("payload_bytes", 0)
    overhead.context_ref_count = bus_stats.get("context_ref_count", 0)

    total_shared_bytes = sum(len(v.encode("utf-8")) for v in shared_context.values())
    agents_count = max(1, len(rt.agents))
    overhead.contextref_bytes_saved = max(0, total_shared_bytes * (agents_count - 1))

    worker_events = [
        e for e in events
        if e.event_type in {"worker.started", "worker.finished", "worker.failed", "worker.timeout"}
    ]
    worker_starts: dict[str, float] = {}
    worker_durations: list[float] = []
    for e in worker_events:
        agent = e.agent_id or ""
        if e.event_type == "worker.started":
            worker_starts[agent] = e.timestamp
        elif e.event_type in {"worker.finished", "worker.failed", "worker.timeout"}:
            if agent in worker_starts:
                dur = e.timestamp - worker_starts[agent]
                worker_durations.append(dur)

    if worker_durations:
        overhead.worker_startup_ms_avg = (sum(worker_durations) / len(worker_durations)) * 1000
        overhead.subprocess_roundtrip_ms_avg = overhead.worker_startup_ms_avg

    total_llm_wait = 0.0
    total_tool_wait = 0.0
    stdin_bytes = 0
    stdout_bytes = 0
    for acb in rt.agents.values():
        total_llm_wait += acb.stats.total_llm_wait_secs
        total_tool_wait += acb.stats.total_tool_wait_secs

    for e in events:
        if e.event_type == "worker.finished":
            stdin_bytes += e.data.get("stdin_bytes", 0)
            stdout_bytes += e.data.get("stdout_bytes", 0)

    overhead.total_llm_wait_secs = total_llm_wait
    overhead.total_tool_wait_secs = total_tool_wait
    overhead.stdin_payload_bytes = stdin_bytes
    overhead.stdout_payload_bytes = stdout_bytes

    return overhead


def _compute_normalized(
    result: WorkloadResult,
    baseline: WorkloadResult | None = None,
) -> NormalizedMetrics:
    n = NormalizedMetrics()
    tc = result.tasks_completed or 1

    n.prompt_tokens_per_completed_task = result.prompt_tokens / tc
    n.total_tokens_per_completed_task = (result.prompt_tokens + result.completion_tokens) / tc
    n.latency_per_completed_task_secs = result.total_latency_secs / tc

    if baseline is not None:
        n.cache_hit_ratio_improvement = result.cache_hit_ratio - baseline.cache_hit_ratio
        n.tasks_completed_delta = result.tasks_completed - baseline.tasks_completed
        if result.fault_injected:
            n.fault_recovery_rate = 1.0 if result.fault_recovered else 0.0

    return n


async def run_workload(
    workload_type: str,
    dag: TaskDAG,
    baseline_handlers: dict[str, AgentHandler],
    runtime_handlers: dict[str, AgentHandler],
    shared_context: dict[str, str],
    backend: Any = None,
    fault_plan: dict[str, str] | None = None,
    event_dir: Path | None = None,
    force_policy: str | None = None,
    adaptive_tuned: bool = False,
) -> dict[str, WorkloadResult]:
    if backend is None:
        backend = CacheSimulatingMockBackend()

    _has_sim = hasattr(backend, "requests") and hasattr(backend, "call_count")

    results: dict[str, WorkloadResult] = {}

    # --- baseline ---
    if hasattr(backend, "reset"):
        backend.reset()
    dag_bl = _clone_dag(dag)
    rt_bl = Runtime(
        llm_backend=backend,
        handlers=baseline_handlers,
        fault_injection=FaultInjectionPlan(fault_plan) if fault_plan else None,
        force_scheduler_policy=force_policy,
    )

    start = time.monotonic()
    try:
        await rt_bl.run(dag_bl)
    except Exception:
        pass
    elapsed_bl = time.monotonic() - start

    completed_bl = sum(
        1 for t in dag_bl.tasks.values() if t.status == TaskStatus.COMPLETED
    )
    tok_bl = _extract_token_metrics(backend) if _has_sim else _extract_token_metrics_from_runtime(rt_bl)
    fault_bl = _extract_fault_info(rt_bl.events)

    _call_count_bl = backend.call_count if _has_sim else sum(a.stats.llm_calls for a in rt_bl.agents.values())
    bl = WorkloadResult(
        workload_type=workload_type,
        mode="baseline",
        total_latency_secs=elapsed_bl,
        makespan_secs=elapsed_bl,
        llm_calls=_call_count_bl,
        prompt_tokens=tok_bl["prompt_tokens"],
        completion_tokens=_call_count_bl * 10,
        cache_hit_tokens=tok_bl["cache_hit_tokens"],
        cache_miss_tokens=tok_bl["cache_miss_tokens"],
        cache_hit_ratio=tok_bl["cache_hit_ratio"],
        tasks_completed=completed_bl,
        tasks_total=len(dag_bl.tasks),
        event_count=len(rt_bl.events),
        **fault_bl,
    )
    bl.throughput_tasks_per_sec = (
        completed_bl / elapsed_bl if elapsed_bl > 0 else 0
    )
    bl.avg_latency_per_task_secs = (
        elapsed_bl / completed_bl if completed_bl > 0 else 0
    )
    bl.interaction_overhead = _extract_interaction_overhead(rt_bl, rt_bl.events, {})
    bl.normalized = _compute_normalized(bl)
    results["baseline"] = bl

    if event_dir:
        _save_events(event_dir / f"{workload_type}_baseline_events.jsonl", rt_bl.events)

    # --- runtime ---
    if hasattr(backend, "reset"):
        backend.reset()
    dag_rt = _clone_dag(dag)
    ctx_mgr = ContextManager()
    for key, val in shared_context.items():
        ctx_mgr.add_shared(key, val)

    log_path = event_dir / f"{workload_type}_runtime_events.jsonl" if event_dir else None

    # Create adaptive tuner if adaptive_tuned mode
    adaptive_tuner = AdaptivePolicyTuner(window_size=5) if adaptive_tuned else None

    rt_rt = Runtime(
        llm_backend=backend,
        handlers=runtime_handlers,
        context_manager=ctx_mgr,
        event_log=EventLog(log_path) if log_path else None,
        fault_injection=FaultInjectionPlan(fault_plan) if fault_plan else None,
        recovery_policy=ContextRetryPolicy(max_retries=1),
        force_scheduler_policy=force_policy,
        adaptive_tuner=adaptive_tuner,
    )

    start = time.monotonic()
    try:
        await rt_rt.run(dag_rt)
    except Exception:
        pass
    elapsed_rt = time.monotonic() - start

    completed_rt = sum(
        1 for t in dag_rt.tasks.values() if t.status == TaskStatus.COMPLETED
    )
    tok_rt = _extract_token_metrics(backend) if _has_sim else _extract_token_metrics_from_runtime(rt_rt)
    fault_rt = _extract_fault_info(rt_rt.events)

    _call_count_rt = backend.call_count if _has_sim else sum(a.stats.llm_calls for a in rt_rt.agents.values())
    rt_res = WorkloadResult(
        workload_type=workload_type,
        mode="runtime",
        total_latency_secs=elapsed_rt,
        makespan_secs=elapsed_rt,
        llm_calls=_call_count_rt,
        prompt_tokens=tok_rt["prompt_tokens"],
        completion_tokens=_call_count_rt * 10,
        cache_hit_tokens=tok_rt["cache_hit_tokens"],
        cache_miss_tokens=tok_rt["cache_miss_tokens"],
        cache_hit_ratio=tok_rt["cache_hit_ratio"],
        tasks_completed=completed_rt,
        tasks_total=len(dag_rt.tasks),
        event_count=len(rt_rt.events),
        **fault_rt,
    )
    rt_res.throughput_tasks_per_sec = (
        completed_rt / elapsed_rt if elapsed_rt > 0 else 0
    )
    rt_res.avg_latency_per_task_secs = (
        elapsed_rt / completed_rt if completed_rt > 0 else 0
    )
    rt_res.interaction_overhead = _extract_interaction_overhead(
        rt_rt, rt_rt.events, shared_context,
    )
    rt_res.normalized = _compute_normalized(rt_res, baseline=bl)
    results["runtime"] = rt_res

    return results


def _clone_dag(dag: TaskDAG) -> TaskDAG:
    new = TaskDAG()
    for t in dag.tasks.values():
        new.add_task(
            Task(
                task_id=t.task_id,
                name=t.name,
                agent_role=t.agent_role,
                dependencies=list(t.dependencies),
                priority=t.priority,
                workload_hint=t.workload_hint,
                io_profile=t.io_profile,
                expected_parallelism=t.expected_parallelism,
                sla_class=t.sla_class,
                deadline_ms=t.deadline_ms,
                previous_failures=t.previous_failures,
                resource_requirements=t.resource_requirements,
            )
        )
    return new


def _save_events(path: Path, events: list[RuntimeEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev.to_dict(), ensure_ascii=False) + "\n")
