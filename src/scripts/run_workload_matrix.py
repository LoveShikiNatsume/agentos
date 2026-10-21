"""Unified workload matrix runner.

Runs all four workload types (ai_inference, compile, compression, network)
in both baseline and runtime modes, producing a unified summary.json for
dashboard consumption and evidence reporting.

Supports multiple backends: mock, cache-sim, vllm, deepseek.

Usage:
    uv run python -m scripts.run_workload_matrix --backend mock --out runs/workload_matrix/mock
    uv run python scripts/run_workload_matrix.py --backend vllm --base-url http://127.0.0.1:8000 --model Qwen/Qwen2-7B
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from agentos.llm.factory import create_backend
from agentos.isolation.sched_ext import probe_sched_ext as runtime_probe_sched_ext

from scripts.probe_backend import probe_backend, BackendProbeResult
from scripts.probe_sched_ext import probe as probe_sched_ext
from scripts.workloads import ai_inference_pressure, compile_pressure, compression_pressure, network_pressure
from scripts.workloads._base import WorkloadResult

WORKLOADS = {
    "ai_inference": ai_inference_pressure,
    "compile": compile_pressure,
    "compression": compression_pressure,
    "network": network_pressure,
}


def _pct_delta(baseline: float, runtime: float) -> str:
    if baseline == 0:
        return "+inf" if runtime > 0 else "0.0%"
    delta = (runtime - baseline) / baseline * 100
    return f"{delta:+.1f}%"


async def _build_interaction_probe(all_results: dict) -> dict:
    """Build standalone MessageBus / ContextRef communication evidence.

    For each workload, sends two messages through the bus:
    - inline: carries the full shared context text in content (message_type="inline")
    - ref:    carries only a context_ref URI + small metadata (message_type="context_ref")

    This allows MessageBus.payload_savings() to compute the real byte difference.
    """
    from agentos.comm.bus import MessageBus
    from agentos.core.types import Message

    SHARED_CONTEXT_TEMPLATE = (
        "You are an AI agent in the AgentOS-Runtime multi-agent system. "
        "Your role is {role}. Follow the project coding standards and "
        "coordinate with other agents via the shared context pool. "
        "This shared system prompt is reused across all agents in the "
        "{workload} workload to demonstrate prefix-caching benefits. "
        "The context manager deduplicates this text so it is only stored "
        "once and passed by reference to subsequent agents. " * 3
    )

    bus = MessageBus()
    for wl_name, wl_data in all_results.items():
        rt = wl_data.get("runtime", {})
        oh = rt.get("interaction_overhead", {})
        ctx_saved = oh.get("contextref_bytes_saved", 0)

        shared_text = SHARED_CONTEXT_TEMPLATE.format(
            role=f"{wl_name}_worker", workload=wl_name,
        )

        num_agents = max(oh.get("runtime_events", 0) // 14, 1)
        for i in range(num_agents):
            await bus.send(Message(
                from_agent=f"{wl_name}_coordinator",
                to_agent=f"{wl_name}_agent_{i}",
                content={"system_prompt": shared_text, "agent_idx": i},
                message_type="inline",
            ))

            await bus.send(Message(
                from_agent=f"{wl_name}_coordinator",
                to_agent=f"{wl_name}_agent_{i}",
                content={"ref": f"ctx://{wl_name}/shared", "agent_idx": i},
                message_type="context_ref",
                context_refs=[f"ctx://{wl_name}/shared"],
            ))

    stats = bus.stats()
    savings = bus.payload_savings()

    total_ctx_saved = sum(
        wl.get("runtime", {}).get("interaction_overhead", {}).get("contextref_bytes_saved", 0)
        for wl in all_results.values()
    )

    return {
        "message_count": stats["message_count"],
        "context_ref_count": stats["context_ref_count"],
        "payload_bytes": stats["payload_bytes"],
        "inline_payload_bytes": savings.get("inline_payload_bytes", 0),
        "ref_payload_bytes": savings.get("ref_payload_bytes", 0),
        "bytes_saved": savings.get("bytes_saved", 0),
        "payload_reduction_ratio": savings.get("payload_reduction_ratio", 0),
        "contextref_bytes_saved_total": total_ctx_saved,
        "workload_count": len(all_results),
    }


async def run_matrix(
    backend_type: str,
    output_dir: Path,
    workloads: list[str] | None = None,
    base_url: str = "",
    model: str = "",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    event_dir = output_dir / "events"
    event_dir.mkdir(parents=True, exist_ok=True)

    # Clean old events to avoid mixing with historical data
    for old_file in event_dir.glob("*.jsonl"):
        old_file.unlink()

    selected = workloads or list(WORKLOADS.keys())

    # Probe backend availability
    bp: BackendProbeResult = await probe_backend(backend_type, base_url, model)

    # Create backend with graceful fallback
    # "mock" maps to "cache-sim" for workload matrix (needs cache simulation)
    actual_backend_type = "cache-sim" if backend_type == "mock" else backend_type
    fallback_reason = ""
    if not bp.available and actual_backend_type not in ("cache-sim", "vllm-sim", "deepseek-sim"):
        print(f"[WARN] Backend '{backend_type}' unavailable: {bp.fallback_reason}")
        print(f"[WARN] Falling back to cache-sim")
        actual_backend_type = "cache-sim"
        fallback_reason = bp.fallback_reason

    kwargs: dict = {}
    if actual_backend_type in ("vllm", "deepseek") and base_url:
        kwargs["base_url"] = base_url
    if actual_backend_type in ("vllm", "deepseek") and (model or bp.model):
        kwargs["model"] = model or bp.model
    try:
        backend = create_backend(actual_backend_type, **kwargs)
    except Exception as e:
        print(f"[WARN] Failed to create backend '{actual_backend_type}': {e}")
        print(f"[WARN] Falling back to cache-sim")
        backend = create_backend("cache-sim")
        actual_backend_type = "cache-sim"
        fallback_reason = str(e)[:200]

    all_results: dict[str, dict[str, dict]] = {}
    summary_table: list[dict] = []

    print(f"{'=' * 70}")
    print(f"  AgentOS-Runtime Workload Matrix Experiment")
    print(f"  Backend: {actual_backend_type} (requested: {backend_type})  |  Workloads: {', '.join(selected)}")
    if fallback_reason:
        print(f"  Fallback reason: {fallback_reason}")
    print(f"{'=' * 70}")
    print()

    matrix_start = time.monotonic()

    for wl_name in selected:
        mod = WORKLOADS[wl_name]
        print(f"[{wl_name}] Running baseline + runtime ...")
        results = await mod.run(backend=backend, event_dir=event_dir)

        bl = results["baseline"]
        rt = results["runtime"]

        all_results[wl_name] = {
            "baseline": bl.to_dict(),
            "runtime": rt.to_dict(),
            "comparison": {
                "latency_delta_pct": (
                    (rt.total_latency_secs - bl.total_latency_secs) / bl.total_latency_secs
                    if bl.total_latency_secs > 0
                    else 0
                ),
                "cache_hit_ratio_improvement": rt.cache_hit_ratio - bl.cache_hit_ratio,
                "tasks_completed_delta": rt.tasks_completed - bl.tasks_completed,
                "token_savings": bl.prompt_tokens - rt.prompt_tokens,
                "fault_recovery": rt.fault_recovered and not bl.fault_recovered,
                "event_count_delta": rt.event_count - bl.event_count,
            },
        }

        bl_norm = bl.normalized
        rt_norm = rt.normalized
        rt_oh = rt.interaction_overhead
        summary_table.append({
            "workload": wl_name,
            "bl_latency": f"{bl.total_latency_secs:.3f}s",
            "rt_latency": f"{rt.total_latency_secs:.3f}s",
            "bl_cache_ratio": f"{bl.cache_hit_ratio:.1%}",
            "rt_cache_ratio": f"{rt.cache_hit_ratio:.1%}",
            "bl_tasks": f"{bl.tasks_completed}/{bl.tasks_total}",
            "rt_tasks": f"{rt.tasks_completed}/{rt.tasks_total}",
            "bl_fault": "CRASHED" if bl.fault_injected and not bl.fault_recovered else "OK",
            "rt_fault": "RECOVERED" if rt.fault_recovered else ("OK" if not rt.fault_injected else "FAILED"),
            "bl_tok_per_task": f"{bl_norm.prompt_tokens_per_completed_task:.0f}",
            "rt_tok_per_task": f"{rt_norm.prompt_tokens_per_completed_task:.0f}",
            "rt_events": str(rt_oh.runtime_events),
            "rt_msg_count": str(rt_oh.message_count),
            "rt_ctx_saved_bytes": str(rt_oh.contextref_bytes_saved),
        })

        print(f"  baseline: {bl.total_latency_secs:.3f}s, cache={bl.cache_hit_ratio:.1%}, tasks={bl.tasks_completed}/{bl.tasks_total}")
        print(f"  runtime:  {rt.total_latency_secs:.3f}s, cache={rt.cache_hit_ratio:.1%}, tasks={rt.tasks_completed}/{rt.tasks_total}")
        print(f"  fault:    baseline={'CRASHED' if bl.fault_injected else 'N/A'} → runtime={'RECOVERED' if rt.fault_recovered else 'N/A'}")
        print()

    matrix_elapsed = time.monotonic() - matrix_start

    aggregate = _compute_aggregate(all_results)

    sched_probe = probe_sched_ext()

    interaction_probe = await _build_interaction_probe(all_results)

    # Collect scheduler evidence, runtime health evidence, SLA/Risk evidence, and Backpressure/Safety evidence from events
    scheduler_evidence = {}
    runtime_health_evidence = {}
    sla_risk_evidence = {}
    backpressure_safety_evidence = {}
    for wl_name in selected:
        # Read runtime events for this workload
        runtime_events_path = event_dir / f"{wl_name}_runtime_events.jsonl"
        if not runtime_events_path.exists():
            continue

        events = []
        for line in runtime_events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        # Extract scheduler evidence
        policies_seen = set()
        workload_class = "unknown"
        fairness_score = 1.0
        starvation_warnings = 0
        kernel_integration = "unavailable"
        bpf_map_mode = "mock"
        fallback_reason = None
        policy_switches = 0
        last_policy = None

        # Extract runtime health evidence
        health_scores = []
        queue_pressures = []
        resource_pressures = []
        fairness_pressures = []
        dependency_pressures = []
        pressure_levels = []
        all_health_warnings = set()

        # Extract SLA/Risk evidence
        sla_counts = {"critical": 0, "interactive": 0, "batch": 0, "best_effort": 0}
        deadline_task_count = 0
        max_urgency_score = 0.0
        max_task_risk_score = 0.0
        risk_aware_count = 0
        all_risk_warnings = set()

        for event in events:
            event_type = event.get("event_type", "")
            data = event.get("data", {})

            if event_type == "scheduler.decision":
                policy = data.get("policy")
                if policy:
                    policies_seen.add(policy)
                    if policy == "risk_aware":
                        risk_aware_count += 1
                    if last_policy is not None and policy != last_policy:
                        policy_switches += 1
                    last_policy = policy

                wl_class = data.get("workload_class")
                if wl_class and wl_class != "unknown":
                    workload_class = wl_class

                fairness = data.get("fairness", {})
                if fairness:
                    score = fairness.get("fairness_score")
                    if score is not None:
                        fairness_score = min(fairness_score, score)

                warnings = data.get("fairness_warnings", [])
                starvation_warnings += len(warnings)

                ki = data.get("kernel_integration")
                if ki:
                    kernel_integration = ki

                fr = data.get("fallback_reason")
                if fr:
                    fallback_reason = fr

                # Extract SLA/Risk from decision
                sla = data.get("sla", {})
                if sla:
                    for key in ["critical_count", "interactive_count", "batch_count", "best_effort_count"]:
                        sla_counts[key.replace("_count", "")] += sla.get(key, 0)
                    deadline_task_count += sla.get("deadline_task_count", 0)
                    max_urgency_score = max(max_urgency_score, sla.get("max_urgency_score", 0.0))

                risk = data.get("risk", {})
                if risk:
                    max_task_risk_score = max(max_task_risk_score, risk.get("max_task_risk_score", 0.0))
                    all_risk_warnings.update(risk.get("risk_warnings", []))

            elif event_type == "scheduler.runtime_health":
                score = data.get("runtime_health_score")
                if score is not None:
                    health_scores.append(score)
                qp = data.get("queue_pressure")
                if qp is not None:
                    queue_pressures.append(qp)
                rp = data.get("resource_pressure")
                if rp is not None:
                    resource_pressures.append(rp)
                fp = data.get("fairness_pressure")
                if fp is not None:
                    fairness_pressures.append(fp)
                dp = data.get("dependency_pressure")
                if dp is not None:
                    dependency_pressures.append(dp)
                pl = data.get("pressure_level")
                if pl:
                    pressure_levels.append(pl)
                hw = data.get("health_warnings", [])
                all_health_warnings.update(hw)

        scheduler_evidence[wl_name] = {
            "workload_class": workload_class,
            "policies_seen": sorted(policies_seen),
            "policy_switches": policy_switches,
            "fairness_score": fairness_score,
            "starvation_warnings": starvation_warnings,
            "kernel_integration": kernel_integration,
            "bpf_map_mode": bpf_map_mode,
            "fallback_reason": fallback_reason,
        }

        # Compute runtime health aggregates
        if health_scores:
            runtime_health_evidence[wl_name] = {
                "avg_health_score": sum(health_scores) / len(health_scores),
                "min_health_score": min(health_scores),
                "worst_pressure_level": _worst_pressure_level(pressure_levels),
                "max_queue_pressure": max(queue_pressures) if queue_pressures else 0.0,
                "max_resource_pressure": max(resource_pressures) if resource_pressures else 0.0,
                "max_fairness_pressure": max(fairness_pressures) if fairness_pressures else 0.0,
                "max_dependency_pressure": max(dependency_pressures) if dependency_pressures else 0.0,
                "health_warnings": sorted(all_health_warnings),
            }

        # Compute SLA/Risk aggregates
        sla_risk_evidence[wl_name] = {
            "critical_count": sla_counts.get("critical", 0),
            "interactive_count": sla_counts.get("interactive", 0),
            "batch_count": sla_counts.get("batch", 0),
            "best_effort_count": sla_counts.get("best_effort", 0),
            "deadline_task_count": deadline_task_count,
            "max_urgency_score": max_urgency_score,
            "max_task_risk_score": max_task_risk_score,
            "risk_aware_count": risk_aware_count,
            "risk_warnings": sorted(all_risk_warnings),
        }

        # Compute Backpressure/Safety aggregates
        max_backpressure_signal = "none"
        backpressure_event_count = 0
        allow_count = 0
        defer_count = 0
        degrade_count = 0
        reject_count = 0
        safety_violation_count = 0
        all_safety_violations = set()

        for event in events:
            event_type = event.get("event_type", "")
            data = event.get("data", {})

            if event_type == "scheduler.backpressure":
                backpressure_event_count += 1
                signal = data.get("signal", "none")
                signal_priority = {"none": 0, "soft": 1, "hard": 2, "rejecting": 3}
                if signal_priority.get(signal, 0) > signal_priority.get(max_backpressure_signal, 0):
                    max_backpressure_signal = signal

            elif event_type == "scheduler.safety_violation":
                safety_violation_count += 1
                violation = data.get("violation")
                if violation:
                    all_safety_violations.add(violation)

            elif event_type == "scheduler.decision":
                safety = data.get("safety", {})
                counts = safety.get("admission_counts", {})
                allow_count += counts.get("allow", 0)
                defer_count += counts.get("defer", 0)
                degrade_count += counts.get("degrade", 0)
                reject_count += counts.get("reject", 0)

        backpressure_safety_evidence[wl_name] = {
            "max_backpressure_signal": max_backpressure_signal,
            "backpressure_event_count": backpressure_event_count,
            "allow_count": allow_count,
            "defer_count": defer_count,
            "degrade_count": degrade_count,
            "reject_count": reject_count,
            "safety_violation_count": safety_violation_count,
            "safety_violations": sorted(all_safety_violations),
        }

    # Collect all events across workloads for kernel adapter evidence
    all_events: list[dict] = []
    for wl_name in selected:
        runtime_events_path = event_dir / f"{wl_name}_runtime_events.jsonl"
        if not runtime_events_path.exists():
            continue
        for line in runtime_events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                all_events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Use runtime probe for kernel_adapter_evidence (has KernelProbeResult format)
    runtime_probe = runtime_probe_sched_ext()
    kernel_adapter_evidence = _extract_kernel_adapter_evidence(
        all_events, kernel_probe=runtime_probe.to_dict(),
    )

    summary = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backend": actual_backend_type,
            "backend_requested": backend_type,
            "workloads": selected,
            "total_duration_secs": matrix_elapsed,
        },
        "backend_probe": bp.to_dict(),
        "workloads": all_results,
        "aggregate": aggregate,
        "interaction_probe": interaction_probe,
        "kernel_probe": sched_probe.summary,
        "kernel_probe_detail": sched_probe.to_dict(),
        "scheduler_evidence": scheduler_evidence,
        "runtime_health_evidence": runtime_health_evidence,
        "sla_risk_evidence": sla_risk_evidence,
        "backpressure_safety_evidence": backpressure_safety_evidence,
        "kernel_adapter_evidence": kernel_adapter_evidence,
    }

    json_path = output_dir / "summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    csv_path = output_dir / "summary.csv"
    _write_csv(csv_path, summary_table)

    md_path = output_dir / "report.md"
    md_path.write_text(_render_report(summary), encoding="utf-8")

    print(f"{'=' * 70}")
    print(f"  Matrix completed in {matrix_elapsed:.3f}s")
    print(f"  Output: {output_dir}")
    print(f"    summary.json  — full results")
    print(f"    summary.csv   — comparison table")
    print(f"    report.md     — markdown report")
    print(f"    events/       — per-workload event logs")
    print(f"{'=' * 70}")

    return summary


def _worst_pressure_level(levels: list[str]) -> str:
    """Find the worst pressure level from a list."""
    if not levels:
        return "normal"
    priority = {"normal": 0, "warning": 1, "overloaded": 2, "critical": 3}
    worst = "normal"
    for level in levels:
        if priority.get(level, 0) > priority.get(worst, 0):
            worst = level
    return worst


def _extract_kernel_adapter_evidence(
    events: list[dict],
    kernel_probe: dict | None = None,
) -> dict:
    """Aggregate kernel adapter evidence from workload events.

    Collects cgroup, pressure, BPF, and probe events into a structured
    summary suitable for ``summary.json.kernel_adapter_evidence``.
    """
    event_counts = {
        "cgroup_quota_applied": 0,
        "cgroup_unavailable": 0,
        "cgroup_cleanup_failed": 0,
        "pressure_sampled": 0,
        "pressure_system": 0,
        "pressure_agent_cgroup": 0,
        "bpf_map_updated": 0,
        "kernel_probe": 0,
        "scheduler_decision": 0,
    }

    pressure_levels: list[str] = []
    pressure_level_counts: dict[str, int] = {}
    quota_paths: set[str] = set()
    bpf_modes: set[str] = set()
    bpf_schema_versions: set[str] = set()
    fallback_reasons: set[str] = set()
    worst_pressure = "normal"
    _priority = {"normal": 0, "warning": 1, "overloaded": 2, "critical": 3}

    for event in events:
        etype = event.get("event_type", "")
        data = event.get("data", {})

        if etype == "resource.cgroup_quota_applied":
            event_counts["cgroup_quota_applied"] += 1
            path = data.get("path")
            if path:
                quota_paths.add(path)

        elif etype == "resource.cgroup_unavailable":
            event_counts["cgroup_unavailable"] += 1

        elif etype == "resource.cgroup_cleanup_failed":
            event_counts["cgroup_cleanup_failed"] += 1

        elif etype == "resource.pressure_sampled":
            event_counts["pressure_sampled"] += 1
            scope = data.get("scope", "")
            if scope == "system":
                event_counts["pressure_system"] += 1
            elif scope == "agent_cgroup":
                event_counts["pressure_agent_cgroup"] += 1

            # Extract pressure level from nested or top-level
            pressure = data.get("pressure") or {}
            level = pressure.get("pressure_level", data.get("pressure_level", "normal"))
            pressure_levels.append(level)
            pressure_level_counts[level] = pressure_level_counts.get(level, 0) + 1
            if _priority.get(level, 0) > _priority.get(worst_pressure, 0):
                worst_pressure = level

        elif etype == "scheduler.bpf_map_updated":
            event_counts["bpf_map_updated"] += 1
            mode = data.get("mode")
            if mode:
                bpf_modes.add(mode)

        elif etype == "scheduler.kernel_probe":
            event_counts["kernel_probe"] += 1
            fr = data.get("reason")
            if fr:
                fallback_reasons.add(fr)

        elif etype == "scheduler.decision":
            event_counts["scheduler_decision"] += 1
            # Extract BPF schema version from bpf_map.tasks
            bpf_map = data.get("bpf_map", {})
            tasks = bpf_map.get("tasks", {})
            for task_record in tasks.values():
                sv = task_record.get("schema_version")
                if sv:
                    bpf_schema_versions.add(sv)

            # Also collect fallback reason from decision
            fr = data.get("fallback_reason")
            if fr:
                fallback_reasons.add(fr)

    # Determine composite adapter_mode: <cgroup_part>+<bpf_part>
    # cgroup part: cgroup_v2 if actual quota writes observed, else user_space
    cgroup_observed = event_counts["cgroup_quota_applied"] > 0
    cgroup_prefix = "cgroup_v2" if cgroup_observed else "user_space"

    # BPF part: derived from kernel_probe.bpf_map
    if kernel_probe is None:
        bpf_suffix = "unavailable"
    else:
        bpf_map_val = kernel_probe.get("bpf_map", "mock")
        if bpf_map_val == "pinned_probe":
            bpf_suffix = "pinned_probe"
        elif bpf_map_val == "unavailable":
            bpf_suffix = "unavailable"
        else:
            bpf_suffix = "mock_bpf"

    adapter_mode = f"{cgroup_prefix}+{bpf_suffix}"

    # Aggregate probe reason into fallback_reasons
    if kernel_probe:
        probe_reason_str = kernel_probe.get("reason", "")
        if isinstance(probe_reason_str, str) and probe_reason_str.strip():
            fallback_reasons.add(probe_reason_str.strip())

    # fallback_active: from events OR from kernel_probe fallback_mode/reason
    probe_fallback_active = False
    if kernel_probe:
        probe_fm = kernel_probe.get("fallback_mode", "")
        probe_fallback_active = bool(probe_fm and probe_fm != "none") or bool(fallback_reasons)
    fallback_active = bool(fallback_reasons) or probe_fallback_active

    return {
        "schema_version": "kernel_adapter_evidence.v1",
        "adapter_mode": adapter_mode,
        "event_counts": event_counts,
        "cgroup_quota_applied_count": event_counts["cgroup_quota_applied"],
        "pressure_sample_count": event_counts["pressure_sampled"],
        "bpf_map_update_count": event_counts["bpf_map_updated"],
        "worst_pressure_level": worst_pressure,
        "pressure_level_counts": pressure_level_counts,
        "quota_paths": sorted(quota_paths),
        "bpf_modes": sorted(bpf_modes),
        "bpf_record_schema_versions": sorted(bpf_schema_versions),
        "probe": kernel_probe or {},
        "fallback_active": fallback_active,
        "fallback_reasons": sorted(fallback_reasons),
        "cgroup_control_observed": cgroup_observed,
    }


def _compute_aggregate(
    all_results: dict[str, dict[str, dict]],
) -> dict:
    total_bl_tokens = 0
    total_rt_tokens = 0
    total_bl_cache_hit = 0
    total_rt_cache_hit = 0
    total_bl_tasks = 0
    total_rt_tasks = 0
    total_tasks_total = 0
    faults_recovered = 0
    faults_total = 0
    total_rt_events = 0
    total_rt_messages = 0
    total_rt_payload_bytes = 0
    total_rt_contextref_saved = 0

    for wl in all_results.values():
        bl = wl["baseline"]
        rt = wl["runtime"]
        total_bl_tokens += bl["prompt_tokens"]
        total_rt_tokens += rt["prompt_tokens"]
        total_bl_cache_hit += bl["cache_hit_tokens"]
        total_rt_cache_hit += rt["cache_hit_tokens"]
        total_bl_tasks += bl["tasks_completed"]
        total_rt_tasks += rt["tasks_completed"]
        total_tasks_total += bl["tasks_total"]
        if rt["fault_injected"]:
            faults_total += 1
            if rt["fault_recovered"]:
                faults_recovered += 1
        rt_overhead = rt.get("interaction_overhead", {})
        total_rt_events += rt_overhead.get("runtime_events", 0)
        total_rt_messages += rt_overhead.get("message_count", 0)
        total_rt_payload_bytes += rt_overhead.get("payload_bytes", 0)
        total_rt_contextref_saved += rt_overhead.get("contextref_bytes_saved", 0)

    bl_ratio = total_bl_cache_hit / total_bl_tokens if total_bl_tokens > 0 else 0
    rt_ratio = total_rt_cache_hit / total_rt_tokens if total_rt_tokens > 0 else 0

    bl_per_task = total_bl_tokens / total_bl_tasks if total_bl_tasks > 0 else 0
    rt_per_task = total_rt_tokens / total_rt_tasks if total_rt_tasks > 0 else 0

    return {
        "total_prompt_tokens_baseline": total_bl_tokens,
        "total_prompt_tokens_runtime": total_rt_tokens,
        "prompt_tokens_per_task_baseline": bl_per_task,
        "prompt_tokens_per_task_runtime": rt_per_task,
        "avg_cache_hit_ratio_baseline": bl_ratio,
        "avg_cache_hit_ratio_runtime": rt_ratio,
        "cache_hit_ratio_improvement": rt_ratio - bl_ratio,
        "total_tasks_completed_baseline": total_bl_tasks,
        "total_tasks_completed_runtime": total_rt_tasks,
        "total_tasks": total_tasks_total,
        "fault_recovery_rate": faults_recovered / faults_total if faults_total > 0 else 0,
        "faults_recovered": faults_recovered,
        "faults_total": faults_total,
        "interaction_overhead": {
            "total_runtime_events": total_rt_events,
            "total_message_count": total_rt_messages,
            "total_payload_bytes": total_rt_payload_bytes,
            "total_contextref_bytes_saved": total_rt_contextref_saved,
        },
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_report(summary: dict) -> str:
    meta = summary["metadata"]
    agg = summary["aggregate"]
    wls = summary["workloads"]

    lines = [
        "# AgentOS-Runtime Workload Matrix Report",
        "",
        f"- **Timestamp**: {meta['timestamp']}",
        f"- **Backend**: {meta['backend']}",
        f"- **Duration**: {meta['total_duration_secs']:.3f}s",
        f"- **Workloads**: {', '.join(meta['workloads'])}",
        "",
        "## Aggregate",
        "",
        f"| Metric | Baseline | Runtime | Delta |",
        f"|--------|----------|---------|-------|",
        f"| Cache Hit Ratio | {agg['avg_cache_hit_ratio_baseline']:.1%} | {agg['avg_cache_hit_ratio_runtime']:.1%} | +{agg['cache_hit_ratio_improvement']:.1%} |",
        f"| Tasks Completed | {agg['total_tasks_completed_baseline']}/{agg['total_tasks']} | {agg['total_tasks_completed_runtime']}/{agg['total_tasks']} | +{agg['total_tasks_completed_runtime'] - agg['total_tasks_completed_baseline']} |",
        f"| Fault Recovery | 0/{agg['faults_total']} | {agg['faults_recovered']}/{agg['faults_total']} | +{agg['faults_recovered']} |",
        "",
        "## Per-Workload Results",
        "",
        "| Workload | BL Latency | RT Latency | BL Cache | RT Cache | BL Tasks | RT Tasks | Fault |",
        "|----------|-----------|-----------|----------|----------|----------|----------|-------|",
    ]

    for wl_name, wl_data in wls.items():
        bl = wl_data["baseline"]
        rt = wl_data["runtime"]
        fault_str = "RECOVERED" if rt["fault_recovered"] else "CRASHED"
        lines.append(
            f"| {wl_name} "
            f"| {bl['total_latency_secs']:.3f}s "
            f"| {rt['total_latency_secs']:.3f}s "
            f"| {bl['cache_hit_ratio']:.1%} "
            f"| {rt['cache_hit_ratio']:.1%} "
            f"| {bl['tasks_completed']}/{bl['tasks_total']} "
            f"| {rt['tasks_completed']}/{rt['tasks_total']} "
            f"| {fault_str} |"
        )

    # Normalized metrics table
    lines += [
        "",
        "## Normalized Metrics (per completed task)",
        "",
        "| Workload | BL Tok/Task | RT Tok/Task | BL Latency/Task | RT Latency/Task | Cache Improvement | Fault Recovery |",
        "|----------|------------|------------|----------------|----------------|-------------------|---------------|",
    ]
    for wl_name, wl_data in wls.items():
        bl = wl_data["baseline"]
        rt = wl_data["runtime"]
        bl_n = bl.get("normalized", {})
        rt_n = rt.get("normalized", {})
        lines.append(
            f"| {wl_name} "
            f"| {bl_n.get('prompt_tokens_per_completed_task', 0):.0f} "
            f"| {rt_n.get('prompt_tokens_per_completed_task', 0):.0f} "
            f"| {bl_n.get('latency_per_completed_task_secs', 0):.3f}s "
            f"| {rt_n.get('latency_per_completed_task_secs', 0):.3f}s "
            f"| +{rt_n.get('cache_hit_ratio_improvement', 0):.1%} "
            f"| {'Yes' if rt_n.get('fault_recovery_rate', 0) > 0 else 'No'} |"
        )

    # Interaction overhead section
    agg_oh = agg.get("interaction_overhead", {})
    lines += [
        "",
        "## Interaction Overhead Evidence",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Runtime Events | {agg_oh.get('total_runtime_events', 0)} |",
        f"| Total MessageBus Messages | {agg_oh.get('total_message_count', 0)} |",
        f"| Total MessageBus Payload Bytes | {agg_oh.get('total_payload_bytes', 0)} |",
        f"| ContextRef Bytes Saved (vs inline) | {agg_oh.get('total_contextref_bytes_saved', 0)} |",
        "",
        "### Per-Workload Overhead",
        "",
        "| Workload | RT Events | Messages | Payload Bytes | ContextRef Saved | Event Latency(ms) |",
        "|----------|----------|----------|--------------|-----------------|-------------------|",
    ]
    for wl_name, wl_data in wls.items():
        rt_oh = wl_data["runtime"].get("interaction_overhead", {})
        lines.append(
            f"| {wl_name} "
            f"| {rt_oh.get('runtime_events', 0)} "
            f"| {rt_oh.get('message_count', 0)} "
            f"| {rt_oh.get('payload_bytes', 0)} "
            f"| {rt_oh.get('contextref_bytes_saved', 0)} "
            f"| {rt_oh.get('event_latency_ms_avg', 0):.1f} |"
        )

    # Backend probe section
    bp = summary.get("backend_probe", {})
    if bp:
        lines += [
            "",
            "## Real Backend Probe",
            "",
            f"- **Backend**: {bp.get('backend', 'unknown')}",
            f"- **Available**: {'Yes' if bp.get('available') else 'No'}",
        ]
        if bp.get("base_url"):
            lines.append(f"- **Base URL**: {bp['base_url']}")
        if bp.get("model"):
            lines.append(f"- **Model**: {bp['model']}")
        if bp.get("models_endpoint"):
            lines.append(f"- **/v1/models**: accessible")
        if bp.get("metrics_endpoint"):
            lines.append(f"- **/metrics**: accessible")
            lines.append(f"- **Prefix Cache Metrics**: {'Yes' if bp.get('prefix_cache_metrics') else 'No'}")
        if bp.get("health_latency_ms"):
            lines.append(f"- **Health Latency**: {bp['health_latency_ms']:.1f}ms")
        if bp.get("fallback_reason"):
            lines.append(f"- **Fallback Reason**: {bp['fallback_reason']}")
        lines.append("")

    # MessageBus / ContextRef Communication Evidence
    ip = summary.get("interaction_probe", {})
    if ip:
        lines += [
            "## MessageBus / ContextRef Communication Evidence",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Message Count | {ip.get('message_count', 0)} |",
            f"| Context Ref Count | {ip.get('context_ref_count', 0)} |",
            f"| Payload Bytes | {ip.get('payload_bytes', 0)} |",
            f"| Inline Payload Bytes | {ip.get('inline_payload_bytes', 0)} |",
            f"| Ref Payload Bytes | {ip.get('ref_payload_bytes', 0)} |",
            f"| Bytes Saved | {ip.get('bytes_saved', 0)} |",
            f"| Payload Reduction Ratio | {ip.get('payload_reduction_ratio', 0):.1%} |",
            f"| ContextRef Bytes Saved (total) | {ip.get('contextref_bytes_saved_total', 0)} |",
            f"| Workload Count | {ip.get('workload_count', 0)} |",
            "",
        ]

    # Kernel probe
    kp = summary.get("kernel_probe", {})
    lines += [
        "## sched_ext / eBPF Probe",
        "",
        f"- **sched_ext**: {kp.get('sched_ext', 'unavailable')}",
        f"- **BPF Map**: {kp.get('bpf_map', 'mock')}",
        f"- **Reason**: {kp.get('reason', '-')}",
    ]
    # cgroup v2 info from kernel_probe_detail
    kpd = summary.get("kernel_probe_detail", {})
    if kpd.get("cgroup_v2_mounted") is not None:
        lines.append(f"- **cgroup v2**: {'mounted' if kpd['cgroup_v2_mounted'] else 'not mounted'}")
    if kpd.get("cgroup_v2_path"):
        lines.append(f"- **cgroup v2 path**: {kpd['cgroup_v2_path']}")
    if kpd.get("is_openeuler") and kpd.get("openeuler_version"):
        lines.append(f"- **openEuler version**: {kpd['openeuler_version']}")
    lines += [
        "",
        "> Note: sched_ext / eBPF status is probed at runtime. When unavailable, "
        "the system falls back to user-space mock adapters. The overhead evidence "
        "above measures user-space Runtime communication costs.",
    ]

    # Scheduler Evidence by Workload
    scheduler_evidence = summary.get("scheduler_evidence", {})
    if scheduler_evidence:
        lines += [
            "",
            "## Scheduler Evidence by Workload",
            "",
            "| Workload | workload_class | policies_seen | policy_switches | fairness_score | starvation_warnings | kernel_integration | bpf_map_mode | fallback_reason |",
            "|----------|---------------|---------------|-----------------|----------------|--------------------|--------------------|--------------|-----------------|",
        ]
        for wl_name, wl_data in scheduler_evidence.items():
            lines.append(
                f"| {wl_name} "
                f"| {wl_data.get('workload_class', 'unknown')} "
                f"| {', '.join(wl_data.get('policies_seen', []))} "
                f"| {wl_data.get('policy_switches', 0)} "
                f"| {wl_data.get('fairness_score', 1.0):.4f} "
                f"| {wl_data.get('starvation_warnings', 0)} "
                f"| {wl_data.get('kernel_integration', 'unavailable')} "
                f"| {wl_data.get('bpf_map_mode', 'mock')} "
                f"| {wl_data.get('fallback_reason', '-')} |"
            )
        lines.append("")

    # Runtime Health Evidence by Workload
    runtime_health_evidence = summary.get("runtime_health_evidence", {})
    if runtime_health_evidence:
        lines += [
            "",
            "## Runtime Health Evidence by Workload",
            "",
            "| Workload | avg_health_score | min_health_score | worst_pressure_level | max_queue_pressure | max_resource_pressure | max_fairness_pressure | max_dependency_pressure | health_warnings |",
            "|----------|------------------|------------------|----------------------|--------------------|-----------------------|-----------------------|-------------------------|-----------------|",
        ]
        for wl_name, wl_data in runtime_health_evidence.items():
            warnings = wl_data.get("health_warnings", [])
            warnings_str = "; ".join(warnings) if warnings else "-"
            lines.append(
                f"| {wl_name} "
                f"| {wl_data.get('avg_health_score', 1.0):.4f} "
                f"| {wl_data.get('min_health_score', 1.0):.4f} "
                f"| {wl_data.get('worst_pressure_level', 'normal')} "
                f"| {wl_data.get('max_queue_pressure', 0.0):.4f} "
                f"| {wl_data.get('max_resource_pressure', 0.0):.4f} "
                f"| {wl_data.get('max_fairness_pressure', 0.0):.4f} "
                f"| {wl_data.get('max_dependency_pressure', 0.0):.4f} "
                f"| {warnings_str} |"
            )
        lines.append("")

    # SLA / Risk Evidence by Workload
    sla_risk_evidence = summary.get("sla_risk_evidence", {})
    if sla_risk_evidence:
        lines += [
            "",
            "## SLA / Risk Evidence by Workload",
            "",
            "| Workload | critical | interactive | batch | best_effort | deadline_tasks | max_urgency | max_risk | risk_aware_count | risk_warnings |",
            "|----------|----------|-------------|-------|-------------|----------------|-------------|----------|------------------|---------------|",
        ]
        for wl_name, wl_data in sla_risk_evidence.items():
            warnings = wl_data.get("risk_warnings", [])
            warnings_str = "; ".join(warnings) if warnings else "-"
            lines.append(
                f"| {wl_name} "
                f"| {wl_data.get('critical_count', 0)} "
                f"| {wl_data.get('interactive_count', 0)} "
                f"| {wl_data.get('batch_count', 0)} "
                f"| {wl_data.get('best_effort_count', 0)} "
                f"| {wl_data.get('deadline_task_count', 0)} "
                f"| {wl_data.get('max_urgency_score', 0.0):.4f} "
                f"| {wl_data.get('max_task_risk_score', 0.0):.4f} "
                f"| {wl_data.get('risk_aware_count', 0)} "
                f"| {warnings_str} |"
            )
        lines.append("")

    # Backpressure / Safety Evidence by Workload
    backpressure_safety_evidence = summary.get("backpressure_safety_evidence", {})
    if backpressure_safety_evidence:
        lines += [
            "",
            "## Backpressure / Safety Evidence by Workload",
            "",
            "| Workload | max_backpressure | allow | defer | degrade | reject | safety_violations | violation_count |",
            "|----------|------------------|-------|-------|---------|--------|-------------------|-----------------|",
        ]
        for wl_name, wl_data in backpressure_safety_evidence.items():
            violations = wl_data.get("safety_violations", [])
            violations_str = "; ".join(violations) if violations else "-"
            lines.append(
                f"| {wl_name} "
                f"| {wl_data.get('max_backpressure_signal', 'none')} "
                f"| {wl_data.get('allow_count', 0)} "
                f"| {wl_data.get('defer_count', 0)} "
                f"| {wl_data.get('degrade_count', 0)} "
                f"| {wl_data.get('reject_count', 0)} "
                f"| {violations_str} "
                f"| {wl_data.get('safety_violation_count', 0)} |"
            )
        lines.append("")

    # Kernel Adapter Evidence section
    kae = summary.get("kernel_adapter_evidence", {})
    if kae:
        ec = kae.get("event_counts", {})
        lines += [
            "",
            "## Kernel Adapter Evidence",
            "",
            f"- **Schema**: {kae.get('schema_version', '-')}",
            f"- **Adapter Mode**: {kae.get('adapter_mode', 'mock')}",
            f"- **Worst Pressure Level**: {kae.get('worst_pressure_level', 'normal')}",
            f"- **Cgroup Control Observed**: {'Yes' if kae.get('cgroup_control_observed') else 'No'}",
            f"- **Fallback Active**: {'Yes' if kae.get('fallback_active') else 'No'}",
            "",
            "| Event Type | Count |",
            "|------------|-------|",
        ]
        for etype, count in ec.items():
            lines.append(f"| {etype} | {count} |")
        lines += [
            "",
            f"- **Pressure Samples**: {kae.get('pressure_sample_count', 0)}",
            f"- **BPF Map Updates**: {kae.get('bpf_map_update_count', 0)}",
            f"- **Cgroup Quota Applied**: {kae.get('cgroup_quota_applied_count', 0)}",
        ]
        if kae.get("quota_paths"):
            lines.append(f"- **Quota Paths**: {', '.join(kae['quota_paths'])}")
        if kae.get("bpf_modes"):
            lines.append(f"- **BPF Modes**: {', '.join(kae['bpf_modes'])}")
        if kae.get("bpf_record_schema_versions"):
            lines.append(f"- **BPF Record Schema**: {', '.join(kae['bpf_record_schema_versions'])}")
        if kae.get("pressure_level_counts"):
            plc = kae["pressure_level_counts"]
            parts = [f"{k}={v}" for k, v in sorted(plc.items())]
            lines.append(f"- **Pressure Level Distribution**: {', '.join(parts)}")
        lines.append("")

    lines += [
        "",
        "## Key Findings",
        "",
        f"1. **Context pooling** improved cache hit ratio by {agg['cache_hit_ratio_improvement']:.1%} across all workloads",
        f"2. **Fault recovery** restored {agg['faults_recovered']}/{agg['faults_total']} faulted tasks vs 0 in baseline",
        f"3. **Task completion** improved from {agg['total_tasks_completed_baseline']} to {agg['total_tasks_completed_runtime']} tasks",
        f"4. **Per-task token efficiency** baseline {agg.get('prompt_tokens_per_task_baseline', 0):.0f} → runtime {agg.get('prompt_tokens_per_task_runtime', 0):.0f} tokens/task",
        f"5. **Runtime observability** collected {agg_oh.get('total_runtime_events', 0)} events across all workloads",
    ]
    if ip.get("contextref_bytes_saved_total", 0) > 0:
        lines.append(
            f"6. **ContextRef savings** {ip['contextref_bytes_saved_total']} bytes saved via pass-by-reference across {ip.get('workload_count', 0)} workloads"
        )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run workload matrix experiment")
    parser.add_argument(
        "--backend",
        choices=["mock", "cache-sim", "vllm-sim", "deepseek-sim", "vllm", "deepseek"],
        default="mock",
        help="Backend type (default: mock)",
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="Base URL for real backends (e.g. http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model name for real backends",
    )
    parser.add_argument(
        "--out",
        default="runs/workload_matrix/mock",
        help="Output directory (default: runs/workload_matrix/mock)",
    )
    parser.add_argument(
        "--workloads",
        nargs="*",
        choices=list(WORKLOADS.keys()),
        help="Specific workloads to run (default: all)",
    )
    args = parser.parse_args()

    output_dir = Path(args.out)
    asyncio.run(run_matrix(
        args.backend, output_dir, args.workloads,
        base_url=args.base_url, model=args.model,
    ))


if __name__ == "__main__":
    main()
