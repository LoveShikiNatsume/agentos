"""Scheduling policy ablation experiment.

Runs all four workload types under each scheduling policy and produces
summary.json / summary.csv / report.md for competition evidence.

Usage:
    .venv/bin/python scripts/ablation_scheduling.py --backend mock --runs 3 --out runs/ablation_scheduling
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.workloads import (
    ai_inference_pressure,
    compile_pressure,
    compression_pressure,
    network_pressure,
)
from agentos.benchmarks.experiment_stats import describe

WORKLOAD_MODULES = {
    "ai_inference": ai_inference_pressure,
    "compile": compile_pressure,
    "compression": compression_pressure,
    "network": network_pressure,
}

POLICIES = [
    "adaptive",
    "adaptive_tuned",
    "latency_first",
    "throughput_first",
    "cache_affinity",
    "fairness_balanced",
    "risk_aware",
]
EXPERIMENT_POLICIES = ["fifo", *POLICIES]

# Metrics we extract from each run
METRIC_FIELDS = [
    "total_latency_secs",
    "completed_tasks",
    "total_tasks",
    "tasks_completed_rate",
    "latency_per_completed_task_secs",
    "throughput_tasks_per_sec",
    "cache_hit_ratio",
    "avg_wait_ms",
    "max_wait_ms",
    "fairness_score",
    "starvation_warning_count",
    "policy_switch_count",
    "unique_high_risk_task_count",
    "max_urgency_score",
    "unique_critical_sla_task_count",
    "unique_deadline_task_count",
    "safety_reject_count",
    "backpressure_signal_max",
    "adaptive_tuning_decision_count",
    "adaptive_tuning_event_count",
    "adaptive_policy_adjustment_count",
    "deadline_miss_count",
    "cpu_throttled_usec",
    "runtime_cpu_secs",
    "runtime_peak_memory_bytes",
]


def _extract_metrics_from_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract scheduler metrics from event list.

    Unique task counts are derived by scanning the per-task decision dict
    inside each scheduler.decision event, NOT by summing the SLA/Risk
    aggregate counts (which would over-count across scheduling rounds).
    """
    metrics: dict[str, Any] = {
        "avg_wait_ms": 0.0,
        "max_wait_ms": 0.0,
        "fairness_score": 1.0,
        "starvation_warning_count": 0,
        "policy_switch_count": 0,
        "unique_high_risk_task_count": 0,
        "max_urgency_score": 0.0,
        "unique_critical_sla_task_count": 0,
        "unique_deadline_task_count": 0,
        "safety_reject_count": 0,
        "backpressure_signal_max": 0,
        "adaptive_tuning_decision_count": 0,
        "adaptive_tuning_event_count": 0,
        "adaptive_policy_adjustment_count": 0,
    }

    policies_seen: list[str] = []
    fairness_scores: list[float] = []
    wait_values: list[float] = []
    max_wait_values: list[float] = []
    urgency_values: list[float] = []
    bp_signals: dict[str, int] = {"none": 0, "defer": 1, "degrade": 2, "reject": 3}

    # Sets for unique task ID tracking across all scheduling rounds
    unique_deadline_ids: set[str] = set()
    unique_critical_ids: set[str] = set()
    unique_high_risk_ids: set[str] = set()

    for event in events:
        etype = event.get("event_type", "")
        data = event.get("data", {})

        if etype == "scheduler.decision":
            policy = data.get("policy", "")
            if policy:
                policies_seen.append(policy)

            # Fairness
            fairness = data.get("fairness", {})
            fs = fairness.get("fairness_score")
            if fs is not None:
                fairness_scores.append(fs)
            aw = fairness.get("avg_wait_ms")
            if aw is not None:
                wait_values.append(aw)
            mw = fairness.get("max_wait_ms")
            if mw is not None:
                max_wait_values.append(mw)

            # Fairness warnings
            warnings = data.get("fairness_warnings", [])
            metrics["starvation_warning_count"] += len(warnings)

            # Per-task unique tracking from the tasks dict
            tasks_dict = data.get("tasks", {})
            for tid, tdata in tasks_dict.items():
                if tdata.get("deadline_ms") is not None:
                    unique_deadline_ids.add(tid)
                if tdata.get("sla_class") == "critical":
                    unique_critical_ids.add(tid)
                if tdata.get("task_risk_score", 0.0) >= 0.7:
                    unique_high_risk_ids.add(tid)

            # Also collect from risk.high_risk_task_ids (complementary)
            risk = data.get("risk", {})
            for tid in risk.get("high_risk_task_ids", []):
                unique_high_risk_ids.add(tid)

            # Max urgency across all decisions
            sla = data.get("sla", {})
            urg = sla.get("max_urgency_score", 0.0)
            if urg:
                urgency_values.append(urg)

            # Safety (cumulative observation count is fine here)
            safety = data.get("safety", {})
            reject_count = safety.get("reject_count", 0)
            if isinstance(reject_count, int):
                metrics["safety_reject_count"] += reject_count

            # Backpressure
            bp = data.get("backpressure", {})
            bp_signal = bp.get("signal", "none")
            bp_val = bp_signals.get(bp_signal, 0)
            metrics["backpressure_signal_max"] = max(
                metrics["backpressure_signal_max"], bp_val
            )

            # Adaptive tuning from decision (count decisions that have it enabled)
            adaptive_tuning = data.get("adaptive_tuning")
            if adaptive_tuning and adaptive_tuning.get("enabled"):
                metrics["adaptive_tuning_decision_count"] += 1
                policy_bias = adaptive_tuning.get("policy_bias", {})
                if any(v != 0 for v in policy_bias.values()):
                    metrics["adaptive_policy_adjustment_count"] += 1

        elif etype == "scheduler.adaptive_tuning":
            # Count standalone adaptive_tuning events (one per decision with tuner)
            metrics["adaptive_tuning_event_count"] += 1

    # Policy switch count
    if len(policies_seen) > 1:
        switches = sum(
            1 for i in range(1, len(policies_seen))
            if policies_seen[i] != policies_seen[i - 1]
        )
        metrics["policy_switch_count"] = switches

    # Aggregate unique counts from sets
    metrics["unique_deadline_task_count"] = len(unique_deadline_ids)
    metrics["unique_critical_sla_task_count"] = len(unique_critical_ids)
    metrics["unique_high_risk_task_count"] = len(unique_high_risk_ids)

    # Aggregate fairness/wait/urgency
    if fairness_scores:
        metrics["fairness_score"] = sum(fairness_scores) / len(fairness_scores)
    if wait_values:
        metrics["avg_wait_ms"] = sum(wait_values) / len(wait_values)
    if max_wait_values:
        metrics["max_wait_ms"] = max(max_wait_values)
    if urgency_values:
        metrics["max_urgency_score"] = max(urgency_values)

    return metrics


def _extract_metrics_from_result(
    result: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Extract full metrics from workload result + events."""
    rt = result.get("runtime", {})
    bl = result.get("baseline", {})

    completed = rt.get("tasks_completed", 0)
    total = rt.get("tasks_total", 0)
    elapsed = rt.get("total_latency_secs", 0.0)

    event_metrics = _extract_metrics_from_events(events)

    return {
        "total_latency_secs": elapsed,
        "completed_tasks": completed,
        "total_tasks": total,
        "tasks_completed_rate": completed / total if total > 0 else 0.0,
        "latency_per_completed_task_secs": elapsed / completed if completed > 0 else 0.0,
        "throughput_tasks_per_sec": rt.get("throughput_tasks_per_sec", 0.0),
        "cache_hit_ratio": rt.get("cache_hit_ratio", 0.0),
        "avg_wait_ms": event_metrics["avg_wait_ms"],
        "max_wait_ms": event_metrics["max_wait_ms"],
        "fairness_score": event_metrics["fairness_score"],
        "starvation_warning_count": event_metrics["starvation_warning_count"],
        "policy_switch_count": event_metrics["policy_switch_count"],
        "unique_high_risk_task_count": event_metrics["unique_high_risk_task_count"],
        "max_urgency_score": event_metrics["max_urgency_score"],
        "unique_critical_sla_task_count": event_metrics["unique_critical_sla_task_count"],
        "unique_deadline_task_count": event_metrics["unique_deadline_task_count"],
        "safety_reject_count": event_metrics["safety_reject_count"],
        "backpressure_signal_max": event_metrics["backpressure_signal_max"],
        "adaptive_tuning_decision_count": event_metrics["adaptive_tuning_decision_count"],
        "adaptive_tuning_event_count": event_metrics["adaptive_tuning_event_count"],
        "adaptive_policy_adjustment_count": event_metrics["adaptive_policy_adjustment_count"],
        "deadline_miss_count": 0,
        "cpu_throttled_usec": 0,
        "runtime_cpu_secs": elapsed,
        "runtime_peak_memory_bytes": rt.get("peak_memory_bytes", 0),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _recommend_policy(workload_metrics: dict[str, dict[str, list[dict[str, Any]]]]) -> str:
    """Pick the best policy for a workload based on simple rules."""
    best_policy = "adaptive"
    best_score = (-1, float("inf"), -1.0, -1.0)  # (completion_rate, -latency, fairness, cache)

    for policy, runs in workload_metrics.items():
        if not runs:
            continue
        cr = _mean([r["tasks_completed_rate"] for r in runs])
        lat = _mean([r["latency_per_completed_task_secs"] for r in runs])
        fs = _mean([r["fairness_score"] for r in runs])
        ch = _mean([r["cache_hit_ratio"] for r in runs])
        score = (cr, -lat, fs, ch)
        if score > best_score:
            best_score = score
            best_policy = policy

    return best_policy


def validate_competition_settings(
    *,
    runs: int,
    warmup_runs: int = 0,
    competition: bool = False,
) -> None:
    if runs < 1:
        raise ValueError("runs must be at least 1")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    if competition and runs < 20:
        raise ValueError("competition mode requires at least 20 runs")


def _successful_samples(
    raw_samples: list[dict[str, Any]],
    *,
    workload: str,
    policy: str,
) -> list[dict[str, Any]]:
    return [
        sample
        for sample in raw_samples
        if sample.get("workload") == workload
        and sample.get("policy") == policy
        and sample.get("success", True) is True
    ]


def build_summary_from_raw(
    raw_samples: list[dict[str, Any]],
    *,
    workloads: list[str],
    policies: list[str],
    runs: int,
    warmup_runs: int,
    backend_name: str,
    competition: bool,
    seed: int,
) -> dict[str, Any]:
    validate_competition_settings(
        runs=runs,
        warmup_runs=warmup_runs,
        competition=competition,
    )
    summary: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backend": backend_name,
            "runs_per_policy": runs,
            "warmup_runs": warmup_runs,
            "competition": competition,
            "seed": seed,
            "workloads": workloads,
            "policies": policies,
        },
        "raw_samples": raw_samples,
        "results": {},
        "recommendations": {},
    }

    for workload_name in workloads:
        summary["results"][workload_name] = {}
        workload_metrics: dict[str, list[dict[str, Any]]] = {}
        for policy in policies:
            samples = _successful_samples(
                raw_samples,
                workload=workload_name,
                policy=policy,
            )
            if competition and len(samples) < 20:
                raise ValueError(
                    f"effective samples for {workload_name}/{policy} "
                    f"must be at least 20"
                )
            workload_metrics[policy] = samples
            cell: dict[str, Any] = {
                "sample_count": len(samples),
                "failed_count": sum(
                    1
                    for sample in raw_samples
                    if sample.get("workload") == workload_name
                    and sample.get("policy") == policy
                    and sample.get("success", True) is not True
                ),
            }
            for field in METRIC_FIELDS:
                values = [
                    float(sample.get(field, 0.0))
                    for sample in samples
                    if isinstance(sample.get(field, 0.0), (int, float))
                    and not isinstance(sample.get(field, 0.0), bool)
                ]
                if not values:
                    continue
                stat = describe(values)
                cell[field] = stat
                cell[f"mean_{field}"] = stat["mean"]
            summary["results"][workload_name][policy] = cell
        summary["recommendations"][workload_name] = _recommend_policy(workload_metrics)

    return summary


async def run_ablation(
    workloads: list[str],
    policies: list[str],
    runs: int,
    backend_name: str,
    out_dir: Path,
    *,
    warmup_runs: int = 0,
    competition: bool = False,
    seed: int = 20260705,
) -> dict[str, Any]:
    """Run the full ablation experiment."""
    from agentos.llm.factory import create_backend

    validate_competition_settings(
        runs=runs,
        warmup_runs=warmup_runs,
        competition=competition,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    events_dir = out_dir / "events"
    events_dir.mkdir(exist_ok=True)

    # Create backend
    backend = create_backend(backend_name)

    raw_samples: list[dict[str, Any]] = []

    total_runs = len(workloads) * len(policies) * (runs + warmup_runs)
    completed_runs = 0

    for workload_name in workloads:
        if workload_name not in WORKLOAD_MODULES:
            print(f"  [WARN] Unknown workload: {workload_name}, skipping")
            continue
        mod = WORKLOAD_MODULES[workload_name]

        for policy in policies:
            if policy in ("adaptive", "adaptive_tuned", "fifo"):
                force_policy = None
            else:
                force_policy = policy

            for run_idx in range(1, runs + warmup_runs + 1):
                completed_runs += 1
                is_warmup = run_idx <= warmup_runs
                effective_index = run_idx - warmup_runs
                label = f"{workload_name}/{policy}/run{run_idx}"
                print(f"  [{completed_runs}/{total_runs}] {label} ...", end=" ", flush=True)

                # Reset backend state
                if hasattr(backend, "reset"):
                    backend.reset()

                # Run workload
                try:
                    result = await mod.run(
                        backend=backend,
                        event_dir=events_dir,
                        force_policy=force_policy,
                        adaptive_tuned=(policy == "adaptive_tuned"),
                    )

                    # Load events for metric extraction
                    events: list[dict[str, Any]] = []
                    # Try to load from event file
                    event_path = events_dir / f"{workload_name}_runtime_events.jsonl"
                    if event_path.exists():
                        with event_path.open("r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    try:
                                        events.append(json.loads(line))
                                    except json.JSONDecodeError:
                                        pass

                    # result is dict[str, WorkloadResult] - convert to dict for extraction
                    result_dict: dict[str, Any] = {}
                    for mode, wr in result.items():
                        result_dict[mode] = wr.to_dict() if hasattr(wr, "to_dict") else wr

                    metrics = _extract_metrics_from_result(result_dict, events)
                    sample = {
                        "workload": workload_name,
                        "policy": policy,
                        "run_index": effective_index,
                        "seed": seed + run_idx - 1,
                        "success": True,
                        **metrics,
                    }
                    if not is_warmup:
                        raw_samples.append(sample)

                    # Save per-run events with policy label
                    if event_path.exists():
                        labeled_path = events_dir / f"{workload_name}_{policy}_run{run_idx}.jsonl"
                        labeled_path.write_text(event_path.read_text())

                    rt = result.get("runtime")
                    rt_dict = rt.to_dict() if hasattr(rt, "to_dict") else rt
                    print(f"OK  completed={rt_dict.get('tasks_completed', 0)}/{rt_dict.get('tasks_total', 0)}  latency={rt_dict.get('total_latency_secs', 0):.3f}s")

                except Exception as exc:
                    print(f"FAIL: {exc}")
                    if not is_warmup:
                        raw_samples.append(
                            {
                                "workload": workload_name,
                                "policy": policy,
                                "run_index": effective_index,
                                "seed": seed + run_idx - 1,
                                "success": False,
                                "failure_reason": str(exc),
                            }
                        )

    return build_summary_from_raw(
        raw_samples,
        workloads=workloads,
        policies=policies,
        runs=runs,
        warmup_runs=warmup_runs,
        backend_name=backend_name,
        competition=competition,
        seed=seed,
    )


def write_summary_json(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Written: {path}")


def write_raw_samples(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary.get("raw_samples", []), f, indent=2, ensure_ascii=False)
    print(f"  Written: {path}")


def write_summary_csv(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for workload_name, policies in summary.get("results", {}).items():
        for policy, metrics in policies.items():
            row: dict[str, Any] = {
                "workload": workload_name,
                "policy": policy,
                "runs": summary["metadata"]["runs_per_policy"],
            }
            row.update(metrics)
            rows.append(row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written: {path}")


def write_report(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = summary.get("metadata", {})
    results = summary.get("results", {})
    recommendations = summary.get("recommendations", {})
    policies = meta.get("policies", [])
    workloads = meta.get("workloads", [])

    lines: list[str] = []
    lines.append("# Scheduling Policy Ablation Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Backend**: {meta.get('backend', 'unknown')}")
    lines.append(f"- **Runs per policy**: {meta.get('runs_per_policy', 0)}")
    lines.append(f"- **Workloads**: {', '.join(workloads)}")
    lines.append(f"- **Policies**: {', '.join(policies)}")
    lines.append(f"- **Timestamp**: {meta.get('timestamp', '')}")
    lines.append("")

    # Results by workload
    lines.append("## Results by Workload")
    lines.append("")
    for workload_name in workloads:
        if workload_name not in results:
            continue
        lines.append(f"### {workload_name}")
        lines.append("")

        # Table header
        header = "| Policy | Completion Rate | Latency (s) | Throughput (task/s) | Cache Hit | Fairness | Max Wait (ms) | Unique High Risk | Max Urgency |"
        sep = "|--------|----------------|-------------|--------------------| ---------|----------| --------------| ----------------| -----------|"
        lines.append(header)
        lines.append(sep)

        for policy in policies:
            if policy not in results[workload_name]:
                continue
            m = results[workload_name][policy]
            cr = m.get("mean_tasks_completed_rate", 0.0)
            lat = m.get("mean_latency_per_completed_task_secs", 0.0)
            tput = m.get("mean_throughput_tasks_per_sec", 0.0)
            cache = m.get("mean_cache_hit_ratio", 0.0)
            fair = m.get("mean_fairness_score", 1.0)
            mw = m.get("mean_max_wait_ms", 0.0)
            hr = m.get("mean_unique_high_risk_task_count", 0.0)
            urg = m.get("mean_max_urgency_score", 0.0)
            lines.append(
                f"| {policy} | {cr:.2f} | {lat:.4f} | {tput:.2f} | {cache:.3f} | {fair:.3f} | {mw:.1f} | {hr:.0f} | {urg:.3f} |"
            )

        lines.append("")

    # Policy comparison
    lines.append("## Policy Comparison")
    lines.append("")
    lines.append("| Workload | Recommended Policy | Rationale |")
    lines.append("|----------|-------------------|-----------|")
    for workload_name in workloads:
        rec = recommendations.get(workload_name, "adaptive")
        # Build rationale
        if workload_name in results and rec in results[workload_name]:
            m = results[workload_name][rec]
            cr = m.get("mean_tasks_completed_rate", 0.0)
            lat = m.get("mean_latency_per_completed_task_secs", 0.0)
            rationale = f"completion_rate={cr:.2f}, latency={lat:.4f}s"
        else:
            rationale = "default"
        lines.append(f"| {workload_name} | {rec} | {rationale} |")
    lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    for workload_name in workloads:
        rec = recommendations.get(workload_name, "adaptive")
        lines.append(f"- **{workload_name}**: `{rec}`")
    lines.append("")
    lines.append("### Recommendation Method")
    lines.append("")
    lines.append("Policy ranking uses multi-criteria tie-break on **mean values across all runs**:")
    lines.append("")
    lines.append("```text")
    lines.append("1. tasks_completed_rate  (higher is better)")
    lines.append("2. latency_per_completed_task_secs  (lower is better, negated)")
    lines.append("3. fairness_score  (higher is better)")
    lines.append("4. cache_hit_ratio  (higher is better)")
    lines.append("```")
    lines.append("")
    lines.append(f"With {meta.get('runs_per_policy', 1)} runs per policy, recommendations are based on")
    lines.append("mean values across all runs, not single-run snapshots. When two policies tie on")
    lines.append("completion rate and latency, fairness_score and cache_hit_ratio serve as tie-breakers.")
    lines.append("")
    lines.append("> Note: Recommendations are based on ablation metrics, not production-grade auto-tuning.")
    lines.append("> In the mock backend, cache differences are simulated and should be interpreted as")
    lines.append("> Runtime-side evidence rather than real KV-cache kernel evidence.")
    lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")
    lines.append("### 1. Which workload benefits most from cache_affinity?")
    lines.append("")
    # Find workload where cache_affinity has highest cache_hit_ratio
    best_cache_wl = None
    best_cache_val = 0.0
    for wl in workloads:
        if wl in results and "cache_affinity" in results[wl]:
            val = results[wl]["cache_affinity"].get("mean_cache_hit_ratio", 0.0)
            if val > best_cache_val:
                best_cache_val = val
                best_cache_wl = wl
    if best_cache_wl:
        lines.append(f"`{best_cache_wl}` shows the highest cache hit ratio ({best_cache_val:.3f}) under cache_affinity,")
        lines.append("indicating prefix-heavy workloads benefit from cache-aware scheduling.")
    else:
        lines.append("No clear cache_affinity advantage observed in mock backend.")
    lines.append("")

    lines.append("### 2. Which workload has lowest latency under latency_first?")
    lines.append("")
    best_lat_wl = None
    best_lat_val = float("inf")
    for wl in workloads:
        if wl in results and "latency_first" in results[wl]:
            val = results[wl]["latency_first"].get("mean_latency_per_completed_task_secs", float("inf"))
            if val < best_lat_val:
                best_lat_val = val
                best_lat_wl = wl
    if best_lat_wl:
        lines.append(f"`{best_lat_wl}` achieves the lowest per-task latency ({best_lat_val:.4f}s) under latency_first.")
    else:
        lines.append("No clear latency_first advantage observed.")
    lines.append("")

    lines.append("### 3. Does fairness_balanced reduce max_wait_ms?")
    lines.append("")
    # Compare fairness_balanced max_wait vs others
    for wl in workloads:
        if wl not in results:
            continue
        fb_mw = results[wl].get("fairness_balanced", {}).get("mean_max_wait_ms", 0.0)
        other_mws = [
            results[wl][p].get("mean_max_wait_ms", 0.0)
            for p in policies if p != "fairness_balanced" and p in results[wl]
        ]
        avg_other = _mean(other_mws) if other_mws else 0.0
        if fb_mw < avg_other and avg_other > 0:
            lines.append(f"- `{wl}`: fairness_balanced max_wait={fb_mw:.1f}ms vs avg={avg_other:.1f}ms — reduced.")
        else:
            lines.append(f"- `{wl}`: fairness_balanced max_wait={fb_mw:.1f}ms vs avg={avg_other:.1f}ms — no clear reduction.")
    lines.append("")

    lines.append("### 4. Does risk_aware protect critical/deadline tasks?")
    lines.append("")
    lines.append("risk_aware is evaluated by its ability to prioritize high-risk and deadline-sensitive tasks,")
    lines.append("not by safety reject counts. A low reject count means tasks are admitted (good), while")
    lines.append("high urgency scores indicate the policy correctly identifies deadline pressure.")
    lines.append("")
    for wl in workloads:
        if wl not in results:
            continue
        ra = results[wl].get("risk_aware", {})
        ra_hr = ra.get("mean_unique_high_risk_task_count", 0.0)
        ra_urg = ra.get("mean_max_urgency_score", 0.0)
        ra_crit = ra.get("mean_unique_critical_sla_task_count", 0.0)
        ra_deadline = ra.get("mean_unique_deadline_task_count", 0.0)
        ra_cr = ra.get("mean_tasks_completed_rate", 0.0)
        # Compare completion rate: risk_aware should still complete tasks
        other_cr = _mean([
            results[wl][p].get("mean_tasks_completed_rate", 0.0)
            for p in policies if p != "risk_aware" and p in results[wl]
        ])
        lines.append(
            f"- `{wl}`: unique_high_risk_tasks={ra_hr:.0f}, max_urgency={ra_urg:.3f}, "
            f"unique_critical_sla={ra_crit:.0f}, unique_deadline_tasks={ra_deadline:.0f}, "
            f"completion_rate={ra_cr:.2f} (avg_others={other_cr:.2f})"
        )
    lines.append("")
    lines.append("> Note: In mock backend, urgency and SLA signals are derived from task metadata")
    lines.append("> (sla_class, deadline_ms) rather than real execution pressure. The risk_aware policy")
    lines.append("> demonstrates correct ordering logic, not production-grade risk prediction.")
    lines.append(">")
    lines.append("> All task counts (unique_high_risk_tasks, unique_critical_sla, unique_deadline_tasks)")
    lines.append("> are **unique task IDs** deduplicated across all scheduling rounds, not cumulative")
    lines.append("> observation counts.")
    lines.append("")

    lines.append("### 5. Is adaptive close to the best strategy per workload?")
    lines.append("")
    for wl in workloads:
        if wl not in results:
            continue
        adaptive_cr = results[wl].get("adaptive", {}).get("mean_tasks_completed_rate", 0.0)
        best_cr = max(
            results[wl][p].get("mean_tasks_completed_rate", 0.0)
            for p in policies if p in results[wl]
        )
        diff = best_cr - adaptive_cr
        if diff <= 0.01:
            lines.append(f"- `{wl}`: adaptive={adaptive_cr:.2f}, best={best_cr:.2f} — adaptive is near-optimal.")
        else:
            lines.append(f"- `{wl}`: adaptive={adaptive_cr:.2f}, best={best_cr:.2f} — gap of {diff:.2f}.")
    lines.append("")

    lines.append("### 6. Is adaptive_tuned more stable than plain adaptive?")
    lines.append("")
    lines.append("adaptive_tuned uses sliding-window rule-based bias on top of the base adaptive policy.")
    lines.append("It provides explainable scheduling adjustments without black-box ML.")
    lines.append("")
    for wl in workloads:
        if wl not in results:
            continue
        adaptive_data = results[wl].get("adaptive", {})
        tuned_data = results[wl].get("adaptive_tuned", {})
        if not tuned_data:
            lines.append(f"- `{wl}`: adaptive_tuned data not available.")
            continue
        at_cr = adaptive_data.get("mean_tasks_completed_rate", 0.0)
        at_lat = adaptive_data.get("mean_latency_per_completed_task_secs", 0.0)
        at_fair = adaptive_data.get("mean_fairness_score", 0.0)
        at_switches = adaptive_data.get("mean_policy_switch_count", 0.0)
        tuned_cr = tuned_data.get("mean_tasks_completed_rate", 0.0)
        tuned_lat = tuned_data.get("mean_latency_per_completed_task_secs", 0.0)
        tuned_fair = tuned_data.get("mean_fairness_score", 0.0)
        tuned_switches = tuned_data.get("mean_policy_switch_count", 0.0)
        lines.append(
            f"- `{wl}`: adaptive(cr={at_cr:.2f},lat={at_lat:.4f},fair={at_fair:.3f},switches={at_switches:.1f})"
            f" vs adaptive_tuned(cr={tuned_cr:.2f},lat={tuned_lat:.4f},fair={tuned_fair:.3f},switches={tuned_switches:.1f})"
        )
    lines.append("")
    lines.append("> Note: If mock workload differences are small, adaptive_tuned primarily provides")
    lines.append("> explainable scheduling bias evidence. Performance improvements require longer")
    lines.append("> run windows or real pressure workloads to validate.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scheduling policy ablation experiment")
    parser.add_argument(
        "--workloads",
        default="ai_inference,compile,compression,network",
        help="Comma-separated workload names",
    )
    parser.add_argument(
        "--policies",
        default=",".join(EXPERIMENT_POLICIES),
        help="Comma-separated policy names",
    )
    parser.add_argument("--runs", type=int, default=3, help="Runs per policy/workload")
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--competition", action="store_true")
    parser.add_argument("--backend", default="mock", help="LLM backend name")
    parser.add_argument(
        "--out",
        default="runs/ablation_scheduling",
        help="Output directory",
    )
    args = parser.parse_args()

    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    out_dir = Path(args.out)

    print(f"Scheduling Policy Ablation Experiment")
    print(f"  Workloads: {workloads}")
    print(f"  Policies:  {policies}")
    print(f"  Runs:      {args.runs}")
    print(f"  Backend:   {args.backend}")
    print(f"  Output:    {out_dir}")
    print()

    summary = asyncio.run(run_ablation(
        workloads=workloads,
        policies=policies,
        runs=args.runs,
        backend_name=args.backend,
        out_dir=out_dir,
        warmup_runs=args.warmup_runs,
        competition=args.competition,
        seed=args.seed,
    ))

    print()
    print("Writing outputs...")
    write_summary_json(summary, out_dir / "summary.json")
    write_raw_samples(summary, out_dir / "raw_samples.json")
    write_summary_csv(summary, out_dir / "summary.csv")
    write_report(summary, out_dir / "report.md")

    print()
    print("Done.")
    print(f"  Recommendations: {summary.get('recommendations', {})}")


if __name__ == "__main__":
    main()
