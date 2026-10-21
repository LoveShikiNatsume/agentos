"""Recompute P2 business metrics exclusively from raw events and results."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from agentos.benchmarks.experiment_stats import describe, percentile
from agentos.benchmarks.sched_ext_stage4 import jain_fairness
from agentos.events.types import RuntimeEvent
from agentos.scenarios.sched_ext_business.pipeline import (
    BUSINESS_TASK_IDS,
    WORKER_TYPES_BY_TASK,
)


CRITICAL_TASK_IDS = ("patch_validation", "security_gate")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _event_dict(event: RuntimeEvent | Mapping[str, Any]) -> dict[str, Any]:
    return event.to_dict() if isinstance(event, RuntimeEvent) else dict(event)


def _monotonic_seconds(event: Mapping[str, Any]) -> float:
    value = event.get("monotonic_ns")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("event monotonic_ns must be a positive integer")
    return value / 1_000_000_000


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _one_event(
    grouped: Mapping[tuple[str, str], list[dict[str, Any]]],
    event_type: str,
    task_id: str,
) -> dict[str, Any]:
    selected = grouped.get((event_type, task_id), [])
    if not selected:
        raise ValueError(f"missing worker events: {event_type}/{task_id}")
    if len(selected) != 1:
        raise ValueError(f"duplicate worker event: {event_type}/{task_id}")
    return selected[0]


def compute_trial_metrics(
    events: Iterable[RuntimeEvent | Mapping[str, Any]],
    task_results: Mapping[str, Mapping[str, Any]],
    *,
    deadlines_ms: Mapping[str, int] | None = None,
    work_units: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Recalculate one trial without trusting pre-computed summary fields."""

    raw = [_event_dict(event) for event in events]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in raw:
        task_id = event.get("task_id")
        event_type = event.get("event_type")
        timestamp = float(event.get("timestamp", math.nan))
        if not math.isfinite(timestamp):
            raise ValueError("event timestamp must be finite")
        _monotonic_seconds(event)
        if isinstance(task_id, str) and isinstance(event_type, str):
            grouped[(event_type, task_id)].append(event)

    release_events = grouped.get(("task.completed", "issue_triage"), [])
    if len(release_events) != 1:
        raise ValueError("exactly one issue_triage completion event is required")
    release_time = _monotonic_seconds(release_events[0])

    starts: dict[str, float] = {}
    finishes: dict[str, float] = {}
    durations: dict[str, float] = {}
    for task_id in BUSINESS_TASK_IDS:
        started = _one_event(grouped, "worker.started", task_id)
        finished = _one_event(grouped, "worker.finished", task_id)
        start = _monotonic_seconds(started)
        finish = _monotonic_seconds(finished)
        if start < release_time or finish < start:
            raise ValueError(f"invalid event ordering for {task_id}")
        starts[task_id] = start
        finishes[task_id] = finish
        durations[task_id] = finish - start

    missing_results = sorted(set(BUSINESS_TASK_IDS) - set(task_results))
    if missing_results:
        raise ValueError(f"missing task results: {missing_results}")
    checksums: dict[str, str] = {}
    for task_id in BUSINESS_TASK_IDS:
        task_result = task_results[task_id]
        checksum = str(task_result.get("checksum", ""))
        if _SHA256.fullmatch(checksum) is None:
            raise ValueError(f"invalid task checksum: {task_id}")
        checksum_payload = {
            "worker_type": task_result.get("worker_type"),
            "input_sha256": task_result.get("input_sha256"),
            "work_units": task_result.get("work_units"),
            "result": task_result.get("result"),
        }
        if task_result.get("worker_type") != WORKER_TYPES_BY_TASK[task_id]:
            raise ValueError(f"worker type mismatch: {task_id}")
        if checksum != _canonical_sha256(checksum_payload):
            raise ValueError(f"task checksum does not match result payload: {task_id}")
        checksums[task_id] = checksum

    critical_ready = max(finishes[task_id] for task_id in CRITICAL_TASK_IDS)
    final_finish = max(finishes.values())
    makespan = final_finish - release_time
    if makespan <= 0:
        raise ValueError("trial makespan must be positive")

    deadlines = dict(deadlines_ms or {})
    deadline_hits = [
        (finishes[task_id] - release_time) * 1000 <= deadline
        for task_id, deadline in deadlines.items()
        if task_id in finishes
    ]
    configured_units = {
        task_id: int((work_units or {}).get(task_id, 1))
        for task_id in BUSINESS_TASK_IDS
    }
    rates = [configured_units[task_id] / durations[task_id] for task_id in BUSINESS_TASK_IDS]

    return {
        "release_timestamp": release_time,
        "critical_ready_ms": (critical_ready - release_time) * 1000,
        "makespan_ms": makespan * 1000,
        "task_wait_ms": {
            task_id: (starts[task_id] - release_time) * 1000
            for task_id in BUSINESS_TASK_IDS
        },
        "task_duration_ms": {
            task_id: durations[task_id] * 1000 for task_id in BUSINESS_TASK_IDS
        },
        "completed_tasks": len(finishes),
        "task_loss": len(BUSINESS_TASK_IDS) - len(finishes),
        "deadline_hit_rate": (
            sum(deadline_hits) / len(deadline_hits) if deadline_hits else 1.0
        ),
        "tasks_per_second": len(BUSINESS_TASK_IDS) / makespan,
        "jain_fairness": jain_fairness(rates),
        "task_checksums": checksums,
        "output_checksum": _canonical_sha256(checksums),
    }


def summarise_trials(trials: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for trial in trials:
        mode = str(trial.get("mode", ""))
        if not mode:
            raise ValueError("trial mode is required")
        by_mode[mode].append(trial)
    if not by_mode:
        raise ValueError("at least one trial is required")

    result: dict[str, Any] = {}
    metric_names = (
        "critical_ready_ms",
        "makespan_ms",
        "deadline_hit_rate",
        "tasks_per_second",
        "jain_fairness",
    )
    for mode, mode_trials in sorted(by_mode.items()):
        metrics: dict[str, Any] = {"trial_count": len(mode_trials)}
        for name in metric_names:
            values = [float(trial[name]) for trial in mode_trials if name in trial]
            if values:
                stats = describe(values)
                stats["p99"] = percentile(values, 0.99)
                metrics[name] = stats
        result[mode] = metrics
    return result


def paired_improvements(
    trials: Iterable[Mapping[str, Any]],
    *,
    baseline: str,
    candidate: str,
    metric: str = "critical_ready_ms",
) -> dict[str, Any]:
    values: dict[str, dict[int, float]] = {baseline: {}, candidate: {}}
    for trial in trials:
        mode = str(trial.get("mode", ""))
        if mode not in values:
            continue
        round_id = int(trial["round"])
        if round_id in values[mode]:
            raise ValueError(f"duplicate paired trial: {mode}/{round_id}")
        value = float(trial[metric])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("paired metric values must be finite and positive")
        values[mode][round_id] = value
    if set(values[baseline]) != set(values[candidate]):
        raise ValueError("paired rounds differ")
    if not values[baseline]:
        raise ValueError("paired samples are empty")

    rounds = sorted(values[baseline])
    improvement = [
        round(
            (values[baseline][round_id] - values[candidate][round_id])
            / values[baseline][round_id]
            * 100,
            10,
        )
        for round_id in rounds
    ]
    stats = describe(improvement)
    return {
        "baseline": baseline,
        "candidate": candidate,
        "metric": metric,
        "rounds": rounds,
        "n": len(rounds),
        "improvement_percent": improvement,
        "negative_sample_count": sum(value < 0 for value in improvement),
        "mean_improvement_percent": stats["mean"],
        "ci95_low_percent": stats["ci95_low"],
        "ci95_high_percent": stats["ci95_high"],
    }
