from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from agentos.events.types import RuntimeEvent
from agentos.scenarios.sched_ext_business.metrics import (
    compute_trial_metrics,
    paired_improvements,
    summarise_trials,
)
from agentos.scenarios.sched_ext_business.pipeline import BUSINESS_TASK_IDS


def _events() -> list[RuntimeEvent]:
    events = [
        RuntimeEvent(
            event_type="task.completed",
            task_id="issue_triage",
            timestamp=10.0,
            monotonic_ns=10_000_000_000,
        )
    ]
    for index, task_id in enumerate(BUSINESS_TASK_IDS):
        started = 10.01 + index * 0.001
        duration = 0.10 + index * 0.01
        events.extend(
            [
                RuntimeEvent(
                    event_type="worker.started",
                    task_id=task_id,
                    timestamp=started,
                    monotonic_ns=round(started * 1_000_000_000),
                ),
                RuntimeEvent(
                    event_type="worker.finished",
                    task_id=task_id,
                    timestamp=started + duration,
                    monotonic_ns=round((started + duration) * 1_000_000_000),
                    data={"duration_secs": duration},
                ),
            ]
        )
    return events


def _results(prefix: str = "same") -> dict[str, dict[str, Any]]:
    from agentos.scenarios.sched_ext_business.pipeline import WORKER_TYPES_BY_TASK

    results = {}
    for task_id in BUSINESS_TASK_IDS:
        payload = {
            "worker_type": WORKER_TYPES_BY_TASK[task_id],
            "input_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
            "work_units": 1,
            "result": {"marker": prefix},
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode()
        results[task_id] = {
            **payload,
            "checksum": hashlib.sha256(encoded).hexdigest(),
        }
    return results


def test_compute_trial_metrics_recalculates_critical_ready() -> None:
    metrics = compute_trial_metrics(
        _events(),
        _results(),
        deadlines_ms={"patch_validation": 250, "security_gate": 250},
        work_units={task_id: 2 for task_id in BUSINESS_TASK_IDS},
    )

    assert metrics["critical_ready_ms"] == pytest.approx(121.0)
    assert metrics["completed_tasks"] == 8
    assert metrics["deadline_hit_rate"] == 1.0
    assert metrics["task_loss"] == 0
    assert metrics["makespan_ms"] > metrics["critical_ready_ms"]
    assert metrics["tasks_per_second"] > 0
    assert 0 < metrics["jain_fairness"] <= 1
    assert len(metrics["output_checksum"]) == 64


def test_compute_trial_metrics_rejects_missing_or_duplicate_events() -> None:
    missing = [event for event in _events() if event.task_id != "symbol_index"]
    with pytest.raises(ValueError, match="missing worker events"):
        compute_trial_metrics(missing, _results())

    duplicate = _events() + [
        RuntimeEvent(
            event_type="worker.finished",
            task_id="artifact_hash",
            timestamp=20.0,
            monotonic_ns=20_000_000_000,
        )
    ]
    with pytest.raises(ValueError, match="duplicate worker event"):
        compute_trial_metrics(duplicate, _results())


def test_summarise_and_pair_trials_preserve_negative_samples() -> None:
    trials = [
        {"mode": "cfs", "round": 0, "critical_ready_ms": 100.0},
        {"mode": "full", "round": 0, "critical_ready_ms": 80.0},
        {"mode": "cfs", "round": 1, "critical_ready_ms": 100.0},
        {"mode": "full", "round": 1, "critical_ready_ms": 110.0},
    ]

    summary = summarise_trials(trials)
    paired = paired_improvements(trials, baseline="cfs", candidate="full")

    assert summary["cfs"]["critical_ready_ms"]["n"] == 2
    assert summary["cfs"]["critical_ready_ms"]["p99"] == pytest.approx(100.0)
    assert paired["n"] == 2
    assert paired["improvement_percent"] == [20.0, -10.0]
    assert paired["negative_sample_count"] == 1


def test_paired_improvements_rejects_missing_round() -> None:
    with pytest.raises(ValueError, match="paired rounds differ"):
        paired_improvements(
            [
                {"mode": "cfs", "round": 0, "critical_ready_ms": 100.0},
                {"mode": "full", "round": 1, "critical_ready_ms": 80.0},
            ],
            baseline="cfs",
            candidate="full",
        )
