"""Frozen P2 formal plan and preregistered performance gate."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from agentos.benchmarks.experiment_stats import percentile
from agentos.scenarios.sched_ext_business.metrics import paired_improvements


FORMAL_MODES = (
    "cfs",
    "agentos-normal",
    "agentos-static-3dsq",
    "agentos-full",
)
WARMUP_ROUNDS = 2
FORMAL_ROUNDS = 20
PRIMARY_BASELINE = "cfs"
PRIMARY_CANDIDATE = "agentos-static-3dsq"
PRIMARY_METRIC = "critical_ready_ms"


def build_formal_plan(sample_kind: str) -> list[dict[str, int | str]]:
    """Return the frozen rotating warmup or formal matrix."""

    rounds = {"warmup": WARMUP_ROUNDS, "formal": FORMAL_ROUNDS}.get(sample_kind)
    if rounds is None:
        raise ValueError(f"unsupported formal sample kind: {sample_kind!r}")
    plan: list[dict[str, int | str]] = []
    for round_id in range(rounds):
        rotated = FORMAL_MODES[round_id % len(FORMAL_MODES) :] + FORMAL_MODES[
            : round_id % len(FORMAL_MODES)
        ]
        for order, mode in enumerate(rotated):
            plan.append(
                {
                    "sample_kind": sample_kind,
                    "round": round_id,
                    "order": order,
                    "mode": mode,
                }
            )
    return plan


def formal_round_seed(base_seed: int, round_id: int, *, sample_kind: str) -> int:
    """Pair all modes within a round and keep warmup seeds outside formal seeds."""

    limits = {"warmup": WARMUP_ROUNDS, "formal": FORMAL_ROUNDS}
    limit = limits.get(sample_kind)
    if limit is None:
        raise ValueError(f"unsupported formal sample kind: {sample_kind!r}")
    if not 0 <= round_id < limit:
        raise ValueError(f"{sample_kind} round_id out of range: {round_id}")
    offset = round_id if sample_kind == "formal" else round_id - WARMUP_ROUNDS
    return int(base_seed) + offset * 100


def _finite_mean(values: list[float], *, name: str) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} samples must be finite and non-empty")
    return sum(values) / len(values)


def evaluate_primary_gate(trials: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate only the preregistered primary claim from all 80 formal trials."""

    raw = list(trials)
    expected = {
        (str(item["mode"]), int(item["round"]))
        for item in build_formal_plan("formal")
    }
    actual = [(str(item.get("mode", "")), int(item.get("round", -1))) for item in raw]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("formal trial plan is incomplete or contains duplicates")

    by_mode: dict[str, list[Mapping[str, Any]]] = {
        mode: [item for item in raw if item.get("mode") == mode]
        for mode in FORMAL_MODES
    }
    for mode, mode_trials in by_mode.items():
        if len(mode_trials) != FORMAL_ROUNDS:
            raise ValueError(f"formal trial plan is incomplete for {mode}")

    baseline_metric = [float(item[PRIMARY_METRIC]) for item in by_mode[PRIMARY_BASELINE]]
    candidate_metric = [float(item[PRIMARY_METRIC]) for item in by_mode[PRIMARY_CANDIDATE]]
    baseline_p95 = percentile(baseline_metric, 0.95)
    candidate_p95 = percentile(candidate_metric, 0.95)
    if baseline_p95 <= 0 or candidate_p95 <= 0:
        raise ValueError("formal critical_ready_ms values must be positive")
    p95_improvement = (baseline_p95 - candidate_p95) / baseline_p95 * 100

    paired = paired_improvements(
        raw,
        baseline=PRIMARY_BASELINE,
        candidate=PRIMARY_CANDIDATE,
        metric=PRIMARY_METRIC,
    )
    baseline_tps = _finite_mean(
        [float(item["tasks_per_second"]) for item in by_mode[PRIMARY_BASELINE]],
        name="baseline tasks_per_second",
    )
    candidate_tps = _finite_mean(
        [float(item["tasks_per_second"]) for item in by_mode[PRIMARY_CANDIDATE]],
        name="candidate tasks_per_second",
    )
    if baseline_tps <= 0 or candidate_tps <= 0:
        raise ValueError("formal tasks_per_second values must be positive")
    tasks_per_second_change = (candidate_tps / baseline_tps - 1) * 100

    checks = {
        "p95_improvement_at_least_25_percent": p95_improvement >= 25.0 - 1e-12,
        "paired_ci95_low_above_zero": float(paired["ci95_low_percent"]) > 0.0,
        "tasks_per_second_regression_at_most_5_percent": (
            tasks_per_second_change >= -5.0 - 1e-12
        ),
    }
    return {
        "passed": all(checks.values()),
        "primary_baseline": PRIMARY_BASELINE,
        "primary_candidate": PRIMARY_CANDIDATE,
        "primary_metric": PRIMARY_METRIC,
        "formal_rounds": FORMAL_ROUNDS,
        "baseline_p95_ms": baseline_p95,
        "candidate_p95_ms": candidate_p95,
        "p95_improvement_percent": p95_improvement,
        "paired_mean_improvement_percent": paired["mean_improvement_percent"],
        "paired_ci95_low_percent": paired["ci95_low_percent"],
        "paired_ci95_high_percent": paired["ci95_high_percent"],
        "paired_negative_sample_count": paired["negative_sample_count"],
        "baseline_tasks_per_second_mean": baseline_tps,
        "candidate_tasks_per_second_mean": candidate_tps,
        "tasks_per_second_change_percent": tasks_per_second_change,
        "checks": checks,
        "secondary_modes_reported": [
            mode
            for mode in FORMAL_MODES
            if mode not in {PRIMARY_BASELINE, PRIMARY_CANDIDATE}
        ],
    }
