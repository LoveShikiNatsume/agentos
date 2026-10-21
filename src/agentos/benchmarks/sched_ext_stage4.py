"""Statistics and acceptance rules for sched_ext Stage 4 experiments."""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable

from agentos.benchmarks.experiment_stats import describe, student_t_critical_95


STAGE4_MODES = (
    "cfs",
    "scx_simple",
    "agentos_normal",
    "agentos_static_3dsq",
    "agentos_full_dag_psi_fairness",
)


def jain_fairness(values: Iterable[float]) -> float:
    samples = [float(value) for value in values]
    if not samples or any(value < 0 or not math.isfinite(value) for value in samples):
        raise ValueError("fairness samples must be finite and non-negative")
    denominator = len(samples) * sum(value * value for value in samples)
    return 1.0 if denominator == 0 else sum(samples) ** 2 / denominator


def paired_improvement_95(
    baseline: Iterable[float], candidate: Iterable[float]
) -> dict[str, float | int]:
    baseline_values = [float(value) for value in baseline]
    candidate_values = [float(value) for value in candidate]
    if not baseline_values or len(baseline_values) != len(candidate_values):
        raise ValueError("paired samples must have equal non-zero length")
    improvements = [
        (base - current) / base
        for base, current in zip(baseline_values, candidate_values)
        if base > 0
    ]
    if len(improvements) != len(baseline_values):
        raise ValueError("baseline values must be positive")
    mean = statistics.fmean(improvements)
    stddev = statistics.stdev(improvements) if len(improvements) > 1 else 0.0
    margin = (
        student_t_critical_95(len(improvements) - 1)
        * stddev
        / math.sqrt(len(improvements))
        if len(improvements) > 1
        else 0.0
    )
    return {
        "n": len(improvements),
        "mean": mean,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def summarize_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    if not trials:
        raise ValueError("at least one trial is required")
    latency_p95 = [float(trial["latency_p95_us"]) for trial in trials]
    throughput = [float(trial["throughput_iterations"]) for trial in trials]
    batch_gap = [float(trial["batch_max_gap_us"]) for trial in trials]
    startup_wait = [float(trial["worker_startup_max_us"]) for trial in trials]
    summary = {
        "runs": len(trials),
        "latency_p95_us": describe(latency_p95),
        "throughput_iterations": describe(throughput),
        "batch_max_gap_us": describe(batch_gap),
        "worker_startup_max_us": describe(startup_wait),
        "fairness": describe(float(trial["jain_fairness"]) for trial in trials),
        "completed_runs": sum(bool(trial.get("completed")) for trial in trials),
        "workers_attempted": sum(len(trial.get("workers", [])) for trial in trials),
        "workers_completed": sum(
            sum(
                worker.get("completed") is True and worker.get("return_code") == 0
                for worker in trial.get("workers", [])
            )
            for trial in trials
        ),
        "batch_progress_runs": sum(
            all(
                int(worker.get("iterations", 0)) > 0
                for worker in trial.get("workers", [])
                if worker.get("role") == "batch"
            )
            for trial in trials
        ),
        "affinity_violations": sum(int(trial.get("affinity_violations", 0)) for trial in trials),
    }
    observation_values: dict[str, list[float]] = {
        "cpu_utilization": [],
        "system_context_switches": [],
        "cpu_user_us": [],
        "cpu_system_us": [],
        "voluntary_context_switches": [],
        "involuntary_context_switches": [],
    }
    for trial in trials:
        for key in ("cpu_utilization", "system_context_switches"):
            if trial.get(key) is not None:
                observation_values[key].append(float(trial[key]))
        workers = [worker for worker in trial.get("workers", []) if isinstance(worker, dict)]
        for key in (
            "cpu_user_us", "cpu_system_us", "voluntary_context_switches",
            "involuntary_context_switches",
        ):
            values = [worker[key] for worker in workers if worker.get(key) is not None]
            if values:
                observation_values[key].append(sum(float(value) for value in values))
    complete = {
        key: describe(values)
        for key, values in observation_values.items()
        if len(values) == len(trials)
    }
    if complete:
        summary["observability"] = complete
    return summary


def acceptance_passed(summary: dict[str, Any], *, expected_runs: int = 20) -> bool:
    modes = summary.get("modes", {})
    static_gain = summary.get("paired_latency_improvement", {}).get(
        "agentos_static_3dsq", {}
    )
    full_gain = summary.get("paired_latency_improvement", {}).get(
        "agentos_full_dag_psi_fairness", {}
    )
    return bool(
        set(modes) == set(STAGE4_MODES)
        and all(
            mode.get("runs") == expected_runs
            and mode.get("completed_runs") == expected_runs
            and mode.get("workers_attempted") == expected_runs * 5
            and mode.get("workers_completed") == expected_runs * 5
            and mode.get("batch_progress_runs") == expected_runs
            and mode.get("affinity_violations") == 0
            for mode in modes.values()
        )
        and static_gain.get("ci95_low", 0) > 0
        and full_gain.get("ci95_low", 0) > 0
        and summary.get("batch_wait_max_ns", 1 << 63) <= 250_000_000
        and summary.get("batch_process_max_gap_us", 1 << 63) <= 500_000
        and summary.get("worker_startup_max_us", 1 << 63) <= 500_000
        and summary.get("weight_direction_verified") is True
        and summary.get("slice_direction_verified") is True
        and summary.get("queue_stats_verified") is True
        and summary.get("checksum_consistent") is True
        and summary.get("pressure_warning_observed") is True
        and summary.get("warning_generation_verified") is True
        and summary.get("policy_leaks") == 0
        and summary.get("task_state_leaks") == 0
        and summary.get("kernel_log_errors") == []
        and summary.get("systemd_failed_units") == 0
        and summary.get("final_sched_ext_state") == "disabled"
        and summary.get("failure_reason") is None
    )


def smoke_acceptance_passed(summary: dict[str, Any], *, expected_runs: int = 3) -> bool:
    modes = summary.get("modes", {})
    return bool(
        set(modes) == set(STAGE4_MODES)
        and all(
            mode.get("runs") == expected_runs
            and mode.get("completed_runs") == expected_runs
            and mode.get("workers_attempted") == expected_runs * 5
            and mode.get("workers_completed") == expected_runs * 5
            and mode.get("batch_progress_runs") == expected_runs
            and mode.get("affinity_violations") == 0
            for mode in modes.values()
        )
        and summary.get("batch_wait_max_ns", 1 << 63) <= 250_000_000
        and summary.get("batch_process_max_gap_us", 1 << 63) <= 500_000
        and summary.get("worker_startup_max_us", 1 << 63) <= 500_000
        and summary.get("checksum_consistent") is True
        and summary.get("policy_leaks") == 0
        and summary.get("task_state_leaks") == 0
        and summary.get("kernel_log_errors") == []
        and summary.get("systemd_failed_units") == 0
        and summary.get("final_sched_ext_state") == "disabled"
        and summary.get("failure_reason") is None
    )
