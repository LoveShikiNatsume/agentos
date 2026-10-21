from __future__ import annotations

import types

import pytest

from agentos.scheduler.policy import SchedulingPolicy
from scripts import ablation_scheduling as ablation


def _sample(value: float, *, success: bool = True) -> dict[str, float | int | bool]:
    return {
        "success": success,
        "total_latency_secs": value,
        "completed_tasks": 1,
        "total_tasks": 1,
        "tasks_completed_rate": 1.0 if success else 0.0,
        "latency_per_completed_task_secs": value,
        "throughput_tasks_per_sec": 10.0 + value,
        "cache_hit_ratio": 0.5,
        "avg_wait_ms": value,
        "max_wait_ms": value * 2,
        "fairness_score": 1.0,
        "starvation_warning_count": 0,
        "deadline_miss_count": 0,
        "cpu_throttled_usec": 0,
        "runtime_cpu_secs": value,
        "runtime_peak_memory_bytes": 1024,
    }


def test_fifo_baseline_is_experiment_only():
    assert "fifo" in ablation.EXPERIMENT_POLICIES
    assert "fifo" not in {item.value for item in SchedulingPolicy}


def test_competition_mode_requires_at_least_twenty_runs(tmp_path):
    with pytest.raises(ValueError, match="at least 20"):
        ablation.validate_competition_settings(
            runs=19,
            warmup_runs=2,
            competition=True,
        )


def test_build_summary_keeps_raw_samples_and_distribution_stats(tmp_path):
    raw = [
        {
            "workload": "compile",
            "policy": "fifo",
            "run_index": index,
            "seed": 20260705 + index,
            **_sample(float(index + 1)),
        }
        for index in range(20)
    ]

    summary = ablation.build_summary_from_raw(
        raw,
        workloads=["compile"],
        policies=["fifo"],
        runs=20,
        warmup_runs=2,
        backend_name="mock",
        competition=True,
        seed=20260705,
    )

    cell = summary["results"]["compile"]["fifo"]
    throughput = cell["throughput_tasks_per_sec"]
    assert summary["raw_samples"] == raw
    assert throughput["n"] == 20
    assert "p50" in throughput
    assert "p95" in throughput
    assert "ci95_low" in throughput
    assert cell["mean_throughput_tasks_per_sec"] == throughput["mean"]


def test_competition_summary_rejects_fewer_than_twenty_successes():
    raw = [
        {
            "workload": "compile",
            "policy": "fifo",
            "run_index": index,
            "seed": 20260705 + index,
            **_sample(float(index + 1), success=index < 19),
        }
        for index in range(20)
    ]

    with pytest.raises(ValueError, match="effective samples"):
        ablation.build_summary_from_raw(
            raw,
            workloads=["compile"],
            policies=["fifo"],
            runs=20,
            warmup_runs=2,
            backend_name="mock",
            competition=True,
            seed=20260705,
        )


@pytest.mark.asyncio
async def test_run_ablation_writes_successful_raw_samples(tmp_path, monkeypatch):
    class _WorkloadResult:
        def to_dict(self):
            return {
                "tasks_completed": 1,
                "tasks_total": 1,
                "total_latency_secs": 0.1,
                "throughput_tasks_per_sec": 10.0,
                "cache_hit_ratio": 0.0,
            }

    async def run(**_kwargs):
        return {"runtime": _WorkloadResult(), "baseline": _WorkloadResult()}

    monkeypatch.setitem(
        ablation.WORKLOAD_MODULES,
        "tiny",
        types.SimpleNamespace(run=run),
    )
    monkeypatch.setattr(
        "agentos.llm.factory.create_backend",
        lambda _name: object(),
    )

    summary = await ablation.run_ablation(
        workloads=["tiny"],
        policies=["fifo"],
        runs=1,
        backend_name="mock",
        out_dir=tmp_path,
    )

    assert summary["raw_samples"][0]["success"] is True
    assert summary["results"]["tiny"]["fifo"]["throughput_tasks_per_sec"]["n"] == 1
