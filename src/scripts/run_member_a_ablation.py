#!/usr/bin/env python3
"""Run lightweight open/closed-loop ablations for Member A Runtime."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentos.core.resources import ResourceQuota
from agentos.core.task import Task, TaskDAG
from agentos.benchmarks.experiment_stats import describe
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import AgentHandler, Runtime
from scripts.demo_member_a_feedback_loop import DemoCgroupV2Manager, run_demo


_MODES = (
    "dag_only",
    "observation_only",
    "feedback_control",
    "feedback_bpf_publish",
)


class _NoopHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        return f"{ctx.task.task_id}-ok"


def _dag_only_dag() -> TaskDAG:
    dag = TaskDAG()
    dag.add_task(Task(task_id="a", name="a", agent_role="noop"))
    dag.add_task(Task(task_id="b", name="b", agent_role="noop", dependencies=["a"]))
    return dag


def _observation_dag() -> TaskDAG:
    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="observe",
            name="observe",
            agent_role="worker",
            resource_requirements=ResourceQuota(
                cpu_max_percent=50.0,
                cpu_shares=512,
                memory_max_mb=128,
                pids_max=16,
            ),
            metadata={
                "execution": "subprocess",
                "worker_type": "sleep",
                "worker_payload": {"seconds": 0.03, "result": "observe-ok"},
            },
        )
    )
    return dag


async def _run_dag_only() -> dict[str, Any]:
    runtime = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"noop": _NoopHandler()},
        max_batch_size=1,
    )
    started = time.monotonic()
    results = await runtime.run(_dag_only_dag())
    return _metrics(runtime.events, results, started)


async def _run_observation_only(output_dir: Path) -> dict[str, Any]:
    runtime = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={},
        execution_mode="subprocess",
        cgroup_manager=DemoCgroupV2Manager(output_dir / "cgroup"),
        worker_sample_interval_seconds=0.01,
    )
    started = time.monotonic()
    results = await runtime.run(_observation_dag())
    return _metrics(runtime.events, results, started)


def _run_feedback(output_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    summary = run_demo(output_dir)
    elapsed = time.monotonic() - started
    metrics = _metrics_from_dicts(summary["events"], summary["results"], elapsed)
    metrics["accepted"] = summary["accepted"]
    return metrics


def _metrics(events: list[Any], results: dict[str, Any], started: float) -> dict[str, Any]:
    return _metrics_from_dicts(
        [event.to_dict() for event in events],
        results,
        time.monotonic() - started,
    )


def _metrics_from_dicts(
    events: list[dict[str, Any]],
    results: dict[str, Any],
    elapsed_secs: float,
) -> dict[str, Any]:
    return {
        "elapsed_secs": elapsed_secs,
        "completed_tasks": len(results),
        "event_count": len(events),
        "usage_samples": _count(events, "resource.usage_sampled"),
        "pressure_samples": _count(events, "resource.pressure_sampled"),
        "feedback_applied": _count(events, "resource.cgroup_feedback_applied"),
        "affected_decisions": sum(
            1
            for event in events
            if event.get("event_type") == "scheduler.decision"
            and event.get("data", {}).get("pressure", {}).get("pressure_source")
            == "system+cgroup"
        ),
        "kernel_decision_published": _count(events, "kernel.decision_published"),
        "cgroup_created": _count(events, "resource.cgroup_created"),
        "cgroup_removed": _count(events, "resource.cgroup_removed"),
        "cleanup_failed": _count(events, "resource.cgroup_cleanup_failed"),
    }


def _count(events: list[dict[str, Any]], event_type: str) -> int:
    return sum(1 for event in events if event.get("event_type") == event_type)


def _aggregate(samples: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = [
        key
        for key, value in samples[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    mean: dict[str, float] = {}
    stdev: dict[str, float] = {}
    statistics: dict[str, dict[str, float | int]] = {}
    for key in keys:
        values = [float(sample[key]) for sample in samples]
        stat = describe(values)
        statistics[key] = stat
        mean[key] = float(stat["mean"])
        stdev[key] = float(stat["stddev"])
    return {"mean": mean, "stdev": stdev, "statistics": statistics}


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# Member A Open/Closed Loop Ablation",
        "",
        f"- Iterations: {result['iterations']}",
        "",
        "## Modes",
        "",
        "| Mode | Usage Samples | Feedback Applied | Affected Decisions | Kernel Publishes | Elapsed Mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, data in result["modes"].items():
        mean = data["mean"]
        elapsed = data.get("statistics", {}).get("elapsed_secs", {})
        lines.append(
            "| "
            f"`{mode}` | "
            f"{mean.get('usage_samples', 0):.2f} | "
            f"{mean.get('feedback_applied', 0):.2f} | "
            f"{mean.get('affected_decisions', 0):.2f} | "
            f"{mean.get('kernel_decision_published', 0):.2f} | "
            f"{elapsed.get('mean', mean.get('elapsed_secs', 0)):.4f}s |"
        )
    lines.extend(["", "## Comparisons", ""])
    for key, value in result["comparisons"].items():
        lines.append(f"- `{key}`: {value}")
    return "\n".join(lines) + "\n"


def _write_raw_samples(path: Path, samples: dict[str, list[dict[str, Any]]]) -> None:
    rows = [
        {"mode": mode, "run_index": index, **sample}
        for mode, mode_samples in samples.items()
        for index, sample in enumerate(mode_samples)
    ]
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_summary_csv(path: Path, result: dict[str, Any]) -> None:
    fieldnames = [
        "mode",
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for mode, data in result["modes"].items():
            for metric, stat in data["statistics"].items():
                writer.writerow({"mode": mode, "metric": metric, **stat})


def run_ablation(
    output_dir: str | Path = "runs/member_a_ablation",
    *,
    iterations: int = 3,
    competition: bool = False,
) -> dict[str, Any]:
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if competition and iterations < 20:
        raise ValueError("competition mode requires at least 20 iterations")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples: dict[str, list[dict[str, Any]]] = {mode: [] for mode in _MODES}
    for index in range(iterations):
        samples["dag_only"].append(asyncio.run(_run_dag_only()))
        samples["observation_only"].append(
            asyncio.run(_run_observation_only(out_dir / f"observation-{index}"))
        )
        samples["feedback_control"].append(
            _run_feedback(out_dir / f"feedback-control-{index}")
        )
        samples["feedback_bpf_publish"].append(
            _run_feedback(out_dir / f"feedback-bpf-publish-{index}")
        )

    modes = {
        mode: {
            "samples": mode_samples,
            **_aggregate(mode_samples),
        }
        for mode, mode_samples in samples.items()
    }
    result = {
        "iterations": iterations,
        "competition": competition,
        "modes": modes,
        "comparisons": {
            "observation_adds_usage_samples": (
                modes["observation_only"]["mean"]["usage_samples"]
                > modes["dag_only"]["mean"]["usage_samples"]
            ),
            "feedback_adds_cgroup_decisions": (
                modes["feedback_control"]["mean"]["affected_decisions"] > 0
            ),
            "bpf_publish_records_kernel_events": (
                modes["feedback_bpf_publish"]["mean"]["kernel_decision_published"] > 0
            ),
        },
    }

    (out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_raw_samples(out_dir / "raw_samples.json", samples)
    _write_summary_csv(out_dir / "summary.csv", result)
    (out_dir / "report.md").write_text(_report(result), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/member_a_ablation")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--competition", action="store_true")
    args = parser.parse_args(argv)

    result = run_ablation(
        args.output_dir,
        iterations=args.iterations,
        competition=args.competition,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
