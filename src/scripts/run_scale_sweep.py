#!/usr/bin/env python3
"""AgentOS 调度层规模扩展压测（模拟推理后端，不调用真实模型）。"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from agentos.core.task import Task, TaskDAG
from agentos.llm.mock import MockLLMBackend
from agentos.runtime import engine as runtime_engine
from agentos.runtime.engine import AgentHandler, Runtime
from agentos.scheduler.dag import DAGScheduler


SCALES = (8, 16, 32, 64)
LABEL = "调度层压测、模拟推理后端"
PLOT_LABEL = "调度层压测·模拟推理后端"
DEFAULT_OUTPUT_DIR = Path("runs/national_experiments/20260710_scale_sweep")


class MockInferenceHandler(AgentHandler):
    """每个节点执行一次零等待 Mock LLM 调用。"""

    async def run(self, ctx: AgentHandler.Context) -> str:
        response = await ctx.llm_call(
            [{"role": "user", "content": f"scale-sweep task {ctx.task.task_id}"}]
        )
        return response.content


class TimedDAGScheduler(DAGScheduler):
    """原样执行 DAGScheduler，并记录每次决策的壁钟耗时。"""

    decision_latencies_ns: list[int] = []

    async def get_next_batch(self) -> list[Task]:
        started = time.perf_counter_ns()
        try:
            return await super().get_next_batch()
        finally:
            self.decision_latencies_ns.append(time.perf_counter_ns() - started)


def build_scaled_dag(node_count: int) -> TaskDAG:
    """复制 8 节点 software_eng 的分工—汇聚结构。"""
    if node_count not in SCALES:
        raise ValueError(f"node_count must be one of {SCALES}")

    dag = TaskDAG()
    for session in range(node_count // 8):
        prefix = f"s{session:02d}"

        def add(name: str, dependencies: list[str] | None = None) -> None:
            dag.add_task(
                Task(
                    task_id=f"{prefix}-{name}",
                    name=f"session {session} {name}",
                    agent_role="mock_worker",
                    dependencies=[f"{prefix}-{dep}" for dep in dependencies or []],
                    metadata={
                        "scale_sweep": True,
                        "session": session,
                        "stage": name,
                        "backend": "mock",
                    },
                )
            )

        add("architect")
        add("db_design", ["architect"])
        add("api_design", ["architect"])
        add("backend", ["db_design", "api_design"])
        add("test", ["backend"])
        add("security", ["backend"])
        add("docs", ["backend"])
        add("integrate", ["test", "security", "docs"])
    return dag


def percentile(values: Sequence[float], percentile_value: float) -> float:
    """线性插值分位数，与常用 inclusive quantile 口径一致。"""
    if not values:
        raise ValueError("cannot calculate percentile of empty values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def current_rss_mib() -> float:
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def peak_rss_mib() -> float:
    # Linux ru_maxrss 的单位是 KiB；本任务运行环境为 Linux。
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


async def execute_single_run(node_count: int, repeat: int, output_dir: Path) -> dict[str, Any]:
    scale_dir = output_dir / f"scale_{node_count:03d}"
    event_path = scale_dir / f"run_{repeat:02d}_events.jsonl"
    result_path = scale_dir / f"run_{repeat:02d}.json"
    if event_path.exists() or result_path.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {result_path}")

    baseline_rss = current_rss_mib()
    dag = build_scaled_dag(node_count)
    TimedDAGScheduler.decision_latencies_ns = []
    original_scheduler = runtime_engine.DAGScheduler
    runtime_engine.DAGScheduler = TimedDAGScheduler
    try:
        runtime = Runtime(
            llm_backend=MockLLMBackend(latency=0.0),
            handlers={"mock_worker": MockInferenceHandler()},
            event_log=event_path,
            enable_kernel_probe=False,
        )
        started_ns = time.perf_counter_ns()
        results = await runtime.run(dag)
        elapsed_ns = time.perf_counter_ns() - started_ns
    finally:
        runtime_engine.DAGScheduler = original_scheduler

    latencies_us = [value / 1_000.0 for value in TimedDAGScheduler.decision_latencies_ns]
    event_count = count_lines(event_path)
    elapsed_seconds = elapsed_ns / 1_000_000_000.0
    completed = len(results)
    if completed != node_count:
        raise RuntimeError(f"only {completed}/{node_count} tasks completed")
    if event_count != len(runtime.events):
        raise RuntimeError(
            f"event log mismatch: {event_count} lines vs {len(runtime.events)} runtime events"
        )

    payload: dict[str, Any] = {
        "experiment": "AgentOS scale sweep",
        "measurement_scope": LABEL,
        "backend": "mock",
        "real_model_called": False,
        "api_key_required": False,
        "node_count": node_count,
        "concurrent_sessions": node_count // 8,
        "max_parallel_ready_nodes": 3 * (node_count // 8),
        "repeat": repeat,
        "graph_shape": "8-node software_eng isomorphic replicas: 1->2->1->3->1",
        "completed_tasks": completed,
        "elapsed_seconds": elapsed_seconds,
        "scheduler_decisions": {
            "count": len(latencies_us),
            "latencies_us": latencies_us,
            "median_us": statistics.median(latencies_us),
            "p95_us": percentile(latencies_us, 95.0),
            "clock": "time.perf_counter_ns",
            "boundary": "DAGScheduler.get_next_batch only",
        },
        "memory": {
            "baseline_rss_mib": baseline_rss,
            "peak_rss_mib": peak_rss_mib(),
            "source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
            "process_isolation": "one fresh subprocess per repetition",
        },
        "events": {
            "count": event_count,
            "throughput_events_per_second": event_count / elapsed_seconds,
            "log_path": str(event_path),
            "boundary": "non-empty JSONL records / Runtime.run wall time",
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pid": os.getpid(),
        },
        "recorded_at": datetime.now().astimezone().isoformat(),
    }
    write_json(result_path, payload)
    return payload


def spread(values: Sequence[float]) -> dict[str, float]:
    median_value = statistics.median(values)
    minimum = min(values)
    maximum = max(values)
    return {
        "median": median_value,
        "min": minimum,
        "max": maximum,
        "range": maximum - minimum,
        "relative_range_percent": (
            (maximum - minimum) / median_value * 100.0 if median_value else 0.0
        ),
    }


def linear_fit(points: Sequence[tuple[float, float]]) -> dict[str, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    intercept = y_mean - slope * x_mean
    predicted = [slope * x + intercept for x in xs]
    residual = sum((y - estimate) ** 2 for y, estimate in zip(ys, predicted))
    total = sum((y - y_mean) ** 2 for y in ys)
    r_squared = 1.0 - residual / total if total else 1.0
    return {"slope": slope, "intercept": intercept, "r_squared": r_squared}


def aggregate(output_dir: Path, repeats: int) -> dict[str, Any]:
    scales: list[dict[str, Any]] = []
    for node_count in SCALES:
        runs = [
            json.loads(
                (output_dir / f"scale_{node_count:03d}" / f"run_{repeat:02d}.json").read_text(
                    encoding="utf-8"
                )
            )
            for repeat in range(1, repeats + 1)
        ]
        scales.append(
            {
                "node_count": node_count,
                "concurrent_sessions": node_count // 8,
                "repeat_count": len(runs),
                "completed_tasks_each_run": [run["completed_tasks"] for run in runs],
                "scheduler_decision_median_us": spread(
                    [run["scheduler_decisions"]["median_us"] for run in runs]
                ),
                "scheduler_decision_p95_us": spread(
                    [run["scheduler_decisions"]["p95_us"] for run in runs]
                ),
                "peak_rss_mib": spread([run["memory"]["peak_rss_mib"] for run in runs]),
                "event_throughput_events_per_second": spread(
                    [run["events"]["throughput_events_per_second"] for run in runs]
                ),
                "event_count": spread([float(run["events"]["count"]) for run in runs]),
                "elapsed_seconds": spread([run["elapsed_seconds"] for run in runs]),
                "scheduler_decisions_each_run": [
                    run["scheduler_decisions"]["count"] for run in runs
                ],
            }
        )

    metric_names = (
        "scheduler_decision_median_us",
        "scheduler_decision_p95_us",
        "peak_rss_mib",
        "event_throughput_events_per_second",
    )
    fits = {
        metric: linear_fit(
            [(float(item["node_count"]), float(item[metric]["median"])) for item in scales]
        )
        for metric in metric_names
    }
    summary: dict[str, Any] = {
        "experiment": "AgentOS scale sweep",
        "measurement_scope": LABEL,
        "prominent_disclaimer": "本产物仅为调度层压测，使用模拟推理后端；不测模型质量，不调用真实模型。",
        "backend": "mock",
        "real_model_called": False,
        "api_key_required": False,
        "scales": list(SCALES),
        "repeats_per_scale": repeats,
        "aggregation": "每档以各次重复结果的中位数为定稿值；波动为最小值—最大值及极差/中位数。P95 先在单次运行的逐决策样本内计算，再跨重复取中位数。",
        "graph_shape": "8-node software_eng isomorphic replicas: 1->2->1->3->1",
        "metric_boundaries": {
            "scheduler_decision_latency": "DAGScheduler.get_next_batch；perf_counter_ns；单位 us",
            "memory": "独立子进程 ru_maxrss 进程峰值；Linux KiB 换算 MiB",
            "event_throughput": "事件 JSONL 非空记录数 / Runtime.run 壁钟时间；单位 events/s",
        },
        "results": scales,
        "linear_fits": fits,
        "generated_at": datetime.now().astimezone().isoformat(),
        "generator": "scripts/run_scale_sweep.py",
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def render_chart(summary: dict[str, Any], output_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/agentos-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    cjk_font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if cjk_font_path.exists():
        cjk_font_name = font_manager.FontProperties(fname=cjk_font_path).get_name()
        plt.rcParams["font.family"] = cjk_font_name
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    results = summary["results"]
    nodes = [item["node_count"] for item in results]

    figure, axes = plt.subplots(3, 1, figsize=(10.5, 11.5), constrained_layout=True)

    median_latency = [item["scheduler_decision_median_us"]["median"] for item in results]
    p95_latency = [item["scheduler_decision_p95_us"]["median"] for item in results]
    axes[0].plot(nodes, median_latency, "o-", linewidth=2, label="决策延迟中位数")
    axes[0].plot(nodes, p95_latency, "s--", linewidth=2, label="决策延迟 P95")
    axes[0].set_ylabel("延迟（μs）")
    axes[0].legend()

    rss = [item["peak_rss_mib"]["median"] for item in results]
    axes[1].plot(nodes, rss, "o-", color="#d55e00", linewidth=2)
    axes[1].set_ylabel("进程峰值 RSS（MiB）")

    throughput = [
        item["event_throughput_events_per_second"]["median"] for item in results
    ]
    axes[2].plot(nodes, throughput, "o-", color="#009e73", linewidth=2)
    axes[2].set_ylabel("事件写入吞吐（条/s）")
    axes[2].set_xlabel(
        f"任务图节点数\n每档 {summary['repeats_per_scale']} 次独立子进程重复；点为跨重复中位数"
    )

    for axis in axes:
        axis.set_xticks(nodes)
        axis.grid(True, alpha=0.25)
    figure.suptitle(f"AgentOS 调度开销随规模增长\n{PLOT_LABEL}", fontsize=16, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, metadata={"Description": LABEL})
    plt.close(figure)


def run_parent(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    if args.repeats < 3:
        raise ValueError("--repeats must be at least 3")
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite existing summary: {summary_path}")

    for node_count in SCALES:
        for repeat in range(1, args.repeats + 1):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--single-run",
                "--nodes",
                str(node_count),
                "--repeat",
                str(repeat),
                "--output-dir",
                str(output_dir),
            ]
            subprocess.run(command, check=True)
            print(f"completed nodes={node_count} repeat={repeat}", flush=True)

    summary = aggregate(output_dir, args.repeats)
    if args.chart_output is not None:
        render_chart(summary, args.chart_output.resolve())
        print(f"chart: {args.chart_output.resolve()}")
    print(f"summary: {summary_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"AgentOS {LABEL}",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--chart-output", type=Path)
    parser.add_argument("--single-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--nodes", type=int, choices=SCALES, help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=int, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.single_run:
        if args.nodes is None or args.repeat is None:
            raise ValueError("--single-run requires --nodes and --repeat")
        asyncio.run(execute_single_run(args.nodes, args.repeat, args.output_dir.resolve()))
        return 0
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
