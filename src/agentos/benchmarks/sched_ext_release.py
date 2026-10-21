"""Stage 6 sched_ext evidence aggregation and release rendering."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterable

from agentos.benchmarks.experiment_stats import describe, percentile


SCHEMA_VERSION = "sched_ext_release.v1"
MODES = (
    "cfs",
    "scx_simple",
    "agentos_normal",
    "agentos_static_3dsq",
    "agentos_full_dag_psi_fairness",
)
OFFICIAL = {
    "stage1_baseline": "stage1_baseline/result.json",
    "stage1_preflight": "stage1_preflight/preflight.json",
    "stage2": "stage2_skeleton/summary.json",
    "stage3": "stage3_runtime/summary.json",
    "stage4_policy_effects": "stage4_policy_effects_v9/summary.json",
    "stage4_smoke_1": "stage4_ablation_smoke_v9_run1/summary.json",
    "stage4_smoke_2": "stage4_ablation_smoke_v9_run2/summary.json",
    "stage4_formal": "stage4_ablation_formal_v9/summary.json",
    "stage5_formal": (
        "stage5_reliability/"
        "formal-accepted-20x-soak-20260717-143313/summary.json"
    ),
}
STATS = (
    "enqueue_latency", "enqueue_normal", "enqueue_batch",
    "dispatch_latency", "dispatch_normal", "dispatch_batch",
    "runtime_latency_ns", "runtime_normal_ns", "runtime_batch_ns",
    "migration", "preempt_requested", "preempt_suppressed", "fallback",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_official_evidence(root: Path) -> dict[str, dict[str, Any]]:
    root = root.resolve()
    lowered = str(root).lower()
    if "before_" in lowered or "failed_" in lowered:
        raise ValueError("process/failure evidence cannot be an official evidence root")
    result: dict[str, dict[str, Any]] = {}
    for name, relative in OFFICIAL.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = read_json(path)
    return result


def _trial_latency(trial: dict[str, Any]) -> list[float]:
    for worker in trial.get("workers", []):
        if isinstance(worker, dict) and worker.get("role") == "latency":
            values = worker.get("latencies_us")
            if isinstance(values, list) and values:
                return [float(value) for value in values]
    raise ValueError("trial has no latency worker samples")


def _optional_sum(trial: dict[str, Any], key: str) -> float | None:
    values = [
        worker.get(key)
        for worker in trial.get("workers", [])
        if isinstance(worker, dict) and worker.get(key) is not None
    ]
    return sum(float(value) for value in values) if values else None


def aggregate_stage4(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trial_dir = root / "stage4_ablation_formal_v9/trials"
    records: list[dict[str, Any]] = []
    for path in sorted(trial_dir.glob("[0-9][0-9]-*.json")):
        if path.name.endswith("-loader-events.json"):
            continue
        trial = read_json(path)
        mode = trial.get("mode")
        if mode not in MODES:
            continue
        latency = _trial_latency(trial)
        duration_ms = 12_000
        record: dict[str, Any] = {
            "source": str(path.relative_to(root)),
            "source_sha256": sha256_file(path),
            "mode": mode,
            "run_index": int(trial.get("run_index", 0)),
            "completed": trial.get("completed") is True,
            "latency_p50_us": percentile(latency, 0.50),
            "latency_p95_us": percentile(latency, 0.95),
            "latency_p99_us": percentile(latency, 0.99),
            "benchmark_iterations_per_second": (
                float(trial["throughput_iterations"]) / (duration_ms / 1000.0)
            ),
            "jain_fairness": float(trial["jain_fairness"]),
            "batch_max_gap_us": float(trial["batch_max_gap_us"]),
            "worker_startup_max_us": float(trial["worker_startup_max_us"]),
            "affinity_violations": int(trial.get("affinity_violations", 0)),
            "policy_leaks": int(trial.get("policy_leaks", 0)),
            "task_state_leaks": int(trial.get("task_state_leaks", 0)),
        }
        for key in STATS:
            record[f"bpf_{key}"] = float(trial.get("stats", {}).get(key, 0))
        for key in (
            "cpu_user_us", "cpu_system_us", "voluntary_context_switches",
            "involuntary_context_switches",
        ):
            value = _optional_sum(trial, key)
            if value is not None:
                record[key] = value
        for key in ("cpu_utilization", "system_context_switches"):
            if trial.get(key) is not None:
                record[key] = float(trial[key])
        records.append(record)

    metrics = (
        "latency_p50_us", "latency_p95_us", "latency_p99_us",
        "benchmark_iterations_per_second", "jain_fairness",
        "batch_max_gap_us", "worker_startup_max_us",
    ) + tuple(f"bpf_{key}" for key in STATS)
    summary: dict[str, Any] = {"schema": SCHEMA_VERSION, "modes": {}}
    for mode in MODES:
        selected = [record for record in records if record["mode"] == mode]
        if len(selected) != 20:
            raise ValueError(f"{mode} requires 20 formal trials, got {len(selected)}")
        if not all(record["completed"] for record in selected):
            raise ValueError(f"{mode} contains incomplete formal trials")
        mode_summary = {
            "runs": len(selected),
            "metrics": {
                metric: describe(record[metric] for record in selected)
                for metric in metrics
            },
            "optional_metrics": {},
        }
        optional = (
            "cpu_user_us", "cpu_system_us", "voluntary_context_switches",
            "involuntary_context_switches", "cpu_utilization",
            "system_context_switches",
        )
        for metric in optional:
            values = [record[metric] for record in selected if metric in record]
            if values:
                mode_summary["optional_metrics"][metric] = describe(values)
        summary["modes"][mode] = mode_summary
    summary["interpretation"] = {
        "primary_sample": "Stage 4 frozen 20-run formal experiment",
        "throughput_unit": "benchmark iterations/second",
        "throughput_is_business_qps": False,
    }
    return summary, records


def benchmark_csv(summary: dict[str, Any]) -> str:
    output = io.StringIO()
    fields = (
        "mode", "metric", "n", "mean", "stddev", "p50", "p95",
        "ci95_low", "ci95_high", "minimum", "maximum",
    )
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for mode in MODES:
        groups = ("metrics", "optional_metrics")
        for group in groups:
            for metric, values in summary["modes"][mode].get(group, {}).items():
                writer.writerow({"mode": mode, "metric": metric, **values})
    return output.getvalue()


def collect_lifecycle_events(stage5_root: Path, evidence_root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    wanted = {
        "kernel.sched_ext_degraded",
        "kernel.sched_ext_fallback",
        "kernel.sched_ext_recovered",
    }
    for path in sorted(stage5_root.glob("trials/*/*/run-*/events.jsonl")):
        source_digest = sha256_file(path)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            event = json.loads(line)
            event_name = event.get("event_type") or event.get("event")
            if event_name not in wanted:
                continue
            parts = path.relative_to(stage5_root).parts
            result.append({
                "mode": parts[1], "case": parts[2], "run": parts[3],
                "event": event_name, "data": event.get("data", {}),
                "timestamp": event.get("timestamp"),
                "source": str(path.relative_to(evidence_root)),
                "source_sha256": source_digest, "line": line_number,
            })
    return result


def render_report(official: dict[str, dict[str, Any]], benchmark: dict[str, Any]) -> str:
    stage5 = official["stage5_formal"]
    lines = [
        "# AgentOS sched_ext Stage 6 冻结报告", "",
        f"- Schema: `{SCHEMA_VERSION}`",
        "- 接管边界：partial mode 仅接管显式绑定的 Agent Worker。",
        "- 性能主样本：阶段 4 五模式各 20 轮；Stage 6 增量样本不并入主样本。",
        f"- 可靠性：`{stage5.get('passed')}/{stage5.get('trials')}`，"
        f"soak `{stage5.get('soak', {}).get('duration_seconds', 0):.3f}s`，"
        f"task loss `{stage5.get('soak', {}).get('task_loss')}`。", "",
        "## 五模式正式指标", "",
        "| 模式 | latency P50 (us) | latency P95 (us) | latency P99 (us) | benchmark iter/s | Jain fairness |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in MODES:
        metrics = benchmark["modes"][mode]["metrics"]
        lines.append(
            f"| {mode} | {metrics['latency_p50_us']['mean']:.3f} | "
            f"{metrics['latency_p95_us']['mean']:.3f} | "
            f"{metrics['latency_p99_us']['mean']:.3f} | "
            f"{metrics['benchmark_iterations_per_second']['mean']:.3f} | "
            f"{metrics['jain_fairness']['mean']:.6f} |"
        )
    lines += [
        "", "## 能力边界", "",
        "允许表述：AgentOS 基于 sched_ext partial mode 真实接管 Agent Worker，并在异常时回退默认调度。",
        "", "禁止表述：替换 Linux 全部调度器、接管实时调度类、达到生产级。", "",
    ]
    return "\n".join(lines)


def artifact_manifest(root: Path, paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [
        {"path": str(path.relative_to(root)), "sha256": sha256_file(path), "size": path.stat().st_size}
        for path in sorted(paths)
        if path.is_file()
    ]
