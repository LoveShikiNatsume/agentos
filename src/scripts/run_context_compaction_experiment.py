"""Compare context compaction policies on the real software-eng workflow.

This is intentionally not a synthetic "repeat the same prompt" microbenchmark.
It runs the existing multi-agent software engineering DAG with the cache-sim
backend and compares:

1. no_compaction: all shared context is kept
2. lru_compaction: existing coldest-first segment eviction
3. prefix_aware_compaction: keep segments with higher expected prefix-cache value
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentos.context.eviction import SegmentEvictor
from agentos.context.manager import ContextManager
from agentos.context.prefix import ContextPriority
from agentos.core.task import TaskDAG, TaskStatus
from agentos.events.log import EventLog
from agentos.llm.factory import create_backend
from agentos.runtime.engine import Runtime
from agentos.scenarios.software_eng.pipeline import (
    PROJECT_CONTEXT,
    REPO_SUMMARY,
    SEGMENT_BUDGET,
    _build_handlers,
    build_software_eng_dag,
)

VARIANTS = (
    "legacy_name_lru_compaction",
    "no_compaction",
    "lru_compaction",
    "prefix_aware_compaction",
)


def _make_context(
    variant: str,
    segment_budget: int,
    merge_threshold: float,
) -> ContextManager:
    if variant == "legacy_name_lru_compaction":
        ctx = ContextManager(
            evictor=SegmentEvictor(similarity_threshold=merge_threshold),
            segment_budget=segment_budget,
            prefix_order="name",
        )
    elif variant == "no_compaction":
        ctx = ContextManager()
    elif variant == "lru_compaction":
        ctx = ContextManager(
            evictor=SegmentEvictor(similarity_threshold=merge_threshold),
            segment_budget=segment_budget,
        )
    elif variant == "prefix_aware_compaction":
        ctx = ContextManager(
            evictor=SegmentEvictor(
                similarity_threshold=merge_threshold,
                prefix_aware=True,
            ),
            segment_budget=segment_budget,
        )
    else:
        raise ValueError(f"unknown variant: {variant}")

    ctx.add_shared_ref(
        "project-context",
        PROJECT_CONTEXT,
        priority=ContextPriority.TASK_BACKGROUND,
    )
    ctx.add_shared_ref(
        "repo-summary",
        REPO_SUMMARY,
        priority=ContextPriority.REPO_SUMMARY,
    )
    return ctx


def _make_runtime(
    *,
    variant: str,
    backend: str,
    log_path: Path,
    segment_budget: int,
    merge_threshold: float,
) -> Runtime:
    return Runtime(
        llm_backend=create_backend(backend),
        handlers=_build_handlers(),
        context_manager=_make_context(variant, segment_budget, merge_threshold),
        event_log=EventLog(log_path),
    )


def _collect(rt: Runtime, dag: TaskDAG, elapsed_secs: float) -> dict[str, Any]:
    llm_events = [e for e in rt.events if e.event_type == "llm.response_received"]
    comp_events = [e for e in rt.events if e.event_type == "context.compressed"]

    prompt_tokens = sum(e.data.get("prompt_tokens", 0) for e in llm_events)
    completion_tokens = sum(e.data.get("tokens_used", 0) for e in llm_events)
    hit_tokens = sum(e.data.get("cache_hit_tokens", 0) for e in llm_events)
    miss_tokens = sum(e.data.get("cache_miss_tokens", 0) for e in llm_events)
    latency_secs = sum(e.data.get("latency_secs", 0.0) for e in llm_events)
    cache_total = hit_tokens + miss_tokens

    total_tasks = len(dag.tasks)
    completed_tasks = sum(
        1 for task in dag.tasks.values() if task.status == TaskStatus.COMPLETED
    )
    failed_tasks = sum(
        1 for task in dag.tasks.values() if task.status == TaskStatus.FAILED
    )
    nonempty_results = sum(1 for task in dag.tasks.values() if task.result)

    context_stats = rt.context.stats()
    compression = context_stats.get("compression", {})
    prefix_diag = context_stats.get("prefix_diagnostics", {})

    return {
        "success": completed_tasks == total_tasks and failed_tasks == 0,
        "elapsed_secs": elapsed_secs,
        "llm_latency_secs": latency_secs,
        "llm_calls": len(llm_events),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cache_hit_tokens": hit_tokens,
        "cache_miss_tokens": miss_tokens,
        "cache_hit_ratio": hit_tokens / cache_total if cache_total else 0.0,
        "prefill_work_tokens": miss_tokens,
        "task_success_rate": completed_tasks / total_tasks if total_tasks else 0.0,
        "nonempty_result_rate": (
            nonempty_results / total_tasks if total_tasks else 0.0
        ),
        "completed_tasks": completed_tasks,
        "failed_tasks": failed_tasks,
        "total_tasks": total_tasks,
        "context_compressed_events": len(comp_events),
        "compressed_tokens_total": compression.get("tokens_saved_total", 0),
        "active_wss_tokens": context_stats.get("active_wss_tokens", 0),
        "context_reuse_ratio": context_stats.get("context_reuse_ratio", 0.0),
        "prefix_overlap_samples": prefix_diag.get("overlap_samples", 0),
        "avg_shared_prefix_tokens": prefix_diag.get("avg_shared_prefix_tokens", 0.0),
        "avg_current_overlap_ratio": prefix_diag.get("avg_current_overlap_ratio", 0.0),
        "avg_shorter_overlap_ratio": prefix_diag.get("avg_shorter_overlap_ratio", 0.0),
        "last_prefix_tokens": prefix_diag.get("last_prefix_tokens", 0),
        "last_prefix_segment_count": prefix_diag.get("segment_count", 0),
        "dropped_segments": sorted(
            {
                name
                for event in comp_events
                for name in event.data.get("dropped_segments", [])
            }
        ),
        "policies_seen": sorted(
            {
                event.data.get("policy", "unknown")
                for event in comp_events
            }
        ),
    }


async def _run_one(
    *,
    variant: str,
    backend: str,
    log_path: Path,
    segment_budget: int,
    merge_threshold: float,
) -> dict[str, Any]:
    if log_path.exists():
        log_path.unlink()
    dag = build_software_eng_dag()
    rt = _make_runtime(
        variant=variant,
        backend=backend,
        log_path=log_path,
        segment_budget=segment_budget,
        merge_threshold=merge_threshold,
    )
    started = time.monotonic()
    try:
        await rt.run(dag)
    except Exception as exc:
        elapsed = time.monotonic() - started
        return {
            "success": False,
            "error": str(exc),
            "elapsed_secs": elapsed,
            "llm_latency_secs": 0.0,
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_hit_tokens": 0,
            "cache_miss_tokens": 0,
            "cache_hit_ratio": 0.0,
            "prefill_work_tokens": 0,
            "task_success_rate": 0.0,
            "nonempty_result_rate": 0.0,
            "completed_tasks": 0,
            "failed_tasks": 1,
            "total_tasks": len(dag.tasks),
            "context_compressed_events": 0,
            "compressed_tokens_total": 0,
            "active_wss_tokens": 0,
            "context_reuse_ratio": 0.0,
            "prefix_overlap_samples": 0,
            "avg_shared_prefix_tokens": 0.0,
            "avg_current_overlap_ratio": 0.0,
            "avg_shorter_overlap_ratio": 0.0,
            "last_prefix_tokens": 0,
            "last_prefix_segment_count": 0,
            "dropped_segments": [],
            "policies_seen": [],
        }
    return _collect(rt, dag, time.monotonic() - started)


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _aggregate(raw: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    numeric_keys = [
        "success",
        "elapsed_secs",
        "llm_latency_secs",
        "llm_calls",
        "prompt_tokens",
        "completion_tokens",
        "cache_hit_tokens",
        "cache_miss_tokens",
        "cache_hit_ratio",
        "prefill_work_tokens",
        "task_success_rate",
        "nonempty_result_rate",
        "context_compressed_events",
        "compressed_tokens_total",
        "active_wss_tokens",
        "context_reuse_ratio",
        "prefix_overlap_samples",
        "avg_shared_prefix_tokens",
        "avg_current_overlap_ratio",
        "avg_shorter_overlap_ratio",
        "last_prefix_tokens",
        "last_prefix_segment_count",
    ]
    summary: dict[str, dict[str, Any]] = {}
    for variant, runs in raw.items():
        entry: dict[str, Any] = {"n_runs": len(runs)}
        for key in numeric_keys:
            values = [float(run.get(key, 0.0)) for run in runs]
            entry[key] = _mean(values)
        entry["dropped_segments"] = sorted(
            {name for run in runs for name in run.get("dropped_segments", [])}
        )
        entry["policies_seen"] = sorted(
            {name for run in runs for name in run.get("policies_seen", [])}
        )
        summary[variant] = entry
    return summary


def _markdown_table(summary: dict[str, dict[str, Any]]) -> str:
    rows = [
        "| Variant | 成功率 | Cache Hit | Prompt Tok | Miss Tok | LLM Lat(s) | Prefix Overlap | 压缩节省Tok | Active WSS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        item = summary[variant]
        rows.append(
            "| {variant} | {success:.0%} | {hit:.1%} | {prompt:.0f} | "
            "{miss:.0f} | {lat:.3f} | {overlap:.1%} | {saved:.0f} | {wss:.0f} |".format(
                variant=variant,
                success=item["task_success_rate"],
                hit=item["cache_hit_ratio"],
                prompt=item["prompt_tokens"],
                miss=item["cache_miss_tokens"],
                lat=item["llm_latency_secs"],
                overlap=item["avg_current_overlap_ratio"],
                saved=item["compressed_tokens_total"],
                wss=item["active_wss_tokens"],
            )
        )
    return "\n".join(rows) + "\n"


async def run_experiment(
    *,
    backend: str,
    n_runs: int,
    output_dir: Path,
    segment_budget: int,
    merge_threshold: float,
) -> dict[str, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / "runs"
    run_dir.mkdir(exist_ok=True)

    raw: dict[str, list[dict[str, Any]]] = {variant: [] for variant in VARIANTS}
    for run_idx in range(n_runs):
        print(f"[run {run_idx + 1}/{n_runs}]")
        for variant in VARIANTS:
            print(f"  {variant:<24}", end="", flush=True)
            metrics = await _run_one(
                variant=variant,
                backend=backend,
                log_path=run_dir / f"{variant}_{run_idx}.jsonl",
                segment_budget=segment_budget,
                merge_threshold=merge_threshold,
            )
            raw[variant].append(metrics)
            status = "ok" if metrics["success"] else "FAIL"
            print(
                f" {status} hit={metrics['cache_hit_ratio']:.1%} "
                f"miss={metrics['cache_miss_tokens']} "
                f"prompt={metrics['prompt_tokens']}"
            )

    summary = _aggregate(raw)
    (output_dir / "raw.json").write_text(
        json.dumps(raw, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    table = _markdown_table(summary)
    (output_dir / "summary.md").write_text(table, encoding="utf-8")
    print(f"\n[saved] {output_dir / 'summary.json'}")
    print(f"[saved] {output_dir / 'summary.md'}\n")
    print(table)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(prog="run-context-compaction")
    parser.add_argument(
        "--backend",
        default="cache-sim",
        choices=["mock", "cache-sim", "vllm-sim", "deepseek-sim", "deepseek", "vllm"],
    )
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--output-dir", default="runs/context_compaction")
    parser.add_argument("--segment-budget", type=int, default=SEGMENT_BUDGET)
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=0.85,
        help=(
            "Near-duplicate merge threshold. Use >1.0 to disable Level-1 merge "
            "when isolating Level-2 eviction policy on cache-sim mock outputs."
        ),
    )
    args = parser.parse_args()

    asyncio.run(
        run_experiment(
            backend=args.backend,
            n_runs=args.n_runs,
            output_dir=Path(args.output_dir),
            segment_budget=args.segment_budget,
            merge_threshold=args.merge_threshold,
        )
    )


if __name__ == "__main__":
    main()
