"""Ablation experiment: bare → +DAG → +DAG+Context → +DAG+Ctx+Fault.

Usage:
    .venv/bin/python scripts/run_ablation_experiment.py --backend mock
    .venv/bin/python scripts/run_ablation_experiment.py --backend deepseek --n-runs 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

from agentos.context.eviction import SegmentEvictor
from agentos.context.manager import ContextManager
from agentos.context.prefix import ContextPriority
from agentos.events.log import EventLog
from agentos.fault.injector import FaultInjectionPlan
from agentos.fault.recovery import ContextRetryPolicy
from agentos.llm.factory import create_backend
from agentos.runtime.engine import Runtime
from agentos.scenarios.software_eng.pipeline import (
    SEGMENT_BUDGET,
    PROJECT_CONTEXT,
    REPO_SUMMARY,
    _build_handlers,
    build_software_eng_dag,
)

MODES = ["bare", "dag_only", "dag_context", "full"]


class _RetryingBackend:
    """把真实后端包一层，仅对**瞬时网络错误**（如 httpx 'peer closed connection /
    incomplete chunked read'）做退避重试，避免一次网络抖动就让无恢复机制的 bare 臂崩溃、
    token 记为 0。只重试失败的调用，不改变任何成功调用的 token/延迟测量，故对消融对比中性。
    """

    def __init__(self, inner: Any, retries: int = 5) -> None:
        self._inner = inner
        self._retries = retries

    async def generate(self, request: Any) -> Any:
        for attempt in range(self._retries):
            try:
                return await self._inner.generate(request)
            except Exception as exc:  # noqa: BLE001 - 瞬时网络错误退避重试
                if attempt + 1 < self._retries:
                    await asyncio.sleep(2.0 * (attempt + 1))
                    continue
                raise
        raise RuntimeError("unreachable")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _make_runtime(mode: str, backend: str, model: str, log_path: Path) -> Runtime:
    backend_kwargs = {"model": model} if backend == "deepseek" else {}
    llm = _RetryingBackend(create_backend(backend, **backend_kwargs))
    handlers = _build_handlers()
    log = EventLog(log_path)

    if mode == "bare":
        return Runtime(
            llm_backend=llm,
            handlers=handlers,
            context_manager=ContextManager(),
            event_log=log,
            max_batch_size=1,
        )
    if mode == "dag_only":
        return Runtime(
            llm_backend=llm,
            handlers=handlers,
            context_manager=ContextManager(),
            event_log=log,
        )
    ctx_mgr = ContextManager(evictor=SegmentEvictor(), segment_budget=SEGMENT_BUDGET)
    ctx_mgr.add_shared_ref("project-context", PROJECT_CONTEXT, priority=ContextPriority.TASK_BACKGROUND)
    ctx_mgr.add_shared_ref("repo-summary", REPO_SUMMARY, priority=ContextPriority.REPO_SUMMARY)
    if mode == "dag_context":
        return Runtime(llm_backend=llm, handlers=handlers, context_manager=ctx_mgr, event_log=log)
    # full
    return Runtime(
        llm_backend=llm,
        handlers=handlers,
        context_manager=ctx_mgr,
        event_log=log,
        fault_injection=FaultInjectionPlan({"security": "kill"}),
        recovery_policy=ContextRetryPolicy(max_retries=1),
    )


def _collect(rt: Runtime, elapsed: float) -> dict[str, Any]:
    events = rt.events
    prompt_tokens = sum(
        e.data.get("prompt_tokens", e.data.get("tokens_used", 0))
        for e in events if e.event_type == "llm.response_received"
    )
    hit = sum(e.data.get("cache_hit_tokens", 0) for e in events if e.event_type == "llm.response_received")
    miss = sum(e.data.get("cache_miss_tokens", 0) for e in events if e.event_type == "llm.response_received")
    total_cache = hit + miss
    # max parallel tasks observed via event ordering
    in_progress: set[str] = set()
    max_parallel = 0
    for e in events:
        if e.event_type == "task.started":
            in_progress.add(e.task_id)
            max_parallel = max(max_parallel, len(in_progress))
        elif e.event_type == "task.completed":
            in_progress.discard(e.task_id)
    return {
        "elapsed_secs": elapsed,
        "prompt_tokens": prompt_tokens,
        "cache_hit_tokens": hit,
        "cache_miss_tokens": miss,
        "cache_hit_ratio": hit / total_cache if total_cache > 0 else 0.0,
        "llm_calls": sum(1 for e in events if e.event_type == "llm.response_received"),
        "recovery_count": sum(1 for e in events if e.event_type == "fault.recovery_completed"),
        "fault_count": sum(1 for e in events if e.event_type == "fault.injected"),
        "max_parallel_tasks": max_parallel,
        "context_compressed": sum(1 for e in events if e.event_type == "context.compressed"),
        "success": True,
    }


async def _run_one(mode: str, backend: str, model: str, log_path: Path) -> dict[str, Any]:
    if log_path.exists():
        log_path.unlink()
    dag = build_software_eng_dag()
    rt = _make_runtime(mode, backend, model, log_path)
    t0 = time.monotonic()
    try:
        await rt.run(dag)
    except Exception as exc:
        elapsed = time.monotonic() - t0
        return {"elapsed_secs": elapsed, "success": False, "error": str(exc),
                "prompt_tokens": 0, "cache_hit_tokens": 0, "cache_miss_tokens": 0,
                "cache_hit_ratio": 0.0, "llm_calls": 0, "recovery_count": 0,
                "fault_count": 0, "max_parallel_tasks": 0, "context_compressed": 0}
    return _collect(rt, time.monotonic() - t0)


async def run_ablation(backend: str, model: str, n_runs: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / "runs"
    run_dir.mkdir(exist_ok=True)

    raw: dict[str, list[dict[str, Any]]] = {m: [] for m in MODES}

    for i in range(n_runs):
        print(f"[run {i + 1}/{n_runs}]")
        for mode in MODES:
            print(f"  {mode:<14} ...", end=" ", flush=True)
            m = await _run_one(mode, backend, model, run_dir / f"{mode}_{i}.jsonl")
            raw[mode].append(m)
            status = "ok" if m["success"] else "FAIL"
            print(f"{status}  {m['elapsed_secs']:.3f}s  tokens={m['prompt_tokens']}")

    results_path = output_dir / "ablation_results.json"
    results_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False))
    print(f"\n[saved] {results_path}")

    def avg(runs: list[dict], key: str) -> float:
        vals = [r[key] for r in runs if key in r]
        return statistics.mean(vals) if vals else 0.0

    summary: dict[str, Any] = {"model": model}
    for mode in MODES:
        runs = raw[mode]
        summary[mode] = {
            "n_runs": len(runs),
            "elapsed_secs": avg(runs, "elapsed_secs"),
            "prompt_tokens": avg(runs, "prompt_tokens"),
            "cache_hit_ratio": avg(runs, "cache_hit_ratio"),
            "recovery_count": avg(runs, "recovery_count"),
            "max_parallel_tasks": avg(runs, "max_parallel_tasks"),
            "success_rate": sum(1 for r in runs if r.get("success")) / len(runs),
        }

    summary_path = output_dir / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[saved] {summary_path}")

    rows = []
    for mode in MODES:
        s = summary[mode]
        rec = f"✓ {s['recovery_count']:.1f}x" if s["recovery_count"] > 0 else "✗ 崩溃"
        rows.append(
            f"| {mode:<16} | {s['elapsed_secs']:>12.3f} | {int(s['prompt_tokens']):>13} |"
            f" {s['cache_hit_ratio']:>9.1%} | {rec} |"
        )
    table = "\n".join([
        "| 模式             | 端到端延迟(s) | Prompt Tokens | 缓存命中率 | 故障恢复 |",
        "|------------------|-------------|---------------|-----------|---------|",
        *rows,
    ])
    table_path = output_dir / "ablation_table.md"
    table_path.write_text(table + "\n")
    print(f"[saved] {table_path}\n")
    print(table)


def main() -> None:
    parser = argparse.ArgumentParser(prog="run-ablation")
    parser.add_argument("--backend", default="mock",
                        choices=["mock", "cache-sim", "deepseek-sim", "deepseek", "vllm"])
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--output-dir", default="runs/ablation")
    args = parser.parse_args()
    asyncio.run(run_ablation(args.backend, args.model, args.n_runs, Path(args.output_dir)))


if __name__ == "__main__":
    main()
