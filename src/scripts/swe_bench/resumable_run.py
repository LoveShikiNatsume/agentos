#!/usr/bin/env python3
"""Task 10: resumable/checkpointing driver for the honest 4-framework SWE-bench run.

Reuses scripts/swe_bench/honest_bench.py WITHOUT modifying it (import & reuse,
same convention as real_workload_matrix.py). Adds what the original run lacked and
what made it die at task 13 (a transient kill on this shared box, same failure the
original national batch hit at task 16):
  - per-TASK checkpoint to per_task.json (atomic write) -> crash-resumable
  - per-(framework,arm) retry with backoff on transient (network) errors
  - flush=True live progress
  - gc.collect() between tasks to keep RSS down (crewai/chromadb can spike)

Run:
  set -a; source .env; set +a
  .venv/bin/python task10_resumable_run.py --tasks runs/swe_bench/tasks_verified_30.json \
      --out-dir runs/national_experiments/20260709_swe_bench_verified_deepseek_30 \
      --frameworks langgraph,autogen,openai_agents,crewai
Re-running after a crash resumes from the checkpoint (skips finished instance_ids).
"""
from __future__ import annotations
import argparse, asyncio, gc, json, sys, time
from pathlib import Path
from typing import Any

ROOT = str(Path(__file__).resolve().parent)
# ensure repo root on path (script lives in scratchpad; repo root is CWD)
if "/OS2026/AgentOS-Runtime" not in sys.path[0:1]:
    sys.path.insert(0, str(Path.cwd()))

from scripts.swe_bench import swe_workload as W
from scripts.swe_bench.honest_bench import FRAMEWORKS, make_backend, aggregate, print_summary

ARMS = [("native", "native", False), ("runtime", "runtime", False),
        ("native_fault", "native", True), ("runtime_fault", "runtime", True)]


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


async def run_arm_retry(runner, backend, task, mode, fault, retries=5):
    for attempt in range(retries):
        try:
            return await runner(backend, task, mode, fault)
        except Exception as e:  # transient network/API errors -> retry whole arm
            if attempt + 1 < retries:
                wait = 3 * (attempt + 1)
                print(f"      [retry] {task.instance_id} {mode} fault={fault}: "
                      f"{type(e).__name__}: {str(e)[:80]} -> sleep {wait}s", flush=True)
                await asyncio.sleep(wait)
                continue
            raise


async def main(tasks_path: Path, out_dir: Path, frameworks: list[str], limit: int | None) -> None:
    data = json.loads(tasks_path.read_text())
    rows = data["tasks"][:limit] if limit else data["tasks"]
    tasks = [W.parse_task(t) for t in rows]
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "per_task.json"

    per_task: list[dict[str, Any]] = json.loads(ckpt.read_text()) if ckpt.exists() else []
    done_ids = {r["instance_id"] for r in per_task}
    if done_ids:
        print(f"[resume] {len(done_ids)} tasks already checkpointed, skipping them", flush=True)

    backend = make_backend("deepseek")
    t_start = time.monotonic()
    for i, task in enumerate(tasks):
        if task.instance_id in done_ids:
            continue
        print(f"[{i + 1}/{len(tasks)}] {task.instance_id} ({task.repo}, ~{task.est_context_tokens} tok)", flush=True)
        rec: dict[str, Any] = {"instance_id": task.instance_id, "repo": task.repo,
                               "est_context_tokens": task.est_context_tokens}
        for fw in frameworks:
            rec[fw] = {}
            for arm, mode, fault in ARMS:
                m = await run_arm_retry(FRAMEWORKS[fw], backend, task, mode, fault)
                rec[fw][arm] = m
                print(f"    {fw:10s} {arm:14s} calls={m['llm_calls']} ptok={m['prompt_tokens']:>7} "
                      f"miss={m['cache_miss_tokens']:>7} hit={m['cache_hit_ratio']:.0%} "
                      f"cost=${m['est_cost_usd']:.4f} done={m['tasks_completed']}/{m['tasks_total']} "
                      f"recov={m['recovered_faults']}", flush=True)
        per_task.append(rec)
        _atomic_write(ckpt, json.dumps(per_task, indent=2, ensure_ascii=False))
        gc.collect()

    summary = aggregate(per_task, frameworks)
    _atomic_write(out_dir / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[saved] {out_dir}/summary.json  ({time.monotonic() - t_start:.0f}s)", flush=True)
    print_summary(summary, frameworks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frameworks", default="langgraph,autogen,openai_agents,crewai")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    fws = [f.strip() for f in args.frameworks.split(",") if f.strip()]
    asyncio.run(main(Path(args.tasks), Path(args.out_dir), fws, args.limit))
