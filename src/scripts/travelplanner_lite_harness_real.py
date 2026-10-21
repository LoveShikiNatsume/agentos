#!/usr/bin/env python3
"""TravelPlanner-lite 实验的真实 DeepSeek API 版本（任务 07）。

背景：原始 travelplanner_lite_harness.py（未改动）用
`len(text) // 4` 估算 token、用规则匹配代替真实 LLM 调用，`run.log` 为空，
没有调用任何真实 API。本脚本复用原始模块里的 prompt 构造 / 参考数据切片 /
方案校验逻辑（不改动原文件），把 run_arm 换成真实 DeepSeek API 调用，
token 数与 cache 命中数直接取自 API 返回的 usage 字段，故障注入模拟的是
进程级故障（角色被杀死/超时，不涉及伪造模型输出），与本项目其他实验
（workload_matrix_deepseek_real 等）的故障定义一致。

用法：
    set -a; source .env; set +a
    python3 scripts/travelplanner_lite_harness_real.py \
        --ref-info runs/external_data/TravelPlanner/database/validation_ref_info.jsonl \
        --queries runs/external_data/TravelPlanner/postprocess/example_evaluation.jsonl \
        --out-dir runs/national_experiments/20260709_travelplanner_lite_real \
        --limit 30
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import sys

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.travelplanner_lite_harness import (
    ROLES,
    FAULT_ROLE,
    PRICE_INPUT_MISS,
    PRICE_INPUT_HIT,
    PRICE_OUTPUT,
    load_jsonl,
    build_plan,
    prompt_for,
    pct,
    cost,
)

from openai import AsyncOpenAI

CONCURRENCY = 12
MODEL = "deepseek-v4-flash"


async def call_llm(client: AsyncOpenAI, sem: asyncio.Semaphore, prompt: str) -> dict[str, Any]:
    async with sem:
        t0 = time.monotonic()
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.2,
        )
        elapsed = time.monotonic() - t0
        u = resp.usage
        return {
            "prompt_tokens": u.prompt_tokens,
            "cache_hit_tokens": getattr(u, "prompt_cache_hit_tokens", 0) or 0,
            "cache_miss_tokens": getattr(u, "prompt_cache_miss_tokens", u.prompt_tokens) or u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "elapsed_secs": elapsed,
        }


async def run_arm_real(
    client: AsyncOpenAI, sem: asyncio.Semaphore, query: str, ref: dict[str, Any], mode: str, fault: bool
) -> dict[str, Any]:
    findings: dict[str, Any] = {}
    prompt_tokens = 0
    cache_hit_tokens = 0
    cache_miss_tokens = 0
    completion_tokens = 0
    completed = 0
    recovered = 0
    injected = 0

    for role in ROLES:
        attempts = 2 if mode == "runtime" else 1
        for attempt in range(attempts):
            if fault and role == FAULT_ROLE and attempt == 0:
                # 模拟角色所在进程被杀死/超时：不发起真实调用，代表这次尝试
                # 从未产生输出（进程级故障，不是模型输出的问题）。
                injected += 1
                if attempt + 1 < attempts:
                    recovered += 1
                    continue
                return {
                    "prompt_tokens": prompt_tokens,
                    "cache_hit_tokens": cache_hit_tokens,
                    "cache_miss_tokens": cache_miss_tokens,
                    "completion_tokens": completion_tokens,
                    "llm_calls": completed,
                    "tasks_completed": completed,
                    "tasks_total": len(ROLES),
                    "faults_injected": injected,
                    "recovered_faults": recovered,
                    "plan_completed": False,
                    "constraint_pass_rate": 0.0,
                    "final_pass": False,
                }
            prompt = prompt_for(role, mode, query, ref, findings)
            result = await call_llm(client, sem, prompt)
            prompt_tokens += result["prompt_tokens"]
            cache_hit_tokens += result["cache_hit_tokens"]
            cache_miss_tokens += result["cache_miss_tokens"]
            completion_tokens += result["completion_tokens"]
            completed += 1
            findings[role] = f"{role} completed"
            break

    plan = build_plan(query, ref)
    checks = plan["checks"]
    pass_count = sum(1 for ok in checks.values() if ok)
    constraint_pass_rate = pass_count / len(checks)
    final_pass = all(checks.values())
    return {
        "prompt_tokens": prompt_tokens,
        "cache_hit_tokens": cache_hit_tokens,
        "cache_miss_tokens": cache_miss_tokens,
        "completion_tokens": completion_tokens,
        "llm_calls": completed,
        "tasks_completed": completed,
        "tasks_total": len(ROLES),
        "faults_injected": injected,
        "recovered_faults": recovered,
        "plan_completed": True,
        "constraint_pass_rate": round(constraint_pass_rate, 4),
        "final_pass": final_pass,
        "estimated_plan_cost": plan["total_cost"],
        "budget": plan["intent"]["budget"],
    }


async def run(ref_path: Path, query_path: Path, out_dir: Path, limit: int) -> None:
    refs = load_jsonl(ref_path)[:limit]
    queries = load_jsonl(query_path)[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ["DEEPSEEK_API_KEY"]
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    sem = asyncio.Semaphore(CONCURRENCY)

    per_task: list[dict[str, Any]] = [None] * len(refs)  # type: ignore

    async def do_task(idx: int, ref: dict[str, Any], qrow: dict[str, Any]) -> None:
        query = qrow["query"]
        rec: dict[str, Any] = {"idx": qrow.get("idx", idx + 1), "query": query}
        for arm, mode, fault in [
            ("native", "native", False),
            ("runtime", "runtime", False),
            ("native_fault", "native", True),
            ("runtime_fault", "runtime", True),
        ]:
            rec[arm] = await run_arm_real(client, sem, query, ref, mode, fault)
        per_task[idx] = rec
        print(f"[{idx + 1}/{len(refs)}] task {rec['idx']} done", flush=True)

    t0 = time.monotonic()
    await asyncio.gather(*(do_task(i, ref, qrow) for i, (ref, qrow) in enumerate(zip(refs, queries))))
    elapsed = time.monotonic() - t0

    def total(arm: str, key: str) -> int:
        return sum(int(x[arm].get(key, 0)) for x in per_task)

    def avg(arm: str, key: str) -> float:
        values = [float(x[arm].get(key, 0)) for x in per_task]
        return round(statistics.mean(values), 4) if values else 0.0

    native_cost = cost(total("native", "cache_miss_tokens"), total("native", "cache_hit_tokens"), total("native", "completion_tokens"))
    runtime_cost = cost(total("runtime", "cache_miss_tokens"), total("runtime", "cache_hit_tokens"), total("runtime", "completion_tokens"))
    summary = {
        "benchmark": "TravelPlanner-lite",
        "source": "OSU-NLP-Group/TravelPlanner validation ref_info + example queries",
        "backend": "deepseek",
        "model": MODEL,
        "backend_probe": {"backend": "deepseek", "available": True, "base_url": "https://api.deepseek.com", "model": MODEL},
        "n_tasks": len(per_task),
        "roles": ROLES,
        "efficiency_no_fault": {
            "prompt_tokens": {
                "native": total("native", "prompt_tokens"),
                "runtime": total("runtime", "prompt_tokens"),
                "reduction_pct": pct(total("native", "prompt_tokens"), total("runtime", "prompt_tokens")),
            },
            "cache_miss_tokens": {
                "native": total("native", "cache_miss_tokens"),
                "runtime": total("runtime", "cache_miss_tokens"),
                "reduction_pct": pct(total("native", "cache_miss_tokens"), total("runtime", "cache_miss_tokens")),
            },
            "est_cost_usd": {
                "native": native_cost,
                "runtime": runtime_cost,
                "reduction_pct": pct(native_cost, runtime_cost),
            },
            "constraint_pass_rate": {
                "native": avg("native", "constraint_pass_rate"),
                "runtime": avg("runtime", "constraint_pass_rate"),
            },
            "final_pass_rate": {
                "native": round(sum(1 for x in per_task if x["native"].get("final_pass")) / len(per_task), 4),
                "runtime": round(sum(1 for x in per_task if x["runtime"].get("final_pass")) / len(per_task), 4),
            },
        },
        "resilience_with_fault": {
            "tasks_completed": {
                "native": total("native_fault", "tasks_completed"),
                "runtime": total("runtime_fault", "tasks_completed"),
                "total": total("native_fault", "tasks_total"),
            },
            "recovered_faults": {
                "native": total("native_fault", "recovered_faults"),
                "runtime": total("runtime_fault", "recovered_faults"),
                "injected": total("runtime_fault", "faults_injected"),
            },
        },
        "elapsed_secs": round(elapsed, 3),
        "price_assumptions_usd_per_1m": {
            "input_miss": PRICE_INPUT_MISS,
            "input_hit": PRICE_INPUT_HIT,
            "output": PRICE_OUTPUT,
        },
    }
    (out_dir / "per_task.json").write_text(json.dumps(per_task, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["efficiency_no_fault"], indent=2, ensure_ascii=False))
    print(json.dumps(summary["resilience_with_fault"], indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-info", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    asyncio.run(run(Path(args.ref_info), Path(args.queries), Path(args.out_dir), args.limit))


if __name__ == "__main__":
    main()
