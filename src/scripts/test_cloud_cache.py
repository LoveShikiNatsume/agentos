"""Cloud API prefix cache experiment: Baseline vs Runtime.

Tests whether Runtime's stable prefix ordering improves DeepSeek prompt cache hit ratio.
Uses the proxy at ruoli.dev with deepseek-v4-flash.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentos.llm.deepseek_backend import DeepSeekBackend
from agentos.llm.base import LLMRequest

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

PROJECT_BG = (
    "The AgentOS-Runtime project builds an OS-level runtime for managing "
    "multiple AI agents sharing GPU resources. It provides agent lifecycle "
    "management, DAG-based task scheduling, shared context pooling with "
    "ContextRef, prefix canonicalization for KV cache optimization, a unified "
    "model gateway with cache metrics tracking, and fault tolerance with "
    "recovery strategies. Architecture follows OS principles: agents as "
    "processes, scheduler manages execution order, context pool as shared "
    "memory, model gateway as device abstraction. Key modules: core "
    "(AgentControlBlock, Task, TaskDAG), scheduler (FIFO, DAG, Priority), "
    "context (SharedContextPool, ContextRef, PrefixCanonicalizer), llm "
    "(ModelGateway with DeepSeek/vLLM/LiteLLM backends), comm (MessageBus), "
    "events (RuntimeEvent)."
)

AGENTS = [
    ("planner", "You are a project planner. Analyze tasks and create execution plans."),
    ("coder", "You are a software developer. Write clean, tested code."),
    ("tester", "You are a QA engineer. Design test cases and verify correctness."),
]

QUESTIONS = [
    "What should we implement next? One sentence.",
    "What is the key challenge? One sentence.",
    "How do we verify correctness? One sentence.",
]


async def call_with_retry(backend: DeepSeekBackend, req: LLMRequest, retries: int = 5) -> any:
    for attempt in range(retries):
        try:
            return await backend.generate(req)
        except Exception as e:
            if attempt < retries - 1:
                wait = 8 * (attempt + 1)
                print(f"  Retry {attempt + 1}/{retries} after {e.__class__.__name__}. Waiting {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise


async def run_baseline(backend: DeepSeekBackend) -> list:
    """Baseline: each agent builds prompt independently, role-first prefix."""
    results = []
    for (name, role), q in zip(AGENTS, QUESTIONS):
        req = LLMRequest(
            agent_id=name,
            messages=[
                {"role": "system", "content": f"{role}\n\nProject context:\n{PROJECT_BG}"},
                {"role": "user", "content": q},
            ],
            max_tokens=30,
        )
        resp = await call_with_retry(backend, req)
        m = resp.metrics
        print(f"  {name}: latency={m.latency_secs:.2f}s prompt={m.prompt_tokens} hit={m.cache_hit_tokens} miss={m.cache_miss_tokens} ratio={m.cache_hit_ratio:.4f}")
        results.append(m)
        await asyncio.sleep(1)
    return results


async def run_runtime(backend: DeepSeekBackend) -> list:
    """Runtime: stable public prefix first, role-specific part after."""
    results = []
    for (name, role), q in zip(AGENTS, QUESTIONS):
        req = LLMRequest(
            agent_id=name,
            messages=[
                {"role": "system", "content": f"{PROJECT_BG}\n\nYour role: {role}"},
                {"role": "user", "content": q},
            ],
            max_tokens=30,
        )
        resp = await call_with_retry(backend, req)
        m = resp.metrics
        print(f"  {name}: latency={m.latency_secs:.2f}s prompt={m.prompt_tokens} hit={m.cache_hit_tokens} miss={m.cache_miss_tokens} ratio={m.cache_hit_ratio:.4f}")
        results.append(m)
        await asyncio.sleep(1)
    return results


def summarize(label: str, results: list) -> dict:
    total_prompt = sum(r.prompt_tokens for r in results)
    total_hit = sum(r.cache_hit_tokens for r in results)
    total_miss = sum(r.cache_miss_tokens for r in results)
    avg_latency = sum(r.latency_secs for r in results) / len(results)
    ratio = total_hit / total_prompt if total_prompt > 0 else 0.0
    return {
        "label": label,
        "total_prompt": total_prompt,
        "total_hit": total_hit,
        "total_miss": total_miss,
        "avg_latency": round(avg_latency, 3),
        "cache_hit_ratio": round(ratio, 4),
    }


async def main():
    backend = DeepSeekBackend(api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=120.0)

    print(f"Model: {MODEL}")
    print(f"Base URL: {BASE_URL}")
    print()

    print("Phase 1: Baseline (role-first, unstable prefix)")
    bl_results = await run_baseline(backend)
    bl = summarize("baseline", bl_results)

    print("\nCooldown 15s...\n")
    await asyncio.sleep(15)

    print("Phase 2: Runtime (background-first, stable prefix)")
    rt_results = await run_runtime(backend)
    rt = summarize("runtime", rt_results)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for s in [bl, rt]:
        print(f"  {s['label']:>10}: hit_ratio={s['cache_hit_ratio']:.4f}  avg_latency={s['avg_latency']:.2f}s  total_hit={s['total_hit']}/{s['total_prompt']}")

    improvement = rt["cache_hit_ratio"] - bl["cache_hit_ratio"]
    print(f"\n  Cache hit ratio improvement: +{improvement:.4f} ({improvement * 100:.1f} pp)")
    if rt["avg_latency"] < bl["avg_latency"]:
        pct = (1 - rt["avg_latency"] / bl["avg_latency"]) * 100
        print(f"  Latency reduction: {pct:.1f}%")

    report = {"model": MODEL, "base_url": BASE_URL, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "baseline": bl, "runtime": rt}
    report_path = os.path.join(os.path.dirname(__file__), "..", "runs", "cloud_cache_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
