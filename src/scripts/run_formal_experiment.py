"""Formal cloud API comparison experiment using BaselineExperiment framework.

Runs Baseline vs Runtime on DeepSeek official API with multiple scenarios
and multiple runs for statistical significance. Exports JSON + CSV.

Usage:
    ./.venv/bin/python scripts/run_formal_experiment.py [--runs N] [--proxy]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentos.benchmarks.baseline_context import (
    BaselineExperiment,
    ExperimentScenario,
    CODING_3AGENT,
    CODING_5AGENT,
    MINIMAL_2AGENT,
)
from agentos.llm.deepseek_backend import DeepSeekBackend

OFFICIAL_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
OFFICIAL_URL = "https://api.deepseek.com"
OFFICIAL_MODEL = "deepseek-chat"

PROXY_KEY = os.environ.get("DEEPSEEK_PROXY_KEY", "")
PROXY_URL = "https://ruoli.dev/v1"
PROXY_MODEL = "deepseek-v4-flash"

EXTENDED_CONTEXT = (
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
    "events (RuntimeEvent). The system targets competitive evaluation with "
    "quantitative benchmarks comparing cache hit ratios, token savings, "
    "latency reduction, and throughput improvements across different agent "
    "configurations and workload patterns."
)


def make_unique_scenario(base: ExperimentScenario) -> ExperimentScenario:
    """Add a unique nonce to shared_context to ensure cold-start cache."""
    nonce = uuid.uuid4().hex[:8]
    return ExperimentScenario(
        name=base.name,
        agents=base.agents,
        shared_context=f"[experiment-id: {nonce}]\n{EXTENDED_CONTEXT}",
        questions=base.questions,
        max_tokens=base.max_tokens,
    )


async def run_scenario(
    backend: DeepSeekBackend,
    scenario: ExperimentScenario,
    runs: int,
    label: str,
) -> dict:
    unique = make_unique_scenario(scenario)
    exp = BaselineExperiment(backend=backend, request_interval=1.0, retries=5)

    print(f"\n{'='*60}")
    print(f"Scenario: {label} ({len(scenario.agents)} agents, {runs} runs)")
    print(f"{'='*60}")

    results = await exp.run_comparison(unique, cooldown=30.0, runs=runs)

    bl = results["baseline"]
    rt = results["runtime"]
    improvement = rt.total_cache_hit_ratio - bl.total_cache_hit_ratio

    print(f"\n  Baseline:  hit_ratio={bl.total_cache_hit_ratio:.4f}  "
          f"avg_latency={bl.avg_latency:.2f}s  "
          f"hit={bl.total_cache_hit_tokens}/{bl.total_prompt_tokens}")
    print(f"  Runtime:   hit_ratio={rt.total_cache_hit_ratio:.4f}  "
          f"avg_latency={rt.avg_latency:.2f}s  "
          f"hit={rt.total_cache_hit_tokens}/{rt.total_prompt_tokens}")
    print(f"  Improvement: +{improvement*100:.1f} pp")

    return {
        "scenario": label,
        "agents": len(scenario.agents),
        "runs": runs,
        "baseline": bl.to_dict(),
        "runtime": rt.to_dict(),
        "improvement_pp": round(improvement * 100, 2),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=2, help="Number of runs per scenario")
    parser.add_argument("--proxy", action="store_true", help="Use proxy instead of official API")
    parser.add_argument("--scenario", choices=["all", "3agent", "5agent", "2agent"], default="all")
    args = parser.parse_args()

    if args.proxy:
        api_key, base_url, model = PROXY_KEY, PROXY_URL, PROXY_MODEL
        provider = "proxy"
    else:
        api_key, base_url, model = OFFICIAL_KEY, OFFICIAL_URL, OFFICIAL_MODEL
        provider = "official"

    backend = DeepSeekBackend(api_key=api_key, base_url=base_url, model=model, timeout=120.0)

    print(f"Provider: {provider} ({base_url})")
    print(f"Model: {model}")
    print(f"Runs per scenario: {args.runs}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    scenarios: list[tuple[str, ExperimentScenario]] = []
    if args.scenario in ("all", "3agent"):
        scenarios.append(("coding_3agent", CODING_3AGENT))
    if args.scenario in ("all", "5agent"):
        scenarios.append(("coding_5agent", CODING_5AGENT))
    if args.scenario in ("all", "2agent"):
        scenarios.append(("minimal_2agent", MINIMAL_2AGENT))

    all_results = []
    for label, scenario in scenarios:
        result = await run_scenario(backend, scenario, args.runs, label)
        all_results.append(result)
        if label != scenarios[-1][0]:
            print("\nCooldown 30s between scenarios...")
            await asyncio.sleep(30)

    import json
    from pathlib import Path

    out_dir = Path(__file__).parent.parent / "runs"
    out_dir.mkdir(exist_ok=True)

    ts = time.strftime("%Y%m%d-%H%M%S")
    report = {
        "provider": provider,
        "model": model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runs_per_scenario": args.runs,
        "results": all_results,
    }
    report_path = out_dir / f"formal_experiment_{provider}_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport saved to {report_path}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in all_results:
        print(f"  {r['scenario']:>20}: baseline={r['baseline']['total_cache_hit_ratio']:.4f}  "
              f"runtime={r['runtime']['total_cache_hit_ratio']:.4f}  "
              f"improvement=+{r['improvement_pp']:.1f}pp")


if __name__ == "__main__":
    asyncio.run(main())
