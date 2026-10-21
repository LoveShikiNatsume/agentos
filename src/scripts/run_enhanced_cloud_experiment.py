"""Enhanced cloud API experiment: longer shared context + more agents.

Previous experiment used ~228 token shared context, yielding only +18.7pp improvement.
DeepSeek caches in 128-token blocks — longer shared prefix amplifies the difference.

This experiment uses ~1200 token shared context and 6-8 agents.

Usage:
    ./.venv/bin/python scripts/run_enhanced_cloud_experiment.py [--runs N] [--proxy]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentos.benchmarks.baseline_context import (
    BaselineExperiment,
    ExperimentScenario,
)
from agentos.llm.deepseek_backend import DeepSeekBackend

OFFICIAL_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
OFFICIAL_URL = "https://api.deepseek.com"
OFFICIAL_MODEL = "deepseek-chat"

PROXY_KEY = os.environ.get("DEEPSEEK_PROXY_KEY", "")
PROXY_URL = "https://ruoli.dev/v1"
PROXY_MODEL = "deepseek-v4-flash"

# ~1200 tokens of realistic project context
LONG_SHARED_CONTEXT = """\
The AgentOS-Runtime project builds an OS-level runtime for managing multiple AI \
agents sharing GPU resources. It provides agent lifecycle management, DAG-based \
task scheduling, shared context pooling with ContextRef, prefix canonicalization \
for KV cache optimization, a unified model gateway with cache metrics tracking, \
and fault tolerance with recovery strategies.

Architecture follows operating system principles: agents are managed as \
first-class processes with defined lifecycle states (CREATED, READY, RUNNING, \
BLOCKED_LLM, BLOCKED_TOOL, SUSPENDED, FINISHED, FAILED, RECOVERING). The \
scheduler manages execution order using DAG dependency analysis and critical \
path scoring. The shared context pool acts as shared memory with content-hash \
deduplication and reference counting. The model gateway serves as a device \
abstraction layer routing requests to local vLLM or cloud DeepSeek backends.

Key modules and their responsibilities:
- core/ (AgentControlBlock, Task, TaskDAG): Agent identity, state machine, \
task dependency graph with dynamic task submission support.
- scheduler/ (FIFO, DAG, Priority, CacheAffinity): Multiple scheduling \
strategies including critical-path-first and KV cache affinity scheduling \
that prioritizes tasks whose prompt prefixes are likely still in GPU memory.
- context/ (SharedContextPool, ContextRef, PrefixCanonicalizer, CacheHeatTracker, \
Compressor, TokenCounter): Context management with URI-based references \
(ctx://name), content deduplication, deterministic prefix ordering for cache \
optimization, cache heat tracking with exponential decay, and pluggable \
compression (sliding window / token budget).
- llm/ (ModelGateway, DeepSeekBackend, VLLMBackend, LiteLLMBackend, MockBackend): \
Unified model interface with per-call ModelCallMetrics recording backend, model, \
prompt_tokens, completion_tokens, cache_hit_tokens, cache_miss_tokens, latency. \
Same-prefix batch scheduling groups requests by prefix for cache locality.
- comm/ (MessageBus): Inter-agent communication with ContextRef-aware message passing.
- events/ (RuntimeEvent, EventLog): Structured event system covering agent, task, \
llm, context, tool, fault, resource, and scheduler categories with JSONL persistence.
- fault/ (FaultInjector, RecoveryPolicy, TimeoutGuard): Fault injection for testing, \
recovery strategies (retry, fallback, skip), timeout detection with configurable limits.
- isolation/ (CgroupV2Manager): OS-level resource isolation using cgroup v2 for \
memory, CPU, and PID limits with graceful fallback when not available.

The system targets competitive evaluation in the China Graduate OS Innovation \
Competition, with quantitative benchmarks comparing cache hit ratios, token \
savings, latency reduction, and throughput improvements. Validation scenarios \
include software engineering pipelines (plan → code → test → review → fix) and \
multi-agent collaborative problem solving."""

ENHANCED_6AGENT = ExperimentScenario(
    name="coding_6agent",
    agents=[
        ("planner", "You are a project planner. Break down requirements into actionable tasks with dependencies."),
        ("architect", "You are a software architect. Design system structure, interfaces, and data flow."),
        ("coder", "You are a senior developer. Write clean, efficient, well-structured code."),
        ("tester", "You are a QA engineer. Design comprehensive test cases covering edge cases."),
        ("reviewer", "You are a code reviewer. Check for bugs, security issues, and code quality."),
        ("deployer", "You are a DevOps engineer. Plan deployment, monitoring, and rollback strategies."),
    ],
    shared_context=LONG_SHARED_CONTEXT,
    questions=[
        "What should be the top priority for the next sprint? Answer in one sentence.",
        "What architectural pattern best fits the scheduler module? Answer in one sentence.",
        "What is the most critical function to implement next? Answer in one sentence.",
        "What test scenario would catch the most regression bugs? Answer in one sentence.",
        "What is the biggest code quality risk in the current codebase? Answer in one sentence.",
        "What deployment configuration needs attention before release? Answer in one sentence.",
    ],
    max_tokens=60,
)

ENHANCED_8AGENT = ExperimentScenario(
    name="coding_8agent",
    agents=[
        ("planner", "You are a project planner. Break down requirements into actionable tasks."),
        ("architect", "You are a software architect. Design system structure and interfaces."),
        ("coder_core", "You are a core module developer. Implement runtime kernel and scheduler."),
        ("coder_context", "You are a context module developer. Implement context management and caching."),
        ("coder_gateway", "You are a gateway module developer. Implement model gateway and backends."),
        ("tester", "You are a QA engineer. Design test cases and verify correctness."),
        ("reviewer", "You are a code reviewer. Check for bugs and code quality issues."),
        ("integrator", "You are an integration engineer. Ensure all modules work together correctly."),
    ],
    shared_context=LONG_SHARED_CONTEXT,
    questions=[
        "What should be the top priority for the next sprint? Answer in one sentence.",
        "What architectural improvement would have the most impact? Answer in one sentence.",
        "What core module function needs implementation first? Answer in one sentence.",
        "What context optimization would improve cache hit rates most? Answer in one sentence.",
        "What gateway feature is most important for benchmarks? Answer in one sentence.",
        "What test scenario would catch the most regressions? Answer in one sentence.",
        "What code quality issue needs immediate attention? Answer in one sentence.",
        "What integration risk should we address first? Answer in one sentence.",
    ],
    max_tokens=60,
)


def make_unique_scenario(base: ExperimentScenario) -> ExperimentScenario:
    nonce = uuid.uuid4().hex[:8]
    return ExperimentScenario(
        name=base.name,
        agents=base.agents,
        shared_context=f"[experiment-id: {nonce}]\n{base.shared_context}",
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
    exp = BaselineExperiment(backend=backend, request_interval=1.5, retries=5)

    print(f"\n{'='*60}")
    print(f"Scenario: {label} ({len(scenario.agents)} agents, {runs} runs)")
    print(f"Shared context length: ~{len(scenario.shared_context)} chars")
    print(f"{'='*60}")

    results = await exp.run_comparison(unique, cooldown=30.0, runs=runs)

    bl = results["baseline"]
    rt = results["runtime"]
    improvement = rt.total_cache_hit_ratio - bl.total_cache_hit_ratio

    print(f"\n  Baseline:  hit_ratio={bl.total_cache_hit_ratio:.4f}  "
          f"avg_latency={bl.avg_latency:.3f}s  "
          f"hit={bl.total_cache_hit_tokens}/{bl.total_prompt_tokens}")
    print(f"  Runtime:   hit_ratio={rt.total_cache_hit_ratio:.4f}  "
          f"avg_latency={rt.avg_latency:.3f}s  "
          f"hit={rt.total_cache_hit_tokens}/{rt.total_prompt_tokens}")
    print(f"  Improvement: {improvement*100:+.1f} pp")

    return {
        "scenario": label,
        "agents": len(scenario.agents),
        "runs": runs,
        "shared_context_chars": len(scenario.shared_context),
        "baseline": bl.to_dict(),
        "runtime": rt.to_dict(),
        "improvement_pp": round(improvement * 100, 2),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--proxy", action="store_true")
    parser.add_argument("--scenario", choices=["all", "6agent", "8agent"], default="all")
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
    print(f"Shared context: ~{len(LONG_SHARED_CONTEXT)} chars")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    scenarios: list[tuple[str, ExperimentScenario]] = []
    if args.scenario in ("all", "6agent"):
        scenarios.append(("coding_6agent", ENHANCED_6AGENT))
    if args.scenario in ("all", "8agent"):
        scenarios.append(("coding_8agent", ENHANCED_8AGENT))

    all_results = []
    for label, scenario in scenarios:
        result = await run_scenario(backend, scenario, args.runs, label)
        all_results.append(result)
        if label != scenarios[-1][0]:
            print("\nCooldown 30s between scenarios...")
            await asyncio.sleep(30)

    out_dir = Path(__file__).parent.parent / "runs"
    out_dir.mkdir(exist_ok=True)

    ts = time.strftime("%Y%m%d-%H%M%S")
    report = {
        "experiment": "enhanced_cloud_prefix_cache",
        "description": "Longer shared context (~1200 tokens) + more agents (6-8) for amplified cache difference",
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runs_per_scenario": args.runs,
        "shared_context_chars": len(LONG_SHARED_CONTEXT),
        "results": all_results,
    }
    report_path = out_dir / f"enhanced_experiment_{provider}_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport saved to {report_path}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  {r['scenario']:>15} ({r['agents']} agents): "
              f"baseline={r['baseline']['total_cache_hit_ratio']:.4f}  "
              f"runtime={r['runtime']['total_cache_hit_ratio']:.4f}  "
              f"improvement={r['improvement_pp']:+.1f}pp")


if __name__ == "__main__":
    asyncio.run(main())
