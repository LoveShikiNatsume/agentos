#!/usr/bin/env python3
"""Cloud framework benchmark runner.

Runs 4 frameworks (OpenAI Agents, LangGraph, AutoGen, CrewAI)
against DeepSeek in native and runtime modes.

Usage:
    DEEPSEEK_API_KEY=sk-xxx python scripts/run_cloud_framework_benchmark.py \
        --framework langgraph --runs 1 --max-tokens 256

    DEEPSEEK_API_KEY=sk-xxx python scripts/run_cloud_framework_benchmark.py \
        --framework all --runs 1 --max-tokens 256
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.cloud_bench.common import RunMetrics, FrameworkResult
from scripts.cloud_bench.software_task import build_task
from agentos.events.log import EventLog

FRAMEWORKS = ["openai_agents", "langgraph", "autogen", "crewai"]
FRAMEWORK_DISPLAY = {
    "openai_agents": "OpenAI Agents SDK",
    "langgraph": "LangGraph",
    "autogen": "AutoGen",
    "crewai": "CrewAI",
    "google_adk": "Google ADK",
}


def _get_adapter(name: str):
    if name == "openai_agents":
        from scripts.cloud_bench.openai_agents_adapter import run_native, run_with_runtime
    elif name == "langgraph":
        from scripts.cloud_bench.langgraph_adapter import run_native, run_with_runtime
    elif name == "autogen":
        from scripts.cloud_bench.autogen_adapter import run_native, run_with_runtime
    elif name == "crewai":
        from scripts.cloud_bench.crewai_adapter import run_native, run_with_runtime
    elif name == "google_adk":
        from scripts.cloud_bench.google_adk_adapter import run_native, run_with_runtime
    else:
        raise ValueError(f"unknown framework: {name}")
    return run_native, run_with_runtime


async def _run_one_framework(
    fw_name: str,
    steps: list,
    runs: int,
    max_tokens: int,
    event_log: EventLog,
) -> FrameworkResult:
    """Run one framework: native + runtime, each `runs` times."""
    result = FrameworkResult(framework=FRAMEWORK_DISPLAY.get(fw_name, fw_name))

    run_native_fn, run_with_runtime_fn = _get_adapter(fw_name)

    # --- native ---
    native_results: list[RunMetrics] = []
    for i in range(runs):
        try:
            m = await run_native_fn(steps, max_tokens)
            native_results.append(m)
        except Exception as exc:
            native_results.append(RunMetrics(status="error", error=str(exc)[:120]))

    ok_native = [m for m in native_results if m.status == "ok"]
    degraded_native = [m for m in native_results if m.status == "degraded"]
    if ok_native:
        first = ok_native[0]
        result.native = RunMetrics(
            avg_latency=sum(m.avg_latency for m in ok_native) / len(ok_native),
            prompt_tokens=int(sum(m.prompt_tokens for m in ok_native) / len(ok_native)),
            completion_tokens=int(sum(m.completion_tokens for m in ok_native) / len(ok_native)),
            cache_hit_tokens=int(sum(m.cache_hit_tokens for m in ok_native) / len(ok_native)),
            cache_miss_tokens=int(sum(m.cache_miss_tokens for m in ok_native) / len(ok_native)),
            observable_events=int(sum(m.observable_events for m in ok_native) / len(ok_native)),
            status="ok",
            token_source=first.token_source,
            adapter_source=first.adapter_source,
            note=first.note,
            orchestration_source=first.orchestration_source,
            model_path=first.model_path,
            runtime_contract=first.runtime_contract,
            prompt_chars_total=first.prompt_chars_total,
            prompt_chars_sent=first.prompt_chars_sent,
            prompt_truncated=first.prompt_truncated,
            framework_entrypoint_invoked=first.framework_entrypoint_invoked,
        )
        result.native.compute_cost()
        result.native.compute_context_reuse()
    elif degraded_native:
        first = degraded_native[0]
        result.native = first
    else:
        err = native_results[0].error if native_results else "no runs"
        status = native_results[0].status if native_results else "error"
        result.native = RunMetrics(status=status, error=err)

    # --- runtime ---
    runtime_results: list[RunMetrics] = []
    for i in range(runs):
        try:
            m = await run_with_runtime_fn(steps, max_tokens, event_log=event_log)
            runtime_results.append(m)
        except Exception as exc:
            runtime_results.append(RunMetrics(status="error", error=str(exc)[:120]))

    ok_runtime = [m for m in runtime_results if m.status == "ok"]
    degraded_runtime = [m for m in runtime_results if m.status == "degraded"]
    if ok_runtime:
        first = ok_runtime[0]
        result.runtime = RunMetrics(
            avg_latency=sum(m.avg_latency for m in ok_runtime) / len(ok_runtime),
            prompt_tokens=int(sum(m.prompt_tokens for m in ok_runtime) / len(ok_runtime)),
            completion_tokens=int(sum(m.completion_tokens for m in ok_runtime) / len(ok_runtime)),
            cache_hit_tokens=int(sum(m.cache_hit_tokens for m in ok_runtime) / len(ok_runtime)),
            cache_miss_tokens=int(sum(m.cache_miss_tokens for m in ok_runtime) / len(ok_runtime)),
            recovered_faults=int(sum(m.recovered_faults for m in ok_runtime) / len(ok_runtime)),
            observable_events=int(sum(m.observable_events for m in ok_runtime) / len(ok_runtime)),
            status="ok",
            token_source=first.token_source,
            adapter_source=first.adapter_source,
            note=first.note,
            orchestration_source=first.orchestration_source,
            model_path=first.model_path,
            runtime_contract=first.runtime_contract,
            prompt_chars_total=first.prompt_chars_total,
            prompt_chars_sent=first.prompt_chars_sent,
            prompt_truncated=first.prompt_truncated,
            framework_entrypoint_invoked=first.framework_entrypoint_invoked,
            model_gateway_invoked=first.model_gateway_invoked,
            runtime_mode_actual=first.runtime_mode_actual,
        )
        result.runtime.compute_cost()
        result.runtime.compute_context_reuse()
    elif degraded_runtime:
        first = degraded_runtime[0]
        result.runtime = first
    else:
        err = runtime_results[0].error if runtime_results else "no runs"
        status = runtime_results[0].status if runtime_results else "error"
        result.runtime = RunMetrics(status=status, error=err)

    result.compute_improvement()
    return result


async def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cloud framework benchmark")
    parser.add_argument("--framework", default="all", help="all or comma-separated: openai_agents,langgraph,autogen,crewai,google_adk")
    parser.add_argument("--runs", type=int, default=1, help="Runs per framework per mode")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per LLM call")
    parser.add_argument("--output", help="Output JSON path (default: runs/cloud_framework_benchmark_<ts>.json)")
    parser.add_argument("--no-install", action="store_true", help="Skip pip install of missing deps")
    parser.add_argument("--runtime-mode", choices=["auto", "gateway-direct", "framework"], default="auto",
                        help="Runtime mode: auto (prefer framework, fallback gateway-direct), "
                             "gateway-direct (force direct loop), framework (force framework-orchestrated)")
    args = parser.parse_args(argv)

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY required", file=sys.stderr)
        return 1

    ts = int(time.time())
    out_dir = _REPO_ROOT / "runs"
    out_dir.mkdir(exist_ok=True)

    output_path = Path(args.output) if args.output else out_dir / f"cloud_framework_benchmark_{ts}.json"
    event_log_path = out_dir / f"cloud_framework_benchmark_{ts}_events.jsonl"

    # Resolve frameworks
    if args.framework == "all":
        fws = FRAMEWORKS[:]
    else:
        fws = [f.strip() for f in args.framework.split(",")]
        for f in fws:
            if f not in FRAMEWORKS:
                print(f"ERROR: unknown framework '{f}'. Choose from: {FRAMEWORKS}", file=sys.stderr)
                return 1

    # Install deps if needed
    if not args.no_install:
        from scripts.cloud_bench.deps import install_deps
        print("=== Checking dependencies ===")
        dep_results = install_deps(fws, venv_python=str(_REPO_ROOT / ".venv" / "bin" / "python"))
        for fw, status in dep_results.items():
            icon = "✅" if status.available else "⚠️ "
            print(f"  {icon} {fw}: {'ok' if status.available else status.error[:80]}")

    # Check which frameworks are available
    from scripts.cloud_bench.deps import check_deps
    dep_status = {fw: check_deps(fw) for fw in fws}

    steps = build_task(max_tokens=args.max_tokens)
    event_log = EventLog(event_log_path)

    print(f"\n=== Running benchmark ({len(fws)} frameworks, {args.runs} runs, max_tokens={args.max_tokens}) ===")
    print(f"  Output: {output_path}")
    print(f"  Events: {event_log_path}")
    print()

    results: list[FrameworkResult] = []
    for fw in fws:
        if not dep_status[fw].available:
            print(f"  ⏭️  {FRAMEWORK_DISPLAY.get(fw, fw)}: skipped ({dep_status[fw].error[:60]})")
            fr = FrameworkResult(framework=FRAMEWORK_DISPLAY.get(fw, fw))
            fr.native = RunMetrics(status="skipped", error=dep_status[fw].error[:120])
            fr.runtime = RunMetrics(status="skipped", error=dep_status[fw].error[:120])
            results.append(fr)
            continue

        print(f"  🔄 {FRAMEWORK_DISPLAY.get(fw, fw)}...")
        try:
            fr = await _run_one_framework(fw, steps, args.runs, args.max_tokens, event_log)
            n_status = fr.native.status if fr.native else "error"
            r_status = fr.runtime.status if fr.runtime else "error"
            print(f"     native={n_status}, runtime={r_status}")
            results.append(fr)
        except Exception as exc:
            print(f"     ❌ error: {str(exc)[:80]}")
            fr = FrameworkResult(framework=FRAMEWORK_DISPLAY.get(fw, fw))
            fr.native = RunMetrics(status="error", error=str(exc)[:120])
            fr.runtime = RunMetrics(status="error", error=str(exc)[:120])
            results.append(fr)

    # Compute contract counts
    fw_orchestrated = [r.framework for r in results
                       if r.runtime and r.runtime.runtime_contract == "framework_orchestrated_runtime"]
    gw_direct = [r.framework for r in results
                 if r.runtime and r.runtime.runtime_contract == "model_gateway_direct_runtime"]
    comparable = [r.framework for r in results
                  if r.native and r.runtime and r.native.status == "ok" and r.runtime.status == "ok"]

    # Build output JSON
    payload = {
        "metadata": {
            "source": "AgentOS Runtime cloud benchmark (DeepSeek)",
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com",
            "scenarios": 1,
            "total_runs": args.runs * len(fws) * 2,
            "mode": "Native vs +Runtime",
            "timestamp": ts,
            "runtime_mode_requested": args.runtime_mode,
            "runtime_metric_convergence_reason": (
                "Runtime paths share AgentOS ModelGateway, shared prefix cache, "
                "one injected backend_coder fault, and common event schema; "
                "therefore runtime prompt/cache/recovery/event metrics may converge."
            ),
            "comparable_runs_count": len(comparable),
            "framework_orchestrated_runtime_count": len(fw_orchestrated),
            "strict_framework_runtime_count": len(fw_orchestrated),
            "gateway_direct_runtime_count": len(gw_direct),
            "strict_framework_runtime_frameworks": fw_orchestrated,
            "gateway_direct_frameworks": gw_direct,
        },
        "frameworks": [r.to_dict() for r in results],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n=== Results written to {output_path} ===")
    for r in results:
        n = r._metrics_or_error(r.native)
        rt = r._metrics_or_error(r.runtime)
        print(f"  {r.framework}:")
        print(f"    native:  {json.dumps(n)}")
        print(f"    runtime: {json.dumps(rt)}")
        if r.improvement:
            print(f"    improvement: {json.dumps(r.improvement)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
