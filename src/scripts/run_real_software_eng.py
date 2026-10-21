#!/usr/bin/env python3
"""Run software_eng scenario with DeepSeek backend, producing real event logs.

Variants:
    base   -- standard 11-task DAG, default parameters (P1 behavior)
    stress -- max_batch_size=2 + delay=8s + lowered fairness thresholds
              → triggers fairness_warning + multiple policy_selected
    cache  -- ModelGateway with CacheHeatTracker + large segment_budget
              → triggers CACHE_AFFINITY policy transitions

Usage:
    DEEPSEEK_API_KEY=sk-xxx python scripts/run_real_software_eng.py --variant base
    DEEPSEEK_API_KEY=sk-xxx python scripts/run_real_software_eng.py --variant stress
    DEEPSEEK_API_KEY=sk-xxx python scripts/run_real_software_eng.py --variant cache
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agentos.events.log import EventLog
from agentos.events.types import RuntimeEvent


def _print_event_stats(log_path: Path) -> Counter[str]:
    """Print event type distribution and return counter."""
    counter: Counter[str] = Counter()
    for event in EventLog(log_path).read():
        counter[event.event_type] += 1

    print(f"\n=== Event type distribution ({log_path}) ===")
    for etype, count in sorted(counter.items()):
        print(f"  {etype:<40} {count}")
    print(f"  {'TOTAL':<40} {sum(counter.values())}")
    return counter


async def _run_base(log_path: Path) -> None:
    """Standard run: 11-task DAG, default parameters, fault injection."""
    from agentos.scenarios.software_eng.pipeline import run_software_eng

    print("  Variant: base (standard 11-task DAG, fault injection ON)")
    await run_software_eng(
        backend="deepseek",
        log_path=str(log_path),
        inject_fault=True,
    )


async def _run_stress(log_path: Path) -> None:
    """Stress run: small batch + delay + lowered fairness thresholds.

    Goal: trigger scheduler.fairness_warning and multiple policy_selected.
    """
    import asyncio as _aio
    from agentos.context.manager import ContextManager
    from agentos.context.eviction import SegmentEvictor
    from agentos.context.prefix import ContextPriority
    from agentos.events.log import EventLog as ELog
    from agentos.llm.base import LLMRequest, LLMResponse
    from agentos.llm.deepseek_backend import DeepSeekBackend
    from agentos.runtime.engine import Runtime
    from agentos.scheduler.fairness import FairnessTracker
    from agentos.scenarios.software_eng.pipeline import (
        SEGMENT_BUDGET,
        PROJECT_CONTEXT,
        REPO_SUMMARY,
        build_software_eng_dag,
        _build_handlers,
    )

    print("  Variant: stress (max_batch_size=2, delay=8s, lowered fairness thresholds)")

    class DelayedBackend:
        """Wraps DeepSeekBackend with an artificial delay per call."""
        def __init__(self, inner: DeepSeekBackend, delay_secs: float):
            self._inner = inner
            self._delay = delay_secs

        async def generate(self, request: LLMRequest) -> LLMResponse:
            resp = await self._inner.generate(request)
            await _aio.sleep(self._delay)
            return resp

    ctx_mgr = ContextManager(
        evictor=SegmentEvictor(),
        segment_budget=SEGMENT_BUDGET,
    )
    ctx_mgr.add_shared_ref(
        "project-context", PROJECT_CONTEXT, priority=ContextPriority.TASK_BACKGROUND,
    )
    ctx_mgr.add_shared_ref(
        "repo-summary", REPO_SUMMARY, priority=ContextPriority.REPO_SUMMARY,
    )

    dag = build_software_eng_dag()
    llm_backend = DelayedBackend(DeepSeekBackend(), delay_secs=8.0)

    # Lower fairness thresholds so warnings trigger with moderate wait times:
    # medium_wait_ms=500 (default 1000), high_wait_ms=1500 (default 3000)
    fairness = FairnessTracker(medium_wait_ms=500, high_wait_ms=1500)

    rt = Runtime(
        llm_backend=llm_backend,
        handlers=_build_handlers(),
        context_manager=ctx_mgr,
        event_log=ELog(log_path),
        max_batch_size=2,           # Limit concurrency → tasks queue up
        scheduler_fairness_tracker=fairness,
    )

    await rt.run(dag)


async def _run_cache(log_path: Path) -> None:
    """Cache run: ModelGateway with CacheHeatTracker, large segment budget.

    Goal: trigger CACHE_AFFINITY policy transitions via prefix heat tracking.
    """
    from agentos.context.cache_tracker import CacheHeatTracker
    from agentos.context.manager import ContextManager
    from agentos.context.eviction import SegmentEvictor
    from agentos.context.prefix import ContextPriority
    from agentos.events.log import EventLog as ELog
    from agentos.llm.deepseek_backend import DeepSeekBackend
    from agentos.llm.gateway import ModelGateway
    from agentos.runtime.engine import Runtime
    from agentos.scenarios.software_eng.pipeline import (
        PROJECT_CONTEXT,
        REPO_SUMMARY,
        build_software_eng_dag,
        _build_handlers,
    )

    print("  Variant: cache (ModelGateway + CacheHeatTracker, segment_budget=2000)")

    # Large segment budget → context not evicted → prefix stays hot
    ctx_mgr = ContextManager(
        evictor=SegmentEvictor(),
        segment_budget=2000,
    )
    ctx_mgr.add_shared_ref(
        "project-context", PROJECT_CONTEXT, priority=ContextPriority.TASK_BACKGROUND,
    )
    ctx_mgr.add_shared_ref(
        "repo-summary", REPO_SUMMARY, priority=ContextPriority.REPO_SUMMARY,
    )

    dag = build_software_eng_dag()

    # Wrap DeepSeekBackend in ModelGateway with CacheHeatTracker
    cache_tracker = CacheHeatTracker(ttl=300, hot_threshold=0.3)
    gateway = ModelGateway(
        backends={"deepseek": DeepSeekBackend()},
        default_backend="deepseek",
        cache_tracker=cache_tracker,
    )

    rt = Runtime(
        llm_backend=gateway,
        handlers=_build_handlers(),
        context_manager=ctx_mgr,
        event_log=ELog(log_path),
        # No max_batch_size → unlimited concurrency (default)
    )

    await rt.run(dag)


async def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run software_eng with DeepSeek")
    parser.add_argument(
        "--variant",
        choices=["base", "stress", "cache"],
        default="base",
        help="Run variant (default: base)",
    )
    args = parser.parse_args(argv)

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY environment variable is required", file=sys.stderr)
        return 1

    ts = int(time.time())
    run_dir = Path(f"runs/software_eng_real_{args.variant}_{ts}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "events.jsonl"

    print(f"=== software_eng [{args.variant}] with DeepSeek backend ===")
    print(f"  Event log: {log_path}")
    print()

    variant_fn = {
        "base": _run_base,
        "stress": _run_stress,
        "cache": _run_cache,
    }[args.variant]

    await variant_fn(log_path)

    counter = _print_event_stats(log_path)

    # --- Check key scheduler events ---
    key_events = {
        "scheduler.decision",
        "scheduler.policy_selected",
        "scheduler.fairness_warning",
        "scheduler.kernel_probe",
    }
    present = set(counter.keys())
    for evt in sorted(key_events):
        count = counter.get(evt, 0)
        status = "✅" if count > 0 else "⚠️ "
        print(f"  {status} {evt}: {count}")

    print(f"\nDone. Results in {run_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
