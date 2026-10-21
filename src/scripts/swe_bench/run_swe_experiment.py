"""Multi-agent SWE-bench experiment: token / cache / cost / fault-recovery.

We run a 6-agent software-engineering DAG (localize -> analyze -> patch ->
{test, review} -> summarize) over REAL SWE-bench Lite tasks. The big shared
repo+issue context (median ~10k tokens) is the dominant token mass, and it is
*shared by every agent* — exactly where an Agent Runtime should help.

We do NOT score correctness (% resolved). We measure systems efficiency:

  Efficiency arms (no fault):
    baseline : naive multi-agent — each agent inlines the FULL repo context in
               its own prompt (role text first), so the shared context is NOT a
               stable cacheable prefix and is paid for, full price, by every
               agent.  -> high cache-MISS tokens.
    runtime  : AgentOS — repo context lives once in a shared, prefix-canonical
               pool; agents send only their role prompt, so the shared context
               becomes a stable prefix reused (prompt-cache hit) across agents.
               -> most context tokens become cheap cache hits.

  Resilience arms (kill fault injected on the patcher task):
    baseline : no recovery -> patcher dies, downstream blocked, partial finish.
    runtime  : ContextRetryPolicy -> patcher recovers from ContextRef, finishes.

Honest reporting: raw prompt_tokens are similar across arms; the real, price-
relevant saving is in cache-MISS (full-price) tokens and estimated cost. We
report all of them separately, never a single inflated percentage.

Usage:
    # dry run, no API cost:
    .venv/bin/python scripts/swe_bench/run_swe_experiment.py --backend mock --limit 2
    # real run (needs DEEPSEEK_API_KEY in env):
    .venv/bin/python scripts/swe_bench/run_swe_experiment.py --backend deepseek
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from agentos.context.eviction import SegmentEvictor
from agentos.context.manager import ContextManager
from agentos.context.prefix import ContextPriority
from agentos.core.task import Task, TaskDAG
from agentos.events.log import EventLog
from agentos.fault.injector import FaultInjectionPlan
from agentos.fault.recovery import ContextRetryPolicy
from agentos.llm.factory import create_backend
from agentos.runtime.engine import AgentHandler, Runtime

# ---- DeepSeek list-price ASSUMPTIONS (USD / 1M tokens). Adjust if prices move.
# Only used for the *estimated cost* line; the cache-miss-token saving below is
# price-independent and is the robust headline.
PRICE_INPUT_MISS = 0.27
PRICE_INPUT_HIT = 0.07
PRICE_OUTPUT = 1.10

MAX_TOKENS = 300

# 6-agent pipeline. Roles are deliberately short — the real content is the
# shared repo context, which is what we want to share rather than duplicate.
ROLES: dict[str, str] = {
    "localizer": "You are a code-localization agent. Using the repository context and the "
                 "issue, list the files and functions most relevant to fixing this issue. "
                 "Be concise.",
    "analyzer": "You are a root-cause-analysis agent. Explain the underlying cause of the "
                "issue based on the code. Be concise.",
    "patcher": "You are a patch-writing agent. Propose a minimal change (unified diff or "
               "short code) that resolves the issue. Be concise.",
    "tester": "You are a test-writing agent. Describe the tests that would verify the fix. "
              "Be concise.",
    "reviewer": "You are a code reviewer. Review the proposed approach for correctness and "
                "side effects. Be concise.",
    "summarizer": "You are the integrator. Summarize localization, fix, tests and review "
                  "into a short final report.",
}


def build_swe_dag() -> TaskDAG:
    dag = TaskDAG()
    dag.add_task(Task(task_id="localizer", name="localizer", agent_role="localizer"))
    dag.add_task(Task(task_id="analyzer", name="analyzer", agent_role="analyzer",
                      dependencies=["localizer"]))
    dag.add_task(Task(task_id="patcher", name="patcher", agent_role="patcher",
                      dependencies=["analyzer"]))
    dag.add_task(Task(task_id="tester", name="tester", agent_role="tester",
                      dependencies=["patcher"]))
    dag.add_task(Task(task_id="reviewer", name="reviewer", agent_role="reviewer",
                      dependencies=["patcher"]))
    dag.add_task(Task(task_id="summarizer", name="summarizer", agent_role="summarizer",
                      dependencies=["tester", "reviewer"]))
    return dag


def _make_handler(role_prompt: str, arm: str, shared_context: str) -> AgentHandler:
    """baseline inlines the full context (role first); runtime relies on the
    shared pool prefix injected by build_messages()."""

    class _H(AgentHandler):
        async def run(self, ctx: "AgentHandler.Context") -> str:
            if arm == "baseline":
                user = (
                    f"{role_prompt}\n\n"
                    f"<repository_context>\n{shared_context}\n</repository_context>"
                )
                messages = [{"role": "user", "content": user}]
            else:  # runtime: shared context comes from the canonical prefix pool
                messages = [{"role": "user", "content": role_prompt}]
            resp = await ctx.llm_call(messages)
            return resp.content

    return _H()


def _build_handlers(arm: str, shared_context: str) -> dict[str, AgentHandler]:
    return {role: _make_handler(prompt, arm, shared_context)
            for role, prompt in ROLES.items()}


def _make_runtime(arm: str, backend: str, shared_context: str,
                  log_path: Path, fault: bool) -> Runtime:
    llm = create_backend(backend)
    handlers = _build_handlers(arm, shared_context)
    log = EventLog(log_path)
    fault_plan = FaultInjectionPlan({"patcher": "kill"}) if fault else None

    if arm == "baseline":
        # naive: plain context manager, no shared pool, no recovery.
        return Runtime(
            llm_backend=llm, handlers=handlers,
            context_manager=ContextManager(), event_log=log,
            fault_injection=fault_plan,
        )
    # runtime: shared, prefix-canonical pool; recovery on when fault is injected.
    ctx_mgr = ContextManager(evictor=SegmentEvictor(), segment_budget=None)
    ctx_mgr.add_shared_ref("repo-context", shared_context,
                           priority=ContextPriority.TASK_BACKGROUND)
    return Runtime(
        llm_backend=llm, handlers=handlers,
        context_manager=ctx_mgr, event_log=log,
        fault_injection=fault_plan,
        recovery_policy=ContextRetryPolicy(max_retries=1) if fault else None,
    )


def _collect(rt: Runtime, dag: TaskDAG, elapsed: float) -> dict[str, Any]:
    events = rt.events
    resp = [e for e in events if e.event_type == "llm.response_received"]
    prompt = sum(e.data.get("prompt_tokens", e.data.get("tokens_used", 0)) for e in resp)
    hit = sum(e.data.get("cache_hit_tokens", 0) for e in resp)
    miss = sum(e.data.get("cache_miss_tokens", 0) for e in resp)
    completion = sum(e.data.get("completion_tokens", 0) for e in resp)
    total_cache = hit + miss
    completed = sum(1 for e in events if e.event_type == "task.completed")
    cost = (miss * PRICE_INPUT_MISS + hit * PRICE_INPUT_HIT
            + completion * PRICE_OUTPUT) / 1e6
    return {
        "elapsed_secs": round(elapsed, 2),
        "llm_calls": len(resp),
        "prompt_tokens": prompt,
        "cache_hit_tokens": hit,
        "cache_miss_tokens": miss,
        "completion_tokens": completion,
        "cache_hit_ratio": round(hit / total_cache, 4) if total_cache else 0.0,
        "est_cost_usd": round(cost, 6),
        "tasks_completed": completed,
        "tasks_total": len(dag.tasks),
        "fault_injected": sum(1 for e in events if e.event_type == "fault.injected"),
        "recovered_faults": sum(1 for e in events
                                if e.event_type == "fault.recovery_completed"),
    }


async def _run_arm(arm: str, backend: str, shared_context: str,
                   log_path: Path, fault: bool) -> dict[str, Any]:
    if log_path.exists():
        log_path.unlink()
    dag = build_swe_dag()
    rt = _make_runtime(arm, backend, shared_context, log_path, fault)
    t0 = time.monotonic()
    try:
        await rt.run(dag)
    except Exception as exc:  # baseline + fault may surface; still collect what ran
        m = _collect(rt, dag, time.monotonic() - t0)
        m["error"] = f"{type(exc).__name__}: {exc}"
        return m
    return _collect(rt, dag, time.monotonic() - t0)


async def run_experiment(backend: str, tasks_path: Path, out_dir: Path,
                         limit: int | None) -> None:
    data = json.loads(tasks_path.read_text())
    tasks = data["tasks"][: limit] if limit else data["tasks"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    per_task: list[dict[str, Any]] = []
    for i, t in enumerate(tasks):
        iid = t["instance_id"]
        ctx = t["shared_context"]
        print(f"[{i + 1}/{len(tasks)}] {iid}  (~{t['est_context_tokens']} ctx tok)")
        rec: dict[str, Any] = {"instance_id": iid, "repo": t["repo"],
                               "est_context_tokens": t["est_context_tokens"]}
        # efficiency arms (no fault) + resilience arms (fault)
        for arm, fault in [("baseline", False), ("runtime", False),
                           ("baseline", True), ("runtime", True)]:
            key = f"{arm}{'_fault' if fault else ''}"
            log_path = log_dir / f"{iid}_{key}.jsonl"
            m = await _run_arm(arm, backend, ctx, log_path, fault)
            rec[key] = m
            print(f"    {key:16s} calls={m['llm_calls']} "
                  f"miss_tok={m['cache_miss_tokens']:>7} hit={m['cache_hit_ratio']:.0%} "
                  f"cost=${m['est_cost_usd']:.4f} done={m['tasks_completed']}/{m['tasks_total']} "
                  f"recov={m['recovered_faults']}")
        per_task.append(rec)

    summary = _aggregate(per_task)
    (out_dir / "per_task.json").write_text(json.dumps(per_task, indent=2, ensure_ascii=False))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[saved] {out_dir}/summary.json")
    _print_summary(summary)


def _aggregate(per_task: list[dict[str, Any]]) -> dict[str, Any]:
    def s(key: str, field: str) -> int:
        return sum(t[key][field] for t in per_task)

    eff_b_miss = s("baseline", "cache_miss_tokens")
    eff_r_miss = s("runtime", "cache_miss_tokens")
    eff_b_cost = sum(t["baseline"]["est_cost_usd"] for t in per_task)
    eff_r_cost = sum(t["runtime"]["est_cost_usd"] for t in per_task)
    eff_b_prompt = s("baseline", "prompt_tokens")
    eff_r_prompt = s("runtime", "prompt_tokens")

    def pct(b: float, r: float) -> float:
        return round((b - r) / b * 100, 1) if b else 0.0

    return {
        "n_tasks": len(per_task),
        "efficiency_no_fault": {
            "prompt_tokens": {"baseline": eff_b_prompt, "runtime": eff_r_prompt,
                              "reduction_pct": pct(eff_b_prompt, eff_r_prompt)},
            "cache_miss_tokens": {"baseline": eff_b_miss, "runtime": eff_r_miss,
                                  "reduction_pct": pct(eff_b_miss, eff_r_miss),
                                  "note": "full-price input tokens — the headline saving"},
            "cache_hit_ratio": {
                "baseline": round(sum(t["baseline"]["cache_hit_ratio"] for t in per_task) / len(per_task), 4),
                "runtime": round(sum(t["runtime"]["cache_hit_ratio"] for t in per_task) / len(per_task), 4)},
            "est_cost_usd": {"baseline": round(eff_b_cost, 4), "runtime": round(eff_r_cost, 4),
                             "reduction_pct": pct(eff_b_cost, eff_r_cost)},
        },
        "resilience_with_fault": {
            "tasks_completed": {
                "baseline": s("baseline_fault", "tasks_completed"),
                "runtime": s("runtime_fault", "tasks_completed"),
                "total": s("baseline_fault", "tasks_total")},
            "recovered_faults": {
                "baseline": s("baseline_fault", "recovered_faults"),
                "runtime": s("runtime_fault", "recovered_faults"),
                "faults_injected": s("runtime_fault", "fault_injected")},
        },
        "price_assumptions_usd_per_1m": {"input_miss": PRICE_INPUT_MISS,
                                         "input_hit": PRICE_INPUT_HIT, "output": PRICE_OUTPUT},
    }


def _print_summary(s: dict[str, Any]) -> None:
    eff = s["efficiency_no_fault"]
    res = s["resilience_with_fault"]
    print("=" * 64)
    print(f"  SWE-bench Lite multi-agent experiment  (n={s['n_tasks']} real tasks)")
    print("=" * 64)
    print("  [Efficiency, no fault]  baseline -> runtime")
    print(f"    full-price (cache-miss) input tok : {eff['cache_miss_tokens']['baseline']:>9} -> "
          f"{eff['cache_miss_tokens']['runtime']:>9}   (-{eff['cache_miss_tokens']['reduction_pct']}%)")
    print(f"    cache hit ratio                   : {eff['cache_hit_ratio']['baseline']:.0%} -> "
          f"{eff['cache_hit_ratio']['runtime']:.0%}")
    print(f"    estimated cost (USD)              : ${eff['est_cost_usd']['baseline']} -> "
          f"${eff['est_cost_usd']['runtime']}   (-{eff['est_cost_usd']['reduction_pct']}%)")
    print(f"    raw prompt tokens (similar)       : {eff['prompt_tokens']['baseline']} -> "
          f"{eff['prompt_tokens']['runtime']}")
    print("  [Resilience, fault on patcher]")
    print(f"    tasks completed                   : baseline "
          f"{res['tasks_completed']['baseline']}/{res['tasks_completed']['total']}  vs  "
          f"runtime {res['tasks_completed']['runtime']}/{res['tasks_completed']['total']}")
    print(f"    faults recovered                  : baseline {res['recovered_faults']['baseline']}  vs  "
          f"runtime {res['recovered_faults']['runtime']}  "
          f"(injected {res['recovered_faults']['faults_injected']})")


def main() -> None:
    ap = argparse.ArgumentParser(prog="run-swe-experiment")
    ap.add_argument("--backend", default="mock",
                    choices=["mock", "cache-sim", "deepseek-sim", "deepseek", "vllm"])
    ap.add_argument("--tasks", default="runs/swe_bench/tasks.json")
    ap.add_argument("--out-dir", default="runs/swe_bench/result")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(run_experiment(args.backend, Path(args.tasks), Path(args.out_dir), args.limit))


if __name__ == "__main__":
    main()
