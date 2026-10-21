"""Benchmark: FIFO vs Critical-Path vs Cache-Affinity scheduling on local vLLM.

Demonstrates that scheduling order affects KV cache hit rate and latency.

Design:
  3 prefix groups × 4 tasks = 12 tasks, all independent (ready simultaneously).
  Each group has a distinct ~1000 token system prompt.
  Tasks are dispatched one-at-a-time (serial) so order = strategy output.

  - Interleaved (worst):  A1 B1 C1 A2 B2 C2 A3 B3 C3 A4 B4 C4
  - Critical-Path:        sorted by score/priority, prefix-unaware
  - Cache-Affinity:       A1 A2 A3 A4 B1 B2 B3 B4 C1 C2 C3 C4

Usage:
    ./.venv/bin/python scripts/benchmark_scheduling_strategies.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentos.context.cache_tracker import CacheHeatTracker
from agentos.context.manager import ContextManager
from agentos.core.task import Task, TaskDAG
from agentos.llm.base import LLMRequest
from agentos.llm.vllm_backend import VLLMBackend
from agentos.metrics.model import ModelCallMetrics
from agentos.scheduler.affinity import CacheAffinityAdvisor
from agentos.scheduler.dag import DAGScheduler


# ---------------------------------------------------------------------------
# Prefix content — 3 distinct groups, each ~600-800 tokens
# ---------------------------------------------------------------------------

def _build_group_context(group_name: str, domain: str, details: str, nonce: str = "",
                         padding_lines: int = 0) -> str:
    nonce_line = f"\n    [Run ID: {nonce}]\n" if nonce else ""
    padding = ""
    if padding_lines > 0:
        padding = "\n" + "\n".join(
            f"    Section {i+1} — {group_name} extended specification: "
            f"The {group_name} subsystem maintains internal state across distributed nodes "
            f"using a consensus protocol with configurable quorum size. Each node runs an "
            f"independent health monitor that reports to the central coordinator. Recovery "
            f"procedures are triggered automatically when node health drops below threshold. "
            f"The coordinator assigns work based on current load and historical performance "
            f"metrics collected over a sliding window of configurable duration."
            for i in range(padding_lines)
        )
    return textwrap.dedent(f"""\
    [Project Group: {group_name}]{nonce_line}

    {domain}

    Technical Architecture:
    The system follows a layered architecture with clear separation of concerns.
    The core layer manages fundamental data structures and state machines.
    The scheduling layer implements multiple strategies including FIFO, priority-based,
    critical-path-first, and cache-affinity scheduling that considers GPU KV cache state.
    The context layer provides shared memory with content-hash deduplication,
    URI-based references (ctx://name), deterministic prefix ordering for cache
    optimization, and pluggable compression strategies.
    The model gateway layer abstracts local and cloud inference backends behind
    a unified interface with per-call metrics tracking.

    {details}

    Implementation Guidelines:
    All components communicate through well-defined interfaces. The runtime engine
    orchestrates agent lifecycle, task scheduling, context building, and model calls.
    Events are structured with categories (agent, task, llm, context, tool, fault,
    resource, scheduler) and persisted as JSONL for offline analysis. Resource
    isolation uses cgroup v2 with graceful fallback. Fault tolerance includes
    retry, fallback, and skip strategies with full trace visibility.
    Recovery preserves context snapshots to avoid redundant token consumption.
    The benchmark framework compares baseline (independent agents) against
    runtime-managed execution across latency, cache hit ratio, token savings,
    and throughput metrics.{padding}""")


GROUP_SPECS = {
    "alpha": (
        "Alpha",
        "Alpha team builds the agent lifecycle and scheduling subsystem. "
        "Agents are first-class processes with states: CREATED, READY, RUNNING, "
        "BLOCKED_LLM, BLOCKED_TOOL, SUSPENDED, FINISHED, FAILED, RECOVERING. "
        "The DAG scheduler computes critical path scores and dispatches ready "
        "tasks respecting dependency constraints and resource quotas.",
        "Alpha-specific modules: core/agent.py (AgentControlBlock with state machine), "
        "core/task.py (Task, TaskDAG with dynamic submission), scheduler/dag.py "
        "(DAGScheduler with critical path and resource awareness), runtime/engine.py "
        "(main execution loop, agent creation, state transitions, event emission). "
        "Key invariant: all agents must be created through Runtime, never directly.",
    ),
    "beta": (
        "Beta",
        "Beta team builds the context management and model gateway subsystem. "
        "Context is managed through ContextRef URIs (ctx://name) with shared and "
        "private scopes, content-hash deduplication, reference counting, and "
        "access control with audit events. The PrefixCanonicalizer ensures "
        "byte-identical prompt prefixes across agents for KV cache optimization.",
        "Beta-specific modules: context/manager.py (ContextManager with resolve_prefix_id), "
        "context/pool.py (SharedContextPool with dedup and token estimation), "
        "context/cache_tracker.py (CacheHeatTracker with exponential decay), "
        "llm/gateway.py (ModelGateway with same-prefix batch scheduling), "
        "llm/deepseek_backend.py and llm/vllm_backend.py (cache metrics extraction). "
        "Key invariant: all model calls go through ModelGateway, never direct.",
    ),
    "gamma": (
        "Gamma",
        "Gamma team builds the execution, observability, and fault tolerance "
        "subsystem. Worker processes run in subprocess isolation with cgroup v2 "
        "resource limits. The event log captures structured RuntimeEvents across "
        "all categories with JSONL persistence. Fault injection supports kill "
        "and timeout scenarios with configurable recovery strategies.",
        "Gamma-specific modules: fault/injector.py (FaultInjectionPlan for testing), "
        "fault/recovery.py (RecoveryPolicy with retry/fallback/skip), "
        "fault/timeout.py (TimeoutGuard with configurable limits), "
        "isolation/cgroup.py (CgroupV2Manager for memory/cpu/pid limits), "
        "worker/process.py (SubprocessWorkerManager with lifecycle tracking), "
        "events/log.py (EventLog with JSONL persistence and query support). "
        "Key invariant: single agent failure must not cascade to the DAG.",
    ),
    "delta": (
        "Delta",
        "Delta team builds the API gateway and request routing layer. "
        "Incoming requests are classified by intent, validated against schemas, "
        "rate-limited per client, and dispatched to the appropriate backend service. "
        "The gateway maintains connection pools, circuit breakers, and request tracing.",
        "Delta-specific modules: api/gateway.py (RequestRouter with intent classification), "
        "api/schema.py (SchemaValidator with OpenAPI 3.1 compliance), api/rate_limit.py "
        "(TokenBucket rate limiter with burst support), api/tracing.py (distributed "
        "trace propagation with W3C TraceContext). Key invariant: all external requests "
        "must pass through the gateway; direct backend access is forbidden.",
    ),
    "epsilon": (
        "Epsilon",
        "Epsilon team builds the data persistence and caching layer. "
        "Structured data uses PostgreSQL with connection pooling and read replicas. "
        "Unstructured data uses object storage with CDN acceleration. "
        "Hot data is cached in Redis with write-through invalidation.",
        "Epsilon-specific modules: storage/pg.py (PostgresPool with read/write splitting), "
        "storage/object.py (ObjectStore with S3-compatible API), storage/cache.py "
        "(RedisCache with write-through and time-based eviction), storage/migration.py "
        "(schema migration runner with rollback support). Key invariant: all writes "
        "go through the primary; reads prefer replicas with configurable staleness.",
    ),
    "zeta": (
        "Zeta",
        "Zeta team builds the authentication and authorization layer. "
        "Users authenticate via OAuth2 / OIDC with support for social login, "
        "SAML federation, and hardware security keys. Authorization uses RBAC "
        "with dynamic policy evaluation at request time.",
        "Zeta-specific modules: auth/oauth.py (OAuth2Server with PKCE and refresh), "
        "auth/saml.py (SAMLProvider with metadata exchange), auth/rbac.py "
        "(PolicyEngine with Rego-based evaluation), auth/session.py (SessionManager "
        "with secure token storage and rotation). Key invariant: every API call "
        "must carry a validated token; anonymous access is explicit and audit-logged.",
    ),
    "eta": (
        "Eta",
        "Eta team builds the notification and messaging infrastructure. "
        "Events from all subsystems flow through a message bus with topic-based "
        "routing, dead-letter queues, and exactly-once delivery guarantees. "
        "User-facing notifications support email, push, SMS, and in-app channels.",
        "Eta-specific modules: messaging/bus.py (MessageBus with topic routing), "
        "messaging/dlq.py (DeadLetterQueue with retry scheduling), messaging/push.py "
        "(PushNotifier with APNs and FCM), messaging/email.py (EmailSender with "
        "template rendering and bounce handling). Key invariant: messages are "
        "persisted before acknowledgment; no message is silently dropped.",
    ),
    "theta": (
        "Theta",
        "Theta team builds the monitoring, alerting, and SRE tooling layer. "
        "Metrics are collected via OpenTelemetry, stored in a time-series database, "
        "and visualized through dashboards. Alerts use multi-signal detection with "
        "configurable escalation policies and on-call rotation.",
        "Theta-specific modules: observability/metrics.py (OTelCollector with "
        "histogram and counter aggregation), observability/alerts.py (AlertManager "
        "with multi-signal detection), observability/dashboard.py (DashboardBuilder "
        "with Grafana JSON export), observability/oncall.py (EscalationPolicy with "
        "PagerDuty integration). Key invariant: all production services must emit "
        "RED metrics (rate, errors, duration) within their first week.",
    ),
}

# Need real token count to exceed ~108k cache capacity.
# At 130 lines, vLLM reports ~10.3k tokens/group (82k total) — still fits.
# At 250 lines, expect ~18k tokens/group × 8 = ~144k → should force evictions.
PADDING_LINES = 250

import uuid


def build_group_contexts(nonce: str) -> dict[str, str]:
    return {
        group: _build_group_context(name, domain, details, nonce=nonce,
                                     padding_lines=PADDING_LINES)
        for group, (name, domain, details) in GROUP_SPECS.items()
    }

QUESTIONS = [
    "What is the highest priority task for this group? One sentence.",
    "What test scenario would catch the most bugs? One sentence.",
    "What interface needs the most attention? One sentence.",
    "What performance bottleneck should we address first? One sentence.",
]


# ---------------------------------------------------------------------------
# Task + ordering strategies
# ---------------------------------------------------------------------------

TASKS_PER_GROUP = 3


def build_tasks() -> list[Task]:
    tasks = []
    for group in GROUP_SPECS:
        for i in range(TASKS_PER_GROUP):
            tasks.append(Task(
                task_id=f"{group}-{i}",
                name=f"{group} task {i}",
                agent_role="worker",
                context_refs=[f"ctx-{group}"],
                priority=i,
            ))
    return tasks


def order_interleaved(tasks: list[Task]) -> list[Task]:
    """Worst case: round-robin across prefix groups."""
    groups: dict[str, list[Task]] = {}
    for t in tasks:
        g = t.task_id.split("-")[0]
        groups.setdefault(g, []).append(t)
    result = []
    max_len = max(len(v) for v in groups.values())
    group_keys = sorted(groups.keys())
    for i in range(max_len):
        for g in group_keys:
            if i < len(groups[g]):
                result.append(groups[g][i])
    return result


def order_critical_path(tasks: list[Task]) -> list[Task]:
    """Sort by critical_path_score (all 0 here) then priority, prefix-unaware."""
    return sorted(
        tasks,
        key=lambda t: (-t.critical_path_score, -t.priority, t.created_at, t.task_id),
    )


def order_affinity(tasks: list[Task], ctx: ContextManager, tracker: CacheHeatTracker) -> list[Task]:
    """Group by prefix heat, same-prefix together."""
    advisor = CacheAffinityAdvisor(ctx, tracker)
    return advisor.sort_by_affinity(list(tasks))


# ---------------------------------------------------------------------------
# Run one strategy
# ---------------------------------------------------------------------------

@dataclass
class StrategyResult:
    name: str
    calls: list[dict[str, Any]] = field(default_factory=list)
    order: list[str] = field(default_factory=list)
    total_latency: float = 0.0
    avg_latency: float = 0.0
    total_prompt_tokens: int = 0
    total_hit_tokens: int = 0
    total_miss_tokens: int = 0
    cache_hit_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "order": self.order,
            "total_latency_s": round(self.total_latency, 3),
            "avg_latency_s": round(self.avg_latency, 3),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_hit_tokens": self.total_hit_tokens,
            "total_miss_tokens": self.total_miss_tokens,
            "cache_hit_ratio": round(self.cache_hit_ratio, 4),
            "calls": self.calls,
        }


async def run_strategy(
    backend: VLLMBackend,
    ctx: ContextManager,
    tracker: CacheHeatTracker,
    ordered_tasks: list[Task],
    strategy_name: str,
    group_contexts: dict[str, str],
) -> StrategyResult:
    result = StrategyResult(name=strategy_name)
    result.order = [t.task_id for t in ordered_tasks]
    start_all = time.monotonic()

    for i, task in enumerate(ordered_tasks):
        group = task.task_id.split("-")[0]
        content = group_contexts[group]
        question = QUESTIONS[int(task.task_id.split("-")[1]) % len(QUESTIONS)]

        req = LLMRequest(
            agent_id=task.task_id,
            messages=[
                {"role": "system", "content": content},
                {"role": "user", "content": question},
            ],
            max_tokens=32,
        )
        resp = await backend.generate(req)
        m = resp.metrics
        assert m is not None

        prefix_id = ctx.resolve_prefix_id(task.context_refs)
        tracker.record_call(prefix_id, m)

        call = {
            "task_id": task.task_id,
            "group": group,
            "prompt_tokens": m.prompt_tokens,
            "hit": m.cache_hit_tokens,
            "miss": m.cache_miss_tokens,
            "ratio": round(m.cache_hit_ratio, 4),
            "latency_s": round(m.latency_secs, 3),
        }
        result.calls.append(call)
        print(
            f"  [{strategy_name:15s}] {task.task_id:10s} "
            f"prompt={m.prompt_tokens:5d} hit={m.cache_hit_tokens:5d} "
            f"miss={m.cache_miss_tokens:5d} ratio={m.cache_hit_ratio:5.1%} "
            f"latency={m.latency_secs:.3f}s"
        )

    elapsed = time.monotonic() - start_all
    result.total_latency = elapsed
    result.total_prompt_tokens = sum(c["prompt_tokens"] for c in result.calls)
    result.total_hit_tokens = sum(c["hit"] for c in result.calls)
    result.total_miss_tokens = sum(c["miss"] for c in result.calls)
    hm = result.total_hit_tokens + result.total_miss_tokens
    result.cache_hit_ratio = result.total_hit_tokens / hm if hm > 0 else 0.0
    result.avg_latency = elapsed / len(result.calls) if result.calls else 0.0
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    backend = VLLMBackend()
    all_tasks = build_tasks()

    results: list[StrategyResult] = []

    strategy_names = ["interleaved", "critical-path", "cache-affinity"]

    for name in strategy_names:
        nonce = uuid.uuid4().hex[:12]
        group_contexts = build_group_contexts(nonce)

        ctx = ContextManager()
        for group, content in group_contexts.items():
            ctx.add_shared_ref(f"ctx-{group}", content, priority=10)

        if name == "interleaved":
            ordered = order_interleaved(list(all_tasks))
        elif name == "critical-path":
            ordered = order_critical_path(list(all_tasks))
        else:
            tracker_pre = CacheHeatTracker(ttl=300.0)
            ordered = order_affinity(list(all_tasks), ctx, tracker_pre)

        print(f"\n{'='*70}")
        print(f"Strategy: {name}  (nonce={nonce})")
        print(f"Order: {' → '.join(t.task_id for t in ordered)}")
        print(f"{'='*70}")

        tracker = CacheHeatTracker(ttl=300.0)
        r = await run_strategy(backend, ctx, tracker, ordered, name, group_contexts)
        results.append(r)

        print(f"\n  Summary: hit_ratio={r.cache_hit_ratio:.1%}  "
              f"avg_latency={r.avg_latency:.3f}s  "
              f"total={r.total_latency:.1f}s")

    # Summary
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Strategy':20s} {'Hit Ratio':>10s} {'Avg Latency':>12s} {'Total Time':>11s} {'Hit Tokens':>11s}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*11} {'-'*11}")
    for r in results:
        print(f"  {r.name:20s} {r.cache_hit_ratio:>9.1%} {r.avg_latency:>11.3f}s {r.total_latency:>10.1f}s {r.total_hit_tokens:>11d}")

    best = max(results, key=lambda r: r.cache_hit_ratio)
    worst = min(results, key=lambda r: r.cache_hit_ratio)
    print(f"\n  Best:  {best.name} ({best.cache_hit_ratio:.1%})")
    print(f"  Worst: {worst.name} ({worst.cache_hit_ratio:.1%})")
    if worst.cache_hit_ratio > 0:
        print(f"  Improvement: {best.cache_hit_ratio / worst.cache_hit_ratio:.1f}x hit ratio")
    if best.avg_latency < worst.avg_latency:
        print(f"  Speedup: {worst.avg_latency / best.avg_latency:.2f}x avg latency")

    # Save
    out_dir = Path(__file__).parent.parent / "runs"
    out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    report = {
        "experiment": "scheduling_strategy_comparison",
        "description": "FIFO-interleaved vs Critical-Path vs Cache-Affinity on local vLLM",
        "timestamp": ts,
        "environment": {
            "backend": "vllm",
            "base_url": "http://localhost:8000",
            "model": "qwen2.5-7b-instruct",
            "prefix_groups": len(GROUP_SPECS),
            "tasks_per_group": TASKS_PER_GROUP,
            "total_tasks": len(GROUP_SPECS) * TASKS_PER_GROUP,
            "padding_lines_per_group": PADDING_LINES,
            "context_chars_per_group": {g: len(_build_group_context(*GROUP_SPECS[g])) for g in GROUP_SPECS},
            "isolation": "per-strategy unique nonce prevents cross-strategy cache reuse",
        },
        "results": {r.name: r.to_dict() for r in results},
        "comparison": {
            "best_strategy": best.name,
            "best_hit_ratio": round(best.cache_hit_ratio, 4),
            "worst_strategy": worst.name,
            "worst_hit_ratio": round(worst.cache_hit_ratio, 4),
        },
    }
    report_path = out_dir / f"scheduling_benchmark_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
