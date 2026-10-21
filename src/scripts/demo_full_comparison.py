"""
Full Comparison Demo: AgentOS Runtime vs Baseline (Sequential, No-Recovery)

Runs a realistic 8-agent software-engineering DAG two ways:

  BASELINE  — agents run SEQUENTIALLY in topological order; each inlines the
              full shared spec with its ROLE-SPECIFIC text FIRST (so system
              prefixes differ across agents → prompt-cache misses every call);
              an injected fault makes the run record as incomplete/failed.

  RUNTIME   — same agents run over the existing Runtime engine with DAG
              parallelism, shared ContextRef + PrefixCanonicalizer (stable
              long shared prefix → cache hits), and the injected fault is
              recovered via ContextRetryPolicy so the DAG still completes.

Usage:
  .venv/bin/python scripts/demo_full_comparison.py --backend mock
  .venv/bin/python scripts/demo_full_comparison.py --backend cache-sim
  .venv/bin/python scripts/demo_full_comparison.py --backend deepseek-sim
  export DEEPSEEK_API_KEY=<key>
  .venv/bin/python scripts/demo_full_comparison.py --backend deepseek
"""
from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentos.benchmarks.runner import _MetricsCollector, BenchmarkResult
from agentos.context.manager import ContextManager
from agentos.context.prefix import ContextPriority
from agentos.core.task import Task, TaskDAG, TaskStatus
from agentos.events.log import EventLog
from agentos.fault.injector import FaultInjectionPlan
from agentos.fault.recovery import ContextRetryPolicy
from agentos.llm.base import LLMBackend, LLMRequest, LLMResponse
from agentos.llm.factory import create_backend
from agentos.runtime.engine import AgentHandler, Runtime

# ---------------------------------------------------------------------------
# Long shared project spec (~700 tokens).  Stable across all calls → cache-
# friendly when placed first in the system prompt.  Intentionally much longer
# than the SOFTWARE_ENG scenario's built-in PROJECT_CONTEXT.
# ---------------------------------------------------------------------------
LONG_PROJECT_SPEC = """\
=== Project Specification: NebulaCommerce Microservices Platform ===

Overview
--------
NebulaCommerce is a cloud-native, multi-tenant e-commerce microservices platform
targeting small-to-medium online retailers.  It replaces a legacy monolith with
independently deployable Python services communicating over gRPC and an async
event bus (Kafka 3.6).  The platform is designed for horizontal scaling,
99.9% SLA, and zero-downtime deployments via canary releases.

Repository Layout
-----------------
  services/
    catalog/       — Product catalogue, search (Elasticsearch 8), media CDN hooks
    inventory/     — Stock levels, warehouse allocation, reservation TTL
    pricing/       — Dynamic pricing rules, discount engine, currency conversion
    cart/          — Session-backed shopping cart, merge on login
    checkout/      — Order orchestration (Saga pattern), payment gateway adapter
    identity/      — OAuth 2.1 + OIDC, JWT RS256, role-based access control
    notifications/ — Email (SES), SMS (Twilio), push (FCM), in-app WebSocket
    analytics/     — ClickHouse ingest, real-time dashboards, funnel reports
  infra/
    terraform/     — AWS EKS, RDS (Postgres 15), ElastiCache (Redis 7), MSK
    helm/          — Per-service Helm charts, ArgoCD app-of-apps pattern
    monitoring/    — Prometheus, Grafana, Loki, OpenTelemetry collector
  shared/
    proto/         — Protobuf definitions, buf.build registry
    libs/          — Python shared: auth middleware, event schema, pagination

Technology Stack
----------------
  Language : Python 3.12 (services), TypeScript 5 (storefront BFF)
  Frameworks: FastAPI 0.111 (HTTP), Strawberry (GraphQL), gRPC 1.64
  ORM      : SQLAlchemy 2.0 async, Alembic migrations
  Messaging: aiokafka producers/consumers, CloudEvents schema registry
  Caching  : aioredis 2.x, write-through cache for catalogue reads
  Testing  : pytest-asyncio, factory_boy, Testcontainers (Postgres + Kafka)
  CI/CD    : GitHub Actions → ECR → ArgoCD progressive delivery
  Secrets  : AWS Secrets Manager, HashiCorp Vault (on-prem fallback)

Architecture Decisions (ADRs in docs/adr/)
------------------------------------------
  ADR-001: Saga orchestration in checkout (vs. choreography) for auditability
  ADR-003: CQRS in analytics service — write to Kafka, project to ClickHouse
  ADR-007: Single Postgres cluster with schema-per-tenant (max 500 tenants)
  ADR-012: Cache-aside pattern for catalogue; TTL 5 min, invalidated on write
  ADR-015: All inter-service calls carry trace-id, span-id (OpenTelemetry W3C)
  ADR-019: API versioning via URL path prefix (/v1/, /v2/) not content negotiation
  ADR-023: Retry with exponential back-off (max 3, jitter) in gRPC interceptors

Current Sprint Objective: Wishlist Feature
------------------------------------------
Add a Wishlist capability across the platform.  Scope:

  1. Data Model — wishlist_items table (id, tenant_id, user_id, product_id,
     added_at, notes, notified_at).  Composite unique index (user_id, product_id).
     Alembic migration with rollback.

  2. Catalogue integration — WishlistService.add() calls catalogue gRPC to
     validate product_id; raises ItemNotFoundError on miss.

  3. API Endpoints (identity service auth required, JWT scope wishlist:write):
       POST   /v1/wishlists/{user_id}/items        — add item
       DELETE /v1/wishlists/{user_id}/items/{id}   — remove item
       GET    /v1/wishlists/{user_id}              — list items (paginated)
       GET    /v1/wishlists/{user_id}/items/{id}   — get single item

  4. Notification trigger — pricing service emits PriceDropEvent on Kafka
     topic price.events; notifications service subscribes, queries wishlist
     service, fans out email + push per user.

  5. Tests — unit tests for WishlistService (mock gRPC, mock Redis), integration
     test with Testcontainers, load test (locust) at 2000 RPS sustained.

  6. Security — OWASP Top-10 review; enforce tenant isolation at ORM query level;
     rate-limit add-to-wishlist at 100 req/min per user (Redis sliding window).

  7. Documentation — OpenAPI 3.1 spec, README with sequence diagrams (Mermaid),
     runbook for on-call (alert thresholds, rollback steps).

Non-Functional Requirements
---------------------------
  Latency  : P99 < 80 ms for wishlist GET (warm cache), < 200 ms cold
  Throughput: sustain 5000 concurrent users per tenant
  Observability: every endpoint emits latency histogram, error counter, trace
  Compliance: GDPR — PII in wishlist (user_id) must be purgeable on request
  Disaster recovery: RPO 1 h, RTO 4 h; wishlist data replicated cross-AZ

Coding Standards
----------------
  - Type hints everywhere; mypy --strict on all service modules
  - Black formatter (line-length 100), ruff linter, pre-commit hooks
  - Conventional Commits; PR must have ≥ 1 reviewer approval + CI green
  - All async I/O; no blocking calls in event loop (enforced by pytest-asyncio
    strict mode with asyncio_mode = "auto")
  - Tests must reach 90% branch coverage; failing coverage gate blocks merge
"""

# Per-agent role-specific preamble (shorter than shared spec; prepended FIRST
# in baseline to break prefix alignment — this is the key fairness constraint
# that makes baseline's system prompts different per agent).
_ROLE_PREAMBLES: dict[str, str] = {
    "architect": (
        "ROLE: System Architect — your job is to produce the high-level design "
        "for the Wishlist feature, define component interfaces, and write the "
        "integration plan across NebulaCommerce microservices.\n\n"
    ),
    "db_designer": (
        "ROLE: Database Designer — your job is to design the wishlist_items "
        "SQLAlchemy model, write the Alembic migration (upgrade + downgrade), "
        "and add composite indexes for query performance.\n\n"
    ),
    "api_designer": (
        "ROLE: API Designer — your job is to draft the OpenAPI 3.1 spec for "
        "all four Wishlist endpoints, define request/response schemas, "
        "and add validation rules and error response shapes.\n\n"
    ),
    "backend_coder": (
        "ROLE: Backend Engineer — implement WishlistService with full CRUD, "
        "gRPC catalogue validation, Redis caching, and FastAPI routers wired "
        "to the service layer with dependency injection.\n\n"
    ),
    "test_writer": (
        "ROLE: Test Engineer — write pytest-asyncio unit tests for WishlistService "
        "(mocking gRPC and Redis), integration tests with Testcontainers, "
        "and edge-case coverage for quota limits and duplicate adds.\n\n"
    ),
    "security_reviewer": (
        "ROLE: Security Reviewer — perform OWASP Top-10 analysis on the Wishlist "
        "feature, verify tenant isolation in ORM queries, check JWT scope "
        "enforcement, and validate the Redis rate-limiter implementation.\n\n"
    ),
    "doc_writer": (
        "ROLE: Technical Writer — produce the Wishlist README with Mermaid sequence "
        "diagrams, update the OpenAPI spec with examples, and write the on-call "
        "runbook with alert thresholds and rollback steps.\n\n"
    ),
    "integrator": (
        "ROLE: Integration Lead — write the final integration report: summarise "
        "all sub-team outputs, flag open issues, confirm that the feature is "
        "production-ready, and list post-merge monitoring tasks.\n\n"
    ),
}

# Topological order for sequential baseline execution
_TOPO_ORDER = [
    "architect",
    "db_designer",
    "api_designer",
    "backend_coder",
    "test_writer",
    "security_reviewer",
    "doc_writer",
    "integrator",
]

# Which task to inject the fault on (same for both sides)
_FAULT_TASK = "security_reviewer"

# Cap completion tokens for both sides (we care about PROMPT cache, not outputs)
_MAX_TOKENS = 200

# ---------------------------------------------------------------------------
# Thin backend wrapper that clamps max_tokens on every request
# ---------------------------------------------------------------------------

class MaxTokensCappedBackend:
    """Wraps any LLMBackend and clamps request.max_tokens to a ceiling."""

    def __init__(self, backend: LLMBackend, max_tokens: int = _MAX_TOKENS) -> None:
        self._backend = backend
        self._max_tokens = max_tokens

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if request.max_tokens > self._max_tokens:
            request = LLMRequest(
                agent_id=request.agent_id,
                messages=request.messages,
                model=request.model,
                backend=request.backend,
                temperature=request.temperature,
                max_tokens=self._max_tokens,
                task_id=request.task_id,
                trace_id=request.trace_id,
                context_refs=request.context_refs,
                prefix_id=request.prefix_id,
                metadata=request.metadata,
                created_at=request.created_at,
            )
        return await self._backend.generate(request)


# ---------------------------------------------------------------------------
# Simple agent handlers for the demo DAG
# (one LLM call per agent — enough to prove cache behaviour)
# ---------------------------------------------------------------------------

class _SingleCallHandler(AgentHandler):
    def __init__(self, role: str, user_prompt: str) -> None:
        self._role = role
        self._user_prompt = user_prompt

    async def run(self, ctx: AgentHandler.Context) -> str:
        resp = await ctx.llm_call([{"role": "user", "content": self._user_prompt}])
        # Share the response so downstream agents get it via ContextManager
        ctx.context_manager.add_shared(f"{self._role}-output", resp.content)
        return resp.content


_USER_PROMPTS: dict[str, str] = {
    "architect": (
        "Design the Wishlist feature for NebulaCommerce: data model, API endpoints, "
        "service layer interfaces, and integration points.  Output a structured "
        "technical design document (< 200 words)."
    ),
    "db_designer": (
        "Based on the architect's design, create the SQLAlchemy 2.0 async model for "
        "wishlist_items and write the Alembic migration (upgrade + downgrade).  "
        "Keep your answer under 200 words."
    ),
    "api_designer": (
        "Draft the OpenAPI 3.1 spec for all four Wishlist endpoints.  Define "
        "request/response schemas, error shapes, and auth requirements.  "
        "Keep your answer under 200 words."
    ),
    "backend_coder": (
        "Implement WishlistService with add, remove, list, and get methods.  "
        "Include FastAPI router, Redis caching, and gRPC catalogue validation.  "
        "Keep your answer under 200 words."
    ),
    "test_writer": (
        "Write pytest-asyncio unit tests for WishlistService covering add, remove, "
        "list, duplicate-add error, and item-not-found.  Keep your answer under 200 words."
    ),
    "security_reviewer": (
        "Perform an OWASP Top-10 review of the Wishlist feature.  Check tenant isolation "
        "in ORM queries, JWT scope enforcement, and the Redis rate-limiter.  "
        "Keep your answer under 200 words."
    ),
    "doc_writer": (
        "Write the Wishlist README with a Mermaid sequence diagram for the add-item flow, "
        "update the OpenAPI spec with curl examples.  Keep your answer under 200 words."
    ),
    "integrator": (
        "Write the final integration report for the Wishlist feature: summarise all "
        "sub-team outputs, flag open issues, confirm production-readiness.  "
        "Keep your answer under 200 words."
    ),
}


def _build_handlers() -> dict[str, AgentHandler]:
    return {
        role: _SingleCallHandler(role, _USER_PROMPTS[role])
        for role in _TOPO_ORDER
    }


# ---------------------------------------------------------------------------
# DAG builder
# ---------------------------------------------------------------------------

def _build_dag() -> TaskDAG:
    dag = TaskDAG()
    dag.add_task(Task(task_id="architect",        name="architect",        agent_role="architect"))
    dag.add_task(Task(task_id="db_designer",      name="db_designer",      agent_role="db_designer",      dependencies=["architect"]))
    dag.add_task(Task(task_id="api_designer",     name="api_designer",     agent_role="api_designer",     dependencies=["architect"]))
    dag.add_task(Task(task_id="backend_coder",    name="backend_coder",    agent_role="backend_coder",    dependencies=["db_designer", "api_designer"]))
    dag.add_task(Task(task_id="test_writer",      name="test_writer",      agent_role="test_writer",      dependencies=["backend_coder"]))
    dag.add_task(Task(task_id="security_reviewer",name="security_reviewer",agent_role="security_reviewer",dependencies=["backend_coder"]))
    dag.add_task(Task(task_id="doc_writer",       name="doc_writer",       agent_role="doc_writer",       dependencies=["backend_coder"]))
    dag.add_task(Task(task_id="integrator",       name="integrator",       agent_role="integrator",       dependencies=["test_writer", "security_reviewer", "doc_writer"]))
    return dag


# ---------------------------------------------------------------------------
# BASELINE: sequential execution, role-first prompts, no fault recovery
# ---------------------------------------------------------------------------

@dataclass
class BaselineResult:
    makespan_secs: float
    total_prompt_tokens: int
    total_completion_tokens: int
    cache_hit_tokens: int
    cache_miss_tokens: int
    cache_hit_ratio: float
    llm_calls: int
    tasks_completed: int
    tasks_total: int
    fault_task: str
    fault_outcome: str  # "failed/incomplete" or "completed"
    per_call: list[dict[str, Any]] = field(default_factory=list)


async def run_baseline(backend: LLMBackend) -> BaselineResult:
    """Sequential baseline: role-first system prompt, no shared context, no recovery."""
    collector = _MetricsCollector(backend)
    tasks_total = len(_TOPO_ORDER)
    tasks_completed = 0
    fault_outcome = "not triggered"
    t0 = time.monotonic()

    for role in _TOPO_ORDER:
        # Role-specific text FIRST → prefix differs across agents → cache miss
        role_preamble = _ROLE_PREAMBLES[role]
        system_content = role_preamble + LONG_PROJECT_SPEC
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": _USER_PROMPTS[role]},
        ]
        req = LLMRequest(
            agent_id=f"baseline-{role}",
            messages=messages,
            max_tokens=_MAX_TOKENS,
        )

        # Simulate the injected fault: skip the LLM call, record as failed
        if role == _FAULT_TASK:
            fault_outcome = "FAILED (no recovery) — run incomplete"
            # Record 0 tokens for the failed task so the table is honest
            break

        try:
            resp = await collector.generate(req)
            tasks_completed += 1
        except Exception:
            fault_outcome = "FAILED (no recovery) — run incomplete"
            break

    makespan = time.monotonic() - t0

    total_prompt = sum(c.get("prompt_tokens", 0) for c in collector.calls)
    total_completion = sum(c.get("completion_tokens", 0) for c in collector.calls)
    total_hit = sum(c.get("cache_hit_tokens", 0) for c in collector.calls)
    total_miss = sum(c.get("cache_miss_tokens", 0) for c in collector.calls)
    ratio = total_hit / (total_hit + total_miss) if (total_hit + total_miss) > 0 else 0.0

    return BaselineResult(
        makespan_secs=makespan,
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
        cache_hit_tokens=total_hit,
        cache_miss_tokens=total_miss,
        cache_hit_ratio=ratio,
        llm_calls=len(collector.calls),
        tasks_completed=tasks_completed,
        tasks_total=tasks_total,
        fault_task=_FAULT_TASK,
        fault_outcome=fault_outcome,
        per_call=collector.calls,
    )


# ---------------------------------------------------------------------------
# RUNTIME: DAG parallel + shared ContextRef + fault recovery
# ---------------------------------------------------------------------------

@dataclass
class RuntimeResult:
    makespan_secs: float
    total_prompt_tokens: int
    total_completion_tokens: int
    cache_hit_tokens: int
    cache_miss_tokens: int
    cache_hit_ratio: float
    llm_calls: int
    tasks_completed: int
    tasks_total: int
    fault_task: str
    fault_outcome: str
    event_count: int
    event_log_path: str
    dashboard_path: str
    per_call: list[dict[str, Any]] = field(default_factory=list)


async def run_runtime(backend: LLMBackend, run_dir: Path) -> RuntimeResult:
    """Runtime: DAG parallel, shared ContextRef prefix, fault recovery."""
    collector = _MetricsCollector(backend)

    # Shared context as a ContextRef — stable long prefix for all agents
    ctx_mgr = ContextManager()
    ctx_mgr.add_shared_ref(
        "nebula-spec",
        LONG_PROJECT_SPEC,
        priority=ContextPriority.TASK_BACKGROUND,
    )

    dag = _build_dag()
    log_path = run_dir / "runtime_events.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    fault_plan = FaultInjectionPlan({_FAULT_TASK: "kill"})
    recovery = ContextRetryPolicy(max_retries=1)

    rt = Runtime(
        llm_backend=collector,
        handlers=_build_handlers(),
        context_manager=ctx_mgr,
        event_log=EventLog(log_path),
        fault_injection=fault_plan,
        recovery_policy=recovery,
    )

    t0 = time.monotonic()
    fault_outcome = "unknown"
    try:
        results = await rt.run(dag)
        tasks_completed = sum(
            1 for t in dag.tasks.values() if t.status == TaskStatus.COMPLETED
        )
        # Check if fault task recovered and completed
        ft = dag.tasks.get(_FAULT_TASK)
        if ft is not None and ft.status == TaskStatus.COMPLETED:
            fault_outcome = "RECOVERED & COMPLETED (context-retry)"
        else:
            fault_outcome = "completed (no fault triggered)"
    except Exception as exc:
        results = {}
        tasks_completed = sum(
            1 for t in dag.tasks.values() if t.status == TaskStatus.COMPLETED
        )
        fault_outcome = f"FAILED: {exc}"

    makespan = time.monotonic() - t0

    total_prompt = sum(c.get("prompt_tokens", 0) for c in collector.calls)
    total_completion = sum(c.get("completion_tokens", 0) for c in collector.calls)
    total_hit = sum(c.get("cache_hit_tokens", 0) for c in collector.calls)
    total_miss = sum(c.get("cache_miss_tokens", 0) for c in collector.calls)
    ratio = total_hit / (total_hit + total_miss) if (total_hit + total_miss) > 0 else 0.0

    tasks_total = len(dag.tasks)

    # Generate dashboard
    from agentos.dashboard.builder import build_dashboard
    dashboard_path = run_dir / "dashboard.html"
    build_dashboard(log_paths=[log_path], output_path=dashboard_path)

    return RuntimeResult(
        makespan_secs=makespan,
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
        cache_hit_tokens=total_hit,
        cache_miss_tokens=total_miss,
        cache_hit_ratio=ratio,
        llm_calls=len(collector.calls),
        tasks_completed=tasks_completed,
        tasks_total=tasks_total,
        fault_task=_FAULT_TASK,
        fault_outcome=fault_outcome,
        event_count=len(rt.events),
        event_log_path=str(log_path),
        dashboard_path=str(dashboard_path),
        per_call=collector.calls,
    )


# ---------------------------------------------------------------------------
# Pretty comparison table
# ---------------------------------------------------------------------------

def print_comparison(b: BaselineResult, r: RuntimeResult, backend_name: str) -> None:
    W = 70
    print()
    print("=" * W)
    print(f"  AgentOS Runtime vs Baseline — Full Comparison")
    print(f"  Backend: {backend_name}")
    print("=" * W)
    fmt = "{:<38} {:>13} {:>13}"
    hdr = fmt.format("Metric", "BASELINE", "RUNTIME")
    sep = "-" * W
    print(hdr)
    print(sep)

    def row(label: str, bval: Any, rval: Any, *, better: str = "lower") -> None:
        """Print a comparison row.  better='lower'|'higher'|'none'."""
        bstr = str(bval)
        rstr = str(rval)
        print(fmt.format(label, bstr, rstr))

    row("Makespan (wall-clock secs)",
        f"{b.makespan_secs:.3f}s", f"{r.makespan_secs:.3f}s")
    row("LLM calls",
        b.llm_calls, r.llm_calls, better="none")
    row("Total prompt tokens",
        b.total_prompt_tokens, r.total_prompt_tokens)
    row("Total completion tokens",
        b.total_completion_tokens, r.total_completion_tokens, better="none")
    row("Cache hit tokens",
        b.cache_hit_tokens, r.cache_hit_tokens, better="higher")
    row("Cache miss tokens",
        b.cache_miss_tokens, r.cache_miss_tokens)
    row("Cache hit ratio",
        f"{b.cache_hit_ratio:.1%}", f"{r.cache_hit_ratio:.1%}", better="higher")
    row("Tasks completed / total",
        f"{b.tasks_completed}/{b.tasks_total}", f"{r.tasks_completed}/{r.tasks_total}")
    row("Fault task",
        b.fault_task, r.fault_task, better="none")
    print(sep)

    # Fault outcome — full width
    print(f"{'Baseline fault outcome:':<38} {b.fault_outcome}")
    print(f"{'Runtime fault outcome:':<38} {r.fault_outcome}")

    print(sep)

    # Highlight deltas
    makespan_delta = r.makespan_secs - b.makespan_secs
    prompt_delta   = r.total_prompt_tokens - b.total_prompt_tokens
    cache_delta    = r.cache_hit_ratio - b.cache_hit_ratio
    print()
    print("  Deltas (Runtime vs Baseline):")
    print(f"    Makespan:        {makespan_delta:+.3f}s  (negative = runtime faster)")
    print(f"    Prompt tokens:   {prompt_delta:+d}  (negative = runtime saves tokens)")
    print(f"    Cache hit ratio: {cache_delta:+.1%}  (positive = runtime better)")
    print()
    if r.event_count:
        print(f"  Runtime events:    {r.event_count}")
    print(f"  Runtime EventLog:  {r.event_log_path}")
    print(f"  Runtime Dashboard: {r.dashboard_path}")
    print("=" * W)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def _async_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="demo_full_comparison",
        description="AgentOS Runtime vs Baseline full comparison demo",
    )
    parser.add_argument(
        "--backend",
        default="mock",
        choices=["mock", "cache-sim", "deepseek-sim", "vllm-sim", "deepseek"],
        help="LLM backend to use",
    )
    parser.add_argument(
        "--out-dir",
        default="runs/full_comparison_demo",
        help="Directory for runtime event logs and dashboard",
    )
    parser.add_argument(
        "--latency",
        type=float,
        default=0.05,
        help="Per-call latency in seconds for mock/sim backends (helps show makespan diff)",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[demo] Building backend: {args.backend!r} (latency={args.latency}s)")

    # DeepSeek real backend doesn't accept a latency kwarg
    if args.backend == "deepseek":
        raw_backend = create_backend(args.backend)
    else:
        raw_backend = create_backend(args.backend, latency=args.latency)
    # Wrap to clamp max_tokens
    capped = MaxTokensCappedBackend(raw_backend, max_tokens=_MAX_TOKENS)

    print("[demo] Running BASELINE (sequential, role-first prompts, no recovery) ...")
    baseline = await run_baseline(capped)
    print(f"       Done in {baseline.makespan_secs:.3f}s — {baseline.tasks_completed}/{baseline.tasks_total} tasks")

    print("[demo] Running RUNTIME (DAG parallel, shared ContextRef, fault recovery) ...")
    runtime = await run_runtime(capped, run_dir)
    print(f"       Done in {runtime.makespan_secs:.3f}s — {runtime.tasks_completed}/{runtime.tasks_total} tasks")

    print_comparison(baseline, runtime, args.backend)
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
