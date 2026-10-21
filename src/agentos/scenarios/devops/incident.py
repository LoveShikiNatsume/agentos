"""DevOps Incident Response scenario.

DAG: monitor → diagnose → fix → verify + independent postmortem
Demonstrates Runtime capabilities in a non-coding context:
- Shared incident context via ContextRef
- Fault injection on 'diagnose' (simulates tool failure)
- ContextRetry recovery preserving diagnostic state
- Independent postmortem branch unaffected by fault

Usage:
    .venv/bin/python -m agentos.scenarios.devops.incident
    .venv/bin/python -m agentos.scenarios.devops.incident --log runs/devops/events.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any, Sequence

from agentos.context.manager import ContextManager
from agentos.core.task import Task, TaskDAG, TaskStatus
from agentos.events.log import EventLog
from agentos.fault.injector import FaultInjectionPlan
from agentos.fault.recovery import ContextRetryPolicy
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import AgentHandler, Runtime

INCIDENT_CONTEXT = (
    "Production alert: API gateway latency spiked from 50ms to 2000ms at "
    "14:32 UTC. Affected services: user-auth, payment-service, order-service. "
    "Error rate increased from 0.1% to 15%. No recent deployments in the last "
    "24 hours. Infrastructure: 3-node Kubernetes cluster on openEuler 24.03."
)

INFRA_SUMMARY = (
    "Cluster: k8s v1.28, 3 worker nodes (8 vCPU, 32GB each). "
    "Database: PostgreSQL 15 primary + 2 read replicas. "
    "Cache: Redis 7 cluster (3 shards). Load balancer: Nginx ingress. "
    "Monitoring: Prometheus + Grafana. Logs: Loki."
)


class MonitorHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        ctx.context_manager.add_shared(
            "alert-summary",
            "API gateway p99 latency: 2000ms (normal: 50ms). "
            "Error rate: 15%. Top errors: 503 Service Unavailable. "
            "Affected pods: user-auth-7f8d9, payment-svc-3a2b1.",
        )
        resp = await ctx.llm_call([
            {"role": "user", "content": "Analyze the monitoring data and identify the incident scope."},
        ])
        return resp.content


class DiagnoseHandler(AgentHandler):
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, ctx: AgentHandler.Context) -> str:
        self.calls += 1
        ctx.context_manager.add_private(
            ctx.acb.agent_id,
            f"[attempt {self.calls}] kubectl logs: OOMKilled in redis-shard-2. "
            f"Memory usage 31.8/32GB. Connection pool exhausted.",
        )
        resp = await ctx.llm_call([
            {"role": "user", "content": "Diagnose the root cause using logs, metrics, and cluster state."},
        ])
        ctx.submit_task(Task(
            task_id="db_check",
            name="db_check",
            agent_role="db_inspector",
            dependencies=["diagnose"],
        ))
        return resp.content


class DbInspectorHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        resp = await ctx.llm_call([
            {"role": "user", "content": "Inspect database connection pools, query latency, and replica lag."},
        ])
        return resp.content


class FixHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        resp = await ctx.llm_call([
            {"role": "user", "content": "Apply the fix: restart affected pods, increase Redis memory limit, flush stale connections."},
        ])
        return resp.content


class VerifyHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        resp = await ctx.llm_call([
            {"role": "user", "content": "Verify the fix: check latency, error rate, and pod health."},
        ])
        return resp.content


class PostmortemHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        resp = await ctx.llm_call([
            {"role": "user", "content": "Write a postmortem: timeline, root cause, impact, action items."},
        ])
        return resp.content


def build_devops_dag() -> TaskDAG:
    dag = TaskDAG()
    dag.add_task(Task(task_id="monitor", name="monitor", agent_role="monitor"))
    dag.add_task(Task(task_id="diagnose", name="diagnose", agent_role="diagnoser", dependencies=["monitor"]))
    dag.add_task(Task(task_id="fix", name="fix", agent_role="fixer", dependencies=["diagnose"]))
    dag.add_task(Task(task_id="verify", name="verify", agent_role="verifier", dependencies=["fix"]))
    dag.add_task(Task(task_id="postmortem", name="postmortem", agent_role="postmortem", dependencies=["monitor"]))
    return dag


async def run_devops_incident(
    *,
    log_path: str | Path = "runs/devops/events.jsonl",
    inject_fault: bool = True,
) -> tuple[dict[str, Any], Runtime, TaskDAG]:
    ctx_mgr = ContextManager()
    ctx_mgr.add_shared("incident-context", INCIDENT_CONTEXT)
    ctx_mgr.add_shared("infra-summary", INFRA_SUMMARY)

    dag = build_devops_dag()
    log = Path(log_path)

    fault_plan = FaultInjectionPlan({"diagnose": "kill"}) if inject_fault else None
    recovery = ContextRetryPolicy(max_retries=1) if inject_fault else None

    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={
            "monitor": MonitorHandler(),
            "diagnoser": DiagnoseHandler(),
            "fixer": FixHandler(),
            "verifier": VerifyHandler(),
            "postmortem": PostmortemHandler(),
            "db_inspector": DbInspectorHandler(),
        },
        context_manager=ctx_mgr,
        event_log=EventLog(log),
        fault_injection=fault_plan,
        recovery_policy=recovery,
    )

    results = await rt.run(dag)
    return results, rt, dag


def print_report(results: dict[str, Any], rt: Runtime, dag: TaskDAG) -> bool:
    print("=" * 64)
    print("  AgentOS-Runtime DevOps Incident Response Scenario")
    print("=" * 64)
    print()

    print("[DAG Task Results]")
    for task_id in ("monitor", "diagnose", "db_check", "fix", "verify", "postmortem"):
        if task_id not in dag.tasks:
            continue
        task = dag.tasks[task_id]
        result = results.get(task_id, "-")
        result_str = str(result)[:60] if result else "-"
        print(f"  {task_id:<12} status={task.status.value:<10} result={result_str}")
    print()

    print("[Event Summary]")
    event_types: dict[str, int] = {}
    for e in rt.events:
        event_types[e.event_type] = event_types.get(e.event_type, 0) + 1
    for etype, count in sorted(event_types.items()):
        print(f"  {etype:<36} x{count}")
    print()

    print("[Fault & Recovery Chain]")
    for e in rt.events:
        if e.event_type.startswith("fault."):
            agent = e.agent_id or "-"
            print(f"  {e.event_type:<36} agent={agent:<16} {e.data}")
    print()

    checks = [
        ("all tasks completed", all(
            dag.tasks[t].status == TaskStatus.COMPLETED
            for t in ("monitor", "diagnose", "fix", "verify", "postmortem")
        )),
        ("db_check dynamically submitted", any(
            e.event_type == "task.dynamic_submitted" and e.task_id == "db_check"
            for e in rt.events
        )),
        ("db_check completed", dag.tasks.get("db_check") is not None
            and dag.tasks["db_check"].status == TaskStatus.COMPLETED),
        ("fault injected on diagnose", any(
            e.event_type == "fault.injected" and e.task_id == "diagnose" for e in rt.events
        )),
        ("context snapshot saved", any(
            e.event_type == "fault.context_snapshot_saved" for e in rt.events
        )),
        ("context restored after failure", any(
            e.event_type == "fault.context_restored" for e in rt.events
        )),
        ("recovery completed (context_retry)", any(
            e.event_type == "fault.recovery_completed"
            and e.data.get("action") == "context_retry"
            for e in rt.events
        )),
        ("postmortem branch unaffected", dag.tasks["postmortem"].status == TaskStatus.COMPLETED),
        ("DAG fully complete", dag.is_complete()),
    ]

    print("[Checks]")
    ok = True
    for label, passed in checks:
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {label}")
        if not passed:
            ok = False
    print()

    restore = [e for e in rt.events if e.event_type == "fault.context_restored"]
    if restore:
        print("[Partial Recovery Metrics]")
        print(f"  Private refs restored:  {restore[0].data.get('context_refs_restored', 0)}")
        print(f"  Private tokens restored (estimate):  {restore[0].data.get('estimated_private_tokens_restored', 0)}")
        print()

    return ok


async def _async_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="devops-incident")
    parser.add_argument("--log", default="runs/devops/events.jsonl")
    parser.add_argument("--no-fault", action="store_true")
    args = parser.parse_args(argv)

    log_path = Path(args.log)
    if log_path.exists():
        log_path.unlink()

    results, rt, dag = await run_devops_incident(
        log_path=log_path,
        inject_fault=not args.no_fault,
    )
    ok = print_report(results, rt, dag)

    print(f"[EventLog] Written to {log_path}")
    status = "ALL PASSED" if ok else "SOME FAILED"
    print(f"Result: {status}")
    return 0 if ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(_async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
