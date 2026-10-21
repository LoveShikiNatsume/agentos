"""Network workload: simulates distributed agent communication with high message-bus traffic."""
from __future__ import annotations

from pathlib import Path

from agentos.benchmarks.cache_sim import CacheSimulatingMockBackend
from agentos.core.task import Task, TaskDAG
from agentos.runtime.engine import AgentHandler

from scripts.workloads._base import WorkloadResult, run_workload

WORKLOAD_TYPE = "network"

SYSTEM_PREFIX = (
    "You are a network operations agent in AgentOS-Runtime. "
    "The system manages distributed agent coordination via MessageBus IPC. "
    "Shared network topology and routing context is pooled across agents. "
    "Each agent handles one stage of incident response."
)

NETWORK_CONTEXT = (
    "Incident: API gateway latency spike detected at 14:23 UTC. "
    "Topology: 3 availability zones, 12 service mesh nodes, 4 GPU clusters. "
    "SLO: p99 < 200ms, current p99 = 1.2s. "
    "Runbook: detect → diagnose → isolate → mitigate → verify → postmortem."
)


def _build_dag() -> TaskDAG:
    dag = TaskDAG()
    dag.add_task(Task(
        task_id="detect", name="detect", agent_role="detector",
        workload_hint="network", io_profile="network_heavy", expected_parallelism=4,
        sla_class="critical", deadline_ms=800,
    ))
    dag.add_task(Task(
        task_id="diagnose", name="diagnose", agent_role="diagnostician", dependencies=["detect"],
        workload_hint="network", io_profile="network_heavy", expected_parallelism=4,
        sla_class="critical", deadline_ms=1000,
    ))
    dag.add_task(Task(
        task_id="isolate", name="isolate", agent_role="isolator", dependencies=["diagnose"],
        workload_hint="network", io_profile="network_heavy", expected_parallelism=4,
        sla_class="critical", deadline_ms=1200,
    ))
    dag.add_task(Task(
        task_id="mitigate", name="mitigate", agent_role="mitigator", dependencies=["isolate"],
        workload_hint="network", io_profile="network_heavy", expected_parallelism=4,
        sla_class="interactive", deadline_ms=1500,
    ))
    dag.add_task(Task(
        task_id="verify", name="verify", agent_role="verifier", dependencies=["mitigate"],
        workload_hint="network", io_profile="network_heavy", expected_parallelism=2,
        sla_class="interactive", deadline_ms=2000,
    ))
    dag.add_task(Task(
        task_id="postmortem", name="postmortem", agent_role="analyst", dependencies=["verify"],
        workload_hint="network", io_profile="network_heavy", expected_parallelism=1,
        sla_class="batch",
    ))
    dag.add_task(Task(
        task_id="alert", name="alert", agent_role="alerter", dependencies=["detect"],
        workload_hint="network", io_profile="network_heavy", expected_parallelism=4,
        sla_class="critical", deadline_ms=500,
    ))
    return dag


class _BaselineHandler(AgentHandler):
    def __init__(self, role: str) -> None:
        self._role = role

    async def run(self, ctx: AgentHandler.Context) -> str:
        resp = await ctx.llm_call([
            {"role": "system", "content": f"{SYSTEM_PREFIX}\n\n{NETWORK_CONTEXT}\n\nYou are a {self._role}."},
            {"role": "user", "content": f"Execute {self._role} work for the incident response."},
        ])
        return resp.content


class _RuntimeHandler(AgentHandler):
    def __init__(self, role: str) -> None:
        self._role = role

    async def run(self, ctx: AgentHandler.Context) -> str:
        resp = await ctx.llm_call([
            {"role": "user", "content": f"Execute {self._role} work for the incident response."},
        ])
        return resp.content


def build_handlers() -> tuple[dict[str, AgentHandler], dict[str, AgentHandler]]:
    roles = ["detector", "diagnostician", "isolator", "mitigator", "verifier", "analyst", "alerter"]
    baseline = {r: _BaselineHandler(r) for r in roles}
    runtime = {r: _RuntimeHandler(r) for r in roles}
    return baseline, runtime


async def run(
    backend: CacheSimulatingMockBackend | None = None,
    event_dir: Path | None = None,
    force_policy: str | None = None,
    adaptive_tuned: bool = False,
) -> dict[str, WorkloadResult]:
    baseline_h, runtime_h = build_handlers()
    return await run_workload(
        workload_type=WORKLOAD_TYPE,
        dag=_build_dag(),
        baseline_handlers=baseline_h,
        runtime_handlers=runtime_h,
        shared_context={
            "system-prefix": SYSTEM_PREFIX,
            "network-context": NETWORK_CONTEXT,
        },
        backend=backend,
        fault_plan={"isolate": "kill"},
        event_dir=event_dir,
        force_policy=force_policy,
        adaptive_tuned=adaptive_tuned,
    )
