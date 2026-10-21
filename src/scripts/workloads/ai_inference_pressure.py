"""AI inference workload: multi-agent pipeline with heavy LLM calls and prefix sharing."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from agentos.benchmarks.cache_sim import CacheSimulatingMockBackend
from agentos.core.task import Task, TaskDAG
from agentos.runtime.engine import AgentHandler

from scripts.workloads._base import WorkloadResult, run_workload

WORKLOAD_TYPE = "ai_inference"

SYSTEM_PREFIX = (
    "You are an AI agent operating within AgentOS-Runtime. "
    "The system manages GPU resources, KV cache, and prompt prefix deduplication. "
    "You share context with other agents through ContextRef pooling. "
    "Optimize token usage by reusing shared context where possible."
)

TASK_CONTEXT = (
    "Given a user request for implementing a recommendation engine: "
    "1. Analyze requirements 2. Design architecture 3. Generate code "
    "4. Review output 5. Optimize performance 6. Deploy"
)


def _build_dag() -> TaskDAG:
    dag = TaskDAG()
    dag.add_task(Task(
        task_id="analyze", name="analyze", agent_role="analyst",
        workload_hint="ai_inference", io_profile="read_heavy", expected_parallelism=1,
        sla_class="interactive", deadline_ms=1000,
    ))
    dag.add_task(Task(
        task_id="design", name="design", agent_role="architect", dependencies=["analyze"],
        workload_hint="ai_inference", io_profile="read_heavy", expected_parallelism=1,
        sla_class="interactive", deadline_ms=1000,
    ))
    dag.add_task(Task(
        task_id="generate", name="generate", agent_role="coder", dependencies=["design"],
        workload_hint="ai_inference", io_profile="read_heavy", expected_parallelism=2,
        sla_class="interactive", deadline_ms=1500,
    ))
    dag.add_task(Task(
        task_id="review", name="review", agent_role="reviewer", dependencies=["generate"],
        workload_hint="ai_inference", io_profile="read_heavy", expected_parallelism=1,
        sla_class="interactive", deadline_ms=800,
    ))
    dag.add_task(Task(
        task_id="optimize", name="optimize", agent_role="optimizer", dependencies=["review"],
        workload_hint="ai_inference", io_profile="read_heavy", expected_parallelism=1,
        sla_class="batch",
    ))
    dag.add_task(Task(
        task_id="deploy", name="deploy", agent_role="deployer", dependencies=["optimize"],
        workload_hint="ai_inference", io_profile="read_heavy", expected_parallelism=1,
        sla_class="batch",
    ))
    dag.add_task(Task(
        task_id="monitor", name="monitor", agent_role="monitor", dependencies=["analyze"],
        workload_hint="ai_inference", io_profile="read_heavy", expected_parallelism=1,
        sla_class="batch",
    ))
    return dag


class _BaselineHandler(AgentHandler):
    def __init__(self, role: str) -> None:
        self._role = role

    async def run(self, ctx: AgentHandler.Context) -> str:
        resp = await ctx.llm_call([
            {"role": "system", "content": f"{SYSTEM_PREFIX}\n\n{TASK_CONTEXT}\n\nYou are a {self._role}."},
            {"role": "user", "content": f"Perform {self._role} work on the recommendation engine task."},
        ])
        return resp.content


class _RuntimeHandler(AgentHandler):
    def __init__(self, role: str) -> None:
        self._role = role

    async def run(self, ctx: AgentHandler.Context) -> str:
        resp = await ctx.llm_call([
            {"role": "user", "content": f"Perform {self._role} work on the recommendation engine task."},
        ])
        return resp.content


def build_handlers() -> tuple[dict[str, AgentHandler], dict[str, AgentHandler]]:
    roles = ["analyst", "architect", "coder", "reviewer", "optimizer", "deployer", "monitor"]
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
            "task-context": TASK_CONTEXT,
        },
        backend=backend,
        fault_plan={"generate": "kill"},
        event_dir=event_dir,
        force_policy=force_policy,
        adaptive_tuned=adaptive_tuned,
    )
