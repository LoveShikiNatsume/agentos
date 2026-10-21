"""Compile workload: simulates multi-stage build pipeline with parallel compilation agents."""
from __future__ import annotations

from pathlib import Path

from agentos.benchmarks.cache_sim import CacheSimulatingMockBackend
from agentos.core.task import Task, TaskDAG
from agentos.runtime.engine import AgentHandler

from scripts.workloads._base import WorkloadResult, run_workload

WORKLOAD_TYPE = "compile"

SYSTEM_PREFIX = (
    "You are a build-system agent in AgentOS-Runtime. "
    "The system manages parallel compilation tasks with dependency tracking. "
    "Shared build configuration and toolchain context is pooled across agents. "
    "Each agent handles one compilation stage."
)

BUILD_CONTEXT = (
    "Project: OS2026 kernel module compilation pipeline. "
    "Stages: configure → parse → compile-frontend → compile-backend → link → package. "
    "Toolchain: GCC 13, CMake 3.28, openEuler 24.03 LTS."
)


def _build_dag() -> TaskDAG:
    dag = TaskDAG()
    dag.add_task(Task(
        task_id="configure", name="configure", agent_role="configurator",
        workload_hint="compile", io_profile="mixed_read_write", expected_parallelism=1,
        sla_class="batch",
    ))
    dag.add_task(Task(
        task_id="parse", name="parse", agent_role="parser", dependencies=["configure"],
        workload_hint="compile", io_profile="mixed_read_write", expected_parallelism=2,
        sla_class="batch",
    ))
    dag.add_task(Task(
        task_id="compile_fe", name="compile_frontend", agent_role="compiler_fe", dependencies=["parse"],
        workload_hint="compile", io_profile="mixed_read_write", expected_parallelism=4,
        sla_class="batch",
    ))
    dag.add_task(Task(
        task_id="compile_be", name="compile_backend", agent_role="compiler_be", dependencies=["parse"],
        workload_hint="compile", io_profile="mixed_read_write", expected_parallelism=4,
        sla_class="batch",
    ))
    dag.add_task(Task(
        task_id="link", name="link", agent_role="linker", dependencies=["compile_fe", "compile_be"],
        workload_hint="compile", io_profile="mixed_read_write", expected_parallelism=2,
        sla_class="batch",
    ))
    dag.add_task(Task(
        task_id="package", name="package", agent_role="packager", dependencies=["link"],
        workload_hint="compile", io_profile="mixed_read_write", expected_parallelism=1,
        sla_class="batch",
    ))
    dag.add_task(Task(
        task_id="lint", name="lint", agent_role="linter", dependencies=["configure"],
        workload_hint="compile", io_profile="mixed_read_write", expected_parallelism=4,
        sla_class="batch",
    ))
    return dag


class _BaselineHandler(AgentHandler):
    def __init__(self, role: str) -> None:
        self._role = role

    async def run(self, ctx: AgentHandler.Context) -> str:
        resp = await ctx.llm_call([
            {"role": "system", "content": f"{SYSTEM_PREFIX}\n\n{BUILD_CONTEXT}\n\nYou are a {self._role}."},
            {"role": "user", "content": f"Execute {self._role} stage of the build pipeline."},
        ])
        return resp.content


class _RuntimeHandler(AgentHandler):
    def __init__(self, role: str) -> None:
        self._role = role

    async def run(self, ctx: AgentHandler.Context) -> str:
        resp = await ctx.llm_call([
            {"role": "user", "content": f"Execute {self._role} stage of the build pipeline."},
        ])
        return resp.content


def build_handlers() -> tuple[dict[str, AgentHandler], dict[str, AgentHandler]]:
    roles = ["configurator", "parser", "compiler_fe", "compiler_be", "linker", "packager", "linter"]
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
            "build-context": BUILD_CONTEXT,
        },
        backend=backend,
        fault_plan={"compile_fe": "kill"},
        event_dir=event_dir,
        force_policy=force_policy,
        adaptive_tuned=adaptive_tuned,
    )
