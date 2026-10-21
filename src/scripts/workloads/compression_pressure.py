"""Compression workload: simulates context-heavy data processing with large prompt prefixes."""
from __future__ import annotations

from pathlib import Path

from agentos.benchmarks.cache_sim import CacheSimulatingMockBackend
from agentos.core.resources import ResourceQuota
from agentos.core.task import Task, TaskDAG
from agentos.runtime.engine import AgentHandler

from scripts.workloads._base import WorkloadResult, run_workload

WORKLOAD_TYPE = "compression"

SYSTEM_PREFIX = (
    "You are a data processing agent in AgentOS-Runtime. "
    "The system manages context compression and eviction for large payloads. "
    "Shared data schemas and transformation rules are pooled across agents. "
    "Each agent handles one stage of the ETL pipeline."
)

DATA_CONTEXT = (
    "Pipeline: ingest → validate → transform → compress → store → verify. "
    "Data: 500MB log corpus from openEuler kernel traces. "
    "Compression target: 10:1 ratio with lossless semantic preservation. "
    "Schema: timestamp, pid, syscall, latency_ns, cpu_id, cgroup_path."
)

LARGE_SCHEMA = (
    "Field definitions:\n"
    "  timestamp: uint64, nanoseconds since boot\n"
    "  pid: int32, process identifier\n"
    "  tid: int32, thread identifier\n"
    "  syscall: string, syscall name (max 64 chars)\n"
    "  latency_ns: uint64, syscall latency in nanoseconds\n"
    "  cpu_id: uint16, CPU core identifier\n"
    "  cgroup_path: string, cgroup v2 path\n"
    "  memory_usage_bytes: uint64, RSS at time of syscall\n"
    "  io_bytes_read: uint64, cumulative read bytes\n"
    "  io_bytes_written: uint64, cumulative write bytes\n"
    "  context_switches: uint32, voluntary + involuntary\n"
    "  gpu_utilization_pct: float32, GPU usage percentage\n"
    "  kv_cache_blocks_used: uint32, vLLM KV cache blocks in use"
)


def _build_dag() -> TaskDAG:
    dag = TaskDAG()
    dag.add_task(Task(
        task_id="ingest", name="ingest", agent_role="ingestor",
        workload_hint="compression", io_profile="read_write_heavy", expected_parallelism=2,
        sla_class="batch",
    ))
    dag.add_task(Task(
        task_id="validate", name="validate", agent_role="validator", dependencies=["ingest"],
        workload_hint="compression", io_profile="read_write_heavy", expected_parallelism=2,
        sla_class="batch",
    ))
    dag.add_task(Task(
        task_id="transform", name="transform", agent_role="transformer", dependencies=["validate"],
        workload_hint="compression", io_profile="read_write_heavy", expected_parallelism=2,
        sla_class="best_effort",
    ))
    dag.add_task(Task(
        task_id="compress", name="compress", agent_role="compressor", dependencies=["transform"],
        workload_hint="compression", io_profile="read_write_heavy", expected_parallelism=2,
        sla_class="best_effort",
    ))
    dag.add_task(Task(
        task_id="store", name="store", agent_role="storer", dependencies=["compress"],
        workload_hint="compression", io_profile="read_write_heavy", expected_parallelism=2,
        sla_class="best_effort",
    ))
    dag.add_task(Task(
        task_id="verify", name="verify", agent_role="verifier", dependencies=["store"],
        workload_hint="compression", io_profile="read_write_heavy", expected_parallelism=1,
        sla_class="best_effort",
    ))
    dag.add_task(Task(
        task_id="index", name="index", agent_role="indexer", dependencies=["validate"],
        workload_hint="compression", io_profile="read_write_heavy", expected_parallelism=2,
        sla_class="batch",
    ))
    # P6: Add a large memory task to trigger safety violation
    dag.add_task(Task(
        task_id="huge_compress", name="huge_compress", agent_role="compressor", dependencies=["transform"],
        workload_hint="compression", io_profile="read_write_heavy", expected_parallelism=1,
        sla_class="best_effort",
        resource_requirements=ResourceQuota(memory_max_mb=8192),
    ))
    return dag


class _BaselineHandler(AgentHandler):
    def __init__(self, role: str) -> None:
        self._role = role

    async def run(self, ctx: AgentHandler.Context) -> str:
        resp = await ctx.llm_call([
            {
                "role": "system",
                "content": f"{SYSTEM_PREFIX}\n\n{DATA_CONTEXT}\n\n{LARGE_SCHEMA}\n\nYou are a {self._role}.",
            },
            {"role": "user", "content": f"Execute {self._role} stage of the data pipeline."},
        ])
        return resp.content


class _RuntimeHandler(AgentHandler):
    def __init__(self, role: str) -> None:
        self._role = role

    async def run(self, ctx: AgentHandler.Context) -> str:
        resp = await ctx.llm_call([
            {"role": "user", "content": f"Execute {self._role} stage of the data pipeline."},
        ])
        return resp.content


def build_handlers() -> tuple[dict[str, AgentHandler], dict[str, AgentHandler]]:
    roles = ["ingestor", "validator", "transformer", "compressor", "storer", "verifier", "indexer"]
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
            "data-context": DATA_CONTEXT,
            "schema": LARGE_SCHEMA,
        },
        backend=backend,
        fault_plan={"compress": "kill"},
        event_dir=event_dir,
        force_policy=force_policy,
        adaptive_tuned=adaptive_tuned,
    )
