"""Unified PPT experiment script — runs all three experiments and writes results to runs/ppt_data/.

Usage:
    .venv/bin/python scripts/run_ppt_experiment.py --backend mock
    .venv/bin/python scripts/run_ppt_experiment.py --backend deepseek-sim
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
from typing import Any

from agentos.comm.bus import MessageBus
from agentos.core.task import Task, TaskDAG
from agentos.core.types import Message
from agentos.fault.injector import FaultInjectionPlan
from agentos.llm.factory import create_backend
from agentos.runtime.engine import AgentHandler, Runtime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _NoopHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        return "done"


def _load_ablation_module():
    spec = importlib.util.spec_from_file_location(
        "run_ablation_experiment",
        Path(__file__).parent / "run_ablation_experiment.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Experiment 1: Ablation (delegates to run_ablation_experiment)
# ---------------------------------------------------------------------------

async def _run_ablation(backend: str, output_dir: Path) -> None:
    mod = _load_ablation_module()
    await mod.run_ablation(backend, 2, output_dir)


# ---------------------------------------------------------------------------
# Experiment 2: Fault propagation control comparison
# DAG: A→B→C→D, A→E→F→D  (B is killed)
# ---------------------------------------------------------------------------

def _propagation_dag() -> TaskDAG:
    # DevOps incident DAG:
    #   monitor → diagnose → apply_fix → restore_service
    #   monitor → notify_team → write_postmortem → restore_service
    # "diagnose" is killed — demonstrates propagation control on apply_fix/restore_service
    dag = TaskDAG()
    dag.add_task(Task(task_id="monitor",         name="monitor",         agent_role="noop"))
    dag.add_task(Task(task_id="diagnose",        name="diagnose",        agent_role="noop", dependencies=["monitor"]))
    dag.add_task(Task(task_id="apply_fix",       name="apply_fix",       agent_role="noop", dependencies=["diagnose"]))
    dag.add_task(Task(task_id="notify_team",     name="notify_team",     agent_role="noop", dependencies=["monitor"]))
    dag.add_task(Task(task_id="write_postmortem",name="write_postmortem",agent_role="noop", dependencies=["notify_team"]))
    dag.add_task(Task(task_id="restore_service", name="restore_service", agent_role="noop", dependencies=["apply_fix", "write_postmortem"]))
    return dag


async def _run_propagation(backend: str) -> dict[str, Any]:
    configs = [
        ("no_propagation", 0),
        ("depth_1", 1),
        ("full_propagation", None),
    ]
    results: dict[str, Any] = {}
    for name, depth in configs:
        dag = _propagation_dag()
        rt = Runtime(
            llm_backend=create_backend(backend),
            handlers={"noop": _NoopHandler()},
            fault_injection=FaultInjectionPlan({"diagnose": "kill"}),
            propagation_depth=depth,
        )
        try:
            await rt.run(dag)
        except Exception:
            pass
        statuses = {t.task_id: t.status.value for t in dag.tasks.values()}
        results[name] = {
            "propagation_depth": depth,
            "task_statuses": statuses,
            "skipped_count": sum(1 for s in statuses.values() if s == "skipped"),
            "propagated_events": sum(1 for e in rt.events if e.event_type == "fault.propagated"),
            "dag_complete": dag.is_complete(),
        }
        skipped = results[name]["skipped_count"]
        evts = results[name]["propagated_events"]
        done = results[name]["dag_complete"]
        print(f"  {name:<22} depth={str(depth):<4} skipped={skipped}  fault.propagated={evts}  complete={done}")
    return results


# ---------------------------------------------------------------------------
# Experiment 3: Message payload savings (inline vs ContextRef pointer)
# ---------------------------------------------------------------------------

async def _run_comm_savings() -> dict[str, Any]:
    bus = MessageBus()
    large_content = "A" * 2000   # ~2 KB inline payload
    ref_content = "ctx://architect-design"   # tiny pointer

    for _ in range(6):
        await bus.send(Message(
            from_agent="sender", to_agent="receiver",
            content={"body": large_content},
            message_type="inline",
        ))
    for _ in range(6):
        await bus.send(Message(
            from_agent="sender", to_agent="receiver",
            content={"ref": ref_content},
            message_type="context_ref",
        ))
    return bus.payload_savings()


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

_README = """\
# PPT 实验数据说明

| 文件 | PPT 页 | 内容 |
|------|--------|------|
| `ablation_results.json` | P13 消融矩阵 | bare/dag_only/dag_context/full 四种模式原始指标（2次取均值） |
| `ablation_table.md` | P13 消融矩阵 | 可直接贴 PPT 的 Markdown 表格 |
| `propagation_comparison.json` | P15 故障传播控制 | no_propagation/depth_1/full_propagation 三种配置下的任务状态与事件数 |
| `comm_savings.json` | P16 通信节省 | inline 全量传输 vs context_ref 引用传递的 payload 字节对比与节省比例 |
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main(backend: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/3] Ablation experiment (backend={backend}) ...")
    await _run_ablation(backend, output_dir)

    print(f"\n[2/3] Fault propagation comparison (backend={backend}) ...")
    prop = await _run_propagation(backend)
    prop_path = output_dir / "propagation_comparison.json"
    prop_path.write_text(json.dumps(prop, indent=2, ensure_ascii=False))
    print(f"  [saved] {prop_path}")

    print("\n[3/3] Message payload savings ...")
    savings = await _run_comm_savings()
    savings_path = output_dir / "comm_savings.json"
    savings_path.write_text(json.dumps(savings, indent=2, ensure_ascii=False))
    pct = savings.get("payload_reduction_ratio", 0.0)
    print(f"  bytes_saved={savings.get('bytes_saved', 0)}  reduction={pct:.1%}")
    print(f"  [saved] {savings_path}")

    (output_dir / "README.md").write_text(_README)
    print(f"\n[saved] {output_dir}/README.md")
    print("\nAll PPT data ready →", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="run-ppt-experiment")
    parser.add_argument("--backend", default="mock",
                        choices=["mock", "cache-sim", "deepseek-sim", "deepseek"])
    parser.add_argument("--output-dir", default="runs/ppt_data")
    args = parser.parse_args()
    asyncio.run(main(args.backend, Path(args.output_dir)))
