#!/usr/bin/env python3
"""Deterministic P0 demo for Member A's cgroup feedback loop.

The demo runs the real Runtime and subprocess worker path, but uses a
temporary cgroup root plus deterministic cgroup-local PSI readings so it can
run in unprivileged CI. Privileged proof is handled by verify_real_cgroup_v2.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentos.core.resources import ResourceQuota, ResourceUsage
from agentos.core.task import Task, TaskDAG
from agentos.isolation.bpf import MockBPFMapAdapter
from agentos.isolation.cgroup import CgroupV2Manager
from agentos.isolation.kernel_adapter import UserSpaceKernelAdapter
from agentos.isolation.sched_ext import KernelProbeResult
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import Runtime


class DemoCgroupV2Manager(CgroupV2Manager):
    """Cgroup manager with deterministic usage and cgroup-local PSI samples."""

    def __init__(self, root: Path) -> None:
        super().__init__(root=root, enable_controllers=False)
        self._sample_index = 0

    def read_usage(self, path: Path) -> ResourceUsage:
        self._sample_index += 1
        return ResourceUsage(
            cpu_usage_usec=20_000 * self._sample_index,
            memory_rss_mb=32.0 + self._sample_index,
            memory_peak_mb=96.0 + self._sample_index,
            pids_current=1,
        )

    def read_pressure(self, path: Path) -> dict[str, tuple[float, bool]]:
        del path
        return {
            "cpu": (4.0, True),
            "memory": (24.0, True),
            "io": (2.0, True),
        }


def _build_dag() -> TaskDAG:
    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="pressure",
            name="pressure producer",
            agent_role="worker",
            resource_requirements=ResourceQuota(
                cpu_max_percent=50.0,
                cpu_shares=512,
                memory_max_mb=128,
                pids_max=16,
            ),
            metadata={
                "execution": "subprocess",
                "worker_type": "sleep",
                "worker_payload": {
                    "seconds": 0.05,
                    "result": "pressure-ok",
                },
                "workload_hint": "ai_inference",
                "sla_class": "batch",
            },
        )
    )
    dag.add_task(
        Task(
            task_id="followup",
            name="feedback consumer",
            agent_role="worker",
            dependencies=["pressure"],
            resource_requirements=ResourceQuota(
                cpu_max_percent=25.0,
                cpu_shares=256,
                memory_max_mb=64,
                pids_max=8,
            ),
            metadata={
                "execution": "subprocess",
                "worker_type": "echo",
                "worker_payload": {"result": "followup-ok"},
                "workload_hint": "interactive",
                "sla_class": "interactive",
            },
        )
    )
    return dag


def _event_dicts(runtime: Runtime) -> list[dict[str, Any]]:
    return [event.to_dict() for event in runtime.events]


def _summarize(events: list[dict[str, Any]], results: dict[str, Any]) -> dict[str, Any]:
    local_phases = sorted(
        {
            event["data"].get("phase")
            for event in events
            if event["event_type"] == "resource.pressure_sampled"
            and event["data"].get("scope") == "agent_cgroup"
            and event["data"].get("pressure", {}).get("pressure_source") == "cgroup"
            and event["data"].get("phase") is not None
        }
    )
    affected_decisions = [
        event
        for event in events
        if event["event_type"] == "scheduler.decision"
        and event["data"].get("pressure", {}).get("pressure_source") == "system+cgroup"
        and event["data"].get("pressure", {}).get("cgroup_local")
    ]
    summary = {
        "accepted": False,
        "results": results,
        "local_sample_phases": local_phases,
        "feedback_applied": sum(
            1
            for event in events
            if event["event_type"] == "resource.cgroup_feedback_applied"
            and event["data"].get("pressure_source") == "system+cgroup"
        ),
        "affected_decisions": len(affected_decisions),
        "kernel_decision_published": sum(
            1 for event in events if event["event_type"] == "kernel.decision_published"
        ),
        "cgroup_removed": sum(
            1 for event in events if event["event_type"] == "resource.cgroup_removed"
        ),
        "events": events,
    }
    summary["accepted"] = bool(
        summary["local_sample_phases"] == ["final", "initial", "running"]
        and summary["feedback_applied"] >= 1
        and summary["affected_decisions"] >= 1
        and summary["kernel_decision_published"] >= 2
        and summary["cgroup_removed"] >= 2
    )
    return summary


async def _run_demo_async(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cgroup_manager = DemoCgroupV2Manager(output_dir / "cgroup")
    adapter = UserSpaceKernelAdapter(
        cgroup_manager=cgroup_manager,
        bpf_map_adapter=MockBPFMapAdapter(),
        probe_result=KernelProbeResult(
            sched_ext="unavailable",
            bpf_map="mock",
            fallback_mode="member_a_p0_demo",
            reason="deterministic unprivileged feedback-loop demo",
        ),
    )
    runtime = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={},
        execution_mode="subprocess",
        kernel_adapter=adapter,
        worker_sample_interval_seconds=0.01,
    )

    results = await runtime.run(_build_dag())
    events = _event_dicts(runtime)
    summary = _summarize(events, results)

    events_path = output_dir / "events.jsonl"
    events_path.write_text(
        "\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n",
        encoding="utf-8",
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def run_demo(output_dir: str | Path = "runs/member_a_feedback_loop_demo") -> dict[str, Any]:
    return asyncio.run(_run_demo_async(Path(output_dir)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/member_a_feedback_loop_demo")
    args = parser.parse_args(argv)

    summary = run_demo(args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
