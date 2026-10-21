#!/usr/bin/env python3
"""Verify the real cgroup-local PSI -> next scheduler decision loop."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentos.core.resources import ResourceQuota
from agentos.core.task import Task, TaskDAG
from agentos.events.log import EventLog
from agentos.events.types import RuntimeEvent
from agentos.isolation.bpftool import (
    BPFMapError,
    BPFMapUnavailableError,
    BpftoolPinnedMapAdapter,
)
from agentos.isolation.cgroup import CgroupV2Manager
from agentos.isolation.delegation import detect_delegation
from agentos.isolation.kernel_adapter import UserSpaceKernelAdapter
from agentos.isolation.sched_ext import KernelProbeResult
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import AgentHandler, Runtime


_REQUIRED_PHASES = ("initial", "running", "final")


class _ImmediateHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        return ctx.task.task_id


def _event_type(event: RuntimeEvent | dict[str, Any]) -> str:
    if isinstance(event, RuntimeEvent):
        return event.event_type
    return str(event.get("event_type", ""))


def _event_data(event: RuntimeEvent | dict[str, Any]) -> dict[str, Any]:
    if isinstance(event, RuntimeEvent):
        return event.data
    data = event.get("data", {})
    return data if isinstance(data, dict) else {}


def summarize_events(
    events: Iterable[RuntimeEvent | dict[str, Any]],
    *,
    scenario_completed: bool,
    skip_reason: str | None = None,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    materialized = list(events)
    created = sum(
        _event_type(event) == "resource.cgroup_created"
        for event in materialized
    )
    quota_applied = sum(
        _event_type(event) == "resource.cgroup_quota_applied"
        for event in materialized
    )
    pid_events = [
        _event_data(event)
        for event in materialized
        if _event_type(event) == "resource.cgroup_pid_added"
    ]
    verified_pids = sum(
        data.get("pid_membership_verified") is True
        and data.get("membership_check_required") is True
        for data in pid_events
    )

    local_phase_counts: Counter[str] = Counter()
    nonlocal_pressure_samples = 0
    local_pressure_samples = 0
    for event in materialized:
        if _event_type(event) != "resource.pressure_sampled":
            continue
        data = _event_data(event)
        if data.get("scope") != "agent_cgroup":
            continue
        pressure = data.get("pressure")
        if not isinstance(pressure, dict):
            nonlocal_pressure_samples += 1
            continue
        sources = pressure.get("pressure_sources")
        all_local = (
            isinstance(sources, dict)
            and all(sources.get(resource) == "cgroup" for resource in ("cpu", "memory", "io"))
        )
        if not all_local:
            nonlocal_pressure_samples += 1
            continue
        local_pressure_samples += 1
        phase = data.get("phase")
        if isinstance(phase, str):
            local_phase_counts[phase] += 1

    applied_positions: dict[str, list[int]] = {}
    for position, event in enumerate(materialized):
        if _event_type(event) != "resource.cgroup_feedback_applied":
            continue
        data = _event_data(event)
        feedback_id = data.get("feedback_id")
        if (
            isinstance(feedback_id, str)
            and data.get("consumed") is True
            and data.get("reusable") is False
            and data.get("pressure_source") == "system+cgroup"
        ):
            applied_positions.setdefault(feedback_id, []).append(position)

    decision_positions: dict[str, list[int]] = {}
    for position, event in enumerate(materialized):
        if _event_type(event) != "scheduler.decision":
            continue
        pressure = _event_data(event).get("pressure")
        if not isinstance(pressure, dict):
            continue
        feedback_id = pressure.get("feedback_id")
        local = pressure.get("cgroup_local")
        if (
            pressure.get("pressure_source") == "system+cgroup"
            and isinstance(feedback_id, str)
            and isinstance(local, dict)
            and local.get("feedback_id") == feedback_id
        ):
            decision_positions.setdefault(feedback_id, []).append(position)

    affected_decisions = 0
    feedback_reuse_violations = 0
    for feedback_id, positions in applied_positions.items():
        decisions = decision_positions.get(feedback_id, [])
        matching = [
            decision_position
            for decision_position in decisions
            if decision_position > positions[0]
        ]
        if matching:
            affected_decisions += 1
        if len(positions) != 1 or len(matching) != 1:
            feedback_reuse_violations += max(
                1,
                abs(len(positions) - 1) + abs(len(matching) - 1),
            )
    for feedback_id, decisions in decision_positions.items():
        if feedback_id not in applied_positions:
            feedback_reuse_violations += len(decisions)

    cleanup_failed = sum(
        _event_type(event) == "resource.cgroup_cleanup_failed"
        for event in materialized
    )
    cleanup_succeeded = sum(
        _event_type(event) == "resource.cgroup_removed"
        for event in materialized
    )
    published = [
        _event_data(event)
        for event in materialized
        if _event_type(event) == "kernel.decision_published"
    ]
    real_updated = sum(
        data.get("real_updated_count", 0)
        for data in published
        if data.get("effective_bpf_mode") == "bpftool"
        and data.get("fallback_active") is False
    )
    readback_verified = any(
        data.get("bpf_readback_verified") is True for data in published
    )
    phases = [
        phase for phase in _REQUIRED_PHASES if local_phase_counts[phase] > 0
    ]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "accepted": False,
        "scenario_completed": scenario_completed,
        "cgroup_created": created,
        "quota_applied": quota_applied,
        "pid_attached": len(pid_events),
        "pid_membership_verified": verified_pids,
        "local_pressure_samples": local_pressure_samples,
        "local_sample_phases": phases,
        "local_phase_counts": dict(local_phase_counts),
        "nonlocal_pressure_samples": nonlocal_pressure_samples,
        "feedback_applied": sum(len(value) for value in applied_positions.values()),
        "affected_decisions": affected_decisions,
        "feedback_ids": sorted(applied_positions),
        "feedback_reuse_violations": feedback_reuse_violations,
        "cleanup_succeeded": cleanup_succeeded,
        "cleanup_failed": cleanup_failed,
        "root_cleanup_failed": 0,
        "real_bpf_published": real_updated > 0,
        "real_bpf_updated_tasks": real_updated,
        "bpf_fallback_active": any(
            data.get("fallback_active") is True for data in published
        ),
        "bpf_readback_verified": readback_verified,
        "bpf_readback_task_ids": [],
        "bpf_cleanup_failed": 0,
        "skip_reason": skip_reason,
        "failure_reason": failure_reason,
        "claims": {
            "real_cgroup_v2": created > 0 and verified_pids > 0,
            "cgroup_local_psi_feedback_loop": affected_decisions > 0,
            "sched_ext_active": False,
            "kernel_scheduling_takeover": False,
        },
    }
    summary["accepted"] = acceptance_passed(summary)
    return summary


def acceptance_passed(summary: dict[str, Any]) -> bool:
    return bool(
        summary.get("scenario_completed") is True
        and summary.get("cgroup_created", 0) >= 1
        and summary.get("quota_applied", 0) >= 1
        and summary.get("pid_attached", 0) >= 1
        and summary.get("pid_membership_verified", 0) >= 1
        and summary.get("local_pressure_samples", 0) >= 3
        and summary.get("local_sample_phases") == list(_REQUIRED_PHASES)
        and summary.get("nonlocal_pressure_samples", 1) == 0
        and summary.get("feedback_applied", 0) >= 1
        and summary.get("affected_decisions", 0) >= 1
        and summary.get("feedback_reuse_violations", 1) == 0
        and summary.get("cleanup_succeeded", 0) >= 1
        and summary.get("cleanup_failed", 1) == 0
        and summary.get("root_cleanup_failed", 1) == 0
        and summary.get("real_bpf_published") is True
        and summary.get("real_bpf_updated_tasks", 0) >= 2
        and summary.get("bpf_fallback_active") is False
        and summary.get("bpf_readback_verified") is True
        and summary.get("bpf_cleanup_failed", 1) == 0
        and summary.get("skip_reason") is None
        and summary.get("failure_reason") is None
    )


def _build_runtime(
    root: Path,
    event_log: EventLog,
    *,
    pin_path: Path,
    bpftool_path: str,
) -> tuple[Runtime, TaskDAG, BpftoolPinnedMapAdapter]:
    manager = CgroupV2Manager(root=root)
    bpf = BpftoolPinnedMapAdapter(
        pin_path=pin_path,
        bpftool_path=bpftool_path,
        owns_pin=True,
    )
    bpf.create_if_missing()
    adapter = UserSpaceKernelAdapter(
        cgroup_manager=manager,
        bpf_map_adapter=bpf,
        requested_bpf_mode="bpftool",
        probe_result=KernelProbeResult(
            sched_ext="unavailable",
            bpf_map="bpftool",
            fallback_mode=None,
            reason="strict real Runtime kernel-chain verification",
        ),
    )
    quota = ResourceQuota(
        cpu_max_percent=50.0,
        cpu_shares=512,
        memory_max_mb=128,
        pids_max=32,
        timeout_seconds=5.0,
    )
    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="local-psi-producer",
            name="real cgroup local PSI producer",
            agent_role="sleep",
            resource_requirements=quota,
            metadata={
                "execution": "subprocess",
                "worker_type": "sleep",
                "worker_payload": {
                    "seconds": 0.35,
                    "result": "local sample complete",
                },
            },
        )
    )
    dag.add_task(
        Task(
            task_id="feedback-consumer",
            name="next scheduler decision consumer",
            agent_role="immediate",
            dependencies=["local-psi-producer"],
        )
    )
    runtime = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"immediate": _ImmediateHandler()},
        event_log=event_log,
        kernel_adapter=adapter,
        worker_sample_interval_seconds=0.1,
    )
    return runtime, dag, bpf


async def run_verification(
    root: Path,
    event_log: EventLog,
    *,
    bpffs_root: Path,
    bpftool_path: str,
) -> dict[str, Any]:
    mode, allowed = detect_delegation()
    if not allowed:
        return summarize_events(
            [],
            scenario_completed=False,
            skip_reason=(
                f"delegation={mode}; set AGENTOS_CGROUP_ALLOW_ROOT=1 to override"
            ),
        )

    pin_path = bpffs_root / "agentos" / f"member_a_chain_{os.getpid()}"
    failure_reason = None
    completed = False
    bpf_cleanup_failed = 0
    bpf_readback_verified = False
    bpf_readback_task_ids: list[str] = []
    try:
        runtime, dag, bpf = _build_runtime(
            root,
            event_log,
            pin_path=pin_path,
            bpftool_path=bpftool_path,
        )
    except BPFMapUnavailableError as exc:
        return summarize_events(
            [],
            scenario_completed=False,
            skip_reason=str(exc),
        )
    except BPFMapError as exc:
        return summarize_events(
            [],
            scenario_completed=False,
            failure_reason=f"{type(exc).__name__}: {exc}",
        )

    try:
        results = await runtime.run(dag)
        completed = (
            results.get("local-psi-producer") == "local sample complete"
            and results.get("feedback-consumer") == "feedback-consumer"
        )
        snapshot = bpf.snapshot()
        required = {"local-psi-producer", "feedback-consumer"}
        bpf_readback_task_ids = sorted(snapshot)
        bpf_readback_verified = required.issubset(snapshot)
    except Exception as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            bpf.clear()
        except Exception as exc:
            bpf_cleanup_failed = 1
            if failure_reason is None:
                failure_reason = (
                    f"BPF cleanup failed: {type(exc).__name__}: {exc}"
                )

    summary = summarize_events(
        runtime.events,
        scenario_completed=completed,
        failure_reason=failure_reason,
    )
    summary["bpf_readback_verified"] = bpf_readback_verified
    summary["bpf_readback_task_ids"] = bpf_readback_task_ids
    summary["bpf_cleanup_failed"] = bpf_cleanup_failed
    summary["delegation_mode"] = mode
    summary["cgroup_root"] = str(root)
    summary["bpf_pin_path"] = str(pin_path)
    summary["accepted"] = acceptance_passed(summary)
    return summary


def _environment_text() -> str:
    os_release = ""
    try:
        os_release = Path("/etc/os-release").read_text(encoding="utf-8").strip()
    except OSError:
        os_release = "unavailable"
    return "\n".join(
        (
            f"python={sys.version.split()[0]}",
            f"platform={platform.platform()}",
            f"pid={os.getpid()}",
            "cgroup_filesystem=cgroup2fs",
            os_release,
            "",
        )
    )


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/sys/fs/cgroup/agentos_stage3_local_psi",
    )
    parser.add_argument("--bpffs-root", default="/sys/fs/bpf")
    parser.add_argument("--bpftool-path", default="bpftool")
    parser.add_argument("--out", default="runs/member_a_stage3_local_psi")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "events.jsonl"
    events_path.unlink(missing_ok=True)
    root = Path(args.root)
    summary = asyncio.run(
        run_verification(
            root,
            EventLog(events_path),
            bpffs_root=Path(args.bpffs_root),
            bpftool_path=args.bpftool_path,
        )
    )
    try:
        root.rmdir()
    except FileNotFoundError:
        pass
    except OSError as exc:
        summary["root_cleanup_failed"] = 1
        if summary.get("failure_reason") is None:
            summary["failure_reason"] = (
                f"verifier root cleanup failed: {type(exc).__name__}: {exc}"
            )
    summary["accepted"] = acceptance_passed(summary)
    _write_json(out_dir / "summary.json", summary)
    (out_dir / "environment.txt").write_text(
        _environment_text(),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary.get("skip_reason"):
        return 2
    return 0 if acceptance_passed(summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
