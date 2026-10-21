#!/usr/bin/env python3
"""Verify Stage 3 Runtime-to-sched_ext policy binding on a real kernel."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentos.core.task import Task, TaskDAG
from agentos.isolation.cgroup import CgroupV2Manager
from agentos.isolation.sched_ext_control import SchedExtControlClient
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import Runtime
from scripts.preflight_sched_ext import probe_sched_ext
from scripts.verify_real_sched_ext import (
    _snapshot,
    _start_loader,
    _stop_loader,
    kernel_log_errors,
)


def associations(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    names = {
        "kernel.sched_ext_policy_published": "published",
        "kernel.sched_ext_worker_attached": "attached",
        "kernel.sched_ext_policy_removed": "removed",
    }
    for event in events:
        field = names.get(str(event.get("event_type")))
        task_id = event.get("task_id")
        if field is None or not isinstance(task_id, str):
            continue
        result.setdefault(task_id, {})[field] = event.get("data", {})
    return result


def acceptance_passed(summary: dict[str, Any]) -> bool:
    workers = summary.get("workers", {})
    return bool(
        summary.get("preflight_accepted") is True
        and summary.get("scheduler_name") == "agentos_scx"
        and summary.get("partial_mode") is True
        and workers.get("attempted") > 0
        and workers.get("completed") == workers.get("attempted")
        and workers.get("complete_associations") == workers.get("attempted")
        and summary.get("policy_readbacks_verified") == workers.get("attempted")
        and summary.get("policy_map_entries_after_runtime") == 0
        and summary.get("policy_hit") > 0
        and summary.get("loader_exit_code") == 0
        and summary.get("final_sched_ext_state") == "disabled"
        and summary.get("kernel_log_errors") == []
        and summary.get("failure_reason") is None
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _seal(output_dir: Path) -> None:
    lines = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
    (output_dir / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


async def _run_dag(worker_count: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dag = TaskDAG()
    for index in range(worker_count):
        task_id = f"stage3-worker-{index:03d}"
        dag.add_task(
            Task(
                task_id=task_id,
                name=task_id,
                agent_role="echo",
                metadata={
                    "execution": "subprocess",
                    "worker_type": "echo",
                    "worker_payload": {"result": task_id},
                },
            )
        )
    runtime = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={},
        execution_mode="subprocess",
        cgroup_manager=CgroupV2Manager(),
        sched_ext_mode="strict",
        sched_ext_control=SchedExtControlClient(),
    )
    results = await runtime.run(dag)
    return results, [event.to_dict() for event in runtime.events]


def run_verification(
    *, scheduler_binary: Path, output_dir: Path, workers: int = 10
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "accepted": False,
        "kernel_release": platform.release(),
        "scheduler_name": None,
        "partial_mode": None,
        "workers": {
            "attempted": workers,
            "completed": 0,
            "complete_associations": 0,
        },
        "policy_readbacks_verified": 0,
        "policy_map_entries_after_runtime": None,
        "policy_hit": 0,
        "loader_exit_code": None,
        "final_sched_ext_state": None,
        "kernel_log_errors": [],
        "failure_reason": None,
    }
    loader = None
    loader_events: list[dict[str, Any]] = []
    started_at = int(time.time())
    try:
        if os.geteuid() != 0:
            raise PermissionError("Stage 3 real verification must run as root")
        if workers <= 0:
            raise ValueError("workers must be positive")
        if not scheduler_binary.is_file():
            raise FileNotFoundError(str(scheduler_binary))
        preflight = probe_sched_ext()
        _write_json(output_dir / "preflight.json", preflight)
        summary["preflight_accepted"] = preflight.get("accepted") is True
        if not summary["preflight_accepted"]:
            raise RuntimeError(f"preflight rejected host: {preflight.get('failures')}")

        _write_json(output_dir / "sysfs-before.json", _snapshot())
        loader, loader_events = _start_loader(scheduler_binary)
        enabled = _snapshot()
        _write_json(output_dir / "sysfs-enabled.json", enabled)
        summary["scheduler_name"] = enabled.get("ops")
        summary["partial_mode"] = enabled.get("switch_all") == "0"

        results, runtime_events = asyncio.run(_run_dag(workers))
        _write_json(output_dir / "runtime-results.json", results)
        with (output_dir / "runtime-events.jsonl").open("w", encoding="utf-8") as stream:
            for event in runtime_events:
                stream.write(json.dumps(event, sort_keys=True) + "\n")

        linked = associations(runtime_events)
        complete = 0
        readbacks = 0
        for task_id in results:
            item = linked.get(task_id, {})
            pids = [item.get(name, {}).get("pid") for name in ("published", "attached", "removed")]
            if len(set(pids)) == 1 and isinstance(pids[0], int):
                complete += 1
            if item.get("published", {}).get("readback_verified") is True:
                readbacks += 1
        summary["workers"] = {
            "attempted": workers,
            "completed": len(results),
            "complete_associations": complete,
        }
        summary["policy_readbacks_verified"] = readbacks

        dumped = subprocess.run(
            ["bpftool", "-j", "map", "dump", "pinned", "/sys/fs/bpf/agentos/sched_ext/task_policy_map"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        entries = json.loads(dumped.stdout)
        summary["policy_map_entries_after_runtime"] = len(entries)
        _write_json(output_dir / "task-policy-map-after-runtime.json", entries)

        loader_rc, loader_stdout, loader_stderr = _stop_loader(
            loader, loader_events, signal.SIGTERM
        )
        loader = None
        summary["loader_exit_code"] = loader_rc
        final_stats = next(
            (event.get("stats", {}) for event in reversed(loader_events) if event.get("event") == "final_stats"),
            {},
        )
        summary["policy_hit"] = int(final_stats.get("policy_hit", 0))
        _write_json(output_dir / "loader-events.json", loader_events)
        (output_dir / "loader-stdout.log").write_text(loader_stdout, encoding="utf-8")
        (output_dir / "loader-stderr.log").write_text(loader_stderr, encoding="utf-8")

        final = _snapshot()
        _write_json(output_dir / "sysfs-after.json", final)
        summary["final_sched_ext_state"] = final.get("state")
        journal = subprocess.run(
            ["journalctl", "-k", "-b", f"--since=@{started_at}", "--no-pager"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        (output_dir / "kernel-window.log").write_text(
            journal.stdout + journal.stderr, encoding="utf-8"
        )
        if journal.returncode:
            raise RuntimeError(f"journalctl failed with code {journal.returncode}")
        summary["kernel_log_errors"] = kernel_log_errors(journal.stdout.splitlines())
    except Exception as exc:
        summary["failure_reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        if loader is not None and loader.poll() is None:
            try:
                _stop_loader(loader, loader_events, signal.SIGTERM)
            except Exception:
                loader.kill()
                loader.communicate()
        summary["final_sched_ext_state"] = _snapshot().get("state")
        summary["accepted"] = acceptance_passed(summary)
        _write_json(output_dir / "summary.json", summary)
        _seal(output_dir)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scheduler-binary",
        type=Path,
        default=PROJECT_ROOT / "native/sched_ext/build/bin/agentos_scx",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs/member_a_sched_ext_validation/stage3_runtime",
    )
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    summary = run_verification(
        scheduler_binary=args.scheduler_binary.resolve(),
        output_dir=args.output_dir.resolve(),
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
