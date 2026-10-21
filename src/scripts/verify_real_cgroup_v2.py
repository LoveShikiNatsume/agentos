#!/usr/bin/env python3
"""Privileged, machine-readable verifier for the real cgroup v2 path."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentos.core.resources import ResourceQuota
from agentos.isolation.bpf import MockBPFMapAdapter
from agentos.isolation.cgroup import CgroupV2Manager
from agentos.isolation.delegation import detect_delegation
from agentos.isolation.kernel_adapter import UserSpaceKernelAdapter
from agentos.isolation.sched_ext import KernelProbeResult


def _empty_result(mode: str) -> dict[str, Any]:
    return {
        "available": False,
        "delegation_mode": mode,
        "quota_applied": 0,
        "pid_attached": 0,
        "pid_membership_verified": False,
        "child_inheritance_verified": False,
        "initial_usage_samples": 0,
        "running_usage_samples": 0,
        "final_usage_samples": 0,
        "initial_pressure_samples": 0,
        "running_pressure_samples": 0,
        "final_pressure_samples": 0,
        "cgroup_local_pressure_samples": 0,
        "usage_samples": 0,
        "pressure_sources": {},
        "pressure_reasons": {},
        "cleanup_failed": 0,
        "skip_reason": None,
        "failure_reason": None,
    }


def acceptance_passed(result: dict[str, Any]) -> bool:
    return bool(
        result.get("available") is True
        and result.get("quota_applied", 0) >= 4
        and result.get("pid_attached", 0) >= 1
        and result.get("pid_membership_verified") is True
        and result.get("child_inheritance_verified") is True
        and result.get("initial_usage_samples", 0) >= 1
        and result.get("running_usage_samples", 0) >= 1
        and result.get("final_usage_samples", 0) >= 1
        and result.get("initial_pressure_samples", 0) >= 1
        and result.get("running_pressure_samples", 0) >= 1
        and result.get("final_pressure_samples", 0) >= 1
        and result.get("cgroup_local_pressure_samples", 0) >= 1
        and result.get("usage_samples", 0) >= 3
        and result.get("cleanup_failed", 1) == 0
        and result.get("skip_reason") is None
        and result.get("failure_reason") is None
    )


def _worker_command() -> list[str]:
    program = (
        "import subprocess,sys,time\n"
        "print('ready', flush=True)\n"
        "if sys.stdin.readline():\n"
        "    child = subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(1.0)'])\n"
        "    print(child.pid, flush=True)\n"
        "    child.wait()\n"
        "time.sleep(0.1)\n"
    )
    return [sys.executable, "-u", "-c", program]


def run_verification(root: Path) -> dict[str, Any]:
    mode, allowed = detect_delegation()
    result = _empty_result(mode)
    if not allowed:
        result["skip_reason"] = (
            f"delegation={mode}; set AGENTOS_CGROUP_ALLOW_ROOT=1 to override"
        )
        return result

    manager = CgroupV2Manager(root=root)
    quota = ResourceQuota(
        cpu_max_percent=50.0,
        cpu_shares=512,
        memory_max_mb=128,
        pids_max=32,
    )
    cgroup_path: Path | None = None
    worker: subprocess.Popen[str] | None = None
    adapter = UserSpaceKernelAdapter(
        cgroup_manager=manager,
        bpf_map_adapter=MockBPFMapAdapter(),
        probe_result=KernelProbeResult(
            sched_ext="unavailable",
            bpf_map="mock",
            fallback_mode="verification_only",
            reason="real cgroup verifier does not claim sched_ext",
        ),
    )

    def sample_phase(phase: str) -> None:
        if cgroup_path is None:
            raise RuntimeError(f"{phase} sample requested without cgroup")
        sample = adapter.sample_pressure(cgroup_path=cgroup_path)
        if sample.cgroup_usage is None:
            raise RuntimeError(f"{phase} cgroup usage unavailable")
        result[f"{phase}_usage_samples"] += 1
        result[f"{phase}_pressure_samples"] += 1
        result["usage_samples"] += 1
        result["pressure_sources"][phase] = sample.pressure_source
        if sample.reason is not None:
            result["pressure_reasons"][phase] = sample.reason
        if sample.pressure_source == "cgroup":
            result["cgroup_local_pressure_samples"] += 1

    try:
        created = manager.create(f"verifier-{os.getpid()}", quota)
        if not created.available:
            result["skip_reason"] = f"create failed: {created.reason}"
            return result

        cgroup_path = created.path
        result["available"] = True
        result["quota_applied"] = 4

        worker = subprocess.Popen(
            _worker_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if worker.stdout is None or worker.stdin is None:
            raise RuntimeError("worker pipes unavailable")
        if worker.stdout.readline().strip() != "ready":
            raise RuntimeError("worker did not become ready")

        attached = manager.add_pid(cgroup_path, worker.pid)
        if not attached.available:
            raise RuntimeError(f"PID attach failed: {attached.reason}")
        result["pid_attached"] = 1
        result["pid_membership_verified"] = manager.verify_pid_membership(
            cgroup_path, worker.pid
        )
        sample_phase("initial")

        worker.stdin.write("spawn\n")
        worker.stdin.flush()
        child_line = worker.stdout.readline().strip()
        if not child_line.isdigit():
            raise RuntimeError(f"worker did not report child PID: {child_line!r}")
        child_pid = int(child_line)
        result["child_inheritance_verified"] = manager.verify_pid_membership(
            cgroup_path, child_pid
        )

        sample_phase("running")

        worker.wait(timeout=10)
        sample_phase("final")
    except Exception as exc:
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        if worker is not None and worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=2)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=2)
        if cgroup_path is not None:
            cleanup = manager.cleanup(cgroup_path)
            if not cleanup.available:
                result["cleanup_failed"] += 1
                if result["failure_reason"] is None:
                    result["failure_reason"] = f"cleanup failed: {cleanup.reason}"

    return result


def _out(data: dict[str, Any], path: str | None) -> None:
    text = json.dumps(data, indent=2, sort_keys=True)
    print(text)
    if path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/sys/fs/cgroup/agentos")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    result = run_verification(Path(args.root))
    _out(result, args.json_out)
    if result.get("skip_reason"):
        return 2
    return 0 if acceptance_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
