#!/usr/bin/env python3
"""Privileged verifier for real cgroup CPU, memory, and PID effects."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentos.core.resources import ResourceQuota
from agentos.isolation.cgroup import CgroupV2Manager
from agentos.isolation.delegation import detect_delegation


def _strict_int(mapping: dict[str, Any], key: str) -> int | None:
    value = mapping.get(key)
    if type(value) is not int:
        return None
    return value


def _delta(before: dict[str, Any], after: dict[str, Any], key: str) -> int | None:
    start = _strict_int(before, key)
    end = _strict_int(after, key)
    if start is None or end is None:
        return None
    return end - start


def summarize_cpu_effect(
    *,
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    nr_delta = _delta(before, after, "nr_throttled")
    usec_delta = _delta(before, after, "throttled_usec")
    accepted = (
        nr_delta is not None
        and usec_delta is not None
        and nr_delta > 0
        and usec_delta > 0
    )
    return {
        "accepted": accepted,
        "before": before,
        "after": after,
        "nr_throttled_delta": nr_delta,
        "throttled_usec_delta": usec_delta,
        "failure_reason": None
        if accepted
        else "cpu throttling counters did not increase",
    }


def summarize_pids_effect(
    *,
    attempted: int,
    started: int,
    failures: list[str],
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
) -> dict[str, Any]:
    valid_counts = (
        type(attempted) is int
        and type(started) is int
        and attempted >= 0
        and started >= 0
        and started <= attempted
    )
    events_delta = _delta(before or {}, after or {}, "max")
    has_limit_failure = any(item in {"EAGAIN", "Resource temporarily unavailable"} for item in failures)
    accepted = bool(
        valid_counts
        and attempted > started
        and (has_limit_failure or (events_delta is not None and events_delta > 0))
    )
    return {
        "accepted": accepted,
        "attempted": attempted,
        "started": started,
        "blocked": attempted - started if valid_counts else None,
        "failures": failures,
        "before": before or {},
        "after": after or {},
        "events_max_delta": events_delta,
        "failure_reason": None
        if accepted
        else "PID limit did not block process creation",
    }


def summarize_memory_effect(
    *,
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    max_delta = _delta(before, after, "max")
    oom_delta = _delta(before, after, "oom")
    kill_delta = _delta(before, after, "oom_kill")
    deltas = (max_delta, oom_delta, kill_delta)
    accepted = all(delta is not None and delta >= 0 for delta in deltas) and any(
        delta > 0 for delta in deltas if delta is not None
    )
    return {
        "accepted": accepted,
        "before": before,
        "after": after,
        "events_max_delta": max_delta,
        "oom_delta": oom_delta,
        "oom_kill_delta": kill_delta,
        "failure_reason": None
        if accepted
        else "memory events did not increase",
    }


def acceptance_passed(summary: dict[str, Any]) -> bool:
    return bool(
        summary.get("cpu", {}).get("accepted") is True
        and summary.get("pids", {}).get("accepted") is True
        and summary.get("memory", {}).get("accepted") is True
        and summary.get("cleanup_failed", 1) == 0
        and summary.get("skip_reason") is None
        and summary.get("failure_reason") is None
    )


def _read_key_values(path: Path) -> dict[str, int]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    result: dict[str, int] = {}
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            result[parts[0]] = int(parts[1])
        except ValueError:
            continue
    return result


def _enable_parent_controllers(root: Path) -> None:
    parent = root.parent
    controllers = set((parent / "cgroup.controllers").read_text(encoding="utf-8").split())
    subtree = parent / "cgroup.subtree_control"
    for controller in ("cpu", "memory", "pids", "io"):
        if controller in controllers:
            subtree.write_text(f"+{controller}", encoding="utf-8")


def _kill_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=2)


def _start_worker(program: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-u", "-c", program],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def _wait_ready(process: subprocess.Popen[str]) -> None:
    if process.stdout is None:
        raise RuntimeError("worker stdout unavailable")
    line = process.stdout.readline().strip()
    if line != "ready":
        raise RuntimeError(f"worker did not become ready: {line!r}")


def _scenario_path(root: Path, name: str, quota: ResourceQuota) -> tuple[CgroupV2Manager, Path]:
    manager = CgroupV2Manager(root=root)
    created = manager.create(name, quota)
    if not created.available:
        raise RuntimeError(f"create {name} failed: {created.reason}")
    return manager, created.path


def run_cpu_effect(root: Path) -> dict[str, Any]:
    manager, path = _scenario_path(
        root,
        "cpu",
        ResourceQuota(cpu_max_percent=10.0, cpu_shares=100, memory_max_mb=256, pids_max=16),
    )
    worker: subprocess.Popen[str] | None = None
    try:
        program = (
            "import sys,time\n"
            "print('ready', flush=True)\n"
            "sys.stdin.readline()\n"
            "deadline=time.monotonic()+1.5\n"
            "x=0\n"
            "while time.monotonic()<deadline:\n"
            "    x=(x+1)%1000003\n"
            "print(x, flush=True)\n"
        )
        worker = _start_worker(program)
        _wait_ready(worker)
        attached = manager.add_pid(path, worker.pid)
        if not attached.available:
            raise RuntimeError(f"cpu worker attach failed: {attached.reason}")
        before = _read_key_values(path / "cpu.stat")
        assert worker.stdin is not None
        worker.stdin.write("go\n")
        worker.stdin.flush()
        worker.wait(timeout=5)
        after = _read_key_values(path / "cpu.stat")
        return summarize_cpu_effect(before=before, after=after)
    except Exception as exc:
        return {
            "accepted": False,
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if worker is not None:
            _kill_process_tree(worker)
        manager.cleanup(path)


def run_pids_effect(root: Path) -> dict[str, Any]:
    manager, path = _scenario_path(
        root,
        "pids",
        ResourceQuota(memory_max_mb=128, pids_max=8),
    )
    worker: subprocess.Popen[str] | None = None
    try:
        program = (
            "import errno,json,subprocess,sys,time\n"
            "print('ready', flush=True)\n"
            "sys.stdin.readline()\n"
            "children=[]; failures=[]; attempted=20\n"
            "for _ in range(attempted):\n"
            "    try:\n"
            "        children.append(subprocess.Popen([sys.executable,'-c','import time; time.sleep(5)']))\n"
            "    except OSError as exc:\n"
            "        failures.append(errno.errorcode.get(exc.errno, str(exc.errno)))\n"
            "        break\n"
            "print(json.dumps({'attempted': attempted, 'started': len(children), 'failures': failures}), flush=True)\n"
            "for child in children:\n"
            "    child.terminate()\n"
            "for child in children:\n"
            "    try: child.wait(timeout=1)\n"
            "    except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=1)\n"
        )
        worker = _start_worker(program)
        _wait_ready(worker)
        attached = manager.add_pid(path, worker.pid)
        if not attached.available:
            raise RuntimeError(f"pids worker attach failed: {attached.reason}")
        before = _read_key_values(path / "pids.events")
        assert worker.stdin is not None and worker.stdout is not None
        worker.stdin.write("go\n")
        worker.stdin.flush()
        payload = json.loads(worker.stdout.readline())
        worker.wait(timeout=10)
        after = _read_key_values(path / "pids.events")
        return summarize_pids_effect(
            attempted=payload["attempted"],
            started=payload["started"],
            failures=payload["failures"],
            before=before,
            after=after,
        )
    except Exception as exc:
        return {
            "accepted": False,
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if worker is not None:
            _kill_process_tree(worker)
        manager.cleanup(path)


def run_memory_effect(root: Path) -> dict[str, Any]:
    manager, path = _scenario_path(
        root,
        "memory",
        ResourceQuota(memory_max_mb=64, pids_max=16),
    )
    worker: subprocess.Popen[str] | None = None
    try:
        program = (
            "import sys,time\n"
            "print('ready', flush=True)\n"
            "sys.stdin.readline()\n"
            "blocks=[]\n"
            "for _ in range(256):\n"
            "    blocks.append(bytearray(1024*1024))\n"
            "    time.sleep(0.005)\n"
            "print(len(blocks), flush=True)\n"
        )
        worker = _start_worker(program)
        _wait_ready(worker)
        attached = manager.add_pid(path, worker.pid)
        if not attached.available:
            raise RuntimeError(f"memory worker attach failed: {attached.reason}")
        before = _read_key_values(path / "memory.events")
        assert worker.stdin is not None
        worker.stdin.write("go\n")
        worker.stdin.flush()
        try:
            worker.wait(timeout=20)
        except subprocess.TimeoutExpired:
            _kill_process_tree(worker)
        after = _read_key_values(path / "memory.events")
        return summarize_memory_effect(before=before, after=after)
    except Exception as exc:
        return {
            "accepted": False,
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if worker is not None:
            _kill_process_tree(worker)
        manager.cleanup(path)


def run_verification(root: Path) -> dict[str, Any]:
    mode, allowed = detect_delegation()
    if not allowed:
        return {
            "accepted": False,
            "cpu": {"accepted": False},
            "pids": {"accepted": False},
            "memory": {"accepted": False},
            "cleanup_failed": 0,
            "delegation_mode": mode,
            "skip_reason": f"delegation={mode}; set AGENTOS_CGROUP_ALLOW_ROOT=1 to override",
            "failure_reason": None,
        }

    cleanup_failed = 0
    failure_reason = None
    root.mkdir(parents=True, exist_ok=True)
    try:
        _enable_parent_controllers(root)
    except Exception as exc:
        failure_reason = f"controller enable failed: {type(exc).__name__}: {exc}"

    if failure_reason is None:
        cpu = run_cpu_effect(root)
        pids = run_pids_effect(root)
        memory = run_memory_effect(root)
    else:
        cpu = {"accepted": False}
        pids = {"accepted": False}
        memory = {"accepted": False}

    try:
        root.rmdir()
    except OSError as exc:
        cleanup_failed += 1
        if failure_reason is None:
            failure_reason = f"root cleanup failed: {exc}"

    summary = {
        "accepted": False,
        "cpu": cpu,
        "pids": pids,
        "memory": memory,
        "cleanup_failed": cleanup_failed,
        "delegation_mode": mode,
        "skip_reason": None,
        "failure_reason": failure_reason,
    }
    summary["accepted"] = acceptance_passed(summary)
    return summary


def _write_outputs(summary: dict[str, Any], out: str | None) -> None:
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if out is None:
        return
    output = Path(out)
    if output.suffix == ".json":
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        return
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(text + "\n", encoding="utf-8")
    for name in ("cpu", "pids", "memory"):
        (output / f"{name}.json").write_text(
            json.dumps(summary.get(name, {}), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-isolated-vm", action="store_true")
    parser.add_argument("--root", default="/sys/fs/cgroup/agentos_member_a_effects")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if not args.confirm_isolated_vm:
        summary = {
            "accepted": False,
            "cpu": {"accepted": False},
            "pids": {"accepted": False},
            "memory": {"accepted": False},
            "cleanup_failed": 0,
            "skip_reason": "isolated VM not confirmed",
            "failure_reason": None,
        }
        _write_outputs(summary, args.out)
        return 2

    summary = run_verification(Path(args.root))
    _write_outputs(summary, args.out)
    if summary.get("skip_reason"):
        return 2
    return 0 if acceptance_passed(summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
