from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Sequence


def _runtime_observation() -> dict[str, Any]:
    ext_enabled = -1
    try:
        with open("/proc/self/sched", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if line.strip().startswith("ext.enabled"):
                    ext_enabled = int(line.rsplit(":", 1)[1].strip())
                    break
    except (OSError, ValueError):
        pass
    return {
        "policy": os.sched_getscheduler(0),
        "ext_enabled": ext_enabled,
        "affinity": sorted(os.sched_getaffinity(0)),
    }


def _clamp_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _clamp_float(value: Any, *, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


async def _apply_memory_pressure(fault: dict[str, Any]) -> None:
    memory_mb = _clamp_int(
        fault.get("memory_mb"),
        default=8,
        minimum=1,
        maximum=256,
    )
    chunk_mb = _clamp_int(
        fault.get("chunk_mb"),
        default=1,
        minimum=1,
        maximum=memory_mb,
    )
    hold_seconds = _clamp_float(
        fault.get("hold_seconds"),
        default=0.05,
        minimum=0.0,
        maximum=5.0,
    )

    remaining = memory_mb
    blocks: list[bytearray] = []
    while remaining > 0:
        current_mb = min(chunk_mb, remaining)
        blocks.append(bytearray(current_mb * 1024 * 1024))
        remaining -= current_mb
        await asyncio.sleep(0)
    if hold_seconds:
        await asyncio.sleep(hold_seconds)
    raise MemoryError(f"injected memory pressure fault allocated {memory_mb} MiB")


async def _run_worker(payload: dict[str, Any]) -> dict[str, Any]:
    worker_type = str(payload.get("worker_type", "echo"))
    task_id = str(payload.get("task_id", ""))
    agent_id = str(payload.get("agent_id", ""))
    trace_id = payload.get("trace_id")
    worker_payload = dict(payload.get("payload") or {})
    attempt = int(payload.get("attempt", 0))

    if worker_type == "bad-json":
        print("{not-json", end="")
        raise SystemExit(0)

    try:
        fault = payload.get("fault")
        if isinstance(fault, dict) and fault.get("fault_type") == "memory_pressure":
            await _apply_memory_pressure(fault)

        if worker_type == "echo":
            result = worker_payload.get("result", worker_payload.get("message", "ok"))
        elif worker_type == "sleep":
            seconds = float(worker_payload.get("seconds", 1.0))
            await asyncio.sleep(seconds)
            result = worker_payload.get("result", f"slept {seconds}s")
        elif worker_type == "fail":
            raise RuntimeError(str(worker_payload.get("error", "worker failed")))
        elif worker_type == "fork":
            child_pid = os.fork()
            if child_pid == 0:
                os._exit(0)
            waited_pid, status = os.waitpid(child_pid, 0)
            result = {
                "child_pid": waited_pid,
                "child_exit_code": os.waitstatus_to_exitcode(status),
            }
        elif worker_type == "flaky":
            if attempt == 0:
                raise RuntimeError(str(worker_payload.get("error", "temporary worker failure")))
            result = worker_payload.get("result", "recovered")
        elif worker_type in {
            "artifact_hash",
            "dependency_analyze",
            "docs_scan",
            "patch_validate",
            "repo_index",
            "security_scan",
            "symbol_index",
            "test_select",
        }:
            from agentos.scenarios.sched_ext_business.workers import (
                run_business_worker,
            )

            result = {
                **run_business_worker(worker_type, worker_payload),
                "runtime_observation": _runtime_observation(),
            }
        else:
            raise RuntimeError(f"unsupported worker_type: {worker_type}")

        return {
            "ok": True,
            "worker_type": worker_type,
            "task_id": task_id,
            "agent_id": agent_id,
            "trace_id": trace_id,
            "result": result,
        }
    except Exception as exc:
        return {
            "ok": False,
            "worker_type": worker_type,
            "task_id": task_id,
            "agent_id": agent_id,
            "trace_id": trace_id,
            "error": str(exc),
        }


async def _main_async() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        print("agentos.worker.entrypoint ready")
        return 0

    payload = json.loads(raw)
    result = await _run_worker(payload)
    print(json.dumps(result, ensure_ascii=False), end="")
    return 0 if result.get("ok") else 1


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    return asyncio.run(_main_async())


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
