#!/usr/bin/env python3
"""Privileged verifier for the real bpftool pinned-map path."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentos.isolation.bpf import BPFTaskRecord
from agentos.isolation.bpf_schema import BPF_TASK_SCHEMA_VERSION
from agentos.isolation.bpftool import (
    BPFMapError,
    BpftoolPinnedMapAdapter,
)


def _empty_result() -> dict[str, Any]:
    return {
        "available": False,
        "bpftool_version": None,
        "bpffs_mounted": False,
        "map_created": False,
        "schema_verified": False,
        "schema_version": BPF_TASK_SCHEMA_VERSION,
        "updates": 0,
        "lookups": 0,
        "readback_verified": False,
        "dumps": 0,
        "cleanup_failed": 0,
        "pin_path": None,
        "skip_reason": None,
        "failure_reason": None,
    }


def acceptance_passed(result: dict[str, Any]) -> bool:
    return bool(
        result.get("available") is True
        and result.get("map_created") is True
        and result.get("schema_verified") is True
        and result.get("updates", 0) >= 1
        and result.get("lookups", 0) >= 1
        and result.get("readback_verified") is True
        and result.get("dumps", 0) >= 1
        and result.get("cleanup_failed", 1) == 0
        and result.get("skip_reason") is None
        and result.get("failure_reason") is None
    )


def probe_environment(
    *,
    bpffs_root: Path,
    bpftool_path: str | None = None,
) -> dict[str, Any]:
    binary = bpftool_path or shutil.which("bpftool")
    if binary is None:
        return {
            "available": False,
            "bpftool_version": None,
            "bpffs_mounted": False,
            "binary": None,
            "reason": "bpftool not found in PATH",
        }

    try:
        completed = subprocess.run(
            [binary, "version"],
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "available": False,
            "bpftool_version": None,
            "bpffs_mounted": False,
            "binary": binary,
            "reason": f"bpftool version failed: {exc}",
        }
    if completed.returncode != 0:
        return {
            "available": False,
            "bpftool_version": None,
            "bpffs_mounted": False,
            "binary": binary,
            "reason": (
                f"bpftool version exited with code {completed.returncode}: "
                f"{(completed.stderr or '').strip()[:300]}"
            ),
        }

    version = (completed.stdout or "").strip().splitlines()
    version_text = version[0] if version else "unknown"
    mounted = _filesystem_type(bpffs_root) == "bpf"
    if not mounted:
        return {
            "available": False,
            "bpftool_version": version_text,
            "bpffs_mounted": False,
            "binary": binary,
            "reason": f"bpffs is not mounted at {bpffs_root}",
        }
    if not os.access(bpffs_root, os.W_OK):
        return {
            "available": False,
            "bpftool_version": version_text,
            "bpffs_mounted": True,
            "binary": binary,
            "reason": f"bpffs is not writable at {bpffs_root}",
        }
    return {
        "available": True,
        "bpftool_version": version_text,
        "bpffs_mounted": True,
        "binary": binary,
        "reason": None,
    }


def _filesystem_type(path: Path) -> str | None:
    try:
        resolved = path.resolve()
        lines = Path("/proc/self/mountinfo").read_text(
            encoding="utf-8"
        ).splitlines()
    except OSError:
        return None

    best_length = -1
    best_type: str | None = None
    for line in lines:
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        fields = before.split()
        fs_fields = after.split()
        if len(fields) < 5 or not fs_fields:
            continue
        mountpoint = Path(
            fields[4]
            .replace("\\040", " ")
            .replace("\\011", "\t")
            .replace("\\134", "\\")
        )
        try:
            resolved.relative_to(mountpoint)
        except ValueError:
            continue
        mount_length = len(str(mountpoint))
        if mount_length > best_length:
            best_length = mount_length
            best_type = fs_fields[0]
    return best_type


def _verification_record() -> BPFTaskRecord:
    return BPFTaskRecord(
        task_id="stage3-verifier-task",
        workload_class="ai_inference",
        policy="latency_first",
        fairness_score=0.875,
        starvation_risk="low",
        selected=True,
        cpu_weight=512,
        cpu_max_percent=50.0,
        memory_max_mb=128,
        pids_max=32,
        pressure_level="warning",
        backpressure_signal="throttle",
        updated_at=123.456,
    )


def run_verification(
    *,
    bpffs_root: Path,
    bpftool_path: str | None = None,
) -> dict[str, Any]:
    result = _empty_result()
    probe = probe_environment(
        bpffs_root=bpffs_root,
        bpftool_path=bpftool_path,
    )
    result["bpftool_version"] = probe.get("bpftool_version")
    result["bpffs_mounted"] = probe.get("bpffs_mounted", False)
    if not probe.get("available"):
        result["skip_reason"] = probe.get("reason") or "BPF unavailable"
        return result

    pin_path = (
        bpffs_root / "agentos" / f"tasks_verify_{os.getpid()}"
    )
    result["pin_path"] = str(pin_path)
    if pin_path.exists():
        result["failure_reason"] = (
            f"refusing to reuse existing verifier pin: {pin_path}"
        )
        return result

    adapter = BpftoolPinnedMapAdapter(
        pin_path=pin_path,
        bpftool_path=str(probe["binary"]),
        owns_pin=True,
    )
    record = _verification_record()

    try:
        created = adapter.create_if_missing()
        result["map_created"] = created.get("created") is True
        result["schema_verified"] = (
            created.get("schema_verified") is True
        )
        result["available"] = True

        adapter.update_task(record)
        result["updates"] = 1

        looked_up = adapter.get_task(record.task_id)
        result["lookups"] = 1
        if looked_up is None:
            raise BPFMapError("lookup returned no record")
        decoded = looked_up["decoded"]
        result["readback_verified"] = bool(
            decoded.get("version") == BPF_TASK_SCHEMA_VERSION
            and decoded.get("selected") is True
            and decoded.get("cpu_weight") == 512
            and decoded.get("cpu_max_milli_percent") == 50_000
            and decoded.get("memory_max_bytes") == 128 * 1024 * 1024
            and decoded.get("pids_max") == 32
            and decoded.get("fairness_micros") == 875_000
        )
        if not result["readback_verified"]:
            raise BPFMapError(
                f"lookup readback mismatch: {decoded!r}"
            )

        dumped = adapter.snapshot()
        result["dumps"] = 1
        if record.task_id not in dumped:
            raise BPFMapError("dump did not contain verifier task")
    except Exception as exc:
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            adapter.clear()
        except Exception as exc:
            result["cleanup_failed"] += 1
            if result["failure_reason"] is None:
                result["failure_reason"] = (
                    f"cleanup failed: {type(exc).__name__}: {exc}"
                )
        if pin_path.exists():
            result["cleanup_failed"] += 1
            if result["failure_reason"] is None:
                result["failure_reason"] = (
                    f"verifier pin still exists after cleanup: {pin_path}"
                )

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
    parser.add_argument("--bpffs-root", default="/sys/fs/bpf")
    parser.add_argument("--bpftool-path", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    result = run_verification(
        bpffs_root=Path(args.bpffs_root),
        bpftool_path=args.bpftool_path,
    )
    _out(result, args.json_out)
    if result.get("skip_reason"):
        return 2
    return 0 if acceptance_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
