#!/usr/bin/env python3
"""Preflight checks for Member A's real cgroup/BPF Runtime path."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentos.isolation.delegation import detect_delegation


def _read_os_release(path: Path = Path("/etc/os-release")) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    result: dict[str, str] = {}
    for line in lines:
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        result[key] = value.strip().strip('"')
    return result


def _kernel_release() -> str:
    return platform.release()


def _path_exists(path: Path) -> bool:
    return path.exists()


def _path_writable(path: Path) -> bool:
    return path.exists() and path.is_dir() and shutil.os.access(path, shutil.os.W_OK)


def _bpffs_mounted(path: Path = Path("/sys/fs/bpf")) -> bool:
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
        target = path.resolve()
    except OSError:
        return False
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
        if mountpoint == target and fs_fields[0] == "bpf":
            return True
    return False


def _cap_bpf(path: Path = Path("/proc/self/status")) -> tuple[bool, str | None]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return False, f"cannot read {path}: {exc}"
    for line in text.splitlines():
        if not line.startswith("CapEff:"):
            continue
        raw = line.split(":", 1)[1].strip()
        try:
            return bool(int(raw, 16) & (1 << 39)), None
        except ValueError:
            return False, f"invalid CapEff value: {raw}"
    return False, "CapEff not found"


def _psi_status(root: Path = Path("/proc/pressure")) -> dict[str, Any]:
    files = {
        name: {
            "path": str(root / name),
            "present": _path_exists(root / name),
        }
        for name in ("cpu", "memory", "io")
    }
    return {
        "available": all(item["present"] for item in files.values()),
        "files": files,
    }


def _recommended_bpf_mode(
    *,
    bpffs_mounted: bool,
    bpftool_available: bool,
    cap_bpf: bool,
) -> tuple[str, bool]:
    if bpffs_mounted and bpftool_available and cap_bpf:
        return "bpftool", False
    return "auto", True


def run_preflight(
    *,
    output_path: str | Path | None = None,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    bpffs_root: Path = Path("/sys/fs/bpf"),
    pressure_root: Path = Path("/proc/pressure"),
) -> dict[str, Any]:
    os_release = _read_os_release(Path("/etc/os-release"))
    os_id = os_release.get("ID", "")
    pretty_name = os_release.get("PRETTY_NAME", os_release.get("NAME", "unknown"))
    delegation_mode, delegation_allowed = detect_delegation()

    cgroup_v2_present = _path_exists(cgroup_root / "cgroup.controllers") or _path_exists(
        cgroup_root / "cgroup.procs"
    )
    cgroup_writable = _path_writable(cgroup_root)
    psi = _psi_status(pressure_root)
    bpftool_path = shutil.which("bpftool")
    cap_bpf, cap_bpf_reason = _cap_bpf(Path("/proc/self/status"))
    mounted = _bpffs_mounted(bpffs_root)
    bpf_mode, fallback_expected = _recommended_bpf_mode(
        bpffs_mounted=mounted,
        bpftool_available=bpftool_path is not None,
        cap_bpf=cap_bpf,
    )

    missing: list[str] = []
    if not cgroup_v2_present:
        missing.append("cgroup v2 not detected")
    if not delegation_allowed:
        missing.append("cgroup delegation not available")
    if not cgroup_writable:
        missing.append("cgroup root not writable")
    if not psi["available"]:
        missing.append("PSI files missing")
    if not mounted:
        missing.append("bpffs not mounted")
    if bpftool_path is None:
        missing.append("bpftool not found")
    if not cap_bpf:
        missing.append(cap_bpf_reason or "CAP_BPF missing")

    result = {
        "os": {
            "pretty_name": pretty_name,
            "id": os_id,
            "kernel": _kernel_release(),
        },
        "is_openeuler": os_id.lower() == "openeuler"
        or "openeuler" in pretty_name.lower(),
        "delegation": {
            "mode": delegation_mode,
            "allowed": delegation_allowed,
        },
        "cgroup_v2": {
            "path": str(cgroup_root),
            "present": cgroup_v2_present,
            "writable": cgroup_writable,
            "available": cgroup_v2_present and delegation_allowed and cgroup_writable,
        },
        "psi": psi,
        "bpf": {
            "bpffs_path": str(bpffs_root),
            "bpffs_mounted": mounted,
            "bpftool_path": bpftool_path,
            "bpftool_available": bpftool_path is not None,
            "cap_bpf": cap_bpf,
            "cap_bpf_reason": cap_bpf_reason,
        },
        "recommended_bpf_mode": bpf_mode,
        "fallback_expected": fallback_expected,
        "missing": missing,
    }

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)
    result = run_preflight(output_path=args.json_out)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
