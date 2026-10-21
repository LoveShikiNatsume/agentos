#!/usr/bin/env python3
"""Validate that the host is ready to build and load AgentOS sched_ext."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


MINIMUM_KERNEL = (6, 12)
MINIMUM_LLVM = (17, 0)
REQUIRED_CONFIG = (
    "CONFIG_BPF",
    "CONFIG_BPF_SYSCALL",
    "CONFIG_BPF_JIT",
    "CONFIG_BPF_JIT_ALWAYS_ON",
    "CONFIG_BPF_JIT_DEFAULT_ON",
    "CONFIG_DEBUG_INFO_BTF",
    "CONFIG_SCHED_CLASS_EXT",
    "CONFIG_CGROUPS",
    "CONFIG_CGROUP_BPF",
    "CONFIG_PSI",
)
REQUIRED_CONTROLLERS = ("cpu", "cpuset", "io", "memory", "pids")
REQUIRED_PSI = ("cpu", "memory", "io")

CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
CommandFinder = Callable[[str], str | None]


@dataclass(frozen=True)
class ProbeEnvironment:
    root: Path = Path("/")
    kernel_release: str | None = None
    euid: int | None = None

    def path(self, absolute: str) -> Path:
        return self.root / absolute.lstrip("/")


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )


def _version(value: str) -> tuple[int, ...] | None:
    match = re.search(r"(?<!\d)(\d+)\.(\d+)(?:\.(\d+))?", value)
    if not match:
        return None
    return tuple(int(item) for item in match.groups(default="0"))


def _version_at_least(actual: tuple[int, ...] | None, minimum: tuple[int, ...]) -> bool:
    if actual is None:
        return False
    width = max(len(actual), len(minimum))
    return actual + (0,) * (width - len(actual)) >= minimum + (0,) * (width - len(minimum))


def _read_text(path: Path) -> tuple[str | None, str | None]:
    try:
        return path.read_text(encoding="utf-8"), None
    except OSError as exc:
        return None, str(exc)


def _parse_kernel_config(text: str) -> dict[str, str]:
    config: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        config[key] = value
    return config


def _mount_type(mountinfo: str, target: str) -> str | None:
    for line in mountinfo.splitlines():
        before, separator, after = line.partition(" - ")
        fields = before.split()
        after_fields = after.split()
        if separator and len(fields) >= 5 and len(after_fields) >= 1 and fields[4] == target:
            return after_fields[0]
    return None


def _check_tool(
    name: str,
    version_args: Sequence[str],
    *,
    finder: CommandFinder,
    runner: CommandRunner,
    minimum: tuple[int, ...] | None = None,
) -> dict[str, object]:
    executable = finder(name)
    if executable is None:
        return {
            "accepted": False,
            "path": None,
            "version": None,
            "error": "command not found",
        }

    try:
        completed = runner((executable, *version_args))
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "accepted": False,
            "path": executable,
            "version": None,
            "error": str(exc),
        }

    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
    parsed = _version(output)
    accepted = completed.returncode == 0 and (minimum is None or _version_at_least(parsed, minimum))
    return {
        "accepted": accepted,
        "path": executable,
        "version": ".".join(str(item) for item in parsed) if parsed else None,
        "minimum": ".".join(str(item) for item in minimum) if minimum else None,
        "error": None if accepted else output or f"exit code {completed.returncode}",
    }


def probe_sched_ext(
    environment: ProbeEnvironment | None = None,
    *,
    finder: CommandFinder = shutil.which,
    runner: CommandRunner = _run,
) -> dict[str, object]:
    env = environment or ProbeEnvironment()
    release = env.kernel_release or platform.release()
    euid = os.geteuid() if env.euid is None else env.euid
    failures: list[str] = []

    kernel_version = _version(release)
    kernel_ok = _version_at_least(kernel_version, MINIMUM_KERNEL)
    if not kernel_ok:
        failures.append("kernel_version_too_old")

    config_path = env.path(f"/boot/config-{release}")
    config_text, config_error = _read_text(config_path)
    config = _parse_kernel_config(config_text or "")
    missing_config = [name for name in REQUIRED_CONFIG if config.get(name) != "y"]
    config_ok = config_text is not None and not missing_config
    if config_text is None:
        failures.append("kernel_config_unreadable")
    elif missing_config:
        failures.append("kernel_config_missing")

    btf_path = env.path("/sys/kernel/btf/vmlinux")
    btf_ok = btf_path.is_file() and os.access(btf_path, os.R_OK)
    if not btf_ok:
        failures.append("kernel_btf_missing")

    sched_ext_dir = env.path("/sys/kernel/sched_ext")
    state_text, state_error = _read_text(sched_ext_dir / "state")
    sequence_text, sequence_error = _read_text(sched_ext_dir / "enable_seq")
    state = state_text.strip() if state_text else None
    try:
        enable_seq = int(sequence_text.strip()) if sequence_text else None
    except ValueError:
        enable_seq = None
    sched_ext_ok = sched_ext_dir.is_dir() and state == "disabled" and enable_seq is not None and enable_seq >= 0
    if not sched_ext_dir.is_dir() or state is None:
        failures.append("sched_ext_unavailable")
    elif state != "disabled":
        failures.append("sched_ext_busy")
    elif enable_seq is None or enable_seq < 0:
        failures.append("sched_ext_enable_seq_invalid")

    mountinfo_text, mountinfo_error = _read_text(env.path("/proc/self/mountinfo"))
    mountinfo = mountinfo_text or ""
    bpffs_type = _mount_type(mountinfo, "/sys/fs/bpf")
    bpffs_ok = bpffs_type == "bpf"
    if not bpffs_ok:
        failures.append("bpffs_not_mounted")

    cgroup_type = _mount_type(mountinfo, "/sys/fs/cgroup")
    controllers_text, controllers_error = _read_text(env.path("/sys/fs/cgroup/cgroup.controllers"))
    controllers = sorted(set((controllers_text or "").split()))
    missing_controllers = [name for name in REQUIRED_CONTROLLERS if name not in controllers]
    cgroup_ok = cgroup_type == "cgroup2" and not missing_controllers
    if cgroup_type != "cgroup2":
        failures.append("cgroup_v2_not_mounted")
    elif missing_controllers:
        failures.append("cgroup_v2_controllers_missing")

    psi_paths = {name: env.path(f"/proc/pressure/{name}") for name in REQUIRED_PSI}
    missing_psi = [name for name, path in psi_paths.items() if not path.is_file() or not os.access(path, os.R_OK)]
    psi_ok = not missing_psi
    if missing_psi:
        failures.append("psi_unavailable")

    root_ok = euid == 0
    if not root_ok:
        failures.append("insufficient_privileges")

    tools = {
        "bpftool": _check_tool("bpftool", ("version",), finder=finder, runner=runner),
        "clang": _check_tool("clang", ("--version",), finder=finder, runner=runner, minimum=MINIMUM_LLVM),
        "llvm-config": _check_tool(
            "llvm-config",
            ("--version",),
            finder=finder,
            runner=runner,
            minimum=MINIMUM_LLVM,
        ),
        "gcc": _check_tool("gcc", ("--version",), finder=finder, runner=runner),
        "make": _check_tool("make", ("--version",), finder=finder, runner=runner),
    }
    missing_tools = [name for name, result in tools.items() if result["accepted"] is not True]
    if missing_tools:
        failures.append("required_tools_unavailable")

    result: dict[str, object] = {
        "schema_version": 1,
        "accepted": not failures,
        "failures": failures,
        "kernel": {
            "accepted": kernel_ok,
            "release": release,
            "version": ".".join(str(item) for item in kernel_version) if kernel_version else None,
            "minimum": ".".join(str(item) for item in MINIMUM_KERNEL),
        },
        "kernel_config": {
            "accepted": config_ok,
            "path": str(config_path),
            "required": list(REQUIRED_CONFIG),
            "missing": missing_config,
            "error": config_error,
        },
        "btf": {"accepted": btf_ok, "path": str(btf_path)},
        "sched_ext": {
            "accepted": sched_ext_ok,
            "path": str(sched_ext_dir),
            "state": state,
            "enable_seq": enable_seq,
            "state_error": state_error,
            "enable_seq_error": sequence_error,
        },
        "bpffs": {
            "accepted": bpffs_ok,
            "path": "/sys/fs/bpf",
            "filesystem": bpffs_type,
            "mountinfo_error": mountinfo_error,
        },
        "cgroup_v2": {
            "accepted": cgroup_ok,
            "path": "/sys/fs/cgroup",
            "filesystem": cgroup_type,
            "controllers": controllers,
            "missing_controllers": missing_controllers,
            "controllers_error": controllers_error,
        },
        "psi": {
            "accepted": psi_ok,
            "required": list(REQUIRED_PSI),
            "missing": missing_psi,
        },
        "privileges": {
            "accepted": root_ok,
            "euid": euid,
            "requirement": "run as root",
        },
        "tools": tools,
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write the JSON report to this path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = probe_sched_ext()
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    sys.stdout.write(output)
    return 0 if result["accepted"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
