from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

from scripts import preflight_sched_ext as preflight


RELEASE = "6.12.95-agentos-scx"


def _write(root: Path, absolute: str, text: str = "") -> None:
    path = root / absolute.lstrip("/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _accepted_root(tmp_path: Path) -> Path:
    config = "\n".join(f"{name}=y" for name in preflight.REQUIRED_CONFIG) + "\n"
    _write(tmp_path, f"/boot/config-{RELEASE}", config)
    _write(tmp_path, "/sys/kernel/btf/vmlinux", "btf")
    _write(tmp_path, "/sys/kernel/sched_ext/state", "disabled\n")
    _write(tmp_path, "/sys/kernel/sched_ext/enable_seq", "5\n")
    _write(
        tmp_path,
        "/proc/self/mountinfo",
        "20 1 0:20 / /sys/fs/bpf rw - bpf bpf rw\n"
        "21 1 0:21 / /sys/fs/cgroup rw - cgroup2 cgroup2 rw\n",
    )
    _write(
        tmp_path,
        "/sys/fs/cgroup/cgroup.controllers",
        "cpuset cpu io memory hugetlb pids rdma misc\n",
    )
    for resource in preflight.REQUIRED_PSI:
        _write(tmp_path, f"/proc/pressure/{resource}", "some avg10=0.00 total=0\n")
    return tmp_path


def _finder(name: str) -> str:
    return f"/usr/bin/{name}"


def _runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    name = Path(command[0]).name
    versions = {
        "bpftool": "bpftool v7.2.0",
        "clang": "clang version 17.0.6",
        "llvm-config": "17.0.6",
        "gcc": "gcc (GCC) 12.3.1",
        "make": "GNU Make 4.4",
    }
    return subprocess.CompletedProcess(command, 0, versions[name], "")


def _probe(tmp_path: Path, **overrides: object) -> dict[str, object]:
    environment = preflight.ProbeEnvironment(
        root=tmp_path,
        kernel_release=str(overrides.get("kernel_release", RELEASE)),
        euid=int(overrides.get("euid", 0)),
    )
    return preflight.probe_sched_ext(environment, finder=_finder, runner=_runner)


def test_accepts_complete_sched_ext_environment(tmp_path: Path) -> None:
    _accepted_root(tmp_path)

    result = _probe(tmp_path)

    assert result["accepted"] is True
    assert result["failures"] == []
    assert result["kernel"]["release"] == RELEASE
    assert result["sched_ext"]["state"] == "disabled"
    assert result["sched_ext"]["enable_seq"] == 5
    assert result["cgroup_v2"]["accepted"] is True
    assert all(tool["accepted"] for tool in result["tools"].values())


def test_rejects_old_kernel_and_missing_config(tmp_path: Path) -> None:
    _accepted_root(tmp_path)
    old_release = "6.6.0-openeuler"
    _write(tmp_path, f"/boot/config-{old_release}", "CONFIG_BPF=y\n")

    result = _probe(tmp_path, kernel_release=old_release)

    assert result["accepted"] is False
    assert "kernel_version_too_old" in result["failures"]
    assert "kernel_config_missing" in result["failures"]
    assert "CONFIG_SCHED_CLASS_EXT" in result["kernel_config"]["missing"]


def test_rejects_missing_btf_and_psi(tmp_path: Path) -> None:
    _accepted_root(tmp_path)
    (tmp_path / "sys/kernel/btf/vmlinux").unlink()
    (tmp_path / "proc/pressure/io").unlink()

    result = _probe(tmp_path)

    assert result["accepted"] is False
    assert "kernel_btf_missing" in result["failures"]
    assert "psi_unavailable" in result["failures"]
    assert result["psi"]["missing"] == ["io"]


def test_rejects_legacy_cgroup_and_missing_bpffs(tmp_path: Path) -> None:
    _accepted_root(tmp_path)
    _write(
        tmp_path,
        "/proc/self/mountinfo",
        "21 1 0:21 / /sys/fs/cgroup rw - tmpfs tmpfs rw\n",
    )

    result = _probe(tmp_path)

    assert result["accepted"] is False
    assert "bpffs_not_mounted" in result["failures"]
    assert "cgroup_v2_not_mounted" in result["failures"]


def test_rejects_busy_scheduler_and_non_root_user(tmp_path: Path) -> None:
    _accepted_root(tmp_path)
    _write(tmp_path, "/sys/kernel/sched_ext/state", "enabled\n")

    result = _probe(tmp_path, euid=1000)

    assert result["accepted"] is False
    assert "sched_ext_busy" in result["failures"]
    assert "insufficient_privileges" in result["failures"]


def test_rejects_missing_or_old_tools(tmp_path: Path) -> None:
    _accepted_root(tmp_path)

    def finder(name: str) -> str | None:
        return None if name == "bpftool" else f"/usr/bin/{name}"

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if Path(command[0]).name == "clang":
            return subprocess.CompletedProcess(command, 0, "clang version 16.0.0", "")
        return _runner(command)

    environment = preflight.ProbeEnvironment(root=tmp_path, kernel_release=RELEASE, euid=0)
    result = preflight.probe_sched_ext(environment, finder=finder, runner=runner)

    assert result["accepted"] is False
    assert "required_tools_unavailable" in result["failures"]
    assert result["tools"]["bpftool"]["accepted"] is False
    assert result["tools"]["clang"]["accepted"] is False
