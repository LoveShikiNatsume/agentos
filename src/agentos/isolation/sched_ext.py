"""sched_ext / eBPF capability probe.

This module detects whether the host kernel supports sched_ext and BPF maps.
When unsupported or running without root privileges, it falls back to mock
mode so that tests and development environments remain stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentos.isolation.bpftool import probe_bpftool, probe_pinned_map


@dataclass
class KernelProbeResult:
    """Result of a sched_ext / BPF capability probe."""

    sched_ext: str  # "available" | "unavailable"
    bpf_map: str  # "mock" | "pinned_probe" | "unavailable"
    fallback_mode: str  # "none" | "mock_bpf_map" | "probe_only"
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sched_ext": self.sched_ext,
            "bpf_map": self.bpf_map,
            "fallback_mode": self.fallback_mode,
            "reason": self.reason,
            "details": dict(self.details),
        }


def _check_bpffs_mounted(proc_mounts_path: Path, bpffs_path: Path) -> bool:
    """Check if bpffs is actually mounted by parsing /proc/mounts.

    Returns True only if the mountpoint corresponds to filesystem type 'bpf'.
    """
    try:
        text = proc_mounts_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False

    target = str(bpffs_path)
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            _device, mountpoint, fstype = parts[0], parts[1], parts[2]
            if mountpoint == target and fstype == "bpf":
                return True
    return False


def _check_psi_files(proc_pressure_path: Path) -> dict[str, Any]:
    """Check PSI (Pressure Stall Information) file availability."""
    psi_files = {"cpu": "cpu", "memory": "memory", "io": "io"}
    result: dict[str, Any] = {
        "psi_path": str(proc_pressure_path),
        "psi_available": False,
        "psi_files": {},
    }
    all_present = True
    for name, filename in psi_files.items():
        fpath = proc_pressure_path / filename
        exists = fpath.exists()
        result["psi_files"][name] = {"path": str(fpath), "present": exists}
        if not exists:
            all_present = False
    result["psi_available"] = all_present
    return result


def _check_cap_bpf(proc_status_path: Path) -> dict[str, Any]:
    """Check CAP_BPF (bit 39) from /proc/self/status CapEff."""
    result: dict[str, Any] = {"cap_bpf": False, "reason": None}
    try:
        text = proc_status_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        result["reason"] = f"cannot read {proc_status_path}: {exc}"
        return result

    for line in text.splitlines():
        if line.startswith("CapEff:"):
            cap_hex = line.split(":", 1)[1].strip()
            try:
                cap_val = int(cap_hex, 16)
                CAP_BPF_BIT = 1 << 39
                result["cap_bpf"] = bool(cap_val & CAP_BPF_BIT)
            except ValueError:
                result["reason"] = f"invalid CapEff value: {cap_hex}"
            return result

    result["reason"] = "CapEff not found in status file"
    return result


def _check_cgroup_v2(cgroup_path: Path) -> dict[str, Any]:
    """Check cgroup v2 presence."""
    controllers_path = cgroup_path / "cgroup.controllers"
    if controllers_path.exists():
        return {
            "cgroup_v2_present": True,
            "cgroup_v2_path": str(cgroup_path),
        }
    # Fallback: check for cgroup.procs in unified hierarchy
    procs_path = cgroup_path / "cgroup.procs"
    if procs_path.exists():
        return {
            "cgroup_v2_present": True,
            "cgroup_v2_path": str(cgroup_path),
        }
    return {
        "cgroup_v2_present": False,
        "cgroup_v2_path": str(cgroup_path),
    }


def probe_sched_ext(
    *,
    sys_kernel_path: Path | str = "/sys/kernel",
    sys_fs_bpf_path: Path | str = "/sys/fs/bpf",
    proc_pressure_path: Path | str = "/proc/pressure",
    proc_mounts_path: Path | str = "/proc/mounts",
    proc_status_path: Path | str = "/proc/self/status",
    cgroup_path: Path | str = "/sys/fs/cgroup",
    bpftool_path: str | None = None,
    pinned_map_path: str | Path | None = None,
    force_mock: bool = False,
) -> KernelProbeResult:
    """Probe the host kernel for sched_ext / BPF capabilities.

    Parameters
    ----------
    sys_kernel_path:
        Path to ``/sys/kernel`` (or a mock for testing).
    sys_fs_bpf_path:
        Path to ``/sys/fs/bpf``.
    proc_pressure_path:
        Path to ``/proc/pressure``.
    proc_mounts_path:
        Path to ``/proc/mounts``.
    proc_status_path:
        Path to ``/proc/self/status``.
    cgroup_path:
        Path to ``/sys/fs/cgroup``.
    bpftool_path:
        Explicit path to ``bpftool`` binary.  ``None`` searches ``$PATH``.
    pinned_map_path:
        Path to a pinned BPF map for read-only probe.
    force_mock:
        When ``True``, skip filesystem/subprocess checks and return mock.

    Returns
    -------
    KernelProbeResult
        A serializable probe result suitable for embedding in scheduler
        decisions and runtime events.
    """
    if force_mock:
        return KernelProbeResult(
            sched_ext="unavailable",
            bpf_map="mock",
            fallback_mode="mock_bpf_map",
            reason="forced mock mode",
        )

    sys_kernel_path = Path(sys_kernel_path)
    sys_fs_bpf_path = Path(sys_fs_bpf_path)
    proc_pressure_path = Path(proc_pressure_path)
    proc_mounts_path = Path(proc_mounts_path)
    proc_status_path = Path(proc_status_path)
    cgroup_path = Path(cgroup_path)

    details: dict[str, Any] = {}

    # sched_ext
    sched_ext_path = sys_kernel_path / "sched_ext"
    sched_ext_present = sched_ext_path.exists()
    details["sched_ext_path"] = str(sched_ext_path)
    details["sched_ext_present"] = sched_ext_present

    # bpffs
    details["bpffs_path"] = str(sys_fs_bpf_path)
    bpffs_present = sys_fs_bpf_path.exists()
    details["bpffs_present"] = bpffs_present
    bpffs_mounted = False
    if bpffs_present:
        bpffs_mounted = _check_bpffs_mounted(proc_mounts_path, sys_fs_bpf_path)
    details["bpffs_mounted"] = bpffs_mounted

    # PSI
    psi_info = _check_psi_files(proc_pressure_path)
    details.update(psi_info)

    # CAP_BPF
    cap_info = _check_cap_bpf(proc_status_path)
    details["cap_bpf"] = cap_info["cap_bpf"]
    if cap_info["reason"]:
        details["cap_bpf_reason"] = cap_info["reason"]

    # cgroup v2
    cgroup_info = _check_cgroup_v2(cgroup_path)
    details.update(cgroup_info)

    # bpftool probe
    bpftool_info = probe_bpftool(bpftool_path=bpftool_path)
    details["bpftool"] = bpftool_info

    # pinned map probe (optional)
    pinned_map_info: dict[str, Any] | None = None
    if pinned_map_path is not None:
        pinned_map_info = probe_pinned_map(
            pinned_map_path=pinned_map_path,
            bpftool_path=bpftool_path,
        )
    details["pinned_map_probe"] = pinned_map_info

    # Determine bpf_map mode:
    # - "pinned_probe" if pinned map was successfully read
    # - "mock" otherwise (never "native" — no real kernel map write)
    bpf_map_mode = "mock"
    if pinned_map_info is not None and pinned_map_info.get("available"):
        bpf_map_mode = "pinned_probe"

    # Build reason with specific status for each component
    status_parts: list[str] = []
    status_parts.append("sched_ext present" if sched_ext_present else "sched_ext not present")
    status_parts.append("bpffs mounted" if bpffs_mounted else "bpffs not mounted")
    status_parts.append("bpftool available" if bpftool_info.get("available") else "bpftool unavailable")
    status_parts.append("CAP_BPF present" if cap_info["cap_bpf"] else "CAP_BPF missing")

    # Check if all prerequisites are available
    prerequisites_available = (
        sched_ext_present
        and bpffs_mounted
        and bpftool_info.get("available", False)
        and cap_info["cap_bpf"]
    )

    # Build conclusion
    if bpf_map_mode == "pinned_probe":
        conclusion = "pinned map read-only probe succeeded; real sched_ext adapter not enabled"
    elif prerequisites_available:
        conclusion = "probe prerequisites available; real sched_ext adapter not enabled"
    else:
        conclusion = "probe prerequisites incomplete; using mock BPF map"

    reason = "; ".join(status_parts) + "; " + conclusion

    return KernelProbeResult(
        sched_ext="available" if sched_ext_present else "unavailable",
        bpf_map=bpf_map_mode,
        fallback_mode="probe_only" if bpf_map_mode == "pinned_probe" else "mock_bpf_map",
        reason=reason,
        details=details,
    )
