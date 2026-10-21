from __future__ import annotations

import os
from pathlib import Path

_TRUE_VALUES = {"1", "true", "yes"}
_CONTAINER_MARKERS = ("docker", "lxc", "kubepods", "containerd")


def detect_delegation() -> tuple[str, bool]:
    """Return the detected cgroup delegation mode and whether writes are allowed."""
    if _in_container():
        return "container", True
    if _systemd_delegated():
        return "systemd_delegation", True
    override = os.environ.get("AGENTOS_CGROUP_ALLOW_ROOT", "").strip().lower()
    if override in _TRUE_VALUES:
        return "host_root_override", True
    return "host_root", False


def _in_container() -> bool:
    return (
        Path("/.dockerenv").exists()
        or Path("/run/.containerenv").exists()
        or _proc1_cgroup_matches_container()
    )


def _proc1_cgroup_matches_container() -> bool:
    try:
        text = Path("/proc/1/cgroup").read_text(encoding="utf-8")
    except OSError:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in _CONTAINER_MARKERS)


def _systemd_delegated() -> bool:
    try:
        uid = os.getuid()
        slice_path = Path(f"/sys/fs/cgroup/user.slice/user-{uid}.slice")
        return slice_path.exists() and os.access(slice_path, os.W_OK)
    except OSError:
        return False
