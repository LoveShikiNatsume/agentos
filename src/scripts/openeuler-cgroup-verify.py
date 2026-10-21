"""Verify real cgroup v2 operations on openEuler.

Run inside a --privileged Docker container:
    docker run --rm --privileged agentos-openeuler \
        python3 scripts/openeuler-cgroup-verify.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from agentos.core.resources import ResourceQuota
from agentos.isolation.cgroup import CgroupV2Manager
from agentos.isolation.delegation import detect_delegation

CGROUP_ROOT = Path("/sys/fs/cgroup/agentos-verify")
PASSED = 0
FAILED = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {label}")
    else:
        FAILED += 1
        msg = f"  [FAIL] {label}"
        if detail:
            msg += f" — {detail}"
        print(msg)


def verify_cgroup_v2_mounted() -> None:
    controllers = Path("/sys/fs/cgroup/cgroup.controllers")
    check(
        "cgroup v2 filesystem mounted",
        controllers.exists(),
        "missing /sys/fs/cgroup/cgroup.controllers",
    )
    if controllers.exists():
        available = controllers.read_text().strip()
        print(f"         controllers: {available}")


def _enable_delegation() -> None:
    """Move root cgroup processes to a child, then enable controllers on root.

    In containers, the root cgroup has processes in it, which prevents
    enabling subtree_control. Move them to an 'init' child first.
    """
    mode, allowed = detect_delegation()
    if not allowed:
        raise RuntimeError(f"unsafe cgroup delegation refused: {mode}")

    root = Path("/sys/fs/cgroup")
    subtree_control = root / "cgroup.subtree_control"
    if not subtree_control.exists():
        return

    # Move all processes out of root cgroup into an 'init' child
    init_cg = root / "init"
    init_cg.mkdir(exist_ok=True)
    procs = (root / "cgroup.procs").read_text().strip().splitlines()
    for pid in procs:
        try:
            (init_cg / "cgroup.procs").write_text(pid)
        except OSError:
            pass

    # Now enable controllers on root
    for ctrl in ("cpu", "memory", "pids"):
        try:
            subtree_control.write_text(f"+{ctrl}")
        except OSError:
            pass


def verify_create_and_limits() -> Path | None:
    _enable_delegation()

    mgr = CgroupV2Manager(root=CGROUP_ROOT)
    quota = ResourceQuota(memory_max_mb=64, pids_max=16, cpu_max_percent=50.0)
    result = mgr.create("test-agent", quota)

    check("cgroup created", result.available, result.reason or "")
    if not result.available:
        return None

    check("cgroup directory exists", result.path.exists())

    memory_max = (result.path / "memory.max").read_text().strip()
    check(
        "memory.max set to 64 MiB",
        memory_max == str(64 * 1024 * 1024),
        f"got {memory_max}",
    )

    pids_max = (result.path / "pids.max").read_text().strip()
    check("pids.max set to 16", pids_max == "16", f"got {pids_max}")

    cpu_max = (result.path / "cpu.max").read_text().strip()
    check(
        "cpu.max set to 50%",
        cpu_max == "50000 100000",
        f"got {cpu_max}",
    )

    return result.path


def verify_read_usage(path: Path) -> None:
    mgr = CgroupV2Manager(root=CGROUP_ROOT)
    usage = mgr.read_usage(path)

    check("cpu_usage_usec is int", isinstance(usage.cpu_usage_usec, int))
    check("pids_current is int", isinstance(usage.pids_current, int))
    check("memory_rss_mb is float", isinstance(usage.memory_rss_mb, float))
    print(
        f"         cpu={usage.cpu_usage_usec}us "
        f"mem={usage.memory_rss_mb:.1f}MB "
        f"pids={usage.pids_current}"
    )


def verify_cleanup(path: Path) -> None:
    mgr = CgroupV2Manager(root=CGROUP_ROOT)
    result = mgr.cleanup(path)

    check("cgroup cleaned up", result.available, result.reason or "")
    check("cgroup directory removed", not path.exists())

    if CGROUP_ROOT.exists():
        try:
            CGROUP_ROOT.rmdir()
            check("root directory removed", True)
        except OSError as exc:
            check("root directory removed", False, str(exc))


def main() -> int:
    if os.getuid() != 0:
        print("SKIP: cgroup verification requires root (use --privileged)")
        print("Run: docker run --rm --privileged agentos-openeuler "
              "python3 scripts/openeuler-cgroup-verify.py")
        return 0

    mode, allowed = detect_delegation()
    if not allowed:
        print(f"SKIP: unsafe cgroup delegation refused: {mode}")
        print("Set AGENTOS_CGROUP_ALLOW_ROOT=1 only on an isolated test host.")
        return 0

    print("=== cgroup v2 Real Verification on openEuler ===")
    print()

    print("[1/4] Check cgroup v2 filesystem")
    verify_cgroup_v2_mounted()
    print()

    print("[2/4] Create cgroup with resource limits")
    path = verify_create_and_limits()
    print()

    if path is None:
        print("Cannot continue without cgroup creation.")
        return 1

    print("[3/4] Read resource usage")
    verify_read_usage(path)
    print()

    print("[4/4] Cleanup cgroup")
    verify_cleanup(path)
    print()

    print(f"=== Results: {PASSED} passed, {FAILED} failed ===")
    return 1 if FAILED > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
