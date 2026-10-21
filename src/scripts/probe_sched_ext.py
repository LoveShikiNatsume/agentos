"""Probe sched_ext / eBPF / BPF Map availability.

Checks kernel and toolchain support for sched_ext-based scheduling,
outputting a structured JSON report with fallback reasons. Designed
to run on any Linux (including containers) without requiring root.

This script delegates core probing to
:func:`agentos.isolation.sched_ext.probe_sched_ext` so that the CLI
and Runtime never give contradictory conclusions.

Usage:
    uv run python scripts/probe_sched_ext.py
    uv run python scripts/probe_sched_ext.py --mock
    uv run python scripts/probe_sched_ext.py --bpftool /usr/sbin/bpftool
    uv run python scripts/probe_sched_ext.py --pinned-map /sys/fs/bpf/my_map
    uv run python scripts/probe_sched_ext.py --json-only
    uv run python scripts/probe_sched_ext.py --out runs/probe_sched_ext.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentos.isolation.sched_ext import KernelProbeResult, probe_sched_ext


@dataclass
class ProbeResult:
    """CLI probe result combining OS info with kernel probe."""

    kernel_version: str = ""
    kernel_release: str = ""
    is_openeuler: bool = False
    openeuler_version: str = ""
    os_release: str = ""
    sched_ext_available: bool = False
    sched_ext_path: str = ""
    sched_ext_reason: str = ""
    bpf_available: bool = False
    bpf_reason: str = ""
    bpftool_available: bool = False
    bpftool_path: str = ""
    bpftool_version: str = ""
    cap_bpf: bool = False
    cap_bpf_reason: str = ""
    bpf_map_mode: str = "mock"
    cgroup_v2_mounted: bool = False
    cgroup_v2_path: str = ""
    cgroup_v2_controllers: list[str] = field(default_factory=list)
    fallback_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def summary(self) -> dict:
        return {
            "sched_ext": "available" if self.sched_ext_available else "unavailable",
            "bpf_map": self.bpf_map_mode,
            "reason": "; ".join(self.fallback_reasons) if self.fallback_reasons else "all checks passed",
        }


def parse_os_release(text: str) -> dict:
    """Extract is_openeuler and openeuler_version from os-release text."""
    result: dict = {"is_openeuler": False, "openeuler_version": "", "os_release": text.strip()[:200]}
    result["is_openeuler"] = "openeuler" in text.lower()
    if result["is_openeuler"]:
        for line in text.splitlines():
            if line.startswith("VERSION_ID="):
                result["openeuler_version"] = line.split("=", 1)[1].strip().strip('"')
                break
        if not result["openeuler_version"]:
            for line in text.splitlines():
                if line.startswith("PRETTY_NAME="):
                    result["openeuler_version"] = line.split("=", 1)[1].strip().strip('"')
                    break
    return result


def probe(
    *,
    bpftool_path: str | None = None,
    pinned_map_path: str | None = None,
) -> ProbeResult:
    """Run the CLI probe, delegating core checks to sched_ext.probe_sched_ext."""
    r = ProbeResult()

    # OS info
    r.kernel_version = platform.release()
    r.kernel_release = platform.version()

    os_release_path = Path("/etc/os-release")
    if os_release_path.exists():
        text = os_release_path.read_text(encoding="utf-8", errors="replace")
        parsed = parse_os_release(text)
        r.os_release = parsed["os_release"]
        r.is_openeuler = parsed["is_openeuler"]
        r.openeuler_version = parsed["openeuler_version"]

    # Delegate core probe to runtime module
    kernel_result: KernelProbeResult = probe_sched_ext(
        bpftool_path=bpftool_path,
        pinned_map_path=pinned_map_path,
    )
    details = kernel_result.details

    # Map kernel probe results to CLI ProbeResult
    r.sched_ext_available = kernel_result.sched_ext == "available"
    r.sched_ext_path = details.get("sched_ext_path", "")
    if not r.sched_ext_available:
        r.sched_ext_reason = kernel_result.reason or ""
        r.fallback_reasons.append(r.sched_ext_reason)

    r.bpf_available = details.get("bpffs_present", False)
    if not r.bpf_available:
        r.bpf_reason = "missing /sys/fs/bpf"
        r.fallback_reasons.append(r.bpf_reason)

    bpftool_info = details.get("bpftool", {})
    r.bpftool_available = bpftool_info.get("available", False)
    r.bpftool_path = bpftool_info.get("binary", "") or ""
    r.bpftool_version = bpftool_info.get("version", "") or ""
    if not r.bpftool_available:
        reason = bpftool_info.get("reason", "bpftool not available")
        r.fallback_reasons.append(reason)

    r.cap_bpf = details.get("cap_bpf", False)
    if not r.cap_bpf:
        r.cap_bpf_reason = details.get("cap_bpf_reason", "missing CAP_BPF")
        r.fallback_reasons.append(r.cap_bpf_reason)

    # cgroup v2
    r.cgroup_v2_mounted = details.get("cgroup_v2_present", False)
    r.cgroup_v2_path = details.get("cgroup_v2_path", "")

    # bpf_map_mode: never "native" — only "mock" or "pinned_probe"
    r.bpf_map_mode = kernel_result.bpf_map  # "mock" or "pinned_probe"

    return r


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe sched_ext/eBPF/BPF availability")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force mock mode (skip real kernel checks)",
    )
    parser.add_argument(
        "--bpftool",
        default=None,
        help="Explicit path to bpftool binary",
    )
    parser.add_argument(
        "--pinned-map",
        default=None,
        help="Path to a pinned BPF map for read-only probe",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Output only JSON (no summary text)",
    )
    args = parser.parse_args()

    if args.mock:
        result = KernelProbeResult(
            sched_ext="unavailable",
            bpf_map="mock",
            fallback_mode="mock_bpf_map",
            reason="forced mock mode",
        )
        output = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    else:
        result = probe(
            bpftool_path=args.bpftool,
            pinned_map_path=args.pinned_map,
        )
        output = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)

    print(output)

    if not args.json_only:
        if args.mock:
            print()
            print("Summary: sched_ext=unavailable, bpf_map=mock")
            print("Reason:  forced mock mode")
        else:
            summary = result.summary
            print()
            print(f"Summary: sched_ext={summary['sched_ext']}, bpf_map={summary['bpf_map']}")
            print(f"Reason:  {summary['reason']}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output + "\n", encoding="utf-8")
        if not args.json_only:
            print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
