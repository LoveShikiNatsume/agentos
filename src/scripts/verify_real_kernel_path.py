#!/usr/bin/env python3
"""Combined strict verifier for Member A's real kernel-facing paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import verify_real_bpf_map as bpf_verifier
from scripts import verify_real_cgroup_v2 as cgroup_verifier


def summarize_acceptance(
    cgroup_result: dict[str, Any],
    bpf_result: dict[str, Any],
) -> dict[str, Any]:
    cgroup_passed = cgroup_verifier.acceptance_passed(cgroup_result)
    bpf_passed = bpf_verifier.acceptance_passed(bpf_result)
    skip_reasons: list[str] = []
    failure_reasons: list[str] = []

    for component, result, passed in (
        ("cgroup", cgroup_result, cgroup_passed),
        ("bpf", bpf_result, bpf_passed),
    ):
        skip_reason = result.get("skip_reason")
        failure_reason = result.get("failure_reason")
        if skip_reason:
            skip_reasons.append(f"{component}: {skip_reason}")
        if failure_reason:
            failure_reasons.append(f"{component}: {failure_reason}")
        elif not passed and not skip_reason:
            failure_reasons.append(
                f"{component} strict acceptance predicate failed"
            )

    return {
        "schema_version": 1,
        "accepted": cgroup_passed and bpf_passed,
        "cgroup_passed": cgroup_passed,
        "bpf_passed": bpf_passed,
        "cgroup": cgroup_result,
        "bpf": bpf_result,
        "skip_reasons": skip_reasons,
        "failure_reasons": failure_reasons,
        "claims": {
            "real_cgroup_v2": cgroup_passed,
            "cgroup_local_psi": cgroup_passed,
            "real_pinned_bpf_map": bpf_passed,
            "bpf_program_attached": False,
            "sched_ext_active": False,
            "kernel_scheduling_takeover": False,
        },
    }


def _exception_result(exc: Exception) -> dict[str, Any]:
    return {
        "available": False,
        "skip_reason": None,
        "failure_reason": f"{type(exc).__name__}: {exc}",
    }


def run_verification(
    *,
    cgroup_root: Path,
    bpffs_root: Path,
    bpftool_path: str | None = None,
    cgroup_runner=None,
    bpf_runner=None,
) -> dict[str, Any]:
    run_cgroup = cgroup_runner or cgroup_verifier.run_verification
    run_bpf = bpf_runner or bpf_verifier.run_verification

    try:
        cgroup_result = run_cgroup(cgroup_root)
    except Exception as exc:
        cgroup_result = _exception_result(exc)

    try:
        bpf_result = run_bpf(
            bpffs_root=bpffs_root,
            bpftool_path=bpftool_path,
        )
    except Exception as exc:
        bpf_result = _exception_result(exc)

    return summarize_acceptance(cgroup_result, bpf_result)


def acceptance_exit_code(summary: dict[str, Any]) -> int:
    if summary.get("accepted") is True:
        return 0
    if summary.get("failure_reasons"):
        return 1
    if summary.get("skip_reasons"):
        return 2
    return 1


def _out(data: dict[str, Any], path: str | None) -> None:
    text = json.dumps(data, indent=2, sort_keys=True)
    print(text)
    if path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cgroup-root", default="/sys/fs/cgroup/agentos")
    parser.add_argument("--bpffs-root", default="/sys/fs/bpf")
    parser.add_argument("--bpftool-path", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    summary = run_verification(
        cgroup_root=Path(args.cgroup_root),
        bpffs_root=Path(args.bpffs_root),
        bpftool_path=args.bpftool_path,
    )
    _out(summary, args.json_out)
    return acceptance_exit_code(summary)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
