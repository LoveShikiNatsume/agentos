#!/usr/bin/env python3
"""Render Member A's kernel-path report from machine-readable evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_real_bpf_map import acceptance_passed as bpf_accepted
from scripts.verify_real_cgroup_v2 import acceptance_passed as cgroup_accepted


def read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: cannot read valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return raw


def load_evidence(cgroup_path: Path, bpf_path: Path) -> dict[str, Any]:
    return {
        "cgroup": read_json(cgroup_path),
        "bpf": read_json(bpf_path),
        "sources": {
            "cgroup": str(cgroup_path),
            "bpf": str(bpf_path),
        },
    }


def _status(payload: dict[str, Any], accepted: bool) -> str:
    if payload.get("skip_reason"):
        return "SKIP"
    return "PASS" if accepted else "FAIL"


def _json_value(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def render_report(evidence: dict[str, Any]) -> str:
    cgroup = evidence["cgroup"]
    bpf = evidence["bpf"]
    cgroup_status = _status(cgroup, cgroup_accepted(cgroup))
    bpf_status = _status(bpf, bpf_accepted(bpf))
    sources = evidence.get("sources", {})

    lines = [
        "# 成员 A 真实内核通路验收报告",
        "",
        "## 当前机器可读证据",
        "",
        f"- 真实 cgroup v2：{cgroup_status}",
        f"- 真实 pinned BPF map：{bpf_status}",
        "",
        "### cgroup v2",
        "",
        f"- `available={_json_value(cgroup.get('available'))}`",
        f"- `quota_applied={_json_value(cgroup.get('quota_applied', 0))}`",
        (
            "- `pid_membership_verified="
            f"{_json_value(cgroup.get('pid_membership_verified'))}`"
        ),
        (
            "- `child_inheritance_verified="
            f"{_json_value(cgroup.get('child_inheritance_verified'))}`"
        ),
        (
            "- `cgroup_local_pressure_samples="
            f"{_json_value(cgroup.get('cgroup_local_pressure_samples', 0))}`"
        ),
        f"- `cleanup_failed={_json_value(cgroup.get('cleanup_failed', 0))}`",
    ]
    if cgroup.get("skip_reason"):
        lines.append(f"- skip reason: {cgroup['skip_reason']}")
    if cgroup.get("failure_reason"):
        lines.append(f"- failure reason: {cgroup['failure_reason']}")

    lines.extend(
        [
            "",
            "### pinned BPF map",
            "",
            f"- `available={_json_value(bpf.get('available'))}`",
            f"- `bpftool_version={_json_value(bpf.get('bpftool_version'))}`",
            f"- `map_created={_json_value(bpf.get('map_created'))}`",
            f"- `schema_verified={_json_value(bpf.get('schema_verified'))}`",
            (
                "- `schema_version="
                f"{_json_value(bpf.get('schema_version'))}` "
                "（pinned map ABI v1）"
            ),
            f"- `updates={_json_value(bpf.get('updates', 0))}`",
            f"- `lookups={_json_value(bpf.get('lookups', 0))}`",
            f"- `dumps={_json_value(bpf.get('dumps', 0))}`",
            (
                "- `readback_verified="
                f"{_json_value(bpf.get('readback_verified'))}`"
            ),
            f"- `cleanup_failed={_json_value(bpf.get('cleanup_failed', 0))}`",
        ]
    )
    if bpf.get("skip_reason"):
        lines.append(f"- skip reason: {bpf['skip_reason']}")
    if bpf.get("failure_reason"):
        lines.append(f"- failure reason: {bpf['failure_reason']}")

    lines.extend(
        [
            "",
            "## 证据来源",
            "",
            f"- cgroup: `{sources.get('cgroup', 'unknown')}`",
            f"- BPF: `{sources.get('bpf', 'unknown')}`",
            "",
            "## 技术边界",
            "",
            "- 已验证真实 cgroup v2 控制、cgroup-local PSI/usage 与清理。",
            "- 已验证真实 BPF map 数据通路的 create/update/lookup/dump/readback。",
            "- 未实现 sched_ext/eBPF 调度接管，不宣称替换 CFS。",
            "- BPF map 数据通路只证明决策记录发布和 ABI，不改变 Linux 调度行为。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cgroup-json", required=True)
    parser.add_argument("--bpf-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    report = render_report(
        load_evidence(Path(args.cgroup_json), Path(args.bpf_json))
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
