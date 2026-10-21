from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.render_member_a_kernel_report import load_evidence, render_report


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def _accepted_cgroup() -> dict[str, object]:
    return {
        "available": True,
        "quota_applied": 4,
        "pid_attached": 1,
        "pid_membership_verified": True,
        "child_inheritance_verified": True,
        "initial_usage_samples": 1,
        "running_usage_samples": 1,
        "final_usage_samples": 1,
        "initial_pressure_samples": 1,
        "running_pressure_samples": 1,
        "final_pressure_samples": 1,
        "cgroup_local_pressure_samples": 3,
        "usage_samples": 3,
        "cleanup_failed": 0,
        "skip_reason": None,
        "failure_reason": None,
    }


def _accepted_bpf() -> dict[str, object]:
    return {
        "available": True,
        "bpftool_version": "bpftool v7.2.0",
        "map_created": True,
        "schema_verified": True,
        "schema_version": 1,
        "updates": 1,
        "lookups": 1,
        "dumps": 1,
        "readback_verified": True,
        "cleanup_failed": 0,
        "skip_reason": None,
        "failure_reason": None,
    }


def test_report_renders_current_passed_results(tmp_path: Path) -> None:
    cgroup = tmp_path / "cgroup.json"
    bpf = tmp_path / "bpf.json"
    _write_json(cgroup, _accepted_cgroup())
    _write_json(bpf, _accepted_bpf())

    report = render_report(load_evidence(cgroup, bpf))

    assert "真实 cgroup v2：PASS" in report
    assert "真实 pinned BPF map：PASS" in report
    assert "`quota_applied=4`" in report
    assert "`readback_verified=true`" in report
    assert "available=false" not in report


def test_report_marks_skip_without_calling_it_passed(tmp_path: Path) -> None:
    cgroup_payload = _accepted_cgroup()
    cgroup_payload.update(
        {
            "available": False,
            "skip_reason": "delegation missing",
        }
    )
    cgroup = tmp_path / "cgroup.json"
    bpf = tmp_path / "bpf.json"
    _write_json(cgroup, cgroup_payload)
    _write_json(bpf, _accepted_bpf())

    report = render_report(load_evidence(cgroup, bpf))

    assert "真实 cgroup v2：SKIP" in report
    assert "真实 cgroup v2：PASS" not in report
    assert "delegation missing" in report


def test_report_keeps_sched_ext_takeover_false(tmp_path: Path) -> None:
    cgroup = tmp_path / "cgroup.json"
    bpf = tmp_path / "bpf.json"
    _write_json(cgroup, _accepted_cgroup())
    _write_json(bpf, _accepted_bpf())

    report = render_report(load_evidence(cgroup, bpf))

    assert "未实现 sched_ext/eBPF 调度接管" in report
    assert "BPF map 数据通路" in report


def test_invalid_json_fails_with_source_path(tmp_path: Path) -> None:
    cgroup = tmp_path / "broken-cgroup.json"
    bpf = tmp_path / "bpf.json"
    cgroup.write_text("[", encoding="utf-8")
    _write_json(bpf, _accepted_bpf())

    with pytest.raises(ValueError, match=str(cgroup)):
        load_evidence(cgroup, bpf)
