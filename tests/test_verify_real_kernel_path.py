from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import verify_real_kernel_path as verifier


ENTRYPOINT = (
    (Path(__file__).resolve().parent.parent / "src")
    / "scripts"
    / "openeuler-real-kernel-verify.sh"
)
VERIFIER_SCRIPT = (
    (Path(__file__).resolve().parent.parent / "src")
    / "scripts"
    / "verify_real_kernel_path.py"
)


def accepted_cgroup_result() -> dict[str, object]:
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


def accepted_bpf_result() -> dict[str, object]:
    return {
        "available": True,
        "map_created": True,
        "schema_verified": True,
        "updates": 1,
        "lookups": 1,
        "readback_verified": True,
        "dumps": 1,
        "cleanup_failed": 0,
        "skip_reason": None,
        "failure_reason": None,
    }


def test_summary_accepts_only_when_both_real_paths_pass():
    cgroup = accepted_cgroup_result()
    bpf = accepted_bpf_result()

    summary = verifier.summarize_acceptance(cgroup, bpf)

    assert summary["accepted"] is True
    assert summary["cgroup_passed"] is True
    assert summary["bpf_passed"] is True
    assert summary["cgroup"] is cgroup
    assert summary["bpf"] is bpf
    assert summary["skip_reasons"] == []
    assert summary["failure_reasons"] == []


@pytest.mark.parametrize("component", ["cgroup", "bpf"])
def test_summary_rejects_when_either_component_fails(component: str):
    cgroup = accepted_cgroup_result()
    bpf = accepted_bpf_result()
    if component == "cgroup":
        cgroup["running_pressure_samples"] = 0
    else:
        bpf["readback_verified"] = False

    summary = verifier.summarize_acceptance(cgroup, bpf)

    assert summary["accepted"] is False
    assert summary[f"{component}_passed"] is False
    assert summary["failure_reasons"]


def test_summary_never_claims_kernel_scheduler_takeover():
    summary = verifier.summarize_acceptance(
        accepted_cgroup_result(),
        accepted_bpf_result(),
    )

    assert summary["claims"] == {
        "real_cgroup_v2": True,
        "cgroup_local_psi": True,
        "real_pinned_bpf_map": True,
        "bpf_program_attached": False,
        "sched_ext_active": False,
        "kernel_scheduling_takeover": False,
    }


def test_run_verification_attempts_both_components_after_first_exception(
    tmp_path,
):
    calls: list[tuple[str, object]] = []

    def fail_cgroup(root):
        calls.append(("cgroup", root))
        raise OSError("cgroup boom")

    def pass_bpf(**kwargs):
        calls.append(("bpf", kwargs["bpffs_root"]))
        return accepted_bpf_result()

    summary = verifier.run_verification(
        cgroup_root=tmp_path / "cg",
        bpffs_root=tmp_path / "bpf",
        cgroup_runner=fail_cgroup,
        bpf_runner=pass_bpf,
    )

    assert [name for name, _value in calls] == ["cgroup", "bpf"]
    assert summary["accepted"] is False
    assert summary["bpf_passed"] is True
    assert "OSError: cgroup boom" in summary["cgroup"]["failure_reason"]


def test_run_verification_contains_bpf_exception_after_cgroup_passes(
    tmp_path,
):
    calls: list[str] = []

    def pass_cgroup(root):
        calls.append("cgroup")
        return accepted_cgroup_result()

    def fail_bpf(**kwargs):
        calls.append("bpf")
        raise RuntimeError("bpf boom")

    summary = verifier.run_verification(
        cgroup_root=tmp_path / "cg",
        bpffs_root=tmp_path / "bpf",
        cgroup_runner=pass_cgroup,
        bpf_runner=fail_bpf,
    )

    assert calls == ["cgroup", "bpf"]
    assert summary["accepted"] is False
    assert summary["cgroup_passed"] is True
    assert "RuntimeError: bpf boom" in summary["bpf"]["failure_reason"]


def test_acceptance_exit_code_prioritizes_failure_over_skip():
    assert verifier.acceptance_exit_code({"accepted": True}) == 0
    assert verifier.acceptance_exit_code(
        {
            "accepted": False,
            "failure_reasons": ["cgroup failed"],
            "skip_reasons": ["bpf skipped"],
        }
    ) == 1
    assert verifier.acceptance_exit_code(
        {
            "accepted": False,
            "failure_reasons": [],
            "skip_reasons": ["bpf skipped"],
        }
    ) == 2
    assert verifier.acceptance_exit_code(
        {
            "accepted": False,
            "failure_reasons": [],
            "skip_reasons": [],
        }
    ) == 1


def test_main_writes_identical_json_before_returning_exit_code(
    monkeypatch,
    capsys,
    tmp_path,
):
    summary = verifier.summarize_acceptance(
        accepted_cgroup_result(),
        accepted_bpf_result(),
    )
    monkeypatch.setattr(
        verifier,
        "run_verification",
        lambda **_kwargs: summary,
    )
    output = tmp_path / "nested" / "summary.json"

    exit_code = verifier.main(
        [
            "--cgroup-root",
            str(tmp_path / "cg"),
            "--bpffs-root",
            str(tmp_path / "bpf"),
            "--bpftool-path",
            "/usr/sbin/bpftool",
            "--json-out",
            str(output),
        ]
    )

    stdout_payload = json.loads(capsys.readouterr().out)
    file_payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert stdout_payload == summary
    assert file_payload == summary


def test_openeuler_entrypoint_bootstraps_both_real_paths():
    text = ENTRYPOINT.read_text(encoding="utf-8")

    assert "set -euo pipefail" in text
    assert "openeuler-cgroup-verify.py" in text
    assert "stat -fc %T /sys/fs/bpf" in text
    assert "mount -t bpf bpf /sys/fs/bpf" in text
    assert "verify_real_kernel_path.py" in text
    assert "--json-out" in text


def test_verifier_script_runs_directly_outside_project_directory(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, str(VERIFIER_SCRIPT), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--cgroup-root" in completed.stdout
