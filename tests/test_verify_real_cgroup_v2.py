from __future__ import annotations

import json

from scripts import verify_real_cgroup_v2 as verifier


def _accepted_result() -> dict[str, object]:
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


def test_acceptance_requires_every_real_cgroup_counter():
    result = _accepted_result()

    assert verifier.acceptance_passed(result)

    for key, bad_value in (
        ("available", False),
        ("quota_applied", 3),
        ("pid_attached", 0),
        ("pid_membership_verified", False),
        ("child_inheritance_verified", False),
        ("initial_usage_samples", 0),
        ("running_usage_samples", 0),
        ("final_usage_samples", 0),
        ("initial_pressure_samples", 0),
        ("running_pressure_samples", 0),
        ("final_pressure_samples", 0),
        ("cgroup_local_pressure_samples", 0),
        ("usage_samples", 2),
        ("cleanup_failed", 1),
        ("skip_reason", "unsupported environment"),
        ("failure_reason", "verification failed"),
    ):
        candidate = dict(result)
        candidate[key] = bad_value
        assert not verifier.acceptance_passed(candidate), key


def test_non_delegated_environment_emits_machine_readable_skip(
    monkeypatch, capsys, tmp_path,
):
    monkeypatch.setattr(
        verifier,
        "detect_delegation",
        lambda: ("host_root", False),
    )

    exit_code = verifier.main(["--root", str(tmp_path / "cg")])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["available"] is False
    assert payload["delegation_mode"] == "host_root"
    assert "AGENTOS_CGROUP_ALLOW_ROOT=1" in payload["skip_reason"]


def test_json_output_writes_parent_directories(tmp_path):
    output = tmp_path / "nested" / "evidence.json"

    verifier._out({"available": False}, str(output))

    assert json.loads(output.read_text(encoding="utf-8")) == {"available": False}
