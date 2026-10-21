from __future__ import annotations

from pathlib import Path

from scripts import verify_real_sched_ext as verifier


def _accepted_summary() -> dict[str, object]:
    return {
        "preflight_accepted": True,
        "scheduler_name": "agentos_scx",
        "partial_mode": True,
        "workers": {
            "attempted": 100,
            "completed": 100,
            "failed": 0,
            "ext_enabled": 100,
            "valid_checksums": 100,
        },
        "controls": {
            "verifier_ext_enabled": 0,
            "loader_ext_enabled": 0,
            "unmanaged_ext_enabled": 0,
            "cfs_worker_ext_enabled": 0,
        },
        "affinity_violations": 0,
        "stats_delta": {name: 1 for name in verifier.STAT_NAMES},
        "direct_probe_attempts": 4,
        "direct_probe_completed": 4,
        "direct_probe_observed": True,
        "loader_exit_code": 0,
        "sigint_loader_exit_code": 0,
        "duplicate_loader_exit_code": 1,
        "nonroot_loader_exit_code": 1,
        "nonroot_guard_verified": True,
        "final_sched_ext_state": "disabled",
        "systemd_failed_units": 0,
        "kernel_log_errors": [],
        "failure_reason": None,
    }


def test_acceptance_requires_complete_real_partial_takeover() -> None:
    assert verifier.acceptance_passed(_accepted_summary())


def test_worker_failure_or_missing_ext_rejects_acceptance() -> None:
    failed = _accepted_summary()
    failed["workers"]["completed"] = 99
    failed["workers"]["failed"] = 1

    missing_ext = _accepted_summary()
    missing_ext["workers"]["ext_enabled"] = 99

    assert not verifier.acceptance_passed(failed)
    assert not verifier.acceptance_passed(missing_ext)


def test_control_task_takeover_rejects_acceptance() -> None:
    summary = _accepted_summary()
    summary["controls"]["unmanaged_ext_enabled"] = 1

    assert not verifier.acceptance_passed(summary)


def test_missing_stats_or_affinity_violation_rejects_acceptance() -> None:
    missing_stats = _accepted_summary()
    missing_stats["stats_delta"]["dispatch"] = 0

    affinity = _accepted_summary()
    affinity["affinity_violations"] = 1

    assert not verifier.acceptance_passed(missing_stats)
    assert not verifier.acceptance_passed(affinity)


def test_cleanup_or_kernel_error_rejects_acceptance() -> None:
    active = _accepted_summary()
    active["final_sched_ext_state"] = "enabled"

    errored = _accepted_summary()
    errored["kernel_log_errors"] = ["sched_ext: runnable task stall"]

    assert not verifier.acceptance_passed(active)
    assert not verifier.acceptance_passed(errored)


def test_loader_guards_are_required() -> None:
    duplicate = _accepted_summary()
    duplicate["duplicate_loader_exit_code"] = 0

    nonroot = _accepted_summary()
    nonroot["nonroot_guard_verified"] = False

    assert not verifier.acceptance_passed(duplicate)
    assert not verifier.acceptance_passed(nonroot)


def test_kernel_log_error_filter_ignores_normal_lifecycle() -> None:
    lines = [
        'sched_ext: BPF scheduler "agentos_scx" enabled (partial)',
        'sched_ext: BPF scheduler "agentos_scx" disabled (unregistered from user space)',
        "sched_ext: qmap: sshd failed to run for 5.061s",
        "sched_ext: runnable task stall",
    ]

    assert verifier.kernel_log_errors(lines) == [
        "sched_ext: runnable task stall"
    ]


def test_sched_ext_enabled_parses_proc_sched(tmp_path: Path) -> None:
    sched = tmp_path / "123/sched"
    sched.parent.mkdir(parents=True)
    sched.write_text(
        "worker (123, #threads: 1)\n"
        "policy : 7\n"
        "ext.enabled : 1\n",
        encoding="utf-8",
    )

    assert verifier.sched_ext_enabled(123, tmp_path) == 1
    assert verifier.sched_ext_enabled(456, tmp_path) is None


def test_nonroot_identity_prefers_sudo_caller() -> None:
    assert verifier._nonroot_identity({"SUDO_UID": "1000", "SUDO_GID": "1001"}) == (
        1000,
        1001,
    )


def test_nonroot_identity_falls_back_for_invalid_or_root_caller() -> None:
    assert verifier._nonroot_identity({}) == (65534, 65534)
    assert verifier._nonroot_identity({"SUDO_UID": "0", "SUDO_GID": "0"}) == (
        65534,
        65534,
    )
    assert verifier._nonroot_identity({"SUDO_UID": "bad"}) == (65534, 65534)


def test_direct_local_is_a_required_acceptance_stat() -> None:
    summary = _accepted_summary()
    summary["stats_delta"]["direct_local"] = 0

    assert verifier.acceptance_passed(summary)

    summary["direct_probe_observed"] = False

    assert not verifier.acceptance_passed(summary)
