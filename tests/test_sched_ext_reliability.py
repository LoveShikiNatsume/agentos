from agentos.benchmarks.sched_ext_reliability import (
    AUTO_FALLBACK_CASES,
    AUTO_FAULT_CASES,
    STRICT_FAULT_CASES,
    acceptance_passed,
    summarize_trials,
)


def _records(iterations: int) -> list[dict]:
    records = []
    for mode, cases in (("auto", AUTO_FAULT_CASES), ("strict", STRICT_FAULT_CASES)):
        for case in cases:
            for iteration in range(iterations):
                records.append(
                    {
                        "mode": mode,
                        "case": case,
                        "iteration": iteration,
                        "passed": True,
                        "outcome": "recovered" if mode == "auto" else "strict_failed",
                        "task_loss": 0,
                        "fallback_events": int(
                            mode == "strict" or case in AUTO_FALLBACK_CASES
                        ),
                        "recovered_events": int(
                            mode == "auto" and case in AUTO_FALLBACK_CASES
                        ),
                        "detection_ms": 500.0,
                        "recovery_ms": 1000.0,
                    }
                )
    return records


def test_stage5_summary_and_acceptance() -> None:
    summary = summarize_trials(_records(3))
    summary.update(
        policy_leaks=0,
        task_state_leaks=0,
        residual_pinned_maps=[],
        residual_bpf_links=[],
        residual_processes=[],
        residual_cgroups=[],
        kernel_log_errors=[],
        systemd_failed_units=0,
        final_sched_ext_state="disabled",
        failure_reason=None,
        soak={"completed": False, "duration_seconds": 0, "task_loss": 0},
    )

    assert acceptance_passed(summary, expected_iterations=3, require_soak=False)
    assert not acceptance_passed(summary, expected_iterations=3, require_soak=True)


def test_stage5_acceptance_rejects_task_loss() -> None:
    records = _records(1)
    records[0]["task_loss"] = 1
    summary = summarize_trials(records)
    summary.update(
        policy_leaks=0,
        task_state_leaks=0,
        residual_pinned_maps=[],
        residual_bpf_links=[],
        residual_processes=[],
        residual_cgroups=[],
        kernel_log_errors=[],
        systemd_failed_units=0,
        final_sched_ext_state="disabled",
        failure_reason=None,
    )

    assert not acceptance_passed(summary, expected_iterations=1, require_soak=False)


def test_stage5_acceptance_enforces_detection_and_recovery_bounds() -> None:
    records = _records(1)
    records[0]["detection_ms"] = 2_001.0
    records[1]["recovered_events"] = 0
    summary = summarize_trials(records)
    summary.update(
        policy_leaks=0,
        task_state_leaks=0,
        residual_pinned_maps=[],
        residual_bpf_links=[],
        residual_processes=[],
        residual_cgroups=[],
        kernel_log_errors=[],
        systemd_failed_units=0,
        final_sched_ext_state="disabled",
        failure_reason=None,
    )

    assert not acceptance_passed(summary, expected_iterations=1, require_soak=False)
