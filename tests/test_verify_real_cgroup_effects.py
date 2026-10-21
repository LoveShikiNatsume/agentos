from __future__ import annotations

import json

from scripts import verify_real_cgroup_effects as verifier


def test_cpu_effect_requires_throttling_delta() -> None:
    result = verifier.summarize_cpu_effect(
        before={"nr_throttled": 2, "throttled_usec": 100},
        after={"nr_throttled": 4, "throttled_usec": 300},
    )

    assert result["accepted"] is True
    assert result["nr_throttled_delta"] == 2
    assert result["throttled_usec_delta"] == 200


def test_cpu_effect_rejects_zero_negative_missing_and_bool_values() -> None:
    assert verifier.summarize_cpu_effect(
        before={"nr_throttled": 2, "throttled_usec": 100},
        after={"nr_throttled": 2, "throttled_usec": 200},
    )["accepted"] is False
    assert verifier.summarize_cpu_effect(
        before={"nr_throttled": 4, "throttled_usec": 300},
        after={"nr_throttled": 2, "throttled_usec": 100},
    )["accepted"] is False
    assert verifier.summarize_cpu_effect(
        before={"nr_throttled": 1},
        after={"nr_throttled": 2, "throttled_usec": 100},
    )["accepted"] is False
    assert verifier.summarize_cpu_effect(
        before={"nr_throttled": False, "throttled_usec": 0},
        after={"nr_throttled": True, "throttled_usec": 1},
    )["accepted"] is False


def test_pids_effect_requires_eagain_or_limit_event() -> None:
    result = verifier.summarize_pids_effect(
        attempted=4,
        started=1,
        failures=["EAGAIN"],
    )

    assert result["accepted"] is True
    assert result["blocked"] == 3


def test_pids_effect_accepts_events_max_delta() -> None:
    result = verifier.summarize_pids_effect(
        attempted=4,
        started=2,
        failures=[],
        before={"max": 0},
        after={"max": 1},
    )

    assert result["accepted"] is True
    assert result["events_max_delta"] == 1


def test_pids_effect_rejects_no_limit_signal_invalid_counts_and_bool_values() -> None:
    assert verifier.summarize_pids_effect(
        attempted=4,
        started=4,
        failures=[],
    )["accepted"] is False
    assert verifier.summarize_pids_effect(
        attempted=1,
        started=2,
        failures=["EAGAIN"],
    )["accepted"] is False
    assert verifier.summarize_pids_effect(
        attempted=True,
        started=False,
        failures=["EAGAIN"],
    )["accepted"] is False
    assert verifier.summarize_pids_effect(
        attempted=4,
        started=1,
        failures=["EPERM"],
    )["accepted"] is False


def test_memory_effect_requires_memory_event_delta() -> None:
    result = verifier.summarize_memory_effect(
        before={"oom": 0, "oom_kill": 0, "max": 0},
        after={"oom": 1, "oom_kill": 1, "max": 2},
    )

    assert result["accepted"] is True
    assert result["events_max_delta"] == 2
    assert result["oom_delta"] == 1
    assert result["oom_kill_delta"] == 1


def test_memory_effect_rejects_zero_negative_missing_and_bool_values() -> None:
    assert verifier.summarize_memory_effect(
        before={"oom": 0, "oom_kill": 0, "max": 0},
        after={"oom": 0, "oom_kill": 0, "max": 0},
    )["accepted"] is False
    assert verifier.summarize_memory_effect(
        before={"oom": 1, "oom_kill": 1, "max": 2},
        after={"oom": 0, "oom_kill": 0, "max": 1},
    )["accepted"] is False
    assert verifier.summarize_memory_effect(
        before={"oom": 0, "max": 0},
        after={"oom": 1, "oom_kill": 1, "max": 2},
    )["accepted"] is False
    assert verifier.summarize_memory_effect(
        before={"oom": False, "oom_kill": False, "max": False},
        after={"oom": True, "oom_kill": True, "max": True},
    )["accepted"] is False


def test_acceptance_requires_all_effects_and_cleanup_success() -> None:
    summary = {
        "cpu": verifier.summarize_cpu_effect(
            before={"nr_throttled": 0, "throttled_usec": 0},
            after={"nr_throttled": 1, "throttled_usec": 10},
        ),
        "pids": verifier.summarize_pids_effect(
            attempted=3,
            started=1,
            failures=["EAGAIN"],
        ),
        "memory": verifier.summarize_memory_effect(
            before={"oom": 0, "oom_kill": 0, "max": 0},
            after={"oom": 0, "oom_kill": 0, "max": 1},
        ),
        "cleanup_failed": 0,
        "skip_reason": None,
        "failure_reason": None,
    }

    assert verifier.acceptance_passed(summary) is True
    summary["cleanup_failed"] = 1
    assert verifier.acceptance_passed(summary) is False


def test_acceptance_ignores_existing_accepted_placeholder() -> None:
    summary = {
        "accepted": False,
        "cpu": {"accepted": True},
        "pids": {"accepted": True},
        "memory": {"accepted": True},
        "cleanup_failed": 0,
        "skip_reason": None,
        "failure_reason": None,
    }

    assert verifier.acceptance_passed(summary) is True


def test_main_without_isolated_vm_confirmation_skips(tmp_path, capsys) -> None:
    out = tmp_path / "summary.json"

    code = verifier.main(["--out", str(out)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert code == 2
    assert payload == written
    assert payload["accepted"] is False
    assert payload["skip_reason"] == "isolated VM not confirmed"
