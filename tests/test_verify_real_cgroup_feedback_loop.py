from __future__ import annotations

from agentos.events.types import EventCategory, RuntimeEvent
from scripts import verify_real_cgroup_feedback_loop as verifier


def _event(event_type: str, **data) -> RuntimeEvent:
    return RuntimeEvent(
        event_type=event_type,
        category=EventCategory.RESOURCE,
        data=data,
    )


def _accepted_events() -> list[RuntimeEvent]:
    feedback_id = "cgroup-feedback-1"
    events = [
        _event("resource.cgroup_created", path="/sys/fs/cgroup/test"),
        _event(
            "resource.cgroup_pid_added",
            pid=123,
            pid_membership_verified=True,
            membership_check_required=True,
        ),
        _event(
            "resource.cgroup_quota_applied",
            mode="cgroup_v2",
            pid_attached=True,
        ),
    ]
    for index, phase in enumerate(("initial", "running", "final")):
        events.append(
            _event(
                "resource.pressure_sampled",
                scope="agent_cgroup",
                phase=phase,
                sample_index=index,
                pressure={
                    "pressure_source": "cgroup",
                    "pressure_sources": {
                        "cpu": "cgroup",
                        "memory": "cgroup",
                        "io": "cgroup",
                    },
                },
            )
        )
    events.extend(
        [
            _event(
                "resource.cgroup_feedback_applied",
                feedback_id=feedback_id,
                pressure_source="system+cgroup",
                consumed=True,
                reusable=False,
                scheduler_round=2,
            ),
            RuntimeEvent(
                event_type="scheduler.decision",
                category=EventCategory.SCHEDULER,
                data={
                    "pressure": {
                        "pressure_source": "system+cgroup",
                        "feedback_id": feedback_id,
                        "scheduler_round": 2,
                        "cgroup_local": {"feedback_id": feedback_id},
                    }
                },
            ),
            RuntimeEvent(
                event_type="kernel.decision_published",
                category=EventCategory.SCHEDULER,
                data={
                    "requested_bpf_mode": "bpftool",
                    "effective_bpf_mode": "bpftool",
                    "fallback_active": False,
                    "real_updated_count": 2,
                    "real_updated_task_ids": [
                        "local-psi-producer",
                        "feedback-consumer",
                    ],
                    "real_failed_task_ids": [],
                    "bpf_readback_verified": True,
                    "pin_path": "/sys/fs/bpf/agentos/member_a_chain_test",
                },
            ),
            _event("resource.cgroup_removed", path="/sys/fs/cgroup/test"),
        ]
    )
    return events


def test_acceptance_requires_complete_real_feedback_chain() -> None:
    summary = verifier.summarize_events(
        _accepted_events(),
        scenario_completed=True,
    )

    assert verifier.acceptance_passed(summary)
    assert summary["accepted"] is True
    assert summary["local_sample_phases"] == ["initial", "running", "final"]
    assert summary["feedback_applied"] == 1
    assert summary["affected_decisions"] == 1
    assert summary["feedback_reuse_violations"] == 0
    assert summary["real_bpf_published"] is True
    assert summary["real_bpf_updated_tasks"] >= 2
    assert summary["bpf_fallback_active"] is False
    assert summary["bpf_readback_verified"] is True
    assert summary["root_cleanup_failed"] == 0


def test_missing_phase_rejects_acceptance() -> None:
    events = [
        event
        for event in _accepted_events()
        if not (
            event.event_type == "resource.pressure_sampled"
            and event.data.get("phase") == "running"
        )
    ]

    summary = verifier.summarize_events(events, scenario_completed=True)

    assert not verifier.acceptance_passed(summary)
    assert summary["accepted"] is False


def test_fallback_pressure_source_rejects_acceptance() -> None:
    events = _accepted_events()
    pressure_event = next(
        event
        for event in events
        if event.event_type == "resource.pressure_sampled"
    )
    pressure_event.data["pressure"]["pressure_sources"]["io"] = "system"

    summary = verifier.summarize_events(events, scenario_completed=True)

    assert summary["nonlocal_pressure_samples"] == 1
    assert not verifier.acceptance_passed(summary)


def test_feedback_must_reach_a_following_matching_decision() -> None:
    events = [
        event
        for event in _accepted_events()
        if event.event_type != "scheduler.decision"
    ]

    summary = verifier.summarize_events(events, scenario_completed=True)

    assert summary["affected_decisions"] == 0
    assert not verifier.acceptance_passed(summary)


def test_reused_feedback_id_rejects_acceptance() -> None:
    events = _accepted_events()
    applied = next(
        event
        for event in events
        if event.event_type == "resource.cgroup_feedback_applied"
    )
    events.insert(events.index(applied) + 1, applied)

    summary = verifier.summarize_events(events, scenario_completed=True)

    assert summary["feedback_reuse_violations"] >= 1
    assert not verifier.acceptance_passed(summary)


def test_cleanup_failure_or_incomplete_scenario_rejects_acceptance() -> None:
    events = _accepted_events()
    events.append(_event("resource.cgroup_cleanup_failed", reason="busy"))

    failed_cleanup = verifier.summarize_events(
        events,
        scenario_completed=True,
    )
    incomplete = verifier.summarize_events(
        _accepted_events(),
        scenario_completed=False,
    )

    assert failed_cleanup["cleanup_failed"] == 1
    assert not verifier.acceptance_passed(failed_cleanup)
    assert not verifier.acceptance_passed(incomplete)


def test_strict_chain_rejects_mock_bpf() -> None:
    events = _accepted_events()
    published = next(
        event
        for event in events
        if event.event_type == "kernel.decision_published"
    )
    published.data["effective_bpf_mode"] = "mock"
    published.data["real_updated_count"] = 0

    summary = verifier.summarize_events(events, scenario_completed=True)

    assert summary["real_bpf_published"] is False
    assert not verifier.acceptance_passed(summary)


def test_strict_chain_rejects_auto_fallback() -> None:
    events = _accepted_events()
    published = next(
        event
        for event in events
        if event.event_type == "kernel.decision_published"
    )
    published.data["fallback_active"] = True

    summary = verifier.summarize_events(events, scenario_completed=True)

    assert summary["bpf_fallback_active"] is True
    assert not verifier.acceptance_passed(summary)


def test_strict_chain_rejects_missing_bpf_readback() -> None:
    events = _accepted_events()
    published = next(
        event
        for event in events
        if event.event_type == "kernel.decision_published"
    )
    published.data["bpf_readback_verified"] = False

    summary = verifier.summarize_events(events, scenario_completed=True)

    assert summary["bpf_readback_verified"] is False
    assert not verifier.acceptance_passed(summary)


def test_strict_chain_rejects_bpf_cleanup_failure() -> None:
    summary = verifier.summarize_events(
        _accepted_events(),
        scenario_completed=True,
    )
    summary["bpf_cleanup_failed"] = 1

    assert not verifier.acceptance_passed(summary)
