from scripts.verify_sched_ext_runtime import acceptance_passed, associations


def _event(event_type: str, task_id: str, pid: int, **data):
    return {
        "event_type": event_type,
        "task_id": task_id,
        "data": {"pid": pid, **data},
    }


def test_associations_links_policy_attach_and_cleanup_by_task() -> None:
    events = [
        _event(
            "kernel.sched_ext_policy_published",
            "task-1",
            123,
            readback_verified=True,
        ),
        _event("kernel.sched_ext_worker_attached", "task-1", 123),
        _event("kernel.sched_ext_policy_removed", "task-1", 123),
    ]

    assert associations(events)["task-1"] == {
        "published": {"pid": 123, "readback_verified": True},
        "attached": {"pid": 123},
        "removed": {"pid": 123},
    }


def test_stage3_acceptance_requires_clean_map_and_policy_hits() -> None:
    summary = {
        "preflight_accepted": True,
        "scheduler_name": "agentos_scx",
        "partial_mode": True,
        "workers": {
            "attempted": 10,
            "completed": 10,
            "complete_associations": 10,
        },
        "policy_readbacks_verified": 10,
        "policy_map_entries_after_runtime": 0,
        "policy_hit": 10,
        "loader_exit_code": 0,
        "final_sched_ext_state": "disabled",
        "kernel_log_errors": [],
        "failure_reason": None,
    }

    assert acceptance_passed(summary) is True
    summary["policy_map_entries_after_runtime"] = 1
    assert acceptance_passed(summary) is False
    summary["policy_map_entries_after_runtime"] = 0
    summary["policy_hit"] = 0
    assert acceptance_passed(summary) is False
