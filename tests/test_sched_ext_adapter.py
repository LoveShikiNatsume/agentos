from __future__ import annotations

from dataclasses import dataclass

import pytest

from agentos.isolation.bpf_schema import SchedPolicyKeyV1, SchedPolicyValueV1
from agentos.isolation.kernel_adapter import UserSpaceKernelAdapter
from agentos.isolation.sched_ext_adapter import SchedExtKernelAdapter
from agentos.isolation.sched_ext_adapter import SchedExtLifecycleState
from agentos.isolation.sched_ext_control import SchedExtPinMissingError
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import Runtime


@dataclass
class _Control:
    fail_validation: bool = False

    def __post_init__(self) -> None:
        self.entries: dict[SchedPolicyKeyV1, SchedPolicyValueV1] = {}
        self.configs: list[int] = []
        self.put_calls = 0
        self.health = {
            "state": "enabled",
            "switch_all": "0",
            "root_ops": "agentos_scx",
            "enable_seq": 1,
            "schema_version": 1,
            "loader_pid": 42,
            "heartbeat_timeout_seconds": 3,
            "heartbeat_age_ns": 0,
        }

    def validate_ready(self):
        if self.fail_validation:
            raise RuntimeError("maps unavailable")
        return {}

    def update_global_config(self, *, generation: int, **_kwargs):
        self.configs.append(generation)
        return {}

    def put_policy(self, key, policy):
        self.put_calls += 1
        self.entries[key] = policy
        return policy

    def delete_policy(self, key):
        self.entries.pop(key, None)

    @staticmethod
    def policy_to_dict(policy):
        return {"version": policy.version, "generation": policy.generation}

    def read_health(self, *, now_ns=None):
        return dict(self.health)


class _Base(UserSpaceKernelAdapter):
    def __init__(self) -> None:
        pass

    def publish_decision(self, decision):
        return []

    def snapshot(self):
        return {"mode": "base"}


def _decision() -> dict:
    return {
        "policy_generation": 4,
        "pressure": {"effective_pressure_level": "warning"},
        "tasks": {
            "task-1": {
                "queue_class": "latency",
                "kernel_weight": 300,
                "slice_us": 2_000,
                "policy_generation": 4,
                "admission_decision": "allow",
                "sla_class": "critical",
            }
        },
    }


def test_adapter_publishes_binds_verifies_and_removes_worker() -> None:
    control = _Control()
    attached: list[int] = []
    adapter = SchedExtKernelAdapter(
        control=control,
        base=_Base(),
        mode="strict",
        sched_setter=attached.append,
        status_reader=lambda _pid: (7, 1),
    )
    decision = _decision()

    assert adapter.publish_decision(decision) == []
    result = adapter.bind_worker_policy(
        agent_id="agent-1", task_id="task-1", pid=123
    )

    assert result.attached is True
    assert attached == [123]
    assert decision["kernel_integration"] == "sched_ext"
    assert list(control.entries) == [SchedPolicyKeyV1(tgid=123, pid=123)]
    assert [event.event_type for event in result.events] == [
        "kernel.sched_ext_policy_published",
        "kernel.sched_ext_worker_attached",
    ]

    events = adapter.remove_worker_policy(
        agent_id="agent-1", task_id="task-1", pid=123
    )
    assert control.entries == {}
    assert events[0].event_type == "kernel.sched_ext_policy_removed"
    assert control.configs == [4]


def test_adapter_policy_leases_are_rate_limited(monkeypatch) -> None:
    control = _Control()
    moment = [1_000_000_000]
    monkeypatch.setattr(
        "agentos.isolation.sched_ext_adapter.time.monotonic_ns",
        lambda: moment[0],
    )
    adapter = SchedExtKernelAdapter(
        control=control,
        base=_Base(),
        heartbeat_interval_ms=500,
        sched_setter=lambda _pid: None,
        status_reader=lambda _pid: (7, 1),
    )

    adapter.publish_decision(_decision())
    adapter.bind_worker_policy(agent_id="agent-1", task_id="task-1", pid=123)
    adapter.maintain_sched_ext(force=True)
    moment[0] = 1_100_000_000
    adapter.maintain_sched_ext(force=True)
    moment[0] = 1_600_000_000
    adapter.maintain_sched_ext(force=True)

    assert control.configs == [4]
    assert control.put_calls == 3


def test_attach_verification_failure_rolls_back_policy() -> None:
    control = _Control()
    adapter = SchedExtKernelAdapter(
        control=control,
        base=_Base(),
        mode="auto",
        sched_setter=lambda _pid: None,
        status_reader=lambda _pid: (0, 0),
    )
    adapter.publish_decision(_decision())

    result = adapter.bind_worker_policy(
        agent_id="agent-1", task_id="task-1", pid=123
    )

    assert result.attached is False
    assert control.entries == {}
    assert [event.event_type for event in result.events] == [
        "kernel.sched_ext_attach_failed",
        "kernel.sched_ext_degraded",
    ]


def test_auto_falls_back_and_strict_propagates_map_validation_failure() -> None:
    control = _Control(fail_validation=True)
    base = _Base()

    assert SchedExtKernelAdapter.from_mode(
        mode="auto", control=control, base=base
    ) is base
    with pytest.raises(RuntimeError, match="maps unavailable"):
        SchedExtKernelAdapter.from_mode(
            mode="strict", control=control, base=base
        )


def test_runtime_sched_ext_mode_has_explicit_auto_and_strict_semantics() -> None:
    auto_runtime = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={},
        sched_ext_mode="auto",
        sched_ext_control=_Control(fail_validation=True),
    )
    assert auto_runtime._kernel_adapter.supports_sched_ext_binding is False

    with pytest.raises(RuntimeError, match="maps unavailable"):
        Runtime(
            llm_backend=MockLLMBackend(),
            handlers={},
            sched_ext_mode="strict",
            sched_ext_control=_Control(fail_validation=True),
        )


def test_runtime_rejects_ambiguous_sched_ext_adapter_configuration() -> None:
    with pytest.raises(ValueError, match="cannot both configure"):
        Runtime(
            llm_backend=MockLLMBackend(),
            handlers={},
            kernel_adapter=_Base(),
            sched_ext_mode="strict",
        )


def test_lifecycle_falls_back_once_and_recovers_after_two_healthy_polls() -> None:
    control = _Control()
    adapter = SchedExtKernelAdapter(control=control, base=_Base(), mode="auto")
    control.health["state"] = "disabled"

    failed = adapter.maintain_sched_ext(trigger="loader_exit", force=True)
    repeated = adapter.maintain_sched_ext(trigger="loader_exit", force=True)

    assert failed.state is SchedExtLifecycleState.FALLBACK
    assert [event.event_type for event in failed.events] == [
        "kernel.sched_ext_fallback"
    ]
    assert repeated.events == []

    control.health["state"] = "enabled"
    # read_health already validated the current pin identities and schemas;
    # recovery must not repeat the expensive full validation.
    control.fail_validation = True
    first = adapter.maintain_sched_ext(trigger="loader_restart", force=True)
    second = adapter.maintain_sched_ext(trigger="loader_restart", force=True)

    assert first.state is SchedExtLifecycleState.FALLBACK
    assert second.state is SchedExtLifecycleState.ACTIVE
    assert [event.event_type for event in second.events] == [
        "kernel.sched_ext_recovered"
    ]
    assert control.configs == [0]


def test_fallback_resets_bound_worker_to_cfs_once_and_keeps_cleanup_tracking() -> None:
    control = _Control()
    reset_pids: list[int] = []
    adapter = SchedExtKernelAdapter(
        control=control,
        base=_Base(),
        mode="auto",
        sched_setter=lambda _pid: None,
        sched_resetter=reset_pids.append,
        status_reader=lambda _pid: (7, 1),
    )
    adapter.publish_decision(_decision())
    adapter.bind_worker_policy(agent_id="agent-1", task_id="task-1", pid=123)
    control.health["state"] = "disabled"

    first = adapter.maintain_sched_ext(trigger="loader_exit", force=True)
    repeated = adapter.maintain_sched_ext(trigger="loader_exit", force=True)

    assert reset_pids == [123]
    assert first.events[0].data["fallback_reset_pids"] == [123]
    assert first.events[0].data["fallback_reset_failures"] == {}
    assert repeated.events == []
    assert adapter.snapshot()["bound_pids"] == [123]

    events = adapter.remove_worker_policy(
        agent_id="agent-1", task_id="task-1", pid=123
    )
    assert events[0].event_type == "kernel.sched_ext_policy_removed"


def test_fallback_records_worker_reset_failure_without_repeating_it() -> None:
    control = _Control()
    attempts: list[int] = []

    def fail_reset(pid: int) -> None:
        attempts.append(pid)
        raise PermissionError("cannot reset scheduler policy")

    adapter = SchedExtKernelAdapter(
        control=control,
        base=_Base(),
        mode="auto",
        sched_setter=lambda _pid: None,
        sched_resetter=fail_reset,
        status_reader=lambda _pid: (7, 1),
    )
    adapter.publish_decision(_decision())
    adapter.bind_worker_policy(agent_id="agent-1", task_id="task-1", pid=123)
    control.health["state"] = "disabled"

    first = adapter.maintain_sched_ext(trigger="loader_exit", force=True)
    adapter.maintain_sched_ext(trigger="loader_exit", force=True)

    assert attempts == [123]
    assert first.events[0].data["fallback_reset_pids"] == []
    assert first.events[0].data["fallback_reset_failures"] == {
        "123": "cannot reset scheduler policy"
    }


def test_transient_control_errors_degrade_then_fallback() -> None:
    control = _Control()
    adapter = SchedExtKernelAdapter(control=control, base=_Base(), mode="auto")

    def fail_health(**_kwargs):
        raise RuntimeError("temporary bpftool failure")

    control.read_health = fail_health
    first = adapter.maintain_sched_ext(force=True)
    second = adapter.maintain_sched_ext(force=True)
    third = adapter.maintain_sched_ext(force=True)

    assert first.state is SchedExtLifecycleState.DEGRADED
    assert second.events == []
    assert third.state is SchedExtLifecycleState.FALLBACK
    assert third.events[0].event_type == "kernel.sched_ext_fallback"


def test_missing_required_pin_falls_back_on_first_health_check() -> None:
    control = _Control()
    adapter = SchedExtKernelAdapter(control=control, base=_Base(), mode="auto")

    def missing_pin(**_kwargs):
        raise SchedExtPinMissingError("required pin task_policy_map is missing")

    control.read_health = missing_pin
    result = adapter.maintain_sched_ext(trigger="policy_map_unpin", force=True)

    assert result.state is SchedExtLifecycleState.FALLBACK
    assert [event.event_type for event in result.events] == [
        "kernel.sched_ext_fallback"
    ]
    assert result.events[0].data["consecutive_failures"] == 1


def test_policy_lease_failures_accumulate_across_healthy_reads(monkeypatch) -> None:
    control = _Control()
    moment = [1_000_000_000]
    monkeypatch.setattr(
        "agentos.isolation.sched_ext_adapter.time.monotonic_ns",
        lambda: moment[0],
    )
    adapter = SchedExtKernelAdapter(
        control=control,
        base=_Base(),
        mode="auto",
        heartbeat_interval_ms=500,
        sched_setter=lambda _pid: None,
        status_reader=lambda _pid: (7, 1),
    )
    adapter.publish_decision(_decision())
    adapter.bind_worker_policy(agent_id="agent-1", task_id="task-1", pid=123)

    def fail_lease(_key, _policy):
        raise RuntimeError("update failed: Operation not permitted")

    control.put_policy = fail_lease
    states = []
    emitted = []
    for value in (1_600_000_000, 2_200_000_000, 2_800_000_000):
        moment[0] = value
        result = adapter.maintain_sched_ext(force=True)
        states.append(result.state)
        emitted.extend(event.event_type for event in result.events)

    assert states == [
        SchedExtLifecycleState.DEGRADED,
        SchedExtLifecycleState.DEGRADED,
        SchedExtLifecycleState.FALLBACK,
    ]
    assert emitted == [
        "kernel.sched_ext_degraded",
        "kernel.sched_ext_fallback",
    ]


def test_strict_lifecycle_failure_is_immediately_fatal() -> None:
    control = _Control()
    adapter = SchedExtKernelAdapter(control=control, base=_Base(), mode="strict")
    control.health["schema_version"] = 999

    result = adapter.maintain_sched_ext(trigger="schema", force=True)

    assert result.fatal is True
    assert result.state is SchedExtLifecycleState.FALLBACK


def test_strict_publish_failure_emits_fallback_and_remains_fatal() -> None:
    control = _Control()
    adapter = SchedExtKernelAdapter(control=control, base=_Base(), mode="strict")

    def fail_update(**_kwargs):
        raise RuntimeError("global config frozen")

    control.update_global_config = fail_update
    events = adapter.publish_decision(_decision())
    health = adapter.maintain_sched_ext(force=True)

    assert [event.event_type for event in events] == [
        "kernel.sched_ext_fallback"
    ]
    assert health.fatal is True
    assert health.state is SchedExtLifecycleState.FALLBACK


def test_snapshot_exposes_lifecycle_health_and_bound_workers() -> None:
    control = _Control()
    adapter = SchedExtKernelAdapter(
        control=control,
        base=_Base(),
        sched_setter=lambda _pid: None,
        status_reader=lambda _pid: (7, 1),
    )
    adapter.publish_decision(_decision())
    adapter.bind_worker_policy(agent_id="agent-1", task_id="task-1", pid=123)
    adapter.maintain_sched_ext(force=True)

    snapshot = adapter.snapshot()

    assert snapshot["lifecycle_state"] == "active"
    assert snapshot["bound_pids"] == [123]
    assert snapshot["last_health"]["loader_pid"] == 42
    assert snapshot["last_lease_refresh_ns"] > 0
