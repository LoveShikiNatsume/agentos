"""Runtime adapter for binding subprocess workers to agentos_scx."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from agentos.events.types import EventCategory, EventSeverity, RuntimeEvent
from agentos.isolation.bpf_schema import SchedPolicyKeyV1, SchedPolicyValueV1
from agentos.isolation.bpf_schema import SCHED_POLICY_SCHEMA_VERSION
from agentos.isolation.kernel_adapter import (
    KernelApplyResult,
    KernelWorkerBindResult,
    PressureSample,
    UserSpaceKernelAdapter,
)
from agentos.isolation.sched_ext import KernelProbeResult
from agentos.isolation.sched_ext_control import (
    SchedExtControlClient,
    SchedExtControlError,
    SchedExtPinMissingError,
)


SCHED_EXT = 7
DEFAULT_POLL_INTERVAL_MS = 500
DEFAULT_TRANSIENT_FAILURE_THRESHOLD = 3
DEFAULT_RECOVERY_SUCCESS_THRESHOLD = 2


class SchedExtLifecycleState(str, Enum):
    DISABLED = "disabled"
    LOADING = "loading"
    ACTIVE = "active"
    DEGRADED = "degraded"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class SchedExtHealthResult:
    state: SchedExtLifecycleState
    usable: bool
    fatal: bool = False
    reason: str | None = None
    health: dict[str, Any] = field(default_factory=dict)
    events: list[RuntimeEvent] = field(default_factory=list)


def _read_ext_status(pid: int, proc_root: Path = Path("/proc")) -> tuple[int, int]:
    policy = -1
    enabled = -1
    text = (proc_root / str(pid) / "sched").read_text(
        encoding="utf-8", errors="replace"
    )
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("policy"):
            policy = int(stripped.rsplit(":", 1)[1].strip())
        elif stripped.startswith("ext.enabled"):
            enabled = int(stripped.rsplit(":", 1)[1].strip())
    return policy, enabled


class SchedExtKernelAdapter:
    supports_sched_ext_binding = True

    def __init__(
        self,
        *,
        control: SchedExtControlClient,
        base: UserSpaceKernelAdapter | None = None,
        mode: str = "auto",
        sched_setter: Callable[[int], None] | None = None,
        sched_resetter: Callable[[int], None] | None = None,
        status_reader: Callable[[int], tuple[int, int]] = _read_ext_status,
        probe_result: KernelProbeResult | None = None,
        config_flags: int = 0,
        heartbeat_interval_ms: int = 500,
        health_check_interval_ms: int = DEFAULT_POLL_INTERVAL_MS,
        transient_failure_threshold: int = DEFAULT_TRANSIENT_FAILURE_THRESHOLD,
        recovery_success_threshold: int = DEFAULT_RECOVERY_SUCCESS_THRESHOLD,
    ) -> None:
        if mode not in {"auto", "strict"}:
            raise ValueError("sched_ext mode must be 'auto' or 'strict'")
        if not 100 <= heartbeat_interval_ms <= 5_000:
            raise ValueError("heartbeat_interval_ms must be in 100..5000")
        if not 100 <= health_check_interval_ms <= 5_000:
            raise ValueError("health_check_interval_ms must be in 100..5000")
        if transient_failure_threshold < 1 or recovery_success_threshold < 1:
            raise ValueError("lifecycle thresholds must be positive")
        self.strict_sched_ext = mode == "strict"
        self._mode = mode
        self._control = control
        self._base = base or UserSpaceKernelAdapter()
        self._sched_setter = sched_setter or self._set_sched_ext
        self._sched_resetter = sched_resetter or self._set_sched_other
        self._status_reader = status_reader
        self._probe = probe_result or KernelProbeResult(
            sched_ext="available",
            bpf_map="pinned_probe",
            fallback_mode="none",
            details={"ops": "agentos_scx"},
        )
        self._decisions: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        self._bound: dict[int, SchedPolicyKeyV1] = {}
        self._bound_policies: dict[int, SchedPolicyValueV1] = {}
        self._renewable_pids: set[int] = set()
        self._config_flags = config_flags
        self._heartbeat_interval_ns = heartbeat_interval_ms * 1_000_000
        self._health_check_interval_ns = health_check_interval_ms * 1_000_000
        self._transient_failure_threshold = transient_failure_threshold
        self._recovery_success_threshold = recovery_success_threshold
        self._last_health_check_ns = 0
        self._last_lease_refresh_ns = 0
        self._last_health: dict[str, Any] = {}
        self._state = SchedExtLifecycleState.ACTIVE
        self._consecutive_failures = 0
        self._consecutive_lease_failures = 0
        self._consecutive_successes = recovery_success_threshold
        self._fallback_count = 0
        self._recovery_count = 0
        self._strict_failure_reason: str | None = None

    @classmethod
    def from_mode(
        cls,
        *,
        mode: str,
        control: SchedExtControlClient,
        base: UserSpaceKernelAdapter | None = None,
        **kwargs: Any,
    ) -> "SchedExtKernelAdapter | UserSpaceKernelAdapter":
        fallback = base or UserSpaceKernelAdapter()
        try:
            control.validate_ready()
        except Exception:
            if mode == "strict":
                raise
            if mode == "auto":
                return fallback
            raise ValueError("sched_ext mode must be 'auto' or 'strict'")
        return cls(control=control, base=fallback, mode=mode, **kwargs)

    @staticmethod
    def _set_sched_ext(pid: int) -> None:
        os.sched_setscheduler(pid, SCHED_EXT, os.sched_param(0))

    @staticmethod
    def _set_sched_other(pid: int) -> None:
        os.sched_setscheduler(pid, os.SCHED_OTHER, os.sched_param(0))

    def probe(self) -> KernelProbeResult:
        return self._probe

    def apply_agent_quota(self, **kwargs: Any) -> KernelApplyResult:
        return self._base.apply_agent_quota(**kwargs)

    def publish_decision(self, decision: dict[str, Any]) -> list[RuntimeEvent]:
        decision["kernel_integration"] = "sched_ext"
        generation = int(decision.get("policy_generation", 0))
        for task_id, task_decision in decision.get("tasks", {}).items():
            if isinstance(task_decision, dict):
                task_decision["policy_generation"] = generation
                self._decisions[task_id] = (decision, task_decision)
        events = self._base.publish_decision(decision)
        if self._state is not SchedExtLifecycleState.ACTIVE:
            return events
        try:
            self._control.update_global_config(
                generation=generation,
                flags=self._config_flags,
            )
        except Exception as exc:
            failure = self._health_failure(
                trigger="publish_decision", reason=str(exc), definitive=False
            )
            events.extend(failure.events)
        return events

    def bind_worker_policy(
        self, *, agent_id: str, task_id: str, pid: int
    ) -> KernelWorkerBindResult:
        key = SchedPolicyKeyV1(tgid=pid, pid=pid)
        if self._state is not SchedExtLifecycleState.ACTIVE:
            return self._failed(
                agent_id,
                task_id,
                pid,
                f"sched_ext lifecycle is {self._state.value}",
            )
        decision_pair = self._decisions.get(task_id)
        if decision_pair is None:
            return self._failed(agent_id, task_id, pid, "task decision is unavailable")
        decision, task = decision_pair
        pressure = decision.get("pressure", {}).get("effective_pressure_level", "normal")
        policy = SchedPolicyValueV1.from_fields(
            queue_class=str(task.get("queue_class", "normal")),
            weight=int(task.get("kernel_weight", 100)),
            slice_us=int(task.get("slice_us", 5_000)),
            pressure_level=str(pressure),
            admitted=task.get("admission_decision") != "reject",
            critical=task.get("sla_class") == "critical",
            generation=int(task.get("policy_generation", 0)),
            updated_at_ns=time.monotonic_ns(),
        )
        try:
            readback = self._control.put_policy(key, policy)
            self._sched_setter(pid)
            actual_policy, ext_enabled = self._status_reader(pid)
            if actual_policy != SCHED_EXT or ext_enabled != 1:
                raise SchedExtControlError(
                    f"worker attach verification failed: policy={actual_policy}, "
                    f"ext.enabled={ext_enabled}"
                )
        except Exception as exc:
            try:
                self._sched_resetter(pid)
            except Exception:
                pass
            try:
                self._control.delete_policy(key)
            except Exception:
                pass
            result = self._failed(agent_id, task_id, pid, str(exc))
            failure = self._health_failure(
                trigger="worker_bind", reason=str(exc), definitive=False
            )
            result.events.extend(failure.events)
            return result

        self._bound[pid] = key
        self._bound_policies[pid] = policy
        self._renewable_pids.add(pid)
        event = RuntimeEvent(
            event_type="kernel.sched_ext_policy_published",
            agent_id=agent_id,
            task_id=task_id,
            category=EventCategory.SCHEDULER,
            data={
                "pid": pid,
                "tgid": pid,
                "schema_version": "sched_policy.v1",
                "policy": self._control.policy_to_dict(readback),
                "policy_base": task.get("kernel_policy_base"),
                "policy_adjustment": task.get("kernel_policy_adjustment", "none"),
                "pressure_level": pressure,
                "readback_verified": True,
            },
        )
        attached = RuntimeEvent(
            event_type="kernel.sched_ext_worker_attached",
            agent_id=agent_id,
            task_id=task_id,
            category=EventCategory.SCHEDULER,
            data={"pid": pid, "policy": SCHED_EXT, "ext_enabled": 1},
        )
        return KernelWorkerBindResult(
            available=True,
            attached=True,
            mode="sched_ext",
            pid=pid,
            events=[event, attached],
        )

    def _failed(
        self, agent_id: str, task_id: str, pid: int, reason: str
    ) -> KernelWorkerBindResult:
        return KernelWorkerBindResult(
            available=False,
            attached=False,
            mode="sched_ext",
            pid=pid,
            reason=reason,
            events=[
                RuntimeEvent(
                    event_type="kernel.sched_ext_attach_failed",
                    agent_id=agent_id,
                    task_id=task_id,
                    category=EventCategory.SCHEDULER,
                    severity=(
                        EventSeverity.ERROR
                        if self.strict_sched_ext
                        else EventSeverity.WARNING
                    ),
                    data={"pid": pid, "reason": reason, "mode": self._mode},
                )
            ],
        )

    def remove_worker_policy(
        self, *, agent_id: str, task_id: str, pid: int
    ) -> list[RuntimeEvent]:
        key = self._bound.pop(pid, None)
        self._bound_policies.pop(pid, None)
        self._renewable_pids.discard(pid)
        if key is None:
            return []
        try:
            self._control.delete_policy(key)
        except Exception as exc:
            events = [
                RuntimeEvent(
                    event_type="kernel.sched_ext_policy_remove_failed",
                    agent_id=agent_id,
                    task_id=task_id,
                    category=EventCategory.SCHEDULER,
                    severity=EventSeverity.WARNING,
                    data={"pid": pid, "tgid": key.tgid, "reason": str(exc)},
                )
            ]
            failure = self._health_failure(
                trigger="policy_remove", reason=str(exc), definitive=False
            )
            events.extend(failure.events)
            return events
        return [
            RuntimeEvent(
                event_type="kernel.sched_ext_policy_removed",
                agent_id=agent_id,
                task_id=task_id,
                category=EventCategory.SCHEDULER,
                data={"pid": pid, "tgid": key.tgid},
            )
        ]

    def sample_pressure(self, **kwargs: Any) -> PressureSample:
        return self._base.sample_pressure(**kwargs)

    def maintain_sched_ext(
        self, *, trigger: str = "poll", force: bool = False
    ) -> SchedExtHealthResult:
        now = time.monotonic_ns()
        if (
            not force
            and self._last_health_check_ns
            and now - self._last_health_check_ns < self._health_check_interval_ns
        ):
            return self._health_result()
        self._last_health_check_ns = now
        try:
            health = self._control.read_health(now_ns=now)
        except Exception as exc:
            return self._health_failure(
                trigger=trigger,
                reason=str(exc),
                definitive=isinstance(exc, SchedExtPinMissingError),
            )
        self._last_health = dict(health)
        if self._strict_failure_reason is not None:
            return self._health_result()
        reason = self._definitive_health_failure(health)
        if reason is not None:
            return self._health_failure(
                trigger=trigger, reason=reason, definitive=True
            )

        if self._state is not SchedExtLifecycleState.ACTIVE:
            if self._consecutive_lease_failures and self._renewable_pids:
                lease_failure = self._refresh_policy_leases(now, force=True)
                if lease_failure is not None:
                    return lease_failure
            self._consecutive_successes += 1
            self._consecutive_failures = 0
            if self._consecutive_successes >= self._recovery_success_threshold:
                try:
                    self._control.update_global_config(
                        generation=self._latest_generation(),
                        flags=self._config_flags,
                    )
                except Exception as exc:
                    return self._health_failure(
                        trigger="recovery_generation",
                        reason=str(exc),
                        definitive=False,
                    )
                event = self._transition(
                    SchedExtLifecycleState.ACTIVE,
                    trigger=trigger,
                    reason="sched_ext health recovered",
                )
                return self._health_result(events=[event] if event else [])
            return self._health_result()

        self._consecutive_failures = 0
        self._consecutive_successes = self._recovery_success_threshold
        if now - self._last_lease_refresh_ns >= self._heartbeat_interval_ns:
            lease_failure = self._refresh_policy_leases(now)
            if lease_failure is not None:
                return lease_failure
        return self._health_result()

    def _refresh_policy_leases(
        self, now: int, *, force: bool = False
    ) -> SchedExtHealthResult | None:
        if not force and now - self._last_lease_refresh_ns < self._heartbeat_interval_ns:
            return None
        try:
            for pid in list(self._renewable_pids):
                key = self._bound.get(pid)
                policy = self._bound_policies.get(pid)
                if key is None or policy is None:
                    continue
                renewed = replace(policy, updated_at_ns=now)
                self._control.put_policy(key, renewed)
                self._bound_policies[pid] = renewed
            self._last_lease_refresh_ns = now
            self._consecutive_lease_failures = 0
            return None
        except Exception as exc:
            self._consecutive_lease_failures += 1
            self._consecutive_failures = max(
                self._consecutive_failures,
                self._consecutive_lease_failures - 1,
            )
            return self._health_failure(
                trigger="policy_lease", reason=str(exc), definitive=False
            )

    def _definitive_health_failure(self, health: dict[str, Any]) -> str | None:
        expected = {
            "state": "enabled",
            "switch_all": "0",
            "root_ops": "agentos_scx",
            "schema_version": SCHED_POLICY_SCHEMA_VERSION,
        }
        for field_name, wanted in expected.items():
            if health.get(field_name) != wanted:
                return f"{field_name}={health.get(field_name)!r}, expected {wanted!r}"
        if int(health.get("enable_seq", -1)) < 1:
            return "enable_seq is not positive"
        if int(health.get("loader_pid", 0)) <= 0:
            return "loader_pid is not positive"
        timeout_ns = int(health.get("heartbeat_timeout_seconds", 3)) * 1_000_000_000
        if int(health.get("heartbeat_age_ns", timeout_ns + 1)) > timeout_ns:
            return "loader heartbeat expired"
        return None

    def _health_failure(
        self, *, trigger: str, reason: str, definitive: bool
    ) -> SchedExtHealthResult:
        self._consecutive_failures += 1
        self._consecutive_successes = 0
        if self.strict_sched_ext:
            self._strict_failure_reason = reason
        target = SchedExtLifecycleState.DEGRADED
        if (
            definitive
            or self.strict_sched_ext
            or self._state is SchedExtLifecycleState.FALLBACK
            or self._consecutive_failures >= self._transient_failure_threshold
        ):
            target = SchedExtLifecycleState.FALLBACK
        event = self._transition(target, trigger=trigger, reason=reason)
        return self._health_result(
            fatal=self.strict_sched_ext,
            reason=reason,
            events=[event] if event else [],
        )

    def _transition(
        self,
        state: SchedExtLifecycleState,
        *,
        trigger: str,
        reason: str,
    ) -> RuntimeEvent | None:
        previous = self._state
        if previous is state:
            return None
        self._state = state
        fallback_reset_pids: list[int] = []
        fallback_reset_failures: dict[str, str] = {}
        if state is SchedExtLifecycleState.FALLBACK:
            self._fallback_count += 1
            for pid in sorted(self._bound):
                try:
                    self._sched_resetter(pid)
                    fallback_reset_pids.append(pid)
                except Exception as exc:
                    fallback_reset_failures[str(pid)] = str(exc)
            self._renewable_pids.clear()
            event_type = "kernel.sched_ext_fallback"
        elif state is SchedExtLifecycleState.DEGRADED:
            event_type = "kernel.sched_ext_degraded"
        elif state is SchedExtLifecycleState.ACTIVE:
            self._recovery_count += 1
            event_type = "kernel.sched_ext_recovered"
        else:
            return None
        return RuntimeEvent(
            event_type=event_type,
            category=EventCategory.SCHEDULER,
            severity=(
                EventSeverity.WARNING
                if state is not SchedExtLifecycleState.ACTIVE
                else EventSeverity.INFO
            ),
            data={
                "previous_state": previous.value,
                "current_state": state.value,
                "trigger": trigger,
                "reason": reason,
                "mode": self._mode,
                "loader_pid": self._last_health.get("loader_pid"),
                "enable_seq": self._last_health.get("enable_seq"),
                "consecutive_failures": self._consecutive_failures,
                "bound_pid_count": len(self._bound),
                "monotonic_ns": time.monotonic_ns(),
                "fallback_reset_pids": fallback_reset_pids,
                "fallback_reset_failures": fallback_reset_failures,
            },
        )

    def _health_result(
        self,
        *,
        fatal: bool = False,
        reason: str | None = None,
        events: list[RuntimeEvent] | None = None,
    ) -> SchedExtHealthResult:
        return SchedExtHealthResult(
            state=self._state,
            usable=self._state is SchedExtLifecycleState.ACTIVE,
            fatal=fatal or self._strict_failure_reason is not None,
            reason=reason or self._strict_failure_reason,
            health=dict(self._last_health),
            events=list(events or []),
        )

    def _latest_generation(self) -> int:
        return max(
            (int(task.get("policy_generation", 0)) for _, task in self._decisions.values()),
            default=0,
        )

    def cleanup_agent(self, **kwargs: Any) -> list[RuntimeEvent]:
        return self._base.cleanup_agent(**kwargs)

    def snapshot(self) -> dict[str, Any]:
        snapshot = self._base.snapshot()
        snapshot.update(
            {
                "mode": "sched_ext",
                "sched_ext_mode": self._mode,
                "bound_pids": sorted(self._bound),
                "policy_task_ids": sorted(self._decisions),
                "lifecycle_state": self._state.value,
                "last_health": dict(self._last_health),
                "consecutive_failures": self._consecutive_failures,
                "fallback_count": self._fallback_count,
                "recovery_count": self._recovery_count,
                "last_lease_refresh_ns": self._last_lease_refresh_ns,
            }
        )
        return snapshot
