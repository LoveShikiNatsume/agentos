from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from agentos.core.resources import ResourceQuota, ResourceUsage
from agentos.events.types import EventCategory, EventSeverity, RuntimeEvent
from agentos.isolation.bpf import (
    BPFMapAdapter,
    BPFTaskRecord,
    MockBPFMapAdapter,
    task_record_from_decision,
)
from agentos.isolation.bpftool import (
    BPFMapError,
    BpftoolPinnedMapAdapter,
    run_bpftool_json,
)
from agentos.isolation.cgroup import CgroupV2Manager
from agentos.isolation.pressure import (
    max_pressure_level,
    pressure_level_from_psi,
    read_psi_avg10_with_status,
)
from agentos.isolation.sched_ext import KernelProbeResult, probe_sched_ext


_DEFAULT_PSI_PATHS: dict[str, str] = {
    "cpu": "/proc/pressure/cpu",
    "memory": "/proc/pressure/memory",
    "io": "/proc/pressure/io",
}


@dataclass(frozen=True)
class KernelApplyResult:
    available: bool
    mode: str
    cgroup_path: Path | None = None
    reason: str | None = None
    events: list[RuntimeEvent] = field(default_factory=list)


@dataclass(frozen=True)
class KernelWorkerBindResult:
    available: bool
    attached: bool
    mode: str
    pid: int
    reason: str | None = None
    events: list[RuntimeEvent] = field(default_factory=list)


@dataclass(frozen=True)
class PressureSample:
    cpu_some_avg10: float = 0.0
    memory_some_avg10: float = 0.0
    io_some_avg10: float = 0.0
    cgroup_usage: ResourceUsage | None = None
    pressure_level: str = "normal"
    pressure_source: str = "system"
    pressure_sources: dict[str, str] = field(
        default_factory=lambda: {
            "cpu": "system",
            "memory": "system",
            "io": "system",
        }
    )
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_some_avg10": self.cpu_some_avg10,
            "memory_some_avg10": self.memory_some_avg10,
            "io_some_avg10": self.io_some_avg10,
            "pressure_level": self.pressure_level,
            "pressure_source": self.pressure_source,
            "pressure_sources": dict(self.pressure_sources),
            "reason": self.reason,
            "cgroup_usage": (
                self.cgroup_usage.to_dict() if self.cgroup_usage is not None else None
            ),
        }


class KernelSchedulerAdapter(Protocol):
    supports_sched_ext_binding: bool
    strict_sched_ext: bool
    def probe(self) -> KernelProbeResult: ...

    def apply_agent_quota(
        self,
        *,
        agent_id: str,
        task_id: str,
        pid: int,
        quota: ResourceQuota,
    ) -> KernelApplyResult: ...

    def publish_decision(self, decision: dict[str, Any]) -> list[RuntimeEvent]: ...

    def bind_worker_policy(
        self, *, agent_id: str, task_id: str, pid: int
    ) -> KernelWorkerBindResult: ...

    def remove_worker_policy(
        self, *, agent_id: str, task_id: str, pid: int
    ) -> list[RuntimeEvent]: ...

    def maintain_sched_ext(
        self, *, trigger: str = "poll", force: bool = False
    ) -> Any: ...

    def sample_pressure(self, *, cgroup_path: Path | None = None) -> PressureSample: ...

    def cleanup_agent(
        self,
        *,
        agent_id: str,
        cgroup_path: Path,
    ) -> list[RuntimeEvent]: ...

    def snapshot(self) -> dict[str, Any]: ...


class UserSpaceKernelAdapter:
    supports_sched_ext_binding = False
    strict_sched_ext = False
    def __init__(
        self,
        *,
        cgroup_manager: CgroupV2Manager | None = None,
        bpf_map_adapter: BPFMapAdapter | None = None,
        requested_bpf_mode: str | None = None,
        bpf_fallback_reason: str | None = None,
        probe_result: KernelProbeResult | None = None,
        psi_paths: dict[str, str | Path] | None = None,
    ) -> None:
        self._cgroup = cgroup_manager or CgroupV2Manager()
        self._bpf: BPFMapAdapter = (
            bpf_map_adapter
            if bpf_map_adapter is not None
            else MockBPFMapAdapter()
        )
        self._requested_bpf_mode = requested_bpf_mode or self._bpf.mode
        self._bpf_fallback_reason = bpf_fallback_reason
        self._probe = probe_result or probe_sched_ext()
        self._agent_cgroups: dict[str, Path] = {}
        # Merge user-supplied PSI paths with defaults so missing keys don't crash
        merged: dict[str, str | Path] = dict(_DEFAULT_PSI_PATHS)
        if psi_paths is not None:
            merged.update(psi_paths)
        self._psi_paths: dict[str, str | Path] = merged

    @classmethod
    def from_bpf_mode(
        cls,
        mode: str = "mock",
        *,
        bpf_pin_path: str | Path = "/sys/fs/bpf/agentos/tasks",
        bpftool_runner=None,
        **kwargs: Any,
    ) -> "UserSpaceKernelAdapter":
        if mode == "mock":
            return cls(
                bpf_map_adapter=MockBPFMapAdapter(),
                requested_bpf_mode="mock",
                **kwargs,
            )
        if mode not in {"bpftool", "auto"}:
            raise ValueError(f"unsupported BPF mode: {mode!r}")

        try:
            real = BpftoolPinnedMapAdapter(
                pin_path=bpf_pin_path,
                runner=bpftool_runner or run_bpftool_json,
            )
            real.create_if_missing()
        except BPFMapError as exc:
            if mode == "bpftool":
                raise
            return cls(
                bpf_map_adapter=MockBPFMapAdapter(),
                requested_bpf_mode="auto",
                bpf_fallback_reason=str(exc),
                **kwargs,
            )

        return cls(
            bpf_map_adapter=real,
            requested_bpf_mode=mode,
            **kwargs,
        )

    def probe(self) -> KernelProbeResult:
        return self._probe

    def maintain_sched_ext(
        self, *, trigger: str = "poll", force: bool = False
    ) -> Any:
        from agentos.isolation.sched_ext_adapter import (
            SchedExtHealthResult,
            SchedExtLifecycleState,
        )

        return SchedExtHealthResult(
            state=SchedExtLifecycleState.DISABLED,
            usable=False,
        )

    def apply_agent_quota(
        self,
        *,
        agent_id: str,
        task_id: str,
        pid: int,
        quota: ResourceQuota,
    ) -> KernelApplyResult:
        events: list[RuntimeEvent] = []
        result = self._cgroup.create(agent_id, quota)
        events.append(result.event)

        if not result.available:
            return KernelApplyResult(
                available=False,
                mode="cgroup_v2",
                cgroup_path=result.path,
                reason=result.reason,
                events=events,
            )

        self._agent_cgroups[agent_id] = result.path

        pid_result = self._cgroup.add_pid(result.path, pid)
        events.append(pid_result.event)

        if not pid_result.available:
            try:
                cleanup_result = self._cgroup.cleanup(result.path)
                events.append(cleanup_result.event)
            except Exception as exc:
                events.append(
                    RuntimeEvent(
                        event_type="resource.cgroup_cleanup_failed",
                        category=EventCategory.RESOURCE,
                        severity=EventSeverity.WARNING,
                        data={"path": str(result.path), "reason": str(exc)},
                    )
                )
            self._agent_cgroups.pop(agent_id, None)
            return KernelApplyResult(
                available=False,
                mode="cgroup_v2",
                cgroup_path=result.path,
                reason=pid_result.reason,
                events=events,
            )

        events.append(
            RuntimeEvent(
                event_type="resource.cgroup_quota_applied",
                agent_id=agent_id,
                task_id=task_id,
                category=EventCategory.RESOURCE,
                severity=EventSeverity.INFO,
                data={
                    "path": str(result.path),
                    "pid": pid,
                    "limits": quota.to_dict(),
                    "mode": "cgroup_v2",
                    "pid_attached": True,
                },
            )
        )

        return KernelApplyResult(
            available=True,
            mode="cgroup_v2",
            cgroup_path=result.path,
            events=events,
        )

    def publish_decision(self, decision: dict[str, Any]) -> list[RuntimeEvent]:
        tasks = decision.get("tasks", {})
        records: list[BPFTaskRecord] = [
            task_record_from_decision(
                task_id=task_id,
                task_decision=task_decision,
                scheduler_decision=decision,
            )
            for task_id, task_decision in tasks.items()
        ]

        attempted_task_ids = [record.task_id for record in records]
        updated_task_ids: list[str] = []
        failed_task_ids: list[str] = []
        real_updated_task_ids: list[str] = []
        real_failed_task_ids: list[str] = []
        started_in_real_mode = self._bpf.mode == "bpftool"

        for index, record in enumerate(records):
            try:
                self._bpf.update_task(record)
            except BPFMapError as exc:
                if self._requested_bpf_mode != "auto":
                    raise
                real_failed_task_ids = [
                    item.task_id for item in records[index:]
                ]
                self._bpf_fallback_reason = str(exc)
                self._bpf = MockBPFMapAdapter()
                updated_task_ids.clear()
                for fallback_record in records:
                    self._bpf.update_task(fallback_record)
                    updated_task_ids.append(fallback_record.task_id)
                failed_task_ids.clear()
                break
            else:
                updated_task_ids.append(record.task_id)
                if started_in_real_mode:
                    real_updated_task_ids.append(record.task_id)

        updated_task_ids = sorted(updated_task_ids)
        failed_task_ids = sorted(failed_task_ids)
        fallback_active = self._bpf_fallback_reason is not None
        effective_mode = self._bpf.mode
        pin_path = getattr(self._bpf, "pin_path", None)
        schema_version = self._bpf.schema_version

        record_count = self._bpf.task_count
        return [
            RuntimeEvent(
                event_type="scheduler.bpf_map_updated",
                category=EventCategory.SCHEDULER,
                severity=EventSeverity.INFO,
                data={
                    "mode": effective_mode,
                    "task_count": record_count,
                    "updated_task_ids": updated_task_ids,
                    "failed_task_ids": failed_task_ids,
                },
            ),
            RuntimeEvent(
                event_type="kernel.decision_published",
                category=EventCategory.SCHEDULER,
                severity=EventSeverity.INFO,
                data={
                    "bpf_mode": effective_mode,
                    "requested_bpf_mode": self._requested_bpf_mode,
                    "effective_bpf_mode": effective_mode,
                    "record_count": record_count,
                    "attempted_count": len(attempted_task_ids),
                    "updated_count": len(updated_task_ids),
                    "failed_count": len(failed_task_ids),
                    "fallback_active": fallback_active,
                    "fallback_reason": self._bpf_fallback_reason,
                    "updated_task_ids": updated_task_ids,
                    "failed_task_ids": failed_task_ids,
                    "real_updated_count": len(real_updated_task_ids),
                    "real_updated_task_ids": sorted(real_updated_task_ids),
                    "real_failed_task_ids": sorted(real_failed_task_ids),
                    "schema_version": schema_version,
                    "pin_path": str(pin_path) if pin_path is not None else None,
                },
            ),
        ]

    def sample_pressure(self, *, cgroup_path: Path | None = None) -> PressureSample:
        usage = self._cgroup.read_usage(cgroup_path) if cgroup_path is not None else None

        system = {
            resource: read_psi_avg10_with_status(self._psi_paths[resource])
            for resource in ("cpu", "memory", "io")
        }
        local = None
        if cgroup_path is not None:
            local_reader = getattr(self._cgroup, "read_pressure", None)
            local = (
                local_reader(cgroup_path)
                if callable(local_reader)
                else {
                    resource: (0.0, False)
                    for resource in ("cpu", "memory", "io")
                }
            )
        values: dict[str, float] = {}
        sources: dict[str, str] = {}
        reason_parts: list[str] = []
        for resource in ("cpu", "memory", "io"):
            system_value, system_available = system[resource]
            if local is None:
                values[resource] = system_value
                sources[resource] = (
                    "system" if system_available else "unavailable"
                )
                if not system_available:
                    reason_parts.append(
                        f"{resource} system PSI missing or unreadable"
                    )
                continue

            local_value, local_available = local[resource]
            if local_available:
                values[resource] = local_value
                sources[resource] = "cgroup"
            elif system_available:
                values[resource] = system_value
                sources[resource] = "system"
                reason_parts.append(
                    f"{resource} cgroup PSI unavailable; using system PSI"
                )
            else:
                values[resource] = 0.0
                sources[resource] = "unavailable"
                reason_parts.append(
                    f"{resource} cgroup and system PSI unavailable"
                )

        if local is not None and all(
            source == "cgroup" for source in sources.values()
        ):
            aggregate_source = "cgroup"
        elif local is not None and any(
            source == "cgroup" for source in sources.values()
        ):
            aggregate_source = "mixed"
        else:
            aggregate_source = "system"

        level = pressure_level_from_psi(
            values["cpu"], values["memory"], values["io"]
        )
        reason = "; ".join(reason_parts) if reason_parts else None

        return PressureSample(
            cpu_some_avg10=values["cpu"],
            memory_some_avg10=values["memory"],
            io_some_avg10=values["io"],
            cgroup_usage=usage,
            pressure_level=level,
            pressure_source=aggregate_source,
            pressure_sources=sources,
            reason=reason,
        )

    def cleanup_agent(
        self,
        *,
        agent_id: str,
        cgroup_path: Path,
    ) -> list[RuntimeEvent]:
        result = self._cgroup.cleanup(cgroup_path)
        if result.available:
            self._agent_cgroups.pop(agent_id, None)
        return [result.event]

    def snapshot(self) -> dict[str, Any]:
        pin_path = getattr(self._bpf, "pin_path", None)
        fallback_active = self._bpf_fallback_reason is not None
        return {
            "mode": f"user_space_cgroup_{self._bpf.mode}_bpf",
            "probe": self._probe.to_dict(),
            "bpf_map": {
                "requested_mode": self._requested_bpf_mode,
                "mode": self._bpf.mode,
                "schema_version": self._bpf.schema_version,
                "pin_path": (
                    str(pin_path) if pin_path is not None else None
                ),
                "fallback_active": fallback_active,
                "fallback_reason": self._bpf_fallback_reason,
                "task_count": self._bpf.task_count,
                "tasks": self._bpf.snapshot(),
            },
            "agent_cgroups": {
                agent_id: str(path)
                for agent_id, path in self._agent_cgroups.items()
            },
            "psi_paths": {
                name: str(path) for name, path in self._psi_paths.items()
            },
            "updated_at": time.time(),
        }
