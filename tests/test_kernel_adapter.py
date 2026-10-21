from pathlib import Path

import pytest

from agentos.core.resources import ResourceQuota
from agentos.isolation.bpf import MockBPFMapAdapter
from agentos.isolation.bpftool import BPFMapError
from agentos.isolation.cgroup import CgroupV2Manager
from agentos.isolation.kernel_adapter import UserSpaceKernelAdapter
from agentos.isolation.sched_ext import KernelProbeResult


class _PidFailingCgroupManager(CgroupV2Manager):
    def add_pid(self, path: Path, pid: int):
        return self._unavailable(path, "pid attach failed")


def test_user_space_kernel_adapter_probe_uses_supplied_probe(tmp_path: Path):
    probe = KernelProbeResult(
        sched_ext="unavailable",
        bpf_map="mock",
        fallback_mode="mock_bpf_map",
        reason="unit test",
    )
    adapter = UserSpaceKernelAdapter(
        cgroup_manager=CgroupV2Manager(root=tmp_path / "cg", enable_controllers=False),
        bpf_map_adapter=MockBPFMapAdapter(),
        probe_result=probe,
    )

    assert adapter.probe().reason == "unit test"
    assert adapter.snapshot()["mode"] == "user_space_cgroup_mock_bpf"


def test_user_space_kernel_adapter_applies_quota_to_fake_cgroup(tmp_path: Path):
    adapter = UserSpaceKernelAdapter(
        cgroup_manager=CgroupV2Manager(root=tmp_path / "cg", enable_controllers=False),
        bpf_map_adapter=MockBPFMapAdapter(),
        probe_result=KernelProbeResult(
            sched_ext="unavailable",
            bpf_map="mock",
            fallback_mode="mock_bpf_map",
            reason="unit test",
        ),
    )

    result = adapter.apply_agent_quota(
        agent_id="agent-1",
        task_id="task-1",
        pid=1234,
        quota=ResourceQuota(cpu_max_percent=50, memory_max_mb=64, pids_max=8),
    )

    assert result.available is True
    assert result.mode == "cgroup_v2"
    assert result.events[0].event_type == "resource.cgroup_created"
    assert result.events[-1].event_type == "resource.cgroup_quota_applied"
    assert (tmp_path / "cg" / "agent-1" / "memory.max").read_text().strip() == str(64 * 1024 * 1024)
    assert (tmp_path / "cg" / "agent-1" / "pids.max").read_text().strip() == "8"


def test_user_space_kernel_adapter_publish_decision_updates_mock_bpf_map(tmp_path: Path):
    adapter = UserSpaceKernelAdapter(
        cgroup_manager=CgroupV2Manager(root=tmp_path / "cg", enable_controllers=False),
        bpf_map_adapter=MockBPFMapAdapter(),
        probe_result=KernelProbeResult(
            sched_ext="unavailable",
            bpf_map="mock",
            fallback_mode="mock_bpf_map",
            reason="unit test",
        ),
    )

    events = adapter.publish_decision({
        "policy": "latency_first",
        "selected": ["task-1"],
        "fairness": {"fairness_score": 0.9},
        "tasks": {
            "task-1": {
                "task_id": "task-1",
                "workload_class": "interactive",
                "starvation_risk": "low",
                "reason": "selected",
                "wait_ms": 0.0,
                "deferred_count": 0,
                "blocked_count": 0,
            }
        },
    })

    snapshot = adapter.snapshot()
    events_by_type = {event.event_type: event for event in events}
    assert "scheduler.bpf_map_updated" in events_by_type
    published = events_by_type["kernel.decision_published"]
    assert published.data["bpf_mode"] == "mock"
    assert published.data["requested_bpf_mode"] == "mock"
    assert published.data["effective_bpf_mode"] == "mock"
    assert published.data["record_count"] == 1
    assert published.data["attempted_count"] == 1
    assert published.data["updated_count"] == 1
    assert published.data["failed_count"] == 0
    assert published.data["fallback_active"] is False
    assert published.data["fallback_reason"] is None
    assert published.data["updated_task_ids"] == ["task-1"]
    assert snapshot["bpf_map"]["task_count"] == 1
    assert snapshot["bpf_map"]["tasks"]["task-1"]["policy"] == "latency_first"


def test_user_space_kernel_adapter_sample_pressure_missing_files_defaults_to_normal(tmp_path: Path):
    adapter = UserSpaceKernelAdapter(
        cgroup_manager=CgroupV2Manager(root=tmp_path / "cg", enable_controllers=False),
        bpf_map_adapter=MockBPFMapAdapter(),
        psi_paths={name: tmp_path / name for name in ("cpu", "memory", "io")},
        probe_result=KernelProbeResult(
            sched_ext="unavailable",
            bpf_map="mock",
            fallback_mode="mock_bpf_map",
            reason="unit test",
        ),
    )

    sample = adapter.sample_pressure()

    assert sample.pressure_level == "normal"
    assert sample.to_dict()["pressure_level"] == "normal"


def test_user_space_kernel_adapter_reports_unavailable_when_pid_attach_fails(tmp_path: Path):
    adapter = UserSpaceKernelAdapter(
        cgroup_manager=_PidFailingCgroupManager(root=tmp_path / "cg", enable_controllers=False),
        bpf_map_adapter=MockBPFMapAdapter(),
        probe_result=KernelProbeResult(
            sched_ext="unavailable",
            bpf_map="mock",
            fallback_mode="mock_bpf_map",
            reason="unit test",
        ),
    )

    result = adapter.apply_agent_quota(
        agent_id="agent-1",
        task_id="task-1",
        pid=1234,
        quota=ResourceQuota(cpu_max_percent=50, memory_max_mb=64, pids_max=8),
    )

    assert result.available is False
    assert result.reason == "pid attach failed"
    event_types = [event.event_type for event in result.events]
    assert event_types == [
        "resource.cgroup_created",
        "resource.cgroup_unavailable",
        "resource.cgroup_removed",
    ]


def test_apply_agent_quota_cleans_up_cgroup_when_pid_attach_fails(tmp_path: Path):
    cgroup_root = tmp_path / "cg"
    adapter = UserSpaceKernelAdapter(
        cgroup_manager=_PidFailingCgroupManager(root=cgroup_root, enable_controllers=False),
        bpf_map_adapter=MockBPFMapAdapter(),
        probe_result=KernelProbeResult(
            sched_ext="unavailable",
            bpf_map="mock",
            fallback_mode="mock_bpf_map",
            reason="unit test",
        ),
    )

    result = adapter.apply_agent_quota(
        agent_id="agent-1",
        task_id="task-1",
        pid=1234,
        quota=ResourceQuota(cpu_max_percent=50, memory_max_mb=64, pids_max=8),
    )

    assert result.available is False
    assert result.reason == "pid attach failed"

    event_types = [event.event_type for event in result.events]
    assert "resource.cgroup_created" in event_types
    assert "resource.cgroup_unavailable" in event_types
    assert "resource.cgroup_removed" in event_types
    assert "resource.cgroup_quota_applied" not in event_types

    cgroup_path = cgroup_root / "agent-1"
    assert not cgroup_path.exists()

    snapshot = adapter.snapshot()
    assert "agent-1" not in snapshot["agent_cgroups"]


def test_mock_bpf_adapter_exposes_mode_and_schema() -> None:
    adapter = MockBPFMapAdapter()

    assert adapter.mode == "mock"
    assert adapter.schema_version == "bpf.task_record.v2"
    assert adapter.pin_path is None


def _decision(task_id: str = "task-1") -> dict:
    return {
        "policy": "latency_first",
        "selected_task_ids": [task_id],
        "fairness": {"fairness_score": 0.9},
        "tasks": {
            task_id: {
                "task_id": task_id,
                "workload_class": "interactive",
                "starvation_risk": "low",
                "reason": "selected",
                "resource_requirements": {
                    "cpu_shares": 512,
                    "cpu_max_percent": 50.0,
                    "memory_max_mb": 128,
                    "pids_max": 32,
                },
            }
        },
    }


def test_kernel_adapter_uses_mock_by_default() -> None:
    snapshot = UserSpaceKernelAdapter().snapshot()

    assert snapshot["bpf_map"]["requested_mode"] == "mock"
    assert snapshot["bpf_map"]["mode"] == "mock"
    assert snapshot["bpf_map"]["fallback_active"] is False
    assert snapshot["bpf_map"]["fallback_reason"] is None


def test_kernel_adapter_can_publish_through_bpftool_mode(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def runner(argv: list[str]) -> object:
        calls.append(list(argv))
        return None

    adapter = UserSpaceKernelAdapter.from_bpf_mode(
        "bpftool",
        bpf_pin_path=tmp_path / "tasks",
        bpftool_runner=runner,
    )

    events = adapter.publish_decision(_decision())

    published = events[-1].data
    assert published["requested_bpf_mode"] == "bpftool"
    assert published["effective_bpf_mode"] == "bpftool"
    assert published["bpf_mode"] == "bpftool"
    assert published["fallback_active"] is False
    assert any(call[1:4] == ["map", "update", "pinned"] for call in calls)


def test_strict_bpftool_mode_does_not_silently_fallback(
    tmp_path: Path,
) -> None:
    def fail(_argv: list[str]) -> object:
        raise BPFMapError("denied")

    with pytest.raises(BPFMapError, match="denied"):
        UserSpaceKernelAdapter.from_bpf_mode(
            "bpftool",
            bpf_pin_path=tmp_path / "tasks",
            bpftool_runner=fail,
        )


def test_explicit_bpftool_update_failure_does_not_silently_fallback(
    tmp_path: Path,
) -> None:
    def fail_on_update(argv: list[str]) -> object:
        if "update" in argv:
            raise BPFMapError("update denied")
        return None

    adapter = UserSpaceKernelAdapter.from_bpf_mode(
        "bpftool",
        bpf_pin_path=tmp_path / "tasks",
        bpftool_runner=fail_on_update,
    )

    with pytest.raises(BPFMapError, match="update denied"):
        adapter.publish_decision(_decision())


def test_auto_mode_initialization_failure_falls_back_with_reason(
    tmp_path: Path,
) -> None:
    def fail(_argv: list[str]) -> object:
        raise BPFMapError("denied")

    adapter = UserSpaceKernelAdapter.from_bpf_mode(
        "auto",
        bpf_pin_path=tmp_path / "tasks",
        bpftool_runner=fail,
    )

    snapshot = adapter.snapshot()
    assert snapshot["bpf_map"]["requested_mode"] == "auto"
    assert snapshot["bpf_map"]["mode"] == "mock"
    assert snapshot["bpf_map"]["fallback_active"] is True
    assert "denied" in snapshot["bpf_map"]["fallback_reason"]


def test_auto_mode_update_failure_republishes_complete_decision_to_mock(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def runner(argv: list[str]) -> object:
        calls.append(list(argv))
        if len(calls) == 1:
            return None
        raise BPFMapError("update denied")

    adapter = UserSpaceKernelAdapter.from_bpf_mode(
        "auto",
        bpf_pin_path=tmp_path / "tasks",
        bpftool_runner=runner,
    )

    events = adapter.publish_decision(_decision())
    snapshot = adapter.snapshot()
    published = events[-1].data

    assert snapshot["bpf_map"]["requested_mode"] == "auto"
    assert snapshot["bpf_map"]["mode"] == "mock"
    assert snapshot["bpf_map"]["task_count"] == 1
    assert published["effective_bpf_mode"] == "mock"
    assert published["fallback_active"] is True
    assert "update denied" in published["fallback_reason"]
    assert published["attempted_count"] == 1
    assert published["updated_count"] == 1
    assert published["failed_count"] == 0
    assert published["real_updated_count"] == 0
    assert published["real_failed_task_ids"] == ["task-1"]
