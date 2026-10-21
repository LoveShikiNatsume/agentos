import asyncio
import os
import signal

import pytest
from pathlib import Path

from agentos.core.resources import ResourceQuota
from agentos.core.task import Task, TaskDAG
from agentos.fault.injector import FaultInjectionPlan
from agentos.fault.recovery import RetryPolicy, SkipPolicy
from agentos.fault.timeout import AgentTimeoutError
from agentos.isolation.bpf import MockBPFMapAdapter
from agentos.isolation.cgroup import CgroupV2Manager
from agentos.events.types import EventCategory, RuntimeEvent
from agentos.isolation.kernel_adapter import (
    KernelWorkerBindResult,
    UserSpaceKernelAdapter,
)
from agentos.isolation.sched_ext import KernelProbeResult
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import Runtime


class _PidFailingCgroupManager(CgroupV2Manager):
    def add_pid(self, path: Path, pid: int):
        return self._unavailable(path, "pid attach failed")


class _RecordingKernelAdapter(UserSpaceKernelAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cleanup_calls: list[str] = []

    def cleanup_agent(self, *, agent_id: str, cgroup_path: Path):
        self.cleanup_calls.append(agent_id)
        return super().cleanup_agent(agent_id=agent_id, cgroup_path=cgroup_path)


class _RecordingSchedExtAdapter(_RecordingKernelAdapter):
    supports_sched_ext_binding = True
    strict_sched_ext = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bound_pids: set[int] = set()
        self.removed_pids: list[int] = []

    def bind_worker_policy(self, *, agent_id: str, task_id: str, pid: int):
        self.bound_pids.add(pid)
        return KernelWorkerBindResult(
            available=True,
            attached=True,
            mode="sched_ext",
            pid=pid,
            events=[
                RuntimeEvent(
                    event_type="kernel.sched_ext_policy_published",
                    agent_id=agent_id,
                    task_id=task_id,
                    category=EventCategory.SCHEDULER,
                    data={"pid": pid, "readback_verified": True},
                ),
                RuntimeEvent(
                    event_type="kernel.sched_ext_worker_attached",
                    agent_id=agent_id,
                    task_id=task_id,
                    category=EventCategory.SCHEDULER,
                    data={"pid": pid, "policy": 7, "ext_enabled": 1},
                ),
            ],
        )

    def remove_worker_policy(self, *, agent_id: str, task_id: str, pid: int):
        self.bound_pids.discard(pid)
        self.removed_pids.append(pid)
        return [
            RuntimeEvent(
                event_type="kernel.sched_ext_policy_removed",
                agent_id=agent_id,
                task_id=task_id,
                category=EventCategory.SCHEDULER,
                data={"pid": pid},
            )
        ]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _kill_pid(pid: int) -> None:
    if not _pid_alive(pid):
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


async def test_runtime_runs_subprocess_worker_and_records_pid(tmp_path):
    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="t1",
            name="subprocess",
            agent_role="echo",
            metadata={
                "execution": "subprocess",
                "worker_type": "echo",
                "worker_payload": {"result": "done"},
            },
        )
    )
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={},
        cgroup_manager=CgroupV2Manager(root=tmp_path / "cgroup"),
    )

    results = await rt.run(dag)

    acb = rt.agents["agent-t1"]
    assert results["t1"] == "done"
    assert acb.pid is not None
    assert acb.cgroup is not None
    assert any(event.event_type == "worker.started" for event in rt.events)
    assert any(event.event_type == "worker.finished" for event in rt.events)
    assert any(event.event_type == "resource.cgroup_pid_added" for event in rt.events)
    assert any(event.event_type == "resource.usage_sampled" for event in rt.events)
    assert any(event.event_type == "resource.cgroup_removed" for event in rt.events)
    assert not Path(acb.cgroup).exists()
    for event in rt.events:
        event.to_dict()


async def test_runtime_subprocess_timeout_is_observable(tmp_path):
    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="t1",
            name="slow",
            agent_role="sleep",
            resource_requirements=ResourceQuota(timeout_seconds=0.01),
            metadata={
                "execution": "subprocess",
                "worker_type": "sleep",
                "worker_payload": {"seconds": 1.0},
            },
        )
    )
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={},
        cgroup_manager=CgroupV2Manager(root=tmp_path / "cgroup"),
    )

    with pytest.raises(AgentTimeoutError):
        await rt.run(dag)

    assert any(event.event_type == "worker.timeout" for event in rt.events)
    assert any(event.event_type == "fault.timeout" for event in rt.events)
    assert any(event.event_type == "resource.usage_sampled" for event in rt.events)
    assert any(event.event_type == "resource.cgroup_removed" for event in rt.events)
    final_index = next(
        index
        for index, event in enumerate(rt.events)
        if event.event_type == "resource.usage_sampled"
        and event.data.get("phase") == "final"
    )
    cleanup_index = next(
        index
        for index, event in enumerate(rt.events)
        if event.event_type == "resource.cgroup_removed"
    )
    assert final_index < cleanup_index
    assert dag.tasks["t1"].status.value == "failed"


async def test_worker_timeout_still_cleans_cgroup(tmp_path):
    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="timeout",
            name="slow",
            agent_role="sleep",
            resource_requirements=ResourceQuota(timeout_seconds=0.01),
            metadata={
                "execution": "subprocess",
                "worker_type": "sleep",
                "worker_payload": {"seconds": 1.0},
            },
        )
    )
    adapter = _RecordingKernelAdapter(
        cgroup_manager=CgroupV2Manager(root=tmp_path / "cgroup"),
        bpf_map_adapter=MockBPFMapAdapter(),
        probe_result=KernelProbeResult(
            sched_ext="unavailable",
            bpf_map="mock",
            fallback_mode="mock_bpf_map",
            reason="unit test",
        ),
    )
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={},
        kernel_adapter=adapter,
    )

    with pytest.raises(AgentTimeoutError):
        await rt.run(dag)

    assert adapter.cleanup_calls == ["agent-timeout"]


async def test_sched_ext_worker_binding_precedes_execution_and_is_cleaned(tmp_path):
    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="bound",
            name="bound",
            agent_role="echo",
            metadata={
                "execution": "subprocess",
                "worker_type": "echo",
                "worker_payload": {"result": "done"},
            },
        )
    )
    adapter = _RecordingSchedExtAdapter(
        cgroup_manager=CgroupV2Manager(root=tmp_path / "cgroup"),
        bpf_map_adapter=MockBPFMapAdapter(),
    )
    runtime = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={},
        kernel_adapter=adapter,
    )

    assert (await runtime.run(dag))["bound"] == "done"

    event_types = [event.event_type for event in runtime.events]
    assert event_types.index("resource.cgroup_pid_added") < event_types.index(
        "kernel.sched_ext_policy_published"
    )
    assert event_types.index("kernel.sched_ext_policy_published") < event_types.index(
        "kernel.sched_ext_worker_attached"
    )
    assert event_types.index("kernel.sched_ext_worker_attached") < event_types.index(
        "worker.finished"
    )
    assert event_types.index("worker.finished") < event_types.index(
        "kernel.sched_ext_policy_removed"
    )
    assert event_types.index("kernel.sched_ext_policy_removed") < event_types.index(
        "resource.cgroup_removed"
    )
    assert adapter.bound_pids == set()
    assert len(adapter.removed_pids) == 1


async def test_sched_ext_worker_timeout_removes_policy(tmp_path):
    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="bound-timeout",
            name="bound-timeout",
            agent_role="sleep",
            resource_requirements=ResourceQuota(timeout_seconds=0.01),
            metadata={
                "execution": "subprocess",
                "worker_type": "sleep",
                "worker_payload": {"seconds": 1.0},
            },
        )
    )
    adapter = _RecordingSchedExtAdapter(
        cgroup_manager=CgroupV2Manager(root=tmp_path / "cgroup"),
        bpf_map_adapter=MockBPFMapAdapter(),
    )
    runtime = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={},
        kernel_adapter=adapter,
    )

    with pytest.raises(AgentTimeoutError):
        await runtime.run(dag)

    assert adapter.bound_pids == set()
    assert len(adapter.removed_pids) == 1
    event_types = [event.event_type for event in runtime.events]
    assert event_types.index("worker.timeout") < event_types.index(
        "kernel.sched_ext_policy_removed"
    )


async def test_runtime_cancellation_cleans_started_worker_cgroup(tmp_path):
    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="cancel",
            name="cancel",
            agent_role="sleep",
            metadata={
                "execution": "subprocess",
                "worker_type": "sleep",
                "worker_payload": {"seconds": 10.0},
            },
        )
    )
    adapter = _RecordingKernelAdapter(
        cgroup_manager=CgroupV2Manager(root=tmp_path / "cgroup"),
        bpf_map_adapter=MockBPFMapAdapter(),
        probe_result=KernelProbeResult(
            sched_ext="unavailable",
            bpf_map="mock",
            fallback_mode="mock_bpf_map",
            reason="unit test",
        ),
    )
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={},
        kernel_adapter=adapter,
    )
    run_task = asyncio.create_task(rt.run(dag))
    worker_pid: int | None = None

    try:
        for _ in range(200):
            started = [
                event
                for event in rt.events
                if event.event_type == "worker.started"
            ]
            if started:
                worker_pid = started[0].data["pid"]
                break
            await asyncio.sleep(0.01)
        assert worker_pid is not None

        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task

        assert adapter.cleanup_calls == ["agent-cancel"]
        assert not _pid_alive(worker_pid)
        assert any(
            event.event_type == "resource.cgroup_removed"
            for event in rt.events
        )
        assert not any(
            event.event_type == "resource.cgroup_cleanup_failed"
            for event in rt.events
        )
    finally:
        if worker_pid is not None:
            _kill_pid(worker_pid)
        if not run_task.done():
            run_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await run_task


async def test_sched_ext_cancellation_removes_policy_before_cgroup(tmp_path):
    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="cancel-ext",
            name="cancel-ext",
            agent_role="sleep",
            metadata={
                "execution": "subprocess",
                "worker_type": "sleep",
                "worker_payload": {"seconds": 10.0},
            },
        )
    )
    adapter = _RecordingSchedExtAdapter(
        cgroup_manager=CgroupV2Manager(root=tmp_path / "cgroup"),
        bpf_map_adapter=MockBPFMapAdapter(),
    )
    runtime = Runtime(
        llm_backend=MockLLMBackend(), handlers={}, kernel_adapter=adapter
    )
    run_task = asyncio.create_task(runtime.run(dag))

    for _ in range(200):
        if any(event.event_type == "worker.started" for event in runtime.events):
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("worker did not start")

    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task

    event_types = [event.event_type for event in runtime.events]
    assert adapter.bound_pids == set()
    assert event_types.index("kernel.sched_ext_policy_removed") < event_types.index(
        "resource.cgroup_removed"
    )


async def test_runtime_subprocess_fault_injection_can_recover(tmp_path):
    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="t1",
            name="subprocess",
            agent_role="echo",
            metadata={
                "execution": "subprocess",
                "worker_type": "echo",
                "worker_payload": {"result": "done"},
            },
        )
    )
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={},
        recovery_policy=SkipPolicy(),
        fault_injection=FaultInjectionPlan({"t1": "kill"}),
        cgroup_manager=CgroupV2Manager(root=tmp_path / "cgroup"),
    )

    results = await rt.run(dag)

    assert results["t1"] is None
    event_types = [event.event_type for event in rt.events]
    assert "fault.injected" in event_types
    assert "fault.triggered" in event_types
    assert "worker.started" in event_types
    killed_events = [
        event
        for event in rt.events
        if event.event_type == "worker.failed" and event.data.get("killed") is True
    ]
    assert killed_events
    assert killed_events[0].data["pid"] is not None
    assert any(event.event_type == "resource.usage_sampled" for event in rt.events)
    assert any(event.event_type == "resource.cgroup_removed" for event in rt.events)
    assert "fault.recovery_completed" in event_types


async def test_runtime_subprocess_memory_pressure_retry_recovers(tmp_path):
    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="t1",
            name="subprocess",
            agent_role="echo",
            metadata={
                "execution": "subprocess",
                "worker_type": "echo",
                "worker_payload": {"result": "done"},
            },
        )
    )
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={},
        recovery_policy=RetryPolicy(max_retries=1),
        fault_injection=FaultInjectionPlan(
            {
                "t1": {
                    "fault_type": "memory_pressure",
                    "memory_mb": 2,
                    "chunk_mb": 1,
                    "hold_seconds": 0,
                }
            }
        ),
        cgroup_manager=CgroupV2Manager(root=tmp_path / "cgroup"),
    )

    results = await rt.run(dag)

    assert results["t1"] == "done"
    event_types = [event.event_type for event in rt.events]
    assert "worker.started" in event_types
    assert "worker.failed" in event_types
    assert "worker.finished" in event_types
    assert "fault.memory_pressure" in event_types
    assert "resource.usage_sampled" in event_types
    assert "resource.cgroup_removed" in event_types
    assert "fault.recovery_completed" in event_types
    assert any(
        "injected memory pressure fault" in (event.data.get("error") or "")
        for event in rt.events
        if event.event_type == "worker.failed"
    )


async def test_runtime_subprocess_retry_recovers_flaky_worker(tmp_path):
    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="t1",
            name="flaky",
            agent_role="flaky",
            metadata={
                "execution": "subprocess",
                "worker_type": "flaky",
                "worker_payload": {"result": "recovered"},
            },
        )
    )
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={},
        recovery_policy=RetryPolicy(max_retries=1),
        cgroup_manager=CgroupV2Manager(root=tmp_path / "cgroup"),
    )

    results = await rt.run(dag)

    assert results["t1"] == "recovered"
    assert any(
        event.event_type == "fault.recovery_completed"
        and event.data["action"] == "retry"
        for event in rt.events
    )


@pytest.mark.asyncio
async def test_runtime_uses_kernel_adapter_for_subprocess_cgroup_quota(tmp_path: Path):
    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="worker-task",
            name="worker-task",
            agent_role="echo",
            metadata={
                "worker_type": "echo",
                "worker_payload": {"value": "ok"},
            },
        )
    )

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

    runtime = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={},
        execution_mode="subprocess",
        kernel_adapter=adapter,
    )

    await runtime.run(dag)

    event_types = [event.event_type for event in runtime.events]

    assert "resource.cgroup_created" in event_types
    assert "resource.cgroup_pid_added" in event_types
    assert "resource.cgroup_quota_applied" in event_types
    assert "resource.usage_sampled" in event_types

    quota_events = [
        event for event in runtime.events
        if event.event_type == "resource.cgroup_quota_applied"
    ]
    assert len(quota_events) == 1
    assert quota_events[0].data["pid_attached"] is True
    assert "limits" in quota_events[0].data


@pytest.mark.asyncio
async def test_runtime_does_not_emit_quota_applied_when_kernel_adapter_pid_attach_fails(tmp_path: Path):
    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="worker-task",
            name="worker-task",
            agent_role="echo",
            metadata={
                "worker_type": "echo",
                "worker_payload": {"value": "ok"},
            },
        )
    )

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

    runtime = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={},
        execution_mode="subprocess",
        kernel_adapter=adapter,
    )

    await runtime.run(dag)

    event_types = [event.event_type for event in runtime.events]

    assert "resource.cgroup_created" in event_types
    assert "resource.cgroup_unavailable" in event_types
    assert "resource.cgroup_quota_applied" not in event_types


@pytest.mark.asyncio
async def test_runtime_uses_injected_kernel_adapter_probe_result(tmp_path: Path):
    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="worker-task",
            name="worker-task",
            agent_role="echo",
            metadata={
                "worker_type": "echo",
                "worker_payload": {"value": "ok"},
            },
        )
    )

    adapter = UserSpaceKernelAdapter(
        cgroup_manager=CgroupV2Manager(root=tmp_path / "cg", enable_controllers=False),
        bpf_map_adapter=MockBPFMapAdapter(),
        probe_result=KernelProbeResult(
            sched_ext="unavailable",
            bpf_map="mock",
            fallback_mode="custom_test_fallback",
            reason="adapter sentinel",
        ),
    )

    runtime = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={},
        execution_mode="subprocess",
        kernel_adapter=adapter,
    )

    await runtime.run(dag)

    probe_events = [
        event for event in runtime.events
        if event.event_type == "scheduler.kernel_probe"
    ]
    assert len(probe_events) == 1
    assert probe_events[0].data["reason"] == "adapter sentinel"
    assert probe_events[0].data["fallback_mode"] == "custom_test_fallback"


@pytest.mark.asyncio
async def test_runtime_quota_writes_cgroup_control_files(tmp_path: Path):
    cgroup_root = tmp_path / "cg"
    manager = CgroupV2Manager(root=cgroup_root, enable_controllers=False)

    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="worker-task",
            name="worker-task",
            agent_role="echo",
            resource_requirements=ResourceQuota(
                cpu_max_percent=50.0,
                cpu_shares=200,
                memory_max_mb=64,
                pids_max=8,
            ),
            metadata={
                "worker_type": "echo",
                "worker_payload": {"value": "ok"},
            },
        )
    )

    adapter = UserSpaceKernelAdapter(
        cgroup_manager=manager,
        bpf_map_adapter=MockBPFMapAdapter(),
        probe_result=KernelProbeResult(
            sched_ext="unavailable",
            bpf_map="mock",
            fallback_mode="mock_bpf_map",
            reason="unit test",
        ),
    )

    runtime = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={},
        execution_mode="subprocess",
        kernel_adapter=adapter,
        cleanup_cgroup_on_exit=False,
    )

    await runtime.run(dag)

    acb = runtime.agents["agent-worker-task"]
    cgroup_path = Path(acb.cgroup)

    assert (cgroup_path / "cpu.max").read_text().strip() == "50000 100000"
    assert (cgroup_path / "memory.max").read_text().strip() == str(64 * 1024 * 1024)
    assert (cgroup_path / "pids.max").read_text().strip() == "8"
    assert (cgroup_path / "cpu.weight").read_text().strip() == "200"

    limits = manager.read_limits(cgroup_path)
    assert limits["cpu_max_percent"] == 50.0
    assert limits["cpu_weight"] == 200
    assert limits["memory_max_mb"] == 64
    assert limits["pids_max"] == 8

    event_types = [event.event_type for event in runtime.events]
    assert event_types.count("resource.cgroup_created") == 1
    assert event_types.count("resource.cgroup_pid_added") == 1
    assert event_types.count("resource.cgroup_quota_applied") == 1
    usage_events = [
        event
        for event in runtime.events
        if event.event_type == "resource.usage_sampled"
    ]
    phases = [event.data["phase"] for event in usage_events]
    assert phases[0] == "initial"
    assert phases[-1] == "final"
    assert phases.count("initial") == 1
    assert phases.count("final") == 1
    assert set(phases) <= {"initial", "running", "final"}

    quota_events = [
        event for event in runtime.events
        if event.event_type == "resource.cgroup_quota_applied"
    ]
    assert len(quota_events) == 1
    assert quota_events[0].data["pid_attached"] is True
    assert quota_events[0].data["limits"]["cpu_max_percent"] == 50.0
    assert quota_events[0].data["limits"]["memory_max_mb"] == 64
    assert quota_events[0].data["limits"]["pids_max"] == 8

    manager.cleanup(cgroup_path)
    assert not cgroup_path.exists()


@pytest.mark.asyncio
async def test_runtime_emits_ordered_worker_sampling_lifecycle(tmp_path: Path):
    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="worker-task",
            name="worker-task",
            agent_role="sleep",
            metadata={
                "execution": "subprocess",
                "worker_type": "sleep",
                "worker_payload": {"seconds": 0.12, "result": "done"},
            },
        )
    )
    adapter = UserSpaceKernelAdapter(
        cgroup_manager=CgroupV2Manager(
            root=tmp_path / "cg", enable_controllers=False
        ),
        bpf_map_adapter=MockBPFMapAdapter(),
        probe_result=KernelProbeResult(
            sched_ext="unavailable",
            bpf_map="mock",
            fallback_mode="mock_bpf_map",
            reason="unit test",
        ),
    )
    runtime = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={},
        kernel_adapter=adapter,
        worker_sample_interval_seconds=0.02,
    )

    results = await runtime.run(dag)

    assert results["worker-task"] == "done"
    usage_events = [
        event
        for event in runtime.events
        if event.event_type == "resource.usage_sampled"
    ]
    pressure_events = [
        event
        for event in runtime.events
        if event.event_type == "resource.pressure_sampled"
        and event.data.get("scope") == "agent_cgroup"
    ]
    phases = [event.data["phase"] for event in usage_events]
    assert phases[0] == "initial"
    assert "running" in phases
    assert phases[-1] == "final"
    assert [event.data["sample_index"] for event in usage_events] == list(
        range(len(usage_events))
    )
    assert [event.data["elapsed_ms"] for event in usage_events] == sorted(
        event.data["elapsed_ms"] for event in usage_events
    )
    assert [
        (event.data["phase"], event.data["sample_index"])
        for event in pressure_events
    ] == [
        (event.data["phase"], event.data["sample_index"])
        for event in usage_events
    ]
    assert all(
        "pressure_source" in event.data["pressure"]
        for event in pressure_events
    )


def test_runtime_rejects_nonpositive_worker_sample_interval():
    with pytest.raises(ValueError, match="worker_sample_interval_seconds"):
        Runtime(
            llm_backend=MockLLMBackend(),
            handlers={},
            worker_sample_interval_seconds=0,
        )
