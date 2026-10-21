import asyncio
import os
import sys
import time
from contextlib import suppress
from pathlib import Path

import pytest

from agentos.worker.process import SubprocessWorkerManager, WorkerSpec


async def test_subprocess_worker_returns_json_result():
    manager = SubprocessWorkerManager()
    result = await manager.run(
        WorkerSpec(
            worker_type="echo",
            task_id="task-1",
            agent_id="agent-1",
            payload={"result": "hello"},
        )
    )

    assert result.ok is True
    assert result.result == "hello"
    assert result.pid is not None
    assert result.exit_code == 0


async def test_subprocess_worker_reports_non_zero_exit():
    manager = SubprocessWorkerManager()
    result = await manager.run(
        WorkerSpec(
            worker_type="fail",
            task_id="task-1",
            agent_id="agent-1",
            payload={"error": "boom"},
        )
    )

    assert result.ok is False
    assert result.exit_code != 0
    assert result.error == "boom"


async def test_subprocess_worker_memory_pressure_fault_is_structured_failure():
    manager = SubprocessWorkerManager()
    result = await manager.run(
        WorkerSpec(
            worker_type="echo",
            task_id="task-1",
            agent_id="agent-1",
            payload={"result": "hello"},
            fault={
                "fault_type": "memory_pressure",
                "memory_mb": 2,
                "chunk_mb": 1,
                "hold_seconds": 0,
            },
        )
    )

    assert result.ok is False
    assert result.exit_code != 0
    assert "injected memory pressure fault" in (result.error or "")


async def test_subprocess_worker_timeout_kills_process():
    manager = SubprocessWorkerManager()
    result = await manager.run(
        WorkerSpec(
            worker_type="sleep",
            task_id="task-1",
            agent_id="agent-1",
            payload={"seconds": 1.0},
        ),
        timeout_seconds=0.01,
    )

    assert result.ok is False
    assert result.timed_out is True
    assert "timed out" in (result.error or "")


@pytest.mark.asyncio
async def test_subprocess_worker_fork_child_exits_normally() -> None:
    manager = SubprocessWorkerManager()

    result = await manager.run(
        WorkerSpec(worker_type="fork", task_id="task-fork", agent_id="agent-1"),
        timeout_seconds=2.0,
    )

    assert result.ok is True
    assert result.result["child_pid"] > 0
    assert result.result["child_exit_code"] == 0


@pytest.mark.asyncio
async def test_subprocess_worker_applies_child_affinity() -> None:
    available = os.sched_getaffinity(0)
    cpu = min(available)
    manager = SubprocessWorkerManager(allowed_cpus={cpu})
    observed: set[int] = set()

    result = await manager.run(
        WorkerSpec(
            worker_type="sleep",
            task_id="task-affinity",
            agent_id="agent-1",
            payload={"seconds": 0.05},
        ),
        timeout_seconds=2.0,
        on_start=lambda pid: observed.update(os.sched_getaffinity(pid)),
    )

    assert result.ok is True
    assert observed == {cpu}


@pytest.mark.asyncio
async def test_subprocess_worker_startup_barrier_binds_while_child_is_sleeping() -> None:
    manager = SubprocessWorkerManager(startup_barrier=True)
    observed_states: list[str] = []

    def observe_state(pid: int) -> None:
        raw = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
        observed_states.append(raw[raw.rfind(")") + 2])

    result = await manager.run(
        WorkerSpec(
            worker_type="echo",
            task_id="task-barrier",
            agent_id="agent-1",
            payload={"result": "released-after-bind"},
        ),
        timeout_seconds=2.0,
        on_start=observe_state,
    )

    assert result.ok is True
    assert result.result == "released-after-bind"
    assert observed_states == ["S"]
    assert "READY" not in result.stderr


async def test_subprocess_worker_can_be_killed_after_start():
    manager = SubprocessWorkerManager()
    result = await manager.run(
        WorkerSpec(
            worker_type="sleep",
            task_id="task-1",
            agent_id="agent-1",
            payload={"seconds": 1.0},
        ),
        kill_after_start=True,
    )

    assert result.ok is False
    assert result.killed is True
    assert result.pid is not None
    assert result.exit_code != 0
    assert "killed" in (result.error or "")


async def test_subprocess_worker_invalid_json_is_structured_failure():
    manager = SubprocessWorkerManager(python_executable=sys.executable)
    result = await manager.run(
        WorkerSpec(worker_type="bad-json", task_id="task-1", agent_id="agent-1")
    )

    assert result.ok is False
    assert result.exit_code == 0
    assert "invalid JSON" in (result.error or "")


async def test_subprocess_worker_samples_while_running():
    manager = SubprocessWorkerManager()
    samples: list[float] = []

    result = await manager.run(
        WorkerSpec(
            worker_type="sleep",
            task_id="task-1",
            agent_id="agent-1",
            payload={"seconds": 0.12},
        ),
        on_sample=lambda: samples.append(time.monotonic()),
        sample_interval_seconds=0.02,
    )

    assert result.ok is True
    assert samples
    assert samples == sorted(samples)


async def test_subprocess_worker_periodic_sampling_does_not_extend_timeout():
    manager = SubprocessWorkerManager()
    started = time.monotonic()

    result = await manager.run(
        WorkerSpec(
            worker_type="sleep",
            task_id="task-1",
            agent_id="agent-1",
            payload={"seconds": 1.0},
        ),
        timeout_seconds=0.05,
        on_sample=lambda: None,
        sample_interval_seconds=0.01,
    )

    assert result.timed_out is True
    assert time.monotonic() - started < 0.25


async def test_subprocess_worker_sampling_error_kills_and_reaps_process():
    manager = SubprocessWorkerManager()
    worker_pids: list[int] = []

    def fail_sample() -> None:
        raise RuntimeError("sched_ext strict health failure")

    with pytest.raises(RuntimeError, match="sched_ext strict health failure"):
        await manager.run(
            WorkerSpec(
                worker_type="sleep",
                task_id="task-sampling-error",
                agent_id="agent-1",
                payload={"seconds": 5.0},
            ),
            timeout_seconds=10.0,
            on_start=worker_pids.append,
            on_sample=fail_sample,
            sample_interval_seconds=0.01,
        )

    assert worker_pids
    worker_path = Path("/proc") / str(worker_pids[0])
    leaked = worker_path.exists()
    if leaked:
        with suppress(ProcessLookupError):
            os.kill(worker_pids[0], 9)
        await asyncio.sleep(0.05)
    assert leaked is False


async def test_subprocess_worker_rejects_nonpositive_periodic_interval():
    manager = SubprocessWorkerManager()

    with pytest.raises(ValueError, match="sample_interval_seconds"):
        await manager.run(
            WorkerSpec(
                worker_type="echo",
                task_id="task-1",
                agent_id="agent-1",
            ),
            on_sample=lambda: None,
            sample_interval_seconds=0,
        )
