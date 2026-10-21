from __future__ import annotations

import math

from agentos.isolation.cgroup_feedback import CgroupFeedbackAccumulator
from agentos.isolation.kernel_adapter import PressureSample


def _sample(
    *,
    cpu: float = 0.0,
    memory: float = 0.0,
    io: float = 0.0,
    cpu_source: str = "cgroup",
    memory_source: str = "cgroup",
    io_source: str = "cgroup",
) -> PressureSample:
    return PressureSample(
        cpu_some_avg10=cpu,
        memory_some_avg10=memory,
        io_some_avg10=io,
        pressure_sources={
            "cpu": cpu_source,
            "memory": memory_source,
            "io": io_source,
        },
    )


def _observe(
    accumulator: CgroupFeedbackAccumulator,
    sample: PressureSample,
    *,
    agent_id: str = "agent-a",
    task_id: str = "task-a",
    phase: str = "running",
    sample_index: int = 1,
    timestamp: float = 100.0,
) -> bool:
    return accumulator.observe(
        sample,
        agent_id=agent_id,
        task_id=task_id,
        phase=phase,
        sample_index=sample_index,
        timestamp=timestamp,
    )


def test_valid_local_zero_is_an_observation() -> None:
    accumulator = CgroupFeedbackAccumulator()

    assert _observe(accumulator, _sample()) is True

    snapshot = accumulator.consume()
    assert snapshot is not None
    assert snapshot.sample_count == 1
    assert snapshot.cpu_some_avg10 == 0.0
    assert snapshot.memory_some_avg10 == 0.0
    assert snapshot.io_some_avg10 == 0.0
    assert snapshot.pressure_level == "normal"


def test_peak_values_survive_later_lower_samples() -> None:
    accumulator = CgroupFeedbackAccumulator()
    _observe(
        accumulator,
        _sample(cpu=25.0, memory=3.0, io=1.0),
        phase="initial",
        sample_index=0,
        timestamp=100.0,
    )
    _observe(
        accumulator,
        _sample(cpu=2.0, memory=31.0, io=4.0),
        phase="running",
        sample_index=1,
        timestamp=101.0,
    )
    _observe(
        accumulator,
        _sample(cpu=1.0, memory=2.0, io=55.0),
        phase="final",
        sample_index=2,
        timestamp=102.0,
    )

    snapshot = accumulator.consume()
    assert snapshot is not None
    assert snapshot.cpu_some_avg10 == 25.0
    assert snapshot.memory_some_avg10 == 31.0
    assert snapshot.io_some_avg10 == 55.0
    assert snapshot.pressure_level == "overloaded"
    assert snapshot.sample_count == 3
    assert snapshot.phases == ("initial", "running", "final")
    assert snapshot.sample_indexes == (0, 1, 2)
    assert snapshot.first_timestamp == 100.0
    assert snapshot.last_timestamp == 102.0


def test_mixed_sources_accept_only_genuine_cgroup_resources() -> None:
    accumulator = CgroupFeedbackAccumulator()

    accepted = _observe(
        accumulator,
        _sample(
            cpu=22.0,
            memory=80.0,
            io=90.0,
            cpu_source="cgroup",
            memory_source="system",
            io_source="unavailable",
        ),
    )

    assert accepted is True
    snapshot = accumulator.consume()
    assert snapshot is not None
    assert snapshot.cpu_some_avg10 == 22.0
    assert snapshot.memory_some_avg10 == 0.0
    assert snapshot.io_some_avg10 == 0.0
    assert snapshot.local_resources == ("cpu",)
    assert snapshot.pressure_level == "critical"


def test_fallback_only_sample_does_not_create_pending_feedback() -> None:
    accumulator = CgroupFeedbackAccumulator()

    accepted = _observe(
        accumulator,
        _sample(
            cpu=50.0,
            memory=50.0,
            io=50.0,
            cpu_source="system",
            memory_source="system",
            io_source="unavailable",
        ),
    )

    assert accepted is False
    assert accumulator.pending is False
    assert accumulator.consume() is None


def test_invalid_local_values_are_ignored_without_raising() -> None:
    accumulator = CgroupFeedbackAccumulator()

    accepted = _observe(
        accumulator,
        _sample(cpu=-1.0, memory=math.nan, io=math.inf),
    )

    assert accepted is False
    assert accumulator.consume() is None


def test_valid_resource_survives_invalid_siblings() -> None:
    accumulator = CgroupFeedbackAccumulator()

    assert _observe(
        accumulator,
        _sample(cpu=12.0, memory=-1.0, io=math.nan),
    )

    snapshot = accumulator.consume()
    assert snapshot is not None
    assert snapshot.local_resources == ("cpu",)
    assert snapshot.cpu_some_avg10 == 12.0
    assert snapshot.pressure_level == "warning"


def test_provenance_is_unique_and_stably_ordered() -> None:
    accumulator = CgroupFeedbackAccumulator()
    _observe(
        accumulator,
        _sample(cpu=1.0),
        agent_id="agent-b",
        task_id="task-b",
        phase="running",
        sample_index=2,
        timestamp=100.0,
    )
    _observe(
        accumulator,
        _sample(cpu=2.0),
        agent_id="agent-a",
        task_id="task-a",
        phase="final",
        sample_index=3,
        timestamp=101.0,
    )
    _observe(
        accumulator,
        _sample(cpu=3.0),
        agent_id="agent-b",
        task_id="task-b",
        phase="running",
        sample_index=2,
        timestamp=102.0,
    )

    snapshot = accumulator.consume()
    assert snapshot is not None
    assert snapshot.agent_ids == ("agent-b", "agent-a")
    assert snapshot.task_ids == ("task-b", "task-a")
    assert snapshot.phases == ("running", "final")
    assert snapshot.sample_indexes == (2, 3)
    assert snapshot.sample_count == 3


def test_consume_returns_snapshot_once_and_clears_state() -> None:
    accumulator = CgroupFeedbackAccumulator()
    _observe(accumulator, _sample(cpu=21.0))

    snapshot = accumulator.consume()

    assert snapshot is not None
    assert snapshot.to_dict()["consumed"] is True
    assert snapshot.to_dict()["reusable"] is False
    assert accumulator.pending is False
    assert accumulator.consume() is None


def test_clear_discards_pending_cross_run_feedback() -> None:
    accumulator = CgroupFeedbackAccumulator()
    _observe(accumulator, _sample(cpu=21.0))
    assert accumulator.pending is True

    accumulator.clear()

    assert accumulator.pending is False
    assert accumulator.consume() is None
