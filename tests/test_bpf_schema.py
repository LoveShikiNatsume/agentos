from __future__ import annotations

import pytest

from agentos.isolation.bpf_schema import (
    BPF_TASK_SCHEMA_VERSION,
    FLAG_CPU_MAX_PRESENT,
    FLAG_CPU_WEIGHT_PRESENT,
    FLAG_MEMORY_MAX_PRESENT,
    FLAG_PIDS_MAX_PRESENT,
    BPFTaskKeyV1,
    BPFTaskValueV1,
)


def test_task_key_is_stable_full_sha256() -> None:
    key = BPFTaskKeyV1.from_task_id("task-123")

    assert len(key.raw) == 32
    assert key == BPFTaskKeyV1.from_task_id("task-123")
    assert len(key.to_hex_tokens()) == 32
    assert all(len(token) == 2 for token in key.to_hex_tokens())


def test_task_key_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="empty"):
        BPFTaskKeyV1.from_task_id("")


def test_task_value_is_exactly_64_bytes_and_round_trips() -> None:
    value = BPFTaskValueV1.from_fields(
        selected=True,
        workload_class="ai_inference",
        policy="latency_first",
        cpu_weight=512,
        cpu_max_percent=50.0,
        memory_max_mb=128,
        pids_max=32,
        fairness_score=0.875,
        pressure_level="warning",
        backpressure_signal="throttle",
        updated_at_ns=123456789,
    )

    encoded = value.to_bytes()
    decoded = BPFTaskValueV1.from_bytes(encoded)

    assert len(encoded) == 64
    assert decoded == value
    assert decoded.version == BPF_TASK_SCHEMA_VERSION
    assert decoded.selected is True
    assert decoded.memory_max_bytes == 128 * 1024 * 1024
    assert decoded.flags & FLAG_CPU_WEIGHT_PRESENT
    assert decoded.flags & FLAG_CPU_MAX_PRESENT
    assert decoded.flags & FLAG_MEMORY_MAX_PRESENT
    assert decoded.flags & FLAG_PIDS_MAX_PRESENT


def test_absent_limits_have_zero_values_without_presence_flags() -> None:
    value = BPFTaskValueV1.from_fields(updated_at_ns=1)

    assert value.cpu_weight == 0
    assert value.cpu_max_milli_percent == 0
    assert value.memory_max_bytes == 0
    assert value.pids_max == 0
    assert value.flags & FLAG_CPU_WEIGHT_PRESENT == 0
    assert value.flags & FLAG_CPU_MAX_PRESENT == 0
    assert value.flags & FLAG_MEMORY_MAX_PRESENT == 0
    assert value.flags & FLAG_PIDS_MAX_PRESENT == 0


@pytest.mark.parametrize("length", [0, 63, 65])
def test_task_value_rejects_wrong_length(length: int) -> None:
    with pytest.raises(ValueError, match="64"):
        BPFTaskValueV1.from_bytes(b"\x00" * length)


def test_task_value_rejects_unsupported_version() -> None:
    value = BPFTaskValueV1.from_fields(updated_at_ns=1)
    encoded = bytearray(value.to_bytes())
    encoded[:4] = (99).to_bytes(4, "little")

    with pytest.raises(ValueError, match="version"):
        BPFTaskValueV1.from_bytes(bytes(encoded))


@pytest.mark.parametrize("cpu_weight", [0, 10_001])
def test_task_value_rejects_out_of_range_cpu_weight(cpu_weight: int) -> None:
    with pytest.raises(ValueError, match="cpu_weight"):
        BPFTaskValueV1.from_fields(cpu_weight=cpu_weight)


@pytest.mark.parametrize("cpu_max_percent", [-0.1, 100.1])
def test_task_value_rejects_out_of_range_cpu_percent(
    cpu_max_percent: float,
) -> None:
    with pytest.raises(ValueError, match="cpu_max_percent"):
        BPFTaskValueV1.from_fields(cpu_max_percent=cpu_max_percent)


@pytest.mark.parametrize("fairness_score", [-0.1, 1.1])
def test_task_value_rejects_out_of_range_fairness(
    fairness_score: float,
) -> None:
    with pytest.raises(ValueError, match="fairness_score"):
        BPFTaskValueV1.from_fields(fairness_score=fairness_score)


def test_unknown_enums_encode_as_zero() -> None:
    value = BPFTaskValueV1.from_fields(
        workload_class="future-workload",
        policy="future-policy",
        pressure_level="future-pressure",
        backpressure_signal="future-signal",
    )

    assert value.workload_class == 0
    assert value.policy == 0
    assert value.pressure_level == 0
    assert value.backpressure_signal == 0
