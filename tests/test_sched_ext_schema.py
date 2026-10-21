from __future__ import annotations

import struct

import pytest

from agentos.isolation.bpf_schema import (
    SCHED_POLICY_FLAG_ADMITTED,
    SCHED_POLICY_FLAG_CRITICAL,
    SCHED_POLICY_KEY_SIZE,
    SCHED_POLICY_VALUE_SIZE,
    SchedPolicyKeyV1,
    SchedPolicyValueV1,
)


def test_sched_policy_key_v1_has_stable_little_endian_encoding() -> None:
    key = SchedPolicyKeyV1(tgid=0x11223344, pid=0x55667788)

    assert SCHED_POLICY_KEY_SIZE == 8
    assert key.to_bytes() == bytes.fromhex("4433221188776655")
    assert SchedPolicyKeyV1.from_bytes(key.to_bytes()) == key


def test_sched_policy_value_v1_round_trips_golden_bytes() -> None:
    value = SchedPolicyValueV1.from_fields(
        queue_class="latency",
        weight=250,
        slice_us=2_000,
        pressure_level="critical",
        admitted=True,
        critical=True,
        generation=9,
        updated_at_ns=123_456,
    )
    expected = struct.pack(
        "<IIIIIIQQ",
        1,
        1,
        250,
        2_000,
        3,
        SCHED_POLICY_FLAG_ADMITTED | SCHED_POLICY_FLAG_CRITICAL,
        9,
        123_456,
    )

    assert SCHED_POLICY_VALUE_SIZE == 40
    assert value.to_bytes() == expected
    assert SchedPolicyValueV1.from_bytes(expected) == value


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("queue_class", "unknown"),
        ("weight", 0),
        ("weight", 10_001),
        ("slice_us", 99),
        ("slice_us", 1_000_001),
        ("pressure_level", "invalid"),
        ("generation", -1),
    ],
)
def test_sched_policy_value_v1_rejects_invalid_fields(field: str, value: object) -> None:
    kwargs = {field: value}
    with pytest.raises(ValueError):
        SchedPolicyValueV1.from_fields(**kwargs)


def test_sched_policy_v1_rejects_wrong_sizes_versions_and_unknown_flags() -> None:
    with pytest.raises(ValueError, match="key must be 8 bytes"):
        SchedPolicyKeyV1.from_bytes(b"short")
    with pytest.raises(ValueError, match="value must be 40 bytes"):
        SchedPolicyValueV1.from_bytes(b"short")

    valid = SchedPolicyValueV1.from_fields().to_bytes()
    invalid_version = struct.pack("<I", 2) + valid[4:]
    with pytest.raises(ValueError, match="unsupported sched policy version"):
        SchedPolicyValueV1.from_bytes(invalid_version)

    values = list(struct.unpack("<IIIIIIQQ", valid))
    values[5] = 1 << 31
    with pytest.raises(ValueError, match="unknown sched policy flags"):
        SchedPolicyValueV1.from_bytes(struct.pack("<IIIIIIQQ", *values))
