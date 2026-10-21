"""Fixed binary schema for AgentOS pinned BPF task maps."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass


BPF_TASK_SCHEMA_VERSION = 1
BPF_TASK_KEY_SIZE = 32
BPF_TASK_VALUE_SIZE = 64

SCHED_POLICY_SCHEMA_VERSION = 1
SCHED_POLICY_KEY_SIZE = 8
SCHED_POLICY_VALUE_SIZE = 40

_VALUE_STRUCT = struct.Struct("<IIIIIIQIIIIQ8s")
assert _VALUE_STRUCT.size == BPF_TASK_VALUE_SIZE

_SCHED_POLICY_KEY_STRUCT = struct.Struct("<II")
_SCHED_POLICY_VALUE_STRUCT = struct.Struct("<IIIIIIQQ")
assert _SCHED_POLICY_KEY_STRUCT.size == SCHED_POLICY_KEY_SIZE
assert _SCHED_POLICY_VALUE_STRUCT.size == SCHED_POLICY_VALUE_SIZE

QUEUE_CLASS_CODES = {"latency": 1, "normal": 2, "batch": 3}
PRESSURE_LEVEL_CODES = {
    "normal": 1,
    "warning": 2,
    "critical": 3,
    "overloaded": 4,
}

SCHED_POLICY_FLAG_ADMITTED = 1 << 0
SCHED_POLICY_FLAG_CRITICAL = 1 << 1
SCHED_POLICY_FLAG_FALLBACK = 1 << 2
SCHED_POLICY_KNOWN_FLAGS = (
    SCHED_POLICY_FLAG_ADMITTED
    | SCHED_POLICY_FLAG_CRITICAL
    | SCHED_POLICY_FLAG_FALLBACK
)

_WORKLOAD = {
    "unknown": 0,
    "ai_inference": 1,
    "compile": 2,
    "compression": 3,
    "network": 4,
    "io_intensive": 5,
    "interactive": 6,
    "batch": 7,
}
_POLICY = {
    "unknown": 0,
    "latency_first": 1,
    "throughput_first": 2,
    "cache_affinity": 3,
    "fairness_balanced": 4,
    "risk_aware": 5,
}
_PRESSURE = {
    "unknown": 0,
    "normal": 1,
    "warning": 2,
    "critical": 3,
    "overloaded": 4,
}
_BACKPRESSURE = {
    "unknown": 0,
    "none": 1,
    "throttle": 2,
    "pause": 3,
    "reject": 4,
    "reduce_admission": 5,
    "defer": 6,
    "shed_load": 7,
}

FLAG_SELECTED = 1 << 0
FLAG_CPU_WEIGHT_PRESENT = 1 << 1
FLAG_CPU_MAX_PRESENT = 1 << 2
FLAG_MEMORY_MAX_PRESENT = 1 << 3
FLAG_PIDS_MAX_PRESENT = 1 << 4


def _require_u32(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if not 0 <= value <= 0xFFFFFFFF:
        raise ValueError(f"{name} outside u32 range: {value}")
    return value


def _require_u64(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if not 0 <= value <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"{name} outside u64 range: {value}")
    return value


def _enum_code(table: dict[str, int], value: object) -> int:
    return table.get(str(value).strip().lower(), 0)


@dataclass(frozen=True)
class SchedPolicyKeyV1:
    """The fixed PID/TGID key used by the sched_ext policy map."""

    tgid: int
    pid: int

    def to_bytes(self) -> bytes:
        return _SCHED_POLICY_KEY_STRUCT.pack(
            _require_u32("tgid", self.tgid),
            _require_u32("pid", self.pid),
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "SchedPolicyKeyV1":
        if len(data) != SCHED_POLICY_KEY_SIZE:
            raise ValueError(
                f"sched policy key must be {SCHED_POLICY_KEY_SIZE} bytes, "
                f"got {len(data)}"
            )
        return cls(*_SCHED_POLICY_KEY_STRUCT.unpack(data))

    def to_hex_tokens(self) -> list[str]:
        return [f"{byte:02x}" for byte in self.to_bytes()]


@dataclass(frozen=True)
class SchedPolicyValueV1:
    """The stable sched_policy.v1 value shared with agentos_scx."""

    version: int
    queue_class: int
    weight: int
    slice_us: int
    pressure_level: int
    flags: int
    generation: int
    updated_at_ns: int

    @classmethod
    def from_fields(
        cls,
        *,
        queue_class: str = "normal",
        weight: int = 100,
        slice_us: int = 5_000,
        pressure_level: str = "normal",
        admitted: bool = True,
        critical: bool = False,
        fallback: bool = False,
        generation: int = 0,
        updated_at_ns: int = 0,
    ) -> "SchedPolicyValueV1":
        queue_code = QUEUE_CLASS_CODES.get(queue_class)
        if queue_code is None:
            raise ValueError(f"unsupported queue_class: {queue_class!r}")
        pressure_code = PRESSURE_LEVEL_CODES.get(pressure_level)
        if pressure_code is None:
            raise ValueError(f"unsupported pressure_level: {pressure_level!r}")
        if not isinstance(weight, int) or isinstance(weight, bool) or not 1 <= weight <= 10_000:
            raise ValueError(f"weight outside 1..10000: {weight}")
        if not isinstance(slice_us, int) or isinstance(slice_us, bool) or not 100 <= slice_us <= 1_000_000:
            raise ValueError(f"slice_us outside 100..1000000: {slice_us}")
        flags = 0
        if admitted:
            flags |= SCHED_POLICY_FLAG_ADMITTED
        if critical:
            flags |= SCHED_POLICY_FLAG_CRITICAL
        if fallback:
            flags |= SCHED_POLICY_FLAG_FALLBACK
        return cls(
            version=SCHED_POLICY_SCHEMA_VERSION,
            queue_class=queue_code,
            weight=weight,
            slice_us=slice_us,
            pressure_level=pressure_code,
            flags=flags,
            generation=_require_u64("generation", generation),
            updated_at_ns=_require_u64("updated_at_ns", updated_at_ns),
        )

    def to_bytes(self) -> bytes:
        if self.version != SCHED_POLICY_SCHEMA_VERSION:
            raise ValueError(f"unsupported sched policy version: {self.version}")
        if self.queue_class not in QUEUE_CLASS_CODES.values():
            raise ValueError(f"invalid queue_class code: {self.queue_class}")
        if self.pressure_level not in PRESSURE_LEVEL_CODES.values():
            raise ValueError(f"invalid pressure_level code: {self.pressure_level}")
        if not 1 <= self.weight <= 10_000:
            raise ValueError(f"weight outside 1..10000: {self.weight}")
        if not 100 <= self.slice_us <= 1_000_000:
            raise ValueError(f"slice_us outside 100..1000000: {self.slice_us}")
        if self.flags & ~SCHED_POLICY_KNOWN_FLAGS:
            raise ValueError(f"unknown sched policy flags: {self.flags:#x}")
        return _SCHED_POLICY_VALUE_STRUCT.pack(
            self.version,
            self.queue_class,
            self.weight,
            self.slice_us,
            self.pressure_level,
            self.flags,
            _require_u64("generation", self.generation),
            _require_u64("updated_at_ns", self.updated_at_ns),
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "SchedPolicyValueV1":
        if len(data) != SCHED_POLICY_VALUE_SIZE:
            raise ValueError(
                f"sched policy value must be {SCHED_POLICY_VALUE_SIZE} bytes, "
                f"got {len(data)}"
            )
        value = cls(*_SCHED_POLICY_VALUE_STRUCT.unpack(data))
        value.to_bytes()
        return value

    def to_hex_tokens(self) -> list[str]:
        return [f"{byte:02x}" for byte in self.to_bytes()]


@dataclass(frozen=True)
class BPFTaskKeyV1:
    """A deterministic 32-byte key derived from a task ID."""

    raw: bytes

    def __post_init__(self) -> None:
        if len(self.raw) != BPF_TASK_KEY_SIZE:
            raise ValueError(
                f"BPF task key must be {BPF_TASK_KEY_SIZE} bytes, "
                f"got {len(self.raw)}"
            )

    @classmethod
    def from_task_id(cls, task_id: str) -> "BPFTaskKeyV1":
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task id must be a non-empty string")
        return cls(hashlib.sha256(task_id.encode("utf-8")).digest())

    def to_hex_tokens(self) -> list[str]:
        return [f"{byte:02x}" for byte in self.raw]


@dataclass(frozen=True)
class BPFTaskValueV1:
    """The fixed 64-byte value stored for each task."""

    version: int
    flags: int
    workload_class: int
    policy: int
    cpu_weight: int
    cpu_max_milli_percent: int
    memory_max_bytes: int
    pids_max: int
    fairness_micros: int
    pressure_level: int
    backpressure_signal: int
    updated_at_ns: int

    @property
    def selected(self) -> bool:
        return bool(self.flags & FLAG_SELECTED)

    @classmethod
    def from_fields(
        cls,
        *,
        selected: bool = False,
        workload_class: str = "unknown",
        policy: str = "unknown",
        cpu_weight: int | None = None,
        cpu_max_percent: float | None = None,
        memory_max_mb: int | None = None,
        pids_max: int | None = None,
        fairness_score: float = 0.0,
        pressure_level: str = "unknown",
        backpressure_signal: str = "unknown",
        updated_at_ns: int = 0,
    ) -> "BPFTaskValueV1":
        flags = FLAG_SELECTED if selected else 0

        encoded_cpu_weight = 0
        if cpu_weight is not None:
            if not isinstance(cpu_weight, int) or isinstance(cpu_weight, bool):
                raise ValueError("cpu_weight must be an integer")
            if not 1 <= cpu_weight <= 10_000:
                raise ValueError(
                    f"cpu_weight outside 1..10000: {cpu_weight}"
                )
            encoded_cpu_weight = cpu_weight
            flags |= FLAG_CPU_WEIGHT_PRESENT

        encoded_cpu_max = 0
        if cpu_max_percent is not None:
            if not isinstance(cpu_max_percent, (int, float)) or isinstance(
                cpu_max_percent, bool
            ):
                raise ValueError("cpu_max_percent must be numeric")
            if not 0.0 <= float(cpu_max_percent) <= 100.0:
                raise ValueError(
                    f"cpu_max_percent outside 0..100: {cpu_max_percent}"
                )
            encoded_cpu_max = round(float(cpu_max_percent) * 1000)
            flags |= FLAG_CPU_MAX_PRESENT

        encoded_memory = 0
        if memory_max_mb is not None:
            if not isinstance(memory_max_mb, int) or isinstance(
                memory_max_mb, bool
            ):
                raise ValueError("memory_max_mb must be an integer")
            encoded_memory = _require_u64(
                "memory_max_bytes", memory_max_mb * 1024 * 1024
            )
            flags |= FLAG_MEMORY_MAX_PRESENT

        encoded_pids = 0
        if pids_max is not None:
            encoded_pids = _require_u32("pids_max", pids_max)
            flags |= FLAG_PIDS_MAX_PRESENT

        if not isinstance(fairness_score, (int, float)) or isinstance(
            fairness_score, bool
        ):
            raise ValueError("fairness_score must be numeric")
        if not 0.0 <= float(fairness_score) <= 1.0:
            raise ValueError(
                f"fairness_score outside 0..1: {fairness_score}"
            )

        return cls(
            version=BPF_TASK_SCHEMA_VERSION,
            flags=_require_u32("flags", flags),
            workload_class=_enum_code(_WORKLOAD, workload_class),
            policy=_enum_code(_POLICY, policy),
            cpu_weight=encoded_cpu_weight,
            cpu_max_milli_percent=_require_u32(
                "cpu_max_milli_percent", encoded_cpu_max
            ),
            memory_max_bytes=encoded_memory,
            pids_max=encoded_pids,
            fairness_micros=_require_u32(
                "fairness_micros",
                round(float(fairness_score) * 1_000_000),
            ),
            pressure_level=_enum_code(_PRESSURE, pressure_level),
            backpressure_signal=_enum_code(
                _BACKPRESSURE, backpressure_signal
            ),
            updated_at_ns=_require_u64("updated_at_ns", updated_at_ns),
        )

    def to_bytes(self) -> bytes:
        return _VALUE_STRUCT.pack(
            _require_u32("version", self.version),
            _require_u32("flags", self.flags),
            _require_u32("workload_class", self.workload_class),
            _require_u32("policy", self.policy),
            _require_u32("cpu_weight", self.cpu_weight),
            _require_u32(
                "cpu_max_milli_percent", self.cpu_max_milli_percent
            ),
            _require_u64("memory_max_bytes", self.memory_max_bytes),
            _require_u32("pids_max", self.pids_max),
            _require_u32("fairness_micros", self.fairness_micros),
            _require_u32("pressure_level", self.pressure_level),
            _require_u32("backpressure_signal", self.backpressure_signal),
            _require_u64("updated_at_ns", self.updated_at_ns),
            b"\x00" * 8,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "BPFTaskValueV1":
        if len(data) != BPF_TASK_VALUE_SIZE:
            raise ValueError(
                f"BPF task value must be {BPF_TASK_VALUE_SIZE} bytes, "
                f"got {len(data)}"
            )
        unpacked = _VALUE_STRUCT.unpack(data)
        version = unpacked[0]
        if version != BPF_TASK_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported BPF task schema version: {version}"
            )
        reserved = unpacked[-1]
        if reserved != b"\x00" * 8:
            raise ValueError("BPF task value reserved bytes must be zero")
        return cls(*unpacked[:-1])

    def to_hex_tokens(self) -> list[str]:
        return [f"{byte:02x}" for byte in self.to_bytes()]
