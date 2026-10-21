from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field


@dataclass
class ResourceQuota:
    cpu_shares: int | None = None
    cpu_max_percent: float | None = None
    memory_max_mb: int | None = None
    pids_max: int | None = None
    io_read_bps: int | None = None
    io_write_bps: int | None = None
    timeout_seconds: float | None = None
    allowed_tools: list[str] = field(default_factory=list)
    allow_network: bool = False
    allow_filesystem_write: bool = False

    def has_limits(self) -> bool:
        return any(
            value is not None
            for value in (
                self.cpu_shares,
                self.cpu_max_percent,
                self.memory_max_mb,
                self.pids_max,
                self.io_read_bps,
                self.io_write_bps,
                self.timeout_seconds,
            )
        ) or bool(self.allowed_tools) or self.allow_network or self.allow_filesystem_write

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ResourceUsage:
    cpu_percent: float = 0.0
    cpu_usage_usec: int = 0
    memory_rss_mb: float = 0.0
    memory_peak_mb: float = 0.0
    pids_current: int = 0
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    gpu_memory_mb: float = 0.0
    gpu_util_percent: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ResourceUsageSummary:
    samples: int = 0
    cpu_percent: float = 0.0
    memory_peak_mb: float = 0.0
    pids_current: int = 0

    def record(self, usage: ResourceUsage) -> None:
        self.samples += 1
        if usage.cpu_percent > 0:
            self.cpu_percent = max(self.cpu_percent, usage.cpu_percent)
        if usage.memory_peak_mb > 0:
            self.memory_peak_mb = max(self.memory_peak_mb, usage.memory_peak_mb)
        if usage.pids_current > 0:
            self.pids_current = max(self.pids_current, usage.pids_current)

    def to_quota(self) -> ResourceQuota | None:
        if self.samples == 0:
            return None

        quota = ResourceQuota()
        if self.cpu_percent > 0:
            quota.cpu_max_percent = self.cpu_percent
        if self.memory_peak_mb > 0:
            quota.memory_max_mb = math.ceil(self.memory_peak_mb)
        if self.pids_current > 0:
            quota.pids_max = self.pids_current

        return quota if quota.has_limits() else None
