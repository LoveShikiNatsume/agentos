from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from agentos.core.resources import ResourceQuota, ResourceUsage
from agentos.events.log import EventLog
from agentos.events.types import EventCategory, EventSeverity, RuntimeEvent
from agentos.isolation.pressure import read_psi_avg10_with_status


_CGROUP_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")


@dataclass(frozen=True)
class CgroupResult:
    available: bool
    path: Path
    event: RuntimeEvent
    reason: str | None = None


class CgroupV2Manager:
    _CONTROL_FILES = (
        "cgroup.procs",
        "cpu.max",
        "cpu.pressure",
        "cpu.stat",
        "cpu.weight",
        "io.pressure",
        "memory.current",
        "memory.max",
        "memory.peak",
        "memory.pressure",
        "pids.current",
        "pids.max",
    )

    def __init__(
        self,
        root: str | Path = "/sys/fs/cgroup/agentos",
        *,
        enable_controllers: bool = True,
    ) -> None:
        self.root = Path(root)
        self.enable_controllers = enable_controllers

    @staticmethod
    def _validate_name(name: str) -> None:
        if (
            not isinstance(name, str)
            or Path(name).is_absolute()
            or name.startswith(".")
            or "/" in name
            or "\\" in name
        ):
            raise ValueError(f"cgroup name traversal rejected: {name!r}")
        if _CGROUP_NAME_RE.fullmatch(name) is None:
            raise ValueError(f"cgroup name illegal chars: {name!r}")

    def create(self, name: str, quota: ResourceQuota | None = None) -> CgroupResult:
        path = self.root / name
        path_created = False
        try:
            self._validate_name(name)
            if self.root.exists() and not self.root.is_dir():
                raise NotADirectoryError(str(self.root))
            self.root.mkdir(parents=True, exist_ok=True)
            self._enable_available_controllers(self.root)
            if not path.exists():
                path.mkdir()
                path_created = True
            elif not path.is_dir():
                raise NotADirectoryError(str(path))
            if quota is not None:
                self.apply(path, quota)
                self._verify_limits(path, quota)
            return CgroupResult(
                available=True,
                path=path,
                event=RuntimeEvent(
                    event_type="resource.cgroup_created",
                    category=EventCategory.RESOURCE,
                    data={
                        "path": str(path),
                        "root": str(self.root),
                        "controllers": sorted(self.available_controllers(self.root)),
                        "limits": quota.to_dict() if quota is not None else {},
                    },
                ),
            )
        except (OSError, ValueError) as exc:
            reason = str(exc)
            if path_created:
                rollback = self.cleanup(path)
                if not rollback.available:
                    reason = f"{reason}; rollback failed: {rollback.reason}"
            return self._unavailable(path, reason)

    def apply(self, path: Path, quota: ResourceQuota) -> None:
        if quota.memory_max_mb is not None:
            (path / "memory.max").write_text(
                str(quota.memory_max_mb * 1024 * 1024), encoding="utf-8"
            )
        if quota.pids_max is not None:
            (path / "pids.max").write_text(str(quota.pids_max), encoding="utf-8")
        if quota.cpu_max_percent is not None:
            period = 100000
            quota_us = int(period * quota.cpu_max_percent / 100)
            (path / "cpu.max").write_text(f"{quota_us} {period}", encoding="utf-8")
        if quota.cpu_shares is not None:
            weight = min(max(int(quota.cpu_shares), 1), 10000)
            (path / "cpu.weight").write_text(str(weight), encoding="utf-8")

    def _verify_limits(self, path: Path, quota: ResourceQuota) -> None:
        limits = self.read_limits(path)
        if quota.memory_max_mb is not None:
            actual = limits.get("memory_max_mb")
            if actual != quota.memory_max_mb:
                raise OSError(
                    f"memory.max readback mismatch: wrote {quota.memory_max_mb}, got {actual}"
                )
        if quota.pids_max is not None:
            actual = limits.get("pids_max")
            if actual != quota.pids_max:
                raise OSError(
                    f"pids.max readback mismatch: wrote {quota.pids_max}, got {actual}"
                )
        if quota.cpu_max_percent is not None:
            actual = limits.get("cpu_max_percent")
            if actual is None or abs(float(actual) - quota.cpu_max_percent) > 1.0:
                raise OSError(
                    f"cpu.max readback mismatch: wrote {quota.cpu_max_percent}, got {actual}"
                )
        if quota.cpu_shares is not None:
            expected = min(max(int(quota.cpu_shares), 1), 10000)
            actual = limits.get("cpu_weight")
            if actual != expected:
                raise OSError(
                    f"cpu.weight readback mismatch: wrote {expected}, got {actual}"
                )

    def add_pid(self, path: Path, pid: int) -> CgroupResult:
        try:
            (path / "cgroup.procs").write_text(str(pid), encoding="utf-8")
            verified = self.verify_pid_membership(path, pid)
            verification_required = self._is_real_cgroup_path(path)
            if verification_required and not verified:
                raise OSError(
                    f"PID {pid} cgroup membership verification failed for {path}"
                )
            return CgroupResult(
                available=True,
                path=path,
                event=RuntimeEvent(
                    event_type="resource.cgroup_pid_added",
                    category=EventCategory.RESOURCE,
                    data={
                        "path": str(path),
                        "pid": pid,
                        "pid_membership_verified": verified,
                        "membership_check_required": verification_required,
                    },
                ),
            )
        except OSError as exc:
            return self._unavailable(path, str(exc))

    @staticmethod
    def verify_pid_membership(
        path: Path,
        pid: int,
        *,
        proc_root: Path = Path("/proc"),
        cgroup_root: Path = Path("/sys/fs/cgroup"),
    ) -> bool:
        """Confirm a PID's unified cgroup entry matches the expected path."""
        try:
            relative = path.resolve().relative_to(cgroup_root.resolve())
            expected = "/" if not relative.parts else "/" + relative.as_posix()
            text = (proc_root / str(pid) / "cgroup").read_text(encoding="utf-8")
        except (OSError, ValueError):
            return False

        for line in text.splitlines():
            parts = line.split(":", 2)
            if len(parts) != 3 or parts[0] != "0":
                continue
            actual = "/" + parts[2].strip().lstrip("/")
            if actual.rstrip("/") or actual == "/":
                actual = actual.rstrip("/") or "/"
            if actual == expected:
                return True
        return False

    def read_usage(self, path: Path) -> ResourceUsage:
        cpu_stat = self._read_key_values(path / "cpu.stat")
        return ResourceUsage(
            cpu_usage_usec=int(cpu_stat.get("usage_usec", 0)),
            memory_rss_mb=self._read_bytes_as_mb(path / "memory.current"),
            memory_peak_mb=self._read_bytes_as_mb(path / "memory.peak"),
            pids_current=self._read_int(path / "pids.current"),
        )

    def read_pressure(self, path: Path) -> dict[str, tuple[float, bool]]:
        return {
            resource: read_psi_avg10_with_status(
                path / f"{resource}.pressure"
            )
            for resource in ("cpu", "memory", "io")
        }

    def read_limits(self, path: Path) -> dict[str, object]:
        memory_raw = self._read_int(path / "memory.max")
        pids_raw = self._read_int(path / "pids.max")

        cpu_raw = ""
        try:
            cpu_raw = (path / "cpu.max").read_text(encoding="utf-8").strip()
        except OSError:
            cpu_raw = ""

        cpu_max_percent = None
        if cpu_raw:
            parts = cpu_raw.split()
            if len(parts) == 2 and parts[0] != "max":
                quota_us = int(parts[0])
                period_us = int(parts[1])
                if period_us > 0:
                    cpu_max_percent = quota_us / period_us * 100

        return {
            "memory_max_mb": (
                int(memory_raw / 1024 / 1024) if memory_raw > 0 else None
            ),
            "pids_max": pids_raw if pids_raw > 0 else None,
            "cpu_max_percent": cpu_max_percent,
            "cpu_weight": self._read_int(path / "cpu.weight") or None,
        }

    def cleanup(self, path: Path) -> CgroupResult:
        try:
            if not self._is_real_cgroup_path(path):
                for name in self._CONTROL_FILES:
                    file_path = path / name
                    if file_path.exists() and file_path.is_file():
                        file_path.unlink()
            path.rmdir()
            return CgroupResult(
                available=True,
                path=path,
                event=RuntimeEvent(
                    event_type="resource.cgroup_removed",
                    category=EventCategory.RESOURCE,
                    data={"path": str(path)},
                ),
            )
        except OSError as exc:
            return CgroupResult(
                available=False,
                path=path,
                reason=str(exc),
                event=RuntimeEvent(
                    event_type="resource.cgroup_cleanup_failed",
                    category=EventCategory.RESOURCE,
                    severity=EventSeverity.WARNING,
                    data={"path": str(path), "reason": str(exc)},
                ),
            )

    def available_controllers(self, path: Path | None = None) -> set[str]:
        controllers_file = (path or self.root) / "cgroup.controllers"
        if not controllers_file.exists():
            return set()
        return set(controllers_file.read_text(encoding="utf-8").split())

    def _enable_available_controllers(self, path: Path) -> None:
        if not self.enable_controllers:
            return
        subtree_control = path / "cgroup.subtree_control"
        if not subtree_control.exists():
            return
        controllers = self.available_controllers(path)
        for controller in ("cpu", "memory", "pids"):
            if controller in controllers:
                subtree_control.write_text(f"+{controller}", encoding="utf-8")

    @staticmethod
    def _read_int(path: Path) -> int:
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            return 0
        if raw == "" or raw == "max":
            return 0
        return int(raw)

    @classmethod
    def _read_bytes_as_mb(cls, path: Path) -> float:
        value = cls._read_int(path)
        if value <= 0:
            return 0.0
        return value / 1024 / 1024

    @staticmethod
    def _read_key_values(path: Path) -> dict[str, int]:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return {}
        values: dict[str, int] = {}
        for line in lines:
            parts = line.split()
            if len(parts) == 2:
                values[parts[0]] = int(parts[1])
        return values

    @staticmethod
    def _is_real_cgroup_path(path: Path) -> bool:
        try:
            path.resolve().relative_to(Path("/sys/fs/cgroup").resolve())
            return True
        except ValueError:
            return False

    @staticmethod
    def _unavailable(path: Path, reason: str) -> CgroupResult:
        return CgroupResult(
            available=False,
            path=path,
            reason=reason,
            event=RuntimeEvent(
                event_type="resource.cgroup_unavailable",
                category=EventCategory.RESOURCE,
                severity=EventSeverity.WARNING,
                data={"path": str(path), "reason": reason},
            ),
        )


def probe_cgroup(
    *,
    root: str | Path = "/sys/fs/cgroup/agentos",
    log: EventLog | str | Path | None = None,
) -> CgroupResult:
    event_log = log if isinstance(log, EventLog) or log is None else EventLog(log)
    result = CgroupV2Manager(root=root).create(
        "probe",
        ResourceQuota(memory_max_mb=64, pids_max=16, cpu_max_percent=50.0),
    )
    if event_log is not None:
        event_log.append(result.event)
    if result.available:
        CgroupV2Manager(root=root).cleanup(result.path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentos.isolation.cgroup")
    parser.add_argument("--root", default="/sys/fs/cgroup/agentos")
    parser.add_argument("--log", default="runs/latest/events.jsonl")
    args = parser.parse_args(argv)

    result = probe_cgroup(root=args.root, log=args.log)
    status = "available" if result.available else "unavailable"
    print(f"cgroup {status}: path={result.path} reason={result.reason or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
