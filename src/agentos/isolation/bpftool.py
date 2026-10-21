"""bpftool probe helpers for read-only kernel evidence.

This module provides two functions:

* :func:`probe_bpftool` – detect whether ``bpftool`` is available and
  report its version.
* :func:`probe_pinned_map` – read a pinned BPF map via ``bpftool map show``
  (read-only; never writes to the kernel).

All probes are best-effort: failures are captured in the return dict and
never propagate to the Runtime.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from agentos.isolation.bpf import BPFTaskRecord
from agentos.isolation.bpf_schema import (
    BPF_TASK_KEY_SIZE,
    BPF_TASK_VALUE_SIZE,
    BPFTaskKeyV1,
    BPFTaskValueV1,
)


class BPFMapError(RuntimeError):
    """Base error for real pinned BPF map operations."""


class BPFMapUnavailableError(BPFMapError):
    """The runtime environment cannot execute the requested BPF operation."""


class BPFMapSchemaError(BPFMapError):
    """A pinned map does not match the AgentOS binary schema."""


def run_bpftool_json(
    argv: list[str],
    *,
    timeout_seconds: float = 3.0,
) -> object:
    """Run one bpftool command and decode optional JSON stdout."""
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise BPFMapUnavailableError(
            f"bpftool binary not found: {argv[0]}"
        ) from exc
    except PermissionError as exc:
        raise BPFMapUnavailableError(
            f"bpftool binary is not executable: {argv[0]}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise BPFMapUnavailableError(
            f"bpftool command timed out after {timeout_seconds}s"
        ) from exc
    except OSError as exc:
        raise BPFMapUnavailableError(
            f"bpftool execution failed: {exc}"
        ) from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()[:500]
        raise BPFMapError(
            f"bpftool exited with code {result.returncode}: {stderr}"
        )

    stdout = (result.stdout or "").strip()
    if not stdout:
        return None
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise BPFMapError(f"bpftool returned invalid JSON: {exc}") from exc


_PIN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class BpftoolPinnedMapAdapter:
    """Read and write an AgentOS task map pinned in bpffs."""

    mode = "bpftool"
    schema_version = "bpf.task_value.v1"

    def __init__(
        self,
        *,
        pin_path: str | Path,
        runner: Callable[[list[str]], object] = run_bpftool_json,
        bpftool_path: str = "bpftool",
        max_entries: int = 4096,
        owns_pin: bool = False,
    ) -> None:
        self.pin_path = Path(pin_path)
        if not _PIN_NAME_RE.fullmatch(self.pin_path.name):
            raise ValueError(f"invalid BPF pin name: {self.pin_path.name!r}")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._runner = runner
        self._bpftool = bpftool_path
        self._max_entries = max_entries
        self._owns_pin = owns_pin
        self._known_task_ids: set[str] = set()

    def create_if_missing(self) -> dict[str, object]:
        """Create the pinned hash map or validate a compatible existing map."""
        if self.pin_path.exists():
            metadata = self._runner([
                self._bpftool,
                "-j",
                "map",
                "show",
                "pinned",
                str(self.pin_path),
            ])
            self._validate_metadata(metadata)
            return {
                "created": False,
                "pin_path": str(self.pin_path),
                "schema_verified": True,
            }

        try:
            self.pin_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise BPFMapUnavailableError(
                f"cannot create BPF pin directory {self.pin_path.parent}: {exc}"
            ) from exc

        self._runner([
            self._bpftool,
            "map",
            "create",
            str(self.pin_path),
            "type",
            "hash",
            "key",
            str(BPF_TASK_KEY_SIZE),
            "value",
            str(BPF_TASK_VALUE_SIZE),
            "entries",
            str(self._max_entries),
            "name",
            "agentos_tasks",
        ])
        self._owns_pin = True
        return {
            "created": True,
            "pin_path": str(self.pin_path),
            "schema_verified": True,
        }

    def _validate_metadata(self, raw: object) -> None:
        metadata: object = raw
        if isinstance(metadata, list):
            if len(metadata) != 1:
                raise BPFMapSchemaError(
                    f"expected one pinned map, got {len(metadata)}"
                )
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            raise BPFMapSchemaError(
                "bpftool map metadata must be a JSON object"
            )

        actual_type = metadata.get("type")
        actual_key = self._metadata_int(
            metadata, "key_size", "bytes_key", "key"
        )
        actual_value = self._metadata_int(
            metadata, "value_size", "bytes_value", "value"
        )
        actual_entries = self._metadata_int(metadata, "max_entries")

        expected = {
            "type": "hash",
            "key_size": BPF_TASK_KEY_SIZE,
            "value_size": BPF_TASK_VALUE_SIZE,
            "max_entries": self._max_entries,
        }
        actual = {
            "type": actual_type,
            "key_size": actual_key,
            "value_size": actual_value,
            "max_entries": actual_entries,
        }
        for field, expected_value in expected.items():
            if actual[field] != expected_value:
                raise BPFMapSchemaError(
                    f"{field} mismatch: expected {expected_value!r}, "
                    f"got {actual[field]!r}"
                )

    @staticmethod
    def _metadata_int(metadata: dict[str, Any], *names: str) -> int | None:
        for name in names:
            value = metadata.get(name)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        return None

    def update_task(self, record: BPFTaskRecord) -> None:
        key = BPFTaskKeyV1.from_task_id(record.task_id)
        value = self._value_from_record(record)
        self._runner([
            self._bpftool,
            "map",
            "update",
            "pinned",
            str(self.pin_path),
            "key",
            "hex",
            *key.to_hex_tokens(),
            "value",
            "hex",
            *value.to_hex_tokens(),
            "any",
        ])
        self._known_task_ids.add(record.task_id)

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        key = BPFTaskKeyV1.from_task_id(task_id)
        raw = self._runner([
            self._bpftool,
            "-j",
            "map",
            "lookup",
            "pinned",
            str(self.pin_path),
            "key",
            "hex",
            *key.to_hex_tokens(),
        ])
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise BPFMapError("bpftool lookup output must be a JSON object")
        return self._decode_entry(raw, task_id=task_id, expected_key=key.raw)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        raw = self._runner([
            self._bpftool,
            "-j",
            "map",
            "dump",
            "pinned",
            str(self.pin_path),
        ])
        if raw is None:
            return {}
        if not isinstance(raw, list):
            raise BPFMapError("bpftool dump output must be a JSON array")

        known_by_key = {
            BPFTaskKeyV1.from_task_id(task_id).raw: task_id
            for task_id in self._known_task_ids
        }
        result: dict[str, dict[str, Any]] = {}
        for item in raw:
            if not isinstance(item, dict):
                raise BPFMapError(
                    "bpftool dump entries must be JSON objects"
                )
            key_bytes = self._parse_bytes(item.get("key"), field="key")
            task_id = known_by_key.get(key_bytes)
            entry = self._decode_entry(
                item,
                task_id=task_id,
                expected_key=None,
            )
            result[task_id or key_bytes.hex()] = entry
        return result

    def clear(self) -> None:
        if self._owns_pin:
            try:
                self.pin_path.unlink(missing_ok=True)
            except OSError as exc:
                raise BPFMapError(
                    f"failed to remove owned BPF pin {self.pin_path}: {exc}"
                ) from exc
            self._known_task_ids.clear()
            return

        for task_id in sorted(self._known_task_ids):
            key = BPFTaskKeyV1.from_task_id(task_id)
            self._runner([
                self._bpftool,
                "map",
                "delete",
                "pinned",
                str(self.pin_path),
                "key",
                "hex",
                *key.to_hex_tokens(),
            ])
        self._known_task_ids.clear()

    @property
    def task_count(self) -> int:
        return len(self._known_task_ids)

    @staticmethod
    def _value_from_record(record: BPFTaskRecord) -> BPFTaskValueV1:
        return BPFTaskValueV1.from_fields(
            selected=record.selected,
            workload_class=record.workload_class,
            policy=record.policy,
            cpu_weight=record.cpu_weight,
            cpu_max_percent=record.cpu_max_percent,
            memory_max_mb=record.memory_max_mb,
            pids_max=record.pids_max,
            fairness_score=record.fairness_score,
            pressure_level=record.pressure_level,
            backpressure_signal=record.backpressure_signal,
            updated_at_ns=round(record.updated_at * 1_000_000_000),
        )

    def _decode_entry(
        self,
        raw: dict[str, Any],
        *,
        task_id: str | None,
        expected_key: bytes | None,
    ) -> dict[str, Any]:
        key_bytes = self._parse_bytes(raw.get("key"), field="key")
        if len(key_bytes) != BPF_TASK_KEY_SIZE:
            raise BPFMapSchemaError(
                f"lookup key must be {BPF_TASK_KEY_SIZE} bytes, "
                f"got {len(key_bytes)}"
            )
        if expected_key is not None and key_bytes != expected_key:
            raise BPFMapSchemaError(
                "bpftool lookup returned a different key"
            )
        value_bytes = self._parse_bytes(raw.get("value"), field="value")
        value = BPFTaskValueV1.from_bytes(value_bytes)
        return {
            "task_id": task_id,
            "key_hex": key_bytes.hex(),
            "decoded": self._value_to_dict(value),
        }

    @staticmethod
    def _parse_bytes(raw: object, *, field: str) -> bytes:
        if isinstance(raw, list):
            values: list[int] = []
            for item in raw:
                if isinstance(item, int) and not isinstance(item, bool):
                    value = item
                elif isinstance(item, str):
                    try:
                        value = int(item, 16)
                    except ValueError as exc:
                        raise BPFMapError(
                            f"invalid hexadecimal {field} byte: {item!r}"
                        ) from exc
                else:
                    raise BPFMapError(
                        f"invalid {field} byte type: {type(item).__name__}"
                    )
                if not 0 <= value <= 255:
                    raise BPFMapError(
                        f"{field} byte outside 0..255: {value}"
                    )
                values.append(value)
            return bytes(values)
        if isinstance(raw, str):
            compact = raw.replace(" ", "").replace(":", "")
            try:
                return bytes.fromhex(compact)
            except ValueError as exc:
                raise BPFMapError(
                    f"invalid hexadecimal {field} string"
                ) from exc
        raise BPFMapError(f"bpftool entry has no valid {field} bytes")

    @staticmethod
    def _value_to_dict(value: BPFTaskValueV1) -> dict[str, Any]:
        return {
            "version": value.version,
            "flags": value.flags,
            "selected": value.selected,
            "workload_class": value.workload_class,
            "policy": value.policy,
            "cpu_weight": value.cpu_weight,
            "cpu_max_milli_percent": value.cpu_max_milli_percent,
            "memory_max_bytes": value.memory_max_bytes,
            "pids_max": value.pids_max,
            "fairness_micros": value.fairness_micros,
            "pressure_level": value.pressure_level,
            "backpressure_signal": value.backpressure_signal,
            "updated_at_ns": value.updated_at_ns,
        }


def probe_bpftool(
    *,
    bpftool_path: str | None = None,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    """Probe for ``bpftool`` availability.

    Parameters
    ----------
    bpftool_path:
        Explicit path to the ``bpftool`` binary.  When ``None``, the
        function searches ``$PATH`` via :func:`shutil.which`.
    timeout_seconds:
        Subprocess timeout in seconds.

    Returns
    -------
    dict
        ``{"available": bool, "mode": str, "binary": str|None,
        "version": str|None, "reason": str|None}``
    """
    binary = bpftool_path or shutil.which("bpftool")
    if binary is None:
        return {
            "available": False,
            "mode": "unavailable",
            "binary": None,
            "version": None,
            "reason": "bpftool not found in PATH",
        }

    try:
        result = subprocess.run(
            [binary, "version"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError:
        return {
            "available": False,
            "mode": "unavailable",
            "binary": binary,
            "version": None,
            "reason": "bpftool binary not found",
        }
    except PermissionError:
        return {
            "available": False,
            "mode": "unavailable",
            "binary": binary,
            "version": None,
            "reason": "bpftool binary not executable (permission denied)",
        }
    except subprocess.TimeoutExpired:
        return {
            "available": False,
            "mode": "unavailable",
            "binary": binary,
            "version": None,
            "reason": f"bpftool version timed out after {timeout_seconds}s",
        }
    except OSError as exc:
        return {
            "available": False,
            "mode": "unavailable",
            "binary": binary,
            "version": None,
            "reason": f"bpftool execution failed: {exc}",
        }

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()[:200]
        return {
            "available": False,
            "mode": "unavailable",
            "binary": binary,
            "version": None,
            "reason": f"bpftool version exited with code {result.returncode}: {stderr}",
        }

    version_str = (result.stdout or "").strip().split("\n")[0] if result.stdout else ""
    return {
        "available": True,
        "mode": "probe_only",
        "binary": binary,
        "version": version_str or None,
        "reason": None,
    }


def probe_pinned_map(
    *,
    pinned_map_path: str | Path,
    bpftool_path: str | None = None,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    """Read a pinned BPF map via ``bpftool -j map show pinned <path>``.

    This is a **read-only** probe.  It never executes ``map update``,
    ``load``, ``attach``, or ``detach``.

    Parameters
    ----------
    pinned_map_path:
        Path to the pinned map in bpffs (e.g. ``/sys/fs/bpf/my_map``).
    bpftool_path:
        Explicit path to the ``bpftool`` binary.
    timeout_seconds:
        Subprocess timeout in seconds.

    Returns
    -------
    dict
        ``{"available": bool, "mode": str, "pinned_map_path": str,
        "map_info": dict|None, "reason": str|None}``
    """
    map_path = Path(pinned_map_path)
    binary = bpftool_path or shutil.which("bpftool")

    if binary is None:
        return {
            "available": False,
            "mode": "unavailable",
            "pinned_map_path": str(map_path),
            "map_info": None,
            "reason": "bpftool not found in PATH",
        }

    if not map_path.exists():
        return {
            "available": False,
            "mode": "unavailable",
            "pinned_map_path": str(map_path),
            "map_info": None,
            "reason": f"pinned map path does not exist: {map_path}",
        }

    cmd = [binary, "-j", "map", "show", "pinned", str(map_path)]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError:
        return {
            "available": False,
            "mode": "unavailable",
            "pinned_map_path": str(map_path),
            "map_info": None,
            "reason": "bpftool binary not found",
        }
    except PermissionError:
        return {
            "available": False,
            "mode": "unavailable",
            "pinned_map_path": str(map_path),
            "map_info": None,
            "reason": "bpftool binary not executable (permission denied)",
        }
    except subprocess.TimeoutExpired:
        return {
            "available": False,
            "mode": "unavailable",
            "pinned_map_path": str(map_path),
            "map_info": None,
            "reason": f"bpftool map show timed out after {timeout_seconds}s",
        }
    except OSError as exc:
        return {
            "available": False,
            "mode": "unavailable",
            "pinned_map_path": str(map_path),
            "map_info": None,
            "reason": f"bpftool execution failed: {exc}",
        }

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()[:200]
        return {
            "available": False,
            "mode": "unavailable",
            "pinned_map_path": str(map_path),
            "map_info": None,
            "reason": f"bpftool map show exited with code {result.returncode}: {stderr}",
        }

    stdout = (result.stdout or "").strip()
    if not stdout:
        return {
            "available": False,
            "mode": "unavailable",
            "pinned_map_path": str(map_path),
            "map_info": None,
            "reason": "bpftool map show returned empty output",
        }

    try:
        map_info = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return {
            "available": False,
            "mode": "unavailable",
            "pinned_map_path": str(map_path),
            "map_info": None,
            "reason": f"bpftool map show returned invalid JSON: {exc}",
        }

    return {
        "available": True,
        "mode": "pinned_probe",
        "pinned_map_path": str(map_path),
        "map_info": map_info,
        "reason": None,
    }
