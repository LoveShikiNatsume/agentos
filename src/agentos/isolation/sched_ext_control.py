"""Control-plane access to the pinned agentos_scx maps."""

from __future__ import annotations

import struct
import time
import fcntl
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from agentos.isolation.bpf_schema import (
    SCHED_POLICY_KEY_SIZE,
    SCHED_POLICY_SCHEMA_VERSION,
    SCHED_POLICY_VALUE_SIZE,
    SchedPolicyKeyV1,
    SchedPolicyValueV1,
)
from agentos.isolation.bpftool import BPFMapError, BPFMapSchemaError, run_bpftool_json


SCHED_EXT_PIN_DIR = Path("/sys/fs/bpf/agentos/sched_ext")
SCHED_EXT_CONTROL_LOCK = Path("/run/agentos/sched_ext-control.lock")
GLOBAL_CONFIG_VALUE_SIZE = 32
TASK_STATE_VALUE_SIZE = 64
DEFAULT_HEARTBEAT_TIMEOUT_SECONDS = 3
_GLOBAL_CONFIG_STRUCT = struct.Struct("<IIQQII")


class SchedExtControlError(BPFMapError):
    """A sched_ext control operation failed validation or readback."""


class SchedExtPinMissingError(SchedExtControlError):
    """A required pinned sched_ext map disappeared from bpffs."""


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
                    raise SchedExtControlError(
                        f"invalid hexadecimal {field} byte: {item!r}"
                    ) from exc
            else:
                raise SchedExtControlError(f"invalid {field} byte")
            if not 0 <= value <= 255:
                raise SchedExtControlError(f"{field} byte outside 0..255")
            values.append(value)
        return bytes(values)
    if isinstance(raw, str):
        try:
            return bytes.fromhex(raw.replace(" ", "").replace(":", ""))
        except ValueError as exc:
            raise SchedExtControlError(f"invalid hexadecimal {field}") from exc
    raise SchedExtControlError(f"missing {field} bytes")


class SchedExtControlClient:
    """Read, write, and verify the maps pinned by the Stage 3 loader."""

    def __init__(
        self,
        *,
        pin_dir: str | Path = SCHED_EXT_PIN_DIR,
        runner: Callable[[list[str]], object] = run_bpftool_json,
        bpftool_path: str = "bpftool",
        sysfs_root: str | Path = "/sys/kernel/sched_ext",
        lock_path: str | Path = SCHED_EXT_CONTROL_LOCK,
    ) -> None:
        self.pin_dir = Path(pin_dir)
        self.task_policy_path = self.pin_dir / "task_policy_map"
        self.global_config_path = self.pin_dir / "global_config_map"
        self.stats_path = self.pin_dir / "stats_map"
        self.task_state_path = self.pin_dir / "task_state_map"
        self._runner = runner
        self._bpftool = bpftool_path
        self.sysfs_root = Path(sysfs_root)
        self.lock_path = Path(lock_path)
        self._generations: dict[SchedPolicyKeyV1, int] = {}
        self._validated_maps: dict[str, dict[str, Any]] | None = None
        self._validated_pin_identity: tuple[tuple[int, int, int], ...] | None = None

    @contextmanager
    def _config_lock(self):
        try:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            with self.lock_path.open("a+b") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            raise SchedExtControlError(
                f"cannot lock sched_ext control plane: {exc}"
            ) from exc

    def _pin_identity(self) -> tuple[tuple[int, int, int], ...] | None:
        if not self.pin_dir.exists():
            # Unit-test and injected runners may model pins without a filesystem.
            return None
        identities: list[tuple[int, int, int]] = []
        for name in (
            "task_policy_map",
            "task_state_map",
            "global_config_map",
            "stats_map",
        ):
            path = self.pin_dir / name
            try:
                stat = path.stat()
            except FileNotFoundError as exc:
                raise SchedExtPinMissingError(
                    f"cannot stat sched_ext pin {path}: {exc}"
                ) from exc
            except OSError as exc:
                raise SchedExtControlError(f"cannot stat sched_ext pin {path}: {exc}") from exc
            identities.append((stat.st_dev, stat.st_ino, stat.st_ctime_ns))
        return tuple(identities)

    def validate_maps(self, *, force: bool = False) -> dict[str, dict[str, Any]]:
        identity = self._pin_identity()
        if (
            not force
            and identity is not None
            and identity == self._validated_pin_identity
            and self._validated_maps is not None
        ):
            return {name: dict(value) for name, value in self._validated_maps.items()}
        expected = {
            "task_policy_map": ("hash", SCHED_POLICY_KEY_SIZE, SCHED_POLICY_VALUE_SIZE),
            "task_state_map": ("hash", SCHED_POLICY_KEY_SIZE, TASK_STATE_VALUE_SIZE),
            "global_config_map": ("array", 4, GLOBAL_CONFIG_VALUE_SIZE),
            "stats_map": ("percpu_array", 4, 8),
        }
        result: dict[str, dict[str, Any]] = {}
        for name, (map_type, key_size, value_size) in expected.items():
            path = self.pin_dir / name
            raw = self._runner(
                [self._bpftool, "-j", "map", "show", "pinned", str(path)]
            )
            metadata = raw[0] if isinstance(raw, list) and len(raw) == 1 else raw
            if not isinstance(metadata, dict):
                raise BPFMapSchemaError(f"{name} metadata must be an object")
            actual = (
                metadata.get("type"),
                metadata.get("key_size", metadata.get("bytes_key")),
                metadata.get("value_size", metadata.get("bytes_value")),
            )
            if actual != (map_type, key_size, value_size):
                raise BPFMapSchemaError(
                    f"{name} schema mismatch: expected "
                    f"{(map_type, key_size, value_size)!r}, got {actual!r}"
                )
            result[name] = dict(metadata)
        if identity is not None:
            self._validated_pin_identity = identity
            self._validated_maps = {
                name: dict(value) for name, value in result.items()
            }
        return result

    def validate_ready(self) -> dict[str, Any]:
        expected = {
            "state": "enabled",
            "switch_all": "0",
            "root/ops": "agentos_scx",
        }
        actual: dict[str, str] = {}
        for relative, wanted in expected.items():
            try:
                value = (self.sysfs_root / relative).read_text(
                    encoding="utf-8", errors="replace"
                ).strip()
            except OSError as exc:
                raise SchedExtControlError(
                    f"cannot read sched_ext {relative}: {exc}"
                ) from exc
            actual[relative] = value
            if value != wanted:
                raise SchedExtControlError(
                    f"sched_ext {relative} must be {wanted!r}, got {value!r}"
                )
        maps = self.validate_maps()
        return {"maps": maps, "sysfs": actual}

    def read_global_config(self) -> dict[str, int]:
        raw = self._runner(
            [
                self._bpftool,
                "-j",
                "map",
                "lookup",
                "pinned",
                str(self.global_config_path),
                "key",
                "hex",
                "00",
                "00",
                "00",
                "00",
            ]
        )
        if not isinstance(raw, dict):
            raise SchedExtControlError("global config lookup must return an object")
        value = _parse_bytes(raw.get("value"), field="global config value")
        if len(value) != GLOBAL_CONFIG_VALUE_SIZE:
            raise SchedExtControlError(
                f"global config must be {GLOBAL_CONFIG_VALUE_SIZE} bytes"
            )
        schema, flags, generation, heartbeat, loader_pid, timeout = (
            _GLOBAL_CONFIG_STRUCT.unpack(value)
        )
        return {
            "schema_version": schema,
            "flags": flags,
            "generation": generation,
            "heartbeat_ns": heartbeat,
            "loader_pid": loader_pid,
            "heartbeat_timeout_seconds": timeout,
        }

    def read_health(self, *, now_ns: int | None = None) -> dict[str, Any]:
        sysfs: dict[str, Any] = {}
        for relative in ("state", "switch_all", "root/ops", "enable_seq"):
            try:
                value = (self.sysfs_root / relative).read_text(
                    encoding="utf-8", errors="replace"
                ).strip()
            except OSError as exc:
                raise SchedExtControlError(
                    f"cannot read sched_ext {relative}: {exc}"
                ) from exc
            sysfs[relative.replace("/", "_")] = (
                int(value) if relative == "enable_seq" else value
            )
        maps = self.validate_maps()
        config = self.read_global_config()
        now = time.monotonic_ns() if now_ns is None else now_ns
        heartbeat = int(config["heartbeat_ns"])
        return {
            **sysfs,
            **config,
            "maps": maps,
            "map_ids": {
                name: metadata.get("id") for name, metadata in maps.items()
            },
            "heartbeat_age_ns": max(0, now - heartbeat),
        }

    def put_policy(
        self,
        key: SchedPolicyKeyV1,
        policy: SchedPolicyValueV1,
    ) -> SchedPolicyValueV1:
        previous = self._generations.get(key)
        if previous is not None and policy.generation < previous:
            raise SchedExtControlError(
                f"generation regression for pid {key.pid}: "
                f"{policy.generation} < {previous}"
            )
        self._runner(
            [
                self._bpftool,
                "map",
                "update",
                "pinned",
                str(self.task_policy_path),
                "key",
                "hex",
                *key.to_hex_tokens(),
                "value",
                "hex",
                *policy.to_hex_tokens(),
                "any",
            ]
        )
        readback = self.get_policy(key)
        if readback != policy:
            try:
                self.delete_policy(key)
            except BPFMapError:
                pass
            raise SchedExtControlError("sched policy readback mismatch")
        self._generations[key] = policy.generation
        return readback

    def get_policy(self, key: SchedPolicyKeyV1) -> SchedPolicyValueV1:
        raw = self._runner(
            [
                self._bpftool,
                "-j",
                "map",
                "lookup",
                "pinned",
                str(self.task_policy_path),
                "key",
                "hex",
                *key.to_hex_tokens(),
            ]
        )
        if not isinstance(raw, dict):
            raise SchedExtControlError("policy lookup must return an object")
        returned_key = SchedPolicyKeyV1.from_bytes(_parse_bytes(raw.get("key"), field="key"))
        if returned_key != key:
            raise SchedExtControlError("policy lookup returned a different key")
        return SchedPolicyValueV1.from_bytes(_parse_bytes(raw.get("value"), field="value"))

    def delete_policy(self, key: SchedPolicyKeyV1) -> None:
        try:
            self._runner(
                [
                    self._bpftool,
                    "map",
                    "delete",
                    "pinned",
                    str(self.task_policy_path),
                    "key",
                    "hex",
                    *key.to_hex_tokens(),
                ]
            )
        except BPFMapError as exc:
            message = str(exc).lower()
            if "no such file or directory" not in message and "not found" not in message:
                raise
        self._generations.pop(key, None)

    def update_global_config(
        self,
        *,
        generation: int,
        heartbeat_ns: int | None = None,
        loader_pid: int | None = None,
        flags: int | None = None,
        heartbeat_timeout_seconds: int | None = None,
    ) -> dict[str, int]:
        if generation < 0:
            raise ValueError("generation must be non-negative")
        with self._config_lock():
            try:
                current = self.read_global_config()
            except (BPFMapError, KeyError):
                current = {}
            timeout = (
                heartbeat_timeout_seconds
                if heartbeat_timeout_seconds is not None
                else int(
                    current.get(
                        "heartbeat_timeout_seconds",
                        DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
                    )
                )
            )
            if not 1 <= timeout <= 60:
                raise ValueError("heartbeat_timeout_seconds must be in 1..60")
            heartbeat = (
                heartbeat_ns
                if heartbeat_ns is not None
                else int(current.get("heartbeat_ns", time.monotonic_ns()))
            )
            resolved_loader_pid = (
                loader_pid
                if loader_pid is not None
                else int(current.get("loader_pid", 0))
            )
            resolved_flags = flags if flags is not None else int(current.get("flags", 0))
            value = _GLOBAL_CONFIG_STRUCT.pack(
                SCHED_POLICY_SCHEMA_VERSION,
                resolved_flags,
                generation,
                heartbeat,
                resolved_loader_pid,
                timeout,
            )
            self._runner(
                [
                    self._bpftool,
                    "map",
                    "update",
                    "pinned",
                    str(self.global_config_path),
                    "key",
                    "hex",
                    "00",
                    "00",
                    "00",
                    "00",
                    "value",
                    "hex",
                    *[f"{byte:02x}" for byte in value],
                    "any",
                ]
            )
        return {
            "schema_version": SCHED_POLICY_SCHEMA_VERSION,
            "generation": generation,
            "heartbeat_ns": heartbeat,
            "loader_pid": resolved_loader_pid,
            "flags": resolved_flags,
            "heartbeat_timeout_seconds": timeout,
        }

    def read_stats(self) -> object:
        return self._runner(
            [self._bpftool, "-j", "map", "dump", "pinned", str(self.stats_path)]
        )

    @staticmethod
    def policy_to_dict(policy: SchedPolicyValueV1) -> dict[str, int]:
        return asdict(policy)
