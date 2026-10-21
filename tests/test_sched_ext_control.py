from __future__ import annotations

import fcntl
from pathlib import Path

import pytest

from agentos.isolation.bpf_schema import SchedPolicyKeyV1, SchedPolicyValueV1
from agentos.isolation.bpftool import BPFMapError
from agentos.isolation.sched_ext_control import (
    SchedExtControlClient,
    SchedExtPinMissingError,
)


class _MapRunner:
    def __init__(self) -> None:
        self.entries: dict[bytes, bytes] = {}
        self.commands: list[list[str]] = []

    def __call__(self, argv: list[str]) -> object:
        self.commands.append(argv)
        if "show" in argv:
            name = argv[-1].rsplit("/", 1)[-1]
            schemas = {
                "task_policy_map": ("hash", 8, 40),
                "task_state_map": ("hash", 8, 64),
                "global_config_map": ("array", 4, 32),
                "stats_map": ("percpu_array", 4, 8),
            }
            map_type, key_size, value_size = schemas[name]
            return {
                "type": map_type,
                "key_size": key_size,
                "value_size": value_size,
            }
        if "update" in argv:
            key_at = argv.index("key") + 2
            value_at = argv.index("value") + 2
            key = bytes(int(value, 16) for value in argv[key_at:value_at - 2])
            value = bytes(int(item, 16) for item in argv[value_at:-1])
            self.entries[key] = value
            return None
        if "lookup" in argv:
            key_at = argv.index("key") + 2
            key = bytes(int(value, 16) for value in argv[key_at:])
            return {"key": list(key), "value": list(self.entries[key])}
        if "delete" in argv:
            key_at = argv.index("key") + 2
            key = bytes(int(value, 16) for value in argv[key_at:])
            self.entries.pop(key)
            return None
        if "dump" in argv:
            return []
        raise AssertionError(argv)


def test_control_client_validates_all_pinned_map_schemas() -> None:
    runner = _MapRunner()
    client = SchedExtControlClient(pin_dir="/pins", runner=runner)

    metadata = client.validate_maps()

    assert set(metadata) == {
        "task_policy_map",
        "task_state_map",
        "global_config_map",
        "stats_map",
    }


def test_control_client_requires_the_expected_partial_scheduler(tmp_path: Path) -> None:
    root = tmp_path / "sched_ext"
    (root / "root").mkdir(parents=True)
    (root / "state").write_text("enabled\n", encoding="utf-8")
    (root / "switch_all").write_text("0\n", encoding="utf-8")
    (root / "root" / "ops").write_text("agentos_scx\n", encoding="utf-8")
    client = SchedExtControlClient(
        pin_dir="/pins", runner=_MapRunner(), sysfs_root=root
    )

    ready = client.validate_ready()

    assert ready["sysfs"]["state"] == "enabled"


def test_control_client_updates_reads_back_and_deletes_policy() -> None:
    runner = _MapRunner()
    client = SchedExtControlClient(pin_dir="/pins", runner=runner)
    key = SchedPolicyKeyV1(tgid=123, pid=123)
    policy = SchedPolicyValueV1.from_fields(generation=7, updated_at_ns=99)

    assert client.put_policy(key, policy) == policy
    assert client.get_policy(key) == policy
    client.delete_policy(key)

    assert runner.entries == {}


def test_control_client_policy_delete_is_idempotent_when_key_is_missing() -> None:
    key = SchedPolicyKeyV1(tgid=123, pid=123)

    def runner(argv: list[str]) -> object:
        if "delete" in argv:
            raise BPFMapError(
                "bpftool exited with code 254: Error: delete failed: "
                "No such file or directory"
            )
        raise AssertionError(argv)

    client = SchedExtControlClient(pin_dir="/pins", runner=runner)

    client.delete_policy(key)


def test_control_client_caches_schema_until_a_pin_identity_changes(
    tmp_path: Path,
) -> None:
    runner = _MapRunner()
    pin_dir = tmp_path / "pins"
    pin_dir.mkdir()
    for name in (
        "task_policy_map",
        "task_state_map",
        "global_config_map",
        "stats_map",
    ):
        (pin_dir / name).touch()
    client = SchedExtControlClient(pin_dir=pin_dir, runner=runner)

    client.validate_maps()
    client.validate_maps()
    assert sum("show" in command for command in runner.commands) == 4

    replacement = pin_dir / "replacement"
    replacement.touch()
    replacement.replace(pin_dir / "task_policy_map")
    client.validate_maps()

    assert sum("show" in command for command in runner.commands) == 8


def test_control_client_reports_missing_required_pin_as_definitive(
    tmp_path: Path,
) -> None:
    pin_dir = tmp_path / "pins"
    pin_dir.mkdir()
    client = SchedExtControlClient(pin_dir=pin_dir, runner=_MapRunner())

    with pytest.raises(SchedExtPinMissingError, match="task_policy_map"):
        client.validate_maps()


def test_control_client_updates_global_config_and_reads_stats(tmp_path: Path) -> None:
    runner = _MapRunner()
    client = SchedExtControlClient(
        pin_dir="/pins", runner=runner, lock_path=tmp_path / "control.lock"
    )

    config = client.update_global_config(
        generation=8,
        heartbeat_ns=100,
        loader_pid=42,
    )

    assert config == {
        "schema_version": 1,
        "generation": 8,
        "heartbeat_ns": 100,
        "loader_pid": 42,
        "flags": 0,
        "heartbeat_timeout_seconds": 3,
    }
    assert client.read_stats() == []


def test_global_config_update_preserves_loader_heartbeat(tmp_path: Path) -> None:
    runner = _MapRunner()
    client = SchedExtControlClient(
        pin_dir="/pins", runner=runner, lock_path=tmp_path / "control.lock"
    )
    client.update_global_config(
        generation=7,
        heartbeat_ns=1234,
        loader_pid=99,
        flags=3,
    )

    updated = client.update_global_config(generation=8, flags=4)

    assert updated["generation"] == 8
    assert updated["heartbeat_ns"] == 1234
    assert updated["loader_pid"] == 99
    assert client.read_global_config() == updated


def test_global_config_update_uses_exclusive_process_lock(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _MapRunner()
    calls: list[int] = []
    monkeypatch.setattr(
        "agentos.isolation.sched_ext_control.fcntl.flock",
        lambda _fd, operation: calls.append(operation),
    )
    client = SchedExtControlClient(
        pin_dir="/pins", runner=runner, lock_path=tmp_path / "control.lock"
    )

    client.update_global_config(generation=1, heartbeat_ns=10, loader_pid=20)

    assert calls == [fcntl.LOCK_EX, fcntl.LOCK_UN]


def test_read_health_combines_sysfs_and_global_config(tmp_path: Path) -> None:
    runner = _MapRunner()
    root = tmp_path / "sched_ext"
    (root / "root").mkdir(parents=True)
    for relative, value in {
        "state": "enabled",
        "switch_all": "0",
        "root/ops": "agentos_scx",
        "enable_seq": "12",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value + "\n", encoding="utf-8")
    client = SchedExtControlClient(
        pin_dir="/pins",
        runner=runner,
        sysfs_root=root,
        lock_path=tmp_path / "control.lock",
    )
    client.update_global_config(
        generation=4,
        heartbeat_ns=1_000,
        loader_pid=42,
    )

    health = client.read_health(now_ns=1_250)

    assert health["state"] == "enabled"
    assert health["root_ops"] == "agentos_scx"
    assert health["enable_seq"] == 12
    assert health["heartbeat_age_ns"] == 250
    assert set(health["maps"]) == {
        "task_policy_map",
        "task_state_map",
        "global_config_map",
        "stats_map",
    }
