"""Tests for bpftool probe adapter.

All subprocess calls are monkeypatched — no real bpftool needed.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from agentos.isolation.bpf import BPFTaskRecord
from agentos.isolation.bpf_schema import BPFTaskKeyV1, BPFTaskValueV1
from agentos.isolation.bpftool import (
    BPFMapSchemaError,
    BpftoolPinnedMapAdapter,
    probe_bpftool,
    probe_pinned_map,
)


class RecordingRunner:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> object:
        self.calls.append(list(argv))
        if not self.responses:
            raise AssertionError(f"unexpected bpftool call: {argv}")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


# ---------------------------------------------------------------------------
# probe_bpftool tests
# ---------------------------------------------------------------------------


class TestProbeBpftool:
    """Test probe_bpftool function."""

    def test_missing_binary_returns_unavailable(self):
        """When bpftool is not in PATH, returns unavailable."""
        with patch("agentos.isolation.bpftool.shutil.which", return_value=None):
            result = probe_bpftool()
        assert result["available"] is False
        assert result["mode"] == "unavailable"
        assert result["binary"] is None
        assert result["version"] is None
        assert "not found" in result["reason"]

    def test_version_success(self, tmp_path):
        """Successful bpftool version returns available + version string."""
        fake_bin = tmp_path / "bpftool"
        fake_bin.write_text("#!/bin/sh\necho 'bpftool v7.5.0'")
        fake_bin.chmod(0o755)

        result = probe_bpftool(bpftool_path=str(fake_bin))
        assert result["available"] is True
        assert result["mode"] == "probe_only"
        assert result["binary"] == str(fake_bin)
        assert "v7.5.0" in result["version"]
        assert result["reason"] is None

    def test_nonzero_exit_returns_unavailable(self, tmp_path):
        """Non-zero exit code returns unavailable with reason."""
        fake_bin = tmp_path / "bpftool"
        fake_bin.write_text("#!/bin/sh\nexit 1")
        fake_bin.chmod(0o755)

        result = probe_bpftool(bpftool_path=str(fake_bin))
        assert result["available"] is False
        assert result["mode"] == "unavailable"
        assert "exited with code 1" in result["reason"]

    def test_timeout_returns_unavailable(self, tmp_path):
        """Timeout returns unavailable with timeout reason."""
        fake_bin = tmp_path / "bpftool"
        fake_bin.write_text("#!/bin/sh\nsleep 10")
        fake_bin.chmod(0o755)

        result = probe_bpftool(bpftool_path=str(fake_bin), timeout_seconds=0.1)
        assert result["available"] is False
        assert result["mode"] == "unavailable"
        assert "timed out" in result["reason"]

    def test_permission_denied_returns_unavailable(self, tmp_path):
        """Permission denied returns unavailable."""
        fake_bin = tmp_path / "bpftool"
        fake_bin.write_text("#!/bin/sh\necho ok")
        fake_bin.chmod(0o000)

        try:
            result = probe_bpftool(bpftool_path=str(fake_bin))
        finally:
            fake_bin.chmod(0o755)  # cleanup

        assert result["available"] is False
        assert result["mode"] == "unavailable"
        assert "permission" in result["reason"].lower()

    def test_explicit_path_overrides_which(self):
        """Explicit bpftool_path is used instead of shutil.which."""
        with patch("agentos.isolation.bpftool.shutil.which", return_value="/usr/sbin/bpftool"):
            with patch("agentos.isolation.bpftool.subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="bpftool v7.0", stderr=""
                )
                result = probe_bpftool(bpftool_path="/custom/bpftool")
        assert result["available"] is True
        assert result["binary"] == "/custom/bpftool"


# ---------------------------------------------------------------------------
# probe_pinned_map tests
# ---------------------------------------------------------------------------


class TestProbePinnedMap:
    """Test probe_pinned_map function."""

    def test_missing_bpftool_returns_unavailable(self, tmp_path):
        """When bpftool not found, returns unavailable."""
        map_path = tmp_path / "my_map"
        map_path.touch()
        with patch("agentos.isolation.bpftool.shutil.which", return_value=None):
            result = probe_pinned_map(pinned_map_path=map_path)
        assert result["available"] is False
        assert result["mode"] == "unavailable"
        assert "not found" in result["reason"]

    def test_missing_map_path_returns_unavailable(self):
        """When pinned map path doesn't exist, returns unavailable."""
        with patch("agentos.isolation.bpftool.shutil.which", return_value="/usr/sbin/bpftool"):
            result = probe_pinned_map(pinned_map_path="/nonexistent/map")
        assert result["available"] is False
        assert result["mode"] == "unavailable"
        assert "does not exist" in result["reason"]

    def test_json_success_returns_pinned_probe(self, tmp_path):
        """Successful JSON output returns pinned_probe mode."""
        map_path = tmp_path / "my_map"
        map_path.touch()
        map_json = {"id": 42, "type": "hash", "name": "test_map", "key_size": 4, "value_size": 8}

        fake_bin = tmp_path / "bpftool"
        fake_bin.write_text(f"#!/bin/sh\necho '{json.dumps(map_json)}'")
        fake_bin.chmod(0o755)

        result = probe_pinned_map(pinned_map_path=map_path, bpftool_path=str(fake_bin))
        assert result["available"] is True
        assert result["mode"] == "pinned_probe"
        assert result["map_info"]["id"] == 42
        assert result["map_info"]["type"] == "hash"
        assert result["reason"] is None

    def test_corrupt_json_returns_unavailable(self, tmp_path):
        """Non-JSON output returns unavailable with parse error."""
        map_path = tmp_path / "my_map"
        map_path.touch()

        fake_bin = tmp_path / "bpftool"
        fake_bin.write_text("#!/bin/sh\necho 'not json'")
        fake_bin.chmod(0o755)

        result = probe_pinned_map(pinned_map_path=map_path, bpftool_path=str(fake_bin))
        assert result["available"] is False
        assert result["mode"] == "unavailable"
        assert "invalid JSON" in result["reason"]

    def test_empty_output_returns_unavailable(self, tmp_path):
        """Empty stdout returns unavailable."""
        map_path = tmp_path / "my_map"
        map_path.touch()

        fake_bin = tmp_path / "bpftool"
        fake_bin.write_text("#!/bin/sh\n:")
        fake_bin.chmod(0o755)

        result = probe_pinned_map(pinned_map_path=map_path, bpftool_path=str(fake_bin))
        assert result["available"] is False
        assert result["mode"] == "unavailable"
        assert "empty" in result["reason"]

    def test_nonzero_exit_returns_unavailable(self, tmp_path):
        """Non-zero exit returns unavailable."""
        map_path = tmp_path / "my_map"
        map_path.touch()

        fake_bin = tmp_path / "bpftool"
        fake_bin.write_text("#!/bin/sh\nexit 2")
        fake_bin.chmod(0o755)

        result = probe_pinned_map(pinned_map_path=map_path, bpftool_path=str(fake_bin))
        assert result["available"] is False
        assert "exited with code 2" in result["reason"]

    def test_timeout_returns_unavailable(self, tmp_path):
        """Timeout returns unavailable."""
        map_path = tmp_path / "my_map"
        map_path.touch()

        fake_bin = tmp_path / "bpftool"
        fake_bin.write_text("#!/bin/sh\nsleep 10")
        fake_bin.chmod(0o755)

        result = probe_pinned_map(pinned_map_path=map_path, bpftool_path=str(fake_bin), timeout_seconds=0.1)
        assert result["available"] is False
        assert "timed out" in result["reason"]


class TestBpftoolPinnedMapLifecycle:
    def test_create_if_missing_builds_expected_hash_map(
        self,
        tmp_path: Path,
    ) -> None:
        pin = tmp_path / "agentos" / "tasks"
        runner = RecordingRunner([None])
        adapter = BpftoolPinnedMapAdapter(
            pin_path=pin,
            runner=runner,
            bpftool_path="bpftool",
        )

        result = adapter.create_if_missing()

        assert result == {
            "created": True,
            "pin_path": str(pin),
            "schema_verified": True,
        }
        assert runner.calls == [[
            "bpftool",
            "map",
            "create",
            str(pin),
            "type",
            "hash",
            "key",
            "32",
            "value",
            "64",
            "entries",
            "4096",
            "name",
            "agentos_tasks",
        ]]

    def test_existing_compatible_map_is_reused(
        self,
        tmp_path: Path,
    ) -> None:
        pin = tmp_path / "tasks"
        pin.touch()
        runner = RecordingRunner([{
            "type": "hash",
            "key_size": 32,
            "value_size": 64,
            "max_entries": 4096,
        }])
        adapter = BpftoolPinnedMapAdapter(pin_path=pin, runner=runner)

        result = adapter.create_if_missing()

        assert result == {
            "created": False,
            "pin_path": str(pin),
            "schema_verified": True,
        }
        assert runner.calls == [[
            "bpftool",
            "-j",
            "map",
            "show",
            "pinned",
            str(pin),
        ]]

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("type", "array"),
            ("key_size", 4),
            ("value_size", 32),
            ("max_entries", 1024),
        ],
    )
    def test_existing_incompatible_map_is_rejected(
        self,
        tmp_path: Path,
        field: str,
        value: object,
    ) -> None:
        pin = tmp_path / "tasks"
        pin.touch()
        metadata = {
            "type": "hash",
            "key_size": 32,
            "value_size": 64,
            "max_entries": 4096,
        }
        metadata[field] = value
        runner = RecordingRunner([metadata])
        adapter = BpftoolPinnedMapAdapter(pin_path=pin, runner=runner)

        with pytest.raises(BPFMapSchemaError, match=field):
            adapter.create_if_missing()


def _task_record(task_id: str = "task-123") -> BPFTaskRecord:
    return BPFTaskRecord(
        task_id=task_id,
        workload_class="ai_inference",
        policy="latency_first",
        fairness_score=0.875,
        starvation_risk="low",
        selected=True,
        cpu_weight=512,
        cpu_max_percent=50.0,
        memory_max_mb=128,
        pids_max=32,
        pressure_level="warning",
        backpressure_signal="throttle",
        updated_at=123.456,
    )


def _encoded_record(record: BPFTaskRecord) -> tuple[list[int], list[int]]:
    key = BPFTaskKeyV1.from_task_id(record.task_id)
    value = BPFTaskValueV1.from_fields(
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
    return list(key.raw), list(value.to_bytes())


class TestBpftoolPinnedMapDataOperations:
    def test_update_builds_fixed_schema_command(self, tmp_path: Path) -> None:
        pin = tmp_path / "tasks"
        pin.touch()
        record = _task_record()
        key_bytes, value_bytes = _encoded_record(record)
        runner = RecordingRunner([None])
        adapter = BpftoolPinnedMapAdapter(pin_path=pin, runner=runner)

        adapter.update_task(record)

        assert runner.calls == [[
            "bpftool",
            "map",
            "update",
            "pinned",
            str(pin),
            "key",
            "hex",
            *[f"{byte:02x}" for byte in key_bytes],
            "value",
            "hex",
            *[f"{byte:02x}" for byte in value_bytes],
            "any",
        ]]
        assert adapter.task_count == 1

    def test_lookup_decodes_value_and_preserves_task_id(
        self,
        tmp_path: Path,
    ) -> None:
        pin = tmp_path / "tasks"
        pin.touch()
        record = _task_record()
        key_bytes, value_bytes = _encoded_record(record)
        runner = RecordingRunner([
            None,
            {"key": key_bytes, "value": value_bytes},
        ])
        adapter = BpftoolPinnedMapAdapter(pin_path=pin, runner=runner)
        adapter.update_task(record)

        result = adapter.get_task(record.task_id)

        assert result is not None
        assert result["task_id"] == record.task_id
        assert result["key_hex"] == bytes(key_bytes).hex()
        assert result["decoded"]["selected"] is True
        assert result["decoded"]["cpu_weight"] == 512
        assert result["decoded"]["memory_max_bytes"] == 128 * 1024 * 1024

    def test_snapshot_correlates_known_and_unknown_keys(
        self,
        tmp_path: Path,
    ) -> None:
        pin = tmp_path / "tasks"
        pin.touch()
        record = _task_record()
        key_bytes, value_bytes = _encoded_record(record)
        unknown_key = [0xA5] * 32
        runner = RecordingRunner([
            None,
            [
                {"key": key_bytes, "value": value_bytes},
                {"key": unknown_key, "value": value_bytes},
            ],
        ])
        adapter = BpftoolPinnedMapAdapter(pin_path=pin, runner=runner)
        adapter.update_task(record)

        result = adapter.snapshot()

        assert set(result) == {record.task_id, bytes(unknown_key).hex()}
        assert result[record.task_id]["decoded"]["selected"] is True
        assert result[bytes(unknown_key).hex()]["task_id"] is None

    def test_clear_unlinks_only_owned_pin(self, tmp_path: Path) -> None:
        pin = tmp_path / "tasks"
        pin.touch()
        adapter = BpftoolPinnedMapAdapter(
            pin_path=pin,
            runner=RecordingRunner([]),
            owns_pin=True,
        )

        adapter.clear()

        assert not pin.exists()
        assert adapter.task_count == 0
