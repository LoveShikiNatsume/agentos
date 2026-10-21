"""Tests for probe_sched_ext: openEuler detection, cgroup v2 fields.

Phase 4 additions: BPFTaskRecord v2 fields, bpffs mount check, PSI,
CAP_BPF, force_mock subprocess guard, sched_ext-still-mock, CLI --json-only.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

# Add project root to path for imports
PROJECT_ROOT = (Path(__file__).parent.parent / "src")
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.probe_sched_ext import parse_os_release, probe


DEBIAN_OS_RELEASE = """\
PRETTY_NAME="Debian GNU/Linux 12 (bookworm)"
NAME="Debian GNU/Linux"
VERSION_ID="12"
VERSION="12 (bookworm)"
ID=debian
"""

OPENEULER_OS_RELEASE = """\
NAME="openEuler"
VERSION="22.03 (LTS-SP3)"
ID="openEuler"
VERSION_ID="22.03"
PRETTY_NAME="openEuler 22.03 (LTS-SP3)"
"""

OPENEULER_NO_VERSION = """\
NAME="openEuler"
ID="openEuler"
PRETTY_NAME="openEuler Custom Build"
"""


def test_debian_not_openeuler():
    result = parse_os_release(DEBIAN_OS_RELEASE)
    assert result["is_openeuler"] is False
    assert result["openeuler_version"] == ""


def test_openeuler_detected_with_version():
    result = parse_os_release(OPENEULER_OS_RELEASE)
    assert result["is_openeuler"] is True
    assert result["openeuler_version"] == "22.03"


def test_openeuler_fallback_to_pretty_name():
    result = parse_os_release(OPENEULER_NO_VERSION)
    assert result["is_openeuler"] is True
    assert result["openeuler_version"] == "openEuler Custom Build"


def test_empty_os_release():
    result = parse_os_release("")
    assert result["is_openeuler"] is False
    assert result["openeuler_version"] == ""


def test_probe_result_has_cgroup_fields():
    result = probe()
    d = result.to_dict()
    assert "cgroup_v2_mounted" in d
    assert "cgroup_v2_path" in d
    assert "cgroup_v2_controllers" in d
    assert isinstance(d["cgroup_v2_controllers"], list)


def test_probe_result_non_openeuler_version_empty():
    result = probe()
    if not result.is_openeuler:
        assert result.openeuler_version == ""


def test_probe_summary_keys():
    result = probe()
    s = result.summary
    assert "sched_ext" in s
    assert "bpf_map" in s
    assert "reason" in s
    assert s["sched_ext"] in ("available", "unavailable")
    # Phase 4: bpf_map is "mock" or "pinned_probe", never "native"
    assert s["bpf_map"] in ("mock", "pinned_probe")


# New tests for P2: KernelProbeResult and probe_sched_ext
import json

from agentos.isolation.sched_ext import KernelProbeResult, probe_sched_ext


class TestKernelProbeResult:
    """Test KernelProbeResult data class."""

    def test_default_values(self):
        result = KernelProbeResult(
            sched_ext="unavailable",
            bpf_map="mock",
            fallback_mode="mock_bpf_map",
        )
        assert result.sched_ext == "unavailable"
        assert result.bpf_map == "mock"
        assert result.fallback_mode == "mock_bpf_map"
        assert result.reason is None
        assert result.details == {}

    def test_to_dict(self):
        result = KernelProbeResult(
            sched_ext="available",
            bpf_map="mock",
            fallback_mode="mock_bpf_map",
            reason="test reason",
            details={"key": "value"},
        )
        d = result.to_dict()
        assert d["sched_ext"] == "available"
        assert d["bpf_map"] == "mock"
        assert d["fallback_mode"] == "mock_bpf_map"
        assert d["reason"] == "test reason"
        assert d["details"] == {"key": "value"}

    def test_to_dict_is_json_serializable(self):
        result = KernelProbeResult(
            sched_ext="unavailable",
            bpf_map="mock",
            fallback_mode="mock_bpf_map",
            reason="test",
        )
        # Should not raise
        json_str = json.dumps(result.to_dict())
        parsed = json.loads(json_str)
        assert parsed["sched_ext"] == "unavailable"


class TestProbeSchedExt:
    """Test probe_sched_ext function."""

    def test_probe_sched_ext_force_mock_returns_mock_result(self):
        """Test 1: force mock 输出稳定."""
        result = probe_sched_ext(force_mock=True)
        assert result.sched_ext == "unavailable"
        assert result.bpf_map == "mock"
        assert result.fallback_mode == "mock_bpf_map"
        assert result.reason == "forced mock mode"

    def test_probe_sched_ext_missing_path_falls_back_to_mock(self, tmp_path):
        """Test 2: 不存在 sys kernel path fallback."""
        result = probe_sched_ext(sys_kernel_path=tmp_path / "missing")
        assert result.sched_ext == "unavailable"
        assert result.bpf_map == "mock"
        assert result.fallback_mode == "mock_bpf_map"
        assert "sched_ext not present" in result.reason

    def test_probe_sched_ext_with_sched_ext_path(self, tmp_path):
        """Test 3: sched_ext path 存在时返回 available."""
        sched_ext_path = tmp_path / "sched_ext"
        sched_ext_path.mkdir()
        result = probe_sched_ext(sys_kernel_path=tmp_path)
        assert result.sched_ext == "available"
        assert result.bpf_map == "mock"  # First version always uses mock
        assert result.fallback_mode == "mock_bpf_map"

    def test_probe_sched_ext_result_is_json_serializable(self):
        """Test 4: result 可 JSON 序列化."""
        result = probe_sched_ext(force_mock=True)
        json_str = json.dumps(result.to_dict())
        parsed = json.loads(json_str)
        assert parsed["bpf_map"] == "mock"


# Tests for MockBPFMapAdapter
from agentos.isolation.bpf import (
    BPFTaskRecord,
    MockBPFMapAdapter,
    task_record_from_decision,
)


class TestBPFTaskRecord:
    """Test BPFTaskRecord data class."""

    def test_to_dict(self):
        record = BPFTaskRecord(
            task_id="t1",
            workload_class="batch",
            policy="fairness_balanced",
            fairness_score=0.75,
            starvation_risk="high",
            selected=True,
            metadata={"reason": "selected"},
        )
        d = record.to_dict()
        assert d["task_id"] == "t1"
        assert d["workload_class"] == "batch"
        assert d["policy"] == "fairness_balanced"
        assert d["fairness_score"] == 0.75
        assert d["starvation_risk"] == "high"
        assert d["selected"] is True
        assert d["metadata"]["reason"] == "selected"

    def test_to_dict_is_json_serializable(self):
        record = BPFTaskRecord(
            task_id="t1",
            workload_class="batch",
            policy="latency_first",
            fairness_score=1.0,
            starvation_risk="low",
        )
        json_str = json.dumps(record.to_dict())
        parsed = json.loads(json_str)
        assert parsed["task_id"] == "t1"


class TestMockBPFMapAdapter:
    """Test MockBPFMapAdapter class."""

    def test_mock_bpf_map_adapter_updates_and_snapshots_task_records(self):
        """Test update_task and snapshot."""
        adapter = MockBPFMapAdapter()
        record = BPFTaskRecord(
            task_id="t1",
            workload_class="batch",
            policy="fairness_balanced",
            fairness_score=0.7,
            starvation_risk="high",
            selected=True,
        )
        adapter.update_task(record)
        snapshot = adapter.snapshot()
        assert snapshot["t1"]["policy"] == "fairness_balanced"
        assert snapshot["t1"]["starvation_risk"] == "high"

    def test_get_task_returns_none_for_missing(self):
        adapter = MockBPFMapAdapter()
        assert adapter.get_task("missing") is None

    def test_get_task_returns_dict(self):
        adapter = MockBPFMapAdapter()
        record = BPFTaskRecord(
            task_id="t1",
            workload_class="batch",
            policy="latency_first",
            fairness_score=1.0,
            starvation_risk="low",
        )
        adapter.update_task(record)
        result = adapter.get_task("t1")
        assert result is not None
        assert result["task_id"] == "t1"

    def test_task_count(self):
        adapter = MockBPFMapAdapter()
        assert adapter.task_count == 0
        adapter.update_task(BPFTaskRecord(
            task_id="t1",
            workload_class="batch",
            policy="latency_first",
            fairness_score=1.0,
            starvation_risk="low",
        ))
        assert adapter.task_count == 1

    def test_clear(self):
        adapter = MockBPFMapAdapter()
        adapter.update_task(BPFTaskRecord(
            task_id="t1",
            workload_class="batch",
            policy="latency_first",
            fairness_score=1.0,
            starvation_risk="low",
        ))
        adapter.clear()
        assert adapter.task_count == 0


class TestTaskRecordFromDecision:
    """Test task_record_from_decision helper."""

    def test_task_record_from_decision_maps_scheduler_fields(self):
        """Test that helper maps scheduler decision fields correctly."""
        scheduler_decision = {
            "policy": "fairness_balanced",
            "fairness": {"fairness_score": 0.75},
            "selected": ["t1"],
        }
        task_decision = {
            "task_id": "t1",
            "workload_class": "batch",
            "starvation_risk": "high",
            "wait_ms": 1200,
            "deferred_count": 3,
            "blocked_count": 1,
            "reason": "selected",
        }
        record = task_record_from_decision(
            task_id="t1",
            task_decision=task_decision,
            scheduler_decision=scheduler_decision,
        )
        assert record.selected is True
        assert record.fairness_score == 0.75
        assert record.policy == "fairness_balanced"
        assert record.workload_class == "batch"
        assert record.starvation_risk == "high"
        assert record.metadata["wait_ms"] == 1200
        assert record.metadata["deferred_count"] == 3

    def test_task_record_from_decision_unselected_task(self):
        scheduler_decision = {
            "policy": "latency_first",
            "fairness": {"fairness_score": 0.9},
            "selected": ["t2"],
        }
        task_decision = {
            "task_id": "t1",
            "workload_class": "interactive",
            "starvation_risk": "low",
            "wait_ms": 100,
            "deferred_count": 0,
            "blocked_count": 0,
            "reason": "resource_batch_full",
        }
        record = task_record_from_decision(
            task_id="t1",
            task_decision=task_decision,
            scheduler_decision=scheduler_decision,
        )
        assert record.selected is False
        assert record.metadata["reason"] == "resource_batch_full"


# Tests for probe_sched_ext.py script
import subprocess


class TestProbeSchedExtScript:
    """Test probe_sched_ext.py script."""

    def test_probe_sched_ext_script_mock_mode(self, tmp_path):
        """Test script with --mock flag."""
        out = tmp_path / "probe.json"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/probe_sched_ext.py",
                "--mock",
                "--out",
                str(out),
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.returncode == 0
        payload = json.loads(out.read_text())
        assert payload["sched_ext"] == "unavailable"
        assert payload["bpf_map"] == "mock"
        assert payload["fallback_mode"] == "mock_bpf_map"
        assert payload["reason"] == "forced mock mode"


# ---------------------------------------------------------------------------
# Phase 4: BPFTaskRecord v2 fields, bpffs mount, PSI, CAP_BPF, force_mock
# ---------------------------------------------------------------------------


class TestBPFTaskRecordV2:
    """Test BPFTaskRecord v2 OS-facing fields."""

    def test_v2_fields_default(self):
        """New v2 fields have correct defaults."""
        record = BPFTaskRecord(
            task_id="t1",
            workload_class="batch",
            policy="latency_first",
            fairness_score=1.0,
            starvation_risk="low",
        )
        assert record.agent_id is None
        assert record.pid is None
        assert record.cpu_weight is None
        assert record.cpu_max_percent is None
        assert record.memory_max_mb is None
        assert record.pids_max is None
        assert record.pressure_level == "normal"
        assert record.backpressure_signal == "none"
        assert record.updated_at == 0.0
        assert record.schema_version == "bpf.task_record.v2"

    def test_v2_fields_in_to_dict(self):
        """to_dict() includes all v2 fields."""
        record = BPFTaskRecord(
            task_id="t1",
            workload_class="batch",
            policy="latency_first",
            fairness_score=1.0,
            starvation_risk="low",
            agent_id="agent-1",
            pid=1234,
            cpu_weight=1024,
            cpu_max_percent=80.0,
            memory_max_mb=512,
            pids_max=100,
            pressure_level="warning",
            backpressure_signal="throttle",
            updated_at=1234567890.0,
        )
        d = record.to_dict()
        assert d["agent_id"] == "agent-1"
        assert d["pid"] == 1234
        assert d["cpu_weight"] == 1024
        assert d["cpu_max_percent"] == 80.0
        assert d["memory_max_mb"] == 512
        assert d["pids_max"] == 100
        assert d["pressure_level"] == "warning"
        assert d["backpressure_signal"] == "throttle"
        assert d["updated_at"] == 1234567890.0
        assert d["schema_version"] == "bpf.task_record.v2"

    def test_v2_fields_json_serializable(self):
        """Full v2 record is JSON serializable."""
        record = BPFTaskRecord(
            task_id="t1",
            workload_class="batch",
            policy="latency_first",
            fairness_score=1.0,
            starvation_risk="low",
            agent_id="agent-1",
            pid=1234,
            cpu_weight=1024,
        )
        json_str = json.dumps(record.to_dict())
        parsed = json.loads(json_str)
        assert parsed["agent_id"] == "agent-1"
        assert parsed["pid"] == 1234
        assert parsed["schema_version"] == "bpf.task_record.v2"


class TestTaskRecordFromDecisionV2:
    """Test task_record_from_decision v2 field mapping."""

    def test_full_decision_maps_v2_fields(self):
        """Full decision with resource_requirements maps all v2 fields."""
        scheduler_decision = {
            "policy": "latency_first",
            "fairness": {"fairness_score": 0.9},
            "selected_task_ids": ["t1"],
        }
        task_decision = {
            "task_id": "t1",
            "workload_class": "interactive",
            "starvation_risk": "low",
            "wait_ms": 50,
            "deferred_count": 0,
            "blocked_count": 0,
            "reason": "selected",
            "agent_id": "agent-1",
            "pid": 42,
            "effective_pressure_level": "warning",
            "backpressure_signal": "throttle",
            "resource_requirements": {
                "cpu_shares": 1024,
                "cpu_max_percent": 75.0,
                "memory_max_mb": 256,
                "pids_max": 50,
            },
        }
        record = task_record_from_decision(
            task_id="t1",
            task_decision=task_decision,
            scheduler_decision=scheduler_decision,
        )
        assert record.agent_id == "agent-1"
        assert record.pid == 42
        assert record.cpu_weight == 1024
        assert record.cpu_max_percent == 75.0
        assert record.memory_max_mb == 256
        assert record.pids_max == 50
        assert record.pressure_level == "warning"
        assert record.backpressure_signal == "throttle"
        assert record.updated_at > 0
        assert record.schema_version == "bpf.task_record.v2"
        assert record.selected is True

    def test_missing_fields_use_defaults(self):
        """Missing v2 fields default to None/normal/none."""
        scheduler_decision = {
            "policy": "latency_first",
            "fairness": {"fairness_score": 1.0},
            "selected": ["t2"],
        }
        task_decision = {
            "task_id": "t1",
            "workload_class": "batch",
            "starvation_risk": "low",
        }
        record = task_record_from_decision(
            task_id="t1",
            task_decision=task_decision,
            scheduler_decision=scheduler_decision,
        )
        assert record.agent_id is None
        assert record.pid is None
        assert record.cpu_weight is None
        assert record.cpu_max_percent is None
        assert record.memory_max_mb is None
        assert record.pids_max is None
        assert record.pressure_level == "normal"
        assert record.backpressure_signal == "none"
        assert record.selected is False

    def test_old_pressure_level_compat(self):
        """Old pressure_level field is used when effective_pressure_level missing."""
        scheduler_decision = {"policy": "latency_first", "fairness": {}, "selected": []}
        task_decision = {
            "task_id": "t1",
            "workload_class": "batch",
            "pressure_level": "critical",
        }
        record = task_record_from_decision(
            task_id="t1",
            task_decision=task_decision,
            scheduler_decision=scheduler_decision,
        )
        assert record.pressure_level == "critical"

    def test_selected_task_ids_compat(self):
        """selected_task_ids is preferred over selected."""
        scheduler_decision = {
            "policy": "latency_first",
            "fairness": {},
            "selected_task_ids": ["t1"],
            "selected": ["t2"],
        }
        task_decision = {"task_id": "t1", "workload_class": "batch"}
        record = task_record_from_decision(
            task_id="t1",
            task_decision=task_decision,
            scheduler_decision=scheduler_decision,
        )
        assert record.selected is True

    def test_top_level_backpressure_fallback(self):
        """Top-level backpressure_signal is used when task-level missing."""
        scheduler_decision = {
            "policy": "latency_first",
            "fairness": {},
            "selected": [],
            "backpressure_signal": "shed",
        }
        task_decision = {"task_id": "t1", "workload_class": "batch"}
        record = task_record_from_decision(
            task_id="t1",
            task_decision=task_decision,
            scheduler_decision=scheduler_decision,
        )
        assert record.backpressure_signal == "shed"

    def test_mock_adapter_update_covers_v2(self):
        """MockBPFMapAdapter.update_task stores v2 fields."""
        adapter = MockBPFMapAdapter()
        record = BPFTaskRecord(
            task_id="t1",
            workload_class="batch",
            policy="latency_first",
            fairness_score=1.0,
            starvation_risk="low",
            agent_id="agent-1",
            pid=99,
            cpu_weight=512,
        )
        adapter.update_task(record)
        snapshot = adapter.snapshot()
        assert snapshot["t1"]["agent_id"] == "agent-1"
        assert snapshot["t1"]["pid"] == 99
        assert snapshot["t1"]["cpu_weight"] == 512


# ---------------------------------------------------------------------------
# Phase 4: bpffs, PSI, CAP_BPF, force_mock, sched_ext-still-mock
# ---------------------------------------------------------------------------


class TestBpffsMountCheck:
    """Test bpffs directory vs. mount detection."""

    def test_bpffs_directory_present_but_not_mounted(self, tmp_path):
        """bpffs directory exists but /proc/mounts has no bpf entry."""
        bpffs = tmp_path / "bpffs"
        bpffs.mkdir()
        mounts = tmp_path / "mounts"
        mounts.write_text("proc /proc proc rw 0 0\n")

        result = probe_sched_ext(
            sys_kernel_path=tmp_path / "kernel",
            sys_fs_bpf_path=bpffs,
            proc_mounts_path=mounts,
            proc_pressure_path=tmp_path / "pressure",
            proc_status_path=tmp_path / "status",
            cgroup_path=tmp_path / "cgroup",
        )
        assert result.details["bpffs_present"] is True
        assert result.details["bpffs_mounted"] is False

    def test_bpffs_correctly_mounted(self, tmp_path):
        """bpffs directory exists and /proc/mounts shows bpf fstype."""
        bpffs = tmp_path / "bpffs"
        bpffs.mkdir()
        mounts = tmp_path / "mounts"
        mounts.write_text(f"bpf {bpffs} bpf rw,nosuid,nodev,noexec 0 0\n")

        result = probe_sched_ext(
            sys_kernel_path=tmp_path / "kernel",
            sys_fs_bpf_path=bpffs,
            proc_mounts_path=mounts,
            proc_pressure_path=tmp_path / "pressure",
            proc_status_path=tmp_path / "status",
            cgroup_path=tmp_path / "cgroup",
        )
        assert result.details["bpffs_present"] is True
        assert result.details["bpffs_mounted"] is True


class TestPSIFiles:
    """Test PSI file detection."""

    def test_psi_all_three_files_present(self, tmp_path):
        """All three PSI files present → psi_available True."""
        psi = tmp_path / "pressure"
        psi.mkdir()
        (psi / "cpu").write_text("some avg10=1.00 avg60=0.50 avg300=0.25 total=100\n")
        (psi / "memory").write_text("some avg10=0.50 avg60=0.25 avg300=0.10 total=50\n")
        (psi / "io").write_text("some avg10=0.10 avg60=0.05 avg300=0.01 total=10\n")

        result = probe_sched_ext(
            sys_kernel_path=tmp_path / "kernel",
            sys_fs_bpf_path=tmp_path / "bpffs",
            proc_pressure_path=psi,
            proc_mounts_path=tmp_path / "mounts",
            proc_status_path=tmp_path / "status",
            cgroup_path=tmp_path / "cgroup",
        )
        assert result.details["psi_available"] is True
        assert result.details["psi_files"]["cpu"]["present"] is True
        assert result.details["psi_files"]["memory"]["present"] is True
        assert result.details["psi_files"]["io"]["present"] is True

    def test_psi_missing_files(self, tmp_path):
        """PSI directory exists but files missing → psi_available False."""
        psi = tmp_path / "pressure"
        psi.mkdir()
        # Only cpu present
        (psi / "cpu").write_text("some avg10=1.00\n")

        result = probe_sched_ext(
            sys_kernel_path=tmp_path / "kernel",
            sys_fs_bpf_path=tmp_path / "bpffs",
            proc_pressure_path=psi,
            proc_mounts_path=tmp_path / "mounts",
            proc_status_path=tmp_path / "status",
            cgroup_path=tmp_path / "cgroup",
        )
        assert result.details["psi_available"] is False
        assert result.details["psi_files"]["cpu"]["present"] is True
        assert result.details["psi_files"]["memory"]["present"] is False
        assert result.details["psi_files"]["io"]["present"] is False


class TestCapBpf:
    """Test CAP_BPF bit 39 detection."""

    def test_cap_bpf_present(self, tmp_path):
        """CapEff with bit 39 set → cap_bpf True."""
        status = tmp_path / "status"
        # bit 39 = 0x8000000000
        status.write_text("Name: test\nCapEff: 8000000000\n")

        result = probe_sched_ext(
            sys_kernel_path=tmp_path / "kernel",
            sys_fs_bpf_path=tmp_path / "bpffs",
            proc_pressure_path=tmp_path / "pressure",
            proc_mounts_path=tmp_path / "mounts",
            proc_status_path=status,
            cgroup_path=tmp_path / "cgroup",
        )
        assert result.details["cap_bpf"] is True

    def test_cap_bpf_absent(self, tmp_path):
        """CapEff without bit 39 → cap_bpf False."""
        status = tmp_path / "status"
        # 0x000000003ffffeff = all caps except bit 39
        status.write_text("Name: test\nCapEff: 000000003ffffeff\n")

        result = probe_sched_ext(
            sys_kernel_path=tmp_path / "kernel",
            sys_fs_bpf_path=tmp_path / "bpffs",
            proc_pressure_path=tmp_path / "pressure",
            proc_mounts_path=tmp_path / "mounts",
            proc_status_path=status,
            cgroup_path=tmp_path / "cgroup",
        )
        assert result.details["cap_bpf"] is False

    def test_cap_bpf_read_failure(self, tmp_path):
        """Missing status file → cap_bpf False with reason."""
        result = probe_sched_ext(
            sys_kernel_path=tmp_path / "kernel",
            sys_fs_bpf_path=tmp_path / "bpffs",
            proc_pressure_path=tmp_path / "pressure",
            proc_mounts_path=tmp_path / "mounts",
            proc_status_path=tmp_path / "nonexistent",
            cgroup_path=tmp_path / "cgroup",
        )
        assert result.details["cap_bpf"] is False
        assert "cap_bpf_reason" in result.details


class TestForceMock:
    """Test force_mock skips subprocess."""

    def test_force_mock_returns_stable_result(self):
        """force_mock=True returns stable mock without subprocess."""
        result = probe_sched_ext(force_mock=True)
        assert result.sched_ext == "unavailable"
        assert result.bpf_map == "mock"
        assert result.fallback_mode == "mock_bpf_map"
        assert result.reason == "forced mock mode"
        assert result.details == {}

    def test_force_mock_skips_bpftool_subprocess(self):
        """force_mock=True never calls probe_bpftool."""
        with patch("agentos.isolation.sched_ext.probe_bpftool") as mock_bpftool:
            probe_sched_ext(force_mock=True)
        mock_bpftool.assert_not_called()


class TestSchedExtPresentStillMock:
    """Test that sched_ext present still returns mock bpf_map."""

    def test_sched_ext_present_returns_mock(self, tmp_path):
        """sched_ext directory exists → sched_ext available but bpf_map mock."""
        kernel = tmp_path / "kernel"
        kernel.mkdir()
        (kernel / "sched_ext").mkdir()

        result = probe_sched_ext(
            sys_kernel_path=kernel,
            sys_fs_bpf_path=tmp_path / "bpffs",
            proc_pressure_path=tmp_path / "pressure",
            proc_mounts_path=tmp_path / "mounts",
            proc_status_path=tmp_path / "status",
            cgroup_path=tmp_path / "cgroup",
        )
        assert result.sched_ext == "available"
        assert result.bpf_map == "mock"
        assert "prerequisites incomplete" in result.reason
        assert "prerequisites available" not in result.reason

    def test_sched_ext_present_but_prerequisites_incomplete(self, tmp_path):
        """sched_ext exists but bpftool/CAP_BPF/bpffs missing → prerequisites incomplete."""
        kernel = tmp_path / "kernel"
        kernel.mkdir()
        (kernel / "sched_ext").mkdir()

        # Create status file without CAP_BPF
        status = tmp_path / "status"
        status.write_text("Name: test\nCapEff: 000000003ffffeff\n")

        result = probe_sched_ext(
            sys_kernel_path=kernel,
            sys_fs_bpf_path=tmp_path / "bpffs",  # not mounted
            proc_pressure_path=tmp_path / "pressure",
            proc_mounts_path=tmp_path / "mounts",
            proc_status_path=status,
            cgroup_path=tmp_path / "cgroup",
        )
        assert result.sched_ext == "available"
        assert result.bpf_map == "mock"
        assert "prerequisites incomplete" in result.reason
        assert "prerequisites available" not in result.reason
        assert "CAP_BPF missing" in result.reason


class TestCliJsonOnly:
    """Test CLI --json-only flag."""

    def test_json_only_output_is_valid_json(self, tmp_path):
        """--json-only output can be json.loads()'d."""
        result = subprocess.run(
            [sys.executable, "scripts/probe_sched_ext.py", "--json-only"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.returncode == 0
        # Output should be pure JSON (no summary text)
        payload = json.loads(result.stdout)
        assert "kernel_version" in payload or "sched_ext" in payload

    def test_mock_json_only_output_is_valid_json(self, tmp_path):
        """--mock --json-only output is valid JSON."""
        result = subprocess.run(
            [sys.executable, "scripts/probe_sched_ext.py", "--mock", "--json-only"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert payload["bpf_map"] == "mock"


class TestNoKernelTakeoverClaims:
    """Test output doesn't claim kernel takeover."""

    def test_probe_result_no_takeover_language(self):
        """probe_sched_ext result doesn't contain takeover claims."""
        result = probe_sched_ext(force_mock=True)
        result_str = json.dumps(result.to_dict()).lower()
        assert "native scheduler enabled" not in result_str
        assert "kernel takeover" not in result_str
        assert "eBPF scheduling active" not in result_str

    def test_cli_output_no_takeover_language(self):
        """CLI output doesn't contain takeover claims."""
        result = subprocess.run(
            [sys.executable, "scripts/probe_sched_ext.py", "--json-only"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        output_lower = result.stdout.lower()
        assert "native scheduler enabled" not in output_lower
        assert "kernel takeover" not in output_lower
        assert "eBPF scheduling active" not in output_lower
