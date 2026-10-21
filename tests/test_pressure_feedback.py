"""Tests for PSI parsing, pressure feedback, and policy integration.

Covers:
- Task 1: PSI parser (parse_psi, read_psi_avg10, pressure_level_from_psi, max_pressure_level)
- Task 2: Adapter PSI integration (sample_pressure with real/injected PSI files)
- Task 3: Scheduler set_pressure and decision.pressure field
- Task 4: PolicySelector pressure-driven policy changes
- Task 5: Runtime feedback loop (resource.pressure_sampled events)
"""

from __future__ import annotations

import pytest
from pathlib import Path

from agentos.core.resources import ResourceQuota, ResourceUsage
from agentos.core.task import Task, TaskDAG
from agentos.isolation.bpf import MockBPFMapAdapter
from agentos.isolation.cgroup import CgroupV2Manager
from agentos.isolation.kernel_adapter import PressureSample, UserSpaceKernelAdapter
from agentos.isolation.pressure import (
    max_pressure_level,
    parse_psi,
    pressure_level_from_psi,
    read_psi_avg10,
    read_psi_avg10_with_status,
)
from agentos.isolation.sched_ext import KernelProbeResult
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import AgentHandler, Runtime
from agentos.scheduler.dag import DAGScheduler
from agentos.scheduler.policy import PolicySelector, SchedulingPolicy, StarvationRisk
from agentos.scheduler.workload import WorkloadClass, WorkloadSample


# ===================================================================
# Task 1: PSI parser tests
# ===================================================================


class TestParsePSI:
    """Tests for parse_psi()."""

    def test_parse_cpu_psi_normal(self):
        text = "some avg10=1.23 avg60=4.56 avg300=7.89 total=123456\n"
        result = parse_psi(text)
        assert "some" in result
        assert result["some"]["avg10"] == 1.23
        assert result["some"]["avg60"] == 4.56
        assert result["some"]["avg300"] == 7.89
        assert result["some"]["total"] == 123456

    def test_parse_memory_psi_with_full_line(self):
        text = (
            "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
            "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
        )
        result = parse_psi(text)
        assert "some" in result
        assert "full" in result
        assert result["some"]["avg10"] == 0.0
        assert result["full"]["avg10"] == 0.0

    def test_parse_io_psi_integer_total(self):
        text = "some avg10=5.50 avg60=3.20 avg300=1.10 total=987654321\n"
        result = parse_psi(text)
        assert result["some"]["total"] == 987654321

    def test_parse_only_some_line(self):
        """CPU PSI only has 'some' line, no 'full'."""
        text = "some avg10=21.0 avg60=10.0 avg300=5.0 total=50000\n"
        result = parse_psi(text)
        assert "some" in result
        assert "full" not in result
        assert result["some"]["avg10"] == 21.0

    def test_parse_empty_string_returns_empty(self):
        assert parse_psi("") == {}

    def test_parse_blank_lines_only(self):
        assert parse_psi("\n\n\n") == {}

    def test_parse_unknown_fields_skipped(self):
        text = "unknown avg10=99.0\nsome avg10=5.0 avg60=3.0 avg300=1.0 total=100\n"
        result = parse_psi(text)
        assert "unknown" not in result
        assert result["some"]["avg10"] == 5.0

    def test_parse_malformed_numeric_values_skipped(self):
        text = "some avg10=abc avg60=1.0 avg300=2.0 total=3\n"
        result = parse_psi(text)
        # avg10 should be skipped, others present
        assert "avg10" not in result.get("some", {})
        assert result["some"]["avg60"] == 1.0

    def test_parse_corrupt_content_no_exception(self):
        assert parse_psi("not a psi file at all") == {}
        assert parse_psi("!!!@@@###") == {}
        assert parse_psi("some\nsome\n") == {}

    def test_parse_token_without_equals_skipped(self):
        text = "some avg10=1.0 junk_token avg60=2.0 avg300=3.0 total=4\n"
        result = parse_psi(text)
        assert result["some"]["avg10"] == 1.0
        assert result["some"]["avg60"] == 2.0


class TestReadPSIAvg10:
    """Tests for read_psi_avg10()."""

    def test_reads_normal_file(self, tmp_path: Path):
        psi_file = tmp_path / "cpu"
        psi_file.write_text(
            "some avg10=15.50 avg60=10.0 avg300=5.0 total=12345\n",
            encoding="utf-8",
        )
        assert read_psi_avg10(psi_file) == 15.50

    def test_missing_file_returns_zero(self, tmp_path: Path):
        assert read_psi_avg10(tmp_path / "nonexistent") == 0.0

    def test_permission_denied_returns_zero(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        psi_file = tmp_path / "cpu"
        psi_file.write_text("some avg10=10.0\n", encoding="utf-8")

        original_read_text = Path.read_text

        def deny_psi_read(path: Path, *args, **kwargs):
            if path == psi_file:
                raise PermissionError("permission denied")
            return original_read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", deny_psi_read)
        assert read_psi_avg10(psi_file) == 0.0

    def test_corrupt_content_returns_zero(self, tmp_path: Path):
        psi_file = tmp_path / "cpu"
        psi_file.write_text("garbage content\n", encoding="utf-8")
        assert read_psi_avg10(psi_file) == 0.0

    def test_empty_file_returns_zero(self, tmp_path: Path):
        psi_file = tmp_path / "cpu"
        psi_file.write_text("", encoding="utf-8")
        assert read_psi_avg10(psi_file) == 0.0

    def test_path_as_string(self, tmp_path: Path):
        psi_file = tmp_path / "cpu"
        psi_file.write_text(
            "some avg10=7.77 avg60=5.0 avg300=2.0 total=100\n",
            encoding="utf-8",
        )
        assert read_psi_avg10(str(psi_file)) == 7.77


class TestPressureLevelFromPSI:
    """Tests for pressure_level_from_psi() with four thresholds."""

    def test_normal_when_all_below_10(self):
        assert pressure_level_from_psi(5.0, 3.0, 1.0) == "normal"

    def test_normal_when_all_zero(self):
        assert pressure_level_from_psi(0.0, 0.0, 0.0) == "normal"

    def test_warning_when_cpu_at_10(self):
        assert pressure_level_from_psi(10.0, 0.0, 0.0) == "warning"

    def test_warning_when_memory_at_15(self):
        assert pressure_level_from_psi(0.0, 15.0, 0.0) == "warning"

    def test_warning_when_io_at_19(self):
        assert pressure_level_from_psi(0.0, 0.0, 19.0) == "warning"

    def test_critical_when_cpu_at_20(self):
        assert pressure_level_from_psi(20.0, 0.0, 0.0) == "critical"

    def test_critical_when_memory_at_30(self):
        assert pressure_level_from_psi(0.0, 30.0, 0.0) == "critical"

    def test_critical_when_io_at_49(self):
        assert pressure_level_from_psi(0.0, 0.0, 49.0) == "critical"

    def test_overloaded_when_cpu_at_50(self):
        assert pressure_level_from_psi(50.0, 0.0, 0.0) == "overloaded"

    def test_overloaded_when_io_at_100(self):
        assert pressure_level_from_psi(0.0, 0.0, 100.0) == "overloaded"

    def test_peak_determines_level(self):
        """Level is determined by the maximum of the three."""
        assert pressure_level_from_psi(5.0, 25.0, 3.0) == "critical"
        assert pressure_level_from_psi(5.0, 3.0, 55.0) == "overloaded"


class TestMaxPressureLevel:
    """Tests for max_pressure_level()."""

    def test_single_normal(self):
        assert max_pressure_level("normal") == "normal"

    def test_single_warning(self):
        assert max_pressure_level("warning") == "warning"

    def test_single_critical(self):
        assert max_pressure_level("critical") == "critical"

    def test_single_overloaded(self):
        assert max_pressure_level("overloaded") == "overloaded"

    def test_normal_and_warning_returns_warning(self):
        assert max_pressure_level("normal", "warning") == "warning"

    def test_warning_and_critical_returns_critical(self):
        assert max_pressure_level("warning", "critical") == "critical"

    def test_critical_and_overloaded_returns_overloaded(self):
        assert max_pressure_level("critical", "overloaded") == "overloaded"

    def test_order_independent(self):
        assert max_pressure_level("overloaded", "normal", "warning") == "overloaded"
        assert max_pressure_level("normal", "overloaded", "warning") == "overloaded"
        assert max_pressure_level("warning", "normal", "overloaded") == "overloaded"

    def test_multiple_same_level(self):
        assert max_pressure_level("critical", "critical", "critical") == "critical"

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError, match="invalid pressure level"):
            max_pressure_level("normal", "bogus")

    def test_empty_args_returns_normal(self):
        assert max_pressure_level() == "normal"


# ===================================================================
# Task 2: Adapter PSI integration tests
# ===================================================================


def _make_adapter(
    tmp_path: Path,
    *,
    psi_paths: dict[str, str | Path] | None = None,
    cgroup_manager: CgroupV2Manager | None = None,
) -> UserSpaceKernelAdapter:
    """Helper to create an adapter with fake cgroup root."""
    cm = cgroup_manager or CgroupV2Manager(
        root=tmp_path / "cg", enable_controllers=False
    )
    return UserSpaceKernelAdapter(
        cgroup_manager=cm,
        bpf_map_adapter=MockBPFMapAdapter(),
        probe_result=KernelProbeResult(
            sched_ext="unavailable",
            bpf_map="mock",
            fallback_mode="mock_bpf_map",
            reason="unit test",
        ),
        psi_paths=psi_paths,
    )


def _write_psi_files(directory: Path, *, cpu: float = 0, memory: float = 0, io: float = 0) -> dict[str, Path]:
    """Write fake PSI files and return path dict."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in [("cpu", cpu), ("memory", memory), ("io", io)]:
        f = directory / name
        f.write_text(
            f"some avg10={value} avg60={value / 2} avg300={value / 4} total=0\n",
            encoding="utf-8",
        )
    return {
        "cpu": directory / "cpu",
        "memory": directory / "memory",
        "io": directory / "io",
    }


class TestAdapterPSI:
    """Tests for UserSpaceKernelAdapter PSI integration."""

    def test_cgroup_psi_source_uses_all_local_files(self, tmp_path: Path):
        system_paths = _write_psi_files(
            tmp_path / "proc" / "pressure", cpu=8.0, memory=9.0, io=10.0
        )
        cgroup_path = tmp_path / "cg" / "agent-1"
        cgroup_path.mkdir(parents=True)
        for name, value in (("cpu", 1.0), ("memory", 2.0), ("io", 3.0)):
            (cgroup_path / f"{name}.pressure").write_text(
                f"some avg10={value} avg60=0.0 avg300=0.0 total=0\n",
                encoding="utf-8",
            )
        adapter = _make_adapter(tmp_path, psi_paths=system_paths)

        sample = adapter.sample_pressure(cgroup_path=cgroup_path)

        assert sample.pressure_source == "cgroup"
        assert sample.pressure_sources == {
            "cpu": "cgroup",
            "memory": "cgroup",
            "io": "cgroup",
        }
        assert sample.cpu_some_avg10 == 1.0
        assert sample.memory_some_avg10 == 2.0
        assert sample.io_some_avg10 == 3.0
        assert sample.reason is None

    def test_cgroup_psi_source_falls_back_per_resource(self, tmp_path: Path):
        system_paths = _write_psi_files(
            tmp_path / "proc" / "pressure", cpu=8.0, memory=9.0, io=10.0
        )
        cgroup_path = tmp_path / "cg" / "agent-1"
        cgroup_path.mkdir(parents=True)
        for name, value in (("cpu", 1.0), ("io", 3.0)):
            (cgroup_path / f"{name}.pressure").write_text(
                f"some avg10={value} avg60=0.0 avg300=0.0 total=0\n",
                encoding="utf-8",
            )
        adapter = _make_adapter(tmp_path, psi_paths=system_paths)

        sample = adapter.sample_pressure(cgroup_path=cgroup_path)

        assert sample.pressure_source == "mixed"
        assert sample.pressure_sources == {
            "cpu": "cgroup",
            "memory": "system",
            "io": "cgroup",
        }
        assert sample.memory_some_avg10 == 9.0
        assert "memory" in (sample.reason or "")

    def test_cgroup_psi_source_reports_all_sources_unavailable(
        self, tmp_path: Path,
    ):
        missing = {
            name: tmp_path / "missing-system" / name
            for name in ("cpu", "memory", "io")
        }
        cgroup_path = tmp_path / "cg" / "agent-1"
        cgroup_path.mkdir(parents=True)
        adapter = _make_adapter(tmp_path, psi_paths=missing)

        sample = adapter.sample_pressure(cgroup_path=cgroup_path)

        assert sample.pressure_source == "system"
        assert sample.pressure_sources == {
            "cpu": "unavailable",
            "memory": "unavailable",
            "io": "unavailable",
        }
        assert sample.pressure_level == "normal"
        assert sample.reason is not None

    def test_cgroup_psi_source_preserves_usage_when_local_reader_is_absent(
        self, tmp_path: Path,
    ):
        class UsageOnlyCgroupManager:
            def read_usage(self, path):
                return ResourceUsage(
                    cpu_percent=7.0,
                    memory_peak_mb=42.2,
                    pids_current=2,
                )

        system_paths = _write_psi_files(
            tmp_path / "proc" / "pressure", cpu=1.0, memory=2.0, io=3.0
        )
        adapter = _make_adapter(
            tmp_path,
            psi_paths=system_paths,
            cgroup_manager=UsageOnlyCgroupManager(),
        )

        sample = adapter.sample_pressure(cgroup_path=tmp_path / "agent-1")

        assert sample.cgroup_usage == ResourceUsage(
            cpu_percent=7.0,
            memory_peak_mb=42.2,
            pids_current=2,
        )
        assert sample.pressure_source == "system"
        assert set(sample.pressure_sources.values()) == {"system"}

    def test_sample_pressure_reads_psi_files(self, tmp_path: Path):
        psi_dir = tmp_path / "proc" / "pressure"
        _write_psi_files(psi_dir, cpu=21.0, memory=5.0, io=2.0)

        adapter = _make_adapter(tmp_path, psi_paths={
            "cpu": psi_dir / "cpu",
            "memory": psi_dir / "memory",
            "io": psi_dir / "io",
        })

        sample = adapter.sample_pressure()
        assert sample.cpu_some_avg10 == 21.0
        assert sample.memory_some_avg10 == 5.0
        assert sample.io_some_avg10 == 2.0
        assert sample.pressure_level == "critical"

    def test_sample_pressure_cpu_only_some_line(self, tmp_path: Path):
        """CPU PSI typically only has 'some' line."""
        psi_dir = tmp_path / "proc" / "pressure"
        psi_dir.mkdir(parents=True, exist_ok=True)
        (psi_dir / "cpu").write_text(
            "some avg10=55.0 avg60=30.0 avg300=10.0 total=999\n",
            encoding="utf-8",
        )
        (psi_dir / "memory").write_text("", encoding="utf-8")
        (psi_dir / "io").write_text("", encoding="utf-8")

        adapter = _make_adapter(tmp_path, psi_paths={
            "cpu": psi_dir / "cpu",
            "memory": psi_dir / "memory",
            "io": psi_dir / "io",
        })

        sample = adapter.sample_pressure()
        assert sample.cpu_some_avg10 == 55.0
        assert sample.pressure_level == "overloaded"

    def test_sample_pressure_missing_psi_degrades_to_normal(self, tmp_path: Path):
        """Missing PSI files degrade to normal without raising."""
        adapter = _make_adapter(tmp_path, psi_paths={
            "cpu": tmp_path / "missing_cpu",
            "memory": tmp_path / "missing_mem",
            "io": tmp_path / "missing_io",
        })

        sample = adapter.sample_pressure()
        assert sample.pressure_level == "normal"
        assert sample.cpu_some_avg10 == 0.0
        assert sample.reason is not None
        assert "psi" in sample.reason.lower() or "missing" in sample.reason.lower()

    def test_sample_pressure_without_overrides_reads_default_system_paths(self, tmp_path: Path, monkeypatch):
        """Default configuration samples the host; simulate its readings deterministically."""
        paths = []

        def read(path):
            paths.append(str(path))
            return (55.0 if str(path) == "/proc/pressure/cpu" else 0.0, True)

        monkeypatch.setattr("agentos.isolation.kernel_adapter.read_psi_avg10_with_status", read)
        adapter = _make_adapter(tmp_path)
        sample = adapter.sample_pressure()
        assert set(paths) == {"/proc/pressure/cpu", "/proc/pressure/memory", "/proc/pressure/io"}
        assert sample.cpu_some_avg10 == 55.0
        assert sample.pressure_level == "overloaded"

    def test_sample_pressure_with_cgroup_usage(self, tmp_path: Path):
        """PSI sampling still includes cgroup usage."""
        psi_dir = tmp_path / "proc" / "pressure"
        _write_psi_files(psi_dir, cpu=5.0, memory=3.0, io=1.0)

        adapter = _make_adapter(tmp_path, psi_paths={
            "cpu": psi_dir / "cpu",
            "memory": psi_dir / "memory",
            "io": psi_dir / "io",
        })

        # Create a cgroup with some usage data
        cgroup_path = tmp_path / "cg" / "agent-1"
        cgroup_path.mkdir(parents=True, exist_ok=True)
        (cgroup_path / "memory.current").write_text(str(32 * 1024 * 1024), encoding="utf-8")
        (cgroup_path / "memory.peak").write_text(str(64 * 1024 * 1024), encoding="utf-8")
        (cgroup_path / "pids.current").write_text("2", encoding="utf-8")
        (cgroup_path / "cpu.stat").write_text("usage_usec 5000\n", encoding="utf-8")

        sample = adapter.sample_pressure(cgroup_path=cgroup_path)
        assert sample.cgroup_usage is not None
        assert sample.cgroup_usage.memory_rss_mb == 32.0
        assert sample.pressure_level == "normal"

    def test_sample_pressure_corrupt_psi_returns_normal(self, tmp_path: Path):
        psi_dir = tmp_path / "proc" / "pressure"
        psi_dir.mkdir(parents=True, exist_ok=True)
        (psi_dir / "cpu").write_text("CORRUPT", encoding="utf-8")
        (psi_dir / "memory").write_text("CORRUPT", encoding="utf-8")
        (psi_dir / "io").write_text("CORRUPT", encoding="utf-8")

        adapter = _make_adapter(tmp_path, psi_paths={
            "cpu": psi_dir / "cpu",
            "memory": psi_dir / "memory",
            "io": psi_dir / "io",
        })

        sample = adapter.sample_pressure()
        assert sample.pressure_level == "normal"
        assert sample.cpu_some_avg10 == 0.0

    def test_snapshot_includes_psi_paths(self, tmp_path: Path):
        psi_dir = tmp_path / "proc" / "pressure"
        _write_psi_files(psi_dir, cpu=0, memory=0, io=0)

        adapter = _make_adapter(tmp_path, psi_paths={
            "cpu": psi_dir / "cpu",
            "memory": psi_dir / "memory",
            "io": psi_dir / "io",
        })

        snapshot = adapter.snapshot()
        assert "psi_paths" in snapshot
        assert "cpu" in snapshot["psi_paths"]

    def test_psi_paths_partial_merge_with_defaults(self, tmp_path: Path):
        """Passing partial psi_paths should merge with defaults, not crash."""
        psi_dir = tmp_path / "proc" / "pressure"
        _write_psi_files(psi_dir, cpu=12.0, memory=0, io=0)

        # Only pass cpu path; memory/io use our injected zero files
        adapter = _make_adapter(tmp_path, psi_paths={
            "cpu": psi_dir / "cpu",
            "memory": psi_dir / "memory",
            "io": psi_dir / "io",
        })

        sample = adapter.sample_pressure()
        assert sample.cpu_some_avg10 == 12.0
        assert sample.memory_some_avg10 == 0.0
        assert sample.io_some_avg10 == 0.0


# ===================================================================
# Task 3: Scheduler pressure tests
# ===================================================================


class TestSchedulerPressure:
    """Tests for DAGScheduler.set_pressure() and decision.pressure field."""

    async def test_set_pressure_normal_by_default(self):
        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

        scheduler = DAGScheduler(dag)
        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        assert "pressure" in decision
        pressure = decision["pressure"]
        assert pressure["os_pressure_level"] == "normal"
        assert pressure["fairness_pressure_level"] == "normal"
        assert pressure["effective_pressure_level"] == "normal"

    async def test_set_pressure_updates_os_level(self):
        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

        scheduler = DAGScheduler(dag)
        scheduler.set_pressure(pressure_level="critical")
        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        pressure = decision["pressure"]
        assert pressure["os_pressure_level"] == "critical"

    async def test_set_pressure_invalid_level_raises(self):
        dag = TaskDAG()
        scheduler = DAGScheduler(dag)

        with pytest.raises(ValueError):
            scheduler.set_pressure(pressure_level="bogus")

    async def test_set_pressure_with_sample_data(self):
        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

        scheduler = DAGScheduler(dag)
        sample = {
            "cpu_some_avg10": 21.0,
            "memory_some_avg10": 5.0,
            "io_some_avg10": 2.0,
        }
        scheduler.set_pressure(pressure_level="critical", pressure_sample=sample)
        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        pressure = decision["pressure"]
        assert pressure["cpu_some_avg10"] == 21.0
        assert pressure["memory_some_avg10"] == 5.0
        assert pressure["io_some_avg10"] == 2.0

    async def test_fairness_pressure_level_mapping(self):
        """fairness_pressure_level: low→normal, medium→warning, high→critical."""
        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

        scheduler = DAGScheduler(dag)
        scheduler.set_pressure(pressure_level="normal")
        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        # With default fairness (low), fairness_pressure_level should be normal
        pressure = decision["pressure"]
        assert pressure["fairness_pressure_level"] == "normal"

    async def test_effective_pressure_is_max_of_os_and_fairness(self):
        """effective_pressure_level is the max of OS and fairness levels."""
        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

        scheduler = DAGScheduler(dag)
        scheduler.set_pressure(pressure_level="warning")
        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        pressure = decision["pressure"]
        # OS is warning, fairness is normal → effective is warning
        assert pressure["effective_pressure_level"] == "warning"

    async def test_decision_pressure_has_runtime_health_pressure_level(self):
        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

        scheduler = DAGScheduler(dag)
        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        pressure = decision["pressure"]
        assert "runtime_health_pressure_level" in pressure
        # runtime_health.pressure_level is independent from PSI
        assert pressure["runtime_health_pressure_level"] in (
            "normal", "warning", "critical", "overloaded"
        )

    async def test_runtime_health_field_preserved(self):
        """runtime_health field must still exist and be compatible."""
        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

        scheduler = DAGScheduler(dag)
        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        assert "runtime_health" in decision
        assert "pressure_level" in decision["runtime_health"]


# ===================================================================
# Task 4: Policy pressure effect tests
# ===================================================================


class TestPolicyPressureEffect:
    """Tests that pressure alone can change the scheduling policy."""

    def test_critical_os_pressure_selects_risk_aware(self):
        """critical OS pressure → risk_aware, all else equal."""
        selector = PolicySelector()
        sample = WorkloadSample(
            task_id="t1",
            agent_role="worker",
            workload_hint="batch",
            priority=0,
        )
        decision = selector.select(
            [sample],
            {"t1": WorkloadClass.BATCH},
            starvation_risk=StarvationRisk.LOW,
            has_critical_sla=False,
            pressure_level="critical",
        )
        assert decision.policy == SchedulingPolicy.RISK_AWARE
        assert "pressure" in decision.score_breakdown.get("pressure_level", "") or \
               decision.score_breakdown.get("pressure_level") == "critical"

    def test_overloaded_os_pressure_selects_risk_aware(self):
        """overloaded OS pressure → risk_aware."""
        selector = PolicySelector()
        sample = WorkloadSample(
            task_id="t1",
            agent_role="worker",
            workload_hint="batch",
            priority=0,
        )
        decision = selector.select(
            [sample],
            {"t1": WorkloadClass.BATCH},
            starvation_risk=StarvationRisk.LOW,
            has_critical_sla=False,
            pressure_level="overloaded",
        )
        assert decision.policy == SchedulingPolicy.RISK_AWARE

    def test_warning_os_pressure_selects_fairness_balanced(self):
        """warning OS pressure → fairness_balanced."""
        selector = PolicySelector()
        sample = WorkloadSample(
            task_id="t1",
            agent_role="worker",
            workload_hint="batch",
            priority=0,
        )
        decision = selector.select(
            [sample],
            {"t1": WorkloadClass.BATCH},
            starvation_risk=StarvationRisk.LOW,
            has_critical_sla=False,
            pressure_level="warning",
        )
        assert decision.policy == SchedulingPolicy.FAIRNESS_BALANCED

    def test_normal_pressure_keeps_workload_policy(self):
        """normal pressure → workload-based policy unchanged."""
        selector = PolicySelector()
        sample = WorkloadSample(
            task_id="t1",
            agent_role="worker",
            workload_hint="interactive",
            priority=8,
        )
        decision = selector.select(
            [sample],
            {"t1": WorkloadClass.INTERACTIVE},
            starvation_risk=StarvationRisk.LOW,
            has_critical_sla=False,
            pressure_level="normal",
        )
        assert decision.policy == SchedulingPolicy.LATENCY_FIRST

    def test_pressure_only_change_drives_policy(self):
        """Only pressure changes, other inputs identical → policy changes."""
        selector = PolicySelector()
        sample = WorkloadSample(
            task_id="t1",
            agent_role="worker",
            workload_hint="interactive",
            priority=5,
        )
        classes = {"t1": WorkloadClass.INTERACTIVE}
        common = dict(
            starvation_risk=StarvationRisk.LOW,
            has_critical_sla=False,
        )

        normal_decision = selector.select([sample], classes, pressure_level="normal", **common)
        critical_decision = selector.select([sample], classes, pressure_level="critical", **common)

        assert normal_decision.policy != critical_decision.policy
        assert normal_decision.policy == SchedulingPolicy.LATENCY_FIRST
        assert critical_decision.policy == SchedulingPolicy.RISK_AWARE

    def test_critical_sla_still_prioritizes_risk_aware(self):
        """critical SLA still → risk_aware even with normal pressure."""
        selector = PolicySelector()
        sample = WorkloadSample(
            task_id="t1",
            agent_role="worker",
            workload_hint="batch",
            priority=0,
        )
        decision = selector.select(
            [sample],
            {"t1": WorkloadClass.BATCH},
            starvation_risk=StarvationRisk.LOW,
            has_critical_sla=True,
            pressure_level="normal",
        )
        assert decision.policy == SchedulingPolicy.RISK_AWARE

    def test_high_starvation_normal_os_pressure_gives_fairness_balanced(self):
        """High starvation with normal OS pressure → fairness_balanced."""
        selector = PolicySelector()
        sample = WorkloadSample(
            task_id="t1",
            agent_role="worker",
            workload_hint="batch",
            priority=0,
        )
        decision = selector.select(
            [sample],
            {"t1": WorkloadClass.BATCH},
            starvation_risk=StarvationRisk.HIGH,
            has_critical_sla=False,
            pressure_level="normal",
        )
        assert decision.policy == SchedulingPolicy.FAIRNESS_BALANCED

    @pytest.mark.asyncio
    async def test_force_policy_overrides_pressure(self):
        """force_policy always wins, but score_breakdown records pressure."""
        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

        scheduler = DAGScheduler(dag, force_policy="latency_first")
        scheduler.set_pressure(pressure_level="critical")
        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        assert decision["policy"] == "latency_first"
        assert decision["score_breakdown"]["pressure_level"] == "critical"


# ===================================================================
# Task 5: Runtime feedback loop tests
# ===================================================================


class TestRuntimePressureFeedback:
    """Tests for Runtime pressure sampling and event emission."""

    async def test_runtime_emits_system_pressure_sampled_event(self, tmp_path: Path):
        """Runtime.run() emits resource.pressure_sampled with scope=system."""
        dag = TaskDAG()
        dag.add_task(
            Task(
                task_id="t1",
                name="work",
                agent_role="echo",
                metadata={
                    "execution": "subprocess",
                    "worker_type": "echo",
                    "worker_payload": {"result": "ok"},
                },
            )
        )

        adapter = UserSpaceKernelAdapter(
            cgroup_manager=CgroupV2Manager(root=tmp_path / "cg", enable_controllers=False),
            bpf_map_adapter=MockBPFMapAdapter(),
            probe_result=KernelProbeResult(
                sched_ext="unavailable",
                bpf_map="mock",
                fallback_mode="mock_bpf_map",
                reason="unit test",
            ),
        )

        rt = Runtime(
            llm_backend=MockLLMBackend(),
            handlers={},
            kernel_adapter=adapter,
        )

        await rt.run(dag)

        system_events = [
            e for e in rt.events
            if e.event_type == "resource.pressure_sampled"
            and e.data.get("scope") == "system"
        ]
        assert len(system_events) >= 1
        sample_data = system_events[0].data
        assert "pressure" in sample_data

    async def test_runtime_emits_agent_cgroup_pressure_sampled_event(self, tmp_path: Path):
        """_sample_worker_usage emits resource.pressure_sampled with scope=agent_cgroup."""
        dag = TaskDAG()
        dag.add_task(
            Task(
                task_id="t1",
                name="work",
                agent_role="echo",
                metadata={
                    "execution": "subprocess",
                    "worker_type": "echo",
                    "worker_payload": {"result": "ok"},
                },
            )
        )

        adapter = UserSpaceKernelAdapter(
            cgroup_manager=CgroupV2Manager(root=tmp_path / "cg", enable_controllers=False),
            bpf_map_adapter=MockBPFMapAdapter(),
            probe_result=KernelProbeResult(
                sched_ext="unavailable",
                bpf_map="mock",
                fallback_mode="mock_bpf_map",
                reason="unit test",
            ),
        )

        rt = Runtime(
            llm_backend=MockLLMBackend(),
            handlers={},
            kernel_adapter=adapter,
        )

        await rt.run(dag)

        cgroup_events = [
            e for e in rt.events
            if e.event_type == "resource.pressure_sampled"
            and e.data.get("scope") == "agent_cgroup"
        ]
        assert len(cgroup_events) >= 1
        assert "path" in cgroup_events[0].data

    async def test_runtime_decision_pressure_has_psi_values(self, tmp_path: Path):
        """scheduler.decision.pressure contains PSI raw values."""
        dag = TaskDAG()
        dag.add_task(
            Task(
                task_id="t1",
                name="work",
                agent_role="echo",
                metadata={
                    "execution": "subprocess",
                    "worker_type": "echo",
                    "worker_payload": {"result": "ok"},
                },
            )
        )

        adapter = UserSpaceKernelAdapter(
            cgroup_manager=CgroupV2Manager(root=tmp_path / "cg", enable_controllers=False),
            bpf_map_adapter=MockBPFMapAdapter(),
            probe_result=KernelProbeResult(
                sched_ext="unavailable",
                bpf_map="mock",
                fallback_mode="mock_bpf_map",
                reason="unit test",
            ),
        )

        rt = Runtime(
            llm_backend=MockLLMBackend(),
            handlers={},
            kernel_adapter=adapter,
        )

        await rt.run(dag)

        decision_events = [
            e for e in rt.events if e.event_type == "scheduler.decision"
        ]
        assert len(decision_events) >= 1
        decision = decision_events[0].data
        assert "pressure" in decision
        pressure = decision["pressure"]
        assert "os_pressure_level" in pressure
        assert "effective_pressure_level" in pressure

    async def test_pressure_sampling_exception_degrades_to_normal(self, tmp_path: Path):
        """Pressure sampling exception degrades to normal, doesn't fail tasks."""
        dag = TaskDAG()
        dag.add_task(
            Task(
                task_id="t1",
                name="work",
                agent_role="echo",
                metadata={
                    "execution": "subprocess",
                    "worker_type": "echo",
                    "worker_payload": {"result": "ok"},
                },
            )
        )

        adapter = UserSpaceKernelAdapter(
            cgroup_manager=CgroupV2Manager(root=tmp_path / "cg", enable_controllers=False),
            bpf_map_adapter=MockBPFMapAdapter(),
            probe_result=KernelProbeResult(
                sched_ext="unavailable",
                bpf_map="mock",
                fallback_mode="mock_bpf_map",
                reason="unit test",
            ),
            psi_paths={
                "cpu": tmp_path / "nonexistent" / "cpu",
                "memory": tmp_path / "nonexistent" / "memory",
                "io": tmp_path / "nonexistent" / "io",
            },
        )

        rt = Runtime(
            llm_backend=MockLLMBackend(),
            handlers={},
            kernel_adapter=adapter,
        )

        results = await rt.run(dag)

        # Task should still succeed
        assert results["t1"] == "ok"
        # Pressure events should still exist (degraded to normal)
        pressure_events = [
            e for e in rt.events if e.event_type == "resource.pressure_sampled"
        ]
        assert len(pressure_events) >= 1

    async def test_no_kernel_scheduling_claims_in_events(self, tmp_path: Path):
        """Events must not claim kernel scheduling takeover."""
        dag = TaskDAG()
        dag.add_task(
            Task(
                task_id="t1",
                name="work",
                agent_role="echo",
                metadata={
                    "execution": "subprocess",
                    "worker_type": "echo",
                    "worker_payload": {"result": "ok"},
                },
            )
        )

        adapter = UserSpaceKernelAdapter(
            cgroup_manager=CgroupV2Manager(root=tmp_path / "cg", enable_controllers=False),
            bpf_map_adapter=MockBPFMapAdapter(),
            probe_result=KernelProbeResult(
                sched_ext="unavailable",
                bpf_map="mock",
                fallback_mode="mock_bpf_map",
                reason="unit test",
            ),
        )

        rt = Runtime(
            llm_backend=MockLLMBackend(),
            handlers={},
            kernel_adapter=adapter,
        )

        await rt.run(dag)

        for event in rt.events:
            event_str = str(event.data).lower()
            assert "内核调度已启用" not in event_str
            assert "内核调度接管" not in event_str
            assert "kernel scheduling enabled" not in event_str


class TestCgroupLocalPressureControlLoop:
    class _ImmediateHandler(AgentHandler):
        async def run(self, ctx: AgentHandler.Context) -> str:
            return ctx.task.task_id

    class _DeterministicPressureAdapter(UserSpaceKernelAdapter):
        def __init__(
            self,
            *,
            cgroup_manager: CgroupV2Manager,
            local_cpu: float,
            local_sources: dict[str, str] | None = None,
            system_cpu: float = 0.0,
        ) -> None:
            super().__init__(
                cgroup_manager=cgroup_manager,
                bpf_map_adapter=MockBPFMapAdapter(),
                probe_result=KernelProbeResult(
                    sched_ext="unavailable",
                    bpf_map="mock",
                    fallback_mode="mock_bpf_map",
                    reason="unit test",
                ),
            )
            self._local_cpu = local_cpu
            self._local_sources = local_sources or {
                "cpu": "cgroup",
                "memory": "cgroup",
                "io": "cgroup",
            }
            self._system_cpu = system_cpu

        def sample_pressure(self, *, cgroup_path=None):
            if cgroup_path is None:
                return PressureSample(
                    cpu_some_avg10=self._system_cpu,
                    pressure_level=pressure_level_from_psi(
                        self._system_cpu, 0.0, 0.0
                    ),
                    pressure_source="system",
                    pressure_sources={
                        "cpu": "system",
                        "memory": "system",
                        "io": "system",
                    },
                )
            return PressureSample(
                cpu_some_avg10=self._local_cpu,
                memory_some_avg10=0.0,
                io_some_avg10=0.0,
                cgroup_usage=ResourceUsage(cpu_percent=1.0, pids_current=1),
                pressure_level=pressure_level_from_psi(
                    self._local_cpu, 0.0, 0.0
                ),
                pressure_source="cgroup",
                pressure_sources=dict(self._local_sources),
            )

    @staticmethod
    def _dependent_dag() -> TaskDAG:
        dag = TaskDAG()
        dag.add_task(
            Task(
                task_id="producer",
                name="local PSI producer",
                agent_role="echo",
                metadata={
                    "execution": "subprocess",
                    "worker_type": "echo",
                    "worker_payload": {"result": "producer"},
                },
            )
        )
        dag.add_task(
            Task(
                task_id="consumer",
                name="feedback consumer",
                agent_role="immediate",
                dependencies=["producer"],
            )
        )
        dag.add_task(
            Task(
                task_id="after",
                name="after consume",
                agent_role="immediate",
                dependencies=["consumer"],
            )
        )
        return dag

    async def test_local_critical_pressure_controls_next_round_once(
        self, tmp_path: Path,
    ):
        dag = self._dependent_dag()
        adapter = self._DeterministicPressureAdapter(
            cgroup_manager=CgroupV2Manager(
                root=tmp_path / "cg", enable_controllers=False
            ),
            local_cpu=25.0,
        )
        runtime = Runtime(
            llm_backend=MockLLMBackend(),
            handlers={"immediate": self._ImmediateHandler()},
            kernel_adapter=adapter,
        )

        results = await runtime.run(dag)

        assert results == {
            "producer": "producer",
            "consumer": "consumer",
            "after": "after",
        }
        applied = [
            event
            for event in runtime.events
            if event.event_type == "resource.cgroup_feedback_applied"
        ]
        assert len(applied) == 1
        assert applied[0].data["system_pressure_level"] == "normal"
        assert applied[0].data["local_pressure_level"] == "critical"
        assert applied[0].data["effective_pressure_level"] == "critical"
        assert applied[0].data["consumed"] is True
        assert applied[0].data["reusable"] is False

        applied_index = runtime.events.index(applied[0])
        following_decision = next(
            event
            for event in runtime.events[applied_index + 1 :]
            if event.event_type == "scheduler.decision"
        )
        pressure = following_decision.data["pressure"]
        assert pressure["pressure_source"] == "system+cgroup"
        assert pressure["os_pressure_level"] == "critical"
        assert pressure["effective_pressure_level"] == "critical"
        assert pressure["cgroup_local"]["cpu_some_avg10"] == 25.0
        assert (
            pressure["cgroup_local"]["feedback_id"]
            == applied[0].data["feedback_id"]
        )

        later_decisions = [
            event
            for event in runtime.events[
                runtime.events.index(following_decision) + 1 :
            ]
            if event.event_type == "scheduler.decision"
        ]
        assert later_decisions
        assert all(
            event.data["pressure"].get("pressure_source") == "system"
            for event in later_decisions
        )

    async def test_fallback_only_worker_pressure_stays_system_only(
        self, tmp_path: Path,
    ):
        dag = self._dependent_dag()
        adapter = self._DeterministicPressureAdapter(
            cgroup_manager=CgroupV2Manager(
                root=tmp_path / "cg", enable_controllers=False
            ),
            local_cpu=25.0,
            local_sources={
                "cpu": "system",
                "memory": "system",
                "io": "unavailable",
            },
        )
        runtime = Runtime(
            llm_backend=MockLLMBackend(),
            handlers={"immediate": self._ImmediateHandler()},
            kernel_adapter=adapter,
        )

        await runtime.run(dag)

        assert not any(
            event.event_type == "resource.cgroup_feedback_applied"
            for event in runtime.events
        )
        decisions = [
            event
            for event in runtime.events
            if event.event_type == "scheduler.decision"
        ]
        assert decisions
        assert all(
            event.data["pressure"].get("pressure_source") == "system"
            for event in decisions
        )

    async def test_local_normal_does_not_lower_system_warning(
        self, tmp_path: Path,
    ):
        dag = self._dependent_dag()
        adapter = self._DeterministicPressureAdapter(
            cgroup_manager=CgroupV2Manager(
                root=tmp_path / "cg", enable_controllers=False
            ),
            local_cpu=1.0,
            system_cpu=12.0,
        )
        runtime = Runtime(
            llm_backend=MockLLMBackend(),
            handlers={"immediate": self._ImmediateHandler()},
            kernel_adapter=adapter,
        )

        await runtime.run(dag)

        applied = next(
            event
            for event in runtime.events
            if event.event_type == "resource.cgroup_feedback_applied"
        )
        assert applied.data["system_pressure_level"] == "warning"
        assert applied.data["local_pressure_level"] == "normal"
        assert applied.data["effective_pressure_level"] == "warning"


# ===================================================================
# Review Fix Tests
# ===================================================================


class TestEffectivePressureControlsSafety:
    """Task 1: effective_pressure_level must control Safety/Backpressure."""

    async def test_effective_pressure_controls_final_safety_and_backpressure(self):
        """Critical OS pressure → batch task deferred, hard backpressure."""
        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

        scheduler = DAGScheduler(dag)
        scheduler.set_pressure(pressure_level="critical")
        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        assert decision["pressure"]["effective_pressure_level"] == "critical"
        assert decision["tasks"]["t1"]["admission_decision"] == "defer"
        assert decision["tasks"]["t1"]["backpressure_signal"] == "hard"
        assert decision["backpressure"]["pressure_level"] == "critical"
        assert decision["backpressure"]["signal"] == "hard"
        assert decision["backpressure"]["defer_count"] == 1

    async def test_runtime_health_pressure_level_still_independent(self):
        """runtime_health_pressure_level is still recorded independently."""
        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

        scheduler = DAGScheduler(dag)
        scheduler.set_pressure(pressure_level="critical")
        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        pressure = decision["pressure"]
        assert "runtime_health_pressure_level" in pressure
        # runtime_health is independent; may differ from effective
        assert pressure["runtime_health_pressure_level"] in (
            "normal", "warning", "critical", "overloaded"
        )

    async def test_normal_pressure_allows_batch_task(self):
        """Normal pressure → batch task allowed."""
        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

        scheduler = DAGScheduler(dag)
        scheduler.set_pressure(pressure_level="normal")
        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        assert decision["pressure"]["effective_pressure_level"] == "normal"
        assert decision["tasks"]["t1"]["admission_decision"] == "allow"
        assert decision["backpressure"]["signal"] == "none"


class TestReadPSIAvg10WithStatus:
    """Tests for read_psi_avg10_with_status() — distinguishing zero from missing."""

    def test_valid_zero_psi_returns_true(self, tmp_path: Path):
        psi_file = tmp_path / "cpu"
        psi_file.write_text(
            "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n",
            encoding="utf-8",
        )
        value, available = read_psi_avg10_with_status(psi_file)
        assert value == 0.0
        assert available is True

    def test_missing_file_returns_false(self, tmp_path: Path):
        value, available = read_psi_avg10_with_status(tmp_path / "nonexistent")
        assert value == 0.0
        assert available is False

    def test_corrupt_file_returns_false(self, tmp_path: Path):
        psi_file = tmp_path / "cpu"
        psi_file.write_text("CORRUPT\n", encoding="utf-8")
        value, available = read_psi_avg10_with_status(psi_file)
        assert value == 0.0
        assert available is False

    def test_file_without_some_line_returns_false(self, tmp_path: Path):
        psi_file = tmp_path / "cpu"
        psi_file.write_text("full avg10=5.0 avg60=3.0 avg300=1.0 total=100\n", encoding="utf-8")
        value, available = read_psi_avg10_with_status(psi_file)
        assert value == 0.0
        assert available is False

    def test_normal_value_returns_true(self, tmp_path: Path):
        psi_file = tmp_path / "cpu"
        psi_file.write_text(
            "some avg10=12.50 avg60=8.0 avg300=4.0 total=100\n",
            encoding="utf-8",
        )
        value, available = read_psi_avg10_with_status(psi_file)
        assert value == 12.50
        assert available is True

    def test_valid_zero_psi_is_not_reported_as_missing(self, tmp_path: Path):
        """Adapter with valid zero PSI must NOT report reason as missing."""
        psi_dir = tmp_path / "proc" / "pressure"
        _write_psi_files(psi_dir, cpu=0, memory=0, io=0)

        adapter = _make_adapter(tmp_path, psi_paths={
            "cpu": psi_dir / "cpu",
            "memory": psi_dir / "memory",
            "io": psi_dir / "io",
        })

        sample = adapter.sample_pressure()
        assert sample.pressure_level == "normal"
        assert sample.cpu_some_avg10 == 0.0
        assert sample.reason is None  # NOT reported as missing


class TestWorkerSamplingException:
    """Task 2: Worker pressure sampling exception must degrade and cleanup."""

    @pytest.mark.asyncio
    async def test_worker_sampling_exception_degrades_and_cleans_cgroup(self, tmp_path: Path):
        """When sample_pressure raises, worker still succeeds and cgroup is cleaned."""

        class _FailingPressureAdapter(UserSpaceKernelAdapter):
            def sample_pressure(self, *, cgroup_path=None):
                raise OSError("simulated PSI read failure")

        dag = TaskDAG()
        dag.add_task(
            Task(
                task_id="t1",
                name="work",
                agent_role="echo",
                metadata={
                    "execution": "subprocess",
                    "worker_type": "echo",
                    "worker_payload": {"result": "ok"},
                },
            )
        )

        adapter = _FailingPressureAdapter(
            cgroup_manager=CgroupV2Manager(root=tmp_path / "cg", enable_controllers=False),
            bpf_map_adapter=MockBPFMapAdapter(),
            probe_result=KernelProbeResult(
                sched_ext="unavailable",
                bpf_map="mock",
                fallback_mode="mock_bpf_map",
                reason="unit test",
            ),
        )

        rt = Runtime(
            llm_backend=MockLLMBackend(),
            handlers={},
            kernel_adapter=adapter,
        )

        results = await rt.run(dag)

        # Worker result still succeeds
        assert results["t1"] == "ok"
        # Task is not failed
        assert dag.tasks["t1"].status.value == "completed"

        # agent_cgroup pressure event has degraded reason
        cgroup_events = [
            e for e in rt.events
            if e.event_type == "resource.pressure_sampled"
            and e.data.get("scope") == "agent_cgroup"
        ]
        assert len(cgroup_events) >= 1
        assert cgroup_events[0].data["pressure"]["pressure_level"] == "normal"
        assert "failed" in cgroup_events[0].data["pressure"].get("reason", "").lower()

        # cgroup was cleaned up
        assert any(e.event_type == "resource.cgroup_removed" for e in rt.events)

        # cgroup path no longer exists
        acb = rt.agents["agent-t1"]
        if acb.cgroup:
            assert not Path(acb.cgroup).exists()


class TestEndToEndPSIToPolicy:
    """Task 4: Full Runtime loop — PSI file → Adapter → Scheduler → policy."""

    async def test_critical_psi_drives_risk_aware_policy_e2e(self, tmp_path: Path):
        """Injected PSI files → Runtime → scheduler.decision shows risk_aware."""
        psi_dir = tmp_path / "proc" / "pressure"
        _write_psi_files(psi_dir, cpu=21, memory=5, io=2)

        adapter = UserSpaceKernelAdapter(
            cgroup_manager=CgroupV2Manager(root=tmp_path / "cg", enable_controllers=False),
            bpf_map_adapter=MockBPFMapAdapter(),
            probe_result=KernelProbeResult(
                sched_ext="unavailable",
                bpf_map="mock",
                fallback_mode="mock_bpf_map",
                reason="unit test",
            ),
            psi_paths={
                "cpu": psi_dir / "cpu",
                "memory": psi_dir / "memory",
                "io": psi_dir / "io",
            },
        )

        dag = TaskDAG()
        dag.add_task(
            Task(
                task_id="t1",
                name="work",
                agent_role="echo",
                metadata={
                    "execution": "subprocess",
                    "worker_type": "echo",
                    "worker_payload": {"result": "ok"},
                },
            )
        )

        rt = Runtime(
            llm_backend=MockLLMBackend(),
            handlers={},
            kernel_adapter=adapter,
        )

        await rt.run(dag)

        # Find the scheduler decision event
        decision_events = [
            e for e in rt.events if e.event_type == "scheduler.decision"
        ]
        assert len(decision_events) >= 1
        decision = decision_events[0].data

        # PSI values propagated through the full chain
        assert decision["pressure"]["os_pressure_level"] == "critical"
        assert decision["pressure"]["cpu_some_avg10"] == 21.0
        assert decision["policy"] == "risk_aware"

        # Task still completes
        assert dag.tasks["t1"].status.value == "completed"
