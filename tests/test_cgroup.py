from pathlib import Path

import pytest

from agentos.core.resources import ResourceQuota
from agentos.events.log import EventLog
from agentos.isolation.cgroup import CgroupV2Manager, probe_cgroup


def test_cgroup_manager_gracefully_falls_back_when_root_is_not_directory(tmp_path):
    root_file = tmp_path / "not-a-dir"
    root_file.write_text("x", encoding="utf-8")
    manager = CgroupV2Manager(root=root_file)

    result = manager.create("agent-1", ResourceQuota(memory_max_mb=128))

    assert not result.available
    assert result.reason
    assert result.event.event_type == "resource.cgroup_unavailable"


def test_cgroup_manager_can_create_under_writable_root(tmp_path):
    manager = CgroupV2Manager(root=tmp_path)

    result = manager.create(
        "agent-1",
        ResourceQuota(
            cpu_shares=250,
            cpu_max_percent=50.0,
            memory_max_mb=128,
            pids_max=32,
        ),
    )

    assert result.path == tmp_path / "agent-1"
    assert result.event.event_type in {"resource.cgroup_created", "resource.cgroup_unavailable"}
    assert (result.path / "memory.max").read_text(encoding="utf-8") == str(128 * 1024 * 1024)
    assert (result.path / "pids.max").read_text(encoding="utf-8") == "32"
    assert (result.path / "cpu.max").read_text(encoding="utf-8") == "50000 100000"
    assert (result.path / "cpu.weight").read_text(encoding="utf-8") == "250"


def test_cgroup_manager_reads_usage_from_control_files(tmp_path):
    manager = CgroupV2Manager(root=tmp_path)
    result = manager.create("agent-1")
    (result.path / "memory.current").write_text(str(64 * 1024 * 1024), encoding="utf-8")
    (result.path / "memory.peak").write_text(str(96 * 1024 * 1024), encoding="utf-8")
    (result.path / "pids.current").write_text("3", encoding="utf-8")
    (result.path / "cpu.stat").write_text(
        "usage_usec 12345\nuser_usec 10000\nsystem_usec 2345\n",
        encoding="utf-8",
    )

    usage = manager.read_usage(result.path)

    assert usage.memory_rss_mb == 64.0
    assert usage.memory_peak_mb == 96.0
    assert usage.pids_current == 3
    assert usage.cpu_usage_usec == 12345


def test_cgroup_cleanup_removes_fake_cgroup_files(tmp_path):
    manager = CgroupV2Manager(root=tmp_path)
    result = manager.create("agent-1", ResourceQuota(memory_max_mb=128))
    (result.path / "memory.current").write_text("0", encoding="utf-8")

    cleanup = manager.cleanup(result.path)

    assert cleanup.available
    assert cleanup.event.event_type == "resource.cgroup_removed"
    assert not result.path.exists()


def test_probe_cgroup_writes_event_log(tmp_path):
    log_path = tmp_path / "events.jsonl"

    result = probe_cgroup(root=tmp_path / "cg", log=EventLog(log_path))

    assert result.event.event_type in {"resource.cgroup_created", "resource.cgroup_unavailable"}
    assert log_path.exists()


def test_cgroup_read_limits_from_fake_root(tmp_path: Path):
    manager = CgroupV2Manager(root=tmp_path / "cg", enable_controllers=False)
    result = manager.create(
        "agent-1",
        ResourceQuota(
            cpu_max_percent=25,
            memory_max_mb=128,
            pids_max=12,
            cpu_shares=200,
        ),
    )

    limits = manager.read_limits(result.path)

    assert limits["memory_max_mb"] == 128
    assert limits["pids_max"] == 12
    assert limits["cpu_max_percent"] == 25.0
    assert limits["cpu_weight"] == 200


def test_cgroup_read_limits_handles_cpu_max_unlimited(tmp_path: Path):
    manager = CgroupV2Manager(root=tmp_path / "cg", enable_controllers=False)
    result = manager.create(
        "agent-1",
        ResourceQuota(memory_max_mb=128, pids_max=12),
    )

    (result.path / "cpu.max").write_text("max 100000", encoding="utf-8")

    limits = manager.read_limits(result.path)

    assert limits["cpu_max_percent"] is None
    assert limits["memory_max_mb"] == 128
    assert limits["pids_max"] == 12


@pytest.mark.parametrize("name", ["../escape", "/etc/passwd", ".hidden", r"..\escape"])
def test_create_rejects_path_traversal_and_absolute_names(tmp_path: Path, name: str):
    manager = CgroupV2Manager(root=tmp_path)

    result = manager.create(name)

    assert not result.available
    assert "traversal" in (result.reason or "").lower()


@pytest.mark.parametrize("name", ["a b", "a;b", "a$b", "a!b", "x" * 64])
def test_create_rejects_illegal_names(tmp_path: Path, name: str):
    manager = CgroupV2Manager(root=tmp_path)

    result = manager.create(name)

    assert not result.available
    assert "illegal" in (result.reason or "").lower()


@pytest.mark.parametrize("name", ["agent-1", "worker.2", "task_abc", "A0"])
def test_create_accepts_valid_names(tmp_path: Path, name: str):
    manager = CgroupV2Manager(root=tmp_path)

    result = manager.create(name)

    assert result.available
    assert result.path == tmp_path / name


def test_create_rolls_back_new_directory_after_partial_apply_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    manager = CgroupV2Manager(root=tmp_path)

    def fail_after_write(path: Path, quota: ResourceQuota) -> None:
        (path / "memory.max").write_text("1", encoding="utf-8")
        raise OSError("write fail")

    monkeypatch.setattr(manager, "apply", fail_after_write)

    result = manager.create("probe", ResourceQuota(memory_max_mb=64))

    assert not result.available
    assert not (tmp_path / "probe").exists()


def test_create_failure_never_removes_preexisting_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    existing = tmp_path / "probe"
    existing.mkdir()
    marker = existing / "owner.marker"
    marker.write_text("keep", encoding="utf-8")
    manager = CgroupV2Manager(root=tmp_path)
    monkeypatch.setattr(
        manager,
        "apply",
        lambda path, quota: (_ for _ in ()).throw(OSError("write fail")),
    )

    result = manager.create("probe", ResourceQuota(memory_max_mb=64))

    assert not result.available
    assert existing.is_dir()
    assert marker.read_text(encoding="utf-8") == "keep"


def test_verify_limits_detects_memory_readback_mismatch(tmp_path: Path):
    manager = CgroupV2Manager(root=tmp_path)
    path = tmp_path / "x"
    path.mkdir()
    (path / "memory.max").write_text("0", encoding="utf-8")

    with pytest.raises(OSError, match="memory.max readback mismatch"):
        manager._verify_limits(path, ResourceQuota(memory_max_mb=128))


def test_verify_limits_detects_cpu_weight_readback_mismatch(tmp_path: Path):
    manager = CgroupV2Manager(root=tmp_path)
    path = tmp_path / "x"
    path.mkdir()
    (path / "cpu.weight").write_text("1", encoding="utf-8")

    with pytest.raises(OSError, match="cpu.weight readback mismatch"):
        manager._verify_limits(path, ResourceQuota(cpu_shares=200))


def test_verify_pid_membership_reads_unified_cgroup_entry(tmp_path: Path):
    proc_root = tmp_path / "proc"
    pid_dir = proc_root / "123"
    pid_dir.mkdir(parents=True)
    (pid_dir / "cgroup").write_text(
        "0::/agentos/agent-1\n",
        encoding="utf-8",
    )

    assert CgroupV2Manager.verify_pid_membership(
        Path("/sys/fs/cgroup/agentos/agent-1"),
        123,
        proc_root=proc_root,
    )


def test_verify_pid_membership_returns_false_for_exited_process(tmp_path: Path):
    assert CgroupV2Manager.verify_pid_membership(
        Path("/sys/fs/cgroup/agentos/agent-1"),
        999999,
        proc_root=tmp_path / "proc",
    ) is False


def test_add_pid_records_membership_check_result_for_fake_cgroup(tmp_path: Path):
    manager = CgroupV2Manager(root=tmp_path)
    result = manager.create("agent-1")

    attached = manager.add_pid(result.path, 999999)

    assert attached.available
    assert attached.event.data["pid_membership_verified"] is False


def test_read_pressure_returns_local_values_and_availability(tmp_path: Path):
    for name, value in (("cpu", 0.0), ("memory", 2.5), ("io", 7.25)):
        (tmp_path / f"{name}.pressure").write_text(
            f"some avg10={value:.2f} avg60=0.00 avg300=0.00 total=0\n",
            encoding="utf-8",
        )

    result = CgroupV2Manager(root=tmp_path).read_pressure(tmp_path)

    assert result["cpu"] == (0.0, True)
    assert result["memory"] == (2.5, True)
    assert result["io"] == (7.25, True)


def test_read_pressure_marks_missing_and_malformed_files_unavailable(
    tmp_path: Path,
):
    (tmp_path / "cpu.pressure").write_text("malformed\n", encoding="utf-8")

    result = CgroupV2Manager(root=tmp_path).read_pressure(tmp_path)

    assert result == {
        "cpu": (0.0, False),
        "memory": (0.0, False),
        "io": (0.0, False),
    }
