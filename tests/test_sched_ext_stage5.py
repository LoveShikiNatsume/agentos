from pathlib import Path

import pytest

from scripts import verify_sched_ext_reliability as reliability


ROOT = (Path(__file__).resolve().parent.parent / "src")


def test_bpf_has_policy_lease_fallback_without_schema_growth() -> None:
    header = (ROOT / "native/sched_ext/include/agentos_scx.h").read_text()
    source = (ROOT / "native/sched_ext/agentos_scx.bpf.c").read_text()

    assert "AGENTOS_SCX_STAT_FALLBACK_POLICY_STALE" in header
    assert "POLICY_STALE" in source
    assert "policy->updated_at_ns" in source
    assert "sizeof(struct agentos_task_policy_v1) == 40" in header


def test_loader_heartbeat_is_locked_and_failure_is_fatal() -> None:
    source = (ROOT / "native/sched_ext/agentos_scx.c").read_text()

    assert "flock(lock_fd, LOCK_EX)" in source
    assert "failed to update loader heartbeat" in source
    assert "unexpected_exit || heartbeat_failed ? 1 : 0" in source


def test_stage5_verifier_is_independent_and_uses_fresh_loader_per_trial() -> None:
    source = (ROOT / "scripts/verify_sched_ext_reliability.py").read_text()

    assert "run_stage5_experiments" not in source
    assert "_start_captured_loader(scheduler_binary)" in source
    assert "threading.Thread(" in source
    assert "recovered_old_worker_ext_enabled" in source
    assert "stage5_reliability" in source
    assert 'choices=(1, 20)' in source


def test_stage5_strict_runtime_does_not_skip_lifecycle_failures(monkeypatch) -> None:
    class FakeRuntime:
        def __init__(self, **kwargs):
            self._recovery_policy = kwargs["recovery_policy"]

    monkeypatch.setattr(reliability, "Runtime", FakeRuntime)
    strict, _dag = reliability._make_runtime("strict", "loader_sigkill")
    auto, _dag = reliability._make_runtime("auto", "loader_sigkill")

    assert strict._recovery_policy is None
    assert auto._recovery_policy is not None


def test_policy_map_unpin_uses_bpffs_unlink(tmp_path: Path) -> None:
    pin = tmp_path / "task_policy_map"
    pin.touch()

    reliability._unpin_policy_map(pin)

    assert not pin.exists()


def test_residual_cgroups_only_reports_directories(tmp_path: Path) -> None:
    (tmp_path / "cgroup.procs").touch()
    (tmp_path / "agent-stage5-primary").mkdir()

    assert reliability._residual_cgroups(tmp_path) == ["agent-stage5-primary"]


def test_trial_cleanup_retries_empty_stage5_cgroups_only(tmp_path: Path) -> None:
    stage5 = tmp_path / "agent-stage5-primary"
    unrelated = tmp_path / "agent-production"
    stage5.mkdir()
    unrelated.mkdir()

    reliability._cleanup_trial_cgroups(tmp_path)

    assert not stage5.exists()
    assert unrelated.is_dir()


def test_reliability_verifier_unpins_sigkill_maps_only_after_disable(
    tmp_path: Path, monkeypatch
) -> None:
    pin_dir = tmp_path / "sched_ext"
    pin_dir.mkdir()
    for name in ("task_policy_map", "global_config_map"):
        (pin_dir / name).write_bytes(b"pin")
    monkeypatch.setattr(reliability, "_snapshot", lambda: {"state": "disabled"})

    reliability._unpin_sched_ext_maps(pin_dir)

    assert list(pin_dir.iterdir()) == []


def test_reliability_verifier_refuses_to_unpin_active_scheduler(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(reliability, "_snapshot", lambda: {"state": "enabled"})

    with pytest.raises(RuntimeError, match="refusing to unpin"):
        reliability._unpin_sched_ext_maps(tmp_path)
