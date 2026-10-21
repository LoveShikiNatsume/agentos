from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from agentos.benchmarks.sched_ext_release import (
    MODES,
    OFFICIAL,
    aggregate_stage4,
    artifact_manifest,
    benchmark_csv,
    load_official_evidence,
    render_report,
)
from scripts.sched_ext_stage4_common import proc_cpu_delta, proc_cpu_snapshot
from scripts.run_sched_ext_stage6_release import assemble
import scripts.run_sched_ext_stage6_release as stage6_runner
from scripts.verify_sched_ext_stage6_release import main as verify_main, verify


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _official(root: Path) -> dict[str, dict]:
    values = {
        "stage1_baseline": {"accepted": True},
        "stage1_preflight": {"accepted": True},
        "stage2": {
            "accepted": True, "partial_mode": True, "scheduler_name": "agentos_scx",
            "workers": {"completed": 100, "ext_enabled": 100},
            "controls": {"loader": 0, "runtime": 0},
        },
        "stage3": {
            "accepted": True, "workers": {"completed": 10, "complete_associations": 10},
            "policy_readbacks_verified": 10, "policy_map_entries_after_runtime": 0,
        },
        "stage4_policy_effects": {"accepted": True},
        "stage4_smoke_1": {"accepted": True},
        "stage4_smoke_2": {"accepted": True},
        "stage4_formal": {
            "accepted": True, "checksum_consistent": True, "policy_leaks": 0,
            "task_state_leaks": 0,
            "modes": {mode: {"workers_completed": 100, "affinity_violations": 0} for mode in MODES},
        },
        "stage5_formal": {
            "accepted": True, "passed": 380, "trials": 380,
            "soak": {"completed": True, "duration_seconds": 3601, "task_loss": 0},
            "outcomes": {"strict_failed": 80}, "kernel_log_errors": [],
            "final_sched_ext_state": "disabled", "systemd_failed_units": 0,
            "policy_leaks": 0, "task_state_leaks": 0,
            "residual_bpf_links": [], "residual_cgroups": [],
            "residual_pinned_maps": [], "residual_processes": [],
        },
    }
    for name, relative in OFFICIAL.items():
        _write(root / relative, values[name])
    trial_dir = root / "stage4_ablation_formal_v9/trials"
    for run in range(20):
        for mode in MODES:
            _write(trial_dir / f"{run:02d}-{mode}.json", {
                "mode": mode, "run_index": run, "completed": True,
                "throughput_iterations": 1200 + run, "jain_fairness": 0.9,
                "batch_max_gap_us": 10, "worker_startup_max_us": 5,
                "affinity_violations": 0, "policy_leaks": 0, "task_state_leaks": 0,
                "stats": {},
                "workers": [
                    {"role": "latency", "latencies_us": [1, 2, 3, 4], "iterations": 1},
                    {"role": "normal", "iterations": 10},
                ],
            })
    return values


def test_aggregate_stage4_recomputes_percentiles_and_csv(tmp_path: Path) -> None:
    _official(tmp_path)
    summary, raw = aggregate_stage4(tmp_path)
    assert len(raw) == 100
    assert summary["modes"]["cfs"]["metrics"]["latency_p50_us"]["mean"] == 2.5
    assert summary["modes"]["cfs"]["metrics"]["latency_p99_us"]["mean"] == pytest.approx(3.97)
    assert "benchmark iterations/second" in summary["interpretation"]["throughput_unit"]
    assert benchmark_csv(summary).startswith("mode,metric,n,mean")


def test_aggregate_requires_twenty_runs(tmp_path: Path) -> None:
    _official(tmp_path)
    next((tmp_path / "stage4_ablation_formal_v9/trials").glob("00-cfs.json")).unlink()
    with pytest.raises(ValueError, match="requires 20"):
        aggregate_stage4(tmp_path)


def test_official_evidence_rejects_failure_root(tmp_path: Path) -> None:
    failed = tmp_path / "failed_probe"
    failed.mkdir()
    with pytest.raises(ValueError, match="cannot be an official"):
        load_official_evidence(failed)


def test_proc_stat_observation(tmp_path: Path) -> None:
    stat = tmp_path / "stat"
    stat.write_text("cpu0 10 0 5 20 2\ncpu1 10 0 5 20 2\nctxt 100\n", encoding="utf-8")
    before = proc_cpu_snapshot(stat, (0, 1))
    after = {"total": before["total"] + 100, "idle": before["idle"] + 40, "context_switches": 125}
    assert proc_cpu_delta(before, after) == {"cpu_utilization": 0.6, "system_context_switches": 25}


def test_stage6_runner_preserves_virtualenv_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stage6_runner.sys, "executable", "/opt/venv/bin/python")
    assert stage6_runner.child_python() == "/opt/venv/bin/python"


def _release_fixture(tmp_path: Path, *, dirty: bool = False) -> Path:
    bundle = tmp_path / "release"
    defense = bundle / "defense_bundle"
    evidence = defense / "evidence/stage1-5"
    official = _official(evidence)
    summary, _raw = aggregate_stage4(evidence)
    _write(bundle / "benchmark_summary.json", summary)
    (bundle / "benchmark_summary.csv").write_text(benchmark_csv(summary), encoding="utf-8")
    (bundle / "report.md").write_text(render_report(official, summary), encoding="utf-8")
    logs = bundle / "logs"
    logs.mkdir(parents=True)
    from agentos.benchmarks.sched_ext_release import sha256_file
    pytest_results = {}
    for host in ("debian", "openeuler"):
        log = logs / f"{host}-pytest.log"
        log.write_text("passed", encoding="utf-8")
        pytest_results[host] = {
            "passed": True, "exit_code": 0, "log": log.name,
            "sha256": sha256_file(log),
        }
    _write(bundle / "pytest_results.json", pytest_results)
    regression = {
        "accepted": True, "sysfs_after": {"state": "disabled"},
        "summaries": {
            "runtime": {"accepted": True}, "policy_effects": {"accepted": True},
            "ablation": {
                "accepted": True, "iterations": 3,
                "modes": {
                    mode: {"observability": {
                        key: {"n": 3, "mean": 1}
                        for key in (
                            "cpu_utilization", "system_context_switches", "cpu_user_us",
                            "cpu_system_us", "voluntary_context_switches",
                            "involuntary_context_switches",
                        )
                    }}
                    for mode in MODES
                },
            },
            "reliability": {"accepted": True, "passed": 19, "trials": 19},
        },
    }
    _write(defense / "evidence/stage6-regression/summary.json", regression)
    events = ["kernel.sched_ext_fallback"] * 240 + ["kernel.sched_ext_recovered"] * 160
    (bundle / "events.jsonl").write_text(
        "".join(json.dumps({"event": event, "source_sha256": "a" * 64}) + "\n" for event in events),
        encoding="utf-8",
    )
    (defense / "README.md").write_text("partial mode 仅接管 Agent Worker。\n", encoding="utf-8")
    paths = [path for path in bundle.rglob("*") if path.is_file()]
    _write(defense / "manifest.json", {
        "schema": "sched_ext_release.v1",
        "git": {"commit": "abc123", "dirty": dirty},
        "artifact_root": "release",
        "artifacts": artifact_manifest(bundle, paths),
    })
    return bundle


def test_assemble_runner_builds_a_verifiable_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "evidence"
    _official(evidence)
    stage5 = evidence / OFFICIAL["stage5_formal"].removesuffix("/summary.json")
    events_path = stage5 / "trials/auto/case/run-01/events.jsonl"
    events_path.parent.mkdir(parents=True)
    events_path.write_text(
        "".join(
            json.dumps({"event_type": event, "data": {}, "timestamp": 1}) + "\n"
            for event in (["kernel.sched_ext_fallback"] * 240 + ["kernel.sched_ext_recovered"] * 160)
        ),
        encoding="utf-8",
    )
    regression = tmp_path / "regression"
    fixture = _release_fixture(tmp_path / "template")
    regression_summary = json.loads(
        (fixture / "defense_bundle/evidence/stage6-regression/summary.json").read_text()
    )
    _write(regression / "summary.json", regression_summary)
    _write(regression / "environment.json", {
        "kernel_release": "6.12.95-agentos-scx",
        "cpu_topology": {"controller": 0, "workers": [1, 2], "pressure": 3},
        "artifacts": {},
    })
    (regression / "raw.log").write_text("ok", encoding="utf-8")
    debian_log = tmp_path / "debian.log"
    openeuler_log = tmp_path / "openeuler.log"
    debian_log.write_text("tests passed", encoding="utf-8")
    openeuler_log.write_text("tests passed", encoding="utf-8")
    monkeypatch.setattr(
        stage6_runner, "_git",
        lambda *args: "abc123" if args[-1] == "HEAD" else "main" if "abbrev-ref" in args else "",
    )
    output = tmp_path / "assembled"
    rc = assemble(Namespace(
        evidence_root=evidence, regression_root=regression,
        debian_pytest_log=debian_log, debian_pytest_exit_code=0,
        openeuler_pytest_log=openeuler_log, openeuler_pytest_exit_code=0,
        output_dir=output,
    ))
    assert rc == 0
    assert verify(output, frozen=True)["passed"] is True


def test_strict_verifier_accepts_complete_bundle(tmp_path: Path) -> None:
    result = verify(_release_fixture(tmp_path), frozen=True)
    assert result["passed"] is True, result


def test_strict_verifier_rejects_dirty_and_tampered_bundle(tmp_path: Path) -> None:
    bundle = _release_fixture(tmp_path, dirty=True)
    (bundle / "defense_bundle/README.md").write_text("tampered", encoding="utf-8")
    result = verify(bundle, frozen=True)
    failed = {item["name"] for item in result["checks"] if not item["passed"]}
    assert {"git_provenance", "manifest_hashes"} <= failed


def test_strict_verifier_check_only_does_not_mutate_bundle(tmp_path: Path) -> None:
    bundle = _release_fixture(tmp_path)
    before = {
        path.relative_to(bundle): path.read_bytes()
        for path in bundle.rglob("*")
        if path.is_file()
    }

    rc = verify_main([
        "--bundle", str(bundle), "--frozen", "--check-only", "--json",
    ])

    after = {
        path.relative_to(bundle): path.read_bytes()
        for path in bundle.rglob("*")
        if path.is_file()
    }
    assert rc == 0
    assert after == before
    assert not list(tmp_path.glob("member-a-sched-ext-stage6-release-*.tar.gz"))
