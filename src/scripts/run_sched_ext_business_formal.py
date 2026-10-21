#!/usr/bin/env python3
"""Run the approved P2 formal matrix on the frozen openEuler host."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentos.benchmarks.sched_ext_business import (
    DEFAULT_FIXTURE_PATHS,
    FROZEN_FIXTURE_MANIFEST,
    FROZEN_FIXTURE_VERSION,
    seal_directory,
    write_json,
    write_jsonl,
    write_summary_csv,
)
from agentos.benchmarks.sched_ext_business_formal import (
    FORMAL_MODES,
    FORMAL_ROUNDS,
    PRIMARY_BASELINE,
    PRIMARY_CANDIDATE,
    PRIMARY_METRIC,
    WARMUP_ROUNDS,
    build_formal_plan,
    evaluate_primary_gate,
    formal_round_seed,
)
from agentos.benchmarks.sched_ext_business_smoke import assess_trial_evidence
from agentos.scenarios.sched_ext_business.fixture import FixtureManifest, freeze_fixture
from agentos.scenarios.sched_ext_business.metrics import (
    paired_improvements,
    summarise_trials,
)
from scripts.preflight_sched_ext import probe_sched_ext
from scripts.run_sched_ext_business_smoke import (
    CONTROLLER_CPU,
    PRESSURE_CPU,
    WORKER_CPUS,
    _git_value,
    _snapshot,
    _trial,
)
from scripts.verify_real_sched_ext import kernel_log_errors


FORMAL_SCHEMA = "sched_ext_business_formal.v1"
FROZEN_HOST = "openeuler-sched-ext-dev"
FROZEN_KERNEL = "6.12.95-agentos-scx"
FROZEN_SCHEDULER_SHA256 = (
    "71b34014318f11383001d7cd30b77d742c324fc5e3ca00149c7fc52e289aba11"
)
FROZEN_FIXTURE_SHA256 = (
    "e65b1c9dd804abfd3b241db4650112d17eb6db2795aa11a0c4b620b07dd7fdd5"
)
FROZEN_WORK_UNITS = 20


def _tag_records(
    records: list[dict[str, Any]],
    *,
    mode: str,
    round_id: int,
    sample_kind: str,
) -> list[dict[str, Any]]:
    return [
        {
            **record,
            "benchmark_mode": mode,
            "benchmark_round": round_id,
            "sample_kind": sample_kind,
        }
        for record in records
    ]


def _execute_plan(
    *,
    plan: list[dict[str, int | str]],
    fixture_root: Path,
    manifest: FixtureManifest,
    scheduler_binary: Path,
    seed: int,
    trials: list[dict[str, Any]],
    events: list[dict[str, Any]],
    loader_events: list[dict[str, Any]],
) -> None:
    for item in plan:
        sample_kind = str(item["sample_kind"])
        round_id = int(item["round"])
        mode = str(item["mode"])
        trial, trial_events, trial_loader_events = _trial(
            mode=mode,
            round_id=round_id,
            order=int(item["order"]),
            fixture_root=fixture_root,
            manifest=manifest,
            scheduler_binary=scheduler_binary,
            work_units=FROZEN_WORK_UNITS,
            seed=formal_round_seed(seed, round_id, sample_kind=sample_kind),
        )
        trial.update(schema=FORMAL_SCHEMA, sample_kind=sample_kind, formal=True)
        trials.append(trial)
        events.extend(
            _tag_records(
                trial_events,
                mode=mode,
                round_id=round_id,
                sample_kind=sample_kind,
            )
        )
        loader_events.extend(
            _tag_records(
                trial_loader_events,
                mode=mode,
                round_id=round_id,
                sample_kind=sample_kind,
            )
        )
        failed_evidence = [
            name
            for name, passed in trial.get("evidence_checks", {}).items()
            if passed is not True
        ]
        if failed_evidence:
            raise RuntimeError(
                "formal runtime evidence failed at "
                f"{sample_kind}/{mode}/round-{round_id}: "
                f"{','.join(failed_evidence)}"
            )


def run_formal(
    *,
    scheduler_binary: Path,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    """Run the exact approved matrix; no count or threshold override is exposed."""

    if os.geteuid() != 0:
        raise PermissionError("P2 formal must run as root")
    if platform.node() != FROZEN_HOST:
        raise RuntimeError(f"unexpected formal host: {platform.node()}")
    if platform.release() != FROZEN_KERNEL:
        raise RuntimeError(f"unexpected formal kernel: {platform.release()}")
    required_affinity = {CONTROLLER_CPU, *WORKER_CPUS, PRESSURE_CPU}
    if not required_affinity.issubset(os.sched_getaffinity(0)):
        raise RuntimeError("P2 formal requires available CPUs 0,1,2,3")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if not scheduler_binary.is_file():
        raise FileNotFoundError(str(scheduler_binary))
    scheduler_sha256 = hashlib.sha256(scheduler_binary.read_bytes()).hexdigest()
    if scheduler_sha256 != FROZEN_SCHEDULER_SHA256:
        raise ValueError(f"scheduler SHA-256 is not frozen: {scheduler_sha256}")

    warmup_plan = build_formal_plan("warmup")
    formal_plan = build_formal_plan("formal")
    output_dir.mkdir(parents=True)
    journal_since = int(time.time())
    preflight = probe_sched_ext()
    write_json(output_dir / "preflight.json", preflight)
    if preflight.get("accepted") is not True:
        raise RuntimeError(f"sched_ext preflight rejected host: {preflight.get('failures')}")
    initial_state = _snapshot()
    if initial_state.get("state") != "disabled" or initial_state.get("switch_all") != "0":
        raise RuntimeError(f"formal host is not clean before execution: {initial_state}")

    fixture_root = output_dir / "fixture"
    manifest = freeze_fixture(PROJECT_ROOT, fixture_root, DEFAULT_FIXTURE_PATHS)
    expected = FixtureManifest.from_dict(
        json.loads(FROZEN_FIXTURE_MANIFEST.read_text(encoding="utf-8"))
    )
    if manifest != expected or manifest.aggregate_sha256 != FROZEN_FIXTURE_SHA256:
        raise ValueError("source does not match frozen fixture v2")
    write_json(output_dir / "fixture_manifest.json", manifest.to_dict())
    write_json(
        output_dir / "environment.json",
        {
            "schema": FORMAL_SCHEMA,
            "phase": "formal",
            "formal": True,
            "host": platform.node(),
            "kernel_release": platform.release(),
            "python": platform.python_version(),
            "controller_cpu": CONTROLLER_CPU,
            "worker_cpus": sorted(WORKER_CPUS),
            "pressure_cpu": PRESSURE_CPU,
            "modes": list(FORMAL_MODES),
            "warmup_rounds_per_mode": WARMUP_ROUNDS,
            "formal_rounds_per_mode": FORMAL_ROUNDS,
            "warmup_in_formal_statistics": False,
        },
    )
    write_json(
        output_dir / "build_manifest.json",
        {
            "schema": FORMAL_SCHEMA,
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_dirty": bool(_git_value("status", "--porcelain")),
            "scheduler_binary": str(scheduler_binary),
            "scheduler_sha256": scheduler_sha256,
            "fixture_sha256": manifest.aggregate_sha256,
            "fixture_version": FROZEN_FIXTURE_VERSION,
            "work_units": FROZEN_WORK_UNITS,
            "seed": seed,
            "primary_baseline": PRIMARY_BASELINE,
            "primary_candidate": PRIMARY_CANDIDATE,
            "primary_metric": PRIMARY_METRIC,
        },
    )

    warmup_trials: list[dict[str, Any]] = []
    warmup_events: list[dict[str, Any]] = []
    warmup_loader_events: list[dict[str, Any]] = []
    formal_trials: list[dict[str, Any]] = []
    formal_events: list[dict[str, Any]] = []
    formal_loader_events: list[dict[str, Any]] = []
    failure_reason: str | None = None
    try:
        _execute_plan(
            plan=warmup_plan,
            fixture_root=fixture_root,
            manifest=manifest,
            scheduler_binary=scheduler_binary,
            seed=seed,
            trials=warmup_trials,
            events=warmup_events,
            loader_events=warmup_loader_events,
        )
        _execute_plan(
            plan=formal_plan,
            fixture_root=fixture_root,
            manifest=manifest,
            scheduler_binary=scheduler_binary,
            seed=seed,
            trials=formal_trials,
            events=formal_events,
            loader_events=formal_loader_events,
        )
    except Exception as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"

    journal = subprocess.run(
        ["journalctl", "-k", "-b", f"--since=@{journal_since}", "--no-pager"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    (output_dir / "kernel_window.txt").write_text(
        journal.stdout + journal.stderr, encoding="utf-8"
    )
    kernel_errors = kernel_log_errors(journal.stdout.splitlines())
    systemd = subprocess.run(
        ["systemctl", "--failed", "--no-legend", "--plain"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    failed_units = [line for line in systemd.stdout.splitlines() if line.strip()]
    final_state = _snapshot()

    all_trials = warmup_trials + formal_trials
    evidence_checks = [assess_trial_evidence(trial) for trial in all_trials]
    output_checksums = {trial.get("output_checksum") for trial in all_trials}
    integrity_accepted = bool(
        failure_reason is None
        and len(warmup_trials) == len(warmup_plan)
        and len(formal_trials) == len(formal_plan)
        and evidence_checks
        and all(all(item.values()) for item in evidence_checks)
        and len(output_checksums) == 1
        and kernel_errors == []
        and failed_units == []
        and journal.returncode == 0
        and final_state.get("state") == "disabled"
        and final_state.get("switch_all") == "0"
    )
    benchmark_summary = summarise_trials(formal_trials) if formal_trials else {}
    paired = (
        {
            mode: paired_improvements(formal_trials, baseline="cfs", candidate=mode)
            for mode in FORMAL_MODES
            if mode != "cfs"
        }
        if len(formal_trials) == len(formal_plan)
        else {}
    )
    primary_gate = (
        evaluate_primary_gate(formal_trials)
        if len(formal_trials) == len(formal_plan)
        else {"passed": False, "reason": "formal trial plan incomplete"}
    )
    summary = {
        "schema": FORMAL_SCHEMA,
        "accepted": integrity_accepted,
        "phase": "formal",
        "formal": True,
        "performance_gate_passed": bool(primary_gate.get("passed")),
        "modes": list(FORMAL_MODES),
        "expected_warmup_trials": len(warmup_plan),
        "completed_warmup_trials": len(warmup_trials),
        "expected_formal_trials": len(formal_plan),
        "completed_formal_trials": len(formal_trials),
        "fixture_sha256": manifest.aggregate_sha256,
        "output_checksum": next(iter(output_checksums)) if len(output_checksums) == 1 else None,
        "task_loss": sum(int(trial.get("task_loss", 0)) for trial in all_trials),
        "kernel_log_errors": kernel_errors,
        "systemd_failed_units": len(failed_units),
        "final_sched_ext_state": final_state.get("state"),
        "final_switch_all": final_state.get("switch_all"),
        "failure_reason": failure_reason,
    }
    write_jsonl(output_dir / "warmup_trials.jsonl", warmup_trials)
    write_jsonl(output_dir / "warmup_events.jsonl", warmup_events)
    write_jsonl(output_dir / "warmup_loader_events.jsonl", warmup_loader_events)
    write_jsonl(output_dir / "trials.jsonl", formal_trials)
    write_jsonl(output_dir / "events.jsonl", formal_events)
    write_jsonl(output_dir / "loader_events.jsonl", formal_loader_events)
    write_json(
        output_dir / "benchmark_raw.json",
        {"schema": FORMAL_SCHEMA, "sample_kind": "formal", "trials": formal_trials},
    )
    write_json(
        output_dir / "benchmark_summary.json",
        {"modes": benchmark_summary, "paired_improvements": paired},
    )
    write_summary_csv(output_dir / "benchmark_summary.csv", benchmark_summary)
    write_json(output_dir / "primary_gate.json", primary_gate)
    write_json(output_dir / "summary.json", summary)
    seal_directory(output_dir)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler-binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args(argv)
    summary = run_formal(
        scheduler_binary=args.scheduler_binary.resolve(),
        output_dir=args.output_dir.resolve(),
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
