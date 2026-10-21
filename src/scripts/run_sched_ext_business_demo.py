#!/usr/bin/env python3
"""Run the Debian functional phase of the P2 real-Agent business demo."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentos.benchmarks.sched_ext_business import (
    BUSINESS_BUNDLE_SCHEMA,
    DEFAULT_FIXTURE_PATHS,
    FROZEN_FIXTURE_MANIFEST,
    FROZEN_FIXTURE_VERSION,
    FUNCTIONAL_MODE,
    seal_directory,
    write_json,
    write_jsonl,
    write_summary_csv,
)
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import Runtime
from agentos.scenarios.sched_ext_business.fixture import FixtureManifest, freeze_fixture
from agentos.scenarios.sched_ext_business.metrics import compute_trial_metrics, summarise_trials
from agentos.scenarios.sched_ext_business.pipeline import BUSINESS_TASK_IDS, build_business_dag


def _git_value(source_root: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(source_root), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


async def _run_trial(
    fixture_root: Path,
    manifest: Any,
    work_units: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dag = build_business_dag(fixture_root, manifest, work_units=work_units)
    runtime = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={},
        execution_mode="subprocess",
        max_batch_size=len(BUSINESS_TASK_IDS),
        enable_kernel_probe=False,
        sched_ext_mode="disabled",
    )
    results = await runtime.run(dag)
    return results, [event.to_dict() for event in runtime.events]


def run_functional_demo(
    *,
    source_root: Path,
    output_dir: Path,
    fixture_paths: Sequence[str] = DEFAULT_FIXTURE_PATHS,
    rounds: int = 1,
    seed: int = 20260719,
    work_units: int = 1,
) -> dict[str, Any]:
    """Create a sealed functional-only bundle; never emits sched_ext claims."""

    source_root = Path(source_root).resolve(strict=True)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if not 1 <= rounds <= 10:
        raise ValueError("functional rounds must be between 1 and 10")
    if not 1 <= work_units <= 10_000:
        raise ValueError("work_units must be between 1 and 10000")
    output_dir.mkdir(parents=True)
    fixture_root = output_dir / "fixture"
    manifest = freeze_fixture(source_root, fixture_root, fixture_paths)
    if tuple(fixture_paths) == DEFAULT_FIXTURE_PATHS:
        expected = FixtureManifest.from_dict(
            json.loads(FROZEN_FIXTURE_MANIFEST.read_text(encoding="utf-8"))
        )
        if manifest != expected:
            raise ValueError(
                f"default fixture differs from frozen fixture v{FROZEN_FIXTURE_VERSION}: "
                f"expected {expected.aggregate_sha256}, got {manifest.aggregate_sha256}"
            )
    write_json(output_dir / "fixture_manifest.json", manifest.to_dict())

    environment = {
        "schema": BUSINESS_BUNDLE_SCHEMA,
        "mode": FUNCTIONAL_MODE,
        "functional_only": True,
        "sched_ext_performance_claim_allowed": False,
        "host": platform.node(),
        "kernel_release": platform.release(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    write_json(output_dir / "environment.json", environment)
    write_json(
        output_dir / "build_manifest.json",
        {
            "schema": BUSINESS_BUNDLE_SCHEMA,
            "git_commit": _git_value(source_root, "rev-parse", "HEAD"),
            "git_dirty": bool(_git_value(source_root, "status", "--porcelain")),
            "fixture_sha256": manifest.aggregate_sha256,
            "fixture_version": FROZEN_FIXTURE_VERSION,
            "seed": seed,
            "work_units": work_units,
            "rounds": rounds,
        },
    )

    random.seed(seed)
    trials: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    units = {task_id: work_units for task_id in BUSINESS_TASK_IDS}
    deadlines = {"patch_validation": 2_000, "security_gate": 2_000}
    for round_id in range(rounds):
        results, events = asyncio.run(_run_trial(fixture_root, manifest, work_units))
        task_results = {task_id: results[task_id] for task_id in BUSINESS_TASK_IDS}
        metrics = compute_trial_metrics(
            events,
            task_results,
            deadlines_ms=deadlines,
            work_units=units,
        )
        for event in events:
            all_events.append(
                {
                    **event,
                    "benchmark_mode": FUNCTIONAL_MODE,
                    "benchmark_round": round_id,
                }
            )
        trials.append(
            {
                "schema": BUSINESS_BUNDLE_SCHEMA,
                "mode": FUNCTIONAL_MODE,
                "round": round_id,
                "seed": seed,
                "fixture_sha256": manifest.aggregate_sha256,
                "work_units": units,
                "task_results": task_results,
                **metrics,
                "cleanup": {
                    "workers": 0,
                    "cgroups": 0,
                    "policies": 0,
                    "sched_ext_state": "not_applicable",
                },
            }
        )

    benchmark_summary = summarise_trials(trials)
    output_checksums = {trial["output_checksum"] for trial in trials}
    summary = {
        "schema": BUSINESS_BUNDLE_SCHEMA,
        "accepted": len(trials) == rounds and len(output_checksums) == 1,
        "functional_only": True,
        "sched_ext_performance_claim_allowed": False,
        "mode": FUNCTIONAL_MODE,
        "rounds": rounds,
        "completed_rounds": len(trials),
        "fixture_sha256": manifest.aggregate_sha256,
        "output_checksum": next(iter(output_checksums)),
        "task_loss": sum(int(trial["task_loss"]) for trial in trials),
        "failure_reason": None,
    }
    write_jsonl(output_dir / "trials.jsonl", trials)
    write_jsonl(output_dir / "events.jsonl", all_events)
    write_json(
        output_dir / "benchmark_raw.json",
        {"schema": BUSINESS_BUNDLE_SCHEMA, "trials": trials},
    )
    write_json(output_dir / "benchmark_summary.json", benchmark_summary)
    write_summary_csv(output_dir / "benchmark_summary.csv", benchmark_summary)
    (output_dir / "kernel_window.txt").write_text(
        "Not applicable: Debian functional validation does not produce sched_ext evidence.\n",
        encoding="utf-8",
    )
    write_json(output_dir / "summary.json", summary)
    seal_directory(output_dir)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--work-units", type=int, default=1)
    parser.add_argument("--fixture-path", action="append", dest="fixture_paths")
    args = parser.parse_args(argv)
    summary = run_functional_demo(
        source_root=args.source_root,
        output_dir=args.output_dir,
        fixture_paths=args.fixture_paths or DEFAULT_FIXTURE_PATHS,
        rounds=args.rounds,
        seed=args.seed,
        work_units=args.work_units,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
