#!/usr/bin/env python3
"""Read-only strict verifier for P2 sched_ext business bundles."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentos.benchmarks.sched_ext_business import (
    BUSINESS_BUNDLE_SCHEMA,
    FUNCTIONAL_MODE,
    read_jsonl,
    verify_seal,
)
from agentos.scenarios.sched_ext_business.fixture import FixtureManifest, verify_fixture
from agentos.scenarios.sched_ext_business.metrics import compute_trial_metrics
from agentos.scenarios.sched_ext_business.pipeline import BUSINESS_TASK_IDS


CHECK_NAMES = (
    "required_files",
    "content_seal",
    "schema",
    "functional_boundary",
    "fixture_verified",
    "trial_set_complete",
    "events_complete",
    "metrics_recomputed",
    "output_checksums_consistent",
    "task_loss_zero",
    "no_residual_resources",
    "summary_consistent",
)


def _same_number(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-9)
    except (TypeError, ValueError):
        return False


def verify_bundle(bundle: Path) -> dict[str, Any]:
    bundle = Path(bundle)
    passed = {name: False for name in CHECK_NAMES}
    errors: dict[str, str] = {}
    required = {
        "environment.json",
        "build_manifest.json",
        "fixture_manifest.json",
        "trials.jsonl",
        "events.jsonl",
        "benchmark_raw.json",
        "benchmark_summary.json",
        "benchmark_summary.csv",
        "kernel_window.txt",
        "summary.json",
        "SHA256SUMS",
    }
    try:
        passed["required_files"] = bundle.is_dir() and all(
            (bundle / name).is_file() for name in required
        )
        passed["content_seal"] = passed["required_files"] and verify_seal(bundle)

        environment = json.loads((bundle / "environment.json").read_text(encoding="utf-8"))
        summary = json.loads((bundle / "summary.json").read_text(encoding="utf-8"))
        manifest = FixtureManifest.from_dict(
            json.loads((bundle / "fixture_manifest.json").read_text(encoding="utf-8"))
        )
        passed["schema"] = (
            environment.get("schema")
            == summary.get("schema")
            == BUSINESS_BUNDLE_SCHEMA
        )
        passed["functional_boundary"] = (
            environment.get("mode") == FUNCTIONAL_MODE
            and environment.get("functional_only") is True
            and environment.get("sched_ext_performance_claim_allowed") is False
            and summary.get("functional_only") is True
            and summary.get("sched_ext_performance_claim_allowed") is False
        )
        try:
            verify_fixture(bundle / "fixture", manifest)
            passed["fixture_verified"] = True
        except Exception as exc:
            errors["fixture_verified"] = str(exc)

        trials = read_jsonl(bundle / "trials.jsonl")
        rounds = int(summary.get("rounds", -1))
        round_ids = [int(trial.get("round", -1)) for trial in trials]
        passed["trial_set_complete"] = rounds > 0 and round_ids == list(range(rounds))

        events = read_jsonl(bundle / "events.jsonl")
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            grouped[int(event.get("benchmark_round", -1))].append(event)
        passed["events_complete"] = set(grouped) == set(range(rounds))

        recomputed_ok = passed["trial_set_complete"] and passed["events_complete"]
        output_checksums: set[str] = set()
        no_loss = True
        no_residual = True
        for trial in trials:
            round_id = int(trial["round"])
            task_results = trial.get("task_results")
            units = trial.get("work_units")
            if not isinstance(task_results, dict) or not isinstance(units, dict):
                recomputed_ok = False
                continue
            recomputed = compute_trial_metrics(
                grouped[round_id],
                task_results,
                deadlines_ms={"patch_validation": 2_000, "security_gate": 2_000},
                work_units=units,
            )
            for key in (
                "critical_ready_ms",
                "makespan_ms",
                "deadline_hit_rate",
                "tasks_per_second",
                "jain_fairness",
            ):
                recomputed_ok &= _same_number(recomputed[key], trial.get(key))
            recomputed_ok &= recomputed["output_checksum"] == trial.get("output_checksum")
            output_checksums.add(str(trial.get("output_checksum", "")))
            no_loss &= int(trial.get("task_loss", -1)) == 0
            cleanup = trial.get("cleanup", {})
            no_residual &= (
                cleanup.get("workers") == 0
                and cleanup.get("cgroups") == 0
                and cleanup.get("policies") == 0
                and cleanup.get("sched_ext_state") == "not_applicable"
            )
        passed["metrics_recomputed"] = bool(recomputed_ok)
        passed["output_checksums_consistent"] = (
            len(output_checksums) == 1 and "" not in output_checksums
        )
        passed["task_loss_zero"] = bool(trials) and no_loss
        passed["no_residual_resources"] = bool(trials) and no_residual
        passed["summary_consistent"] = (
            summary.get("accepted") is True
            and summary.get("completed_rounds") == len(trials) == rounds
            and summary.get("fixture_sha256") == manifest.aggregate_sha256
            and summary.get("output_checksum") in output_checksums
            and summary.get("task_loss") == 0
            and summary.get("failure_reason") is None
        )
    except Exception as exc:
        errors["verification"] = f"{type(exc).__name__}: {exc}"

    failed = [name for name in CHECK_NAMES if not passed[name]]
    return {
        "passed": not failed,
        "check_only": True,
        "checks": f"{len(CHECK_NAMES) - len(failed)}/{len(CHECK_NAMES)}",
        "failed_checks": failed,
        "errors": errors,
        "functional_only": True,
        "sched_ext_performance_claim_allowed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args(argv)
    result = verify_bundle(args.bundle)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
