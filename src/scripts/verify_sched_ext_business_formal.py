#!/usr/bin/env python3
"""Read-only strict verifier for approved P2 formal bundles."""

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

from agentos.benchmarks.sched_ext_business import read_jsonl, verify_seal
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
from agentos.scenarios.sched_ext_business.fixture import FixtureManifest, verify_fixture
from agentos.scenarios.sched_ext_business.metrics import (
    compute_trial_metrics,
    paired_improvements,
    summarise_trials,
)
from scripts.run_sched_ext_business_formal import (
    FORMAL_SCHEMA,
    FROZEN_FIXTURE_SHA256,
    FROZEN_HOST,
    FROZEN_KERNEL,
    FROZEN_SCHEDULER_SHA256,
    FROZEN_WORK_UNITS,
)
from scripts.run_sched_ext_business_smoke import _policy_evidence, _stats_delta
from scripts.verify_real_sched_ext import kernel_log_errors


CHECK_NAMES = (
    "required_files",
    "content_seal",
    "schema_and_formal_boundary",
    "frozen_topology",
    "frozen_build_identity",
    "preflight",
    "fixture_verified",
    "warmup_plan_complete",
    "formal_plan_complete",
    "paired_seeds_and_rotation",
    "events_complete",
    "metrics_recomputed",
    "business_checksums_consistent",
    "runtime_evidence",
    "loader_stats_recomputed",
    "warmup_excluded_from_statistics",
    "statistics_and_primary_gate_recomputed",
    "kernel_clean",
    "summary_consistent",
)


def _same_number(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-9)
    except (TypeError, ValueError):
        return False


def _expected_keys(sample_kind: str) -> list[tuple[str, int]]:
    return [
        (str(item["mode"]), int(item["round"]))
        for item in build_formal_plan(sample_kind)
    ]


def verify_formal_bundle(bundle: Path) -> dict[str, Any]:
    bundle = Path(bundle)
    checks = {name: False for name in CHECK_NAMES}
    errors: dict[str, str] = {}
    required = {
        "SHA256SUMS",
        "benchmark_raw.json",
        "benchmark_summary.csv",
        "benchmark_summary.json",
        "build_manifest.json",
        "environment.json",
        "events.jsonl",
        "fixture_manifest.json",
        "kernel_window.txt",
        "loader_events.jsonl",
        "preflight.json",
        "primary_gate.json",
        "summary.json",
        "trials.jsonl",
        "warmup_events.jsonl",
        "warmup_loader_events.jsonl",
        "warmup_trials.jsonl",
    }
    try:
        checks["required_files"] = bundle.is_dir() and all(
            (bundle / name).is_file() for name in required
        )
        checks["content_seal"] = checks["required_files"] and verify_seal(bundle)
        environment = json.loads((bundle / "environment.json").read_text(encoding="utf-8"))
        build = json.loads((bundle / "build_manifest.json").read_text(encoding="utf-8"))
        preflight = json.loads((bundle / "preflight.json").read_text(encoding="utf-8"))
        summary = json.loads((bundle / "summary.json").read_text(encoding="utf-8"))
        stored_primary_gate = json.loads(
            (bundle / "primary_gate.json").read_text(encoding="utf-8")
        )
        manifest = FixtureManifest.from_dict(
            json.loads((bundle / "fixture_manifest.json").read_text(encoding="utf-8"))
        )
        checks["schema_and_formal_boundary"] = (
            environment.get("schema") == summary.get("schema") == FORMAL_SCHEMA
            and environment.get("phase") == summary.get("phase") == "formal"
            and environment.get("formal") is True
            and summary.get("formal") is True
            and environment.get("warmup_in_formal_statistics") is False
        )
        checks["frozen_topology"] = (
            environment.get("host") == FROZEN_HOST
            and environment.get("kernel_release") == FROZEN_KERNEL
            and environment.get("controller_cpu") == 0
            and environment.get("worker_cpus") == [1, 2]
            and environment.get("pressure_cpu") == 3
            and environment.get("modes") == list(FORMAL_MODES)
            and environment.get("warmup_rounds_per_mode") == WARMUP_ROUNDS
            and environment.get("formal_rounds_per_mode") == FORMAL_ROUNDS
        )
        checks["frozen_build_identity"] = (
            build.get("schema") == FORMAL_SCHEMA
            and build.get("scheduler_sha256") == FROZEN_SCHEDULER_SHA256
            and build.get("fixture_sha256") == FROZEN_FIXTURE_SHA256
            and build.get("fixture_version") == 2
            and build.get("work_units") == FROZEN_WORK_UNITS
            and build.get("primary_baseline") == PRIMARY_BASELINE
            and build.get("primary_candidate") == PRIMARY_CANDIDATE
            and build.get("primary_metric") == PRIMARY_METRIC
        )
        checks["preflight"] = preflight.get("accepted") is True
        try:
            verify_fixture(bundle / "fixture", manifest)
            checks["fixture_verified"] = manifest.aggregate_sha256 == FROZEN_FIXTURE_SHA256
        except Exception as exc:
            errors["fixture_verified"] = str(exc)

        warmup_trials = read_jsonl(bundle / "warmup_trials.jsonl")
        formal_trials = read_jsonl(bundle / "trials.jsonl")
        warmup_expected = _expected_keys("warmup")
        formal_expected = _expected_keys("formal")
        warmup_actual = [
            (str(item.get("mode", "")), int(item.get("round", -1)))
            for item in warmup_trials
        ]
        formal_actual = [
            (str(item.get("mode", "")), int(item.get("round", -1)))
            for item in formal_trials
        ]
        checks["warmup_plan_complete"] = warmup_actual == warmup_expected
        checks["formal_plan_complete"] = formal_actual == formal_expected

        base_seed = int(build.get("seed", -1))
        seed_rotation_ok = True
        for sample_kind, trials, plan in (
            ("warmup", warmup_trials, build_formal_plan("warmup")),
            ("formal", formal_trials, build_formal_plan("formal")),
        ):
            if len(trials) != len(plan):
                seed_rotation_ok = False
                continue
            for trial, item in zip(trials, plan, strict=True):
                seed_rotation_ok &= trial.get("sample_kind") == sample_kind
                seed_rotation_ok &= trial.get("formal") is True
                seed_rotation_ok &= trial.get("schema") == FORMAL_SCHEMA
                seed_rotation_ok &= trial.get("mode") == item["mode"]
                seed_rotation_ok &= trial.get("round") == item["round"]
                seed_rotation_ok &= trial.get("order") == item["order"]
                seed_rotation_ok &= trial.get("seed") == formal_round_seed(
                    base_seed, int(item["round"]), sample_kind=sample_kind
                )
                seed_rotation_ok &= set(trial.get("work_units", {}).values()) == {
                    FROZEN_WORK_UNITS
                }
        checks["paired_seeds_and_rotation"] = bool(seed_rotation_ok)

        record_sets = {
            "warmup": (
                warmup_trials,
                read_jsonl(bundle / "warmup_events.jsonl"),
                read_jsonl(bundle / "warmup_loader_events.jsonl"),
                set(warmup_expected),
            ),
            "formal": (
                formal_trials,
                read_jsonl(bundle / "events.jsonl"),
                read_jsonl(bundle / "loader_events.jsonl"),
                set(formal_expected),
            ),
        }
        metrics_ok = checks["warmup_plan_complete"] and checks["formal_plan_complete"]
        evidence_ok = bool(warmup_trials and formal_trials)
        loader_ok = bool(warmup_trials and formal_trials)
        events_ok = True
        output_checksums: set[str] = set()
        for sample_kind, (trials, events, loader_events, expected_keys) in record_sets.items():
            event_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
            for event in events:
                key = (
                    str(event.get("benchmark_mode", "")),
                    int(event.get("benchmark_round", -1)),
                )
                events_ok &= event.get("sample_kind") == sample_kind
                event_groups[key].append(event)
            events_ok &= set(event_groups) == expected_keys
            loader_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
            for event in loader_events:
                key = (
                    str(event.get("benchmark_mode", "")),
                    int(event.get("benchmark_round", -1)),
                )
                loader_ok &= event.get("sample_kind") == sample_kind
                loader_groups[key].append(event)
            for trial in trials:
                key = (str(trial["mode"]), int(trial["round"]))
                task_results = trial.get("task_results")
                work_units = trial.get("work_units")
                if not isinstance(task_results, dict) or not isinstance(work_units, dict):
                    metrics_ok = False
                    continue
                recalculated = compute_trial_metrics(
                    event_groups[key],
                    task_results,
                    deadlines_ms={"patch_validation": 2_000, "security_gate": 2_000},
                    work_units=work_units,
                )
                for name in (
                    "critical_ready_ms",
                    "makespan_ms",
                    "deadline_hit_rate",
                    "tasks_per_second",
                    "jain_fairness",
                ):
                    metrics_ok &= _same_number(recalculated[name], trial.get(name))
                metrics_ok &= recalculated["output_checksum"] == trial.get("output_checksum")
                output_checksums.add(str(trial.get("output_checksum", "")))
                raw_evidence = assess_trial_evidence(trial)
                associations, queues = _policy_evidence(event_groups[key])
                evidence_ok &= associations == trial.get("policy_associations")
                evidence_ok &= queues == trial.get("policy_queues")
                evidence_ok &= all(raw_evidence.values())
                evidence_ok &= raw_evidence == trial.get("evidence_checks")
                if key[0] == "cfs":
                    loader_ok &= key not in loader_groups and trial.get("stats_delta") == {}
                else:
                    group = loader_groups.get(key, [])
                    loader_ok &= bool(group)
                    loader_ok &= _stats_delta(group) == trial.get("stats_delta")
                    loader_ok &= any(event.get("event") == "ready" for event in group)
                    loader_ok &= any(event.get("event") == "final_stats" for event in group)
        checks["events_complete"] = bool(events_ok)
        checks["metrics_recomputed"] = bool(metrics_ok)
        checks["business_checksums_consistent"] = (
            len(output_checksums) == 1 and "" not in output_checksums
        )
        checks["runtime_evidence"] = bool(evidence_ok)
        checks["loader_stats_recomputed"] = bool(loader_ok)

        benchmark_raw = json.loads(
            (bundle / "benchmark_raw.json").read_text(encoding="utf-8")
        )
        checks["warmup_excluded_from_statistics"] = (
            benchmark_raw.get("schema") == FORMAL_SCHEMA
            and benchmark_raw.get("sample_kind") == "formal"
            and benchmark_raw.get("trials") == formal_trials
            and all(item.get("sample_kind") == "formal" for item in formal_trials)
            and all(item.get("sample_kind") == "warmup" for item in warmup_trials)
        )
        recomputed_summary = {
            "modes": summarise_trials(formal_trials),
            "paired_improvements": {
                mode: paired_improvements(formal_trials, baseline="cfs", candidate=mode)
                for mode in FORMAL_MODES
                if mode != "cfs"
            },
        }
        stored_summary = json.loads(
            (bundle / "benchmark_summary.json").read_text(encoding="utf-8")
        )
        recomputed_primary_gate = evaluate_primary_gate(formal_trials)
        checks["statistics_and_primary_gate_recomputed"] = (
            stored_summary == recomputed_summary
            and stored_primary_gate == recomputed_primary_gate
        )

        kernel_text = (bundle / "kernel_window.txt").read_text(
            encoding="utf-8", errors="replace"
        )
        checks["kernel_clean"] = (
            kernel_log_errors(kernel_text.splitlines()) == []
            and summary.get("kernel_log_errors") == []
            and summary.get("systemd_failed_units") == 0
            and summary.get("final_sched_ext_state") == "disabled"
            and summary.get("final_switch_all") == "0"
        )
        checks["summary_consistent"] = (
            summary.get("accepted") is True
            and summary.get("performance_gate_passed")
            == recomputed_primary_gate.get("passed")
            and summary.get("expected_warmup_trials") == len(warmup_expected)
            and summary.get("completed_warmup_trials") == len(warmup_trials)
            and summary.get("expected_formal_trials") == len(formal_expected)
            and summary.get("completed_formal_trials") == len(formal_trials)
            and summary.get("fixture_sha256") == manifest.aggregate_sha256
            and summary.get("output_checksum") in output_checksums
            and summary.get("task_loss") == 0
            and summary.get("failure_reason") is None
        )
    except Exception as exc:
        errors["verification"] = f"{type(exc).__name__}: {exc}"

    failed = [name for name in CHECK_NAMES if not checks[name]]
    return {
        "passed": not failed,
        "check_only": True,
        "checks": f"{len(CHECK_NAMES) - len(failed)}/{len(CHECK_NAMES)}",
        "failed_checks": failed,
        "errors": errors,
        "formal": True,
        "performance_gate_passed": (
            stored_primary_gate.get("passed")
            if "stored_primary_gate" in locals()
            else False
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args(argv)
    result = verify_formal_bundle(args.bundle)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
