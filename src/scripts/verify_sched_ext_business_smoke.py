#!/usr/bin/env python3
"""Read-only strict verifier for P2 four-mode openEuler smoke bundles."""

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
from agentos.benchmarks.sched_ext_business_smoke import (
    SMOKE_MODES,
    assess_trial_evidence,
    build_smoke_plan,
)
from agentos.scenarios.sched_ext_business.fixture import FixtureManifest, verify_fixture
from agentos.scenarios.sched_ext_business.metrics import compute_trial_metrics
from scripts.run_sched_ext_business_smoke import (
    SMOKE_SCHEMA,
    _policy_evidence,
    _stats_delta,
)
from scripts.verify_real_sched_ext import kernel_log_errors


CHECK_NAMES = (
    "required_files",
    "content_seal",
    "schema",
    "authorized_phase",
    "fixed_topology",
    "preflight",
    "fixture_verified",
    "trial_plan_complete",
    "events_complete",
    "metrics_recomputed",
    "business_checksums_consistent",
    "runtime_evidence",
    "loader_stats_recomputed",
    "kernel_clean",
    "summary_consistent",
)


def _same_number(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-9)
    except (TypeError, ValueError):
        return False


def verify_smoke_bundle(bundle: Path) -> dict[str, Any]:
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
        "summary.json",
        "trials.jsonl",
    }
    try:
        checks["required_files"] = bundle.is_dir() and all(
            (bundle / name).is_file() for name in required
        )
        checks["content_seal"] = checks["required_files"] and verify_seal(bundle)
        environment = json.loads((bundle / "environment.json").read_text(encoding="utf-8"))
        preflight = json.loads((bundle / "preflight.json").read_text(encoding="utf-8"))
        summary = json.loads((bundle / "summary.json").read_text(encoding="utf-8"))
        manifest = FixtureManifest.from_dict(
            json.loads((bundle / "fixture_manifest.json").read_text(encoding="utf-8"))
        )
        checks["schema"] = (
            environment.get("schema") == summary.get("schema") == SMOKE_SCHEMA
        )
        phase = str(summary.get("phase", ""))
        plan = build_smoke_plan(phase)
        checks["authorized_phase"] = (
            environment.get("phase") == phase
            and environment.get("formal") is False
            and summary.get("formal") is False
            and len(plan) in {4, 12}
        )
        checks["fixed_topology"] = (
            environment.get("kernel_release") == "6.12.95-agentos-scx"
            and environment.get("controller_cpu") == 0
            and environment.get("worker_cpus") == [1, 2]
            and environment.get("pressure_cpu") == 3
            and environment.get("modes") == list(SMOKE_MODES)
        )
        checks["preflight"] = preflight.get("accepted") is True
        try:
            verify_fixture(bundle / "fixture", manifest)
            checks["fixture_verified"] = True
        except Exception as exc:
            errors["fixture_verified"] = str(exc)

        trials = read_jsonl(bundle / "trials.jsonl")
        expected_keys = [(str(item["mode"]), int(item["round"])) for item in plan]
        actual_keys = [(str(item.get("mode")), int(item.get("round", -1))) for item in trials]
        checks["trial_plan_complete"] = actual_keys == expected_keys

        events = read_jsonl(bundle / "events.jsonl")
        event_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            key = (str(event.get("benchmark_mode")), int(event.get("benchmark_round", -1)))
            event_groups[key].append(event)
        checks["events_complete"] = set(event_groups) == set(expected_keys)

        loader_events = read_jsonl(bundle / "loader_events.jsonl")
        loader_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for event in loader_events:
            key = (str(event.get("benchmark_mode")), int(event.get("benchmark_round", -1)))
            loader_groups[key].append(event)

        metrics_ok = checks["trial_plan_complete"] and checks["events_complete"]
        evidence_ok = bool(trials)
        loader_ok = bool(trials)
        output_checksums: set[str] = set()
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
        checks["metrics_recomputed"] = bool(metrics_ok)
        checks["business_checksums_consistent"] = (
            len(output_checksums) == 1 and "" not in output_checksums
        )
        checks["runtime_evidence"] = bool(evidence_ok)
        checks["loader_stats_recomputed"] = bool(loader_ok)

        kernel_text = (bundle / "kernel_window.txt").read_text(
            encoding="utf-8", errors="replace"
        )
        recomputed_kernel_errors = kernel_log_errors(kernel_text.splitlines())
        checks["kernel_clean"] = (
            recomputed_kernel_errors == []
            and summary.get("kernel_log_errors") == []
            and summary.get("systemd_failed_units") == 0
            and summary.get("final_sched_ext_state") == "disabled"
        )
        checks["summary_consistent"] = (
            summary.get("accepted") is True
            and summary.get("expected_trials") == len(plan)
            and summary.get("completed_trials") == len(trials) == len(plan)
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
        "formal": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args(argv)
    result = verify_smoke_bundle(args.bundle)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
