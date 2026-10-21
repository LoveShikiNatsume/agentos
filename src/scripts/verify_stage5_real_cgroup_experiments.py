#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentos.benchmarks.experiment_stats import describe
from agentos.benchmarks.real_cgroup_experiment import (
    METRICS,
    MODES,
    SCENARIOS,
    SCHEMA_VERSION,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_directory(output_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    required = (
        "trial_plan.json",
        "raw_trials.jsonl",
        "summary.json",
        "summary.csv",
        "report.md",
        "environment.txt",
    )
    for name in required:
        if not (output_dir / name).exists():
            errors.append(f"missing artifact: {name}")
    if errors:
        return {
            "passed": False,
            "errors": errors,
            "counts": {},
            "enforcement": {},
        }

    try:
        plan = _load_json(output_dir / "trial_plan.json")
        summary = _load_json(output_dir / "summary.json")
        raw = [
            json.loads(line)
            for line in (output_dir / "raw_trials.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "passed": False,
            "errors": [f"artifact parse failed: {type(exc).__name__}: {exc}"],
            "counts": {},
            "enforcement": {},
        }

    metadata = summary.get("metadata", {})
    repetitions = metadata.get("repetitions")
    if summary.get("schema_version") != SCHEMA_VERSION:
        errors.append("summary schema_version mismatch")
    if not isinstance(repetitions, int) or repetitions < 10:
        errors.append("formal repetitions must be at least 10")
        repetitions = 0
    if tuple(metadata.get("scenarios", ())) != SCENARIOS:
        errors.append("scenario set or order mismatch")
    if tuple(metadata.get("modes", ())) != MODES:
        errors.append("mode set or order mismatch")
    boundary = summary.get("claim_boundary", {})
    if boundary != {
        "real_cgroup_v2": True,
        "bpf_program_attached": False,
        "sched_ext_active": False,
        "kernel_scheduling_takeover": False,
    }:
        errors.append("capability boundary mismatch")

    plan_ids = [row.get("trial_id") for row in plan]
    raw_ids = [row.get("trial_id") for row in raw]
    if plan_ids != raw_ids:
        errors.append("raw trial IDs or order do not match plan")
    if len(plan_ids) != len(set(plan_ids)):
        errors.append("duplicate trial ID")

    measured = [row for row in raw if not row.get("warmup")]
    warmups = [row for row in raw if row.get("warmup")]
    expected_measured = len(SCENARIOS) * len(MODES) * repetitions
    if len(measured) != expected_measured:
        errors.append(
            f"measured trial count mismatch: {len(measured)} != {expected_measured}"
        )
    if len(warmups) != len(SCENARIOS) * len(MODES):
        errors.append("warmup count mismatch")
    if any(row.get("trial_status") != "recorded" for row in measured):
        errors.append("one or more measured trials have harness errors")
    if any(
        row.get("actual_position") != row.get("planned_position")
        for row in raw
    ):
        errors.append("actual order differs from trial plan")

    blocks: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    orders: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for row in measured:
        blocks[(str(row["scenario"]), int(row["repetition"]))].append(row)
    for (scenario, _repetition), rows in blocks.items():
        ordered = sorted(rows, key=lambda row: row["planned_position"])
        modes = tuple(str(row["mode"]) for row in ordered)
        if set(modes) != set(MODES) or len(modes) != 2:
            errors.append(f"incomplete paired block: {scenario}")
        orders[scenario].add(modes)
    for scenario in SCENARIOS:
        if len(orders[scenario]) < 2:
            errors.append(f"mode order was not randomized: {scenario}")

    if any(row.get("pid_membership_verified") is not True for row in measured):
        errors.append("one or more measured PID membership checks failed")
    if any(row.get("limit_readback_verified") is not True for row in measured):
        errors.append("one or more controlled limit readbacks failed")
    if any(int(row.get("cleanup_failed", 0)) != 0 for row in measured):
        errors.append("one or more measured cgroup cleanup operations failed")
    if any(
        row.get("pressure_source") != "cgroup"
        or set(row.get("pressure_sources", {}).values()) != {"cgroup"}
        for row in measured
    ):
        errors.append("one or more pressure samples were not cgroup-local")

    cells = summary.get("cells", {})
    for scenario in SCENARIOS:
        for mode in MODES:
            rows = [
                row
                for row in measured
                if row.get("scenario") == scenario
                and row.get("mode") == mode
                and row.get("trial_status") == "recorded"
            ]
            if len(rows) != repetitions:
                errors.append(f"cell count mismatch: {scenario}/{mode}")
                continue
            cell = cells.get(scenario, {}).get(mode, {})
            for metric in METRICS:
                try:
                    recomputed = describe(
                        [float(row["metrics"][metric]) for row in rows]
                    )
                    recorded = cell["metrics"][metric]
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append(
                        f"metric missing: {scenario}/{mode}/{metric}: {exc}"
                    )
                    continue
                for field, expected in recomputed.items():
                    actual = recorded.get(field)
                    if not isinstance(actual, (int, float)) or not math.isclose(
                        float(actual),
                        float(expected),
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    ):
                        errors.append(
                            f"{field} mismatch: {scenario}/{mode}/{metric}"
                        )

    def rows_for(scenario: str, mode: str) -> list[dict[str, Any]]:
        return [
            row
            for row in measured
            if row.get("scenario") == scenario and row.get("mode") == mode
        ]

    cpu_unrestricted = rows_for("cpu", "unrestricted")
    cpu_controlled = rows_for("cpu", "controlled")
    unrestricted_cpu_mean = (
        sum(float(row["metrics"]["cpu_usage_usec"]) for row in cpu_unrestricted)
        / len(cpu_unrestricted)
        if cpu_unrestricted
        else 0.0
    )
    controlled_cpu_mean = (
        sum(float(row["metrics"]["cpu_usage_usec"]) for row in cpu_controlled)
        / len(cpu_controlled)
        if cpu_controlled
        else 0.0
    )
    cpu_ratio = (
        controlled_cpu_mean / unrestricted_cpu_mean
        if unrestricted_cpu_mean > 0
        else math.inf
    )
    if not math.isfinite(cpu_ratio) or cpu_ratio > 0.70:
        errors.append(
            f"CPU enforcement too weak: controlled/unrestricted={cpu_ratio:.3f}"
        )

    memory_unrestricted = rows_for("memory", "unrestricted")
    memory_controlled = rows_for("memory", "controlled")
    unrestricted_memory_successes = sum(
        row.get("exit_code") == 0
        and row.get("worker_result", {}).get("completed") is True
        for row in memory_unrestricted
    )
    memory_oom_trials = sum(
        float(row["metrics"]["oom_kill_count"]) >= 1.0
        for row in memory_controlled
    )
    if unrestricted_memory_successes != len(memory_unrestricted):
        errors.append("unrestricted memory worker did not complete every trial")
    minimum_oom = math.ceil(0.8 * len(memory_controlled))
    if memory_oom_trials < minimum_oom:
        errors.append(
            f"controlled memory enforcement insufficient: {memory_oom_trials} "
            f"< {minimum_oom} OOM trials"
        )

    pids_unrestricted = rows_for("pids", "unrestricted")
    pids_controlled = rows_for("pids", "controlled")
    if any(
        float(row["metrics"]["children_spawned"]) < 16
        for row in pids_unrestricted
    ):
        errors.append("unrestricted PIDs worker did not create all children")
    if any(
        float(row["metrics"]["children_spawned"]) > 7
        or float(row["metrics"]["spawn_failures"]) < 1
        for row in pids_controlled
    ):
        errors.append("controlled PIDs limit was not enforced")

    report = (output_dir / "report.md").read_text(encoding="utf-8")
    for marker in (
        "real cgroup v2",
        "No BPF program",
        "No sched_ext",
        "Student-t",
    ):
        if marker not in report:
            errors.append(f"report missing boundary marker: {marker}")

    return {
        "passed": not errors,
        "errors": errors,
        "checks": [
            "schema",
            "paired_randomization",
            "real_pid_membership",
            "limit_readback",
            "cgroup_local_pressure",
            "cleanup",
            "statistics_recomputed",
            "cpu_enforcement",
            "memory_enforcement",
            "pids_enforcement",
            "claim_boundary",
        ],
        "counts": {
            "plan_entries": len(plan),
            "raw_trials": len(raw),
            "measured_trials": len(measured),
            "warmups": len(warmups),
        },
        "enforcement": {
            "cpu_ratio": cpu_ratio,
            "memory_oom_trials": memory_oom_trials,
            "unrestricted_memory_successes": unrestricted_memory_successes,
            "controlled_pids_max_spawned": max(
                (
                    float(row["metrics"]["children_spawned"])
                    for row in pids_controlled
                ),
                default=0.0,
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/member_a_stage5_real_cgroup"),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = verify_directory(args.out)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for error in result["errors"]:
            print(f"[FAIL] {error}")
        print(
            "[PASS] stage5 real cgroup evidence"
            if result["passed"]
            else "[FAIL] stage5 real cgroup evidence"
        )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
