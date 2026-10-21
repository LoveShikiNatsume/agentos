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
from agentos.benchmarks.stage5 import (
    GROUP_PROFILES,
    LANES,
    REQUIRED_METRICS,
    SCHEMA_VERSION,
    WORKLOAD_NAMES,
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
    )
    for name in required:
        if not (output_dir / name).exists():
            errors.append(f"missing artifact: {name}")
    if errors:
        return {"passed": False, "errors": errors, "checks": [], "counts": {}}

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
            "checks": [],
            "counts": {},
        }

    if summary.get("schema_version") != SCHEMA_VERSION:
        errors.append("summary schema_version mismatch")
    metadata = summary.get("metadata", {})
    repetitions = metadata.get("repetitions")
    if not isinstance(repetitions, int) or repetitions < 10:
        errors.append("formal repetitions must be at least 10")
        repetitions = 0
    if set(metadata.get("workloads", [])) != set(WORKLOAD_NAMES):
        errors.append("workload set mismatch")
    if set(metadata.get("groups", [])) != set(GROUP_PROFILES):
        errors.append("group set mismatch")
    if set(metadata.get("lanes", [])) != set(LANES):
        errors.append("lane set mismatch")

    plan_ids = [entry.get("trial_id") for entry in plan]
    raw_ids = [entry.get("trial_id") for entry in raw]
    if len(plan_ids) != len(set(plan_ids)):
        errors.append("duplicate trial ID in plan")
    if len(raw_ids) != len(set(raw_ids)):
        errors.append("duplicate trial ID in raw trials")
    if plan_ids != raw_ids:
        errors.append("raw trial order or IDs do not match plan")

    measured = [row for row in raw if not row.get("warmup")]
    warmups = [row for row in raw if row.get("warmup")]
    expected_measured = len(WORKLOAD_NAMES) * len(LANES) * 5 * repetitions
    expected_warmups = len(WORKLOAD_NAMES) * len(LANES) * 5
    if len(measured) != expected_measured:
        errors.append(
            f"measured trial count mismatch: {len(measured)} != {expected_measured}"
        )
    if len(warmups) != expected_warmups:
        errors.append(
            f"warmup count mismatch: {len(warmups)} != {expected_warmups}"
        )
    if any(row.get("trial_status") != "recorded" for row in measured):
        errors.append("one or more measured trials have harness errors")
    if any(
        row.get("actual_position") != row.get("planned_position")
        for row in raw
    ):
        errors.append("actual execution position differs from plan")

    orders: dict[tuple[str, str], set[tuple[str, ...]]] = defaultdict(set)
    blocks: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in measured:
        blocks[
            (str(row["workload"]), str(row["lane"]), int(row["repetition"]))
        ].append(row)
    for (workload, lane, _repetition), rows in blocks.items():
        ordered = sorted(rows, key=lambda row: row["planned_position"])
        groups = tuple(str(row["group"]) for row in ordered)
        if set(groups) != set(GROUP_PROFILES) or len(groups) != 5:
            errors.append(f"incomplete randomized block: {workload}/{lane}")
        orders[(workload, lane)].add(groups)
    for key, group_orders in orders.items():
        if len(group_orders) < 2:
            errors.append(f"group order was not randomized for {key[0]}/{key[1]}")

    manifests = summary.get("capability_manifests", {})
    expected_manifests = {
        name: profile.to_dict() for name, profile in GROUP_PROFILES.items()
    }
    if manifests != expected_manifests:
        errors.append("capability manifests mismatch")
    boundary = summary.get("claim_boundary", {})
    if boundary.get("real_cgroup_control") is not False:
        errors.append("cache-sim evidence must not claim real cgroup control")
    if boundary.get("real_pinned_bpf_map") is not False:
        errors.append("cache-sim evidence must not claim real pinned BPF map")
    if boundary.get("sched_ext_active") is not False:
        errors.append("cache-sim evidence must not claim sched_ext")

    cells = summary.get("cells", {})
    for workload in WORKLOAD_NAMES:
        for lane in LANES:
            for group, profile in GROUP_PROFILES.items():
                rows = [
                    row
                    for row in measured
                    if row.get("workload") == workload
                    and row.get("lane") == lane
                    and row.get("group") == group
                    and row.get("trial_status") == "recorded"
                ]
                if len(rows) != repetitions:
                    errors.append(
                        f"cell count mismatch: {workload}/{lane}/{group}"
                    )
                    continue
                if any(row.get("capabilities") != profile.to_dict() for row in rows):
                    errors.append(
                        f"trial capability mismatch: {workload}/{lane}/{group}"
                    )
                if group == "workload_aware" and any(
                    row["evidence"].get("pressure_control_applied") is not False
                    for row in rows
                ):
                    errors.append("observation-only group claims pressure control")
                if group in {"feedback_control", "full_runtime"} and any(
                    row["evidence"].get("pressure_control_applied") is not True
                    for row in rows
                ):
                    errors.append(f"{group} lacks pressure control evidence")

                cell = (
                    cells.get(workload, {})
                    .get(lane, {})
                    .get(group, {})
                )
                for metric in REQUIRED_METRICS:
                    try:
                        recomputed = describe(
                            [float(row["metrics"][metric]) for row in rows]
                        )
                        recorded = cell["metrics"][metric]
                    except (KeyError, TypeError, ValueError) as exc:
                        errors.append(
                            f"metric missing: {workload}/{lane}/{group}/{metric}: {exc}"
                        )
                        continue
                    for field, expected in recomputed.items():
                        actual = recorded.get(field)
                        if not isinstance(actual, (int, float)) or not math.isclose(
                            float(actual),
                            float(expected),
                            rel_tol=1e-9,
                            abs_tol=1e-12,
                        ):
                            errors.append(
                                f"{field} mismatch: "
                                f"{workload}/{lane}/{group}/{metric}"
                            )

    event_files = list((output_dir / "events").glob("*.jsonl"))
    if len(event_files) != len(raw):
        errors.append("event file count does not match raw trial count")
    report = (output_dir / "report.md").read_text(encoding="utf-8")
    for marker in ("Student-t", "not real cgroup", "No BPF program"):
        if marker not in report:
            errors.append(f"report missing boundary marker: {marker}")

    return {
        "passed": not errors,
        "errors": errors,
        "checks": [
            "schema",
            "trial_integrity",
            "randomization",
            "capability_boundaries",
            "statistics_recomputed",
            "artifacts",
        ],
        "counts": {
            "plan_entries": len(plan),
            "raw_trials": len(raw),
            "measured_trials": len(measured),
            "warmups": len(warmups),
            "event_files": len(event_files),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/member_a_stage5"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = verify_directory(args.out)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for error in result["errors"]:
            print(f"[FAIL] {error}")
        print("[PASS] stage5 strict evidence" if result["passed"] else "[FAIL] stage5 strict evidence")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
