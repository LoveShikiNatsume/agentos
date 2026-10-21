#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentos.benchmarks.real_cgroup_experiment import (
    aggregate_trials,
    build_trial_plan,
    run_trial,
    write_artifacts,
)
from agentos.isolation.delegation import detect_delegation
from scripts.verify_stage5_real_cgroup_experiments import verify_directory


def _clear_directory(path: Path) -> None:
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def run_experiment(
    *,
    root: Path,
    output_dir: Path,
    repetitions: int,
    seed: int,
    overwrite: bool,
) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"output directory is not empty: {output_dir}; use --overwrite"
            )
        _clear_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    incomplete = output_dir / ".incomplete"
    incomplete.write_text(
        "stage5 real cgroup experiment in progress\n",
        encoding="utf-8",
    )

    plan = build_trial_plan(repetitions=repetitions, seed=seed)
    trials: list[dict] = []
    live_path = output_dir / "raw_trials.live.jsonl"
    with live_path.open("w", encoding="utf-8") as live:
        for index, entry in enumerate(plan, start=1):
            result = run_trial(entry, root=root)
            trials.append(result)
            live.write(json.dumps(result, sort_keys=True) + "\n")
            live.flush()
            if index == 1 or index % 10 == 0 or index == len(plan):
                print(
                    f"[{index:02d}/{len(plan)}] "
                    f"{entry.scenario}/{entry.mode}"
                )

    summary = aggregate_trials(
        trials,
        repetitions=repetitions,
        seed=seed,
    )
    write_artifacts(
        output_dir=output_dir,
        plan=plan,
        trials=trials,
        summary=summary,
    )
    live_path.unlink(missing_ok=True)
    verification = verify_directory(output_dir)
    (output_dir / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    incomplete.unlink(missing_ok=True)
    try:
        root.rmdir()
    except OSError:
        pass
    return verification


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/sys/fs/cgroup/agentos_stage5_real"),
    )
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/member_a_stage5_real_cgroup"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.repetitions < 10:
        print(
            "formal real cgroup experiments require at least 10 repetitions",
            file=sys.stderr,
        )
        return 2
    mode, allowed = detect_delegation()
    if not allowed:
        result = {
            "passed": False,
            "delegation_mode": mode,
            "skip_reason": (
                f"delegation={mode}; set AGENTOS_CGROUP_ALLOW_ROOT=1 to override"
            ),
            "errors": [],
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2
    try:
        result = run_experiment(
            root=args.root,
            output_dir=args.out,
            repetitions=args.repetitions,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    except (FileExistsError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
