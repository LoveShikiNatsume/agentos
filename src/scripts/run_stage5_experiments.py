#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentos.benchmarks.stage5 import (
    WORKLOAD_NAMES,
    aggregate_trials,
    build_trial_plan,
    run_trial,
    write_artifacts,
)
from scripts.verify_stage5_experiments import verify_directory


async def run_experiment(
    *,
    output_dir: Path,
    repetitions: int,
    seed: int,
    cache_latency_ms: float,
    overwrite: bool,
) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"output directory is not empty: {output_dir}; use --overwrite"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    incomplete = output_dir / ".incomplete"
    incomplete.write_text("stage5 experiment in progress\n", encoding="utf-8")

    plan = build_trial_plan(
        WORKLOAD_NAMES,
        repetitions=repetitions,
        seed=seed,
    )
    trials: list[dict] = []
    live_path = output_dir / "raw_trials.live.jsonl"
    with live_path.open("w", encoding="utf-8") as live:
        for index, entry in enumerate(plan, start=1):
            result = await run_trial(
                entry,
                cache_latency_ms=cache_latency_ms,
            )
            trials.append(result)
            raw = dict(result)
            raw.pop("events", None)
            live.write(json.dumps(raw, sort_keys=True) + "\n")
            live.flush()
            if index == 1 or index % 25 == 0 or index == len(plan):
                print(
                    f"[{index:03d}/{len(plan)}] "
                    f"{entry.workload}/{entry.lane}/{entry.group}"
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
    return verification


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["cache-sim"], default="cache-sim")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--cache-latency-ms", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=Path("runs/member_a_stage5"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.repetitions < 10:
        print(
            "formal stage5 experiments require at least 10 repetitions",
            file=sys.stderr,
        )
        return 2
    if args.cache_latency_ms <= 0:
        print("--cache-latency-ms must be positive", file=sys.stderr)
        return 2
    try:
        result = asyncio.run(
            run_experiment(
                output_dir=args.out,
                repetitions=args.repetitions,
                seed=args.seed,
                cache_latency_ms=args.cache_latency_ms,
                overwrite=args.overwrite,
            )
        )
    except (FileExistsError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
