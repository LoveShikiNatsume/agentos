#!/usr/bin/env python3
"""Merge multiple events.jsonl files into one, sorted by timestamp.

Each event gets a trace_id prefix (run=base / run=stress / run=cache)
so the dashboard can distinguish which run it came from.

Usage:
    python scripts/merge_events.py \
        runs/software_eng_real_base_<ts>/events.jsonl \
        runs/software_eng_real_stress_<ts>/events.jsonl \
        runs/software_eng_real_cache_<ts>/events.jsonl \
        --output runs/software_eng_real_merged/events.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def merge_events(
    inputs: list[Path],
    output: Path,
    labels: list[str] | None = None,
) -> dict[str, int]:
    """Merge JSONL event files, sorted by timestamp.

    Returns a dict of {event_type: count} for the merged output.
    """
    if labels is None:
        labels = [f"run{i}" for i in range(len(inputs))]

    all_events: list[tuple[float, dict]] = []

    for input_path, label in zip(inputs, labels):
        if not input_path.exists():
            print(f"  WARNING: {input_path} not found, skipping", file=sys.stderr)
            continue
        count = 0
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                # Add trace_id prefix to distinguish runs
                existing_trace = event.get("trace_id") or ""
                event["trace_id"] = f"{label}:{existing_trace}" if existing_trace else label
                ts = float(event.get("timestamp", 0))
                all_events.append((ts, event))
                count += 1
        print(f"  {label}: {count} events from {input_path}")

    # Sort by timestamp
    all_events.sort(key=lambda x: x[0])

    # Write merged output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for _, event in all_events:
            f.write(json.dumps(event, ensure_ascii=False))
            f.write("\n")

    # Count by event_type
    from collections import Counter
    counter: Counter[str] = Counter()
    for _, event in all_events:
        counter[event["event_type"]] += 1

    return dict(counter)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge event JSONL files")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input events.jsonl files (in order: base, stress, cache)",
    )
    parser.add_argument(
        "--output",
        default="runs/software_eng_real_merged/events.jsonl",
        help="Output merged events.jsonl path",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Trace ID labels for each input (default: run0, run1, ...)",
    )
    args = parser.parse_args()

    inputs = [Path(p) for p in args.inputs]
    output = Path(args.output)

    # Default labels based on directory names
    labels = args.labels
    if labels is None:
        labels = []
        for p in inputs:
            # Extract variant name from path like runs/software_eng_real_stress_123/events.jsonl
            parent_name = p.parent.name  # software_eng_real_stress_123
            parts = parent_name.split("_")
            # Find the variant part (base/stress/cache)
            variant = "unknown"
            for candidate in ["base", "stress", "cache"]:
                if candidate in parts:
                    variant = candidate
                    break
            labels.append(variant)

    print(f"=== Merging {len(inputs)} event logs ===")
    counter = merge_events(inputs, output, labels)

    print(f"\n=== Merged event type distribution ({output}) ===")
    for etype, count in sorted(counter.items()):
        print(f"  {etype:<40} {count}")
    print(f"  {'TOTAL':<40} {sum(counter.values())}")

    # Check key scheduler events
    key_events = [
        "scheduler.policy_selected",
        "scheduler.fairness_warning",
        "scheduler.kernel_probe",
        "scheduler.bpf_map_updated",
    ]
    print(f"\n=== Key event checks ===")
    for evt in key_events:
        count = counter.get(evt, 0)
        print(f"  {evt}: {count}")

    print(f"\nMerged output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
