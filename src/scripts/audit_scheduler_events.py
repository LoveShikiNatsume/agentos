"""Audit scheduler events from an EventLog JSONL file.

Usage:
    .venv/bin/python scripts/audit_scheduler_events.py runs/demo/events.jsonl
    .venv/bin/python scripts/audit_scheduler_events.py runs/demo/events.jsonl --out runs/defense/scheduler_audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentos.scheduler.audit import audit_scheduler_events


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit scheduler events from an EventLog JSONL file"
    )
    parser.add_argument(
        "input",
        help="Path to events.jsonl file",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON file path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Read events from JSONL
    events = []
    for line_num, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            events.append(event)
        except json.JSONDecodeError as e:
            print(f"Warning: skipping invalid JSON on line {line_num}: {e}", file=sys.stderr)

    # Audit
    result = audit_scheduler_events(events)

    # Output
    output = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    print(output)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output + "\n", encoding="utf-8")
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
