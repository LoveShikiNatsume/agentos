#!/usr/bin/env python3
"""Start the AgentOS dashboard server with optional preloaded events.

Usage:
    DASHBOARD_EVENT_LOG=runs/software_eng_real_merged/events.jsonl \
    DASHBOARD_WORKLOAD_MATRIX=runs/workload_matrix_deepseek_real/summary.json \
    CLOUD_BENCHMARK_JSON=runs/cloud_framework_benchmark_1782191230.json \
    python scripts/run_dashboard.py [--host 127.0.0.1] [--port 9400]

If CLOUD_BENCHMARK_JSON is not set, agentos.dashboard.cloud_benchmark falls
back to its own pinned exploratory-pilot file (see that module's docstring) — this script no longer auto-selects "whatever
file sorts last by filename", since that silently changed what the dashboard
displayed as new pilot files were dropped into runs/.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agentos.dashboard.broadcast import EventBroadcaster
from agentos.dashboard.server import preload_events, run_dashboard_server


def main() -> int:
    parser = argparse.ArgumentParser(description="AgentOS live dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=9400, help="Server port")
    args = parser.parse_args()

    event_log_path = os.environ.get("DASHBOARD_EVENT_LOG")
    workload_matrix_path = os.environ.get("DASHBOARD_WORKLOAD_MATRIX")
    cloud_json = os.environ.get("CLOUD_BENCHMARK_JSON")

    broadcaster = EventBroadcaster()

    # Preload historical events from event log
    if event_log_path:
        count = preload_events(broadcaster, Path(event_log_path))
        print(f"  Preloaded {count} events from {event_log_path}")
    else:
        print("  No DASHBOARD_EVENT_LOG set, starting with empty history")

    # Store workload matrix path for the /api/workload-matrix route
    if workload_matrix_path:
        os.environ["DASHBOARD_WORKLOAD_MATRIX"] = workload_matrix_path
        print(f"  Workload matrix: {workload_matrix_path}")
    else:
        print("  No DASHBOARD_WORKLOAD_MATRIX set, /api/workload-matrix will return 404")

    # Print cloud benchmark source
    if cloud_json:
        print(f"  Cloud benchmark: {cloud_json}")
    else:
        print("  CLOUD_BENCHMARK_JSON not set, /api/cloud-benchmark will use the pinned "
              "exploratory-pilot default in agentos/dashboard/cloud_benchmark.py")

    print()
    asyncio.run(run_dashboard_server(broadcaster, host=args.host, port=args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
