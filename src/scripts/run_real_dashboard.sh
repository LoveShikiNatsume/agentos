#!/usr/bin/env bash
# One-command startup for the AgentOS dashboard with REAL data.
#
# Usage:
#   export DEEPSEEK_API_KEY=your_key_here
#   bash scripts/run_real_dashboard.sh [--port 9400]
#
# Data sources:
#   - Local events:    runs/software_eng_real_merged/events.jsonl
#   - Workload matrix: runs/workload_matrix_deepseek_real/summary.json
#   - Cloud benchmark: pinned exploratory-pilot default in
#                       agentos/dashboard/cloud_benchmark.py, unless
#                       CLOUD_BENCHMARK_JSON is already set in the environment

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PORT=9400
case "${1:-}" in
  --port) PORT="${2:-9400}" ;;
  [0-9]*) PORT="$1" ;;
  "") ;;
  *) echo "Usage: bash scripts/run_real_dashboard.sh [PORT | --port PORT]" >&2; exit 1 ;;
esac

# --- Validate data files exist ---
EVENT_LOG="runs/software_eng_real_merged/events.jsonl"
WORKLOAD_MATRIX="runs/workload_matrix_deepseek_real/summary.json"

if [ ! -f "$EVENT_LOG" ]; then
  echo "ERROR: $EVENT_LOG not found. Run Phase 1/1.5 first." >&2
  exit 1
fi
if [ ! -f "$WORKLOAD_MATRIX" ]; then
  echo "ERROR: $WORKLOAD_MATRIX not found. Run Phase 1 first." >&2
  exit 1
fi

# --- Check DEEPSEEK_API_KEY (warn, don't fail — dashboard doesn't need it) ---
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  echo "WARNING: DEEPSEEK_API_KEY not set. Dashboard will work but benchmarks won't re-run." >&2
fi

echo "=== AgentOS Dashboard (real data) ==="
echo "  Event log:      $EVENT_LOG ($(wc -l < "$EVENT_LOG") events)"
echo "  Workload matrix: $WORKLOAD_MATRIX"
echo "  Cloud benchmark: ${CLOUD_BENCHMARK_JSON:-<pinned exploratory-pilot default, see cloud_benchmark.py>}"
echo "  Port:           $PORT"
echo ""

# --- Start dashboard ---
DASHBOARD_EVENT_LOG="$EVENT_LOG" \
DASHBOARD_WORKLOAD_MATRIX="$WORKLOAD_MATRIX" \
.venv/bin/python scripts/run_dashboard.py --port "$PORT"
