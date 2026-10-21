#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$RUNTIME_DIR"

MODE="${1:-help}"
TASKS="runs/swe_bench/tasks_national_30.json"
FRAMEWORKS_ALL="langgraph,autogen,openai_agents,crewai"
FRAMEWORKS_CORE3="langgraph,autogen,openai_agents"

case "$MODE" in
  prepare)
    PYTHONPATH=. .venv/bin/python scripts/swe_bench/prepare_swe_tasks.py \
      --n 30 \
      --min-tok 3000 \
      --max-tok 12000 \
      --out "$TASKS"
    ;;

  mock-smoke)
    PYTHONPATH=. .venv/bin/python scripts/swe_bench/honest_bench.py \
      --backend mock \
      --frameworks "$FRAMEWORKS_ALL" \
      --tasks "$TASKS" \
      --limit 2 \
      --out-dir runs/swe_bench/national_30_mock_smoke
    ;;

  mock-full)
    PYTHONPATH=. .venv/bin/python scripts/swe_bench/honest_bench.py \
      --backend mock \
      --frameworks "$FRAMEWORKS_ALL" \
      --tasks "$TASKS" \
      --out-dir runs/swe_bench/national_30_mock_full
    ;;

  deepseek-limit5)
    if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
      echo "DEEPSEEK_API_KEY is required for deepseek-limit5" >&2
      exit 1
    fi
    PYTHONPATH=. .venv/bin/python scripts/swe_bench/honest_bench.py \
      --backend deepseek \
      --frameworks "$FRAMEWORKS_ALL" \
      --tasks "$TASKS" \
      --limit 5 \
      --out-dir runs/swe_bench/national_30_deepseek_limit5
    ;;

  deepseek-full)
    if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
      echo "DEEPSEEK_API_KEY is required for deepseek-full" >&2
      exit 1
    fi
    PYTHONPATH=. .venv/bin/python scripts/swe_bench/honest_bench.py \
      --backend deepseek \
      --frameworks "$FRAMEWORKS_ALL" \
      --tasks "$TASKS" \
      --out-dir runs/swe_bench/national_30_deepseek_full
    ;;

  deepseek-core3)
    if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
      echo "DEEPSEEK_API_KEY is required for deepseek-core3" >&2
      exit 1
    fi
    PYTHONPATH=. .venv/bin/python scripts/swe_bench/honest_bench.py \
      --backend deepseek \
      --frameworks "$FRAMEWORKS_CORE3" \
      --tasks "$TASKS" \
      --out-dir runs/swe_bench/national_30_deepseek_core3
    ;;

  deepseek-crewai)
    if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
      echo "DEEPSEEK_API_KEY is required for deepseek-crewai" >&2
      exit 1
    fi
    PYTHONPATH=. .venv/bin/python scripts/swe_bench/honest_bench.py \
      --backend deepseek \
      --frameworks crewai \
      --tasks "$TASKS" \
      --out-dir runs/swe_bench/national_30_deepseek_crewai
    ;;

  *)
    cat <<'EOF'
Usage:
  bash run_swe_bench_national.sh prepare
  bash run_swe_bench_national.sh mock-smoke
  bash run_swe_bench_national.sh mock-full
  bash run_swe_bench_national.sh deepseek-limit5
  bash run_swe_bench_national.sh deepseek-full
  bash run_swe_bench_national.sh deepseek-core3
  bash run_swe_bench_national.sh deepseek-crewai

DeepSeek modes require DEEPSEEK_API_KEY in the environment.
EOF
    ;;
esac
