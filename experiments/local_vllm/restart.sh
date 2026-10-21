#!/usr/bin/env bash
# 按 pid 文件停掉上一个 vLLM，再用当前环境变量和参数重启，等到服务可用为止。
# 用法：LOG=xxx.log AGENTOS_MODEL_PATH=... SERVED_NAME=... restart.sh [额外 vLLM 参数]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="${PIDFILE:-$HOME/agentos-runs/vllm.pid}"
LOG="${LOG:-$HOME/agentos-runs/vllm.log}"
PORT="${AGENTOS_VLLM_PORT:-18095}"
if [[ -f "$PIDFILE" ]]; then
  pid="$(cat "$PIDFILE")"
  kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
fi
for _ in $(seq 60); do
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
  [[ "$used" -lt 1000 ]] && break
  sleep 2
done
export AGENTOS_VLLM_PORT="$PORT"
setsid nohup "$HERE/serve.sh" "$@" > "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
for _ in $(seq 120); do
  if curl -s -m 2 "localhost:$PORT/v1/models" > /dev/null; then
    grep -E "KV cache size|Overriding num_gpu_blocks" "$LOG" | sed 's/.*\] //'
    exit 0
  fi
  if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    tail -5 "$LOG"; exit 1
  fi
  sleep 2
done
echo "vLLM 启动超时"; exit 1
