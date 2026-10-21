#!/usr/bin/env bash
set -euo pipefail

AGENTOS_RUNTIME_ROOT="${AGENTOS_RUNTIME_ROOT:-$HOME/os2026-runtime}"
AGENTOS_VLLM_PYTHON="${AGENTOS_VLLM_PYTHON:-$HOME/venvs/os2026-vllm/bin/python}"
AGENTOS_EXEC_CACHE="${AGENTOS_EXEC_CACHE:-$HOME/.cache/os2026-exec}"
AGENTOS_MODEL_PATH="${AGENTOS_MODEL_PATH:-$HOME/models/Qwen3.8-27B-AWQ-INT4}"
AGENTOS_VLLM_PORT="${AGENTOS_VLLM_PORT:-18094}"

export TMPDIR="$AGENTOS_EXEC_CACHE/tmp"
export HF_HOME="$AGENTOS_RUNTIME_ROOT/cache/huggingface"
export VLLM_CACHE_ROOT="$AGENTOS_EXEC_CACHE/vllm"
export TRITON_CACHE_DIR="$AGENTOS_EXEC_CACHE/triton"
export TORCH_HOME="$AGENTOS_RUNTIME_ROOT/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$AGENTOS_EXEC_CACHE/torchinductor"
export CUDA_CACHE_PATH="$AGENTOS_EXEC_CACHE/cuda"
export FLASHINFER_WORKSPACE_BASE="$AGENTOS_EXEC_CACHE/flashinfer"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
# Use the supported native sampler; this host has no nvcc toolchain.
export VLLM_USE_FLASHINFER_SAMPLER=0
export DO_NOT_TRACK=1
mkdir -p "$TMPDIR" "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCH_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$FLASHINFER_WORKSPACE_BASE"

exec "$AGENTOS_VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$AGENTOS_MODEL_PATH" \
  --served-model-name qwen3.8-27b-int4 \
  --host 127.0.0.1 --port "$AGENTOS_VLLM_PORT" \
  --language-model-only --dtype bfloat16 \
  --max-model-len 8192 --max-num-seqs 2 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.90 --enforce-eager \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --reasoning-parser qwen3 \
  "$@"
