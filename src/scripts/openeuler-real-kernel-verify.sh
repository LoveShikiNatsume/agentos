#!/usr/bin/env bash
set -euo pipefail

json_out="${1:-/evidence/summary.json}"
evidence_dir="$(dirname "$json_out")"

python3 scripts/openeuler-cgroup-verify.py

python3 scripts/verify_real_cgroup_feedback_loop.py \
  --root /sys/fs/cgroup/agentos_stage3_local_psi \
  --out "$evidence_dir/stage3_local_psi"

if [[ "$(stat -fc %T /sys/fs/bpf 2>/dev/null || true)" != "bpf" ]]; then
  mkdir -p /sys/fs/bpf
  mount -t bpf bpf /sys/fs/bpf
fi

exec python3 scripts/verify_real_kernel_path.py \
  --cgroup-root /sys/fs/cgroup/agentos \
  --bpffs-root /sys/fs/bpf \
  --json-out "$json_out"
