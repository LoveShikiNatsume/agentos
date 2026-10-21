#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"

if [[ $# -gt 1 ]]; then
  echo "用法：$0 [输出目录]" >&2
  exit 2
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
  python_bin="$repo_root/.venv/bin/python"
else
  python_bin="python3"
fi

if [[ $# -eq 1 ]]; then
  output_dir="$1"
  [[ "$output_dir" = /* ]] || output_dir="$repo_root/$output_dir"
else
  output_dir="$repo_root/demo/output/离线功能演示"
fi

if [[ -e "$output_dir" ]]; then
  echo "输出目录已存在，为避免混合批次而停止：$output_dir" >&2
  exit 2
fi

mkdir -p "$(dirname -- "$output_dir")"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

"$python_bin" -m agentos.cli finals-demo --output "$output_dir"

# The generated panel only displays the source log path. Keep the archived
# result portable when the repository is opened on another machine.
"$python_bin" - "$output_dir" <<'PY'
from pathlib import Path
import sys

output = Path(sys.argv[1]).resolve()
dashboard = output / "dashboard.html"
text = dashboard.read_text(encoding="utf-8")
text = text.replace(str(output / "events.jsonl"), "events.jsonl")
dashboard.write_text(text, encoding="utf-8")
PY

"$python_bin" "$script_dir/verify_offline_bundle.py" --output "$output_dir"

echo "离线验证通过，结果目录：$output_dir"
