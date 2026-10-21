#!/usr/bin/env bash
# Verify all three real data API endpoints on a running dashboard.
#
# Usage:
#   bash scripts/verify_real_data.sh [port]
#
# Requires: dashboard running on the specified port (default 9400).

set -euo pipefail

PORT="${1:-9400}"
BASE="http://127.0.0.1:$PORT"
PASS=0
FAIL=0

check() {
  local desc="$1" url="$2" expect="$3"
  local body
  body=$(curl -sf "$url" 2>/dev/null) || {
    echo "  ❌ $desc: fetch failed (is dashboard running? Try: bash scripts/run_real_dashboard.sh)"
    FAIL=$((FAIL+1))
    return
  }
  if echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); assert $expect" 2>/dev/null; then
    echo "  ✅ $desc"
    PASS=$((PASS+1))
  else
    echo "  ❌ $desc: assertion failed"
    FAIL=$((FAIL+1))
  fi
}

echo "=== Verifying dashboard at $BASE ==="
echo ""

# --- /api/snapshot ---
echo "[1] /api/snapshot"
check "events > 0" "$BASE/api/snapshot" "len(d['events']) > 0"
check "policy_selected >= 1" "$BASE/api/snapshot" "sum(1 for e in d['events'] if e['event_type']=='scheduler.policy_selected') >= 1"
check "fairness_warning >= 1" "$BASE/api/snapshot" "sum(1 for e in d['events'] if e['event_type']=='scheduler.fairness_warning') >= 1"
check "kernel_probe >= 1" "$BASE/api/snapshot" "sum(1 for e in d['events'] if e['event_type']=='scheduler.kernel_probe') >= 1"
check "model is deepseek-chat" "$BASE/api/snapshot" "any('deepseek' in str(e.get('data',{}).get('model','')).lower() for e in d['events'] if e['event_type']=='llm.response_received')"

echo ""

# --- /api/workload-matrix ---
echo "[2] /api/workload-matrix"
check "has workloads" "$BASE/api/workload-matrix" "'workloads' in d"
check "4 workload types" "$BASE/api/workload-matrix" "len(d['workloads']) == 4"
check "has ai_inference" "$BASE/api/workload-matrix" "'ai_inference' in d['workloads']"

echo ""

# --- /api/cloud-benchmark ---
echo "[3] /api/cloud-benchmark"
check "has metadata" "$BASE/api/cloud-benchmark" "'metadata' in d"
check "model is deepseek-chat" "$BASE/api/cloud-benchmark" "d['metadata'].get('model') == 'deepseek-chat'"
check "4 frameworks" "$BASE/api/cloud-benchmark" "len(d['frameworks']) == 4"
check "has CrewAI" "$BASE/api/cloud-benchmark" "any('CrewAI' in fw['framework'] for fw in d['frameworks'])"
check "no Google ADK" "$BASE/api/cloud-benchmark" "not any('Google ADK' in fw['framework'] for fw in d['frameworks'])"
check "LangGraph native=ok" "$BASE/api/cloud-benchmark" "any(fw['framework']=='LangGraph' and fw.get('native',{}).get('status','ok')=='ok' for fw in d['frameworks'])"
check "all frameworks have native data" "$BASE/api/cloud-benchmark" "all(fw.get('native') is not None for fw in d['frameworks'])"

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
