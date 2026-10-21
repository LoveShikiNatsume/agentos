#!/usr/bin/env bash
# openEuler Adaptation Verification
#
# Usage:
#   cd <仓库根>
#   bash src/scripts/openeuler-verify.sh
#
# Prerequisites: Docker installed and running

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$SUBMISSION_ROOT"

IMAGE_NAME="agentos-openeuler"

echo "============================================"
echo "  AgentOS-Runtime openEuler Verification"
echo "  Target: openEuler 24.03 LTS (x86_64)"
echo "============================================"
echo ""

# Step 1: Build
echo ">>> Step 1/4: Building openEuler image..."
docker build -f src/Dockerfile.openeuler -t "$IMAGE_NAME" .
echo ""

# Step 2: Python version check
echo ">>> Step 2/4: Checking Python version..."
docker run --rm "$IMAGE_NAME" python3 -c "
import sys
v = sys.version_info
print(f'Python {v.major}.{v.minor}.{v.micro}')
assert v >= (3, 11), f'Need Python >= 3.11, got {v.major}.{v.minor}'
print('OK: Python version meets requirement')
"
echo ""

# Step 3: Full test suite
echo ">>> Step 3/4: Running full test suite..."
docker run --rm "$IMAGE_NAME" python3 -m pytest -c src/pyproject.toml tests -q
echo ""

# Step 4: Real cgroup v2 verification
echo ">>> Step 4/4: Running cgroup v2 verification (privileged)..."
docker run --rm --privileged "$IMAGE_NAME" \
    python3 src/scripts/openeuler-cgroup-verify.py
echo ""

echo "============================================"
echo "  All verifications PASSED"
echo "  Platform: openEuler 24.03 LTS"
echo "============================================"
