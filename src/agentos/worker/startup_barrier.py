"""Block a subprocess worker before Runtime applies cgroup and sched_ext policy."""

from __future__ import annotations

import os
import runpy
import sys


def main() -> int:
    target = os.environ.get("AGENTOS_WORKER_TARGET_MODULE", "").strip()
    if not target or target == __name__:
        print("invalid worker startup barrier target", file=sys.stderr)
        return 2
    print("READY", file=sys.stderr, flush=True)
    runpy.run_module(target, run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
