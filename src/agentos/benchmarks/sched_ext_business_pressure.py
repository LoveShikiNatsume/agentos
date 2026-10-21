"""Controlled single-CPU SHA-256 pressure used only by the P2 smoke runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time


_RUNNING = True


def _stop(_signum: int, _frame: object) -> None:
    global _RUNNING
    _RUNNING = False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    os.sched_setaffinity(0, {args.cpu})
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    print("READY", file=sys.stderr, flush=True)
    if sys.stdin.buffer.read(1) != b"1":
        raise RuntimeError("pressure release byte is missing")

    digest = hashlib.sha256(str(args.seed).encode("ascii")).digest()
    iterations = 0
    started_ns = time.monotonic_ns()
    while _RUNNING or iterations == 0:
        for _ in range(1024):
            digest = hashlib.sha256(digest).digest()
            iterations += 1
    print(
        json.dumps(
            {
                "completed": True,
                "cpu": args.cpu,
                "affinity": sorted(os.sched_getaffinity(0)),
                "iterations": iterations,
                "elapsed_ns": time.monotonic_ns() - started_ns,
                "checksum": digest.hex(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
