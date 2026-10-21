#!/usr/bin/env python3
"""Serve the dashboard in director/replay mode.

Starts the SSE dashboard server, waits for a browser client, then plays
the storyboard once.  Used directly for rehearsal, and driven by
scripts/record_demo.py for one-command recording.

Usage:
    .venv/bin/python scripts/replay_dashboard.py                 # rehearse
    .venv/bin/python scripts/replay_dashboard.py --exit-after-show
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agentos.dashboard.broadcast import EventBroadcaster
from agentos.dashboard.replay import run_replay
from agentos.dashboard.server import run_dashboard_server

DEMO_DIR = _REPO_ROOT.parent / "demo" / "replay"


async def main_async(args: argparse.Namespace) -> int:
    broadcaster = EventBroadcaster()
    server_task = asyncio.create_task(
        run_dashboard_server(broadcaster, host=args.host, port=args.port)
    )
    await asyncio.sleep(0.5)
    url = f"http://{args.host}:{args.port}/?director=1"
    print(f"  Director mode: open {url}  (show starts when the page connects)")
    print(f"  Storyboard: {args.storyboard}")
    try:
        await run_replay(broadcaster, args.storyboard, args.base_dir)
        print("  Show finished.")
        if args.exit_after_show:
            await asyncio.sleep(args.grace)
            server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await server_task
            return 0
        print("  Server keeps running for inspection, Ctrl+C to quit ...")
        await server_task
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9420)
    parser.add_argument("--storyboard", default=str(DEMO_DIR / "storyboard.json"))
    parser.add_argument("--base-dir", default=str(DEMO_DIR),
                        help="directory containing the replay JSONL files")
    parser.add_argument("--exit-after-show", action="store_true")
    parser.add_argument("--grace", type=float, default=5.0,
                        help="seconds to keep serving after show end (recording tail)")
    args = parser.parse_args()
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
