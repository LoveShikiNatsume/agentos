#!/usr/bin/env python3
"""Rehearse the director show headlessly and screenshot given timestamps.

Usage:
    .venv/bin/python scripts/capture_director_frames.py \
        --storyboard runs/demo_video/storyboard_final.json \
        --times 1,3,8,25,45,70,100,130,170,200,215,227 \
        --out /tmp/director_frames
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--storyboard", required=True)
    ap.add_argument("--times", required=True,
                    help="comma separated seconds, ascending")
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=9431)
    ap.add_argument("--width", type=int, default=2560)
    ap.add_argument("--height", type=int, default=1440)
    args = ap.parse_args()

    marks = sorted(float(t) for t in args.times.split(","))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    srv = subprocess.Popen([
        sys.executable, str(_REPO / "scripts" / "replay_dashboard.py"),
        "--port", str(args.port), "--storyboard", args.storyboard,
        "--exit-after-show", "--grace", "5",
    ])
    try:
        base = f"http://127.0.0.1:{args.port}"
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(base + "/api/snapshot", timeout=2)
                break
            except OSError:
                time.sleep(0.3)

        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_context(
                viewport={"width": args.width, "height": args.height}
            ).new_page()
            t0 = time.monotonic()
            page.goto(base + "/?director=1")
            for mark in marks:
                delay = mark - (time.monotonic() - t0)
                if delay > 0:
                    page.wait_for_timeout(delay * 1000)
                page.screenshot(path=str(out / f"t{mark:07.1f}.png"))
                print(f"captured t={mark}")
            browser.close()
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=10)
        except subprocess.TimeoutExpired:
            srv.kill()
    print(f"frames in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
