#!/usr/bin/env python3
"""One-command demo recording: replay server + headless Chromium + video.

Launches scripts/replay_dashboard.py as a subprocess, opens the director
page in headless Chromium (1920x1080) with Playwright video recording,
waits for the show to finish (document.title becomes DIRECTOR_DONE),
then saves the .webm (and converts to .mp4 if ffmpeg is available).

Usage:
    .venv/bin/python scripts/record_demo.py
    .venv/bin/python scripts/record_demo.py --port 9421 --out runs/demo_video/recordings
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = _REPO_ROOT.parent / "demo" / "replay"


def wait_for_port(url: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except OSError:
            time.sleep(0.3)
    raise TimeoutError(f"replay server did not come up at {url}")


def find_ffmpeg() -> str | None:
    """ffmpeg from PATH, else the binary bundled with imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9420)
    parser.add_argument("--storyboard", default=str(DEMO_DIR / "storyboard.json"))
    parser.add_argument("--out", default=str(DEMO_DIR / "recordings"))
    parser.add_argument(
        "--filename",
        default="demo_raw.webm",
        help="output WebM filename inside --out",
    )
    parser.add_argument(
        "--rehearsal",
        action="store_true",
        help="show a visible mock/rehearsal provenance badge",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="recording viewport width in pixels",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="recording viewport height in pixels",
    )
    args = parser.parse_args(argv)
    if args.width < 320 or args.height < 180:
        parser.error("--width/--height must be at least 320×180")
    return args


def main() -> int:
    args = parse_args()

    total = json.loads(Path(args.storyboard).read_text(encoding="utf-8"))["total_duration"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Master logs: {DEMO_DIR}")
    print(f"  Storyboard total: {total}s")
    print(f"  Recording size: {args.width}x{args.height}")

    server = subprocess.Popen([
        sys.executable, str(_REPO_ROOT / "scripts" / "replay_dashboard.py"),
        "--port", str(args.port), "--storyboard", args.storyboard,
        "--exit-after-show", "--grace", "8",
    ])
    try:
        base = f"http://127.0.0.1:{args.port}"
        wait_for_port(base + "/api/snapshot")

        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch()
            ctx = browser.new_context(
                viewport={"width": args.width, "height": args.height},
                record_video_dir=str(out_dir),
                record_video_size={"width": args.width, "height": args.height},
            )
            page = ctx.new_page()
            t0 = time.monotonic()
            query = "/?director=1" + ("&rehearsal=1" if args.rehearsal else "")
            page.goto(base + query)
            page.wait_for_function(
                "document.title === 'DIRECTOR_DONE'",
                timeout=(total + 120) * 1000,
            )
            elapsed = time.monotonic() - t0
            page.wait_for_timeout(2500)   # recording tail
            video = page.video
            ctx.close()
            webm = Path(video.path())
            browser.close()

        final_webm = out_dir / args.filename
        shutil.move(str(webm), final_webm)
        print(f"  Show wall time: {elapsed:.1f}s (storyboard {total}s)")
        if abs(elapsed - total) > 5:
            print("  WARNING: wall time deviates >5s from storyboard — check for jank")
        print(f"  Saved: {final_webm}")

        ffmpeg = find_ffmpeg()
        if ffmpeg:
            mp4 = out_dir / Path(args.filename).with_suffix(".mp4").name
            subprocess.run([ffmpeg, "-y", "-i", str(final_webm),
                            "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                            "-r", "30", str(mp4)], check=True)
            print(f"  Converted: {mp4}")
        else:
            print("  ffmpeg not found — deliverable is the .webm; convert on a machine with ffmpeg")
        return 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    raise SystemExit(main())
