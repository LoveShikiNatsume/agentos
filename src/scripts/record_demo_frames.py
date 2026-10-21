#!/usr/bin/env python3
"""High-quality demo recording: CDP screencast frames + ffmpeg assembly.

Replaces Playwright's built-in recorder (whose ~0.7 Mbps VP8 output is the
quality bottleneck) with lossless PNG frames captured via the Chrome
DevTools Protocol screencast, assembled by ffmpeg at high bitrate.

Frames arrive only when the screen changes and carry timestamps; assembly
uses the concat demuxer with per-frame durations so the timeline matches
the live show exactly.

Usage:
    .venv/bin/python scripts/record_demo_frames.py \
        --storyboard runs/demo_video/storyboard_final.json \
        --filename demo_final_hq.mp4
"""
from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import sys
import tempfile
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
    """Prefer a modern ffmpeg (imageio bundle) over ancient PATH builds.

    PATH may resolve to conda ffmpeg 4.x that lacks options used by this
    pipeline; imageio-ffmpeg ships a recent static binary.
    """
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg")


def write_concat_list(
    frames: list[tuple[float, Path]], list_path: Path, tail: float = 1.0
) -> None:
    """Write an ffmpeg concat-demuxer list with per-frame durations."""
    lines = ["ffconcat version 1.0"]
    for i, (ts, path) in enumerate(frames):
        nxt = frames[i + 1][0] if i + 1 < len(frames) else ts + tail
        dur = max(nxt - ts, 1 / 60)
        lines.append(f"file '{path.as_posix()}'")
        lines.append(f"duration {dur:.6f}")
    # concat demuxer quirk: repeat the last frame so its duration applies
    lines.append(f"file '{frames[-1][1].as_posix()}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def assemble(
    ffmpeg: str, list_path: Path, out_mp4: Path, fps: int, crf: int
) -> None:
    # UI screencast: low CRF + stillimage tune keeps text sharp.
    # Prefer -fps_mode (ffmpeg 5+) with -vsync fallback for older builds.
    base = [
        ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
    ]
    encode = [
        "-r", str(fps),
        "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
        "-tune", "stillimage",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    # Try modern fps_mode first; fall back to vsync for ffmpeg 4.x
    for sync_flag in (["-fps_mode", "cfr"], ["-vsync", "cfr"]):
        result = subprocess.run(base + sync_flag + encode, capture_output=True)
        if result.returncode == 0:
            return
        err = (result.stderr or b"").decode("utf-8", errors="replace")
        if "Unrecognized option" not in err and "Option not found" not in err:
            # Real encode failure — surface it
            raise subprocess.CalledProcessError(
                result.returncode, base + sync_flag + encode, result.stdout, result.stderr
            )
    raise RuntimeError("ffmpeg assemble failed with both -fps_mode and -vsync")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9422)
    parser.add_argument("--storyboard", default=str(DEMO_DIR / "storyboard_final.json"))
    parser.add_argument("--out", default=str(DEMO_DIR / "recordings"))
    parser.add_argument("--filename", default="demo_final_hq.mp4")
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--fps", type=int, default=30, help="output fps")
    parser.add_argument(
        "--crf", type=int, default=12,
        help="x264 quality (lower=sharper; 12 is near-lossless for UI)",
    )
    parser.add_argument("--keep-frames", action="store_true",
                        help="keep the temporary PNG frame directory")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    total = json.loads(Path(args.storyboard).read_text(encoding="utf-8"))["total_duration"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("ERROR: no ffmpeg available (PATH or imageio-ffmpeg)")
        return 1

    frame_dir = Path(tempfile.mkdtemp(prefix="demo_frames_"))
    print(f"  Storyboard total: {total}s")
    print(f"  Frames -> {frame_dir}")

    server = subprocess.Popen([
        sys.executable, str(_REPO_ROOT / "scripts" / "replay_dashboard.py"),
        "--port", str(args.port), "--storyboard", args.storyboard,
        "--exit-after-show", "--grace", "8",
    ])
    frames: list[tuple[float, Path]] = []
    try:
        base = f"http://127.0.0.1:{args.port}"
        wait_for_port(base + "/api/snapshot")

        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_context(
                viewport={"width": args.width, "height": args.height}
            ).new_page()
            cdp = page.context.new_cdp_session(page)
            counter = 0

            def on_frame(params: dict) -> None:
                nonlocal counter
                path = frame_dir / f"f{counter:06d}.png"
                path.write_bytes(base64.b64decode(params["data"]))
                frames.append((params["metadata"]["timestamp"], path))
                counter += 1
                cdp.send("Page.screencastFrameAck",
                         {"sessionId": params["sessionId"]})

            cdp.on("Page.screencastFrame", on_frame)
            cdp.send("Page.startScreencast", {
                "format": "png", "everyNthFrame": 1,
                "maxWidth": args.width, "maxHeight": args.height,
            })
            t0 = time.monotonic()
            page.goto(base + "/?director=1")
            page.wait_for_function(
                "document.title === 'DIRECTOR_DONE'",
                timeout=(total + 120) * 1000,
            )
            page.wait_for_timeout(1500)   # capture the final fade tail
            cdp.send("Page.stopScreencast")
            elapsed = time.monotonic() - t0
            browser.close()

        print(f"  Show wall time: {elapsed:.1f}s (storyboard {total}s)")
        print(f"  Captured {len(frames)} frames "
              f"({sum(p.stat().st_size for _, p in frames) / 1e9:.2f} GB)")
        if abs(elapsed - total) > 8:
            print("  WARNING: wall time deviates from storyboard — check for jank")
        if not frames:
            print("ERROR: no frames captured")
            return 1

        frames.sort(key=lambda fr: fr[0])
        list_path = frame_dir / "concat.txt"
        write_concat_list(frames, list_path)
        out_mp4 = out_dir / args.filename
        assemble(ffmpeg, list_path, out_mp4, args.fps, args.crf)
        print(f"  Saved: {out_mp4} "
              f"({out_mp4.stat().st_size / 1e6:.1f} MB)")
        return 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        if not args.keep_frames:
            shutil.rmtree(frame_dir, ignore_errors=True)
            print("  Frame dir cleaned")


if __name__ == "__main__":
    raise SystemExit(main())
