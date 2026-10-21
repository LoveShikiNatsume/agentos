#!/usr/bin/env python3
"""Capture dashboard screenshots for PPT/video review.

Uses system Chrome via playwright (channel='chrome'), waits for SSE data
to render, switches pages/tabs, and saves PNGs.

Usage:
    .venv/bin/python scripts/screenshot_dashboard.py \
        --url http://127.0.0.1:9412 --outdir ../OS-PPT/dashboard截图
"""
from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:9412")
    parser.add_argument("--outdir", default="runs/demo_video/screenshots")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto(args.url, wait_until="domcontentloaded")
        page.wait_for_timeout(4000)  # let SSE snapshot render

        shots: list[tuple[str, str]] = []

        # --- local (本地实测) page: capture full page then each nav section
        page.wait_for_timeout(500)
        page.screenshot(path=str(outdir / "01-local-总览.png"))
        shots.append(("local", "01-local-总览.png"))

        # Click through left-nav items on the local page, if present
        nav_items = page.locator("#page-local .nav a, #page-local nav a, .sidebar a")
        count = nav_items.count()
        for i in range(count):
            item = nav_items.nth(i)
            label = (item.inner_text() or f"nav{i}").strip().replace("/", "-")[:20]
            try:
                item.click()
                page.wait_for_timeout(800)
                fname = f"02-local-{i:02d}-{label}.png"
                page.screenshot(path=str(outdir / fname))
                shots.append(("local-nav", fname))
            except Exception as exc:  # noqa: BLE001
                print(f"  skip nav {label}: {exc}")

        # --- cloud (云端实验) page
        cloud_tab = page.locator('[data-page="cloud"]')
        if cloud_tab.count():
            cloud_tab.first.click()
            page.wait_for_timeout(1500)
            page.screenshot(path=str(outdir / "03-cloud-云端实验.png"))
            shots.append(("cloud", "03-cloud-云端实验.png"))
            # full-page capture of cloud view
            page.screenshot(path=str(outdir / "04-cloud-全页.png"), full_page=True)
            shots.append(("cloud", "04-cloud-全页.png"))

        # back to local, full page
        local_tab = page.locator('[data-page="local"]')
        if local_tab.count():
            local_tab.first.click()
            page.wait_for_timeout(1000)
            page.screenshot(path=str(outdir / "05-local-全页.png"), full_page=True)
            shots.append(("local", "05-local-全页.png"))

        browser.close()

    print(f"\nSaved {len(shots)} screenshots to {outdir}:")
    for section, name in shots:
        print(f"  [{section}] {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
