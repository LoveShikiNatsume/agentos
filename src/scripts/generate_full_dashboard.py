"""Generate a comprehensive dashboard from multiple data sources.

Combines event logs from e2e fault demo, DevOps scenario, and benchmark
data into a single dashboard. Verifies the HTML contains all expected sections.

Usage:
    .venv/bin/python scripts/generate_full_dashboard.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


async def _collect_event_logs(output_dir: Path) -> list[Path]:
    logs: list[Path] = []

    demo_log = output_dir / "demo_events.jsonl"
    if demo_log.exists():
        demo_log.unlink()
    from scripts.demo_e2e_fault import run_demo
    await run_demo(demo_log)
    logs.append(demo_log)
    print(f"  [1/3] E2E fault demo events → {demo_log}")

    devops_log = output_dir / "devops_events.jsonl"
    if devops_log.exists():
        devops_log.unlink()
    from agentos.scenarios.devops.incident import run_devops_incident
    await run_devops_incident(log_path=devops_log, inject_fault=True)
    logs.append(devops_log)
    print(f"  [2/3] DevOps incident events → {devops_log}")

    benchmark_log = output_dir / "benchmark_events.jsonl"
    if benchmark_log.exists():
        benchmark_log.unlink()
    from scripts.run_benchmark_comparison import run_runtime, CacheSimulatingMockBackend
    backend = CacheSimulatingMockBackend()
    await run_runtime(backend, log_path=benchmark_log)
    logs.append(benchmark_log)
    print(f"  [3/3] Benchmark runtime events → {benchmark_log}")

    return logs


def _collect_benchmark_summary(output_dir: Path) -> Path | None:
    comparison_file = output_dir / "comparison_results.json"
    if not comparison_file.exists():
        return None

    data = json.loads(comparison_file.read_text(encoding="utf-8"))
    bl = data.get("baseline", {})
    rt = data.get("runtime", {})

    summary = {
        "run_count": 1,
        "modes": {
            "baseline": {
                "total_latency_secs": {"mean": bl.get("total_latency_secs", 0), "min": bl.get("total_latency_secs", 0), "max": bl.get("total_latency_secs", 0)},
                "total_prompt_tokens": {"mean": bl.get("prompt_tokens", 0), "min": bl.get("prompt_tokens", 0), "max": bl.get("prompt_tokens", 0)},
                "total_completion_tokens": {"mean": bl.get("completion_tokens", 0) if "completion_tokens" in bl else 0, "min": 0, "max": 0},
                "cache_hit_ratio": {"mean": bl.get("cache_hit_ratio", 0), "min": bl.get("cache_hit_ratio", 0), "max": bl.get("cache_hit_ratio", 0)},
                "fault_count": {"mean": 1 if not bl.get("fault_recovered", False) else 0, "min": 0, "max": 1},
                "event_count": {"mean": bl.get("event_count", 0), "min": bl.get("event_count", 0), "max": bl.get("event_count", 0)},
            },
            "runtime": {
                "total_latency_secs": {"mean": rt.get("total_latency_secs", 0), "min": rt.get("total_latency_secs", 0), "max": rt.get("total_latency_secs", 0)},
                "total_prompt_tokens": {"mean": rt.get("prompt_tokens", 0), "min": rt.get("prompt_tokens", 0), "max": rt.get("prompt_tokens", 0)},
                "total_completion_tokens": {"mean": rt.get("completion_tokens", 0) if "completion_tokens" in rt else 0, "min": 0, "max": 0},
                "cache_hit_ratio": {"mean": rt.get("cache_hit_ratio", 0), "min": rt.get("cache_hit_ratio", 0), "max": rt.get("cache_hit_ratio", 0)},
                "fault_count": {"mean": 0, "min": 0, "max": 0},
                "event_count": {"mean": rt.get("event_count", 0), "min": rt.get("event_count", 0), "max": rt.get("event_count", 0)},
            },
        },
        "comparison": data.get("comparison", {}),
    }

    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary_path


def verify_dashboard(html: str) -> list[tuple[str, bool]]:
    checks = [
        ("contains title", "AgentOS Runtime Dashboard" in html),
        ("contains agent data", "agent-" in html),
        ("contains fault section", "Fault" in html),
        ("contains recovery data", "recovery" in html.lower()),
        ("contains KPI section", "kpi" in html),
        ("contains resource section", "Resource" in html),
        ("contains benchmark section", "Benchmark" in html or "benchmark" in html),
        ("contains workload matrix", "Workload Matrix" in html),
        ("HTML is well-formed", html.strip().startswith("<!doctype") or html.strip().startswith("<!")),
        ("not empty", len(html) > 1000),
    ]
    return checks


async def async_main(argv: Sequence[str] | None = None) -> int:
    output_dir = Path("runs/full_dashboard")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  Full Dashboard Generation & Verification")
    print("=" * 64)
    print()

    print("[Collecting Event Logs]")
    log_paths = await _collect_event_logs(output_dir)
    print()

    print("[Running Benchmark Comparison]")
    from scripts.run_benchmark_comparison import (
        CacheSimulatingMockBackend,
        run_baseline,
        run_runtime,
        save_report,
    )

    backend = CacheSimulatingMockBackend()
    bl = await run_baseline(backend)
    rt = await run_runtime(backend)
    save_report(bl, rt, output_dir)
    print(f"  Comparison results saved to {output_dir}")
    print()

    print("[Running Workload Matrix]")
    matrix_path = output_dir / "workload_matrix_summary.json"
    try:
        from scripts.run_workload_matrix import run_matrix
        await run_matrix("mock", output_dir / "workload_matrix")
        wl_summary = output_dir / "workload_matrix" / "summary.json"
        if wl_summary.exists():
            import shutil
            shutil.copy2(wl_summary, matrix_path)
            print(f"  Workload matrix summary → {matrix_path}")
        else:
            matrix_path = None
    except Exception as exc:
        print(f"  Workload matrix skipped: {exc}")
        matrix_path = None
    print()

    print("[Generating Dashboard]")
    summary_path = _collect_benchmark_summary(output_dir)

    from agentos.dashboard.builder import build_dashboard
    dashboard_path = build_dashboard(
        log_paths=log_paths,
        output_path=output_dir / "dashboard.html",
        benchmark_summary_path=summary_path,
        workload_matrix_path=matrix_path,
    )
    print(f"  Dashboard written to {dashboard_path}")
    print()

    html = dashboard_path.read_text(encoding="utf-8")
    checks = verify_dashboard(html)

    print("[Verification]")
    ok = True
    for label, passed in checks:
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {label}")
        if not passed:
            ok = False
    print()

    print(f"  HTML size: {len(html):,} bytes")
    print(f"  Agent rows: {html.count('<tr>') - 5}")  # subtract header rows
    print()

    status = "ALL PASSED" if ok else "SOME FAILED"
    print(f"Result: {status}")
    return 0 if ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
