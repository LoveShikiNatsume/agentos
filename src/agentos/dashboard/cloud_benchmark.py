"""Cloud benchmark data provider.

Reads real benchmark JSON from:
1. CLOUD_BENCHMARK_JSON env var (explicit path)
2. Pinned exploratory-pilot file (see _PINNED_JSON below)
3. Fallback to hardcoded mock with source/status annotation

Scope note (2026-07-09):
This page's data source (runs/cloud_framework_benchmark_*.json, produced by
scripts/run_cloud_framework_benchmark.py + scripts/cloud_bench/) is a small
exploratory pilot — 1 scenario x 4 frameworks x 2 modes = 8 runs, dated
2026-06-23/24 — independent of and predating the official 30-task national
batch (runs/national_experiments/20260703_swe_bench_deepseek_30_aggregated/).
It previously auto-selected "whatever file sorts last by filename", which
silently changed what this page showed as new pilot files were dropped in.
That has been replaced with a pinned path so the displayed pilot batch is
stable and deliberate. Do not treat this page's numbers as the official
national-batch result; see REAL_DATA_BOUNDARIES.md.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Most complete run of the exploratory pilot pipeline (all 4 frameworks status=ok,
# 8/8 runs). Pinned explicitly instead of auto-discovering the latest file by
# filename sort — see module docstring.
_PINNED_JSON = _REPO_ROOT / "runs" / "cloud_framework_benchmark_1782279008.json"


def _find_latest_json() -> Path | None:
    """Return the pinned pilot-batch JSON (no longer auto-discovers latest)."""
    if _PINNED_JSON.exists():
        return _PINNED_JSON
    return None


_PILOT_SCOPE_NOTE = (
    "探索性小样本（1 任务 x 4 框架 x 2 模式 = 8 次运行，2026-06-23/24），"
    "非国赛 30 任务正式批次。正式数字见 runs/national_experiments/"
    "20260703_swe_bench_deepseek_30_aggregated/。"
)


def _load_real_json(path: Path) -> dict[str, Any] | None:
    """Load and validate a benchmark JSON file.

    Annotates metadata with a data-scope note in-memory only; the JSON file
    on disk (a measurement artifact) is never modified.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "metadata" in data and "frameworks" in data:
            data["metadata"]["data_scope_note"] = _PILOT_SCOPE_NOTE
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return None


def cloud_benchmark_payload() -> dict[str, Any]:
    """Return benchmark data for the cloud framework page.

    Priority:
    1. CLOUD_BENCHMARK_JSON env var
    2. Pinned exploratory-pilot file (_PINNED_JSON)
    3. Fallback mock with annotation
    """
    # 1. Explicit env var
    env_path = os.environ.get("CLOUD_BENCHMARK_JSON")
    if env_path:
        p = Path(env_path)
        data = _load_real_json(p)
        if data:
            return data

    # 2. Auto-discover latest
    latest = _find_latest_json()
    if latest:
        data = _load_real_json(latest)
        if data:
            return data

    # 3. Fallback mock
    return _fallback_mock()


def _fallback_mock() -> dict[str, Any]:
    """Return mock data annotated as fallback."""
    return {
        "metadata": {
            "source": "AgentOS Runtime fallback mock",
            "model": "mock",
            "scenarios": 1,
            "total_runs": 0,
            "mode": "Native vs +Runtime",
            "status": "fallback_mock",
            "note": "No real benchmark JSON found; showing placeholder data",
        },
        "frameworks": [
            {
                "framework": "OpenAI Agents SDK",
                "native": {
                    "avg_latency": 4.32, "prompt_tokens": 612.7, "api_cost": 0.184,
                    "context_reuse": 18.6, "recovered_faults": 16, "observable_events": 2843,
                    "status": "mock",
                },
                "runtime": {
                    "avg_latency": 2.94, "prompt_tokens": 378.6, "api_cost": 0.134,
                    "context_reuse": 34.5, "recovered_faults": 16, "observable_events": 13652,
                    "status": "mock",
                },
                "improvement": {},
            },
            {
                "framework": "LangGraph",
                "native": {
                    "avg_latency": 4.78, "prompt_tokens": 653.4, "api_cost": 0.196,
                    "context_reuse": 17.3, "recovered_faults": 15, "observable_events": 2701,
                    "status": "mock",
                },
                "runtime": {
                    "avg_latency": 3.21, "prompt_tokens": 402.1, "api_cost": 0.142,
                    "context_reuse": 33.9, "recovered_faults": 15, "observable_events": 13128,
                    "status": "mock",
                },
                "improvement": {},
            },
            {
                "framework": "AutoGen",
                "native": {
                    "avg_latency": 4.15, "prompt_tokens": 589.6, "api_cost": 0.176,
                    "context_reuse": 19.4, "recovered_faults": 13, "observable_events": 2329,
                    "status": "mock",
                },
                "runtime": {
                    "avg_latency": 2.82, "prompt_tokens": 360.7, "api_cost": 0.128,
                    "context_reuse": 35.1, "recovered_faults": 13, "observable_events": 11574,
                    "status": "mock",
                },
                "improvement": {},
            },
            {
                "framework": "CrewAI",
                "native": {
                    "avg_latency": 0, "prompt_tokens": 0, "api_cost": 0,
                    "context_reuse": 0, "recovered_faults": 0, "observable_events": 0,
                    "status": "mock",
                },
                "runtime": {
                    "avg_latency": 0, "prompt_tokens": 0, "api_cost": 0,
                    "context_reuse": 0, "recovered_faults": 0, "observable_events": 0,
                    "status": "mock",
                },
                "improvement": {},
            },
        ],
    }
