#!/usr/bin/env python
"""用真实 DeepSeek 后端跑四类压力测试矩阵(不改动只读的原始源码)。

背景:原始 run_workload_matrix.py + workloads/_base.py 直接用 `--backend deepseek`
会拿不到真实数据,有两个原因:

  1. probe_backend() 请求 DeepSeek 的 /v1/models 时没带 Authorization header,
     DeepSeek 返回 401 → available=False → 整个 matrix 自动 fallback 到 cache-sim。
  2. _base._extract_token_metrics_from_runtime() 对真实后端把缓存命中硬编码成 0,
     而真实的 prompt_cache_hit/miss tokens 其实已经写进了 llm.response_received 事件。

本脚本不修改上述两个文件,而是用 monkey-patch 在
运行时修正这两点,然后复用原有的聚合 / 报告 / dashboard 输出逻辑,产出真实数据。

用法:
    export DEEPSEEK_API_KEY=...
    .venv/bin/python scripts/real_workload_matrix.py --out runs/workload_matrix/deepseek_real
    .venv/bin/python scripts/real_workload_matrix.py --workloads ai_inference --out runs/_probe
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import scripts.workloads._base as base
import scripts.run_workload_matrix as M
from scripts.probe_backend import BackendProbeResult


# --- patch 1: 真实后端从 events 聚合真实缓存命中 ----------------------------
def _extract_token_metrics_from_events(rt):
    if rt is None:
        return {"prompt_tokens": 0, "cache_hit_tokens": 0, "cache_miss_tokens": 0, "cache_hit_ratio": 0.0}
    events = getattr(rt, "events", [])
    hit = sum(
        e.data.get("cache_hit_tokens", 0)
        for e in events if e.event_type == "llm.response_received"
    )
    miss = sum(
        e.data.get("cache_miss_tokens", 0)
        for e in events if e.event_type == "llm.response_received"
    )
    prompt = sum(
        e.data.get("prompt_tokens", e.data.get("tokens_used", 0))
        for e in events if e.event_type == "llm.response_received"
    )
    total_cache = hit + miss
    if prompt <= 0 and total_cache <= 0:
        total_sent = sum(acb.stats.tokens_sent for acb in rt.agents.values())
        return {
            "prompt_tokens": total_sent,
            "cache_hit_tokens": 0,
            "cache_miss_tokens": total_sent,
            "cache_hit_ratio": 0.0,
        }
    if prompt <= 0:
        prompt = total_cache
    return {
        "prompt_tokens": prompt,
        "cache_hit_tokens": hit,
        "cache_miss_tokens": miss,
        "cache_hit_ratio": hit / total_cache if total_cache > 0 else 0.0,
    }


base._extract_token_metrics_from_runtime = _extract_token_metrics_from_events


# --- patch 2: 跳过会误判的 probe(DeepSeek 真实连通性已单独确认) -------------
_orig_probe = M.probe_backend


async def _patched_probe(backend, base_url="", model="", timeout=10.0):
    if backend == "deepseek":
        return BackendProbeResult(
            backend="deepseek",
            available=True,
            base_url=base_url or "https://api.deepseek.com",
            model=model or "deepseek-chat",
            models_endpoint=True,
        )
    return await _orig_probe(backend, base_url, model, timeout)


M.probe_backend = _patched_probe


def main() -> None:
    ap = argparse.ArgumentParser(description="Workload matrix on real DeepSeek backend")
    ap.add_argument("--out", default="runs/workload_matrix/deepseek_real")
    ap.add_argument("--workloads", nargs="*", choices=list(M.WORKLOADS.keys()))
    ap.add_argument("--model", default="deepseek-chat")
    args = ap.parse_args()

    asyncio.run(M.run_matrix(
        "deepseek",
        Path(args.out),
        args.workloads,
        base_url="https://api.deepseek.com",
        model=args.model,
    ))


if __name__ == "__main__":
    main()
