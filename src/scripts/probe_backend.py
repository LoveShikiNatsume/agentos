"""Probe LLM backend availability and capabilities.

Checks whether a given backend (vllm, deepseek, etc.) is reachable
and what metrics it exposes. Outputs a structured JSON report.

Usage:
    uv run python scripts/probe_backend.py --backend vllm --base-url http://127.0.0.1:8000
    uv run python scripts/probe_backend.py --backend mock
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class BackendProbeResult:
    backend: str = "mock"
    available: bool = True
    base_url: str = ""
    model: str = ""
    models_endpoint: bool = False
    metrics_endpoint: bool = False
    prefix_cache_metrics: bool = False
    prefill_decode_metrics: bool = False
    health_latency_ms: float = 0.0
    fallback_reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def probe_backend(
    backend: str,
    base_url: str = "",
    model: str = "",
    timeout: float = 10.0,
) -> BackendProbeResult:
    result = BackendProbeResult(backend=backend, base_url=base_url, model=model)

    if backend in ("mock", "cache-sim", "vllm-sim", "deepseek-sim"):
        result.available = True
        result.fallback_reason = ""
        return result

    if backend in ("vllm", "deepseek"):
        if not base_url:
            base_url = (
                "http://127.0.0.1:8000" if backend == "vllm"
                else "https://api.deepseek.com"
            )
            result.base_url = base_url

        try:
            import httpx
        except ImportError:
            result.available = False
            result.fallback_reason = "httpx not installed"
            return result

        import time
        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                # /v1/models
                try:
                    resp = await client.get(f"{base_url.rstrip('/')}/v1/models")
                    if resp.status_code == 200:
                        result.models_endpoint = True
                        data = resp.json()
                        models = data.get("data", [])
                        if models and not model:
                            result.model = models[0].get("id", "")
                        result.detail["models"] = [
                            m.get("id", "") for m in models[:5]
                        ]
                except Exception:
                    pass

                # /metrics (vllm only)
                if backend == "vllm":
                    try:
                        resp = await client.get(f"{base_url.rstrip('/')}/metrics")
                        if resp.status_code == 200:
                            result.metrics_endpoint = True
                            text = resp.text
                            result.prefix_cache_metrics = (
                                "prefix_cache" in text
                            )
                            result.prefill_decode_metrics = (
                                "prefill_tokens" in text
                                or "prompt_tokens_total" in text
                            )
                    except Exception:
                        pass

                # /health or /v1/models as liveness
                result.health_latency_ms = (time.monotonic() - start) * 1000
                result.available = result.models_endpoint

        except Exception as e:
            result.available = False
            result.fallback_reason = str(e)[:200]
            result.health_latency_ms = (time.monotonic() - start) * 1000

        if not result.available and not result.fallback_reason:
            result.fallback_reason = f"{backend} at {base_url} not reachable"

        return result

    result.available = False
    result.fallback_reason = f"unknown backend: {backend}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe LLM backend")
    parser.add_argument("--backend", default="mock")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = asyncio.run(
        probe_backend(args.backend, args.base_url, args.model)
    )
    output = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    print(output)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output + "\n", encoding="utf-8")
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
