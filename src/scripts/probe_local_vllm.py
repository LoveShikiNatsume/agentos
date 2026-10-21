"""Small local vLLM load/cache/concurrency probe; not a model-quality benchmark."""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from agentos.llm.base import LLMRequest
from agentos.llm.vllm_backend import VLLMBackend
from agentos.scenarios.repository_review import collect_sources


async def probe(output: Path, base_url: str, model: str, source_root: Path):
    if urlsplit(base_url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("probe requires a local model endpoint")
    texts, facts = collect_sources(source_root)
    prefix = "Use only this source to answer briefly.\n" + texts["source/context"]
    backend = VLLMBackend(base_url=base_url, model=model, timeout=240,
                          max_output_tokens=64, chat_template_kwargs={"enable_thinking": False})
    output.mkdir(parents=True, exist_ok=False)
    gpu_samples = []
    collecting = True

    async def sample_gpu():
        while collecting:
            p = await asyncio.create_subprocess_exec(
                "nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await p.communicate()
            if p.returncode == 0:
                memory, util = stdout.decode().strip().split(",")
                gpu_samples.append({"time": time.time(), "memory_mib": int(memory), "utilization_pct": int(util)})
            await asyncio.sleep(0.5)

    records = []
    async def call(name, question):
        req = LLMRequest(agent_id=name, task_id=name, prefix_id="context_source",
                         messages=[{"role":"system","content":prefix},{"role":"user","content":question}],
                         temperature=0.7, max_tokens=64)
        result = await backend.generate(req)
        record = {"case": name, "response": result.content, "usage": result.metadata.get("usage"),
                  "metrics": result.metrics.to_dict(), "nonempty": bool(result.content.strip())}
        records.append(record)
        return record

    async with httpx.AsyncClient(trust_env=False, timeout=15) as client:
        identity = (await client.get(base_url + "/v1/models")).json()
        before = await client.get(base_url + "/metrics")
        (output / "metrics_before.txt").write_text(before.text)
    sampler = asyncio.create_task(sample_gpu())
    try:
        await call("first", "Explain how shared context is selected in two short sentences.")
        warm = await call("repeat", "Explain how shared context is selected in two short sentences.")
        started = time.monotonic()
        parallel = await asyncio.gather(
            call("parallel_a", "Explain what context retry restores in one short sentence."),
            call("parallel_b", "Explain which concurrent updates survive retry in one short sentence."),
        )
        concurrent_elapsed = time.monotonic() - started
    finally:
        collecting = False
        await sampler
        (output / "requests.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
        (output / "gpu_samples.json").write_text(json.dumps(gpu_samples, indent=2))
    async with httpx.AsyncClient(trust_env=False, timeout=15) as client:
        after = await client.get(base_url + "/metrics")
        (output / "metrics_after.txt").write_text(after.text)
    result = {
        "schema": "agentos_local_vllm_probe.v1", "model": model,
        "server_identity": identity, "source_files": facts,
        "all_responses_nonempty": all(x["nonempty"] for x in records),
        "cache_details_available": all(x["metrics"]["metadata"]["cache_metrics_available"] for x in records),
        "repeat_cached_tokens": warm["metrics"]["cache_hit_tokens"],
        "cases": records, "parallel_wall_secs": concurrent_elapsed,
        "parallel_completion_tokens_per_wall_second": sum(x["metrics"]["completion_tokens"] for x in parallel) / concurrent_elapsed,
        "gpu_memory_peak_mib": max((x["memory_mib"] for x in gpu_samples), default=None),
        "quality_benchmark": False, "cache_probe_order": "first, repeat, parallel pair",
        "timing_note": "wall-clock completion throughput includes prefill and queueing; not pure decode throughput",
    }
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18094")
    parser.add_argument("--model", default="qwen3.8-27b-int4")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = asyncio.run(probe(args.output, args.base_url, args.model, args.source_root))
    print(json.dumps({k: result[k] for k in ("all_responses_nonempty", "cache_details_available", "repeat_cached_tokens", "gpu_memory_peak_mib", "parallel_wall_secs")},ensure_ascii=False))
    return 0 if result["all_responses_nonempty"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
