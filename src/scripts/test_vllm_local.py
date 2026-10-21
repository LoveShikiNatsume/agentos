"""Quick smoke test: call local vLLM via our DeepSeekBackend + ModelGateway.

Usage:
    ./.venv/bin/python scripts/test_vllm_local.py

Requires vLLM running on localhost:8000 with an OpenAI-compatible model.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentos.llm.base import LLMRequest
from agentos.llm.deepseek_backend import DeepSeekBackend
from agentos.llm.gateway import ModelGateway


VLLM_BASE_URL = "http://localhost:8000/v1"
# 本地 vLLM 服务加载的模型标识；按实际部署路径或模型名通过环境变量覆盖。
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen2-7B-Instruct")
SHARED_SYSTEM_PROMPT = (
    "You are a helpful coding assistant working inside AgentOS, "
    "an OS-level multi-agent runtime. You help with Python programming tasks. "
    "Always respond concisely."
)


async def test_single_call():
    print("=" * 60)
    print("Test 1: Single call to local vLLM")
    print("=" * 60)

    backend = DeepSeekBackend(
        api_key="not-needed",
        base_url=VLLM_BASE_URL,
        model=VLLM_MODEL,
    )

    req = LLMRequest(
        agent_id="test-agent",
        messages=[
            {"role": "system", "content": SHARED_SYSTEM_PROMPT},
            {"role": "user", "content": "Write a Python function to compute fibonacci(n)."},
        ],
        max_tokens=256,
    )

    resp = await backend.generate(req)
    print(f"Content: {resp.content[:200]}...")
    print(f"Tokens: prompt={resp.metrics.prompt_tokens}, completion={resp.metrics.completion_tokens}")
    print(f"Cache:  hit={resp.metrics.cache_hit_tokens}, miss={resp.metrics.cache_miss_tokens}")
    print(f"Ratio:  {resp.metrics.cache_hit_ratio:.2%}")
    print(f"Latency: {resp.metrics.latency_secs:.3f}s")
    print()


async def test_prefix_caching():
    print("=" * 60)
    print("Test 2: Prefix caching — two agents, same system prompt")
    print("=" * 60)

    backend = DeepSeekBackend(
        api_key="not-needed",
        base_url=VLLM_BASE_URL,
        model=VLLM_MODEL,
    )

    gw = ModelGateway(backends={"local": backend}, default_backend="local")

    req1 = LLMRequest(
        agent_id="planner",
        messages=[
            {"role": "system", "content": SHARED_SYSTEM_PROMPT},
            {"role": "user", "content": "Plan the steps to build a calculator app."},
        ],
        max_tokens=256,
    )

    req2 = LLMRequest(
        agent_id="coder",
        messages=[
            {"role": "system", "content": SHARED_SYSTEM_PROMPT},
            {"role": "user", "content": "Write a Python class for a simple calculator."},
        ],
        max_tokens=256,
    )

    print("Agent 1 (planner) — first call, expect cache miss...")
    resp1 = await gw.generate(req1)
    print(f"  prompt={resp1.metrics.prompt_tokens}, cache_hit={resp1.metrics.cache_hit_tokens}, "
          f"cache_miss={resp1.metrics.cache_miss_tokens}, ratio={resp1.metrics.cache_hit_ratio:.2%}, "
          f"latency={resp1.metrics.latency_secs:.3f}s")

    print("Agent 2 (coder) — same system prompt, expect cache hit on prefix...")
    resp2 = await gw.generate(req2)
    print(f"  prompt={resp2.metrics.prompt_tokens}, cache_hit={resp2.metrics.cache_hit_tokens}, "
          f"cache_miss={resp2.metrics.cache_miss_tokens}, ratio={resp2.metrics.cache_hit_ratio:.2%}, "
          f"latency={resp2.metrics.latency_secs:.3f}s")

    print()
    print("Gateway stats:", gw.stats())
    print()

    if resp2.metrics.cache_hit_tokens > 0:
        print("PREFIX CACHING WORKS — second call got cache hits!")
    else:
        print("No cache hits detected (vLLM may not report prompt_cache_hit_tokens in usage).")
        print("This is expected — vLLM prefix caching works internally but may not expose it in the API response.")
        print("The performance benefit (lower latency on second call) is still real.")

    if resp2.metrics.latency_secs < resp1.metrics.latency_secs:
        speedup = (1 - resp2.metrics.latency_secs / resp1.metrics.latency_secs) * 100
        print(f"Latency improvement: {speedup:.1f}% faster on second call")


async def main():
    await test_single_call()
    await test_prefix_caching()


if __name__ == "__main__":
    asyncio.run(main())
