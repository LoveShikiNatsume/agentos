from __future__ import annotations

from agentos.llm.base import LLMBackend


def create_backend(name: str, **kwargs) -> LLMBackend:
    if name == "mock":
        from agentos.llm.mock import MockLLMBackend
        latency = kwargs.pop("latency", None)
        if latency is not None:
            kwargs["latency"] = latency
        return MockLLMBackend(**kwargs)

    if name == "cache-sim":
        from agentos.benchmarks.cache_sim import CacheSimulatingMockBackend
        latency = kwargs.pop("latency", None)
        if latency is not None:
            kwargs["latency_base"] = latency
        return CacheSimulatingMockBackend(**kwargs)

    if name == "vllm-sim":
        from agentos.benchmarks.cache_sim import CacheSimulatingMockBackend
        latency = kwargs.pop("latency", None)
        return CacheSimulatingMockBackend(
            latency_base=latency or 0.8,
            cache_speedup=0.7,
            model_name="vllm/Qwen2.5-72B",
            backend_name="vllm-local",
            **kwargs,
        )

    if name == "deepseek-sim":
        from agentos.benchmarks.cache_sim import CacheSimulatingMockBackend
        latency = kwargs.pop("latency", None)
        return CacheSimulatingMockBackend(
            latency_base=latency or 2.0,
            cache_speedup=0.5,
            model_name="deepseek-v3",
            backend_name="deepseek-cloud",
            **kwargs,
        )

    if name == "deepseek":
        from agentos.llm.deepseek_backend import DeepSeekBackend
        return DeepSeekBackend(**kwargs)

    if name == "vllm":
        from agentos.llm.vllm_backend import VLLMBackend
        return VLLMBackend(**kwargs)

    raise ValueError(
        f"Unknown backend: {name!r}. "
        f"Available: mock, cache-sim, vllm-sim, deepseek-sim, deepseek, vllm"
    )
