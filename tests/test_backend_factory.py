from agentos.llm.factory import create_backend


def test_create_mock_backend():
    backend = create_backend("mock")
    from agentos.llm.mock import MockLLMBackend
    assert isinstance(backend, MockLLMBackend)


def test_create_cache_sim_backend():
    backend = create_backend("cache-sim")
    from agentos.benchmarks.cache_sim import CacheSimulatingMockBackend
    assert isinstance(backend, CacheSimulatingMockBackend)


def test_create_unknown_backend_raises():
    import pytest
    with pytest.raises(ValueError, match="Unknown backend"):
        create_backend("nonexistent")


def test_create_deepseek_backend():
    backend = create_backend("deepseek")
    from agentos.llm.deepseek_backend import DeepSeekBackend
    assert isinstance(backend, DeepSeekBackend)


def test_create_vllm_backend():
    backend = create_backend("vllm")
    from agentos.llm.vllm_backend import VLLMBackend
    assert isinstance(backend, VLLMBackend)


def test_create_vllm_sim_backend():
    backend = create_backend("vllm-sim")
    from agentos.benchmarks.cache_sim import CacheSimulatingMockBackend
    assert isinstance(backend, CacheSimulatingMockBackend)
    assert backend._model_name == "vllm/Qwen2.5-72B"
    assert backend._backend_name == "vllm-local"
    assert backend._latency_base == 0.8
    assert backend._cache_speedup == 0.7


def test_create_deepseek_sim_backend():
    backend = create_backend("deepseek-sim")
    from agentos.benchmarks.cache_sim import CacheSimulatingMockBackend
    assert isinstance(backend, CacheSimulatingMockBackend)
    assert backend._model_name == "deepseek-v3"
    assert backend._backend_name == "deepseek-cloud"
    assert backend._latency_base == 2.0
    assert backend._cache_speedup == 0.5


def test_vllm_sim_faster_than_deepseek_sim():
    vllm = create_backend("vllm-sim")
    deepseek = create_backend("deepseek-sim")
    assert vllm._latency_base < deepseek._latency_base


def test_sim_backends_accept_latency_override():
    vllm = create_backend("vllm-sim", latency=1.5)
    assert vllm._latency_base == 1.5
    deepseek = create_backend("deepseek-sim", latency=3.0)
    assert deepseek._latency_base == 3.0
