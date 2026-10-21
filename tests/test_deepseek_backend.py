from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentos.llm.base import LLMRequest
from agentos.llm.deepseek_backend import DeepSeekBackend
from agentos.metrics.model import ModelCallMetrics


def _make_request(content="hello") -> LLMRequest:
    return LLMRequest(
        agent_id="test-agent",
        messages=[{"role": "user", "content": content}],
    )


def _make_httpx_response(
    content="Hello!",
    prompt_tokens=100,
    completion_tokens=20,
    cache_hit=80,
    cache_miss=20,
    status_code=200,
):
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_cache_hit_tokens": cache_hit,
        "prompt_cache_miss_tokens": cache_miss,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    body = {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": usage,
        "model": "deepseek-chat",
    }
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = body
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def _patch_httpx(mock_response):
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return patch("agentos.llm.deepseek_backend.httpx.AsyncClient", return_value=mock_client)


class TestDeepSeekBackend:
    async def test_parses_cache_hit_tokens(self):
        resp = _make_httpx_response(cache_hit=80, cache_miss=20)
        with _patch_httpx(resp):
            backend = DeepSeekBackend(api_key="test-key")
            result = await backend.generate(_make_request())
        assert result.metrics.cache_hit_tokens == 80

    async def test_parses_cache_miss_tokens(self):
        resp = _make_httpx_response(cache_hit=30, cache_miss=70)
        with _patch_httpx(resp):
            backend = DeepSeekBackend(api_key="test-key")
            result = await backend.generate(_make_request())
        assert result.metrics.cache_miss_tokens == 70

    async def test_cache_hit_ratio(self):
        resp = _make_httpx_response(cache_hit=80, cache_miss=20)
        with _patch_httpx(resp):
            backend = DeepSeekBackend(api_key="test-key")
            result = await backend.generate(_make_request())
        assert result.metrics.cache_hit_ratio == pytest.approx(0.8)

    async def test_fallback_when_no_cache_fields(self):
        body = {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
            "model": "deepseek-chat",
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = body
        mock_resp.raise_for_status = MagicMock()
        with _patch_httpx(mock_resp):
            backend = DeepSeekBackend(api_key="test-key")
            result = await backend.generate(_make_request())
        assert result.metrics.cache_hit_tokens == 0
        assert result.metrics.cache_miss_tokens == 50

    async def test_returns_model_call_metrics(self):
        resp = _make_httpx_response()
        with _patch_httpx(resp):
            backend = DeepSeekBackend(api_key="test-key")
            result = await backend.generate(_make_request())
        assert isinstance(result.metrics, ModelCallMetrics)
        assert result.metrics.backend == "deepseek"

    async def test_content_extraction(self):
        resp = _make_httpx_response(content="The answer is 42.")
        with _patch_httpx(resp):
            backend = DeepSeekBackend(api_key="test-key")
            result = await backend.generate(_make_request())
        assert result.content == "The answer is 42."

    async def test_metadata_preserves_raw_usage(self):
        resp = _make_httpx_response(prompt_tokens=100, completion_tokens=20, cache_hit=80, cache_miss=20)
        with _patch_httpx(resp):
            backend = DeepSeekBackend(api_key="test-key")
            result = await backend.generate(_make_request())
        assert "usage" in result.metadata
        assert result.metadata["usage"]["prompt_tokens"] == 100
        assert result.metadata["usage"]["prompt_cache_hit_tokens"] == 80

    async def test_uses_env_api_key(self):
        resp = _make_httpx_response()
        with _patch_httpx(resp) as mock_cls, patch.dict("os.environ", {"DEEPSEEK_API_KEY": "env-key-123"}):
            backend = DeepSeekBackend()
            assert backend._api_key == "env-key-123"

    async def test_custom_model_from_request(self):
        resp = _make_httpx_response()
        with _patch_httpx(resp) as mock_cls:
            backend = DeepSeekBackend(api_key="test-key")
            req = LLMRequest(agent_id="a", messages=[{"role": "user", "content": "hi"}], model="deepseek-coder")
            result = await backend.generate(req)
        assert result.model == "deepseek-coder"
        assert result.metrics.model == "deepseek-coder"

    async def test_tokens_used_sum(self):
        resp = _make_httpx_response(prompt_tokens=100, completion_tokens=25)
        with _patch_httpx(resp):
            backend = DeepSeekBackend(api_key="test-key")
            result = await backend.generate(_make_request())
        assert result.tokens_used == 125

    async def test_backend_field(self):
        resp = _make_httpx_response()
        with _patch_httpx(resp):
            backend = DeepSeekBackend(api_key="test-key")
            result = await backend.generate(_make_request())
        assert result.backend == "deepseek"
