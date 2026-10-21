from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentos.llm.base import LLMRequest, LLMResponse
from agentos.llm.litellm_backend import LiteLLMBackend
from agentos.llm.mock import MockLLMBackend
from agentos.metrics.model import ModelCallMetrics


@pytest.fixture
def mock_llm():
    return MockLLMBackend(responses={"default": "I am a mock response."})


async def test_mock_backend_returns_response(mock_llm: MockLLMBackend):
    req = LLMRequest(
        agent_id="a1",
        messages=[{"role": "user", "content": "hello"}],
    )
    resp = await mock_llm.generate(req)
    assert isinstance(resp, LLMResponse)
    assert resp.content == "I am a mock response."
    assert resp.tokens_used > 0


async def test_mock_backend_custom_responses():
    llm = MockLLMBackend(
        responses={
            "agent-coder": "def hello(): pass",
            "default": "generic",
        }
    )
    req = LLMRequest(
        agent_id="agent-coder", messages=[{"role": "user", "content": "write code"}]
    )
    resp = await llm.generate(req)
    assert resp.content == "def hello(): pass"


async def test_mock_backend_tracks_call_count():
    llm = MockLLMBackend(responses={"default": "ok"})
    req = LLMRequest(agent_id="a1", messages=[{"role": "user", "content": "hi"}])
    await llm.generate(req)
    await llm.generate(req)
    assert llm.call_count == 2


async def test_llm_request_has_model_field():
    req = LLMRequest(
        agent_id="a1",
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-4o-mini",
    )
    assert req.model == "gpt-4o-mini"


async def test_mock_backend_returns_metrics():
    llm = MockLLMBackend(responses={"default": "hello world"})
    req = LLMRequest(agent_id="a1", messages=[{"role": "user", "content": "hi"}])
    resp = await llm.generate(req)
    assert resp.metrics is not None
    assert isinstance(resp.metrics, ModelCallMetrics)
    assert resp.metrics.backend == "mock"
    assert resp.metrics.model == "mock"
    assert resp.backend == "mock"


async def test_mock_backend_metrics_token_counts():
    llm = MockLLMBackend(responses={"default": "one two three"})
    req = LLMRequest(
        agent_id="a1",
        messages=[{"role": "user", "content": "a" * 100}],
    )
    resp = await llm.generate(req)
    assert resp.metrics.prompt_tokens == 25
    assert resp.metrics.completion_tokens == 6
    assert resp.metrics.completion_tokens == resp.tokens_used


async def test_mock_backend_metrics_no_cache_hits():
    llm = MockLLMBackend(responses={"default": "ok"})
    req = LLMRequest(
        agent_id="a1",
        messages=[{"role": "user", "content": "a" * 40}],
    )
    resp = await llm.generate(req)
    assert resp.metrics.cache_hit_tokens == 0
    assert resp.metrics.cache_miss_tokens == resp.metrics.prompt_tokens
    assert resp.metrics.cache_hit_ratio == 0.0


class _FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


def _mock_litellm_response(prompt_tokens=50, completion_tokens=10, content="mock reply"):
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = _FakeUsage(prompt_tokens, completion_tokens)
    return resp


class TestLiteLLMBackendMetrics:
    async def test_returns_model_call_metrics(self):
        with patch("agentos.llm.litellm_backend.litellm.acompletion", new_callable=AsyncMock) as mock_ac:
            mock_ac.return_value = _mock_litellm_response()
            backend = LiteLLMBackend(default_model="test-model")
            req = LLMRequest(agent_id="a1", messages=[{"role": "user", "content": "hi"}])
            resp = await backend.generate(req)
        assert resp.metrics is not None
        assert isinstance(resp.metrics, ModelCallMetrics)
        assert resp.metrics.backend == "litellm"
        assert resp.backend == "litellm"

    async def test_metrics_token_counts(self):
        with patch("agentos.llm.litellm_backend.litellm.acompletion", new_callable=AsyncMock) as mock_ac:
            mock_ac.return_value = _mock_litellm_response(prompt_tokens=80, completion_tokens=15)
            backend = LiteLLMBackend()
            req = LLMRequest(agent_id="a1", messages=[{"role": "user", "content": "hello"}])
            resp = await backend.generate(req)
        assert resp.metrics.prompt_tokens == 80
        assert resp.metrics.completion_tokens == 15
        assert resp.tokens_used == 95

    async def test_cache_fallback_all_miss(self):
        with patch("agentos.llm.litellm_backend.litellm.acompletion", new_callable=AsyncMock) as mock_ac:
            mock_ac.return_value = _mock_litellm_response(prompt_tokens=60)
            backend = LiteLLMBackend()
            req = LLMRequest(agent_id="a1", messages=[{"role": "user", "content": "hello"}])
            resp = await backend.generate(req)
        assert resp.metrics.cache_hit_tokens == 0
        assert resp.metrics.cache_miss_tokens == 60

