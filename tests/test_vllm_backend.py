from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentos.llm.base import LLMRequest
from agentos.llm.vllm_backend import VLLMBackend
from agentos.metrics.model import ModelCallMetrics


def _make_request(content="hello") -> LLMRequest:
    return LLMRequest(
        agent_id="test-agent",
        messages=[{"role": "user", "content": content}],
    )


def _make_chat_response(content="Hi!", prompt_tokens=50, completion_tokens=10):
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": None,
        },
        "model": "Qwen2-7B-Instruct",
    }


def _make_models_response(model_id="Qwen2-7B-Instruct"):
    return {"data": [{"id": model_id}]}


def _make_metrics_text(
    queries=0.0,
    hits=0.0,
    prefill_tokens=0.0,
    decode_tokens=0.0,
    prefill_secs=0.0,
    decode_secs=0.0,
):
    return (
        f'vllm:prefix_cache_queries_total{{engine="0",model_name="m"}} {queries}\n'
        f'vllm:prefix_cache_hits_total{{engine="0",model_name="m"}} {hits}\n'
        f'vllm:prompt_tokens_total{{engine="0",model_name="m"}} {prefill_tokens}\n'
        f'vllm:generation_tokens_total{{engine="0",model_name="m"}} {decode_tokens}\n'
        f'vllm:prefill_time_seconds_sum{{engine="0",model_name="m"}} {prefill_secs}\n'
        f'vllm:decode_time_seconds_sum{{engine="0",model_name="m"}} {decode_secs}\n'
        f'vllm:kv_cache_usage_perc{{engine="0",model_name="m"}} 0.0\n'
    )


def _setup_mock_client(chat_resp, metrics_before, metrics_after, models_resp=None):
    """Create a mock httpx.AsyncClient that handles /v1/chat/completions, /metrics, and /v1/models."""
    call_count = {"metrics": 0}

    mock_client = AsyncMock()

    async def mock_get(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "/v1/models" in url:
            resp.json.return_value = models_resp or _make_models_response()
            resp.status_code = 200
        elif "/metrics" in url:
            call_count["metrics"] += 1
            resp.text = metrics_before if call_count["metrics"] <= 1 else metrics_after
            resp.status_code = 200
        return resp

    async def mock_post(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = chat_resp
        resp.status_code = 200
        return resp

    mock_client.get = mock_get
    mock_client.post = mock_post
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


class TestVLLMBackend:
    async def test_basic_generation(self):
        chat = _make_chat_response(content="Hello world")
        client = _setup_mock_client(chat, _make_metrics_text(0, 0), _make_metrics_text(50, 0))
        with patch("agentos.llm.vllm_backend.httpx.AsyncClient", return_value=client):
            backend = VLLMBackend(metrics_mode="exclusive", model="Qwen2-7B-Instruct")
            resp = await backend.generate(_make_request())
        assert resp.content == "Hello world"
        assert resp.backend == "vllm"

    async def test_returns_model_call_metrics(self):
        chat = _make_chat_response()
        client = _setup_mock_client(chat, _make_metrics_text(0, 0), _make_metrics_text(50, 0))
        with patch("agentos.llm.vllm_backend.httpx.AsyncClient", return_value=client):
            backend = VLLMBackend(metrics_mode="exclusive", model="Qwen2-7B-Instruct")
            resp = await backend.generate(_make_request())
        assert isinstance(resp.metrics, ModelCallMetrics)
        assert resp.metrics.backend == "vllm"

    async def test_cache_hit_from_prometheus_delta(self):
        chat = _make_chat_response(prompt_tokens=75)
        client = _setup_mock_client(
            chat,
            _make_metrics_text(queries=100, hits=30),
            _make_metrics_text(queries=175, hits=94),
        )
        with patch("agentos.llm.vllm_backend.httpx.AsyncClient", return_value=client):
            backend = VLLMBackend(metrics_mode="exclusive", model="Qwen2-7B-Instruct")
            resp = await backend.generate(_make_request())
        assert resp.metrics.cache_hit_tokens == 64
        assert resp.metrics.cache_miss_tokens == 11

    async def test_cache_hit_ratio(self):
        chat = _make_chat_response(prompt_tokens=77)
        client = _setup_mock_client(
            chat,
            _make_metrics_text(queries=100, hits=0),
            _make_metrics_text(queries=177, hits=64),
        )
        with patch("agentos.llm.vllm_backend.httpx.AsyncClient", return_value=client):
            backend = VLLMBackend(metrics_mode="exclusive", model="Qwen2-7B-Instruct")
            resp = await backend.generate(_make_request())
        ratio = resp.metrics.cache_hit_ratio
        assert ratio == pytest.approx(64 / (64 + 13), rel=1e-3)

    async def test_no_cache_hits_first_call(self):
        chat = _make_chat_response(prompt_tokens=50)
        client = _setup_mock_client(
            chat,
            _make_metrics_text(queries=0, hits=0),
            _make_metrics_text(queries=50, hits=0),
        )
        with patch("agentos.llm.vllm_backend.httpx.AsyncClient", return_value=client):
            backend = VLLMBackend(metrics_mode="exclusive", model="Qwen2-7B-Instruct")
            resp = await backend.generate(_make_request())
        assert resp.metrics.cache_hit_tokens == 0
        assert resp.metrics.cache_miss_tokens == 50

    async def test_auto_resolve_model(self):
        chat = _make_chat_response()
        client = _setup_mock_client(
            chat,
            _make_metrics_text(0, 0),
            _make_metrics_text(50, 0),
            models_resp=_make_models_response("auto-detected-model"),
        )
        with patch("agentos.llm.vllm_backend.httpx.AsyncClient", return_value=client):
            backend = VLLMBackend()
            resp = await backend.generate(_make_request())
        assert resp.model == "auto-detected-model"

    async def test_metrics_fallback_on_prometheus_failure(self):
        chat = _make_chat_response(prompt_tokens=40)

        mock_client = AsyncMock()

        async def mock_get(url, **kwargs):
            if "/metrics" in url:
                raise ConnectionError("metrics unavailable")
            resp = MagicMock()
            resp.json.return_value = _make_models_response()
            return resp

        async def mock_post(url, **kwargs):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = chat
            return resp

        mock_client.get = mock_get
        mock_client.post = mock_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("agentos.llm.vllm_backend.httpx.AsyncClient", return_value=mock_client):
            backend = VLLMBackend(metrics_mode="exclusive", model="test")
            resp = await backend.generate(_make_request())
        assert resp.metrics.cache_hit_tokens == 0
        assert resp.metrics.cache_miss_tokens == 40

    async def test_token_counts(self):
        chat = _make_chat_response(prompt_tokens=100, completion_tokens=25)
        client = _setup_mock_client(chat, _make_metrics_text(0, 0), _make_metrics_text(100, 0))
        with patch("agentos.llm.vllm_backend.httpx.AsyncClient", return_value=client):
            backend = VLLMBackend(metrics_mode="exclusive", model="Qwen2-7B-Instruct")
            resp = await backend.generate(_make_request())
        assert resp.metrics.prompt_tokens == 100
        assert resp.metrics.completion_tokens == 25
        assert resp.tokens_used == 125

    async def test_metadata_includes_cache_queries(self):
        chat = _make_chat_response()
        client = _setup_mock_client(
            chat,
            _make_metrics_text(queries=10, hits=5),
            _make_metrics_text(queries=60, hits=35),
        )
        with patch("agentos.llm.vllm_backend.httpx.AsyncClient", return_value=client):
            backend = VLLMBackend(metrics_mode="exclusive", model="Qwen2-7B-Instruct")
            resp = await backend.generate(_make_request())
        assert resp.metadata["cache_queries"] == 50

    async def test_prefill_decode_metrics_from_prometheus_delta(self):
        chat = _make_chat_response(prompt_tokens=100, completion_tokens=25)
        client = _setup_mock_client(
            chat,
            _make_metrics_text(
                queries=10,
                hits=5,
                prefill_tokens=1000,
                decode_tokens=200,
                prefill_secs=3.0,
                decode_secs=7.0,
            ),
            _make_metrics_text(
                queries=110,
                hits=65,
                prefill_tokens=1100,
                decode_tokens=225,
                prefill_secs=3.4,
                decode_secs=7.8,
            ),
        )
        with patch("agentos.llm.vllm_backend.httpx.AsyncClient", return_value=client):
            backend = VLLMBackend(metrics_mode="exclusive", model="Qwen2-7B-Instruct")
            resp = await backend.generate(_make_request())

        assert resp.metrics.metadata["prefill_tokens"] == 100
        assert resp.metrics.metadata["decode_tokens"] == 25
        assert resp.metrics.metadata["prefill_secs"] == pytest.approx(0.4)
        assert resp.metrics.metadata["decode_secs"] == pytest.approx(0.8)
        assert resp.metrics.metadata["prefill_decode_available"] is True
        assert resp.metrics.metadata["prefill_decode_fallback"] is False
        assert resp.metadata["prefill_saved_tokens"] == 60

    async def test_prefill_decode_fallback_when_prometheus_split_missing(self):
        chat = _make_chat_response(prompt_tokens=40, completion_tokens=9)
        client = _setup_mock_client(
            chat,
            _make_metrics_text(queries=0, hits=0),
            _make_metrics_text(queries=40, hits=0),
        )
        with patch("agentos.llm.vllm_backend.httpx.AsyncClient", return_value=client):
            backend = VLLMBackend(metrics_mode="exclusive", model="Qwen2-7B-Instruct")
            resp = await backend.generate(_make_request())

        assert resp.metrics.metadata["prefill_tokens"] == 40
        assert resp.metrics.metadata["decode_tokens"] == 9
        assert resp.metrics.metadata["prefill_decode_available"] is False
        assert resp.metrics.metadata["prefill_decode_fallback"] is True


async def test_default_vllm_metrics_use_response_cache_details_only():
    chat = _make_chat_response(prompt_tokens=100)
    chat["usage"]["prompt_tokens_details"] = {"cached_tokens": 64}
    client = _setup_mock_client(chat, _make_metrics_text(1000, 900), _make_metrics_text(9000, 8000))
    with patch("agentos.llm.vllm_backend.httpx.AsyncClient", return_value=client):
        response = await VLLMBackend(model="test").generate(_make_request())
    assert response.metrics.cache_hit_tokens == 64
    assert response.metrics.cache_miss_tokens == 36
    assert response.metrics.metadata["cache_metric_source"] == "response_usage"
    assert response.metrics.metadata["prefill_decode_available"] is False


@pytest.mark.parametrize("details", [None, {}, {"cached_tokens": None}])
async def test_missing_vllm_cache_details_remain_unknown_to_gateway(details):
    from agentos.context.cache_tracker import CacheHeatTracker
    from agentos.llm.gateway import ModelGateway
    chat = _make_chat_response()
    chat["usage"]["prompt_tokens_details"] = details
    client = _setup_mock_client(chat, _make_metrics_text(), _make_metrics_text(10000, 9000))
    tracker = CacheHeatTracker()
    with patch("agentos.llm.vllm_backend.httpx.AsyncClient", return_value=client):
        gateway = ModelGateway({"local": VLLMBackend(model="test")}, default_backend="local", cache_tracker=tracker)
        response = await gateway.generate(_make_request())
    assert response.metrics.metadata["cache_metrics_available"] is False
    assert response.metrics.cache_hit_tokens == response.metrics.cache_miss_tokens == 0
    assert tracker.get_hot_prefixes() == []
    assert gateway._call_log[0]["cache_hit_ratio"] is None
    assert gateway.stats()["cache_hit_ratio"] is None
    assert gateway.stats()["cache_metrics_available_calls"] == 0
    assert gateway.stats()["cache_metrics_complete"] is False


async def test_request_mode_forwards_thinking_control_and_bounded_output():
    client = _setup_mock_client(_make_chat_response(), "", "")
    posted = []
    original_post = client.post
    async def post(url, **kwargs):
        posted.append(kwargs["json"])
        return await original_post(url, **kwargs)
    client.post = post
    with patch("agentos.llm.vllm_backend.httpx.AsyncClient", return_value=client):
        await VLLMBackend(model="test", max_output_tokens=128, chat_template_kwargs={"enable_thinking": False}).generate(_make_request())
    assert posted[0]["max_tokens"] == 128
    assert posted[0]["chat_template_kwargs"] == {"enable_thinking": False}
