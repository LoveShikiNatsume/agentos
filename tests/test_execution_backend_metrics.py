"""Synthetic HTTP responses only: no real API calls or repository reruns."""
import json

import httpx
import pytest

from agentos.adapters.workflow import PlannedTask, adapt_plan
from agentos.context.cache_tracker import CacheHeatTracker
from agentos.context.manager import ContextManager
from agentos.execution.backend import OpenAIBackend, RecordedBackend
from agentos.execution.records import Records, usage_summary
from agentos.llm.base import LLMRequest
from agentos.llm.gateway import ModelGateway
from agentos.runtime.engine import Runtime


CONFIG = {'model': 'requested', 'base_url': 'http://synthetic.test',
          'api_key_env': 'UNUSED_TEST_KEY', 'timeout': 1, 'max_tokens': None,
          'temperature': 0, 'thinking': 'enabled', 'reasoning_effort': 'high', 'retries': 0}


def wire(monkeypatch, usages):
    usages = iter(usages)
    def handler(request):
        return httpx.Response(200, json={'model': 'returned-model', 'choices': [
            {'message': {'role': 'assistant', 'content': 'observed'}, 'finish_reason': 'stop'}],
            'usage': next(usages)})
    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs))


@pytest.mark.parametrize('usage,expected,cache_available,source', [
    ({'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120,
      'prompt_cache_hit_tokens': 60, 'prompt_cache_miss_tokens': 40},
     (100, 20, 60, 40), True, 'response_usage'),
    ({'prompt_tokens': 100, 'completion_tokens': 20, 'prompt_tokens_details': {'cached_tokens': 60}},
     (100, 20, 60, 40), True, 'reported_hit_and_prompt_minus_hit'),
    ({'prompt_tokens': 100, 'completion_tokens': 20, 'prompt_tokens_details': {'cached_tokens': 0}},
     (100, 20, 0, 100), True, 'reported_hit_and_prompt_minus_hit'),
    ({'prompt_tokens': 100, 'completion_tokens': 20},
     (100, 20, None, None), False, 'unavailable'),
    (None, (None, None, None, None), False, 'unavailable'),
    ({'prompt_tokens': 5}, (5, None, None, None), False, 'unavailable'),
    ({'prompt_tokens': 100, 'completion_tokens': 20, 'prompt_cache_hit_tokens': 120},
     (100, 20, None, None), False, 'invalid_response_usage'),
    ({'prompt_tokens': 100, 'completion_tokens': 20, 'prompt_cache_hit_tokens': 60, 'prompt_cache_miss_tokens': 50},
     (100, 20, None, None), False, 'invalid_response_usage'),
    ({'prompt_tokens': True, 'completion_tokens': '20', 'prompt_tokens_details': {'cached_tokens': -1}},
     (None, None, None, None), False, 'invalid_response_usage'),
])
async def test_response_usage_populates_existing_metrics(monkeypatch, usage, expected, cache_available, source):
    wire(monkeypatch, [usage])
    response = await OpenAIBackend(CONFIG).generate(LLMRequest('a', [{'role': 'user', 'content': 'task'}]))
    metrics = response.metrics
    assert metrics is not None and metrics.model == 'returned-model'
    fields = ('prompt_tokens', 'completion_tokens', 'cache_hit_tokens', 'cache_miss_tokens')
    for field, value in zip(fields, expected):
        assert metrics.metadata['observed_tokens'][field] == value
        assert metrics.metadata[field + '_available'] == (value is not None)
        assert getattr(metrics, field) == (value if value is not None else 0)
    assert metrics.metadata['cache_metrics_available'] is cache_available
    assert metrics.metadata['cache_metric_source'] == source
    assert metrics.metadata['prefill_decode_available'] is False
    assert metrics.metadata['queue_wait_available'] is False
    assert response.metadata['usage'] == usage
    if expected[:2] == (100, 20):
        assert response.tokens_used == 120


async def test_same_response_reaches_runtime_events_and_gateway_cache_feedback(monkeypatch, tmp_path):
    known = {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120,
             'prompt_cache_hit_tokens': 60, 'prompt_cache_miss_tokens': 40}
    # Missing cache usage must not become an observed miss/zero-hit sample.
    missing_cache = {'prompt_tokens': 28, 'completion_tokens': 7, 'total_tokens': 35}
    wire(monkeypatch, [known, missing_cache])
    records = Records(tmp_path, 'synthetic-metrics')
    recorder = RecordedBackend(OpenAIBackend(CONFIG), records, CONFIG)
    tracker = CacheHeatTracker()
    gateway = ModelGateway({'configured': recorder}, default_backend='configured', cache_tracker=tracker)
    context = ContextManager()
    context.add_shared('policy', 'shared task policy')
    async def run(ctx):
        for _ in range(2):
            await ctx.llm_call([{'role': 'user', 'content': 'task'}])
        return 'done'
    dag, handlers = adapt_plan([PlannedTask('one', 'task', 'r', context_refs=('policy',))], {'r': run}, framework='synthetic')
    runtime = Runtime(gateway, handlers, context_manager=context, event_log=records, enable_kernel_probe=False)
    await runtime.run(dag)
    events = list(records.read(event_type='llm.response_received'))
    assert [(e.data['prompt_tokens'], e.data['completion_tokens']) for e in events] == [(100, 20), (28, 7)]
    assert events[0].data['cache_hit_tokens'] == 60 and events[0].data['cache_miss_tokens'] == 40
    prefix = next(records.read(event_type='context.assembled')).data['prefix_id']
    assert tracker.predict_reuse(prefix).samples == 1
    stats = gateway.stats()
    assert stats['total_prompt_tokens'] == 128 and stats['total_completion_tokens'] == 27
    assert stats['total_cache_hit_tokens'] == 60 and stats['total_cache_miss_tokens'] == 40
    assert stats['cache_metrics_available_calls'] == 1 and not stats['cache_metrics_complete']
    assert stats['cache_hit_ratio'] == .6
    calls = [json.loads(line) for line in (tmp_path/'calls.jsonl').read_text().splitlines()]
    assert [c['usage'] for c in calls] == [known, missing_cache]
    assert calls[1]['metrics']['metadata']['observed_tokens']['cache_miss_tokens'] is None
    completed = list(records.read(event_type='execution.model_completed'))
    assert completed[1].data['metrics']['metadata']['cache_metrics_available'] is False
    assert usage_summary(calls)['prompt_tokens'] == 128


async def test_missing_usage_does_not_feed_cache_or_become_measured_zero(monkeypatch, tmp_path):
    wire(monkeypatch, [None])
    recorder = RecordedBackend(OpenAIBackend(CONFIG), Records(tmp_path, 'missing'), CONFIG)
    tracker = CacheHeatTracker()
    gateway = ModelGateway({'configured': recorder}, default_backend='configured', cache_tracker=tracker)
    response = await gateway.generate(LLMRequest('a', [{'role': 'user', 'content': 'task'}], prefix_id='p'))
    assert response.metrics.metadata['observed_tokens'] == {
        'prompt_tokens': None, 'completion_tokens': None, 'total_tokens': None,
        'cache_hit_tokens': None, 'cache_miss_tokens': None}
    assert tracker.predict_reuse('p').samples == 0
    assert gateway.stats()['cache_hit_ratio'] is None
    assert usage_summary(recorder.calls)['prompt_tokens'] is None
    assert not usage_summary(recorder.calls)['usage_complete']
