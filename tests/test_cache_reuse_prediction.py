import pytest

from agentos.context.cache_tracker import CacheHeatTracker
from agentos.metrics.model import ModelCallMetrics


def _metrics(prompt=1000, hit=800, miss=200, prefill_secs=0.2):
    return ModelCallMetrics(
        backend="vllm",
        model="qwen",
        prompt_tokens=prompt,
        cache_hit_tokens=hit,
        cache_miss_tokens=miss,
        metadata={
            "prefill_miss_tokens": miss,
            "prefill_secs": prefill_secs,
        },
    )


def test_predict_reuse_estimates_expected_hit_tokens_and_latency():
    tracker = CacheHeatTracker(ttl=60.0)
    tracker.record_call("hot", _metrics(), timestamp=0.0)
    tracker.record_call("hot", _metrics(prompt=1200, hit=900, miss=300), timestamp=1.0)

    prediction = tracker.predict_reuse("hot", now=1.0)

    assert prediction.reuse_probability > 0.7
    assert prediction.expected_hit_tokens > 800
    assert prediction.expected_prefill_saved_secs > 0.0
    assert prediction.confidence == pytest.approx(0.25)
    assert prediction.samples == 2


def test_predict_reuse_decays_with_time():
    tracker = CacheHeatTracker(ttl=60.0)
    tracker.record_call("hot", _metrics(), timestamp=0.0)

    fresh = tracker.predict_reuse("hot", now=0.0)
    stale = tracker.predict_reuse("hot", now=180.0)

    assert fresh.reuse_probability > stale.reuse_probability
    assert stale.expected_hit_tokens < fresh.expected_hit_tokens


def test_reuse_predictions_sorted_by_expected_hit_tokens():
    tracker = CacheHeatTracker(ttl=60.0)
    tracker.record_call("small", _metrics(prompt=100, hit=80, miss=20), timestamp=0.0)
    tracker.record_call("large", _metrics(prompt=2000, hit=1500, miss=500), timestamp=1.0)

    predictions = tracker.get_reuse_predictions(now=1.0)

    assert predictions[0].prefix_id == "large"
    assert tracker.stats(now=1.0)["predictions"][0]["prefix_id"] == "large"
