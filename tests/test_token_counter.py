from agentos.context.token_counter import (
    CharEstimateCounter,
    TiktokenCounter,
    TokenCounter,
    get_default_counter,
    set_default_counter,
)
from agentos.context.pool import SharedContextPool


class TestCharEstimateCounter:
    def test_basic_estimate(self):
        counter = CharEstimateCounter()
        assert counter.count("hello world") >= 1

    def test_empty_string(self):
        counter = CharEstimateCounter()
        assert counter.count("") == 1

    def test_long_string(self):
        counter = CharEstimateCounter()
        text = "a" * 400
        assert counter.count(text) == 100

    def test_implements_protocol(self):
        assert isinstance(CharEstimateCounter(), TokenCounter)


class TestTiktokenCounter:
    def test_fallback_without_tiktoken(self):
        counter = TiktokenCounter(model="nonexistent_model_xyz")
        assert counter.count("hello world") >= 1

    def test_implements_protocol(self):
        assert isinstance(TiktokenCounter(), TokenCounter)


class TestDefaultCounter:
    def test_get_default_is_char_estimate(self):
        counter = get_default_counter()
        assert isinstance(counter, CharEstimateCounter)

    def test_set_and_get_default(self):
        original = get_default_counter()
        try:
            custom = CharEstimateCounter()
            set_default_counter(custom)
            assert get_default_counter() is custom
        finally:
            set_default_counter(original)


class TestPoolWithTokenCounter:
    def test_pool_uses_injected_counter(self):
        class FixedCounter:
            def count(self, text: str) -> int:
                return 42

        pool = SharedContextPool(token_counter=FixedCounter())
        pool.store("key", "any content here")
        stats = pool.stats()
        assert stats["total_estimated_tokens"] == 42

    def test_pool_uses_default_counter(self):
        pool = SharedContextPool()
        pool.store("key", "a" * 100)
        stats = pool.stats()
        assert stats["total_estimated_tokens"] == 25

    def test_pool_dedup_with_counter(self):
        class CountingCounter:
            def __init__(self):
                self.calls = 0

            def count(self, text: str) -> int:
                self.calls += 1
                return 10

        counter = CountingCounter()
        pool = SharedContextPool(token_counter=counter)
        pool.store("k1", "same content")
        pool.store("k2", "same content")
        assert counter.calls == 1
