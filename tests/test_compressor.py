import pytest

from agentos.context.compressor import (
    Compressor,
    SlidingWindowCompressor,
    TokenBudgetCompressor,
)
from agentos.context.manager import ContextManager


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


class TestSlidingWindowCompressor:
    async def test_keeps_all_when_under_window(self):
        comp = SlidingWindowCompressor(window_size=10)
        msgs = [_msg("system", "sys"), _msg("user", "hi"), _msg("assistant", "hello")]
        result = await comp.compress(msgs)
        assert result == msgs

    async def test_trims_old_turns(self):
        comp = SlidingWindowCompressor(window_size=2)
        msgs = [
            _msg("system", "sys"),
            _msg("user", "q1"),
            _msg("assistant", "a1"),
            _msg("user", "q2"),
            _msg("assistant", "a2"),
        ]
        result = await comp.compress(msgs)
        assert len(result) == 3
        assert result[0] == _msg("system", "sys")
        assert result[1] == _msg("user", "q2")
        assert result[2] == _msg("assistant", "a2")

    async def test_preserves_system_prompt(self):
        comp = SlidingWindowCompressor(window_size=1)
        msgs = [
            _msg("system", "you are helpful"),
            _msg("user", "old"),
            _msg("user", "new"),
        ]
        result = await comp.compress(msgs)
        assert result[0]["role"] == "system"
        assert result[1]["content"] == "new"

    async def test_empty_messages(self):
        comp = SlidingWindowCompressor(window_size=5)
        result = await comp.compress([])
        assert result == []

    async def test_system_only(self):
        comp = SlidingWindowCompressor(window_size=3)
        msgs = [_msg("system", "sys")]
        result = await comp.compress(msgs)
        assert result == [_msg("system", "sys")]

    async def test_multiple_system_messages(self):
        comp = SlidingWindowCompressor(window_size=1)
        msgs = [
            _msg("system", "sys1"),
            _msg("system", "sys2"),
            _msg("user", "old"),
            _msg("user", "new"),
        ]
        result = await comp.compress(msgs)
        assert len(result) == 3
        assert result[0]["content"] == "sys1"
        assert result[1]["content"] == "sys2"
        assert result[2]["content"] == "new"


class TestTokenBudgetCompressor:
    async def test_keeps_all_when_under_budget(self):
        comp = TokenBudgetCompressor(budget=10000)
        msgs = [_msg("system", "sys"), _msg("user", "hi")]
        result = await comp.compress(msgs)
        assert result == msgs

    async def test_trims_to_budget(self):
        comp = TokenBudgetCompressor(budget=10)
        msgs = [
            _msg("system", "ok"),
            _msg("user", "a" * 100),
            _msg("user", "b" * 8),
        ]
        result = await comp.compress(msgs)
        assert result[0]["role"] == "system"
        assert result[-1]["content"] == "b" * 8
        assert len(result) == 2

    async def test_max_tokens_override(self):
        comp = TokenBudgetCompressor(budget=10000)
        msgs = [
            _msg("system", "ok"),
            _msg("user", "a" * 100),
            _msg("user", "short"),
        ]
        result = await comp.compress(msgs, max_tokens=5)
        assert result[-1]["content"] == "short"

    async def test_custom_estimator(self):
        comp = TokenBudgetCompressor(budget=3, token_estimator=lambda s: len(s))
        msgs = [
            _msg("system", "x"),
            _msg("user", "ab"),
        ]
        result = await comp.compress(msgs)
        assert len(result) == 2

    async def test_system_exceeds_budget(self):
        comp = TokenBudgetCompressor(budget=1)
        msgs = [
            _msg("system", "a" * 100),
            _msg("user", "hi"),
        ]
        result = await comp.compress(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "system"


class TestCompressorProtocol:
    def test_sliding_window_is_compressor(self):
        assert isinstance(SlidingWindowCompressor(), Compressor)

    def test_token_budget_is_compressor(self):
        assert isinstance(TokenBudgetCompressor(), Compressor)


class TestContextManagerWithCompressor:
    async def test_async_build_messages_without_compressor(self):
        mgr = ContextManager()
        mgr.add_shared("bg", "background info")
        result = await mgr.async_build_messages("agent-1", [_msg("user", "hello")])
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["content"] == "hello"

    async def test_async_build_messages_with_compressor(self):
        comp = SlidingWindowCompressor(window_size=1)
        mgr = ContextManager(compressor=comp)
        mgr.add_shared("bg", "background info")
        user_msgs = [_msg("user", "old"), _msg("user", "new")]
        result = await mgr.async_build_messages("agent-1", user_msgs)
        assert result[0]["role"] == "system"
        assert result[-1]["content"] == "new"
        assert len(result) == 2

    async def test_sync_build_messages_unaffected_by_compressor(self):
        comp = SlidingWindowCompressor(window_size=1)
        mgr = ContextManager(compressor=comp)
        mgr.add_shared("bg", "background")
        user_msgs = [_msg("user", "old"), _msg("user", "new")]
        result = mgr.build_messages("agent-1", user_msgs)
        assert len(result) == 3
