"""Tests for MessageBus benchmark script."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentos.comm.bus import MessageBus
from agentos.core.types import Message


async def test_inline_payload_statistics_correct():
    """Inline messages should accumulate payload bytes."""
    bus = MessageBus()
    large = "X" * 500
    for i in range(5):
        await bus.send(Message(
            from_agent="a1", to_agent="a2",
            content={"text": large, "i": i},
            message_type="inline",
        ))
    stats = bus.stats(message_type="inline")
    assert stats["message_count"] == 5
    assert stats["payload_bytes"] > 500 * 5  # JSON overhead


async def test_context_ref_payload_less_than_inline():
    """ContextRef messages should have significantly less payload than inline."""
    bus = MessageBus()
    large = "Y" * 1000

    # Inline
    for i in range(10):
        await bus.send(Message(
            from_agent="a1", to_agent="a2",
            content={"full_text": large},
            message_type="inline",
        ))

    # ContextRef
    for i in range(10):
        await bus.send(Message(
            from_agent="a1", to_agent="a2",
            content={"ref": "ctx://shared"},
            message_type="context_ref",
            context_refs=["ctx://shared"],
        ))

    savings = bus.payload_savings(
        inline_message_type="inline",
        ref_message_type="context_ref",
    )
    assert savings["inline_payload_bytes"] > savings["ref_payload_bytes"]
    assert savings["payload_reduction_ratio"] > 0.5


async def test_payload_reduction_ratio_positive():
    """Payload reduction ratio should be > 0 when inline > ref."""
    bus = MessageBus()
    await bus.send(Message(
        from_agent="a1", to_agent="a2",
        content={"data": "A" * 500},
        message_type="inline",
    ))
    await bus.send(Message(
        from_agent="a1", to_agent="a2",
        content={"ref": "ctx://x"},
        message_type="context_ref",
        context_refs=["ctx://x"],
    ))
    savings = bus.payload_savings(
        inline_message_type="inline",
        ref_message_type="context_ref",
    )
    assert savings["payload_reduction_ratio"] > 0


async def test_priority_ordering_correct():
    """High-priority messages should be received first."""
    bus = MessageBus()
    await bus.send(Message(
        from_agent="s", to_agent="w",
        content={"n": 1}, priority=1,
    ))
    await bus.send(Message(
        from_agent="s", to_agent="w",
        content={"n": 2}, priority=10,
    ))
    first = await bus.receive("w")
    second = await bus.receive("w")
    assert first is not None and first.content["n"] == 2
    assert second is not None and second.content["n"] == 1


async def test_history_query_by_agent_task_type():
    """History queries should filter by agent, task, and type."""
    bus = MessageBus()
    for i in range(6):
        await bus.send(Message(
            from_agent="a1", to_agent="a2",
            content={"i": i},
            message_type="status" if i % 2 == 0 else "result",
            task_id=f"t-{i % 3}",
        ))

    by_task = bus.get_history("a2", task_id="t-0")
    assert all(m.task_id == "t-0" for m in by_task)
    assert len(by_task) > 0

    by_type = bus.get_history("a2", message_type="result")
    assert all(m.message_type == "result" for m in by_type)

    by_all = bus.get_history("a2", task_id="t-1", message_type="result")
    assert all(m.task_id == "t-1" and m.message_type == "result" for m in by_all)


async def test_benchmark_generates_summary_json(tmp_path: Path):
    """Benchmark script should generate summary.json."""
    from scripts.benchmark_messagebus import run_benchmark

    out_path = tmp_path / "summary.json"
    summary = await run_benchmark(
        message_count=10,
        payload_size=1024,
        out_path=out_path,
    )

    assert out_path.exists()
    loaded = json.loads(out_path.read_text())

    assert "metadata" in loaded
    assert "aggregate" in loaded
    assert loaded["aggregate"]["inline_payload_bytes"] > 0
    assert loaded["aggregate"]["ref_payload_bytes"] > 0
    assert loaded["aggregate"]["bytes_saved"] > 0
    assert loaded["aggregate"]["payload_reduction_ratio"] > 0
    assert loaded["aggregate"]["priority_order_correct"] is True
    assert loaded["aggregate"]["history_query_correct"] is True

    # report.md should also be generated
    report_path = tmp_path / "report.md"
    assert report_path.exists()
    report_text = report_path.read_text()
    assert "MessageBus Communication Benchmark Report" in report_text
    assert "Payload Comparison" in report_text
