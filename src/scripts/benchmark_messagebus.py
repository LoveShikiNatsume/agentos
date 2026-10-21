"""MessageBus communication benchmark.

Demonstrates that ContextRef-based messaging reduces payload compared to
inline text, and validates priority ordering and history query capabilities.

Usage:
    .venv/bin/python scripts/benchmark_messagebus.py --messages 100 --payload-size 4096 --out runs/messagebus_benchmark/summary.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from agentos.comm.bus import MessageBus
from agentos.core.types import Message


async def run_inline_mode(
    bus: MessageBus,
    message_count: int,
    payload_size: int,
) -> dict[str, Any]:
    """Send messages with full inline context text."""
    large_text = "A" * payload_size
    for i in range(message_count):
        await bus.send(Message(
            from_agent=f"agent_{i % 5}",
            to_agent=f"agent_{(i + 1) % 5}",
            content={"full_context": large_text, "seq": i},
            message_type="inline",
            task_id=f"task-{i % 10}",
        ))
    stats = bus.stats(message_type="inline")
    return {
        "message_count": stats["message_count"],
        "payload_bytes": stats["payload_bytes"],
        "context_ref_count": stats["context_ref_count"],
    }


async def run_context_ref_mode(
    bus: MessageBus,
    message_count: int,
    payload_size: int,
) -> dict[str, Any]:
    """Send messages with ContextRef instead of inline text."""
    for i in range(message_count):
        await bus.send(Message(
            from_agent=f"agent_{i % 5}",
            to_agent=f"agent_{(i + 1) % 5}",
            content={"context_ref": f"ctx://shared-context-{i % 3}", "seq": i},
            message_type="context_ref",
            task_id=f"task-{i % 10}",
            context_refs=[f"ctx://shared-context-{i % 3}"],
        ))
    stats = bus.stats(message_type="context_ref")
    return {
        "message_count": stats["message_count"],
        "payload_bytes": stats["payload_bytes"],
        "context_ref_count": stats["context_ref_count"],
    }


async def run_priority_mode(bus: MessageBus) -> dict[str, Any]:
    """Verify high-priority messages are received first."""
    # Send low priority first, then high priority
    await bus.send(Message(
        from_agent="scheduler",
        to_agent="worker",
        content={"task": "low-priority-task"},
        message_type="task_assignment",
        priority=1,
    ))
    await bus.send(Message(
        from_agent="scheduler",
        to_agent="worker",
        content={"task": "high-priority-task"},
        message_type="task_assignment",
        priority=10,
    ))

    first = await bus.receive("worker")
    second = await bus.receive("worker")

    priority_order_correct = (
        first is not None
        and second is not None
        and first.content.get("task") == "high-priority-task"
        and second.content.get("task") == "low-priority-task"
    )

    return {
        "priority_order_correct": priority_order_correct,
        "first_received": first.content.get("task") if first else None,
        "second_received": second.content.get("task") if second else None,
    }


async def run_history_mode(bus: MessageBus) -> dict[str, Any]:
    """Verify history queries work by agent, task, and message type."""
    # Send a mix of messages
    for i in range(10):
        await bus.send(Message(
            from_agent="a1",
            to_agent="a2",
            content={"n": i},
            message_type="status" if i % 2 == 0 else "result",
            task_id=f"task-{i % 3}",
        ))

    # Query by agent
    by_agent = bus.get_history("a2")
    agent_query_ok = len(by_agent) == 10

    # Query by task
    by_task = bus.get_history("a2", task_id="task-0")
    task_query_ok = len(by_task) > 0 and all(
        m.task_id == "task-0" for m in by_task
    )

    # Query by type
    by_type = bus.get_history("a2", message_type="status")
    type_query_ok = len(by_type) > 0 and all(
        m.message_type == "status" for m in by_type
    )

    # Query by all three
    by_all = bus.get_history("a2", task_id="task-1", message_type="result")
    all_query_ok = all(
        m.task_id == "task-1" and m.message_type == "result"
        for m in by_all
    )

    history_query_correct = all([agent_query_ok, task_query_ok, type_query_ok, all_query_ok])

    return {
        "history_query_correct": history_query_correct,
        "agent_query_ok": agent_query_ok,
        "task_query_ok": task_query_ok,
        "type_query_ok": type_query_ok,
        "combined_query_ok": all_query_ok,
        "total_messages": len(by_agent),
    }


async def run_benchmark(
    message_count: int,
    payload_size: int,
    out_path: Path,
) -> dict[str, Any]:
    """Run the full MessageBus benchmark."""
    start = time.monotonic()

    # Mode 1: Inline
    bus_inline = MessageBus()
    inline_result = await run_inline_mode(bus_inline, message_count, payload_size)

    # Mode 2: ContextRef
    bus_ref = MessageBus()
    ref_result = await run_context_ref_mode(bus_ref, message_count, payload_size)

    # Mode 3: Priority
    bus_priority = MessageBus()
    priority_result = await run_priority_mode(bus_priority)

    # Mode 4: History
    bus_history = MessageBus()
    history_result = await run_history_mode(bus_history)

    elapsed = time.monotonic() - start

    # Compute savings
    inline_bytes = inline_result["payload_bytes"]
    ref_bytes = ref_result["payload_bytes"]
    bytes_saved = max(0, inline_bytes - ref_bytes)
    reduction_ratio = bytes_saved / inline_bytes if inline_bytes > 0 else 0.0

    summary: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message_count": message_count,
            "payload_size": payload_size,
            "benchmark_elapsed_secs": elapsed,
        },
        "inline": inline_result,
        "context_ref": ref_result,
        "priority": priority_result,
        "history": history_result,
        "aggregate": {
            "message_count": message_count * 2,  # inline + ref
            "context_ref_count": ref_result["context_ref_count"],
            "inline_payload_bytes": inline_bytes,
            "ref_payload_bytes": ref_bytes,
            "bytes_saved": bytes_saved,
            "payload_reduction_ratio": reduction_ratio,
            "avg_inline_payload_bytes": inline_bytes / message_count if message_count > 0 else 0,
            "avg_ref_payload_bytes": ref_bytes / message_count if message_count > 0 else 0,
            "priority_order_correct": priority_result["priority_order_correct"],
            "history_query_correct": history_result["history_query_correct"],
        },
    }

    # Write summary
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Written: {out_path}")

    # Write report
    report_path = out_path.parent / "report.md"
    report_lines = [
        "# MessageBus Communication Benchmark Report",
        "",
        "## Summary",
        "",
        f"- **Messages**: {message_count} per mode",
        f"- **Payload size**: {payload_size} bytes per inline message",
        f"- **Elapsed**: {elapsed:.3f}s",
        "",
        "## Payload Comparison",
        "",
        "| Mode | Messages | Payload Bytes | Avg Bytes/Msg |",
        "|------|----------|--------------|---------------|",
        f"| Inline | {inline_result['message_count']} | {inline_bytes} | {inline_bytes / message_count:.0f} |",
        f"| ContextRef | {ref_result['message_count']} | {ref_bytes} | {ref_bytes / message_count:.0f} |",
        "",
        f"**Bytes saved**: {bytes_saved}",
        f"**Payload reduction ratio**: {reduction_ratio:.2%}",
        "",
        "## Priority Ordering",
        "",
        f"- Priority order correct: {priority_result['priority_order_correct']}",
        f"- First received: {priority_result['first_received']}",
        f"- Second received: {priority_result['second_received']}",
        "",
        "## History Query",
        "",
        f"- History query correct: {history_result['history_query_correct']}",
        f"- Agent query OK: {history_result['agent_query_ok']}",
        f"- Task query OK: {history_result['task_query_ok']}",
        f"- Type query OK: {history_result['type_query_ok']}",
        f"- Combined query OK: {history_result['combined_query_ok']}",
        "",
        "## Interpretation",
        "",
        "Inline mode carries full context text in every message, resulting in large payloads.",
        "ContextRef mode carries only `ctx://...` references, dramatically reducing per-message payload.",
        "The Runtime's MessageBus supports priority ordering (high-priority messages received first)",
        "and multi-dimensional history queries (by agent, task, message_type).",
        "",
        "> Note: ContextRef savings depend on the shared context size. In real workloads,",
        "> system prompts and task context can be several KB, making ContextRef significantly",
        "> more efficient than inline transmission.",
        "",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  Written: {report_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="MessageBus communication benchmark")
    parser.add_argument("--messages", type=int, default=100, help="Number of messages per mode")
    parser.add_argument("--payload-size", type=int, default=4096, help="Inline payload size in bytes")
    parser.add_argument("--out", default="runs/messagebus_benchmark/summary.json", help="Output path")
    args = parser.parse_args()

    print("MessageBus Communication Benchmark")
    print(f"  Messages: {args.messages}")
    print(f"  Payload size: {args.payload_size}")
    print(f"  Output: {args.out}")
    print()

    summary = asyncio.run(run_benchmark(args.messages, args.payload_size, Path(args.out)))

    agg = summary["aggregate"]
    print()
    print("Results:")
    print(f"  Inline payload: {agg['inline_payload_bytes']} bytes")
    print(f"  ContextRef payload: {agg['ref_payload_bytes']} bytes")
    print(f"  Bytes saved: {agg['bytes_saved']}")
    print(f"  Reduction ratio: {agg['payload_reduction_ratio']:.2%}")
    print(f"  Priority order correct: {agg['priority_order_correct']}")
    print(f"  History query correct: {agg['history_query_correct']}")


if __name__ == "__main__":
    main()
