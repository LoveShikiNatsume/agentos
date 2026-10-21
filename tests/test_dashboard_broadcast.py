import asyncio
from pathlib import Path
from agentos.events.types import RuntimeEvent, EventCategory


async def test_broadcaster_delivers_event_to_subscriber():
    from agentos.dashboard.broadcast import EventBroadcaster

    broadcaster = EventBroadcaster()
    queue = broadcaster.subscribe()
    event = RuntimeEvent(
        event_type="agent.created",
        agent_id="agent-1",
        category=EventCategory.AGENT,
        data={"role": "planner"},
    )
    broadcaster.publish(event)
    received = queue.get_nowait()
    assert received.event_type == "agent.created"
    assert received.agent_id == "agent-1"


async def test_broadcaster_delivers_to_multiple_subscribers():
    from agentos.dashboard.broadcast import EventBroadcaster

    broadcaster = EventBroadcaster()
    q1 = broadcaster.subscribe()
    q2 = broadcaster.subscribe()
    event = RuntimeEvent(event_type="task.completed", category=EventCategory.TASK)
    broadcaster.publish(event)
    assert q1.get_nowait().event_type == "task.completed"
    assert q2.get_nowait().event_type == "task.completed"


async def test_unsubscribe_removes_queue():
    from agentos.dashboard.broadcast import EventBroadcaster

    broadcaster = EventBroadcaster()
    queue = broadcaster.subscribe()
    broadcaster.unsubscribe(queue)
    event = RuntimeEvent(event_type="llm.request_sent", category=EventCategory.LLM)
    broadcaster.publish(event)
    assert queue.empty()


async def test_broadcaster_snapshot_returns_history():
    from agentos.dashboard.broadcast import EventBroadcaster

    broadcaster = EventBroadcaster()
    for i in range(3):
        broadcaster.publish(
            RuntimeEvent(event_type=f"event.{i}", category=EventCategory.SYSTEM)
        )
    snapshot = broadcaster.snapshot()
    assert len(snapshot) == 3
    assert snapshot[0].event_type == "event.0"


async def test_runtime_publishes_to_broadcaster(tmp_path: Path):
    from agentos.dashboard.broadcast import EventBroadcaster
    from agentos.llm.mock import MockLLMBackend
    from agentos.runtime.engine import AgentHandler, Runtime
    from agentos.core.task import Task, TaskDAG

    class EchoHandler(AgentHandler):
        async def run(self, ctx):
            resp = await ctx.llm_call([{"role": "user", "content": "hi"}])
            return resp.content

    broadcaster = EventBroadcaster()
    queue = broadcaster.subscribe()
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"worker": EchoHandler()},
        broadcaster=broadcaster,
    )
    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="t1", agent_role="worker"))
    await rt.run(dag)

    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    assert len(events) > 0
    types = {e.event_type for e in events}
    assert "agent.created" in types


async def test_llm_response_event_includes_token_breakdown(tmp_path: Path):
    from agentos.dashboard.broadcast import EventBroadcaster
    from agentos.benchmarks.cache_sim import CacheSimulatingMockBackend
    from agentos.runtime.engine import AgentHandler, Runtime
    from agentos.core.task import Task, TaskDAG

    class TokenHandler(AgentHandler):
        async def run(self, ctx):
            resp = await ctx.llm_call([{"role": "user", "content": "hello"}])
            return resp.content

    broadcaster = EventBroadcaster()
    queue = broadcaster.subscribe()
    backend = CacheSimulatingMockBackend(latency_base=0.01)
    rt = Runtime(
        llm_backend=backend,
        handlers={"worker": TokenHandler()},
        broadcaster=broadcaster,
    )
    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="t1", agent_role="worker"))
    await rt.run(dag)

    llm_events = []
    while not queue.empty():
        e = queue.get_nowait()
        if e.event_type == "llm.response_received":
            llm_events.append(e)

    assert len(llm_events) > 0
    evt = llm_events[0]
    assert "cache_hit_tokens" in evt.data or "prompt_tokens" in evt.data
