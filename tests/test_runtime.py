import asyncio

import pytest

from agentos.context.cache_tracker import CacheHeatTracker
from agentos.context.manager import ContextManager
from agentos.core.resources import ResourceQuota, ResourceUsage
from agentos.core.task import Task, TaskDAG, TaskStatus
from agentos.core.types import AgentState
from agentos.events.log import EventLog
from agentos.llm.gateway import ModelGateway
from agentos.llm.mock import MockLLMBackend
from agentos.metrics.model import ModelCallMetrics
from agentos.runtime.engine import AgentHandler, Runtime


class EchoHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        response = await ctx.llm_call(
            messages=[{"role": "user", "content": f"Task: {ctx.task.name}"}]
        )
        return response.content


class InspectRegistrationHandler(AgentHandler):
    def __init__(self, runtime_holder: dict[str, Runtime]) -> None:
        self._runtime_holder = runtime_holder

    async def run(self, ctx: AgentHandler.Context) -> str:
        runtime = self._runtime_holder["runtime"]
        assert runtime.agents[ctx.acb.agent_id] is ctx.acb
        assert ctx.task.assigned_agent_id == ctx.acb.agent_id
        assert ctx.acb.task_id == ctx.task.task_id
        assert ctx.acb.trace_id is not None
        assert any(
            event.event_type == "agent.created"
            and event.agent_id == ctx.acb.agent_id
            and event.task_id == ctx.task.task_id
            for event in runtime.events
        )
        return "registered"


class FailingHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        raise RuntimeError("boom")


class DynamicSubmitHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        ctx.submit_task(
            Task(
                task_id="child",
                name="child",
                agent_role="echo",
                dependencies=[ctx.task.task_id],
            )
        )
        return "parent done"


class DynamicFanoutHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        ctx.submit_task(
            Task(
                task_id="short",
                name="short",
                agent_role="echo",
                dependencies=[ctx.task.task_id],
            )
        )
        ctx.submit_task(
            Task(
                task_id="long",
                name="long",
                agent_role="echo",
                dependencies=[ctx.task.task_id],
            )
        )
        ctx.submit_task(
            Task(
                task_id="leaf",
                name="leaf",
                agent_role="echo",
                dependencies=["long"],
            )
        )
        return "fanout done"


class DuplicateDynamicSubmitHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        ctx.submit_task(
            Task(
                task_id="child",
                name="child",
                agent_role="echo",
                dependencies=[ctx.task.task_id],
            )
        )
        ctx.submit_task(
            Task(
                task_id="child",
                name="duplicate child",
                agent_role="echo",
                dependencies=[ctx.task.task_id],
            )
        )
        return "duplicate done"


class MissingDependencyDynamicSubmitHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        ctx.submit_task(
            Task(
                task_id="orphan-child",
                name="orphan-child",
                agent_role="echo",
                dependencies=["missing-parent"],
            )
        )
        return "missing dep done"


class DynamicBatchSubmitHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        ctx.submit_tasks([
            Task(
                task_id="child-a",
                name="child-a",
                agent_role="echo",
                dependencies=[ctx.task.task_id],
            ),
            Task(
                task_id="child-b",
                name="child-b",
                agent_role="echo",
                dependencies=[ctx.task.task_id],
            ),
        ])
        return "batch parent done"


class InvalidDynamicBatchSubmitHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        ctx.submit_tasks([
            Task(
                task_id="child-a",
                name="child-a",
                agent_role="echo",
                dependencies=[ctx.task.task_id],
            ),
            Task(
                task_id="child-b",
                name="child-b",
                agent_role="echo",
                dependencies=["missing-parent"],
            ),
        ])
        return "invalid batch done"


class CancelDownstreamHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        cancelled = ctx.cancel_downstream(reason="obsolete branch")
        return ",".join(cancelled)


class ReplaceDownstreamHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        cancelled, replacements = ctx.replace_downstream(
            None,
            [
                Task(
                    task_id="new-a",
                    name="new-a",
                    agent_role="echo",
                    dependencies=[ctx.task.task_id],
                ),
                Task(
                    task_id="new-b",
                    name="new-b",
                    agent_role="echo",
                    dependencies=["new-a"],
                ),
            ],
            reason="better recovery branch",
        )
        return f"{','.join(cancelled)}->{','.join(replacements)}"


class BatchEchoHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> list[str]:
        responses = await ctx.llm_batch_call(
            [
                [{"role": "user", "content": "first"}],
                [{"role": "user", "content": "second"}],
            ]
        )
        return [response.content for response in responses]


class BatchMockLLMBackend(MockLLMBackend):
    def __init__(self) -> None:
        super().__init__(responses={"default": "batch ok"}, latency=0)
        self.batch_count = 0
        self.batch_sizes: list[int] = []

    async def generate_batch(self, requests):
        self.batch_count += 1
        self.batch_sizes.append(len(requests))
        return [await self.generate(request) for request in requests]


class FailingBatchLLMBackend(MockLLMBackend):
    async def generate_batch(self, requests):
        raise RuntimeError("batch backend unavailable")


async def test_single_task():
    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="greet", agent_role="echo"))

    llm = MockLLMBackend(responses={"default": "hello world"})
    rt = Runtime(llm_backend=llm, handlers={"echo": EchoHandler()})

    results = await rt.run(dag)
    assert results["t1"] == "hello world"


async def test_sequential_dag():
    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="plan", agent_role="echo"))
    dag.add_task(
        Task(task_id="t2", name="execute", agent_role="echo", dependencies=["t1"])
    )

    llm = MockLLMBackend(responses={"default": "done"})
    rt = Runtime(llm_backend=llm, handlers={"echo": EchoHandler()})

    results = await rt.run(dag)
    assert results["t1"] == "done"
    assert results["t2"] == "done"
    assert llm.call_count == 2


async def test_parallel_dag():
    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="root", agent_role="echo"))
    dag.add_task(
        Task(task_id="t2", name="branch-a", agent_role="echo", dependencies=["t1"])
    )
    dag.add_task(
        Task(task_id="t3", name="branch-b", agent_role="echo", dependencies=["t1"])
    )

    llm = MockLLMBackend(responses={"default": "ok"})
    rt = Runtime(llm_backend=llm, handlers={"echo": EchoHandler()})

    results = await rt.run(dag)
    assert len(results) == 3
    assert llm.call_count == 3


async def test_events_are_emitted():
    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="work", agent_role="echo"))

    llm = MockLLMBackend(responses={"default": "ok"})
    rt = Runtime(llm_backend=llm, handlers={"echo": EchoHandler()})
    await rt.run(dag)

    event_types = [event.event_type for event in rt.events]
    assert "agent.state_changed" in event_types
    assert "task.completed" in event_types


async def test_llm_call_sends_resolved_prefix_id():
    ctx = ContextManager()
    ctx.add_shared_ref("repo", "repository summary")

    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="t1",
            name="work",
            agent_role="echo",
            context_refs=["repo"],
        )
    )

    llm = MockLLMBackend(responses={"default": "ok"})
    rt = Runtime(
        llm_backend=llm,
        handlers={"echo": EchoHandler()},
        context_manager=ctx,
    )

    await rt.run(dag)

    assert llm.requests[0].prefix_id == ctx.resolve_prefix_id(["repo"])


async def test_runtime_creates_agent_before_handler_runs():
    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="t1",
            name="inspect",
            agent_role="inspect",
            context_refs=["ctx://repo-summary"],
            resource_requirements=ResourceQuota(memory_max_mb=256),
        )
    )

    holder: dict[str, Runtime] = {}
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"inspect": InspectRegistrationHandler(holder)},
    )
    holder["runtime"] = rt

    await rt.run(dag)

    acb = rt.agents["agent-t1"]
    assert acb.context_refs == ["ctx://repo-summary"]
    assert acb.task_id == "t1"
    assert acb.role == "inspect"
    assert acb.name == "inspect"
    assert acb.resource_quota.memory_max_mb == 256
    assert acb.trace_id == "trace-agent-t1"
    assert acb.metadata == {}


async def test_agent_created_event_contains_registration_payload():
    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="t1",
            name="inspect",
            agent_role="inspect",
            context_refs=["ctx://repo-summary", "ctx://task-bg"],
            resource_requirements=ResourceQuota(memory_max_mb=512),
        )
    )

    holder: dict[str, Runtime] = {}
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"inspect": InspectRegistrationHandler(holder)},
    )
    holder["runtime"] = rt

    await rt.run(dag)

    created_events = [
        event for event in rt.events if event.event_type == "agent.created"
    ]
    assert len(created_events) == 1
    event = created_events[0]
    assert event.agent_id == "agent-t1"
    assert event.task_id == "t1"
    assert event.trace_id == "trace-agent-t1"
    assert event.data["name"] == "inspect"
    assert event.data["role"] == "inspect"
    assert event.data["context_refs"] == ["ctx://repo-summary", "ctx://task-bg"]
    assert event.data["resource_quota"]["memory_max_mb"] == 512
    assert "source" not in event.data


async def test_runtime_copies_task_source_into_acb_and_event_log(tmp_path):
    source = {
        "framework": "langgraph",
        "node": "inspect",
        "graph": "review",
        "details": {"tags": ["static"]},
    }
    task = Task(
        task_id="t1", name="inspect", agent_role="inspect",
        metadata={"source": source, "application_only": object()},
    )
    dag = TaskDAG()
    dag.add_task(task)
    log = EventLog(tmp_path / "events.jsonl")
    holder: dict[str, Runtime] = {}
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"inspect": InspectRegistrationHandler(holder)},
        event_log=log,
        enable_kernel_probe=False,
    )
    holder["runtime"] = rt

    await rt.run(dag)

    acb = rt.agents["agent-t1"]
    event = next(e for e in rt.events if e.event_type == "agent.created")
    restored = next(log.read(event_type="agent.created"))
    assert acb.metadata == {"source": source}
    assert event.data["source"] == source
    assert restored.data["source"] == source
    assert restored.agent_id == acb.agent_id
    assert restored.task_id == task.task_id
    assert restored.trace_id == acb.trace_id
    assert "application_only" not in event.data

    source["node"] = "changed-task"
    source["details"]["tags"].append("task")
    assert acb.metadata["source"]["node"] == "inspect"
    assert acb.metadata["source"]["details"]["tags"] == ["static"]
    assert event.data["source"]["node"] == "inspect"
    assert event.data["source"]["details"]["tags"] == ["static"]

    acb.metadata["source"]["node"] = "changed-acb"
    acb.metadata["source"]["details"]["tags"].append("acb")
    assert event.data["source"]["node"] == "inspect"
    assert event.data["source"]["details"]["tags"] == ["static"]
    assert restored.data["source"]["node"] == "inspect"
    assert restored.data["source"]["details"]["tags"] == ["static"]


async def test_runtime_rejects_duplicate_agent_creation_for_same_task():
    task = Task(task_id="t1", name="inspect", agent_role="inspect")
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"inspect": InspectRegistrationHandler({})},
    )

    acb = rt._create_agent_for_task(task)

    with pytest.raises(RuntimeError, match="already exists"):
        rt._create_agent_for_task(task)

    assert rt.agents["agent-t1"] is acb
    assert task.assigned_agent_id == "agent-t1"


async def test_runtime_rejects_task_preassigned_to_unexpected_agent():
    task = Task(task_id="t1", name="inspect", agent_role="inspect")
    task.assigned_agent_id = "external-agent"
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"inspect": InspectRegistrationHandler({})},
    )

    with pytest.raises(RuntimeError, match="already assigned"):
        rt._create_agent_for_task(task)

    assert rt.agents == {}


async def test_runtime_emits_complete_lifecycle_trace_with_task_and_trace_ids():
    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="work", agent_role="echo"))

    rt = Runtime(
        llm_backend=MockLLMBackend(responses={"default": "ok"}),
        handlers={"echo": EchoHandler()},
    )
    await rt.run(dag)

    state_events = [
        event for event in rt.events if event.event_type == "agent.state_changed"
    ]
    state_pairs = [
        (event.data["old_state"], event.data["new_state"]) for event in state_events
    ]

    assert state_pairs == [
        (AgentState.CREATED.value, AgentState.READY.value),
        (AgentState.READY.value, AgentState.RUNNING.value),
        (AgentState.RUNNING.value, AgentState.BLOCKED_LLM.value),
        (AgentState.BLOCKED_LLM.value, AgentState.RUNNING.value),
        (AgentState.RUNNING.value, AgentState.FINISHED.value),
    ]
    assert all(event.task_id == "t1" for event in state_events)
    assert all(event.trace_id == "trace-agent-t1" for event in state_events)
    assert rt.agents["agent-t1"].state == AgentState.FINISHED


async def test_runtime_lifecycle_trace_links_llm_and_task_events():
    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="work", agent_role="echo"))

    rt = Runtime(
        llm_backend=MockLLMBackend(responses={"default": "ok"}),
        handlers={"echo": EchoHandler()},
    )
    await rt.run(dag)

    linked_events = [
        event
        for event in rt.events
        if event.event_type
        in {
            "agent.created",
            "agent.state_changed",
            "llm.request_sent",
            "llm.response_received",
            "task.completed",
        }
    ]

    assert linked_events
    assert all(event.agent_id == "agent-t1" for event in linked_events)
    assert all(event.task_id == "t1" for event in linked_events)
    assert all(event.trace_id == "trace-agent-t1" for event in linked_events)


async def test_handler_exception_marks_agent_and_task_failed():
    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="fail", agent_role="failing"))

    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"failing": FailingHandler()},
    )

    with pytest.raises(RuntimeError, match="boom"):
        await rt.run(dag)

    assert rt.agents["agent-t1"].state == AgentState.FAILED
    assert dag.tasks["t1"].status.value == "failed"
    state_events = [
        event for event in rt.events if event.event_type == "agent.state_changed"
    ]
    state_pairs = [
        (event.data["old_state"], event.data["new_state"]) for event in state_events
    ]
    assert state_pairs == [
        (AgentState.CREATED.value, AgentState.READY.value),
        (AgentState.READY.value, AgentState.RUNNING.value),
        (AgentState.RUNNING.value, AgentState.FAILED.value),
    ]
    assert all(event.agent_id == "agent-t1" for event in state_events)
    assert all(event.task_id == "t1" for event in state_events)
    assert all(event.trace_id == "trace-agent-t1" for event in state_events)

    failed_events = [
        event for event in rt.events if event.event_type == "task.failed"
    ]
    assert len(failed_events) == 1
    assert failed_events[0].agent_id == "agent-t1"
    assert failed_events[0].task_id == "t1"
    assert failed_events[0].trace_id == "trace-agent-t1"


async def test_agent_can_submit_dynamic_child_task():
    dag = TaskDAG()
    dag.add_task(Task(task_id="parent", name="parent", agent_role="dynamic"))

    rt = Runtime(
        llm_backend=MockLLMBackend(responses={"default": "child done"}),
        handlers={
            "dynamic": DynamicSubmitHandler(),
            "echo": EchoHandler(),
        },
    )

    results = await rt.run(dag)

    assert results["parent"] == "parent done"
    assert results["child"] == "child done"
    assert dag.tasks["child"].dependencies == ["parent"]
    assert dag.tasks["child"].status == TaskStatus.COMPLETED
    assert rt.agents["agent-child"].task_id == "child"
    assert rt.agents["agent-child"].trace_id == "trace-agent-child"

    child_events = [
        event
        for event in rt.events
        if event.agent_id == "agent-child"
        and event.event_type
        in {"agent.created", "agent.state_changed", "task.completed"}
    ]
    assert {event.event_type for event in child_events} == {
        "agent.created",
        "agent.state_changed",
        "task.completed",
    }
    assert all(event.task_id == "child" for event in child_events)
    assert all(event.trace_id == "trace-agent-child" for event in child_events)


async def test_dynamic_task_submission_emits_trace_event():
    dag = TaskDAG()
    dag.add_task(Task(task_id="parent", name="parent", agent_role="dynamic"))

    rt = Runtime(
        llm_backend=MockLLMBackend(responses={"default": "child done"}),
        handlers={"dynamic": DynamicSubmitHandler(), "echo": EchoHandler()},
    )

    await rt.run(dag)

    dynamic_events = [
        event for event in rt.events if event.event_type == "task.dynamic_submitted"
    ]
    assert len(dynamic_events) == 1
    event = dynamic_events[0]
    assert event.agent_id == "agent-parent"
    assert event.task_id == "child"
    assert event.trace_id == "trace-agent-parent"
    assert event.data == {
        "name": "child",
        "agent_role": "echo",
        "dependencies": ["parent"],
        "parent_task_id": "parent",
        "parent_agent_id": "agent-parent",
        "context_refs": [],
        "resource_requirements": ResourceQuota().to_dict(),
    }


async def test_dynamic_task_submission_rejects_duplicate_task_id():
    dag = TaskDAG()
    dag.add_task(Task(task_id="parent", name="parent", agent_role="duplicate"))

    rt = Runtime(
        llm_backend=MockLLMBackend(responses={"default": "ok"}),
        handlers={"duplicate": DuplicateDynamicSubmitHandler(), "echo": EchoHandler()},
    )

    with pytest.raises(ValueError, match="already exists"):
        await rt.run(dag)

    assert dag.tasks["parent"].status == TaskStatus.FAILED
    assert "child" in dag.tasks


async def test_dynamic_task_submission_rejects_missing_dependency():
    dag = TaskDAG()
    dag.add_task(Task(task_id="parent", name="parent", agent_role="missing"))

    rt = Runtime(
        llm_backend=MockLLMBackend(responses={"default": "ok"}),
        handlers={"missing": MissingDependencyDynamicSubmitHandler()},
    )

    with pytest.raises(ValueError, match="Dependency missing-parent not found"):
        await rt.run(dag)

    assert dag.tasks["parent"].status == TaskStatus.FAILED
    assert "orphan-child" not in dag.tasks


async def test_dynamic_task_submission_requires_active_dag():
    rt = Runtime(llm_backend=MockLLMBackend(), handlers={})
    acb = rt.agents.setdefault(
        "agent-parent",
        rt._create_agent_for_task(Task(task_id="parent", name="parent", agent_role="echo")),
    )

    with pytest.raises(RuntimeError, match="No active DAG"):
        rt._submit_dynamic_task(
            Task(task_id="orphan", name="orphan", agent_role="echo"),
            acb,
        )


async def test_agent_can_submit_dynamic_task_batch():
    dag = TaskDAG()
    dag.add_task(Task(task_id="parent", name="parent", agent_role="dynamic-batch"))

    rt = Runtime(
        llm_backend=MockLLMBackend(responses={"default": "child done"}),
        handlers={
            "dynamic-batch": DynamicBatchSubmitHandler(),
            "echo": EchoHandler(),
        },
    )

    results = await rt.run(dag)

    assert results["parent"] == "batch parent done"
    assert results["child-a"] == "child done"
    assert results["child-b"] == "child done"
    assert dag.tasks["child-a"].status == TaskStatus.COMPLETED
    assert dag.tasks["child-b"].status == TaskStatus.COMPLETED


async def test_dynamic_task_batch_submission_emits_trace_event():
    dag = TaskDAG()
    dag.add_task(Task(task_id="parent", name="parent", agent_role="dynamic-batch"))

    rt = Runtime(
        llm_backend=MockLLMBackend(responses={"default": "child done"}),
        handlers={
            "dynamic-batch": DynamicBatchSubmitHandler(),
            "echo": EchoHandler(),
        },
    )

    await rt.run(dag)

    events = [
        event for event in rt.events if event.event_type == "task.dynamic_batch_submitted"
    ]
    assert len(events) == 1
    event = events[0]
    assert event.agent_id == "agent-parent"
    assert event.task_id == "parent"
    assert event.trace_id == "trace-agent-parent"
    assert event.data["parent_task_id"] == "parent"
    assert event.data["parent_agent_id"] == "agent-parent"
    assert event.data["batch_size"] == 2
    assert event.data["task_ids"] == ["child-a", "child-b"]
    assert [task["task_id"] for task in event.data["tasks"]] == [
        "child-a",
        "child-b",
    ]


async def test_dynamic_task_batch_submission_rejects_invalid_batch_atomically():
    dag = TaskDAG()
    dag.add_task(Task(task_id="parent", name="parent", agent_role="invalid-batch"))

    rt = Runtime(
        llm_backend=MockLLMBackend(responses={"default": "ok"}),
        handlers={
            "invalid-batch": InvalidDynamicBatchSubmitHandler(),
            "echo": EchoHandler(),
        },
    )

    with pytest.raises(ValueError, match="Dependency missing-parent not found"):
        await rt.run(dag)

    assert dag.tasks["parent"].status == TaskStatus.FAILED
    assert "child-a" not in dag.tasks
    assert "child-b" not in dag.tasks
    assert not any(
        event.event_type == "task.dynamic_batch_submitted" for event in rt.events
    )


async def test_dynamic_tasks_recompute_critical_path_before_scheduling():
    dag = TaskDAG()
    dag.add_task(Task(task_id="parent", name="parent", agent_role="fanout"))

    rt = Runtime(
        llm_backend=MockLLMBackend(responses={"default": "ok"}),
        handlers={
            "fanout": DynamicFanoutHandler(),
            "echo": EchoHandler(),
        },
        max_batch_size=1,
    )

    results = await rt.run(dag)

    assert results["parent"] == "fanout done"
    assert results["long"] == "ok"
    assert results["short"] == "ok"
    assert results["leaf"] == "ok"
    assert dag.tasks["long"].critical_path_score > dag.tasks["short"].critical_path_score

    created_order = [
        event.task_id
        for event in rt.events
        if event.event_type == "agent.created"
    ]
    assert created_order[0] == "parent"
    assert created_order.index("long") < created_order.index("short")
    assert set(created_order) == {"parent", "long", "short", "leaf"}


async def test_agent_can_cancel_pending_downstream_branch():
    dag = TaskDAG()
    dag.add_task(Task(task_id="parent", name="parent", agent_role="cancel"))
    dag.add_task(
        Task(
            task_id="old-a",
            name="old-a",
            agent_role="echo",
            dependencies=["parent"],
        )
    )
    dag.add_task(
        Task(
            task_id="old-b",
            name="old-b",
            agent_role="echo",
            dependencies=["old-a"],
        )
    )

    rt = Runtime(
        llm_backend=MockLLMBackend(responses={"default": "old done"}),
        handlers={"cancel": CancelDownstreamHandler(), "echo": EchoHandler()},
    )

    results = await rt.run(dag)

    assert results["parent"] == "old-a,old-b"
    assert "old-a" not in results
    assert "old-b" not in results
    assert dag.tasks["old-a"].status == TaskStatus.CANCELLED
    assert dag.tasks["old-b"].status == TaskStatus.CANCELLED
    cancelled_events = [
        event for event in rt.events if event.event_type == "task.cancelled"
    ]
    assert [event.task_id for event in cancelled_events] == ["old-a", "old-b"]
    assert all(event.agent_id == "agent-parent" for event in cancelled_events)
    assert all(
        event.data["reason"] == "obsolete branch" for event in cancelled_events
    )


async def test_agent_can_replace_pending_downstream_branch():
    dag = TaskDAG()
    dag.add_task(Task(task_id="parent", name="parent", agent_role="replace"))
    dag.add_task(
        Task(
            task_id="old-a",
            name="old-a",
            agent_role="echo",
            dependencies=["parent"],
        )
    )
    dag.add_task(
        Task(
            task_id="old-b",
            name="old-b",
            agent_role="echo",
            dependencies=["old-a"],
        )
    )

    rt = Runtime(
        llm_backend=MockLLMBackend(responses={"default": "new done"}),
        handlers={"replace": ReplaceDownstreamHandler(), "echo": EchoHandler()},
    )

    results = await rt.run(dag)

    assert results["parent"] == "old-a,old-b->new-a,new-b"
    assert results["new-a"] == "new done"
    assert results["new-b"] == "new done"
    assert "old-a" not in results
    assert "old-b" not in results
    assert dag.tasks["old-a"].status == TaskStatus.CANCELLED
    assert dag.tasks["old-b"].status == TaskStatus.CANCELLED
    assert dag.tasks["new-a"].status == TaskStatus.COMPLETED
    assert dag.tasks["new-b"].status == TaskStatus.COMPLETED

    replacement_events = [
        event
        for event in rt.events
        if event.event_type == "task.dynamic_branch_replaced"
    ]
    assert len(replacement_events) == 1
    event = replacement_events[0]
    assert event.agent_id == "agent-parent"
    assert event.task_id == "parent"
    assert event.data["cancelled_task_ids"] == ["old-a", "old-b"]
    assert event.data["replacement_task_ids"] == ["new-a", "new-b"]
    assert event.data["reason"] == "better recovery branch"


async def test_context_defines_batch_llm_call_interface():
    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="batch", agent_role="batch"))

    llm = BatchMockLLMBackend()
    rt = Runtime(llm_backend=llm, handlers={"batch": BatchEchoHandler()})

    results = await rt.run(dag)

    assert results["t1"] == ["batch ok", "batch ok"]
    assert llm.batch_count == 1


async def test_batch_llm_call_sends_resolved_prefix_id():
    ctx = ContextManager()
    ctx.add_shared_ref("repo", "repository summary")

    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="t1",
            name="batch",
            agent_role="batch",
            context_refs=["repo"],
        )
    )

    llm = BatchMockLLMBackend()
    rt = Runtime(
        llm_backend=llm,
        handlers={"batch": BatchEchoHandler()},
        context_manager=ctx,
    )

    await rt.run(dag)

    expected_prefix_id = ctx.resolve_prefix_id(["repo"])
    assert [request.prefix_id for request in llm.requests] == [
        expected_prefix_id,
        expected_prefix_id,
    ]


async def test_batch_llm_call_emits_batch_events():
    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="batch", agent_role="batch"))

    llm = BatchMockLLMBackend()
    rt = Runtime(llm_backend=llm, handlers={"batch": BatchEchoHandler()})

    await rt.run(dag)

    batch_started = [
        event for event in rt.events if event.event_type == "llm.batch_started"
    ]
    batch_completed = [
        event for event in rt.events if event.event_type == "llm.batch_completed"
    ]
    assert len(batch_started) == 1
    assert len(batch_completed) == 1
    assert batch_started[0].agent_id == "agent-t1"
    assert batch_started[0].task_id == "t1"
    assert batch_started[0].trace_id == "trace-agent-t1"
    assert batch_started[0].data["batch_size"] == 2
    assert batch_started[0].data["backend"] == "default"
    assert batch_started[0].data["prefix_group_count"] == 1
    assert batch_started[0].data["prefix_groups"] == [
        {"prefix_id": "__no_prefix__", "size": 2}
    ]
    assert batch_completed[0].data["batch_size"] == 2
    assert batch_completed[0].data["response_count"] == 2
    assert batch_completed[0].data["tokens_used"] > 0
    assert batch_completed[0].data["prefix_group_count"] == 1
    assert batch_completed[0].data["prefix_groups"] == [
        {"prefix_id": "__no_prefix__", "size": 2}
    ]


async def test_batch_llm_call_emits_failure_event():
    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="batch", agent_role="batch"))

    rt = Runtime(
        llm_backend=FailingBatchLLMBackend(),
        handlers={"batch": BatchEchoHandler()},
    )

    with pytest.raises(RuntimeError, match="batch backend unavailable"):
        await rt.run(dag)

    failed = [event for event in rt.events if event.event_type == "llm.batch_failed"]
    assert len(failed) == 1
    assert failed[0].agent_id == "agent-t1"
    assert failed[0].task_id == "t1"
    assert failed[0].trace_id == "trace-agent-t1"
    assert failed[0].data["batch_size"] == 2
    assert failed[0].data["backend"] == "default"
    assert failed[0].data["error"] == "batch backend unavailable"
    assert failed[0].data["prefix_group_count"] == 1
    assert failed[0].data["prefix_groups"] == [
        {"prefix_id": "__no_prefix__", "size": 2}
    ]
    assert rt.agents["agent-t1"].state == AgentState.FAILED


async def test_global_llm_pool_batches_parallel_llm_calls_by_prefix():
    dag = TaskDAG()
    dag.add_task(Task(task_id="a", name="a", agent_role="echo"))
    dag.add_task(Task(task_id="b", name="b", agent_role="echo"))

    llm = BatchMockLLMBackend()
    rt = Runtime(
        llm_backend=llm,
        handlers={"echo": EchoHandler()},
        max_llm_batch_size=2,
        llm_batch_wait_ms=50,
    )

    results = await rt.run(dag)

    assert results == {"a": "batch ok", "b": "batch ok"}
    assert llm.batch_count == 1
    assert llm.batch_sizes == [2]
    batch_started = [
        event
        for event in rt.events
        if event.event_type == "llm.batch_started"
        and event.data.get("source") == "global_pool"
    ]
    assert len(batch_started) == 1
    assert batch_started[0].data["flush_reason"] == "max_batch_size"
    assert batch_started[0].data["batch_size"] == 2
    assert batch_started[0].data["prefix_group_count"] == 1


async def test_global_llm_pool_flushes_after_wait_window():
    dag = TaskDAG()
    dag.add_task(Task(task_id="solo", name="solo", agent_role="echo"))

    llm = BatchMockLLMBackend()
    rt = Runtime(
        llm_backend=llm,
        handlers={"echo": EchoHandler()},
        max_llm_batch_size=8,
        llm_batch_wait_ms=1,
    )

    results = await rt.run(dag)

    assert results["solo"] == "batch ok"
    assert llm.batch_count == 1
    assert llm.batch_sizes == [1]
    batch_started = [
        event
        for event in rt.events
        if event.event_type == "llm.batch_started"
        and event.data.get("source") == "global_pool"
    ]
    assert batch_started[0].data["flush_reason"] == "wait_window"
    assert batch_started[0].data["batch_wait_ms"] == 1


async def test_global_llm_pool_splits_batches_by_max_llm_batch_size():
    dag = TaskDAG()
    for task_id in ("a", "b", "c"):
        dag.add_task(Task(task_id=task_id, name=task_id, agent_role="echo"))

    llm = BatchMockLLMBackend()
    rt = Runtime(
        llm_backend=llm,
        handlers={"echo": EchoHandler()},
        max_llm_batch_size=2,
        llm_batch_wait_ms=1,
    )

    results = await rt.run(dag)

    assert results == {"a": "batch ok", "b": "batch ok", "c": "batch ok"}
    assert llm.batch_count == 2
    assert llm.batch_sizes == [2, 1]
    completed = [
        event
        for event in rt.events
        if event.event_type == "llm.batch_completed"
        and event.data.get("source") == "global_pool"
    ]
    assert [event.data["batch_size"] for event in completed] == [2, 1]


def test_model_gateway_exposes_cache_tracker_for_runtime_integration():
    tracker = CacheHeatTracker(ttl=300.0)
    gateway = ModelGateway(
        backends={"default": MockLLMBackend(responses={"default": "ok"})},
        cache_tracker=tracker,
    )

    assert gateway.cache_tracker is tracker


def test_runtime_records_resource_usage_history_by_role():
    rt = Runtime(llm_backend=MockLLMBackend(), handlers={})
    task = Task(task_id="lint-1", name="lint", agent_role="lint")

    rt._record_resource_usage(
        task,
        ResourceUsage(
            cpu_percent=12.5,
            memory_peak_mb=83.4,
            pids_current=3,
        ),
    )
    rt._record_resource_usage(
        task,
        ResourceUsage(
            cpu_percent=9.0,
            memory_peak_mb=70.0,
            pids_current=2,
        ),
    )

    history = rt.resource_usage_by_role["lint"]
    estimate = rt._estimate_task_resources(task)

    assert history.samples == 2
    assert history.cpu_percent == 12.5
    assert history.memory_peak_mb == 83.4
    assert history.pids_current == 3
    assert estimate == ResourceQuota(
        cpu_max_percent=12.5,
        memory_max_mb=84,
        pids_max=3,
    )


def test_runtime_resource_estimator_returns_none_without_history():
    rt = Runtime(llm_backend=MockLLMBackend(), handlers={})
    task = Task(task_id="unknown-1", name="unknown", agent_role="unknown")

    assert rt._estimate_task_resources(task) is None


def test_runtime_records_sampled_worker_usage_by_role():
    class FakeCgroupManager:
        def read_usage(self, path):
            return ResourceUsage(memory_peak_mb=42.2, cpu_percent=7.0, pids_current=2)

    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={},
        cgroup_manager=FakeCgroupManager(),
    )
    task = Task(task_id="build-1", name="build", agent_role="build")
    acb = rt._create_agent_for_task(task)

    rt._sample_worker_usage(acb, task, "/sys/fs/cgroup/build-1")

    estimate = rt._estimate_task_resources(task)
    assert estimate == ResourceQuota(
        cpu_max_percent=7.0,
        memory_max_mb=43,
        pids_max=2,
    )


async def test_runtime_emits_resource_reservation_lifecycle_events():
    class TaskIdHandler(AgentHandler):
        async def run(self, ctx: AgentHandler.Context) -> str:
            return ctx.task.task_id

    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="build",
            name="build",
            agent_role="build",
            resource_requirements=ResourceQuota(memory_max_mb=128),
        )
    )
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"build": TaskIdHandler()},
        available_resources=ResourceQuota(memory_max_mb=256),
    )

    await rt.run(dag)

    reserved = [event for event in rt.events if event.event_type == "resource.reserved"]
    released = [event for event in rt.events if event.event_type == "resource.released"]

    assert len(reserved) == 1
    assert len(released) == 1
    assert reserved[0].category.value == "resource"
    assert reserved[0].task_id == "build"
    assert reserved[0].data["quota"]["memory_max_mb"] == 128
    assert "build" in reserved[0].data["active_reservations"]
    assert released[0].task_id == "build"
    assert released[0].data["quota"]["memory_max_mb"] == 128
    assert released[0].data["active_reservations"] == {}
    assert rt.active_resource_reservations == {}


async def test_runtime_releases_resource_reservation_after_task_failure():
    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="bad",
            name="bad",
            agent_role="bad",
            resource_requirements=ResourceQuota(memory_max_mb=64),
        )
    )
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"bad": FailingHandler()},
        available_resources=ResourceQuota(memory_max_mb=128),
        recovery_policy=None,
    )

    with pytest.raises(RuntimeError, match="boom"):
        await rt.run(dag)

    released = [event for event in rt.events if event.event_type == "resource.released"]
    assert len(released) == 1
    assert released[0].task_id == "bad"
    assert released[0].data["active_reservations"] == {}
    assert rt.active_resource_reservations == {}


def test_runtime_scheduler_available_resources_excludes_active_reservations():
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={},
        available_resources=ResourceQuota(
            cpu_max_percent=100.0,
            memory_max_mb=512,
            pids_max=32,
        ),
    )
    task = Task(
        task_id="active",
        name="active",
        agent_role="worker",
        resource_requirements=ResourceQuota(
            cpu_max_percent=25.0,
            memory_max_mb=128,
            pids_max=4,
        ),
    )

    rt._reserve_task_resources(task)
    available = rt._scheduler_available_resources()

    assert available == ResourceQuota(
        cpu_max_percent=75.0,
        memory_max_mb=384,
        pids_max=28,
    )

    rt._release_task_resources(task)
    assert rt.active_resource_reservations == {}


async def test_runtime_scheduler_uses_resource_feedback_for_next_batch():
    class ConcurrentHandler(AgentHandler):
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0

        async def run(self, ctx: AgentHandler.Context) -> str:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.02)
            self.active -= 1
            return ctx.task.task_id

    handler = ConcurrentHandler()
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"lint": handler},
        available_resources=ResourceQuota(memory_max_mb=512),
    )
    seed = Task(
        task_id="seed",
        name="seed",
        agent_role="lint",
        resource_requirements=ResourceQuota(memory_max_mb=300),
    )
    rt._record_resource_usage(seed, ResourceUsage(memory_peak_mb=80.0))

    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="lint-a",
            name="lint-a",
            agent_role="lint",
            resource_requirements=ResourceQuota(memory_max_mb=300),
        )
    )
    dag.add_task(
        Task(
            task_id="lint-b",
            name="lint-b",
            agent_role="lint",
            resource_requirements=ResourceQuota(memory_max_mb=300),
        )
    )

    await rt.run(dag)

    assert handler.max_active == 2


async def test_runtime_emits_scheduler_decision_event():
    class TaskIdHandler(AgentHandler):
        async def run(self, ctx: AgentHandler.Context) -> str:
            return ctx.task.task_id

    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="lint-a",
            name="lint-a",
            agent_role="lint",
            resource_requirements=ResourceQuota(memory_max_mb=300),
        )
    )
    dag.add_task(
        Task(
            task_id="lint-b",
            name="lint-b",
            agent_role="lint",
            resource_requirements=ResourceQuota(memory_max_mb=300),
        )
    )

    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"lint": TaskIdHandler()},
        available_resources=ResourceQuota(memory_max_mb=512),
    )

    await rt.run(dag)

    decisions = [
        event for event in rt.events if event.event_type == "scheduler.decision"
    ]
    assert decisions
    first_decision = decisions[0]
    assert first_decision.category.value == "scheduler"
    assert first_decision.data["strategy"] == "critical_path"
    assert first_decision.data["schema_version"] == "scheduler.decision.v2"
    assert first_decision.data["policy"] == "latency_first"
    assert first_decision.data["workload_class"] == "unknown"
    assert first_decision.data["policy_reason"] == "fallback critical path ordering"
    assert first_decision.data["selected_task_ids"] == ["lint-a"]
    assert first_decision.data["deferred_task_ids"] == ["lint-b"]
    assert first_decision.data["selected"] == ["lint-a"]
    assert first_decision.data["deferred"] == ["lint-b"]
    assert (
        first_decision.data["tasks"]["lint-b"]["reason"]
        == "resource_batch_full"
    )


async def test_runtime_prioritizes_cache_hot_context_refs_from_gateway_tracker():
    ctx = ContextManager()
    ctx.add_shared_ref("hot-ctx", "hot context content")
    ctx.add_shared_ref("cold-ctx", "cold context content")

    tracker = CacheHeatTracker(ttl=300.0)
    hot_prefix_id = ctx.resolve_prefix_id(["hot-ctx"])
    tracker.record_call(
        hot_prefix_id,
        ModelCallMetrics(
            backend="mock",
            model="mock",
            prompt_tokens=100,
            cache_hit_tokens=90,
            cache_miss_tokens=10,
        ),
    )

    gateway = ModelGateway(
        backends={"default": MockLLMBackend(responses={"default": "ok"})},
        cache_tracker=tracker,
    )
    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="cold-task",
            name="cold",
            agent_role="echo",
            context_refs=["cold-ctx"],
        )
    )
    dag.add_task(
        Task(
            task_id="hot-task",
            name="hot",
            agent_role="echo",
            context_refs=["hot-ctx"],
        )
    )
    rt = Runtime(
        llm_backend=gateway,
        handlers={"echo": EchoHandler()},
        context_manager=ctx,
    )

    await rt.run(dag)

    created_order = [
        event.task_id for event in rt.events if event.event_type == "agent.created"
    ]
    assert created_order[:2] == ["hot-task", "cold-task"]


async def test_runtime_accepts_propagation_depth():
    """Runtime should accept propagation_depth parameter."""
    llm = MockLLMBackend(responses={"default": "ok"})
    rt = Runtime(
        llm_backend=llm,
        handlers={"echo": EchoHandler()},
        propagation_depth=2,
    )
    assert rt._propagation_depth == 2


async def test_runtime_default_propagation_depth_is_none():
    """Default propagation_depth should be None (unlimited)."""
    llm = MockLLMBackend(responses={"default": "ok"})
    rt = Runtime(llm_backend=llm, handlers={"echo": EchoHandler()})
    assert rt._propagation_depth is None


async def test_propagate_skips_direct_downstream():
    """A fails -> B skipped."""
    dag = TaskDAG()
    dag.add_task(Task(task_id="a", name="a", agent_role="fail"))
    dag.add_task(Task(task_id="b", name="b", agent_role="echo", dependencies=["a"]))

    llm = MockLLMBackend(responses={"default": "ok"})
    rt = Runtime(
        llm_backend=llm,
        handlers={"fail": FailingHandler(), "echo": EchoHandler()},
        recovery_policy=None,
    )

    with pytest.raises(RuntimeError, match="boom"):
        await rt.run(dag)

    assert dag.tasks["a"].status == TaskStatus.FAILED
    assert dag.tasks["b"].status == TaskStatus.SKIPPED


async def test_propagate_skips_all_downstream_by_default():
    """A fails -> B, C skipped (default depth=None means unlimited)."""
    dag = TaskDAG()
    dag.add_task(Task(task_id="a", name="a", agent_role="fail"))
    dag.add_task(Task(task_id="b", name="b", agent_role="echo", dependencies=["a"]))
    dag.add_task(Task(task_id="c", name="c", agent_role="echo", dependencies=["b"]))

    llm = MockLLMBackend(responses={"default": "ok"})
    rt = Runtime(
        llm_backend=llm,
        handlers={"fail": FailingHandler(), "echo": EchoHandler()},
        recovery_policy=None,
    )

    with pytest.raises(RuntimeError, match="boom"):
        await rt.run(dag)

    assert dag.tasks["a"].status == TaskStatus.FAILED
    assert dag.tasks["b"].status == TaskStatus.SKIPPED
    assert dag.tasks["c"].status == TaskStatus.SKIPPED


async def test_propagate_respects_depth_limit():
    """depth=1: A fails -> B skipped, but C stays PENDING."""
    dag = TaskDAG()
    dag.add_task(Task(task_id="a", name="a", agent_role="fail"))
    dag.add_task(Task(task_id="b", name="b", agent_role="echo", dependencies=["a"]))
    dag.add_task(Task(task_id="c", name="c", agent_role="echo", dependencies=["b"]))

    llm = MockLLMBackend(responses={"default": "ok"})
    rt = Runtime(
        llm_backend=llm,
        handlers={"fail": FailingHandler(), "echo": EchoHandler()},
        recovery_policy=None,
        propagation_depth=1,
    )

    with pytest.raises(RuntimeError, match="boom"):
        await rt.run(dag)

    assert dag.tasks["a"].status == TaskStatus.FAILED
    assert dag.tasks["b"].status == TaskStatus.SKIPPED
    assert dag.tasks["c"].status == TaskStatus.PENDING


async def test_propagate_emits_events():
    """Each skipped task should emit fault.propagated event."""
    dag = TaskDAG()
    dag.add_task(Task(task_id="a", name="a", agent_role="fail"))
    dag.add_task(Task(task_id="b", name="b", agent_role="echo", dependencies=["a"]))

    llm = MockLLMBackend(responses={"default": "ok"})
    rt = Runtime(
        llm_backend=llm,
        handlers={"fail": FailingHandler(), "echo": EchoHandler()},
        recovery_policy=None,
    )

    with pytest.raises(RuntimeError, match="boom"):
        await rt.run(dag)

    propagated_events = [
        e for e in rt.events if e.event_type == "fault.propagated"
    ]
    assert len(propagated_events) == 1
    assert propagated_events[0].task_id == "b"
    assert propagated_events[0].data["source_task_id"] == "a"
    assert propagated_events[0].data["depth"] == 1


async def test_propagate_skips_only_pending_tasks():
    """Completed tasks should not be skipped."""
    dag = TaskDAG()
    dag.add_task(Task(task_id="a", name="a", agent_role="fail"))
    dag.add_task(Task(task_id="b", name="b", agent_role="echo"))
    dag.add_task(Task(task_id="c", name="c", agent_role="echo", dependencies=["a", "b"]))

    llm = MockLLMBackend(responses={"default": "ok"})
    rt = Runtime(
        llm_backend=llm,
        handlers={"fail": FailingHandler(), "echo": EchoHandler()},
        recovery_policy=None,
    )

    with pytest.raises(RuntimeError, match="boom"):
        await rt.run(dag)

    assert dag.tasks["b"].status == TaskStatus.COMPLETED
    assert dag.tasks["c"].status == TaskStatus.SKIPPED


async def test_dag_continues_other_branches():
    """A fails -> B skipped, but independent branch D continues."""
    dag = TaskDAG()
    dag.add_task(Task(task_id="a", name="a", agent_role="fail"))
    dag.add_task(Task(task_id="b", name="b", agent_role="echo", dependencies=["a"]))
    dag.add_task(Task(task_id="d", name="d", agent_role="echo"))

    llm = MockLLMBackend(responses={"default": "ok"})
    rt = Runtime(
        llm_backend=llm,
        handlers={"fail": FailingHandler(), "echo": EchoHandler()},
        recovery_policy=None,
    )

    with pytest.raises(RuntimeError, match="boom"):
        await rt.run(dag)

    assert dag.tasks["a"].status == TaskStatus.FAILED
    assert dag.tasks["b"].status == TaskStatus.SKIPPED
    assert dag.tasks["d"].status == TaskStatus.COMPLETED


async def test_coding_pipeline_with_propagation():
    """
    Planner succeeds -> Coder fails -> Tester skipped -> Fixer skipped.
    Verify: DAG doesn't hang, outputs partial result, event chain complete.
    """
    dag = TaskDAG()
    dag.add_task(Task(task_id="plan", name="plan", agent_role="echo"))
    dag.add_task(Task(task_id="code", name="code", agent_role="fail", dependencies=["plan"]))
    dag.add_task(Task(task_id="test", name="test", agent_role="echo", dependencies=["code"]))
    dag.add_task(Task(task_id="fix", name="fix", agent_role="echo", dependencies=["test"]))

    llm = MockLLMBackend(responses={"default": "ok"})
    rt = Runtime(
        llm_backend=llm,
        handlers={"echo": EchoHandler(), "fail": FailingHandler()},
        recovery_policy=None,
    )

    with pytest.raises(RuntimeError, match="boom"):
        await rt.run(dag)

    assert dag.tasks["plan"].status == TaskStatus.COMPLETED
    assert dag.tasks["code"].status == TaskStatus.FAILED
    assert dag.tasks["test"].status == TaskStatus.SKIPPED
    assert dag.tasks["fix"].status == TaskStatus.SKIPPED

    assert dag.is_complete() is True

    propagated_events = [
        e for e in rt.events if e.event_type == "fault.propagated"
    ]
    assert len(propagated_events) == 2
    assert propagated_events[0].task_id == "test"
    assert propagated_events[0].data["source_task_id"] == "code"
    assert propagated_events[0].data["depth"] == 1
    assert propagated_events[1].task_id == "fix"
    assert propagated_events[1].data["source_task_id"] == "code"
    assert propagated_events[1].data["depth"] == 2


async def test_runtime_emits_scheduler_decision_with_fairness():
    """Test that scheduler.decision event includes fairness data."""

    class TaskIdHandler(AgentHandler):
        async def run(self, ctx: AgentHandler.Context) -> str:
            return ctx.task.task_id

    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))
    dag.add_task(Task(task_id="t2", name="work2", agent_role="worker"))

    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"worker": TaskIdHandler()},
    )

    await rt.run(dag)

    decisions = [
        event for event in rt.events if event.event_type == "scheduler.decision"
    ]
    assert decisions
    first_decision = decisions[0]
    assert "fairness" in first_decision.data
    assert "fairness_score" in first_decision.data["fairness"]
    assert "starvation_risk" in first_decision.data["fairness"]
    assert 0.0 <= first_decision.data["fairness"]["fairness_score"] <= 1.0


async def test_runtime_emits_policy_selected_event():
    """Test that Runtime emits scheduler.policy_selected event."""

    class TaskIdHandler(AgentHandler):
        async def run(self, ctx: AgentHandler.Context) -> str:
            return ctx.task.task_id

    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"worker": TaskIdHandler()},
    )

    await rt.run(dag)

    policy_events = [
        event for event in rt.events if event.event_type == "scheduler.policy_selected"
    ]
    assert policy_events
    assert policy_events[0].data["policy"] == "latency_first"
    assert "workload_class" in policy_events[0].data
    assert "reason" in policy_events[0].data
    assert "candidate_count" in policy_events[0].data


async def test_runtime_emits_fairness_warning_event():
    """Test that Runtime emits scheduler.fairness_warning events."""
    from agentos.scheduler.fairness import FairnessTracker

    class TaskIdHandler(AgentHandler):
        async def run(self, ctx: AgentHandler.Context) -> str:
            return ctx.task.task_id

    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))
    dag.add_task(Task(task_id="t2", name="work2", agent_role="worker"))

    # Use low thresholds to trigger warnings quickly
    tracker = FairnessTracker(
        medium_wait_ms=10,
        high_wait_ms=20,
        medium_deferred_count=1,
        high_deferred_count=2,
    )

    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"worker": TaskIdHandler()},
        max_batch_size=1,
        scheduler_fairness_tracker=tracker,
    )

    await rt.run(dag)

    warning_events = [
        event for event in rt.events if event.event_type == "scheduler.fairness_warning"
    ]
    # There should be at least one fairness warning
    assert len(warning_events) > 0
    warning = warning_events[0]
    assert "task_id" in warning.data
    assert "starvation_risk" in warning.data
    assert warning.data["starvation_risk"] in {"medium", "high"}


async def test_runtime_emits_scheduler_kernel_probe_event():
    """Test that Runtime emits scheduler.kernel_probe event."""
    from agentos.isolation.sched_ext import KernelProbeResult

    class TaskIdHandler(AgentHandler):
        async def run(self, ctx: AgentHandler.Context) -> str:
            return ctx.task.task_id

    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

    probe = KernelProbeResult(
        sched_ext="unavailable",
        bpf_map="mock",
        fallback_mode="mock_bpf_map",
        reason="test",
    )
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"worker": TaskIdHandler()},
        kernel_probe=probe,
    )

    await rt.run(dag)

    events = [e for e in rt.events if e.event_type == "scheduler.kernel_probe"]
    assert len(events) == 1
    assert events[0].data["sched_ext"] == "unavailable"
    assert events[0].data["bpf_map"] == "mock"
    assert events[0].data["fallback_mode"] == "mock_bpf_map"


async def test_runtime_emits_kernel_probe_only_once():
    """Test that scheduler.kernel_probe is emitted only once per run."""
    from agentos.isolation.sched_ext import KernelProbeResult

    class TaskIdHandler(AgentHandler):
        async def run(self, ctx: AgentHandler.Context) -> str:
            return ctx.task.task_id

    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))
    dag.add_task(Task(task_id="t2", name="work2", agent_role="worker"))

    probe = KernelProbeResult(
        sched_ext="unavailable",
        bpf_map="mock",
        fallback_mode="mock_bpf_map",
        reason="test",
    )
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"worker": TaskIdHandler()},
        kernel_probe=probe,
    )

    await rt.run(dag)

    events = [e for e in rt.events if e.event_type == "scheduler.kernel_probe"]
    assert len(events) == 1


async def test_runtime_emits_bpf_map_updated_event():
    """Runtime publishes decisions only through its kernel adapter."""
    from agentos.isolation.bpf import MockBPFMapAdapter
    from agentos.isolation.cgroup import CgroupV2Manager
    from agentos.isolation.kernel_adapter import UserSpaceKernelAdapter
    from agentos.isolation.sched_ext import KernelProbeResult

    class RecordingKernelAdapter(UserSpaceKernelAdapter):
        def __init__(self):
            super().__init__(
                cgroup_manager=CgroupV2Manager(enable_controllers=False),
                bpf_map_adapter=MockBPFMapAdapter(),
                probe_result=KernelProbeResult(
                    sched_ext="unavailable",
                    bpf_map="mock",
                    fallback_mode="mock_bpf_map",
                    reason="unit test",
                ),
            )
            self.published_decisions = []

        def publish_decision(self, decision):
            self.published_decisions.append(decision)
            return super().publish_decision(decision)

    class TaskIdHandler(AgentHandler):
        async def run(self, ctx: AgentHandler.Context) -> str:
            return ctx.task.task_id

    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

    adapter = RecordingKernelAdapter()
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"worker": TaskIdHandler()},
        kernel_adapter=adapter,
    )

    await rt.run(dag)

    events = [e for e in rt.events if e.event_type == "scheduler.bpf_map_updated"]
    assert len(events) > 0
    assert events[0].data["mode"] == "mock"
    assert events[0].data["task_count"] >= 1
    assert "updated_task_ids" in events[0].data
    publication_events = [
        event for event in rt.events
        if event.event_type == "kernel.decision_published"
    ]
    assert len(publication_events) == 1
    assert publication_events[0].data["bpf_mode"] == "mock"
    assert publication_events[0].data["record_count"] == 1
    assert len(adapter.published_decisions) == 1
    assert set(adapter.published_decisions[0]["tasks"]) == {"t1"}


async def test_runtime_decision_event_carries_kernel_probe_and_bpf_map():
    """Test that scheduler.decision event carries kernel_probe and bpf_map."""
    from agentos.isolation.bpf import MockBPFMapAdapter
    from agentos.isolation.sched_ext import KernelProbeResult

    class TaskIdHandler(AgentHandler):
        async def run(self, ctx: AgentHandler.Context) -> str:
            return ctx.task.task_id

    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

    probe = KernelProbeResult(
        sched_ext="unavailable",
        bpf_map="mock",
        fallback_mode="mock_bpf_map",
        reason="test",
    )
    adapter = MockBPFMapAdapter()
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"worker": TaskIdHandler()},
        kernel_probe=probe,
        bpf_map_adapter=adapter,
    )

    await rt.run(dag)

    decision_events = [e for e in rt.events if e.event_type == "scheduler.decision"]
    assert len(decision_events) > 0
    first_decision = decision_events[0]
    assert "kernel_probe" in first_decision.data
    assert "bpf_map" in first_decision.data
    assert first_decision.data["kernel_probe"]["bpf_map"] == "mock"
    assert first_decision.data["bpf_map"]["mode"] == "mock"


async def test_runtime_emits_scheduler_runtime_health_event():
    """Test that Runtime emits scheduler.runtime_health event."""

    class TaskIdHandler(AgentHandler):
        async def run(self, ctx: AgentHandler.Context) -> str:
            return ctx.task.task_id

    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"worker": TaskIdHandler()},
    )

    await rt.run(dag)

    health_events = [e for e in rt.events if e.event_type == "scheduler.runtime_health"]
    assert len(health_events) > 0
    health_event = health_events[0]
    assert "runtime_health_score" in health_event.data
    assert "queue_pressure" in health_event.data
    assert "resource_pressure" in health_event.data
    assert "fairness_pressure" in health_event.data
    assert "dependency_pressure" in health_event.data
    assert "pressure_level" in health_event.data
    assert "health_warnings" in health_event.data


@pytest.mark.parametrize("explicit", [False, True])
async def test_empty_context_refs_can_explicitly_exclude_shared_context(explicit):
    cm = ContextManager()
    cm.add_shared_ref("policy", "shared policy that must be opt-in")
    dag = TaskDAG()
    dag.add_task(Task(
        task_id="empty-refs", name="empty refs", agent_role="echo",
        metadata={"explicit_context_refs": explicit},
    ))
    backend = MockLLMBackend(latency=0)
    rt = Runtime(backend, {"echo": EchoHandler()}, context_manager=cm,
                 enable_kernel_probe=False)
    await rt.run(dag)
    assembled = next(e for e in rt.events if e.event_type == "context.assembled")
    assert assembled.data["shared_refs"] == ([] if explicit else ["policy"])
    assert any("shared policy that must be opt-in" in m["content"]
               for m in backend.requests[0].messages) is (not explicit)
