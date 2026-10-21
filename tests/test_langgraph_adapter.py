import asyncio
import json
import operator
from typing import Annotated, TypedDict

import pytest

pytest.importorskip("langgraph")
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.managed import IsLastStep
from langgraph.types import CachePolicy, Command, RetryPolicy, Send

from agentos.adapters.langgraph import UnsupportedGraphError, adapt_state_graph
from agentos.context.manager import ContextManager
from agentos.core.task import TaskStatus
from agentos.events.log import EventLog
from agentos.llm.gateway import ModelGateway
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import Runtime


class State(TypedDict, total=False):
    seed: int
    count: int
    left: str
    right: str
    merged: str
    text: str
    payload: dict


def one_node(action=None, *, schema=State, **node_options):
    graph = StateGraph(schema)
    graph.add_node("work", action or (lambda state: {}), **node_options)
    graph.add_edge(START, "work")
    graph.add_edge("work", END)
    return graph


def runtime(handlers, **options):
    return Runtime(
        MockLLMBackend(latency=0), handlers,
        enable_kernel_probe=False, **options,
    )


def test_topology_waiting_edges_refs_and_provenance_derive_from_graph():
    graph = StateGraph(State)
    # Registration order intentionally differs from dependency order.
    for name in ("join", "right_node", "left_node", "root"):
        graph.add_node(name, lambda state: {})
    graph.add_edge(START, "root")
    graph.add_edge("root", "left_node")
    graph.add_edge("root", "right_node")
    graph.add_edge(["left_node", "right_node"], "join")
    graph.add_edge("join", END)

    dag, handlers = adapt_state_graph(
        graph, graph_name="review",
        base_context_refs=("task", "policy", "task"), output_prefix="outputs/",
    )

    assert set(dag.tasks) == set(handlers) == set(graph.nodes)
    assert dag.tasks["root"].dependencies == []
    assert dag.tasks["join"].dependencies == ["right_node", "left_node"]
    assert dag.tasks["root"].context_refs == ["task", "policy"]
    assert dag.tasks["join"].context_refs == [
        "task", "policy", "outputs/right_node", "outputs/left_node",
    ]
    for name, task in dag.tasks.items():
        assert task.task_id == task.agent_role == name
        assert task.metadata["output_ref"] == f"outputs/{name}"
        assert task.metadata["source"] == {
            "framework": "langgraph", "node": name, "graph": "review",
        }
        assert task.metadata["is_terminal"] is (name == "join")
    assert not graph.compiled


async def test_partial_patches_retain_initial_and_transitive_state_and_publish_raw_results():
    seen = {}

    def first(state):
        assert state == {"seed": 4, "count": 0}
        return {"count": state["seed"] + 1}

    async def middle(state):
        seen["middle"] = dict(state)
        return {"left": "middle"}

    def last(state):
        seen["last"] = dict(state)
        return {"merged": f'{state["seed"]}:{state["count"]}:{state["left"]}'}

    graph = StateGraph(State)
    graph.add_node("first", first)
    graph.add_node("middle", middle)
    graph.add_node("last", last)
    graph.add_edge(START, "first")
    graph.add_edge("first", "middle")
    graph.add_edge("middle", "last")
    graph.add_edge("last", END)
    cm = ContextManager()
    dag, handlers = adapt_state_graph(
        graph, graph_name="chain", initial_state={"seed": 4, "count": 0},
    )
    rt = runtime(handlers, context_manager=cm)
    results = await rt.run(dag)

    assert results == {
        "first": {"count": 5}, "middle": {"left": "middle"},
        "last": {"merged": "4:5:middle"},
    }
    assert seen["middle"] == {"seed": 4, "count": 5}
    assert seen["last"] == {"seed": 4, "count": 5, "left": "middle"}
    assert dag.tasks["last"].context_refs == ["result/middle"]
    for name, patch in results.items():
        assert json.loads(cm.read(f"agent-{name}", f"result/{name}")) == patch
    assert sum(e.event_type == "context.result_published" for e in rt.events) == 3


async def test_diamond_merges_common_ancestor_once_and_preserves_later_overwrite():
    graph = StateGraph(State)
    seen = []

    def join(state):
        seen.append(dict(state))
        return {"merged": "done"}

    graph.add_node("root", lambda state: {"count": 1, "seed": 9})
    graph.add_node("left_node", lambda state: {"count": 2, "left": "L"})
    graph.add_node("right_node", lambda state: {"right": "R"})
    graph.add_node("join", join)
    graph.add_edge(START, "root")
    graph.add_edge("root", "left_node")
    graph.add_edge("root", "right_node")
    graph.add_edge(["left_node", "right_node"], "join")
    graph.add_edge("join", END)
    dag, handlers = adapt_state_graph(graph, graph_name="diamond")

    await runtime(handlers).run(dag)

    assert seen == [{"count": 2, "seed": 9, "left": "L", "right": "R"}]


async def test_incomparable_writes_fail_before_join_execution_or_publication():
    graph = StateGraph(State)
    joined = []

    def join(state):
        joined.append(state)
        return {}

    graph.add_node("left_node", lambda state: {"count": 1})
    graph.add_node("right_node", lambda state: {"count": 2})
    graph.add_node("join", join)
    graph.add_edge(START, "left_node")
    graph.add_edge(START, "right_node")
    graph.add_edge(["left_node", "right_node"], "join")
    graph.add_edge("join", END)
    cm = ContextManager()
    dag, handlers = adapt_state_graph(graph, graph_name="conflict")
    rt = runtime(handlers, context_manager=cm)

    with pytest.raises(ValueError, match="count|conflict|concurrent|incomparable"):
        await rt.run(dag)

    assert not joined
    assert cm.get_ref("result/join") is None
    assert dag.tasks["join"].status == TaskStatus.FAILED


async def test_node_input_schema_filters_unrequested_fields():
    class SeedInput(TypedDict):
        seed: int

    def narrow(state):
        assert state == {"seed": 7}
        return {"text": "filtered"}

    graph = one_node(narrow, input_schema=SeedInput)
    dag, handlers = adapt_state_graph(
        graph, graph_name="narrow", initial_state={"seed": 7, "count": 99},
    )
    assert await runtime(handlers).run(dag) == {"work": {"text": "filtered"}}


async def test_parallel_ctx_calls_use_mock_gateway_and_each_node_has_traceable_acb(tmp_path):
    started = set()
    both_started = asyncio.Event()

    async def branch(state, *, ctx):
        started.add(ctx.task.task_id)
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=2)
        assert state == {"seed": 3}
        reply = await ctx.llm_call(messages=[{"role": "user", "content": ctx.task.task_id}])
        return {"left" if ctx.task.task_id == "left_node" else "right": reply.content}

    graph = StateGraph(State)
    graph.add_node("left_node", branch)
    graph.add_node("right_node", branch)
    for name in ("left_node", "right_node"):
        graph.add_edge(START, name)
        graph.add_edge(name, END)
    cm = ContextManager()
    cm.add_shared_ref("policy", "POLICY")
    cm.add_shared_ref("unselected", "SHOULD NOT APPEAR")
    backend = MockLLMBackend(responses={"default": "ok"}, latency=0)
    gateway = ModelGateway({"default": backend})
    log = EventLog(tmp_path / "events.jsonl")
    dag, handlers = adapt_state_graph(
        graph, graph_name="parallel", initial_state={"seed": 3},
        base_context_refs=("policy",),
    )
    rt = Runtime(
        gateway, handlers, context_manager=cm, event_log=log,
        max_batch_size=2, enable_kernel_probe=False,
    )

    assert await rt.run(dag) == {"left_node": {"left": "ok"}, "right_node": {"right": "ok"}}
    assert backend.call_count == gateway.stats()["total_calls"] == 2
    assert {request.agent_id for request in backend.requests} == {"agent-left_node", "agent-right_node"}
    for request in backend.requests:
        assert request.messages[0] == {"role": "system", "content": "POLICY"}
        assert "SHOULD NOT APPEAR" not in str(request.messages)
    created = list(log.read(event_type="agent.created"))
    assert len(created) == 2
    for event in created:
        acb = rt.agents[event.agent_id]
        expected = {"framework": "langgraph", "node": event.task_id, "graph": "parallel"}
        assert acb.metadata["source"] == event.data["source"] == expected
        assert acb.task_id == event.task_id
        assert acb.trace_id == event.trace_id
        assert acb.context_refs == ["policy"]
        assert acb.stats.llm_calls == 1
    response_events = [e for e in rt.events if e.event_type == "llm.response_received"]
    assert {e.agent_id for e in response_events} == set(rt.agents)


async def test_empty_base_refs_keep_root_model_prompt_free_of_ambient_shared_refs():
    async def work(state, *, ctx):
        reply = await ctx.llm_call(messages=[{"role": "user", "content": "hello"}])
        return {"text": reply.content}

    cm = ContextManager()
    cm.add_shared_ref("task", "ambient task")
    cm.add_shared_ref("policy", "ambient policy")
    graph = one_node(work)
    dag, handlers = adapt_state_graph(graph, graph_name="empty-base")
    backend = MockLLMBackend(latency=0)
    rt = Runtime(backend, handlers, context_manager=cm, enable_kernel_probe=False)

    await rt.run(dag)

    assert dag.tasks["work"].context_refs == []
    assert backend.requests[0].messages == [{"role": "user", "content": "hello"}]
    assert backend.requests[0].prefix_id == "__no_prefix__"


@pytest.mark.parametrize("result", [
    None,
    ["not", "a", "patch"],
    {"unknown": 1},
    {"payload": {"bad": {1, 2}}},
    {"count": float("nan")},
    Command(update={"count": 1}, goto=END),
    Send("work", {"count": 1}),
], ids=["none", "list", "unknown-field", "non-json", "nan", "command", "send"])
async def test_invalid_node_result_fails_without_publishing(result):
    graph = one_node(lambda state: result)
    dag, handlers = adapt_state_graph(graph, graph_name="invalid-result")
    cm = ContextManager()
    rt = runtime(handlers, context_manager=cm)

    with pytest.raises((ValueError, TypeError), match="dict|mapping|JSON|json|Command|Send|field|key|unknown"):
        await rt.run(dag)

    assert cm.get_ref("result/work") is None
    assert not any(e.event_type == "context.result_published" for e in rt.events)


def test_conditional_edges_are_rejected_even_if_all_destinations_are_known():
    graph = one_node()
    graph.add_conditional_edges("work", lambda state: "end", {"end": END})
    with pytest.raises(UnsupportedGraphError, match="conditional|branch"):
        adapt_state_graph(graph, graph_name="conditional")


def test_cycle_is_rejected_instead_of_being_silently_unrolled():
    graph = StateGraph(State)
    graph.add_node("first", lambda state: {})
    graph.add_node("second", lambda state: {})
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", "first")
    graph.add_edge("second", END)
    with pytest.raises(UnsupportedGraphError, match="cycl|DAG|acyclic"):
        adapt_state_graph(graph, graph_name="cycle")


@pytest.mark.parametrize("missing", ["start", "end"])
def test_missing_start_or_end_is_rejected(missing):
    graph = StateGraph(State)
    graph.add_node("work", lambda state: {})
    if missing != "start":
        graph.add_edge(START, "work")
    if missing != "end":
        graph.add_edge("work", END)
    with pytest.raises(UnsupportedGraphError, match="START|END|start|end"):
        adapt_state_graph(graph, graph_name="missing-boundary")


def test_unreachable_node_is_rejected():
    graph = one_node()
    graph.add_node("orphan", lambda state: {})
    graph.add_edge("orphan", END)
    with pytest.raises(UnsupportedGraphError, match="reach|orphan"):
        adapt_state_graph(graph, graph_name="unreachable")


def test_empty_graph_is_rejected():
    with pytest.raises(UnsupportedGraphError):
        adapt_state_graph(StateGraph(State), graph_name="empty")


@pytest.mark.parametrize("compiled_builder", [False, True])
def test_compiled_graph_and_its_builder_are_rejected(compiled_builder):
    builder = one_node()
    compiled = builder.compile()
    with pytest.raises(UnsupportedGraphError, match="compil|StateGraph"):
        adapt_state_graph(builder if compiled_builder else compiled, graph_name="compiled")


class ReducedState(TypedDict):
    count: Annotated[int, operator.add]


class ManagedState(TypedDict):
    count: int
    is_last: IsLastStep


@pytest.mark.parametrize("schema", [dict, ReducedState, ManagedState], ids=["non-typeddict", "reducer", "managed"])
def test_unsupported_state_semantics_are_rejected(schema):
    with pytest.raises(UnsupportedGraphError, match="TypedDict|reduc|managed|channel|LastValue"):
        adapt_state_graph(one_node(schema=schema), graph_name="unsupported-schema")


@pytest.mark.parametrize("options", [
    {"defer": True},
    {"retry_policy": RetryPolicy(max_attempts=2)},
    {"cache_policy": CachePolicy()},
    {"timeout": 1.0},
    {"error_handler": lambda state: {}},
    {"destinations": (END,)},
], ids=["defer", "retry", "cache", "timeout", "error-handler", "destinations"])
def test_node_policies_and_declared_dynamic_destinations_are_rejected(options):
    with pytest.raises(UnsupportedGraphError, match="defer|retry|cache|timeout|error|ends|destination|dynamic"):
        adapt_state_graph(one_node(**options), graph_name="node-policy")


@pytest.mark.parametrize("options", [
    {"retry_policy": RetryPolicy(max_attempts=2)},
    {"cache_policy": CachePolicy()},
    {"timeout": 1.0},
    {"error_handler": lambda state: {}},
], ids=["retry", "cache", "timeout", "error-handler"])
def test_graph_default_policies_are_rejected_even_if_set_after_node_registration(options):
    graph = one_node()
    graph.set_node_defaults(**options)
    with pytest.raises(UnsupportedGraphError, match="default|retry|cache|timeout|error"):
        adapt_state_graph(graph, graph_name="default-policy")


def test_context_schema_is_rejected_instead_of_ignoring_injection():
    class Context(TypedDict):
        user: str

    graph = StateGraph(State, context_schema=Context)
    graph.add_node("work", lambda state: {})
    graph.add_edge(START, "work")
    graph.add_edge("work", END)
    with pytest.raises(UnsupportedGraphError, match="context_schema|context schema"):
        adapt_state_graph(graph, graph_name="context")


@pytest.mark.parametrize("action", [
    RunnableLambda(lambda state: {}),
    one_node().compile(),
], ids=["runnable", "subgraph"])
def test_complex_runnable_and_subgraph_are_rejected(action):
    with pytest.raises(UnsupportedGraphError, match="function|runnable|subgraph|callable"):
        adapt_state_graph(one_node(action), graph_name="complex-node")


def test_extra_node_arguments_are_rejected_before_execution():
    with pytest.raises(UnsupportedGraphError, match="signature|argument|parameter|ctx|state"):
        adapt_state_graph(one_node(lambda state, config: {}), graph_name="signature")


async def test_sync_ctx_and_nested_initial_state_are_snapshotted_and_isolated():
    initial = {"payload": {"tags": ["original"]}}
    observed = {}

    def left(state, *, ctx):
        observed[ctx.task.task_id] = list(state["payload"]["tags"])
        state["payload"]["tags"].append("left-local")
        return {"left": "done"}

    def right(state, *, ctx):
        observed[ctx.task.task_id] = list(state["payload"]["tags"])
        return {"right": "done"}

    graph = StateGraph(State)
    graph.add_node("left_node", left)
    graph.add_node("right_node", right)
    for name in ("left_node", "right_node"):
        graph.add_edge(START, name)
        graph.add_edge(name, END)
    dag, handlers = adapt_state_graph(graph, graph_name="snapshot", initial_state=initial)
    initial["payload"]["tags"].append("changed-after-adaptation")
    rt = runtime(handlers)

    assert await rt.run(dag) == {
        "left_node": {"left": "done"}, "right_node": {"right": "done"},
    }
    assert observed == {"left_node": ["original"], "right_node": ["original"]}
    assert initial == {"payload": {"tags": ["original", "changed-after-adaptation"]}}


@pytest.mark.parametrize("value", [
    {"payload": {"control": Command(goto=END)}},
    {"payload": {"control": [Send("work", {})]}},
], ids=["nested-command", "nested-send"])
async def test_dynamic_control_nested_inside_patch_is_rejected(value):
    graph = one_node(lambda state: value)
    dag, handlers = adapt_state_graph(graph, graph_name="nested-control")
    cm = ContextManager()
    rt = runtime(handlers, context_manager=cm)

    with pytest.raises(UnsupportedGraphError, match="Command|Send|dynamic"):
        await rt.run(dag)

    assert cm.get_ref("result/work") is None


@pytest.mark.parametrize("kwargs", [
    {"graph_name": ""},
    {"graph_name": "bad", "output_prefix": ""},
    {"graph_name": "bad", "base_context_refs": "policy"},
    {"graph_name": "bad", "base_context_refs": ("result/work",)},
    {"graph_name": "bad", "initial_state": {"unknown": 1}},
    {"graph_name": "bad", "initial_state": {"payload": {"not-json": object()}}},
], ids=["name", "prefix", "refs-string", "ref-collision", "initial-field", "initial-json"])
def test_invalid_adapter_configuration_is_rejected_before_any_node_runs(kwargs):
    with pytest.raises(ValueError):
        adapt_state_graph(one_node(), **kwargs)


def test_node_without_any_path_to_end_is_rejected():
    graph = one_node()
    graph.add_node("dead_end", lambda state: {})
    graph.add_edge(START, "dead_end")
    with pytest.raises(UnsupportedGraphError, match="END|end"):
        adapt_state_graph(graph, graph_name="dead-end")


def test_custom_runnable_with_distinct_sync_and_async_behavior_is_rejected():
    from langgraph._internal._runnable import RunnableCallable

    def sync(state):
        return {"count": 1}

    async def async_version(state):
        return {"count": 2}

    action = RunnableCallable(sync, async_version)
    with pytest.raises(UnsupportedGraphError, match="sync/async|Runnable|implementation"):
        adapt_state_graph(one_node(action), graph_name="ambiguous-runnable")


async def test_decorated_sync_node_preserves_wrapper_behavior():
    from functools import wraps

    def decorate(function):
        @wraps(function)
        def wrapper(state):
            patch = function(state)
            return {**patch, "text": "decorator-ran"}
        return wrapper

    @decorate
    def node(state):
        return {"count": state["seed"] + 1, "text": "original"}

    dag, handlers = adapt_state_graph(
        one_node(node), graph_name="decorated", initial_state={"seed": 4},
    )

    assert await runtime(handlers).run(dag) == {
        "work": {"count": 5, "text": "decorator-ran"},
    }
