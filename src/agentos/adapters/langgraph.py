"""Lower an uncompiled, static LangGraph into Runtime-owned tasks.

This is a causal DAG adapter, not a LangGraph Pregel executor. Node results
remain raw JSON state patches; inputs replay each ancestor patch exactly once.
LangGraph is optional and imported only when adaptation is requested.
"""
from __future__ import annotations

import asyncio
import heapq
import inspect
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from agentos.adapters.workflow import PlannedTask, adapt_plan
from agentos.core.task import TaskDAG
from agentos.runtime.engine import AgentHandler

if TYPE_CHECKING:
    from langgraph.graph import StateGraph


class UnsupportedGraphError(ValueError):
    """The graph requires semantics outside the static Runtime DAG subset."""


def _json_dict(value: Any, label: str) -> dict[str, Any]:
    # A JSON round trip gives every invocation an independent, transport-safe
    # value. Reject lossy conversions (integer keys, tuples, custom objects).
    def check(item: Any) -> None:
        if type(item) is dict:
            if not all(type(key) is str for key in item):
                raise ValueError(f"{label} must have JSON string keys")
            for child in item.values():
                check(child)
        elif type(item) is list:
            for child in item:
                check(child)
        elif type(item) not in (str, int, float, bool, type(None)):
            raise ValueError(f"{label} must contain only JSON values")

    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON dict state patch")
    check(value)
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must contain only finite JSON values") from exc


def _node_function(name: str, runnable: Any) -> tuple[Any, bool]:
    from langgraph._internal._runnable import RunnableCallable

    if (type(runnable) is not RunnableCallable
            or runnable.kwargs or runnable.explode_args):
        raise UnsupportedGraphError(
            f"node {name!r}: only plain sync/async functions are supported; "
            "nested graphs and composite/configured Runnables require adapt_plan"
        )
    if runnable.func is not None and getattr(runnable.afunc, "__wrapped__", None) is not runnable.func:
        raise UnsupportedGraphError(
            f"node {name!r}: custom sync/async Runnable implementations are unsupported"
        )
    function = runnable.func if runnable.func is not None else runnable.afunc
    if (function is None or inspect.isgeneratorfunction(function)
            or inspect.isasyncgenfunction(function)):
        raise UnsupportedGraphError(f"node {name!r}: generator nodes are unsupported")
    try:
        signature = inspect.signature(function)
        parameters = signature.parameters
        injected = {"config", "runtime", "store", "writer", "previous", "error"}
        if injected.intersection(parameters):
            raise UnsupportedGraphError(
                f"node {name!r}: LangGraph config/runtime injection is unsupported; "
                "use an explicit ctx parameter for the Runtime context"
            )
        accepts_ctx = "ctx" in parameters
        signature.bind({}, **({"ctx": object()} if accepts_ctx else {}))
    except (TypeError, ValueError) as exc:
        if isinstance(exc, UnsupportedGraphError):
            raise
        raise UnsupportedGraphError(
            f"node {name!r}: expected (state) or (state, *, ctx) signature"
        ) from exc
    return function, accepts_ctx


def adapt_state_graph(
    graph: StateGraph, *, graph_name: str,
    initial_state: Mapping[str, Any] | None = None,
    base_context_refs: Sequence[str] = (), output_prefix: str = "result/",
) -> tuple[TaskDAG, dict[str, AgentHandler]]:
    """Create one task/handler per node without compiling or mutating the graph.

    TypedDict state with LastValue channels is supported. Inputs consist of
    initial_state plus all causal ancestor patches. Concurrent writes to the
    same field are rejected. Each node's raw patch is published at output_prefix
    + node; only direct predecessor refs and base_context_refs enter its model
    context. A declared ctx parameter receives AgentHandler.Context.
    """
    from langgraph.channels.last_value import LastValue
    from langgraph.graph import END, START, StateGraph
    from typing_extensions import is_typeddict

    if not isinstance(graph, StateGraph) or graph.compiled:
        raise UnsupportedGraphError("expected an uncompiled LangGraph StateGraph")
    if not isinstance(graph_name, str) or not graph_name.strip():
        raise ValueError("graph_name must be a nonempty string")
    if not isinstance(output_prefix, str) or not output_prefix or output_prefix.startswith("ctx://"):
        raise ValueError("output_prefix must be a nonempty ref prefix without ctx://")
    if isinstance(base_context_refs, str) or not all(
        isinstance(ref, str) and ref for ref in base_context_refs
    ):
        raise ValueError("base_context_refs must be a sequence of nonempty ref names")
    base_refs = tuple(dict.fromkeys(base_context_refs))
    if graph.branches:
        raise UnsupportedGraphError("conditional edges and dynamic Send routing are unsupported")
    if graph.context_schema is not None or graph.managed:
        raise UnsupportedGraphError("LangGraph context_schema and managed state are unsupported")
    defaults = getattr(graph, "_node_defaults", None)
    if defaults is not None and any(
        getattr(defaults, field, None) is not None
        for field in ("retry_policy", "cache_policy", "error_handler", "timeout")
    ):
        raise UnsupportedGraphError("LangGraph default retry/cache/error/timeout policies are unsupported")
    schemas = (graph.state_schema, graph.input_schema, graph.output_schema,
               *(spec.input_schema for spec in graph.nodes.values()))
    if not all(is_typeddict(schema) for schema in schemas):
        raise UnsupportedGraphError("only TypedDict state/input/output schemas are supported")
    if not all(type(channel) is LastValue for channel in graph.channels.values()):
        raise UnsupportedGraphError("reducers and non-LastValue channels are unsupported")
    if not graph.nodes:
        raise UnsupportedGraphError("a static graph requires nodes and START/END edges")
    functions = {}
    for name, spec in graph.nodes.items():
        if any((spec.retry_policy, spec.cache_policy, spec.defer, spec.timeout,
                spec.ends, spec.is_error_handler, spec.error_handler_node)):
            raise UnsupportedGraphError(
                f"node {name!r}: dynamic destinations, defer, retry/cache/error/timeout "
                "policies are unsupported"
            )
        functions[name] = _node_function(name, spec.runnable)

    names = tuple(graph.nodes)
    rank = {name: index for index, name in enumerate(names)}
    parents: dict[str, set[str]] = {name: set() for name in names}
    children: dict[str, set[str]] = {name: set() for name in names}
    edges = set(graph.edges)
    for sources, destination in graph.waiting_edges:
        if not sources:
            raise UnsupportedGraphError("empty static join edges are unsupported")
        edges.update((source, destination) for source in sources)
    starts, ends = set(), set()
    for source, destination in edges:
        if source != START and source not in parents:
            raise UnsupportedGraphError(f"unknown static edge source: {source!r}")
        if destination != END and destination not in parents:
            raise UnsupportedGraphError(f"unknown static edge destination: {destination!r}")
        if source == START:
            if destination == END:
                raise UnsupportedGraphError("START to END bypass edges are unsupported")
            starts.add(destination)
        elif destination == END:
            ends.add(source)
        else:
            parents[destination].add(source)
            children[source].add(destination)
    if not starts or not ends:
        raise UnsupportedGraphError("a static graph requires explicit START and END edges")
    # Stable Kahn order handles nodes registered before their dependencies.
    pending = {name: len(parents[name]) for name in names}
    ready = [rank[name] for name in names if not pending[name]]
    heapq.heapify(ready)
    order = []
    while ready:
        name = names[heapq.heappop(ready)]
        order.append(name)
        for child in children[name]:
            pending[child] -= 1
            if not pending[child]:
                heapq.heappush(ready, rank[child])
    if len(order) != len(names):
        raise UnsupportedGraphError("cycles/loops cannot be lowered to a static TaskDAG")
    if any(parents[name] for name in starts):
        raise UnsupportedGraphError("START nodes with incoming node edges require repeated activation")
    reachable = set(starts)
    for name in order:
        if name in reachable:
            reachable.update(children[name])
    if reachable != set(names):
        raise UnsupportedGraphError(f"nodes unreachable from START: {sorted(set(names) - reachable)}")
    finishing = set(ends)
    for name in reversed(order):
        if name in finishing:
            finishing.update(parents[name])
    if finishing != set(names):
        raise UnsupportedGraphError("all nodes must have a path to END")

    initial = _json_dict(dict(initial_state) if initial_state is not None else {}, "initial_state")
    if set(initial) - set(graph.input_schema.__annotations__):
        raise ValueError("initial_state contains fields outside the graph input schema")
    fields = frozenset(graph.channels)
    output_refs = {name: output_prefix + name for name in names}
    if set(output_refs.values()).intersection(ref.removeprefix("ctx://") for ref in base_refs):
        raise ValueError("base_context_refs cannot overlap generated output refs")
    ancestors: dict[str, frozenset[str]] = {}
    for name in order:
        ancestors[name] = frozenset().union(
            parents[name], *(ancestors[parent] for parent in parents[name])
        )
    ancestor_order = {name: tuple(a for a in order if a in ancestors[name]) for name in names}

    def make_runner(name: str):
        function, accepts_ctx = functions[name]
        input_fields = frozenset(graph.nodes[name].input_schema.__annotations__)

        async def run(ctx: AgentHandler.Context) -> dict[str, Any]:
            from langgraph.types import Command, Send

            state = deepcopy(initial)
            writers: dict[str, str] = {}
            for ancestor in ancestor_order[name]:
                patch = json.loads(ctx.context_manager.read(ctx.acb.agent_id, output_refs[ancestor]))
                patch = _json_dict(patch, f"published result of {ancestor!r}")
                for key, value in patch.items():
                    previous = writers.get(key)
                    if previous is not None and previous not in ancestors[ancestor]:
                        raise UnsupportedGraphError(
                            f"node {name!r}: concurrent ancestors {previous!r} and "
                            f"{ancestor!r} write field {key!r}; reducers are unsupported"
                        )
                    state[key] = value
                    writers[key] = ancestor
            state = {key: value for key, value in state.items() if key in input_fields}
            kwargs = {"ctx": ctx} if accepts_ctx else {}
            if inspect.iscoroutinefunction(function):
                result = await function(state, **kwargs)
            else:
                result = await asyncio.to_thread(function, state, **kwargs)
                if inspect.isawaitable(result):
                    result = await result

            def reject_dynamic(value: Any) -> None:
                if isinstance(value, (Command, Send)):
                    raise UnsupportedGraphError(
                        f"node {name!r}: Command/Send dynamic control flow is unsupported"
                    )
                if isinstance(value, dict):
                    for child in value.values():
                        reject_dynamic(child)
                elif isinstance(value, (list, tuple)):
                    for child in value:
                        reject_dynamic(child)

            reject_dynamic(result)
            result = _json_dict(result, f"node {name!r} result")
            if set(result) - fields:
                raise ValueError(f"node {name!r} returned fields outside the graph state schema")
            return result

        return run

    plans, runners = [], {}
    for name in order:
        dependencies = tuple(sorted(parents[name], key=rank.__getitem__))
        runners[name] = make_runner(name)
        plans.append(PlannedTask(
            task_id=name, goal=name, runner=name, dependencies=dependencies,
            context_refs=tuple(dict.fromkeys((*base_refs, *(output_refs[p] for p in dependencies)))),
            output_ref=output_refs[name], metadata={
                "source": {"framework": "langgraph", "node": name, "graph": graph_name},
                "is_terminal": name in ends,
                "explicit_context_refs": True,
            },
        ))
    return adapt_plan(plans, runners, framework="langgraph")
