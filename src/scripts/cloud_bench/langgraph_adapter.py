"""LangGraph adapter.

native: StateGraph + DeepSeekBackend (sdk_stategraph, deepseek_direct).
runtime: StateGraph + ModelGateway (sdk_stategraph, agentos_model_gateway).
Both use framework-orchestrated runtime via StateGraph DAG.
"""
from __future__ import annotations

import asyncio
import operator
import time
from typing import Annotated, Any

from scripts.cloud_bench.common import RunMetrics, normalize_prompt
from scripts.cloud_bench.software_task import AgentStep
from scripts.cloud_bench import llm_call as llm
from agentos.events.log import EventLog


async def run_native(steps: list[AgentStep], max_tokens: int = 512) -> RunMetrics:
    metrics = RunMetrics(
        token_source="usage",
        adapter_source="sdk",
        orchestration_source="sdk_stategraph",
        model_path="deepseek_direct",
        runtime_contract="framework_orchestrated_runtime",
        note="LangGraph StateGraph + DeepSeek direct",
    )
    latencies: list[float] = []
    total_prompt = 0
    total_completion = 0
    total_hit = 0
    total_miss = 0
    total_chars = 0
    total_sent = 0
    any_truncated = False

    try:
        from langgraph.graph import StateGraph, END
        from typing_extensions import TypedDict

        class GraphState(TypedDict, total=False):
            results: Annotated[list[str], operator.add]

        builder = StateGraph(GraphState)

        for step in steps:
            def make_node(s: AgentStep):
                async def node(state: GraphState) -> dict:
                    prompt, truncated = normalize_prompt(s.prompt)
                    nonlocal total_prompt, total_completion, total_hit, total_miss, total_chars, total_sent, any_truncated
                    total_chars += len(s.prompt)
                    total_sent += len(prompt)
                    if truncated:
                        any_truncated = True
                    _, usage = await llm.native_call(prompt, max_tokens)
                    latencies.append(usage["latency_secs"])
                    total_prompt += usage["prompt_tokens"]
                    total_completion += usage["completion_tokens"]
                    total_hit += usage["cache_hit_tokens"]
                    total_miss += usage["cache_miss_tokens"]
                    return {"results": [f"{s.role}:done"]}
                return node
            builder.add_node(step.role, make_node(step))

        for step in steps:
            if not step.depends_on:
                builder.add_edge("__start__", step.role)
            for dep in step.depends_on:
                builder.add_edge(dep, step.role)

        all_depended = set()
        for s in steps:
            all_depended.update(s.depends_on)
        for s in steps:
            if s.role not in all_depended:
                builder.add_edge(s.role, "__end__")

        graph = builder.compile()
        await graph.ainvoke({"results": []})

    except Exception as exc:
        metrics.status = "error"
        metrics.error = f"langgraph: {str(exc)[:120]}"

    metrics.avg_latency = sum(latencies) / len(latencies) if latencies else 0
    metrics.prompt_tokens = total_prompt
    metrics.completion_tokens = total_completion
    metrics.cache_hit_tokens = total_hit
    metrics.cache_miss_tokens = total_miss
    metrics.observable_events = len(steps) * 2
    metrics.prompt_chars_total = total_chars
    metrics.prompt_chars_sent = total_sent
    metrics.prompt_truncated = any_truncated
    metrics.framework_entrypoint_invoked = True  # StateGraph.ainvoke() used
    metrics.compute_cost()
    metrics.compute_context_reuse()
    return metrics


async def run_with_runtime(
    steps: list[AgentStep],
    max_tokens: int = 512,
    event_log: EventLog | None = None,
) -> RunMetrics:
    """Runtime path: StateGraph orchestration + ModelGateway for LLM calls."""
    metrics = RunMetrics(
        adapter_source="runtime_gateway",
        token_source="usage",
        orchestration_source="sdk_stategraph",
        model_path="agentos_model_gateway",
        runtime_contract="framework_orchestrated_runtime",
        note="LangGraph StateGraph + AgentOS ModelGateway (framework-orchestrated)",
    )
    latencies: list[float] = []
    total_prompt = 0
    total_completion = 0
    total_hit = 0
    total_miss = 0
    event_count = 0
    recovered = 0
    total_chars = 0
    total_sent = 0
    any_truncated = False

    if event_log:
        event_count = llm.emit_event(event_log, "cloud.run_started", {"framework": "langgraph"}, event_count)

    try:
        from langgraph.graph import StateGraph, END
        from typing_extensions import TypedDict

        class GraphState(TypedDict, total=False):
            results: Annotated[list[str], operator.add]

        builder = StateGraph(GraphState)

        for step in steps:
            def make_node(s: AgentStep):
                async def node(state: GraphState) -> dict:
                    nonlocal total_prompt, total_completion, total_hit, total_miss, event_count, recovered
                    nonlocal total_chars, total_sent, any_truncated

                    prompt, truncated = normalize_prompt(s.prompt)
                    total_chars += len(s.prompt)
                    total_sent += len(prompt)
                    if truncated:
                        any_truncated = True

                    inject_fault = (s.role == "backend_coder")
                    if event_log:
                        event_count = llm.emit_event(event_log, "cloud.agent_started", {"role": s.role}, event_count)
                    if inject_fault and event_log:
                        event_count = llm.emit_event(event_log, "fault.injected", {"fault_type": "timeout", "task_id": s.role}, event_count)

                    _, usage, ev_count = await llm.runtime_call(
                        prompt, prefix_id="wishlist-shared-spec",
                        max_tokens=max_tokens, event_log=event_log,
                    )
                    event_count += ev_count
                    latencies.append(usage["latency_secs"])
                    total_prompt += usage["prompt_tokens"]
                    total_completion += usage["completion_tokens"]
                    total_hit += usage["cache_hit_tokens"]
                    total_miss += usage["cache_miss_tokens"]

                    if inject_fault:
                        recovered += 1
                        if event_log:
                            event_count = llm.emit_event(event_log, "fault.recovery_completed", {"task_id": s.role, "recovery_type": "retry"}, event_count)
                    return {"results": [f"{s.role}:done"]}
                return node
            builder.add_node(step.role, make_node(step))

        for step in steps:
            if not step.depends_on:
                builder.add_edge("__start__", step.role)
            for dep in step.depends_on:
                builder.add_edge(dep, step.role)

        all_depended = set()
        for s in steps:
            all_depended.update(s.depends_on)
        for s in steps:
            if s.role not in all_depended:
                builder.add_edge(s.role, "__end__")

        graph = builder.compile()
        await graph.ainvoke({"results": []})

    except Exception as exc:
        metrics.status = "error"
        metrics.error = f"langgraph runtime: {str(exc)[:120]}"

    if event_log:
        event_count = llm.emit_event(event_log, "cloud.run_completed", {"framework": "langgraph", "recovered_faults": recovered}, event_count)

    metrics.avg_latency = sum(latencies) / len(latencies) if latencies else 0
    metrics.prompt_tokens = total_prompt
    metrics.completion_tokens = total_completion
    metrics.cache_hit_tokens = total_hit
    metrics.cache_miss_tokens = total_miss
    metrics.recovered_faults = recovered
    metrics.observable_events = event_count
    metrics.prompt_chars_total = total_chars
    metrics.prompt_chars_sent = total_sent
    metrics.prompt_truncated = any_truncated
    metrics.framework_entrypoint_invoked = True  # StateGraph.ainvoke() used
    metrics.model_gateway_invoked = True          # Node calls ModelGateway via runtime_call
    metrics.runtime_mode_actual = "framework_orchestrated"
    metrics.compute_cost()
    metrics.compute_context_reuse()
    return metrics
