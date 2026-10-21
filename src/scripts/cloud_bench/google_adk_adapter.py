"""Google ADK adapter.

Uses google-adk LlmAgent + LiteLlm for agent orchestration.
Falls back to skipped if google-adk is not installed.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from scripts.cloud_bench.common import RunMetrics
from scripts.cloud_bench.software_task import AgentStep
from scripts.cloud_bench import llm_call as llm
from agentos.events.log import EventLog


async def run_native(steps: list[AgentStep], max_tokens: int = 512) -> RunMetrics:
    """Run agents natively via Google ADK."""
    metrics = RunMetrics()

    try:
        from google.adk.agents import LlmAgent
        from google.adk.models.lite_llm import LiteLlm
    except ImportError:
        metrics.status = "skipped"
        metrics.error = "google-adk not installed"
        return metrics

    latencies: list[float] = []
    total_prompt = 0
    total_completion = 0
    total_hit = 0
    total_miss = 0

    try:
        for step in steps:
            agent = LlmAgent(
                name=step.role,
                model="deepseek-v4-flash",
                instruction="You are a senior software engineer. Be concise.",
            )
            start = time.monotonic()
            content = ""
            async for event in agent.run_async(step.prompt):
                if hasattr(event, "content"):
                    content += str(event.content)
                elif hasattr(event, "text"):
                    content += str(event.text)
                elif isinstance(event, str):
                    content += event
            elapsed = time.monotonic() - start
            latencies.append(elapsed)
            est_prompt = len(step.prompt) // 4
            est_completion = len(content) // 4
            total_prompt += est_prompt
            total_completion += est_completion
            total_miss += est_prompt

        metrics.avg_latency = sum(latencies) / len(latencies) if latencies else 0
        metrics.prompt_tokens = total_prompt
        metrics.completion_tokens = total_completion
        metrics.cache_hit_tokens = total_hit
        metrics.cache_miss_tokens = total_miss
        metrics.observable_events = len(steps) * 2
        metrics.compute_cost()
        metrics.compute_context_reuse()

    except Exception as exc:
        metrics.status = "error"
        metrics.error = f"google_adk: {str(exc)[:120]}"

    return metrics


async def run_with_runtime(
    steps: list[AgentStep],
    max_tokens: int = 512,
    event_log: EventLog | None = None,
) -> RunMetrics:
    """Run agents via Google ADK + ModelGateway."""
    metrics = RunMetrics()

    try:
        from google.adk.agents import LlmAgent
        from google.adk.models.lite_llm import LiteLlm
    except ImportError:
        metrics.status = "skipped"
        metrics.error = "google-adk not installed"
        return metrics

    latencies: list[float] = []
    total_prompt = 0
    total_completion = 0
    total_hit = 0
    total_miss = 0
    event_count = 0
    recovered = 0

    if event_log:
        event_count = llm.emit_event(event_log, "cloud.run_started", {"framework": "google_adk"}, event_count)

    try:
        for step in steps:
            inject_fault = (step.role == "backend_coder")

            if event_log:
                event_count = llm.emit_event(event_log, "cloud.agent_started", {"role": step.role}, event_count)

            if inject_fault and event_log:
                event_count = llm.emit_event(event_log, "fault.injected", {
                    "fault_type": "timeout", "task_id": step.role,
                }, event_count)

            _, usage, ev_count = await llm.runtime_call(
                step.prompt, prefix_id="wishlist-shared-spec",
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
                    event_count = llm.emit_event(event_log, "fault.recovery_completed", {
                        "task_id": step.role, "recovery_type": "retry",
                    }, event_count)

    except Exception as exc:
        metrics.status = "error"
        metrics.error = f"google_adk runtime: {str(exc)[:120]}"

    if event_log:
        event_count = llm.emit_event(event_log, "cloud.run_completed", {
            "framework": "google_adk", "recovered_faults": recovered,
        }, event_count)

    metrics.avg_latency = sum(latencies) / len(latencies) if latencies else 0
    metrics.prompt_tokens = total_prompt
    metrics.completion_tokens = total_completion
    metrics.cache_hit_tokens = total_hit
    metrics.cache_miss_tokens = total_miss
    metrics.recovered_faults = recovered
    metrics.observable_events = event_count
    metrics.compute_cost()
    metrics.compute_context_reuse()
    return metrics
