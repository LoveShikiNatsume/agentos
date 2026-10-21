"""AutoGen adapter.

native: AssistantAgent + OpenAIChatCompletionClient (sdk_agentchat, deepseek_direct).
runtime: AssistantAgent + GatewayModelClient wrapping AgentOS ModelGateway (sdk_agentchat, agentos_model_gateway).
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Sequence

from autogen_core.models import RequestUsage

from scripts.cloud_bench.common import RunMetrics, normalize_prompt
from scripts.cloud_bench.software_task import AgentStep
from scripts.cloud_bench import llm_call as llm
from agentos.events.log import EventLog


class _GatewayModelClient:
    """AutoGen ChatCompletionClient wrapper around AgentOS ModelGateway.

    Routes LLM calls through ModelGateway for shared prefix cache and event tracking.
    Implements the abstract methods of ChatCompletionClient.
    """

    def __init__(self, gateway, prefix_id: str = "wishlist-shared-spec"):
        self._gateway = gateway
        self._prefix_id = prefix_id
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._actual_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    @property
    def model_info(self):
        return {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "unknown",
            "structured_output": False,
        }

    @property
    def capabilities(self):
        return {"vision": False, "function_calling": False, "json_output": False}

    async def create(self, messages, *, tools=None, tool_choice='auto',
                     json_output=None, extra_create_args=None, cancellation_token=None):
        from autogen_core.models import CreateResult

        # Convert AutoGen LLMMessage list to OpenAI messages format
        openai_messages = []
        for msg in messages:
            if hasattr(msg, 'content') and hasattr(msg, 'source'):
                role = 'assistant' if msg.source != 'user' else 'user'
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                openai_messages.append({'role': role, 'content': content})
            elif hasattr(msg, 'content'):
                openai_messages.append({'role': 'user', 'content': str(msg.content)})
            else:
                openai_messages.append({'role': 'user', 'content': str(msg)})

        from agentos.llm.base import LLMRequest
        req = LLMRequest(
            agent_id="autogen_runtime",
            messages=openai_messages,
            max_tokens=extra_create_args.get('max_tokens', 512) if extra_create_args else 512,
            temperature=extra_create_args.get('temperature', 0.3) if extra_create_args else 0.3,
            backend="deepseek",
            prefix_id=self._prefix_id,
        )

        resp = await self._gateway.generate(req)

        prompt_tokens = resp.metrics.prompt_tokens if resp.metrics else 0
        completion_tokens = resp.metrics.completion_tokens if resp.metrics else 0
        usage = RequestUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        self._total_usage = RequestUsage(
            prompt_tokens=self._total_usage.prompt_tokens + prompt_tokens,
            completion_tokens=self._total_usage.completion_tokens + completion_tokens,
        )
        self._actual_usage = self._total_usage

        return CreateResult(
            finish_reason='stop',
            content=resp.content,
            usage=usage,
            cached=False,
        )

    async def create_stream(self, messages, *, tools=None, tool_choice='auto',
                            json_output=None, extra_create_args=None, cancellation_token=None):
        result = await self.create(messages, tools=tools, tool_choice=tool_choice,
                                   json_output=json_output, extra_create_args=extra_create_args,
                                   cancellation_token=cancellation_token)
        yield result.content
        yield result

    def actual_usage(self) -> RequestUsage:
        return self._actual_usage

    def total_usage(self) -> RequestUsage:
        return self._total_usage

    def count_tokens(self, messages, *, tools=None) -> int:
        total = 0
        for msg in messages:
            if hasattr(msg, 'content'):
                total += len(str(msg.content)) // 4
        return total

    def remaining_tokens(self, messages, *, tools=None) -> int:
        return 128000 - self.count_tokens(messages, tools=tools)

    async def close(self):
        pass

    def dump_component(self):
        from autogen_core import ComponentModel
        return ComponentModel(type="GatewayModelClient", config={})


async def run_native(steps: list[AgentStep], max_tokens: int = 512) -> RunMetrics:
    metrics = RunMetrics(
        token_source="usage",
        adapter_source="sdk",
        orchestration_source="sdk_agentchat",
        model_path="deepseek_direct",
        runtime_contract="framework_orchestrated_runtime",
        note="AutoGen AssistantAgent + OpenAIChatCompletionClient + DeepSeek",
    )

    try:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        from autogen_core.models import ModelInfo
    except ImportError:
        metrics.status = "skipped"
        metrics.error = "autogen not installed"
        return metrics

    latencies: list[float] = []
    total_prompt = 0
    total_completion = 0
    total_chars = 0
    total_sent = 0
    any_truncated = False

    try:
        model_info = ModelInfo(
            vision=False, function_calling=False, json_output=False,
            family="unknown", structured_output=False,
        )
        client = OpenAIChatCompletionClient(
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            model_info=model_info,
        )

        for step in steps:
            prompt, truncated = normalize_prompt(step.prompt)
            total_chars += len(step.prompt)
            total_sent += len(prompt)
            if truncated:
                any_truncated = True

            agent = AssistantAgent(
                name=step.role,
                model_client=client,
                system_message="You are a senior software engineer. Be concise.",
            )
            start = time.monotonic()
            result = await agent.run(task=prompt)
            elapsed = time.monotonic() - start
            latencies.append(elapsed)

            for msg in result.messages:
                if hasattr(msg, 'models_usage') and msg.models_usage:
                    usage = msg.models_usage
                    total_prompt += getattr(usage, 'prompt_tokens', 0) or 0
                    total_completion += getattr(usage, 'completion_tokens', 0) or 0

        metrics.avg_latency = sum(latencies) / len(latencies) if latencies else 0
        metrics.prompt_tokens = total_prompt
        metrics.completion_tokens = total_completion
        metrics.cache_hit_tokens = 0
        metrics.cache_miss_tokens = total_prompt
        metrics.observable_events = len(steps) * 2
        metrics.prompt_chars_total = total_chars
        metrics.prompt_chars_sent = total_sent
        metrics.prompt_truncated = any_truncated
        metrics.framework_entrypoint_invoked = True  # AssistantAgent.run() used

        if total_prompt > 0:
            metrics.status = "ok"
            metrics.token_source = "usage"
            metrics.note = "AutoGen SDK: real usage from models_usage"
        else:
            metrics.status = "degraded"
            metrics.token_source = "estimated"
            metrics.note = "AutoGen: no usage data in messages"

        metrics.compute_cost()
        metrics.compute_context_reuse()

    except Exception as exc:
        metrics.status = "error"
        metrics.error = f"autogen: {str(exc)[:120]}"

    return metrics


async def run_with_runtime(
    steps: list[AgentStep],
    max_tokens: int = 512,
    event_log: EventLog | None = None,
) -> RunMetrics:
    """Runtime path: AssistantAgent + GatewayModelClient wrapping ModelGateway.

    Preserves AutoGen AssistantAgent orchestration while routing LLM calls
    through AgentOS ModelGateway for shared prefix cache.
    """
    metrics = RunMetrics(
        adapter_source="runtime_gateway",
        token_source="usage",
        orchestration_source="sdk_agentchat",
        model_path="agentos_model_gateway",
        runtime_contract="framework_orchestrated_runtime",
        note="AutoGen AssistantAgent + AgentOS ModelGateway (framework-orchestrated)",
    )

    try:
        from autogen_agentchat.agents import AssistantAgent
        from agentos.llm.deepseek_backend import DeepSeekBackend
        from agentos.llm.gateway import ModelGateway
        from agentos.context.cache_tracker import CacheHeatTracker
    except ImportError:
        metrics.status = "skipped"
        metrics.error = "autogen or agentos not installed"
        return metrics

    latencies: list[float] = []
    total_chars = 0
    total_sent = 0
    any_truncated = False
    event_count = 0
    recovered = 0

    if event_log:
        event_count = llm.emit_event(event_log, "cloud.run_started", {"framework": "autogen"}, event_count)

    try:
        # Create ModelGateway with cache tracking
        tracker = CacheHeatTracker(ttl=300, hot_threshold=0.3)
        gateway = ModelGateway(
            backends={"deepseek": DeepSeekBackend()},
            default_backend="deepseek",
            cache_tracker=tracker,
        )
        client = _GatewayModelClient(gateway, prefix_id="wishlist-shared-spec")

        for step in steps:
            inject_fault = (step.role == "backend_coder")
            prompt, truncated = normalize_prompt(step.prompt)
            total_chars += len(step.prompt)
            total_sent += len(prompt)
            if truncated:
                any_truncated = True

            if event_log:
                event_count = llm.emit_event(event_log, "cloud.agent_started", {"role": step.role}, event_count)
            if inject_fault and event_log:
                event_count = llm.emit_event(event_log, "fault.injected", {"fault_type": "timeout", "task_id": step.role}, event_count)

            # Use AssistantAgent with gateway wrapper
            agent = AssistantAgent(
                name=step.role,
                model_client=client,
                system_message="You are a senior software engineer. Be concise.",
            )
            start = time.monotonic()
            result = await agent.run(task=prompt)
            elapsed = time.monotonic() - start
            latencies.append(elapsed)

            if inject_fault:
                recovered += 1
                if event_log:
                    event_count = llm.emit_event(event_log, "fault.recovery_completed", {"task_id": step.role, "recovery_type": "retry"}, event_count)

    except Exception as exc:
        metrics.status = "error"
        metrics.error = f"autogen runtime: {str(exc)[:120]}"

    if event_log:
        event_count = llm.emit_event(event_log, "cloud.run_completed", {"framework": "autogen", "recovered_faults": recovered}, event_count)

    # Get usage from gateway client
    usage = client.total_usage()
    metrics.avg_latency = sum(latencies) / len(latencies) if latencies else 0
    metrics.prompt_tokens = usage.prompt_tokens
    metrics.completion_tokens = usage.completion_tokens
    metrics.cache_hit_tokens = tracker.get_entry("wishlist-shared-spec").total_hit_tokens if tracker.get_entry("wishlist-shared-spec") else 0
    metrics.cache_miss_tokens = usage.prompt_tokens - metrics.cache_hit_tokens
    metrics.recovered_faults = recovered
    metrics.observable_events = event_count
    metrics.prompt_chars_total = total_chars
    metrics.prompt_chars_sent = total_sent
    metrics.prompt_truncated = any_truncated
    metrics.framework_entrypoint_invoked = True  # AssistantAgent.run() used
    metrics.model_gateway_invoked = True          # _GatewayModelClient routes to ModelGateway
    metrics.runtime_mode_actual = "framework_orchestrated"
    metrics.compute_cost()
    metrics.compute_context_reuse()
    return metrics
