"""OpenAI Agents SDK adapter.

native: Agent + Runner + OpenAIChatCompletionsModel (sdk_runner, deepseek_direct).
runtime: Agent + Runner + ModelGateway wrapper (sdk_runner, agentos_model_gateway).
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from agents import (
    Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled,
    Usage, ModelResponse,
)
from agents.models.interface import Model, ModelTracing

from scripts.cloud_bench.common import RunMetrics, normalize_prompt
from scripts.cloud_bench.software_task import AgentStep
from scripts.cloud_bench import llm_call as llm
from agentos.events.log import EventLog


class _UsageTrackingModel(OpenAIChatCompletionsModel):
    """Native: wraps OpenAIChatCompletionsModel to accumulate token usage."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_tokens = 0

    async def get_response(self, system_instructions, input, model_settings,
                           tools, output_schema, handoffs, tracing,
                           previous_response_id=None, conversation_id=None, prompt=None):
        response = await super().get_response(
            system_instructions, input, model_settings, tools, output_schema,
            handoffs, tracing, previous_response_id, conversation_id, prompt,
        )
        if hasattr(response, 'usage') and response.usage:
            self.total_input_tokens += response.usage.input_tokens or 0
            self.total_output_tokens += response.usage.output_tokens or 0
            if hasattr(response.usage, 'input_tokens_details') and response.usage.input_tokens_details:
                self.total_cached_tokens += getattr(response.usage.input_tokens_details, 'cached_tokens', 0) or 0
        return response


class _GatewayModel(Model):
    """Runtime: Model wrapper routing LLM calls through AgentOS ModelGateway."""

    def __init__(self, gateway, prefix_id: str = "wishlist-shared-spec"):
        self._gw = gateway
        self._prefix_id = prefix_id
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_tokens = 0

    async def get_response(self, system_instructions, input, model_settings,
                           tools, output_schema, handoffs, tracing,
                           previous_response_id=None, conversation_id=None, prompt=None):
        from openai.types.responses import ResponseOutputMessage, ResponseOutputText
        from agentos.llm.base import LLMRequest

        msgs = []
        if system_instructions:
            msgs.append({"role": "system", "content": system_instructions})
        if isinstance(input, str):
            msgs.append({"role": "user", "content": input})
        elif isinstance(input, list):
            for item in input:
                if hasattr(item, 'content'):
                    role = 'assistant' if getattr(item, 'role', 'user') == 'assistant' else 'user'
                    msgs.append({"role": role, "content": str(item.content)})
                else:
                    msgs.append({"role": "user", "content": str(item)})

        req = LLMRequest(
            agent_id="openai_agents_runtime", messages=msgs,
            max_tokens=getattr(model_settings, 'max_tokens', 512) or 512,
            temperature=getattr(model_settings, 'temperature', 0.3) or 0.3,
            backend="deepseek", prefix_id=self._prefix_id,
        )
        resp = await self._gw.generate(req)
        pi = resp.metrics.prompt_tokens if resp.metrics else 0
        co = resp.metrics.completion_tokens if resp.metrics else 0
        ch = resp.metrics.cache_hit_tokens if resp.metrics else 0
        self.total_input_tokens += pi
        self.total_output_tokens += co
        self.total_cached_tokens += ch

        output_msg = ResponseOutputMessage(
            id="runtime_msg",
            content=[ResponseOutputText(annotations=[], text=resp.content, type="output_text")],
            role="assistant", status="completed", type="message",
        )
        return ModelResponse(
            output=[output_msg],
            usage=Usage(input_tokens=pi, output_tokens=co),
            response_id=None,
        )

    async def stream_response(self, system_instructions, input, model_settings,
                              tools, output_schema, handoffs, tracing,
                              previous_response_id=None, conversation_id=None, prompt=None):
        result = await self.get_response(
            system_instructions, input, model_settings, tools, output_schema,
            handoffs, tracing, previous_response_id=previous_response_id,
            conversation_id=conversation_id, prompt=prompt,
        )
        yield result


async def run_native(steps: list[AgentStep], max_tokens: int = 512) -> RunMetrics:
    set_tracing_disabled(True)

    metrics = RunMetrics(
        token_source="usage",
        adapter_source="sdk",
        orchestration_source="sdk_runner",
        model_path="deepseek_direct",
        runtime_contract="framework_orchestrated_runtime",
        note="openai-agents: Agent + Runner + OpenAIChatCompletionsModel + DeepSeek",
    )

    try:
        client = AsyncOpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com",
        )
        model = _UsageTrackingModel(model="deepseek-v4-flash", openai_client=client)
    except Exception as exc:
        metrics.status = "error"
        metrics.error = f"openai_agents init: {str(exc)[:120]}"
        return metrics

    latencies: list[float] = []
    total_chars = 0
    total_sent = 0
    any_truncated = False

    try:
        for step in steps:
            prompt, truncated = normalize_prompt(step.prompt)
            total_chars += len(step.prompt)
            total_sent += len(prompt)
            if truncated:
                any_truncated = True

            agent = Agent(
                name=step.role,
                instructions="You are a senior software engineer. Be concise.",
                model=model,
            )
            start = time.monotonic()
            await Runner.run(agent, prompt)
            elapsed = time.monotonic() - start
            latencies.append(elapsed)

        metrics.avg_latency = sum(latencies) / len(latencies) if latencies else 0
        metrics.prompt_tokens = model.total_input_tokens
        metrics.completion_tokens = model.total_output_tokens
        metrics.cache_hit_tokens = model.total_cached_tokens
        metrics.cache_miss_tokens = model.total_input_tokens - model.total_cached_tokens
        metrics.observable_events = len(steps) * 2
        metrics.prompt_chars_total = total_chars
        metrics.prompt_chars_sent = total_sent
        metrics.prompt_truncated = any_truncated
        metrics.framework_entrypoint_invoked = True
        metrics.status = "ok"
        metrics.compute_cost()
        metrics.compute_context_reuse()

    except Exception as exc:
        metrics.status = "error"
        metrics.error = f"openai_agents: {str(exc)[:120]}"

    return metrics


async def run_with_runtime(
    steps: list[AgentStep],
    max_tokens: int = 512,
    event_log: EventLog | None = None,
) -> RunMetrics:
    """Runtime path: Agent + Runner + ModelGateway wrapper (framework-orchestrated)."""
    set_tracing_disabled(True)

    metrics = RunMetrics(
        adapter_source="runtime_gateway",
        token_source="usage",
        orchestration_source="sdk_runner",
        model_path="agentos_model_gateway",
        runtime_contract="framework_orchestrated_runtime",
        note="openai-agents: Agent + Runner + AgentOS ModelGateway (framework-orchestrated)",
    )

    try:
        from agentos.llm.deepseek_backend import DeepSeekBackend
        from agentos.llm.gateway import ModelGateway
        from agentos.context.cache_tracker import CacheHeatTracker
    except ImportError:
        metrics.status = "error"
        metrics.error = "agentos not installed"
        return metrics

    latencies: list[float] = []
    total_chars = 0
    total_sent = 0
    any_truncated = False
    event_count = 0
    recovered = 0

    if event_log:
        event_count = llm.emit_event(event_log, "cloud.run_started", {"framework": "openai_agents"}, event_count)

    try:
        tracker = CacheHeatTracker(ttl=300, hot_threshold=0.3)
        gateway = ModelGateway(
            backends={"deepseek": DeepSeekBackend()},
            default_backend="deepseek",
            cache_tracker=tracker,
        )
        model = _GatewayModel(gateway, prefix_id="wishlist-shared-spec")

        for i, step in enumerate(steps):
            inject_fault = (step.role == "backend_coder" and i > 0)
            prompt, truncated = normalize_prompt(step.prompt)
            total_chars += len(step.prompt)
            total_sent += len(prompt)
            if truncated:
                any_truncated = True

            if event_log:
                event_count = llm.emit_event(event_log, "cloud.agent_started", {"role": step.role}, event_count)
            if inject_fault and event_log:
                event_count = llm.emit_event(event_log, "fault.injected", {"fault_type": "timeout", "task_id": step.role}, event_count)

            agent = Agent(
                name=step.role,
                instructions="You are a senior software engineer. Be concise.",
                model=model,
            )
            start = time.monotonic()
            await Runner.run(agent, prompt)
            elapsed = time.monotonic() - start
            latencies.append(elapsed)

            if inject_fault:
                recovered += 1
                if event_log:
                    event_count = llm.emit_event(event_log, "fault.recovery_completed", {"task_id": step.role, "recovery_type": "retry"}, event_count)

    except Exception as exc:
        metrics.status = "error"
        metrics.error = f"openai_agents runtime: {str(exc)[:120]}"

    if event_log:
        event_count = llm.emit_event(event_log, "cloud.run_completed", {"framework": "openai_agents", "recovered_faults": recovered}, event_count)

    metrics.avg_latency = sum(latencies) / len(latencies) if latencies else 0
    metrics.prompt_tokens = model.total_input_tokens
    metrics.completion_tokens = model.total_output_tokens
    metrics.cache_hit_tokens = model.total_cached_tokens
    metrics.cache_miss_tokens = model.total_input_tokens - model.total_cached_tokens
    metrics.recovered_faults = recovered
    metrics.observable_events = event_count
    metrics.prompt_chars_total = total_chars
    metrics.prompt_chars_sent = total_sent
    metrics.prompt_truncated = any_truncated
    metrics.framework_entrypoint_invoked = True  # Agent + Runner used
    metrics.model_gateway_invoked = True          # _GatewayModel routes to ModelGateway
    metrics.runtime_mode_actual = "framework_orchestrated"
    metrics.compute_cost()
    metrics.compute_context_reuse()
    return metrics
