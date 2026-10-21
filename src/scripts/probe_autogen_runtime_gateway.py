#!/usr/bin/env python3
"""Feasibility probe: AutoGen AssistantAgent + AgentOS ModelGateway.

Tests whether we can create a custom AutoGen ChatCompletionClient
that routes LLM calls through AgentOS ModelGateway, allowing
AssistantAgent orchestration while using the shared prefix cache.

Usage:
    DEEPSEEK_API_KEY=sk-xxx python scripts/probe_autogen_runtime_gateway.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from autogen_core.models import RequestUsage


class GatewayModelClient:
    """AutoGen ChatCompletionClient wrapper around AgentOS ModelGateway.

    Implements the abstract methods of ChatCompletionClient that AssistantAgent needs.
    Routes LLM calls through ModelGateway for shared prefix cache and event tracking.
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
        from autogen_core.models import CreateResult, RequestUsage
        from autogen_core import FunctionCall

        # Convert AutoGen LLMMessage list to OpenAI messages format
        openai_messages = []
        for msg in messages:
            if hasattr(msg, 'content') and hasattr(msg, 'source'):
                # It's an assistant/user message
                role = 'assistant' if msg.source != 'user' else 'user'
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                openai_messages.append({'role': role, 'content': content})
            elif hasattr(msg, 'content'):
                openai_messages.append({'role': 'user', 'content': str(msg.content)})
            else:
                openai_messages.append({'role': 'user', 'content': str(msg)})

        # Use ModelGateway
        from agentos.llm.base import LLMRequest
        req = LLMRequest(
            agent_id="autogen_probe",
            messages=openai_messages,
            max_tokens=extra_create_args.get('max_tokens', 512) if extra_create_args else 512,
            temperature=extra_create_args.get('temperature', 0.3) if extra_create_args else 0.3,
            backend="deepseek",
            prefix_id=self._prefix_id,
        )

        resp = await self._gateway.generate(req)

        # Extract usage
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

    def actual_usage(self) -> RequestUsage:
        return self._actual_usage

    def total_usage(self) -> RequestUsage:
        return self._total_usage

    def count_tokens(self, messages, *, tools=None) -> int:
        # Rough estimate
        total = 0
        for msg in messages:
            if hasattr(msg, 'content'):
                total += len(str(msg.content)) // 4
        return total

    def remaining_tokens(self, messages, *, tools=None) -> int:
        return 128000 - self.count_tokens(messages, tools=tools)

    async def create_stream(self, messages, *, tools=None, tool_choice='auto',
                            json_output=None, extra_create_args=None, cancellation_token=None):
        """Stream implementation - just yield the full result."""
        result = await self.create(messages, tools=tools, tool_choice=tool_choice,
                                   json_output=json_output, extra_create_args=extra_create_args,
                                   cancellation_token=cancellation_token)
        yield result.content
        yield result

    async def close(self):
        pass

    def dump_component(self):
        from autogen_core import ComponentModel
        return ComponentModel(type="GatewayModelClient", config={})


async def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY required", file=sys.stderr)
        return 1

    # Set up ModelGateway
    from agentos.llm.deepseek_backend import DeepSeekBackend
    from agentos.llm.gateway import ModelGateway
    from agentos.context.cache_tracker import CacheHeatTracker

    tracker = CacheHeatTracker(ttl=300, hot_threshold=0.3)
    gateway = ModelGateway(
        backends={"deepseek": DeepSeekBackend()},
        default_backend="deepseek",
        cache_tracker=tracker,
    )

    # Create AutoGen AssistantAgent with gateway wrapper
    from autogen_agentchat.agents import AssistantAgent
    from autogen_core.models import ModelInfo

    client = GatewayModelClient(gateway, prefix_id="wishlist-shared-spec")

    agent = AssistantAgent(
        name="test_agent",
        model_client=client,
        system_message="You are a senior software engineer. Be concise.",
    )

    print("=== Feasibility Probe: AutoGen + ModelGateway ===")
    print()

    # Test 1: simple call
    print("[1] Single agent call...")
    start = time.monotonic()
    result = await agent.run(task="Design a wishlist API in one paragraph")
    elapsed = time.monotonic() - start
    print(f"  Output: {str(result)[:100]}...")
    print(f"  Latency: {elapsed:.2f}s")
    print(f"  Usage: {client.total_usage()}")
    print()

    # Test 2: second call (same prefix, should have cache)
    print("[2] Second call (same prefix)...")
    start = time.monotonic()
    result2 = await agent.run(task="Now design the database schema")
    elapsed2 = time.monotonic() - start
    print(f"  Output: {str(result2)[:100]}...")
    print(f"  Latency: {elapsed2:.2f}s")
    print(f"  Total usage: {client.total_usage()}")
    print()

    # Test 3: multi-turn conversation
    print("[3] Multi-turn conversation...")
    agent2 = AssistantAgent(
        name="architect",
        model_client=client,
        system_message="You are a system architect. Be concise.",
    )
    result3 = await agent2.run(task="Review the wishlist API design and suggest improvements")
    print(f"  Output: {str(result3)[:100]}...")
    print(f"  Total usage: {client.total_usage()}")
    print()

    print("=== Probe complete ===")
    print(f"Total tokens: {client.total_usage().prompt_tokens} prompt + {client.total_usage().completion_tokens} completion")
    print("SUCCESS: AutoGen AssistantAgent works with ModelGateway wrapper")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
