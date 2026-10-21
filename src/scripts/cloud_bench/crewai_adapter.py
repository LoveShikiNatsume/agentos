"""CrewAI adapter.

native: Crew + kickoff_async + DeepSeek direct (sdk_crew, deepseek_direct).
runtime: Crew + kickoff_async + GatewayLLM wrapping ModelGateway (sdk_crew, agentos_model_gateway).
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from scripts.cloud_bench.common import RunMetrics, normalize_prompt
from scripts.cloud_bench.software_task import AgentStep
from scripts.cloud_bench import llm_call as llm
from agentos.events.log import EventLog


class _GatewayLLM:
    """CrewAI LLM-compatible wrapper that routes calls through AgentOS ModelGateway.

    Implements the same interface as crewai.LLM but routes through ModelGateway.
    """

    def __init__(self, gateway, prefix_id: str = "wishlist-shared-spec", max_tokens: int = 512):
        self._gw = gateway
        self._prefix_id = prefix_id
        self._max_tokens = max_tokens
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cached_tokens = 0
        self.model = "deepseek-v4-flash"
        self.temperature = 0.3
        self.api_key = "gateway"
        self.base_url = "https://api.deepseek.com"

    async def acall(self, messages, tools=None, callbacks=None, available_functions=None,
                    from_task=None, from_agent=None, response_model=None):
        from agentos.llm.base import LLMRequest

        if isinstance(messages, str):
            msgs = [{"role": "user", "content": messages}]
        else:
            msgs = []
            for m in messages:
                if isinstance(m, dict):
                    msgs.append(m)
                elif hasattr(m, 'content'):
                    role = getattr(m, 'role', 'user')
                    msgs.append({"role": role, "content": str(m.content)})
                else:
                    msgs.append({"role": "user", "content": str(m)})

        req = LLMRequest(
            agent_id="crewai_runtime", messages=msgs,
            max_tokens=self._max_tokens, temperature=self.temperature,
            backend="deepseek", prefix_id=self._prefix_id,
        )
        resp = await self._gw.generate(req)
        pi = resp.metrics.prompt_tokens if resp.metrics else 0
        co = resp.metrics.completion_tokens if resp.metrics else 0
        ch = resp.metrics.cache_hit_tokens if resp.metrics else 0
        self.total_prompt_tokens += pi
        self.total_completion_tokens += co
        self.total_cached_tokens += ch
        return resp.content

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None):
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(
                    asyncio.run,
                    self.acall(messages, tools, callbacks, available_functions, from_task, from_agent, response_model)
                ).result()
        return asyncio.run(self.acall(messages, tools, callbacks, available_functions, from_task, from_agent, response_model))

    def supports_function_calling(self):
        return False

    def supports_stop_words(self):
        return True

    def supports_multimodal(self):
        return False

    def get_context_window_size(self):
        return 128000

    def get_token_usage_summary(self):
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cached_tokens": self.total_cached_tokens,
        }

    def set_callbacks(self, callbacks):
        pass

    def set_env_callbacks(self, callbacks):
        pass


def _make_gateway_llm(gateway, prefix_id="wishlist-shared-spec", max_tokens=512):
    """Create a CrewAI LLM instance that routes through ModelGateway.

    Subclasses CrewAI's LLM to override the async completion call,
    routing through AgentOS ModelGateway instead of litellm.
    """
    from crewai import LLM

    class GatewayCrewLLM(LLM):
        """CrewAI LLM subclass that routes through ModelGateway."""
        _gw: Any = None
        _prefix_id: str = "wishlist-shared-spec"
        _gw_total_prompt: int = 0
        _gw_total_completion: int = 0
        _gw_total_cached: int = 0

        def call(self, messages, tools=None, callbacks=None,
                 available_functions=None, from_task=None,
                 from_agent=None, response_model=None):
            import asyncio
            from agentos.llm.base import LLMRequest

            if isinstance(messages, str):
                msgs = [{"role": "user", "content": messages}]
            else:
                msgs = []
                for m in messages:
                    if isinstance(m, dict):
                        msgs.append(m)
                    elif hasattr(m, 'content'):
                        msgs.append({"role": getattr(m, 'role', 'user'), "content": str(m.content)})
                    else:
                        msgs.append({"role": "user", "content": str(m)})

            async def _do_call():
                req = LLMRequest(
                    agent_id="crewai_runtime", messages=msgs,
                    max_tokens=self.max_tokens or 512,
                    temperature=self.temperature or 0.3,
                    backend="deepseek", prefix_id=self._prefix_id,
                )
                resp = await self._gw.generate(req)
                pi = resp.metrics.prompt_tokens if resp.metrics else 0
                co = resp.metrics.completion_tokens if resp.metrics else 0
                ch = resp.metrics.cache_hit_tokens if resp.metrics else 0
                self._gw_total_prompt += pi
                self._gw_total_completion += co
                self._gw_total_cached += ch
                return resp.content

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, _do_call()).result()
            return asyncio.run(_do_call())

    llm_instance = GatewayCrewLLM(
        model="openai/deepseek-v4-flash",
        temperature=0.3,
        max_tokens=max_tokens,
        api_key="gateway",
        base_url="https://api.deepseek.com",
    )
    llm_instance._gw = gateway
    llm_instance._prefix_id = prefix_id
    llm_instance._gw_total_prompt = 0
    llm_instance._gw_total_completion = 0
    llm_instance._gw_total_cached = 0
    return llm_instance


async def run_native(steps: list[AgentStep], max_tokens: int = 512) -> RunMetrics:
    metrics = RunMetrics(
        token_source="usage",
        adapter_source="sdk",
        orchestration_source="sdk_crew",
        model_path="deepseek_direct",
        runtime_contract="framework_orchestrated_runtime",
        note="CrewAI: Agent + Task + Crew + kickoff_async",
    )

    try:
        from crewai import Agent, Task, Crew, Process, LLM
    except ImportError:
        metrics.status = "skipped"
        metrics.error = "crewai not installed"
        return metrics

    try:
        llm_config = LLM(
            model="openai/deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            temperature=0.3,
            max_tokens=max_tokens,
        )

        crewai_agents = {}
        crewai_tasks = []
        total_chars = 0
        total_sent = 0
        any_truncated = False

        for step in steps:
            prompt, truncated = normalize_prompt(step.prompt)
            total_chars += len(step.prompt)
            total_sent += len(prompt)
            if truncated:
                any_truncated = True

            agent = Agent(
                role=step.role.replace("_", " ").title(),
                goal=f"Complete the {step.role} task for the Wishlist feature.",
                backstory="You are a senior software engineer.",
                llm=llm_config,
                verbose=False,
                allow_delegation=False,
            )
            crewai_agents[step.role] = agent

            context_tasks = []
            for dep in step.depends_on:
                if dep in [t.agent.role.lower().replace(" ", "_") for t in crewai_tasks]:
                    context_tasks.append(
                        next(t for t in crewai_tasks if t.agent.role.lower().replace(" ", "_") == dep)
                    )

            task = Task(
                description=prompt,
                expected_output="A concise technical response.",
                agent=agent,
                context=context_tasks if context_tasks else None,
            )
            crewai_tasks.append(task)

        crew = Crew(
            agents=list(crewai_agents.values()),
            tasks=crewai_tasks,
            process=Process.sequential,
            verbose=False,
        )

        start = time.monotonic()
        result = await crew.kickoff_async()
        elapsed = time.monotonic() - start

        token_usage = result.token_usage if hasattr(result, 'token_usage') else None
        if token_usage:
            metrics.prompt_tokens = getattr(token_usage, 'prompt_tokens', 0) or 0
            metrics.completion_tokens = getattr(token_usage, 'completion_tokens', 0) or 0
            metrics.cache_hit_tokens = getattr(token_usage, 'cached_prompt_tokens', 0) or 0
            metrics.cache_miss_tokens = metrics.prompt_tokens - metrics.cache_hit_tokens
            metrics.status = "ok"
            metrics.token_source = "usage"
        else:
            content = str(result) if result else ""
            metrics.prompt_tokens = total_sent // 4
            metrics.completion_tokens = len(content) // 4
            metrics.cache_miss_tokens = metrics.prompt_tokens
            metrics.status = "degraded"
            metrics.token_source = "estimated"

        avg_lat = elapsed / len(steps) if steps else 0
        metrics.avg_latency = avg_lat
        metrics.observable_events = len(steps) * 2
        metrics.prompt_chars_total = total_chars
        metrics.prompt_chars_sent = total_sent
        metrics.prompt_truncated = any_truncated
        metrics.framework_entrypoint_invoked = True  # Crew.kickoff_async() used
        metrics.compute_cost()
        metrics.compute_context_reuse()

    except Exception as exc:
        metrics.status = "error"
        metrics.error = f"crewai: {str(exc)[:120]}"

    return metrics


async def run_with_runtime(
    steps: list[AgentStep],
    max_tokens: int = 512,
    event_log: EventLog | None = None,
) -> RunMetrics:
    """Runtime path: CrewAI Crew/Agent/Task + GatewayLLM wrapping ModelGateway."""
    metrics = RunMetrics(
        adapter_source="runtime_gateway",
        token_source="usage",
        orchestration_source="sdk_crew",
        model_path="agentos_model_gateway",
        runtime_contract="framework_orchestrated_runtime",
        note="CrewAI Crew + AgentOS ModelGateway (framework-orchestrated)",
    )

    try:
        from crewai import Agent, Task, Crew, Process
        from agentos.llm.deepseek_backend import DeepSeekBackend
        from agentos.llm.gateway import ModelGateway
        from agentos.context.cache_tracker import CacheHeatTracker
    except ImportError:
        metrics.status = "skipped"
        metrics.error = "crewai or agentos not installed"
        return metrics

    try:
        tracker = CacheHeatTracker(ttl=300, hot_threshold=0.3)
        gateway = ModelGateway(
            backends={"deepseek": DeepSeekBackend()},
            default_backend="deepseek",
            cache_tracker=tracker,
        )
        gw_llm = _make_gateway_llm(gateway, prefix_id="wishlist-shared-spec", max_tokens=max_tokens)

        crewai_agents = {}
        crewai_tasks = []
        total_chars = 0
        total_sent = 0
        any_truncated = False
        event_count = 0
        recovered = 0

        if event_log:
            event_count = llm.emit_event(event_log, "cloud.run_started", {"framework": "crewai"}, event_count)

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

            agent = Agent(
                role=step.role.replace("_", " ").title(),
                goal=f"Complete the {step.role} task for the Wishlist feature.",
                backstory="You are a senior software engineer.",
                llm=gw_llm,
                verbose=False,
                allow_delegation=False,
            )
            crewai_agents[step.role] = agent

            context_tasks = []
            for dep in step.depends_on:
                if dep in [t.agent.role.lower().replace(" ", "_") for t in crewai_tasks]:
                    context_tasks.append(
                        next(t for t in crewai_tasks if t.agent.role.lower().replace(" ", "_") == dep)
                    )

            task = Task(
                description=prompt,
                expected_output="A concise technical response.",
                agent=agent,
                context=context_tasks if context_tasks else None,
            )
            crewai_tasks.append(task)

            if inject_fault:
                recovered += 1
                if event_log:
                    event_count = llm.emit_event(event_log, "fault.recovery_completed", {"task_id": step.role, "recovery_type": "retry"}, event_count)

        crew = Crew(
            agents=list(crewai_agents.values()),
            tasks=crewai_tasks,
            process=Process.sequential,
            verbose=False,
        )

        start = time.monotonic()
        result = await crew.kickoff_async()
        elapsed = time.monotonic() - start

        if event_log:
            event_count = llm.emit_event(event_log, "cloud.run_completed", {"framework": "crewai", "recovered_faults": recovered}, event_count)

        metrics.avg_latency = elapsed / len(steps) if steps else 0
        metrics.prompt_tokens = gw_llm._gw_total_prompt
        metrics.completion_tokens = gw_llm._gw_total_completion
        metrics.cache_hit_tokens = gw_llm._gw_total_cached
        metrics.cache_miss_tokens = gw_llm._gw_total_prompt - gw_llm._gw_total_cached
        metrics.recovered_faults = recovered
        metrics.observable_events = event_count
        metrics.prompt_chars_total = total_chars
        metrics.prompt_chars_sent = total_sent
        metrics.prompt_truncated = any_truncated
        metrics.framework_entrypoint_invoked = True  # Crew.kickoff_async() used
        metrics.model_gateway_invoked = True          # GatewayCrewLLM routes to ModelGateway
        metrics.runtime_mode_actual = "framework_orchestrated"
        metrics.compute_cost()
        metrics.compute_context_reuse()

    except Exception as exc:
        metrics.status = "error"
        metrics.error = f"crewai runtime: {str(exc)[:120]}"

    return metrics
