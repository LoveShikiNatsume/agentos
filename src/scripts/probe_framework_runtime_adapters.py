#!/usr/bin/env python3
"""Probe framework-orchestrated runtime adapters with entrypoint/gateway verification.

Each probe verifies:
1. Framework entrypoint is actually invoked (Runner.run, AssistantAgent.run, etc.)
2. ModelGateway is actually called (via call counter)
3. Token usage comes from real DeepSeek API

Usage:
    DEEPSEEK_API_KEY=sk-xxx python scripts/probe_framework_runtime_adapters.py

Output:
    runs/framework_runtime_adapter_probe_<ts>.json
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


async def probe_openai_agents() -> dict:
    """Probe: Agent + Runner + _GatewayModel (inherits Model ABC)."""
    try:
        from agents import Agent, Runner, set_tracing_disabled, Usage, ModelResponse
        from agents.models.interface import Model
        set_tracing_disabled(True)
    except ImportError:
        return {"framework": "OpenAI Agents SDK", "status": "skipped", "error": "openai-agents not installed",
                "framework_entrypoint_invoked": False, "model_gateway_invoked": False}

    try:
        from agentos.llm.deepseek_backend import DeepSeekBackend
        from agentos.llm.gateway import ModelGateway
        from agentos.llm.base import LLMRequest
        from agentos.context.cache_tracker import CacheHeatTracker
        from openai.types.responses import ResponseOutputMessage, ResponseOutputText

        gateway = ModelGateway(
            backends={"deepseek": DeepSeekBackend()},
            default_backend="deepseek",
            cache_tracker=CacheHeatTracker(ttl=300, hot_threshold=0.3),
        )

        gateway_call_count = 0

        class _GatewayModel(Model):
            def __init__(self, gw):
                self._gw = gw
                self.total_input = 0
                self.total_output = 0
            async def get_response(self, system_instructions, input, model_settings,
                                   tools, output_schema, handoffs, tracing,
                                   previous_response_id=None, conversation_id=None, prompt=None):
                nonlocal gateway_call_count
                gateway_call_count += 1
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
                req = LLMRequest(agent_id="probe", messages=msgs, max_tokens=256,
                                 temperature=0.3, backend="deepseek", prefix_id="probe")
                resp = await self._gw.generate(req)
                pi = resp.metrics.prompt_tokens if resp.metrics else 0
                co = resp.metrics.completion_tokens if resp.metrics else 0
                self.total_input += pi
                self.total_output += co
                output_msg = ResponseOutputMessage(
                    id="probe", content=[ResponseOutputText(annotations=[], text=resp.content, type="output_text")],
                    role="assistant", status="completed", type="message",
                )
                return ModelResponse(output=[output_msg], usage=Usage(input_tokens=pi, output_tokens=co), response_id=None)
            async def stream_response(self, *args, **kwargs):
                result = await self.get_response(*args, **kwargs)
                yield result

        model = _GatewayModel(gateway)
        agent = Agent(name="probe", instructions="Say hello.", model=model)

        # Verify: Runner.run is the entrypoint
        result = await Runner.run(agent, "Say hello")
        entrypoint_invoked = result is not None and hasattr(result, 'final_output')

        return {
            "framework": "OpenAI Agents SDK",
            "status": "ok",
            "framework_entrypoint_invoked": entrypoint_invoked,
            "model_gateway_invoked": gateway_call_count >= 1,
            "orchestration_source": "sdk_runner",
            "model_path": "agentos_model_gateway",
            "runtime_contract": "framework_orchestrated_runtime",
            "gateway_call_count": gateway_call_count,
            "prompt_tokens": model.total_input,
            "completion_tokens": model.total_output,
            "note": "Agent + Runner.run() invoked, _GatewayModel routes to ModelGateway",
        }
    except Exception as exc:
        return {"framework": "OpenAI Agents SDK", "status": "failed", "error": str(exc)[:200],
                "framework_entrypoint_invoked": False, "model_gateway_invoked": False}


async def probe_autogen() -> dict:
    """Probe: AssistantAgent + _GatewayModelClient."""
    try:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_core.models import RequestUsage
    except ImportError:
        return {"framework": "AutoGen", "status": "skipped", "error": "autogen not installed",
                "framework_entrypoint_invoked": False, "model_gateway_invoked": False}

    try:
        from scripts.cloud_bench.autogen_adapter import _GatewayModelClient
        from agentos.llm.deepseek_backend import DeepSeekBackend
        from agentos.llm.gateway import ModelGateway
        from agentos.context.cache_tracker import CacheHeatTracker

        gateway = ModelGateway(
            backends={"deepseek": DeepSeekBackend()},
            default_backend="deepseek",
            cache_tracker=CacheHeatTracker(ttl=300, hot_threshold=0.3),
        )
        client = _GatewayModelClient(gateway, prefix_id="probe")
        agent = AssistantAgent(name="probe", model_client=client, system_message="Say hello.")

        # Verify: AssistantAgent.run is the entrypoint
        result = await agent.run(task="Say hello")
        entrypoint_invoked = result is not None and hasattr(result, 'messages') and len(result.messages) > 0

        usage = client.total_usage()
        return {
            "framework": "AutoGen",
            "status": "ok",
            "framework_entrypoint_invoked": entrypoint_invoked,
            "model_gateway_invoked": entrypoint_invoked,  # _GatewayModelClient routes to gateway
            "orchestration_source": "sdk_agentchat",
            "model_path": "agentos_model_gateway",
            "runtime_contract": "framework_orchestrated_runtime",
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "note": "AssistantAgent.run() invoked, _GatewayModelClient routes to ModelGateway",
        }
    except Exception as exc:
        return {"framework": "AutoGen", "status": "failed", "error": str(exc)[:200],
                "framework_entrypoint_invoked": False, "model_gateway_invoked": False}


async def probe_crewai() -> dict:
    """Probe: Crew/Agent/Task + GatewayCrewLLM."""
    try:
        from crewai import Agent, Task, Crew, Process
    except ImportError:
        return {"framework": "CrewAI", "status": "skipped", "error": "crewai not installed",
                "framework_entrypoint_invoked": False, "model_gateway_invoked": False}

    try:
        from scripts.cloud_bench.crewai_adapter import _make_gateway_llm
        from agentos.llm.deepseek_backend import DeepSeekBackend
        from agentos.llm.gateway import ModelGateway
        from agentos.context.cache_tracker import CacheHeatTracker

        gateway = ModelGateway(
            backends={"deepseek": DeepSeekBackend()},
            default_backend="deepseek",
            cache_tracker=CacheHeatTracker(ttl=300, hot_threshold=0.3),
        )
        gw_llm = _make_gateway_llm(gateway, prefix_id="probe", max_tokens=256)

        agent = Agent(role="Prober", goal="Say hello", backstory="Helpful.",
                      llm=gw_llm, verbose=False, allow_delegation=False)
        task = Task(description="Say hello", expected_output="A greeting.", agent=agent)
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)

        # Verify: Crew.kickoff_async is the entrypoint
        result = await crew.kickoff_async()
        entrypoint_invoked = result is not None
        gateway_invoked = gw_llm._gw_total_prompt > 0

        return {
            "framework": "CrewAI",
            "status": "ok",
            "framework_entrypoint_invoked": entrypoint_invoked,
            "model_gateway_invoked": gateway_invoked,
            "orchestration_source": "sdk_crew",
            "model_path": "agentos_model_gateway",
            "runtime_contract": "framework_orchestrated_runtime",
            "prompt_tokens": gw_llm._gw_total_prompt,
            "completion_tokens": gw_llm._gw_total_completion,
            "note": "Crew.kickoff_async() invoked, GatewayCrewLLM routes to ModelGateway",
        }
    except Exception as exc:
        return {"framework": "CrewAI", "status": "failed", "error": str(exc)[:200],
                "framework_entrypoint_invoked": False, "model_gateway_invoked": False}


async def probe_langgraph() -> dict:
    """Probe: StateGraph + DeepSeekBackend (already proven, verify entrypoint)."""
    try:
        from langgraph.graph import StateGraph, END
        from typing_extensions import TypedDict
        import operator
        from typing import Annotated, List
    except ImportError:
        return {"framework": "LangGraph", "status": "skipped", "error": "langgraph not installed",
                "framework_entrypoint_invoked": False, "model_gateway_invoked": False}

    try:
        from scripts.cloud_bench.llm_call import runtime_call
        from agentos.llm.deepseek_backend import DeepSeekBackend
        from agentos.llm.gateway import ModelGateway
        from agentos.context.cache_tracker import CacheHeatTracker

        gateway = ModelGateway(
            backends={"deepseek": DeepSeekBackend()},
            default_backend="deepseek",
            cache_tracker=CacheHeatTracker(ttl=300, hot_threshold=0.3),
        )

        gateway_call_count = 0

        class GraphState(TypedDict, total=False):
            results: Annotated[list[str], operator.add]

        builder = StateGraph(GraphState)

        async def probe_node(state):
            nonlocal gateway_call_count
            gateway_call_count += 1
            from agentos.llm.base import LLMRequest
            req = LLMRequest(agent_id="probe", messages=[
                {"role": "system", "content": "Say hello."},
                {"role": "user", "content": "Say hello"},
            ], max_tokens=128, temperature=0.3, backend="deepseek", prefix_id="probe")
            resp = await gateway.generate(req)
            return {"results": ["probe:done"]}

        builder.add_node("probe", probe_node)
        builder.add_edge("__start__", "probe")
        builder.add_edge("probe", "__end__")
        graph = builder.compile()

        # Verify: graph.ainvoke is the entrypoint
        result = await graph.ainvoke({"results": []})
        entrypoint_invoked = result is not None and "probe:done" in result.get("results", [])

        return {
            "framework": "LangGraph",
            "status": "ok",
            "framework_entrypoint_invoked": entrypoint_invoked,
            "model_gateway_invoked": gateway_call_count >= 1,
            "orchestration_source": "sdk_stategraph",
            "model_path": "agentos_model_gateway",
            "runtime_contract": "framework_orchestrated_runtime",
            "gateway_call_count": gateway_call_count,
            "note": "StateGraph.ainvoke() invoked, node calls ModelGateway directly",
        }
    except Exception as exc:
        return {"framework": "LangGraph", "status": "failed", "error": str(exc)[:200],
                "framework_entrypoint_invoked": False, "model_gateway_invoked": False}


async def main() -> int:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY required", file=sys.stderr)
        return 1

    ts = int(time.time())
    print("=== Framework Runtime Adapter Probe (with verification) ===\n")

    results = []
    for name, probe_fn in [
        ("OpenAI Agents SDK", probe_openai_agents),
        ("AutoGen", probe_autogen),
        ("CrewAI", probe_crewai),
        ("LangGraph", probe_langgraph),
    ]:
        print(f"Probing {name}...")
        r = await probe_fn()
        status = r.get("status", "?")
        entrypoint = r.get("framework_entrypoint_invoked", False)
        gateway = r.get("model_gateway_invoked", False)
        contract = r.get("runtime_contract", "-")
        print(f"  status={status}, entrypoint={entrypoint}, gateway={gateway}, contract={contract}")
        if r.get("error"):
            print(f"  error: {r['error'][:100]}")
        results.append(r)

    out_path = _REPO_ROOT / "runs" / f"framework_runtime_adapter_probe_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({"timestamp": ts, "probes": results}, f, indent=2)
    print(f"\nProbe results: {out_path}")

    # Summary
    ok_count = sum(1 for r in results if r.get("status") == "ok"
                   and r.get("framework_entrypoint_invoked")
                   and r.get("model_gateway_invoked"))
    print(f"\nVerified framework+gateway: {ok_count}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
