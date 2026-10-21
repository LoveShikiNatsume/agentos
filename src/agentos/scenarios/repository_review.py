"""An actual repository review through LangGraph runners and AgentOS Runtime.

A predefined outer plan is adapted to a TaskDAG. Each LangGraph runner owns
plan -> inspect -> answer, using Runtime model and tool entry points. This is
an execution-chain probe, not autonomous patch generation or a model benchmark.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from agentos.adapters.workflow import PlannedTask, adapt_plan
from agentos.context.cache_tracker import CacheHeatTracker
from agentos.context.manager import ContextManager
from agentos.events.types import EventCategory, RuntimeEvent
from agentos.llm.gateway import ModelGateway
from agentos.llm.mock import MockLLMBackend
from agentos.llm.vllm_backend import VLLMBackend
from agentos.runtime.engine import Runtime
from agentos.tools.manager import ToolManager, ToolSpec

POLICY = (
    "Review this repository using only supplied source and actual inspection evidence. "
    "Distinguish behavior implemented in a component from behavior enabled in a scenario. "
    "Do not claim a deployment, performance improvement, test pass, or security boundary "
    "unless the evidence establishes it. Keep observations concise and cite class or method names."
)
SOURCES = {
    "source/context": ("agentos/context/manager.py", "ContextManager", {"prepare_context", "import_agent_context"}),
    "source/execution": ("agentos/core/task.py", "TaskDAG", {"get_ready_tasks", "complete_task", "fail_task"}),
}


def collect_sources(root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    texts, facts = {}, {}
    for ref, (relative, class_name, methods) in SOURCES.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("source escapes repository")
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        members = [node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        selected = [node for node in members if node.name in methods]
        if not selected:
            raise ValueError(f"no inspection methods found in {relative}")
        texts[ref] = relative + "\n" + "\n\n".join(ast.get_source_segment(source, node) or "" for node in selected)
        facts[ref] = {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "class": class_name, "inspected_methods": [node.name for node in selected],
                      "all_methods": [node.name for node in members]}
    return texts, facts


class RecordedBackend:
    def __init__(self, backend):
        self.backend = backend
        self.records = []

    async def generate(self, request):
        response = await self.backend.generate(request)
        self.records.append({
            "task_id": request.task_id, "agent_id": request.agent_id,
            "prefix_id": request.prefix_id, "context_refs": request.context_refs,
            "messages": request.messages, "response": response.content,
            "metrics": response.metrics.to_dict() if response.metrics else None,
            "usage": response.metadata.get("usage"),
        })
        return response


def make_langgraph_runner():
    from langgraph.graph import END, START, StateGraph
    from typing_extensions import TypedDict

    class State(TypedDict):
        ctx: Any
        plan: str
        observations: dict[str, Any]
        answer: str

    def step(ctx, name):
        ctx._emit(RuntimeEvent(
            event_type="runner.step", category=EventCategory.AGENT,
            task_id=ctx.task.task_id, agent_id=ctx.acb.agent_id, trace_id=ctx.acb.trace_id,
            data={"framework": "langgraph", "step": name},
        ))

    async def plan(state):
        ctx = state["ctx"]
        step(ctx, "plan")
        response = await ctx.llm_call([{"role": "user", "content": ctx.task.name + " First, list exactly two checks in at most 80 words."}])
        return {"plan": response.content}

    async def inspect(state):
        ctx = state["ctx"]
        step(ctx, "inspect")
        notifications = []
        if ctx.task.task_id == "report":
            for _ in ctx.task.dependencies:
                message = await ctx.bus.receive(ctx.acb.agent_id)
                if message is None:
                    raise RuntimeError("missing upstream result notification")
                notifications.append(message.content["context_ref"])
        result = await ctx.tool_call("inspect_evidence", refs=ctx.task.context_refs)
        if not result.ok:
            raise RuntimeError(result.error)
        return {"observations": {"inspection": result.output, "notifications": notifications}}

    async def answer(state):
        ctx = state["ctx"]
        step(ctx, "answer")
        response = await ctx.llm_call([
            {"role": "user", "content": ctx.task.name + " First, list exactly two checks in at most 80 words."},
            {"role": "assistant", "content": state["plan"]},
            {"role": "user", "content": "Actual inspection: " + json.dumps(state["observations"], ensure_ascii=False)
             + "\nNow report the supported behavior and one remaining limitation in at most 150 words."},
        ])
        return {"answer": response.content}

    graph = StateGraph(State)
    graph.add_node("plan", plan)
    graph.add_node("inspect", inspect)
    graph.add_node("answer", answer)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "inspect")
    graph.add_edge("inspect", "answer")
    graph.add_edge("answer", END)
    compiled = graph.compile()

    async def run(ctx):
        result = await compiled.ainvoke({"ctx": ctx})
        return result["answer"]
    return run


async def run_repository_review(
    output: Path, *, source_root: Path, backend_name: str = "mock",
    base_url: str = "http://127.0.0.1:18094", model: str = "qwen3.8-27b-int4",
) -> dict[str, Any]:
    if backend_name not in {"mock", "vllm"}:
        raise ValueError("only mock or local vllm is supported")
    if urlsplit(base_url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("this verification uses a loopback model endpoint")
    texts, facts = collect_sources(source_root)
    cm = ContextManager(prefix_order="name")
    cm.add_shared("00/policy", POLICY)
    for ref, text in texts.items():
        cm.add_shared(ref, text)
    backend = (MockLLMBackend(responses={
        "agent-context": "Context review from the supplied source.",
        "agent-execution": "Execution review from the supplied source.",
        "agent-report": "Combined repository review; runtime wiring needs separate evidence.",
    }, latency=0) if backend_name == "mock" else VLLMBackend(
        base_url=base_url, model=model, timeout=240, max_output_tokens=512,
        chat_template_kwargs={"enable_thinking": False},
    ))
    recorder = RecordedBackend(backend)
    tracker = CacheHeatTracker()
    gateway = ModelGateway({"local": recorder}, default_backend="local", cache_tracker=tracker)
    tools = ToolManager()

    def inspect_evidence(refs):
        result = {}
        for ref in refs:
            if ref in facts:
                result[ref] = facts[ref]
            elif ref.startswith("result/"):
                context_ref = cm.get_ref(ref)
                if context_ref is None:
                    raise ValueError(f"upstream output not published: {ref}")
                result[ref] = {"content_hash": context_ref.content_hash, "published": True}
        return result

    tools.register(ToolSpec("inspect_evidence", inspect_evidence, timeout_seconds=5))
    plan = [
        PlannedTask("context", "Explain task-specific context selection and the scope of context retry.", "review",
                    context_refs=("00/policy", "source/context"), output_ref="result/context",
                    allowed_tools=("inspect_evidence",), metadata={"notify_agent": "agent-report"}),
        PlannedTask("execution", "Explain when a dependent task becomes ready and how failure is represented.", "review",
                    context_refs=("00/policy", "source/execution"), output_ref="result/execution",
                    allowed_tools=("inspect_evidence",), metadata={"notify_agent": "agent-report"}),
        PlannedTask("report", "Combine the two reviews and identify what is still needed to prove the full execution chain.", "review",
                    dependencies=("context", "execution"), context_refs=("00/policy", "result/context", "result/execution"),
                    output_ref="result/report", allowed_tools=("inspect_evidence",)),
    ]
    dag, handlers = adapt_plan(plan, {"review": make_langgraph_runner()}, framework="langgraph")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rt = Runtime(gateway, handlers, context_manager=cm, tool_manager=tools, event_log=output / "events.jsonl",
                 max_batch_size=2, max_llm_batch_size=2, llm_batch_wait_ms=5,
                 failure_policy="isolate", enable_kernel_probe=False)
    try:
        results = await rt.run(dag)
    finally:
        (output / "requests.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in recorder.records), encoding="utf-8")
    counts = Counter(e.event_type for e in rt.events)
    checks = {
        "three_runtime_tasks_completed": len(results) == 3,
        "real_langgraph_steps_observed": counts["runner.step"] == 9,
        "tools_ran_through_runtime": counts["tool.call_finished"] == 3,
        "all_six_calls_used_gateway": gateway.stats()["total_calls"] == len(recorder.records) == 6,
        "all_requests_selected_context": counts["context.assembled"] == 6 and all(item["context_refs"] for item in recorder.records),
        "only_successful_results_published": counts["context.result_published"] == 3,
        "upstream_refs_notified": counts["message.reference_sent"] == 2,
        "report_received_results_without_source": all(
            "result/context" in item["context_refs"] and not any(ref.startswith("source/") for ref in item["context_refs"])
            for item in recorder.records if item["task_id"] == "report"
        ),
    }
    if backend_name == "vllm":
        checks["all_model_responses_complete"] = all(
            (item["metrics"] or {}).get("metadata", {}).get("finish_reason") == "stop"
            for item in recorder.records
        )
    summary = {
        "schema": "agentos_execution_chain.v1", "passed": all(checks.values()), "checks": checks,
        "backend": backend_name, "model": model if backend_name == "vllm" else "mock",
        "framework": "langgraph", "plan_source": "predefined_application_plan",
        "source_files": facts, "task_statuses": {k:t.status.value for k,t in dag.tasks.items()},
        "gateway": gateway.stats(), "cache_feedback_samples": sum(e.total_calls for e in tracker._entries.values()),
        "events": dict(counts), "answers": results,
        "model_quality_independently_scored": False, "kernel_isolation_verified": False,
        "execution_mode": "coroutine", "network_inference": "local_only",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--backend", choices=["mock", "vllm"], default="mock")
    parser.add_argument("--base-url", default="http://127.0.0.1:18094")
    parser.add_argument("--model", default="qwen3.8-27b-int4")
    args = parser.parse_args()
    result = asyncio.run(run_repository_review(args.output, source_root=args.source_root,
                         backend_name=args.backend, base_url=args.base_url, model=args.model))
    print(json.dumps({"passed":result["passed"], "checks":result["checks"], "gateway":result["gateway"]}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
