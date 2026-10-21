from pathlib import Path

import pytest

from agentos.adapters.workflow import PlannedTask, adapt_plan
from agentos.context.manager import ContextManager
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import Runtime, WorkflowExecutionError


async def test_adapter_retains_dependency_and_result_publication_contract():
    cm = ContextManager()
    cm.add_shared("rules", "shared rules")

    async def produce(ctx):
        return "completed upstream result"

    async def consume(ctx):
        assert ctx.context_manager.read(ctx.acb.agent_id, "result/first") == "completed upstream result"
        notification = await ctx.bus.receive(ctx.acb.agent_id)
        assert notification.content == {"context_ref": "result/first"}
        return "consumed"

    plan = [
        PlannedTask("first", "produce", "producer", context_refs=("rules",), output_ref="result/first",
                    metadata={"notify_agent": "agent-second"}),
        PlannedTask("second", "consume", "consumer", dependencies=("first",), context_refs=("result/first",)),
    ]
    dag, handlers = adapt_plan(plan, {"producer": produce, "consumer": consume}, framework="test")
    rt = Runtime(MockLLMBackend(), handlers, context_manager=cm, enable_kernel_probe=False)
    assert await rt.run(dag) == {"first": "completed upstream result", "second": "consumed"}
    assert dag.tasks["second"].dependencies == ["first"]
    events = [e.event_type for e in rt.events]
    assert events.count("adapter.runner_completed") == 2
    assert events.count("context.result_published") == 1


async def test_failed_runner_does_not_publish_a_final_result():
    cm = ContextManager()
    async def fail(ctx):
        ctx.context_manager.add_private(ctx.acb.agent_id, "partial scratch")
        raise RuntimeError("runner failed before final result")
    dag, handlers = adapt_plan([PlannedTask("bad", "fail", "runner", output_ref="result/bad")],
                               {"runner": fail}, framework="test")
    rt = Runtime(MockLLMBackend(), handlers, context_manager=cm, failure_policy="isolate", enable_kernel_probe=False)
    with pytest.raises(WorkflowExecutionError):
        await rt.run(dag)
    assert cm.get_ref("result/bad") is None
    assert not any(e.event_type == "context.result_published" for e in rt.events)


def test_plan_rejects_unbound_runners_and_ambiguous_outputs():
    with pytest.raises(ValueError, match="unregistered runner"):
        adapt_plan([PlannedTask("a", "a", "missing")], {}, framework="test")
    with pytest.raises(ValueError, match="same output_ref"):
        adapt_plan([PlannedTask("a", "a", "r", output_ref="same"),
                    PlannedTask("b", "b", "r", output_ref="same")], {}, framework="test")


async def test_langgraph_probe_traverses_runtime_tools_context_gateway_and_results(tmp_path):
    pytest.importorskip("langgraph")
    from agentos.scenarios.repository_review import run_repository_review
    result = await run_repository_review(tmp_path / "review", source_root=(Path(__file__).resolve().parents[1] / "src"))
    assert result["passed"] is True, result["checks"]
    assert result["framework"] == "langgraph"
    assert result["gateway"]["total_calls"] == 6
    assert result["cache_feedback_samples"] == 6
    assert result["events"]["runner.step"] == 9
    assert result["events"]["tool.call_finished"] == 3
    assert result["kernel_isolation_verified"] is False
    assert result["model_quality_independently_scored"] is False


async def test_completed_workflow_does_not_hide_truncated_model_responses(tmp_path, monkeypatch):
    pytest.importorskip("langgraph")
    import agentos.scenarios.repository_review as review
    from agentos.llm.base import LLMResponse
    from agentos.metrics.model import ModelCallMetrics

    class Truncated:
        def __init__(self, **kwargs):
            pass
        async def generate(self, request):
            return LLMResponse(
                content="unfinished output", tokens_used=12, model="test", latency_secs=0,
                metrics=ModelCallMetrics(backend="test", model="test", prompt_tokens=10,
                    completion_tokens=2, metadata={"finish_reason": "length", "cache_metrics_available": False}),
            )
    monkeypatch.setattr(review, "VLLMBackend", Truncated)
    result = await review.run_repository_review(tmp_path / "truncated", source_root=(Path(__file__).resolve().parents[1] / "src"), backend_name="vllm")
    assert result["checks"]["three_runtime_tasks_completed"] is True
    assert result["checks"]["all_model_responses_complete"] is False
    assert result["passed"] is False
