import asyncio
from copy import deepcopy

import pytest

from agentos.adapters.workflow import PlannedTask, adapt_plan
from agentos.context.manager import ContextManager
from agentos.core.resources import ResourceQuota
from agentos.core.task import Task, TaskDAG, TaskStatus
from agentos.fault.recovery import (
    RecoveryAction, RepetitionError, RetryPolicy, SemanticRecoveryPolicy,
    SemanticValidationError, ValidationExecutionError, ValidationResult,
    normalized_result_hash,
)
from agentos.fault.timeout import AgentTimeoutError
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import AgentHandler, Runtime
from agentos.tools.manager import ToolManager, ToolSpec
from agentos.worker.process import WorkerResult


def make_runtime(handler, *, validator=None, metadata=None, policy=None, **kwargs):
    dag = TaskDAG()
    dag.add_task(Task(task_id="t", name="answer", agent_role="worker",
                      metadata=metadata if metadata is not None else {"validator": "answer"}))
    rt = Runtime(MockLLMBackend(latency=0), {"worker": handler},
                 validators={"answer": validator or (lambda result: ValidationResult(result == "correct", "answer is incorrect"))},
                 recovery_policy=policy, enable_kernel_probe=False, **kwargs)
    return dag, rt


def events(rt, kind):
    return [event for event in rt.events if event.event_type == kind]


class SequenceHandler(AgentHandler):
    def __init__(self, values):
        self.values = iter(values)
        self.attempts = []
        self.feedback = []

    async def run(self, ctx):
        self.attempts.append(ctx.attempt)
        self.feedback.append(deepcopy(ctx.recovery_feedback))
        value = next(self.values)
        if isinstance(value, BaseException):
            raise value
        return value


@pytest.mark.parametrize("restore_context,remaining", [(False, 2), (True, 1)])
async def test_feedback_retry_restores_only_snapshot_arm_and_reaches_llm(restore_context, remaining):
    class Partial(AgentHandler):
        async def run(self, ctx):
            ctx.context_manager.add_private(ctx.acb.agent_id, f"attempt {ctx.attempt} fact")
            if ctx.attempt == 1:
                return "wrong"
            assert len(ctx.recovery_feedback) == 1
            assert "answer is incorrect" in ctx.recovery_feedback[-1]["content"]
            assert ctx.context_manager.get_ref("background") is not None
            await ctx.llm_call([{"role": "user", "content": "please retry"}])
            return "correct"

    cm = ContextManager()
    cm.add_shared("background", "stable shared content")
    dag, rt = make_runtime(Partial(), context_manager=cm,
                          policy=SemanticRecoveryPolicy(max_attempts=2, restore_context=restore_context))
    assert await rt.run(dag) == {"t": "correct"}
    private = cm.export_agent_context("agent-t")["private"]
    assert len(private) == remaining
    assert "answer is incorrect" in rt._llm.requests[0].messages[-1]["content"]
    validations = events(rt, "fault.validation_completed")
    assert [(e.data["passed"], e.data["attempt"]) for e in validations] == [(False, 1), (True, 2)]
    assert validations[0].data["failure_type"] == "semantic"
    assert events(rt, "fault.retry_scheduled")[0].data["next_attempt"] == 2
    assert len(events(rt, "fault.context_restored")) == int(restore_context)
    feedback = rt._bus.get_history(message_type="recovery_feedback")
    assert len(feedback) == 1 and feedback[0].content["failure_type"] == "semantic"


async def test_execution_failure_gets_feedback_and_uses_same_attempt_budget():
    handler = SequenceHandler([RuntimeError("temporary execution failure"), "correct"])
    dag, rt = make_runtime(handler, policy=SemanticRecoveryPolicy(max_attempts=2))
    assert await rt.run(dag) == {"t": "correct"}
    assert handler.attempts == [1, 2]
    assert "temporary execution failure" in handler.feedback[1][0]["content"]
    assert events(rt, "fault.retry_scheduled")[0].data["failure_type"] == "execution"
    assert len(events(rt, "fault.validation_completed")) == 1


@pytest.mark.parametrize("max_attempts", [1, 2])
async def test_semantic_max_attempts_includes_initial_attempt(max_attempts):
    handler = SequenceHandler(["wrong1", "wrong2", "correct"])
    dag, rt = make_runtime(handler, policy=SemanticRecoveryPolicy(max_attempts=max_attempts))
    with pytest.raises(SemanticValidationError):
        await rt.run(dag)
    assert handler.attempts == list(range(1, max_attempts + 1))
    assert events(rt, "fault.recovery_failed")[-1].data["failure_type"] == "semantic"
    assert dag.tasks["t"].status == TaskStatus.FAILED


async def test_equal_normalized_rejected_results_stop_even_with_ordinary_retry_policy():
    handler = SequenceHandler([{"a": " wrong  answer", "b": 2}, {"b": 2, "a": "wrong answer "}, "correct"])
    dag, rt = make_runtime(handler, validator=lambda value: ValidationResult(False, "incorrect facts"),
                          policy=RetryPolicy(max_retries=10))
    with pytest.raises(RepetitionError, match="consecutive"):
        await rt.run(dag)
    assert handler.attempts == [1, 2]
    stop = events(rt, "fault.recovery_stopped")[-1]
    assert stop.data["failure_type"] == "repetition" and stop.data["attempt"] == 2
    assert len(events(rt, "fault.validation_completed")) == 2
    assert len(events(rt, "fault.retry_scheduled")) == 1


async def test_intervening_execution_failure_breaks_result_repetition_chain():
    handler = SequenceHandler(["wrong", RuntimeError("interrupted"), "wrong", "correct"])
    dag, rt = make_runtime(handler, policy=SemanticRecoveryPolicy(max_attempts=4))
    assert await rt.run(dag) == {"t": "correct"}
    assert handler.attempts == [1, 2, 3, 4]
    assert not events(rt, "fault.repetition_detected")


async def test_same_result_that_now_passes_is_accepted_before_repetition_check():
    decisions = iter([ValidationResult(False, "not yet"), ValidationResult(True)])
    handler = SequenceHandler(["same", "same"])
    dag, rt = make_runtime(handler, validator=lambda _: next(decisions),
                          policy=SemanticRecoveryPolicy(max_attempts=4))
    assert await rt.run(dag) == {"t": "same"}
    assert not events(rt, "fault.repetition_detected")


@pytest.mark.parametrize("catch_error", [False, True])
async def test_repeated_tool_calls_stop_before_extra_side_effect_and_cannot_be_swallowed(catch_error):
    calls = []
    tools = ToolManager()
    tools.register(ToolSpec("read", lambda **kwargs: calls.append(kwargs)))

    class Repeats(AgentHandler):
        async def run(self, ctx):
            try:
                for _ in range(8):
                    await ctx.tool_call("read", path="x")
            except RepetitionError:
                if not catch_error:
                    raise
            return "correct"

    dag, rt = make_runtime(Repeats(), tool_manager=tools,
                          metadata={"validator": "answer", "repeated_tool_call_threshold": 2},
                          policy=SemanticRecoveryPolicy(max_attempts=8))
    dag.tasks["t"].resource_requirements.allowed_tools = ["read"]
    with pytest.raises(RepetitionError, match="threshold 2"):
        await rt.run(dag)
    assert len(calls) == 2
    assert not events(rt, "fault.retry_scheduled")
    assert events(rt, "fault.repetition_detected")[0].data["attempt"] == 1


async def test_tool_repetition_counts_reset_between_attempts():
    calls = []
    tools = ToolManager()
    tools.register(ToolSpec("read", lambda **kwargs: calls.append(kwargs)))

    class Work(AgentHandler):
        async def run(self, ctx):
            await ctx.tool_call("read", a=1, b=2)
            await ctx.tool_call("read", b=2, a=1)
            if ctx.attempt == 1:
                raise RuntimeError("retry after two legitimate identical calls")
            return "correct"

    dag, rt = make_runtime(Work(), tool_manager=tools,
                          metadata={"validator": "answer", "repeated_tool_call_threshold": 2},
                          policy=SemanticRecoveryPolicy(max_attempts=2))
    dag.tasks["t"].resource_requirements.allowed_tools = ["read"]
    assert await rt.run(dag) == {"t": "correct"}
    assert len(calls) == 4


async def test_no_validator_preserves_tool_calls_and_unvalidated_result():
    calls = []
    tools = ToolManager()
    tools.register(ToolSpec("read", lambda **kwargs: calls.append(kwargs)))

    class Legacy:
        async def run(self, ctx):
            for _ in range(6):
                await ctx.tool_call("read", path="same")
            return "wrong"

    dag, rt = make_runtime(Legacy(), metadata={}, tool_manager=tools)
    dag.tasks["t"].resource_requirements.allowed_tools = ["read"]
    assert await rt.run(dag) == {"t": "wrong"}
    assert len(calls) == 6 and not events(rt, "fault.validation_completed")


async def test_invalid_result_is_not_published_or_notified_or_released_to_dependents():
    cm = ContextManager()
    seen = []

    async def bad(ctx):
        return "wrong"

    async def downstream(ctx):
        seen.append("ran")
        return "correct"

    dag, handlers = adapt_plan([
        PlannedTask("t", "bad", "bad", output_ref="result/t",
                    metadata={"validator": "answer", "notify_agent": "agent-next"}),
        PlannedTask("next", "next", "next", dependencies=("t",)),
    ], {"bad": bad, "next": downstream}, framework="test")
    rt = Runtime(MockLLMBackend(latency=0), handlers, context_manager=cm,
                 validators={"answer": lambda _: ValidationResult(False, "wrong")},
                 recovery_policy=SemanticRecoveryPolicy(max_attempts=1), enable_kernel_probe=False)
    with pytest.raises(SemanticValidationError):
        await rt.run(dag)
    assert cm.get_ref("result/t") is None
    assert not seen and dag.tasks["next"].status == TaskStatus.SKIPPED
    assert not rt._bus.get_history(message_type="result_ref")
    assert not events(rt, "context.result_published")


async def test_successful_retry_publishes_once_and_only_after_validation():
    cm = ContextManager()
    async def work(ctx):
        assert cm.get_ref("result/t") is None
        return "wrong" if ctx.attempt == 1 else "correct"

    dag, handlers = adapt_plan([PlannedTask("t", "answer", "worker", output_ref="result/t",
                                           metadata={"validator": "answer"})],
                              {"worker": work}, framework="test")
    rt = Runtime(MockLLMBackend(latency=0), handlers, context_manager=cm,
                 validators={"answer": lambda v: ValidationResult(v == "correct", "wrong")},
                 recovery_policy=SemanticRecoveryPolicy(max_attempts=2), enable_kernel_probe=False)
    assert await rt.run(dag) == {"t": "correct"}
    assert cm.read("agent-t", "result/t") == "correct"
    assert len(events(rt, "context.result_published")) == 1
    kinds = [e.event_type for e in rt.events]
    assert max(i for i, kind in enumerate(kinds) if kind == "fault.validation_completed") < kinds.index("context.result_published")


async def test_async_validator_is_supported():
    async def validator(value):
        await asyncio.sleep(0)
        return ValidationResult(value == "correct")
    dag, rt = make_runtime(SequenceHandler(["correct"]), validator=validator)
    assert await rt.run(dag) == {"t": "correct"}


@pytest.mark.parametrize("validator", [lambda _: "false", lambda _: 1 / 0])
async def test_validator_defects_are_terminal_and_not_semantic_rejections(validator):
    handler = SequenceHandler(["correct", "correct"])
    dag, rt = make_runtime(handler, validator=validator, policy=RetryPolicy(max_retries=5))
    with pytest.raises(ValidationExecutionError):
        await rt.run(dag)
    assert handler.attempts == [1]
    validation = events(rt, "fault.validation_completed")[0]
    assert validation.data["failure_type"] == "validation_error"
    assert not events(rt, "fault.retry_scheduled")


async def test_async_validator_obeys_single_attempt_timeout():
    async def validator(value):
        await asyncio.sleep(10)
        return ValidationResult(True)
    dag, rt = make_runtime(SequenceHandler(["correct"]), validator=validator)
    dag.tasks["t"].resource_requirements.timeout_seconds = 0.01
    with pytest.raises(AgentTimeoutError):
        await asyncio.wait_for(rt.run(dag), timeout=1)
    assert events(rt, "fault.validation_completed")[0].data["reason"] == "validation cancelled"
    assert events(rt, "fault.timeout")


async def test_unknown_validator_rejected_before_handler_run():
    handler = SequenceHandler(["correct"])
    dag, rt = make_runtime(handler, metadata={"validator": "missing"})
    with pytest.raises(ValueError, match="unregistered validator"):
        await rt.run(dag)
    assert handler.attempts == []


async def test_subprocess_retry_receives_feedback_in_payload_without_mutating_metadata():
    class FakeWorkers:
        def __init__(self):
            self.specs = []

        async def run(self, spec, **kwargs):
            self.specs.append(deepcopy(spec))
            return WorkerResult(ok=True, worker_type=spec.worker_type, task_id=spec.task_id,
                                agent_id=spec.agent_id,
                                result="wrong" if len(self.specs) == 1 else "correct")

    workers = FakeWorkers()
    metadata = {"validator": "answer", "execution": "subprocess", "worker_payload": {"existing": 1}}
    dag, rt = make_runtime(None, metadata=metadata, worker_manager=workers,
                          policy=SemanticRecoveryPolicy(max_attempts=2))
    assert await rt.run(dag) == {"t": "correct"}
    assert "recovery_feedback" not in workers.specs[0].payload
    assert "answer is incorrect" in workers.specs[1].payload["recovery_feedback"][0]["content"]
    assert metadata["worker_payload"] == {"existing": 1}


def test_normalization_and_policy_contracts():
    assert normalized_result_hash({"b": [1, "e\u0301 \n x"], "a": 2}) == normalized_result_hash({"a": 2, "b": [1, "\u00e9 x"]})
    assert normalized_result_hash([1, 2]) != normalized_result_hash([2, 1])
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            SemanticRecoveryPolicy(max_attempts=bad)
    with pytest.raises(TypeError):
        ValidationResult("false")
    assert SemanticRecoveryPolicy(2, restore_context=False).decide(RuntimeError("x"), 1).action == RecoveryAction.RETRY
    assert SemanticRecoveryPolicy(2).decide(RepetitionError("loop"), 1).action == RecoveryAction.FAIL


async def test_tool_argument_whitespace_remains_significant():
    calls = []
    tools = ToolManager()
    tools.register(ToolSpec("write", lambda **kwargs: calls.append(kwargs)))

    class Work(AgentHandler):
        async def run(self, ctx):
            await ctx.tool_call("write", content="a b")
            await ctx.tool_call("write", content="a  b")
            return "correct"

    dag, rt = make_runtime(Work(), tool_manager=tools,
                          metadata={"validator": "answer", "repeated_tool_call_threshold": 1})
    dag.tasks["t"].resource_requirements.allowed_tools = ["write"]
    assert await rt.run(dag) == {"t": "correct"}
    assert len(calls) == 2


async def test_raw_json_result_reformatting_does_not_bypass_repetition_stop():
    handler = SequenceHandler(['{"a":1,"b":2}', '{ "b": 2, "a": 1 }'])
    dag, rt = make_runtime(handler, validator=lambda _: ValidationResult(False, "wrong"),
                          policy=SemanticRecoveryPolicy(max_attempts=5))
    with pytest.raises(RepetitionError):
        await rt.run(dag)
    assert handler.attempts == [1, 2]


@pytest.mark.parametrize("validator", [None, False, 42])
def test_validator_registry_rejects_noncallables(validator):
    with pytest.raises(ValueError, match="callable"):
        Runtime(MockLLMBackend(latency=0), {}, validators={"check": validator},
                enable_kernel_probe=False)


async def test_repetition_latch_blocks_later_llm_calls_after_swallowed_tool_error():
    tools = ToolManager()
    tools.register(ToolSpec("read", lambda **kwargs: "ok"))

    class Work(AgentHandler):
        async def run(self, ctx):
            try:
                await ctx.tool_call("read", path="same")
                await ctx.tool_call("read", path="same")
            except RepetitionError:
                pass
            await ctx.llm_call([{"role": "user", "content": "should not be sent"}])
            return "correct"

    dag, rt = make_runtime(Work(), tool_manager=tools,
                          metadata={"validator": "answer", "repeated_tool_call_threshold": 1},
                          policy=SemanticRecoveryPolicy(max_attempts=5))
    dag.tasks["t"].resource_requirements.allowed_tools = ["read"]
    with pytest.raises(RepetitionError):
        await rt.run(dag)
    assert rt._llm.requests == []
