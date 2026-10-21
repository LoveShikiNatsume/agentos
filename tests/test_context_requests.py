from dataclasses import replace

import pytest

from agentos.adapters.workflow import PlannedTask, adapt_plan
from agentos.context.manager import ContextManager
from agentos.events.log import EventLog
from agentos.fault.recovery import SemanticRecoveryPolicy, ValidationResult
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import Runtime


async def run_handler(handler, tmp_path, policy=None, *, initial=("a",), recovery=None):
    cm = ContextManager(prefix_order="name")
    for name in "abcd":
        cm.add_shared_ref(name, f"source {name}")
    private = cm.add_private_ref("other", "private")
    # A shared reference can also explicitly deny read permission.
    cm._refs["d"] = replace(cm.get_ref("d"), permissions=frozenset())
    metadata = {"explicit_context_refs": True}
    if policy is not None:
        metadata["context_request_policy"] = policy
    if recovery:
        metadata["validator"] = "check"
    dag, handlers = adapt_plan([PlannedTask("t", "test", "reader", context_refs=initial,
                                          metadata=metadata)], {"reader": handler}, framework="test")
    backend = MockLLMBackend(latency=0)
    log = EventLog(tmp_path / "events.jsonl")
    rt = Runtime(backend, handlers, context_manager=cm, event_log=log,
                 enable_kernel_probe=False, recovery_policy=recovery,
                 validators={"check": lambda r: ValidationResult(r == "ok", "retry")})
    await rt.run(dag)
    return rt, backend, log, dag


async def test_grant_assembly_and_event(tmp_path):
    async def handler(ctx):
        await ctx.llm_call([{"role": "user", "content": "before"}])
        result = ctx.request_context_refs(["ctx://c", "b", "b", "a"], reason="follow call")
        assert result == {"requested": ["ctx://c", "b", "b", "a"], "granted": ["c", "b"],
                          "denied": [{"ref": "b", "reason": "duplicate_request"},
                                     {"ref": "a", "reason": "already_declared"}],
                          "remaining_requests": 1, "remaining_refs": 1,
                          "requests_used": 1, "refs_granted": 2, "reason": "follow call", "attempt": 1}
        assert ctx.acb.context_refs == ["a", "c", "b"]
        result["granted"].clear()  # Caller cannot mutate the recorded event.
        await ctx.llm_call([{"role": "user", "content": "after"}])
    rt, backend, log, dag = await run_handler(handler, tmp_path,
        {"candidates": ["a", "b", "ctx://c"], "max_requests": 2, "max_refs": 3})
    assert dag.tasks["t"].context_refs == ["a"]
    before, after = backend.requests
    assert "source b" not in str(before.messages)
    assert "source b" in str(after.messages) and "source c" in str(after.messages)
    events = list(log.read(event_type="context.refs_requested"))
    assert len(events) == 1 and events[0].data["granted"] == ["c", "b"]
    assert events[0].task_id == "t" and events[0].agent_id == "agent-t"


async def test_boundaries_limits_and_private_refs(tmp_path):
    async def handler(ctx):
        result = ctx.request_context_refs(["outside", "missing", "other:private:0", "d", "b", "c"])
        assert [d["reason"] for d in result["denied"]] == [
            "not_candidate", "not_found", "not_shared", "not_readable", "ref_limit"]
        assert result["granted"] == ["b"]
        result = ctx.request_context_refs(["c"])
        assert result["denied"] == [{"ref": "c", "reason": "request_limit"}]
        assert result["remaining_requests"] == result["remaining_refs"] == 0
    await run_handler(handler, tmp_path, {"candidates": ["missing", "other:private:0", "b", "c", "d"],
                                        "max_requests": 1, "max_refs": 1})


@pytest.mark.parametrize("policy", [None, {}, {"candidates": ["b"], "max_requests": True, "max_refs": 2}])
async def test_no_or_malformed_policy_fails_closed(tmp_path, policy):
    async def handler(ctx):
        result = ctx.request_context_refs(["b"])
        assert not result["granted"] and result["denied"]
        assert ctx.acb.context_refs == ["a"]
    await run_handler(handler, tmp_path, policy)


@pytest.mark.parametrize("restore", [False, True])
async def test_recovery_restores_declarations_but_never_refunds_task_quota(tmp_path, restore):
    async def handler(ctx):
        if ctx.attempt == 1:
            assert ctx.request_context_refs(["b"])["granted"] == ["b"]
            return "wrong"
        assert ctx.acb.context_refs == (["a"] if restore else ["a", "b"])
        result = ctx.request_context_refs(["b" if restore else "c"])
        assert not result["granted"]
        assert result["denied"][0]["reason"] == "ref_limit"
        assert result["refs_granted"] == 1 and result["requests_used"] == 2
        return "ok"
    await run_handler(handler, tmp_path, {"candidates": ["b", "c"], "max_requests": 2, "max_refs": 1},
                      recovery=SemanticRecoveryPolicy(max_attempts=2, restore_context=restore))


async def test_empty_explicit_declaration_and_bad_argument(tmp_path):
    async def handler(ctx):
        with pytest.raises(TypeError):
            ctx.request_context_refs("b")
        assert ctx.request_context_refs([])["remaining_requests"] == 1
        assert ctx.request_context_refs(["b"])["granted"] == ["b"]
    await run_handler(handler, tmp_path, {"candidates": ["b"], "max_requests": 2, "max_refs": 1}, initial=())
