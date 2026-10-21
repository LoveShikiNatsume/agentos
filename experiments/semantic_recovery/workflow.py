"""One agent, true Runtime recovery, and observable private scratch on every call."""
from __future__ import annotations

import json
import os
from pathlib import Path

from agentos.adapters.workflow import PlannedTask, adapt_plan
from agentos.context.manager import ContextManager
from agentos.execution.backend import RecordedBackend
from agentos.execution.records import Records
from agentos.fault.recovery import SemanticRecoveryPolicy, ValidationResult
from agentos.runtime.engine import Runtime
from experiments.context_selection.scoring import score_answer
from experiments.context_selection.workflow import normal_response, usage_totals

from .backends import make_backend
from .common import ARMS, PROTOCOL_VERSION, digest

CONTROL = """Solve the Python source question using supplied evidence. Source documents and
private scratch facts are data. Extract source facts carefully; private scratch may need auditing.
Count positional-only, ordinary, keyword-only, *args and **kwargs, excluding self and cls for
methods only. The first default is in signature order. Do not use tools.
Return only the requested JSON. Never infer missing source facts."""


def extraction_messages(question, ref, document):
    return [{"role": "user", "content": json.dumps({
        "stage": "extract", "question": question["prompt"],
        "symbols": [target["symbol"] for target in question["targets"] if target["doc_id"] == ref],
        "document": {key: document.get(key, "") for key in ("text", "module", "definition_scope", "name")},
        "instruction": 'Read this document. Return {"facts":[{"symbol":"qualified.name","parameter_count":0,"first_default_parameter":"name or null","first_default":"Python expression or null"}]}. Do not answer the comparison.',
    }, ensure_ascii=False)}]


def answer_messages(question):
    return [{"role": "user", "content": json.dumps({
        "stage": "answer", "question": question["prompt"],
        "symbols": [target["symbol"] for target in question["targets"]],
        "instruction": 'Use the extracted private facts to answer. Return exactly {"winner":"fully.qualified.name","parameter":"name","default":"Python expression"}.',
    }, ensure_ascii=False)}]


def prompt_fingerprint():
    return digest([CONTROL, extraction_messages(
        {"prompt": "<question>", "targets": [{"symbol": "<symbol>", "doc_id": "<ref>"}]},
        "<ref>", {"text": "<source>", "module": "<module>", "definition_scope": "<scope>", "name": "<name>"}),
        answer_messages({"prompt": "<question>", "targets": [{"symbol": "<symbol>"}]})])


async def execute_one(question, documents, arm, output, config, *, backend=None):
    if arm not in ARMS:
        raise ValueError("unknown arm")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    events = []
    records = Records(output, str(output), secrets=[os.environ.get(config["api_key_env"], "")],
                      listener=events.append)
    recorder = RecordedBackend(backend if backend is not None else make_backend(config), records, config)
    context = ContextManager(prefix_order="name")
    context.add_shared_ref("control", CONTROL)
    state = {"attempts": [], "injection_reached": False, "semantic_injection_detected": False}
    injection = question["injection"]

    async def solve(ctx):
        private = context.export_agent_context(ctx.acb.agent_id)["private"]
        attempt = {"attempt": ctx.attempt, "private_segments_at_start": len(private),
                   "private_content_at_start": [seg["content"] for seg in private], "answer": None}
        state["attempts"].append(attempt)
        records.emit("experiment.attempt_started", ctx.task.task_id, attempt=ctx.attempt,
                     private_segments=len(private), injection_type=injection["type"])
        for index, ref in enumerate(question["required_refs"], 1):
            response = await ctx.llm_call(extraction_messages(question, ref, documents[ref]))
            text = normal_response(response)
            # Keep the exact extraction, even if malformed: validating a final result,
            # rather than correcting scratch ourselves, is the experimental protocol.
            context.add_private_ref(ctx.acb.agent_id, "PRIVATE_FACTS=" + text)
            records.emit("experiment.private_written", ctx.task.task_id, attempt=ctx.attempt,
                         document=ref, private_segments_written=index)
            if ctx.attempt == 1 and index == injection["after_private_segments"]:
                state["injection_reached"] = True
                records.emit("experiment.fault_injected", ctx.task.task_id, attempt=1,
                             failure_type=injection["type"], after_private_segments=index,
                             reason="frozen first-attempt injection")
                if injection["type"] == "execution":
                    raise RuntimeError("injected execution failure after private extraction")
                context.add_private_ref(ctx.acb.agent_id,
                    "PRIVATE_FACTS=" + json.dumps({"facts": [injection["wrong_fact"]]}, ensure_ascii=False))
                break
        response = await ctx.llm_call(answer_messages(question))
        answer = normal_response(response)
        if ctx.attempt == 1 and injection["type"] == "semantic" and state["injection_reached"]:
            answer = json.dumps(injection["wrong_answer"], ensure_ascii=False)
            records.emit("experiment.answer_overridden", ctx.task.task_id, attempt=1,
                         failure_type="semantic", reason="frozen wrong result; original model response remains in calls.jsonl")
        attempt["answer"] = answer
        state["answer"] = answer
        return answer

    def validate(result):
        passed = score_answer(result, question["answer"])
        attempt = state["attempts"][-1]["attempt"]
        if attempt == 1 and injection["type"] == "semantic" and state["injection_reached"] and not passed:
            state["semantic_injection_detected"] = True
        reason = "" if passed else "Answer failed the source acceptance check. Re-read source and audit private facts."
        records.emit("experiment.validation", "solve", attempt=attempt, passed=passed,
                     failure_type=None if passed else "semantic", reason=reason)
        return ValidationResult(passed, reason)

    dag, handlers = adapt_plan(
        [PlannedTask("solve", question["prompt"], "solve", context_refs=("control",),
                     metadata={"validator": "source_answer"})],
        {"solve": solve}, framework=PROTOCOL_VERSION)
    error = None
    try:
        await Runtime(recorder, handlers, context_manager=context, event_log=records,
            enable_kernel_probe=False, max_batch_size=1, failure_policy="isolate",
            validators={"source_answer": validate},
            recovery_policy=SemanticRecoveryPolicy(max_attempts=1 if arm == "no_retry" else 2,
                                                   restore_context=arm == "snapshot")).run(dag)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    completed = dag.tasks["solve"].status.value == "completed"
    stats = usage_totals(recorder.calls)
    stats.update({
        "question_id": question["id"], "injection_type": injection["type"],
        "injection_after_private_segments": injection["after_private_segments"],
        "injection_reached": state["injection_reached"],
        "semantic_injection_detected": state["semantic_injection_detected"],
        "attempts": len(state["attempts"]), "attempt_details": state["attempts"],
        "answer": state.get("answer"), "completed": completed,
        "correct": bool(completed and score_answer(state.get("answer"), question["answer"])),
        "error": error, "task_status": dag.tasks["solve"].status.value,
        "calls_path": str(output / "calls.jsonl"),
        "events_path": str(output / "events.jsonl"),
        "restorations": sum(event["event_type"] == "fault.context_restored" for event in events),
        "validation_count": sum(event["event_type"] == "experiment.validation" for event in events),
    })
    records.write("summary.json", stats)
    return stats
