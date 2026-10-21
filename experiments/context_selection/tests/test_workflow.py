from __future__ import annotations
import asyncio
import json
from pathlib import Path
import re
import pytest

from agentos.llm.base import LLMRequest
from experiments.context_selection.backends import SeededOpenAIBackend, StubBackend
from experiments.context_selection.common import digest, model_config
from experiments.context_selection.freeze import freeze
from experiments.context_selection.policies import retrieval_budget, selected_documents
from experiments.context_selection.run import run_arm
from experiments.context_selection.workflow import validate_plan, usage_totals


def fixture_manifest():
    docs = {f"doc{i}": {"id": f"doc{i}", "name": f"module{i}.func", "description": "source definition",
                         "text": f"def func{i}(a, value={i}):\n    return value\n", "estimated_tokens": 12}
            for i in range(8)}
    questions = [{"id": f"q{i}", "split": "probe", "prompt": "Compare module0.func and module2.func",
                  "candidate_refs": list(docs), "required_refs": ["doc0", "doc2"],
                  "answer": {"winner": "module0.func", "parameter": "value", "default": "0"}}
                 for i in range(2)]
    return {"documents": docs, "questions": questions}


def test_real_payload_includes_seed_temperature_and_limit():
    config = model_config(base_url="http://example.invalid", model="test", seed=41)
    payload = SeededOpenAIBackend(config).request_payload(LLMRequest("a", [{"role": "user", "content": "x"}]))
    assert payload["seed"] == 41 and payload["temperature"] == 0 and payload["max_tokens"] == 4096
    assert "tools" not in payload


@pytest.mark.parametrize("bad", [
    {}, {"readers": []}, {"readers": [{"id": "reader1", "goal": "x", "context_refs": ["unknown"]}]},
    {"readers": [{"id": "reader1", "goal": "x", "context_refs": ["doc0", "doc0"]}]},
    {"readers": [{"id": "answerer", "goal": "x", "context_refs": []}]},
])
def test_plan_validation(bad):
    with pytest.raises(ValueError):
        validate_plan(bad, fixture_manifest()["questions"][0])


def test_unknown_usage_is_not_zero():
    values = usage_totals([{"status": "completed", "usage": None, "finish_reason": "stop"}])
    assert values["prompt_tokens"] is None and values["cached_tokens"] is None
    assert usage_totals([])["prompt_tokens"] == 0


def test_bm25_matches_average_budget_exactly():
    manifest = fixture_manifest()
    bundle = {"config": {"seed": 7}, "questions": {
        q["id"]: {"status": "completed", "plan": {"readers": [
            {"id": "reader1", "context_refs": ["doc0"]},
            {"id": "reader2", "context_refs": ["doc0", "doc1"]}]}}
        for q in manifest["questions"]}}
    budget, info = retrieval_budget(manifest, bundle, "probe")
    assert sorted(budget.values()) == [1, 1, 2, 2]
    assert info["mean_k"] == 1.5
    assert budget == retrieval_budget(manifest, bundle, "probe")[0]


class NeverPlanner(StubBackend):
    async def generate(self, request):
        stage = json.loads(next(m["content"] for m in request.messages if m["role"] == "user"))["stage"]
        assert stage != "planner", "arm must never invoke planner"
        return await super().generate(request)


def test_four_arms_reuse_one_plan_and_send_exact_documents(tmp_path):
    async def scenario():
        manifest = fixture_manifest()
        config = model_config(stub=True)
        bundle = await freeze(manifest, tmp_path / "plans", config, split="probe", concurrency=2)
        original = digest(bundle)
        budget, _ = retrieval_budget(manifest, bundle, "probe")
        summaries = []
        for arm in ("all", "declared", "bm25", "oracle"):
            output = tmp_path / arm
            summary = await run_arm(manifest, bundle, arm, output, config, repeats=2,
                                    concurrency=2, backend_factory=NeverPlanner)
            summaries.append(summary)
            assert summary["plans_sha256"] == original
            assert len(summary["questions"]) == 4
            assert all(row["valid"] and row["model_calls"] == 3 for row in summary["questions"])
            for q in manifest["questions"]:
                frozen = bundle["questions"][q["id"]]
                taskmap = {row["id"]: row for row in frozen["plan"]["readers"]}
                for repeat in range(2):
                    calls = [json.loads(line) for line in
                             (output / q["id"] / f"repeat-{repeat}" / "calls.jsonl").read_text().splitlines()]
                    assert len(calls) == 3
                    for call in calls:
                        task = taskmap.get(call["task_id"])
                        expected = set(selected_documents(arm, q, manifest["documents"], task,
                                       k=budget.get((q["id"], call["task_id"]))))
                        sent = set(re.findall(r'<document id="(doc\d+)">',
                                              "\n".join(m.get("content", "") for m in call["messages"])))
                        assert sent == expected
                        for ref in expected:
                            assert "Name: " + manifest["documents"][ref]["name"] in call["messages"][0]["content"]
                        assert set(call["context_refs"]) & set(q["candidate_refs"]) == expected
                        assert call["request_parameters"]["seed"] == config["seed"] + repeat
                        if call["task_id"] == "answerer":
                            assert {"result/reader1", "result/reader2"} <= set(call["context_refs"])
                        else:
                            assert not any(r.startswith("result/") for r in call["context_refs"])
            assert all(row["plan_sha256"] == bundle["questions"][row["question_id"]]["plan_sha256"]
                       for row in summary["questions"])
        assert digest(bundle) == original
        # Rerunning another output reuses the same plan as well.
        await run_arm(manifest, bundle, "declared", tmp_path / "again", config,
                      repeats=1, backend_factory=NeverPlanner)
    asyncio.run(scenario())


class EmptyPlan(StubBackend):
    async def generate(self, request):
        response = await super().generate(request)
        payload = json.loads(next(m["content"] for m in request.messages if m["role"] == "user"))
        if payload["stage"] == "planner":
            response.content = json.dumps({"readers": [{"id": "reader1", "goal": "No source", "context_refs": []}]})
        return response


def test_empty_declared_context_does_not_broadcast(tmp_path):
    async def scenario():
        manifest, config = fixture_manifest(), model_config(stub=True)
        bundle = await freeze(manifest, tmp_path / "plans", config, backend_factory=EmptyPlan)
        summary = await run_arm(manifest, bundle, "declared", tmp_path / "run", config,
                                repeats=1, backend_factory=NeverPlanner)
        assert all(not refs for row in summary["questions"] for refs in row["actual_documents"].values())
        assert all(row["required_recall"] == 0 for row in summary["questions"])
    asyncio.run(scenario())


class MalformedPlan(StubBackend):
    def __init__(self, config, always_bad=False):
        super().__init__(config)
        self.planner_calls, self.always_bad = 0, always_bad
    async def generate(self, request):
        response = await super().generate(request)
        stage = json.loads(next(m["content"] for m in request.messages if m["role"] == "user"))["stage"]
        if stage == "planner":
            self.planner_calls += 1
            if self.planner_calls == 1 or self.always_bad:
                response.content = "invalid"
        return response


@pytest.mark.parametrize("always_bad", [False, True])
def test_exactly_one_planner_format_correction(tmp_path, always_bad):
    async def scenario():
        manifest, config = fixture_manifest(), model_config(stub=True)
        bundle = await freeze(manifest, tmp_path / "plans", config,
                              backend_factory=lambda c: MalformedPlan(c, always_bad))
        for row in bundle["questions"].values():
            assert row["usage"]["model_calls"] == 2
            assert row["format_corrections"] == 1
            assert (row["status"] == "failed") == always_bad
    asyncio.run(scenario())


class Truncated(StubBackend):
    def __init__(self, config, stage):
        super().__init__(config)
        self.stage = stage
    async def generate(self, request):
        response = await super().generate(request)
        payload = json.loads(next(m["content"] for m in request.messages if m["role"] == "user"))
        if payload["stage"] == self.stage:
            response.metadata["finish_reason"] = "length"
        return response


@pytest.mark.parametrize("stage", ["planner", "reader", "answerer"])
def test_truncation_invalidates_question_at_every_stage(tmp_path, stage):
    async def scenario():
        manifest, config = fixture_manifest(), model_config(stub=True)
        bundle = await freeze(manifest, tmp_path / "plans", config,
                              backend_factory=lambda c: Truncated(c, stage))
        summary = await run_arm(manifest, bundle, "all", tmp_path / "run", config, repeats=1,
                                backend_factory=lambda c: Truncated(c, stage))
        assert all(not row["valid"] and row["truncated_calls"] >= 1 and not row["correct"]
                   for row in summary["questions"])
        if stage == "planner":
            assert all(row["model_calls"] == 0 for row in summary["questions"])
    asyncio.run(scenario())


def test_manifest_or_configuration_mismatch_rejected(tmp_path):
    async def scenario():
        manifest, config = fixture_manifest(), model_config(stub=True)
        bundle = await freeze(manifest, tmp_path / "plans", config)
        with pytest.raises(ValueError, match="manifest"):
            await run_arm({**manifest, "changed": True}, bundle, "all", tmp_path / "run", config)
        with pytest.raises(ValueError, match="seed"):
            await run_arm(manifest, bundle, "all", tmp_path / "run", {**config, "seed": 10})
    asyncio.run(scenario())

@pytest.mark.parametrize("usage", [
    {"prompt_tokens": 10, "completion_tokens": 2, "prompt_tokens_details": []},
    {"prompt_tokens": 10, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 11}},
    {"prompt_tokens": 10, "completion_tokens": 2, "prompt_cache_hit_tokens": 1,
     "prompt_tokens_details": {"cached_tokens": 2}},
])
def test_malformed_or_inconsistent_cache_usage_is_unknown(usage):
    result = usage_totals([{"usage": usage, "finish_reason": "stop"}])
    assert result["prompt_tokens"] == 10
    assert result["cached_tokens"] is None

def test_duplicate_plan_keys_are_rejected():
    raw = '{"readers":[],"readers":[{"id":"reader1","goal":"x","context_refs":[]}]}'
    with pytest.raises(ValueError, match="duplicate"):
        validate_plan(raw, fixture_manifest()["questions"][0])

class MalformedAnswer(StubBackend):
    async def generate(self, request):
        response = await super().generate(request)
        if request.task_id == "answerer":
            response.content = "not JSON"
        return response

def test_answer_format_error_is_wrong_but_remains_valid(tmp_path):
    async def scenario():
        manifest, config = fixture_manifest(), model_config(stub=True)
        bundle = await freeze(manifest, tmp_path/"plans", config)
        result = await run_arm(manifest,bundle,"declared",tmp_path/"run",config,repeats=1,
                               backend_factory=MalformedAnswer)
        assert all(r["valid"] and not r["correct"] for r in result["questions"])
        assert all(r["failures"][0]["kind"] == "answer_format" for r in result["questions"])
    asyncio.run(scenario())

def test_concurrency_limits_actual_backend_calls(tmp_path):
    class Measured(StubBackend):
        active = peak = 0
        async def generate(self,request):
            type(self).active += 1
            type(self).peak = max(type(self).peak,type(self).active)
            try:
                await asyncio.sleep(.001)
                return await super().generate(request)
            finally:
                type(self).active -= 1
    async def scenario():
        manifest, config = fixture_manifest(),model_config(stub=True)
        bundle = await freeze(manifest,tmp_path/"plans",config)
        await run_arm(manifest,bundle,"all",tmp_path/"run",config,concurrency=1,
                      repeats=2,backend_factory=Measured)
        assert Measured.peak == 1
    asyncio.run(scenario())
