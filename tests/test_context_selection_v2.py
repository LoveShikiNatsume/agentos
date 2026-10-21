"""Protocol invariants checked without any network/model access."""
from copy import deepcopy
import json
from pathlib import Path
import re

import pytest

from experiments.context_selection_v2.common import ARMS, HERE, digest, load, model_config
from experiments.context_selection_v2.generate import build_manifest, derive_answer, symbol_leak
from experiments.context_selection_v2.scoring import normalize_answer, score_answer, parse_json
from experiments.context_selection_v2.backends import StubBackend
from experiments.context_selection_v2.freeze import freeze
from experiments.context_selection_v2.run import run_arm, verify_bundle
from experiments.context_selection_v2.policies import retrieval_budget, selected_documents
from experiments.context_selection_v2.workflow import validate_reader, validate_plan
from experiments.context_selection_v2.analyze import analyze, render_markdown
from experiments.context_selection_v2.estimate import estimate


def fixture_manifest():
    docs = {f"doc{i}": {"id": f"doc{i}", "name": f"module{i}.func", "description": f"module{i}.func",
                         "text": f"def func{i}(a, value={i}):\n    return value\n", "estimated_tokens": 12}
            for i in range(8)}
    questions = [{"id": f"q{i}", "split": "probe", "family": family,
                  "prompt": "Follow `module0.func`" if family == "hop" else "Compare first description and second description",
                  "candidate_refs": list(docs), "required_refs": ["doc0", "doc2"],
                  "answer": {"winner": "module0.func", "parameter": "value", "default": "0"}
                  if family == "describe" else {"callee": "module0.func", "parameter_count": 2, "parameter": "value", "default": "0"}}
                 for i, family in enumerate(("hop", "describe"))]
    return {"documents": docs, "questions": questions}


def test_generation_matches_frozen_manifest_and_all_answers_derive_from_text():
    frozen = load(HERE / "manifest.json")
    generated = build_manifest()
    assert generated == frozen
    assert [(split, family, sum(q["split"] == split and q["family"] == family for q in frozen["questions"]))
            for split in ("probe", "formal") for family in ("hop", "describe")] == [
                ("probe", "hop", 10), ("probe", "describe", 10), ("formal", "hop", 100), ("formal", "describe", 100)]
    for q in frozen["questions"]:
        assert len(set(q["candidate_refs"])) == 8 and len(set(q["required_refs"])) == 2
        assert set(q["required_refs"]) <= set(q["candidate_refs"])
        blind = {k: v for k, v in q.items() if k not in ("answer", "targets")}
        assert derive_answer(blind, frozen["documents"]) == q["answer"]
        if q["family"] == "hop":
            leaf = q["answer"]["callee"].rsplit(".", 1)[-1]
            assert leaf.casefold() not in q["prompt"].casefold()
        else:
            assert not symbol_leak(q["prompt"], q["candidate_refs"], frozen["documents"])
            assert all(len(re.findall(r"\b[A-Za-z]+\b", s)) >= 5 for s in q["descriptions"])


@pytest.mark.parametrize("family,answer", [
    ("hop", {"callee": "m.f", "parameter_count": 2, "parameter": "x", "default": " ( 'a' ) "}),
    ("hop", {"callee": "m.f", "parameter_count": 0, "parameter": None, "default": None}),
    ("describe", {"winner": "m.f", "parameter": "x", "default": " ( 'a' ) "}),
])
def test_exact_scorer_single_fence_normalization(family, answer):
    raw = json.dumps(answer)
    assert score_answer(raw, answer, family)
    assert score_answer("```json\n"+raw+"\n```", answer, family)
    assert not score_answer("```json\n```json\n"+raw+"\n```\n```", answer, family)
    assert not score_answer(raw+" explanation", answer, family)
    assert not score_answer({**answer, "extra": 1}, answer, family)
    assert not score_answer(raw[:-1] + ',"default":"wrong"}', answer, family)


@pytest.mark.parametrize("change", [{"parameter_count": True}, {"parameter_count": -1},
                                  {"parameter": "a", "default": None}, {"callee": "f"}])
def test_hop_invalid_values(change):
    answer = {"callee": "m.f", "parameter_count": 0, "parameter": None, "default": None}
    assert not score_answer({**answer, **change}, answer, "hop")


class NeverPlanner(StubBackend):
    async def generate(self, request):
        payload = json.loads(next(m["content"] for m in request.messages if m["role"] == "user"))
        assert payload["stage"] != "planner"
        return await super().generate(request)


async def six_arms(tmp_path):
    manifest, config = fixture_manifest(), model_config(stub=True)
    bundle = await freeze(manifest, tmp_path / "plans", config, concurrency=2)
    original = digest(bundle)
    assert bundle["usage"]["model_calls"] == 2
    assert all(p["fence_stripped"] == 1 for p in bundle["questions"].values())
    results = {}
    for arm in ARMS:
        results[arm] = await run_arm(manifest, bundle, arm, tmp_path / arm, config, repeats=2, concurrency=2,
                                    backend_factory=NeverPlanner, append_summary=results.get("declared_append"))
    assert digest(bundle) == original
    return manifest, bundle, results


async def test_six_arms_share_plan_and_actual_documents_policies(tmp_path):
    manifest, bundle, summaries = await six_arms(tmp_path)
    for arm, summary in summaries.items():
        assert summary["plans_sha256"] == digest(bundle)
        assert all(r["valid"] for r in summary["questions"])
        budget, _ = retrieval_budget(manifest, bundle, "probe", repeats=2,
            append_summary=summaries["declared_append"] if arm == "bm25_append" else None)
        for q in manifest["questions"]:
            plan = bundle["questions"][q["id"]]["plan"]
            for rep in range(2):
                path = tmp_path / arm / q["id"] / f"repeat-{rep}"
                calls = [json.loads(line) for line in (path / "calls.jsonl").read_text().splitlines()]
                events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]
                requests = [e for e in events if e["event_type"] == "context.refs_requested"]
                assert all(not e["data"]["granted"] for e in requests) if arm != "declared_append" else requests
                readers = {r["id"]: r for r in plan["readers"]}
                for tid, task in readers.items():
                    seen = [c for c in calls if c["task_id"] == tid]
                    initial = set(selected_documents(arm, q, manifest["documents"], task, k=budget[q["id"], tid, rep]))
                    assert set(seen[0]["context_refs"]) & set(q["candidate_refs"]) == initial
                    granted = [ref for e in requests if e["task_id"] == tid for ref in e["data"]["granted"]]
                    assert len(granted) <= 3
                    assert set(seen[-1]["context_refs"]) & set(q["candidate_refs"]) == initial | set(granted)
                    assert len([e for e in requests if e["task_id"] == tid]) <= 2
                for call in calls:
                    text = "\n".join(m.get("content") or "" for m in call["messages"])
                    sent = set(re.findall(r'<document id="([^"<>]+)">', text))
                    assert sent == set(call["context_refs"]) & set(q["candidate_refs"])
                    assert call["request_parameters"]["seed"] == bundle["config"]["seed"] + rep
                    payload = json.loads(next(m["content"] for m in call["messages"] if m["role"] == "user"))
                    if call["task_id"] == "answerer":
                        assert not sent and {"result/reader1", "result/reader2"} <= set(call["context_refs"])
                    else:
                        assert len(payload["catalog"]) == 8
                        assert all(set(r) == {"id", "name", "description"} for r in payload["catalog"])
                        assert not any(r.startswith("result/") for r in call["context_refs"])
                row = next(r for r in summary["questions"] if r["question_id"] == q["id"] and r["replicate"] == rep)
                assert row["fence_stripped"] == len(calls)
    report = analyze(summaries, bootstrap_samples=30)
    assert set(report["families"]) == {"hop", "describe"}
    assert report["overall"]["planner_declaration"]["partial"] == 2
    assert report["overall"]["arms"]["declared"]["append"]["denied_requests"] == 8
    assert report["overall"]["append_minus_declared"]["model_calls"]["difference"] == 2
    assert "declared_append" in render_markdown(report)
    budget = estimate(manifest, bundle, split="probe", repeats=2, append_summary=summaries["declared_append"])
    assert len(budget["arms"]) == 6
    assert budget["arms"]["declared_append"]["bounded_need"]["execution_requests"] == 28


class Broken(StubBackend):
    def __init__(self, config, stage, *, truncate=False):
        super().__init__(config)
        self.stage, self.truncate = stage, truncate
    async def generate(self, request):
        response = await super().generate(request)
        payload = json.loads(next(m["content"] for m in request.messages if m["role"] == "user"))
        if payload["stage"] == self.stage:
            if self.truncate:
                response.metadata["finish_reason"] = "length"
            else:
                response.content = "invalid JSON"
        return response


@pytest.mark.parametrize("stage", ["planner", "reader", "answerer"])
async def test_truncation_invalidates_and_is_reported(tmp_path, stage):
    manifest, config = fixture_manifest(), model_config(stub=True)
    factory = lambda c: Broken(c, stage, truncate=True)
    bundle = await freeze(manifest, tmp_path / "plans", config, backend_factory=factory)
    summary = await run_arm(manifest, bundle, "all", tmp_path / "arm", config, repeats=1, backend_factory=factory)
    assert all(not r["valid"] and r["truncated_calls"] and not r["correct"] for r in summary["questions"])
    if stage == "planner":
        assert all(p["usage"]["model_calls"] == 1 for p in bundle["questions"].values())


async def test_planner_correction_actionable_and_config_bound(tmp_path):
    manifest, config = fixture_manifest(), model_config(stub=True)
    bundle = await freeze(manifest, tmp_path / "plans", config, backend_factory=lambda c: Broken(c, "planner"))
    assert all(p["usage"]["model_calls"] == 2 and p["status"] == "failed" for p in bundle["questions"].values())
    calls = [json.loads(l) for l in (tmp_path / "plans/q0/calls.jsonl").read_text().splitlines()]
    assert "no Markdown code fences" in calls[1]["messages"][-1]["content"]
    with pytest.raises(ValueError, match="enable_thinking"):
        verify_bundle(manifest, bundle, {**config, "enable_thinking": False}, "probe")


async def test_secondary_budget_exact_mean_and_wrong_source_rejected(tmp_path):
    manifest, bundle, summaries = await six_arms(tmp_path)
    source = deepcopy(summaries["declared_append"])
    source["questions"][0]["final_documents"]["reader1"].pop()
    budget, info = retrieval_budget(manifest, bundle, "probe", repeats=2, append_summary=source)
    assert sum(budget.values()) == 31 and info["mean_k"] == 31/8
    assert sorted(set(budget.values())) == [3, 4]
    for key, val in [("plans_sha256", "wrong"), ("arm", "oracle")]:
        with pytest.raises(ValueError):
            retrieval_budget(manifest, bundle, "probe", repeats=2, append_summary={**source, key: val})


@pytest.mark.parametrize("value", [{"facts": "bad"}, {"need": [], "reason": "x"},
                                  {"need": [3], "reason": "x"}, {"facts": [{"parameter_count": 1}]},
                                  {"facts": [], "need": ["doc1"], "extra": 1}])
def test_reader_schema(value):
    with pytest.raises(ValueError):
        validate_reader(value)


def test_plan_single_fence_and_duplicate_key():
    q = fixture_manifest()["questions"][0]
    raw = '{"readers":[{"id":"reader1","goal":"x","context_refs":[]}]}'
    assert validate_plan("```json\n"+raw+"\n```", q)["readers"][0]["context_refs"] == []
    with pytest.raises(ValueError):
        validate_plan('{"readers":[],"readers":[]}', q)


class OutOfBounds(StubBackend):
    async def generate(self, request):
        response = await super().generate(request)
        payload = json.loads(next(m["content"] for m in request.messages if m["role"] == "user"))
        if payload["stage"] == "reader" and not any(m["role"] == "assistant" for m in request.messages):
            response.content = json.dumps({"need": ["external", "doc2", "doc3", "doc4", "doc5"], "reason": "test"})
        return response


async def test_append_denies_outside_candidates_and_excess_then_returns_facts(tmp_path):
    m, c = fixture_manifest(), model_config(stub=True)
    bundle = await freeze(m, tmp_path / "plans", c)
    result = await run_arm(m, bundle, "declared_append", tmp_path / "arm", c, repeats=1, backend_factory=OutOfBounds)
    for row in result["questions"]:
        assert row["valid"] and row["model_calls"] == 5
        for event in row["context_requests"]:
            assert len(event["granted"]) == 3
            assert [r["reason"] for r in event["denied"]] == ["not_candidate", "ref_limit"]
        assert all("external" not in refs for refs in row["actual_documents"].values())


class EmptyPlan(StubBackend):
    async def generate(self, request):
        response = await super().generate(request)
        payload = json.loads(next(m["content"] for m in request.messages if m["role"] == "user"))
        if payload["stage"] == "planner":
            response.content = json.dumps({"readers": [{"id": "reader1", "goal": "extract facts", "context_refs": []}]})
        return response


async def test_empty_declaration_never_broadcasts_and_budget_is_zero(tmp_path):
    m, c = fixture_manifest(), model_config(stub=True)
    bundle = await freeze(m, tmp_path / "plans", c, backend_factory=EmptyPlan)
    for arm in ("declared", "bm25"):
        result = await run_arm(m, bundle, arm, tmp_path / arm, c, repeats=1)
        assert all(r["valid"] and r["required_recall"] == 0 for r in result["questions"])
        assert all(not refs for r in result["questions"] for refs in r["actual_documents"].values())


async def test_analysis_question_level_truncation_and_shared_planner_cost(tmp_path):
    _, _, summaries = await six_arms(tmp_path)
    row = summaries["declared"]["questions"][0]
    row.update(valid=False, truncated_calls=1, execution_truncated_calls=1)
    report = analyze(summaries, bootstrap_samples=20)
    r = report["overall"]
    assert r["arms"]["declared"]["itt_questions"] == 1
    assert r["comparisons"]["declared"]["accuracy"]["questions"] == 1
    assert r["planning_once"]["model_calls"] == 2
    assert r["planner_format"]["fence_stripped"] == 2
    assert report == analyze(summaries, bootstrap_samples=20)


def test_offline_bm25_query_and_neutral_candidates_are_deterministic():
    from experiments.context_selection_v2.generate import neutral_candidates
    from experiments.context_selection_v2.retrieval_analysis import curves
    from experiments.context_selection_v2.policies import bm25
    import random
    m = fixture_manifest()
    for i, doc in enumerate(m["documents"].values()):
        doc["module"] = "target" if i < 3 else "package." + str(i)
    q = m["questions"][0]
    assert neutral_candidates(m["documents"], q["required_refs"], random.Random(7)) == neutral_candidates(m["documents"], q["required_refs"], random.Random(7))
    report = curves(m["questions"], m["documents"])
    for modes in report.values():
        assert modes["goal"] == modes["question_plus_goal"]
        assert modes["goal"]["recall_at_k"][-1] == 1
    doc = m["documents"]["doc7"]
    doc["description"] = "needle_very_unique"
    assert bm25(q, m["documents"], "needle_very_unique", 1) == ["doc7"]


class IgnoresDenial(StubBackend):
    async def generate(self, request):
        response = await super().generate(request)
        payload = json.loads(next(m["content"] for m in request.messages if m["role"] == "user"))
        if payload["stage"] == "reader":
            followup = any(m["role"] == "assistant" for m in request.messages)
            response.content = json.dumps({"need": ["doc2" if followup else "outside"], "reason": "test"})
        return response


async def test_repeated_need_after_denial_cannot_grant_unsent_documents(tmp_path):
    m, c = fixture_manifest(), model_config(stub=True)
    b = await freeze(m, tmp_path / "plans", c)
    s = await run_arm(m, b, "declared_append", tmp_path / "arm", c, repeats=1, backend_factory=IgnoresDenial)
    for row in s["questions"]:
        assert not row["valid"] and not row["append_granted"]
        assert all("doc2" not in refs for refs in row["final_documents"].values())


async def test_secondary_budget_excludes_shared_planner_failures(tmp_path):
    m, c = fixture_manifest(), model_config(stub=True)
    b = await freeze(m, tmp_path / "plans", c)
    b["questions"]["q1"].update(status="failed", plan=None, plan_sha256=digest(None), error="invalid plan")
    s = await run_arm(m, b, "declared_append", tmp_path / "append", c, repeats=1)
    budget, info = retrieval_budget(m, b, "probe", repeats=1, append_summary=s)
    assert info["readers"] == 2 and sum(budget.values()) == 8
    result = await run_arm(m, b, "bm25_append", tmp_path / "secondary", c, repeats=1, append_summary=s)
    assert result["questions"][0]["valid"] and not result["questions"][1]["valid"]


def test_reader_facts_accept_loose_types_and_answer_literals():
    validate_reader({"facts": [{"symbol": "m.f", "first_default": 1, "extra": [1]}]})
    assert score_answer({"winner": "m.f", "parameter": "x", "default": 1},
                        {"winner": "m.f", "parameter": "x", "default": "1"}, "describe")
    assert score_answer({"callee": "m.g", "parameter_count": 2, "parameter": "flag", "default": False},
                        {"callee": "m.g", "parameter_count": 2, "parameter": "flag", "default": "False"}, "hop")
    assert not score_answer({"callee": "m.g", "parameter_count": 2, "parameter": None, "default": None},
                            {"callee": "m.g", "parameter_count": 2, "parameter": "flag", "default": "False"}, "hop")


def test_real_payload_requests_json_object():
    from experiments.context_selection_v2.backends import SeededOpenAIBackend
    from agentos.llm.base import LLMRequest
    backend = SeededOpenAIBackend({"backend": "openai", "model": "m", "base_url": "http://127.0.0.1:9/v1",
                                   "api_key_env": "OPENAI_API_KEY", "seed": 1, "max_tokens": 4096,
                                   "temperature": 0, "enable_thinking": False})
    payload = backend.request_payload(LLMRequest(agent_id="a", messages=[{"role": "user", "content": "{}"}]))
    assert payload["response_format"] == {"type": "json_object"}


def test_reader_need_with_facts_is_a_request():
    value = validate_reader({"facts": [], "need": ["doc1"], "reason": "x"})
    assert value == {"need": ["doc1"], "reason": "x"}
    assert validate_reader({"facts": [{"symbol": "m.f"}], "need": None}) == {"facts": [{"symbol": "m.f"}]}
    assert validate_reader({"facts": [], "need": []}) == {"facts": []}
    with pytest.raises(ValueError):
        validate_reader({"need": None})
