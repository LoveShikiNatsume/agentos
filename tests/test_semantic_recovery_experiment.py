"""Experiment C verifies Runtime recovery with fake evidence, without HTTP."""
from copy import deepcopy
import json
import socket

import httpx
import pytest

from agentos.execution.backend import OpenAIBackend
from experiments.context_selection.common import model_config
from experiments.semantic_recovery.analyze import analyze
from experiments.semantic_recovery.backends import StubBackend
from experiments.semantic_recovery.common import materialize
from experiments.semantic_recovery.estimate import estimate
from experiments.semantic_recovery.generate import build_manifest
from experiments.semantic_recovery.run import run_arm
from experiments.semantic_recovery.workflow import execute_one


def corpus():
    documents = {
        "loser": {"text": "def small(value=1):\n    return value\n", "module": "fixture",
                  "definition_scope": "", "name": "small"},
        "winner": {"text": "def large(left, right=2):\n    return right\n", "module": "fixture",
                   "definition_scope": "", "name": "large"},
    }
    question = {"prompt": "Which function has most parameters? Give its first default.",
        "split": "probe", "required_refs": ["loser", "winner"],
        "targets": [{"symbol": "fixture.small", "doc_id": "loser"}, {"symbol": "fixture.large", "doc_id": "winner"}],
        "answer": {"winner": "fixture.large", "parameter": "right", "default": "2"}}
    return {"documents": documents, "questions": [{**question, "id": "q1"}, {**question, "id": "q2"}]}


def manifest():
    source = corpus()
    spec = build_manifest(source)
    for injection in spec["injections"]:
        injection["after_private_segments"] = 1
    return materialize(spec, source)


@pytest.fixture
def no_http(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("stub attempted network access")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(OpenAIBackend, "generate", blocked)
    monkeypatch.setattr(httpx.AsyncClient, "request", blocked)


def test_manifest_is_seeded_balanced_and_hash_checked():
    source = corpus()
    spec = build_manifest(source)
    assert spec == build_manifest(source)
    assert sorted(row["type"] for row in spec["injections"]) == ["execution", "semantic"]
    assert all(1 <= row["after_private_segments"] <= 2 for row in spec["injections"])
    changed = deepcopy(source)
    changed["documents"]["winner"]["text"] += "# changed"
    with pytest.raises(ValueError, match="hash"):
        materialize(spec, changed)
    duplicate = deepcopy(spec)
    duplicate["injections"][1]["question_id"] = duplicate["injections"][0]["question_id"]
    with pytest.raises(ValueError, match="exactly once"):
        materialize(duplicate, source)


@pytest.mark.asyncio
async def test_three_arms_real_runtime_restore_and_feedback(tmp_path, no_http):
    data = manifest()
    config = model_config(stub=True)
    summaries = {}
    for arm in ("dirty", "snapshot", "no_retry"):
        summaries[arm] = await run_arm(data, arm, tmp_path / arm, config, split="probe", concurrency=2)
    dirty = {row["question_id"]: row for row in summaries["dirty"]["questions"]}
    clean = {row["question_id"]: row for row in summaries["snapshot"]["questions"]}
    assert all(row["attempts"] == 2 for row in clean.values())
    assert all(row["completed"] and row["correct"] for row in clean.values())
    assert all(row["restorations"] == 1 for row in clean.values())
    assert all(row["attempt_details"][1]["private_segments_at_start"] == 0 for row in clean.values())
    assert all(row["attempt_details"][1]["private_segments_at_start"] > 0 for row in dirty.values())
    assert not any(row["completed"] for row in summaries["no_retry"]["questions"])
    for row in clean.values():
        calls = [json.loads(line) for line in open(row["calls_path"])]
        assert all(call["usage"]["prompt_tokens"] > 0 and call["finish_reason"] == "stop" for call in calls)
        assert all(call["request_parameters"]["max_tokens"] >= 2048 for call in calls)
        assert any("audit private facts" in message.get("content", "") or "injected execution failure" in message.get("content", "")
                   for call in calls for message in call["messages"])
        first_calls = row["injection_after_private_segments"] + int(row["injection_type"] == "semantic")
        dirty_calls = [json.loads(line) for line in open(dirty[row["question_id"]]["calls_path"])]
        assert [call["messages"] for call in calls[:first_calls]] == [call["messages"] for call in dirty_calls[:first_calls]]
        assert not any("PRIVATE_FACTS=" in message.get("content", "")
                       for message in calls[first_calls]["messages"])
        assert any("PRIVATE_FACTS=" in message.get("content", "")
                   for message in dirty_calls[first_calls]["messages"])
        if row["injection_type"] == "semantic":
            assert row["semantic_injection_detected"]
            assert not dirty[row["question_id"]]["completed"]
            assert any("__injected_wrong_parameter__" in message.get("content", "")
                       for message in dirty_calls[first_calls]["messages"])
    report = analyze(summaries, bootstrap_samples=100)
    assert report["paired_snapshot_minus_dirty"]["all"]["metrics"]["correct"]["questions"] == 2
    assert report["arms"]["no_retry"]["completion_rate"] == 0
    assert report["arms"]["snapshot"]["semantic_detection_rate_reached"] == 1


@pytest.mark.asyncio
async def test_truncation_is_recorded_and_missing_usage_is_not_zero(tmp_path, no_http):
    class TruncatedThenMissing(StubBackend):
        count = 0
        async def generate(self, request):
            response = await super().generate(request)
            self.count += 1
            response.metadata["usage"] = None
            if self.count == 1:
                response.metadata["finish_reason"] = "length"
            return response
    data = manifest()
    config = model_config(stub=True)
    row = await execute_one(data["questions"][0], data["documents"], "snapshot", tmp_path / "trial", config,
                            backend=TruncatedThenMissing(config))
    assert row["truncated_calls"] == 1
    assert row["prompt_tokens"] is None
    assert row["completion_tokens"] is None
    assert row["attempts"] == 2
    assert not row["injection_reached"]


@pytest.mark.asyncio
async def test_analysis_rejects_incomparable_runs_and_keeps_missing_pairs(tmp_path, no_http):
    data = manifest()
    config = model_config(stub=True)
    summaries = {arm: await run_arm(data, arm, tmp_path / arm, config, concurrency=1)
                 for arm in ("dirty", "snapshot")}
    for field, change in (
        ("manifest_sha256", "other"), ("prompt_sha256", "other"), ("split", "formal")):
        bad = deepcopy(summaries)
        bad["snapshot"][field] = change
        with pytest.raises(ValueError, match="incomparable"):
            analyze(bad, bootstrap_samples=10)
    bad = deepcopy(summaries)
    bad["snapshot"]["config"]["model"] = "other"
    with pytest.raises(ValueError, match="configuration"):
        analyze(bad, bootstrap_samples=10)
    bad = deepcopy(summaries)
    bad["snapshot"]["questions"].pop()
    with pytest.raises(ValueError, match="grid"):
        analyze(bad, bootstrap_samples=10)
    bad = deepcopy(summaries)
    bad["snapshot"]["config"]["timeout"] = 1
    with pytest.raises(ValueError, match="configuration"):
        analyze(bad, bootstrap_samples=10)
    bad = deepcopy(summaries)
    bad["snapshot"]["questions"][0]["prompt_tokens"] = True
    with pytest.raises(ValueError, match="token counter"):
        analyze(bad, bootstrap_samples=10)
    missing = deepcopy(summaries)
    missing["snapshot"]["questions"][0]["prompt_tokens"] = None
    report = analyze(missing, bootstrap_samples=10)
    paired = report["paired_snapshot_minus_dirty"]["all"]["metrics"]
    assert paired["correct"]["questions"] == 2
    assert paired["prompt_tokens"]["questions"] == 1
    assert paired["prompt_tokens"]["coverage"] == .5
    assert report["arms"]["snapshot"]["total_prompt_tokens"] is None


def test_request_estimate_matches_manifest_protocol():
    result = estimate(manifest(), split="probe")
    # K=1; 1 execution + 1 semantic => 3 first-attempt calls, 6 retry calls.
    assert result["arms"]["no_retry"]["scheduled_calls"] == 3
    assert result["arms"]["snapshot"]["scheduled_calls"] == 9
    assert result["arms"]["dirty"]["scheduled_calls"] == 9
    assert result["arms"]["snapshot"]["output_tokens_hard_cap"] == 9 * 4096
    assert result["arms"]["snapshot"]["estimated_input_tokens"] < result["arms"]["dirty"]["estimated_input_tokens"]
    with pytest.raises(ValueError):
        estimate(manifest(), split="probe", max_tokens=1024)
