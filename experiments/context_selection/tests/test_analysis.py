"""Question-level analysis tests; no backend, network, or GPU is involved."""
from copy import deepcopy
import json

import pytest

from experiments.context_selection.analyze import (
    analyze, load_summaries, main, render_markdown, validate_summaries,
)


def summaries():
    result = {}
    for arm, tokens in (("all", 100), ("declared", 40)):
        result[arm] = {
            "schema_version": 1, "arm": arm, "split": "probe",
            "manifest_sha256": "manifest-frozen", "plans_sha256": "plans-frozen",
            "config": {"model": "stub", "seed": 19, "repeats": 2,
                       "temperature": 0, "max_tokens": 2048, "backend": "stub"},
            "planning": {"model_calls": 2, "prompt_tokens": 20, "completion_tokens": 8,
                         "cached_tokens": 0, "truncated_calls": 0},
            "questions": [
                {"question_id": qid, "replicate": rep, "valid": True,
                 "correct": True, "prompt_tokens": tokens, "completion_tokens": 5,
                 "cached_tokens": 0, "model_calls": 2, "truncated_calls": 0,
                 "execution_truncated_calls": 0, "planner_truncated_calls": 0,
                 "failures": [], "required_recall": 1.0, "complete_required": True}
                for qid in ("q1", "q2") for rep in (0, 1)
            ],
        }
    return result


def test_paired_bootstrap_is_deterministic_and_averages_replicates_first():
    data = summaries()
    # Each question has an accuracy difference of exactly -0.5. Sampling runs
    # instead of question means would produce a non-degenerate interval.
    for row in data["declared"]["questions"]:
        row["correct"] = row["replicate"] == 0
    report = analyze(data, bootstrap_samples=500, seed=43)
    assert report == analyze(deepcopy(data), bootstrap_samples=500, seed=43)
    paired = report["comparisons"]["declared"]
    assert paired["accuracy"]["difference"] == -.5
    assert paired["accuracy"]["ci95"] == [-.5, -.5]
    assert paired["prompt_tokens"]["difference"] == -60
    assert paired["prompt_tokens"]["ci95"] == [-60, -60]
    assert paired["prompt_tokens"]["relative_change"] == pytest.approx(-.6)
    assert paired["paired_valid_coverage"] == 1
    assert report["arms"]["declared"]["execution"]["prompt_tokens"] == 160
    assert report["arms"]["declared"]["including_planning"]["prompt_tokens"] == 180


def test_failed_nontruncated_questions_remain_in_itt_and_details():
    data = summaries()
    row = data["declared"]["questions"][0]
    row.update(valid=False, correct=True, failures=[{"kind": "runtime", "message": "stopped"}])
    report = analyze(data, bootstrap_samples=100)
    arm = report["arms"]["declared"]
    assert arm["itt_questions"] == 2
    assert arm["itt_accuracy"] == .75
    assert arm["valid_questions"] == 1
    assert arm["valid_accuracy"] == 1
    assert len(arm["failures"]) == 1
    assert len(arm["invalid"]) == 1
    assert report["comparisons"]["declared"]["paired_valid_question_ids"] == ["q2"]
    assert report["comparisons"]["declared"]["paired_valid_coverage"] == .5


def test_json_answer_error_is_wrong_but_stays_valid():
    data = summaries()
    data["declared"]["questions"][0].update(correct=False, failures=[{"kind": "answer_format"}])
    report = analyze(data, bootstrap_samples=50)
    arm = report["arms"]["declared"]
    assert arm["itt_accuracy"] == .75
    assert arm["valid_accuracy"] == .75
    assert arm["valid_questions"] == 2
    assert not arm["invalid"]
    assert len(arm["failures"]) == 1


def test_planner_truncation_excludes_entire_question_without_duplicate_request_count():
    data = summaries()
    for summary in data.values():
        summary["planning"]["truncated_calls"] = 1
        summary["planning"]["failures"] = [{"question_id": "q1", "kind": "length"}]
        for row in summary["questions"][:2]:
            row.update(valid=False, truncated_calls=1, planner_truncated_calls=1)
    report = analyze(data, bootstrap_samples=50)
    arm = report["arms"]["declared"]
    assert arm["itt_questions"] == 1
    assert arm["valid_questions"] == 1
    assert arm["truncation"]["questions"] == 1
    assert arm["truncation"]["runs"] == 2
    assert arm["truncation"]["execution_calls"] == 0
    assert arm["truncation"]["execution_rate"] == 0
    assert arm["truncation"]["planning_calls"] == 1
    assert arm["planning"]["failures"] == [{"question_id": "q1", "kind": "length"}]
    assert arm["truncation"]["planning_rate"] == .5
    assert arm["including_planning"]["model_calls"] == 10
    assert report["comparisons"]["declared"]["excluded_question_ids"] == ["q1"]


def test_one_execution_truncation_excludes_other_replicates_for_same_question():
    data = summaries()
    data["declared"]["questions"][0].update(truncated_calls=1, execution_truncated_calls=1)
    report = analyze(data, bootstrap_samples=50)
    arm = report["arms"]["declared"]
    assert arm["itt_questions"] == 1
    assert arm["truncation"]["execution_rate"] == 1 / 8
    assert len(arm["invalid"]) == 1


def test_missing_usage_is_not_zero_and_reduces_token_pair_coverage_only():
    data = summaries()
    data["declared"]["questions"][0]["prompt_tokens"] = None
    report = analyze(data, bootstrap_samples=50)
    arm = report["arms"]["declared"]
    assert arm["execution"]["prompt_tokens"] is None
    assert arm["execution"]["known_prompt_tokens"] == 120
    assert arm["execution"]["missing_prompt_tokens_runs"] == 1
    assert arm["including_planning"]["prompt_tokens"] is None
    comparison = report["comparisons"]["declared"]
    assert comparison["paired_valid_questions"] == 2
    assert comparison["accuracy"]["questions"] == 2
    assert comparison["prompt_tokens"]["questions"] == 1
    assert comparison["prompt_usage_coverage"] == .5


def test_required_document_strata_use_question_means():
    data = summaries()
    for row in data["declared"]["questions"][:2]:
        row.update(complete_required=False, required_recall=.5, correct=False)
    report = analyze(data, bootstrap_samples=50)
    arm = report["arms"]["declared"]
    assert arm["mean_required_recall"] == .75
    assert arm["strata"]["complete"]["itt_accuracy"] == 1
    assert arm["strata"]["incomplete"]["itt_accuracy"] == 0
    assert arm["strata"]["unknown"]["itt_accuracy"] is None


@pytest.mark.parametrize("field", ["manifest_sha256", "plans_sha256", "split"])
def test_rejects_mismatched_experiment_identity(field):
    data = summaries()
    data["declared"][field] = "different"
    with pytest.raises(ValueError, match=field):
        validate_summaries(data)


@pytest.mark.parametrize("field,value", [("model", "other"), ("seed", 42), ("repeats", 3),
                                          ("max_tokens", 4096), ("temperature", 1), ("backend", "real")])
def test_rejects_mismatched_config(field, value):
    data = summaries()
    data["declared"]["config"][field] = value
    with pytest.raises(ValueError, match=field):
        validate_summaries(data)


def test_rejects_duplicate_and_missing_replicates():
    data = summaries()
    data["declared"]["questions"].append(deepcopy(data["declared"]["questions"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        validate_summaries(data)
    data = summaries()
    data["declared"]["questions"].pop()
    with pytest.raises(ValueError, match="replicate"):
        validate_summaries(data)


def test_requires_all_and_two_arms_and_rejects_conflicting_planner_cost():
    data = summaries()
    with pytest.raises(ValueError, match="all baseline"):
        validate_summaries({"declared": data["declared"]})
    data["declared"]["planning"]["prompt_tokens"] += 1
    with pytest.raises(ValueError, match="shared planning"):
        validate_summaries(data)


def test_empty_paired_subset_is_explicit_and_json_serializable():
    data = summaries()
    for row in data["declared"]["questions"]:
        row.update(valid=False, failures=["transport failure"])
    report = analyze(data, bootstrap_samples=50)
    comparison = report["comparisons"]["declared"]
    assert comparison["paired_valid_coverage"] == 0
    assert comparison["accuracy"]["ci95"] is None
    assert report["arms"]["declared"]["itt_accuracy"] == 0
    json.dumps(report, allow_nan=False)
    assert "NA" in render_markdown(report)


def test_cli_discovers_roots_and_writes_json_and_markdown(tmp_path):
    data = summaries()
    root = tmp_path / "runs"
    for arm, summary in data.items():
        path = root / arm / "summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(summary), encoding="utf-8")
    assert load_summaries([root]) == data
    out = tmp_path / "analysis"
    assert main(["--input", str(root), "--output", str(out), "--bootstrap-samples", "50", "--seed", "9"]) == 0
    result = json.loads((out / "analysis.json").read_text())
    assert result["bootstrap"]["seed"] == 9
    markdown = (out / "analysis.md").read_text()
    assert "| declared |" in markdown
    assert "Known input / missing runs" in markdown
    assert "Required-document selection" in markdown
    assert load_summaries([root / "all", root / "declared" / "summary.json"]) == data


def test_duplicate_arm_files_are_rejected(tmp_path):
    summary = summaries()["all"]
    for name in ("first.json", "second.json"):
        (tmp_path / name).write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate arm"):
        load_summaries([tmp_path / "first.json", tmp_path / "second.json"])
