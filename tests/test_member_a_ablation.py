from __future__ import annotations

import json

import pytest

from scripts.run_member_a_ablation import run_ablation


def test_member_a_ablation_runs_four_modes_and_writes_report(tmp_path):
    result = run_ablation(tmp_path / "ablation", iterations=1)

    assert result["iterations"] == 1
    assert list(result["modes"]) == [
        "dag_only",
        "observation_only",
        "feedback_control",
        "feedback_bpf_publish",
    ]
    assert result["comparisons"]["feedback_adds_cgroup_decisions"] is True
    assert result["comparisons"]["observation_adds_usage_samples"] is True
    assert result["comparisons"]["bpf_publish_records_kernel_events"] is True

    dag_only = result["modes"]["dag_only"]
    observation = result["modes"]["observation_only"]
    feedback = result["modes"]["feedback_control"]
    bpf = result["modes"]["feedback_bpf_publish"]

    assert dag_only["mean"]["usage_samples"] == 0
    assert observation["mean"]["usage_samples"] > 0
    assert feedback["mean"]["feedback_applied"] > 0
    assert feedback["mean"]["affected_decisions"] > 0
    assert bpf["mean"]["kernel_decision_published"] > 0

    summary_path = tmp_path / "ablation" / "summary.json"
    raw_path = tmp_path / "ablation" / "raw_samples.json"
    csv_path = tmp_path / "ablation" / "summary.csv"
    report_path = tmp_path / "ablation" / "report.md"
    assert summary_path.exists()
    assert raw_path.exists()
    assert csv_path.exists()
    assert report_path.exists()
    assert json.loads(summary_path.read_text(encoding="utf-8"))["iterations"] == 1
    assert "Member A Open/Closed Loop Ablation" in report_path.read_text(
        encoding="utf-8"
    )


def test_competition_mode_requires_at_least_twenty_iterations(tmp_path):
    with pytest.raises(ValueError, match="at least 20"):
        run_ablation(tmp_path / "ablation", iterations=19, competition=True)


def test_summary_contains_distribution_statistics(tmp_path):
    result = run_ablation(tmp_path / "ablation", iterations=2)

    elapsed = result["modes"]["dag_only"]["statistics"]["elapsed_secs"]
    assert elapsed["n"] == 2
    assert "p50" in elapsed
    assert "p95" in elapsed
    assert "ci95_low" in elapsed
