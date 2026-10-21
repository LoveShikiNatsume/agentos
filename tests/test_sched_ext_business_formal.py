from __future__ import annotations

import pytest

from agentos.benchmarks.sched_ext_business_formal import (
    FORMAL_MODES,
    FORMAL_ROUNDS,
    PRIMARY_BASELINE,
    PRIMARY_CANDIDATE,
    WARMUP_ROUNDS,
    build_formal_plan,
    evaluate_primary_gate,
    formal_round_seed,
)
from scripts import run_sched_ext_business_formal as formal_runner


def test_formal_plan_freezes_warmup_formal_counts_and_rotation() -> None:
    warmup = build_formal_plan("warmup")
    formal = build_formal_plan("formal")

    assert len(warmup) == WARMUP_ROUNDS * len(FORMAL_MODES) == 8
    assert len(formal) == FORMAL_ROUNDS * len(FORMAL_MODES) == 80
    assert [item["mode"] for item in formal[:4]] == list(FORMAL_MODES)
    assert formal[4]["mode"] == FORMAL_MODES[1]
    assert formal[-1]["round"] == 19
    assert {item["sample_kind"] for item in warmup} == {"warmup"}
    assert {item["sample_kind"] for item in formal} == {"formal"}
    with pytest.raises(ValueError, match="unsupported formal sample kind"):
        build_formal_plan("statistical")


def test_formal_round_seed_pairs_modes_and_separates_warmup() -> None:
    warmup = build_formal_plan("warmup")
    formal = build_formal_plan("formal")

    assert {
        formal_round_seed(20260721, int(item["round"]), sample_kind="warmup")
        for item in warmup
        if item["round"] == 0
    } == {20260521}
    assert {
        formal_round_seed(20260721, int(item["round"]), sample_kind="formal")
        for item in formal
        if item["round"] == 0
    } == {20260721}
    assert formal_round_seed(20260721, 19, sample_kind="formal") == 20262621
    with pytest.raises(ValueError, match="out of range"):
        formal_round_seed(1, 20, sample_kind="formal")


def _formal_trials(
    *,
    cfs_p95: float = 100.0,
    candidate_p95: float = 75.0,
    candidate_tasks_per_second: float = 95.0,
) -> list[dict[str, float | int | str]]:
    trials: list[dict[str, float | int | str]] = []
    for round_id in range(FORMAL_ROUNDS):
        cfs = cfs_p95
        candidate = candidate_p95
        trials.extend(
            [
                {
                    "mode": PRIMARY_BASELINE,
                    "round": round_id,
                    "critical_ready_ms": cfs,
                    "tasks_per_second": 100.0,
                },
                {
                    "mode": "agentos-normal",
                    "round": round_id,
                    "critical_ready_ms": 110.0,
                    "tasks_per_second": 90.0,
                },
                {
                    "mode": PRIMARY_CANDIDATE,
                    "round": round_id,
                    "critical_ready_ms": candidate,
                    "tasks_per_second": candidate_tasks_per_second,
                },
                {
                    "mode": "agentos-full",
                    "round": round_id,
                    "critical_ready_ms": 105.0,
                    "tasks_per_second": 80.0,
                },
            ]
        )
    return trials


def test_primary_gate_accepts_frozen_boundaries_and_reports_secondary_modes() -> None:
    result = evaluate_primary_gate(_formal_trials())

    assert result["passed"] is True
    assert result["primary_candidate"] == PRIMARY_CANDIDATE
    assert result["p95_improvement_percent"] == pytest.approx(25.0)
    assert result["paired_ci95_low_percent"] > 0
    assert result["tasks_per_second_change_percent"] == pytest.approx(-5.0)
    assert set(result["secondary_modes_reported"]) == {
        "agentos-normal",
        "agentos-full",
    }


def test_primary_gate_rejects_any_failed_preregistered_threshold() -> None:
    assert evaluate_primary_gate(
        _formal_trials(candidate_p95=75.1)
    )["checks"]["p95_improvement_at_least_25_percent"] is False
    assert evaluate_primary_gate(
        _formal_trials(candidate_tasks_per_second=94.9)
    )["checks"]["tasks_per_second_regression_at_most_5_percent"] is False

    mixed = _formal_trials()
    for trial in mixed:
        if trial["mode"] == PRIMARY_CANDIDATE and int(trial["round"]) < 10:
            trial["critical_ready_ms"] = 125.0
    assert evaluate_primary_gate(mixed)["checks"]["paired_ci95_low_above_zero"] is False


def test_primary_gate_rejects_missing_or_duplicate_formal_pairs() -> None:
    missing = _formal_trials()[:-1]
    with pytest.raises(ValueError, match="formal trial plan is incomplete"):
        evaluate_primary_gate(missing)

    duplicate = _formal_trials()
    duplicate.append(dict(duplicate[0]))
    with pytest.raises(ValueError, match="formal trial plan is incomplete"):
        evaluate_primary_gate(duplicate)


def test_formal_execution_stops_on_first_runtime_evidence_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    calls: list[int] = []

    def fake_trial(**kwargs):
        round_id = int(kwargs["round_id"])
        calls.append(round_id)
        passed = round_id == 0
        trial = {
            "mode": kwargs["mode"],
            "round": round_id,
            "order": kwargs["order"],
            "seed": kwargs["seed"],
            "evidence_checks": {
                "tasks_complete": True,
                "worker_class": True,
                "queue_mapping": True,
                "dsq_consumption": passed,
                "no_residuals": True,
                "loader_exit": True,
            },
        }
        return trial, [], []

    monkeypatch.setattr(formal_runner, "_trial", fake_trial)
    trials: list[dict] = []
    events: list[dict] = []
    loader_events: list[dict] = []

    with pytest.raises(RuntimeError, match="dsq_consumption"):
        formal_runner._execute_plan(
            plan=[
                {"sample_kind": "formal", "round": 0, "order": 0, "mode": "cfs"},
                {
                    "sample_kind": "formal",
                    "round": 1,
                    "order": 0,
                    "mode": "agentos-full",
                },
                {
                    "sample_kind": "formal",
                    "round": 2,
                    "order": 0,
                    "mode": "agentos-static-3dsq",
                },
            ],
            fixture_root=tmp_path,
            manifest=object(),
            scheduler_binary=tmp_path / "scheduler",
            seed=20260721,
            trials=trials,
            events=events,
            loader_events=loader_events,
        )

    assert calls == [0, 1]
    assert len(trials) == 2
    assert trials[-1]["evidence_checks"]["dsq_consumption"] is False
