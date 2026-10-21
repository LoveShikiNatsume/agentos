from collections import defaultdict
import json

from agentos.benchmarks.stage5 import (
    GROUP_PROFILES,
    LANES,
    REQUIRED_METRICS,
    WORKLOAD_NAMES,
    aggregate_trials,
    build_trial_plan,
    run_trial,
    write_artifacts,
)
from scripts.verify_stage5_experiments import verify_directory
from scripts import run_stage5_experiments


def test_profiles_have_distinct_declared_capabilities():
    assert list(GROUP_PROFILES) == [
        "bare_fifo",
        "dag_fixed",
        "workload_aware",
        "feedback_control",
        "full_runtime",
    ]
    assert GROUP_PROFILES["bare_fifo"].dag_scheduler is False
    assert GROUP_PROFILES["dag_fixed"].dag_scheduler is True
    assert GROUP_PROFILES["dag_fixed"].workload_aware is False
    assert GROUP_PROFILES["workload_aware"].workload_aware is True
    assert GROUP_PROFILES["workload_aware"].feedback_control is False
    assert GROUP_PROFILES["feedback_control"].feedback_control is True
    assert GROUP_PROFILES["full_runtime"].context_refs is True
    assert GROUP_PROFILES["full_runtime"].adaptive_tuning is True
    assert GROUP_PROFILES["full_runtime"].fault_recovery is True


def test_trial_plan_is_complete_balanced_and_randomized():
    plan = build_trial_plan(
        workloads=WORKLOAD_NAMES,
        repetitions=10,
        seed=20260703,
    )

    assert len(plan) == 440
    assert len({entry.trial_id for entry in plan}) == len(plan)
    assert sum(entry.warmup for entry in plan) == 40
    measured = [entry for entry in plan if not entry.warmup]
    assert len(measured) == 400

    blocks = defaultdict(list)
    for entry in measured:
        blocks[(entry.workload, entry.lane, entry.repetition)].append(entry)

    assert len(blocks) == len(WORKLOAD_NAMES) * len(LANES) * 10
    orders = defaultdict(set)
    for (workload, lane, _repetition), entries in blocks.items():
        ordered = sorted(entries, key=lambda item: item.planned_position)
        groups = tuple(item.group for item in ordered)
        assert set(groups) == set(GROUP_PROFILES)
        assert [item.planned_position for item in ordered] == list(range(5))
        orders[(workload, lane)].add(groups)

    assert all(len(group_orders) >= 2 for group_orders in orders.values())


def test_trial_plan_is_reproducible_without_global_random_state():
    first = build_trial_plan(WORKLOAD_NAMES, repetitions=10, seed=17)
    second = build_trial_plan(WORKLOAD_NAMES, repetitions=10, seed=17)
    third = build_trial_plan(WORKLOAD_NAMES, repetitions=10, seed=18)

    assert first == second
    assert first != third


async def test_bare_and_full_runtime_trials_record_required_metrics():
    plan = build_trial_plan(
        workloads=["compile"],
        repetitions=1,
        seed=20260703,
    )
    entries = {
        entry.group: entry
        for entry in plan
        if not entry.warmup and entry.lane == "steady_state"
    }

    bare = await run_trial(entries["bare_fifo"], cache_latency_ms=0.1)
    full = await run_trial(entries["full_runtime"], cache_latency_ms=0.1)

    for result in (bare, full):
        assert result["trial_status"] == "recorded"
        assert result["metrics"]["makespan_secs"] > 0
        assert result["metrics"]["throughput_tasks_per_sec"] > 0
        assert 0 <= result["metrics"]["cache_hit_ratio"] <= 1
        assert 0 <= result["metrics"]["jain_fairness_index"] <= 1
        assert result["capabilities"] == GROUP_PROFILES[
            result["group"]
        ].to_dict()
        assert result["metric_sources"]["process_rss_peak_mb"]
    assert bare["evidence"]["scheduler_decisions"] == 0
    assert full["evidence"]["scheduler_decisions"] > 0
    assert full["evidence"]["kernel_publications"] > 0


async def test_dag_fixed_does_not_claim_kernel_publication():
    entry = next(
        item
        for item in build_trial_plan(["network"], repetitions=1, seed=3)
        if not item.warmup
        and item.lane == "steady_state"
        and item.group == "dag_fixed"
    )

    result = await run_trial(entry, cache_latency_ms=0.1)

    assert result["capabilities"]["kernel_adapter"] is False
    assert result["evidence"]["kernel_publications"] == 0


async def test_observation_and_feedback_profiles_distinguish_control():
    plan = build_trial_plan(
        workloads=["ai_inference"],
        repetitions=1,
        seed=7,
    )
    entries = {
        entry.group: entry
        for entry in plan
        if not entry.warmup and entry.lane == "steady_state"
    }

    observed = await run_trial(
        entries["workload_aware"], cache_latency_ms=0.1
    )
    controlled = await run_trial(
        entries["feedback_control"], cache_latency_ms=0.1
    )

    assert observed["evidence"]["pressure_trace_observed"] is True
    assert observed["evidence"]["pressure_control_applied"] is False
    assert controlled["evidence"]["pressure_trace_observed"] is True
    assert controlled["evidence"]["pressure_control_applied"] is True
    assert controlled["evidence"]["pressure_samples"] > 0
    assert observed["metrics"]["pressure_affected_decisions"] == 0
    assert controlled["metrics"]["pressure_affected_decisions"] > 0


def _synthetic_result(entry, value: float = 1.0):
    return {
        **entry.to_dict(),
        "actual_position": entry.planned_position,
        "backend": "cache-sim",
        "trial_status": "recorded",
        "workload_success": True,
        "error": None,
        "capabilities": GROUP_PROFILES[entry.group].to_dict(),
        "metrics": {metric: value for metric in REQUIRED_METRICS},
        "metric_sources": {"synthetic": "unit test"},
        "evidence": {
            "pressure_trace_observed": GROUP_PROFILES[
                entry.group
            ].pressure_observation,
            "pressure_control_applied": GROUP_PROFILES[
                entry.group
            ].feedback_control,
            "real_cgroup_control": False,
            "bpf_mode": "mock",
        },
        "events": [],
    }


def test_aggregate_describes_every_required_metric():
    plan = build_trial_plan(["compile"], repetitions=10, seed=11)
    trials = [
        _synthetic_result(entry, float(entry.repetition or 1))
        for entry in plan
    ]

    summary = aggregate_trials(trials, repetitions=10, seed=11)

    cell = summary["cells"]["compile"]["steady_state"]["bare_fifo"]
    assert cell["recorded_trials"] == 10
    assert cell["metrics"]["makespan_secs"]["n"] == 10
    assert cell["metrics"]["makespan_secs"]["mean"] == 5.5
    assert cell["metrics"]["makespan_secs"]["stddev"] > 0
    assert summary["metadata"]["warmup_count"] == 10
    assert summary["metadata"]["measured_trial_count"] == 100


def test_artifact_writer_preserves_raw_trials_and_events(tmp_path):
    plan = build_trial_plan(["compile"], repetitions=1, seed=11)
    trials = [_synthetic_result(entry) for entry in plan]
    summary = aggregate_trials(trials, repetitions=1, seed=11)

    write_artifacts(
        output_dir=tmp_path,
        plan=plan,
        trials=trials,
        summary=summary,
    )

    assert (tmp_path / "trial_plan.json").exists()
    assert len((tmp_path / "raw_trials.jsonl").read_text().splitlines()) == 20
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.csv").exists()
    assert "Student-t" in (tmp_path / "report.md").read_text()
    assert len(list((tmp_path / "events").glob("*.jsonl"))) == 20


def _write_complete_synthetic_experiment(tmp_path):
    plan = build_trial_plan(WORKLOAD_NAMES, repetitions=10, seed=20260703)
    trials = [
        _synthetic_result(
            entry,
            value=float((entry.repetition or 1) + entry.planned_position),
        )
        for entry in plan
    ]
    summary = aggregate_trials(
        trials,
        repetitions=10,
        seed=20260703,
    )
    write_artifacts(
        output_dir=tmp_path,
        plan=plan,
        trials=trials,
        summary=summary,
    )


def test_strict_verifier_accepts_complete_recomputable_evidence(tmp_path):
    _write_complete_synthetic_experiment(tmp_path)

    result = verify_directory(tmp_path)

    assert result["passed"] is True
    assert result["errors"] == []
    assert result["counts"]["measured_trials"] == 400


def test_strict_verifier_rejects_corrupted_summary_statistic(tmp_path):
    _write_complete_synthetic_experiment(tmp_path)
    summary_path = tmp_path / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["cells"]["compile"]["steady_state"]["bare_fifo"]["metrics"][
        "makespan_secs"
    ]["mean"] += 1
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = verify_directory(tmp_path)

    assert result["passed"] is False
    assert any("mean" in error for error in result["errors"])


def test_runner_rejects_formal_repetitions_below_ten(tmp_path, capsys):
    exit_code = run_stage5_experiments.main(
        [
            "--repetitions",
            "9",
            "--out",
            str(tmp_path),
        ]
    )

    assert exit_code == 2
    assert "at least 10" in capsys.readouterr().err
