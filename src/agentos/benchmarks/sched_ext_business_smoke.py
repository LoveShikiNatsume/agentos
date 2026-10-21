"""Frozen plan and evidence checks for the authorized P2 openEuler smoke."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agentos.scenarios.sched_ext_business.pipeline import BUSINESS_TASK_IDS


SMOKE_MODES = (
    "cfs",
    "agentos-normal",
    "agentos-static-3dsq",
    "agentos-full",
)
SMOKE_PHASE_ROUNDS = {"functional": 1, "statistical": 3}

_THREE_DSQ_QUEUES = {
    task_id: (
        "latency"
        if task_id in {"patch_validation", "security_gate"}
        else "normal"
        if task_id in {"unit_test_select", "dependency_analysis"}
        else "batch"
    )
    for task_id in BUSINESS_TASK_IDS
}


def build_smoke_plan(phase: str) -> list[dict[str, int | str]]:
    """Return a fixed rotating plan; no API exists for 20-round formal runs."""

    rounds = SMOKE_PHASE_ROUNDS.get(phase)
    if rounds is None:
        raise ValueError(f"unsupported smoke phase: {phase!r}")
    plan: list[dict[str, int | str]] = []
    for round_id in range(rounds):
        rotated = SMOKE_MODES[round_id:] + SMOKE_MODES[:round_id]
        for order, mode in enumerate(rotated):
            plan.append(
                {"phase": phase, "round": round_id, "order": order, "mode": mode}
            )
    return plan


def has_complete_mode_round_pairs(
    trials: list[Mapping[str, Any]],
    phase: str,
) -> bool:
    expected = {
        (str(item["mode"]), int(item["round"]))
        for item in build_smoke_plan(phase)
    }
    actual = {
        (str(trial.get("mode")), int(trial.get("round", -1)))
        for trial in trials
    }
    return actual >= expected


def smoke_round_seed(base_seed: int, round_id: int) -> int:
    """Use one pressure seed for every mode in the same paired round."""

    if round_id < 0:
        raise ValueError("round_id must be non-negative")
    return int(base_seed) + round_id * 100


def expected_policy_queues(mode: str) -> dict[str, str]:
    if mode == "agentos-normal":
        return {task_id: "normal" for task_id in BUSINESS_TASK_IDS}
    if mode in {"agentos-static-3dsq", "agentos-full"}:
        return dict(_THREE_DSQ_QUEUES)
    if mode == "cfs":
        return {}
    raise ValueError(f"unsupported smoke mode: {mode!r}")


def _worker_observations(trial: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_results = trial.get("task_results")
    if not isinstance(raw_results, Mapping):
        return []
    observations: list[Mapping[str, Any]] = []
    for task_id in BUSINESS_TASK_IDS:
        result = raw_results.get(task_id)
        if not isinstance(result, Mapping):
            return []
        observation = result.get("runtime_observation")
        if not isinstance(observation, Mapping):
            return []
        observations.append(observation)
    return observations


def assess_trial_evidence(trial: Mapping[str, Any]) -> dict[str, bool]:
    """Assess raw per-trial evidence without accepting precomputed booleans."""

    mode = str(trial.get("mode", ""))
    observations = _worker_observations(trial)
    agentos_mode = mode != "cfs"
    if agentos_mode:
        worker_class = len(observations) == len(BUSINESS_TASK_IDS) and all(
            item.get("policy") == 7
            and item.get("ext_enabled") == 1
            and item.get("affinity") == [1, 2]
            for item in observations
        )
    else:
        worker_class = len(observations) == len(BUSINESS_TASK_IDS) and all(
            item.get("policy") == 0
            and item.get("ext_enabled") == 0
            and item.get("affinity") == [1, 2]
            for item in observations
        )

    policy_queues = trial.get("policy_queues")
    associations = trial.get("policy_associations")
    if agentos_mode:
        queue_mapping = (
            isinstance(policy_queues, Mapping)
            and dict(policy_queues) == expected_policy_queues(mode)
            and associations == len(BUSINESS_TASK_IDS)
        )
    else:
        queue_mapping = associations == 0

    stats = trial.get("stats_delta")
    if not isinstance(stats, Mapping):
        stats = {}
    if mode == "cfs":
        dsq_consumption = not stats
    elif mode == "agentos-normal":
        dsq_consumption = (
            int(stats.get("policy_hit", 0)) > 0
            and int(stats.get("dispatch_normal", 0)) > 0
        )
    else:
        dsq_consumption = all(
            int(stats.get(name, 0)) > 0
            for name in (
                "policy_hit",
                "dispatch_latency",
                "dispatch_normal",
                "dispatch_batch",
            )
        )

    cleanup = trial.get("cleanup")
    if not isinstance(cleanup, Mapping):
        cleanup = {}
    no_residuals = (
        cleanup.get("workers") == 0
        and cleanup.get("cgroups") == 0
        and cleanup.get("policies") == 0
        and cleanup.get("task_states") == 0
        and cleanup.get("bound_pids") == 0
        and cleanup.get("loader") == 0
        and cleanup.get("sched_ext_state") == "disabled"
    )
    loader_ok = (
        trial.get("loader_exit_code") == 0
        if agentos_mode
        else trial.get("loader_exit_code") is None
    )
    return {
        "tasks_complete": (
            trial.get("completed_tasks") == len(BUSINESS_TASK_IDS)
            and trial.get("task_loss") == 0
        ),
        "worker_class": worker_class,
        "queue_mapping": queue_mapping,
        "dsq_consumption": dsq_consumption,
        "no_residuals": no_residuals,
        "loader_exit": loader_ok,
    }
