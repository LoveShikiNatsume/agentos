"""Stage 5 sched_ext reliability matrix and acceptance rules."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


AUTO_FAULT_CASES = (
    "loader_sigkill",
    "loader_sigterm",
    "sysrq_unload",
    "heartbeat_timeout",
    "schema_mismatch",
    "policy_map_unpin",
    "policy_map_freeze",
    "worker_fork",
    "worker_failure",
    "worker_sigkill",
    "worker_timeout",
    "runtime_cancel",
    "single_cpu_cpuset",
    "cfs_competition",
    "rt_coexistence",
)

STRICT_FAULT_CASES = (
    "loader_sigkill",
    "heartbeat_timeout",
    "schema_mismatch",
    "policy_map_freeze",
)

AUTO_FALLBACK_CASES = frozenset(
    {
        "loader_sigkill",
        "loader_sigterm",
        "sysrq_unload",
        "heartbeat_timeout",
        "schema_mismatch",
        "policy_map_unpin",
        "policy_map_freeze",
        "cfs_competition",
    }
)
AUTO_RECOVERY_CASES = AUTO_FALLBACK_CASES
TIMEOUT_CASES = frozenset({"heartbeat_timeout", "policy_map_freeze"})


def summarize_trials(trials: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(trials)
    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        key = f"{record.get('mode')}:{record.get('case')}"
        item = grouped.setdefault(
            key,
            {
                "mode": record.get("mode"),
                "case": record.get("case"),
                "attempted": 0,
                "passed": 0,
                "task_loss": 0,
                "fallback_events": 0,
                "recovered_events": 0,
                "detection_max_ms": 0.0,
                "recovery_max_ms": 0.0,
            },
        )
        item["attempted"] += 1
        item["passed"] += int(record.get("passed") is True)
        item["task_loss"] += int(record.get("task_loss", 0))
        item["fallback_events"] += int(record.get("fallback_events", 0))
        item["recovered_events"] += int(record.get("recovered_events", 0))
        item["detection_max_ms"] = max(
            item["detection_max_ms"], float(record.get("detection_ms", 0.0))
        )
        item["recovery_max_ms"] = max(
            item["recovery_max_ms"], float(record.get("recovery_ms", 0.0))
        )
    return {
        "trials": len(records),
        "passed": sum(record.get("passed") is True for record in records),
        "outcomes": dict(Counter(str(record.get("outcome")) for record in records)),
        "cases": grouped,
    }


def acceptance_passed(
    summary: dict[str, Any],
    *,
    expected_iterations: int,
    require_soak: bool,
) -> bool:
    cases = summary.get("cases", {})
    expected = {
        *(f"auto:{case}" for case in AUTO_FAULT_CASES),
        *(f"strict:{case}" for case in STRICT_FAULT_CASES),
    }
    return bool(
        set(cases) == expected
        and all(
            item.get("attempted") == expected_iterations
            and item.get("passed") == expected_iterations
            and item.get("task_loss") == 0
            for item in cases.values()
        )
        and all(
            item.get("fallback_events") == expected_iterations
            for key, item in cases.items()
            if (key.startswith("auto:") and item.get("case") in AUTO_FALLBACK_CASES)
            or key.startswith("strict:")
        )
        and all(
            item.get("recovered_events") == expected_iterations
            for key, item in cases.items()
            if key.startswith("auto:") and item.get("case") in AUTO_RECOVERY_CASES
        )
        and all(
            item.get("detection_max_ms", 0) <= (
                4_000 if item.get("case") in TIMEOUT_CASES else 2_000
            )
            for item in cases.values()
        )
        and all(
            item.get("recovery_max_ms", 0) <= 2_000
            for key, item in cases.items()
            if key.startswith("auto:") and item.get("case") in AUTO_RECOVERY_CASES
        )
        and summary.get("policy_leaks") == 0
        and summary.get("task_state_leaks") == 0
        and summary.get("residual_pinned_maps") == []
        and summary.get("residual_bpf_links") == []
        and summary.get("residual_processes") == []
        and summary.get("residual_cgroups") == []
        and summary.get("kernel_log_errors") == []
        and summary.get("systemd_failed_units") == 0
        and summary.get("final_sched_ext_state") == "disabled"
        and summary.get("failure_reason") is None
        and (
            not require_soak
            or (
                summary.get("soak", {}).get("completed") is True
                and summary.get("soak", {}).get("duration_seconds", 0) >= 3600
                and summary.get("soak", {}).get("task_loss") == 0
            )
        )
    )
