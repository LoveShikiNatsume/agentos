"""Schema validation for scheduler.decision.v2 events.

This module provides lightweight validation to ensure scheduler decision
events maintain a stable shape for downstream consumers (dashboard,
audit tools, defense scripts).
"""

from __future__ import annotations

from typing import Any

# Required top-level fields in scheduler.decision.v2
REQUIRED_DECISION_V2_FIELDS = {
    "schema_version",
    "policy",
    "workload_class",
    "policy_reason",
    "selected_task_ids",
    "blocked_task_ids",
    "deferred_task_ids",
    "ready_task_ids",
    "candidate_task_ids",
    "ordered_candidate_task_ids",
    "tasks",
    "fairness",
    "runtime_mode",
    "kernel_integration",
    "fallback_reason",
    "runtime_health",
    "sla",
    "risk",
    "backpressure",
    "safety",
}

# Valid values for kernel_integration
VALID_KERNEL_INTEGRATION_VALUES = {"mock", "probe", "sched_ext", "unavailable"}

VALID_QUEUE_CLASSES = {"latency", "normal", "batch"}
REQUIRED_SCHED_EXT_TASK_FIELDS = {
    "queue_class",
    "kernel_weight",
    "slice_us",
    "policy_generation",
}

# Valid values for pressure_level
VALID_PRESSURE_LEVELS = {"normal", "warning", "critical", "overloaded"}

# Valid values for policy
VALID_POLICY_VALUES = {
    "latency_first",
    "throughput_first",
    "cache_affinity",
    "fairness_balanced",
    "risk_aware",
}

# Valid values for SLA class
VALID_SLA_CLASSES = {"critical", "interactive", "batch", "best_effort"}

# Valid values for admission decision
VALID_ADMISSION_DECISIONS = {"allow", "defer", "degrade", "reject"}

# Valid values for backpressure signal
VALID_BACKPRESSURE_SIGNALS = {"none", "soft", "hard", "rejecting"}

# Required fields in backpressure sub-dict
REQUIRED_BACKPRESSURE_FIELDS = {
    "signal",
    "pressure_level",
    "defer_count",
    "degrade_count",
    "reject_count",
    "reasons",
}

# Required fields in safety sub-dict
REQUIRED_SAFETY_FIELDS = {
    "admission_counts",
    "rejected_task_ids",
    "safety_deferred_task_ids",
    "degraded_task_ids",
    "safety_violations",
}

# Required fields in runtime_health sub-dict
REQUIRED_RUNTIME_HEALTH_FIELDS = {
    "runtime_health_score",
    "queue_pressure",
    "resource_pressure",
    "fairness_pressure",
    "dependency_pressure",
    "pressure_level",
    "health_warnings",
}

# Required fields in sla sub-dict
REQUIRED_SLA_FIELDS = {
    "critical_count",
    "interactive_count",
    "batch_count",
    "best_effort_count",
    "deadline_task_count",
    "max_urgency_score",
}

# Required fields in risk sub-dict
REQUIRED_RISK_FIELDS = {
    "max_task_risk_score",
    "avg_task_risk_score",
    "high_risk_task_ids",
    "risk_warnings",
}

# Required fields in fairness sub-dict
REQUIRED_FAIRNESS_FIELDS = {
    "fairness_score",
    "starvation_risk",
    "max_wait_ms",
    "avg_wait_ms",
}

# Required fields in kernel_probe sub-dict (if present)
REQUIRED_KERNEL_PROBE_FIELDS = {
    "sched_ext",
    "bpf_map",
    "fallback_mode",
}

# Required fields in bpf_map sub-dict (if present)
REQUIRED_BPF_MAP_FIELDS = {
    "mode",
    "task_count",
    "tasks",
}


def validate_scheduler_decision_v2(decision: dict[str, Any]) -> list[str]:
    """Validate a scheduler.decision.v2 event.

    Parameters
    ----------
    decision:
        The decision dict (typically from ``scheduler.decision`` event data).

    Returns
    -------
    list[str]
        A list of error messages.  Empty list means valid.
    """
    errors: list[str] = []

    # Check schema version
    schema_version = decision.get("schema_version")
    if schema_version != "scheduler.decision.v2":
        errors.append(f"schema_version must be 'scheduler.decision.v2', got '{schema_version}'")

    # Check required top-level fields
    for field in REQUIRED_DECISION_V2_FIELDS:
        if field not in decision:
            errors.append(f"missing required field: {field}")

    # Validate runtime_mode
    runtime_mode = decision.get("runtime_mode")
    if runtime_mode is not None and runtime_mode != "user_space_runtime":
        errors.append(f"runtime_mode must be 'user_space_runtime', got '{runtime_mode}'")

    # Validate kernel_integration
    kernel_integration = decision.get("kernel_integration")
    if kernel_integration is not None and kernel_integration not in VALID_KERNEL_INTEGRATION_VALUES:
        errors.append(f"kernel_integration must be one of {VALID_KERNEL_INTEGRATION_VALUES}, got '{kernel_integration}'")
    if kernel_integration == "sched_ext":
        generation = decision.get("policy_generation")
        if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
            errors.append("policy_generation must be a positive integer for sched_ext")

    # Validate fairness sub-dict
    fairness = decision.get("fairness")
    if isinstance(fairness, dict):
        for field in REQUIRED_FAIRNESS_FIELDS:
            if field not in fairness:
                errors.append(f"missing required fairness field: {field}")
    elif fairness is not None:
        errors.append("fairness must be a dict or None")

    # Validate kernel_probe sub-dict (if present)
    kernel_probe = decision.get("kernel_probe")
    if kernel_probe is not None:
        if isinstance(kernel_probe, dict):
            for field in REQUIRED_KERNEL_PROBE_FIELDS:
                if field not in kernel_probe:
                    errors.append(f"missing required kernel_probe field: {field}")
        else:
            errors.append("kernel_probe must be a dict or None")

    # Validate bpf_map sub-dict (if present)
    bpf_map = decision.get("bpf_map")
    if bpf_map is not None:
        if isinstance(bpf_map, dict):
            for field in REQUIRED_BPF_MAP_FIELDS:
                if field not in bpf_map:
                    errors.append(f"missing required bpf_map field: {field}")
        else:
            errors.append("bpf_map must be a dict or None")

    # Validate runtime_health sub-dict (required, must be dict)
    runtime_health = decision.get("runtime_health")
    if runtime_health is None:
        errors.append("runtime_health must be a dict, got None")
    elif not isinstance(runtime_health, dict):
        errors.append(f"runtime_health must be a dict, got {type(runtime_health).__name__}")
    else:
        for field in REQUIRED_RUNTIME_HEALTH_FIELDS:
            if field not in runtime_health:
                errors.append(f"missing required runtime_health field: {field}")

        # Validate runtime_health_score range
        score = runtime_health.get("runtime_health_score")
        if score is not None and (not isinstance(score, (int, float)) or score < 0.0 or score > 1.0):
            errors.append(f"runtime_health_score must be a number in [0.0, 1.0], got {score}")

        # Validate pressure dimensions range
        for dim in ["queue_pressure", "resource_pressure", "fairness_pressure", "dependency_pressure"]:
            val = runtime_health.get(dim)
            if val is not None and (not isinstance(val, (int, float)) or val < 0.0 or val > 1.0):
                errors.append(f"{dim} must be a number in [0.0, 1.0], got {val}")

        # Validate pressure_level
        pressure_level = runtime_health.get("pressure_level")
        if pressure_level is not None and pressure_level not in VALID_PRESSURE_LEVELS:
            errors.append(f"pressure_level must be one of {VALID_PRESSURE_LEVELS}, got '{pressure_level}'")

        # Validate health_warnings is a list
        health_warnings = runtime_health.get("health_warnings")
        if health_warnings is not None and not isinstance(health_warnings, list):
            errors.append(f"health_warnings must be a list, got {type(health_warnings).__name__}")

    # Validate policy value
    policy = decision.get("policy")
    if policy is not None and policy not in VALID_POLICY_VALUES:
        errors.append(f"policy must be one of {VALID_POLICY_VALUES}, got '{policy}'")

    # Validate sla sub-dict
    sla = decision.get("sla")
    if sla is None:
        errors.append("sla must be a dict, got None")
    elif not isinstance(sla, dict):
        errors.append(f"sla must be a dict, got {type(sla).__name__}")
    else:
        for field in REQUIRED_SLA_FIELDS:
            if field not in sla:
                errors.append(f"missing required sla field: {field}")

    # Validate risk sub-dict
    risk = decision.get("risk")
    if risk is None:
        errors.append("risk must be a dict, got None")
    elif not isinstance(risk, dict):
        errors.append(f"risk must be a dict, got {type(risk).__name__}")
    else:
        for field in REQUIRED_RISK_FIELDS:
            if field not in risk:
                errors.append(f"missing required risk field: {field}")

    # Validate backpressure sub-dict
    backpressure = decision.get("backpressure")
    if backpressure is None:
        errors.append("backpressure must be a dict, got None")
    elif not isinstance(backpressure, dict):
        errors.append(f"backpressure must be a dict, got {type(backpressure).__name__}")
    else:
        for field in REQUIRED_BACKPRESSURE_FIELDS:
            if field not in backpressure:
                errors.append(f"missing required backpressure field: {field}")
        # Validate signal
        signal = backpressure.get("signal")
        if signal is not None and signal not in VALID_BACKPRESSURE_SIGNALS:
            errors.append(f"backpressure signal must be one of {VALID_BACKPRESSURE_SIGNALS}, got '{signal}'")

    # Validate safety sub-dict
    safety = decision.get("safety")
    if safety is None:
        errors.append("safety must be a dict, got None")
    elif not isinstance(safety, dict):
        errors.append(f"safety must be a dict, got {type(safety).__name__}")
    else:
        for field in REQUIRED_SAFETY_FIELDS:
            if field not in safety:
                errors.append(f"missing required safety field: {field}")

    # Validate task-level fields
    tasks = decision.get("tasks", {})
    if isinstance(tasks, dict):
        for task_id, task_data in tasks.items():
            if not isinstance(task_data, dict):
                continue
            # Required task-level fields
            for required_field in ("task_id", "reason", "workload_class", "policy"):
                if required_field not in task_data:
                    errors.append(f"task {task_id}: missing required field '{required_field}'")
            if kernel_integration == "sched_ext":
                for required_field in REQUIRED_SCHED_EXT_TASK_FIELDS:
                    if required_field not in task_data:
                        errors.append(
                            f"task {task_id}: missing required sched_ext field "
                            f"'{required_field}'"
                        )
                queue_class = task_data.get("queue_class")
                if queue_class not in VALID_QUEUE_CLASSES:
                    errors.append(
                        f"task {task_id}: queue_class must be one of "
                        f"{VALID_QUEUE_CLASSES}, got '{queue_class}'"
                    )
                kernel_weight = task_data.get("kernel_weight")
                if (
                    not isinstance(kernel_weight, int)
                    or isinstance(kernel_weight, bool)
                    or not 1 <= kernel_weight <= 10_000
                ):
                    errors.append(
                        f"task {task_id}: kernel_weight must be in 1..10000"
                    )
                slice_us = task_data.get("slice_us")
                if (
                    not isinstance(slice_us, int)
                    or isinstance(slice_us, bool)
                    or not 100 <= slice_us <= 1_000_000
                ):
                    errors.append(
                        f"task {task_id}: slice_us must be in 100..1000000"
                    )
                if task_data.get("policy_generation") != generation:
                    errors.append(
                        f"task {task_id}: policy_generation must match decision"
                    )
            # Validate sla_class
            sla_class = task_data.get("sla_class")
            if sla_class is not None and sla_class not in VALID_SLA_CLASSES:
                errors.append(f"task {task_id}: sla_class must be one of {VALID_SLA_CLASSES}, got '{sla_class}'")
            # Validate urgency_score range
            urgency_score = task_data.get("urgency_score")
            if urgency_score is not None and (not isinstance(urgency_score, (int, float)) or urgency_score < 0.0 or urgency_score > 1.0):
                errors.append(f"task {task_id}: urgency_score must be a number in [0.0, 1.0], got {urgency_score}")
            # Validate task_risk_score range
            task_risk_score = task_data.get("task_risk_score")
            if task_risk_score is not None and (not isinstance(task_risk_score, (int, float)) or task_risk_score < 0.0 or task_risk_score > 1.0):
                errors.append(f"task {task_id}: task_risk_score must be a number in [0.0, 1.0], got {task_risk_score}")
            # Validate non-negative fields
            for nonneg_field in ("wait_ms", "deferred_count", "blocked_count"):
                val = task_data.get(nonneg_field)
                if val is not None and isinstance(val, (int, float)) and val < 0:
                    errors.append(f"task {task_id}: {nonneg_field} must be >= 0, got {val}")
            # Validate admission_decision
            admission_decision = task_data.get("admission_decision")
            if admission_decision is not None and admission_decision not in VALID_ADMISSION_DECISIONS:
                errors.append(f"task {task_id}: admission_decision must be one of {VALID_ADMISSION_DECISIONS}, got '{admission_decision}'")
            # Validate admission_decision + reason consistency
            reason = task_data.get("reason", "")
            if admission_decision == "reject" and reason != "rejected_by_safety_envelope":
                errors.append(f"task {task_id}: admission_decision=reject but reason='{reason}' (expected 'rejected_by_safety_envelope')")
            if admission_decision == "defer" and reason != "deferred_by_backpressure":
                errors.append(f"task {task_id}: admission_decision=defer but reason='{reason}' (expected 'deferred_by_backpressure')")
            # Validate backpressure_signal
            bp_signal = task_data.get("backpressure_signal")
            if bp_signal is not None and bp_signal not in VALID_BACKPRESSURE_SIGNALS:
                errors.append(f"task {task_id}: backpressure_signal must be one of {VALID_BACKPRESSURE_SIGNALS}, got '{bp_signal}'")
            # Validate safety_violations is a list
            safety_violations = task_data.get("safety_violations")
            if safety_violations is not None and not isinstance(safety_violations, list):
                errors.append(f"task {task_id}: safety_violations must be a list, got {type(safety_violations).__name__}")

    # Validate adaptive_tuning sub-dict (if present)
    adaptive_tuning = decision.get("adaptive_tuning")
    if adaptive_tuning is not None:
        if not isinstance(adaptive_tuning, dict):
            errors.append(f"adaptive_tuning must be a dict, got {type(adaptive_tuning).__name__}")
        else:
            # enabled must be bool
            enabled = adaptive_tuning.get("enabled")
            if enabled is not None and not isinstance(enabled, bool):
                errors.append(f"adaptive_tuning.enabled must be a bool, got {type(enabled).__name__}")
            # policy_bias must be dict
            policy_bias = adaptive_tuning.get("policy_bias")
            if policy_bias is not None and not isinstance(policy_bias, dict):
                errors.append(f"adaptive_tuning.policy_bias must be a dict, got {type(policy_bias).__name__}")
            # reasons must be list
            reasons = adaptive_tuning.get("reasons")
            if reasons is not None and not isinstance(reasons, list):
                errors.append(f"adaptive_tuning.reasons must be a list, got {type(reasons).__name__}")
            # observed_decisions >= 0
            observed_decisions = adaptive_tuning.get("observed_decisions")
            if observed_decisions is not None and isinstance(observed_decisions, (int, float)) and observed_decisions < 0:
                errors.append(f"adaptive_tuning.observed_decisions must be >= 0, got {observed_decisions}")

    # Validate selected/blocked/deferred set semantics
    selected = set(decision.get("selected_task_ids", []))
    blocked = set(decision.get("blocked_task_ids", []))
    deferred = set(decision.get("deferred_task_ids", []))
    candidates = set(decision.get("candidate_task_ids", []))

    if not selected.issubset(candidates) and candidates:
        errors.append("selected_task_ids must be a subset of candidate_task_ids")
    if selected & blocked:
        errors.append("selected_task_ids and blocked_task_ids must not overlap")
    if selected & deferred:
        errors.append("selected_task_ids and deferred_task_ids must not overlap")
    if blocked & deferred:
        errors.append("blocked_task_ids and deferred_task_ids must not overlap")

    # Rejected tasks should not appear in selected or deferred
    safety = decision.get("safety", {})
    if isinstance(safety, dict):
        rejected = set(safety.get("rejected_task_ids", []))
        if rejected & selected:
            errors.append("rejected tasks must not appear in selected_task_ids")
        if rejected & deferred:
            errors.append("rejected tasks must not appear in deferred_task_ids")

    return errors


def validate_scheduler_event(event: dict[str, Any]) -> list[str]:
    """Validate a scheduler event (event_type + data wrapper).

    Parameters
    ----------
    event:
        The event dict with ``event_type`` and ``data`` keys.

    Returns
    -------
    list[str]
        A list of error messages.  Empty list means valid.
    """
    errors: list[str] = []

    event_type = event.get("event_type")
    if event_type != "scheduler.decision":
        errors.append(f"event_type must be 'scheduler.decision', got '{event_type}'")
        return errors

    data = event.get("data")
    if not isinstance(data, dict):
        errors.append("event data must be a dict")
        return errors

    return validate_scheduler_decision_v2(data)
