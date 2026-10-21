from __future__ import annotations

import time
from typing import Any, Callable, TYPE_CHECKING

from agentos.core.resources import ResourceQuota
from agentos.core.task import Task, TaskDAG, TaskStatus
from agentos.isolation.pressure import max_pressure_level
from agentos.isolation.sched_ext import KernelProbeResult
from agentos.scheduler.fairness import FairnessTracker, StarvationRisk
from agentos.scheduler.health import RuntimeHealthSummary, compute_runtime_health
from agentos.scheduler.policy import PolicyDecision, PolicySelector, SchedulingPolicy
from agentos.scheduler.adaptive import AdaptivePolicyTuner
from agentos.scheduler.risk import TaskRiskSummary, compute_task_risk
from agentos.scheduler.safety import (
    SafetyEnvelopeConfig,
    TaskSafetyDecision,
    evaluate_task_safety,
    summarize_safety,
)
from agentos.scheduler.workload import (
    WorkloadClass,
    WorkloadClassifier,
    WorkloadSample,
)

if TYPE_CHECKING:
    from agentos.scheduler.affinity import CacheAffinityAdvisor


class DAGScheduler:
    """Schedule a ready DAG with optional service-time fairness and aging.

    ``fairness_scale="absolute"`` retains the fixed wait/deferral limits.
    ``"service_time"`` uses queue budgets and observed batch wall times;
    standalone callers report completed batches via ``record_batch_service_time``.
    An injected fairness tracker must use the same scale as this scheduler.
    """

    def __init__(
        self,
        dag: TaskDAG,
        available_resources: ResourceQuota | None = None,
        max_batch_size: int | None = None,
        affinity_advisor: "CacheAffinityAdvisor | None" = None,
        resource_estimator: Callable[[Task], ResourceQuota | None] | None = None,
        workload_classifier: WorkloadClassifier | None = None,
        policy_selector: PolicySelector | None = None,
        fairness_tracker: FairnessTracker | None = None,
        kernel_probe: KernelProbeResult | None = None,
        safety_config: SafetyEnvelopeConfig | None = None,
        force_policy: SchedulingPolicy | str | None = None,
        adaptive_tuner: AdaptivePolicyTuner | None = None,
        pressure_level: str = "normal",
        pressure_sample: dict[str, Any] | None = None,
        fairness_scale: str = "absolute",
    ) -> None:
        if max_batch_size is not None and max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        self._dag = dag
        self._dispatched: set[str] = set()
        self._available_resources = available_resources
        self._max_batch_size = max_batch_size
        self._affinity_advisor = affinity_advisor
        self._resource_estimator = resource_estimator
        self._workload_classifier = workload_classifier or WorkloadClassifier()
        self._policy_selector = policy_selector or PolicySelector()
        if fairness_scale not in {"absolute", "service_time"}:
            raise ValueError("fairness_scale must be 'absolute' or 'service_time'")
        self._fairness_tracker = fairness_tracker or FairnessTracker(
            fairness_scale=fairness_scale,
        )
        if self._fairness_tracker.fairness_scale != fairness_scale:
            raise ValueError("fairness_scale must match the supplied fairness_tracker")
        self._fairness_scale = fairness_scale
        self._kernel_probe = kernel_probe
        self._safety_config = safety_config or SafetyEnvelopeConfig()
        self._last_decision: dict[str, Any] | None = None
        self._policy_generation = 0
        self._adaptive_tuner = adaptive_tuner

        # Ablation support: force a specific policy, bypassing PolicySelector
        if force_policy is not None and not isinstance(force_policy, SchedulingPolicy):
            force_policy = SchedulingPolicy(force_policy)
        self._force_policy: SchedulingPolicy | None = force_policy

        # Pressure state
        _VALID_LEVELS = {"normal", "warning", "critical", "overloaded"}
        if pressure_level not in _VALID_LEVELS:
            raise ValueError(f"invalid pressure level: {pressure_level!r}")
        self._os_pressure_level: str = pressure_level
        self._pressure_sample: dict[str, Any] | None = pressure_sample

    def record_batch_service_time(self, duration_ms: float) -> None:
        """Observe wall time of a nonempty, synchronously executed task batch."""
        self._fairness_tracker.record_service_time(duration_ms)

    def set_available_resources(self, resources: ResourceQuota | None) -> None:
        self._available_resources = resources

    def set_pressure(
        self,
        *,
        pressure_level: str,
        pressure_sample: dict[str, Any] | None = None,
    ) -> None:
        """Update the OS pressure level before the next scheduling round.

        Raises ``ValueError`` for invalid pressure levels.
        """
        _VALID_LEVELS = {"normal", "warning", "critical", "overloaded"}
        if pressure_level not in _VALID_LEVELS:
            raise ValueError(f"invalid pressure level: {pressure_level!r}")
        self._os_pressure_level = pressure_level
        self._pressure_sample = pressure_sample

    async def get_next_batch(self) -> list[Task]:
        self._dag.compute_critical_path_scores()
        ready = self._dag.get_ready_tasks()

        # Step 1: Fairness tracker observes ready tasks
        ready_task_ids = [task.task_id for task in ready if task.task_id not in self._dispatched]
        if self._fairness_scale == "service_time":
            self._fairness_tracker.observe_ready(
                ready_task_ids, batch_size=self._max_batch_size,
            )
        else:
            self._fairness_tracker.observe_ready(ready_task_ids)

        # Step 2: Resource filtering
        blocked = [
            task
            for task in ready
            if task.task_id not in self._dispatched and not self._fits_resources(task)
        ]
        candidates = [
            task
            for task in ready
            if task.task_id not in self._dispatched and self._fits_resources(task)
        ]

        # Step 3: Workload classification
        workload_samples = {
            task.task_id: self._workload_sample(task)
            for task in ready
            if task.task_id not in self._dispatched
        }
        workload_classes = {
            task_id: self._workload_classifier.classify(sample)
            for task_id, sample in workload_samples.items()
        }

        # Step 4: Fairness summary
        fairness_summary = self._fairness_tracker.summarize()

        # Step 4.5: Compute SLA/Risk aggregates for policy selection
        has_critical_sla = any(
            task.sla_class == "critical" for task in candidates
        )
        max_urgency_score = 0.0
        max_task_risk_score = 0.0
        for task in candidates:
            fairness_snapshot = self._fairness_tracker.task_snapshot(task.task_id)
            risk_summary = compute_task_risk(
                sla_class=task.sla_class,
                deadline_ms=task.deadline_ms,
                wait_ms=fairness_snapshot["wait_ms"],
                starvation_risk=fairness_snapshot["starvation_risk"],
                dependency_count=len(task.dependencies),
                unmet_dependency_count=sum(
                    1 for dep_id in task.dependencies
                    if dep_id in self._dag.tasks
                    and self._dag.tasks[dep_id].status != TaskStatus.COMPLETED
                ),
                previous_failures=task.previous_failures,
            )
            max_urgency_score = max(max_urgency_score, risk_summary.urgency_score)
            max_task_risk_score = max(max_task_risk_score, risk_summary.task_risk_score)

        # Compute fairness_pressure_level: low→normal, medium→warning, high→critical
        fairness_pressure_level = "normal"
        if fairness_summary.starvation_risk == StarvationRisk.HIGH:
            fairness_pressure_level = "critical"
        elif fairness_summary.starvation_risk == StarvationRisk.MEDIUM:
            fairness_pressure_level = "warning"

        # effective_pressure_level = max(OS, fairness)
        os_pressure_level = self._os_pressure_level
        effective_pressure_level = max_pressure_level(
            os_pressure_level, fairness_pressure_level
        )

        # PolicySelector receives OS pressure (not effective)
        pressure_level_for_policy = os_pressure_level

        # Step 4.6: Get adaptive bias if tuner is available
        adaptive_bias = None
        if self._adaptive_tuner is not None:
            adaptive_bias = self._adaptive_tuner.recommend_policy_bias()

        # Step 5: Policy selector with real starvation risk and SLA/Risk info
        if self._force_policy is not None:
            # Ablation mode: force a specific policy, bypass PolicySelector
            policy_decision = PolicyDecision(
                policy=self._force_policy,
                reason="forced by ablation experiment",
                workload_class=self._policy_selector._dominant_class(
                    [workload_samples[task.task_id] for task in candidates],
                    workload_classes,
                ),
                score_breakdown={
                    "forced_policy": True,
                    "starvation_risk": fairness_summary.starvation_risk.value,
                    "candidate_count": len(candidates),
                    "has_critical_sla": has_critical_sla,
                    "max_urgency_score": max_urgency_score,
                    "max_task_risk_score": max_task_risk_score,
                    "pressure_level": pressure_level_for_policy,
                },
            )
        else:
            policy_decision = self._policy_selector.select(
                [workload_samples[task.task_id] for task in candidates],
                workload_classes,
                starvation_risk=fairness_summary.starvation_risk,
                has_critical_sla=has_critical_sla,
                max_urgency_score=max_urgency_score,
                max_task_risk_score=max_task_risk_score,
                pressure_level=pressure_level_for_policy,
                # A learned preference cannot bypass the relative hard limit.
                adaptive_bias=(
                    None
                    if self._fairness_scale == "service_time"
                    and fairness_summary.starvation_risk == StarvationRisk.HIGH
                    else adaptive_bias
                ),
            )

        # Preserve the inputs used to choose the policy. The main fairness
        # summary is recomputed below after this round's deferrals are counted.
        policy_decision.score_breakdown["fairness"] = fairness_summary.to_dict()

        # Step 6: Policy-based sorting
        sorted_candidates = self._sort_candidates_by_policy(
            candidates,
            policy_decision,
        )

        # Step 7: Resource batch selection
        resource_fit_batch = self._select_resource_fit_batch(sorted_candidates)

        # Step 8: Max batch size truncation
        batch = resource_fit_batch
        if self._max_batch_size is not None:
            batch = batch[: self._max_batch_size]
        batch, pressure_control = self._apply_explicit_pressure_control(
            batch,
            pressure_sample=self._pressure_sample,
        )

        # Step 8.5: Safety evaluation - enforce admission control
        # Safety/Backpressure uses effective pressure (max of OS + fairness)
        pressure_level = effective_pressure_level

        # Evaluate safety for each task in batch
        safety_filtered_batch: list[Task] = []
        safety_rejected_ids: list[str] = []
        safety_deferred_ids: list[str] = []

        for task in batch:
            # Compute task risk for safety evaluation
            fairness_snapshot = self._fairness_tracker.task_snapshot(task.task_id)
            risk_summary = compute_task_risk(
                sla_class=task.sla_class,
                deadline_ms=task.deadline_ms,
                wait_ms=fairness_snapshot["wait_ms"],
                starvation_risk=fairness_snapshot["starvation_risk"],
                dependency_count=len(task.dependencies),
                unmet_dependency_count=sum(
                    1 for dep_id in task.dependencies
                    if dep_id in self._dag.tasks
                    and self._dag.tasks[dep_id].status != TaskStatus.COMPLETED
                ),
                previous_failures=task.previous_failures,
            )

            safety_decision = evaluate_task_safety(
                sla_class=task.sla_class or "batch",
                workload_class=workload_classes.get(task.task_id, WorkloadClass.UNKNOWN).value,
                pressure_level=pressure_level,
                task_risk_score=risk_summary.task_risk_score,
                previous_failures=task.previous_failures,
                memory_max_mb=task.resource_requirements.memory_max_mb,
                cpu_max_percent=task.resource_requirements.cpu_max_percent,
                policy=policy_decision.policy.value,
                config=self._safety_config,
            )

            if safety_decision.admission_decision == "reject":
                safety_rejected_ids.append(task.task_id)
            elif safety_decision.admission_decision == "defer":
                safety_deferred_ids.append(task.task_id)
                safety_filtered_batch.append(task)
            else:
                safety_filtered_batch.append(task)

        # Update batch to exclude rejected tasks
        batch = safety_filtered_batch

        # Mark rejected tasks as SKIPPED so they don't block completion
        for task_id in safety_rejected_ids:
            task = self._dag.tasks.get(task_id)
            if task is not None and task.status == TaskStatus.PENDING:
                task.status = TaskStatus.SKIPPED
                task.updated_at = time.time()
                task.error = "rejected_by_safety_envelope"

        # Step 9: Record fairness decision
        selected_ids = [task.task_id for task in batch]
        deferred_ids = [
            task.task_id for task in candidates
            if task.task_id not in set(selected_ids) and task.task_id not in set(safety_rejected_ids)
        ]
        blocked_ids = [task.task_id for task in blocked]
        self._fairness_tracker.record_decision(
            selected_ids=selected_ids,
            deferred_ids=deferred_ids,
            blocked_ids=blocked_ids,
        )

        # Step 10: Re-summarize after record_decision to reflect latest state
        fairness_summary = self._fairness_tracker.summarize()

        # Step 11: Build decision with updated fairness data
        self._policy_generation += 1
        self._last_decision = self._build_decision(
            ready=ready,
            candidates=candidates,
            sorted_candidates=sorted_candidates,
            resource_fit_batch=resource_fit_batch,
            selected=batch,
            blocked=blocked,
            policy_decision=policy_decision,
            workload_classes=workload_classes,
            fairness_summary=fairness_summary,
            safety_rejected_ids=safety_rejected_ids,
            safety_deferred_ids=safety_deferred_ids,
            adaptive_bias=adaptive_bias,
            os_pressure_level=os_pressure_level,
            fairness_pressure_level=fairness_pressure_level,
            effective_pressure_level=effective_pressure_level,
            pressure_control=pressure_control,
        )

        # Step 11.5: Observe decision in adaptive tuner (after building)
        if self._adaptive_tuner is not None and self._last_decision is not None:
            self._adaptive_tuner.observe_decision(self._last_decision)

        # Step 12: Mark tasks as dispatched (only non-rejected tasks)
        for task in batch:
            self._dispatched.add(task.task_id)
            task.status = TaskStatus.RUNNING
            task.updated_at = time.time()
        return batch

    @property
    def last_decision(self) -> dict[str, Any] | None:
        return self._last_decision

    def _fits_resources(self, task: Task) -> bool:
        if self._available_resources is None:
            return True

        required = task.resource_requirements
        available = self._available_resources
        scalar_limits = (
            ("cpu_max_percent", required.cpu_max_percent, available.cpu_max_percent),
            ("memory_max_mb", required.memory_max_mb, available.memory_max_mb),
            ("pids_max", required.pids_max, available.pids_max),
            ("io_read_bps", required.io_read_bps, available.io_read_bps),
            ("io_write_bps", required.io_write_bps, available.io_write_bps),
        )
        for _name, required_value, available_value in scalar_limits:
            if required_value is not None and available_value is not None:
                if required_value > available_value:
                    return False

        if required.allow_network and not available.allow_network:
            return False
        if required.allow_filesystem_write and not available.allow_filesystem_write:
            return False
        if required.allowed_tools and available.allowed_tools:
            if not set(required.allowed_tools).issubset(set(available.allowed_tools)):
                return False

        return True

    def _select_resource_fit_batch(self, candidates: list[Task]) -> list[Task]:
        if self._available_resources is None:
            return list(candidates)

        selected: list[Task] = []
        reserved = ResourceQuota()

        for task in candidates:
            if self._fits_reserved_resources(task, reserved):
                selected.append(task)
                self._reserve_resources(task, reserved)

        return selected

    def _fits_reserved_resources(self, task: Task, reserved: ResourceQuota) -> bool:
        available = self._available_resources
        if available is None:
            return True

        required = self._reservation_quota(task)
        scalar_limits = (
            ("cpu_max_percent", required.cpu_max_percent, available.cpu_max_percent),
            ("memory_max_mb", required.memory_max_mb, available.memory_max_mb),
            ("pids_max", required.pids_max, available.pids_max),
            ("io_read_bps", required.io_read_bps, available.io_read_bps),
            ("io_write_bps", required.io_write_bps, available.io_write_bps),
        )
        for name, required_value, available_value in scalar_limits:
            if required_value is None or available_value is None:
                continue
            reserved_value = getattr(reserved, name) or 0
            if reserved_value + required_value > available_value:
                return False

        return True

    def _reservation_quota(self, task: Task) -> ResourceQuota:
        if self._resource_estimator is None:
            return task.resource_requirements
        estimated = self._resource_estimator(task)
        return estimated if estimated is not None else task.resource_requirements

    def _reserve_resources(self, task: Task, reserved: ResourceQuota) -> None:
        required = self._reservation_quota(task)
        for name in (
            "cpu_max_percent",
            "memory_max_mb",
            "pids_max",
            "io_read_bps",
            "io_write_bps",
        ):
            required_value = getattr(required, name)
            if required_value is None:
                continue
            current_value = getattr(reserved, name) or 0
            setattr(reserved, name, current_value + required_value)

    def _apply_explicit_pressure_control(
        self,
        batch: list[Task],
        *,
        pressure_sample: dict[str, Any] | None,
    ) -> tuple[list[Task], dict[str, Any]]:
        control: dict[str, Any] = {
            "enabled": bool(pressure_sample),
            "memory_pressure_deferred_task_ids": [],
            "cpu_pressure_batch_limit": None,
            "reasons": [],
        }
        if not batch or not pressure_sample:
            return batch, control

        memory_avg10 = float(pressure_sample.get("memory_some_avg10", 0.0) or 0.0)
        cpu_avg10 = float(pressure_sample.get("cpu_some_avg10", 0.0) or 0.0)
        controlled = list(batch)

        if memory_avg10 >= 20.0 and len(controlled) > 1:
            lower_memory = [
                task
                for task in controlled
                if (task.resource_requirements.memory_max_mb or 0) < 256
            ]
            high_memory = [
                task
                for task in controlled
                if (task.resource_requirements.memory_max_mb or 0) >= 256
                and task.sla_class != "critical"
            ]
            if lower_memory and high_memory:
                high_ids = {task.task_id for task in high_memory}
                controlled = [
                    task for task in controlled if task.task_id not in high_ids
                ]
                control["memory_pressure_deferred_task_ids"] = sorted(high_ids)
                control["reasons"].append(
                    "memory pressure deferred high-memory non-critical tasks"
                )

        if cpu_avg10 >= 20.0 and len(controlled) > 1:
            controlled = controlled[:1]
            control["cpu_pressure_batch_limit"] = 1
            control["reasons"].append(
                "cpu pressure limited scheduler batch width"
            )

        return controlled, control

    def _workload_sample(self, task: Task) -> WorkloadSample:
        prefix_heat = None
        if self._affinity_advisor is not None:
            prefix_heat = self._affinity_advisor.get_task_prefix_heat(task)
        return WorkloadSample.from_task(task, prefix_heat=prefix_heat)

    def _sort_candidates_by_policy(
        self,
        candidates: list[Task],
        policy_decision: PolicyDecision,
    ) -> list[Task]:
        if policy_decision.policy == SchedulingPolicy.CACHE_AFFINITY:
            ordered = (
                self._affinity_advisor.sort_by_affinity(candidates)
                if self._affinity_advisor is not None
                else self._sort_latency_first(candidates)
            )
            if self._fairness_scale == "service_time":
                # Once a task consumes its normal queue budget, affinity may
                # no longer bypass it. FIFO among aged tasks prevents a stream
                # of newly hot prefixes from extending the bypass indefinitely.
                aged = [
                    task for task in ordered
                    if self._fairness_tracker.task_snapshot(task.task_id)["aging_due"]
                ]
                aged_ids = {task.task_id for task in aged}
                aged.sort(key=lambda task: (
                    self._fairness_tracker.task_snapshot(task.task_id)["ready_since"],
                    task.created_at,
                    task.task_id,
                ))
                return aged + [task for task in ordered if task.task_id not in aged_ids]
            return ordered
        if policy_decision.policy == SchedulingPolicy.THROUGHPUT_FIRST:
            return sorted(
                candidates,
                key=lambda task: (
                    -(task.expected_parallelism or 0),
                    -task.critical_path_score,
                    task.created_at,
                    task.task_id,
                ),
            )
        if policy_decision.policy == SchedulingPolicy.FAIRNESS_BALANCED:
            return self._sort_fairness_balanced(candidates)
        if policy_decision.policy == SchedulingPolicy.RISK_AWARE:
            return self._sort_risk_aware(candidates)
        return self._sort_latency_first(candidates)

    def _sort_latency_first(self, candidates: list[Task]) -> list[Task]:
        return sorted(
            candidates,
            key=lambda task: (
                -task.critical_path_score,
                -task.priority,
                task.created_at,
                task.task_id,
            ),
        )

    def _sort_fairness_balanced(self, candidates: list[Task]) -> list[Task]:
        """Sort candidates for fairness_balanced policy.

        Priority order:
        1. Starvation risk: high > medium > low
        2. Wait time: higher wait_ms first
        3. Deferred count: higher deferred_count first
        4. Critical path score: higher first
        5. Priority: higher first
        6. Created at: earlier first
        7. Task ID: ascending for stability
        """
        risk_order = {"high": 0, "medium": 1, "low": 2}

        def sort_key(task: Task) -> tuple:
            snapshot = self._fairness_tracker.task_snapshot(task.task_id)
            risk = snapshot.get("starvation_risk", "low")
            wait_ms = snapshot.get("wait_ms", 0.0)
            deferred_count = snapshot.get("deferred_count", 0)
            return (
                risk_order.get(risk, 2),  # Starvation risk: high first
                -wait_ms,  # Wait time: higher first
                -deferred_count,  # Deferred count: higher first
                -task.critical_path_score,  # Critical path: higher first
                -task.priority,  # Priority: higher first
                task.created_at,  # Created at: earlier first
                task.task_id,  # Task ID: ascending
            )

        return sorted(candidates, key=sort_key)

    def _sort_risk_aware(self, candidates: list[Task]) -> list[Task]:
        """Sort candidates for risk_aware policy.

        Priority order:
        1. SLA priority: critical > interactive > batch > best_effort
        2. Urgency score: higher first
        3. Starvation risk: high > medium > low
        4. Timeout risk: higher first
        5. Task risk score: higher first
        6. Critical path score: higher first
        7. Priority: higher first
        8. Created at: earlier first
        9. Task ID: ascending for stability
        """
        from agentos.scheduler.risk import SLA_PRIORITY, normalize_sla_class

        sla_priority_map = {cls.value: prio for cls, prio in SLA_PRIORITY.items()}
        risk_order = {"high": 0, "medium": 1, "low": 2}

        def sort_key(task: Task) -> tuple:
            # Get fairness data
            snapshot = self._fairness_tracker.task_snapshot(task.task_id)
            starvation_risk = snapshot.get("starvation_risk", "low")
            wait_ms = snapshot.get("wait_ms", 0.0)

            # Compute risk summary for sorting
            risk_summary = compute_task_risk(
                sla_class=task.sla_class,
                deadline_ms=task.deadline_ms,
                wait_ms=wait_ms,
                starvation_risk=starvation_risk,
                dependency_count=len(task.dependencies),
                unmet_dependency_count=sum(
                    1 for dep_id in task.dependencies
                    if dep_id in self._dag.tasks
                    and self._dag.tasks[dep_id].status != TaskStatus.COMPLETED
                ),
                previous_failures=task.previous_failures,
            )

            sla_prio = sla_priority_map.get(risk_summary.sla_class, 2)

            return (
                -sla_prio,  # SLA priority: higher first
                -risk_summary.urgency_score,  # Urgency: higher first
                risk_order.get(starvation_risk, 2),  # Starvation risk: high first
                -risk_summary.timeout_risk,  # Timeout risk: higher first
                -risk_summary.task_risk_score,  # Task risk: higher first
                -task.critical_path_score,  # Critical path: higher first
                -task.priority,  # Priority: higher first
                task.created_at,  # Created at: earlier first
                task.task_id,  # Task ID: ascending
            )

        return sorted(candidates, key=sort_key)

    def _build_decision(
        self,
        *,
        ready: list[Task],
        candidates: list[Task],
        sorted_candidates: list[Task],
        resource_fit_batch: list[Task],
        selected: list[Task],
        blocked: list[Task],
        policy_decision: PolicyDecision,
        workload_classes: dict[str, WorkloadClass],
        fairness_summary: FairnessSummary | None = None,
        safety_rejected_ids: list[str] | None = None,
        safety_deferred_ids: list[str] | None = None,
        adaptive_bias: dict[str, Any] | None = None,
        os_pressure_level: str = "normal",
        fairness_pressure_level: str = "normal",
        effective_pressure_level: str = "normal",
        pressure_control: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from agentos.scheduler.fairness import FairnessSummary

        if fairness_summary is None:
            fairness_summary = FairnessSummary()
        if safety_rejected_ids is None:
            safety_rejected_ids = []
        if safety_deferred_ids is None:
            safety_deferred_ids = []

        selected_ids = {task.task_id for task in selected}
        blocked_ids = {task.task_id for task in blocked}
        resource_fit_ids = {task.task_id for task in resource_fit_batch}
        candidate_ids = {task.task_id for task in candidates}
        safety_rejected_set = set(safety_rejected_ids)
        safety_deferred_set = set(safety_deferred_ids)
        deferred = [
            task for task in candidates
            if task.task_id not in selected_ids and task.task_id not in safety_rejected_set
        ]

        task_reasons: dict[str, str] = {}
        for task in ready:
            if task.task_id in safety_rejected_set:
                task_reasons[task.task_id] = "rejected_by_safety_envelope"
            elif task.task_id in safety_deferred_set:
                task_reasons[task.task_id] = "deferred_by_backpressure"
            elif task.task_id in selected_ids:
                task_reasons[task.task_id] = "selected"
            elif task.task_id in blocked_ids:
                task_reasons[task.task_id] = "exceeds_available_resources"
            elif task.task_id in resource_fit_ids:
                task_reasons[task.task_id] = "max_batch_size"
            elif task.task_id in candidate_ids:
                task_reasons[task.task_id] = "resource_batch_full"
            else:
                task_reasons[task.task_id] = "not_considered"

        policy_data = policy_decision.to_dict()

        # Build task decisions
        task_decisions = {
            task.task_id: self._task_decision_data(
                task,
                reason=task_reasons[task.task_id],
                workload_class=workload_classes.get(
                    task.task_id,
                    WorkloadClass.UNKNOWN,
                ),
                policy_decision=policy_decision,
                effective_pressure_level=effective_pressure_level,
            )
            for task in ready
        }
        for task_decision in task_decisions.values():
            task_decision["policy_generation"] = self._policy_generation

        # Build decision dict
        decision = {
            "schema_version": "scheduler.decision.v2",
            "strategy": self._strategy_name(),
            "policy": policy_decision.policy.value,
            "workload_class": policy_decision.workload_class.value,
            "policy_reason": policy_decision.reason,
            "policy_score": policy_data["score_breakdown"],
            "score_breakdown": policy_data["score_breakdown"],
            "fairness": fairness_summary.to_dict(),
            "fairness_warnings": fairness_summary.warnings,
            "ready_task_ids": [task.task_id for task in ready],
            "candidate_task_ids": [task.task_id for task in candidates],
            "ordered_candidate_task_ids": [
                task.task_id for task in sorted_candidates
            ],
            "selected_task_ids": [task.task_id for task in selected],
            "blocked_task_ids": [task.task_id for task in blocked],
            "deferred_task_ids": [task.task_id for task in deferred],
            "selected": [task.task_id for task in selected],
            "blocked": [task.task_id for task in blocked],
            "deferred": [task.task_id for task in deferred],
            "available_resources": (
                self._available_resources.to_dict()
                if self._available_resources is not None
                else None
            ),
            "max_batch_size": self._max_batch_size,
            "tasks": task_decisions,
            "policy_generation": self._policy_generation,
        }

        # Add OS integration fields
        decision["runtime_mode"] = "user_space_runtime"
        decision["pressure_control"] = pressure_control or {
            "enabled": False,
            "memory_pressure_deferred_task_ids": [],
            "cpu_pressure_batch_limit": None,
            "reasons": [],
        }

        # Determine kernel_integration based on kernel_probe
        if self._kernel_probe is not None:
            decision["kernel_probe"] = self._kernel_probe.to_dict()
            if self._kernel_probe.bpf_map == "mock":
                decision["kernel_integration"] = "mock"
            elif self._kernel_probe.sched_ext == "available":
                decision["kernel_integration"] = "probe"
            else:
                decision["kernel_integration"] = "unavailable"
            decision["fallback_reason"] = self._kernel_probe.reason
        else:
            decision["kernel_integration"] = "unavailable"
            decision["fallback_reason"] = None

        # Compute runtime health
        resource_blocked_count = sum(
            1 for td in task_decisions.values()
            if td.get("blocked_resource") is not None
        )
        total_incomplete_count = len(self._dag.tasks) - sum(
            1 for t in self._dag.tasks.values()
            if t.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED}
        )
        dependency_blocked_count = self._dependency_blocked_count(ready)

        health_summary = compute_runtime_health(
            ready_count=len(ready),
            candidate_count=len(candidates),
            selected_count=len(selected),
            deferred_count=len(deferred),
            blocked_count=len(blocked),
            total_incomplete_count=total_incomplete_count,
            dependency_blocked_count=dependency_blocked_count,
            fairness_score=fairness_summary.fairness_score,
            starvation_risk=fairness_summary.starvation_risk.value,
            resource_blocked_count=resource_blocked_count,
            max_batch_size=self._max_batch_size,
        )
        decision["runtime_health"] = health_summary.to_dict()

        # Add pressure feedback fields (after health_summary is available)
        decision["pressure"] = {
            "os_pressure_level": os_pressure_level,
            "fairness_pressure_level": fairness_pressure_level,
            "effective_pressure_level": effective_pressure_level,
            "runtime_health_pressure_level": health_summary.pressure_level,
            "cpu_some_avg10": self._pressure_sample.get("cpu_some_avg10", 0.0) if self._pressure_sample else 0.0,
            "memory_some_avg10": self._pressure_sample.get("memory_some_avg10", 0.0) if self._pressure_sample else 0.0,
            "io_some_avg10": self._pressure_sample.get("io_some_avg10", 0.0) if self._pressure_sample else 0.0,
        }
        if self._pressure_sample:
            for key in (
                "pressure_source",
                "scheduler_round",
                "feedback_id",
                "system",
                "cgroup_local",
            ):
                if key in self._pressure_sample:
                    decision["pressure"][key] = self._pressure_sample[key]

        # Compute SLA summary
        sla_counts = {"critical": 0, "interactive": 0, "batch": 0, "best_effort": 0}
        deadline_task_count = 0
        max_urgency_score = 0.0
        for task_decision in task_decisions.values():
            sla_class = task_decision.get("sla_class", "batch")
            sla_counts[sla_class] = sla_counts.get(sla_class, 0) + 1
            if task_decision.get("deadline_ms") is not None:
                deadline_task_count += 1
            max_urgency_score = max(max_urgency_score, task_decision.get("urgency_score", 0.0))

        decision["sla"] = {
            "critical_count": sla_counts.get("critical", 0),
            "interactive_count": sla_counts.get("interactive", 0),
            "batch_count": sla_counts.get("batch", 0),
            "best_effort_count": sla_counts.get("best_effort", 0),
            "deadline_task_count": deadline_task_count,
            "max_urgency_score": max_urgency_score,
        }

        # Compute Risk summary
        risk_scores = [td.get("task_risk_score", 0.0) for td in task_decisions.values()]
        max_task_risk_score = max(risk_scores) if risk_scores else 0.0
        avg_task_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
        high_risk_task_ids = [
            td["task_id"] for td in task_decisions.values()
            if td.get("task_risk_score", 0.0) >= 0.7
        ]
        risk_warnings: list[str] = []
        if sla_counts.get("critical", 0) > 0:
            risk_warnings.append("critical task present")
        if deadline_task_count > 0 and max_task_risk_score >= 0.7:
            risk_warnings.append("deadline risk detected")
        if max_task_risk_score >= 0.7:
            risk_warnings.append("high task risk detected")
        if fairness_summary.starvation_risk.value == "high":
            risk_warnings.append("starvation risk detected")

        decision["risk"] = {
            "max_task_risk_score": max_task_risk_score,
            "avg_task_risk_score": avg_task_risk_score,
            "high_risk_task_ids": high_risk_task_ids,
            "risk_warnings": risk_warnings,
        }

        # Compute Safety/Backpressure for each task
        # MUST use effective_pressure_level (OS + fairness max), NOT
        # health_summary.pressure_level which is an independent metric.
        pressure_level = effective_pressure_level
        for task_id, task_decision in task_decisions.items():
            safety_decision = evaluate_task_safety(
                sla_class=task_decision.get("sla_class", "batch"),
                workload_class=task_decision.get("workload_class", "unknown"),
                pressure_level=pressure_level,
                task_risk_score=task_decision.get("task_risk_score", 0.0),
                previous_failures=task_decision.get("previous_failures", 0),
                memory_max_mb=task_decision.get("resource_requirements", {}).get("memory_max_mb"),
                cpu_max_percent=task_decision.get("resource_requirements", {}).get("cpu_max_percent"),
                policy=decision.get("policy"),
                config=self._safety_config,
            )
            task_decision["admission_decision"] = safety_decision.admission_decision
            task_decision["backpressure_signal"] = safety_decision.backpressure_signal
            task_decision["degrade_reason"] = safety_decision.degrade_reason
            task_decision["safety_violations"] = safety_decision.safety_violations
            task_decision["safety"] = safety_decision.to_dict()

        # Compute Backpressure summary
        from agentos.scheduler.safety import backpressure_signal_from_pressure
        bp_signal = backpressure_signal_from_pressure(pressure_level)
        defer_count = sum(1 for td in task_decisions.values() if td.get("admission_decision") == "defer")
        degrade_count = sum(1 for td in task_decisions.values() if td.get("admission_decision") == "degrade")
        reject_count = sum(1 for td in task_decisions.values() if td.get("admission_decision") == "reject")
        bp_reasons: list[str] = []
        if defer_count > 0:
            bp_reasons.append(f"{defer_count} tasks deferred")
        if degrade_count > 0:
            bp_reasons.append(f"{degrade_count} tasks degraded")
        if reject_count > 0:
            bp_reasons.append(f"{reject_count} tasks rejected")

        decision["backpressure"] = {
            "signal": bp_signal.value,
            "pressure_level": pressure_level,
            "defer_count": defer_count,
            "degrade_count": degrade_count,
            "reject_count": reject_count,
            "reasons": bp_reasons,
        }

        # Compute Safety summary
        safety_summary = summarize_safety(task_decisions)
        decision["safety"] = safety_summary

        # Add adaptive tuning data if available
        if adaptive_bias is not None and adaptive_bias.get("enabled"):
            decision["adaptive_tuning"] = {
                "enabled": True,
                "window_size": adaptive_bias.get("window_size", 5),
                "observed_decisions": adaptive_bias.get("observed_decisions", 0),
                "policy_bias": adaptive_bias.get("policy_bias", {}),
                "reasons": adaptive_bias.get("reasons", []),
            }
        elif self._adaptive_tuner is not None:
            # Tuner exists but no bias applied this round
            decision["adaptive_tuning"] = {
                "enabled": True,
                "window_size": self._adaptive_tuner._window_size,
                "observed_decisions": len(self._adaptive_tuner._observations),
                "policy_bias": {},
                "reasons": [],
            }

        return decision

    def _dependency_blocked_count(self, ready: list[Task]) -> int:
        """Count tasks blocked due to unmet dependencies."""
        ready_ids = {task.task_id for task in ready}
        count = 0
        for task in self._dag.tasks.values():
            if task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED}:
                continue
            if task.task_id in ready_ids:
                continue
            # Check if any dependency is not completed
            for dep_id in task.dependencies:
                dep_task = self._dag.tasks.get(dep_id)
                if dep_task is not None and dep_task.status != TaskStatus.COMPLETED:
                    count += 1
                    break
        return count

    def _strategy_name(self) -> str:
        if self._affinity_advisor is not None:
            return "cache_affinity"
        return "critical_path"

    def _task_decision_data(
        self,
        task: Task,
        *,
        reason: str,
        workload_class: WorkloadClass,
        policy_decision: PolicyDecision,
        effective_pressure_level: str = "normal",
    ) -> dict[str, Any]:
        # Get task-level fairness data
        fairness_snapshot = self._fairness_tracker.task_snapshot(task.task_id)

        # Determine blocked resource details
        blocked_resource = None
        required_value = None
        available_value = None
        reserved_value = None

        if reason == "exceeds_available_resources" and self._available_resources is not None:
            # Find which resource is blocking
            required = task.resource_requirements
            available = self._available_resources
            scalar_limits = (
                ("cpu", required.cpu_max_percent, available.cpu_max_percent),
                ("memory", required.memory_max_mb, available.memory_max_mb),
                ("pids", required.pids_max, available.pids_max),
                ("io_read", required.io_read_bps, available.io_read_bps),
                ("io_write", required.io_write_bps, available.io_write_bps),
            )
            for name, req_val, avail_val in scalar_limits:
                if req_val is not None and avail_val is not None and req_val > avail_val:
                    blocked_resource = name
                    required_value = req_val
                    available_value = avail_val
                    break

        # Compute task risk summary
        risk_summary = compute_task_risk(
            sla_class=task.sla_class,
            deadline_ms=task.deadline_ms,
            wait_ms=fairness_snapshot["wait_ms"],
            starvation_risk=fairness_snapshot["starvation_risk"],
            dependency_count=len(task.dependencies),
            unmet_dependency_count=sum(
                1 for dep_id in task.dependencies
                if dep_id in self._dag.tasks
                and self._dag.tasks[dep_id].status != TaskStatus.COMPLETED
            ),
            blocked_resource=blocked_resource,
            required_value=required_value,
            available_value=available_value,
            previous_failures=task.previous_failures,
        )

        requested_sla = (task.sla_class or "normal").strip().lower()
        if requested_sla in {"critical", "interactive"}:
            queue_class = "latency"
            slice_us = 2_000
        elif requested_sla in {"batch", "best_effort"}:
            queue_class = "batch"
            slice_us = 10_000
        else:
            queue_class = "normal"
            slice_us = 5_000
        requested_weight = task.resource_requirements.cpu_shares
        kernel_weight = requested_weight if requested_weight is not None else 100
        kernel_weight = max(1, min(10_000, kernel_weight))
        base_kernel_policy = {
            "queue_class": queue_class,
            "kernel_weight": kernel_weight,
            "slice_us": slice_us,
        }
        kernel_policy_adjustment = "none"
        if queue_class == "batch" and effective_pressure_level == "warning":
            kernel_weight = max(1, kernel_weight // 2)
            kernel_policy_adjustment = (
                "batch_weight_halved_for_warning_pressure"
            )

        return {
            "task_id": task.task_id,
            "agent_role": task.agent_role,
            "reason": reason,
            "workload_class": workload_class.value,
            "policy": policy_decision.policy.value,
            "policy_reason": policy_decision.reason,
            "queue_class": queue_class,
            "kernel_weight": kernel_weight,
            "slice_us": slice_us,
            "kernel_policy_base": base_kernel_policy,
            "kernel_policy_adjustment": kernel_policy_adjustment,
            "critical_path_score": task.critical_path_score,
            "priority": task.priority,
            "dependencies": list(task.dependencies),
            "context_refs": list(task.context_refs),
            "resource_requirements": task.resource_requirements.to_dict(),
            "resource_estimate": self._reservation_quota(task).to_dict(),
            # Task-level fairness fields
            "wait_ms": fairness_snapshot["wait_ms"],
            "slowdown": fairness_snapshot["slowdown"],
            "selected_count": fairness_snapshot["selected_count"],
            "deferred_count": fairness_snapshot["deferred_count"],
            "blocked_count": fairness_snapshot["blocked_count"],
            "starvation_risk": fairness_snapshot["starvation_risk"],
            # Resource blocked explanation fields
            "blocked_resource": blocked_resource,
            "required_value": required_value,
            "available_value": available_value,
            "reserved_value": None,
            # P5: SLA and Risk fields
            "sla_class": risk_summary.sla_class,
            "deadline_ms": risk_summary.deadline_ms,
            "urgency_score": risk_summary.urgency_score,
            "task_risk_score": risk_summary.task_risk_score,
            "risk": risk_summary.to_dict(),
            # P6: Safety fields (will be populated in _build_decision)
            "previous_failures": task.previous_failures,
        }

    @property
    def is_complete(self) -> bool:
        return self._dag.is_complete()
