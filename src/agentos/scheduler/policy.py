from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agentos.scheduler.fairness import StarvationRisk
from agentos.scheduler.workload import WorkloadClass, WorkloadSample


class SchedulingPolicy(str, Enum):
    LATENCY_FIRST = "latency_first"
    THROUGHPUT_FIRST = "throughput_first"
    CACHE_AFFINITY = "cache_affinity"
    FAIRNESS_BALANCED = "fairness_balanced"
    RISK_AWARE = "risk_aware"


# Re-export StarvationRisk for backward compatibility
__all__ = [
    "SchedulingPolicy",
    "StarvationRisk",
    "PolicyDecision",
    "PolicySelector",
]


@dataclass
class PolicyDecision:
    policy: SchedulingPolicy
    reason: str
    workload_class: WorkloadClass
    score_breakdown: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.value,
            "reason": self.reason,
            "workload_class": self.workload_class.value,
            "score_breakdown": dict(self.score_breakdown),
        }


class PolicySelector:
    def __init__(self, *, hot_prefix_threshold: float = 0.7) -> None:
        self._hot_prefix_threshold = hot_prefix_threshold

    def select(
        self,
        samples: list[WorkloadSample],
        workload_classes: dict[str, WorkloadClass],
        *,
        starvation_risk: StarvationRisk = StarvationRisk.LOW,
        has_critical_sla: bool = False,
        max_urgency_score: float = 0.0,
        max_task_risk_score: float = 0.0,
        pressure_level: str = "normal",
        adaptive_bias: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        dominant_class = self._dominant_class(samples, workload_classes)
        max_prefix_heat = max(
            (sample.prefix_heat or 0.0 for sample in samples),
            default=0.0,
        )
        expected_parallelism = max(
            (sample.expected_parallelism or 0 for sample in samples),
            default=0,
        )
        priority = max((sample.priority for sample in samples), default=0)
        score_breakdown = {
            "starvation_risk": starvation_risk.value,
            "max_prefix_heat": max_prefix_heat,
            "expected_parallelism": expected_parallelism,
            "priority": priority,
            "candidate_count": len(samples),
            "has_critical_sla": has_critical_sla,
            "max_urgency_score": max_urgency_score,
            "max_task_risk_score": max_task_risk_score,
            "pressure_level": pressure_level,
        }

        # Determine base policy using existing rules
        base_decision = self._select_base_policy(
            dominant_class=dominant_class,
            max_prefix_heat=max_prefix_heat,
            expected_parallelism=expected_parallelism,
            priority=priority,
            starvation_risk=starvation_risk,
            has_critical_sla=has_critical_sla,
            max_urgency_score=max_urgency_score,
            max_task_risk_score=max_task_risk_score,
            pressure_level=pressure_level,
            score_breakdown=score_breakdown,
        )

        # Apply adaptive bias if provided
        if adaptive_bias and adaptive_bias.get("enabled"):
            bias = adaptive_bias.get("policy_bias", {})
            reasons = adaptive_bias.get("reasons", [])
            if bias:
                adjusted = self._apply_adaptive_bias(
                    base_decision=base_decision,
                    bias=bias,
                    dominant_class=dominant_class,
                    score_breakdown=score_breakdown,
                    adaptive_reasons=reasons,
                )
                if adjusted is not None:
                    return adjusted

        return base_decision

    def _select_base_policy(
        self,
        *,
        dominant_class: WorkloadClass,
        max_prefix_heat: float,
        expected_parallelism: int,
        priority: int,
        starvation_risk: StarvationRisk,
        has_critical_sla: bool,
        max_urgency_score: float,
        max_task_risk_score: float,
        pressure_level: str,
        score_breakdown: dict[str, Any],
    ) -> PolicyDecision:
        """Core policy selection logic with pressure-driven rules.

        Priority order:
        1. critical SLA / high task risk / high urgency → risk_aware (always)
        2. critical/overloaded OS pressure → risk_aware
        3. warning OS pressure → fairness_balanced
        4. high starvation (OS normal) → fairness_balanced
        5. workload-based rules (cache_affinity, latency_first, throughput_first)
        """
        # Highest priority: critical SLA, high risk, or high urgency
        if has_critical_sla or max_task_risk_score >= 0.7 or max_urgency_score >= 0.85:
            return PolicyDecision(
                policy=SchedulingPolicy.RISK_AWARE,
                reason="risk-aware scheduling triggered",
                workload_class=dominant_class,
                score_breakdown=score_breakdown,
            )

        # critical/overloaded OS pressure → risk_aware
        if pressure_level in {"critical", "overloaded"}:
            return PolicyDecision(
                policy=SchedulingPolicy.RISK_AWARE,
                reason=f"OS pressure {pressure_level}",
                workload_class=dominant_class,
                score_breakdown=score_breakdown,
            )

        # warning OS pressure → fairness_balanced
        if pressure_level == "warning":
            return PolicyDecision(
                policy=SchedulingPolicy.FAIRNESS_BALANCED,
                reason="OS pressure warning",
                workload_class=dominant_class,
                score_breakdown=score_breakdown,
            )

        # high starvation with OS normal → fairness_balanced
        if starvation_risk == StarvationRisk.HIGH:
            return PolicyDecision(
                policy=SchedulingPolicy.FAIRNESS_BALANCED,
                reason="high starvation risk",
                workload_class=dominant_class,
                score_breakdown=score_breakdown,
            )

        # Workload-based rules
        if max_prefix_heat >= self._hot_prefix_threshold:
            return PolicyDecision(
                policy=SchedulingPolicy.CACHE_AFFINITY,
                reason="hot prefix and cache reuse opportunity",
                workload_class=dominant_class,
                score_breakdown=score_breakdown,
            )
        if dominant_class == WorkloadClass.INTERACTIVE or priority >= 8:
            return PolicyDecision(
                policy=SchedulingPolicy.LATENCY_FIRST,
                reason="interactive workload or high priority",
                workload_class=dominant_class,
                score_breakdown=score_breakdown,
            )
        if dominant_class == WorkloadClass.BATCH or expected_parallelism >= 4:
            return PolicyDecision(
                policy=SchedulingPolicy.THROUGHPUT_FIRST,
                reason="batch workload or high expected parallelism",
                workload_class=dominant_class,
                score_breakdown=score_breakdown,
            )
        return PolicyDecision(
            policy=SchedulingPolicy.LATENCY_FIRST,
            reason="fallback critical path ordering",
            workload_class=dominant_class,
            score_breakdown=score_breakdown,
        )

    def _apply_adaptive_bias(
        self,
        *,
        base_decision: PolicyDecision,
        bias: dict[str, int],
        dominant_class: WorkloadClass,
        score_breakdown: dict[str, Any],
        adaptive_reasons: list[str],
    ) -> PolicyDecision | None:
        """Apply adaptive bias to potentially adjust the base policy.

        The bias is a dict of policy_name -> score.  The policy with the
        highest positive bias that differs from the base policy is considered
        as an alternative.  We only switch if the bias score >= 2 (i.e., the
        signal is strong enough to justify overriding the base rule).

        Returns a new PolicyDecision if the policy changes, or None to keep
        the base decision.
        """
        if not bias:
            return None

        # Find the policy with the highest positive bias
        best_bias_policy = None
        best_bias_score = 0
        for policy_name, score in bias.items():
            if score > best_bias_score:
                best_bias_score = score
                best_bias_policy = policy_name

        if best_bias_policy is None or best_bias_score < 2:
            return None

        # Map bias policy name to SchedulingPolicy enum
        try:
            target_policy = SchedulingPolicy(best_bias_policy)
        except ValueError:
            return None

        # Only switch if the target differs from the base
        if target_policy == base_decision.policy:
            return None

        # Record the adaptive adjustment in score_breakdown
        adjusted_breakdown = dict(score_breakdown)
        adjusted_breakdown["adaptive_bias"] = dict(bias)
        adjusted_breakdown["base_policy"] = base_decision.policy.value
        adjusted_breakdown["adaptive_reasons"] = adaptive_reasons

        reason = f"adaptive tuning: {', '.join(adaptive_reasons)}"
        return PolicyDecision(
            policy=target_policy,
            reason=reason,
            workload_class=dominant_class,
            score_breakdown=adjusted_breakdown,
        )

    def _dominant_class(
        self,
        samples: list[WorkloadSample],
        workload_classes: dict[str, WorkloadClass],
    ) -> WorkloadClass:
        if not samples:
            return WorkloadClass.UNKNOWN
        counts: dict[WorkloadClass, int] = {}
        for sample in samples:
            if sample.task_id is None:
                cls = WorkloadClass.UNKNOWN
            else:
                cls = workload_classes.get(sample.task_id, WorkloadClass.UNKNOWN)
            counts[cls] = counts.get(cls, 0) + 1
        priority = {
            WorkloadClass.AI_INFERENCE: 0,
            WorkloadClass.INTERACTIVE: 1,
            WorkloadClass.BATCH: 2,
            WorkloadClass.COMPILE: 3,
            WorkloadClass.COMPRESSION: 4,
            WorkloadClass.NETWORK: 5,
            WorkloadClass.IO_INTENSIVE: 6,
            WorkloadClass.UNKNOWN: 7,
        }
        return max(
            counts,
            key=lambda cls: (counts[cls], -priority.get(cls, 99)),
        )
