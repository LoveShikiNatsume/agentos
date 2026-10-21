"""Adaptive policy tuner for lightweight, explainable scheduling adjustments.

This module provides a sliding-window, rule-based adaptive tuner that observes
recent scheduler decisions and recommends policy bias adjustments. It does NOT
use machine learning or reinforcement learning -- all adjustments are
explainable, auditable, and based on transparent threshold rules.

The tuner outputs "policy bias" recommendations (not hard policy overrides) that
the PolicySelector can consult when making its final decision.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AdaptiveObservation:
    """A single observation extracted from a scheduler decision."""

    policy: str = ""
    workload_class: str = ""
    avg_wait_ms: float = 0.0
    max_wait_ms: float = 0.0
    fairness_score: float = 1.0
    starvation_warning_count: int = 0
    cache_hit_ratio: float = 0.0
    max_urgency_score: float = 0.0
    high_risk_task_count: int = 0
    unique_high_risk_task_count: int = 0
    deferred_task_count: int = 0
    blocked_task_count: int = 0
    backpressure_signal: str = "none"
    safety_reject_count: int = 0
    health_score: float = 1.0
    pressure_level: str = "normal"


def _extract_observation(decision: dict[str, Any]) -> AdaptiveObservation:
    """Extract an AdaptiveObservation from a scheduler.decision.v2 dict."""
    fairness = decision.get("fairness", {})
    health = decision.get("runtime_health", {})
    sla = decision.get("sla", {})
    risk = decision.get("risk", {})
    backpressure = decision.get("backpressure", {})
    safety = decision.get("safety", {})

    return AdaptiveObservation(
        policy=decision.get("policy", ""),
        workload_class=decision.get("workload_class", ""),
        avg_wait_ms=fairness.get("avg_wait_ms", 0.0),
        max_wait_ms=fairness.get("max_wait_ms", 0.0),
        fairness_score=fairness.get("fairness_score", 1.0),
        starvation_warning_count=len(decision.get("fairness_warnings", [])),
        cache_hit_ratio=decision.get("cache_hit_ratio", 0.0),
        max_urgency_score=sla.get("max_urgency_score", 0.0),
        high_risk_task_count=len(risk.get("high_risk_task_ids", [])),
        unique_high_risk_task_count=len(risk.get("high_risk_task_ids", [])),
        deferred_task_count=len(decision.get("deferred_task_ids", [])),
        blocked_task_count=len(decision.get("blocked_task_ids", [])),
        backpressure_signal=backpressure.get("signal", "none"),
        safety_reject_count=safety.get(
            "admission_counts", {}
        ).get("reject", 0),
        health_score=health.get("runtime_health_score", 1.0),
        pressure_level=health.get("pressure_level", "normal"),
    )


class AdaptivePolicyTuner:
    """Sliding-window adaptive policy tuner.

    Observes recent scheduler decisions and produces explainable policy bias
    recommendations. All rules are transparent and auditable -- no black-box
    ML is involved.

    Parameters
    ----------
    window_size:
        Number of recent observations to retain.  Default is 5.
    """

    def __init__(self, window_size: int = 5) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        self._window_size = window_size
        self._observations: deque[AdaptiveObservation] = deque(maxlen=window_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe_decision(self, decision: dict[str, Any]) -> None:
        """Record a scheduler decision for adaptive tuning.

        Parameters
        ----------
        decision:
            A ``scheduler.decision.v2`` dict (or equivalent).
        """
        obs = _extract_observation(decision)
        self._observations.append(obs)

    def recommend_policy_bias(self) -> dict[str, Any]:
        """Compute policy bias recommendations from the observation window.

        Returns
        -------
        dict
            A dict with keys:
            - ``policy_bias``: ``dict[str, int]`` -- cumulative bias scores
            - ``reasons``: ``list[str]`` -- human-readable explanations
            - ``observed_decisions``: ``int`` -- how many observations exist
            - ``window_size``: ``int`` -- configured window size
            - ``enabled``: ``bool`` -- always True
        """
        policy_bias: dict[str, int] = {}
        reasons: list[str] = []

        if not self._observations:
            return {
                "enabled": True,
                "window_size": self._window_size,
                "observed_decisions": 0,
                "policy_bias": policy_bias,
                "reasons": reasons,
            }

        # Rule 1: Fairness adaptive
        self._rule_fairness(policy_bias, reasons)

        # Rule 2: Cache affinity adaptive
        self._rule_cache_affinity(policy_bias, reasons)

        # Rule 3: SLA/Risk adaptive
        self._rule_sla_risk(policy_bias, reasons)

        # Rule 4: Backpressure/Safety adaptive
        self._rule_backpressure_safety(policy_bias, reasons)

        return {
            "enabled": True,
            "window_size": self._window_size,
            "observed_decisions": len(self._observations),
            "policy_bias": policy_bias,
            "reasons": reasons,
        }

    def snapshot(self) -> dict[str, Any]:
        """Return a snapshot of the tuner state for debugging/auditing."""
        return {
            "window_size": self._window_size,
            "observed_decisions": len(self._observations),
            "observations": [
                {
                    "policy": obs.policy,
                    "workload_class": obs.workload_class,
                    "avg_wait_ms": obs.avg_wait_ms,
                    "max_wait_ms": obs.max_wait_ms,
                    "fairness_score": obs.fairness_score,
                    "starvation_warning_count": obs.starvation_warning_count,
                    "cache_hit_ratio": obs.cache_hit_ratio,
                    "max_urgency_score": obs.max_urgency_score,
                    "backpressure_signal": obs.backpressure_signal,
                    "safety_reject_count": obs.safety_reject_count,
                    "pressure_level": obs.pressure_level,
                }
                for obs in self._observations
            ],
        }

    # ------------------------------------------------------------------
    # Adaptive rules (all explainable)
    # ------------------------------------------------------------------

    def _rule_fairness(
        self,
        policy_bias: dict[str, int],
        reasons: list[str],
    ) -> None:
        """Rule 1: Fairness adaptive.

        If max_wait_ms is trending upward, or fairness_score is consistently
        low, or starvation warnings appear, bias toward fairness_balanced.
        """
        obs_list = list(self._observations)
        n = len(obs_list)
        if n == 0:
            return

        # Check for starvation warnings
        has_starvation = any(obs.starvation_warning_count > 0 for obs in obs_list)

        # Check for consistently low fairness score (below 0.7)
        low_fairness_count = sum(1 for obs in obs_list if obs.fairness_score < 0.7)
        consistently_low_fairness = low_fairness_count >= max(2, n // 2)

        # Check for increasing max_wait_ms trend
        wait_increasing = False
        if n >= 2:
            recent_waits = [obs.max_wait_ms for obs in obs_list[-3:]]
            if len(recent_waits) >= 2:
                wait_increasing = all(
                    recent_waits[i] <= recent_waits[i + 1]
                    for i in range(len(recent_waits) - 1)
                ) and recent_waits[-1] > 0

        if has_starvation or consistently_low_fairness or wait_increasing:
            policy_bias["fairness_balanced"] = policy_bias.get("fairness_balanced", 0) + 1
            reasons.append("recent wait/fairness pressure increased")

    def _rule_cache_affinity(
        self,
        policy_bias: dict[str, int],
        reasons: list[str],
    ) -> None:
        """Rule 2: Cache affinity adaptive.

        If workload_class is ai_inference or interactive, and cache_hit_ratio
        is high, bias toward cache_affinity.

        Note: Under mock backend, cache signals are Runtime-side simulated
        evidence, not real cloud KV cache.
        """
        obs_list = list(self._observations)
        if not obs_list:
            return

        cache_workloads = {"ai_inference", "interactive"}
        has_cache_workload = any(
            obs.workload_class in cache_workloads for obs in obs_list
        )

        # Check for high cache hit ratio or strong cache signal
        high_cache_count = sum(1 for obs in obs_list if obs.cache_hit_ratio >= 0.5)
        has_cache_signal = high_cache_count >= max(1, len(obs_list) // 2)

        if has_cache_workload and has_cache_signal:
            policy_bias["cache_affinity"] = policy_bias.get("cache_affinity", 0) + 1
            reasons.append("prefix/cache locality signal observed")

    def _rule_sla_risk(
        self,
        policy_bias: dict[str, int],
        reasons: list[str],
    ) -> None:
        """Rule 3: SLA/Risk adaptive.

        If max_urgency_score >= 0.8, or high risk tasks are present, or
        risk warnings appear, bias toward risk_aware.
        """
        obs_list = list(self._observations)
        if not obs_list:
            return

        high_urgency = any(obs.max_urgency_score >= 0.8 for obs in obs_list)
        has_high_risk = any(obs.high_risk_task_count > 0 for obs in obs_list)
        # Also check for critical/deadline pressure via high urgency
        deadline_pressure = any(obs.max_urgency_score >= 0.7 for obs in obs_list)

        if high_urgency or has_high_risk:
            policy_bias["risk_aware"] = policy_bias.get("risk_aware", 0) + 1
            reasons.append("deadline/SLA urgency increased")

    def _rule_backpressure_safety(
        self,
        policy_bias: dict[str, int],
        reasons: list[str],
    ) -> None:
        """Rule 4: Backpressure/Safety adaptive.

        If backpressure is hard/rejecting, or safety rejects tasks, or
        resource pressure is high, reduce throughput_first bias and increase
        risk_aware bias.
        """
        obs_list = list(self._observations)
        if not obs_list:
            return

        hard_pressure = any(
            obs.backpressure_signal in {"hard", "rejecting"} for obs in obs_list
        )
        has_safety_reject = any(obs.safety_reject_count > 0 for obs in obs_list)
        high_pressure = any(
            obs.pressure_level in {"critical", "overloaded"} for obs in obs_list
        )

        if hard_pressure or has_safety_reject or high_pressure:
            policy_bias["throughput_first"] = policy_bias.get("throughput_first", 0) - 1
            policy_bias["risk_aware"] = policy_bias.get("risk_aware", 0) + 1
            reasons.append("backpressure or safety envelope pressure observed")
