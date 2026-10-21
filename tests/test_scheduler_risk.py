"""Tests for Risk Assessment and SLA-aware scheduling."""

from __future__ import annotations

import pytest

from agentos.scheduler.risk import (
    SLAClass,
    SLA_PRIORITY,
    TaskRiskSummary,
    clamp01,
    compute_dependency_risk,
    compute_failure_risk,
    compute_resource_risk,
    compute_sla_risk,
    compute_task_risk,
    compute_timeout_risk,
    compute_urgency_score,
    normalize_sla_class,
    starvation_risk_to_score,
)


class TestSLAClass:
    """Test SLAClass enum."""

    def test_sla_class_values(self):
        assert SLAClass.CRITICAL.value == "critical"
        assert SLAClass.INTERACTIVE.value == "interactive"
        assert SLAClass.BATCH.value == "batch"
        assert SLAClass.BEST_EFFORT.value == "best_effort"

    def test_sla_priority(self):
        assert SLA_PRIORITY[SLAClass.CRITICAL] > SLA_PRIORITY[SLAClass.INTERACTIVE]
        assert SLA_PRIORITY[SLAClass.INTERACTIVE] > SLA_PRIORITY[SLAClass.BATCH]
        assert SLA_PRIORITY[SLAClass.BATCH] > SLA_PRIORITY[SLAClass.BEST_EFFORT]


class TestClamp01:
    """Test clamp01 function."""

    def test_within_range(self):
        assert clamp01(0.5) == 0.5

    def test_below_zero(self):
        assert clamp01(-0.5) == 0.0

    def test_above_one(self):
        assert clamp01(1.5) == 1.0


class TestNormalizeSLAClass:
    """Test normalize_sla_class function."""

    def test_valid_values(self):
        assert normalize_sla_class("critical") == SLAClass.CRITICAL
        assert normalize_sla_class("interactive") == SLAClass.INTERACTIVE
        assert normalize_sla_class("batch") == SLAClass.BATCH
        assert normalize_sla_class("best_effort") == SLAClass.BEST_EFFORT

    def test_none_defaults_to_batch(self):
        assert normalize_sla_class(None) == SLAClass.BATCH

    def test_invalid_defaults_to_batch(self):
        assert normalize_sla_class("invalid") == SLAClass.BATCH


class TestComputeUrgencyScore:
    """Test compute_urgency_score function."""

    def test_critical_higher_than_batch(self):
        critical = compute_urgency_score(sla_class="critical", wait_ms=0, deadline_ms=None)
        batch = compute_urgency_score(sla_class="batch", wait_ms=0, deadline_ms=None)
        assert critical > batch

    def test_deadline_increases_urgency(self):
        no_deadline = compute_urgency_score(sla_class="batch", wait_ms=0, deadline_ms=None)
        urgent = compute_urgency_score(sla_class="batch", wait_ms=0, deadline_ms=300)
        assert urgent > no_deadline

    def test_wait_increases_urgency(self):
        no_wait = compute_urgency_score(sla_class="batch", wait_ms=0, deadline_ms=None)
        waited = compute_urgency_score(sla_class="batch", wait_ms=5000, deadline_ms=None)
        assert waited > no_wait

    def test_score_range(self):
        score = compute_urgency_score(sla_class="critical", wait_ms=10000, deadline_ms=100)
        assert 0.0 <= score <= 1.0


class TestStarvationRiskToScore:
    """Test starvation_risk_to_score function."""

    def test_high_risk(self):
        assert starvation_risk_to_score("high") == 0.9

    def test_medium_risk(self):
        assert starvation_risk_to_score("medium") == 0.5

    def test_low_risk(self):
        assert starvation_risk_to_score("low") == 0.1

    def test_none_risk(self):
        assert starvation_risk_to_score(None) == 0.1


class TestComputeDependencyRisk:
    """Test compute_dependency_risk function."""

    def test_no_dependencies(self):
        assert compute_dependency_risk(dependency_count=0, unmet_dependency_count=0) == 0.0

    def test_all_met(self):
        assert compute_dependency_risk(dependency_count=5, unmet_dependency_count=0) == 0.0

    def test_all_unmet(self):
        assert compute_dependency_risk(dependency_count=5, unmet_dependency_count=5) == 1.0

    def test_partial_unmet(self):
        risk = compute_dependency_risk(dependency_count=10, unmet_dependency_count=7)
        assert risk == 0.7


class TestComputeResourceRisk:
    """Test compute_resource_risk function."""

    def test_blocked_resource(self):
        risk = compute_resource_risk(blocked_resource="memory", required_value=None, available_value=None)
        assert risk == 0.9

    def test_no_values(self):
        risk = compute_resource_risk(blocked_resource=None, required_value=None, available_value=None)
        assert risk == 0.0

    def test_high_usage(self):
        risk = compute_resource_risk(blocked_resource=None, required_value=90, available_value=100)
        assert risk == 0.9


class TestComputeTimeoutRisk:
    """Test compute_timeout_risk function."""

    def test_no_deadline(self):
        assert compute_timeout_risk(deadline_ms=None, wait_ms=1000) == 0.0

    def test_zero_deadline(self):
        assert compute_timeout_risk(deadline_ms=0, wait_ms=1000) == 1.0

    def test_wait_exceeds_deadline(self):
        risk = compute_timeout_risk(deadline_ms=1000, wait_ms=900)
        assert risk == 0.9


class TestComputeFailureRisk:
    """Test compute_failure_risk function."""

    def test_no_failures(self):
        assert compute_failure_risk(previous_failures=0) == 0.0

    def test_one_failure(self):
        risk = compute_failure_risk(previous_failures=1)
        assert risk > 0.0

    def test_three_failures(self):
        risk = compute_failure_risk(previous_failures=3)
        assert risk == 1.0


class TestComputeTaskRisk:
    """Test compute_task_risk function."""

    def test_basic_risk_computation(self):
        summary = compute_task_risk(
            sla_class="batch",
            deadline_ms=None,
            wait_ms=0,
            starvation_risk="low",
            dependency_count=0,
            unmet_dependency_count=0,
        )
        assert summary.sla_class == "batch"
        assert summary.task_risk_score >= 0.0
        assert summary.task_risk_score <= 1.0

    def test_critical_sla_adds_reason(self):
        summary = compute_task_risk(
            sla_class="critical",
            deadline_ms=None,
            wait_ms=0,
            starvation_risk="low",
            dependency_count=0,
            unmet_dependency_count=0,
        )
        assert summary.sla_class == "critical"
        assert "critical SLA" in summary.risk_reasons

    def test_deadline_risk_adds_reason(self):
        summary = compute_task_risk(
            sla_class="interactive",
            deadline_ms=1000,
            wait_ms=900,
            starvation_risk="low",
            dependency_count=0,
            unmet_dependency_count=0,
        )
        assert summary.timeout_risk >= 0.7
        assert "deadline risk" in summary.risk_reasons

    def test_invalid_sla_defaults_to_batch(self):
        summary = compute_task_risk(
            sla_class="invalid",
            deadline_ms=None,
            wait_ms=0,
            starvation_risk="low",
            dependency_count=0,
            unmet_dependency_count=0,
        )
        assert summary.sla_class == "batch"

    def test_to_dict(self):
        summary = compute_task_risk(
            sla_class="critical",
            deadline_ms=500,
            wait_ms=100,
            starvation_risk="high",
            dependency_count=3,
            unmet_dependency_count=1,
        )
        d = summary.to_dict()
        assert "sla_class" in d
        assert "deadline_ms" in d
        assert "urgency_score" in d
        assert "task_risk_score" in d
        assert "sla_risk" in d
        assert "risk_reasons" in d

    def test_critical_task_risk_higher_than_batch(self):
        """Critical task should have higher risk than batch task."""
        critical = compute_task_risk(
            sla_class="critical",
            deadline_ms=None,
            wait_ms=0,
            starvation_risk="low",
            dependency_count=0,
            unmet_dependency_count=0,
        )
        batch = compute_task_risk(
            sla_class="batch",
            deadline_ms=None,
            wait_ms=0,
            starvation_risk="low",
            dependency_count=0,
            unmet_dependency_count=0,
        )
        assert critical.task_risk_score > batch.task_risk_score
        assert critical.sla_risk > batch.sla_risk

    def test_interactive_with_deadline_risk_higher_than_batch(self):
        """Interactive task with deadline should have higher risk than batch."""
        interactive = compute_task_risk(
            sla_class="interactive",
            deadline_ms=1000,
            wait_ms=0,
            starvation_risk="low",
            dependency_count=0,
            unmet_dependency_count=0,
        )
        batch = compute_task_risk(
            sla_class="batch",
            deadline_ms=None,
            wait_ms=0,
            starvation_risk="low",
            dependency_count=0,
            unmet_dependency_count=0,
        )
        assert interactive.task_risk_score > batch.task_risk_score
        assert interactive.urgency_score > batch.urgency_score

    def test_high_urgency_adds_reason(self):
        """High urgency should add 'high urgency' to risk_reasons."""
        summary = compute_task_risk(
            sla_class="critical",
            deadline_ms=500,
            wait_ms=0,
            starvation_risk="low",
            dependency_count=0,
            unmet_dependency_count=0,
        )
        assert summary.urgency_score >= 0.8
        assert "high urgency" in summary.risk_reasons

    def test_sla_risk_values(self):
        """Test SLA risk values for different SLA classes."""
        critical = compute_sla_risk("critical")
        interactive = compute_sla_risk("interactive")
        batch = compute_sla_risk("batch")
        best_effort = compute_sla_risk("best_effort")

        assert critical >= 0.8
        assert interactive >= 0.5
        assert batch == 0.2
        assert best_effort == 0.1
        assert critical > interactive > batch > best_effort
