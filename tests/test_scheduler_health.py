"""Tests for Runtime Health Score computation."""

from __future__ import annotations

import pytest

from agentos.scheduler.health import (
    RuntimeHealthSummary,
    clamp01,
    compute_dependency_pressure,
    compute_fairness_pressure,
    compute_queue_pressure,
    compute_resource_pressure,
    compute_runtime_health,
    pressure_level_from_score,
)


class TestClamp01:
    """Test clamp01 function."""

    def test_clamp01_within_range(self):
        assert clamp01(0.5) == 0.5

    def test_clamp01_below_zero(self):
        assert clamp01(-0.5) == 0.0

    def test_clamp01_above_one(self):
        assert clamp01(1.5) == 1.0

    def test_clamp01_boundaries(self):
        assert clamp01(0.0) == 0.0
        assert clamp01(1.0) == 1.0


class TestPressureLevelFromScore:
    """Test pressure_level_from_score function."""

    def test_normal_level(self):
        assert pressure_level_from_score(0.90) == "normal"
        assert pressure_level_from_score(0.80) == "normal"

    def test_warning_level(self):
        assert pressure_level_from_score(0.70) == "warning"
        assert pressure_level_from_score(0.60) == "warning"

    def test_critical_level(self):
        assert pressure_level_from_score(0.50) == "critical"
        assert pressure_level_from_score(0.35) == "critical"

    def test_overloaded_level(self):
        assert pressure_level_from_score(0.20) == "overloaded"
        assert pressure_level_from_score(0.0) == "overloaded"

    def test_escalate_on_high_pressure(self):
        # Normal score but high pressure -> critical
        assert pressure_level_from_score(0.90, max_pressure=0.95) == "critical"
        # Warning score but high pressure -> critical
        assert pressure_level_from_score(0.70, max_pressure=0.95) == "critical"


class TestComputeQueuePressure:
    """Test compute_queue_pressure function."""

    def test_no_pressure(self):
        pressure = compute_queue_pressure(
            ready_count=1, deferred_count=0, blocked_count=0
        )
        assert pressure == 0.0

    def test_pressure_from_deferred(self):
        pressure = compute_queue_pressure(
            ready_count=1, deferred_count=5, blocked_count=0
        )
        assert pressure > 0.0

    def test_pressure_from_blocked(self):
        pressure = compute_queue_pressure(
            ready_count=1, deferred_count=0, blocked_count=5
        )
        assert pressure > 0.0

    def test_pressure_from_max_batch_size(self):
        pressure = compute_queue_pressure(
            ready_count=10, deferred_count=0, blocked_count=0, max_batch_size=2
        )
        assert pressure > 0.0


class TestComputeFairnessPressure:
    """Test compute_fairness_pressure function."""

    def test_no_pressure(self):
        pressure = compute_fairness_pressure(fairness_score=1.0, starvation_risk="low")
        assert pressure == 0.0

    def test_pressure_from_low_score(self):
        pressure = compute_fairness_pressure(fairness_score=0.4, starvation_risk="low")
        assert pressure == 0.6

    def test_pressure_from_medium_risk(self):
        pressure = compute_fairness_pressure(fairness_score=1.0, starvation_risk="medium")
        assert pressure == 0.5

    def test_pressure_from_high_risk(self):
        pressure = compute_fairness_pressure(fairness_score=1.0, starvation_risk="high")
        assert pressure == 0.8


class TestComputeResourcePressure:
    """Test compute_resource_pressure function."""

    def test_no_blocked(self):
        pressure = compute_resource_pressure(resource_blocked_count=0)
        assert pressure == 0.0

    def test_one_blocked(self):
        pressure = compute_resource_pressure(resource_blocked_count=1)
        assert pressure == 0.7

    def test_multiple_blocked(self):
        pressure = compute_resource_pressure(resource_blocked_count=3)
        assert pressure > 0.7


class TestComputeDependencyPressure:
    """Test compute_dependency_pressure function."""

    def test_no_incomplete(self):
        pressure = compute_dependency_pressure(
            total_incomplete_count=0, dependency_blocked_count=0
        )
        assert pressure == 0.0

    def test_no_dependency_blocked(self):
        pressure = compute_dependency_pressure(
            total_incomplete_count=10, dependency_blocked_count=0
        )
        assert pressure == 0.0

    def test_dependency_blocked(self):
        pressure = compute_dependency_pressure(
            total_incomplete_count=10, dependency_blocked_count=7
        )
        assert pressure == 0.7


class TestComputeRuntimeHealth:
    """Test compute_runtime_health function."""

    def test_runtime_health_normal_when_no_pressure(self):
        """Test normal health when no pressure."""
        summary = compute_runtime_health(
            ready_count=1,
            candidate_count=1,
            selected_count=1,
            deferred_count=0,
            blocked_count=0,
            total_incomplete_count=1,
            dependency_blocked_count=0,
            fairness_score=1.0,
            starvation_risk="low",
            resource_blocked_count=0,
            max_batch_size=None,
        )

        assert summary.runtime_health_score == 1.0
        assert summary.pressure_level == "normal"
        assert summary.health_warnings == []

    def test_runtime_health_drops_with_queue_pressure(self):
        """Test health drops with queue pressure."""
        summary = compute_runtime_health(
            ready_count=10,
            candidate_count=10,
            selected_count=2,
            deferred_count=8,
            blocked_count=0,
            total_incomplete_count=10,
            dependency_blocked_count=0,
            fairness_score=1.0,
            starvation_risk="low",
            max_batch_size=2,
        )

        assert summary.queue_pressure > 0.0
        assert summary.runtime_health_score < 1.0

    def test_runtime_health_reflects_fairness_pressure(self):
        """Test health reflects fairness pressure."""
        summary = compute_runtime_health(
            ready_count=3,
            candidate_count=3,
            selected_count=1,
            deferred_count=2,
            blocked_count=0,
            total_incomplete_count=3,
            dependency_blocked_count=0,
            fairness_score=0.4,
            starvation_risk="high",
        )

        assert summary.fairness_pressure >= 0.8
        assert summary.runtime_health_score < 0.8
        assert summary.pressure_level in {"warning", "critical", "overloaded"}

    def test_runtime_health_reflects_dependency_pressure(self):
        """Test health reflects dependency pressure."""
        summary = compute_runtime_health(
            ready_count=1,
            candidate_count=1,
            selected_count=1,
            deferred_count=0,
            blocked_count=0,
            total_incomplete_count=10,
            dependency_blocked_count=7,
            fairness_score=1.0,
            starvation_risk="low",
        )

        assert summary.dependency_pressure == 0.7
        assert summary.runtime_health_score < 1.0

    def test_runtime_health_reflects_resource_pressure(self):
        """Test health reflects resource pressure."""
        summary = compute_runtime_health(
            ready_count=1,
            candidate_count=1,
            selected_count=0,
            deferred_count=0,
            blocked_count=1,
            total_incomplete_count=1,
            dependency_blocked_count=0,
            fairness_score=1.0,
            starvation_risk="low",
            resource_blocked_count=1,
        )

        assert summary.resource_pressure == 0.7
        assert summary.runtime_health_score < 1.0

    def test_runtime_health_to_dict(self):
        """Test to_dict serialization."""
        summary = compute_runtime_health(
            ready_count=1,
            candidate_count=1,
            selected_count=1,
            deferred_count=0,
            blocked_count=0,
            total_incomplete_count=1,
            dependency_blocked_count=0,
            fairness_score=1.0,
            starvation_risk="low",
        )

        d = summary.to_dict()
        assert "runtime_health_score" in d
        assert "queue_pressure" in d
        assert "resource_pressure" in d
        assert "fairness_pressure" in d
        assert "dependency_pressure" in d
        assert "pressure_level" in d
        assert "health_warnings" in d

    def test_multiple_pressures_warning_level(self):
        """Test that multiple moderate pressures trigger warning level."""
        summary = compute_runtime_health(
            ready_count=10,
            candidate_count=10,
            selected_count=2,
            deferred_count=8,
            blocked_count=0,
            total_incomplete_count=10,
            dependency_blocked_count=5,
            fairness_score=0.6,
            starvation_risk="medium",
            resource_blocked_count=0,
            max_batch_size=2,
        )

        # Multiple pressures should combine
        assert summary.queue_pressure > 0.0
        assert summary.fairness_pressure >= 0.5
        assert summary.dependency_pressure == 0.5
        assert summary.runtime_health_score < 0.8
        assert summary.pressure_level in {"warning", "critical", "overloaded"}
        assert len(summary.health_warnings) > 0

    def test_critical_fairness_pressure(self):
        """Test that high fairness pressure triggers warning level."""
        summary = compute_runtime_health(
            ready_count=3,
            candidate_count=3,
            selected_count=1,
            deferred_count=2,
            blocked_count=0,
            total_incomplete_count=3,
            dependency_blocked_count=0,
            fairness_score=0.2,
            starvation_risk="high",
        )

        assert summary.fairness_pressure >= 0.8
        # fairness_pressure >= 0.7 should at least trigger warning
        assert summary.pressure_level in {"warning", "critical", "overloaded"}
        assert any("fairness pressure" in w for w in summary.health_warnings)

    def test_resource_and_queue_pressure_warning(self):
        """Test that resource + queue pressure triggers warning."""
        summary = compute_runtime_health(
            ready_count=5,
            candidate_count=5,
            selected_count=1,
            deferred_count=4,
            blocked_count=2,
            total_incomplete_count=7,
            dependency_blocked_count=0,
            fairness_score=1.0,
            starvation_risk="low",
            resource_blocked_count=2,
        )

        assert summary.resource_pressure > 0.7
        assert summary.queue_pressure > 0.5
        assert summary.pressure_level in {"warning", "critical", "overloaded"}
        assert any("resource pressure" in w for w in summary.health_warnings)
        assert any("queue pressure" in w for w in summary.health_warnings)
