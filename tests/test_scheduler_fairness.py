"""Tests for scheduler fairness tracking."""

from __future__ import annotations

import pytest

from agentos.scheduler.fairness import (
    FairnessSummary,
    FairnessTracker,
    StarvationRisk,
    TaskFairnessState,
)


class TestTaskFairnessState:
    """Test TaskFairnessState data class."""

    def test_default_values(self):
        state = TaskFairnessState(task_id="t1")
        assert state.task_id == "t1"
        assert state.ready_since is None
        assert state.wait_ms == 0.0
        assert state.selected_count == 0
        assert state.deferred_count == 0
        assert state.blocked_count == 0
        assert state.last_selected_at is None
        assert state.estimated_service_ms == 1.0
        assert state.slowdown == 1.0
        assert state.starvation_risk == StarvationRisk.LOW

    def test_to_dict(self):
        state = TaskFairnessState(
            task_id="t1",
            ready_since=10.0,
            wait_ms=500.0,
            selected_count=2,
            deferred_count=1,
            blocked_count=0,
            last_selected_at=15.0,
            estimated_service_ms=2.0,
            slowdown=1.5,
            starvation_risk=StarvationRisk.MEDIUM,
        )
        d = state.to_dict()
        assert d["task_id"] == "t1"
        assert d["ready_since"] == 10.0
        assert d["wait_ms"] == 500.0
        assert d["selected_count"] == 2
        assert d["deferred_count"] == 1
        assert d["blocked_count"] == 0
        assert d["last_selected_at"] == 15.0
        assert d["estimated_service_ms"] == 2.0
        assert d["slowdown"] == 1.5
        assert d["starvation_risk"] == "medium"


class TestFairnessSummary:
    """Test FairnessSummary data class."""

    def test_default_values(self):
        summary = FairnessSummary()
        assert summary.max_wait_ms == 0.0
        assert summary.avg_wait_ms == 0.0
        assert summary.max_slowdown == 1.0
        assert summary.fairness_score == 1.0
        assert summary.starvation_risk == StarvationRisk.LOW
        assert summary.warnings == []

    def test_to_dict(self):
        summary = FairnessSummary(
            max_wait_ms=1000.0,
            avg_wait_ms=500.0,
            max_slowdown=2.0,
            fairness_score=0.8,
            starvation_risk=StarvationRisk.MEDIUM,
            warnings=[{"task_id": "t1", "reason": "high wait"}],
        )
        d = summary.to_dict()
        assert d["max_wait_ms"] == 1000.0
        assert d["avg_wait_ms"] == 500.0
        assert d["max_slowdown"] == 2.0
        assert d["fairness_score"] == 0.8
        assert d["starvation_risk"] == "medium"
        assert len(d["warnings"]) == 1


class TestFairnessTracker:
    """Test FairnessTracker class."""

    def test_tracker_sets_ready_since_for_first_ready_task(self):
        """Test 1: 首次 ready 开始累计 wait."""
        tracker = FairnessTracker(now=lambda: 10.0)
        tracker.observe_ready(["t1"])
        state = tracker.task_snapshot("t1")
        assert state["ready_since"] == 10.0
        assert state["wait_ms"] == 0.0

    def test_tracker_updates_wait_ms_on_subsequent_observe(self):
        """Test wait_ms updates on subsequent observe_ready calls."""
        tracker = FairnessTracker(now=lambda: 10.0)
        tracker.observe_ready(["t1"])

        # Advance time to 15.0
        tracker._now = lambda: 15.0
        tracker.observe_ready(["t1"])
        state = tracker.task_snapshot("t1")
        assert state["ready_since"] == 10.0
        assert state["wait_ms"] == 5000.0  # 5 seconds = 5000ms

    def test_tracker_records_selected_deferred_and_blocked_counts(self):
        """Test 2: deferred/blocked/selected 计数."""
        tracker = FairnessTracker(now=lambda: 10.0)
        tracker.observe_ready(["a", "b", "c"])
        tracker.record_decision(
            selected_ids=["a"],
            deferred_ids=["b"],
            blocked_ids=["c"],
        )
        state_a = tracker.task_snapshot("a")
        state_b = tracker.task_snapshot("b")
        state_c = tracker.task_snapshot("c")
        assert state_a["selected_count"] == 1
        assert state_a["last_selected_at"] == 10.0
        assert state_b["deferred_count"] == 1
        assert state_c["blocked_count"] == 1

    def test_fairness_score_range(self):
        """Test 3: fairness_score 范围 0 到 1."""
        tracker = FairnessTracker(now=lambda: 10.0)
        tracker.observe_ready(["t1", "t2"])
        tracker.record_decision(selected_ids=["t1"], deferred_ids=["t2"])

        summary = tracker.summarize()
        assert 0.0 <= summary.fairness_score <= 1.0

    def test_fairness_score_decreases_with_worse_conditions(self):
        """Test 4: wait/deferred 变差时 score 降低."""
        # Early state: no wait, no deferrals
        tracker1 = FairnessTracker(now=lambda: 10.0)
        tracker1.observe_ready(["t1", "t2"])
        tracker1.record_decision(selected_ids=["t1", "t2"], deferred_ids=[], blocked_ids=[])
        early_summary = tracker1.summarize()

        # Late state: high wait, many deferrals
        tracker2 = FairnessTracker(now=lambda: 10.0)
        tracker2.observe_ready(["t1", "t2"])
        # Simulate time passing
        tracker2._now = lambda: 20.0
        tracker2.observe_ready(["t1", "t2"])
        tracker2.record_decision(selected_ids=[], deferred_ids=["t1", "t2"], blocked_ids=[])
        late_summary = tracker2.summarize()

        assert late_summary.fairness_score < early_summary.fairness_score

    def test_medium_starvation_risk_from_wait(self):
        """Test 5a: wait 超阈值 -> medium."""
        tracker = FairnessTracker(
            medium_wait_ms=10,
            high_wait_ms=20,
            medium_deferred_count=2,
            high_deferred_count=4,
            now=lambda: 10.0,
        )
        tracker.observe_ready(["t1"])
        # Advance time to exceed medium threshold
        tracker._now = lambda: 10.015  # 15ms > 10ms
        tracker.observe_ready(["t1"])

        summary = tracker.summarize()
        assert summary.starvation_risk in {StarvationRisk.MEDIUM, StarvationRisk.HIGH}

    def test_high_starvation_risk_from_wait(self):
        """Test 5b: wait 超高阈值 -> high."""
        tracker = FairnessTracker(
            medium_wait_ms=10,
            high_wait_ms=20,
            medium_deferred_count=2,
            high_deferred_count=4,
            now=lambda: 10.0,
        )
        tracker.observe_ready(["t1"])
        # Advance time to exceed high threshold
        tracker._now = lambda: 10.025  # 25ms > 20ms
        tracker.observe_ready(["t1"])

        summary = tracker.summarize()
        assert summary.starvation_risk == StarvationRisk.HIGH

    def test_medium_starvation_risk_from_deferred(self):
        """Test 5c: deferred_count 超阈值 -> medium."""
        tracker = FairnessTracker(
            medium_wait_ms=1000,
            high_wait_ms=3000,
            medium_deferred_count=1,
            high_deferred_count=2,
            now=lambda: 10.0,
        )
        tracker.observe_ready(["t1"])
        tracker.record_decision(deferred_ids=["t1"])
        tracker.record_decision(deferred_ids=["t1"])

        summary = tracker.summarize()
        assert summary.starvation_risk in {StarvationRisk.MEDIUM, StarvationRisk.HIGH}

    def test_high_starvation_risk_from_deferred(self):
        """Test 5d: deferred_count 超高阈值 -> high."""
        tracker = FairnessTracker(
            medium_wait_ms=1000,
            high_wait_ms=3000,
            medium_deferred_count=1,
            high_deferred_count=2,
            now=lambda: 10.0,
        )
        tracker.observe_ready(["t1"])
        tracker.record_decision(deferred_ids=["t1"])
        tracker.record_decision(deferred_ids=["t1"])
        tracker.record_decision(deferred_ids=["t1"])

        summary = tracker.summarize()
        assert summary.starvation_risk == StarvationRisk.HIGH

    def test_fairness_warnings_generated(self):
        """Test that warnings are generated for high-risk tasks."""
        tracker = FairnessTracker(
            medium_wait_ms=10,
            high_wait_ms=20,
            medium_deferred_count=1,
            high_deferred_count=2,
            now=lambda: 10.0,
        )
        tracker.observe_ready(["t1"])
        tracker._now = lambda: 10.025  # 25ms > high threshold
        tracker.observe_ready(["t1"])
        tracker.record_decision(deferred_ids=["t1"])
        tracker.record_decision(deferred_ids=["t1"])
        tracker.record_decision(deferred_ids=["t1"])

        summary = tracker.summarize()
        assert len(summary.warnings) > 0
        warning = summary.warnings[0]
        assert warning["task_id"] == "t1"
        assert warning["starvation_risk"] in {"medium", "high"}

    def test_task_snapshot_returns_dict(self):
        """Test task_snapshot returns proper dictionary."""
        tracker = FairnessTracker(now=lambda: 10.0)
        tracker.observe_ready(["t1"])
        snapshot = tracker.task_snapshot("t1")
        assert isinstance(snapshot, dict)
        assert "task_id" in snapshot
        assert "wait_ms" in snapshot
        assert "selected_count" in snapshot
        assert "deferred_count" in snapshot
        assert "blocked_count" in snapshot
        assert "starvation_risk" in snapshot

    def test_unknown_task_snapshot(self):
        """Test snapshot for unknown task returns defaults."""
        tracker = FairnessTracker(now=lambda: 10.0)
        snapshot = tracker.task_snapshot("unknown")
        assert snapshot["task_id"] == "unknown"
        assert snapshot["wait_ms"] == 0.0
        assert snapshot["starvation_risk"] == "low"
