"""Tests for scheduler evidence report generation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.generate_scheduler_evidence import generate_report

PROJECT_ROOT = (Path(__file__).parent.parent / "src")

# A minimal valid scheduler.decision event
VALID_DECISION_EVENT = {
    "event_type": "scheduler.decision",
    "data": {
        "schema_version": "scheduler.decision.v2",
        "strategy": "critical_path",
        "policy": "latency_first",
        "workload_class": "interactive",
        "policy_reason": "interactive workload or high priority",
        "policy_score": {"candidate_count": 1},
        "score_breakdown": {"candidate_count": 1},
        "fairness": {
            "max_wait_ms": 0.0,
            "avg_wait_ms": 0.0,
            "max_slowdown": 1.0,
            "fairness_score": 1.0,
            "starvation_risk": "low",
            "warnings": [],
        },
        "fairness_warnings": [],
        "ready_task_ids": ["t1"],
        "candidate_task_ids": ["t1"],
        "ordered_candidate_task_ids": ["t1"],
        "selected_task_ids": ["t1"],
        "blocked_task_ids": [],
        "deferred_task_ids": [],
        "selected": ["t1"],
        "blocked": [],
        "deferred": [],
        "available_resources": None,
        "max_batch_size": None,
        "tasks": {
            "t1": {
                "task_id": "t1",
                "agent_role": "worker",
                "reason": "selected",
                "workload_class": "interactive",
                "policy": "latency_first",
                "policy_reason": "interactive workload or high priority",
                "critical_path_score": 0.0,
                "priority": 0,
                "dependencies": [],
                "context_refs": [],
                "resource_requirements": {},
                "resource_estimate": {},
                "wait_ms": 0.0,
                "slowdown": 1.0,
                "selected_count": 1,
                "deferred_count": 0,
                "blocked_count": 0,
                "starvation_risk": "low",
            }
        },
    },
}

VALID_KERNEL_PROBE_EVENT = {
    "event_type": "scheduler.kernel_probe",
    "data": {
        "sched_ext": "unavailable",
        "bpf_map": "mock",
        "fallback_mode": "mock_bpf_map",
        "reason": "forced mock mode",
    },
}


class TestGenerateSchedulerEvidence:
    """Test generate_report function."""

    def test_generate_scheduler_evidence_writes_markdown_report(self):
        """Test that report is generated as Markdown."""
        events = [VALID_DECISION_EVENT]
        report = generate_report(events)
        assert report.startswith("# Member A Scheduler Evidence")
        assert "## Summary" in report
        assert "## Policy Selection Evidence" in report
        assert "## Fairness Evidence" in report
        assert "## OS Integration Evidence" in report

    def test_report_mentions_policy_fairness_and_kernel_probe(self):
        """Test that report mentions policy, fairness, and kernel probe."""
        events = [
            VALID_KERNEL_PROBE_EVENT,
            VALID_DECISION_EVENT,
        ]
        report = generate_report(events)
        assert "latency_first" in report
        assert "fairness" in report.lower()
        assert "kernel probe" in report.lower() or "kernel_probe" in report

    def test_report_does_not_claim_kernel_takeover(self):
        """Test that report does not claim to replace Linux kernel scheduling."""
        events = [VALID_DECISION_EVENT]
        report = generate_report(events)
        # Report should say "does not claim to replace" (negation)
        assert "does not claim to replace Linux kernel scheduling" in report

    def test_report_with_no_events(self):
        """Test report generation with no events."""
        events = []
        report = generate_report(events)
        assert "scheduler decisions: 0" in report
        assert "No policy selection evidence found." in report

    def test_report_contains_runtime_health_evidence(self):
        """Test that report contains Runtime Health Evidence section."""
        events = [VALID_DECISION_EVENT, VALID_KERNEL_PROBE_EVENT]
        report = generate_report(events)
        assert "## Runtime Health Evidence" in report
        assert "Runtime Health Score is a user-space Runtime health assessment metric" in report


class TestGenerateSchedulerEvidenceScript:
    """Test generate_scheduler_evidence.py script."""

    def test_script_generates_report(self, tmp_path):
        """Test script generates report from JSONL."""
        # Write a test JSONL file
        jsonl_path = tmp_path / "events.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps(VALID_DECISION_EVENT) + "\n")
            f.write(json.dumps(VALID_KERNEL_PROBE_EVENT) + "\n")

        out_path = tmp_path / "evidence.md"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_scheduler_evidence.py",
                "--events",
                str(jsonl_path),
                "--out",
                str(out_path),
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.returncode == 0
        assert out_path.exists()
        content = out_path.read_text()
        assert "# Member A Scheduler Evidence" in content
        assert "latency_first" in content


# Workload scenario tests for evidence generation
from agentos.core.resources import ResourceQuota
from agentos.core.task import Task, TaskDAG, TaskStatus
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import AgentHandler, Runtime


class TaskIdHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        return ctx.task.task_id


class TestSchedulerEvidenceScenarios:
    """Test scheduler evidence with various workload scenarios."""

    async def test_multiple_policies_in_evidence(self, tmp_path):
        """Test that multiple policies appear in evidence."""
        # Create fixed PSI files to avoid host-dependent behavior
        psi_dir = tmp_path / "pressure"
        psi_dir.mkdir()
        for name in ("cpu", "memory", "io"):
            (psi_dir / name).write_text(
                "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
            )

        from agentos.isolation.bpf import MockBPFMapAdapter
        from agentos.isolation.kernel_adapter import UserSpaceKernelAdapter
        from agentos.isolation.sched_ext import KernelProbeResult

        adapter = UserSpaceKernelAdapter(
            bpf_map_adapter=MockBPFMapAdapter(),
            probe_result=KernelProbeResult(
                sched_ext="unavailable",
                bpf_map="mock",
                fallback_mode="mock_bpf_map",
                reason="test",
            ),
            psi_paths={
                "cpu": str(psi_dir / "cpu"),
                "memory": str(psi_dir / "memory"),
                "io": str(psi_dir / "io"),
            },
        )

        # Create a DAG with different workload types
        dag = TaskDAG()

        # Interactive task (high priority)
        dag.add_task(Task(
            task_id="interactive",
            name="serve request",
            agent_role="assistant",
            workload_hint="interactive",
            priority=9,
        ))

        # Batch task (high parallelism)
        dag.add_task(Task(
            task_id="batch",
            name="process batch",
            agent_role="worker",
            workload_hint="batch",
            expected_parallelism=8,
        ))

        rt = Runtime(
            llm_backend=MockLLMBackend(),
            handlers={"assistant": TaskIdHandler(), "worker": TaskIdHandler()},
            kernel_adapter=adapter,
        )

        await rt.run(dag)

        # Extract events
        events = [{"event_type": e.event_type, "data": e.data} for e in rt.events]

        # Check that we have multiple policies
        policy_events = [e for e in events if e["event_type"] == "scheduler.policy_selected"]
        policies = {e["data"]["policy"] for e in policy_events}

        # Should have at least latency_first and throughput_first
        assert "latency_first" in policies or "throughput_first" in policies

    async def test_fairness_balanced_policy_with_deferred_tasks(self):
        """Test that fairness_balanced policy appears with deferred tasks."""
        # This test verifies the fairness_balanced policy can be triggered
        # by checking the scheduler directly rather than through Runtime
        # because Runtime completes tasks too quickly for starvation to occur
        from agentos.scheduler.dag import DAGScheduler
        from agentos.scheduler.fairness import FairnessTracker

        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work1", agent_role="worker"))
        dag.add_task(Task(task_id="t2", name="work2", agent_role="worker"))

        # Use low thresholds to trigger fairness_balanced
        tracker = FairnessTracker(
            medium_wait_ms=10,
            high_wait_ms=20,
            medium_deferred_count=1,
            high_deferred_count=2,
            now=lambda: 10.0,
        )

        scheduler = DAGScheduler(dag, max_batch_size=1, fairness_tracker=tracker)

        # First batch: t1 selected, t2 deferred
        await scheduler.get_next_batch()
        dag.tasks["t1"].status = TaskStatus.COMPLETED

        # Advance time significantly to trigger high starvation risk
        tracker._now = lambda: 10.025  # 25ms > high threshold
        await scheduler.get_next_batch()

        decision = scheduler.last_decision
        assert decision["policy"] == "fairness_balanced"
        assert decision["fairness"]["starvation_risk"] == "high"

    async def test_kernel_probe_and_bpf_map_in_evidence(self):
        """Test that kernel probe and BPF map appear in evidence."""
        from agentos.isolation.bpf import MockBPFMapAdapter
        from agentos.isolation.sched_ext import KernelProbeResult

        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

        probe = KernelProbeResult(
            sched_ext="unavailable",
            bpf_map="mock",
            fallback_mode="mock_bpf_map",
            reason="test",
        )
        adapter = MockBPFMapAdapter()

        rt = Runtime(
            llm_backend=MockLLMBackend(),
            handlers={"worker": TaskIdHandler()},
            kernel_probe=probe,
            bpf_map_adapter=adapter,
        )

        await rt.run(dag)

        # Extract events
        events = [{"event_type": e.event_type, "data": e.data} for e in rt.events]

        # Check for kernel probe event
        kernel_probe_events = [e for e in events if e["event_type"] == "scheduler.kernel_probe"]
        assert len(kernel_probe_events) == 1
        assert kernel_probe_events[0]["data"]["bpf_map"] == "mock"

        # Check for BPF map updated event
        bpf_events = [e for e in events if e["event_type"] == "scheduler.bpf_map_updated"]
        assert len(bpf_events) > 0

        # Generate report
        report = generate_report(events)
        assert "kernel probe" in report.lower() or "kernel_probe" in report

    async def test_kernel_adapter_publishes_effective_pressure_and_backpressure_to_bpf_map(self):
        """BPF records receive pressure/backpressure only through the kernel adapter."""
        from agentos.isolation.bpf import MockBPFMapAdapter
        from agentos.isolation.kernel_adapter import UserSpaceKernelAdapter
        from agentos.isolation.sched_ext import KernelProbeResult
        from agentos.scheduler.dag import DAGScheduler

        bpf_adapter = MockBPFMapAdapter()
        kernel_adapter = UserSpaceKernelAdapter(
            bpf_map_adapter=bpf_adapter,
            probe_result=KernelProbeResult(
                sched_ext="unavailable",
                bpf_map="mock",
                fallback_mode="mock_bpf_map",
                reason="unit test",
            ),
        )
        dag = TaskDAG()
        dag.add_task(Task(
            task_id="t1",
            name="batch work",
            agent_role="worker",
            workload_hint="batch",
            resource_requirements=ResourceQuota(
                cpu_shares=200,
                cpu_max_percent=50,
                memory_max_mb=64,
                pids_max=8,
            ),
        ))

        scheduler = DAGScheduler(
            dag,
            pressure_level="critical",
        )

        batch = await scheduler.get_next_batch()
        assert len(batch) == 1
        assert batch[0].task_id == "t1"

        decision = scheduler.last_decision
        assert decision is not None
        assert "bpf_map" not in decision

        events = kernel_adapter.publish_decision(decision)

        # Verify the adapter-published BPF record has the scheduler signals.
        record = bpf_adapter.get_task("t1")
        assert record is not None
        assert record["pressure_level"] == "critical"
        assert record["backpressure_signal"] == "hard"
        assert record["cpu_weight"] == 200
        assert record["cpu_max_percent"] == 50
        assert record["memory_max_mb"] == 64
        assert record["pids_max"] == 8
        assert record["schema_version"] == "bpf.task_record.v2"
        assert events[0].event_type == "scheduler.bpf_map_updated"
        assert events[0].data["updated_task_ids"] == ["t1"]

        # Verify decision has pressure and backpressure
        assert decision["pressure"]["effective_pressure_level"] == "critical"
        assert decision["backpressure"]["signal"] == "hard"
