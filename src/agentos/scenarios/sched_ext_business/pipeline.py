"""Business DAG definition and frozen SLA-to-queue intent."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from agentos.core.resources import ResourceQuota
from agentos.core.task import Task, TaskDAG
from agentos.scenarios.sched_ext_business.fixture import FixtureManifest


BUSINESS_TASK_IDS = (
    "patch_validation",
    "security_gate",
    "unit_test_select",
    "dependency_analysis",
    "repository_index",
    "artifact_hash",
    "documentation_scan",
    "symbol_index",
)


WORKER_TYPES_BY_TASK = {
    "patch_validation": "patch_validate",
    "security_gate": "security_scan",
    "unit_test_select": "test_select",
    "dependency_analysis": "dependency_analyze",
    "repository_index": "repo_index",
    "artifact_hash": "artifact_hash",
    "documentation_scan": "docs_scan",
    "symbol_index": "symbol_index",
}


_SLA = {
    "patch_validation": ("critical", "latency", 2_000),
    "security_gate": ("critical", "latency", 2_000),
    "unit_test_select": (None, "normal", None),
    "dependency_analysis": (None, "normal", None),
    "repository_index": ("batch", "batch", None),
    "artifact_hash": ("batch", "batch", None),
    "documentation_scan": ("batch", "batch", None),
    "symbol_index": ("batch", "batch", None),
}


def _work_units(
    configured: int | Mapping[str, int],
    task_id: str,
) -> int:
    value = configured[task_id] if isinstance(configured, Mapping) else configured
    parsed = int(value)
    if not 1 <= parsed <= 10_000:
        raise ValueError(f"invalid work_units for {task_id}: {parsed}")
    return parsed


def build_business_dag(
    fixture_root: Path,
    manifest: FixtureManifest,
    *,
    work_units: int | Mapping[str, int],
    kernel_queue_mode: str = "sla",
) -> TaskDAG:
    """Build the triage fan-out DAG used identically by every mode."""

    if kernel_queue_mode not in {"sla", "normal"}:
        raise ValueError(f"unsupported kernel_queue_mode: {kernel_queue_mode!r}")

    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="issue_triage",
            name="freeze issue input and release business workers",
            agent_role="issue_triage",
            metadata={
                "execution": "subprocess",
                "worker_type": "echo",
                "worker_payload": {
                    "result": {
                        "fixture_sha256": manifest.aggregate_sha256,
                        "business_task_count": len(BUSINESS_TASK_IDS),
                    }
                },
                "benchmark_scope": "release_barrier_not_measured_business_work",
            },
            resource_requirements=ResourceQuota(timeout_seconds=5.0),
        )
    )

    for task_id in BUSINESS_TASK_IDS:
        sla_class, queue_class, deadline_ms = _SLA[task_id]
        business_sla_class = sla_class or "normal"
        kernel_sla_class = None if kernel_queue_mode == "normal" else sla_class
        kernel_queue_class = "normal" if kernel_queue_mode == "normal" else queue_class
        task_units = _work_units(work_units, task_id)
        dag.add_task(
            Task(
                task_id=task_id,
                name=task_id,
                agent_role=WORKER_TYPES_BY_TASK[task_id],
                dependencies=["issue_triage"],
                sla_class=kernel_sla_class,
                deadline_ms=deadline_ms,
                workload_hint="cpu_bound",
                io_profile="read_only_fixture",
                resource_requirements=ResourceQuota(
                    timeout_seconds=60.0,
                    allow_network=False,
                    allow_filesystem_write=False,
                ),
                metadata={
                    "execution": "subprocess",
                    "worker_type": WORKER_TYPES_BY_TASK[task_id],
                    "worker_payload": {
                        "fixture_root": str(Path(fixture_root).resolve()),
                        "manifest": manifest.to_dict(),
                        "work_units": task_units,
                    },
                    "expected_queue_class": kernel_queue_class,
                    "business_sla_class": business_sla_class,
                    "business_task": True,
                },
            )
        )
    dag.compute_critical_path_scores()
    return dag
