import pytest

from agentos.core.resources import ResourceQuota
from agentos.core.task import Task
from agentos.scheduler.workload import (
    WorkloadClass,
    WorkloadClassifier,
    WorkloadSample,
)


def classify(task: Task) -> WorkloadClass:
    return WorkloadClassifier().classify(WorkloadSample.from_task(task))


def test_workload_sample_preserves_scheduler_visible_task_fields():
    task = Task(
        task_id="infer",
        name="answer user",
        agent_role="assistant",
        workload_hint="ai_inference",
        resource_requirements=ResourceQuota(memory_max_mb=512),
        context_refs=["shared-system"],
        metadata={"prefix_heat": 0.84, "custom": "kept"},
    )

    sample = WorkloadSample.from_task(task)

    assert sample.task_id == "infer"
    assert sample.agent_role == "assistant"
    assert sample.workload_hint == "ai_inference"
    assert sample.resource_requirements["memory_max_mb"] == 512
    assert sample.prefix_heat == pytest.approx(0.84)
    assert sample.metadata["custom"] == "kept"
    assert sample.feature.task_id == "infer"


def test_explicit_workload_hint_wins_over_heuristics():
    task = Task(
        task_id="batch-network",
        name="bulk http fetch",
        agent_role="crawler",
        workload_hint="batch",
        resource_requirements=ResourceQuota(allow_network=True),
    )

    assert classify(task) == WorkloadClass.BATCH


def test_classifier_recognizes_ai_inference_from_llm_signals():
    task = Task(
        task_id="infer",
        name="generate response",
        agent_role="assistant",
        metadata={"prompt_tokens": 1200, "cache_hit_tokens": 800},
    )

    assert classify(task) == WorkloadClass.AI_INFERENCE


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        (
            Task(task_id="compile", name="cargo build kernel", agent_role="builder"),
            WorkloadClass.COMPILE,
        ),
        (
            Task(task_id="zip", name="archive compress logs", agent_role="worker"),
            WorkloadClass.COMPRESSION,
        ),
        (
            Task(
                task_id="net",
                name="download dataset",
                agent_role="fetcher",
                resource_requirements=ResourceQuota(allow_network=True),
            ),
            WorkloadClass.NETWORK,
        ),
        (
            Task(
                task_id="io",
                name="scan files",
                agent_role="worker",
                io_profile="sequential_read",
                resource_requirements=ResourceQuota(io_read_bps=4096),
            ),
            WorkloadClass.IO_INTENSIVE,
        ),
        (
            Task(
                task_id="chat",
                name="serve user request",
                agent_role="assistant",
                priority=9,
                metadata={"deadline_ms": 250},
            ),
            WorkloadClass.INTERACTIVE,
        ),
        (
            Task(
                task_id="batch",
                name="offline map reduce",
                agent_role="worker",
                expected_parallelism=8,
            ),
            WorkloadClass.BATCH,
        ),
    ],
)
def test_classifier_recognizes_p0_workload_classes(task, expected):
    assert classify(task) == expected


def test_classifier_returns_unknown_without_clear_signal():
    task = Task(task_id="plain", name="misc task", agent_role="worker")

    assert classify(task) == WorkloadClass.UNKNOWN
