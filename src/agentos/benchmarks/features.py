from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

from agentos.core.resources import ResourceQuota
from agentos.core.task import Task
from agentos.llm.base import LLMRequest
from agentos.metrics.model import ModelCallMetrics


def _prefix_id_from_request(request: LLMRequest) -> str:
    if request.prefix_id:
        return request.prefix_id
    system_parts = [
        message["content"]
        for message in request.messages
        if message.get("role") == "system"
    ]
    if not system_parts:
        return "__no_prefix__"
    raw = "\n".join(system_parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _resource_dict(quota: ResourceQuota | None) -> dict[str, object]:
    return quota.to_dict() if quota is not None else ResourceQuota().to_dict()


@dataclass
class WorkloadFeature:
    """Scheduler-facing workload feature bundle.

    This is intentionally broader than the current scheduler needs. B-line
    modules can fill LLM/context/cache fields now, while A/C can later add
    classifier and workload-matrix consumers without reaching into internals.
    """

    source: str = "unknown"
    task_id: str | None = None
    agent_id: str | None = None
    agent_role: str | None = None
    workload_class: str | None = None
    workload_hint: str | None = None
    io_profile: str | None = None
    expected_parallelism: int | None = None

    prefix_id: str | None = None
    prefix_heat: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    cache_hit_ratio: float = 0.0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    prefill_secs: float = 0.0
    decode_secs: float = 0.0
    prefill_decode_available: bool = False
    prefill_decode_fallback: bool = False
    prefill_saved_tokens: int = 0
    prefill_miss_tokens: int = 0
    latency_secs: float = 0.0

    shared_refs: int = 0
    private_refs: int = 0
    total_refs: int = 0
    duplicate_tokens: int = 0
    compressed_tokens: int = 0
    active_wss_tokens: int = 0
    context_reuse_ratio: float = 0.0

    resource_requirements: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_task(
        cls,
        task: Task,
        *,
        context_stats: dict[str, Any] | None = None,
        prefix_heat: float | None = None,
        source: str = "task",
    ) -> "WorkloadFeature":
        metadata = dict(task.metadata)
        feature = cls(
            source=source,
            task_id=task.task_id,
            agent_role=task.agent_role,
            workload_class=metadata.get("workload_class"),
            workload_hint=task.workload_hint or metadata.get("workload_hint"),
            io_profile=task.io_profile or metadata.get("io_profile"),
            expected_parallelism=(
                task.expected_parallelism
                if task.expected_parallelism is not None
                else metadata.get("expected_parallelism")
            ),
            prefix_heat=prefix_heat,
            resource_requirements=_resource_dict(task.resource_requirements),
            metadata=metadata,
        )
        if context_stats is not None:
            feature.apply_context_stats(context_stats)
        return feature

    @classmethod
    def from_llm_call(
        cls,
        request: LLMRequest,
        metrics: ModelCallMetrics,
        *,
        backend: str,
        model: str,
        prefix_id: str | None = None,
        prefix_heat: float | None = None,
    ) -> "WorkloadFeature":
        metadata = dict(request.metadata)
        metric_metadata = dict(metrics.metadata)
        resolved_prefix = prefix_id or _prefix_id_from_request(request)
        return cls(
            source="llm",
            task_id=request.task_id,
            agent_id=request.agent_id,
            agent_role=metadata.get("agent_role"),
            workload_class="ai_inference",
            workload_hint=metadata.get("workload_hint", "ai_inference"),
            io_profile=metadata.get("io_profile"),
            expected_parallelism=metadata.get("expected_parallelism"),
            prefix_id=resolved_prefix,
            prefix_heat=prefix_heat,
            prompt_tokens=metrics.prompt_tokens,
            completion_tokens=metrics.completion_tokens,
            cache_hit_tokens=metrics.cache_hit_tokens,
            cache_miss_tokens=metrics.cache_miss_tokens,
            cache_hit_ratio=metrics.cache_hit_ratio,
            prefill_tokens=int(metric_metadata.get("prefill_tokens", metrics.prompt_tokens)),
            decode_tokens=int(metric_metadata.get("decode_tokens", metrics.completion_tokens)),
            prefill_secs=float(metric_metadata.get("prefill_secs", 0.0)),
            decode_secs=float(metric_metadata.get("decode_secs", 0.0)),
            prefill_decode_available=bool(
                metric_metadata.get("prefill_decode_available", False)
            ),
            prefill_decode_fallback=bool(
                metric_metadata.get("prefill_decode_fallback", False)
            ),
            prefill_saved_tokens=int(
                metric_metadata.get("prefill_saved_tokens", metrics.cache_hit_tokens)
            ),
            prefill_miss_tokens=int(
                metric_metadata.get("prefill_miss_tokens", metrics.cache_miss_tokens)
            ),
            latency_secs=metrics.latency_secs,
            metadata={
                **metadata,
                "backend": backend,
                "model": model,
                "batch_size": metrics.batch_size,
                "queue_wait_secs": metrics.queue_wait_secs,
                **metric_metadata,
            },
        )

    def apply_context_stats(self, stats: dict[str, Any]) -> None:
        self.shared_refs = int(stats.get("shared_refs", stats.get("shared_segments", 0)))
        self.private_refs = int(stats.get("private_refs", 0))
        self.total_refs = int(stats.get("total_refs", 0))
        self.duplicate_tokens = int(stats.get("duplicate_tokens", 0))
        self.compressed_tokens = int(stats.get("compressed_tokens", 0))
        self.active_wss_tokens = int(stats.get("active_wss_tokens", 0))
        self.context_reuse_ratio = float(stats.get("context_reuse_ratio", 0.0))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_workload_feature(
    *,
    task: Task | None = None,
    llm_request: LLMRequest | None = None,
    model_metrics: ModelCallMetrics | None = None,
    context_stats: dict[str, Any] | None = None,
    backend: str = "default",
    model: str = "default",
    prefix_id: str | None = None,
    prefix_heat: float | None = None,
    source: str = "manual",
) -> WorkloadFeature:
    """Build one scheduler-facing feature bundle from available signals."""
    if llm_request is not None and model_metrics is not None:
        feature = WorkloadFeature.from_llm_call(
            llm_request,
            model_metrics,
            backend=backend,
            model=model,
            prefix_id=prefix_id,
            prefix_heat=prefix_heat,
        )
    elif task is not None:
        feature = WorkloadFeature.from_task(
            task,
            context_stats=context_stats,
            prefix_heat=prefix_heat,
            source=source,
        )
    else:
        feature = WorkloadFeature(source=source, prefix_id=prefix_id, prefix_heat=prefix_heat)
        if context_stats is not None:
            feature.apply_context_stats(context_stats)

    if task is not None:
        feature.task_id = feature.task_id or task.task_id
        feature.agent_role = feature.agent_role or task.agent_role
        feature.resource_requirements = _resource_dict(task.resource_requirements)
        task_metadata = dict(task.metadata)
        feature.workload_hint = feature.workload_hint or task_metadata.get("workload_hint")
        feature.io_profile = feature.io_profile or task_metadata.get("io_profile")
        feature.expected_parallelism = (
            feature.expected_parallelism
            if feature.expected_parallelism is not None
            else task.expected_parallelism or task_metadata.get("expected_parallelism")
        )
        feature.metadata = {**task_metadata, **feature.metadata}

    if context_stats is not None:
        feature.apply_context_stats(context_stats)

    return feature
