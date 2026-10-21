from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agentos.benchmarks.features import WorkloadFeature, build_workload_feature
from agentos.core.task import Task


class WorkloadClass(str, Enum):
    AI_INFERENCE = "ai_inference"
    COMPILE = "compile"
    COMPRESSION = "compression"
    NETWORK = "network"
    IO_INTENSIVE = "io_intensive"
    INTERACTIVE = "interactive"
    BATCH = "batch"
    UNKNOWN = "unknown"


@dataclass
class WorkloadSample:
    task_id: str | None = None
    agent_role: str | None = None
    task_name: str | None = None
    workload_hint: str | None = None
    resource_requirements: dict[str, object] = field(default_factory=dict)
    prefix_heat: float | None = None
    io_profile: str | None = None
    expected_parallelism: int | None = None
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    feature: WorkloadFeature = field(default_factory=WorkloadFeature)

    @classmethod
    def from_task(
        cls,
        task: Task,
        *,
        prefix_heat: float | None = None,
        feature: WorkloadFeature | None = None,
    ) -> "WorkloadSample":
        metadata = dict(task.metadata)
        resolved_heat = prefix_heat
        if resolved_heat is None and metadata.get("prefix_heat") is not None:
            resolved_heat = float(metadata["prefix_heat"])
        workload_feature = feature or build_workload_feature(
            task=task,
            prefix_heat=resolved_heat,
            source="scheduler",
        )
        return cls(
            task_id=task.task_id,
            agent_role=task.agent_role,
            task_name=task.name,
            workload_hint=task.workload_hint or metadata.get("workload_hint"),
            resource_requirements=dict(workload_feature.resource_requirements),
            prefix_heat=workload_feature.prefix_heat,
            io_profile=task.io_profile or metadata.get("io_profile"),
            expected_parallelism=(
                task.expected_parallelism
                if task.expected_parallelism is not None
                else metadata.get("expected_parallelism")
            ),
            priority=task.priority,
            metadata=metadata,
            feature=workload_feature,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent_role": self.agent_role,
            "task_name": self.task_name,
            "workload_hint": self.workload_hint,
            "resource_requirements": dict(self.resource_requirements),
            "prefix_heat": self.prefix_heat,
            "io_profile": self.io_profile,
            "expected_parallelism": self.expected_parallelism,
            "priority": self.priority,
            "metadata": dict(self.metadata),
            "feature": self.feature.to_dict(),
        }


class WorkloadClassifier:
    _HINTS = {
        "ai": WorkloadClass.AI_INFERENCE,
        "ai_inference": WorkloadClass.AI_INFERENCE,
        "llm": WorkloadClass.AI_INFERENCE,
        "inference": WorkloadClass.AI_INFERENCE,
        "compile": WorkloadClass.COMPILE,
        "build": WorkloadClass.COMPILE,
        "test": WorkloadClass.COMPILE,
        "compression": WorkloadClass.COMPRESSION,
        "compress": WorkloadClass.COMPRESSION,
        "decompress": WorkloadClass.COMPRESSION,
        "archive": WorkloadClass.COMPRESSION,
        "network": WorkloadClass.NETWORK,
        "net": WorkloadClass.NETWORK,
        "io": WorkloadClass.IO_INTENSIVE,
        "io_intensive": WorkloadClass.IO_INTENSIVE,
        "interactive": WorkloadClass.INTERACTIVE,
        "latency": WorkloadClass.INTERACTIVE,
        "batch": WorkloadClass.BATCH,
        "bulk": WorkloadClass.BATCH,
        "unknown": WorkloadClass.UNKNOWN,
    }

    def classify(self, sample: WorkloadSample) -> WorkloadClass:
        explicit = self._class_from_hint(sample.workload_hint)
        if explicit is not None:
            return explicit

        feature_class = self._class_from_hint(sample.feature.workload_class)
        if feature_class is not None:
            return feature_class

        metadata = sample.metadata
        if self._has_llm_signal(sample):
            return WorkloadClass.AI_INFERENCE
        if self._has_network_signal(sample):
            return WorkloadClass.NETWORK
        text = " ".join(
            part.lower()
            for part in (sample.task_name or "", sample.agent_role or "")
        )
        if self._contains_any(text, ("compile", "build", "cargo", "make", "test")):
            return WorkloadClass.COMPILE
        if self._contains_any(
            text,
            ("compress", "decompress", "archive", "zip", "gzip", "tar"),
        ):
            return WorkloadClass.COMPRESSION
        if self._has_io_signal(sample):
            return WorkloadClass.IO_INTENSIVE
        if self._has_interactive_signal(sample):
            return WorkloadClass.INTERACTIVE
        if self._has_batch_signal(sample):
            return WorkloadClass.BATCH
        if metadata.get("workload_class"):
            return WorkloadClass.UNKNOWN
        return WorkloadClass.UNKNOWN

    def _class_from_hint(self, value: object) -> WorkloadClass | None:
        if value is None:
            return None
        normalized = str(value).strip().lower().replace("-", "_")
        return self._HINTS.get(normalized)

    def _has_llm_signal(self, sample: WorkloadSample) -> bool:
        feature = sample.feature
        metadata = sample.metadata
        return any(
            (
                feature.prompt_tokens > 0,
                feature.completion_tokens > 0,
                feature.cache_hit_tokens > 0,
                feature.cache_miss_tokens > 0,
                metadata.get("prompt_tokens", 0) > 0,
                metadata.get("completion_tokens", 0) > 0,
                metadata.get("cache_hit_tokens", 0) > 0,
                metadata.get("cache_miss_tokens", 0) > 0,
                metadata.get("llm_backend") is not None,
            )
        )

    def _has_network_signal(self, sample: WorkloadSample) -> bool:
        metadata = sample.metadata
        resources = sample.resource_requirements
        return bool(
            resources.get("allow_network")
            or metadata.get("allow_network")
            or metadata.get("request_count")
            or metadata.get("network_requests")
            or metadata.get("throughput_rps")
        )

    def _has_io_signal(self, sample: WorkloadSample) -> bool:
        resources = sample.resource_requirements
        return bool(
            sample.io_profile
            or resources.get("io_read_bps")
            or resources.get("io_write_bps")
            or sample.metadata.get("io_profile")
        )

    def _has_interactive_signal(self, sample: WorkloadSample) -> bool:
        metadata = sample.metadata
        return bool(
            sample.priority >= 8
            or metadata.get("deadline_ms") is not None
            or metadata.get("deadline_secs") is not None
            or metadata.get("user_request")
            or metadata.get("interactive")
        )

    def _has_batch_signal(self, sample: WorkloadSample) -> bool:
        metadata = sample.metadata
        expected_parallelism = sample.expected_parallelism or 0
        return bool(
            expected_parallelism >= 4
            or metadata.get("batch_size", 0) >= 4
            or metadata.get("can_wait")
        )

    @staticmethod
    def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
        return any(needle in text for needle in needles)
