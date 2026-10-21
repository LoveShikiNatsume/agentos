from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from agentos.context.cache_tracker import CacheReusePrediction
from agentos.llm.base import LLMRequest


@dataclass(frozen=True)
class BatchPolicy:
    max_wait_ms: int = 30
    max_batch_size: int = 8
    min_prefix_group_size: int = 2
    respect_deadline: bool = True


@dataclass
class BatchGroup:
    request_indices: list[int]
    prefix_id: str
    reason: str
    predicted_hit_tokens: int = 0
    max_wait_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BatchPlan:
    groups: list[BatchGroup] = field(default_factory=list)
    policy: BatchPolicy = field(default_factory=BatchPolicy)
    created_at: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict[str, Any]:
        return {
            "groups": [group.to_dict() for group in self.groups],
            "policy": asdict(self.policy),
            "created_at": self.created_at,
        }


class AdaptiveBatcher:
    """Plans model batches using prefix affinity and latency budgets."""

    def __init__(
        self,
        prefix_resolver: Callable[[LLMRequest], str],
        *,
        prediction_resolver: Callable[[str], CacheReusePrediction | None] | None = None,
        policy: BatchPolicy | None = None,
    ) -> None:
        self._prefix_resolver = prefix_resolver
        self._prediction_resolver = prediction_resolver
        self._policy = policy or BatchPolicy()

    def plan(self, requests: list[LLMRequest]) -> BatchPlan:
        if not requests:
            return BatchPlan(policy=self._policy)

        groups_by_prefix: dict[str, list[int]] = {}
        for idx, req in enumerate(requests):
            prefix_id = self._prefix_resolver(req)
            groups_by_prefix.setdefault(prefix_id, []).append(idx)

        group_specs: list[tuple[str, list[int], int]] = []
        for prefix_id, indices in groups_by_prefix.items():
            predicted = self._predicted_hit_tokens(prefix_id)
            group_specs.append((prefix_id, indices, predicted))

        group_specs.sort(
            key=lambda item: (
                -item[2],
                self._earliest_deadline_ms(requests, item[1]),
                item[0],
            )
        )

        planned: list[BatchGroup] = []
        for prefix_id, indices, predicted in group_specs:
            chunks = [
                indices[i:i + self._policy.max_batch_size]
                for i in range(0, len(indices), self._policy.max_batch_size)
            ]
            for chunk in chunks:
                reason = self._group_reason(requests, chunk, predicted)
                planned.append(
                    BatchGroup(
                        request_indices=chunk,
                        prefix_id=prefix_id,
                        reason=reason,
                        predicted_hit_tokens=predicted,
                        max_wait_ms=self._max_wait_for_group(requests, chunk),
                    )
                )

        return BatchPlan(groups=planned, policy=self._policy)

    def _group_reason(
        self,
        requests: list[LLMRequest],
        indices: list[int],
        predicted_hit_tokens: int,
    ) -> str:
        if self._deadline_is_tight(requests, indices):
            return "deadline_flush"
        if len(indices) >= self._policy.min_prefix_group_size:
            return "prefix_affinity"
        if predicted_hit_tokens > 0:
            return "predicted_cache_reuse"
        return "single_request"

    def _max_wait_for_group(
        self,
        requests: list[LLMRequest],
        indices: list[int],
    ) -> int:
        if not self._policy.respect_deadline:
            return self._policy.max_wait_ms
        deadline = self._earliest_deadline_ms(requests, indices)
        if deadline is None:
            return self._policy.max_wait_ms
        return max(0, min(self._policy.max_wait_ms, deadline))

    def _deadline_is_tight(
        self,
        requests: list[LLMRequest],
        indices: list[int],
    ) -> bool:
        if not self._policy.respect_deadline:
            return False
        deadline = self._earliest_deadline_ms(requests, indices)
        return deadline is not None and deadline <= self._policy.max_wait_ms

    @staticmethod
    def _earliest_deadline_ms(
        requests: list[LLMRequest],
        indices: list[int],
    ) -> int | None:
        deadlines = [
            int(requests[idx].metadata["deadline_ms"])
            for idx in indices
            if "deadline_ms" in requests[idx].metadata
        ]
        return min(deadlines) if deadlines else None

    def _predicted_hit_tokens(self, prefix_id: str) -> int:
        if self._prediction_resolver is None:
            return 0
        prediction = self._prediction_resolver(prefix_id)
        return prediction.expected_hit_tokens if prediction is not None else 0
