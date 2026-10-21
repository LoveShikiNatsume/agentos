from __future__ import annotations

import asyncio
import hashlib
import time
from collections import defaultdict
from typing import Any

from agentos.benchmarks.features import WorkloadFeature
from agentos.context.cache_tracker import CacheHeatTracker
from agentos.llm.base import LLMBackend, LLMRequest, LLMResponse
from agentos.llm.batcher import AdaptiveBatcher, BatchPlan, BatchPolicy
from agentos.metrics.model import ModelCallMetrics


class ModelGateway:
    """Unified model routing gateway implementing the LLMBackend protocol.

    Routes requests to named backends based on request.backend field.
    Ensures every response carries ModelCallMetrics.
    """

    def __init__(
        self,
        backends: dict[str, LLMBackend] | None = None,
        default_backend: str = "default",
        cache_tracker: CacheHeatTracker | None = None,
    ) -> None:
        self._backends: dict[str, LLMBackend] = dict(backends) if backends else {}
        self._default_backend = default_backend
        self._call_log: list[dict[str, Any]] = []
        self._workload_features: list[WorkloadFeature] = []
        self._cache_tracker = cache_tracker
        self._batch_plans: list[dict[str, Any]] = []

    def register(self, name: str, backend: LLMBackend) -> None:
        self._backends[name] = backend

    def get_backend(self, name: str) -> LLMBackend | None:
        return self._backends.get(name)

    @property
    def backend_names(self) -> list[str]:
        return list(self._backends.keys())

    @property
    def cache_tracker(self) -> CacheHeatTracker | None:
        return self._cache_tracker

    async def generate(self, request: LLMRequest) -> LLMResponse:
        backend_name = request.backend if request.backend != "default" else self._default_backend
        backend = self._backends.get(backend_name)
        if backend is None and backend_name != self._default_backend:
            backend = self._backends.get(self._default_backend)
        if backend is None:
            raise ValueError(
                f"No backend registered for '{backend_name}' "
                f"(registered: {list(self._backends.keys())})"
            )

        start = time.monotonic()
        resp = await backend.generate(request)
        elapsed = time.monotonic() - start

        if resp.metrics is not None:
            metrics = ModelCallMetrics(
                backend=backend_name,
                model=resp.metrics.model,
                prompt_tokens=resp.metrics.prompt_tokens,
                completion_tokens=resp.metrics.completion_tokens,
                cache_hit_tokens=resp.metrics.cache_hit_tokens,
                cache_miss_tokens=resp.metrics.cache_miss_tokens,
                latency_secs=elapsed,
                queue_wait_secs=resp.metrics.queue_wait_secs,
                batch_size=resp.metrics.batch_size,
                metadata=resp.metrics.metadata,
            )
        else:
            metrics = ModelCallMetrics(
                backend=backend_name,
                model=resp.model,
                prompt_tokens=resp.metadata.get("prompt_tokens", 0),
                completion_tokens=resp.metadata.get("completion_tokens", 0),
                latency_secs=elapsed,
            )

        result = LLMResponse(
            content=resp.content,
            tokens_used=resp.tokens_used,
            model=resp.model,
            latency_secs=elapsed,
            backend=backend_name,
            metrics=metrics,
            metadata=resp.metadata,
        )

        prefix_id = self._prefix_id(request)
        prefix_heat = None
        cache_available = metrics.metadata.get("cache_metrics_available", True)
        if self._cache_tracker is not None and cache_available:
            self._cache_tracker.record_call(prefix_id, metrics)
            prefix_heat = self._cache_tracker.get_prefix_heat(prefix_id)

        feature = WorkloadFeature.from_llm_call(
            request,
            metrics,
            backend=backend_name,
            model=resp.model,
            prefix_id=prefix_id,
            prefix_heat=prefix_heat,
        )
        self._workload_features.append(feature)

        self._call_log.append({
            "backend": backend_name,
            "model": resp.model,
            "tokens_used": resp.tokens_used,
            "prompt_tokens": metrics.prompt_tokens,
            "completion_tokens": metrics.completion_tokens,
            "cache_hit_tokens": metrics.cache_hit_tokens,
            "cache_miss_tokens": metrics.cache_miss_tokens,
            "cache_hit_ratio": metrics.cache_hit_ratio if cache_available else None,
            "cache_metrics_available": cache_available,
            "prefix_id": prefix_id,
            "prefill_tokens": feature.prefill_tokens,
            "decode_tokens": feature.decode_tokens,
            "prefill_secs": feature.prefill_secs,
            "decode_secs": feature.decode_secs,
            "prefill_decode_available": feature.prefill_decode_available,
            "prefill_decode_fallback": feature.prefill_decode_fallback,
            "prefill_saved_tokens": feature.prefill_saved_tokens,
            "prefill_miss_tokens": feature.prefill_miss_tokens,
            "task_id": request.task_id,
            "latency_secs": elapsed,
            "agent_id": request.agent_id,
            "timestamp": time.time(),
        })

        return result

    @staticmethod
    def _prefix_id(request: LLMRequest) -> str:
        if request.prefix_id:
            return request.prefix_id
        system_parts = [
            m["content"]
            for m in request.messages
            if m.get("role") == "system"
        ]
        if system_parts:
            raw = "\n".join(system_parts)
            return hashlib.sha256(raw.encode()).hexdigest()[:16]
        return "__no_prefix__"

    def group_by_prefix(
        self,
        requests: list[LLMRequest],
    ) -> dict[str, list[LLMRequest]]:
        groups: dict[str, list[LLMRequest]] = defaultdict(list)
        for req in requests:
            groups[self._prefix_id(req)].append(req)
        return dict(groups)

    async def generate_batch(
        self,
        requests: list[LLMRequest],
    ) -> list[LLMResponse]:
        """Same-prefix requests run sequentially (to build cache), different prefixes run in parallel."""
        if not requests:
            return []

        index_groups: dict[str, list[int]] = defaultdict(list)
        for i, req in enumerate(requests):
            index_groups[self._prefix_id(req)].append(i)

        if len(index_groups) == 1:
            return [await self.generate(req) for req in requests]

        async def _run_group(indices: list[int]) -> list[tuple[int, LLMResponse]]:
            results = []
            for idx in indices:
                resp = await self.generate(requests[idx])
                results.append((idx, resp))
            return results

        group_results = await asyncio.gather(
            *[_run_group(indices) for indices in index_groups.values()]
        )

        all_results: list[tuple[int, LLMResponse]] = []
        for group in group_results:
            all_results.extend(group)
        all_results.sort(key=lambda x: x[0])
        return [resp for _, resp in all_results]

    def plan_adaptive_batch(
        self,
        requests: list[LLMRequest],
        policy: BatchPolicy | None = None,
    ) -> BatchPlan:
        batcher = AdaptiveBatcher(
            self._prefix_id,
            prediction_resolver=(
                self._cache_tracker.predict_reuse
                if self._cache_tracker is not None
                else None
            ),
            policy=policy,
        )
        return batcher.plan(requests)

    async def generate_adaptive_batch(
        self,
        requests: list[LLMRequest],
        policy: BatchPolicy | None = None,
    ) -> list[LLMResponse]:
        """Execute requests according to an adaptive prefix/deadline batch plan."""
        if not requests:
            return []

        plan = self.plan_adaptive_batch(requests, policy=policy)
        self._batch_plans.append(plan.to_dict())

        async def _run_group(group_indices: list[int]) -> list[tuple[int, LLMResponse]]:
            results = []
            for idx in group_indices:
                resp = await self.generate(requests[idx])
                results.append((idx, resp))
            return results

        group_results = await asyncio.gather(
            *[_run_group(group.request_indices) for group in plan.groups]
        )
        all_results: list[tuple[int, LLMResponse]] = []
        for group in group_results:
            all_results.extend(group)
        all_results.sort(key=lambda item: item[0])
        return [resp for _, resp in all_results]

    def workload_features(self) -> list[WorkloadFeature]:
        return list(self._workload_features)

    def stats(self) -> dict[str, Any]:
        by_backend: dict[str, int] = {}
        for entry in self._call_log:
            by_backend[entry["backend"]] = by_backend.get(entry["backend"], 0) + 1
        total_hit = sum(e.get("cache_hit_tokens", 0) for e in self._call_log)
        total_miss = sum(e.get("cache_miss_tokens", 0) for e in self._call_log)
        hit_miss_total = total_hit + total_miss
        available_calls = sum(e.get("cache_metrics_available", True) for e in self._call_log)
        return {
            "total_calls": len(self._call_log),
            "total_tokens": sum(e["tokens_used"] for e in self._call_log),
            "total_prompt_tokens": sum(e.get("prompt_tokens", 0) for e in self._call_log),
            "total_completion_tokens": sum(
                e.get("completion_tokens", 0) for e in self._call_log
            ),
            "total_cache_hit_tokens": total_hit,
            "total_cache_miss_tokens": total_miss,
            "cache_metrics_available_calls": available_calls,
            "cache_metrics_complete": available_calls == len(self._call_log),
            "cache_hit_ratio": (
                total_hit / hit_miss_total if hit_miss_total > 0
                else None if self._call_log and not available_calls else 0.0
            ),
            "total_prefill_tokens": sum(
                e.get("prefill_tokens", 0) for e in self._call_log
            ),
            "total_decode_tokens": sum(
                e.get("decode_tokens", 0) for e in self._call_log
            ),
            "total_prefill_secs": sum(
                e.get("prefill_secs", 0.0) for e in self._call_log
            ),
            "total_decode_secs": sum(
                e.get("decode_secs", 0.0) for e in self._call_log
            ),
            "total_prefill_saved_tokens": sum(
                e.get("prefill_saved_tokens", 0) for e in self._call_log
            ),
            "calls_by_backend": by_backend,
            "registered_backends": list(self._backends.keys()),
            "workload_feature_count": len(self._workload_features),
            "adaptive_batch_plan_count": len(self._batch_plans),
            "last_adaptive_batch_plan": (
                self._batch_plans[-1] if self._batch_plans else None
            ),
        }
