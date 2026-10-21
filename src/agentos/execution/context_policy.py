"""Combined context selection and cache feedback; no lossy compression or budgets."""
from dataclasses import replace

from agentos.context.cache_tracker import CacheHeatTracker
from agentos.context.manager import ContextManager
from agentos.context.types import ContextScope
from agentos.llm.gateway import ModelGateway


class ExecutionContextManager(ContextManager):
    def __init__(self, policy):
        self.policy = policy
        super().__init__(prefix_order="name" if policy == "on" else "append_stable")

    def prepare_context(self, agent_id, user_messages, *, context_ref_names=None):
        declared = list(context_ref_names) if context_ref_names is not None else None
        requested = None if self.policy == "off" else declared
        if requested is not None:
            # Instructions remain present regardless of filtering decisions.
            requested = list(dict.fromkeys([name for name in ("policy", "task") if self.get_ref(name)] + requested))
        eligible = [ref for ref in self.get_refs_for_agent(agent_id)
                    if ref.scope == ContextScope.SHARED and self._can_read(agent_id, ref)]
        sizes = {ref.name: len((self._pool.peek(ref.content_hash) or "").encode()) for ref in eligible}
        prepared = super().prepare_context(agent_id, user_messages, context_ref_names=requested)
        selected = set(prepared.shared_refs)
        filtered = [ref.name for ref in eligible if ref.name not in selected]
        details = {"policy": self.policy, "selection": "all_shared" if requested is None else "explicit",
                   "declared_refs": declared, "requested_refs": requested,
                   "selected_refs": list(prepared.shared_refs), "eligible_shared_refs": [ref.name for ref in eligible],
                   "filtered_refs": filtered, "filtered_ref_count": len(filtered),
                   "eligible_shared_bytes": sum(sizes.values()),
                   "selected_shared_bytes": sum(sizes[name] for name in selected),
                   "filtered_shared_bytes": sum(sizes[name] for name in filtered),
                   "byte_metric_scope": "UTF-8 shared segment content, excluding separators; not model tokens",
                   "prefix_id": prepared.prefix_id, "prefix_order": self._prefix_order,
                   "private_context_policy": "existing_owner_only", "lossy_compression": False,
                   "segment_budget": None}
        return replace(prepared, selection_details=details)


class ObservedCacheHeatTracker(CacheHeatTracker):
    def record_call(self, prefix_id, metrics, timestamp=None):
        # Gateway's legacy fallback lacks explicit availability metadata. It must
        # not create a fabricated zero-hit observation in this execution path.
        if metrics.metadata.get("cache_metrics_available") is True:
            super().record_call(prefix_id, metrics, timestamp)


class ExecutionModelGateway(ModelGateway):
    def __init__(self, backend, records, policy):
        self.records, self.policy = records, policy
        super().__init__({"configured": backend}, default_backend="configured",
                         cache_tracker=ObservedCacheHeatTracker() if policy == "on" else None)

    async def generate(self, request):
        prefix = self._prefix_id(request)
        before = self.cache_tracker.predict_reuse(prefix).samples if self.cache_tracker else 0
        response = await super().generate(request)
        available = response.metrics.metadata.get("cache_metrics_available") is True
        after = self.cache_tracker.predict_reuse(prefix).samples if self.cache_tracker else 0
        self.records.emit("execution.cache_feedback", request.task_id, policy=self.policy,
            prefix_id=prefix, enabled=self.cache_tracker is not None, cache_metrics_available=available,
            feedback_applied=available and self.cache_tracker is not None and after > before, prefix_samples=after,
            cache_metric_source=response.metrics.metadata.get("cache_metric_source", "unavailable"),
            cache_hit_tokens=response.metrics.cache_hit_tokens if available else None,
            cache_miss_tokens=response.metrics.cache_miss_tokens if available else None,
            source="provider_request_usage" if available else "unavailable")
        return response
