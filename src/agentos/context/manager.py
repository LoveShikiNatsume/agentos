from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import Any

from agentos.context.compressor import Compressor
from agentos.context.eviction import SegmentCandidate, SegmentEvictor
from agentos.context.pool import SharedContextPool
from agentos.context.prefix import ContextPriority, PrefixCanonicalizer, PrefixSegment
from agentos.context.types import (
    DEFAULT_CONTEXT_PRIORITY,
    ContextAccessDenied,
    ContextPermission,
    ContextRef,
    ContextScope,
)
from agentos.events.types import EventCategory, EventSeverity, RuntimeEvent

SNAPSHOT_VERSION = 1


@dataclass(frozen=True)
class PreparedContext:
    """A request payload and the prefix identity of the shared content it contains."""

    messages: list[dict[str, Any]]
    prefix_id: str
    shared_refs: tuple[str, ...]
    selection_details: dict[str, Any] = field(default_factory=dict)


class ContextManager:
    def __init__(
        self,
        compressor: Compressor | None = None,
        event_hook: Callable[[RuntimeEvent], None] | None = None,
        evictor: SegmentEvictor | None = None,
        segment_budget: int | None = None,
        prefix_order: str = "append_stable",
    ) -> None:
        if prefix_order not in {"append_stable", "name"}:
            raise ValueError("prefix_order must be 'append_stable' or 'name'")
        self._pool = SharedContextPool()
        self._refs: dict[str, ContextRef] = {}
        self._shared_order: list[str] = []
        self._private_order: dict[str, list[str]] = defaultdict(list)
        self._compressor = compressor
        self._event_hook = event_hook
        self._evictor = evictor
        self._segment_budget = segment_budget
        self._prefix_order = prefix_order
        self._compressed_tokens_total = 0
        self._last_compression_tokens_before = 0
        self._last_compression_tokens_after = 0
        self._last_shared_prefix_segments: list[PrefixSegment] | None = None
        self._last_prefix_id: str | None = None
        self._last_prefix_tokens = 0
        self._prefix_overlap_samples: list[dict[str, Any]] = []
        self._prefix_overlap_sample_limit = 256

    def set_event_hook(
        self, hook: Callable[[RuntimeEvent], None] | None
    ) -> None:
        """Wire (or rewire) the event hook. Used by the engine to route
        context events (compressed / access_denied) into its emitter."""
        self._event_hook = hook

    def add_shared(self, key: str, content: str) -> str:
        content_hash = self._pool.store(key, content)
        ref = ContextRef(
            name=key,
            scope=ContextScope.SHARED,
            content_hash=content_hash,
            permissions=frozenset({ContextPermission.READ, ContextPermission.SHARE}),
        )
        if key not in self._refs:
            self._shared_order.append(key)
        self._refs[key] = ref
        return content_hash

    def add_private(self, agent_id: str, content: str) -> str:
        idx = len(self._private_order[agent_id])
        private_key = f"{agent_id}:private:{idx}"
        content_hash = self._pool.store(private_key, content)
        ref = ContextRef(
            name=private_key,
            owner_agent_id=agent_id,
            scope=ContextScope.PRIVATE,
            content_hash=content_hash,
            permissions=frozenset({ContextPermission.READ, ContextPermission.WRITE}),
        )
        if private_key not in self._refs:
            self._private_order[agent_id].append(private_key)
        self._refs[private_key] = ref
        return content_hash

    def add_shared_ref(
        self,
        key: str,
        content: str,
        priority: int = DEFAULT_CONTEXT_PRIORITY,
    ) -> ContextRef:
        content_hash = self._pool.store(key, content)
        ref = ContextRef(
            name=key,
            scope=ContextScope.SHARED,
            content_hash=content_hash,
            permissions=frozenset({ContextPermission.READ, ContextPermission.SHARE}),
            priority=priority,
        )
        if key not in self._refs:
            self._shared_order.append(key)
        self._refs[key] = ref
        return ref

    def add_private_ref(self, agent_id: str, content: str) -> ContextRef:
        idx = len(self._private_order[agent_id])
        private_key = f"{agent_id}:private:{idx}"
        content_hash = self._pool.store(private_key, content)
        ref = ContextRef(
            name=private_key,
            owner_agent_id=agent_id,
            scope=ContextScope.PRIVATE,
            content_hash=content_hash,
            permissions=frozenset({ContextPermission.READ, ContextPermission.WRITE}),
        )
        if private_key not in self._refs:
            self._private_order[agent_id].append(private_key)
        self._refs[private_key] = ref
        return ref

    def _select_shared_names(
        self,
        agent_id: str,
        context_ref_names: Collection[str] | None,
    ) -> set[str]:
        if context_ref_names is None:
            return set(self._shared_order)
        if isinstance(context_ref_names, str):
            raise TypeError("context_ref_names must be a collection of ref names")
        selected: set[str] = set()
        for raw_name in context_ref_names:
            name = raw_name.removeprefix("ctx://")
            ref = self._refs.get(name)
            if ref is None:
                raise KeyError(f"unknown ContextRef: {raw_name}")
            if not self._can_read(agent_id, ref):
                self._emit_access_denied(agent_id, ref)
                raise ContextAccessDenied(
                    f"agent {agent_id} has no read access to {ref.uri}"
                )
            if ref.scope == ContextScope.SHARED:
                selected.add(name)
        return selected

    def _apply_eviction(
        self, agent_id: str, shared_names: Collection[str] | None = None
    ) -> set[str] | None:
        """Run segment eviction and return the set of kept segment names (or None
        if no evictor/budget is configured).

        Builds candidates with READ-ONLY pool accessors (peek/hits/token_estimate)
        so the current agent's own read does not inflate the LRU hit counts before
        the evictor decides. The caller calls get() (which increments hits) only on
        the kept segments afterward. A context.compressed event is emitted on every
        active call (including zero-savings calls) to give the dashboard a per-call
        data point.
        """
        if self._evictor is None or self._segment_budget is None:
            return None
        candidates: list[SegmentCandidate] = []
        for ordinal, name in enumerate(self._shared_order):
            if shared_names is not None and name not in shared_names:
                continue
            ref = self._refs[name]
            if not ref.can_read():
                continue
            raw = self._pool.peek(ref.content_hash)
            if not raw:
                continue
            hits = self._pool.hits(ref.content_hash)
            tokens = self._pool.token_estimate(ref.content_hash)
            candidates.append(
                SegmentCandidate(
                    name=name,
                    content=raw,
                    priority=ref.priority,
                    hits=hits,
                    pinned=ref.priority <= self._evictor.pin_threshold,
                    tokens=tokens,
                    ordinal=self._prefix_ordinal(name, ordinal),
                    prefix_shared=ref.priority <= ContextPriority.SHARED_CONTEXT,
                    reuse_score=float(hits),
                    cache_value=self._segment_cache_value(ref, tokens, hits),
                )
            )
        survivors, report = self._evictor.evict(candidates, self._segment_budget)
        kept_names = {c.name for c in survivors}
        self._compressed_tokens_total += report.tokens_saved
        self._last_compression_tokens_before = report.tokens_before
        self._last_compression_tokens_after = report.tokens_after
        if self._event_hook is not None:
            self._event_hook(
                RuntimeEvent(
                    event_type="context.compressed",
                    agent_id=agent_id,
                    category=EventCategory.CONTEXT,
                    data={
                        "agent_id": agent_id,
                        "tokens_before": report.tokens_before,
                        "tokens_after": report.tokens_after,
                        "tokens_saved": report.tokens_saved,
                        "dropped_segments": report.dropped,
                        "merged_segments": [list(m) for m in report.merged],
                        "kept_count": len(report.kept),
                        "policy": report.policy,
                        "layout_score": report.score,
                        "layout_baseline_score": report.baseline_score,
                        "layout_score_delta": report.score_delta,
                        "kept_prefix_tokens": report.kept_prefix_tokens,
                        "layout_decision": report.layout_decision,
                    },
                )
            )
        return kept_names

    def _prefix_ordinal(self, name: str, ordinal: int) -> int:
        if self._prefix_order == "name":
            return 0
        return ordinal

    @staticmethod
    def _segment_cache_value(ref: ContextRef, tokens: int, hits: int) -> float:
        priority_bonus = max(
            0.0,
            (ContextPriority.AGENT_ROLE - min(ref.priority, ContextPriority.AGENT_ROLE))
            / ContextPriority.AGENT_ROLE,
        )
        return tokens * (1.0 + hits) * (1.0 + priority_bonus)

    def prepare_context(
        self,
        agent_id: str,
        user_messages: list[dict[str, Any]],
        *,
        context_ref_names: Collection[str] | None = None,
    ) -> PreparedContext:
        """Select refs before eviction and bind the actual prefix to this request.

        None includes all shared refs for compatibility; an explicit empty
        collection includes none. The agent's own readable private context is
        still appended after the shared prefix.
        """
        selected = self._select_shared_names(agent_id, context_ref_names)
        kept_names = self._apply_eviction(agent_id, selected)
        shared_segments: list[PrefixSegment] = []
        for ordinal, name in enumerate(self._shared_order):
            if name not in selected:
                continue
            ref = self._refs[name]
            if not self._can_read(agent_id, ref):
                continue
            if kept_names is not None and name not in kept_names:
                continue
            content = self._pool.get(ref.content_hash)
            if content:
                shared_segments.append(
                    PrefixSegment(
                        name=name,
                        content=content,
                        priority=ref.priority,
                        ordinal=self._prefix_ordinal(name, ordinal),
                    )
                )

        canonical = PrefixCanonicalizer.canonicalize(shared_segments)
        self._record_prefix_diagnostics(agent_id, canonical)
        prefix_id = (
            PrefixCanonicalizer.compute_prefix_id(canonical)
            if canonical else "__no_prefix__"
        )
        system_parts = [seg.content for seg in canonical]
        for name in self._private_order.get(agent_id, []):
            ref = self._refs[name]
            if not self._can_read(agent_id, ref):
                continue
            content = self._pool.get(ref.content_hash)
            if content:
                system_parts.append(content)

        messages: list[dict[str, Any]] = []
        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})
        messages.extend(user_messages)
        return PreparedContext(
            messages=messages,
            prefix_id=prefix_id,
            shared_refs=tuple(seg.name for seg in canonical),
        )

    def build_messages(
        self,
        agent_id: str,
        user_messages: list[dict[str, Any]],
        *,
        context_ref_names: Collection[str] | None = None,
    ) -> list[dict[str, Any]]:
        return self.prepare_context(
            agent_id, user_messages, context_ref_names=context_ref_names
        ).messages

    def _record_prefix_diagnostics(
        self,
        agent_id: str,
        segments: list[PrefixSegment],
    ) -> None:
        if self._last_shared_prefix_segments is not None:
            overlap = PrefixCanonicalizer.compute_overlap(
                self._last_shared_prefix_segments,
                segments,
            ).to_dict()
            overlap["agent_id"] = agent_id
            self._prefix_overlap_samples.append(overlap)
            if len(self._prefix_overlap_samples) > self._prefix_overlap_sample_limit:
                del self._prefix_overlap_samples[0]

        self._last_shared_prefix_segments = list(segments)
        prefix_content = PrefixCanonicalizer.build_prefix_content(segments)
        self._last_prefix_id = (
            PrefixCanonicalizer.compute_prefix_id(segments)
            if segments else "__no_prefix__"
        )
        self._last_prefix_tokens = PrefixCanonicalizer.estimate_tokens(prefix_content)

    async def async_build_messages(
        self,
        agent_id: str,
        user_messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        *,
        context_ref_names: Collection[str] | None = None,
    ) -> list[dict[str, Any]]:
        messages = self.build_messages(
            agent_id, user_messages, context_ref_names=context_ref_names
        )
        if self._compressor is not None:
            messages = await self._compressor.compress(messages, max_tokens=max_tokens)
        return messages

    def read(self, agent_id: str, name: str) -> str:
        if name not in self._refs:
            raise KeyError(f"unknown ContextRef: {name}")
        ref = self._refs[name]
        if not self._can_read(agent_id, ref):
            self._emit_access_denied(agent_id, ref)
            raise ContextAccessDenied(
                f"agent {agent_id} has no read access to {ref.uri}"
            )
        content = self._pool.get(ref.content_hash)
        if content is None:
            raise KeyError(f"content missing for ContextRef: {name}")
        return content

    def _can_read(self, agent_id: str, ref: ContextRef) -> bool:
        if not ref.can_read():
            return False
        if ref.scope == ContextScope.PRIVATE and ref.owner_agent_id != agent_id:
            return False
        return True

    def _emit_access_denied(self, agent_id: str, ref: ContextRef) -> None:
        if self._event_hook is None:
            return
        self._event_hook(
            RuntimeEvent(
                event_type="context.access_denied",
                agent_id=agent_id,
                category=EventCategory.CONTEXT,
                severity=EventSeverity.WARNING,
                data={
                    "ref": ref.uri,
                    "owner_agent_id": ref.owner_agent_id,
                    "scope": ref.scope.value,
                },
            )
        )

    def export_agent_context(self, agent_id: str) -> dict[str, Any]:
        shared = []
        for name in self._shared_order:
            ref = self._refs[name]
            shared.append(
                {
                    "name": name,
                    "content": self._pool.peek(ref.content_hash),
                    "priority": ref.priority,
                }
            )
        private = []
        for name in self._private_order.get(agent_id, []):
            ref = self._refs[name]
            private.append(
                {
                    "name": name,
                    "content": self._pool.peek(ref.content_hash),
                }
            )
        return {
            "version": SNAPSHOT_VERSION,
            "agent_id": agent_id,
            "shared": shared,
            "private": private,
        }

    def import_agent_context(
        self, snapshot: dict[str, Any], *, restore_shared: bool = True
    ) -> None:
        """Replace one agent's private state; optionally import shared context.

        In-place retries set restore_shared=False so concurrent shared updates
        survive. This restores context references, not external tool side effects.
        """
        if snapshot.get("version") != SNAPSHOT_VERSION:
            raise ValueError(f"unsupported snapshot version: {snapshot.get('version')}")
        agent_id = snapshot["agent_id"]
        private = snapshot["private"]
        names: set[str] = set()
        for seg in private:
            name = seg["name"]
            if name in names:
                raise ValueError(f"duplicate private snapshot ref: {name}")
            names.add(name)
            current = self._refs.get(name)
            if current is not None and (
                current.scope != ContextScope.PRIVATE
                or current.owner_agent_id != agent_id
            ):
                raise ContextAccessDenied(f"snapshot cannot replace {current.uri}")
            if not isinstance(seg["content"], str):
                raise ValueError("private snapshot content must be text")
        if restore_shared:
            for seg in snapshot["shared"]:
                current = self._refs.get(seg["name"])
                if seg["name"] in names or (
                    current is not None and current.scope != ContextScope.SHARED
                ):
                    raise ContextAccessDenied("shared snapshot ref conflicts with private context")
            for seg in snapshot["shared"]:
                self.add_shared_ref(seg["name"], seg["content"], priority=seg["priority"])

        # Remove scratch written after the snapshot as well as replaced refs.
        # Releasing references does not physically purge the content-addressed pool.
        for name in self._private_order.get(agent_id, []):
            ref = self._refs.pop(name)
            self._pool.release(ref.content_hash)
        self._private_order[agent_id] = []
        for seg in private:
            name = seg["name"]
            content_hash = self._pool.store(name, seg["content"])
            self._refs[name] = ContextRef(
                name=name,
                owner_agent_id=agent_id,
                scope=ContextScope.PRIVATE,
                content_hash=content_hash,
                permissions=frozenset({ContextPermission.READ, ContextPermission.WRITE}),
            )
            self._private_order[agent_id].append(name)

    def get_ref(self, name: str) -> ContextRef | None:
        return self._refs.get(name)

    def get_refs_for_agent(self, agent_id: str) -> list[ContextRef]:
        refs: list[ContextRef] = []
        for name in self._shared_order:
            refs.append(self._refs[name])
        for name in self._private_order.get(agent_id, []):
            refs.append(self._refs[name])
        return refs

    def resolve_prefix_id(self, context_ref_names: list[str] | None = None) -> str:
        """Compute the prefix_id a task will use based on its shared context refs.

        If *context_ref_names* is given, only those shared refs are considered.
        Otherwise all shared refs are included (the common case).
        """
        ordinals = {name: idx for idx, name in enumerate(self._shared_order)}
        if context_ref_names is not None:
            normalized = dict.fromkeys(n.removeprefix("ctx://") for n in context_ref_names)
            names = [n for n in normalized if n in self._refs
                     and self._refs[n].scope == ContextScope.SHARED]
        else:
            names = list(self._shared_order)

        segments: list[PrefixSegment] = []
        for name in names:
            ref = self._refs[name]
            if not ref.can_read():
                continue
            # peek(), not get(): resolving a prefix id for scheduling is not an
            # agent consuming the context, so it must not inflate LRU hit counts.
            content = self._pool.peek(ref.content_hash)
            if content:
                segments.append(
                    PrefixSegment(
                        name=name,
                        content=content,
                        priority=ref.priority,
                        ordinal=self._prefix_ordinal(name, ordinals.get(name, 0)),
                    )
                )

        if not segments:
            return "__no_prefix__"
        return PrefixCanonicalizer.compute_prefix_id(segments)

    def prefix_diagnostics(self) -> dict[str, Any]:
        samples = self._prefix_overlap_samples
        avg_shared_tokens = (
            sum(s["shared_prefix_tokens"] for s in samples) / len(samples)
            if samples
            else 0.0
        )
        avg_current_ratio = (
            sum(s["right_overlap_ratio"] for s in samples) / len(samples)
            if samples
            else 0.0
        )
        avg_shorter_ratio = (
            sum(s["shorter_overlap_ratio"] for s in samples) / len(samples)
            if samples
            else 0.0
        )
        segments = self._last_shared_prefix_segments or []
        return {
            "last_prefix_id": self._last_prefix_id or "__no_prefix__",
            "last_prefix_tokens": self._last_prefix_tokens,
            "segment_count": len(segments),
            "prefix_order": self._prefix_order,
            "segment_fingerprints": [
                fp.to_dict()
                for fp in PrefixCanonicalizer.segment_fingerprints(segments)
            ],
            "overlap_samples": len(samples),
            "avg_shared_prefix_tokens": avg_shared_tokens,
            "avg_current_overlap_ratio": avg_current_ratio,
            "avg_shorter_overlap_ratio": avg_shorter_ratio,
            "last_overlap": samples[-1] if samples else None,
        }

    def stats(self) -> dict[str, Any]:
        per_segment: list[dict[str, Any]] = []
        shared_tokens = 0
        for ordinal, name in enumerate(self._shared_order):
            ref = self._refs[name]
            tokens = self._pool.token_estimate(ref.content_hash)
            hits = self._pool.hits(ref.content_hash)
            shared_tokens += tokens
            per_segment.append({
                "name": name,
                "hits": hits,
                "tokens": tokens,
                "pinned": ref.priority <= ContextPriority.REPO_SUMMARY,
                "ordinal": ordinal,
                "prefix_shared": ref.priority <= ContextPriority.SHARED_CONTEXT,
                "cache_value": self._segment_cache_value(ref, tokens, hits),
            })
        private_tokens = 0
        private_refs = 0
        for names in self._private_order.values():
            private_refs += len(names)
            for name in names:
                ref = self._refs[name]
                private_tokens += self._pool.token_estimate(ref.content_hash)

        pool_stats = self._pool.stats()
        active_shared_tokens = (
            self._last_compression_tokens_after
            if self._last_compression_tokens_before > 0
            else shared_tokens
        )
        active_wss_tokens = active_shared_tokens + private_tokens
        duplicate_tokens = int(pool_stats.get("token_savings", 0))
        reuse_denominator = active_wss_tokens + duplicate_tokens
        context_reuse_ratio = (
            duplicate_tokens / reuse_denominator
            if reuse_denominator > 0
            else 0.0
        )
        return {
            "pool": pool_stats,
            "shared_segments": len(self._shared_order),
            "shared_refs": len(self._shared_order),
            "private_refs": private_refs,
            "agents_with_private": len(self._private_order),
            "total_refs": len(self._refs),
            "shared_tokens": shared_tokens,
            "private_tokens": private_tokens,
            "duplicate_tokens": duplicate_tokens,
            "compressed_tokens": self._compressed_tokens_total,
            "active_wss_tokens": active_wss_tokens,
            "context_reuse_ratio": context_reuse_ratio,
            "compression": {
                "tokens_before": self._last_compression_tokens_before,
                "tokens_after": self._last_compression_tokens_after,
                "tokens_saved_total": self._compressed_tokens_total,
            },
            "prefix_diagnostics": self.prefix_diagnostics(),
            "per_segment": per_segment,
        }
