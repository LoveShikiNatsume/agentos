from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from agentos.context.layout import ContextLayoutOptimizer
from agentos.context.prefix import ContextPriority


@dataclass
class SegmentCandidate:
    name: str
    content: str
    priority: int
    hits: int
    pinned: bool
    tokens: int
    ordinal: int = 0
    prefix_shared: bool = False
    reuse_score: float = 0.0
    cache_value: float = 0.0


@dataclass
class EvictionReport:
    kept: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    merged: list[tuple[str, str]] = field(default_factory=list)
    tokens_before: int = 0
    tokens_after: int = 0
    tokens_saved: int = 0
    policy: str = "lru"
    score: float = 0.0
    baseline_score: float = 0.0
    score_delta: float = 0.0
    kept_prefix_tokens: int = 0
    layout_decision: dict[str, Any] | None = None


class SegmentEvictor:
    """Segment-level context compressor.

    Level 1: merge near-duplicate (non-pinned) segments.
    Level 2: when over budget, keep pinned segments (mlock analogy) and evict
             the coldest (fewest-hits) evictable segments first (LRU analogy).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        pin_threshold: int = ContextPriority.REPO_SUMMARY,
        prefix_aware: bool = False,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        # Read by the caller (ContextManager) to decide which segments are pinned (priority <= pin_threshold).
        self.pin_threshold = pin_threshold
        self.prefix_aware = prefix_aware

    @property
    def policy_name(self) -> str:
        return "prefix_aware" if self.prefix_aware else "lru"

    def evict(
        self,
        candidates: list[SegmentCandidate],
        budget: int,
    ) -> tuple[list[SegmentCandidate], EvictionReport]:
        tokens_before = sum(c.tokens for c in candidates)

        survivors, merged = self._merge_similar(candidates)

        dropped: list[str] = []
        layout_decision = None
        total = sum(c.tokens for c in survivors)
        if total > budget:
            pinned = [c for c in survivors if c.pinned]
            evictable = [c for c in survivors if not c.pinned]

            running = sum(c.tokens for c in pinned)
            if self.prefix_aware:
                optimizer = ContextLayoutOptimizer()
                layout_decision = optimizer.optimize(survivors, budget)
                kept_names = set(layout_decision.kept)
                kept_evictable = [
                    c for c in evictable if c.name in kept_names
                ]
                dropped = layout_decision.dropped
            else:
                kept_evictable = []
                for c in sorted(evictable, key=self._keep_order_key):
                    if running + c.tokens <= budget:
                        kept_evictable.append(c)
                        running += c.tokens
                    else:
                        dropped.append(c.name)
            survivors = pinned + kept_evictable

        tokens_after = sum(c.tokens for c in survivors)
        report = EvictionReport(
            kept=[c.name for c in survivors],
            dropped=dropped,
            merged=merged,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_saved=tokens_before - tokens_after,
            policy=self.policy_name,
            score=layout_decision.score if layout_decision is not None else 0.0,
            baseline_score=(
                layout_decision.baseline_score
                if layout_decision is not None
                else 0.0
            ),
            score_delta=(
                layout_decision.score_delta
                if layout_decision is not None
                else 0.0
            ),
            kept_prefix_tokens=(
                layout_decision.kept_prefix_tokens
                if layout_decision is not None
                else 0
            ),
            layout_decision=(
                layout_decision.to_dict()
                if layout_decision is not None
                else None
            ),
        )
        return survivors, report

    def _select_prefix_aware(
        self,
        candidates: list[SegmentCandidate],
        running_tokens: int,
        budget: int,
    ) -> tuple[list[SegmentCandidate], list[str]]:
        kept: list[SegmentCandidate] = []
        kept_names: set[str] = set()
        running = running_tokens

        # KV prefix cache rewards contiguous bytes from the beginning. Keep the
        # earliest shared prefix contributors first, then spend leftover budget
        # on value-ranked non-contiguous context.
        prefix_candidates = sorted(
            (c for c in candidates if c.prefix_shared),
            key=lambda c: (c.ordinal, c.name),
        )
        for cand in prefix_candidates:
            if running + cand.tokens > budget:
                break
            kept.append(cand)
            kept_names.add(cand.name)
            running += cand.tokens

        leftovers = [c for c in candidates if c.name not in kept_names]
        for cand in sorted(leftovers, key=self._keep_order_key):
            if running + cand.tokens <= budget:
                kept.append(cand)
                kept_names.add(cand.name)
                running += cand.tokens

        dropped = [c.name for c in candidates if c.name not in kept_names]
        return kept, dropped

    def _keep_order_key(self, cand: SegmentCandidate) -> tuple[float, ...]:
        if not self.prefix_aware:
            # Keep hottest segments first; on a tie, prefer smaller segments so
            # the fixed budget retains more distinct context.
            return (-float(cand.hits), float(cand.tokens), float(cand.ordinal))

        value = cand.cache_value or self._derived_cache_value(cand)
        # Prefix-aware mode optimizes expected cache-hit tokens, not just
        # segment count: keep higher-value prefix contributors first.
        return (
            -value,
            0.0 if cand.prefix_shared else 1.0,
            -float(cand.reuse_score or cand.hits),
            float(cand.ordinal),
            float(cand.tokens),
        )

    @staticmethod
    def _derived_cache_value(cand: SegmentCandidate) -> float:
        priority_bonus = max(
            0.0,
            (ContextPriority.AGENT_ROLE - min(cand.priority, ContextPriority.AGENT_ROLE))
            / ContextPriority.AGENT_ROLE,
        )
        shared_bonus = 1.5 if cand.prefix_shared else 1.0
        return cand.tokens * (1.0 + cand.hits) * shared_bonus * (1.0 + priority_bonus)

    def _merge_similar(
        self,
        candidates: list[SegmentCandidate],
    ) -> tuple[list[SegmentCandidate], list[tuple[str, str]]]:
        survivors: list[SegmentCandidate] = []
        merged: list[tuple[str, str]] = []
        for cand in candidates:
            if cand.pinned:
                survivors.append(cand)
                continue
            absorbed_into: SegmentCandidate | None = None
            for kept in survivors:
                if kept.pinned:
                    continue
                ratio = SequenceMatcher(None, cand.content, kept.content).ratio()
                if ratio >= self.similarity_threshold:
                    absorbed_into = kept
                    break
            if absorbed_into is None:
                survivors.append(cand)
            elif cand.tokens > absorbed_into.tokens:
                # New candidate is larger: it survives, the prior one is absorbed.
                # Content is unchanged by a merge, so the survivor keeps its own token count; the absorbed segment's tokens are freed.
                idx = next(i for i, k in enumerate(survivors) if k is absorbed_into)
                survivors[idx] = cand
                merged.append((absorbed_into.name, cand.name))
            else:
                merged.append((cand.name, absorbed_into.name))
        return survivors, merged
