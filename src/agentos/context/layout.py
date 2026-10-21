from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from agentos.context.prefix import ContextPriority


@dataclass(frozen=True)
class LayoutObjectiveWeights:
    priority: float = 1.0
    reuse: float = 1.0
    cache: float = 1.0
    prefix_continuity: float = 2.0
    token_cost: float = 0.2
    fragmentation: float = 1.0


@dataclass
class SegmentScore:
    name: str
    total: float
    priority_score: float
    reuse_score: float
    cache_score: float
    prefix_continuity_score: float
    token_cost_penalty: float
    fragmentation_penalty: float
    tokens: int
    forced: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LayoutDecision:
    policy: str
    budget: int
    kept: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    tokens_before: int = 0
    tokens_after: int = 0
    tokens_saved: int = 0
    score: float = 0.0
    baseline_policy: str = "lru"
    baseline_kept: list[str] = field(default_factory=list)
    baseline_score: float = 0.0
    score_delta: float = 0.0
    kept_prefix_tokens: int = 0
    fragmentation_penalty: float = 0.0
    segment_scores: list[SegmentScore] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["segment_scores"] = [score.to_dict() for score in self.segment_scores]
        return data


class ContextLayoutOptimizer:
    """Budgeted optimizer for cache-aware context layouts.

    The optimizer treats pinned segments as hard constraints. For evictable
    segments it enumerates every contiguous prefix length, then solves a sparse
    0/1 knapsack over the remaining segments. This keeps the implementation
    deterministic and small while making prefix continuity an explicit part of
    the objective instead of a side effect of sorting.
    """

    def __init__(
        self,
        weights: LayoutObjectiveWeights | None = None,
        policy_name: str = "budgeted_prefix_optimizer",
    ) -> None:
        self._weights = weights or LayoutObjectiveWeights()
        self._policy_name = policy_name

    def optimize(self, candidates: list[Any], budget: int) -> LayoutDecision:
        tokens_before = sum(int(c.tokens) for c in candidates)
        pinned = [c for c in candidates if bool(c.pinned)]
        evictable = [c for c in candidates if not bool(c.pinned)]

        pinned_tokens = sum(int(c.tokens) for c in pinned)
        pinned_names = [c.name for c in pinned]
        if pinned_tokens >= budget:
            kept = list(pinned_names)
            dropped = [c.name for c in evictable]
            score = self._score_layout(
                pinned,
                contiguous_prefix_names=set(),
            )
            return LayoutDecision(
                policy=self._policy_name,
                budget=budget,
                kept=kept,
                dropped=dropped,
                tokens_before=tokens_before,
                tokens_after=pinned_tokens,
                tokens_saved=tokens_before - pinned_tokens,
                score=score,
                baseline_kept=kept,
                baseline_score=score,
                score_delta=0.0,
                segment_scores=self._segment_scores(
                    candidates,
                    contiguous_prefix_names=set(),
                    kept_names=set(kept),
                ),
            )

        prefix_chain = sorted(
            (c for c in evictable if bool(c.prefix_shared)),
            key=lambda c: (int(c.ordinal), c.name),
        )
        pinned_prefix_names = {
            c.name
            for c in pinned
            if bool(c.prefix_shared)
        }

        best_kept: list[Any] = list(pinned)
        best_prefix_names: set[str] = set(pinned_prefix_names)
        best_score = self._score_layout(best_kept, best_prefix_names)

        forced_prefix: list[Any] = []
        forced_tokens = pinned_tokens
        for prefix_len in range(len(prefix_chain) + 1):
            if prefix_len > 0:
                cand = prefix_chain[prefix_len - 1]
                forced_tokens += int(cand.tokens)
                forced_prefix.append(cand)
            if forced_tokens > budget:
                break

            prefix_names = pinned_prefix_names | {c.name for c in forced_prefix}
            remaining = [
                c
                for c in evictable
                if c.name not in prefix_names
            ]
            remaining_budget = budget - forced_tokens
            chosen = self._knapsack(
                remaining,
                remaining_budget,
                contiguous_prefix_names=prefix_names,
            )
            kept = list(pinned) + list(forced_prefix) + chosen
            score = self._score_layout(kept, prefix_names)
            if score > best_score or (
                math.isclose(score, best_score)
                and sum(int(c.tokens) for c in kept) < sum(
                    int(c.tokens) for c in best_kept
                )
            ):
                best_score = score
                best_kept = kept
                best_prefix_names = prefix_names

        kept_names = {c.name for c in best_kept}
        dropped = [c.name for c in candidates if c.name not in kept_names]
        tokens_after = sum(int(c.tokens) for c in best_kept)
        baseline_kept = self._lru_baseline(candidates, budget)
        baseline_score = self._score_layout(
            baseline_kept,
            contiguous_prefix_names=self._contiguous_prefix_names(baseline_kept),
        )

        return LayoutDecision(
            policy=self._policy_name,
            budget=budget,
            kept=[c.name for c in best_kept],
            dropped=dropped,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_saved=tokens_before - tokens_after,
            score=best_score,
            baseline_kept=[c.name for c in baseline_kept],
            baseline_score=baseline_score,
            score_delta=best_score - baseline_score,
            kept_prefix_tokens=sum(
                int(c.tokens)
                for c in best_kept
                if c.name in best_prefix_names
            ),
            fragmentation_penalty=self._layout_fragmentation_penalty(
                best_kept,
                best_prefix_names,
            ),
            segment_scores=self._segment_scores(
                candidates,
                contiguous_prefix_names=best_prefix_names,
                kept_names=kept_names,
            ),
        )

    def _knapsack(
        self,
        candidates: list[Any],
        budget: int,
        *,
        contiguous_prefix_names: set[str],
    ) -> list[Any]:
        states: dict[int, tuple[float, tuple[Any, ...]]] = {0: (0.0, tuple())}
        for cand in sorted(candidates, key=lambda c: (int(c.ordinal), c.name)):
            cand_tokens = int(cand.tokens)
            cand_score = self._score_segment(
                cand,
                contiguous_prefix=bool(cand.name in contiguous_prefix_names),
                fragmented=bool(cand.prefix_shared)
                and cand.name not in contiguous_prefix_names,
            ).total
            next_states = dict(states)
            for used, (score, chosen) in states.items():
                new_used = used + cand_tokens
                if new_used > budget:
                    continue
                new_score = score + cand_score
                old = next_states.get(new_used)
                if old is None or new_score > old[0]:
                    next_states[new_used] = (new_score, chosen + (cand,))
            states = self._prune_states(next_states)

        best = max(states.values(), key=lambda item: (item[0], -sum(c.tokens for c in item[1])))
        return list(best[1])

    @staticmethod
    def _prune_states(
        states: dict[int, tuple[float, tuple[Any, ...]]],
    ) -> dict[int, tuple[float, tuple[Any, ...]]]:
        ordered = sorted(states.items())
        pruned: dict[int, tuple[float, tuple[Any, ...]]] = {}
        best_score = -math.inf
        for used, value in ordered:
            score = value[0]
            if score > best_score:
                pruned[used] = value
                best_score = score
        return pruned

    def _segment_scores(
        self,
        candidates: list[Any],
        *,
        contiguous_prefix_names: set[str],
        kept_names: set[str],
    ) -> list[SegmentScore]:
        scores: list[SegmentScore] = []
        for cand in candidates:
            scores.append(
                self._score_segment(
                    cand,
                    contiguous_prefix=cand.name in contiguous_prefix_names,
                    fragmented=bool(cand.prefix_shared)
                    and cand.name not in contiguous_prefix_names,
                    forced=bool(cand.pinned),
                    kept=cand.name in kept_names,
                )
            )
        return scores

    def _score_layout(
        self,
        candidates: list[Any],
        contiguous_prefix_names: set[str],
    ) -> float:
        return sum(
            self._score_segment(
                cand,
                contiguous_prefix=cand.name in contiguous_prefix_names,
                fragmented=bool(cand.prefix_shared)
                and cand.name not in contiguous_prefix_names,
                forced=bool(cand.pinned),
            ).total
            for cand in candidates
        )

    def _score_segment(
        self,
        cand: Any,
        *,
        contiguous_prefix: bool,
        fragmented: bool,
        forced: bool = False,
        kept: bool = True,
    ) -> SegmentScore:
        max_priority = float(ContextPriority.AGENT_ROLE)
        priority_score = max(
            0.0,
            (max_priority - min(float(cand.priority), max_priority)) / max_priority,
        )
        reuse_signal = float(cand.reuse_score or cand.hits or 0.0)
        reuse_score = math.log1p(reuse_signal)
        cache_score = math.log1p(float(cand.cache_value or 0.0))
        prefix_score = 1.0 if contiguous_prefix else 0.0
        token_penalty = math.log1p(float(cand.tokens)) / 10.0
        fragmentation_penalty = 1.0 if fragmented else 0.0

        total = (
            self._weights.priority * priority_score
            + self._weights.reuse * reuse_score
            + self._weights.cache * cache_score
            + self._weights.prefix_continuity * prefix_score
            - self._weights.token_cost * token_penalty
            - self._weights.fragmentation * fragmentation_penalty
        )
        if forced:
            total += 1.0
        if not kept:
            total = 0.0

        return SegmentScore(
            name=cand.name,
            total=total,
            priority_score=priority_score,
            reuse_score=reuse_score,
            cache_score=cache_score,
            prefix_continuity_score=prefix_score,
            token_cost_penalty=token_penalty,
            fragmentation_penalty=fragmentation_penalty,
            tokens=int(cand.tokens),
            forced=forced,
        )

    @staticmethod
    def _lru_baseline(candidates: list[Any], budget: int) -> list[Any]:
        pinned = [c for c in candidates if bool(c.pinned)]
        kept = list(pinned)
        running = sum(int(c.tokens) for c in kept)
        evictable = [c for c in candidates if not bool(c.pinned)]
        for cand in sorted(
            evictable,
            key=lambda c: (-float(c.hits), float(c.tokens), float(c.ordinal), c.name),
        ):
            if running + int(cand.tokens) <= budget:
                kept.append(cand)
                running += int(cand.tokens)
        return kept

    @staticmethod
    def _contiguous_prefix_names(candidates: list[Any]) -> set[str]:
        prefix = sorted(
            (c for c in candidates if bool(c.prefix_shared)),
            key=lambda c: (int(c.ordinal), c.name),
        )
        names: set[str] = set()
        expected = 0
        for cand in prefix:
            if int(cand.ordinal) != expected:
                break
            names.add(cand.name)
            expected += 1
        return names

    @staticmethod
    def _layout_fragmentation_penalty(
        candidates: list[Any],
        contiguous_prefix_names: set[str],
    ) -> float:
        fragmented = [
            c
            for c in candidates
            if bool(c.prefix_shared) and c.name not in contiguous_prefix_names
        ]
        return float(len(fragmented))
