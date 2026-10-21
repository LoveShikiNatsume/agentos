from agentos.context.eviction import SegmentCandidate, SegmentEvictor
from agentos.context.layout import ContextLayoutOptimizer, LayoutObjectiveWeights
from agentos.context.prefix import ContextPriority


def _candidate(
    name: str,
    tokens: int,
    *,
    hits: int = 0,
    ordinal: int = 0,
    priority: int = ContextPriority.SHARED_CONTEXT,
    pinned: bool = False,
    prefix_shared: bool = True,
    cache_value: float | None = None,
) -> SegmentCandidate:
    return SegmentCandidate(
        name=name,
        content=name,
        priority=priority,
        hits=hits,
        pinned=pinned,
        tokens=tokens,
        ordinal=ordinal,
        prefix_shared=prefix_shared,
        reuse_score=float(hits),
        cache_value=cache_value if cache_value is not None else float(tokens * (hits + 1)),
    )


def test_layout_optimizer_keeps_pinned_segments_as_hard_constraints():
    optimizer = ContextLayoutOptimizer()
    candidates = [
        _candidate(
            "system",
            120,
            pinned=True,
            priority=ContextPriority.SYSTEM_INSTRUCTION,
        ),
        _candidate("cold", 20, ordinal=1),
    ]

    decision = optimizer.optimize(candidates, budget=80)

    assert decision.kept == ["system"]
    assert decision.dropped == ["cold"]
    assert decision.tokens_after == 120


def test_layout_optimizer_values_contiguous_prefix_under_budget():
    optimizer = ContextLayoutOptimizer(
        LayoutObjectiveWeights(prefix_continuity=6.0, cache=0.2, reuse=0.2)
    )
    candidates = [
        _candidate(
            "system",
            20,
            pinned=True,
            priority=ContextPriority.SYSTEM_INSTRUCTION,
            ordinal=0,
        ),
        _candidate("prefix-a", 30, ordinal=1, hits=1),
        _candidate("prefix-b", 30, ordinal=2, hits=1),
        _candidate(
            "hot-notes",
            50,
            ordinal=3,
            hits=10,
            prefix_shared=False,
            cache_value=5000,
        ),
    ]

    decision = optimizer.optimize(candidates, budget=80)

    assert "prefix-a" in decision.kept
    assert "prefix-b" in decision.kept
    assert "hot-notes" not in decision.kept
    assert decision.kept_prefix_tokens == 80
    assert decision.score > decision.baseline_score


def test_prefix_aware_evictor_reports_layout_counterfactual():
    evictor = SegmentEvictor(prefix_aware=True, similarity_threshold=1.01)
    candidates = [
        _candidate(
            "system",
            20,
            pinned=True,
            priority=ContextPriority.SYSTEM_INSTRUCTION,
            ordinal=0,
        ),
        _candidate("prefix-a", 30, ordinal=1),
        _candidate("prefix-b", 30, ordinal=2),
        _candidate("tail", 80, ordinal=3, prefix_shared=False, hits=2),
    ]

    survivors, report = evictor.evict(candidates, budget=90)

    assert [candidate.name for candidate in survivors] == [
        "system",
        "prefix-a",
        "prefix-b",
    ]
    assert report.policy == "prefix_aware"
    assert report.layout_decision is not None
    assert report.kept_prefix_tokens == 80
    assert report.score_delta >= 0.0
