from agentos.context.eviction import SegmentCandidate, SegmentEvictor
from agentos.context.prefix import ContextPriority


def _cand(name, tokens, *, priority=30, hits=0, content=None):
    return SegmentCandidate(
        name=name,
        content=content if content is not None else (name + " ") * tokens,
        priority=priority,
        hits=hits,
        pinned=priority <= ContextPriority.REPO_SUMMARY,
        tokens=tokens,
    )


def test_no_eviction_when_under_budget():
    ev = SegmentEvictor()
    cands = [_cand("a", 10), _cand("b", 10)]
    survivors, report = ev.evict(cands, budget=100)
    assert {c.name for c in survivors} == {"a", "b"}
    assert report.dropped == []
    assert report.tokens_saved == 0


def test_pinned_segments_never_dropped():
    ev = SegmentEvictor()
    cands = [
        _cand("pin", 200, priority=ContextPriority.TASK_BACKGROUND),
        _cand("cold", 50, hits=0),
    ]
    survivors, report = ev.evict(cands, budget=100)
    names = {c.name for c in survivors}
    assert "pin" in names          # pinned kept even though it alone exceeds budget
    assert "cold" in report.dropped


def test_lru_evicts_coldest_first():
    ev = SegmentEvictor()
    # budget leaves room for only one of the two evictable segments
    cands = [
        _cand("pin", 40, priority=ContextPriority.REPO_SUMMARY),
        _cand("hot", 40, hits=5),
        _cand("cold", 40, hits=1),
    ]
    survivors, report = ev.evict(cands, budget=80)
    names = {c.name for c in survivors}
    assert names == {"pin", "hot"}     # coldest (fewest hits) evicted
    assert report.dropped == ["cold"]


def test_lru_tie_breaks_on_larger_tokens():
    ev = SegmentEvictor()
    cands = [
        _cand("pin", 40, priority=ContextPriority.REPO_SUMMARY),
        _cand("small", 10, hits=2),
        _cand("big", 40, hits=2),
    ]
    survivors, report = ev.evict(cands, budget=60)
    names = {c.name for c in survivors}
    assert names == {"pin", "small"}   # equal hits -> drop larger ("big") first
    assert report.dropped == ["big"]


def test_prefix_aware_keeps_high_cache_value_segment():
    ev = SegmentEvictor(prefix_aware=True)
    cands = [
        _cand("pin", 40, priority=ContextPriority.REPO_SUMMARY),
        _cand("small-hot", 10, hits=5),
        _cand("large-prefix", 40, hits=2),
    ]
    cands[2].cache_value = 400.0
    survivors, report = ev.evict(cands, budget=80)

    names = {c.name for c in survivors}
    assert names == {"pin", "large-prefix"}
    assert report.dropped == ["small-hot"]
    assert report.policy == "prefix_aware"


def test_prefix_aware_prefers_contiguous_prefix_segments():
    ev = SegmentEvictor(prefix_aware=True)
    cands = [
        _cand("pin", 40, priority=ContextPriority.REPO_SUMMARY),
        _cand("early", 20, hits=0),
        _cand("late-hot", 20, hits=10),
    ]
    cands[1].ordinal = 1
    cands[1].prefix_shared = True
    cands[2].ordinal = 2
    cands[2].prefix_shared = True

    survivors, report = ev.evict(cands, budget=60)

    names = {c.name for c in survivors}
    assert names == {"pin", "early"}
    assert report.dropped == ["late-hot"]


def test_near_duplicates_merged():
    ev = SegmentEvictor(similarity_threshold=0.85)
    a = "The database schema uses PostgreSQL with three tables"
    b = "The database schema uses PostgreSQL with 3 tables"
    cands = [
        _cand("seg-a", 12, content=a),
        _cand("seg-b", 11, content=b),
    ]
    survivors, report = ev.evict(cands, budget=1000)
    # larger-token survivor kept, the other recorded as merged
    assert {c.name for c in survivors} == {"seg-a"}
    assert report.merged == [("seg-b", "seg-a")]


def test_distinct_segments_not_merged():
    ev = SegmentEvictor(similarity_threshold=0.85)
    cands = [
        _cand("seg-a", 8, content="mock response for agent-architect"),
        _cand("seg-b", 8, content="mock response for agent-backend"),
    ]
    survivors, report = ev.evict(cands, budget=1000)
    assert {c.name for c in survivors} == {"seg-a", "seg-b"}
    assert report.merged == []


def test_empty_input():
    ev = SegmentEvictor()
    survivors, report = ev.evict([], budget=100)
    assert survivors == []
    assert report.kept == []
    assert report.dropped == []
    assert report.tokens_before == 0
    assert report.tokens_after == 0
    assert report.tokens_saved == 0


def test_merge_saves_segment_from_eviction():
    # Two near-dups + one distinct, budget fits two segments. After merge the
    # two dups collapse to one, so nothing needs to be evicted.
    ev = SegmentEvictor(similarity_threshold=0.85)
    dup_a = "The database schema uses PostgreSQL with three tables"
    dup_b = "The database schema uses PostgreSQL with 3 tables"
    cands = [
        _cand("dup-a", 40, content=dup_a),
        _cand("dup-b", 40, content=dup_b),
        _cand("other", 40, content="completely unrelated cache warming notes"),
    ]
    survivors, report = ev.evict(cands, budget=80)
    names = {c.name for c in survivors}
    # one dup absorbed (merge), the other dup + other survive, nothing evicted
    assert report.merged == [("dup-b", "dup-a")]
    assert report.dropped == []
    assert names == {"dup-a", "other"}


def test_report_token_accounting():
    ev = SegmentEvictor()
    cands = [
        _cand("pin", 40, priority=ContextPriority.REPO_SUMMARY),
        _cand("cold", 40, hits=0),
    ]
    survivors, report = ev.evict(cands, budget=50)
    assert report.tokens_before == 80
    assert report.tokens_after == 40
    assert report.tokens_saved == 40
    assert report.kept == ["pin"]
