from agentos.context.manager import ContextManager
from agentos.context.eviction import SegmentEvictor
from agentos.context.prefix import ContextPriority


def test_context_manager_accepts_evictor_and_budget():
    ev = SegmentEvictor()
    cm = ContextManager(evictor=ev, segment_budget=180)
    assert cm._evictor is ev
    assert cm._segment_budget == 180


def test_context_manager_defaults_have_no_evictor():
    cm = ContextManager()
    assert cm._evictor is None
    assert cm._segment_budget is None


def test_set_event_hook_replaces_hook():
    captured = []
    hook = captured.append
    cm = ContextManager()
    cm.set_event_hook(hook)
    assert cm._event_hook is hook


# Distinct draft contents — pairwise similarity well below the 0.85 merge
# threshold, so Level-1 merge does NOT fire and Level-2 eviction is what we
# exercise here. (Near-dup merge is covered separately in test_segment_eviction.)
_DRAFT_TOPICS = [
    "authentication flow", "database indexing", "cache warming",
    "queue draining", "retry backoff", "rate limiting",
    "schema migration", "log rotation",
]


def _add_distinct_drafts(cm, n=8):
    for i in range(n):
        cm.add_shared(f"draft-{i}", f"engineering notes on {_DRAFT_TOPICS[i]} with details")


def _foundational_cm():
    """Two pinned foundational segments (~ priority 10/20) + an evictor."""
    captured = []
    # 180 ≈ pinned foundational (~150 tokens) + room for ~3 small drafts; low enough that eviction fires.
    cm = ContextManager(evictor=SegmentEvictor(), segment_budget=180)
    cm.set_event_hook(captured.append)
    cm.add_shared_ref("project-context", "P" * 360, priority=ContextPriority.TASK_BACKGROUND)
    cm.add_shared_ref("repo-summary", "R" * 240, priority=ContextPriority.REPO_SUMMARY)
    return cm, captured


def test_build_messages_emits_compressed_event():
    cm, captured = _foundational_cm()
    # Add distinct evictable intermediate segments (priority 30).
    _add_distinct_drafts(cm)

    cm.build_messages("agent-x", [{"role": "user", "content": "hi"}])

    events = [e for e in captured if e.event_type == "context.compressed"]
    assert len(events) == 1
    data = events[0].data
    assert data["tokens_before"] >= data["tokens_after"]
    assert data["tokens_saved"] > 0
    assert data["dropped_segments"]            # something was evicted
    # pinned foundational segments are never dropped
    assert "project-context" not in data["dropped_segments"]
    assert "repo-summary" not in data["dropped_segments"]


def test_build_messages_keeps_pinned_content_in_prefix():
    cm, _ = _foundational_cm()
    _add_distinct_drafts(cm)

    messages = cm.build_messages("agent-x", [{"role": "user", "content": "hi"}])
    system = next(m["content"] for m in messages if m["role"] == "system")
    assert "P" * 360 in system      # pinned project-context survived
    assert "R" * 240 in system      # pinned repo-summary survived


def test_build_messages_no_evictor_emits_nothing():
    captured = []
    cm = ContextManager()             # no evictor / budget
    cm.set_event_hook(captured.append)
    cm.add_shared("a", "x" * 100)
    cm.build_messages("agent-x", [{"role": "user", "content": "hi"}])
    assert [e for e in captured if e.event_type == "context.compressed"] == []


def test_build_messages_hits_not_polluted_by_self_read():
    # The evictor must read hits BEFORE this agent's own get() inflates them.
    cm, captured = _foundational_cm()
    _add_distinct_drafts(cm)
    # Run twice; second run sees hits from the first and still behaves.
    cm.build_messages("agent-1", [{"role": "user", "content": "hi"}])
    cm.build_messages("agent-2", [{"role": "user", "content": "hi"}])
    events = [e for e in captured if e.event_type == "context.compressed"]
    assert len(events) == 2
    assert all(e.data["tokens_saved"] > 0 for e in events)


def test_build_messages_emits_event_even_when_nothing_evicted():
    # When everything fits the budget, the evictor still runs and an event is
    # emitted (per-call data point for the dashboard), with tokens_saved == 0.
    captured = []
    cm = ContextManager(evictor=SegmentEvictor(), segment_budget=10_000)
    cm.set_event_hook(captured.append)
    cm.add_shared_ref("project-context", "P" * 360, priority=ContextPriority.TASK_BACKGROUND)
    cm.add_shared("draft-0", "engineering notes on authentication flow with details")

    cm.build_messages("agent-x", [{"role": "user", "content": "hi"}])

    events = [e for e in captured if e.event_type == "context.compressed"]
    assert len(events) == 1
    assert events[0].data["tokens_saved"] == 0
    assert events[0].data["dropped_segments"] == []


def test_engine_wires_context_event_hook():
    from agentos.runtime.engine import Runtime
    from agentos.llm.mock import MockLLMBackend

    cm = ContextManager()
    rt = Runtime(llm_backend=MockLLMBackend(), handlers={}, context_manager=cm)

    # After construction the engine should own the context event hook,
    # so an access-denied read surfaces in rt.events.
    cm.add_private("agent-owner", "secret")
    try:
        cm.read("agent-intruder", "agent-owner:private:0")
    except Exception:
        pass

    assert any(e.event_type == "context.access_denied" for e in rt.events)
