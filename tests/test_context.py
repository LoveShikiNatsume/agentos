from agentos.context.manager import ContextManager
from agentos.context.pool import SharedContextPool
from agentos.context.prefix import ContextPriority
from agentos.context.types import ContextPermission, ContextRef, ContextScope


class TestSharedContextPool:
    def test_store_and_retrieve(self):
        pool = SharedContextPool()
        ref = pool.store("system_prompt", "You are a helpful assistant.")
        content = pool.get(ref)
        assert content == "You are a helpful assistant."

    def test_dedup_identical_content(self):
        pool = SharedContextPool()
        ref1 = pool.store("key1", "same content")
        ref2 = pool.store("key2", "same content")
        assert ref1 == ref2
        assert pool.stats()["unique_segments"] == 1

    def test_different_content_different_refs(self):
        pool = SharedContextPool()
        ref1 = pool.store("k1", "content A")
        ref2 = pool.store("k2", "content B")
        assert ref1 != ref2

    def test_hit_tracking(self):
        pool = SharedContextPool()
        ref = pool.store("k", "data")
        pool.get(ref)
        pool.get(ref)
        stats = pool.stats()
        assert stats["total_hits"] == 2

    def test_ref_counting_increments_on_store(self):
        pool = SharedContextPool()
        ref = pool.store("k1", "same content")
        assert pool.ref_count(ref) == 1
        pool.store("k2", "same content")
        assert pool.ref_count(ref) == 2

    def test_acquire_release(self):
        pool = SharedContextPool()
        ref = pool.store("k", "data")
        assert pool.ref_count(ref) == 1
        pool.acquire(ref)
        assert pool.ref_count(ref) == 2
        pool.release(ref)
        assert pool.ref_count(ref) == 1
        pool.release(ref)
        assert pool.ref_count(ref) == 0
        pool.release(ref)
        assert pool.ref_count(ref) == 0

    def test_token_estimation(self):
        pool = SharedContextPool()
        content = "a" * 400
        ref = pool.store("k", content)
        assert pool.token_estimate(ref) == 100

    def test_token_estimation_minimum_one(self):
        pool = SharedContextPool()
        ref = pool.store("k", "hi")
        assert pool.token_estimate(ref) == 1

    def test_stats_includes_token_savings(self):
        pool = SharedContextPool()
        ref = pool.store("k", "a" * 400)
        pool.get(ref)
        pool.get(ref)
        pool.get(ref)
        stats = pool.stats()
        assert stats["total_estimated_tokens"] == 100
        assert stats["token_savings"] == 300

    def test_stats_backward_compatible(self):
        pool = SharedContextPool()
        pool.store("k", "data")
        stats = pool.stats()
        assert "unique_segments" in stats
        assert "total_hits" in stats
        assert "total_bytes" in stats

    def test_pool_hits_is_read_only(self):
        pool = SharedContextPool()
        h = pool.store("k", "some shared content")

        # hits() must not change the count (unlike get())
        assert pool.hits(h) == 0
        assert pool.hits(h) == 0

        pool.get(h)  # get() DOES increment
        assert pool.hits(h) == 1
        assert pool.hits(h) == 1  # still read-only

    def test_pool_hits_unknown_hash_returns_zero(self):
        pool = SharedContextPool()
        assert pool.hits("nonexistent") == 0


class TestContextManager:
    async def test_add_shared_context(self):
        cm = ContextManager()
        cm.add_shared("project_background", "We are building an OS for agents.")
        messages = cm.build_messages("agent-1", [{"role": "user", "content": "hello"}])
        system_msgs = [message for message in messages if message["role"] == "system"]
        assert len(system_msgs) == 1
        assert "OS for agents" in system_msgs[0]["content"]

    async def test_private_context_isolated(self):
        cm = ContextManager()
        cm.add_private("agent-1", "I am a coder.")
        cm.add_private("agent-2", "I am a tester.")

        msgs1 = cm.build_messages("agent-1", [{"role": "user", "content": "hi"}])
        msgs2 = cm.build_messages("agent-2", [{"role": "user", "content": "hi"}])

        sys1 = [message for message in msgs1 if message["role"] == "system"]
        sys2 = [message for message in msgs2 if message["role"] == "system"]

        assert any("coder" in message["content"] for message in sys1)
        assert not any("coder" in message["content"] for message in sys2)
        assert any("tester" in message["content"] for message in sys2)

    async def test_shared_context_dedup_stats(self):
        cm = ContextManager()
        cm.add_shared("bg", "same background info")
        cm.build_messages("agent-1", [])
        cm.build_messages("agent-2", [])
        stats = cm.stats()
        assert stats["pool"]["unique_segments"] == 1

    async def test_token_estimate(self):
        cm = ContextManager()
        cm.add_shared("bg", "word " * 100)
        messages = cm.build_messages("a1", [{"role": "user", "content": "hi"}])
        total_chars = sum(len(message.get("content", "")) for message in messages)
        assert total_chars > 100


class TestContextManagerRefs:
    def test_add_shared_still_returns_str(self):
        cm = ContextManager()
        result = cm.add_shared("bg", "background info")
        assert isinstance(result, str)

    def test_add_shared_ref_returns_context_ref(self):
        cm = ContextManager()
        ref = cm.add_shared_ref("project", "We are building an OS.")
        assert isinstance(ref, ContextRef)
        assert ref.name == "project"
        assert ref.scope == ContextScope.SHARED
        assert ref.content_hash is not None
        assert ref.can_read()
        assert ref.can_share()

    def test_add_private_ref_returns_context_ref(self):
        cm = ContextManager()
        ref = cm.add_private_ref("agent-1", "I am a coder.")
        assert isinstance(ref, ContextRef)
        assert ref.owner_agent_id == "agent-1"
        assert ref.scope == ContextScope.PRIVATE
        assert ref.can_read()
        assert ref.can_write()

    def test_add_shared_ref_with_priority(self):
        cm = ContextManager()
        ref = cm.add_shared_ref("sys", "system instruction", priority=0)
        assert ref.priority == 0

    async def test_permission_check_skips_unreadable(self):
        cm = ContextManager()
        cm.add_shared("readable", "I should appear")
        content_hash = cm._pool.store("no_read", "I should NOT appear")
        no_read_ref = ContextRef(
            name="no_read",
            scope=ContextScope.SHARED,
            content_hash=content_hash,
            permissions=frozenset(),
        )
        cm._refs["no_read"] = no_read_ref
        cm._shared_order.append("no_read")

        messages = cm.build_messages("agent-1", [{"role": "user", "content": "hi"}])
        system_content = messages[0]["content"]
        assert "I should appear" in system_content
        assert "I should NOT appear" not in system_content

    def test_get_ref_by_name(self):
        cm = ContextManager()
        cm.add_shared_ref("project", "OS for agents")
        ref = cm.get_ref("project")
        assert ref is not None
        assert ref.name == "project"
        assert cm.get_ref("nonexistent") is None

    def test_get_refs_for_agent(self):
        cm = ContextManager()
        cm.add_shared_ref("bg", "background")
        cm.add_private_ref("agent-1", "coder role")

        refs = cm.get_refs_for_agent("agent-1")
        assert len(refs) == 2
        assert refs[0].name == "bg"
        assert refs[0].scope == ContextScope.SHARED
        assert refs[1].scope == ContextScope.PRIVATE
        assert refs[1].owner_agent_id == "agent-1"

    def test_get_refs_for_agent_excludes_other_private(self):
        cm = ContextManager()
        cm.add_shared_ref("bg", "background")
        cm.add_private_ref("agent-1", "coder")
        cm.add_private_ref("agent-2", "tester")

        refs = cm.get_refs_for_agent("agent-1")
        names = [r.name for r in refs]
        assert any("agent-1" in n for n in names)
        assert not any("agent-2" in n for n in names)

    def test_stats_includes_total_refs(self):
        cm = ContextManager()
        cm.add_shared("bg", "background")
        cm.add_private("agent-1", "coder")
        stats = cm.stats()
        assert stats["total_refs"] == 2

    def test_stats_includes_per_segment_breakdown(self):
        cm = ContextManager()
        cm.add_shared_ref("bg", "background " * 50, priority=ContextPriority.TASK_BACKGROUND)
        cm.add_shared("draft", "some draft content")
        cm.build_messages("agent-x", [{"role": "user", "content": "hi"}])  # triggers reads -> hits

        seg = {s["name"]: s for s in cm.stats()["per_segment"]}
        assert "bg" in seg and "draft" in seg
        assert seg["bg"]["pinned"] is True       # TASK_BACKGROUND (10) <= REPO_SUMMARY (20)
        assert seg["draft"]["pinned"] is False   # default priority 30
        assert seg["bg"]["tokens"] > 0
        assert all("hits" in s for s in cm.stats()["per_segment"])


class TestContextManagerPrefixCanonicalization:
    async def test_two_agents_get_identical_shared_prefix(self):
        cm = ContextManager()
        cm.add_shared_ref("sys", "System instruction", priority=ContextPriority.SYSTEM_INSTRUCTION)
        cm.add_shared_ref("bg", "Task background", priority=ContextPriority.TASK_BACKGROUND)
        cm.add_private_ref("agent-1", "I am a coder")
        cm.add_private_ref("agent-2", "I am a tester")

        msgs1 = cm.build_messages("agent-1", [{"role": "user", "content": "go"}])
        msgs2 = cm.build_messages("agent-2", [{"role": "user", "content": "go"}])

        sys1 = msgs1[0]["content"]
        sys2 = msgs2[0]["content"]

        shared_prefix_1 = sys1.rsplit("\n\n", 1)[0]
        shared_prefix_2 = sys2.rsplit("\n\n", 1)[0]
        assert shared_prefix_1 == shared_prefix_2
        assert shared_prefix_1 == "System instruction\n\nTask background"

    async def test_priority_overrides_insertion_order(self):
        cm = ContextManager()
        cm.add_shared_ref("repo", "Repo summary", priority=ContextPriority.REPO_SUMMARY)
        cm.add_shared_ref("sys", "System instruction", priority=ContextPriority.SYSTEM_INSTRUCTION)
        cm.add_shared_ref("bg", "Task background", priority=ContextPriority.TASK_BACKGROUND)

        msgs = cm.build_messages("agent-1", [])
        sys_content = msgs[0]["content"]
        parts = sys_content.split("\n\n")
        assert parts == ["System instruction", "Task background", "Repo summary"]

    async def test_same_priority_keeps_append_stable_order(self):
        cm = ContextManager()
        cm.add_shared_ref("zebra", "Z content", priority=ContextPriority.SHARED_CONTEXT)
        cm.add_shared_ref("alpha", "A content", priority=ContextPriority.SHARED_CONTEXT)

        msgs = cm.build_messages("agent-1", [])
        sys_content = msgs[0]["content"]
        parts = sys_content.split("\n\n")
        assert parts == ["Z content", "A content"]

    async def test_legacy_name_prefix_order_still_available(self):
        cm = ContextManager(prefix_order="name")
        cm.add_shared_ref("zebra", "Z content", priority=ContextPriority.SHARED_CONTEXT)
        cm.add_shared_ref("alpha", "A content", priority=ContextPriority.SHARED_CONTEXT)

        msgs = cm.build_messages("agent-1", [])
        sys_content = msgs[0]["content"]
        parts = sys_content.split("\n\n")
        assert parts == ["A content", "Z content"]

    async def test_private_appended_after_shared(self):
        cm = ContextManager()
        cm.add_shared_ref("bg", "Shared background", priority=ContextPriority.TASK_BACKGROUND)
        cm.add_private_ref("agent-1", "Private role")

        msgs = cm.build_messages("agent-1", [])
        sys_content = msgs[0]["content"]
        assert sys_content == "Shared background\n\nPrivate role"

    def test_prefix_diagnostics_records_overlap(self):
        cm = ContextManager()
        cm.add_shared_ref("bg", "Shared background " * 20, priority=ContextPriority.TASK_BACKGROUND)

        cm.build_messages("agent-1", [{"role": "user", "content": "go"}])
        cm.add_shared("draft", "new downstream notes " * 10)
        cm.build_messages("agent-2", [{"role": "user", "content": "go"}])

        diagnostics = cm.stats()["prefix_diagnostics"]
        assert diagnostics["overlap_samples"] == 1
        assert diagnostics["avg_current_overlap_ratio"] > 0
        assert diagnostics["last_overlap"]["reason"] == "prefix_extension"
        assert diagnostics["segment_count"] == 2
