import json

from agentos.context.manager import ContextManager


def _populated_manager() -> ContextManager:
    manager = ContextManager()
    manager.add_shared_ref("task-bg", "project background", priority=10)
    manager.add_shared_ref("repo-summary", "repo structure summary", priority=20)
    manager.add_private_ref("agent-1", "agent-1 step 1 result")
    manager.add_private_ref("agent-1", "agent-1 step 2 result")
    manager.add_private_ref("agent-2", "agent-2 private scratch")
    return manager


def test_export_agent_context_is_json_serializable():
    manager = _populated_manager()
    snapshot = manager.export_agent_context("agent-1")
    json.dumps(snapshot)


def test_export_contains_only_own_private_refs():
    manager = _populated_manager()
    snapshot = manager.export_agent_context("agent-1")
    dumped = json.dumps(snapshot)
    assert "agent-1 step 1 result" in dumped
    assert "agent-2 private scratch" not in dumped


def test_import_rebuilds_identical_messages():
    manager = _populated_manager()
    user_messages = [{"role": "user", "content": "continue the task"}]
    before = manager.build_messages("agent-1", user_messages)

    snapshot = manager.export_agent_context("agent-1")
    # 模拟跨进程恢复：快照经过 JSON 序列化往返
    snapshot = json.loads(json.dumps(snapshot))

    fresh = ContextManager()
    fresh.import_agent_context(snapshot)
    after = fresh.build_messages("agent-1", user_messages)

    assert after == before


def test_import_preserves_ref_names_and_hashes():
    manager = _populated_manager()
    snapshot = manager.export_agent_context("agent-1")

    fresh = ContextManager()
    fresh.import_agent_context(snapshot)

    for ref in manager.get_refs_for_agent("agent-1"):
        restored = fresh.get_ref(ref.name)
        assert restored is not None
        assert restored.content_hash == ref.content_hash
        assert restored.scope == ref.scope
        assert restored.owner_agent_id == ref.owner_agent_id
        assert restored.priority == ref.priority


def test_export_does_not_inflate_pool_stats():
    manager = _populated_manager()
    before = manager.stats()["pool"]["total_hits"]
    manager.export_agent_context("agent-1")
    after = manager.stats()["pool"]["total_hits"]
    assert after == before


def test_import_then_append_private_does_not_collide():
    manager = _populated_manager()
    snapshot = manager.export_agent_context("agent-1")

    fresh = ContextManager()
    fresh.import_agent_context(snapshot)
    # 恢复后 Agent 继续工作，新的私有上下文不能覆盖已恢复的条目
    new_ref = fresh.add_private_ref("agent-1", "post-recovery result")
    names = [ref.name for ref in fresh.get_refs_for_agent("agent-1")]
    assert len(names) == len(set(names))
    assert fresh.read("agent-1", new_ref.name) == "post-recovery result"
