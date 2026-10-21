import pytest

from agentos.context.types import ContextPermission, ContextRef, ContextScope


def test_context_ref_builds_ctx_uri():
    ref = ContextRef(name="task-bg", owner_agent_id="agent-1")
    assert ref.uri == "ctx://task-bg"
    assert str(ref) == "ctx://task-bg"
    assert ref.name == "task-bg"
    assert ref.scope == ContextScope.SHARED


def test_context_ref_parses_uri():
    ref = ContextRef.from_uri("ctx://repo-summary", owner_agent_id="agent-2")
    assert ref.name == "repo-summary"
    assert ref.uri == "ctx://repo-summary"
    assert ref.owner_agent_id == "agent-2"


def test_context_ref_parses_uri_with_positional_owner():
    ref = ContextRef.from_uri("ctx://task-bg", "agent-1")
    assert ref.owner_agent_id == "agent-1"
    assert ref.uri == "ctx://task-bg"


def test_context_ref_rejects_invalid_uri():
    with pytest.raises(ValueError, match="ContextRef URI must start with ctx://"):
        ContextRef.from_uri("repo-summary")


def test_context_ref_rejects_empty_name():
    with pytest.raises(ValueError, match="ContextRef name cannot be empty"):
        ContextRef(name="")


def test_context_ref_permissions():
    ref = ContextRef(
        name="private-notes",
        owner_agent_id="agent-1",
        scope=ContextScope.PRIVATE,
        permissions={ContextPermission.READ},
    )
    assert ref.can_read()
    assert not ref.can_write()


def test_context_ref_permissions_are_immutable():
    ref = ContextRef(name="immutable")
    assert isinstance(ref.permissions, frozenset)
    with pytest.raises(AttributeError):
        ref.permissions.add(ContextPermission.WRITE)  # type: ignore[attr-defined]


def test_context_ref_defensively_copies_permissions():
    permissions = {ContextPermission.READ}
    ref = ContextRef(name="copy", permissions=permissions)
    permissions.add(ContextPermission.WRITE)
    assert ref.can_read()
    assert not ref.can_write()


def test_context_ref_from_uri_defensively_copies_permissions():
    permissions = {ContextPermission.READ}
    ref = ContextRef.from_uri("ctx://copy", permissions=permissions)
    permissions.add(ContextPermission.WRITE)
    assert ref.can_read()
    assert not ref.can_write()
