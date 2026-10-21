from __future__ import annotations

import agentos.isolation.delegation as delegation


def test_detect_delegation_prefers_container(monkeypatch):
    monkeypatch.setattr(delegation, "_in_container", lambda: True)
    monkeypatch.setattr(delegation, "_systemd_delegated", lambda: True)
    monkeypatch.setenv("AGENTOS_CGROUP_ALLOW_ROOT", "1")

    assert delegation.detect_delegation() == ("container", True)


def test_detect_delegation_accepts_systemd_delegation(monkeypatch):
    monkeypatch.setattr(delegation, "_in_container", lambda: False)
    monkeypatch.setattr(delegation, "_systemd_delegated", lambda: True)
    monkeypatch.delenv("AGENTOS_CGROUP_ALLOW_ROOT", raising=False)

    assert delegation.detect_delegation() == ("systemd_delegation", True)


def test_detect_delegation_accepts_explicit_host_override(monkeypatch):
    monkeypatch.setattr(delegation, "_in_container", lambda: False)
    monkeypatch.setattr(delegation, "_systemd_delegated", lambda: False)
    monkeypatch.setenv("AGENTOS_CGROUP_ALLOW_ROOT", "yes")

    assert delegation.detect_delegation() == ("host_root_override", True)


def test_detect_delegation_rejects_unapproved_host_root(monkeypatch):
    monkeypatch.setattr(delegation, "_in_container", lambda: False)
    monkeypatch.setattr(delegation, "_systemd_delegated", lambda: False)
    monkeypatch.delenv("AGENTOS_CGROUP_ALLOW_ROOT", raising=False)

    assert delegation.detect_delegation() == ("host_root", False)


def test_detect_delegation_does_not_accept_arbitrary_override(monkeypatch):
    monkeypatch.setattr(delegation, "_in_container", lambda: False)
    monkeypatch.setattr(delegation, "_systemd_delegated", lambda: False)
    monkeypatch.setenv("AGENTOS_CGROUP_ALLOW_ROOT", "sometimes")

    assert delegation.detect_delegation() == ("host_root", False)
