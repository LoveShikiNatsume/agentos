from __future__ import annotations

import json

from scripts import openeuler_member_a_preflight as preflight


def test_preflight_recommends_bpftool_when_real_path_ready(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight, "detect_delegation", lambda: ("container", True))
    monkeypatch.setattr(preflight, "_read_os_release", lambda _path: {"ID": "openEuler"})
    monkeypatch.setattr(preflight, "_kernel_release", lambda: "6.6.0")
    monkeypatch.setattr(preflight, "_path_exists", lambda path: True)
    monkeypatch.setattr(preflight, "_path_writable", lambda path: True)
    monkeypatch.setattr(preflight, "_bpffs_mounted", lambda _path: True)
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/sbin/bpftool")
    monkeypatch.setattr(preflight, "_cap_bpf", lambda _path: (True, None))

    result = preflight.run_preflight(output_path=tmp_path / "preflight.json")

    assert result["is_openeuler"] is True
    assert result["cgroup_v2"]["available"] is True
    assert result["psi"]["available"] is True
    assert result["bpf"]["bpffs_mounted"] is True
    assert result["bpf"]["bpftool_available"] is True
    assert result["bpf"]["cap_bpf"] is True
    assert result["recommended_bpf_mode"] == "bpftool"
    assert result["fallback_expected"] is False
    assert json.loads((tmp_path / "preflight.json").read_text()) == result


def test_preflight_recommends_auto_with_fallback_when_bpf_missing(monkeypatch):
    monkeypatch.setattr(preflight, "detect_delegation", lambda: ("host_root", False))
    monkeypatch.setattr(preflight, "_read_os_release", lambda _path: {"ID": "debian"})
    monkeypatch.setattr(preflight, "_kernel_release", lambda: "6.1.0")

    def exists(path):
        text = str(path)
        return (
            text.endswith("cgroup.controllers")
            or text.endswith("/pressure/cpu")
            or text.endswith("/pressure/memory")
            or text.endswith("/pressure/io")
        )

    monkeypatch.setattr(preflight, "_path_exists", exists)
    monkeypatch.setattr(preflight, "_path_writable", lambda path: False)
    monkeypatch.setattr(preflight, "_bpffs_mounted", lambda _path: False)
    monkeypatch.setattr(preflight.shutil, "which", lambda name: None)
    monkeypatch.setattr(preflight, "_cap_bpf", lambda _path: (False, "CAP_BPF missing"))

    result = preflight.run_preflight()

    assert result["is_openeuler"] is False
    assert result["delegation"]["allowed"] is False
    assert result["cgroup_v2"]["available"] is False
    assert result["psi"]["available"] is True
    assert result["bpf"]["bpftool_available"] is False
    assert result["recommended_bpf_mode"] == "auto"
    assert result["fallback_expected"] is True
    assert "bpftool not found" in result["missing"]
    assert "cgroup delegation not available" in result["missing"]
